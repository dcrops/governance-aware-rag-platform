import os
import tempfile
import requests
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

import streamlit as st

from dotenv import load_dotenv

load_dotenv()

from app.config import PERSIST_DIR
from app.ingestion.ingest import ingest_document
from app.chunking.chunker import (
    chunk_document,
    chunk_document_by_delimiter,
    chunk_document_by_page,
    chunk_document_by_heading,
)
from app.embeddings.embeddings import EmbeddingClient
from app.models.vector_record import VectorRecord
from app.vector_store.vector_store import VectorStore
from app.ingestion.diagnostics import run_ingestion_diagnostics
from app.ingestion.diagnostics_store import (
    load_diagnostics,
    save_document_diagnostics,
    delete_document_diagnostics,
    delete_client_diagnostics,
)
from app.document_management.document_registry import (
    save_document_record,
    load_registry,
    delete_document_record,
    delete_client_registry,
)

from app.config import PERSIST_DIR, DEFAULT_CLIENT_NAME, DEFAULT_TOP_K, DEFAULT_MIN_SCORE

st.set_page_config(page_title="CRC Document Intelligence Copilot", layout="wide")

persist_dir = PERSIST_DIR

rewrite_client = OpenAI()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

domain_profile = os.getenv("DOMAIN_PROFILE", "general")

APP_TITLE = "Governance-Aware Document Intelligence Copilot"
APP_SUBTITLE = (
    "Ask questions across indexed documents with orchestration-aware retrieval, "
    "grounding checks, source evidence, and safe clarification handling."
)

LOGO_CANDIDATES = [
    Path("assets/icononly_transparent_nobuffer.png"),
    Path("icononly_transparent_nobuffer.png"),
    Path("/mnt/data/icononly_transparent_nobuffer.png"),
]
LOGO_PATH = next((path for path in LOGO_CANDIDATES if path.exists()), None)

st.markdown(
    """
    <style>
        .crc-hero {
            padding: 1.25rem 1.35rem;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(176,255,31,0.12), rgba(20,180,140,0.06));
            margin-bottom: 1.25rem;
        }
        .crc-hero-logo-wrap {
            height: 100%;
            min-height: 165px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .crc-eyebrow {
            font-size: 0.85rem;
            color: #B6FF25;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }
        .crc-title {
            font-size: 2rem;
            line-height: 1.15;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }
        .crc-subtitle {
            color: rgba(255,255,255,0.72);
            font-size: 1rem;
            max-width: 900px;
        }
        .crc-muted {
            color: rgba(255,255,255,0.65);
            font-size: 0.92rem;
        }
        .crc-footer {
            color: rgba(255,255,255,0.52);
            font-size: 0.82rem;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def build_contextual_question(question: str, chat_history: list[dict]) -> str:
    if not chat_history:
        return question

    recent_history = chat_history[-3:]

    context_lines = []

    for turn in recent_history:
        context_lines.append(f"Previous question: {turn.get('question', '')}")
        context_lines.append(f"Previous answer: {turn.get('answer', '')}")

    conversation_context = "\n".join(context_lines)

    prompt = f"""
You rewrite follow-up questions into standalone questions for a RAG system.

Use the recent conversation only to resolve references such as:
it, that, this, he, she, they, the event, the date, the person.

Do not answer the question.
Return only the rewritten standalone question.

Recent conversation:
{conversation_context}

Current question:
{question}

Standalone question:
"""

    response = rewrite_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        max_tokens=80,
    )

    return response.choices[0].message.content.strip()

def get_effective_top_k(
    question: str,
    selected_top_k: int,
    retrieval_mode: str,
    document_count: int,
) -> int:
    if retrieval_mode == "Standard chunk retrieval":
        return selected_top_k

    broad_top_k = max(
        selected_top_k,
        min(document_count * 5, 50),
    )

    if retrieval_mode == "Broad retrieval":
        return broad_top_k

    broad_query_terms = [
        "all",
        "every",
        "list",
        "summarise",
        "summarize",
        "across",
        "all documents",
        "all docs",
        "mentioned",
        "complete",
    ]

    question_lower = question.lower()

    if any(term in question_lower for term in broad_query_terms):
        return broad_top_k

    return selected_top_k

def is_aggregation_question(question: str) -> bool:
    aggregation_terms = [
        "how many",
        "count",
        "list all",
        "all the",
        "all documents",
        "across",
        "mentioned in all",
    ]

    question_lower = question.lower()

    return any(term in question_lower for term in aggregation_terms)


# --- Session Conversation State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: Client / KB State ---
if LOGO_PATH:
    st.sidebar.image(str(LOGO_PATH), width=68)

st.sidebar.markdown("### Chase Risk & Compliance")
st.sidebar.caption("Governance-Aware Document Intelligence")
st.sidebar.divider()
st.sidebar.header("Client Knowledge Base")

client_name = st.sidebar.text_input("Client Name", value=DEFAULT_CLIENT_NAME)
client_name_clean = client_name.strip()
collection_name = f"client_{client_name_clean}"

registry = load_registry(
    persist_dir=persist_dir,
    client_name=client_name_clean,
)

registry_documents = registry.get("documents", {})

diagnostics_by_document = load_diagnostics(
    persist_dir=persist_dir,
    client_name=client_name_clean,
)

# --- Knowledge Base Summary ---
st.sidebar.subheader("Knowledge Base Summary")

document_count = len(registry_documents)
total_chunk_count = sum(
    record.get("chunk_count", 0)
    for record in registry_documents.values()
)

readiness_scores = [
    record.get("readiness_score")
    for record in registry_documents.values()
    if isinstance(record.get("readiness_score"), (int, float))
]

average_readiness_score = (
    sum(readiness_scores) / len(readiness_scores)
    if readiness_scores
    else None
)

st.sidebar.write(f"**Documents:** {document_count}")
st.sidebar.write(f"**Total Chunks:** {total_chunk_count:,}")

if average_readiness_score is not None:
    st.sidebar.write(f"**Average Readiness:** {average_readiness_score:.1f}/100")
else:
    st.sidebar.write("**Average Readiness:** N/A")

# --- Upload / Indexing Controls ---
st.sidebar.subheader("Document Upload")

uploaded_files = st.sidebar.file_uploader(
    "Upload Documents",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
)

advanced_indexing_controls = st.sidebar.checkbox(
    "Show document processing settings",
    value=False,
)

chunking_strategy = "character"
chunk_size = 1000
chunk_overlap = 150
delimiter = None

if advanced_indexing_controls:
    chunking_strategy = st.sidebar.selectbox(
        "Document segmentation strategy",
        options=["character", "delimiter", "page", "heading"],
        index=0,
    )

    st.sidebar.caption(
        "Use character chunking for general documents. Use delimiter chunking when "
        "documents have clear repeated sections such as 'Section:' or 'Question:'."
    )

    if chunking_strategy == "character":
        chunk_size = st.sidebar.slider(
            "Document segment size",
            min_value=250,
            max_value=4000,
            value=1000,
            step=50,
        )

        chunk_overlap = st.sidebar.slider(
            "Segment overlap",
            min_value=0,
            max_value=1000,
            value=150,
            step=25,
        )

        st.sidebar.caption(
            "Suggested starting points: 800–1200 chars for general documents, "
            "1500–2500 for long policies/manuals, 500–900 for short notes. "
            "Overlap is usually 10–20% of chunk size."
        )

    elif chunking_strategy == "delimiter":
        delimiter = st.sidebar.text_input(
            "Delimiter",
            value="Book Title -",
        )

        st.sidebar.caption(
            "Delimiter chunking keeps repeated sections together. "
            "Examples: 'Question:', 'Section:', 'Policy:', 'Agenda Item:'."
        )

    elif chunking_strategy == "page":
        st.sidebar.caption(
            "Page-based chunking keeps PDF pages together. Best when page boundaries "
            "are meaningful and citations by page are useful."
        )

    elif chunking_strategy == "heading":
        st.sidebar.caption(
            "Heading-aware chunking attempts to preserve sections using detected headings. "
            "Best for policies, procedures, manuals, and structured documents."
        )

else:
    st.sidebar.info(
        "Indexing uses default document processing settings. Enable document processing "
        "settings to customise segmentation strategy, segment size, overlap, or delimiter."
    )

index_btn = st.sidebar.button("Index Document")

replace_existing_index = st.sidebar.checkbox(
    "Replace existing client index before indexing",
    value=False,
)

clear_btn = st.sidebar.button("Clear Client Index")

index_status = st.sidebar.empty()

# --- Retrieval Controls ---
st.sidebar.subheader("Retrieval Settings")

advanced_retrieval_controls = st.sidebar.checkbox(
    "Show search & retrieval settings",
    value=False,
)

selected_retrieval_documents = []

top_k = DEFAULT_TOP_K
min_score = DEFAULT_MIN_SCORE
retrieval_mode = "Auto retrieval"

if advanced_retrieval_controls:
    selected_retrieval_documents = st.sidebar.multiselect(
        "Search scope",
        options=list(registry_documents.keys()),
        default=[],
    )

    st.sidebar.caption("Leave empty to search across all indexed documents.")

    top_k = st.sidebar.slider(
        "Evidence segments to retrieve",
        min_value=3,
        max_value=30,
        value=DEFAULT_TOP_K,
        step=1,
    )

    min_score = st.sidebar.slider(
        "Minimum evidence score",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_MIN_SCORE,
        step=0.01,
    )

    retrieval_mode = st.sidebar.selectbox(
        "Search mode",
        options=[
            "Standard chunk retrieval",
            "Broad retrieval",
            "Auto retrieval",
            "Document-level retrieval",
        ],
        index=2,
    )
else:
    st.sidebar.info(
        "Search and evidence retrieval are handled automatically by the orchestration layer."
    )


# --- Indexed Documents Registry UI ---
st.sidebar.subheader("Indexed Documents")

if registry_documents:
    for file_name, record in registry_documents.items():
        with st.sidebar.expander(file_name):
            st.write(f"**Chunking Strategy:** {record.get('chunking_strategy')}")
            st.write(f"**Chunk Count:** {record.get('chunk_count')}")
            st.write(f"**Indexed At:** {record.get('indexed_at')}")

            readiness_score = record.get("readiness_score")
            readiness_status = record.get("readiness_status")

            if readiness_status == "Good":
                st.success(f"Readiness: {readiness_score}/100 — Good")
            elif readiness_status == "Needs Review":
                st.warning(f"Readiness: {readiness_score}/100 — Needs Review")
            elif readiness_status == "Poor":
                st.error(f"Readiness: {readiness_score}/100 — Poor")
            else:
                st.info("Readiness: Unknown")

            doc_diagnostics = diagnostics_by_document.get(file_name, {})

            if doc_diagnostics:
                st.write("---")
                st.write(
                    "**Recommended Chunking Strategy:**",
                    doc_diagnostics.get("recommended_chunking_strategy", "Unknown"),
                )
                st.write(
                    "**Recommendation Reason:**",
                    doc_diagnostics.get(
                        "recommendation_reason",
                        "No recommendation available.",
                    ),
                )

                warnings = doc_diagnostics.get("warnings", [])

                if warnings:
                    st.warning("Review suggested")
                    for warning in warnings:
                        st.write(f"- {warning}")
                else:
                    st.success("No major ingestion issues detected.")

            if st.button(
                f"Delete Document: {file_name}",
                key=f"delete_document_{file_name}",
            ):
                try:
                    vector_store = VectorStore(
                        persist_dir=persist_dir,
                        collection_name=collection_name,
                    )

                    deleted_count = vector_store.delete_document(file_name)

                    delete_document_diagnostics(
                        persist_dir=persist_dir,
                        client_name=client_name_clean,
                        file_name=file_name,
                    )

                    delete_document_record(
                        persist_dir=persist_dir,
                        client_name=client_name_clean,
                        file_name=file_name,
                    )

                    st.sidebar.success(
                        f"Deleted '{file_name}' and {deleted_count} vector chunk(s)."
                    )

                    st.rerun()

                except Exception as e:
                    st.sidebar.error(f"Failed to delete document: {e}")

else:
    st.sidebar.info("No indexed documents.")

# --- Clear Index Logic ---
if clear_btn:
    try:
        vector_store = VectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )

        vector_store.delete_collection()

        delete_client_diagnostics(
            persist_dir=persist_dir,
            client_name=client_name_clean,
        )

        delete_client_registry(
            persist_dir=persist_dir,
            client_name=client_name_clean,
        )

        st.sidebar.success(f"Cleared index for client '{client_name_clean}'.")
        st.rerun()

    except Exception as e:
        st.sidebar.error(f"Failed to clear index: {e}")

# --- Indexing Logic ---
if index_btn:
    if not client_name_clean:
        index_status.error("Please enter a client name.")
    elif not uploaded_files:
        index_status.error("Please upload one or more documents to index.")
    else:
        tmp_paths = []

        try:
            vector_store = VectorStore(
                persist_dir=persist_dir,
                collection_name=collection_name,
            )

            existing_documents = vector_store.list_documents()

            existing_file_names = {
                doc["file_name"]
                for doc in existing_documents
            }

            if replace_existing_index:
                try:
                    vector_store.delete_collection()

                    delete_client_diagnostics(
                        persist_dir=persist_dir,
                        client_name=client_name_clean,
                    )

                    delete_client_registry(
                        persist_dir=persist_dir,
                        client_name=client_name_clean,
                    )

                    vector_store = VectorStore(
                        persist_dir=persist_dir,
                        collection_name=collection_name,
                    )

                    existing_file_names = set()

                    index_status.info("Existing client index cleared before indexing.")

                except Exception:
                    index_status.info("No existing client index found to clear.")

            all_records = []
            document_records = []
            total_chunks = 0

            for uploaded_file in uploaded_files:
                if uploaded_file.name in existing_file_names:
                    deleted_count = vector_store.delete_document(uploaded_file.name)

                    delete_document_diagnostics(
                        persist_dir=persist_dir,
                        client_name=client_name_clean,
                        file_name=uploaded_file.name,
                    )

                    delete_document_record(
                        persist_dir=persist_dir,
                        client_name=client_name_clean,
                        file_name=uploaded_file.name,
                    )

                    index_status.info(
                        f"Removed existing indexed version of "
                        f"'{uploaded_file.name}' "
                        f"({deleted_count} chunks) before re-indexing."
                    )

                index_status.info(f"Processing document: {uploaded_file.name}")

                file_extension = os.path.splitext(uploaded_file.name)[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=file_extension,
                    mode="wb",
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                    tmp_paths.append(tmp_path)

                doc = ingest_document(tmp_path)
                doc.metadata["file_name"] = uploaded_file.name
                doc.metadata["original_file_name"] = uploaded_file.name

                if chunking_strategy == "character":
                    chunks = chunk_document(
                        doc,
                        chunk_size=chunk_size,
                        overlap=chunk_overlap,
                    )

                elif chunking_strategy == "delimiter":
                    chunks = chunk_document_by_delimiter(
                        doc,
                        delimiter=delimiter,
                    )

                elif chunking_strategy == "page":
                    chunks = chunk_document_by_page(doc)

                elif chunking_strategy == "heading":
                    chunks = chunk_document_by_heading(doc)

                else:
                    raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")

                if not chunks:
                    raise RuntimeError(
                        f"No chunks created from document: {uploaded_file.name}"
                    )

                diagnostics = run_ingestion_diagnostics(doc, chunks)

                save_document_diagnostics(
                    persist_dir=persist_dir,
                    client_name=client_name_clean,
                    file_name=uploaded_file.name,
                    diagnostics=diagnostics,
                )

                with st.expander(
                    f"Document readiness: {diagnostics.file_name}",
                    expanded=False,
                ):
                    st.write("**Readiness status:**", diagnostics.readiness_status)
                    st.write("**Readiness score:**", f"{diagnostics.readiness_score}/100")

                    st.write(
                        "**Recommended chunking strategy:**",
                        diagnostics.recommended_chunking_strategy,
                    )

                    st.write(
                        "**Recommendation reason:**",
                        diagnostics.recommendation_reason,
                    )

                    st.write("**File type:**", diagnostics.file_type)
                    st.write("**File size:**", f"{diagnostics.file_size:,} bytes")
                    st.write(
                        "**Characters extracted:**",
                        f"{diagnostics.character_count:,} characters",
                    )
                    st.write(
                        "**Words extracted:**",
                        f"{diagnostics.word_count:,} words",
                    )

                    if diagnostics.page_count is not None:
                        st.write(
                            "**Pages extracted:**",
                            f"{diagnostics.page_count:,} pages",
                        )

                    st.write(
                        "**Chunks created:**",
                        f"{diagnostics.chunk_count:,} chunks",
                    )
                    st.write(
                        "**Average chunk length:**",
                        f"{diagnostics.average_chunk_length:,.1f} characters per chunk",
                    )

                    if diagnostics.warnings:
                        st.warning("Review suggested:")
                        for warning in diagnostics.warnings:
                            st.write(f"- {warning}")
                    else:
                        st.success("No major ingestion issues detected.")

                total_chunks += len(chunks)

                chunk_texts = [chunk.text for chunk in chunks]

                all_records = []
                document_records = []
                total_chunks = 0
                embedding_client = EmbeddingClient()

                embeddings = embedding_client.embed_texts(chunk_texts)

                if len(embeddings) != len(chunks):
                    raise RuntimeError(
                        f"Number of embeddings does not match number of chunks for {uploaded_file.name}."
                    )

                records = [
                    VectorRecord(chunk=chunk, embedding=embedding)
                    for chunk, embedding in zip(chunks, embeddings)
                ]

                all_records.extend(records)

                document_records.append(
                    {
                        "file_name": uploaded_file.name,
                        "file_type": doc.file_type,
                        "chunk_count": len(chunks),
                        "chunking_strategy": chunking_strategy,
                        "readiness_status": diagnostics.readiness_status,
                        "readiness_score": diagnostics.readiness_score,
                        "indexed_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

            if not all_records:
                raise RuntimeError("No records were created from uploaded documents.")

            vector_store.upsert_records(all_records)

            for record in document_records:
                save_document_record(
                    persist_dir=persist_dir,
                    client_name=client_name_clean,
                    record=record,
                )

            index_status.success(
                f"Successfully indexed {len(uploaded_files)} document(s), "
                f"{total_chunks} chunks, for client '{client_name_clean}'."
            )

            st.rerun()

        except Exception as e:
            index_status.error(f"Indexing failed: {e}")

        finally:
            for tmp_path in tmp_paths:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)


# --- Main: QA Section ---

# --- Conversation Controls ---

if LOGO_PATH:
    hero_col_logo, hero_col_text = st.columns([0.10, 0.90], vertical_alignment="center")

    with hero_col_logo:
        st.image(str(LOGO_PATH), width=84)

    with hero_col_text:
        st.markdown(
            f"""
            <div class="crc-hero">
                <div class="crc-eyebrow">Chase Risk & Compliance</div>
                <div class="crc-title">{APP_TITLE}</div>
                <div class="crc-subtitle">{APP_SUBTITLE}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        f"""
        <div class="crc-hero">
            <div class="crc-eyebrow">Chase Risk & Compliance</div>
            <div class="crc-title">{APP_TITLE}</div>
            <div class="crc-subtitle">{APP_SUBTITLE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

col1, col2 = st.columns([1, 5])

with col1:
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()

st.subheader("Ask a Question")

question = st.text_input(
    "Type your question here:",
    placeholder="Ask a question about the indexed knowledge base...",
)
ask_btn = st.button("Ask Question")
qa_status = st.empty()

if ask_btn:
    st.write("---")

    if not client_name_clean:
        qa_status.error("Please enter a client name in the sidebar before asking a question.")
    elif not question.strip():
        qa_status.error("Please enter a question to ask.")
    else:
        try:
            vector_store = VectorStore(
                persist_dir=persist_dir,
                collection_name=collection_name,
            )

            count = vector_store.count()

            if count == 0:
                qa_status.error(
                    f"No records found in collection '{collection_name}'. Please index a document first."
                )

            else:
                metadata_filter = None

                if len(selected_retrieval_documents) == 1:
                    metadata_filter = {
                        "file_name": selected_retrieval_documents[0]
                    }

                elif len(selected_retrieval_documents) > 1:
                    metadata_filter = {
                        "$or": [
                            {"file_name": doc_name}
                            for doc_name in selected_retrieval_documents
                        ]
                    }

                if retrieval_mode == "Document-level retrieval" and not selected_retrieval_documents:
                    qa_status.error(
                        "Document-level retrieval requires one or more documents to be selected in Search scope."
                    )
                    st.stop()

                with st.spinner("Retrieving..."):
                    retrieval_question = build_contextual_question(
                        question=question,
                        chat_history=st.session_state.chat_history,
                    )

                    conversation_context = "\n".join(
                        [
                            f"Previous question: {turn.get('question', '')}\nPrevious answer: {turn.get('answer', '')}"
                            for turn in st.session_state.chat_history[-3:]
                        ]
                    )

                    effective_top_k = get_effective_top_k(
                        question=retrieval_question,
                        selected_top_k=top_k,
                        retrieval_mode=retrieval_mode,
                        document_count=document_count,
                    )

                    api_headers = {
                        "x-api-key": os.getenv("API_KEY", ""),
                    }

                    api_response = requests.post(
                        f"{API_BASE_URL}/ask",
                        headers=api_headers,
                        json={
                            "question": question,
                            "client_name": client_name_clean,
                            "retrieval_question": retrieval_question,
                            "conversation_context": conversation_context,
                            "top_k": effective_top_k,
                            "min_score": min_score,
                            "retrieval_mode": retrieval_mode,
                            "selected_documents": selected_retrieval_documents,
                            "metadata_filter": metadata_filter,
                            "allow_adaptive_routing": not advanced_retrieval_controls,
                        },
                        timeout=60,
                    )

                    api_response.raise_for_status()
                    response_data = api_response.json()

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": response_data.get("answer"),
                        "answer_status": response_data.get("answer_status"),
                        "retrieval_confidence": response_data.get("retrieval_confidence"),
                        "sources": response_data.get("sources", []),
                        "retrieved_chunks": len(response_data.get("sources", [])),
                    }
                )

                sources = response_data.get("sources", [])

                retrieved_document_names = sorted(
                    {
                        source.get("file_name")
                        for source in sources
                        if source.get("file_name")
                    }
                )

                with st.container(border=True):
                    st.subheader("Answer")
                    st.markdown(response_data.get("answer") or "*No answer generated.*")

                    status_col, confidence_col, grounding_col, chunks_col = st.columns(4)
                    status_col.markdown(
                        f"""
                        <div style="
                            padding: 0.75rem;
                            border: 1px solid rgba(255,255,255,0.12);
                            border-radius: 12px;
                            background-color: rgba(255,255,255,0.03);
                            min-height: 105px;
                        ">
                            <div style="font-size:0.8rem; color:rgba(255,255,255,0.65);">
                                Answer Status
                            </div>
                            <div style="
                                font-size:1.4rem;
                                font-weight:700;
                                line-height:1.15;
                                word-break: break-word;
                                overflow-wrap: anywhere;
                            ">
                                {response_data.get("answer_status", "Unknown")}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    confidence_col.metric("Retrieval Confidence", response_data.get("retrieval_confidence", "Unknown"))
                    grounding_col.metric(
                        "Grounding Check",
                        response_data.get("grounding_check", "Unknown"),
                    )

                    chunks_col.metric("Retrieved Chunks", len(sources))

                    st.markdown("**Documents Used:**")
                    st.write(
                        ", ".join(retrieved_document_names)
                        if retrieved_document_names
                        else "None"
                    )

                    with st.expander("Orchestration details", expanded=False):

                        st.write("**Retrieval Query:**", response_data.get("retrieval_query", "Unknown"))

                        st.write(
                            "**Orchestration Intent:**",
                            response_data.get("orchestration_intent", "Unknown"),
                        )

                        st.write(
                            "**Retrieval Strategy:**",
                            response_data.get("retrieval_strategy", "Unknown"),
                        )

                        st.write(
                            "**Clarification Triggered:**",
                            response_data.get("clarification_triggered", "Unknown"),
                        )

                        st.write(
                            "**Orchestration Reasoning:**",
                            response_data.get("orchestration_reasoning", "Unknown"),
                        )

                        timing = response_data.get("timing", {})

                        st.write(
                            "**Total API Duration:**",
                            f"{timing.get('total_duration_ms', 'Unknown')} ms",
                        )

                        st.write(
                            "**Pipeline Duration:**",
                            f"{timing.get('pipeline_duration_ms', 'Unknown')} ms",
                        )

                        st.write(
                            "**Setup Duration:**",
                            f"{timing.get('setup_duration_ms', 'Unknown')} ms",
                        )

                        stage_timings = timing.get("stage_timings", {})

                        if stage_timings:
                            st.write("**Stage Timings:**")
                            for stage_name, duration_ms in stage_timings.items():
                                st.write(f"- {stage_name}: {duration_ms} ms")

                        st.write("**Requested Retrieval Depth:**", effective_top_k)

                    if selected_retrieval_documents:
                        st.write(
                            "**Retrieval Scope:**",
                            ", ".join(selected_retrieval_documents),
                        )
                    else:
                        st.write("**Retrieval Scope:** All indexed documents")

                    st.write("**UI Retrieval Mode:**", retrieval_mode)

                    if retrieval_mode == "Document-level retrieval" and selected_retrieval_documents:
                        selected_doc_count = len(selected_retrieval_documents)
                        chunks_per_document = max(1, effective_top_k // selected_doc_count)

                        st.write(
                            "**Document-Level Retrieval Detail:**",
                            f"{chunks_per_document} chunk(s) retrieved per selected document.",
                        )

                st.subheader("Sources")

                if sources:
                    for source in sources:
                        label_parts = [source.get("file_name", "Unknown document")]

                        chunk_index = source.get("chunk_index")
                        if chunk_index is not None:
                            label_parts.append(f"chunk {chunk_index}")

                        score = source.get("score")
                        if score is not None:
                            label_parts.append(f"score {score:.3f}")

                        label = " | ".join(label_parts)

                        with st.expander(label):
                            preview = source.get("preview")
                            if preview:
                                st.write(preview)
                            else:
                                st.write("_No preview available._")
                else:
                    st.write("_No sources found._")


        except Exception as e:
            qa_status.error(f"Question answering failed: {e}")

# --- Conversation History ---
if st.session_state.chat_history:
    st.write("---")
    st.subheader("Conversation History")
    st.caption("Previous questions are kept below so the current question area stays focused.")

    for i, turn in enumerate(st.session_state.chat_history, start=1):

        turn_label = (
            "Conversation Start"
            if i == 1
            else f"Conversation Turn {i}"
        )

        with st.expander(turn_label):

            st.markdown("**Question:**")
            st.write(turn["question"])

            if turn.get("retrieval_question"):
                st.markdown("**Interpreted question:**")
                st.write(turn["retrieval_question"])

            st.markdown("**Answer:**")
            st.write(turn["answer"])

            st.markdown(f"**Answer Status:** {turn['answer_status']}")
            st.markdown(
                f"**Retrieval Confidence:** {turn['retrieval_confidence']}"
            )
            
            st.markdown("**Retrieval Details:**")
            st.write("**Retrieval Strategy:**", turn.get("retrieval_mode", "Unknown"))
            st.write("**Retrieved Chunks:**", turn.get("retrieved_chunks", "Unknown"))
            st.write("**Requested Retrieval Depth:**", turn.get("requested_depth", "Unknown"))

            retrieval_scope = turn.get("retrieval_scope")

            if retrieval_scope:
                st.write("**Retrieval Scope:**", ", ".join(retrieval_scope))
            else:
                st.write("**Retrieval Scope:** All indexed documents")

            sources = turn.get("sources", [])

            if sources:
                for source in sources:
                    label_parts = [source.get("file_name", "Unknown document")]

                    chunk_index = source.get("chunk_index")
                    if chunk_index is not None:
                        label_parts.append(f"chunk {chunk_index}")

                    score = source.get("score")
                    if score is not None:
                        label_parts.append(f"score {score:.3f}")

                    label = " | ".join(label_parts)

                    with st.expander(label):
                        preview = source.get("preview")
                        if preview:
                            st.write(preview)
                        else:
                            st.write("_No preview available._")
            else:
                st.write("_No sources found._")

st.markdown(
    """
---
<div class="crc-footer">Chase Risk & Compliance &nbsp; | &nbsp; Governance-Aware Document Intelligence Copilot &nbsp; | &nbsp; OpenAI + Streamlit</div>
""",
    unsafe_allow_html=True,
)