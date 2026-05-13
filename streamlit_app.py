import os
import tempfile
from datetime import datetime, timezone
from openai import OpenAI

import streamlit as st

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
from app.query_processing.query_rewriter import QueryRewriter
from app.retrieval.retriever import Retriever
from app.generation.answer_generator import AnswerGenerator
from app.orchestration.rag_pipeline import RAGPipeline
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
from app.telemetry.telemetry_logger import TelemetryLogger
from app.reranking.simple_reranker import SimpleReranker


from app.config import PERSIST_DIR, DEFAULT_CLIENT_NAME, DEFAULT_TOP_K, DEFAULT_MIN_SCORE

st.set_page_config(page_title="Client RAG UI", layout="wide")

persist_dir = PERSIST_DIR

rewrite_client = OpenAI()

domain_profile = "tgbc"

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

telemetry_logger = TelemetryLogger()

# --- Sidebar: Client / KB State ---
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

chunking_strategy = st.sidebar.selectbox(
    "Chunking strategy",
    options=["character", "delimiter", "page", "heading"],
    index=0,
)

st.sidebar.caption(
    "Use character chunking for general documents. Use delimiter chunking when "
    "documents have clear repeated sections such as 'Section:' or 'Question:'."
)

chunk_size = None
chunk_overlap = None
delimiter = None

if chunking_strategy == "character":
    chunk_size = st.sidebar.slider(
        "Chunk size",
        min_value=250,
        max_value=4000,
        value=1000,
        step=50,
    )

    chunk_overlap = st.sidebar.slider(
        "Chunk overlap",
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

elif chunking_strategy == "heading":
    st.sidebar.caption(
        "Heading-aware chunking attempts to preserve sections using detected headings. "
        "Best for policies, procedures, manuals, and structured documents."
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

selected_retrieval_documents = st.sidebar.multiselect(
    "Search scope",
    options=list(registry_documents.keys()),
    default=[],
)

st.sidebar.caption("Leave empty to search across all indexed documents.")

top_k = st.sidebar.slider(
    "Number of chunks to retrieve",
    min_value=3,
    max_value=30,
    value=DEFAULT_TOP_K,
    step=1,
)

min_score = st.sidebar.slider(
    "Minimum retrieval score",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_MIN_SCORE,
    step=0.01,
)

retrieval_mode = st.sidebar.selectbox(
    "Retrieval mode",
    options=[
        "Standard chunk retrieval",
        "Broad retrieval",
        "Auto retrieval",
        "Document-level retrieval",
    ],
    index=0,
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

            embedding_client = EmbeddingClient()

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
col1, col2 = st.columns([1, 5])

with col1:
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()

st.header("Ask a Question")

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
                embedding_client = EmbeddingClient()
                query_rewriter = QueryRewriter()

                reranker = SimpleReranker()

                retriever = Retriever(
                    embedding_client=embedding_client,
                    vector_store=vector_store,
                    query_rewriter=query_rewriter,
                    reranker=reranker,
                )

                answer_generator = AnswerGenerator()

                pipeline = RAGPipeline(
                    retriever=retriever,
                    answer_generator=answer_generator,
                )

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

                    effective_top_k = get_effective_top_k(
                        question=retrieval_question,
                        selected_top_k=top_k,
                        retrieval_mode=retrieval_mode,
                        document_count=document_count,
                    )

                    answer_mode = (
                        "aggregation"
                        if is_aggregation_question(retrieval_question)
                        else "standard"
                    )

                    response = pipeline.answer_question(
                        question=question,
                        retrieval_question=retrieval_question,
                        top_k=effective_top_k,
                        min_score=min_score,
                        metadata_filter=metadata_filter,
                        answer_mode=answer_mode,
                        domain_profile=domain_profile,
                        retrieval_mode=retrieval_mode,
                        selected_documents=selected_retrieval_documents,
                    )

                    if response.log:
                        telemetry_logger.log_retrieval(response.log)

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "retrieval_question": retrieval_question,
                        "answer": response.answer,
                        "answer_status": response.answer_status,
                        "retrieval_confidence": response.retrieval_confidence,

                        "sources": response.sources,
                        "retrieval_mode": retrieval_mode,
                        "retrieval_scope": selected_retrieval_documents,
                        "retrieved_chunks": len(response.sources),
                        "requested_depth": effective_top_k,

                        "telemetry": (
                            {
                                "original_query": response.log.original_query,
                                "retrieval_query": response.log.retrieval_query,
                                "scores": response.log.scores,
                            }
                            if response.log
                            else None
                        ),
                    }
                )

                st.write("**Grounding Check:**", response.log.grounding_check)

                st.subheader("Answer")
                st.markdown(response.answer if response.answer else "*No answer generated.*")

                st.write("**Answer Status**:", response.answer_status)
                st.write("**Retrieval Confidence**:", response.retrieval_confidence)

                retrieved_document_names = sorted(
                    {
                        source.file_name
                        for source in response.sources
                        if source.file_name
                    }
                )

                st.write(
                    "**Documents Used:**",
                    ", ".join(retrieved_document_names)
                    if retrieved_document_names
                    else "None",
                )

                st.metric("Retrieved Chunks", len(response.sources))
                st.write("**Requested Retrieval Depth:**", effective_top_k)

                if selected_retrieval_documents:
                    st.write(
                        "**Retrieval Scope:**",
                        ", ".join(selected_retrieval_documents),
                    )
                else:
                    st.write("**Retrieval Scope:** All indexed documents")

                st.write("**Retrieval Strategy:**", retrieval_mode)

                if retrieval_mode == "Document-level retrieval":
                    selected_doc_count = len(selected_retrieval_documents)
                    chunks_per_document = max(1, effective_top_k // selected_doc_count)

                    st.write(
                        "**Document-Level Retrieval Detail:**",
                        f"{chunks_per_document} chunk(s) retrieved per selected document.",
                    )

                st.write("**Sources:**")

                if response.sources:
                    for source in response.sources:
                        label_parts = [source.file_name]

                        if getattr(source, "metadata", None):
                            metadata = source.metadata

                            section_title = metadata.get("section_title")
                            page_number = metadata.get("page_number")
                            source_chunking_strategy = metadata.get("chunking_strategy")

                            if section_title:
                                if len(section_title) > 50:
                                    section_title = section_title[:47] + "..."

                                label_parts.append(f"section: {section_title}")

                            elif page_number is not None:
                                label_parts.append(f"page {page_number}")

                            else:
                                label_parts.append(f"chunk {source.chunk_index}")

                            if source_chunking_strategy:
                                label_parts.append(f"strategy: {source_chunking_strategy}")

                        else:
                            label_parts.append(f"chunk {source.chunk_index}")

                        score_parts = []

                        if source.vector_score is not None:
                            score_parts.append(f"vector {source.vector_score:.3f}")

                        if source.rerank_bonus is not None:
                            score_parts.append(f"bonus {source.rerank_bonus:.3f}")

                        if source.final_score is not None:
                            score_parts.append(f"final {source.final_score:.3f}")

                        elif source.score is not None:
                            score_parts.append(f"score {source.score:.3f}")

                        if score_parts:
                            label_parts.append(" | ".join(score_parts))

                        label = " | ".join(label_parts)

                        with st.expander(label):
                            if source.text_preview:
                                st.write(source.text_preview)
                            else:
                                st.write("_No preview available._")

                else:
                    st.write("_No sources found._")

                st.write("**Telemetry:**")

                if response.log:
                    st.code(
                        f"Original Query: {response.log.original_query}\n"
                        f"Retrieval Query: {response.log.retrieval_query}\n"
                        f"Scores: {', '.join([str(round(s, 3)) for s in getattr(response.log, 'scores', [])])}"
                    )

        except Exception as e:
            qa_status.error(f"Question answering failed: {e}")

# --- Conversation History ---
if st.session_state.chat_history:
    st.write("---")
    st.header("Conversation History")

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
                st.markdown("**Sources:**")

                for source in sources:
                    label_parts = [source.file_name]

                    if source.metadata:
                        section_title = source.metadata.get("section_title")
                        page_number = source.metadata.get("page_number")
                        source_chunking_strategy = source.metadata.get("chunking_strategy")

                        if section_title:
                            if len(section_title) > 50:
                                section_title = section_title[:47] + "..."
                            label_parts.append(f"section: {section_title}")

                        elif page_number is not None:
                            label_parts.append(f"page {page_number}")

                        else:
                            label_parts.append(f"chunk {source.chunk_index}")

                        if source_chunking_strategy:
                            label_parts.append(f"strategy: {source_chunking_strategy}")

                    score_parts = []

                    if source.vector_score is not None:
                        score_parts.append(f"vector {source.vector_score:.3f}")

                    if source.rerank_bonus is not None:
                        score_parts.append(f"bonus {source.rerank_bonus:.3f}")

                    if source.final_score is not None:
                        score_parts.append(f"final {source.final_score:.3f}")

                    elif source.score is not None:
                        score_parts.append(f"score {source.score:.3f}")

                    if score_parts:
                        label_parts.append(" | ".join(score_parts))

                    label = " | ".join(label_parts)

                    with st.expander(label):
                        if source.text_preview:
                            st.write(source.text_preview)
                        else:
                            st.write("_No preview available._")

st.markdown(
    """
---
<small>Lightweight MVP RAG UI &nbsp; | &nbsp; [OpenAI + Streamlit Demo]</small>
""",
    unsafe_allow_html=True,
)