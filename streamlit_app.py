import os
import tempfile

import streamlit as st

from app.ingestion.ingest import ingest_document
from app.chunking.chunker import chunk_document, chunk_document_by_delimiter, chunk_document_by_page
from app.embeddings.embeddings import EmbeddingClient
from app.models.vector_record import VectorRecord
from app.vector_store.vector_store import VectorStore
from app.query_processing.query_rewriter import QueryRewriter
from app.retrieval.retriever import Retriever
from app.generation.answer_generator import AnswerGenerator
from app.orchestration.rag_pipeline import RAGPipeline


st.set_page_config(page_title="Client RAG UI", layout="wide")

# --- Sidebar: Indexing Section ---
st.sidebar.header("Client Document Indexing")

client_name = st.sidebar.text_input("Client Name", value="demo_client")
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
)

chunking_strategy = st.sidebar.selectbox(
    "Chunking strategy",
    options=["character", "delimiter", "page"],
    index=0,
)

st.sidebar.caption(
    "Chunking guidance: use character chunking for general documents. "
    "Use delimiter chunking when the document has clear repeated sections, "
    "such as 'Book Title -', 'Section:', or 'Agenda Item:'."
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
        "Suggested starting points: "
        "800–1200 chars for general documents, "
        "1500–2500 for long policies/manuals, "
        "500–900 for short notes or dense content. "
        "Overlap is usually 10–20% of chunk size."
    )

elif chunking_strategy == "delimiter":
    delimiter = st.sidebar.text_input(
        "Delimiter",
        value="Book Title -",
    )

    st.sidebar.caption(
        "Delimiter chunking keeps each repeated section together. "
        "Example delimiters: 'Book Title -', 'Section:', 'Policy:', 'Agenda Item:'."
    )

index_btn = st.sidebar.button("Index Document")
clear_btn = st.sidebar.button("Clear Client Index")

replace_existing_index = st.sidebar.checkbox(
    "Replace existing client index before indexing",
    value=False,
)

top_k = st.sidebar.slider(
    "Number of chunks to retrieve",
    min_value=3,
    max_value=30,
    value=5,
    step=1,
)

min_score = st.sidebar.slider(
    "Minimum retrieval score",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.01,
)

index_status = st.sidebar.empty()

# --- Clear Index Logic ---
if clear_btn:
    try:
        persist_dir = "data/index"
        collection_name = f"client_{client_name.strip()}"

        vector_store = VectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )

        vector_store.delete_collection()
        st.sidebar.success(f"Cleared index for client '{client_name.strip()}'.")
    except Exception as e:
        st.sidebar.error(f"Failed to clear index: {e}")

with st.sidebar.expander("Indexed Documents", expanded=False):
    try:
        persist_dir = "data/index"
        collection_name = f"client_{client_name.strip()}"

        vector_store = VectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )

        documents = vector_store.list_documents()

        if not documents:
            st.info("No indexed documents found.")
        else:
            for doc in documents:
                st.markdown(
                    f"""
**{doc['file_name']}**
- Type: `{doc['file_type']}`
- Chunks: `{doc['chunk_count']}`
"""
                )

            document_names = [doc["file_name"] for doc in documents]

            selected_document = st.selectbox(
                "Select document to delete",
                options=document_names,
            )

            if st.button("Delete Selected Document"):
                deleted_count = vector_store.delete_document(selected_document)
                st.success(
                    f"Deleted {deleted_count} chunks for '{selected_document}'."
                )
                st.rerun()

    except Exception as e:
        st.error(f"Failed to load indexed documents: {e}")

# --- Indexing Logic ---
if index_btn:
    if not client_name.strip():
        index_status.error("Please enter a client name.")
    elif not uploaded_files:
        index_status.error("Please upload one or more documents to index.")
    else:
        tmp_paths = []

        try:
            persist_dir = "data/index"
            collection_name = f"client_{client_name.strip()}"

            vector_store = VectorStore(
                persist_dir=persist_dir,
                collection_name=collection_name,
            )

            if replace_existing_index:
                try:
                    vector_store.delete_collection()
                    vector_store = VectorStore(
                        persist_dir=persist_dir,
                        collection_name=collection_name,
                    )
                    index_status.info("Existing client index cleared before indexing.")
                except Exception:
                    index_status.info("No existing client index found to clear.")

            embedding_client = EmbeddingClient()

            all_records = []
            total_chunks = 0

            for uploaded_file in uploaded_files:
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

                else:
                    raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")

                if not chunks:
                    raise RuntimeError(f"No chunks created from document: {uploaded_file.name}")

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

            if not all_records:
                raise RuntimeError("No records were created from uploaded documents.")

            vector_store.upsert_records(all_records)

            index_status.success(
                f"Successfully indexed {len(uploaded_files)} document(s), "
                f"{total_chunks} chunks, for client '{client_name.strip()}'."
            )

        except Exception as e:
            index_status.error(f"Indexing failed: {e}")

        finally:
            for tmp_path in tmp_paths:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

# --- Main: QA Section ---
st.header("Ask a Question")

question = st.text_input("Type your question here:")
ask_btn = st.button("Ask Question")
qa_status = st.empty()

if ask_btn:
    st.write("---")
    if not client_name.strip():
        qa_status.error("Please enter a client name in the sidebar before asking a question.")
    elif not question.strip():
        qa_status.error("Please enter a question to ask.")
    else:
        try:
            persist_dir = "data/index"
            collection_name = f"client_{client_name.strip()}"

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

                retriever = Retriever(
                    embedding_client=embedding_client,
                    vector_store=vector_store,
                    query_rewriter=query_rewriter,
                )

                answer_generator = AnswerGenerator()

                pipeline = RAGPipeline(
                    retriever=retriever,
                    answer_generator=answer_generator,
                )

                with st.spinner("Retrieving..."):
                    response = pipeline.answer_question(
                        question,
                        top_k=top_k,
                        min_score=min_score,
                    )

                st.subheader("Answer")
                st.markdown(response.answer if response.answer else "*No answer generated.*")

                st.write("**Retrieval Confidence**:", response.retrieval_confidence)
                st.write("**Answer Status**:", response.answer_status)

                st.write("**Sources:**")

                if response.sources:
                    for source in response.sources:

                        label = (
                            f"{source.file_name} | "
                            f"chunk {source.chunk_index} | "
                            f"score {source.score:.3f}"
                        )

                        # Prefer page number if available
                        if getattr(source, "metadata", None):
                            page_number = source.metadata.get("page_number")

                            if page_number is not None:
                                label = (
                                    f"{source.file_name} | "
                                    f"page {page_number} | "
                                    f"score {source.score:.3f}"
                                )

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

st.markdown(
    """
---
<small>Lightweight MVP RAG UI &nbsp; | &nbsp; [OpenAI + Streamlit Demo]</small>
""",
    unsafe_allow_html=True,
)