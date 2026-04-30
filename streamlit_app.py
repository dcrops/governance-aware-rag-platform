import streamlit as st
import os
import tempfile
from app.ingestion.ingest import ingest_document
from app.chunking.chunker import chunk_document
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
uploaded_file = st.sidebar.file_uploader("Upload .txt File", type=["txt"])
index_btn = st.sidebar.button("Index Document")

top_k = st.sidebar.slider(
    "Number of chunks to retrieve",
    min_value=3,
    max_value=30,
    value=5,
    step=1,
)

# For feedback messages
index_status = st.sidebar.empty()

# --- Indexing Logic ---
if index_btn:
    if not client_name.strip():
        index_status.error("Please enter a client name.")
    elif not uploaded_file:
        index_status.error("Please upload a .txt file to index.")
    else:
        # Save uploaded file to a temporary location
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            index_status.info("Ingesting document...")

            # Ingest document
            doc = ingest_document(tmp_path)
            index_status.info("Chunking document...")
            chunks = chunk_document(doc)
            if not chunks:
                index_status.error("No chunks created from document (empty file?). Aborting indexing.")
                os.remove(tmp_path)
            else:
                index_status.info(f"Chunked into {len(chunks)} segments.")

                # Embedding
                embedding_client = EmbeddingClient()
                chunk_texts = [chunk.text for chunk in chunks]
                embeddings = embedding_client.embed_texts(chunk_texts)
                if len(embeddings) != len(chunks):
                    index_status.error("Number of embeddings does not match number of chunks.")
                    os.remove(tmp_path)
                else:
                    index_status.info(f"Embedding completed for {len(embeddings)} chunks.")

                    # Build VectorRecords
                    records = [
                        VectorRecord(chunk=chunk, embedding=embedding)
                        for chunk, embedding in zip(chunks, embeddings)
                    ]

                    # Initialize VectorStore and upsert
                    persist_dir = "data/index"
                    collection_name = f"client_{client_name.strip()}"
                    vector_store = VectorStore(
                        persist_dir=persist_dir, collection_name=collection_name
                    )
                    vector_store.upsert_records(records)
                    index_status.success(f"Successfully indexed {len(records)} chunks for client '{client_name.strip()}'.")
            # Clean up temp file
            os.remove(tmp_path)
        except Exception as e:
            index_status.error(f"Indexing failed: {e}")

# --- Main: QA Section ---
st.header("Ask a Question")

question = st.text_input("Type your question here:")
ask_btn = st.button("Ask Question")
qa_status = st.empty()

if ask_btn:
    st.write("---")
    if not client_name.strip():
        qa_status.error("Please enter a client name (in sidebar) before asking a question.")
    elif not question.strip():
        qa_status.error("Please enter a question to ask.")
    else:
        try:
            # --- Initialize RAG Pipeline ---
            persist_dir = "data/index"
            collection_name = f"client_{client_name.strip()}"

            # Check if collection exists and is non-empty
            vector_store = VectorStore(
                persist_dir=persist_dir, collection_name=collection_name
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

                # --- Get Answer ---
                with st.spinner("Retrieving..."):
                    response = pipeline.answer_question(
                        question,
                        top_k=top_k,
                        min_score=0.35,
                        # Optionally, support metadata_filter here as needed
                    )

                # --- Display Response ---
                st.subheader("Answer")
                st.markdown(response.answer if response.answer else "*No answer generated.*")

                st.write("**Retrieval Confidence**:", response.retrieval_confidence)
                st.write("**Answer Status**:", response.answer_status)

                st.write("**Sources:**")
                if response.sources:
                    for source in response.sources:
                        st.markdown(
                            f"- `{source.file_name}` | chunk {source.chunk_index} | score {source.score:.3f}"
                        )

                        if source.text_preview:
                            st.caption(source.text_preview)
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

# --- Page Footer ---
st.markdown("""
---
<small>Lightweight MVP RAG UI &nbsp; | &nbsp; [OpenAI + Streamlit Demo]</small>
""", unsafe_allow_html=True)