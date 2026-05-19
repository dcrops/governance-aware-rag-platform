import os
from fastapi import FastAPI
from pydantic import BaseModel

from app.config import PERSIST_DIR, DEFAULT_CLIENT_NAME, DEFAULT_TOP_K, DEFAULT_MIN_SCORE
from app.embeddings.embeddings import EmbeddingClient
from app.vector_store.vector_store import VectorStore
from app.query_processing.query_rewriter import QueryRewriter
from app.retrieval.retriever import Retriever
from app.generation.answer_generator import AnswerGenerator
from app.orchestration.rag_pipeline import RAGPipeline
from app.reranking.simple_reranker import SimpleReranker

app = FastAPI(
    title="CRC Governance-Aware RAG API",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str
    client_name: str | None = None
    retrieval_question: str | None = None
    conversation_context: str | None = None
    top_k: int = DEFAULT_TOP_K
    min_score: float | None = DEFAULT_MIN_SCORE
    retrieval_mode: str = "Auto retrieval"
    selected_documents: list[str] | None = None
    metadata_filter: dict | None = None
    allow_adaptive_routing: bool = True


@app.get("/")
def healthcheck():

    return {
        "status": "running",
        "service": "crc-rag-api",
    }


@app.post("/ask")
def ask_question(request: AskRequest):
    client_name = request.client_name or os.getenv(
        "RAG_DEFAULT_CLIENT_NAME",
        DEFAULT_CLIENT_NAME,
    )

    collection_name = f"client_{client_name}"

    vector_store = VectorStore(
        persist_dir=PERSIST_DIR,
        collection_name=collection_name,
    )

    if vector_store.count() == 0:
        return {
            "answer": None,
            "answer_status": "NO_INDEX_FOUND",
            "message": f"No indexed documents found for client '{client_name}'.",
            "sources": [],
        }

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

    metadata_filter = request.metadata_filter

    if request.selected_documents:
        metadata_filter = None

    response = pipeline.answer_question(
        question=request.question,
        retrieval_question=request.retrieval_question,
        top_k=request.top_k,
        min_score=request.min_score,
        metadata_filter=metadata_filter,
        retrieval_mode=request.retrieval_mode,
        selected_documents=request.selected_documents or [],
        answer_mode=None,
        conversation_context=request.conversation_context,
        allow_adaptive_routing=request.allow_adaptive_routing,
    )

    return {
        "question": request.question,
        "retrieval_query": response.log.retrieval_query if response.log else None,
        "answer": response.answer,
        "answer_status": response.answer_status,
        "retrieval_confidence": response.retrieval_confidence,
        "grounding_check": response.log.grounding_check if response.log else None,
        "orchestration_intent": response.log.orchestration_intent if response.log else None,
        "retrieval_strategy": response.log.retrieval_strategy if response.log else None,
        "orchestration_reasoning": response.log.orchestration_reasoning if response.log else None,
        "clarification_triggered": response.log.clarification_triggered if response.log else None,
        "sources": [
            {
                "file_name": source.file_name,
                "chunk_index": source.chunk_index,
                "score": source.score,
                "preview": source.text_preview,
            }
            for source in response.sources
        ],
    }