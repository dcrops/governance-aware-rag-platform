import os
import time
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException
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

APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
SERVICE_NAME = "crc-rag-api"
API_KEY = os.getenv("API_KEY")


def validate_api_key(x_api_key: str | None = Header(None)):
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
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
def root():
    return {
        "status": "running",
        "service": SERVICE_NAME,
        "version": APP_VERSION,
    }


@app.post("/ask")
def ask_question(
    request: AskRequest,
    _: None = Depends(validate_api_key),
):
    try:
        request_started_at = time.perf_counter()

        client_name = request.client_name or os.getenv(
            "RAG_DEFAULT_CLIENT_NAME",
            DEFAULT_CLIENT_NAME,
        )

        collection_name = f"client_{client_name}"

        setup_started_at = time.perf_counter()

        vector_store = VectorStore(
            persist_dir=PERSIST_DIR,
            collection_name=collection_name,
        )

        if vector_store.count() == 0:
            total_duration_ms = round(
                (time.perf_counter() - request_started_at) * 1000,
                2,
            )

            return {
                "answer": None,
                "answer_status": "NO_INDEX_FOUND",
                "message": f"No indexed documents found for client '{client_name}'.",
                "retrieval_confidence": "NONE",
                "grounding_check": "SKIPPED",
                "sources": [],
                "timing": {
                    "total_duration_ms": total_duration_ms,
                    "setup_duration_ms": None,
                    "pipeline_duration_ms": None,
                    "stage_timings": {},
                },
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

        setup_duration_ms = round(
            (time.perf_counter() - setup_started_at) * 1000,
            2,
        )

        metadata_filter = request.metadata_filter

        if request.selected_documents:
            metadata_filter = None

        pipeline_started_at = time.perf_counter()

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

        pipeline_duration_ms = round((time.perf_counter() - pipeline_started_at) * 1000, 2)
        total_duration_ms = round((time.perf_counter() - request_started_at) * 1000, 2)

        total_duration_ms = round(
            (time.perf_counter() - request_started_at) * 1000,
            2,
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
            "timing": {
                "total_duration_ms": total_duration_ms,
                "setup_duration_ms": setup_duration_ms,
                "pipeline_duration_ms": pipeline_duration_ms,
                "stage_timings": response.log.stage_timings if response.log else {},
            },
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

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline failure: {str(e)}",
        )


@app.get("/health")
def healthcheck():
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "version": APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/version")
def version():
    return {
        "service": SERVICE_NAME,
        "version": APP_VERSION,
    }