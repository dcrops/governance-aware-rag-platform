from pydantic import BaseModel
from typing import List


class RetrievalLog(BaseModel):
    """
    Represents telemetry data for a single RAG pipeline execution,
    including the user's question, rewritten retrieval query,
    retrieved evidence, answer metadata, and timestamp.
    """

    question: str

    original_query: str
    retrieval_query: str

    retrieved_chunk_ids: List[str]
    scores: List[float]

    answer: str

    answer_status: str | None = None
    retrieval_confidence: str | None = None

    requested_retrieval_depth: int | None = None

    documents_used: List[str] = []

    timestamp: str

    grounding_check: str | None = None  #   PASS, FAIL, SKIPPED

    orchestration_intent: str | None = None
    retrieval_strategy: str | None = None
    orchestration_reasoning: str | None = None
    clarification_triggered: bool = False

    stage_timings: dict | None = None