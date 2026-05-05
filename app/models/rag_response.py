from pydantic import BaseModel
from app.models.citation import Citation
from app.models.retrieval_log import RetrievalLog
from app.models.retrieval_result import RetrievalResult


class RAGResponse(BaseModel):
    """
    Represents the final response from a Retrieval-Augmented Generation (RAG) pipeline,
    including the generated answer and metadata about source contexts used for grounding.
    """
    answer: str
    sources: list[Citation]
    retrieval_result: RetrievalResult | None = None
    log: RetrievalLog | None = None
    retrieval_confidence: str | None = None
    answer_status: str | None = None