from pydantic import BaseModel
from typing import List

class RetrievalLog(BaseModel):
    """
    Represents telemetry data for a single RAG pipeline execution,
    including the user's question, retrieved evidence, answer, and timestamp.
    """
    question: str
    retrieved_chunk_ids: List[str]
    scores: List[float]
    answer: str
    timestamp: str