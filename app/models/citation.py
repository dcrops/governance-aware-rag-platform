from pydantic import BaseModel
from typing import Any


class Citation(BaseModel):
    """
    Represents attribution metadata for a retrieved evidence chunk referenced
    in a generated answer, including provenance, score, and relevant metadata.
    """

    id: str
    file_name: str | None = None
    chunk_index: int | None = None
    text_preview: str | None = None

    score: float
    vector_score: float | None = None
    rerank_bonus: float | None = None
    final_score: float | None = None

    metadata: dict[str, Any]