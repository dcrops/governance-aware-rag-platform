from pydantic import BaseModel
from typing import Any, Optional


class Citation(BaseModel):
    """
    Represents the attribution metadata for a retrieved evidence chunk referenced
    in a generated answer, including provenance, score, and relevant metadata.
    """
    id: str
    file_name: str | None = None
    chunk_index: int | None = None
    text_preview: str | None = None
    score: float
    metadata: dict[str, Any]