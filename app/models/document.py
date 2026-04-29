from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Represents a successfully ingested document with extracted text
    and associated metadata.
    """
    doc_id: str
    source_path: str
    file_type: str
    raw_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
