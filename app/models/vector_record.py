from pydantic import BaseModel
from app.models.chunk import Chunk

class VectorRecord(BaseModel):
    """
    A model representing a document chunk and its associated embedding vector,
    suitable for storage in a vector database within a RAG system.
    """
    chunk: Chunk
    embedding: list[float]