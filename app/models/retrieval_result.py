from pydantic import BaseModel, Field
from typing import List

from app.vector_store.vector_store import SearchResult

class RetrievalResult(BaseModel):
    """
    Represents the result of the retrieval phase in a RAG system.

    Contains the original user query, the query actually used for retrieval
    (which may have been rewritten), and the list of retrieved search results.
    """
    search_results: List[SearchResult] = Field(
        ...,
        description="List of search results returned from the retrieval layer."
    )
    original_query: str = Field(
        ...,
        description="The original input query from the user."
    )
    retrieval_query: str = Field(
        ...,
        description="The query string used for retrieval (may differ from original if rewritten)."
    )