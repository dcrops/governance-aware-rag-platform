from abc import ABC, abstractmethod

from app.models.retrieval_result import RetrievalResult
from app.retrieval.retriever import Retriever


class BaseRetrievalStrategy(ABC):
    """
    Base class for retrieval execution strategies.

    Strategies decide HOW retrieval should run after orchestration has decided
    WHAT kind of retrieval behaviour is needed.
    """

    @abstractmethod
    def retrieve(
        self,
        retriever: Retriever,
        query: str,
        top_k: int,
        min_score: float | None = None,
        metadata_filter: dict | None = None,
        selected_documents: list[str] | None = None,
    ) -> RetrievalResult:
        raise NotImplementedError