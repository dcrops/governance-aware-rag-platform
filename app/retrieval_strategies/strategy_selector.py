from app.retrieval_strategies.standard_retrieval_strategy import StandardRetrievalStrategy
from app.retrieval_strategies.document_balanced_retrieval_strategy import (
    DocumentBalancedRetrievalStrategy,
)
from app.retrieval_strategies.retrieval_strategy import BaseRetrievalStrategy


class RetrievalStrategySelector:
    """
    Selects the concrete retrieval execution strategy from the routed strategy name.
    """

    def __init__(self):
        self.standard_strategy = StandardRetrievalStrategy()
        self.document_balanced_strategy = DocumentBalancedRetrievalStrategy()

    def select(self, retrieval_strategy: str) -> BaseRetrievalStrategy:
        if retrieval_strategy == "document_balanced":
            return self.document_balanced_strategy

        return self.standard_strategy