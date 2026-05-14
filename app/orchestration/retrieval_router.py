from pydantic import BaseModel

from app.orchestration.orchestration_models import OrchestrationDecision


class RetrievalRoutingDecision(BaseModel):
    retrieval_strategy: str
    effective_top_k: int
    reasoning: str


class RetrievalRouter:
    """
    Converts an orchestration decision into concrete retrieval settings.

    This keeps retrieval adaptation explainable and separate from the main
    RAG pipeline orchestration flow.
    """

    def route(
        self,
        orchestration_decision: OrchestrationDecision,
        requested_top_k: int,
    ) -> RetrievalRoutingDecision:
        if not isinstance(requested_top_k, int) or requested_top_k <= 0:
            raise ValueError("requested_top_k must be a positive integer.")

        if orchestration_decision.clarification_required:
            return RetrievalRoutingDecision(
                retrieval_strategy="none",
                effective_top_k=0,
                reasoning="Clarification is required, so retrieval is skipped.",
            )

        if orchestration_decision.retrieval_strategy == "broad":
            effective_top_k = max(requested_top_k, 10)

            return RetrievalRoutingDecision(
                retrieval_strategy="document_balanced",
                effective_top_k=effective_top_k,
                reasoning=(
                    "Aggregation-style query detected, so document-balanced retrieval was selected with expanded retrieval depth."
                ),
            )

        return RetrievalRoutingDecision(
            retrieval_strategy=orchestration_decision.retrieval_strategy,
            effective_top_k=requested_top_k,
            reasoning="Standard retrieval depth retained.",
        )