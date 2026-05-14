from app.orchestration.orchestration_models import (
    OrchestrationDecision,
    RetrievalIntent,
)


class IntentClassifier:
    """
    Lightweight deterministic intent classifier for conversational RAG orchestration.

    This intentionally avoids LLM-based routing for MVP V1 so orchestration decisions
    remain predictable, explainable, and easy to evaluate.
    """

    def classify(
        self,
        question: str,
        conversation_context: str | None = None,
    ) -> OrchestrationDecision:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        normalized_question = " ".join(question.lower().split())

        if self._is_ambiguous_follow_up(normalized_question, conversation_context):
            return OrchestrationDecision(
                intent=RetrievalIntent.CLARIFICATION,
                answer_mode="clarification",
                clarification_required=True,
                retrieval_strategy="none",
                reasoning="Question appears to be an ambiguous conversational follow-up.",
            )

        if self._is_comparison_question(normalized_question):
            return OrchestrationDecision(
                intent=RetrievalIntent.COMPARISON,
                answer_mode="comparison",
                retrieval_strategy="comparison",
                reasoning="Question contains comparison language.",
            )

        if self._is_aggregation_question(normalized_question):
            return OrchestrationDecision(
                intent=RetrievalIntent.AGGREGATION,
                answer_mode="aggregation",
                retrieval_strategy="broad",
                reasoning="Question asks for broad, count, list, or cross-document information.",
            )

        return OrchestrationDecision(
            intent=RetrievalIntent.STANDARD,
            answer_mode="standard",
            retrieval_strategy="standard",
            reasoning="No comparison, aggregation, or clarification trigger detected.",
        )

    def _is_aggregation_question(self, question: str) -> bool:
        aggregation_terms = [
            "how many",
            "count",
            "list all",
            "all the",
            "all documents",
            "across",
            "mentioned in all",
            "summarise all",
            "summarize all",
        ]

        return any(term in question for term in aggregation_terms)

    def _is_comparison_question(self, question: str) -> bool:
        comparison_terms = [
            "difference between",
            "compare",
            "versus",
            " vs ",
            "how does",
            "differ",
        ]

        return any(term in question for term in comparison_terms)

    def _is_ambiguous_follow_up(
        self,
        question: str,
        conversation_context: str | None,
    ) -> bool:
        ambiguous_terms = [
            "what about",
            "how about",
            "and payroll",
            "and hr",
            "and finance",
        ]

        if not conversation_context:
            return False

        return any(term in question for term in ambiguous_terms)