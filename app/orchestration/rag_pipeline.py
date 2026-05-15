from datetime import datetime, timezone

from app.retrieval.retriever import Retriever
from app.generation.answer_generator import AnswerGenerator
from app.models.rag_response import RAGResponse
from app.models.citation import Citation
from app.models.retrieval_log import RetrievalLog
from app.orchestration.intent_classifier import IntentClassifier
from app.orchestration.retrieval_router import RetrievalRouter
from app.retrieval_strategies.strategy_selector import RetrievalStrategySelector


class RAGPipeline:
    """
    Orchestrates retrieval and grounded answer generation for RAG applications.
    """

    def __init__(
        self,
        retriever: Retriever,
        answer_generator: AnswerGenerator,
        intent_classifier: IntentClassifier | None = None,
        retrieval_router: RetrievalRouter | None = None,
        strategy_selector: RetrievalStrategySelector | None = None,
    ):
        self.retriever = retriever
        self.answer_generator = answer_generator
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.retrieval_router = retrieval_router or RetrievalRouter()
        self.strategy_selector = strategy_selector or RetrievalStrategySelector()

    def _calculate_confidence(self, search_results) -> str:
        if not search_results:
            return "NONE"

        top_score = max(result.score for result in search_results)
        result_count = len(search_results)

        if top_score >= 0.45 and result_count >= 3:
            return "HIGH"

        if top_score >= 0.35 and result_count >= 2:
            return "MEDIUM"

        return "LOW"

    def retrieve_per_document(
        self,
        question: str,
        document_names: list[str],
        chunks_per_document: int = 3,
        min_score: float | None = None,
    ):
        """
        Retrieve top chunks separately per document,
        then combine all results together.
        """
        combined_results = []

        for document_name in document_names:
            metadata_filter = {
                "file_name": document_name
            }

            retrieval_result = self.retriever.retrieve(
                question,
                top_k=chunks_per_document,
                min_score=min_score,
                metadata_filter=metadata_filter,
            )

            combined_results.extend(retrieval_result.search_results)

        return combined_results

    def answer_question(
        self,
        question: str,
        retrieval_question: str | None = None,
        top_k: int = 5,
        min_score: float | None = None,
        metadata_filter: dict | None = None,
        answer_mode: str | None = None,
        conversation_context: str | None = None,
        domain_profile: str | None = None,
        retrieval_mode: str = "Standard chunk retrieval",
        selected_documents: list[str] | None = None,
        allow_adaptive_routing: bool = True,
    ) -> RAGResponse:
        """
        Retrieve relevant context and generate a grounded answer.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if min_score is not None:
            if not isinstance(min_score, (int, float)) or min_score < 0:
                raise ValueError("min_score must be a non-negative number.")

        orchestration_decision = self.intent_classifier.classify(
            question=question,
            conversation_context=conversation_context,
        )

        routing_decision = self.retrieval_router.route(
            orchestration_decision=orchestration_decision,
            requested_top_k=top_k,
        )

        if allow_adaptive_routing:
            top_k = routing_decision.effective_top_k

        if answer_mode is None:
            answer_mode = orchestration_decision.answer_mode

        if orchestration_decision.clarification_required:
            clarification_answer = "Can you clarify what you mean?"

            log = RetrievalLog(
                question=question,
                original_query=question,
                retrieval_query=retrieval_question or question,
                retrieved_chunk_ids=[],
                scores=[],
                answer=clarification_answer,
                answer_status="CLARIFICATION_REQUIRED",
                retrieval_confidence="NONE",
                requested_retrieval_depth=top_k,
                documents_used=[],
                grounding_check="SKIPPED",
                timestamp=datetime.now(timezone.utc).isoformat(),
                orchestration_intent=orchestration_decision.intent.value,
                retrieval_strategy=routing_decision.retrieval_strategy,
                orchestration_reasoning=orchestration_decision.reasoning,
                clarification_triggered=orchestration_decision.clarification_required,
            )

            return RAGResponse(
                answer=clarification_answer,
                sources=[],
                log=log,
                retrieval_result=None,
                retrieval_confidence="NONE",
                answer_status="CLARIFICATION_REQUIRED",
            )

        query_for_retrieval = retrieval_question or question
        retrieval_result = None

        if (
            routing_decision.retrieval_strategy == "document_balanced"
            and selected_documents
        ):
            retrieval_mode = "Document-level retrieval"

        if retrieval_mode == "Document-level retrieval":
            if not selected_documents:
                raise ValueError(
                    "Document-level retrieval requires one or more selected documents."
                )

            chunks_per_document = max(
                1,
                top_k // len(selected_documents),
            )

            search_results = self.retrieve_per_document(
                question=query_for_retrieval,
                document_names=selected_documents,
                chunks_per_document=chunks_per_document,
                min_score=min_score,
            )

        else:
            strategy = self.strategy_selector.select(routing_decision.retrieval_strategy)

            retrieval_result = strategy.retrieve(
                retriever=self.retriever,
                query=query_for_retrieval,
                top_k=top_k,
                min_score=min_score,
                metadata_filter=metadata_filter,
                selected_documents=selected_documents,
            )

            search_results = retrieval_result.search_results

        if not search_results:
            fallback_answer = (
                "I couldn't find relevant information in the indexed documents. "
                "Try rephrasing the question or asking about a more specific topic."
            )

            log = RetrievalLog(
                question=question,
                original_query=question,
                retrieval_query=query_for_retrieval,
                retrieved_chunk_ids=[],
                scores=[],
                answer=fallback_answer,
                answer_status="NO_RESULTS",
                retrieval_confidence="NONE",
                requested_retrieval_depth=top_k,
                documents_used=[],
                grounding_check="SKIPPED",
                timestamp=datetime.now(timezone.utc).isoformat(),
                orchestration_intent=orchestration_decision.intent.value,
                retrieval_strategy=routing_decision.retrieval_strategy,
                orchestration_reasoning=orchestration_decision.reasoning,
                clarification_triggered=orchestration_decision.clarification_required,
            )

            return RAGResponse(
                answer=fallback_answer,
                sources=[],
                log=log,
                retrieval_result=retrieval_result,
                retrieval_confidence="NONE",
                answer_status="NO_RESULTS",
            )

        retrieval_confidence = self._calculate_confidence(search_results)

        answer, grounding_check = self.answer_generator.generate_answer(
            question=question,
            search_results=search_results,
            answer_mode=answer_mode,
            domain_profile=domain_profile,
        )

        answer_status = (
            "INSUFFICIENT_EVIDENCE"
            if "don't have enough information" in answer.lower()
            else "ANSWERED"
        )

        sources = [
            Citation(
                id=result.id,
                file_name=result.metadata.get("file_name"),
                chunk_index=result.metadata.get("chunk_index"),
                score=result.score,
                vector_score=result.vector_score,
                rerank_bonus=result.rerank_bonus,
                final_score=result.final_score,
                metadata=result.metadata,
                text_preview=result.text[:300],
            )
            for result in search_results
        ]

        documents_used = sorted(
            {
                result.metadata.get("file_name")
                for result in search_results
                if result.metadata.get("file_name")
            }
        )

        log = RetrievalLog(
            question=question,
            original_query=question,
            retrieval_query=query_for_retrieval,
            retrieved_chunk_ids=[
                result.id
                for result in search_results
            ],
            scores=[
                result.score
                for result in search_results
            ],
            answer=answer,
            answer_status=answer_status,
            retrieval_confidence=retrieval_confidence,
            grounding_check=grounding_check,
            requested_retrieval_depth=top_k,
            documents_used=documents_used,
            timestamp=datetime.now(timezone.utc).isoformat(),
            orchestration_intent=orchestration_decision.intent.value,
            retrieval_strategy=routing_decision.retrieval_strategy,
            orchestration_reasoning=orchestration_decision.reasoning,
            clarification_triggered=orchestration_decision.clarification_required,
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            log=log,
            retrieval_result=retrieval_result,
            retrieval_confidence=retrieval_confidence,
            answer_status=answer_status,
        )