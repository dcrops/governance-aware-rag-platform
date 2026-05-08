from app.retrieval.retriever import Retriever
from app.generation.answer_generator import AnswerGenerator
from app.models.rag_response import RAGResponse
from app.models.citation import Citation
from datetime import datetime, timezone
from app.models.retrieval_log import RetrievalLog

class RAGPipeline:
    """
    Orchestrates retrieval and grounded answer generation for RAG applications.
    """

    def __init__(
        self,
        retriever: Retriever,
        answer_generator: AnswerGenerator,
    ):
        """
        Initialize the RAGPipeline with required components.

        Args:
            retriever: An object with a .retrieve(query: str, top_k: int) -> List[SearchResult] interface.
            answer_generator: An object with a .generate_answer(question: str, search_results: List[SearchResult]) -> str interface.
        """
        self.retriever = retriever
        self.answer_generator = answer_generator

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

    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        min_score: float | None = None,
        metadata_filter: dict | None = None
    ) -> RAGResponse:
        """
        Retrieve relevant context and generate a grounded answer to the user question.

        Args:
            question (str): The user's question.
            top_k (int, optional): Number of top results to retrieve. Default is 5.

        Returns:
            str: Answer generated using retrieved context.

        Raises:
            ValueError: If question is not a non-empty string or top_k is not a positive integer.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if min_score is not None:
            if not isinstance(min_score, (int, float)) or min_score < 0:
                raise ValueError("min_score must be a non-negative number.")

        # Retrieve relevant documents
        retrieval_result = self.retriever.retrieve(
            question,
            top_k=top_k,
            min_score=min_score,
            metadata_filter=metadata_filter,
        )

        search_results = retrieval_result.search_results

        if not search_results:
            fallback_answer = (
                "I couldn't find relevant information in the indexed documents. "
                "Try rephrasing the question or asking about a more specific topic."
            )

            log = RetrievalLog(
                question=question,

                original_query=retrieval_result.original_query,
                retrieval_query=retrieval_result.retrieval_query,

                retrieved_chunk_ids=[],
                scores=[],

                answer=fallback_answer,

                answer_status="NO_RESULTS",
                retrieval_confidence="NONE",

                requested_retrieval_depth=top_k,

                documents_used=[],

                timestamp=datetime.now(timezone.utc).isoformat(),
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

        answer = self.answer_generator.generate_answer(question, search_results)

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
                metadata=result.metadata,
                text_preview=result.text[:300],
            )
            for result in search_results
        ]

        log = RetrievalLog(
            question=question,
            original_query=retrieval_result.original_query,
            retrieval_query=retrieval_result.retrieval_query,

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

            requested_retrieval_depth=top_k,

            documents_used=list(
                {
                    result.metadata.get("file_name")
                    for result in search_results
                    if result.metadata.get("file_name")
                }
            ),

            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            log=log,
            retrieval_result=retrieval_result,
            retrieval_confidence=retrieval_confidence,
            answer_status=answer_status,
        )