from app.models.retrieval_result import RetrievalResult
from app.retrieval.retriever import Retriever
from app.retrieval_strategies.retrieval_strategy import BaseRetrievalStrategy


class StandardRetrievalStrategy(BaseRetrievalStrategy):

    def retrieve(
        self,
        retriever: Retriever,
        query: str,
        top_k: int,
        min_score: float | None = None,
        metadata_filter: dict | None = None,
        selected_documents: list[str] | None = None,
    ) -> RetrievalResult:

        adjusted_top_k = max(top_k * 2, 10)

        if not selected_documents:
            return retriever.retrieve(
                query=query,
                top_k=adjusted_top_k,
                min_score=min_score,
                metadata_filter=metadata_filter,
            )

        combined_results = []

        for document in selected_documents:

            document_filter = {"file_name": document}

            if metadata_filter:
                document_filter.update(metadata_filter)

            result = retriever.retrieve(
                query=query,
                top_k=top_k,
                min_score=min_score,
                metadata_filter=document_filter,
            )

            combined_results.extend(result.search_results)

        combined_results = sorted(
            combined_results,
            key=lambda x: x.score,
            reverse=True,
        )[:adjusted_top_k]

        return RetrievalResult(
            search_results=combined_results,
            original_query=query,
            retrieval_query=query,
        )