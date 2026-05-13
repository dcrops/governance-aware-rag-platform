from app.vector_store.vector_store import SearchResult


class SimpleReranker:
    """
    Simple deterministic reranker that boosts search results
    based on keyword overlap with the query.
    """

    def rerank(
        self,
        query: str,
        search_results: list[SearchResult],
    ) -> list[SearchResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if not isinstance(search_results, list):
            raise ValueError("search_results must be a list.")

        query_tokens = set(query.lower().split())

        reranked_results = []

        for result in search_results:
            result_tokens = set(result.text.lower().split())
            shared_tokens = query_tokens & result_tokens

            keyword_overlap_bonus = len(shared_tokens) * 0.05
            base_score = result.final_score or result.score

            final_score = base_score + keyword_overlap_bonus

            reranked_result = result.model_copy(
                update={
                    "score": final_score,
                    "final_score": final_score,
                    "rerank_bonus": keyword_overlap_bonus,
                }
            )
            reranked_results.append(reranked_result)

        return sorted(
            reranked_results,
            key=lambda result: result.score,
            reverse=True,
        )