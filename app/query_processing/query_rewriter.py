

class QueryRewriter:
    """
    Simple rule-based query rewriter to optimize user questions for retrieval.
    Expands known technical keywords with deterministic expansion rules.
    """

    def rewrite(self, query: str) -> str:
        """
        Rewrite the input query into a retrieval-optimized query.

        Args:
            query (str): User's input query.

        Returns:
            str: Rewritten query string suitable for retrieval.

        Raises:
            ValueError: If query is not a non-empty string.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        normalized_query = " ".join(query.split())

        # Expansion rules mapping (lowercased keyword to phrase)
        expansions = []

        query_lc = normalized_query.lower()

        if "api" in query_lc or "apis" in query_lc:
            expansions.append("FastAPI OpenAI APIs backend services")
        if "llm" in query_lc or "llms" in query_lc:
            expansions.append("large language model OpenAI LangChain")
        if "rag" in query_lc:
            expansions.append("retrieval augmented generation vector search Chroma embeddings")
        if "embedding" in query_lc or "embeddings" in query_lc:
            expansions.append("vector representation semantic search OpenAI embeddings")

        if expansions:
            return f"{normalized_query} {' '.join(expansions)}"
        else:
            return normalized_query