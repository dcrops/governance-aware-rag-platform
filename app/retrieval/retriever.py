from app.embeddings.embeddings import EmbeddingClient
from app.vector_store.vector_store import VectorStore, SearchResult
from app.models.retrieval_result import RetrievalResult

class Retriever:
    """
    Coordinates retrieval of relevant chunks for a given query in a RAG system.
    
    Uses an embedding client and a vector store to embed user queries and find similar document chunks.
    """

    def __init__(self, embedding_client: EmbeddingClient, vector_store: VectorStore, query_rewriter=None, reranker=None) -> None:
        """
        Initialize the Retriever.

        Args:
            embedding_client: An instance of EmbeddingClient for creating embeddings.
            vector_store: An instance of VectorStore for similarity search.
        """
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.query_rewriter = query_rewriter
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 5, min_score: float | None = None, metadata_filter: dict | None = None) -> RetrievalResult:
        """
        Retrieve relevant SearchResult objects for the input query.

        Args:
            query: User query string to search with.
            top_k: Number of top similar results to return.
            min_score: Optional minimum similarity score required for results.
            metadata_filter: Optional metadata filters to apply during vector search.

        Returns:
            List of SearchResult objects sorted by similarity score.

        Raises:
            ValueError: If query is invalid or if top_k is not a positive integer.
        """
        if not isinstance(query, str) or query.strip() == "":
            raise ValueError("Query must be a non-empty string.")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if min_score is not None:
            if not isinstance(min_score, (int, float)) or min_score < 0:
                raise ValueError("min_score must be a non-negative number.")

        query_for_retrieval = query

        if self.query_rewriter is not None:
            query_for_retrieval = self.query_rewriter.rewrite(query)

        query_embedding = self.embedding_client.embed_query(query_for_retrieval)
        results = self.vector_store.similarity_search(
            query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        if self.reranker is not None:
            results = self.reranker.rerank(
                query=query_for_retrieval,
                search_results=results,
            )

        if min_score is not None:
            results = [result for result in results if result.score >= min_score]

        return RetrievalResult(
            search_results=results,
            original_query=query,
            retrieval_query=query_for_retrieval,
        )