from openai import OpenAI


class EmbeddingClient:
    """
    OpenAI embedding client for converting text strings into embedding vectors.
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        """
        Initialize the EmbeddingClient.

        Args:
            model: Name of the OpenAI embedding model to use.
        """
        self.model = model
        self.client = OpenAI()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts and return their embedding vectors.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors (one for each text).

        Raises:
            ValueError: If any input text is empty or whitespace-only.
            RuntimeError: If the API call fails.
        """
        if not texts:
            return []

        for idx, text in enumerate(texts):
            if not isinstance(text, str) or text.strip() == "":
                raise ValueError(f"Text at index {idx} is empty or whitespace-only.")

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            sorted_data = sorted(response.data, key=lambda item: item.index)
            return [item.embedding for item in sorted_data]
        except Exception as e:
            raise RuntimeError(f"Failed to embed texts: {e}")

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string and return its embedding vector.

        Args:
            query: Query string to embed.

        Returns:
            Embedding vector for the query.

        Raises:
            ValueError: If the query is empty or whitespace-only.
            RuntimeError: If the API call fails.
        """
        if not isinstance(query, str) or query.strip() == "":
            raise ValueError("Query is empty or whitespace-only.")

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[query],
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Failed to embed query: {e}")