from openai import OpenAI

from app.config import apply_chat_model_params, resolve_chat_model

class LLMQueryRewriter:
    """
    Rewrites user queries into concise, retrieval-optimized queries using an OpenAI chat model.
    Preserves user intent and avoids adding information not present or implied in the original question.
    """

    def __init__(self, model: str | None = None, client=None):
        """
        Initialize the LLMQueryRewriter.

        Args:
            model: Explicit chat model. When omitted, resolve_chat_model applies.
            client: Optional OpenAI client, used by tests and controlled validation.
        """
        self._client = client or OpenAI()
        self._model = resolve_chat_model(model)

        # Template system prompt for query rewriting.
        self._rewrite_system_prompt = (
            "You are a helpful assistant tasked with rewriting user questions into concise, "
            "retrieval-optimized queries for a search system. "
            "Preserve the user's original intent. Do not answer the question. "
            "Expand ambiguous terms only when useful for improving retrieval. "
            "Do not add factual claims. You may add closely related search terms when they clarify the likely retrieval intent."
            "Return only the rewritten query, with no explanation or formatting. "
            "Keep the rewritten query as short as possible while ensuring clarity for retrieval."
        )

    def rewrite(self, query: str) -> str:
        """
        Rewrite a user query for retrieval optimization.

        Args:
            query (str): The user's original question.

        Returns:
            str: A rewritten, retrieval-optimized query.

        Raises:
            ValueError: If query is not a non-empty string.
            RuntimeError: If the OpenAI API call fails.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        messages = [
            {"role": "system", "content": self._rewrite_system_prompt},
            {"role": "user", "content": query.strip()},
        ]

        try:
            response = self._client.chat.completions.create(
                **apply_chat_model_params(
                    self._model,
                    {
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": 64,
                        "top_p": 1.0,
                        "stop": None,
                    },
                )
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

        # Extract the rewritten query text
        try:
            rewritten = response.choices[0].message.content.strip()
        except (AttributeError, IndexError, KeyError):
            raise RuntimeError("Failed to parse response from OpenAI API.")

        rewritten = rewritten.strip().strip('"').strip("'")

        if not rewritten:
            return query.strip()

        if len(rewritten) > 300:
            return query.strip()

        return rewritten
