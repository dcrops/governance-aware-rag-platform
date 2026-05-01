from typing import List
from openai import OpenAI

from app.vector_store.vector_store import SearchResult


class AnswerGenerator:
    """
    Generates answers grounded in retrieved document context using an OpenAI chat model.
    """
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        """
        Initialize the OpenAI client and select the model to use.

        Args:
            model_name (str): Name of the OpenAI chat model.
        """

        self.client = OpenAI()
        self.model_name = model_name

    def generate_answer(self, question: str, search_results: List[SearchResult]) -> str:
        """
        Generate a grounded answer using only the context from search_results.

        Args:
            question (str): User's question.
            search_results (List[SearchResult]): Retrieved document chunks.

        Returns:
            str: Model-generated grounded answer.

        Raises:
            ValueError: For invalid question or empty search results.
            RuntimeError: For API failures.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        if not search_results or not all(getattr(sr, "text", None) for sr in search_results):
            raise ValueError("search_results must be a non-empty list of SearchResult objects with text.")

        # Construct the context from the search results
        context_chunks = [sr.text.strip() for sr in search_results]
        context_parts = []

        for i, result in enumerate(search_results, start=1):
            file_name = result.metadata.get("file_name", "Unknown document")
            page_number = result.metadata.get("page_number")
            chunk_index = result.metadata.get("chunk_index")

            source_label = f"Source {i} | Document: {file_name}"

            if page_number is not None:
                source_label += f" | Page: {page_number}"

            if chunk_index is not None:
                source_label += f" | Chunk: {chunk_index}"

            context_parts.append(
                f"{source_label}\n{result.text.strip()}"
            )

        context = "\n\n".join(context_parts)

        # Instruction for the model
        system_prompt = (
            "You are an expert assistant. Use ONLY the provided context to answer the user's question. "
            "If the context does not provide enough information, reply: "
            "\"I'm sorry, I don't have enough information to answer that question based on the provided context.\" "
            "When the user asks where information came from, refer to the document name and page number if available, not only the chunk number."
            "Do NOT use any outside knowledge or invent any information."
        )

        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=512,
                n=1,
            )
            answer = response.choices[0].message.content.strip()
            return answer
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}") from e