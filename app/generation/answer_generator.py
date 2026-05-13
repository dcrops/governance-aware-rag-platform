from typing import List
from openai import OpenAI

from app.vector_store.vector_store import SearchResult
from app.orchestration.domain_profiles import get_domain_profile


class AnswerGenerator:
    """
    Generates answers grounded in retrieved document context using an OpenAI chat model.
    """

    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.client = OpenAI()
        self.model_name = model_name

    def validate_context_relevance(
        self,
        question: str,
        context: str,
        answer_mode: str = "standard",
    ) -> bool:
        """
        Checks whether the retrieved context directly supports answering the user's question.
        """

        if answer_mode == "aggregation":
            validation_prompt = (
                "You are validating whether retrieved context is relevant enough for an extraction/list/count question.\n\n"
                "Question:\n"
                f"{question}\n\n"
                "Retrieved context:\n"
                f"{context}\n\n"
                "Does the retrieved context contain relevant evidence that can help answer the user's extraction, list, count, or aggregation question?\n"
                "Answer only YES or NO."
            )

        elif answer_mode == "comparison":
            validation_prompt = (
                "You are validating whether retrieved context is relevant enough for a comparison question.\n\n"
                "Question:\n"
                f"{question}\n\n"
                "Retrieved context:\n"
                f"{context}\n\n"
                "Does the retrieved context contain relevant evidence about the things being compared, even if the difference must be inferred from multiple sources?\n"
                "Answer only YES or NO."
            )

        else:
            validation_prompt = (
                "You are validating whether retrieved context is relevant enough to answer a question.\n\n"
                "Question:\n"
                f"{question}\n\n"
                "Retrieved context:\n"
                f"{context}\n\n"
                "Does the retrieved context directly contain the information needed to answer the question?\n"
                "Answer only YES or NO."
            )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": validation_prompt}
            ],
            temperature=0,
            max_tokens=5,
        )

        response_text = response.choices[0].message.content.strip()

        verdict = response_text.upper()

        return verdict.startswith("YES")

    def generate_answer(
        self,
        question: str,
        search_results: List[SearchResult],
        answer_mode: str = "standard",
        domain_profile: str | None = None,
    ) -> tuple[str, str]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        if not search_results or not all(getattr(sr, "text", None) for sr in search_results):
            raise ValueError(
                "search_results must be a non-empty list of SearchResult objects with text."
            )

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

        grounding_passed = self.validate_context_relevance(
            question=question,
            context=context,
            answer_mode=answer_mode,
        )

        if not grounding_passed:
            return (
                "I'm sorry, I don't have enough information to answer that question "
                "based on the provided context.",
                "FAIL",
            )

        domain_instruction = get_domain_profile(domain_profile)

        aggregation_instruction = ""

        if answer_mode == "aggregation":
            aggregation_instruction = (
                "The user's question requires exhaustive extraction across all retrieved evidence. "
                "You must inspect EVERY provided source carefully before answering. "
                "Extract ALL relevant entities, names, items, or references mentioned across the retrieved context. "
                "Do NOT provide only examples, the most prominent items, or the first items found. "
                "Deduplicate repeated items and return the COMPLETE set of extracted items supported by the evidence. "
                "If counting items, first extract the full list, then count them. "
            )

        system_prompt = (
            "You are an expert assistant. Use ONLY the provided context to answer the user's question. "
            f"{domain_instruction}"
            f"{aggregation_instruction}"
            "Answer the user's specific question directly first, using the most concise supported answer available from the retrieved context. "
            "Do not begin with unnecessary background information if the answer can be stated clearly and briefly. "
            "After the direct answer, provide brief supporting context only if helpful. "
            "If the context does not provide enough information, reply: "
            "\"I'm sorry, I don't have enough information to answer that question based on the provided context.\" "
            "When the user asks where information came from, refer to the document name and page number if available, not only the chunk number. "
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
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=512,
                n=1,
            )

            answer = response.choices[0].message.content.strip()

            return answer, "PASS"

        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}") from e