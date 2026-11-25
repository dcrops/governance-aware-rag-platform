from openai import OpenAI

from app.config import apply_chat_model_params, resolve_chat_model


def rewrite_contextual_question(
    question: str,
    chat_history: list[dict],
    model: str | None = None,
    client=None,
) -> str:
    """
    Rewrite a follow-up into a standalone retrieval question.

    This is the production implementation used by the Streamlit UI.
    The prompt and token budget match the previous inline call.
    """
    if not chat_history:
        return question

    recent_history = chat_history[-3:]

    context_lines = []

    for turn in recent_history:
        context_lines.append(f"Previous question: {turn.get('question', '')}")
        context_lines.append(f"Previous answer: {turn.get('answer', '')}")

    conversation_context = "\n".join(context_lines)

    prompt = f"""
You rewrite follow-up questions into standalone questions for a RAG system.

Use the recent conversation only to resolve references such as:
it, that, this, he, she, they, the event, the date, the person.

Do not answer the question.
Return only the rewritten standalone question.

Recent conversation:
{conversation_context}

Current question:
{question}

Standalone question:
"""

    selected_model = resolve_chat_model(model)
    active_client = client or OpenAI()
    response = active_client.chat.completions.create(
        **apply_chat_model_params(
            selected_model,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "temperature": 0,
                "max_tokens": 80,
            },
        )
    )

    return response.choices[0].message.content.strip()
