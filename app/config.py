import os
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "data/index")

DEFAULT_CLIENT_NAME = os.getenv(
    "RAG_DEFAULT_CLIENT_NAME",
    "default_client",
)

DEFAULT_TOP_K = int(
    os.getenv("RAG_DEFAULT_TOP_K", "5")
)

DEFAULT_MIN_SCORE = float(
    os.getenv("RAG_DEFAULT_MIN_SCORE", "0.35")
)

# Operational chat default. Override with RAG_CHAT_MODEL, for example gpt-4.1-mini.
DEFAULT_CHAT_MODEL = "gpt-6-luna"
LUNA_CHAT_MODEL = "gpt-6-luna"
LEGACY_CHAT_MODEL = "gpt-4.1-mini"


def resolve_chat_model(explicit_model: str | None = None) -> str:
    """
    Select the chat model for a call.

    An explicit constructor argument wins. Otherwise RAG_CHAT_MODEL is used.
    When that variable is unset, the operational default is gpt-6-luna.
    The environment is read on each call so tests and process overrides apply.
    """
    if isinstance(explicit_model, str) and explicit_model.strip():
        return explicit_model.strip()

    configured = os.getenv("RAG_CHAT_MODEL")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()

    return DEFAULT_CHAT_MODEL


def apply_chat_model_params(model: str, params: dict) -> dict:
    """
    Apply the validated parameter profile for the selected chat model.

    Call sites pass the historical Chat Completions arguments, including
    max_tokens and sampling parameters. gpt-4.1-mini keeps those arguments.
    gpt-6-luna uses reasoning_effort none and max_completion_tokens, and omits
    temperature, top_p, and stop. Numeric token budgets and n are preserved.
    """
    effective = dict(params)
    effective["model"] = model

    if model != LUNA_CHAT_MODEL:
        return effective

    if "max_tokens" in effective:
        effective["max_completion_tokens"] = effective.pop("max_tokens")

    effective.pop("temperature", None)
    effective.pop("top_p", None)
    effective.pop("stop", None)
    effective["reasoning_effort"] = "none"
    return effective
