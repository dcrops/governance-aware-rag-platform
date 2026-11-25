"""Offline checks for centralised chat-model selection. No network calls."""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _setup_path() -> None:
    os.chdir(PROJECT_ROOT)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _Details:
    def __init__(self, reasoning_tokens: int):
        self.reasoning_tokens = reasoning_tokens


class _Usage:
    def __init__(self, completion_tokens: int, reasoning_tokens: int = 0):
        self.completion_tokens = completion_tokens
        self.completion_tokens_details = _Details(reasoning_tokens)


class _Response:
    def __init__(self, content: str, completion_tokens: int = 4):
        self.model = "fake-model"
        self.choices = [_Choice(content)]
        self.usage = _Usage(completion_tokens)


class _Completions:
    def __init__(self, contents: list[str]):
        self.contents = list(contents)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.contents:
            raise AssertionError("Unexpected chat call")
        return _Response(self.contents.pop(0))


class _Chat:
    def __init__(self, completions: _Completions):
        self.completions = completions


class _Client:
    def __init__(self, contents: list[str]):
        self.chat = _Chat(_Completions(contents))


def _params(call: dict) -> dict:
    return {key: value for key, value in call.items() if key != "messages"}


@pytest.fixture(autouse=True)
def project_path():
    _setup_path()


def test_luna_is_the_default_chat_model(monkeypatch):
    monkeypatch.delenv("RAG_CHAT_MODEL", raising=False)
    from app.config import resolve_chat_model

    assert resolve_chat_model() == "gpt-6-luna"
    assert resolve_chat_model(None) == "gpt-6-luna"
    assert resolve_chat_model("  ") == "gpt-6-luna"


def test_environment_override_selects_legacy_model(monkeypatch):
    monkeypatch.setenv("RAG_CHAT_MODEL", "gpt-4.1-mini")
    from app.config import resolve_chat_model

    assert resolve_chat_model() == "gpt-4.1-mini"


def test_explicit_model_selection_wins_over_environment(monkeypatch):
    monkeypatch.setenv("RAG_CHAT_MODEL", "gpt-6-luna")
    from app.config import resolve_chat_model

    assert resolve_chat_model("gpt-4.1-mini") == "gpt-4.1-mini"


def test_legacy_parameters_are_preserved():
    from app.config import apply_chat_model_params

    effective = apply_chat_model_params(
        "gpt-4.1-mini",
        {
            "messages": [{"role": "user", "content": "question"}],
            "temperature": 0.0,
            "max_tokens": 64,
            "top_p": 1.0,
            "stop": None,
        },
    )

    assert effective["model"] == "gpt-4.1-mini"
    assert effective["temperature"] == 0.0
    assert effective["max_tokens"] == 64
    assert effective["top_p"] == 1.0
    assert effective["stop"] is None
    assert "reasoning_effort" not in effective
    assert "max_completion_tokens" not in effective


def test_luna_receives_the_validated_parameter_profile():
    from app.config import apply_chat_model_params

    effective = apply_chat_model_params(
        "gpt-6-luna",
        {
            "messages": [{"role": "user", "content": "question"}],
            "temperature": 0.2,
            "max_tokens": 512,
            "top_p": 1.0,
            "stop": None,
            "n": 1,
        },
    )

    assert effective == {
        "messages": [{"role": "user", "content": "question"}],
        "max_completion_tokens": 512,
        "n": 1,
        "model": "gpt-6-luna",
        "reasoning_effort": "none",
    }


def test_answer_generator_default_and_explicit_models(monkeypatch):
    monkeypatch.delenv("RAG_CHAT_MODEL", raising=False)
    from app.generation.answer_generator import AnswerGenerator

    default_client = _Client(["YES", "Seven years."])
    default_generator = AnswerGenerator(client=default_client)
    assert default_generator.model_name == "gpt-6-luna"

    from app.vector_store.vector_store import SearchResult

    answer, grounding = default_generator.generate_answer(
        "How long must payroll records be retained?",
        [
            SearchResult(
                id="policy-3",
                text="UNIQUE_CONTEXT_MARKER Payroll records must be retained for seven years.",
                score=0.9,
                metadata={"file_name": "policy_style.txt", "chunk_index": 3},
            )
        ],
    )

    assert grounding == "PASS"
    assert answer == "Seven years."
    relevance_call, answer_call = default_client.chat.completions.calls
    assert _params(relevance_call) == {
        "model": "gpt-6-luna",
        "max_completion_tokens": 5,
        "reasoning_effort": "none",
    }
    assert _params(answer_call) == {
        "model": "gpt-6-luna",
        "max_completion_tokens": 512,
        "n": 1,
        "reasoning_effort": "none",
    }
    observation = default_generator.last_relevance_observation
    assert observation.raw_response == "YES"
    assert observation.finish_reason == "stop"
    assert observation.completion_tokens == 4
    assert observation.reasoning_tokens == 0
    assert "UNIQUE_CONTEXT_MARKER" not in str(observation)

    monkeypatch.setenv("RAG_CHAT_MODEL", "gpt-6-luna")
    legacy_client = _Client(["NO"])
    legacy_generator = AnswerGenerator(model_name="gpt-4.1-mini", client=legacy_client)
    assert legacy_generator.model_name == "gpt-4.1-mini"
    assert legacy_generator.validate_context_relevance("question", "context") is False
    assert _params(legacy_client.chat.completions.calls[0]) == {
        "model": "gpt-4.1-mini",
        "temperature": 0,
        "max_tokens": 5,
    }


def test_llm_query_rewriter_uses_central_model_selection(monkeypatch):
    monkeypatch.delenv("RAG_CHAT_MODEL", raising=False)
    from app.query_processing.llm_query_rewriter import LLMQueryRewriter

    luna_client = _Client(["Payroll records retention period"])
    luna_rewriter = LLMQueryRewriter(client=luna_client)
    assert luna_rewriter._model == "gpt-6-luna"
    assert luna_rewriter.rewrite("How long do we have to keep payroll records for?") == (
        "Payroll records retention period"
    )
    assert _params(luna_client.chat.completions.calls[0]) == {
        "model": "gpt-6-luna",
        "max_completion_tokens": 64,
        "reasoning_effort": "none",
    }

    monkeypatch.setenv("RAG_CHAT_MODEL", "gpt-6-luna")
    legacy_client = _Client(["Payroll records retention period"])
    legacy_rewriter = LLMQueryRewriter(model="gpt-4.1-mini", client=legacy_client)
    legacy_rewriter.rewrite("How long do we have to keep payroll records for?")
    assert _params(legacy_client.chat.completions.calls[0]) == {
        "model": "gpt-4.1-mini",
        "temperature": 0.0,
        "max_tokens": 64,
        "top_p": 1.0,
        "stop": None,
    }


def test_contextual_rewriter_uses_central_model_selection(monkeypatch):
    monkeypatch.delenv("RAG_CHAT_MODEL", raising=False)
    from app.query_processing.contextual_rewriter import rewrite_contextual_question

    assert (
        rewrite_contextual_question("Who can access them?", [])
        == "Who can access them?"
    )

    client = _Client(["Who is allowed to access payroll records?"])
    rewritten = rewrite_contextual_question(
        "Who is allowed to access them?",
        [
            {
                "question": "How long must payroll records be retained?",
                "answer": "Payroll records must be retained for seven years.",
            }
        ],
        client=client,
    )
    assert rewritten == "Who is allowed to access payroll records?"
    assert _params(client.chat.completions.calls[0]) == {
        "model": "gpt-6-luna",
        "max_completion_tokens": 80,
        "reasoning_effort": "none",
    }
    assert "temperature" not in client.chat.completions.calls[0]


def test_streamlit_delegates_to_the_shared_rewriter():
    source = open(
        os.path.join(PROJECT_ROOT, "streamlit_app.py"),
        encoding="utf-8",
    ).read()
    assert "rewrite_contextual_question" in source
    assert 'model="gpt-4.1-mini"' not in source
    assert 'model="gpt-6-luna"' not in source
    assert "max_tokens=80" not in source


def test_every_chat_call_site_uses_centralised_selection():
    production_files = [
        "app/generation/answer_generator.py",
        "app/query_processing/llm_query_rewriter.py",
        "app/query_processing/contextual_rewriter.py",
    ]
    for relative_path in production_files:
        source = open(os.path.join(PROJECT_ROOT, relative_path), encoding="utf-8").read()
        assert "apply_chat_model_params" in source
        assert "resolve_chat_model" in source
        assert 'model="gpt-4.1-mini"' not in source
        assert 'model="gpt-6-luna"' not in source


def test_embedding_configuration_remains_unchanged():
    import inspect
    from app.embeddings.embeddings import EmbeddingClient

    default_model = inspect.signature(EmbeddingClient.__init__).parameters["model"].default
    assert default_model == "text-embedding-3-small"
    source = open(
        os.path.join(PROJECT_ROOT, "app", "embeddings", "embeddings.py"),
        encoding="utf-8",
    ).read()
    assert "RAG_CHAT_MODEL" not in source
    assert "gpt-6-luna" not in source


def test_pipeline_records_relevance_telemetry_without_context():
    from app.generation.answer_generator import AnswerGenerator
    from app.models.retrieval_result import RetrievalResult
    from app.orchestration.rag_pipeline import RAGPipeline
    from app.vector_store.vector_store import SearchResult

    client = _Client(["YES", "Payroll records must be retained for seven years."])
    generator = AnswerGenerator(client=client)

    class FakeRetriever:
        def retrieve(self, query, top_k=5, min_score=None, metadata_filter=None):
            return RetrievalResult(
                search_results=[
                    SearchResult(
                        id="policy-3",
                        text="UNIQUE_CONTEXT_MARKER retained for seven years.",
                        score=0.9,
                        metadata={"file_name": "policy_style.txt", "chunk_index": 3},
                    )
                ],
                original_query=query,
                retrieval_query=query,
            )

    pipeline = RAGPipeline(retriever=FakeRetriever(), answer_generator=generator)
    response = pipeline.answer_question("How long must payroll records be retained?")

    assert response.log.chat_model == "gpt-6-luna"
    assert response.log.relevance_response == "YES"
    assert response.log.relevance_finish_reason == "stop"
    assert response.log.relevance_completion_tokens == 4
    assert response.log.relevance_reasoning_tokens == 0
    assert "UNIQUE_CONTEXT_MARKER" not in (response.log.relevance_response or "")
    assert response.answer_status == "ANSWERED"


def test_clarification_does_not_call_the_chat_model():
    from app.generation.answer_generator import AnswerGenerator
    from app.orchestration.rag_pipeline import RAGPipeline

    client = _Client([])
    generator = AnswerGenerator(model_name="gpt-6-luna", client=client)

    class FailingRetriever:
        def retrieve(self, *args, **kwargs):
            raise AssertionError("Clarification must not retrieve.")

    pipeline = RAGPipeline(retriever=FailingRetriever(), answer_generator=generator)
    response = pipeline.answer_question(
        "What about payroll?",
        conversation_context="Previous question: How long are records retained?",
    )

    assert response.answer_status == "CLARIFICATION_REQUIRED"
    assert response.log.chat_model == "gpt-6-luna"
    assert response.log.relevance_response is None
    assert client.chat.completions.calls == []
