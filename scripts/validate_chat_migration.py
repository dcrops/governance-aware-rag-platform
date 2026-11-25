"""
Final live validation of the migrated production chat integrations.

Uses the existing synthetic comparison index and frozen retrieval contexts.
Does not query operational indexes, retry failed parameter combinations,
or change prompts and budgets.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from app.config import resolve_chat_model
from app.embeddings.embeddings import EmbeddingClient
from app.generation.answer_generator import AnswerGenerator
from app.orchestration.rag_pipeline import RAGPipeline
from app.query_processing.contextual_rewriter import rewrite_contextual_question
from app.query_processing.llm_query_rewriter import LLMQueryRewriter
from app.query_processing.query_rewriter import QueryRewriter
from app.reranking.simple_reranker import SimpleReranker
from app.retrieval.retriever import Retriever
from app.vector_store.vector_store import SearchResult, VectorStore

MANIFEST_PATH = PROJECT_ROOT / "logs" / "evaluation_runs" / "model_comparison_manifest.json"
RESULTS_PATH = PROJECT_ROOT / "logs" / "evaluation_runs" / "migration_validation_20261125.json"
PERSIST_DIR = PROJECT_ROOT / "logs" / "model_comparison_index"
OPERATIONAL_INDEX = (PROJECT_ROOT / "data" / "index").resolve()
COLLECTION_NAME = "synthetic_workplace"
ALLOWED_FILE_NAMES = {"policy_style.txt", "faq_style.txt"}

# Scenario relevance calls count toward five supported and five unsupported checks.
# Four further direct relevance calls complete each set of five.
PLANNED_CHAT_CALLS = 21
CHAT_CAP = 22
PLANNED_EMBEDDINGS = 1

BANNED_PATTERNS = (
    "cropper",
    "luxoft",
    "compare club",
    "rmit",
    "resume",
    "curriculum vitae",
    "confidential",
    "@",
)


def assert_safe(label: str, text: str) -> None:
    lowered = text.lower()
    for pattern in BANNED_PATTERNS:
        if pattern in lowered:
            raise RuntimeError(f"Safety check failed for {label}: found {pattern!r}.")


def load_results(payload: list[dict]) -> list[SearchResult]:
    return [SearchResult(**item) for item in payload]


def build_context(search_results: list[SearchResult]) -> str:
    """Match AnswerGenerator's source labels so repeat calls see the same context."""
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
        context_parts.append(f"{source_label}\n{result.text.strip()}")
    return "\n\n".join(context_parts)


def usage_fields(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "completion_tokens": None,
            "reasoning_tokens": None,
        }
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details is not None else None
    return {
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": reasoning,
    }


def main() -> None:
    if PERSIST_DIR.resolve() == OPERATIONAL_INDEX or OPERATIONAL_INDEX in PERSIST_DIR.resolve().parents:
        raise RuntimeError("Refusing to open the operational index.")
    if os.getenv("RAG_CHAT_MODEL"):
        raise RuntimeError("Unset RAG_CHAT_MODEL so this run proves the code default.")
    if resolve_chat_model() != "gpt-6-luna":
        raise RuntimeError(f"Default chat model is {resolve_chat_model()}, not gpt-6-luna.")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    selected_comparison = manifest["scenarios"]["scenario_3_selected"]
    scenarios = {
        "scenario_1": manifest["scenarios"]["scenario_1"],
        "scenario_2": manifest["scenarios"]["scenario_2"],
        "scenario_3": manifest["scenarios"][selected_comparison],
        "scenario_4": manifest["scenarios"]["scenario_4"],
    }
    for name, scenario in scenarios.items():
        for chunk in scenario["chunks"]:
            assert_safe(name, chunk["text"])
            if chunk["metadata"].get("file_name") not in ALLOWED_FILE_NAMES:
                raise RuntimeError(f"{name} contains an unexpected document.")

    plan = {
        "default_model": resolve_chat_model(),
        "planned_chat_calls": PLANNED_CHAT_CALLS,
        "chat_cap": CHAT_CAP,
        "planned_embeddings": PLANNED_EMBEDDINGS,
        "retries": "disabled",
        "breakdown": {
            "scenario_1": "1 relevance + 1 answer; counts as supported relevance 1 of 5",
            "supported_repeats": "4 additional relevance calls",
            "scenario_2": "1 relevance; counts as unsupported relevance 1 of 5",
            "unsupported_repeats": "4 additional relevance calls",
            "scenario_3": "1 relevance + 1 answer",
            "scenario_4": "1 relevance + 1 answer",
            "scenario_5": "1 production contextual rewrite + 1 downstream relevance + 1 downstream answer",
            "scenario_6": "1 production evaluation rewrite",
            "cross_document": "1 relevance + 1 answer on both access rules",
        },
    }
    print("PLANNED API CALLS")
    print(json.dumps(plan, indent=2))

    state = {"chat": 0, "embeddings": 0, "calls": []}
    label = {"value": "unspecified"}
    client = OpenAI(max_retries=0, timeout=120.0)
    original_create = client.chat.completions.create

    def create(**kwargs):
        if state["chat"] >= CHAT_CAP:
            raise RuntimeError("Chat call cap reached. No further request was sent.")
        state["chat"] += 1
        safe_params = {key: value for key, value in kwargs.items() if key != "messages"}
        started = time.perf_counter()
        try:
            response = original_create(**kwargs)
        except Exception as exc:
            state["calls"].append(
                {
                    "label": label["value"],
                    "ok": False,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "effective_parameters": safe_params,
                    "error": f"{type(exc).__name__}: {exc}"[:1000],
                }
            )
            raise
        choice = response.choices[0]
        content = choice.message.content
        state["calls"].append(
            {
                "label": label["value"],
                "ok": True,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "requested_model": kwargs.get("model"),
                "returned_model": getattr(response, "model", None),
                "effective_parameters": safe_params,
                "finish_reason": choice.finish_reason,
                "content": content,
                "content_is_empty": content is None or str(content).strip() == "",
                **usage_fields(response),
            }
        )
        return response

    client.chat.completions.create = create

    embed_client = EmbeddingClient()
    embed_client.client = OpenAI(max_retries=0, timeout=120.0)
    original_embed = embed_client.embed_query

    def embed_query(query: str):
        assert_safe("embedding query", query)
        if state["embeddings"] >= PLANNED_EMBEDDINGS:
            raise RuntimeError("Embedding call cap reached.")
        state["embeddings"] += 1
        return original_embed(query)

    embed_client.embed_query = embed_query

    store = VectorStore(persist_dir=str(PERSIST_DIR), collection_name=COLLECTION_NAME)
    listed = {item["file_name"] for item in store.list_documents()}
    if listed != ALLOWED_FILE_NAMES:
        raise RuntimeError(f"Synthetic collection contains unexpected files: {listed}")

    generator = AnswerGenerator(client=client)
    if generator.model_name != "gpt-6-luna":
        raise RuntimeError(f"AnswerGenerator default is {generator.model_name}.")

    results = {
        "plan": plan,
        "resolved_default_model": generator.model_name,
        "scenarios": {},
        "relevance_repeatability": {"supported": [], "unsupported": []},
    }

    def record_observation(bucket: list[dict], passed: bool) -> None:
        observation = generator.last_relevance_observation
        bucket.append(
            {
                "passed": passed,
                "raw_response": None if observation is None else observation.raw_response,
                "finish_reason": None if observation is None else observation.finish_reason,
                "completion_tokens": None if observation is None else observation.completion_tokens,
                "reasoning_tokens": None if observation is None else observation.reasoning_tokens,
                "selected_model": None if observation is None else observation.model,
            }
        )

    def run_generation(name: str, question: str, payload: dict, answer_mode: str | None = None) -> dict:
        label["value"] = name
        chunks = load_results(payload["chunks"])
        started = len(state["calls"])
        answer, grounding = generator.generate_answer(
            question=question,
            search_results=chunks,
            answer_mode=answer_mode or payload["answer_mode"],
        )
        observation = generator.last_relevance_observation
        record = {
            "question": question,
            "answer_mode": answer_mode or payload["answer_mode"],
            "answer": answer,
            "grounding_check": grounding,
            "retrieval_confidence": RAGPipeline._calculate_confidence(None, chunks),
            "selected_model": generator.model_name,
            "relevance_response": None if observation is None else observation.raw_response,
            "relevance_finish_reason": None if observation is None else observation.finish_reason,
            "relevance_completion_tokens": None if observation is None else observation.completion_tokens,
            "relevance_reasoning_tokens": None if observation is None else observation.reasoning_tokens,
            "chunk_ids": [chunk.id for chunk in chunks],
            "calls": state["calls"][started:],
        }
        results["scenarios"][name] = record
        return record

    supported_chunks = load_results(scenarios["scenario_1"]["chunks"])
    unsupported_chunks = load_results(scenarios["scenario_2"]["chunks"])
    supported_question = "How long must payroll records be retained?"
    unsupported_question = "What university did the author attend?"
    supported_context = build_context(supported_chunks)
    unsupported_context = build_context(unsupported_chunks)
    assert_safe("supported_context", supported_context)
    assert_safe("unsupported_context", unsupported_context)

    supported_record = run_generation("scenario_1", supported_question, scenarios["scenario_1"])
    record_observation(
        results["relevance_repeatability"]["supported"],
        supported_record["relevance_response"] == "YES",
    )
    for index in range(4):
        label["value"] = f"supported_relevance_{index + 2}"
        passed = generator.validate_context_relevance(
            supported_question,
            supported_context,
            answer_mode="standard",
        )
        record_observation(results["relevance_repeatability"]["supported"], passed)

    unsupported_record = run_generation("scenario_2", unsupported_question, scenarios["scenario_2"])
    record_observation(
        results["relevance_repeatability"]["unsupported"],
        unsupported_record["relevance_response"] == "NO",
    )
    for index in range(4):
        label["value"] = f"unsupported_relevance_{index + 2}"
        passed = generator.validate_context_relevance(
            unsupported_question,
            unsupported_context,
            answer_mode="standard",
        )
        record_observation(
            results["relevance_repeatability"]["unsupported"],
            (not passed) and generator.last_relevance_observation.raw_response == "NO",
        )

    run_generation(
        "scenario_3",
        "What is the difference between the policy and the FAQ on who can access payroll records?",
        scenarios["scenario_3"],
    )
    run_generation(
        "scenario_4",
        "List all sections across the policy document",
        scenarios["scenario_4"],
    )

    follow_up = "Who is allowed to access them?"
    history = [
        {
            "question": "How long must payroll records be retained?",
            "answer": "Payroll records must be retained for seven years.",
        }
    ]
    label["value"] = "scenario_5_rewrite"
    rewrite_started = len(state["calls"])
    rewritten = rewrite_contextual_question(follow_up, history, client=client)
    assert_safe("scenario_5_rewrite", rewritten)

    retriever = Retriever(
        embedding_client=embed_client,
        vector_store=store,
        query_rewriter=QueryRewriter(),
        reranker=SimpleReranker(),
    )
    original_retrieve = retriever.retrieve

    def guarded_retrieve(*args, **kwargs):
        retrieved = original_retrieve(*args, **kwargs)
        for item in retrieved.search_results:
            assert_safe("scenario_5_retrieved", item.text)
            if item.metadata.get("file_name") not in ALLOWED_FILE_NAMES:
                raise RuntimeError("Downstream retrieval left the synthetic corpus.")
        return retrieved

    retriever.retrieve = guarded_retrieve
    pipeline = RAGPipeline(retriever=retriever, answer_generator=generator)
    label["value"] = "scenario_5_downstream"
    downstream_started = len(state["calls"])
    downstream = pipeline.answer_question(
        question=follow_up,
        retrieval_question=rewritten,
        top_k=5,
        min_score=0.35,
        conversation_context=(
            "Previous question: How long must payroll records be retained?\n"
            "Previous answer: Payroll records must be retained for seven years."
        ),
    )
    results["scenarios"]["scenario_5"] = {
        "follow_up": follow_up,
        "rewritten_question": rewritten,
        "rewrite_calls": state["calls"][rewrite_started:downstream_started],
        "log_retrieval_query": downstream.log.retrieval_query,
        "retriever_retrieval_query": (
            downstream.retrieval_result.retrieval_query
            if downstream.retrieval_result
            else None
        ),
        "clarification_triggered": downstream.log.clarification_triggered,
        "orchestration_intent": downstream.log.orchestration_intent,
        "answer": downstream.answer,
        "answer_status": downstream.answer_status,
        "grounding_check": downstream.log.grounding_check,
        "retrieval_confidence": downstream.retrieval_confidence,
        "selected_model": downstream.log.chat_model,
        "relevance_response": downstream.log.relevance_response,
        "relevance_finish_reason": downstream.log.relevance_finish_reason,
        "relevance_completion_tokens": downstream.log.relevance_completion_tokens,
        "relevance_reasoning_tokens": downstream.log.relevance_reasoning_tokens,
        "sources": [
            {
                "file_name": source.file_name,
                "chunk_index": source.chunk_index,
                "score": source.score,
            }
            for source in downstream.sources
        ],
        "downstream_calls": state["calls"][downstream_started:],
    }

    label["value"] = "scenario_6"
    rewrite_started = len(state["calls"])
    evaluation_rewriter = LLMQueryRewriter(client=client)
    rewritten_query = evaluation_rewriter.rewrite(
        "How long do we have to keep payroll records for?"
    )
    assert_safe("scenario_6_rewrite", rewritten_query)
    results["scenarios"]["scenario_6"] = {
        "selected_model": evaluation_rewriter._model,
        "original_query": "How long do we have to keep payroll records for?",
        "rewritten_query": rewritten_query,
        "calls": state["calls"][rewrite_started:],
    }

    run_generation(
        "cross_document_access",
        "Who can access payroll records?",
        scenarios["scenario_3"],
        answer_mode="standard",
    )

    results["actual_chat_calls"] = state["chat"]
    results["actual_embeddings"] = state["embeddings"]
    results["all_calls"] = state["calls"]
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results written to {RESULTS_PATH}")
    print(
        json.dumps(
            {
                "actual_chat_calls": state["chat"],
                "actual_embeddings": state["embeddings"],
                "default_model": generator.model_name,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
