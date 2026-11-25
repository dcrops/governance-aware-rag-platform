"""
Narrow evaluation adapter for one controlled gpt-4.1-mini vs gpt-6-luna comparison.

This does not change production model defaults, prompts, token budgets, retrieval
thresholds, or operational indexes. Chat calls are not retried with alternate
parameters. OpenAI client retries are disabled.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
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

from app.chunking.chunker import chunk_document_by_delimiter
from app.embeddings.embeddings import EmbeddingClient
from app.generation.answer_generator import AnswerGenerator
from app.ingestion.ingest import ingest_document
from app.models.vector_record import VectorRecord
from app.orchestration.rag_pipeline import RAGPipeline
from app.query_processing.llm_query_rewriter import LLMQueryRewriter
from app.query_processing.query_rewriter import QueryRewriter
from app.reranking.simple_reranker import SimpleReranker
from app.retrieval.retriever import Retriever
from app.vector_store.vector_store import SearchResult, VectorStore

OPERATIONAL_INDEX = (PROJECT_ROOT / "data" / "index").resolve()
PERSIST_DIR = (PROJECT_ROOT / "logs" / "model_comparison_index").resolve()
COLLECTION_NAME = "synthetic_workplace"
MANIFEST_PATH = PROJECT_ROOT / "logs" / "evaluation_runs" / "model_comparison_manifest.json"
RESULTS_PATH = PROJECT_ROOT / "logs" / "evaluation_runs" / "model_comparison_20261125.json"

SOURCE_DOCUMENTS = [
    {
        "path": PROJECT_ROOT / "data" / "evaluation_docs" / "policy_style.txt",
        "delimiter": "SECTION:",
    },
    {
        "path": PROJECT_ROOT / "data" / "evaluation_docs" / "faq_style.txt",
        "delimiter": "QUESTION:",
    },
]

TOP_K = 5
MIN_SCORE = 0.35
ALLOWED_FILE_NAMES = {"policy_style.txt", "faq_style.txt"}
MAX_CHAT_CALLS = 24
MAX_COMPARE_EMBEDDINGS = 2

MODELS = [
    {"id": "gpt-4.1-mini", "profile": "baseline"},
    {"id": "gpt-6-luna", "profile": "luna"},
]

# Planned request policy. Each live call also stores the kwargs actually sent.
PARAMETER_PROFILES = {
    "baseline": {
        "model": "gpt-4.1-mini",
        "api": "chat.completions",
        "reasoning_effort": "omitted",
        "token_limit_parameter": "max_tokens",
        "temperature": "preserved from the existing call site",
        "top_p": "preserved when the existing call site sends it",
        "stop": "preserved when the existing call site sends it",
        "n": "preserved when the existing call site sends it",
        "budgets": {
            "relevance": 5,
            "answer": 512,
            "contextual_rewrite": 80,
            "evaluation_query_rewrite": 64,
        },
    },
    "luna": {
        "model": "gpt-6-luna",
        "api": "chat.completions",
        "reasoning_effort": "none",
        "token_limit_parameter": "max_completion_tokens",
        "temperature": "omitted",
        "top_p": "omitted",
        "stop": "omitted",
        "n": "preserved when the existing call site sends it",
        "budgets": {
            "relevance": 5,
            "answer": 512,
            "contextual_rewrite": 80,
            "evaluation_query_rewrite": 64,
        },
    },
}

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


def assert_isolated_storage() -> None:
    if PERSIST_DIR == OPERATIONAL_INDEX or OPERATIONAL_INDEX in PERSIST_DIR.parents:
        raise RuntimeError(
            f"Refusing to use operational index path: {PERSIST_DIR}"
        )


def assert_safe_text(label: str, text: str) -> None:
    lowered = text.lower()
    for pattern in BANNED_PATTERNS:
        if pattern in lowered:
            raise RuntimeError(
                f"Safety check failed for {label}: found {pattern!r}."
            )
    if " cv " in f" {lowered} ":
        raise RuntimeError(f"Safety check failed for {label}: found a CV reference.")


def make_client() -> OpenAI:
    return OpenAI(max_retries=0, timeout=120.0)


def apply_profile(kwargs: dict, profile: str, model_id: str) -> dict:
    effective = dict(kwargs)
    effective["model"] = model_id
    if profile == "baseline":
        return effective
    if profile != "luna":
        raise ValueError(f"Unknown parameter profile: {profile}")

    effective.pop("temperature", None)
    effective.pop("top_p", None)
    effective.pop("stop", None)
    if "max_tokens" in effective:
        effective["max_completion_tokens"] = effective.pop("max_tokens")
    effective["reasoning_effort"] = "none"
    return effective


def usage_payload(response) -> dict | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {"unparsed": str(usage)}


def reasoning_tokens(usage: dict | None) -> int | None:
    if not isinstance(usage, dict):
        return None
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict) or "reasoning_tokens" not in details:
        return None
    return details.get("reasoning_tokens")


def serialise_error(exc: Exception) -> dict:
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    if not isinstance(body, (dict, list, str, int, float, type(None))):
        body = str(body)
    return {
        "type": type(exc).__name__,
        "status_code": status,
        "message": str(exc)[:2000],
        "body": body,
    }


class CallLog:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record(self, entry: dict) -> None:
        self.calls.append(entry)
        if len(self.calls) > MAX_CHAT_CALLS:
            raise RuntimeError("Chat call cap exceeded. Stopping without another request.")


def invoke_chat(inner_create, profile: str, model_id: str, kwargs: dict, label: str, call_log: CallLog):
    effective = apply_profile(kwargs, profile, model_id)
    safe_params = {
        key: value
        for key, value in effective.items()
        if key != "messages"
    }
    started = time.perf_counter()
    try:
        response = inner_create(**effective)
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        call_log.record(
            {
                "label": label,
                "ok": False,
                "latency_ms": latency_ms,
                "requested_model": model_id,
                "effective_parameters": safe_params,
                "error": serialise_error(exc),
            }
        )
        raise

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    choice = response.choices[0]
    message = choice.message
    content = message.content
    usage = usage_payload(response)
    call_log.record(
        {
            "label": label,
            "ok": True,
            "latency_ms": latency_ms,
            "requested_model": model_id,
            "returned_model": getattr(response, "model", None),
            "effective_parameters": safe_params,
            "finish_reason": choice.finish_reason,
            "content": content,
            "content_is_empty": content is None or str(content).strip() == "",
            "refusal": getattr(message, "refusal", None),
            "usage": usage,
            "prompt_tokens": None if usage is None else usage.get("prompt_tokens"),
            "completion_tokens": None if usage is None else usage.get("completion_tokens"),
            "reasoning_tokens": reasoning_tokens(usage),
        }
    )
    return response


def patch_create(owner, profile: str, model_id: str, label_box: dict, call_log: CallLog) -> None:
    completions = owner.chat.completions
    original = completions.create

    def create(**kwargs):
        return invoke_chat(
            original,
            profile,
            model_id,
            kwargs,
            label_box["label"],
            call_log,
        )

    completions.create = create


def dump_results(results: list[SearchResult]) -> list[dict]:
    payload = []
    for result in results:
        payload.append(
            {
                "id": result.id,
                "text": result.text,
                "score": result.score,
                "vector_score": result.vector_score,
                "rerank_bonus": result.rerank_bonus,
                "final_score": result.final_score,
                "metadata": result.metadata,
            }
        )
    return payload


def load_results(payload: list[dict]) -> list[SearchResult]:
    return [SearchResult(**item) for item in payload]


def result_brief(results: list[SearchResult]) -> list[dict]:
    return [
        {
            "id": result.id,
            "file_name": result.metadata.get("file_name"),
            "chunk_index": result.metadata.get("chunk_index"),
            "score": result.score,
            "text": result.text,
        }
        for result in results
    ]


def existing_answer_status(answer: str) -> str:
    """Same wording rule currently used by RAGPipeline."""
    if "don't have enough information" in answer.lower():
        return "INSUFFICIENT_EVIDENCE"
    return "ANSWERED"


def build_retriever(store: VectorStore, embed_client: EmbeddingClient) -> Retriever:
    return Retriever(
        embedding_client=embed_client,
        vector_store=store,
        query_rewriter=QueryRewriter(),
        reranker=SimpleReranker(),
    )


def retrieve_like_pipeline(
    pipeline: RAGPipeline,
    question: str,
    metadata_filter: dict | None = None,
    retrieval_mode: str = "Standard chunk retrieval",
    selected_documents: list[str] | None = None,
) -> dict:
    decision = pipeline.intent_classifier.classify(question, None)
    routing = pipeline.retrieval_router.route(decision, TOP_K)
    if decision.clarification_required:
        return {
            "intent": decision.intent.value,
            "answer_mode": decision.answer_mode,
            "retrieval_strategy": routing.retrieval_strategy,
            "effective_top_k": routing.effective_top_k,
            "reasoning": decision.reasoning,
            "chunks": [],
        }

    top_k = routing.effective_top_k
    if retrieval_mode == "Document-level retrieval":
        if not selected_documents:
            raise ValueError("Document-level retrieval requires selected documents.")
        chunks_per_document = max(1, top_k // len(selected_documents))
        search_results = pipeline.retrieve_per_document(
            question,
            selected_documents,
            chunks_per_document=chunks_per_document,
            min_score=MIN_SCORE,
        )
    else:
        strategy = pipeline.strategy_selector.select(routing.retrieval_strategy)
        retrieval = strategy.retrieve(
            retriever=pipeline.retriever,
            query=question,
            top_k=top_k,
            min_score=MIN_SCORE,
            metadata_filter=metadata_filter,
            selected_documents=selected_documents,
        )
        search_results = retrieval.search_results
        if decision.intent != "aggregation":
            search_results = pipeline._prune_generation_context(search_results)

    return {
        "intent": decision.intent.value,
        "answer_mode": decision.answer_mode,
        "retrieval_strategy": routing.retrieval_strategy,
        "effective_top_k": top_k,
        "reasoning": decision.reasoning,
        "retrieval_mode": retrieval_mode,
        "chunks": dump_results(search_results),
    }


def index_synthetic_corpus(embed_client: EmbeddingClient) -> dict:
    assert_isolated_storage()
    store = VectorStore(persist_dir=str(PERSIST_DIR), collection_name=COLLECTION_NAME)
    indexed_chunks = []

    for source in SOURCE_DOCUMENTS:
        document = ingest_document(str(source["path"]))
        assert document.metadata["file_name"] in ALLOWED_FILE_NAMES
        assert_safe_text(document.metadata["file_name"], document.raw_text)
        chunks = chunk_document_by_delimiter(document, source["delimiter"])
        indexed_chunks.extend(chunks)

    for chunk in indexed_chunks:
        file_name = chunk.metadata.get("file_name")
        if file_name not in ALLOWED_FILE_NAMES:
            raise RuntimeError(f"Unexpected file in synthetic index: {file_name}")
        assert_safe_text(file_name, chunk.text)

    vectors = embed_client.embed_texts([chunk.text for chunk in indexed_chunks])
    store.upsert_records(
        [
            VectorRecord(chunk=chunk, embedding=vector)
            for chunk, vector in zip(indexed_chunks, vectors)
        ]
    )
    listed = store.list_documents()
    listed_names = {item["file_name"] for item in listed}
    if listed_names != ALLOWED_FILE_NAMES:
        raise RuntimeError(f"Isolated index contains unexpected files: {listed_names}")

    return {
        "persist_dir": str(PERSIST_DIR),
        "collection_name": COLLECTION_NAME,
        "chunk_count": store.count(),
        "documents": listed,
        "embedding_calls": 1,
    }


def prepare() -> None:
    assert_isolated_storage()
    if MANIFEST_PATH.exists():
        print(f"Manifest already exists: {MANIFEST_PATH}")
        print("Prepare was not run again, so no additional embedding call was made.")
        return

    embed_client = EmbeddingClient()
    embed_client.client = make_client()
    index_summary = index_synthetic_corpus(embed_client)
    store = VectorStore(persist_dir=str(PERSIST_DIR), collection_name=COLLECTION_NAME)
    pipeline = RAGPipeline(
        retriever=build_retriever(store, embed_client),
        answer_generator=AnswerGenerator(model_name="gpt-4.1-mini"),
    )

    scenarios = {
        "scenario_1": retrieve_like_pipeline(
            pipeline,
            "How long must payroll records be retained?",
            metadata_filter={"file_name": "policy_style.txt"},
        ),
        "scenario_2_natural": retrieve_like_pipeline(
            pipeline,
            "What university did the author attend?",
        ),
        "scenario_3": retrieve_like_pipeline(
            pipeline,
            "What is the difference between the policy and the FAQ on who can access payroll records?",
        ),
        "scenario_4": retrieve_like_pipeline(
            pipeline,
            "List all sections across the policy document",
            metadata_filter={"file_name": "policy_style.txt"},
        ),
    }
    embedding_calls = 1 + 4

    scenario_3_files = {
        chunk["metadata"].get("file_name")
        for chunk in scenarios["scenario_3"]["chunks"]
    }
    if scenario_3_files != ALLOWED_FILE_NAMES:
        scenarios["scenario_3_document_level"] = retrieve_like_pipeline(
            pipeline,
            "What is the difference between the policy and the FAQ on who can access payroll records?",
            retrieval_mode="Document-level retrieval",
            selected_documents=["policy_style.txt", "faq_style.txt"],
        )
        embedding_calls += 2
        scenarios["scenario_3_selected"] = "scenario_3_document_level"
    else:
        scenarios["scenario_3_selected"] = "scenario_3"

    natural_unsupported = scenarios["scenario_2_natural"]["chunks"]
    if natural_unsupported:
        scenarios["scenario_2_selected"] = "scenario_2_natural"
        scenarios["scenario_2"] = scenarios["scenario_2_natural"]
    else:
        scenarios["scenario_2_selected"] = "scenario_1_context_for_relevance"
        scenarios["scenario_2"] = {
            **scenarios["scenario_1"],
            "question_override": "What university did the author attend?",
            "answer_mode": "standard",
            "note": (
                "Natural retrieval returned no chunks above the existing minimum score. "
                "The relevance call uses the scenario 1 policy chunks, which do not state a university."
            ),
        }

    for name, scenario in scenarios.items():
        if not isinstance(scenario, dict) or "chunks" not in scenario:
            continue
        for chunk in scenario["chunks"]:
            assert_safe_text(name, chunk["text"])
            file_name = chunk["metadata"].get("file_name")
            if file_name not in ALLOWED_FILE_NAMES:
                raise RuntimeError(f"{name} retrieved unexpected file {file_name}")

    required = ["scenario_1", "scenario_2", "scenario_3", "scenario_4"]
    if scenarios["scenario_3_selected"] == "scenario_3_document_level":
        required.append("scenario_3_document_level")
    for name in required:
        if not scenarios[name]["chunks"]:
            raise RuntimeError(f"{name} produced no retrieved context. Comparison stopped.")

    manifest = {
        "index": index_summary,
        "embedding_calls_prepare": embedding_calls,
        "top_k": TOP_K,
        "min_score": MIN_SCORE,
        "sources": [str(source["path"].relative_to(PROJECT_ROOT)) for source in SOURCE_DOCUMENTS],
        "scenarios": scenarios,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(MANIFEST_PATH), "embedding_calls_prepare": embedding_calls}, indent=2))
    for name in ("scenario_1", "scenario_2", scenarios["scenario_3_selected"], "scenario_4"):
        chunks = scenarios[name]["chunks"]
        print(f"\n{name}: {len(chunks)} chunks")
        for chunk in chunks:
            print(
                f"- {chunk['metadata'].get('file_name')} "
                f"chunk {chunk['metadata'].get('chunk_index')} "
                f"score {chunk['score']:.4f}"
            )
            print(chunk["text"])


def contextual_prompt(question: str, conversation_context: str) -> str:
    """Same prompt text as streamlit_app.build_contextual_question."""
    return f"""
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


def call_record(call_log: CallLog, start_index: int) -> list[dict]:
    return call_log.calls[start_index:]


def compare() -> None:
    assert_isolated_storage()
    if not MANIFEST_PATH.exists():
        raise RuntimeError("Missing manifest. Run prepare before compare.")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for name in ("scenario_1", "scenario_2", manifest["scenarios"]["scenario_3_selected"], "scenario_4"):
        for chunk in manifest["scenarios"][name]["chunks"]:
            assert_safe_text(name, chunk["text"])

    anticipated = {
        "chat_calls_if_relevance_passes_and_unsupported_refuses": 22,
        "chat_calls_upper_bound_if_unsupported_also_answers": 24,
        "chat_call_cap": MAX_CHAT_CALLS,
        "compare_embedding_calls": 2,
        "retries": "disabled",
        "per_model": {
            "scenario_1": "1 relevance + 1 answer",
            "scenario_2": "1 relevance, plus 1 answer only if relevance returns YES",
            "scenario_3": "1 relevance + 1 answer",
            "scenario_4": "1 relevance + 1 answer",
            "scenario_5": "1 contextual rewrite + 1 downstream relevance + 1 downstream answer",
            "scenario_6": "1 evaluation rewrite",
        },
    }
    print("ANTICIPATED API CALLS")
    print(json.dumps(anticipated, indent=2))

    embed_client = EmbeddingClient()
    embed_client.client = make_client()
    embedding_calls = {"count": 0}
    original_embed_query = embed_client.embed_query

    def counted_embed_query(query: str):
        assert_safe_text("embedding query", query)
        if embedding_calls["count"] >= MAX_COMPARE_EMBEDDINGS:
            raise RuntimeError("Embedding call cap exceeded.")
        embedding_calls["count"] += 1
        return original_embed_query(query)

    embed_client.embed_query = counted_embed_query
    store = VectorStore(persist_dir=str(PERSIST_DIR), collection_name=COLLECTION_NAME)
    listed_names = {item["file_name"] for item in store.list_documents()}
    if listed_names != ALLOWED_FILE_NAMES:
        raise RuntimeError(f"Isolated collection changed: {listed_names}")

    results = {
        "parameter_profiles": PARAMETER_PROFILES,
        "anticipated": anticipated,
        "manifest": str(MANIFEST_PATH),
        "models": {},
    }

    previous_question = "How long must payroll records be retained?"
    previous_answer = "Payroll records must be retained for seven years."
    follow_up = "Who is allowed to access them?"
    conversation_context = (
        f"Previous question: {previous_question}\n"
        f"Previous answer: {previous_answer}"
    )
    assert_safe_text("scenario_5_context", conversation_context)
    assert_safe_text("scenario_5_question", follow_up)

    for model in MODELS:
        model_id = model["id"]
        profile = model["profile"]
        call_log = CallLog()
        model_result = {"scenarios": {}, "stopped_early": False}
        results["models"][model_id] = model_result

        label_box = {"label": "unspecified"}
        generator = AnswerGenerator(model_name=model_id)
        generator.client = make_client()
        patch_create(generator.client, profile, model_id, label_box, call_log)

        rewriter = LLMQueryRewriter(model=model_id)
        rewriter._client = make_client()
        patch_create(rewriter._client, profile, model_id, label_box, call_log)

        direct_client = make_client()
        patch_create(direct_client, profile, model_id, label_box, call_log)

        retriever = build_retriever(store, embed_client)
        pipeline = RAGPipeline(retriever=retriever, answer_generator=generator)

        def run_generation(scenario_name: str, question: str, scenario_payload: dict) -> None:
            label_box["label"] = scenario_name
            start = len(call_log.calls)
            chunks = load_results(scenario_payload["chunks"])
            try:
                answer, grounding = generator.generate_answer(
                    question=question,
                    search_results=chunks,
                    answer_mode=scenario_payload["answer_mode"],
                )
                error = None
            except Exception as exc:
                answer = None
                grounding = None
                error = serialise_error(exc)
                error["traceback"] = traceback.format_exc(limit=5)
            model_result["scenarios"][scenario_name] = {
                "question": question,
                "answer_mode": scenario_payload["answer_mode"],
                "intent": scenario_payload.get("intent"),
                "retrieval_strategy": scenario_payload.get("retrieval_strategy"),
                "retrieval_confidence": pipeline._calculate_confidence(chunks),
                "chunks": result_brief(chunks),
                "answer": answer,
                "grounding_check": grounding,
                "answer_status_by_existing_wording_rule": (
                    None if answer is None else existing_answer_status(answer)
                ),
                "calls": call_record(call_log, start),
                "error": error,
            }

        try:
            run_generation(
                "scenario_1",
                "How long must payroll records be retained?",
                manifest["scenarios"]["scenario_1"],
            )
            scenario_2 = manifest["scenarios"]["scenario_2"]
            run_generation(
                "scenario_2",
                scenario_2.get("question_override", "What university did the author attend?"),
                scenario_2,
            )
            selected_name = manifest["scenarios"]["scenario_3_selected"]
            run_generation(
                "scenario_3",
                "What is the difference between the policy and the FAQ on who can access payroll records?",
                manifest["scenarios"][selected_name],
            )
            run_generation(
                "scenario_4",
                "List all sections across the policy document",
                manifest["scenarios"]["scenario_4"],
            )

            label_box["label"] = "scenario_5_rewrite"
            start = len(call_log.calls)
            rewrite_error = None
            rewritten = None
            try:
                response = direct_client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": contextual_prompt(follow_up, conversation_context),
                        }
                    ],
                    temperature=0,
                    max_tokens=80,
                )
                rewritten = response.choices[0].message.content
                if rewritten is not None:
                    rewritten = rewritten.strip()
            except Exception as exc:
                rewrite_error = serialise_error(exc)

            downstream = None
            if rewritten:
                assert_safe_text("scenario_5_rewrite", rewritten)
                label_box["label"] = "scenario_5_downstream"
                downstream_start = len(call_log.calls)
                try:
                    downstream_response = pipeline.answer_question(
                        question=follow_up,
                        retrieval_question=rewritten,
                        top_k=TOP_K,
                        min_score=MIN_SCORE,
                        conversation_context=conversation_context,
                    )
                    retrieved = []
                    if downstream_response.retrieval_result is not None:
                        retrieved = result_brief(
                            downstream_response.retrieval_result.search_results
                        )
                        for item in retrieved:
                            assert_safe_text("scenario_5_retrieved", item["text"])
                    downstream = {
                        "answer": downstream_response.answer,
                        "answer_status": downstream_response.answer_status,
                        "grounding_check": (
                            downstream_response.log.grounding_check
                            if downstream_response.log
                            else None
                        ),
                        "retrieval_confidence": downstream_response.retrieval_confidence,
                        "log_retrieval_query": (
                            downstream_response.log.retrieval_query
                            if downstream_response.log
                            else None
                        ),
                        "retriever_retrieval_query": (
                            downstream_response.retrieval_result.retrieval_query
                            if downstream_response.retrieval_result
                            else None
                        ),
                        "clarification_triggered": (
                            downstream_response.log.clarification_triggered
                            if downstream_response.log
                            else None
                        ),
                        "orchestration_intent": (
                            downstream_response.log.orchestration_intent
                            if downstream_response.log
                            else None
                        ),
                        "retrieved_chunks": retrieved,
                        "generation_sources": [
                            {
                                "file_name": source.file_name,
                                "chunk_index": source.chunk_index,
                                "score": source.score,
                                "preview": source.text_preview,
                            }
                            for source in downstream_response.sources
                        ],
                        "calls": call_record(call_log, downstream_start),
                    }
                except Exception as exc:
                    downstream = {"error": serialise_error(exc)}

            model_result["scenarios"]["scenario_5"] = {
                "follow_up": follow_up,
                "conversation_context": conversation_context,
                "rewritten_question": rewritten,
                "rewrite_error": rewrite_error,
                "rewrite_consumed_by_pipeline_log": (
                    downstream is not None
                    and downstream.get("log_retrieval_query") == rewritten
                ),
                "downstream": downstream,
                "calls": call_record(call_log, start),
            }

            label_box["label"] = "scenario_6"
            start = len(call_log.calls)
            rewrite_query = "How long do we have to keep payroll records for?"
            assert_safe_text("scenario_6_query", rewrite_query)
            rewrite_error = None
            rewritten = None
            try:
                rewritten = rewriter.rewrite(rewrite_query)
                assert_safe_text("scenario_6_rewrite", rewritten)
            except Exception as exc:
                rewrite_error = serialise_error(exc)
            model_result["scenarios"]["scenario_6"] = {
                "original_query": rewrite_query,
                "rewritten_query": rewritten,
                "error": rewrite_error,
                "calls": call_record(call_log, start),
            }
        except Exception as exc:
            model_result["stopped_early"] = True
            model_result["stop_error"] = serialise_error(exc)

        model_result["all_calls"] = call_log.calls
        RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    results["actual_chat_calls"] = sum(
        len(model_result.get("all_calls", []))
        for model_result in results["models"].values()
    )
    results["actual_compare_embedding_calls"] = embedding_calls["count"]
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results written to {RESULTS_PATH}")
    print(
        json.dumps(
            {
                "actual_chat_calls": results["actual_chat_calls"],
                "actual_compare_embedding_calls": results["actual_compare_embedding_calls"],
            },
            indent=2,
        )
    )


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"prepare", "compare"}:
        raise SystemExit("Usage: python scripts/compare_chat_models.py prepare|compare")
    if sys.argv[1] == "prepare":
        prepare()
    else:
        compare()


if __name__ == "__main__":
    main()
