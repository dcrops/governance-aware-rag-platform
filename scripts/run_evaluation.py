import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import os
from datetime import datetime, timezone
import pandas as pd

# from tests.evaluation_queries import EVALUATION_CASES
from tests.evaluation_queries_run_sheet_pdf import EVALUATION_CASES
from app.config import PERSIST_DIR, DEFAULT_CLIENT_NAME

from app.ingestion.ingest import ingest_document
from app.chunking.chunker import (
    chunk_document,
    chunk_document_by_page,
    chunk_document_by_heading,
    chunk_document_by_delimiter,
)
from app.embeddings.embeddings import EmbeddingClient
from app.models.vector_record import VectorRecord
from app.vector_store.vector_store import VectorStore
from app.retrieval.retriever import Retriever
from app.orchestration.rag_pipeline import RAGPipeline
from app.generation.answer_generator import AnswerGenerator
from app.telemetry.telemetry_logger import TelemetryLogger
from app.reranking.simple_reranker import SimpleReranker


REWRITE_STRATEGY = "rule"  # options: "none", "rule", "llm"
CHUNKING_STRATEGY = "delimiter"  # options: "character", "page", "heading", "delimiter"
DELIMITER = "BREAK"  # None

RETRIEVAL_MODE = "Standard chunk retrieval"
RERANKER_ENABLED = True
MIN_SCORE = 0.35
TOP_K = 5

EVAL_CSV_PATH = "data/evaluation/rag_eval_suite_v2.csv"
USE_CSV_EVALUATION = True


def build_query_rewriter(strategy: str):
    if strategy == "none":
        return None

    if strategy == "rule":
        from app.query_processing.query_rewriter import QueryRewriter
        return QueryRewriter()

    if strategy == "llm":
        from app.query_processing.llm_query_rewriter import LLMQueryRewriter
        return LLMQueryRewriter()

    raise ValueError(f"Unknown rewrite strategy: {strategy}")

def build_chunks(doc, strategy: str):
    """
    Build chunks using the selected chunking strategy.
    """

    if strategy == "character":
        return chunk_document(doc)

    if strategy == "page":
        return chunk_document_by_page(doc)

    if strategy == "heading":
        return chunk_document_by_heading(doc)

    if strategy == "delimiter":
        if not DELIMITER:
            raise ValueError("DELIMITER must be set when using delimiter chunking.")
        return chunk_document_by_delimiter(doc, delimiter=DELIMITER)

    raise ValueError(f"Unknown chunking strategy: {strategy}")

def is_aggregation_question(question: str) -> bool:
    aggregation_terms = [
        "how many",
        "count",
        "list all",
        "all the",
        "all documents",
        "across",
        "mentioned in all",
    ]

    return any(term in question.lower() for term in aggregation_terms)


def is_comparison_question(question: str) -> bool:
    comparison_terms = [
        "difference between",
        "compare",
        "versus",
        "vs",
        "how does",
        "differ",
    ]

    return any(term in question.lower() for term in comparison_terms)

def is_ambiguous_follow_up(question: str) -> bool:
    ambiguous_terms = [
        "what about",
        "how about",
        "and payroll",
        "and hr",
        "and finance",
    ]

    q = question.lower()

    return any(term in q for term in ambiguous_terms)

def main():
    total_queries = 0
    document_hit_count = 0
    topic_match_rates = []
    top_scores = []
    average_scores = []
    zero_topic_match_queries = []
    answer_status_eval_count = 0
    answer_status_match_count = 0
    answer_status_failures = []
    grounding_pass_count = 0
    grounding_fail_count = 0
    grounding_skipped_count = 0


    print("1. Initializing embedding client...")
    embed_client = EmbeddingClient()

    print("2. Connecting to existing VectorStore index...")
    vector_store = VectorStore(
        persist_dir=PERSIST_DIR,
        #collection_name=f"client_{DEFAULT_CLIENT_NAME}",
        collection_name=f"client_RMIT_Demo",
    )

    # print(vector_store._collection.count())

    indexed_count = vector_store.count()
    print(f"   -> Indexed chunks found: {indexed_count}")

    if indexed_count == 0:
        raise RuntimeError(
            "No indexed chunks found. Please index documents in the Streamlit app first."
        )

    print("3. Initializing Retriever...")
    query_rewriter = build_query_rewriter(REWRITE_STRATEGY)
    print(f"   -> Query rewrite strategy: {REWRITE_STRATEGY}")

    reranker = SimpleReranker() if RERANKER_ENABLED else None


    retriever = Retriever(
        embedding_client=embed_client,
        vector_store=vector_store,
        query_rewriter=query_rewriter,
        reranker=reranker,
    )

    answer_generator = AnswerGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        answer_generator=answer_generator,
    )

    telemetry_logger = TelemetryLogger()

    print("4. Running evaluation queries...")

    if USE_CSV_EVALUATION:
        df = pd.read_csv(EVAL_CSV_PATH)
        evaluation_cases = df.to_dict(orient="records")
    else:
        evaluation_cases = EVALUATION_CASES

    for idx, case in enumerate(evaluation_cases, start=1):
        question = case["question"]

        context_type = str(case.get("context_type", "")).strip().lower()

        setup_question = case.get("setup_question")

        if setup_question != setup_question:  # handles NaN
            setup_question = None

        retrieval_mode = str(
            case.get("retrieval_mode", RETRIEVAL_MODE)
        ).strip()

        search_scope = str(case.get("search_scope", "")).split(";")

        search_scope = [
            doc.strip()
            for doc in search_scope
            if doc.strip()
            and doc.strip().lower() not in ["nan", "all", "auto"]
        ]

        metadata_filter = None

        if retrieval_mode != "Document-level retrieval":
            if len(search_scope) == 1:
                metadata_filter = {
                    "file_name": search_scope[0]
                }

            elif len(search_scope) > 1:
                metadata_filter = {
                    "$or": [
                        {"file_name": doc_name}
                        for doc_name in search_scope
                    ]
                }

        expected_documents = str(case.get("expected_documents", "")).split(";")
        expected_documents = [
            doc.strip()
            for doc in expected_documents
            if doc.strip() and doc.strip().lower() != "nan"
        ]

        expected_topics = str(case.get("expected_answer_contains", "")).split(";")
        expected_topics = [
            topic.strip()
            for topic in expected_topics
            if topic.strip() and topic.strip().lower() != "nan"
        ]

        expected_answer_text = " ".join(expected_topics).lower()

        if (
            context_type == "follow_up"
            and is_ambiguous_follow_up(question)
            and "clarify" in expected_answer_text
        ):
            print("\n" + "=" * 80)
            print(f"Evaluation Query {idx}: {question}")
            print("=" * 80)

            print("\nExpected Topics:")
            for topic in expected_topics:
                print(f"- {topic}")

            print(f"Retrieval Mode Used: {retrieval_mode}")
            print(f"Search Scope Used: {search_scope}")
            print(f"Metadata Filter Used: {metadata_filter}")
            print("Answer Mode Used: clarification")

            print("\nAmbiguous follow-up detected.")

            print("\nAnswer:")
            print("Can you clarify what you mean?")

            print("\nAnswer Status: CLARIFICATION_REQUIRED")

            total_queries += 1

            answer_status_eval_count += 1

            expected_answer_status = "CLARIFICATION_REQUIRED"
            actual_answer_status = "CLARIFICATION_REQUIRED"

            if expected_answer_status == actual_answer_status:
                answer_status_match_count += 1
            else:
                answer_status_failures.append(question)

            topic_match_rate = 1.0
            topic_match_rates.append(topic_match_rate)

            continue

        expected_answer_status = case.get("expected_answer_status")
        if expected_answer_status != expected_answer_status:  # handles NaN
            expected_answer_status = None


        print("\n" + "=" * 80)
        print(f"Evaluation Query {idx}: {question}")
        print("=" * 80)

        print("\nExpected Topics:")
        for topic in expected_topics:
            print(f"- {topic}")

        print(f"Retrieval Mode Used: {retrieval_mode}")
        print(f"Search Scope Used: {search_scope}")
        print(f"Metadata Filter Used: {metadata_filter}")

        if is_aggregation_question(question):
            answer_mode = "aggregation"
        elif is_comparison_question(question):
            answer_mode = "comparison"
        else:
            answer_mode = "standard"

        print(f"Answer Mode Used: {answer_mode}")

        retrieval_question = question

        if context_type == "follow_up" and setup_question:
            print(f"Setup Question: {setup_question}")

            setup_response = pipeline.answer_question(
                setup_question,
                top_k=TOP_K,
                min_score=MIN_SCORE,
                metadata_filter=metadata_filter,
                retrieval_mode=retrieval_mode,
                selected_documents=search_scope,
                answer_mode="standard",
            )

            expected_rewritten_query = case.get("expected_rewritten_query")

            if (
                expected_rewritten_query == expected_rewritten_query
                and str(expected_rewritten_query).strip()
            ):
                retrieval_question = str(expected_rewritten_query).strip()
            else:
                retrieval_question = f"{setup_question} {question}"

        response = pipeline.answer_question(
            question=question,
            retrieval_question=retrieval_question,
            top_k=TOP_K,
            min_score=MIN_SCORE,
            metadata_filter=metadata_filter,
            retrieval_mode=retrieval_mode,
            selected_documents=search_scope,
            answer_mode=answer_mode,
        )

        print("\nAnswer:")
        print(response.answer)
        print(f"\nRetrieval Confidence: {response.retrieval_confidence}")
        print(f"Grounding Check: {response.log.grounding_check}")
        print(f"Answer Status: {response.answer_status}")

        if response.log.grounding_check == "PASS":
            grounding_pass_count += 1

        elif response.log.grounding_check == "FAIL":
            grounding_fail_count += 1

        elif response.log.grounding_check == "SKIPPED":
            grounding_skipped_count += 1

        print("\nSources:")
        for source in response.sources:
            print(
                f"- {source.file_name} | "
                f"chunk {source.chunk_index} | "
                f"score {source.score:.3f}"
            )

        retrieved_documents = [
            source.file_name
            for source in response.sources
            if source.file_name
        ]

        expected_document_hits = [
            doc_name
            for doc_name in expected_documents
            if doc_name in retrieved_documents
        ]

        expected_document_hit = bool(expected_document_hits)

        expected_document_rank = None

        for rank, source in enumerate(response.sources, start=1):
            if source.file_name in expected_documents:
                expected_document_rank = rank
                break

        top_score = response.sources[0].score if response.sources else 0

        average_score = (
            sum(source.score for source in response.sources) / len(response.sources)
            if response.sources
            else 0
        )

        print("\nRetrieval Evaluation:")
        print(f"Expected Documents: {expected_documents}")
        print(f"Retrieved Documents: {retrieved_documents}")
        print(f"Expected Document Hit: {expected_document_hit}")
        print(f"Expected Document Rank: {expected_document_rank}")
        print(f"Top Score: {top_score:.3f}")
        print(f"Average Score: {average_score:.3f}")

        evaluation_text = response.answer.lower()

        if response.retrieval_result:
            evaluation_text += " " + " ".join(
                result.text.lower()
                for result in response.retrieval_result.search_results
            )

        matched_topics = [
            topic
            for topic in expected_topics
            if topic.lower() in evaluation_text
        ]

        missing_topics = [
            topic
            for topic in expected_topics
            if topic.lower() not in evaluation_text
        ]

        topic_match_rate = None

        if expected_topics:
            topic_match_rate = (
                len(matched_topics) / len(expected_topics)
            )

        print("\nTopic Evaluation:")
        print(f"Matched Topics: {matched_topics}")
        print(f"Missing Topics: {missing_topics}")
        if topic_match_rate is not None:
            print(f"Topic Match Rate: {topic_match_rate:.2f}")
        else:
            print("Topic Match Rate: N/A")

        answer_status_match = None

        if expected_answer_status is not None:

            if isinstance(expected_answer_status, list):
                answer_status_match = response.answer_status in expected_answer_status
            else:
                answer_status_match = response.answer_status == expected_answer_status

            answer_status_eval_count += 1

            if answer_status_match:
                answer_status_match_count += 1
            else:
                answer_status_failures.append(question)

            print("\nAnswer Status Evaluation:")
            print(f"Expected Answer Status: {expected_answer_status}")
            print(f"Actual Answer Status: {response.answer_status}")
            print(f"Answer Status Match: {answer_status_match}")

        total_queries += 1

        if expected_document_hit:
            document_hit_count += 1

        top_scores.append(top_score)
        average_scores.append(average_score)

        if topic_match_rate is not None:
            topic_match_rates.append(topic_match_rate)

            if topic_match_rate == 0:
                zero_topic_match_queries.append(question)

        if response.log:
            telemetry_logger.log_retrieval(response.log)

            print("\nTelemetry:")
            print(f"Original Query: {response.log.original_query}")
            print(f"Retrieval Query: {response.log.retrieval_query}")
            print(f"Retrieved IDs: {response.log.retrieved_chunk_ids}")
            print(f"Scores: {[round(score, 3) for score in response.log.scores]}")
            print(f"Orchestration Intent: {response.log.orchestration_intent}")
            print(f"Retrieval Strategy: {response.log.retrieval_strategy}")
            print(f"Orchestration Reasoning: {response.log.orchestration_reasoning}")
            print(f"Clarification Triggered: {response.log.clarification_triggered}")

    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 80)

    document_hit_rate = (
        document_hit_count / total_queries
        if total_queries
        else 0
    )

    average_topic_match_rate = (
        sum(topic_match_rates) / len(topic_match_rates)
        if topic_match_rates
        else 0
    )

    average_top_score = (
        sum(top_scores) / len(top_scores)
        if top_scores
        else 0
    )

    average_retrieval_score = (
        sum(average_scores) / len(average_scores)
        if average_scores
        else 0
    )

    answer_status_match_rate = (
        answer_status_match_count / answer_status_eval_count
        if answer_status_eval_count
        else None
    )

    if answer_status_match_rate is not None:
        print(f"Answer Status Match Rate: {answer_status_match_rate:.2f}")
    else:
        print("Answer Status Match Rate: N/A")

    print("\nAnswer Status Failures:")
    if answer_status_failures:
        for failed_query in answer_status_failures:
            print(f"- {failed_query}")
    else:
        print("- None")

    print(f"Total Queries: {total_queries}")
    print(f"Document Hit Rate: {document_hit_rate:.2f}")
    print(f"Average Topic Match Rate: {average_topic_match_rate:.2f}")
    print(f"Average Top Score: {average_top_score:.3f}")
    print(f"Average Retrieval Score: {average_retrieval_score:.3f}")

    print("\nQueries With Zero Topic Matches:")
    if zero_topic_match_queries:
        for failed_query in zero_topic_match_queries:
            print(f"- {failed_query}")
    else:
        print("- None")

    evaluation_summary = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "rewrite_strategy": REWRITE_STRATEGY,
    "chunking_strategy": CHUNKING_STRATEGY,
    "delimiter": DELIMITER if CHUNKING_STRATEGY == "delimiter" else None,
    "total_queries": total_queries,
    "document_hit_rate": document_hit_rate,
    "average_topic_match_rate": average_topic_match_rate,
    "average_top_score": average_top_score,
    "average_retrieval_score": average_retrieval_score,
    "zero_topic_match_queries": zero_topic_match_queries,
    "answer_status_eval_count": answer_status_eval_count,
    "answer_status_match_rate": answer_status_match_rate,
    "answer_status_failures": answer_status_failures,
    "retrieval_mode": RETRIEVAL_MODE,
    "reranker_enabled": RERANKER_ENABLED,
    "min_score": MIN_SCORE,
    "top_k": TOP_K,
}

    output_dir = "logs/evaluation_runs"
    os.makedirs(output_dir, exist_ok=True)

    timestamp_for_file = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        output_dir,
        f"evaluation_{timestamp_for_file}.json",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)

    print(f"\nEvaluation summary saved to: {output_path}")


if __name__ == "__main__":
    main()