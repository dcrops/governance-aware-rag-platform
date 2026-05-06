import json
import os
from datetime import datetime, timezone

# from tests.evaluation_queries import EVALUATION_CASES
from tests.evaluation_queries_run_sheet_pdf import EVALUATION_CASES

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


REWRITE_STRATEGY = "rule"  # options: "none", "rule", "llm"
CHUNKING_STRATEGY = "delimiter"  # options: "character", "page", "heading", "delimiter"
DELIMITER = "BREAK"  # None


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

    # file_path = "data/raw/sample.txt"
    file_path = "data/evaluation_docs/run_sheet_style.pdf"

    print("1. Ingesting document...")
    doc = ingest_document(file_path)
    print(f"   -> Document ID: {doc.doc_id}")

    print(f"2. Chunking document using strategy: {CHUNKING_STRATEGY}")
    chunks = build_chunks(doc, CHUNKING_STRATEGY)
    print(f"   -> Total Chunks: {len(chunks)}")
    if not chunks:
        raise RuntimeError("No chunks created from document.")

    print("3. Initializing embedding client...")
    embed_client = EmbeddingClient()

    print("4. Embedding chunk texts...")
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embed_client.embed_texts(chunk_texts)
    print(f"   -> Got {len(embeddings)} embeddings")

    if len(chunks) != len(embeddings):
        raise RuntimeError("Mismatch between chunks and embeddings.")

    print("5. Creating VectorRecords...")
    records = [
        VectorRecord(chunk=chunk, embedding=embedding)
        for chunk, embedding in zip(chunks, embeddings)
    ]
    print(f"   -> Created {len(records)} VectorRecords")

    print("6. Initializing VectorStore...")
    vector_store = VectorStore()
    print("   -> Upserting records...")
    vector_store.upsert_records(records)
    print(f"   -> Upserted {len(records)} records")

    print("7. Initializing Retriever...")
    query_rewriter = build_query_rewriter(REWRITE_STRATEGY)
    print(f"   -> Query rewrite strategy: {REWRITE_STRATEGY}")

    retriever = Retriever(
        embedding_client=embed_client,
        vector_store=vector_store,
        query_rewriter=query_rewriter,
    )

    answer_generator = AnswerGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        answer_generator=answer_generator,
    )

    telemetry_logger = TelemetryLogger()

    print("8. Running evaluation queries...")

    for idx, case in enumerate(EVALUATION_CASES, start=1):
        question = case["question"]
        expected_documents = case.get("expected_documents", [])
        expected_topics = case["expected_topics"]
        expected_answer_status = case.get("expected_answer_status")

        print("\n" + "=" * 80)
        print(f"Evaluation Query {idx}: {question}")
        print("=" * 80)

        print("\nExpected Topics:")
        for topic in expected_topics:
            print(f"- {topic}")

        response = pipeline.answer_question(
            question,
            top_k=5,
            min_score=0.35,
            metadata_filter={"file_name": doc.metadata.get("file_name")},
        )

        print("\nAnswer:")
        print(response.answer)
        print(f"\nRetrieval Confidence: {response.retrieval_confidence}")
        print(f"Answer Status: {response.answer_status}")

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

        retrieved_text = ""

        if response.retrieval_result:
            retrieved_text = " ".join(
                result.text.lower()
                for result in response.retrieval_result.search_results
            )

        matched_topics = [
            topic
            for topic in expected_topics
            if topic.lower() in retrieved_text
        ]

        missing_topics = [
            topic
            for topic in expected_topics
            if topic.lower() not in retrieved_text
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