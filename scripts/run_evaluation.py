from tests.evaluation_queries import EVALUATION_CASES

from app.ingestion.ingest import ingest_document
from app.chunking.chunker import chunk_document
from app.embeddings.embeddings import EmbeddingClient
from app.models.vector_record import VectorRecord
from app.vector_store.vector_store import VectorStore
from app.retrieval.retriever import Retriever
from app.orchestration.rag_pipeline import RAGPipeline
from app.generation.answer_generator import AnswerGenerator
from app.telemetry.telemetry_logger import TelemetryLogger


REWRITE_STRATEGY = "rule"  # options: "none", "rule", "llm"


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


def main():
    file_path = "data/raw/sample.txt"

    print("1. Ingesting document...")
    doc = ingest_document(file_path)
    print(f"   -> Document ID: {doc.doc_id}")

    print("2. Chunking document...")
    chunks = chunk_document(doc)
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
        expected_topics = case["expected_topics"]

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
            metadata_filter={"file_name": "sample.txt"},
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

        if response.log:
            telemetry_logger.log_retrieval(response.log)

            print("\nTelemetry:")
            print(f"Original Query: {response.log.original_query}")
            print(f"Retrieval Query: {response.log.retrieval_query}")
            print(f"Retrieved IDs: {response.log.retrieved_chunk_ids}")
            print(f"Scores: {[round(score, 3) for score in response.log.scores]}")


if __name__ == "__main__":
    main()