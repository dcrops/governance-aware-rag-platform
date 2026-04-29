from tests.evaluation_queries import EVALUATION_CASES
from app.ingestion.ingest import ingest_document
from app.chunking.chunker import chunk_document

# Import EmbeddingClient, VectorRecord, VectorStore, and Retriever
from app.embeddings.embeddings import EmbeddingClient
from app.models.vector_record import VectorRecord
from app.vector_store.vector_store import VectorStore
from app.retrieval.retriever import Retriever
from app.orchestration.rag_pipeline import RAGPipeline
from app.generation.answer_generator import AnswerGenerator


def main():
    # EDIT THIS PATH FOR YOUR TEST FILE
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
    records = []

    for chunk, embedding in zip(chunks, embeddings):
        vr = VectorRecord(
            chunk=chunk,
            embedding=embedding,
        )
        records.append(vr)
    print(f"   -> Created {len(records)} VectorRecords")

    print("6. Initializing VectorStore...")
    vector_store = VectorStore()
    print("   -> Upserting records...")
    vector_store.upsert_records(records)
    print(f"   -> Upserted {len(records)} records")

    print("7. Initializing Retriever...")
    retriever = Retriever(embedding_client=embed_client, vector_store=vector_store)

    answer_generator = AnswerGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        answer_generator=answer_generator,
    )

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
        )

        print("\nAnswer:")
        print(response.answer)
        print(f"\nRetrieval Confidence: {response.retrieval_confidence}")
        print(f"Answer Status: {response.answer_status}")

        print("\nSources:")
        for source in response.sources:
            print(f"- {source.file_name} | chunk {source.chunk_index} | score {source.score:.3f}")

        if response.log:
            print("\nTelemetry:")
            print(f"Retrieved IDs: {response.log.retrieved_chunk_ids}")
            print(f"Scores: {[round(score, 3) for score in response.log.scores]}")

if __name__ == "__main__":
    main()