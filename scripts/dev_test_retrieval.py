from app.ingestion.ingest import ingest_document
from app.chunking.chunker import chunk_document

# Import EmbeddingClient, VectorRecord, VectorStore, and Retriever
from app.embeddings.embeddings import EmbeddingClient
from app.models.vector_record import VectorRecord
from app.vector_store.vector_store import VectorStore
from app.retrieval.retriever import Retriever
from app.orchestration.rag_pipeline import RAGPipeline
from app.generation.answer_generator import AnswerGenerator

def print_search_results(results):
    print("\n=== Retrieval Results ===")
    if not results:
        print("NO RESULTS FOUND")
        raise RuntimeError("No search results returned by retriever.")
    for idx, res in enumerate(results):
        print(f"\nResult {idx + 1}:")
        print(f"  ID: {getattr(res, 'id', None) or getattr(res, 'chunk_id', None)}")
        print(f"  Score: {getattr(res, 'score', None)}")
        print(f"  Metadata: {getattr(res, 'metadata', {})}")
        text = getattr(res, "text", None)
        if text is not None:
            preview = res.text[:300].replace("\n", " ")
        else:
            preview = getattr(res, 'chunk', {}).get('text', '')[:300]
        print(f"  Text Preview: {preview!r}")

def main():
    # EDIT THIS PATH FOR YOUR TEST FILE
    file_path = "data/raw/sample.txt"
    test_query = "What is this document about?"

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

    print("8. Running test retrieval query:")
    print(f"   -> Query: {test_query!r}")

    answer_generator = AnswerGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        answer_generator=answer_generator,
    )
    
    response = pipeline.answer_question(test_query, top_k=5)

    print("\n=== GENERATED ANSWER ===")
    print(response.answer)

    print("\n=== SOURCES ===")
    for idx, source in enumerate(response.sources, start=1):
        print(f"  ID: {source.id}")
        print(f"  File: {source.file_name}")
        print(f"  Chunk: {source.chunk_index}")
        print(f"  Score: {source.score}")
        char_start = source.metadata.get("char_start")
        char_end = source.metadata.get("char_end")

        if char_start is not None and char_end is not None:
            print(f"  Location: chars {char_start}-{char_end}")

        if source.text_preview:
            print(f"  Preview: {source.text_preview}")

        if response.log:
            print("\n=== TELEMETRY LOG ===")
            print(f"Question: {response.log.question}")
            print(f"Retrieved IDs: {response.log.retrieved_chunk_ids}")
            print(f"Scores: {response.log.scores}")
            print(f"Timestamp: {response.log.timestamp}")

        print("-" * 50)

if __name__ == "__main__":
    main()