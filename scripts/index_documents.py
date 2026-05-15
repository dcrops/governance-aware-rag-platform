import os
import argparse
from app.ingestion.ingest import ingest_document
from app.chunking.chunker import chunk_document
from app.embeddings.embeddings import EmbeddingClient
from app.models.vector_record import VectorRecord
from app.vector_store.vector_store import VectorStore
from app.config import PERSIST_DIR


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--doc-path",
        required=True,
        help="Path to document for indexing",
    )

    args = parser.parse_args()

    doc_path = args.doc_path

    client_name = os.getenv(
        "RAG_DEFAULT_CLIENT_NAME",
        "default_client",
    )

    persist_dir = PERSIST_DIR
    collection_name = f"client_{client_name}"

    print(f"[Indexing] Starting indexing for client: '{client_name}'")
    print(f"[Indexing] Document path: {doc_path}")

    print("1. Ingesting document...")
    doc = ingest_document(doc_path)
    print(f"   -> Document ID: {doc.doc_id}")

    print("2. Chunking document...")
    chunks = chunk_document(doc)
    print(f"   -> Total chunks: {len(chunks)}")
    if not chunks:
        raise RuntimeError("No chunks created from document. Aborting indexing.")

    print("3. Initializing embedding client...")
    embedding_client = EmbeddingClient()

    print("4. Embedding chunk texts...")
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embedding_client.embed_texts(chunk_texts)
    print(f"   -> Got {len(embeddings)} embeddings")

    if len(chunks) != len(embeddings):
        raise RuntimeError(
            f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})."
        )

    print("5. Creating VectorRecords...")
    records = [
        VectorRecord(chunk=chunk, embedding=embedding)
        for chunk, embedding in zip(chunks, embeddings)
    ]
    print(f"   -> Created {len(records)} VectorRecords")

    print("6. Initializing VectorStore...")
    vector_store = VectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name
    )

    print(f"   -> Upserting records into collection '{collection_name}'...")
    vector_store.upsert_records(records)
    print(f"   -> Upserted {len(records)} records successfully.")

    print(f"\n[Indexing] Indexing complete for client '{client_name}'.")
    print(f"[Indexing] Data persisted to '{persist_dir}', collection '{collection_name}'.")

if __name__ == "__main__":
    main()