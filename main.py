from app.ingestion.ingest import ingest_document
from app.chunking.chunker import chunk_document


def main() -> None:
    print("Running main...")
    file_path = "data/raw/sample.txt"

    document = ingest_document(file_path)
    chunks = chunk_document(document)

    print(f"Document ID: {document.doc_id}")
    print(f"File: {document.metadata.get('file_name')}")
    print(f"Total chunks: {len(chunks)}")
    print("-" * 50)

    for chunk in chunks[:2]:
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Chunk Index: {chunk.chunk_index}")
        print(f"Char Range: {chunk.metadata.get('char_start')} -> {chunk.metadata.get('char_end')}")
        print(f"Text Preview: {chunk.text[:100]!r}")
        print("-" * 50)


if __name__ == "__main__":
    main()