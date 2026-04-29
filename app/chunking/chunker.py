from app.models.document import Document
from app.models.chunk import Chunk

def chunk_document(doc: Document) -> list[Chunk]:
    """
    Splits a Document's raw_text into overlapping fixed-size chunks.

    Args:
        doc: Document object to chunk.

    Returns:
        List of Chunk objects.
    """
    chunk_size = 1000
    overlap = 150
    step = chunk_size - overlap

    text = doc.raw_text
    doc_id = doc.doc_id
    file_name = doc.metadata.get("file_name", "")
    file_type = doc.file_type

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0

    if not doc.raw_text or doc.raw_text.strip() == "":
        raise ValueError("Document raw_text is empty or whitespace-only.")

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunk_id = f"{doc_id}_chunk_{chunk_index}"

        chunk_metadata = {
            "file_name": file_name,
            "file_type": file_type,
            "char_start": start,
            "char_end": end,
        }

        chunk = Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            chunk_index=chunk_index,
            text=chunk_text,
            metadata=chunk_metadata
        )
        chunks.append(chunk)

        if end == len(text):
            break
        start += step
        chunk_index += 1

    return chunks