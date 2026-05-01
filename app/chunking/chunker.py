from app.models.document import Document
from app.models.chunk import Chunk
import re


def chunk_document(
    doc: Document,
    chunk_size: int = 1500,
    overlap: int = 150,
) -> list[Chunk]:
    """
    Splits a Document's raw_text into overlapping fixed-size chunks.

    Args:
        doc: Document object to chunk.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of Chunk objects.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

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

def chunk_document_by_delimiter(document, delimiter: str) -> list["Chunk"]:
    """
    Chunk a document into sections using a specified delimiter string.
    Each chunk starts with the delimiter text and includes all content up to the next delimiter.

    Args:
        document: Document to be chunked.
        delimiter: Non-empty string used to break the document into chunks.

    Returns:
        List of Chunk objects.

    Raises:
        ValueError: If delimiter is empty/whitespace or not found in the document.
    """
    # Validate delimiter
    if not isinstance(delimiter, str) or not delimiter.strip():
        raise ValueError("Delimiter must be a non-empty string.")

    text = document.raw_text
    doc_id = document.doc_id
    file_name = document.metadata.get("file_name", "")
    file_type = document.file_type

    # Find all positions of the delimiter in the text
    matches = [m.start() for m in re.finditer(re.escape(delimiter), text)]
    if not matches:
        raise ValueError(f"Delimiter '{delimiter}' not found in document.")

    chunks: list["Chunk"] = []
    chunk_index = 0

    for i, delim_start in enumerate(matches):
        # The chunk starts at the delimiter, ends at the next delimiter or end of text
        chunk_start = delim_start
        chunk_end = matches[i + 1] if i + 1 < len(matches) else len(text)
        chunk_text = text[chunk_start:chunk_end]
        if not chunk_text or chunk_text.strip() == "":
            continue  # ignore empty or whitespace-only chunks

        chunk_id = f"{doc_id}_chunk_{chunk_index}"
        chunk_metadata = {
            "doc_id": doc_id,
            "file_name": file_name,
            "file_type": file_type,
            "chunk_index": chunk_index,
            "char_start": chunk_start,
            "char_end": chunk_end,
            "chunking_strategy": "delimiter",
            "delimiter": delimiter,
        }
        chunk = Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            chunk_index=chunk_index,
            text=chunk_text,
            metadata=chunk_metadata,
        )
        chunks.append(chunk)
        chunk_index += 1

    if not chunks:
        raise ValueError("No non-empty chunks were found using the given delimiter.")

    return chunks

def chunk_document_by_page(doc: Document) -> list[Chunk]:
    """
    Splits a PDF Document into one chunk per extracted page.

    Requires PDF ingestion to store page text in doc.metadata["pages"].
    """

    pages = doc.metadata.get("pages")

    if not pages:
        raise ValueError(
            "Page-based chunking requires page metadata. "
            "Use this strategy with PDFs ingested through the PDF extractor."
        )

    chunks: list[Chunk] = []

    for chunk_index, page in enumerate(pages):
        page_number = page.get("page_number")
        page_text = page.get("text", "")

        if not page_text or not page_text.strip():
            continue

        chunk_metadata = {
            "doc_id": doc.doc_id,
            "file_name": doc.metadata.get("file_name", ""),
            "file_type": doc.file_type,
            "chunk_index": chunk_index,
            "page_number": page_number,
            "chunking_strategy": "page",
        }

        chunk = Chunk(
            chunk_id=f"{doc.doc_id}_page_{page_number}",
            doc_id=doc.doc_id,
            chunk_index=chunk_index,
            text=page_text,
            metadata=chunk_metadata,
        )

        chunks.append(chunk)

    if not chunks:
        raise ValueError("No non-empty page chunks were created.")

    return chunks