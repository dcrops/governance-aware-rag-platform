import os
from uuid import uuid4
import hashlib

from app.models.document import Document

def ingest_document(file_path: str) -> Document:
    """
    Ingests a .txt file and returns a Document object.
    
    Raises exceptions for invalid or unsupported files.
    """
    # Check if file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check file extension
    _, ext = os.path.splitext(file_path)
    if ext.lower() != ".txt":
        raise ValueError("Only .txt files are supported.")

    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise ValueError("File is empty.")
    max_size_bytes = 10 * 1024 * 1024  # 10 MB
    if file_size > max_size_bytes:
        raise ValueError("File exceeds maximum allowed size of 10 MB.")

    # Read file content
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        raise IOError(f"Failed to read file as UTF-8: {e}")

    if not text or text.strip() == "":
        raise ValueError("Extracted text is empty or whitespace only.")

    doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
    file_name = os.path.basename(file_path)
    metadata = {
        "file_name": file_name,
        "file_size": file_size,
        "file_type": "txt",
    }

    return Document(
        doc_id=doc_id,
        source_path=file_path,
        file_type="txt",
        raw_text=text,
        metadata=metadata,
    )