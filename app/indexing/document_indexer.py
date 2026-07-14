import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from app.chunking.chunker import (
    chunk_document,
    chunk_document_by_delimiter,
    chunk_document_by_heading,
    chunk_document_by_page,
)
from app.document_management.document_registry import save_document_record
from app.embeddings.embeddings import EmbeddingClient
from app.ingestion.diagnostics import run_ingestion_diagnostics
from app.ingestion.diagnostics_store import save_document_diagnostics
from app.ingestion.ingest import ingest_document
from app.models.vector_record import VectorRecord
from app.vector_store.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class DocumentIndexResult:
    file_name: str
    success: bool
    chunk_count: int = 0
    error_message: str | None = None
    document_record: dict[str, Any] | None = None
    diagnostics: Any | None = None


def _chunk_document(
    doc,
    *,
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    delimiter: str | None,
):
    if chunking_strategy == "character":
        return chunk_document(doc, chunk_size=chunk_size, overlap=chunk_overlap)
    if chunking_strategy == "delimiter":
        return chunk_document_by_delimiter(doc, delimiter=delimiter)
    if chunking_strategy == "page":
        return chunk_document_by_page(doc)
    if chunking_strategy == "heading":
        return chunk_document_by_heading(doc)
    raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")


def _user_friendly_error(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        return "Indexing failed for an unknown reason."
    if len(message) > 240:
        return message[:237] + "..."
    return message


def index_single_document(
    *,
    file_path: str,
    file_name: str,
    vector_store: VectorStore,
    persist_dir: str,
    client_name: str,
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    delimiter: str | None,
    embedding_client: EmbeddingClient,
    remove_existing_version: bool = True,
) -> DocumentIndexResult:
    """
    Index one document through extract → chunk → embed → vector upsert → registry update.
    """
    try:
        if remove_existing_version:
            existing_documents = vector_store.list_documents()
            existing_file_names = {doc["file_name"] for doc in existing_documents}
            if file_name in existing_file_names:
                vector_store.delete_document(file_name)
                from app.ingestion.diagnostics_store import delete_document_diagnostics
                from app.document_management.document_registry import delete_document_record

                delete_document_diagnostics(
                    persist_dir=persist_dir,
                    client_name=client_name,
                    file_name=file_name,
                )
                delete_document_record(
                    persist_dir=persist_dir,
                    client_name=client_name,
                    file_name=file_name,
                )

        doc = ingest_document(file_path)
        doc.metadata["file_name"] = file_name
        doc.metadata["original_file_name"] = file_name

        chunks = _chunk_document(
            doc,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            delimiter=delimiter,
        )

        if not chunks:
            raise RuntimeError("Failed to extract non-empty text")

        diagnostics = run_ingestion_diagnostics(doc, chunks)
        save_document_diagnostics(
            persist_dir=persist_dir,
            client_name=client_name,
            file_name=file_name,
            diagnostics=diagnostics,
        )

        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = embedding_client.embed_texts(chunk_texts)

        if len(embeddings) != len(chunks):
            raise RuntimeError("Number of embeddings does not match number of chunks.")

        records = [
            VectorRecord(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ]

        vector_store.upsert_records(records)

        document_record = {
            "file_name": file_name,
            "file_type": doc.file_type,
            "chunk_count": len(chunks),
            "chunking_strategy": chunking_strategy,
            "readiness_status": diagnostics.readiness_status,
            "readiness_score": diagnostics.readiness_score,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }

        save_document_record(
            persist_dir=persist_dir,
            client_name=client_name,
            record=document_record,
        )

        return DocumentIndexResult(
            file_name=file_name,
            success=True,
            chunk_count=len(chunks),
            document_record=document_record,
            diagnostics=diagnostics,
        )

    except Exception as exc:
        logger.exception("Failed to index document '%s'", file_name)
        return DocumentIndexResult(
            file_name=file_name,
            success=False,
            error_message=_user_friendly_error(exc),
        )


@dataclass
class BatchIndexResult:
    results: list[DocumentIndexResult]

    @property
    def successful_results(self) -> list[DocumentIndexResult]:
        return [result for result in self.results if result.success]

    @property
    def failed_results(self) -> list[DocumentIndexResult]:
        return [result for result in self.results if not result.success]

    @property
    def total_chunks(self) -> int:
        return sum(result.chunk_count for result in self.successful_results)


def clear_client_index(
    *,
    persist_dir: str,
    client_name: str,
    collection_name: str,
) -> VectorStore:
    """Clear a client's Chroma collection, registry, and diagnostics once."""
    from app.ingestion.diagnostics_store import delete_client_diagnostics
    from app.document_management.document_registry import delete_client_registry

    vector_store = VectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )

    try:
        vector_store.delete_collection()
    except Exception:
        pass

    delete_client_diagnostics(persist_dir=persist_dir, client_name=client_name)
    delete_client_registry(persist_dir=persist_dir, client_name=client_name)

    return VectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )


def index_document_batch(
    *,
    persist_dir: str,
    client_name: str,
    collection_name: str,
    file_items: list[tuple[str, str]],
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    delimiter: str | None,
    replace_existing_index: bool = False,
    embedding_client: EmbeddingClient | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> BatchIndexResult:
    """
    Index multiple documents for one client in a single action.

    Each file is processed independently. Failures do not roll back prior successes.
    """
    if not file_items:
        return BatchIndexResult(results=[])

    vector_store: VectorStore | None = None
    results: list[DocumentIndexResult] = []

    try:
        vector_store = VectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )

        if replace_existing_index:
            vector_store = clear_client_index(
                persist_dir=persist_dir,
                client_name=client_name,
                collection_name=collection_name,
            )

        embed_client = embedding_client or EmbeddingClient()
        total_files = len(file_items)

        for index, (file_path, file_name) in enumerate(file_items, start=1):
            if progress_callback is not None:
                progress_callback(index, total_files, file_name)

            result = index_single_document(
                file_path=file_path,
                file_name=file_name,
                vector_store=vector_store,
                persist_dir=persist_dir,
                client_name=client_name,
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                delimiter=delimiter,
                embedding_client=embed_client,
                remove_existing_version=not replace_existing_index,
            )
            results.append(result)

        return BatchIndexResult(results=results)

    finally:
        if vector_store is not None:
            vector_store.release_persisted_client()
