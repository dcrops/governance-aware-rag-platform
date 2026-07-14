import os
import sys
import tempfile
import uuid

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _setup_path() -> None:
    os.chdir(PROJECT_ROOT)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    import truststore

    truststore.inject_into_ssl()


def _write_text_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


@pytest.fixture
def temp_persist_dir(tmp_path):
    _setup_path()
    persist_dir = tmp_path / "index"
    persist_dir.mkdir()
    (persist_dir / "registries").mkdir()
    (persist_dir / "diagnostics").mkdir()
    return str(persist_dir)


@pytest.fixture
def batch_client_name() -> str:
    return f"batch_test_{uuid.uuid4().hex[:8]}"


def _index_batch(
    persist_dir: str,
    client_name: str,
    file_items: list[tuple[str, str]],
    *,
    replace_existing_index: bool = False,
):
    from app.document_management.client_registry import collection_name_for_client
    from app.indexing.document_indexer import index_document_batch

    return index_document_batch(
        persist_dir=persist_dir,
        client_name=client_name,
        collection_name=collection_name_for_client(client_name),
        file_items=file_items,
        chunking_strategy="character",
        chunk_size=500,
        chunk_overlap=50,
        delimiter=None,
        replace_existing_index=replace_existing_index,
    )


def _reader_count(persist_dir: str, client_name: str) -> int:
    from app.document_management.client_registry import collection_name_for_client
    from app.vector_store.vector_store import VectorStore

    VectorStore.refresh_persisted_client(persist_dir)
    vector_store = VectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name_for_client(client_name),
    )
    return vector_store.count()


def _cleanup_client(persist_dir: str, client_name: str) -> None:
    from app.document_management.client_registry import collection_name_for_client
    from app.vector_store.vector_store import VectorStore

    VectorStore.refresh_persisted_client(persist_dir)
    vector_store = VectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name_for_client(client_name),
    )
    try:
        vector_store.delete_collection()
    except Exception:
        pass
    vector_store.release_persisted_client()


def test_two_valid_documents_indexed_in_one_action(
    temp_persist_dir: str,
    batch_client_name: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        doc_a = os.path.join(tmp_dir, "Policy_A.txt")
        doc_b = os.path.join(tmp_dir, "Policy_B.txt")
        _write_text_file(doc_a, "Alpha policy content about academic integrity.")
        _write_text_file(
            doc_b,
            "Beta policy content about workplace safety and reporting.",
        )

        batch_result = _index_batch(
            temp_persist_dir,
            batch_client_name,
            [(doc_a, "Policy_A.txt"), (doc_b, "Policy_B.txt")],
        )

    try:
        assert len(batch_result.successful_results) == 2
        assert not batch_result.failed_results
        assert batch_result.total_chunks >= 2
        assert _reader_count(temp_persist_dir, batch_client_name) == batch_result.total_chunks
    finally:
        _cleanup_client(temp_persist_dir, batch_client_name)


def test_partial_failure_keeps_successful_document_retrievable(
    temp_persist_dir: str,
    batch_client_name: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        valid_doc = os.path.join(tmp_dir, "Valid.txt")
        invalid_doc = os.path.join(tmp_dir, "Empty.txt")
        _write_text_file(valid_doc, "Distinctive valid token for retrieval testing.")
        with open(invalid_doc, "wb"):
            pass

        batch_result = _index_batch(
            temp_persist_dir,
            batch_client_name,
            [(valid_doc, "Valid.txt"), (invalid_doc, "Empty.txt")],
        )

    try:
        assert len(batch_result.successful_results) == 1
        assert len(batch_result.failed_results) == 1
        assert batch_result.successful_results[0].file_name == "Valid.txt"
        assert batch_result.failed_results[0].file_name == "Empty.txt"
        assert _reader_count(temp_persist_dir, batch_client_name) > 0
    finally:
        _cleanup_client(temp_persist_dir, batch_client_name)


def test_replace_existing_index_clears_once_then_indexes_batch(
    temp_persist_dir: str,
    batch_client_name: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        seed_doc = os.path.join(tmp_dir, "Seed.txt")
        doc_a = os.path.join(tmp_dir, "Policy_A.txt")
        doc_b = os.path.join(tmp_dir, "Policy_B.txt")
        _write_text_file(seed_doc, "Seed document that should be replaced.")
        _write_text_file(doc_a, "Replacement document A with unique content.")
        _write_text_file(doc_b, "Replacement document B with different content.")

        seed_result = _index_batch(
            temp_persist_dir,
            batch_client_name,
            [(seed_doc, "Seed.txt")],
        )
        assert len(seed_result.successful_results) == 1

        replace_result = _index_batch(
            temp_persist_dir,
            batch_client_name,
            [(doc_a, "Policy_A.txt"), (doc_b, "Policy_B.txt")],
            replace_existing_index=True,
        )

    try:
        from app.document_management.document_registry import load_registry

        registry = load_registry(temp_persist_dir, batch_client_name)
        assert "Seed.txt" not in registry["documents"]
        assert set(registry["documents"]) == {"Policy_A.txt", "Policy_B.txt"}
        assert len(replace_result.successful_results) == 2
        assert _reader_count(temp_persist_dir, batch_client_name) == replace_result.total_chunks
    finally:
        _cleanup_client(temp_persist_dir, batch_client_name)


def test_second_batch_appends_when_replace_not_selected(
    temp_persist_dir: str,
    batch_client_name: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        doc_a = os.path.join(tmp_dir, "Policy_A.txt")
        doc_b = os.path.join(tmp_dir, "Policy_B.txt")
        _write_text_file(doc_a, "First batch document A content.")
        _write_text_file(doc_b, "Second batch document B content.")

        first_result = _index_batch(
            temp_persist_dir,
            batch_client_name,
            [(doc_a, "Policy_A.txt")],
        )
        second_result = _index_batch(
            temp_persist_dir,
            batch_client_name,
            [(doc_b, "Policy_B.txt")],
        )

    try:
        from app.document_management.document_registry import load_registry

        registry = load_registry(temp_persist_dir, batch_client_name)
        assert set(registry["documents"]) == {"Policy_A.txt", "Policy_B.txt"}
        assert len(first_result.successful_results) == 1
        assert len(second_result.successful_results) == 1
        assert _reader_count(temp_persist_dir, batch_client_name) == (
            first_result.total_chunks + second_result.total_chunks
        )
    finally:
        _cleanup_client(temp_persist_dir, batch_client_name)


def test_batch_index_release_allows_immediate_cross_process_read(
    temp_persist_dir: str,
    batch_client_name: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        doc_path = os.path.join(tmp_dir, "Immediate.txt")
        _write_text_file(doc_path, "Immediate retrieval token without restart.")

        batch_result = _index_batch(
            temp_persist_dir,
            batch_client_name,
            [(doc_path, "Immediate.txt")],
        )

    try:
        assert len(batch_result.successful_results) == 1
        assert _reader_count(temp_persist_dir, batch_client_name) == batch_result.total_chunks
    finally:
        _cleanup_client(temp_persist_dir, batch_client_name)
