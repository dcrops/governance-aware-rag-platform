"""Cross-process retrieval freshness tests for Chroma PersistentClient cache."""
import multiprocessing as mp
import os
import sys
import time
import uuid

import chromadb
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIR = os.path.join(PROJECT_ROOT, "data", "index")


def _setup_path() -> None:
    os.chdir(PROJECT_ROOT)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    import truststore

    truststore.inject_into_ssl()


def _index_document(
    collection_name: str,
    file_name: str,
    record_id: str,
    unique_token: str,
    release_after_write: bool,
) -> int:
    _setup_path()
    from app.models.chunk import Chunk
    from app.models.vector_record import VectorRecord
    from app.embeddings.embeddings import EmbeddingClient
    from app.vector_store.vector_store import VectorStore

    vs = VectorStore(persist_dir=PERSIST_DIR, collection_name=collection_name)
    ec = EmbeddingClient()
    emb = ec.embed_query(unique_token)
    chunk = Chunk(
        chunk_id=record_id,
        doc_id=file_name,
        chunk_index=0,
        text=f"{unique_token} document body",
        metadata={"file_name": file_name, "file_type": "txt"},
    )
    vs.upsert_records([VectorRecord(chunk=chunk, embedding=emb)])
    count_after = vs.count()
    if release_after_write:
        vs.release_persisted_client()
    return count_after


def _warm_reader_count(collection_name: str, refresh_before_read: bool) -> int:
    _setup_path()
    from app.vector_store.vector_store import VectorStore

    chromadb.PersistentClient(path=PERSIST_DIR)
    if refresh_before_read:
        VectorStore.refresh_persisted_client(PERSIST_DIR)
    vs = VectorStore(persist_dir=PERSIST_DIR, collection_name=collection_name)
    return vs.count()


def _cleanup_collection(collection_name: str) -> None:
    _setup_path()
    from app.vector_store.vector_store import VectorStore

    VectorStore.refresh_persisted_client(PERSIST_DIR)
    vs = VectorStore(persist_dir=PERSIST_DIR, collection_name=collection_name)
    try:
        vs.delete_collection()
    except Exception:
        pass
    vs.release_persisted_client()


def _writer_stale_case(collection_name: str, done: mp.Event) -> None:
    _setup_path()
    from app.vector_store.vector_store import VectorStore

    vs = VectorStore(persist_dir=PERSIST_DIR, collection_name=collection_name)
    try:
        vs.delete_collection()
    except Exception:
        pass
    _index_document(
        collection_name,
        "doc_a.txt",
        "doc_a_chunk_0",
        "stale_without_refresh_token_a",
        release_after_write=False,
    )
    done.set()


def _reader_stale_case(collection_name: str, done: mp.Event, result_queue: mp.Queue) -> None:
    _setup_path()
    chromadb.PersistentClient(path=PERSIST_DIR)
    done.wait(timeout=120)
    result_queue.put(_warm_reader_count(collection_name, refresh_before_read=False))


def _writer_refresh_case(collection_name: str, done: mp.Event) -> None:
    _setup_path()
    from app.vector_store.vector_store import VectorStore

    vs = VectorStore(persist_dir=PERSIST_DIR, collection_name=collection_name)
    try:
        vs.delete_collection()
    except Exception:
        pass
    _index_document(
        collection_name,
        "doc_a.txt",
        "doc_a_chunk_0",
        "fresh_with_refresh_token_a",
        release_after_write=True,
    )
    done.set()


def _reader_refresh_case(collection_name: str, done: mp.Event, result_queue: mp.Queue) -> None:
    _setup_path()
    chromadb.PersistentClient(path=PERSIST_DIR)
    done.wait(timeout=120)
    result_queue.put(_warm_reader_count(collection_name, refresh_before_read=True))


def _writer_sequence_case(collection_name: str, done: mp.Event, stage: mp.Value) -> None:
    _setup_path()
    from app.vector_store.vector_store import VectorStore

    vs = VectorStore(persist_dir=PERSIST_DIR, collection_name=collection_name)
    try:
        vs.delete_collection()
    except Exception:
        pass
    _index_document(
        collection_name,
        "doc_a.txt",
        "doc_a_chunk_0",
        "sequence_token_doc_a",
        release_after_write=True,
    )
    with stage.get_lock():
        stage.value = 1
    done.wait(timeout=120)
    _index_document(
        collection_name,
        "doc_b.txt",
        "doc_b_chunk_0",
        "sequence_token_doc_b",
        release_after_write=True,
    )
    with stage.get_lock():
        stage.value = 2


def _reader_sequence_case(
    collection_name: str,
    done: mp.Event,
    stage: mp.Value,
    result_queue: mp.Queue,
) -> None:
    _setup_path()
    chromadb.PersistentClient(path=PERSIST_DIR)
    while True:
        with stage.get_lock():
            current = stage.value
        if current >= 1:
            break
        time.sleep(0.05)
    count_a = _warm_reader_count(collection_name, refresh_before_read=True)
    done.set()
    while True:
        with stage.get_lock():
            current = stage.value
        if current >= 2:
            break
        time.sleep(0.05)
    count_b = _warm_reader_count(collection_name, refresh_before_read=True)
    result_queue.put({"count_a": count_a, "count_b": count_b})


@pytest.fixture
def collection_name() -> str:
    return f"client_cross_process_test_{uuid.uuid4().hex[:8]}"


def test_stale_without_refresh(collection_name: str) -> None:
    """Documents that cross-process writes are persisted even if a warm reader cache exists."""
    mp.set_start_method("spawn", force=True)
    done = mp.Event()
    reader_q: mp.Queue = mp.Queue()
    reader = mp.Process(target=_reader_stale_case, args=(collection_name, done, reader_q))
    writer = mp.Process(target=_writer_stale_case, args=(collection_name, done))
    reader.start()
    writer.start()
    writer.join(timeout=120)
    reader.join(timeout=120)
    warm_count = reader_q.get(timeout=10)

    _setup_path()
    from app.vector_store.vector_store import VectorStore

    VectorStore.refresh_persisted_client(PERSIST_DIR)
    vs = VectorStore(persist_dir=PERSIST_DIR, collection_name=collection_name)
    cold_count = vs.count()
    _cleanup_collection(collection_name)

    assert cold_count == 1
    assert warm_count <= cold_count


def test_fresh_with_reader_refresh(collection_name: str) -> None:
    mp.set_start_method("spawn", force=True)
    done = mp.Event()
    reader_q: mp.Queue = mp.Queue()
    reader = mp.Process(target=_reader_refresh_case, args=(collection_name, done, reader_q))
    writer = mp.Process(target=_writer_refresh_case, args=(collection_name, done))
    reader.start()
    writer.start()
    writer.join(timeout=120)
    reader.join(timeout=120)
    warm_count = reader_q.get(timeout=10)
    _cleanup_collection(collection_name)
    assert warm_count == 1


def test_index_a_then_b_without_restart(collection_name: str) -> None:
    mp.set_start_method("spawn", force=True)
    done = mp.Event()
    stage = mp.Value("i", 0)
    reader_q: mp.Queue = mp.Queue()
    reader = mp.Process(
        target=_reader_sequence_case,
        args=(collection_name, done, stage, reader_q),
    )
    writer = mp.Process(target=_writer_sequence_case, args=(collection_name, done, stage))
    reader.start()
    writer.start()
    writer.join(timeout=180)
    reader.join(timeout=180)
    counts = reader_q.get(timeout=10)

    _setup_path()
    from app.vector_store.vector_store import VectorStore

    VectorStore.refresh_persisted_client(PERSIST_DIR)
    vs = VectorStore(persist_dir=PERSIST_DIR, collection_name=collection_name)
    final_count = vs.count()
    _cleanup_collection(collection_name)

    assert counts["count_a"] == 1
    assert counts["count_b"] == 2
    assert final_count == 2
