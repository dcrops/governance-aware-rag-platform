import json
import os
import sys
import uuid

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _setup_path() -> None:
    os.chdir(PROJECT_ROOT)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def temp_persist_dir(tmp_path):
    _setup_path()
    persist_dir = tmp_path / "index"
    persist_dir.mkdir()
    (persist_dir / "registries").mkdir()
    return str(persist_dir)


def _write_registry(persist_dir: str, client_name: str, documents: dict | None = None) -> None:
    registry_path = os.path.join(
        persist_dir,
        "registries",
        f"client_{client_name}_registry.json",
    )
    payload = {
        "client_name": client_name,
        "documents": documents or {},
    }
    with open(registry_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_discover_clients_from_registries(temp_persist_dir: str) -> None:
    from app.document_management.client_registry import discover_clients

    _write_registry(temp_persist_dir, "RMIT_Demo_01", {"Policy_A.pdf": {}})
    _write_registry(temp_persist_dir, "Batch_Index_Demo", {"Handbook.docx": {}})

    clients = discover_clients(temp_persist_dir)

    assert clients == ["Batch_Index_Demo", "RMIT_Demo_01"]


def test_collection_name_mapping() -> None:
    from app.document_management.client_registry import (
        client_name_from_collection,
        collection_name_for_client,
    )

    assert collection_name_for_client("RMIT_Demo_01") == "client_RMIT_Demo_01"
    assert client_name_from_collection("client_RMIT_Demo_01") == "RMIT_Demo_01"
    assert client_name_from_collection("internal_collection") is None


def test_excluded_internal_clients_are_not_discovered(temp_persist_dir: str) -> None:
    from app.document_management.client_registry import discover_clients

    _write_registry(temp_persist_dir, "cross_process_test_abc", {})
    _write_registry(temp_persist_dir, "stale_fix_demo", {})
    _write_registry(temp_persist_dir, "RMIT_Demo_01", {})

    clients = discover_clients(temp_persist_dir)

    assert clients == ["RMIT_Demo_01"]


def test_validate_blank_and_invalid_client_names() -> None:
    from app.document_management.client_registry import validate_client_name

    is_valid, error = validate_client_name("   ")
    assert not is_valid
    assert "empty" in error.lower()

    is_valid, error = validate_client_name("Bad/Name")
    assert not is_valid
    assert "\\" in error


def test_validate_duplicate_client_case_insensitive() -> None:
    from app.document_management.client_registry import validate_client_name

    is_valid, error = validate_client_name(
        " rmit_demo_01 ",
        existing_clients=["RMIT_Demo_01"],
    )

    assert not is_valid
    assert "already exists" in error.lower()


def test_resolve_active_client_name_for_existing_and_new() -> None:
    from app.document_management.client_registry import (
        CREATE_NEW_CLIENT_OPTION,
        resolve_active_client_name,
    )

    client_name, error = resolve_active_client_name(
        "RMIT_Demo_01",
        "",
        ["RMIT_Demo_01"],
    )
    assert client_name == "RMIT_Demo_01"
    assert error is None

    client_name, error = resolve_active_client_name(
        CREATE_NEW_CLIENT_OPTION,
        "  New_Client  ",
        ["RMIT_Demo_01"],
    )
    assert client_name == "New_Client"
    assert error is None

    client_name, error = resolve_active_client_name(
        CREATE_NEW_CLIENT_OPTION,
        "",
        ["RMIT_Demo_01"],
    )
    assert client_name is None
    assert error is not None


def test_client_specific_registries_do_not_mix_documents(temp_persist_dir: str) -> None:
    from app.document_management.document_registry import load_registry

    _write_registry(
        temp_persist_dir,
        "RMIT_Demo_01",
        {"Policy_A.pdf": {"chunk_count": 10}},
    )
    _write_registry(
        temp_persist_dir,
        "Batch_Index_Demo",
        {"Handbook.docx": {"chunk_count": 5}},
    )

    rmit_registry = load_registry(temp_persist_dir, "RMIT_Demo_01")
    batch_registry = load_registry(temp_persist_dir, "Batch_Index_Demo")

    assert set(rmit_registry["documents"]) == {"Policy_A.pdf"}
    assert set(batch_registry["documents"]) == {"Handbook.docx"}


def test_selecting_client_uses_expected_collection() -> None:
    from app.document_management.client_registry import collection_name_for_client

    client_name = "RMIT_Demo_01"
    assert collection_name_for_client(client_name) == "client_RMIT_Demo_01"


def test_new_client_collection_name(temp_persist_dir: str) -> None:
    from app.document_management.client_registry import collection_name_for_client
    from app.vector_store.vector_store import VectorStore

    client_name = f"Batch_Index_Demo_{uuid.uuid4().hex[:6]}"
    collection_name = collection_name_for_client(client_name)

    vector_store = VectorStore(
        persist_dir=temp_persist_dir,
        collection_name=collection_name,
    )
    try:
        assert vector_store.count() == 0
    finally:
        try:
            vector_store.delete_collection()
        except Exception:
            pass
        vector_store.release_persisted_client()
