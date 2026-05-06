import json
import os
from typing import Any


def get_registry_path(persist_dir: str, client_name: str) -> str:
    """Return the path to a client's document registry JSON file."""
    registry_dir = os.path.join(persist_dir, "registries")
    os.makedirs(registry_dir, exist_ok=True)

    return os.path.join(
        registry_dir,
        f"client_{client_name}_registry.json",
    )


def _default_registry(client_name: str) -> dict[str, Any]:
    """Return the default empty registry structure."""
    return {
        "client_name": client_name,
        "documents": {},
    }


def load_registry(persist_dir: str, client_name: str) -> dict[str, Any]:
    """
    Load the document registry for a client.

    Returns a default empty registry if the file does not exist,
    is empty, invalid, or has the wrong structure.
    """
    registry_path = get_registry_path(persist_dir, client_name)
    default_registry = _default_registry(client_name)

    if not os.path.exists(registry_path):
        return default_registry

    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return default_registry

        if not isinstance(data.get("documents"), dict):
            return default_registry

        data["client_name"] = data.get("client_name", client_name)

        return data

    except (json.JSONDecodeError, OSError):
        return default_registry


def save_document_record(
    persist_dir: str,
    client_name: str,
    record: dict[str, Any],
) -> None:
    """
    Add or update a document record in the client's registry.

    The record must include a non-empty 'file_name' value.
    """
    file_name = record.get("file_name")

    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("record must include a non-empty 'file_name' value")

    registry = load_registry(persist_dir, client_name)
    registry["documents"][file_name] = record

    registry_path = get_registry_path(persist_dir, client_name)

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def delete_document_record(
    persist_dir: str,
    client_name: str,
    file_name: str,
) -> None:
    """Delete a document record from the client's registry."""
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("file_name must be a non-empty string")

    registry = load_registry(persist_dir, client_name)

    if file_name not in registry["documents"]:
        return

    del registry["documents"][file_name]

    registry_path = get_registry_path(persist_dir, client_name)

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def delete_client_registry(persist_dir: str, client_name: str) -> None:
    """Delete the entire registry file for a client."""
    registry_path = get_registry_path(persist_dir, client_name)

    if os.path.exists(registry_path):
        os.remove(registry_path)