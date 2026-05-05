import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any


def get_diagnostics_path(persist_dir: str, client_name: str) -> str:
    """Return the JSON file path for a client's document diagnostics."""
    diagnostics_dir = os.path.join(persist_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)

    file_name = f"client_{client_name}_diagnostics.json"
    return os.path.join(diagnostics_dir, file_name)


def load_diagnostics(persist_dir: str, client_name: str) -> dict[str, Any]:
    """
    Load document diagnostics for a client.

    Returns an empty dictionary if the diagnostics file does not exist,
    is empty, or contains invalid JSON.
    """
    path = get_diagnostics_path(persist_dir, client_name)

    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

        return {}

    except (json.JSONDecodeError, OSError):
        return {}

def delete_client_diagnostics(
    persist_dir: str,
    client_name: str,
) -> None:
    """Delete the diagnostics file for a client."""
    path = get_diagnostics_path(persist_dir, client_name)

    if os.path.exists(path):
        os.remove(path)


def _diagnostics_to_dict(diagnostics: Any) -> dict[str, Any]:
    """Convert supported diagnostics objects into a serialisable dictionary."""
    if is_dataclass(diagnostics):
        return asdict(diagnostics)

    if isinstance(diagnostics, dict):
        return diagnostics

    raise TypeError(
        "Diagnostics must be a dataclass instance or dictionary."
    )


def save_document_diagnostics(
    persist_dir: str,
    client_name: str,
    file_name: str,
    diagnostics: Any,
) -> None:
    """Save or update diagnostics for a document in a client's diagnostics file."""
    all_diagnostics = load_diagnostics(persist_dir, client_name)
    all_diagnostics[file_name] = _diagnostics_to_dict(diagnostics)

    path = get_diagnostics_path(persist_dir, client_name)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_diagnostics, f, indent=2, ensure_ascii=False)


def delete_document_diagnostics(
    persist_dir: str,
    client_name: str,
    file_name: str,
) -> None:
    """Delete diagnostics for a document from a client's diagnostics file."""
    all_diagnostics = load_diagnostics(persist_dir, client_name)

    if file_name not in all_diagnostics:
        return

    del all_diagnostics[file_name]

    path = get_diagnostics_path(persist_dir, client_name)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_diagnostics, f, indent=2, ensure_ascii=False)