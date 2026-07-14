import os
import re
from pathlib import Path

try:
    import chromadb
except ImportError:
    chromadb = None

CLIENT_COLLECTION_PREFIX = "client_"
CREATE_NEW_CLIENT_OPTION = "+ Create new client"

_INVALID_CLIENT_NAME_CHARS = re.compile(r'[\\/:*?"<>|]')

_EXCLUDED_CLIENT_NAMES = frozenset({
    "string",
    "demo_client",
    "nonexistent",
    "default_client",
    "documents",
})

_EXCLUDED_CLIENT_PREFIXES = (
    "cross_process",
    "cross_",
    "stale_",
    "baseline",
    "stale_fix",
    "reader_refresh",
    "warm_",
    "disk_",
    "cross_process_",
)


def collection_name_for_client(client_name: str) -> str:
    """Map a user-facing client name to its Chroma collection name."""
    return f"{CLIENT_COLLECTION_PREFIX}{client_name}"


def client_name_from_collection(collection_name: str) -> str | None:
    """Convert a Chroma collection name back to a user-facing client name."""
    if not collection_name.startswith(CLIENT_COLLECTION_PREFIX):
        return None
    return collection_name[len(CLIENT_COLLECTION_PREFIX):]


def normalize_client_name(client_name: str) -> str:
    """Strip surrounding whitespace from a client name."""
    return client_name.strip()


def validate_client_name(client_name: str, existing_clients: list[str] | None = None) -> tuple[bool, str]:
    """
    Validate a client name for indexing and retrieval.

    Returns:
        (is_valid, error_message)
    """
    normalized = normalize_client_name(client_name)

    if not normalized:
        return False, "Client name cannot be empty."

    if _INVALID_CLIENT_NAME_CHARS.search(normalized):
        return (
            False,
            'Client name cannot contain any of these characters: \\ / : * ? " < > |',
        )

    if existing_clients:
        lowered_existing = {name.casefold() for name in existing_clients}
        if normalized.casefold() in lowered_existing:
            return (
                False,
                f"A client named '{normalized}' already exists. Select it from the list instead.",
            )

    return True, ""


def _is_excluded_client_name(client_name: str) -> bool:
    if client_name in _EXCLUDED_CLIENT_NAMES:
        return True
    lowered = client_name.lower()
    return any(lowered.startswith(prefix) for prefix in _EXCLUDED_CLIENT_PREFIXES)


def list_clients_from_registries(persist_dir: str) -> list[str]:
    """Discover client names from persisted registry JSON files."""
    registry_dir = Path(persist_dir) / "registries"
    if not registry_dir.exists():
        return []

    clients: set[str] = set()
    suffix = "_registry.json"
    prefix = CLIENT_COLLECTION_PREFIX

    for registry_path in registry_dir.glob(f"{prefix}*{suffix}"):
        stem = registry_path.stem
        if not stem.startswith(prefix) or not stem.endswith("_registry"):
            continue
        client_name = stem[len(prefix):-len("_registry")]
        if client_name and not _is_excluded_client_name(client_name):
            clients.add(client_name)

    return sorted(clients, key=str.casefold)


def list_clients_from_chroma(persist_dir: str) -> list[str]:
    """Discover client names from persisted Chroma collections."""
    if chromadb is None:
        return []

    try:
        client = chromadb.PersistentClient(path=str(Path(persist_dir).resolve()))
        collections = client.list_collections()
    except Exception:
        return []

    clients: set[str] = set()
    for collection in collections:
        client_name = client_name_from_collection(collection.name)
        if client_name and not _is_excluded_client_name(client_name):
            clients.add(client_name)

    return sorted(clients, key=str.casefold)


def discover_clients(persist_dir: str) -> list[str]:
    """
    Discover existing client knowledge bases from registry files and Chroma collections.
    """
    clients = set(list_clients_from_registries(persist_dir))
    clients.update(list_clients_from_chroma(persist_dir))
    return sorted(clients, key=str.casefold)


def resolve_active_client_name(
    selected_option: str,
    new_client_name_input: str,
    existing_clients: list[str],
) -> tuple[str | None, str | None]:
    """
    Resolve the active client name from selector state.

    Returns:
        (client_name, error_message)
    """
    if selected_option == CREATE_NEW_CLIENT_OPTION:
        candidate = normalize_client_name(new_client_name_input)
        is_valid, error = validate_client_name(candidate, existing_clients=existing_clients)
        if not is_valid:
            return None, error
        return candidate, None

    return selected_option, None
