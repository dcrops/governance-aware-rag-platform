"""Programmatic acceptance check for batch indexing and client selection."""
import os
import sys
import tempfile

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import truststore

truststore.inject_into_ssl()
from dotenv import load_dotenv

load_dotenv()

from app.config import PERSIST_DIR
from app.document_management.client_registry import (
    collection_name_for_client,
    discover_clients,
)
from app.document_management.document_registry import load_registry
from app.indexing.document_indexer import index_document_batch
from app.vector_store.vector_store import VectorStore

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "crc-dev-key")


def main() -> None:
    print("=== Step 1-3: Discover existing clients ===")
    clients = discover_clients(PERSIST_DIR)
    print("Discovered clients:", clients)
    rmit = "RMIT_Demo_01" if "RMIT_Demo_01" in clients else (clients[0] if clients else None)
    if rmit:
        reg = load_registry(PERSIST_DIR, rmit)
        print(f"RMIT registry docs ({rmit}):", list(reg.get("documents", {}).keys()))

    print("=== Step 4-9: Create Batch_Index_Demo and index 3 files ===")
    batch_client = "Batch_Index_Demo"
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        contents = {
            "Distinctive_Alpha_Policy.txt": (
                "ALPHA_UNIQUE_TOKEN: Academic integrity requires honest submission of work."
            ),
            "Distinctive_Beta_Handbook.txt": (
                "BETA_UNIQUE_TOKEN: Workplace safety reporting must occur within 24 hours."
            ),
            "Distinctive_Gamma_Guide.txt": (
                "GAMMA_UNIQUE_TOKEN: Research ethics approval is mandatory for human studies."
            ),
        }
        for name, text in contents.items():
            path = os.path.join(tmp, name)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            files.append((path, name))

        result = index_document_batch(
            persist_dir=PERSIST_DIR,
            client_name=batch_client,
            collection_name=collection_name_for_client(batch_client),
            file_items=files,
            chunking_strategy="character",
            chunk_size=800,
            chunk_overlap=100,
            delimiter=None,
            replace_existing_index=True,
        )

    print(
        "Batch success:",
        len(result.successful_results),
        "failed:",
        len(result.failed_results),
    )
    for item in result.results:
        print(("OK" if item.success else "FAIL"), item.file_name, item.chunk_count, item.error_message)

    reg = load_registry(PERSIST_DIR, batch_client)
    print("Batch registry docs:", list(reg.get("documents", {}).keys()))
    assert len(result.successful_results) == 3

    print("=== Step 10-11: Ask distinctive questions via API without restart ===")
    questions = [
        ("What is ALPHA_UNIQUE_TOKEN about?", "Distinctive_Alpha_Policy.txt"),
        ("What is BETA_UNIQUE_TOKEN about?", "Distinctive_Beta_Handbook.txt"),
        ("What is GAMMA_UNIQUE_TOKEN about?", "Distinctive_Gamma_Guide.txt"),
    ]
    for question, expected_file in questions:
        response = requests.post(
            f"{API_BASE}/ask",
            headers={"X-API-Key": API_KEY},
            json={
                "question": question,
                "client_name": batch_client,
                "top_k": 5,
                "min_score": 0.2,
            },
            timeout=120,
        )
        data = response.json()
        sources = [
            source.get("file_name")
            for source in data.get("sources", [])
            if isinstance(source, dict)
        ]
        print("Q:", question)
        print("  status:", data.get("answer_status"), "sources:", sources[:3])
        assert expected_file in sources, f"Missing source {expected_file}"

    print("=== Step 12-15: Verify RMIT client still available ===")
    clients_after = discover_clients(PERSIST_DIR)
    print("Clients after batch:", clients_after)
    assert batch_client in clients_after
    if rmit:
        rmit_reg = load_registry(PERSIST_DIR, rmit)
        print(f"{rmit} docs preserved:", list(rmit_reg.get("documents", {}).keys()))
        VectorStore.refresh_persisted_client(PERSIST_DIR)
        vector_store = VectorStore(PERSIST_DIR, collection_name_for_client(rmit))
        print(f"{rmit} vector count:", vector_store.count())

    batch_reg2 = load_registry(PERSIST_DIR, batch_client)
    print("Batch docs still present:", list(batch_reg2.get("documents", {}).keys()))
    print("ACCEPTANCE SCRIPT PASSED")


if __name__ == "__main__":
    main()
