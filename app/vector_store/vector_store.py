from typing import List, Dict, Any
from pydantic import BaseModel, Field, ValidationError

from app.models.vector_record import VectorRecord

try:
    import chromadb
except ImportError:
    raise ImportError("ChromaDB is required. Please install with `pip install chromadb`.")


class SearchResult(BaseModel):
    """
    Represents a similarity search hit from the vector store.
    """
    id: str
    text: str
    score: float
    vector_score: float | None = None
    rerank_bonus: float | None = None
    final_score: float | None = None
    metadata: dict = Field(default_factory=dict)


class VectorStore:
    """
    A persistent vector store using ChromaDB for storing and searching VectorRecord objects.

    Stores vector embeddings for document chunks and supports similarity search.
    """

    def __init__(
        self,
        persist_dir: str = "data/index",
        collection_name: str = "documents"
    ) -> None:
        """
        Initializes the persistent ChromaDB client and collection.

        Args:
            persist_dir: Directory path for persisted index storage.
            collection_name: The name of the vector collection.
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(
            path=self.persist_dir
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name
        )

    def delete_collection(self) -> None:
        """Delete the current Chroma collection."""
        self._client.delete_collection(self.collection_name)

    def count(self) -> int:
        """Return the number of records in the collection."""
        return self._collection.count()

    def upsert_records(self, records: List[VectorRecord]) -> None:
        """
        Upsert a batch of VectorRecord objects into the vector store.

        Args:
            records: List of VectorRecord objects (must not be empty).

        Raises:
            ValueError: If records is empty or contains invalid items.
            TypeError: If records items are not VectorRecord instances.
        """
        if not records or not isinstance(records, list):
            raise ValueError("Input 'records' must be a non-empty list of VectorRecord objects.")

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, record in enumerate(records):
            if not isinstance(record, VectorRecord):
                raise TypeError(f"Record at index {i} is not a VectorRecord instance.")
            # Validate vector shape and contents
            if not record.chunk or not isinstance(record.embedding, list) or not record.embedding:
                raise ValueError(f"Record at index {i} is malformed or missing embedding/chunk.")
            ids.append(record.chunk.chunk_id)
            embeddings.append(record.embedding)
            documents.append(record.chunk.text)
            md = record.chunk.metadata.copy() if isinstance(record.chunk.metadata, dict) else {}
            md.update({"doc_id": record.chunk.doc_id, "chunk_index": record.chunk.chunk_index})
            metadatas.append(md)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> List[SearchResult]:
        """
        Perform a similarity search using the input embedding.

        Args:
            query_embedding: Embedding vector of the query (must not be empty).
            top_k: Number of top results to return (must be > 0).

        Returns:
            List of SearchResult objects, sorted by similarity score descending.

        Raises:
            ValueError: For invalid embeddings or top_k.
        """
        if not isinstance(query_embedding, list) or len(query_embedding) == 0:
            raise ValueError("query_embedding must be a non-empty list of floats.")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        # Perform query with Chroma, requesting distance metric 'l2' or default
        try:
            query_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["metadatas", "documents", "distances"],
            }

            if metadata_filter is not None:
                query_kwargs["where"] = metadata_filter

            response = self._collection.query(**query_kwargs)
        except Exception as exc:
            raise RuntimeError(f"ChromaDB similarity search failed: {exc}")

        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        if not ids or len(ids) == 0:
            return []

        # Normalization: similarity = 1 / (1 + distance)
        # Higher similarity = more similar
        search_results = []
        for i in range(len(ids)):
            dist = distances[i]
            sim_score = 1.0 / (1.0 + float(dist)) if dist is not None else 0.0
            try:
                sr = SearchResult(
                    id=ids[i],
                    text=documents[i],
                    score=sim_score,
                    vector_score=sim_score,
                    final_score=sim_score,
                    metadata=metadatas[i] if isinstance(metadatas[i], dict) else {},
                )
            except ValidationError:
                # Skip invalid result
                continue
            search_results.append(sr)
        # Sort descending by score, in case Chroma does not guarantee order
        search_results.sort(key=lambda r: -r.score)

        return search_results

    def list_documents(self) -> list[dict]:
        """
        List indexed documents in the current collection, grouping by file_name.

        Returns:
            A list of dictionaries with file_name, file_type, and chunk_count.
            Returns an empty list if the collection is empty.
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])

        grouped = {}

        for metadata in metadatas:
            file_name = metadata.get("file_name")
            file_type = metadata.get("file_type")

            if not file_name:
                continue

            if file_name not in grouped:
                grouped[file_name] = {
                    "file_name": file_name,
                    "file_type": file_type,
                    "chunk_count": 0,
                }

            grouped[file_name]["chunk_count"] += 1

        return sorted(
            grouped.values(),
            key=lambda item: item["file_name"].lower(),
        )

    def delete_document(self, file_name: str) -> int:
        """
        Delete all chunks for the given file_name from the collection.

        Args:
            file_name (str): Name of the document to delete.

        Returns:
            int: Number of records deleted (0 if none found or deleted).
        """
        # Validate file_name
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError("file_name must be a non-empty string.")

        # Retrieve all ids whose file_name metadata matches
        results = self._collection.get(include=["metadatas"])
        ids_to_delete = [
            result_id
            for result_id, metadata in zip(
                results.get("ids", []), results.get("metadatas", [])
            )
            if isinstance(metadata, dict)
            and metadata.get("file_name") == file_name
        ]

        if not ids_to_delete:
            return 0

        self._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)