import argparse

from app.embeddings.embeddings import EmbeddingClient
from app.vector_store.vector_store import VectorStore
from app.retrieval.retriever import Retriever
from app.generation.answer_generator import AnswerGenerator
from app.orchestration.rag_pipeline import RAGPipeline
from app.config import PERSIST_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Ask a question against an indexed client RAG collection."
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask the client knowledge base.",
    )
    parser.add_argument(
        "--client",
        default="demo_client",
        help="Client name / collection suffix.",
    )
    parser.add_argument(
        "--file-name",
        default=None,
        help="Optional file name to scope retrieval to a single indexed document.",
    )

    args = parser.parse_args()

    client_name = args.client
    question = args.question

    persist_dir = PERSIST_DIR
    collection_name = f"client_{client_name}"

    print(f"[Ask Client] Using collection: '{collection_name}' from '{persist_dir}'")
    print(f"[Ask Client] Question: {question}")

    if args.file_name:
        print(f"[Ask Client] Retrieval scoped to file: {args.file_name}")

    vector_store = VectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )

    try:
        count = vector_store.count()
        if count == 0:
            print(f"[Error] No records found in '{collection_name}'. Did you index the documents?")
            return
    except Exception as e:
        print(f"[Error] Failed to access the vector store: {e}")
        return

    embedding_client = EmbeddingClient()

    from app.query_processing.query_rewriter import QueryRewriter

    query_rewriter = QueryRewriter()

    retriever = Retriever(
        embedding_client=embedding_client,
        vector_store=vector_store,
        query_rewriter=query_rewriter,
    )

    answer_generator = AnswerGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        answer_generator=answer_generator,
    )

    metadata_filter = None

    if args.file_name:
        metadata_filter = {
            "file_name": args.file_name,
        }

    response = pipeline.answer_question(
        question,
        top_k=5,
        min_score=0.35,
        metadata_filter=metadata_filter,
    )

    print("\n" + "=" * 60)
    print("[ANSWER]")
    print(response.answer)
    print("\n[Retrieval Confidence]:", response.retrieval_confidence)
    print("[Answer Status]:", response.answer_status)

    print("\n[SOURCES]:")
    for source in response.sources:
        print(
            f"- {source.file_name} | chunk {source.chunk_index} | score {source.score:.3f}"
        )

    if response.log:
        print("\n[TELEMETRY]:")
        print("Original Query:", response.log.original_query)
        print("Retrieval Query:", response.log.retrieval_query)
        print("Scores:", [round(score, 3) for score in response.log.scores])


if __name__ == "__main__":
    main()