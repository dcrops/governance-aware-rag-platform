# CRC Governance-Aware RAG Platform — MVP V1 Architecture

## High-Level System Architecture

```mermaid
flowchart TB

    User[User]

    subgraph UI[Frontend Layer]
        Streamlit[Streamlit UI]
        Swagger[Swagger / API Docs]
    end

    subgraph API[API Layer]
        FastAPI[FastAPI Backend]
        Auth[API Key Authentication]
    end

    subgraph Orch[Orchestration Layer]
        Pipeline[RAGPipeline]
        Intent[Intent Classification]
        Routing[Retrieval Strategy Routing]
        Clarification[Clarification Detection]
        QueryRewrite[Query Rewriting]
    end

    subgraph Retrieval[Retrieval Layer]
        Retriever[Retriever]
        Reranker[Simple Reranker]
        Pruning[Context Pruning]
        Metadata[Metadata Filtering]
    end

    subgraph Vector[Vector Infrastructure]
        Embeddings[OpenAI Embeddings]
        Chroma[ChromaDB Vector Store]
    end

    subgraph Generation[Generation Layer]
        Generator[Answer Generator]
        Grounding[Grounding Check]
        Confidence[Confidence Evaluation]
    end

    subgraph Telemetry[Observability & Telemetry]
        RequestLogs[Request Telemetry Logger]
        Timing[Stage Timing Diagnostics]
        RetrievalTrace[Retrieval Trace Logging]
        Eval[Evaluation Framework]
    end

    subgraph Ingestion[Ingestion Pipeline]
        Upload[Document Upload]
        Validation[Document Validation]
        Chunking[Chunking Strategies]
        Diagnostics[Readiness Diagnostics]
        Indexing[Indexing Pipeline]
    end

    subgraph Storage[Storage Layer]
        Docs[(Client Documents)]
        Logs[(JSONL Telemetry Logs)]
        EvalRuns[(Evaluation Runs)]
    end

    User --> Streamlit
    User --> Swagger

    Streamlit --> FastAPI
    Swagger --> FastAPI

    FastAPI --> Auth
    FastAPI --> Pipeline

    Pipeline --> Intent
    Pipeline --> Routing
    Pipeline --> Clarification
    Pipeline --> QueryRewrite

    Routing --> Retriever

    Retriever --> Metadata
    Retriever --> Reranker
    Retriever --> Pruning

    Retriever --> Chroma
    Embeddings --> Chroma

    Pipeline --> Generator

    Generator --> Grounding
    Generator --> Confidence

    Pipeline --> RequestLogs
    Pipeline --> Timing
    Pipeline --> RetrievalTrace

    Eval --> Pipeline

    Upload --> Validation
    Validation --> Chunking
    Chunking --> Diagnostics
    Diagnostics --> Indexing

    Indexing --> Embeddings
    Indexing --> Chroma

    Docs --> Upload

    RequestLogs --> Logs
    Eval --> EvalRuns
```

---

# System Flow Overview

## 1. Document Ingestion

Documents are uploaded through the Streamlit UI and passed through the ingestion pipeline.

The ingestion layer performs:

* document validation
* metadata extraction
* readiness diagnostics
* chunking strategy selection
* embedding generation
* vector indexing

Supported formats:

* TXT
* PDF
* DOCX

Supported chunking strategies:

* Character chunking
* Page chunking
* Heading-aware chunking
* Delimiter chunking

---

# 2. Retrieval Orchestration

The orchestration layer acts as the central intelligence layer of the system.

Responsibilities include:

* orchestration intent classification
* retrieval strategy routing
* conversational follow-up handling
* clarification detection
* query rewriting
* retrieval pruning
* telemetry integration

Supported orchestration intents:

* standard
* aggregation
* comparison
* clarification

Supported retrieval strategies:

* standard
* comparison
* document_balanced
* document-level retrieval

---

# 3. Retrieval Pipeline

The retrieval pipeline performs:

* semantic retrieval
* metadata filtering
* reranking
* retrieval pruning
* evidence packaging

Vector infrastructure:

* OpenAI embeddings
* ChromaDB persistent vector store

Retrieval controls:

* top_k
* min_score
* document filters
* metadata filters

---

# 4. Grounded Answer Generation

The generation layer creates grounded responses using retrieved evidence.

The system includes:

* grounding checks
* insufficient evidence handling
* confidence evaluation
* clarification handling

Supported answer states:

* ANSWERED
* INSUFFICIENT_EVIDENCE
* CLARIFICATION_REQUIRED
* NO_INDEX_FOUND

---

# 5. Observability & Telemetry

The platform includes extensive observability infrastructure.

Telemetry includes:

* orchestration reasoning
* retrieval strategy selection
* retrieval confidence
* grounding checks
* retrieval traces
* timing diagnostics
* request telemetry
* evaluation benchmarking

Telemetry persistence:

* JSONL request logs
* evaluation benchmark runs

---

# 6. Deployment Architecture

Deployment stack:

* FastAPI backend
* Streamlit frontend
* Docker Compose orchestration
* ChromaDB persistent storage

Authentication:

* API key authentication

API endpoints:

* `/ask`
* `/health`
* `/version`

---

# Engineering Focus Areas Demonstrated

This MVP demonstrates:

* retrieval engineering
* orchestration-aware AI systems
* conversational retrieval
* grounded answer generation
* observability and telemetry
* governance-aware AI workflows
* deployment architecture
* containerisation
* evaluation infrastructure
* operational AI engineering
