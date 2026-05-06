# CRC AI Services — Client RAG MVP

A governance-aware Retrieval Augmented Generation (RAG) platform for ingesting, indexing, retrieving, and querying client knowledge bases.

This project is part of the CRC AI Services roadmap and focuses on:
- trustworthy retrieval
- grounded responses
- document lifecycle management
- retrieval evaluation
- ingestion diagnostics
- operational knowledge-base workflows

---

# Current MVP Capabilities

## Document Ingestion
- TXT ingestion
- PDF ingestion
- DOCX ingestion

## Chunking Strategies
- Character chunking
- Delimiter chunking
- Page chunking
- Heading-aware chunking

## Retrieval Features
- Semantic vector retrieval
- Metadata-scoped retrieval
- Retrieval confidence reporting
- Source tracing
- Unsupported-answer handling

## Knowledge Base Management
- Persistent document registry
- Multi-document indexing
- Document deletion
- Re-indexing workflows
- Client-scoped collections

## Governance Features
- Ingestion diagnostics
- Readiness scoring
- Retrieval evaluation framework
- Retrieval benchmarking
- Synthetic evaluation corpus

---

# Project Structure

```text
app/
  chunking/
  config.py
  document_management/
  embeddings/
  generation/
  ingestion/
  orchestration/
  query_processing/
  retrieval/
  telemetry/
  vector_store/

data/
  evaluation_docs/
  index/

logs/
  evaluation_runs/

tests/
```

---

# Installation

## Create virtual environment

```bash
python -m venv .venv
```

## Activate environment

### Windows

```bash
.venv\\Scripts\\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in project root:

```env
RAG_PERSIST_DIR=data/index
RAG_DEFAULT_CLIENT_NAME=demo_client
RAG_DEFAULT_TOP_K=5
RAG_DEFAULT_MIN_SCORE=0.35
```

---

# Running The Streamlit App

```bash
streamlit run streamlit_app.py
```

---

# Running Evaluation Framework

Example:

```bash
python scripts/run_evaluation.py
```

---

# Current Status

Current state:
- Retrieval Evaluation V1 complete
- Client MVP operational workflows in progress
- Deployment readiness underway

---

# Future Roadmap

Planned:
- deployment support
- authentication
- reranking
- delimiter candidate detection
- ingestion governance
- AI-ready document transformation
- governance dashboards
- multi-user support

---

# Important Notes

This project is:
- a learning and engineering platform
- a governance-aware AI system prototype
- not production hardened
- not intended for legal or compliance advice

---

# CRC AI Services

This project forms part of the CRC AI Services roadmap focused on trustworthy and operational AI knowledge systems.