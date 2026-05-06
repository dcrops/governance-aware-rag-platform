# RAG Platform Documentation V1

# SYSTEM_OVERVIEW.md

## Project Purpose

This project is a governance-aware Retrieval Augmented Generation (RAG) platform being developed as part of CRC AI Services.

The system is designed to:

- ingest documents
- chunk and index content
- retrieve grounded information
- generate evidence-supported answers
- evaluate retrieval quality
- provide observability and governance capabilities

The project supports two future operational modes:

1. CRC Internal Governance / Engineering Mode
2. Client-Facing SME Knowledge Assistant Mode

---

# Current System Architecture

## Ingestion Layer

Location:

- `app/ingestion/`

Capabilities:

- TXT ingestion
- PDF ingestion
- document metadata extraction
- ingestion diagnostics
- readiness scoring
- chunking strategy recommendations

Current diagnostics include:

- file type
- file size
- page count
- character count
- word count
- chunk count
- average chunk length
- readiness score
- readiness status
- warnings
- recommended chunking strategy
- recommendation reason

Supported chunking recommendations:

- character
- page
- heading
- delimiter

Future roadmap:

- delimiter candidate detection
- duplicate document detection
- document freshness analysis
- AI-ready document transformation layer
- OCR quality scoring

---

## Chunking Layer

Location:

- `app/chunking/chunker.py`

Current supported chunking strategies:

### Character Chunking

- default fallback strategy
- overlap-based chunking
- strongest general-purpose strategy
- best for conversational or poorly structured text

### Page Chunking

- PDF-only
- uses extracted page metadata
- useful for scanned documents and page-aware retrieval

### Heading Chunking

- section-aware chunking
- useful for policy and structured documents

### Delimiter Chunking

- user-specified delimiter splitting
- strongest performer for structured FAQ and repeated-format documents

Example delimiters tested:

- `SECTION:`
- `QUESTION:`
- `BREAK`

Current retrieval findings:

- FAQ documents strongly benefit from delimiter chunking
- Policy documents benefit from delimiter or heading chunking
- Meeting notes show little sensitivity to chunking strategy
- Poorly structured documents perform similarly across all strategies
- Semi-structured mixed documents often perform best with character chunking

---

## Embeddings Layer

Location:

- `app/embeddings/`

Current embedding model:

- OpenAI `text-embedding-3-small`

Capabilities:

- chunk embedding generation
- query embedding generation

---

## Vector Store Layer

Location:

- `app/vector_store/`

Current vector database:

- ChromaDB

Capabilities:

- persistent collections
- similarity search
- metadata filtering
- client-scoped collections
- collection count validation

Current metadata usage:

- file_name
- chunk strategy
- chunk index
- page metadata

---

## Retrieval Layer

Location:

- `app/retrieval/`

Capabilities:

- top-k retrieval
- metadata-scoped retrieval
- minimum similarity thresholds
- query rewriting integration
- reranker-ready architecture

Current retrieval controls:

- `top_k`
- `min_score`
- metadata filters

---

## Query Processing Layer

Location:

- `app/query_processing/`

Current capability:

- deterministic rule-based query rewriting

Purpose:

- improve semantic retrieval
- expand AI-related terminology
- improve recall

Future roadmap:

- LLM-based query rewriting
- adaptive retrieval-aware rewriting
- retrieval feedback loops

---

## Generation Layer

Location:

- `app/generation/`

Capabilities:

- grounded answer generation
- evidence-aware responses
- refusal handling
- confidence reporting

Current answer statuses:

- `ANSWERED`
- `INSUFFICIENT_EVIDENCE`
- `NO_RESULTS`

Current retrieval confidence levels:

- LOW
- MEDIUM
- HIGH

---

## Orchestration Layer

Location:

- `app/orchestration/`

Primary pipeline:

- `RAGPipeline`

Responsibilities:

- retrieval orchestration
- generation orchestration
- telemetry integration
- response packaging

---

## Telemetry Layer

Location:

- `app/telemetry/`

Current capabilities:

- retrieval logging
- query tracking
- retrieval score tracking
- retrieved chunk tracking

Telemetry fields:

- original query
- rewritten query
- retrieved chunk IDs
- similarity scores

Storage:

- JSONL logging

---

## Evaluation Framework

Location:

- `scripts/run_evaluation.py`
- `tests/`

Current capabilities:

- retrieval benchmarking
- chunking strategy comparison
- unsupported-answer evaluation
- topic match evaluation
- retrieval score evaluation
- benchmark persistence

Current metrics:

- document hit rate
- topic match rate
- average top score
- average retrieval score
- answer status match rate
- zero topic match tracking

Supported answer evaluation:

- accepts multiple valid refusal states
- supports `NO_RESULTS`
- supports `INSUFFICIENT_EVIDENCE`

Evaluation persistence:

- JSON benchmark runs saved under:
    - `logs/evaluation_runs/`

Stored benchmark metadata:

- rewrite strategy
- chunking strategy
- delimiter
- retrieval metrics
- answer status metrics

---

# Synthetic Evaluation Corpus

Location:

- `data/evaluation_docs/`

Current datasets:

## policy_style

Purpose:

- structured governance/policy testing

Findings:

- delimiter and heading chunking perform best

---

## faq_style

Purpose:

- repeated Q&A structure testing

Findings:

- delimiter chunking using `QUESTION:` significantly improved retrieval quality

---

## meeting_notes_style

Purpose:

- conversational note testing

Findings:

- chunking strategy had minimal impact

---

## poor_structure

Purpose:

- low-quality document testing

Findings:

- advanced chunking offered little benefit
- character chunking safest fallback

---

## run_sheet_style

Purpose:

- semi-structured mixed-content testing

Findings:

- advanced chunking slightly reduced retrieval quality
- character chunking preserved broader semantic continuity

---

# Streamlit UI

Location:

- `streamlit_app.py`

Current capabilities:

- document upload
- chunking strategy selection
- retrieval parameter controls
- indexed document display
- retrieval filters
- question answering
- source inspection
- readiness display

Current controls:

- chunking strategy
- top_k retrieval slider
- minimum score slider
- document retrieval filtering
- clear index
- replace index

---

# ask_client.py

Purpose:

- CLI interface for querying indexed client collections

Capabilities:

- client collection selection
- optional file-scoped retrieval
- telemetry display