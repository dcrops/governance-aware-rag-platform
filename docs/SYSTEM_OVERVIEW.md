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

## Offline regression baseline — 25 November 2026

Recorded before the controlled `gpt-4.1-mini` and `gpt-6-luna` comparison. Interpreter: the `ai311` environment, Python 3.11.14. Command: `python -m pytest tests/test_client_registry.py`.

- Passed: 9
- Failed: 0
- Skipped: 0
- Additional tests requiring external APIs were not executed: 8
  - `tests/test_batch_indexing.py`: 5
  - `tests/test_cross_process_retrieval.py`: 3
- Offline pytest execution time: 1.85 seconds
- `scripts/test_intent_classifier.py` smoke check passed

This offline suite covers client discovery, name validation, registry isolation, and an empty local collection. It does not establish generation, grounding, citation, or rewriting correctness.

## Controlled comparison — 25 November 2026

One execution of six synthetic scenarios was run for `gpt-4.1-mini` and `gpt-6-luna`. The detailed record is `logs/evaluation_runs/model_comparison_20261125.json`.

- 22 successful chat calls.
- 11 calls per model.
- Zero API errors.
- Zero retries.
- Zero empty or truncated responses.
- Zero reasoning tokens.
- GPT-4.1 Mini: 2,277 prompt tokens, 155 completion tokens, 12.8 seconds combined latency.
- GPT-6 Luna: 2,266 prompt tokens, 138 completion tokens, 10.2 seconds combined latency.

Each scenario ran once per model. These figures describe that single execution. They do not establish a statistical performance improvement.

Luna requests used `reasoning_effort` `none`, `max_completion_tokens` at the existing budgets of 5, 512, 80 and 64, and omitted temperature, `top_p` and `stop`. The baseline model kept its existing parameters. On the contextual follow-up, Luna joined the policy’s “authorised personnel” rule and the FAQ’s payroll, HR and finance rule with “this includes”. The FAQ does not define the policy term. Grounding still returned `PASS`.

## Production validation — 25 November 2026

The migrated production classes were exercised once on the same synthetic index. The record is `logs/evaluation_runs/migration_validation_20261125.json`.

- Default model resolved to `gpt-6-luna`, and all 21 chat responses returned that id.
- 21 chat calls and 1 query-embedding call, matching the plan. No retries and no API errors.
- Supported relevance returned exact `YES` on all 5 calls. Unsupported relevance returned exact `NO` on all 5 calls. Each used 4 completion tokens, finished with `stop`, and reported 0 reasoning tokens.
- The standard, comparison, aggregation, contextual rewrite, and evaluation rewrite scenarios succeeded.
- The additional cross-document question, supplied with both access rules, answered with the FAQ rule and did not repeat the earlier “this includes” bridge. It also did not restate the policy’s broader wording.
- Scenario answers were still a single execution. The five relevance repeats do not establish general answer-quality statistics.

## Chat model configuration

`RAG_CHAT_MODEL` selects the operational chat model. When it is unset, the default is `gpt-6-luna`. Set `RAG_CHAT_MODEL=gpt-4.1-mini` to keep the previous parameter profile. An explicit constructor argument overrides the environment variable.

The Luna profile applies only to the exact model id `gpt-6-luna`. `gpt-4.1-mini` and any other explicit model keep the historical `max_tokens` and sampling arguments. The embedding model remains `text-embedding-3-small`.

## Existing architectural limitations

These behaviours were left unchanged by the model migration:

- Relevance accepts any response whose text starts with `YES`.
- An empty answer is labelled `ANSWERED` unless it contains the refusal sentence.
- Grounding `PASS` records that relevance returned `YES` and an answer was generated. It does not check each claim.
- Citations are the retrieved chunks, including chunks returned with a refusal.
- Refusal status depends on the answer containing “don't have enough information”.
- A follow-up whose original wording matches the clarification list is not retrieved, including when a standalone rewrite was produced.
- The historical topic-match metric searches the answer and the retrieved chunk text together.

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

# UPDATED SYSTEM_OVERVIEW.md ADDITIONS

# API Layer

Location:

- `api/`

Capabilities:

- FastAPI backend
- REST API querying
- API key authentication
- structured error handling
- Docker deployment support
- orchestration-aware responses

Endpoints:

- `/ask`
- `/health`
- `/version`

---

# Expanded Orchestration Layer

Location:

- `app/orchestration/`

Primary pipeline:

- `RAGPipeline`

Responsibilities:

- orchestration intent classification
- retrieval strategy routing
- conversational follow-up handling
- clarification detection
- query rewriting integration
- retrieval pruning
- generation orchestration
- telemetry integration
- response packaging

Supported orchestration intents:

- standard
- aggregation
- comparison
- clarification

Supported retrieval strategies:

- standard
- comparison
- document_balanced
- document-level retrieval

Current orchestration behaviours:

- aggregation-aware retrieval
- comparison-aware retrieval
- conversational ambiguity handling
- unsupported-answer handling
- retrieval trace generation

---

# Expanded Telemetry Layer

Current capabilities:

- retrieval logging
- request telemetry logging
- orchestration reasoning logging
- retrieval trace logging
- pipeline timing diagnostics
- stage timing diagnostics
- retrieval score tracking
- retrieved chunk tracking

Telemetry fields:

- original query
- rewritten query
- orchestration intent
- retrieval strategy
- grounding check
- retrieval confidence
- retrieved chunk IDs
- similarity scores
- stage timings
- pipeline timings
- request timings
- retrieval traces

Storage:

- JSONL logging

Current telemetry files:

- `logs/request_telemetry.jsonl`
- `logs/evaluation_runs/`