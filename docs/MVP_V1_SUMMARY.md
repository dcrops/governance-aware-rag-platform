# CRC Governance-Aware RAG System — MVP V1

## Overview

CRC Governance-Aware RAG is a production-minded Retrieval-Augmented Generation (RAG) system designed to provide grounded, explainable, and governance-aware answers over organisational document collections.

Unlike basic chatbot-style RAG systems, this project focuses heavily on:

- retrieval orchestration
- explainability
- grounded answer generation
- conversational query handling
- retrieval observability
- governance-aware AI behaviour
- insufficient evidence handling
- telemetry and diagnostics

The project was developed as part of the broader CRC AI Services direction focused on trustworthy operational AI systems.

---

# Core MVP Objectives

The MVP V1 goals were to:

- ingest organisational documents into a searchable vector index
- support grounded question answering over indexed documents
- support conversational follow-up questions
- support multiple retrieval strategies
- support document-scoped retrieval
- avoid unsupported hallucinated answers
- provide orchestration transparency
- expose retrieval telemetry and diagnostics
- provide deployable API + UI architecture

---

# Key Features

## Multi-Format Document Ingestion

Supported formats:

- TXT
- PDF
- DOCX

Features:

- document validation
- ingestion diagnostics
- readiness scoring
- client-scoped indexing
- metadata preservation

---

# Chunking Strategies

Implemented chunking approaches:

- character chunking
- delimiter chunking
- page-based chunking
- heading-aware chunking

The ingestion pipeline can recommend chunking strategies based on document diagnostics.

---

# Embedding & Retrieval

## Embeddings

- OpenAI embeddings (`text-embedding-3-small`)

## Vector Store

- ChromaDB
- persistent client-scoped collections

## Retrieval Capabilities

- semantic retrieval
- document-scoped retrieval
- metadata filtering
- reranking
- adaptive retrieval orchestration

---

# Retrieval Orchestration

The system includes orchestration-aware routing logic that determines:

- retrieval intent
- retrieval strategy
- clarification requirements
- conversational query rewriting

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

---

# Conversational Retrieval

The system supports conversational follow-up handling.

Examples:

Previous question:

> What is the appeals process?

Follow-up:

> What about casual staff?

The system detects ambiguous conversational references and can:

- rewrite the retrieval query
- request clarification
- avoid unsupported assumptions

---

# Grounded Answer Generation

The system includes grounding-aware answer generation behaviour.

Supported answer states:

- ANSWERED
- INSUFFICIENT_EVIDENCE
- CLARIFICATION_REQUIRED
- NO_INDEX_FOUND

The system is designed to avoid hallucinated unsupported answers when sufficient grounding evidence is unavailable.

---

# Observability & Telemetry

A major focus of MVP V1 is observability.

The system captures:

- orchestration reasoning
- retrieval strategy selection
- retrieval confidence
- grounding status
- API timing
- pipeline timing
- stage timing
- retrieval traces
- retrieved chunk metadata

Telemetry is stored in structured JSONL logs.

---

# API Architecture

## Backend

- FastAPI
- Dockerised deployment
- API key authentication
- structured error handling

Endpoints:

- `/ask`
- `/health`
- `/version`

---

# User Interface

## Streamlit UI

Features:

- document upload/indexing
- retrieval mode controls
- conversational querying
- orchestration transparency
- source visibility
- timing diagnostics

---

# Example Orchestration Behaviour

## Standard Question

Question:

> What does the Academic Promotion Policy say about appeals?

System Behaviour:

- orchestration intent: standard
- retrieval strategy: standard
- focused retrieval pruning
- grounded answer generation

---

## Aggregation Question

Question:

> List all eligibility requirements mentioned in the academic promotion documents

System Behaviour:

- orchestration intent: aggregation
- retrieval strategy: document_balanced
- broad cross-document retrieval
- aggregation-aware answer generation

---

## Unsupported Question

Question:

> What colour should academic staff dye their hair?

System Behaviour:

- retrieval attempted
- grounding failed
- insufficient evidence returned
- hallucination avoided

---

# Deployment

The MVP is containerised using Docker Compose.

Services:

- FastAPI backend
- Streamlit frontend

Environment configuration managed via `.env`.

---

# Current MVP Limitations

Known limitations include:

- global collection retrieval can introduce retrieval noise
- aggregation retrieval may retrieve semantically broad documents
- no hybrid lexical + semantic retrieval yet
- no advanced caching layer
- no role-based access control
- no multi-tenant authentication layer
- limited corpus/domain partitioning
- evaluation framework still evolving

---

# Engineering Focus Areas Learned

This project was designed as an AI engineering learning platform focused on:

- retrieval engineering
- orchestration systems
- grounded AI generation
- AI observability
- governance-aware AI architecture
- conversational retrieval
- telemetry systems
- evaluation harnesses
- deployment architecture
- containerisation
- production-minded AI workflows

---

# Future Directions (Post-MVP)

Potential future enhancements:

- hybrid retrieval
- adaptive reranking
- corpus/domain partitioning
- governance-layer orchestration
- retrieval analytics dashboards
- structured retrieval evaluation datasets
- operational intelligence integration
- multi-agent orchestration
- enterprise authentication and permissions

---

# Technology Stack

- Python
- FastAPI
- Streamlit
- ChromaDB
- OpenAI API
- Docker
- Docker Compose
- JSONL telemetry logging

---

# MVP V1 Status

MVP V1 successfully demonstrates:

- governance-aware retrieval orchestration
- grounded question answering
- conversational retrieval
- explainable orchestration decisions
- telemetry-driven observability
- deployable AI system architecture

The project has evolved significantly beyond a basic tutorial RAG implementation into a modular, observable, and production-minded AI engineering system.