# VIDEO_PLAN.md

# CRC Governance-Aware RAG Platform — Demo Video Plan

## Video Goal

Demonstrate the architecture, orchestration behaviour, grounded retrieval workflows, observability, and deployment maturity of the CRC Governance-Aware RAG Platform MVP V1.

The video should communicate that this project is:

- more than a basic “chat with PDFs” system
- a modular AI engineering platform
- focused on trustworthy and explainable AI workflows
- designed with operational AI engineering principles

---

# Video Style

Target style:

- clean
- technical
- modern
- concise
- engineering-focused
- visually guided

Avoid:

- excessive hype language
- buzzword overload
- overlong coding walkthroughs
- overly academic explanations

Recommended approach:

- screen recordings
- architecture visuals
- orchestration demonstrations
- telemetry walkthroughs
- subtle background music
- text-to-speech narration

Target duration:

- 5–10 minutes

---

# Video Structure

# 1. Introduction

## Goal

Introduce the platform and its purpose.

## On Screen

- CRC Governance-Aware RAG title
- architecture diagram
- Streamlit UI preview
- orchestration screenshots

## Narration Topics

Explain:

- this is a governance-aware RAG platform
- focus areas include:
  - orchestration
  - grounded generation
  - conversational retrieval
  - telemetry
  - observability
  - deployment architecture

Mention:

- the system uses indexed client-scoped document collections
- current demo corpus uses `RMIT_Demo`

Estimated duration:

- 30–60 seconds

---

# 2. High-Level Architecture Walkthrough

## Goal

Explain the major system layers.

## On Screen

Use:

- architecture diagram
- highlighted system flow

## Explain

System flow:

```text
User
→ API / Streamlit
→ Orchestration Layer
→ Retrieval Layer
→ Vector Store
→ Grounded Generation
→ Telemetry
```

Discuss:

- FastAPI backend
- Streamlit frontend
- orchestration layer
- ChromaDB vector store
- telemetry layer
- Docker deployment

Estimated duration:

- 60–90 seconds

---

# 3. Document Ingestion Demonstration

## Goal

Show ingestion and indexing workflow.

## On Screen

- Streamlit ingestion UI
- indexed documents
- ingestion diagnostics

## Explain

Discuss:

- TXT/PDF/DOCX ingestion
- chunking strategies
- embeddings
- vector indexing
- readiness diagnostics

Mention:

- client-scoped indexing
- metadata-aware ingestion

Estimated duration:

- 45–60 seconds

---

# 4. Standard Grounded Retrieval Demonstration

## Goal

Show focused grounded retrieval behaviour.

## Demo Question

```text
What does the Academic Promotion Policy say about appeals?
```

## On Screen

Show:

- answer generation
- sources used
- orchestration details
- timing diagnostics

## Explain

Highlight:

- orchestration intent = standard
- retrieval strategy = standard
- focused retrieval pruning
- grounded answer generation
- telemetry visibility

Estimated duration:

- 60–90 seconds

---

# 5. Aggregation Retrieval Demonstration

## Goal

Show orchestration-aware broad retrieval behaviour.

## Demo Question

```text
List all eligibility requirements mentioned in the academic promotion documents.
```

## On Screen

Show:

- broader retrieval results
- orchestration details
- aggregation retrieval strategy

## Explain

Highlight:

- orchestration intent = aggregation
- retrieval strategy = document_balanced
- broader cross-document retrieval
- synthesis across multiple chunks

Mention:

- retrieval precision vs breadth tradeoffs

Estimated duration:

- 60–90 seconds

---

# 6. Unsupported Answer Demonstration

## Goal

Demonstrate grounded refusal behaviour.

## Demo Question

```text
What colour should academic staff dye their hair?
```

## On Screen

Show:

- insufficient evidence response
- grounding failure
- orchestration diagnostics

## Explain

Highlight:

- hallucination avoidance
- grounded answer handling
- governance-aware behaviour
- unsupported-answer workflows

Estimated duration:

- 30–45 seconds

---

# 7. Conversational Retrieval Demonstration

## Goal

Show conversational query rewriting.

## Demo Flow

Question 1:

```text
What committees are involved in promotions?
```

Follow-up:

```text
What are their responsibilities?
```

## On Screen

Show:

- rewritten retrieval query
- orchestration details
- conversational retrieval behaviour

## Explain

Discuss:

- conversational context handling
- query rewriting
- orchestration-aware retrieval

Estimated duration:

- 45–60 seconds

---

# 8. Telemetry & Observability Demonstration

## Goal

Demonstrate operational observability infrastructure.

## On Screen

Show:

- request telemetry logs
- retrieval traces
- timing diagnostics
- orchestration reasoning

## Explain

Highlight:

- request telemetry
- retrieval traces
- stage timings
- retrieval diagnostics
- operational visibility

Mention:

- JSONL telemetry persistence
- retrieval engineering insights

Estimated duration:

- 45–60 seconds

---

# 9. API & Deployment Demonstration

## Goal

Show deployment maturity and API architecture.

## On Screen

Show:

- Swagger UI
- authenticated `/ask` endpoint
- Docker Compose startup

## Explain

Discuss:

- FastAPI backend
- API authentication
- Docker Compose deployment
- persistent vector storage
- operational deployment readiness

Estimated duration:

- 45–60 seconds

---

# 10. Closing Summary

## Goal

Summarise engineering capabilities demonstrated.

## On Screen

- architecture diagram
- orchestration screenshots
- telemetry screenshots
- Streamlit UI montage

## Closing Message

Summarise:

This project demonstrates:

- retrieval engineering
- orchestration-aware AI systems
- conversational retrieval
- grounded generation
- telemetry and observability
- API architecture
- deployment engineering
- operational AI platform design

Conclude with:

> The project evolved significantly beyond a tutorial-style RAG implementation into a modular, observable, and governance-aware AI engineering platform prototype.

Estimated duration:

- 30–45 seconds

---

# Suggested Assets

Recommended visuals:

- architecture diagram
- Streamlit screenshots
- orchestration screenshots
- telemetry screenshots
- Swagger screenshots
- Docker terminal startup
- retrieval trace examples

---

# Suggested Narration Style

Recommended narration:

- calm
- technical
- explanatory
- concise
- engineering-oriented

Avoid:

- excessive marketing language
- exaggerated AI claims
- hype terminology

---

# Future Video Expansion Ideas

Potential future videos:

- retrieval engineering deep dive
- orchestration architecture deep dive
- telemetry & observability walkthrough
- deployment walkthrough
- ingestion pipeline deep dive
- governance-aware AI architecture discussion
- operational AI roadmap walkthrough