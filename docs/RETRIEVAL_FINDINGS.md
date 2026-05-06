# RETRIEVAL_FINDINGS.md

## Purpose

The Retrieval Evaluation Framework was developed to experimentally evaluate how different chunking strategies affect retrieval quality across varying document archetypes.

The goal was to:

- compare retrieval strategies empirically
- identify document-type-aware retrieval patterns
- validate ingestion diagnostics and chunking recommendations
- support future governance-aware ingestion logic
- improve retrieval trustworthiness and observability

The evaluation framework tested:

- character chunking
- page chunking
- heading chunking
- delimiter chunking

Evaluation metrics included:

- document hit rate
- topic match rate
- average top retrieval score
- average retrieval score
- answer status match rate
- unsupported-answer handling

---

## Document Archetypes Tested

|   |   |   |
|---|---|---|
|Document|Type|Purpose|
|policy_style|Structured governance policy|Test section-aware retrieval|
|faq_style|Repeated Q&A format|Test delimiter-aware retrieval|
|meeting_notes_style|Conversational notes|Test semantic continuity|
|poor_structure|Weakly structured text|Test fallback behaviour|
|run_sheet_style|Semi-structured mixed content|Test mixed semantic retrieval|

---

## Retrieval Strategy Findings

|   |   |   |
|---|---|---|
|Document Type|Best Strategy|Observation|
|FAQ documents|Delimiter|Repeated QUESTION: structure significantly improved retrieval quality|
|Structured policies|Delimiter / Heading|Section-aware chunking improved semantic isolation|
|Meeting notes|No significant difference|Conversational blended semantics reduced chunking impact|
|Poor structure documents|Character|Advanced chunking offered little benefit|
|Semi-structured run sheets|Character|Aggressive splitting reduced semantic continuity|

---

## Detailed Findings

### FAQ Documents

Delimiter chunking using:

QUESTION:

produced the strongest retrieval scores.

Key observation:

- precise delimiter selection materially improved retrieval quality
- fuzzy delimiters underperformed compared to exact structural delimiters

This validated the value of delimiter-aware retrieval for repeated-format documents.

---

### Structured Policy Documents

Policy documents benefited from:

- delimiter chunking
- heading chunking

because semantic sections were:

- strongly isolated
- operationally independent
- structurally consistent

Example sections:

- LEAVE MANAGEMENT
- PAYROLL APPROVALS
- TERMINATION PROCESS

This suggests structured governance documents are well suited to section-aware retrieval.

---

### Meeting Notes

Meeting notes showed minimal variation across chunking strategies.

Observation:

- semantic meaning was distributed conversationally across the document
- structural chunking boundaries provided little retrieval advantage

This suggests:

- conversational blended documents may not require advanced chunking logic
- simple character chunking may be operationally sufficient

---

### Poorly Structured Documents

Poorly structured documents performed similarly across all chunking strategies.

Delimiter and heading chunking produced little benefit because:

- headings were inconsistent
- semantic structure was weak
- sections were not clearly separated

This validated a major governance insight:

> Retrieval quality is constrained by document quality.

Character chunking proved the safest fallback strategy.

---

### Semi-Structured Run Sheets

Run sheet documents produced weaker retrieval when aggressively segmented.

Observation:

- heading and delimiter chunking reduced semantic continuity
- character chunking preserved broader contextual windows

This demonstrated:

- more advanced chunking is not automatically better
- preserving semantic continuity can matter more than structural isolation

---

## Governance Insights

### Document Quality Matters

The evaluation framework demonstrated that:

- retrieval quality depends heavily on document quality
- poorly structured documents reduce retrieval optimisation effectiveness
- governance-aware ingestion is important for trustworthy AI systems

---

### Retrieval Strategy Should Adapt To Document Type

No single chunking strategy performed best across all document archetypes.

The evaluation findings support future:

- document-aware retrieval
- ingestion diagnostics
- strategy recommendation systems

---

### Unsupported-Answer Handling Is Critical

The system successfully evaluated:

- grounded answers
- insufficient evidence handling
- no-results handling

This supports governance-aware AI behaviour and reduces unsupported responses.

---

### Observability Is Essential

The project now supports:

- retrieval telemetry
- retrieval benchmarking
- answer status evaluation
- persisted benchmark runs

This enables:

- retrieval experimentation
- regression testing
- evidence-based tuning

---

## Future Retrieval Roadmap

### Planned Enhancements

- delimiter candidate detection
- reranker comparison
- embedding model comparison
- groundedness scoring
- hallucination analysis
- automated retrieval benchmarking
- ingestion governance scoring
- duplicate document detection
- AI-ready document transformation layer

---

## Strategic Outcome

The Retrieval Evaluation Framework evolved the project from:

> "basic RAG experimentation"

into:

> governance-aware retrieval engineering infrastructure.

The system now supports:

- measurable retrieval evaluation
- evidence-based chunking strategy analysis
- ingestion diagnostics
- unsupported-answer governance
- retrieval observability

This foundation directly supports future CRC AI Services client-facing RAG systems.

---

# Key Engineering Insights Learned

## Retrieval Strategy Findings

- Chunking strategy effectiveness depends heavily on document structure.
- Structured repeated-format documents benefit strongly from delimiter-aware chunking.
- Poorly structured documents limit the usefulness of advanced chunking strategies.
- Conversational documents often perform well with simple character chunking.
- More advanced chunking is not automatically better.

---

## Governance Insights

- Retrieval quality depends heavily on document quality.
- Unsupported-answer handling is a critical governance feature.
- Evaluation metrics require semantic interpretation.
- Observability and diagnostics are essential for trustworthy AI systems.
- AI knowledge systems require governance-aware ingestion and evaluation layers.