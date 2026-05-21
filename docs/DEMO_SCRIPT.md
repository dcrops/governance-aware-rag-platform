# CRC Governance-Aware RAG — Demo Script

## Demo Goal

Show how the system answers document questions with grounded evidence, orchestration visibility, and retrieval telemetry.

---

## Demo Dataset

This demo uses the `RMIT_Demo` indexed client collection.

The indexed corpus includes governance and academic policy documents such as:

- Academic Promotion Policy
- Academic Integrity Policy
- Academic Integrity Procedure
- Academic Board Regulations
- Academic Dress Regulations

The system performs retrieval and grounded answer generation over these indexed documents.

## 1. Standard Grounded Question

Question:

> What does the Academic Promotion Policy say about appeals?

Expected behaviour:

- focused retrieval
- answer status: ANSWERED
- grounding check: PASS
- sources from Academic Promotion Policy
- orchestration intent: standard

---

## 2. Aggregation Question

Question:

> List all eligibility requirements mentioned in the academic promotion documents.

Expected behaviour:

- orchestration intent: aggregation
- retrieval strategy: document_balanced
- broader evidence retrieval
- answer summarises multiple eligibility conditions

---

## 3. Unsupported Question

Question:

> What colour should academic staff dye their hair?

Expected behaviour:

- system refuses unsupported answer
- answer status: INSUFFICIENT_EVIDENCE
- grounding check: FAIL
- hallucination avoided

---

## 4. Clarification Handling

First ask:

> What is the appeals process?

Then ask:

> What about payroll?

Expected behaviour:

- system detects ambiguous follow-up
- answer status: CLARIFICATION_REQUIRED
- retrieval skipped
- asks user to clarify

---

## 5. Conversational Follow-Up

First ask:

> What committees are involved in promotions?

Then ask:

> What are their responsibilities?

Expected behaviour:

- conversational query rewriting occurs
- retrieval query becomes more specific
- orchestration details show rewritten retrieval query

Known limitation:

- retrieval may still include nearby governance actors due to semantic overlap.

---

## 6. Observability Demonstration

Open orchestration details and show:

- retrieval query
- orchestration intent
- retrieval strategy
- grounding status
- total API duration
- setup duration
- pipeline duration
- orchestration duration
- retrieval duration
- generation duration

---

## 7. API Demonstration

Open:

```text
http://localhost:8000/docs

Show:

- /ask
- /health
- /version
- API key authentication
- structured error handling

## 8. Docker Compose Demonstration

Run:

docker compose up --build

Explain:

- Streamlit frontend service
- FastAPI backend service
- shared data volume
- API key protected backend
- persistent vector store