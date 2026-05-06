# FUTURE_ROADMAP.md

## Near-Term Priorities

### Client MVP Phase 1

Focus:

- usability
    
- operational workflows
    
- document lifecycle management
    
- deployment
    
- trust indicators
    

Planned features:

- indexed document registry
    
- delete/reindex workflows
    
- simplified client UI
    
- deployment support
    
- authentication baseline
    

---

## Governance Layer Expansion

Planned:

- retrieval governance dashboards
    
- hallucination analysis
    
- groundedness scoring
    
- ingestion governance
    
- retrieval QA tooling
    
- strategy recommendation engine
    

---

## AI-Ready Document Representation

Future direction:

- detect weak document structure
    
- identify duplicate content
    
- identify stale content
    
- transform documents into AI-ready retrieval representations
    
- preserve original source documents while improving retrieval quality
    

---

## Delimiter Candidate Detection

Future diagnostics enhancement:

- automatically detect repeated structural delimiters
    
- suggest delimiters such as:
    
    - `SECTION:`
        
    - `QUESTION:`
        
    - `Agenda Item:`
        
    - `Book Title -`
        

Purpose:

- improve delimiter-aware retrieval
    
- reduce manual ingestion tuning
    

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
    

---

# Current Project Status

Current state:

- strong RAG engineering prototype
    
- governance-aware retrieval foundation
    
- evaluation infrastructure operational
    
- client MVP groundwork established
    

Next major transition:

- pivot from retrieval experimentation toward operational client workflows and deployment readiness