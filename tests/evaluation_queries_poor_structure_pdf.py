EVALUATION_CASES = [
    {
        "question": "What topics are mentioned in this document?",
        "expected_documents": ["poor_structure.pdf"],
        "expected_topics": ["payroll", "leave", "termination"],
        "expected_answer_status": "ANSWERED",
    },
    {
        "question": "Does the document mention FastAPI or Streamlit?",
        "expected_documents": ["poor_structure.pdf"],
        "expected_topics": ["fastapi", "streamlit"],
        "expected_answer_status": "ANSWERED",
    },
    {
        "question": "Is the document well structured?",
        "expected_documents": ["poor_structure.pdf"],
        "expected_topics": ["no headings", "disconnected thoughts"],
        "expected_answer_status": "ANSWERED",
    },
    {
        "question": "What university did the author attend?",
        "expected_documents": ["poor_structure.pdf"],
        "expected_topics": [],
        "expected_answer_status": ["INSUFFICIENT_EVIDENCE", "NO_RESULTS"],
    },
]