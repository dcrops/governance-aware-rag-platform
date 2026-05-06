EVALUATION_CASES = [
    {
        "question": "What improvements were discussed for payroll diagnostics reporting?",
        "expected_documents": ["meeting_notes_style.pdf"],
        "expected_topics": ["payroll diagnostics reporting", "retrieval evaluation"],
        "expected_answer_status": "ANSWERED",
    },
    {
        "question": "What concern was raised about client policy packs?",
        "expected_documents": ["meeting_notes_style.pdf"],
        "expected_topics": ["inconsistent document formatting", "client policy packs"],
        "expected_answer_status": "ANSWERED",
    },
    {
        "question": "What did the group agree to evaluate?",
        "expected_documents": ["meeting_notes_style.pdf"],
        "expected_topics": ["heading chunking", "character chunking", "synthetic test datasets"],
        "expected_answer_status": "ANSWERED",
    },
    {
        "question": "What university did the author attend?",
        "expected_documents": ["meeting_notes_style.pdf"],
        "expected_topics": [],
        "expected_answer_status": ["INSUFFICIENT_EVIDENCE", "NO_RESULTS"],
    },
]