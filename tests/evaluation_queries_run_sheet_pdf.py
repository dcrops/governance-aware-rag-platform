EVALUATION_CASES = [
    {
        "question": "What is the book of the month?",
        "expected_documents": ["run_sheet_style.pdf"],
        "expected_topics": ["The Dispossessed"],
        "expected_answer_status": "ANSWERED",
    },
    {
        "question": "What discussion questions are listed?",
        "expected_documents": ["run_sheet_style.pdf"],
        "expected_topics": ["power structures", "setting", "society"],
        "expected_answer_status": "ANSWERED",
    },
    {
        "question": "What member updates were discussed?",
        "expected_documents": ["run_sheet_style.pdf"],
        "expected_topics": ["Melbourne chapter", "online participation", "hybrid events"],
        "expected_answer_status": "ANSWERED",
    },
    {
        "question": "What university did the author attend?",
        "expected_documents": ["run_sheet_style.pdf"],
        "expected_topics": [],
        "expected_answer_status": ["INSUFFICIENT_EVIDENCE", "NO_RESULTS"],
    },
]