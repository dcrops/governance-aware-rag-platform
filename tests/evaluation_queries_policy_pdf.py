EVALUATION_CASES = [
    {
        "question": "How long must payroll records be retained?",
        "expected_documents": ["policy_style.pdf"],
        "expected_topics": [
            "seven years",
            "payroll records",
            "retained",
        ],
        "expected_answer_status": "ANSWERED",
    },
    {
        "question": "Who can access payroll records?",
        "expected_documents": ["policy_style.pdf"],
        "expected_topics": [
            "authorised personnel",
            "payroll records",
        ],
        "expected_answer_status": "ANSWERED",
    },
    {
        "question": "What time must payroll approvals occur?",
        "expected_documents": ["policy_style.pdf"],
        "expected_topics": [
            "3:00 PM",
            "Wednesday",
            "payroll approval",
        ],
        "expected_answer_status": "ANSWERED",
    },
    {
        "question": "What university did the author attend?",
        "expected_documents": ["policy_style.pdf"],
        "expected_topics": [],
        "expected_answer_status": ["INSUFFICIENT_EVIDENCE", "NO_RESULTS"],
    },
]