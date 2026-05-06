EVALUATION_CASES = [
    {
        "question": "How do employees request annual leave?",
        "expected_documents": ["faq_style.pdf"],
        "expected_topics": ["HR portal", "leave requests", "managers"],
        "expected_answer_status": "ANSWERED",
    },
    {
        "question": "When is payroll processed?",
        "expected_documents": ["faq_style.pdf"],
        "expected_topics": ["Wednesday afternoon", "approvals"],
        "expected_answer_status": "ANSWERED",
    },
    {
        "question": "How long are payroll records retained?",
        "expected_documents": ["faq_style.pdf"],
        "expected_topics": ["seven years", "payroll records"],
        "expected_answer_status": "ANSWERED",
    },
    {
        "question": "What university did the author attend?",
        "expected_documents": ["faq_style.pdf"],
        "expected_topics": [],
        "expected_answer_status": ["INSUFFICIENT_EVIDENCE", "NO_RESULTS"],
    },
]