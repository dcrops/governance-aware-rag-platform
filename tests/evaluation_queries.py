EVALUATION_CASES = [
    {
        "question": "What is this document about?",
        "expected_documents": ["sample.txt"],
        "expected_topics": [
            "AI learning",
            "projects",
            "LLMs",
            "FastAPI",
            "personal reflections",
        ],
    },
    {
        "question": "What APIs are mentioned?",
        "expected_documents": ["sample.txt"],
        "expected_topics": [
            "FastAPI",
            "OpenAI APIs",
            "backend services",
        ],
    },
    {
        "question": "What AI technologies are mentioned?",
        "expected_documents": ["sample.txt"],
        "expected_topics": [
            "LLMs",
            "Streamlit",
            "LangChain",
            "FastAPI",
            "OpenAI APIs",
        ],
    },
    {
        "question": "What projects has the author worked on?",
        "expected_documents": ["sample.txt"],
        "expected_topics": [
            "custom AI models",
            "chatbots",
            "Address to Public Holiday compliance app",
        ],
    },
    {
        "question": "Does the document discuss FastAPI?",
        "expected_documents": ["sample.txt"],
        "expected_topics": [
            "FastAPI",
            "backend engineering",
            "API development",
        ],
    },
    {
        "question": "What personal topics are mentioned?",
        "expected_documents": ["sample.txt"],
        "expected_topics": [
            "dog adoption",
            "personal reflections",
            "community involvement",
        ],
    },
    {
        "question": "What is the author's favourite food?",
        "expected_documents": ["sample.txt"],
        "expected_topics": [],
        "expected_answer_status": "INSUFFICIENT_EVIDENCE",
    },
    {
        "question": "What company does the author currently work for?",
        "expected_documents": ["sample.txt"],
        "expected_topics": [],
        "expected_answer_status": "INSUFFICIENT_EVIDENCE",
    },
    {
        "question": "What university did the author attend?",
        "expected_documents": ["sample.txt"],
        "expected_topics": [],
        "expected_answer_status": "INSUFFICIENT_EVIDENCE",
    },
]