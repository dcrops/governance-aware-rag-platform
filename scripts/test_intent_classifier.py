import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.orchestration.intent_classifier import IntentClassifier


classifier = IntentClassifier()

test_questions = [
    "What is the leave policy?",
    "Compare annual leave and sick leave",
    "How many documents mention payroll?",
    "What about payroll?",
]

for question in test_questions:
    decision = classifier.classify(
        question=question,
        conversation_context="Previous discussion about HR appeals process."
    )

    print("=" * 60)
    print("QUESTION:")
    print(question)

    print("\nORCHESTRATION DECISION:")
    print(decision.model_dump(mode="json"))