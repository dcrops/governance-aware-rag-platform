from enum import Enum
from pydantic import BaseModel


class RetrievalIntent(str, Enum):
    STANDARD = "standard"
    COMPARISON = "comparison"
    AGGREGATION = "aggregation"
    CLARIFICATION = "clarification"


class OrchestrationDecision(BaseModel):
    intent: RetrievalIntent
    answer_mode: str
    clarification_required: bool = False
    retrieval_strategy: str = "standard"
    reasoning: str