from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages




class InvestigationState(TypedDict):
    transaction: dict
    fraud_score: dict
    account_checks: list[dict]
    retrieved_patterns: list[str]
    account_feedback_history: list[dict]  
    investigation_report: str
    human_decision: str | None
    human_notes: str | None
    messages: Annotated[list, add_messages]                       
    iteration_count: int
    recommendation: str | None