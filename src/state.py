from typing import TypedDict



class InvestigationState(TypedDict):
    transaction: dict
    fraud_score: dict
    account_checks: list[dict]
    retrieved_patterns: list[str]
    account_feedback_history: list[dict]  
    investigation_report: str
    human_decision: str | None
    human_notes: str | None
    messages: list                         
    iteration_count: int
    