"""
LangGraph workflow for the Fraud Investigation Copilot.

Build order (deliberate): score_node + branch first, tested in isolation.
investigate_node, report_node, and the HITL gate are added in later tasks
once this foundation is confirmed working.
"""

import uuid
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver  # swap for SqliteSaver later (Task 9)

from state import InvestigationState
from scoring import Transaction, score_transaction
from history_store import get_account_history, update_score


# ---------------------------------------------------------------------
# score_node: deterministic, no LLM involved. Pulls sender history,
# scores the transaction, writes the score into state, and persists
# the score to the DB as a side effect.
# ---------------------------------------------------------------------
def score_node(state: InvestigationState) -> InvestigationState:
    txn_dict = state["transaction"]
    txn = Transaction(**txn_dict)

    sender_history = get_account_history(txn.sender_account, as_sender=True)
    fraud_score = score_transaction(txn, sender_history)

    # Side effect: persist the score against the transaction row.
    # Nodes are allowed DB side effects — only investigation LOGIC flows
    # through state, per our earlier design.
    update_score(
        transaction_id=txn_dict["transaction_id"],
        fraud_probability=fraud_score["fraud_probability"],
        is_flagged=fraud_score["is_flagged"],
    )

    return {**state, "fraud_score": fraud_score}


# ---------------------------------------------------------------------
# Conditional routing: only branch that exists right now.
# ---------------------------------------------------------------------
def route_after_scoring(state: InvestigationState) -> str:
    return "investigate_node" if state["fraud_score"]["is_flagged"] else END


# ---------------------------------------------------------------------
# Placeholder nodes — stubbed so the graph is runnable end-to-end now,
# but these are NOT the real implementations. Replace in later tasks.
# ---------------------------------------------------------------------
def investigate_node(state: InvestigationState) -> InvestigationState:
    # TODO (Task 6): bind tools.get_account_history and
    # knowledge_base.search_fraud_patterns to an LLM, loop until no more
    # tool calls, capped at N iterations.
    print("[stub] investigate_node reached — flagged transaction, "
          "real tool-calling logic not implemented yet.")
    return {
        **state,
        "account_checks": [],
        "retrieved_patterns": [],
        "messages": [],
    }


# ---------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------
def build_graph():
    builder = StateGraph(InvestigationState)

    builder.add_node("score_node", score_node)
    builder.add_node("investigate_node", investigate_node)  # stub for now

    builder.add_edge(START, "score_node")
    builder.add_conditional_edges(
        "score_node",
        route_after_scoring,
        {"investigate_node": "investigate_node", END: END},
    )
    builder.add_edge("investigate_node", END)  # temporary — will route to report_node later

    # Checkpointer needed even at this stage if you want to test resuming;
    # harmless to include now, becomes load-bearing once interrupt() (Task 7) exists.
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()


# ---------------------------------------------------------------------
# Manual test harness — run this file directly to sanity-check the
# score_node + branch in isolation, per the recommended build order.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sample_transaction = {
        "transaction_id": str(uuid.uuid4()),
        "sender_account": "ACC1001",
        "receiver_account": "ACC2002",
        "amount": 4500.0,
        "timestamp": "2026-07-13T02:15:00",
        "merchant_category": "electronics",
        "transaction_type": "transfer",
        "spending_deviation_score": 3.2,
        "velocity_score": 15.0,
        "geo_anomaly_score": 0.8,
        "ip_address": "10.0.0.1",
        "device_used": "mobile",
        "location": "Delhi",
        "payment_channel": "UPI",
        "device_hash": "abc123",
        "time_since_last_transaction": 3600.0
    }

    initial_state: InvestigationState = {
        "transaction": sample_transaction,
        "fraud_score": {},
        "account_checks": [],
        "retrieved_patterns": [],
        "account_feedback_history": [],
        "investigation_report": "",
        "human_decision": None,
        "human_notes": None,
        "messages": [],
        "iteration_count": 0,
    }

    config = {"configurable": {"thread_id": sample_transaction["transaction_id"]}}
    result = graph.invoke(initial_state, config=config)

    print("\nFinal state:")
    print("Fraud score:", result["fraud_score"])