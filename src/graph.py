"""
LangGraph workflow for the Fraud Investigation Copilot.
"""

import uuid
import sqlite3
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage, HumanMessage, RemoveMessage
from pydantic import BaseModel, Field
from typing import Literal

from state import InvestigationState
from scoring import Transaction, score_transaction
from history_store import (
    get_account_history,
    update_score,
    update_investigation_status,
    record_investigator_feedback,
    get_account_feedback,
)
from knowledge_base import search_fraud_patterns
from tools import get_account_history as get_account_history_tool

load_dotenv()

# ---------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.1,
    timeout=45,
    max_output_tokens=8192,
    max_retries=3,
)

# Scoring is deterministic and already ran in score_node — it's not
# exposed here as an LLM-callable tool during investigation.
tools = [get_account_history_tool, search_fraud_patterns]

SYSTEM_PROMPT = """
You are a Senior Fraud Investigation Copilot. Your objective is to investigate financial transactions that have been flagged by our XGBoost machine learning model.

You have access to the flagged transaction details and its initial fraud score. You must act as a detective to determine WHY the transaction was flagged and whether it constitutes genuine fraud or a false positive.

### Available Tools:
1. `get_account_history`: Use this to pull the last 50 transactions of either the sender or the receiver.
   - Call this if the ML model indicates "receiver_degree" or "velocity_score" as top contributing features.
   - Look for signs of mule accounts (new accounts receiving many transfers) or account takeovers (sudden spikes in spending).
2. `search_fraud_patterns`: Use this to query the vector database of known fraud typologies.
   - Call this if the transaction pattern is ambiguous or if you need precedent for specific merchant categories or geographic anomalies.

### Investigation Protocol:
1. Analyze the Context: Review the transaction details and the specific features that drove the high fraud probability score.
2. Gather Evidence: You MUST use the available tools to gather more context. Do not guess. If the sender's history wasn't enough to clear the flag, check the receiver's history.
3. Synthesize: Compare the transaction against the retrieved history and known fraud patterns.
4. Formulate a Conclusion: Determine your findings.
"""

llm_with_tools = model.bind_tools(tools)


class InvestigationReport(BaseModel):
    detailed_markdown_report: str = Field(
        description="A highly detailed, professional investigation report formatted in Markdown."
    )
    recommendation: Literal["clear", "investigate_further", "likely_fraud"] = Field(
        description="The final definitive recommendation for routing."
    )


report_llm = model.with_structured_output(InvestigationReport)

REPORT_PROMPT = """
You are a Senior Fraud Investigation Analyst. You have completed the evidence-gathering phase for a flagged transaction.

Review the preceding conversation history containing the initial transaction details and raw tool outputs. Synthesize this evidence into a final, structured Investigation Report.

CRITICAL CONSTRAINTS:
1. STRICT CONCISENESS: Your detailed Markdown report must be highly analytical but strictly under 300 words.
2. STRUCTURE: Use Markdown headers (e.g., ### Executive Summary, ### Anomaly Analysis).
3. ZERO HALLUCINATION: Ground your analysis STRICTLY in the provided tool outputs. If a tool returned "No prior history", explicitly state this as a risk factor.
4. NO FURTHER ACTIONS: Do not attempt to use any more tools.
5. DEFINITIVE CONCLUSION: Provide a clear recommendation (clear, investigate_further, or likely_fraud).
"""

DB_PATH = Path(__file__).parent.parent / "db" / "checkpoints.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# score_node: deterministic, no LLM involved.
# ---------------------------------------------------------------------
@traceable
def score_node(state: InvestigationState) -> InvestigationState:
    txn_dict = state["transaction"]
    txn = Transaction(**txn_dict)

    sender_history = get_account_history(txn.sender_account, as_sender=True)
    fraud_score = score_transaction(txn, sender_history)

    update_score(
        transaction_id=txn_dict["transaction_id"],
        fraud_probability=fraud_score["fraud_probability"],
        is_flagged=fraud_score["is_flagged"],
    )

    return {**state, "fraud_score": fraud_score}


@traceable
def route_after_scoring(state: InvestigationState) -> str:
    return "investigate_node" if state["fraud_score"]["is_flagged"] else END


# ---------------------------------------------------------------------
# investigate_node: agentic tool-calling loop.
# ---------------------------------------------------------------------
@traceable
def investigate_node(state: InvestigationState) -> InvestigationState:
    transaction = state["transaction"]
    fraud_score = state["fraud_score"]

    if not state["messages"]:
        sender_feedback = get_account_feedback(transaction["sender_account"])
        receiver_feedback = get_account_feedback(transaction["receiver_account"])

        feedback_context = ""
        if sender_feedback or receiver_feedback:
            feedback_context = (
                f"\n\nRelevant history of past human investigator decisions:\n"
                f"Sender ({transaction['sender_account']}): {sender_feedback}\n"
                f"Receiver ({transaction['receiver_account']}): {receiver_feedback}\n"
                f"Weigh this prior context appropriately."
            )

        new_messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Investigate the following flagged transaction: {transaction} "
                    f"with fraud score: {fraud_score}{feedback_context}"
                )
            ),
        ]
        conversation = new_messages  # what we send to the LLM THIS call
        account_feedback_history = sender_feedback + receiver_feedback
    else:
        conversation = state["messages"]
        new_messages = []  # nothing new to prepend — state already has everything
        account_feedback_history = state["account_feedback_history"]

    response = llm_with_tools.invoke(conversation)

    return {
        **state,
        # First call: [System, Human, AIResponse] — all new.
        # Later calls: just [AIResponse] — the rest is already persisted.
        "messages": new_messages + [response],
        "iteration_count": state["iteration_count"] + 1,
        "account_feedback_history": account_feedback_history,
    }


@traceable
def route_after_investigate(state: InvestigationState) -> str:
    if state["iteration_count"] >= 4:
        return "report_node"
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "report_node"


@traceable
def report_node(state: InvestigationState) -> InvestigationState:
    conversation = state["messages"]
    state_updates = {}

    if getattr(conversation[-1], "tool_calls", None):
        # Dangling tool call — iteration cap fired before it was resolved.
        # Actually remove it from persisted state via RemoveMessage, not
        # just skip it locally, or it corrupts any future continuation.
        dangling_id = conversation[-1].id
        conversation = conversation[:-1]
        state_updates["messages"] = [RemoveMessage(id=dangling_id)]

    conversation_for_llm = conversation + [HumanMessage(content=REPORT_PROMPT)]
    report_response = report_llm.invoke(conversation_for_llm)

    return {
        **state,
        **state_updates,
        "investigation_report": report_response.detailed_markdown_report,
        "recommendation": report_response.recommendation,
    }


tool_node = ToolNode(tools)


# ---------------------------------------------------------------------
# Task 7: HITL gate
# ---------------------------------------------------------------------
@traceable
def human_review_node(state: InvestigationState) -> InvestigationState:
    decision_payload = interrupt(
        {
            "report": state["investigation_report"],
            "recommendation": state["recommendation"],
            "fraud_score": state["fraud_score"],
            "transaction": state["transaction"],
        }
    )
    decision = decision_payload.get("decision")
    notes = decision_payload.get("notes", "")

    txn_id = state["transaction"]["transaction_id"]
    account_id = state["transaction"]["sender_account"]

    status_map = {
        "approve": "cleared",
        "reject": "confirmed_fraud",
        "need_more_info": "pending",
    }
    update_investigation_status(txn_id, status_map.get(decision, "pending"))

    if decision in ("approve", "reject"):
        record_investigator_feedback(txn_id, account_id, decision, notes)

    new_messages = state["messages"]
    if decision == "need_more_info":
        followup_note = (
            f"A human reviewer has requested more information before making "
            f"a final decision on this transaction. Reviewer's note: "
            f"\"{notes or 'No specific note provided — re-examine the evidence more thoroughly.'}\" "
            f"Continue your investigation, using additional tool calls if needed, "
            f"and address this specific concern in your next report."
        )
        new_messages = new_messages + [HumanMessage(content=followup_note)]

    return {
        **state,
        "human_decision": decision,
        "human_notes": notes,
        "messages": new_messages,
    }


# ---------------------------------------------------------------------
# Task 8: iterative loop
# ---------------------------------------------------------------------
@traceable
def route_after_human(state: InvestigationState) -> str:
    if state["human_decision"] == "need_more_info" and state["iteration_count"] < 6:
        return "investigate_node"
    return END


# ---------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------
def build_graph():
    builder = StateGraph(InvestigationState)

    builder.add_node("score_node", score_node)
    builder.add_node("investigate_node", investigate_node)
    builder.add_node("report_node", report_node)
    builder.add_node("tools", tool_node)
    builder.add_node("human_review_node", human_review_node)

    builder.add_edge(START, "score_node")
    builder.add_conditional_edges(
        "score_node",
        route_after_scoring,
        {"investigate_node": "investigate_node", END: END},
    )
    builder.add_conditional_edges("investigate_node", route_after_investigate)
    builder.add_edge("tools", "investigate_node")
    builder.add_edge("report_node", "human_review_node")
    builder.add_conditional_edges(
        "human_review_node",
        route_after_human,
        {"investigate_node": "investigate_node", END: END},
    )

    # timeout=30 makes concurrent writers wait for the lock instead of
    # raising "database is locked" immediately. SqliteSaver already
    # serializes access to this connection internally via its own
    # threading.Lock (checked its source directly), so ONE shared
    # compiled graph using ONE connection is the correct, supported
    # pattern — not a per-request connection. Recompiling the graph and
    # opening a fresh SqliteSaver.from_conn_string() on every single
    # transaction (as a previous version of this file did) means every
    # worker thread hits the same SQLite file with its own unmanaged
    # connection simultaneously, which is what was actually causing the
    # "database is locked" failures right after score_node.
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    checkpointer = SqliteSaver(conn)
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()