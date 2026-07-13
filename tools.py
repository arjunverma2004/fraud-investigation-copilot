"""
Investigation-time tools exposed to the LangGraph agent. Unlike
check_transaction_fraud (called exactly once, deterministically),
these are genuinely optional — the LLM decides whether and how many
times to call them based on the flagged transaction's evidence.
"""

from langchain_core.tools import tool


@tool
def get_account_history(account_id: str, role_hint: str = "unspecified") -> dict:
    """
    Fetch recent transaction history for a given account ID.

    Use this to investigate either the sender or the receiver of a
    flagged transaction — call it once per account you want to examine.
    You do not need to check both accounts every time; only fetch
    history for an account if there's a specific reason to suspect it
    (e.g. unusually high receiver_degree suggesting a mule account, or
    the top contributing features point to receiver-side signals).

    Args:
        account_id: The account to look up.
        role_hint: "sender" or "receiver" — for logging/context only,
            does not affect the lookup itself.

    Returns:
        dict with the account's recent transactions and summary stats
        (total transaction count, average amount, distinct counterparties).
    """
    from src.history_store import get_account_history as _get_history

    history_df = _get_history(account_id, limit=50)

    if history_df.empty:
        return {
            "account_id": account_id,
            "role_hint": role_hint,
            "transaction_count": 0,
            "note": "No prior history found — likely a new account.",
        }

    return {
        "account_id": account_id,
        "role_hint": role_hint,
        "transaction_count": len(history_df),
        "avg_amount": round(float(history_df["amount"].mean()), 2),
        "distinct_counterparties": int(
            history_df["receiver_account"].nunique()
            if role_hint == "sender"
            else history_df["sender_account"].nunique()
        ),
        "recent_transactions": history_df.tail(5).to_dict(orient="records"),
    }