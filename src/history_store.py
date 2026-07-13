"""
SQLite-backed transaction history store.

Responsibilities:
  - insert_transaction(): write a new (scored) transaction
  - get_account_history(): pull recent history for feature engineering
    and for the get_account_history agent tool
  - update_investigation_status(): write back the human's decision
  - record_investigator_feedback() / get_account_feedback(): long-term
    memory (Task 9/10) — what humans have decided about this account before
"""

import sqlite3
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

DB_PATH = Path(__file__).parent.parent / "db" / "fraud_copilot.db"
SCHEMA_PATH = Path(__file__).parent.parent / "db" / "schema.sql"


@contextmanager
def get_connection():
    """
    Context-managed connection. Using a fresh connection per call rather
    than one long-lived global connection — SQLite connections aren't
    thread-safe by default, and Streamlit/FastAPI both run handlers on
    different threads, so a shared connection would eventually corrupt
    state under concurrent requests.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables/indexes if they don't exist yet. Safe to call every
    startup — CREATE TABLE IF NOT EXISTS is idempotent."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn, open(SCHEMA_PATH) as f:
        conn.executescript(f.read())


def insert_transaction(transaction: dict, fraud_probability: float | None = None,
                        is_flagged: bool = False) -> None:
    """
    Write a transaction row. Called by the CSV-replay stream for every
    transaction, and again (as an UPDATE) once scoring completes — see
    update_score() below for that second step.
    """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO transactions (
                transaction_id, sender_account, receiver_account, amount,
                timestamp, merchant_category, transaction_type,
                spending_deviation_score, velocity_score, geo_anomaly_score,
                ip_address, location, payment_channel, device_hash,
                fraud_probability, is_flagged
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transaction["transaction_id"],
                transaction["sender_account"],
                transaction["receiver_account"],
                transaction["amount"],
                transaction["timestamp"],
                transaction.get("merchant_category"),
                transaction.get("transaction_type"),
                transaction.get("spending_deviation_score"),
                transaction.get("velocity_score"),
                transaction.get("geo_anomaly_score"),
                transaction.get("ip_address"),
                transaction.get("location"),
                transaction.get("payment_channel"),
                transaction.get("device_hash"),
                fraud_probability,
                int(is_flagged),
            ),
        )


def update_score(transaction_id: str, fraud_probability: float, is_flagged: bool) -> None:
    """Called right after score_transaction() runs, to attach the score
    to a transaction that was already inserted pre-scoring."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE transactions SET fraud_probability = ?, is_flagged = ? WHERE transaction_id = ?",
            (fraud_probability, int(is_flagged), transaction_id),
        )


def get_account_history(account_id: str, limit: int = 50, as_sender: bool = True) -> pd.DataFrame:
    """
    Pull an account's recent transactions, most recent first, capped at
    `limit`. Capping matters: unbounded history makes the groupby-based
    feature engineering in scoring.py slower as the table grows, without
    adding meaningful signal past the recent window.

    as_sender=True looks up transactions where this account was the
    sender; set False to look up receiver-side history (used when the
    agent decides to investigate the receiver instead of/in addition to
    the sender — see tools.py's get_account_history tool).
    """
    column = "sender_account" if as_sender else "receiver_account"
    query = f"""
        SELECT * FROM transactions
        WHERE {column} = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=(account_id, limit))
    return df.iloc[::-1].reset_index(drop=True)  # oldest-first, matches training-time ordering assumptions


def update_investigation_status(transaction_id: str, status: str) -> None:
    """status: 'pending' | 'cleared' | 'confirmed_fraud'."""
    valid = {"none", "pending", "cleared", "confirmed_fraud"}
    if status not in valid:
        raise ValueError(f"status must be one of {valid}, got '{status}'")
    with get_connection() as conn:
        conn.execute(
            "UPDATE transactions SET investigation_status = ? WHERE transaction_id = ?",
            (status, transaction_id),
        )


def record_investigator_feedback(transaction_id: str, account_id: str,
                                  decision: str, notes: str = "") -> None:
    """
    Long-term memory write. Called once the human resolves the HITL gate
    (Task 6). This is what get_account_feedback() below reads from on
    future investigations of the same account.
    """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO investigator_feedback
                (transaction_id, account_id, decision, notes, decided_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (transaction_id, account_id, decision, notes,
             datetime.now(timezone.utc).isoformat()),
        )


def get_account_feedback(account_id: str, limit: int = 5) -> list[dict]:
    """
    Past human decisions for this account — this is what makes long-term
    memory load-bearing rather than decorative (Task 9/10). Feed this into
    the investigation report prompt: "this account was previously cleared
    twice and flagged once" changes how the LLM should weigh new evidence.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT decision, notes, decided_at FROM investigator_feedback
            WHERE account_id = ?
            ORDER BY decided_at DESC
            LIMIT ?
            """,
            (account_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]