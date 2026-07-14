"""
Fraud scoring module.

Loads the trained XGBoost model + preprocessing artifacts ONCE at import
time, and exposes:
  - Transaction: Pydantic schema for input validation
  - score_transaction(): core scoring function (deterministic, not a tool)
  - check_transaction_fraud(): LangChain @tool wrapper around scoring,
    for use inside the agent graph
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool

# ---------------------------------------------------------------------
# 1. Load artifacts once at import time (not per call)
# ---------------------------------------------------------------------
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

_model = joblib.load(ARTIFACTS_DIR / "fraud_xgb_model.joblib")
_ordinal_encoder = joblib.load(ARTIFACTS_DIR / "ordinal_encoder.joblib")
_imputer = joblib.load(ARTIFACTS_DIR / "imputer.joblib")

with open(ARTIFACTS_DIR / "metadata.json") as f:
    _metadata = json.load(f)

_FEATURE_COLUMNS = _metadata["feature_columns"]
_CATEGORICAL_COLUMNS = _metadata["categorical_columns"]
_NUMERIC_COLUMNS = _metadata["numeric_columns_for_imputation"]
_FLAG_THRESHOLD = _metadata["flag_threshold"]


# ---------------------------------------------------------------------
# 2. Pydantic schema — validates every transaction before it touches
#    the model. Catches malformed agent-generated calls early instead
#    of letting bad data silently produce a meaningless probability.
# ---------------------------------------------------------------------
class Transaction(BaseModel):
    sender_account: str
    receiver_account: str
    amount: float = Field(gt=0, description="Transaction amount, must be positive")
    timestamp: str  # ISO format, parsed downstream
    merchant_category: str
    transaction_type: str
    spending_deviation_score: float
    velocity_score: float
    geo_anomaly_score: float
    device_used: str
    ip_address: str
    location: str
    payment_channel: str
    device_hash: str
    time_since_last_transaction: float

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        # Fail fast with a clear error instead of a silent NaT downstream
        parsed = pd.to_datetime(v, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"timestamp '{v}' is not a valid datetime string")
        return v


# ---------------------------------------------------------------------
# 3. Core scoring function — deterministic, plain Python.
#    NOT wrapped as a tool itself: scoring doesn't need LLM judgment,
#    only the investigation step (history lookup) does.
# ---------------------------------------------------------------------
def score_transaction(transaction: Transaction, sender_history: pd.DataFrame) -> dict:
    """
    Score a single transaction for fraud risk.

    Args:
        transaction: validated Transaction object.
        sender_history: DataFrame of the sender's past transactions
            (raw schema, same columns as the training CSV). Can be empty
            for a brand-new sender.

    Returns:
        dict with fraud_probability, is_flagged, threshold_used, and
        top_contributing_features.
    """
    new_row = transaction.model_dump()

    combined = pd.concat(
        [sender_history, pd.DataFrame([new_row])],
        ignore_index=True,
    )

    # --- datetime features ---
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined["hour"] = combined["timestamp"].dt.hour
    combined["day"] = combined["timestamp"].dt.day
    combined["day_of_week"] = combined["timestamp"].dt.weekday
    combined["month"] = combined["timestamp"].dt.month

    # --- impute numeric columns using the SAVED imputer ---
    num_cols = [c for c in _NUMERIC_COLUMNS if c in combined.columns]
    combined[num_cols] = _imputer.transform(combined[num_cols])

    # --- encode categoricals using the SAVED encoder (unseen -> -1) ---
    cat_cols = [c for c in _CATEGORICAL_COLUMNS if c in combined.columns]
    combined[cat_cols] = _ordinal_encoder.transform(combined[cat_cols].astype(str))

    # --- re-derive engineered features (must mirror training notebook) ---
    combined["amount_per_velocity"] = combined["amount"] / (combined["velocity_score"] + 1)
    combined["amount_log"] = np.log1p(combined["amount"])
    combined["amount_to_avg_ratio"] = (
        combined["amount"] / combined.groupby("sender_account")["amount"].transform("mean")
    )
    combined["transaction_per_day"] = (
        combined.groupby(["sender_account", "day"])["amount"].transform("count")
    )
    combined["transaction_gap"] = (
        combined.groupby("sender_account")["timestamp"].diff().dt.total_seconds().fillna(0)
    )
    combined["is_night_transaction"] = combined["hour"].between(18, 24).astype(int)
    combined["is_weekend"] = combined["day_of_week"].isin([5, 6]).astype(int)
    combined["is_self_transfer"] = (
        combined["sender_account"] == combined["receiver_account"]
    ).astype(int)
    combined["sender_degree"] = (
        combined.groupby("sender_account")["receiver_account"].transform("nunique")
    )
    combined["receiver_degree"] = (
        combined.groupby("receiver_account")["sender_account"].transform("nunique")
    )
    combined["sender_total_transaction"] = (
        combined.groupby("sender_account")["amount"].transform("count")
    )
    combined["receiver_total_transaction"] = (
        combined.groupby("receiver_account")["amount"].transform("count")
    )
    combined["sender_avg_amount"] = (
        combined.groupby("sender_account")["amount"].transform("mean")
    )
    combined["sender_std_amount"] = (
        combined.groupby("sender_account")["amount"].transform("std").fillna(0)
    )
    combined["deviation_squared"] = combined["spending_deviation_score"] ** 2

    # --- select the new transaction's row, in the exact trained column order ---
    row = combined.iloc[[-1]][_FEATURE_COLUMNS]

    # --- predict ---
    fraud_probability = float(_model.predict_proba(row)[0, 1])
    is_flagged = fraud_probability >= _FLAG_THRESHOLD

    # --- explain: which features drove this score ---
    contribution = row.iloc[0] * pd.Series(_model.feature_importances_, index=_FEATURE_COLUMNS)
    top_features = contribution.abs().sort_values(ascending=False).head(5).index.tolist()

    return {
        "fraud_probability": round(fraud_probability, 4),
        "is_flagged": bool(is_flagged),
        "threshold_used": _FLAG_THRESHOLD,
        "top_contributing_features": top_features,
    }


# ---------------------------------------------------------------------
# 4. LangChain tool wrapper.
#    Flattened primitive args (not a Pydantic object) because tool-calling
#    LLMs generate JSON-serializable arguments, not Python objects.
# ---------------------------------------------------------------------
@tool
def check_transaction_fraud(
    sender_account: str,
    receiver_account: str,
    amount: float,
    timestamp: str,
    merchant_category: str,
    transaction_type: str,
    spending_deviation_score: float,
    velocity_score: float,
    geo_anomaly_score: float,
    ip_address: str,
    device_used: str,
    location: str,
    payment_channel: str,
    device_hash: str,
    time_since_last_transaction: float
) -> dict:
    """
    Score a financial transaction for fraud risk using a trained XGBoost model.

    Returns a fraud probability (0-1), a boolean flag indicating whether
    the transaction crosses the investigation threshold, and the top
    features that drove the score. Use this once per transaction — it
    already incorporates the sender's recent transaction history
    internally, so there's no need to call it more than once for the
    same transaction.
    """
    # Local import avoids a circular import between scoring.py and
    # history_store.py at module load time.
    from history_store import get_account_history

    txn = Transaction(
        sender_account=sender_account,
        receiver_account=receiver_account,
        amount=amount,
        timestamp=timestamp,
        merchant_category=merchant_category,
        transaction_type=transaction_type,
        spending_deviation_score=spending_deviation_score,
        velocity_score=velocity_score,
        geo_anomaly_score=geo_anomaly_score,
        ip_address=ip_address,
        device_used=device_used,
        location=location,
        payment_channel=payment_channel,
        device_hash=device_hash,
        time_since_last_transaction=time_since_last_transaction
    )
    history = get_account_history(sender_account)
    return score_transaction(txn, history)