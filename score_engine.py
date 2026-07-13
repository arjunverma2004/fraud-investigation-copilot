import pandas as pd
import numpy as np
import joblib
import json
from langchain_core.tools import tool





@tool
def score_transaction(new_transaction: dict, sender_history: pd.DataFrame) -> dict:
    """
    Score a single incoming transaction.

    Args:
        new_transaction: dict of raw transaction fields, same schema as the
            original CSV columns (sender_account, receiver_account, amount,
            timestamp, merchant_category, transaction_type,
            spending_deviation_score, velocity_score, geo_anomaly_score,
            ip_address, location, payment_channel, device_hash, ...).
        sender_history: DataFrame of that sender's past transactions
            (same raw schema), pulled from the history store. Can be empty
            for a brand-new sender — engineered ratios will just fall back
            to the transaction's own values.

    Returns:
        dict with fraud_probability, is_flagged, and top contributing
        features (for the agent/human to see *why*).
    """
    # 1. Combine history + new transaction so groupby-based features are
    #    computed the same way they were at training time.

    metadata = json.load(open("artifacts/metadata.json"))
    combined = pd.concat(
        [sender_history, pd.DataFrame([new_transaction])],
        ignore_index=True
    )
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined["hour"] = combined["timestamp"].dt.hour
    combined["day"] = combined["timestamp"].dt.day
    combined["day_of_week"] = combined["timestamp"].dt.weekday
    combined["month"] = combined["timestamp"].dt.month

    # 2. Impute numeric columns using the SAVED imputer (fit at training
    #    time) — never re-fit on a single row / small history slice.
    num_cols = [c for c in metadata["numeric_columns_for_imputation"] if c in combined.columns]
    combined[num_cols] = imputer.transform(combined[num_cols])

    # 3. Encode categoricals with the SAVED encoder — unseen categories
    #    (new device_hash, new location, etc.) safely map to -1 instead of
    #    crashing the pipeline.
    cat_cols = [c for c in metadata["categorical_columns"] if c in combined.columns]
    combined[cat_cols] = ordinal_encoder.transform(combined[cat_cols].astype(str))

    # 4. Re-derive the same engineered features as training (kept in sync
    #    with cell 8 above — if you change feature engineering there,
    #    mirror the change here).
    combined["amount_per_velocity"] = combined["amount"] / (combined["velocity_score"] + 1)
    combined["amount_log"] = np.log1p(combined["amount"])
    combined["amount_to_avg_ratio"] = combined["amount"] / combined.groupby("sender_account")["amount"].transform("mean")
    combined["transaction_per_day"] = combined.groupby(["sender_account", "day"])["amount"].transform("count")
    combined["transaction_gap"] = combined.groupby("sender_account")["timestamp"].diff().dt.total_seconds().fillna(0)
    combined["is_night_transaction"] = combined["hour"].between(18, 24).astype(int)
    combined["is_weekend"] = combined["day_of_week"].isin([5, 6]).astype(int)
    combined["is_self_transfer"] = (combined["sender_account"] == combined["receiver_account"]).astype(int)
    combined["sender_degree"] = combined.groupby("sender_account")["receiver_account"].transform("nunique")
    combined["receiver_degree"] = combined.groupby("receiver_account")["sender_account"].transform("nunique")
    combined["sender_total_transaction"] = combined.groupby("sender_account")["amount"].transform("count")
    combined["receiver_total_transaction"] = combined.groupby("receiver_account")["amount"].transform("count")
    combined["sender_avg_amount"] = combined.groupby("sender_account")["amount"].transform("mean")
    combined["sender_std_amount"] = combined.groupby("sender_account")["amount"].transform("std").fillna(0)
    combined["deviation_squared"] = combined["spending_deviation_score"] ** 2

    # 5. Take only the new transaction's row (the last one) and select
    #    columns in the EXACT saved training order.
    row = combined.iloc[[-1]][metadata["feature_columns"]]

    # 6. Predict.
    fraud_probability = float(xgb_model.predict_proba(row)[0, 1])
    is_flagged = fraud_probability >= metadata["flag_threshold"]

    # 7. Surface the features that most drove this specific score, so the
    #    agent's report and the human reviewer see something more useful
    #    than a bare number.
    contribution = (row.iloc[0] * pd.Series(xgb_model.feature_importances_, index=FEATURE_COLUMNS))
    top_features = contribution.abs().sort_values(ascending=False).head(5).index.tolist()

    return {
        "fraud_probability": round(fraud_probability, 4),
        "is_flagged": bool(is_flagged),
        "threshold_used": metadata["flag_threshold"],
        "top_contributing_features": top_features,
    }
