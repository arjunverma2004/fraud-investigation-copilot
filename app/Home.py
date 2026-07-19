import streamlit as st
import requests
import pandas as pd
from style import apply_theme

apply_theme()
API = "http://localhost:8000"

st.title("Fraud Investigation Copilot")
st.caption("Real-time transaction monitoring dashboard")

col1, col2, col3 = st.columns([2, 1, 1])
with col2:
    delay = st.slider("Replay speed (sec/txn)", 0.1, 2.0, 0.5)
with col3:
    limit = st.number_input("Transactions to replay", min_value=1, value=50)

c1, c2 = st.columns(2)
if c1.button("▶ Start Stream", use_container_width=True):
    requests.post(f"{API}/stream/start", json={"delay_seconds": delay, "limit": int(limit)})
if c2.button("■ Stop Stream", use_container_width=True):
    requests.post(f"{API}/stream/stop")

status = requests.get(f"{API}/stream/status").json()
st.caption(f"Stream status: {'🟢 running' if status['active'] else '⚪ stopped'}")

txns = requests.get(f"{API}/dashboard/transactions", params={"limit": 200}).json()
df = pd.DataFrame(txns)

if not df.empty:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Processed", len(df))
    m2.metric("Flagged", int(df["is_flagged"].fillna(0).sum()))
    m3.metric("Avg Fraud Score", f"{df['fraud_probability'].dropna().mean():.3f}" if df["fraud_probability"].notna().any() else "—")
    pending = df[(df["is_flagged"] == 1) & (df["investigation_status"].isin(["none", "pending"]))]
    m4.metric("Awaiting Review", len(pending))

    st.subheader("Live Transaction Feed")
    st.dataframe(
        df[["transaction_id", "sender_account", "receiver_account", "amount",
            "fraud_probability", "is_flagged", "investigation_status"]],
        use_container_width=True, height=400,
    )
else:
    st.info("No transactions yet — start the stream above.")