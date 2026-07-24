import streamlit as st
import requests
import pandas as pd
from style import apply_theme, risk_badge
import time


apply_theme()
API = "http://localhost:8000"

if "stream_running" not in st.session_state:
    status = requests.get(f"{API}/stream/status").json()
    st.session_state.stream_running = status["active"]

st.title("Fraud Investigation Copilot")
st.caption("Real-time transaction monitoring")

ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.3, 1, 1, 1.3])

with ctrl2:
    delay = st.slider("Speed (sec/txn)", 0.1, 2.0, 0.5, label_visibility="visible")
with ctrl3:
    limit = st.number_input("Batch size", min_value=1, value=50)

with ctrl1:
    if st.session_state.stream_running:
        st.markdown(
            '<div class="stream-active" style="padding:10px 14px;border-radius:8px;'
            'text-align:center;font-weight:600;">🟢 Stream Running</div>',
            unsafe_allow_html=True,
        )
    else:
        if st.button("▶ Start Stream", type="primary", use_container_width=True):
            requests.post(f"{API}/stream/start", json={"delay_seconds": delay, "limit": int(limit)})
            st.session_state.stream_running = True
            st.rerun()

with ctrl4:
    if st.session_state.stream_running:
        if st.button("■ Stop Stream", use_container_width=True):
            requests.post(f"{API}/stream/stop")
            st.session_state.stream_running = False
            st.rerun()

# Reconcile with actual backend state in case it stopped on its own
live_status = requests.get(f"{API}/stream/status").json()
if live_status["active"] != st.session_state.stream_running:
    st.session_state.stream_running = live_status["active"]
    st.rerun()

st.divider()

txns = requests.get(f"{API}/dashboard/transactions", params={"limit": 200}).json()
df = pd.DataFrame(txns)

if not df.empty:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Processed", len(df))
    m2.metric("Flagged", int(df["is_flagged"].fillna(0).sum()))
    avg_score = df["fraud_probability"].dropna().mean()
    m3.metric("Avg Fraud Score", f"{avg_score:.3f}" if pd.notna(avg_score) else "—")
    pending = df[(df["is_flagged"] == 1) & (df["investigation_status"].isin(["none", "pending"]))]
    m4.metric("Awaiting Review", len(pending))

    st.subheader("Live Transaction Feed")
    display_df = df[["transaction_id", "sender_account", "receiver_account", "amount",
                      "fraud_probability", "is_flagged", "investigation_status"]].copy()
    st.dataframe(display_df, use_container_width=True, height=420, hide_index=True)
else:
    st.info("No transactions yet — start the stream above.")

# Auto-refresh while a stream is running so the feed updates live,
# instead of only refreshing on the next manual interaction.
if st.session_state.stream_running:
    time.sleep(2)
    st.rerun()