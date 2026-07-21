import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from style import apply_theme

apply_theme()
API = "http://localhost:8000"

st.title("Analytics")

txns = requests.get(f"{API}/dashboard/transactions", params={"limit": 1000}).json()
df = pd.DataFrame(txns)

if df.empty:
    st.info("No transaction data yet.")
    st.stop()

tab1, tab2 = st.tabs(["Overview", "All Transactions"])

with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Fraud Score Distribution")
        scored = df[df["fraud_probability"].notna()]
        if not scored.empty:
            fig = px.histogram(
                scored, x="fraud_probability", nbins=30,
                color_discrete_sequence=["#4a90e2"],
            )
            fig.update_layout(
                plot_bgcolor="#0b0e14", paper_bgcolor="#0b0e14",
                font_color="#b8c0cc", margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Investigation Outcomes")
        flagged = df[df["is_flagged"] == 1]
        if not flagged.empty:
            status_counts = flagged["investigation_status"].value_counts().reset_index()
            status_counts.columns = ["status", "count"]
            fig = px.pie(
                status_counts, values="count", names="status", hole=0.5,
                color_discrete_sequence=["#4ade80", "#ff6b6b", "#ffc94a", "#8fa3bf"],
            )
            fig.update_layout(
                plot_bgcolor="#0b0e14", paper_bgcolor="#0b0e14",
                font_color="#b8c0cc", margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No flagged transactions yet.")

    st.subheader("Transaction Amount vs Fraud Score")
    scored = df[df["fraud_probability"].notna()]
    if not scored.empty:
        fig = px.scatter(
            scored, x="amount", y="fraud_probability",
            color="is_flagged", color_discrete_map={0: "#4ade80", 1: "#ff6b6b"},
            hover_data=["sender_account", "receiver_account"],
        )
        fig.update_layout(
            plot_bgcolor="#0b0e14", paper_bgcolor="#0b0e14",
            font_color="#b8c0cc", margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("All Transactions")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        show_only_flagged = st.checkbox("Flagged only")
    with filter_col2:
        status_filter = st.multiselect(
            "Investigation status",
            options=df["investigation_status"].dropna().unique().tolist(),
        )

    filtered = df.copy()
    if show_only_flagged:
        filtered = filtered[filtered["is_flagged"] == 1]
    if status_filter:
        filtered = filtered[filtered["investigation_status"].isin(status_filter)]

    st.dataframe(filtered, use_container_width=True, height=500, hide_index=True)