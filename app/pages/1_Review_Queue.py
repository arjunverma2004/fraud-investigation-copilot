import streamlit as st
import requests
from style import apply_theme, risk_badge

apply_theme()
API = "http://localhost:8000"

st.title("Review Queue")

if st.button("🔄 Refresh"):
    st.rerun()

pending = requests.get(f"{API}/investigations/pending").json()

if not pending:
    st.success("No transactions awaiting review.")
    st.stop()

if "selected_txn" not in st.session_state:
    st.session_state.selected_txn = pending[0]["transaction_id"]

list_col, detail_col = st.columns([1, 2])

with list_col:
    st.subheader(f"Pending ({len(pending)})")
    for txn in pending:
        is_selected = txn["transaction_id"] == st.session_state.selected_txn
        label = f"${txn['amount']:.2f} · {txn['sender_account']}"
        btn_type = "primary" if is_selected else "secondary"
        if st.button(label, key=f"select_{txn['transaction_id']}", use_container_width=True, type=btn_type):
            st.caption(f"DEBUG: selected={st.session_state.selected_txn!r}")
            st.session_state.selected_txn = txn["transaction_id"]
            st.rerun()
        st.markdown(risk_badge(txn.get("fraud_probability")), unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

with detail_col:
    txn = next((t for t in pending if t["transaction_id"] == st.session_state.selected_txn), None)
    if txn is None:
        st.info("Select a transaction from the list.")
        st.stop()

    detail = requests.get(f"{API}/investigations/{txn['transaction_id']}").json()
    values = detail["values"]

    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.markdown(f"### Transaction `{txn['transaction_id'][:8]}…`")
    c1, c2, c3 = st.columns(3)
    c1.metric("Amount", f"${txn['amount']:.2f}")
    c2.metric("Sender", txn["sender_account"])
    c3.metric("Receiver", txn["receiver_account"])
    st.markdown(risk_badge(txn.get("fraud_probability")), unsafe_allow_html=True)

    report = values.get("investigation_report") or ""
    if report:
        st.divider()
        st.markdown(report)
        st.caption(f"Model recommendation: **{values.get('recommendation', 'n/a')}**")

        notes = st.text_input("Reviewer notes", key=f"notes_{txn['transaction_id']}")
        b1, b2, b3 = st.columns(3)
        if b1.button("✅ Clear (Not Fraud)", key=f"a_{txn['transaction_id']}", type="primary", use_container_width=True):
            requests.post(f"{API}/investigations/{txn['transaction_id']}/decision",
                          json={"decision": "approve", "notes": notes})
            st.rerun()
        if b2.button("🚫 Confirm Fraud", key=f"r_{txn['transaction_id']}", use_container_width=True):
            requests.post(f"{API}/investigations/{txn['transaction_id']}/decision",
                          json={"decision": "reject", "notes": notes})
            st.rerun()
        if b3.button("🔄 Need More Info", key=f"m_{txn['transaction_id']}", use_container_width=True):
            requests.post(f"{API}/investigations/{txn['transaction_id']}/decision",
                          json={"decision": "need_more_info", "notes": notes})
            st.rerun()
    else:
        st.info("⏳ Investigation still in progress…")
    st.markdown('</div>', unsafe_allow_html=True)