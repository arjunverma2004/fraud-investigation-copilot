import streamlit as st
import requests
from style import apply_theme

apply_theme()
API = "http://localhost:8000"

st.title("Review Queue")

if st.button("🔄 Refresh"):
    st.rerun()

pending = requests.get(f"{API}/investigations/pending").json()

if not pending:
    st.success("No transactions awaiting review.")
else:
    for txn in pending:
        detail = requests.get(f"{API}/investigations/{txn['transaction_id']}").json()
        values = detail["values"]

        with st.container():
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.markdown(
                f"**Transaction** `{txn['transaction_id']}` — "
                f"${txn['amount']:.2f} — fraud score `{txn.get('fraud_probability', '—')}`"
            )

            report = values.get("investigation_report") or ""
            if report:
                st.markdown(report)
                st.caption(f"Model recommendation: **{values.get('recommendation', 'n/a')}**")

                notes_key = f"notes_{txn['transaction_id']}"
                notes = st.text_input("Reviewer notes", key=notes_key)

                b1, b2, b3 = st.columns(3)
                if b1.button("✅ Approve", key=f"a_{txn['transaction_id']}"):
                    requests.post(
                        f"{API}/investigations/{txn['transaction_id']}/decision",
                        json={"decision": "approve", "notes": notes},
                    )
                    st.rerun()
                if b2.button("🚫 Reject", key=f"r_{txn['transaction_id']}"):
                    requests.post(
                        f"{API}/investigations/{txn['transaction_id']}/decision",
                        json={"decision": "reject", "notes": notes},
                    )
                    st.rerun()
                if b3.button("🔄 Need More Info", key=f"m_{txn['transaction_id']}"):
                    requests.post(
                        f"{API}/investigations/{txn['transaction_id']}/decision",
                        json={"decision": "need_more_info", "notes": notes},
                    )
                    st.rerun()
            else:
                st.caption("⏳ Investigation still in progress…")
            st.markdown('</div>', unsafe_allow_html=True)