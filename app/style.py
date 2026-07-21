import streamlit as st

def apply_theme():
    st.set_page_config(page_title="Fraud Copilot", layout="wide", page_icon="◆")
    st.markdown("""
        <style>
        .stApp { background-color: #0b0e14; color: #e6e6e6; }
        [data-testid="stSidebar"] { background-color: #0f1420; border-right: 1px solid #1f2733; }

        h1, h2, h3 { font-weight: 650; letter-spacing: -0.02em; color: #f0f2f5; }
        p, span, label { color: #b8c0cc; }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #141a24 0%, #10141c 100%);
            border: 1px solid #232c3a;
            border-radius: 12px; padding: 18px;
        }
        div[data-testid="stMetricLabel"] { color: #7d8898; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
        div[data-testid="stMetricValue"] { color: #f0f2f5; font-weight: 650; }

        .stButton button {
            border-radius: 8px; font-weight: 550; border: 1px solid #2a3542;
            transition: all 0.15s ease;
        }
        .stButton button:hover { border-color: #4a90e2; }

        .stButton button[kind="primary"] {
            background: #1f6feb; border: none;
        }

        .report-card {
            background: #121722; border: 1px solid #232c3a;
            border-radius: 14px; padding: 20px; margin-bottom: 12px;
        }
        .report-card:hover { border-color: #37455a; }

        .badge {
            display: inline-block; padding: 3px 10px; border-radius: 999px;
            font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em;
        }
        .badge-high { background: #3a1620; color: #ff6b6b; border: 1px solid #5c2530; }
        .badge-med { background: #3a2e14; color: #ffc94a; border: 1px solid #5c4a20; }
        .badge-low { background: #123a24; color: #4ade80; border: 1px solid #205c38; }
        .badge-neutral { background: #1a2230; color: #8fa3bf; border: 1px solid #2a3648; }

        .stream-active {
            background: #123a24 !important; border: 1px solid #2f9e5a !important;
            color: #4ade80 !important;
        }

        [data-testid="stDataFrame"] { border: 1px solid #232c3a; border-radius: 10px; }
        </style>
    """, unsafe_allow_html=True)


# FIX: Use float | None (or typing.Optional[float] for older Python versions)
def risk_badge(prob: float | None) -> str:
    if prob is None:
        return '<span class="badge badge-neutral">unscored</span>'
    if prob >= 0.7:
        return f'<span class="badge badge-high">HIGH · {prob:.2f}</span>'
    if prob >= 0.34:
        return f'<span class="badge badge-med">MED · {prob:.2f}</span>'
    return f'<span class="badge badge-low">LOW · {prob:.2f}</span>'