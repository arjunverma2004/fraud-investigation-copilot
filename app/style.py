import streamlit as st

def apply_theme():
    st.set_page_config(page_title="Fraud Copilot", layout="wide", page_icon="◆")
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; }
        h1, h2, h3 { font-weight: 600; letter-spacing: -0.02em; }
        div[data-testid="stMetric"] {
            background: #161b22; border: 1px solid #2a2f3a;
            border-radius: 10px; padding: 16px;
        }
        div[data-testid="stMetricLabel"] { color: #8b949e; font-size: 0.8rem; }
        .stButton button { border-radius: 8px; font-weight: 500; }
        .report-card {
            background: #161b22; border: 1px solid #2a2f3a;
            border-radius: 12px; padding: 24px; margin-bottom: 16px;
        }
        </style>
    """, unsafe_allow_html=True)