"""
Page 9 — MLOps Experiment Tracker
Historical logs, model selection comparison tables, leakage audits, and stage transitions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Experiment Tracker", page_icon="🧬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.markdown("# MLOps Experiment Tracker & Benchmarking")
st.markdown("Chronological run history for model exploration iterations, debugging logs, and target leakage resolution.")
st.divider()

# ── 1. Model Comparison Table (WOW Feature) ──
st.markdown("### Model Evaluation Leaderboard")
st.markdown("All candidate models evaluated under identical 5-fold cross-validation folds on the standardized training set split.")

leaderboard_data = pd.DataFrame([
    {
        "Model": "Linear Regression",
        "CV R²": "0.5623",
        "RMSE (BRL)": "240.80",
        "Train Time": "0.02 s",
        "Inference Latency": "0.2 ms",
        "Status": "Archived"
    },
    {
        "Model": "Ridge Regression",
        "CV R²": "0.5619",
        "RMSE (BRL)": "240.85",
        "Train Time": "0.03 s",
        "Inference Latency": "0.3 ms",
        "Status": "Archived"
    },
    {
        "Model": "Random Forest",
        "CV R²": "0.9971",
        "RMSE (BRL)": "22.82",
        "Train Time": "2.50 s",
        "Inference Latency": "20.0 ms",
        "Status": "Candidate (Heavy)"
    },
    {
        "Model": "LightGBM",
        "CV R²": "0.9961",
        "RMSE (BRL)": "23.27",
        "Train Time": "0.60 s",
        "Inference Latency": "5.0 ms",
        "Status": "Candidate"
    },
    {
        "Model": "XGBoost (Best)",
        "CV R²": "0.9973",
        "RMSE (BRL)": "19.31",
        "Train Time": "0.78 s",
        "Inference Latency": "8.5 ms",
        "Status": "Production Deploy"
    }
])

st.dataframe(leaderboard_data, use_container_width=True)

st.divider()

# ── 2. Run Details & Rejections ──
st.markdown("### Experiment Run History Log")

# List of experiments
experiments = [
    {
        "ID": "EXP-001",
        "Phase": "Baseline Model Evaluation",
        "Algorithm": "Linear Regression",
        "Parameters": "Default parameters",
        "CV R² Score": "0.5623",
        "Test R² Score": "0.5600",
        "Leakage": "No",
        "Status": "Archived",
        "Rejection Reason": "Underfitting baseline score",
        "Timestamp": "Aug 2, 2026 10:15 AM"
    },
    {
        "ID": "EXP-002",
        "Phase": "Feature Engineering Integration",
        "Algorithm": "Random Forest Regressor",
        "Parameters": "n_estimators=200, max_depth=12",
        "CV R² Score": "0.9999 (Leaked)",
        "Test R² Score": "0.9999 (Leaked)",
        "Leakage": "Leakage Detected",
        "Status": "Rejected",
        "Rejection Reason": "Target leakage detected on price_per_km column",
        "Timestamp": "Aug 2, 2026 11:22 AM"
    },
    {
        "ID": "EXP-003",
        "Phase": "Leakage Audits & Features Cleaned",
        "Algorithm": "XGBoost Regressor",
        "Parameters": "n_estimators=300, max_depth=7",
        "CV R² Score": "0.9974",
        "Test R² Score": "0.9972",
        "Leakage": "Cleaned",
        "Status": "Archived",
        "Rejection Reason": "Superseded by hyperparameter tuned model",
        "Timestamp": "Aug 3, 2026 02:10 PM"
    },
    {
        "ID": "EXP-004",
        "Phase": "Hyperparameter Optimization Tuning",
        "Algorithm": "XGBoost Regressor (Tuned)",
        "Parameters": "learning_rate=0.08, max_depth=6, n_estimators=100, subsample=0.8",
        "CV R² Score": "0.9973",
        "Test R² Score": "0.9972",
        "Leakage": "Cleaned",
        "Status": "Deployed (Production)",
        "Rejection Reason": "N/A - Winner Model",
        "Timestamp": "Aug 4, 2026 02:50 PM"
    }
]

for exp in experiments:
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"#### Run: {exp['ID']} — {exp['Phase']}")
            st.markdown(f"**Algorithm:** `{exp['Algorithm']}`")
            st.markdown(f"**Tuned Parameters:** `{exp['Parameters']}`")
            st.markdown(f"**Cross-Validation R²:** `{exp['CV R² Score']}` &nbsp;|&nbsp; **Test R²:** `{exp['Test R² Score']}`")
            st.markdown(f"**Leakage Audit:** `{exp['Leakage']}`")
            st.markdown(f"**Timestamp:** {exp['Timestamp']}")
        with col2:
            st.markdown("##### Deployment Stage")
            if exp["Status"] == "Deployed (Production)":
                st.success("Deployed")
            elif exp["Status"] == "Rejected":
                st.error("Rejected")
            else:
                st.info(exp["Status"])
            
            if exp["Rejection Reason"] != "N/A - Winner Model":
                st.caption(f"**Reason/Note:** {exp['Rejection Reason']}")
