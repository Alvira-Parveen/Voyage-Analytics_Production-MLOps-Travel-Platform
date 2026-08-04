"""
Page 10 — MLOps Pipeline Architecture
Visualizes the end-to-end MLOps pipeline using Mermaid diagrams.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Architecture Map", page_icon="🏗️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.desc-box {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 16px;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# Voyage Analytics MLOps System Architecture")
st.markdown("Automated pipeline mapping showing flow from raw travel CSV files to active serving endpoints.")
st.divider()

# Render Mermaid diagram
st.markdown("### System Flow Diagram")
st.markdown("""
```mermaid
graph TD
    subgraph Data Pipeline Layer
        CSV["Raw Data CSVs"] --> VAL["Data Validation (validate.py)"]
        VAL --> CLEAN["Cleaning & Dedup (preprocess.py)"]
        CLEAN --> FE["Feature Store & Feature Engineering"]
    end
    subgraph MLOps Model Center
        FE --> TRAIN["K-Fold Model Selection (XGBoost/RF)"]
        TRAIN --> MLFLOW["Experiment Run Tracking (MLflow)"]
        MLFLOW --> REG["Model Registry version transitions"]
    end
    subgraph Deployed System Layer
        REG --> API["FastAPI Inference serving Gateway"]
        API --> MON["Monitoring (PSI & KS Drift metrics)"]
        MON --> DASH["Executive Dashboard Presentation Layer"]
    end
```
""")

st.divider()

# Description boxes
st.markdown("### Pipeline Component Directory")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="desc-box">
        <h4>1. Data Engineering Stage</h4>
        <ul>
            <li><strong>Raw CSV:</strong> Raw relational data feeds containing demographic details, flight logs, and hotel stays.</li>
            <li><strong>Validation:</strong> Executes schema tests, range boundaries, null checks, and category verification.</li>
            <li><strong>Cleaning:</strong> Normalizes datatypes and handles outlier values.</li>
            <li><strong>Feature Engineering:</strong> Generates interaction features (time aggregates, popularity mapping) with target-leak protection.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="desc-box">
        <h4>2. MLOps Serving Stage</h4>
        <ul>
            <li><strong>Training:</strong> Executes K-Fold cross-validation to select the optimal model pipeline.</li>
            <li><strong>MLflow Tracking:</strong> Saves run parameters, metrics, and artifact dictionaries.</li>
            <li><strong>FastAPI App:</strong> Provides sub-10ms predictions, SHAP explainers, and API key authorization.</li>
            <li><strong>Monitoring:</strong> Scrapes request footprints via Prometheus and updates PSI drift indicators.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
