"""
Page 11 — Feature Store Registry
Mock Catalog for Voyage Analytics feature group variables.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Feature Store", page_icon="🗄️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.markdown("# Enterprise Feature Store Registry")
st.markdown("Centralized feature catalog containing engineered inputs, distributions, version tags, and upstream transformations.")
st.divider()

# Tab layout for different feature groups
tab1, tab2 = st.tabs(["Flight Feature Group", "User Feature Group"])

with tab1:
    st.markdown("### Feature Group: `flight_features_v2`")
    st.markdown("**Entity Key:** `flight_id` | **Upstream Source:** `flights_clean.csv`")
    
    flight_features = [
        {
            "Feature Name": "distance",
            "Type": "Float",
            "Encoding": "MinMax Normalized",
            "Version": "v1.0",
            "Drift PSI": "0.045",
            "Status": "Active"
        },
        {
            "Feature Name": "time",
            "Type": "Float",
            "Encoding": "Raw Duration (Hours)",
            "Version": "v1.0",
            "Drift PSI": "0.012",
            "Status": "Active"
        },
        {
            "Feature Name": "month",
            "Type": "Int",
            "Encoding": "Cyclic Encode (Sin/Cos)",
            "Version": "v2.0",
            "Drift PSI": "0.000",
            "Status": "Active"
        },
        {
            "Feature Name": "season_enc",
            "Type": "Int",
            "Encoding": "Ordinal Encodings",
            "Version": "v1.0",
            "Drift PSI": "0.018",
            "Status": "Active"
        },
        {
            "Feature Name": "route_enc",
            "Type": "Int",
            "Encoding": "One-Hot / Target Hash",
            "Version": "v2.1",
            "Drift PSI": "0.024",
            "Status": "Active"
        }
    ]
    st.dataframe(pd.DataFrame(flight_features), use_container_width=True)

with tab2:
    st.markdown("### Feature Group: `user_features_v1`")
    st.markdown("**Entity Key:** `user_code` | **Upstream Source:** `users_clean.csv`")
    
    user_features = [
        {
            "Feature Name": "avg_flight_price",
            "Type": "Float",
            "Encoding": "Rolling average spend (BRL)",
            "Version": "v1.0",
            "Drift PSI": "0.034",
            "Status": "Active"
        },
        {
            "Feature Name": "spending_category_enc",
            "Type": "Int",
            "Encoding": "Ordinal Encoded Spend Tier",
            "Version": "v3.0",
            "Drift PSI": "0.021",
            "Status": "Active"
        },
        {
            "Feature Name": "preferred_flight_type_enc",
            "Type": "Int",
            "Encoding": "One-Hot Encoded Preference",
            "Version": "v2.0",
            "Drift PSI": "0.009",
            "Status": "Active"
        },
        {
            "Feature Name": "total_flight_spend",
            "Type": "Float",
            "Encoding": "Cumulative sum calculation",
            "Version": "v1.1",
            "Drift PSI": "0.051",
            "Status": "Active"
        }
    ]
    st.dataframe(pd.DataFrame(user_features), use_container_width=True)

st.divider()
st.info("**Why a Feature Store?** Centralizing feature mappings ensures identical data transformations during training and real-time API inference (preventing train-serving skew).")
