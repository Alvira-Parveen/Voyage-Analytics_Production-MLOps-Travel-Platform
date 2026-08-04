"""
Page 3 — Gender Classifier & Customer Profiler
Provides demographic classification, traveler personas, behavior charts,
real-time feature importances from the production model, and SHAP explainability.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="User Profiler", page_icon="👤", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.pred-card {
    background: rgba(240, 147, 251, 0.08);
    border: 1px solid rgba(240, 147, 251, 0.3);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    text-align: center;
}
.pred-title { font-size: 1.1rem; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.05em; }
.pred-val { font-size: 2.8rem; font-weight: 700; color: #f093fb; margin: 8px 0; }
.persona-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    margin-top: 15px;
}
.ai-summary {
    background: rgba(240, 147, 251, 0.08);
    border-left: 4px solid #f093fb;
    padding: 16px;
    border-radius: 4px;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY  = os.getenv("API_KEY", "voyage-dev-key-2024")
MODEL_PATH = Path("models/gender_classifier_v1.0.pkl")

# Load model locally to extract real feature importances
@st.cache_resource
def load_production_gender_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

gender_model_artifact = load_production_gender_model()

st.markdown("# Traveler Profiling & Classification")
st.markdown("Classify demographics based on spending habits, identify traveler personas, and inspect features.")

st.info(
    "💡 **How to Use:**\n\n"
    "1. **Set Metrics:** Enter behavioral stats (flight counts, lodging expenditure, average stay length) on the left-hand column.\n"
    "2. **Predict Demographics:** Click the **Classify Traveler Profile** button at the bottom of the input form.\n"
    "3. **Examine Persona:** The right column will show the predicted gender classification, confidence probabilities, and traveler persona tags."
)

st.divider()

col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.markdown("### Customer Travel History")

    age = st.slider("User Age", 18, 80, 34)
    company_enc = st.slider("Corporate Agency ID (encoded)", 0, 20, 2)
    travel_freq = st.number_input("Annual Trips Taken", min_value=0, max_value=100, value=8)
    avg_flight_price = st.number_input("Average Flight Price (BRL)", min_value=0.0, value=1100.0, step=50.0)
    total_flight_spend = st.number_input("Total Flight Spend (BRL)", min_value=0.0, value=8800.0, step=100.0)

    ft_enc = {"Economy": 1, "Business": 0, "First": 2}
    selected_ft = st.selectbox("Preferred Flight Cabin Class", list(ft_enc.keys()))

    hotel_bookings = st.number_input("Annual Hotel Reservations", min_value=0, value=4)
    avg_hotel_spend = st.number_input("Average Hotel Cost (BRL)", min_value=0.0, value=420.0, step=20.0)

    age_grp_enc = {"Young (<25)": 3, "Adult (25-39)": 1, "Middle-aged (40-59)": 2, "Senior (60+)": 0}
    selected_age_grp = st.selectbox("Demographic Age Bracket", list(age_grp_enc.keys()))

    spend_cat_enc = {"Budget": 0, "Economy": 1, "Premium": 2, "Luxury": 3}
    selected_spend = st.selectbox("Account Spending Tier", list(spend_cat_enc.keys()))

    predict_btn = st.button("Analyze Traveler Profile", type="primary", use_container_width=True)

with col_result:
    if predict_btn:
        payload = {
            "age": age,
            "company_enc": company_enc,
            "travel_frequency": travel_freq,
            "avg_flight_price": avg_flight_price,
            "total_flight_spend": total_flight_spend,
            "preferred_flight_type_enc": ft_enc[selected_ft],
            "hotel_bookings": hotel_bookings,
            "avg_hotel_spend": avg_hotel_spend,
            "age_group_enc": age_grp_enc[selected_age_grp],
            "spending_category_enc": spend_cat_enc[selected_spend],
        }

        with st.spinner("Executing customer classification..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/predict/gender",
                    json=payload,
                    headers={"X-API-Key": API_KEY},
                    timeout=10,
                )
                if resp.status_code == 200:
                    res = resp.json()
                    gender = res["predicted_gender"].capitalize()
                    conf = res["confidence"]
                    proba = res.get("probabilities", {})
                    shap = res.get("explanation", [])
                    model_name = res.get("model_name", "RandomForest")
                    persona = res.get("customer_persona", "Occasional Traveller")
                    profile = res.get("travel_profile", "Standard leisure traveler with balanced spending profile.")
                    decision_summary = res.get("decision_summary", "")
                    latency = res.get("inference_time_ms", 0.0)

                    # ── Classification Card ──
                    icon = "👨" if gender == "Male" else "👩"
                    st.markdown(f"""
                    <div class="pred-card">
                        <div style="font-size:3rem;">{icon}</div>
                        <div class="pred-title">Predicted Demographics</div>
                        <div class="pred-val">{gender}</div>
                        <div style="font-size:0.85rem;color:#cbd5e1;">
                            <strong>Model:</strong> {model_name} | <strong>F1 Score Status:</strong> Balanced 0.572<br/>
                            <strong>Confidence:</strong> {conf * 100:.1f}% | <strong>Latency:</strong> {latency:.1f}ms
                        </div>
                    </div>""", unsafe_allow_html=True)

                    # ── Traveler Persona ──
                    st.markdown("### AI Customer Segment Persona")
                    st.markdown(f"""
                    <div class="persona-card">
                        <h4 style="margin:0 0 8px 0;color:#00f2fe;">{persona}</h4>
                        <p style="margin:0;font-size:0.9rem;color:#cbd5e1;line-height:1.4;">{profile}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── AI Explanation Summary Center ──
                    if decision_summary:
                        st.markdown("### AI Decision Center Summary")
                        st.markdown(f'<div class="ai-summary">{decision_summary}</div>', unsafe_allow_html=True)

                    # ── Real Feature Importance from Deployed Model (WOW Feature) ──
                    st.markdown("### Model Feature Importance (Real-Time)")
                    if gender_model_artifact is not None and "model" in gender_model_artifact:
                        model_obj = gender_model_artifact["model"]
                        feat_names = gender_model_artifact.get("feature_cols", [])
                        if hasattr(model_obj, "feature_importances_"):
                            importances = model_obj.feature_importances_
                            fi_series = pd.Series(importances, index=feat_names).sort_values(ascending=False)
                            
                            for col_name, val in fi_series.head(5).items():
                                pct = val * 100
                                bar_char = "█" * int(pct / 4) + "░" * (25 - int(pct / 4))
                                st.markdown(f"- **{col_name}**: {pct:.1f}% &nbsp; [{bar_char}]")
                        else:
                            st.caption("Model does not expose feature importances.")
                    else:
                        st.caption("Feature importance is not available (Artifact not loaded).")

                    # ── SHAP Bar Chart ──
                    if shap:
                        st.markdown("### SHAP Feature Drivers (Impact)")
                        feats = [s["feature"] for s in shap[:6]]
                        impacts = [s["impact"] for s in shap[:6]]
                        colors = ["#f093fb" if v > 0 else "#4facfe" for v in impacts]

                        fig = go.Figure(go.Bar(
                            x=impacts, y=feats, orientation="h",
                            marker_color=colors,
                            text=[f"{'+' if v>0 else ''}{v:.3f}" for v in impacts],
                            textposition="outside",
                        ))
                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            xaxis_title="SHAP Value Importance",
                            height=220,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(f"API Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.warning(f"⚠️ Prediction API is offline: {e}")
                st.markdown("""
                <div class="pred-card">
                    <div class="pred-title">Predicted Demographics (API Offline)</div>
                    <div class="pred-val">Male</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03);border-radius:16px;padding:50px;text-align:center;margin-top:20px;">
            <h4>Demographics Workspace</h4>
            <p style="color:#94a3b8;font-size:0.9rem;">Fill in traveler metrics on the left and trigger analysis to review classification models.</p>
        </div>""", unsafe_allow_html=True)
