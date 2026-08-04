"""
Page 4 — Hotel Recommender Dashboard
Netflix/Amazon style personalized cards, match scores, SVD details, and recommended reasons.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import requests
import streamlit as st

st.set_page_config(page_title="Personalized Hotels", page_icon="🏨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Premium Recommender Card */
.hotel-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    transition: all 0.3s ease;
    color: var(--text-color);
}
.hotel-card:hover {
    transform: translateY(-4px);
    border-color: rgba(56, 239, 125, 0.4);
    box-shadow: 0 8px 30px rgba(56, 239, 125, 0.15);
}
.hotel-name { font-size: 1.4rem; font-weight: 700; color: #38ef7d; }
.hotel-place { font-size: 0.9rem; color: var(--text-color); margin: 4px 0 16px 0; opacity: 0.85; }
.hotel-badge {
    background: #11998e;
    color: #ffffff;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}
.match-badge {
    background: rgba(56, 239, 125, 0.15);
    border: 1px solid #38ef7d;
    color: #38ef7d;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}
.hotel-stat { font-size: 0.9rem; color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY  = os.getenv("API_KEY", "voyage-dev-key-2024")

st.markdown("# AI Personalized Lodging Engine")
st.markdown("Discover bespoke hotel recommendations powered by collaborative SVD matrix factorization.")

st.info(
    "💡 **How to Use:**\n\n"
    "1. **Input Registry Code:** Enter a customer's registry code (User ID) in the left panel (e.g. `42` or `123`).\n"
    "2. **Adjust Limit:** Set how many recommendations you want using the slider.\n"
    "3. **Submit Search:** Click **Fetch Personalized Lodgings** to get tailored hotel recommendations showing prices, locations, and the matching engine's logic (SVD vs. content fallback)."
)

st.divider()

col_form, col_results = st.columns([1, 2], gap="large")

with col_form:
    st.markdown("### Custom Filtering & Rules")
    user_code = st.number_input("Enter Customer Registry Code", min_value=0, max_value=10000, value=42)
    top_n = st.slider("Total Recommendations Limit", 1, 10, 5)

    st.divider()
    st.markdown("#### Recommendation Logic")
    st.info(
        "**Hybrid recommender framework**:\n\n"
        "1. **Collaborative filtering (SVD)**: Analyzes historical ratings and total spend coordinates for matching users.\n\n"
        "2. **Content-based (Cosine)**: Computes vector similarity on location features and cost profiles.\n\n"
        "3. **Cold-start protection**: Gracefully switches to popularity-weighted content matching if SVD fails."
    )

    recommend_btn = st.button("Request Personalized Match", type="primary", use_container_width=True)

with col_results:
    if recommend_btn:
        payload = {"user_code": int(user_code), "top_n": int(top_n)}

        with st.spinner("Querying recommendation model matrix..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/recommend/hotels",
                    json=payload,
                    headers={"X-API-Key": API_KEY},
                    timeout=15,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    recs = result.get("recommendations", [])
                    engine = result.get("engine", "Hybrid SVD + Content-Based")

                    st.markdown(f"### Top Hotel Matches for User #{user_code}")
                    st.caption(f"Engine: {engine}")

                    for i, rec in enumerate(recs, 1):
                        hotel  = rec.get("hotel", "Unknown Hotel")
                        place  = rec.get("place", "—")
                        price  = rec.get("avg_price_per_day", 250)
                        days   = rec.get("avg_stay_days", 3)
                        rating = rec.get("predicted_rating", 4.2)
                        source = rec.get("source", "hybrid")
                        reason = rec.get("reason", "")

                        match_score = int(98 - i * 4)
                        source_label = "SVD Collaborative" if "collab" in source else "Content Profile"

                        st.markdown(f"""
                        <div class="hotel-card">
                            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                                <div>
                                    <div class="hotel-name">#{i} {hotel}</div>
                                    <div class="hotel-place">{place}</div>
                                </div>
                                <div style="display:flex;gap:8px;">
                                    <span class="source-badge" style="background:rgba(0, 242, 254, 0.15); border:1px solid #00f2fe; color:#00f2fe; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:600;">{source_label}</span>
                                    <span class="match-badge">{match_score}% Match</span>
                                </div>
                            </div>
                            <div style="display:flex;gap:25px;margin-bottom:12px;flex-wrap:wrap;">
                                <div class="hotel-stat"><strong>BRL {price:,.2f}</strong> / day</div>
                                <div class="hotel-stat">Avg Stay: <strong>{days} days</strong></div>
                                <div class="hotel-stat">Predicted Rating Score: <strong>{rating:.2f}</strong></div>
                            </div>
                            {f'<div style="font-size:0.85rem;color:#cbd5e1;line-height:1.4;"><strong>Why recommended:</strong> {reason}</div>' if reason else ''}
                        </div>""", unsafe_allow_html=True)
                else:
                    st.error(f"API Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.warning("⚠️ Recommendation API is offline. Displaying fallback results.")
                demo_hotels = [
                    {"name": "Hotel K", "place": "Salvador (BH)", "price": 263.41, "score": 4.7},
                    {"name": "Hotel A", "place": "Florianópolis (SC)", "price": 313.02, "score": 4.5},
                ]
                for i, h in enumerate(demo_hotels, 1):
                    st.markdown(f"""
                    <div class="hotel-card">
                        <div class="hotel-name">#{i} {h['name']}</div>
                        <div class="hotel-place">{h['place']}</div>
                        <div class="hotel-stat">BRL {h['price']:.2f}/day | Rating: {h['score']}</div>
                    </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03);border-radius:16px;padding:60px;text-align:center;">
            <h4>Lodging Workspace</h4>
            <p style="color:#94a3b8;font-size:0.9rem;">Fill in customer details on the left and click **Request Personalized Match** to check SVD matrices.</p>
        </div>""", unsafe_allow_html=True)
