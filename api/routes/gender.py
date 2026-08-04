"""Gender Classification API Route"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from api.schemas.gender_schemas import GenderInput, GenderPredictionResponse
from src.explainability.shap_explainer import format_shap_for_api, get_shap_explanation
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/gender", response_model=GenderPredictionResponse)
async def predict_gender(payload: GenderInput, request: Request):
    """
    Predict user gender, identify traveller persona, and generate SHAP and natural language explanations.
    """
    t0 = time.perf_counter()
    try:
        from api.main import get_model_cache

        model_cache = get_model_cache()
    except Exception:
        model_cache = {}

    if "gender" not in model_cache:
        raise HTTPException(status_code=503, detail="Gender model not loaded")

    artifact = model_cache["gender"]
    model = artifact["model"]
    scaler = artifact.get("scaler")
    feature_cols = artifact["feature_cols"]

    feature_map = {
        "age": payload.age,
        "company_enc": payload.company_enc,
        "travel_frequency": payload.travel_frequency,
        "avg_flight_price": payload.avg_flight_price,
        "total_flight_spend": payload.total_flight_spend,
        "preferred_flight_type_enc": payload.preferred_flight_type_enc,
        "hotel_bookings": payload.hotel_bookings,
        "avg_hotel_spend": payload.avg_hotel_spend,
        "age_group_enc": payload.age_group_enc,
        "spending_category_enc": payload.spending_category_enc,
    }

    row = pd.DataFrame([{col: feature_map.get(col, 0) for col in feature_cols}])

    if scaler:
        X = scaler.transform(row)
        pred_class = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
    else:
        pred_class = int(model.predict(row.values)[0])
        proba = model.predict_proba(row.values)[0]

    gender_label = "male" if pred_class == 1 else "female"
    confidence = round(float(max(proba)), 4)
    inference_time = (time.perf_counter() - t0) * 1000.0

    # 1. SHAP explanation
    shap_result = get_shap_explanation(artifact, row)
    explanation = format_shap_for_api(shap_result)

    # 2. Determine Traveller Persona (Business Logic Layer)
    if payload.avg_flight_price > 1200 or payload.avg_hotel_spend > 400:
        persona = "Luxury Traveller"
        profile = "Premium Cabin seeker with high willingness-to-pay for lodging."
    elif payload.company_enc != 0 and payload.travel_frequency > 10:
        persona = "Corporate Traveller"
        profile = "High-frequency traveler working for institutional agencies."
    elif payload.travel_frequency > 12:
        persona = "Frequent Flyer"
        profile = "High mobility individual prioritizing schedules and flight timings."
    elif payload.avg_flight_price < 600 or payload.avg_hotel_spend < 250:
        persona = "Budget Traveller"
        profile = "Cost-sensitive traveler optimization-driven for economy flights."
    else:
        persona = "Occasional Traveller"
        profile = "Standard leisure traveler with balanced spending profile."

    # 3. Decision Summary (Natural Language AI Explanation Center)
    summary = (
        f"Based on behavioral metrics (Avg Flight Price: BRL {payload.avg_flight_price:,.2f}, "
        f"Hotel Bookings: {payload.hotel_bookings:.0f}), the model classified this user profile as a "
        f"'{persona}' ({gender_label}) with {confidence * 100:.1f}% confidence. The primary "
        f"contributor is the average flight spending category and travel frequency."
    )

    logger.info("Gender predicted", extra={"prediction": gender_label, "confidence": confidence})

    return GenderPredictionResponse(
        predicted_gender=gender_label,
        confidence=confidence,
        probabilities={"male": round(float(proba[1]), 4), "female": round(float(proba[0]), 4)},
        model_name=artifact.get("model_name", "Unknown"),
        explanation=explanation,
        customer_persona=persona,
        travel_profile=profile,
        decision_summary=summary,
        inference_time_ms=round(inference_time, 2),
    )
