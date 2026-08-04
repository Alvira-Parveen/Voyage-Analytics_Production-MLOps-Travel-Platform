"""Flight Price Prediction API Route"""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from api.schemas.flight_schemas import FlightInput, FlightPredictionResponse, SHAPFeature
from src.explainability.shap_explainer import format_shap_for_api, get_shap_explanation
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/flight-price", response_model=FlightPredictionResponse)
async def predict_flight_price(payload: FlightInput, request: Request):
    """
    Predict flight price, compute inference latency, and generate human-readable explanations.
    """
    t0 = time.perf_counter()
    try:
        from api.main import get_model_cache
        model_cache = get_model_cache()
    except Exception:
        model_cache = {}

    if "flight" not in model_cache:
        raise HTTPException(status_code=503, detail="Flight model not loaded")

    artifact = model_cache["flight"]
    model = artifact["model"]
    scaler = artifact.get("scaler")
    feature_cols = artifact["feature_cols"]

    # Build feature map matching trained features exactly (NO TARGET LEAKS)
    feature_map = {
        "flightType_enc":     payload.flightType_enc,
        "agency_enc":         payload.agency_enc,
        "from_enc":           payload.from_enc,
        "to_enc":             payload.to_enc,
        "distance":           payload.distance,
        "time":               payload.time,
        "speed_proxy":        round(payload.distance / max(payload.time, 0.01), 4),
        "month":              payload.month,
        "weekday":            payload.weekday,
        "year":               payload.year,
        "season_enc":         payload.season_enc,
        "is_weekend":         int(payload.weekday >= 5),
        "is_holiday":         int(payload.is_holiday),
        "agency_popularity":  payload.agency_popularity,
    }

    row = pd.DataFrame([{col: feature_map.get(col, 0) for col in feature_cols}])

    if scaler:
        X = scaler.transform(row)
        pred = float(model.predict(X)[0])
    else:
        pred = float(model.predict(row.values)[0])

    pred = round(max(pred, 0), 2)
    inference_time = (time.perf_counter() - t0) * 1000.0

    # 1. SHAP explanation
    shap_result = get_shap_explanation(artifact, row)
    explanation = format_shap_for_api(shap_result)

    # 2. Decision Summary (WOW Feature: Premium Natural Language AI Decision Center)
    flight_type = "First Class" if payload.flightType_enc == 2 else ("Business Class" if payload.flightType_enc == 0 else "Economy Class")
    is_holiday_str = "during a holiday season" if payload.is_holiday else "on a standard travel day"
    
    summary = (
        f"The predicted fare is BRL {pred:,.2f} for a {flight_type} flight covering {payload.distance:,} km ({payload.time:.1f} hours). "
        f"The trip occurs {is_holiday_str}. The model predicts this price based on airline agency supply dynamics and seasonal demand."
    )

    # 3. Expected Price Range & Booking Recommendation
    expected_range = [round(pred * 0.92, 2), round(pred * 1.08, 2)]

    logger.info("Flight price predicted", extra={
        "prediction": pred, "model": artifact.get("model_name", "unknown")
    })

    return FlightPredictionResponse(
        predicted_price=pred,
        currency="BRL",
        model_name=artifact.get("model_name", "Unknown"),
        explanation=explanation,
        decision_summary=summary,
        expected_range=expected_range,
        inference_time_ms=round(inference_time, 2)
    )
