"""Hotel Recommendation API Route"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException, Request

from api.schemas.hotel_schemas import HotelRecommendInput, HotelRecommendResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/hotels", response_model=HotelRecommendResponse)
async def recommend_hotels(payload: HotelRecommendInput, request: Request):
    """
    Get top-N hotel recommendations for a user.
    Uses hybrid SVD + content-based filtering.
    Falls back to content-based on cold-start.

    Requires `X-API-Key` header.
    """
    try:
        from api.main import get_model_cache

        model_cache = get_model_cache()
    except Exception:
        model_cache = {}

    if "recommender" not in model_cache:
        raise HTTPException(status_code=503, detail="Recommender model not loaded")

    recommender = model_cache["recommender"]
    top_n = min(payload.top_n, 10)

    recommendations = recommender.recommend(
        user_code=payload.user_code,
        top_n=top_n,
        reason=True,
    )

    logger.info("Hotels recommended", extra={"user_code": payload.user_code, "count": len(recommendations)})

    return HotelRecommendResponse(
        user_code=payload.user_code,
        recommendations=recommendations,
        total=len(recommendations),
        engine="Hybrid (SVD + Content-Based)",
    )
