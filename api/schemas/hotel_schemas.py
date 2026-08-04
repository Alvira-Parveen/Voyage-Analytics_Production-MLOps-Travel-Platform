"""Pydantic schemas for Hotel Recommendation API"""

from typing import Any

from pydantic import BaseModel, Field


class HotelRecommendInput(BaseModel):
    user_code: int = Field(..., ge=0, description="User code from users dataset")
    top_n: int = Field(5, ge=1, le=10, description="Number of recommendations")

    class Config:
        json_schema_extra = {
            "example": {
                "user_code": 42,
                "top_n": 5,
            }
        }


class HotelRecommendResponse(BaseModel):
    user_code: int
    recommendations: list[dict[str, Any]]
    total: int
    engine: str
