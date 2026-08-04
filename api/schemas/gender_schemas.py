"""Pydantic schemas for Gender Classification API"""

from typing import Optional
from pydantic import BaseModel, Field


class GenderInput(BaseModel):
    age: float = Field(..., ge=0, le=120)
    company_enc: int = Field(0, ge=0)
    travel_frequency: float = Field(0.0, ge=0)
    avg_flight_price: float = Field(500.0, ge=0)
    total_flight_spend: float = Field(0.0, ge=0)
    preferred_flight_type_enc: int = Field(1, ge=0)
    hotel_bookings: float = Field(0.0, ge=0)
    avg_hotel_spend: float = Field(300.0, ge=0)
    age_group_enc: int = Field(1, ge=0)
    spending_category_enc: int = Field(1, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 34,
                "company_enc": 2,
                "travel_frequency": 8,
                "avg_flight_price": 1100.50,
                "total_flight_spend": 8800.0,
                "preferred_flight_type_enc": 2,
                "hotel_bookings": 4,
                "avg_hotel_spend": 420.0,
                "age_group_enc": 1,
                "spending_category_enc": 2,
            }
        }


class SHAPFeature(BaseModel):
    feature: str
    input_value: float
    impact: float
    direction: str


class GenderPredictionResponse(BaseModel):
    predicted_gender: str
    confidence: float
    probabilities: dict[str, float]
    model_name: str
    explanation: list[SHAPFeature] = []
    customer_persona: Optional[str] = None
    travel_profile: Optional[str] = None
    decision_summary: Optional[str] = None
    inference_time_ms: Optional[float] = None
