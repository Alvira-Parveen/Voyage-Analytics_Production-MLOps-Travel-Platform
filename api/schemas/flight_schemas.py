"""Pydantic schemas for Flight Price API"""

from typing import Any, Optional
from pydantic import BaseModel, Field


class FlightInput(BaseModel):
    flightType_enc: int = Field(..., ge=0, description="Encoded flight type (0=business, 1=economy, 2=first)")
    agency_enc: int = Field(..., ge=0, description="Encoded agency ID")
    from_enc: int = Field(..., ge=0, description="Encoded departure city")
    to_enc: int = Field(..., ge=0, description="Encoded arrival city")
    distance: float = Field(..., gt=0, description="Flight distance in km")
    time: float = Field(..., gt=0, description="Flight duration in hours")
    month: int = Field(..., ge=1, le=12)
    weekday: int = Field(..., ge=0, le=6, description="0=Monday, 6=Sunday")
    year: int = Field(..., ge=2018, le=2030)
    season_enc: int = Field(0, ge=0, le=3, description="0=autumn,1=spring,2=summer,3=winter")
    is_holiday: bool = Field(False)
    agency_popularity: int = Field(100, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "flightType_enc": 2,
                "agency_enc": 1,
                "from_enc": 5,
                "to_enc": 10,
                "distance": 676.53,
                "time": 1.76,
                "month": 9,
                "weekday": 3,
                "year": 2024,
                "season_enc": 2,
                "is_holiday": False,
                "agency_popularity": 250,
            }
        }


class SHAPFeature(BaseModel):
    feature: str
    input_value: float
    impact: float
    direction: str


class FlightPredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "BRL"
    model_name: str
    explanation: list[SHAPFeature] = []
    decision_summary: Optional[str] = None
    expected_range: Optional[list[float]] = None
    inference_time_ms: Optional[float] = None
