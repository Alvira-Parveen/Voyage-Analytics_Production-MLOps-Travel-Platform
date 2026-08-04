"""
Voyage Analytics 2.0 — FastAPI Application Entry Point
Production-grade API with security, logging, and monitoring.
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.routes import flight_price, gender, recommend
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger("api", log_file="logs/api.log")

# Rate Limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

# Model Cache (loaded at startup)
MODEL_CACHE: dict = {}

MODELS_PATH = Path(os.getenv("MODELS_PATH", "models"))

def _load_model(filename: str):
    path = MODELS_PATH / filename
    candidates = sorted(MODELS_PATH.glob(filename.replace("vX", "v*")),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return joblib.load(candidates[0]) if candidates else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at startup."""
    logger.info("Starting Voyage Analytics API — loading models...")

    try:
        candidates = sorted(MODELS_PATH.glob("flight_price_v*.pkl"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            MODEL_CACHE["flight"] = joblib.load(candidates[0])
            logger.info(f"Flight model loaded: {candidates[0].name}")
    except Exception as e:
        logger.error(f"Failed to load flight model: {e}")

    try:
        candidates = sorted(MODELS_PATH.glob("gender_classifier_v*.pkl"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            MODEL_CACHE["gender"] = joblib.load(candidates[0])
            logger.info(f"Gender model loaded: {candidates[0].name}")
    except Exception as e:
        logger.error(f"Failed to load gender model: {e}")

    try:
        candidates = sorted(MODELS_PATH.glob("hotel_recommender_v*.pkl"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            MODEL_CACHE["recommender"] = joblib.load(candidates[0])
            logger.info(f"Recommender loaded: {candidates[0].name}")
    except Exception as e:
        logger.error(f"Failed to load recommender: {e}")

    logger.info(f"Models loaded: {list(MODEL_CACHE.keys())}")
    yield
    logger.info("Voyage Analytics API shutting down.")


# FastAPI App
app = FastAPI(
    title="Voyage Analytics 2.0 API",
    description=(
        "Production-grade Travel Intelligence Platform.\n\n"
        "Provides flight price prediction, gender classification, "
        "and hotel recommendations with SHAP explainability."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

#  Rate limiting 
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

#  CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Prometheus metrics 
Instrumentator().instrument(app).expose(app)

#  API Key Security 
API_KEY = os.getenv("API_KEY", "voyage-dev-key-2024")
API_KEY_NAME = os.getenv("API_KEY_NAME", "X-API-Key")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return key


#  Request logging middleware 
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    logger.info(
        f"Request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": duration,
        }
    )
    return response


# Routes
app.include_router(
    flight_price.router,
    prefix="/predict",
    tags=["Flight Price"],
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    gender.router,
    prefix="/predict",
    tags=["Gender Classification"],
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    recommend.router,
    prefix="/recommend",
    tags=["Hotel Recommendation"],
    dependencies=[Depends(verify_api_key)],
)


# Core Endpoints
@app.get("/health", tags=["System"])
async def health():
    """System health check and model version info."""
    from src.utils.model_registry import get_production_model

    return {
        "status": "healthy",
        "api_version": "2.0.0",
        "models_loaded": list(MODEL_CACHE.keys()),
        "production_models": {
            "flight":      get_production_model("FlightPricePredictor"),
            "gender":      get_production_model("GenderClassifier"),
            "recommender": get_production_model("HotelRecommender"),
        }
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "🚀 Voyage Analytics 2.0 API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


#  Expose MODEL_CACHE to routes 
def get_model_cache() -> dict:
    return MODEL_CACHE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=False,
        workers=1,
    )
