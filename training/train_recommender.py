"""
Voyage Analytics 2.0 — Hotel Recommendation System Training
Uses the HybridRecommender from src.models.hotel_recommender.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import mlflow
import pandas as pd

from src.models.hotel_recommender import HybridRecommender
from src.utils.logger import get_logger
from src.utils.model_registry import promote_to_production, save_model_metadata

logger = get_logger(__name__)

MODEL_NAME = "HotelRecommender"
EXPERIMENT = "hotel-recommendation"
MODEL_PATH = Path("models")


def train_recommender(data_dir: str = "data/processed", version: str = "1.0"):
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    hotels_df = pd.read_csv(Path(data_dir) / "hotels_features.csv")

    mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
    mlflow.set_experiment(EXPERIMENT)

    print("\n" + "=" * 65)
    print("  HOTEL RECOMMENDATION — HYBRID ENGINE TRAINING")
    print("=" * 65)
    print(f"  Hotels dataset   : {hotels_df.shape[0]:,} bookings")
    print(f"  Unique hotels    : {hotels_df['name'].nunique()}")
    print(f"  Unique users     : {hotels_df['userCode'].nunique()}")
    print(f"  Unique places    : {hotels_df['place'].nunique()}")
    print("=" * 65)

    with mlflow.start_run(run_name="HybridRecommender"):
        recommender = HybridRecommender()
        recommender.fit(hotels_df)

        metrics = {
            "unique_hotels": int(hotels_df["name"].nunique()),
            "unique_users": int(hotels_df["userCode"].nunique()),
            **recommender.svd_metrics,
        }

        mlflow.log_params({"cf_model": "SVD", "n_factors": 50, "cb_model": "cosine_similarity"})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

        run_id = mlflow.active_run().info.run_id
        exp_id = mlflow.active_run().info.experiment_id

    model_file = MODEL_PATH / f"hotel_recommender_v{version}.pkl"
    joblib.dump(recommender, model_file)
    print(f"  💾  Saved: {model_file}")

    if recommender.svd_metrics:
        print(f"  📊  SVD RMSE = {recommender.svd_metrics.get('svd_rmse', 'N/A'):.4f}")
        print(f"  📊  SVD MAE  = {recommender.svd_metrics.get('svd_mae',  'N/A'):.4f}")

    save_model_metadata(
        model_name=MODEL_NAME,
        version=version,
        algorithm="HybridSVD+ContentBased",
        metrics=metrics,
        mlflow_run_id=run_id,
        mlflow_experiment_id=exp_id,
        feature_columns=["userCode", "name", "total", "place", "days"],
        stage="Production",
    )
    promote_to_production(MODEL_NAME, version)
    print(f"  📋  Registered v{version} → Production")
    print("=" * 65 + "\n")

    return model_file, recommender


if __name__ == "__main__":
    train_recommender()
