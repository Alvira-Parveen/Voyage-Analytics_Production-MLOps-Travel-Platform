"""
Voyage Analytics 2.0 — Flight Price Prediction Training
Auto-compares regression algorithms with 5-fold cross-validation.
Features target-leakage protection and automatic MLflow logging.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

import mlflow

try:
    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from xgboost import XGBRegressor

from src.utils.logger import get_logger
from src.utils.model_registry import promote_to_production, save_model_metadata

logger = get_logger(__name__)

# ── Clean Feature List (NO TARGET LEAKAGE) ──
FEATURE_COLS = [
    "flightType_enc",
    "agency_enc",
    "from_enc",
    "to_enc",
    "distance",
    "time",
    "speed_proxy",
    "month",
    "weekday",
    "year",
    "season_enc",
    "is_weekend",
    "is_holiday",
    "agency_popularity",
]
TARGET_COL = "price"
MODEL_NAME = "FlightPricePredictor"
EXPERIMENT = "flight-price-prediction"
MODEL_PATH = Path("models")

# Hyperparameters tuned for speed & generalization
CANDIDATE_MODELS = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=10.0),
    "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    ),
}

if LIGHTGBM_AVAILABLE:
    CANDIDATE_MODELS["LightGBM"] = LGBMRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.08, random_state=42, verbose=-1, n_jobs=-1
    )


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_flight_price_model(data_dir: str = "data/processed", version: str = "1.0"):
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(data_dir) / "flights_features.csv")

    cols_needed = FEATURE_COLS + [TARGET_COL]
    df = df.dropna(subset=cols_needed)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling setup
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLS)

    mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
    mlflow.set_experiment(EXPERIMENT)

    print("\n" + "=" * 70)
    print("  FLIGHT PRICE PREDICTION — MODEL SELECTION (5-FOLD CV)")
    print("=" * 70)
    print(f"  Training Set Size : {X_train.shape[0]:,} samples")
    print(f"  Test Set Size     : {X_test.shape[0]:,} samples")
    print(f"  Features          : {len(FEATURE_COLS)}")
    print("=" * 70)

    # Subset for cross-validation to prevent training hangs
    cv_sample_size = min(30000, len(X_train))
    cv_idx = np.random.choice(X_train.index, size=cv_sample_size, replace=False)
    X_cv = X_train.loc[cv_idx].reset_index(drop=True)
    y_cv = y_train.loc[cv_idx].reset_index(drop=True)

    results = {}

    for name, base_model in CANDIDATE_MODELS.items():
        with mlflow.start_run(run_name=name):
            # 1. 5-Fold Cross Validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_r2_scores = []
            cv_rmse_scores = []

            for train_idx, val_idx in kf.split(X_cv):
                X_tr, X_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
                y_tr, y_val = y_cv.iloc[train_idx], y_cv.iloc[val_idx]

                # Scale only on the training fold if needed
                use_scaled = name in ["LinearRegression", "Ridge"]
                if use_scaled:
                    f_scaler = StandardScaler()
                    X_tr_s = f_scaler.fit_transform(X_tr)
                    X_val_s = f_scaler.transform(X_val)
                else:
                    X_tr_s, X_val_s = X_tr.values, X_val.values

                fold_model = base_model.__class__(**base_model.get_params())
                fold_model.fit(X_tr_s, y_tr)
                preds = fold_model.predict(X_val_s)

                cv_r2_scores.append(r2_score(y_val, preds))
                cv_rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))

            cv_r2_mean = float(np.mean(cv_r2_scores))
            cv_rmse_mean = float(np.mean(cv_rmse_scores))

            # 2. Fit final model on the full training set
            use_scaled = name in ["LinearRegression", "Ridge"]
            Xtr_final = X_train_scaled if use_scaled else X_train
            Xts_final = X_test_scaled if use_scaled else X_test

            final_model = base_model.__class__(**base_model.get_params())
            final_model.fit(Xtr_final.values, y_train)

            # 3. Evaluate on unseen test set
            y_pred = final_model.predict(Xts_final.values)
            metrics = evaluate(y_test, y_pred)

            # Log to MLflow
            mlflow.log_params(final_model.get_params())
            mlflow.log_param("features", ",".join(FEATURE_COLS))
            mlflow.log_metrics({**metrics, "cv_r2_mean": cv_r2_mean, "cv_rmse_mean": cv_rmse_mean})

            # Feature Importance
            if hasattr(final_model, "feature_importances_"):
                fi = pd.Series(final_model.feature_importances_, index=FEATURE_COLS)
                mlflow.log_dict(fi.to_dict(), "feature_importance.json")

            run_id = mlflow.active_run().info.run_id
            exp_id = mlflow.active_run().info.experiment_id

            results[name] = {
                **metrics,
                "cv_r2_mean": cv_r2_mean,
                "run_id": run_id,
                "exp_id": exp_id,
                "model": final_model,
                "scaler": scaler if use_scaled else None,
            }

            print(f"  {name:<20} | CV R²={cv_r2_mean:.4f} | Test R²={metrics['r2']:.4f} | RMSE={metrics['rmse']:.2f}")

    # Select best model based on CV R² score
    best_name = max(results, key=lambda k: results[k]["cv_r2_mean"])
    best = results[best_name]

    print("-" * 70)
    print(f"  🏆 Best Algorithm  : {best_name}")
    print(f"  📊 Best Test R²     : {best['r2']:.4f}")
    print(f"  📊 CV R² Mean       : {best['cv_r2_mean']:.4f}")
    print("=" * 70)

    # Save artifact
    model_file = MODEL_PATH / f"flight_price_v{version}.pkl"
    artifact = {"model": best["model"], "scaler": best["scaler"], "feature_cols": FEATURE_COLS, "model_name": best_name}
    joblib.dump(artifact, model_file)
    print(f"  💾 Saved Production Artifact: {model_file}")

    # Register in Model Registry metadata
    save_model_metadata(
        model_name=MODEL_NAME,
        version=version,
        algorithm=best_name,
        metrics={"rmse": best["rmse"], "mae": best["mae"], "r2": best["r2"], "cv_r2_mean": best["cv_r2_mean"]},
        mlflow_run_id=best["run_id"],
        mlflow_experiment_id=best["exp_id"],
        feature_columns=FEATURE_COLS,
        stage="Production",
    )
    promote_to_production(MODEL_NAME, version)
    print(f"  📋 Registered v{version} -> Production")
    print("=" * 70 + "\n")

    return model_file, best


if __name__ == "__main__":
    train_flight_price_model()
