"""
Voyage Analytics 2.0 — Gender Classification Training
Auto-compares multiple classifiers via MLflow.
Automatically promotes the best model to Production.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import mlflow
from src.utils.logger import get_logger
from src.utils.model_registry import promote_to_production, save_model_metadata

logger = get_logger(__name__)

# ─────────────────────────────────────────────
FEATURE_COLS = [
    "age",
    "company_enc",
    "travel_frequency",
    "avg_flight_price",
    "total_flight_spend",
    "preferred_flight_type_enc",
    "hotel_bookings",
    "avg_hotel_spend",
    "age_group_enc",
    "spending_category_enc",
]
TARGET_COL = "gender_binary"
MODEL_NAME = "GenderClassifier"
EXPERIMENT = "gender-classification"
MODEL_PATH = Path("models")

CANDIDATE_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
    "DecisionTree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    ),
}


def evaluate_classifier(y_true, y_pred, y_proba) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision_male": float(report.get("1", {}).get("precision", 0)),
        "recall_male": float(report.get("1", {}).get("recall", 0)),
    }


def train_gender_classifier(data_dir: str = "data/processed", version: str = "1.0"):
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(data_dir) / "users_features.csv")

    cols_needed = FEATURE_COLS + [TARGET_COL]
    df = df.dropna(subset=cols_needed)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
    mlflow.set_experiment(EXPERIMENT)

    print("\n" + "=" * 65)
    print("  GENDER CLASSIFICATION — AUTO MODEL SELECTION")
    print("=" * 65)
    print(f"  Training set : {X_train.shape[0]:,} samples")
    print(f"  Test set     : {X_test.shape[0]:,} samples")
    print(f"  Class balance: {y.value_counts().to_dict()}")
    print("=" * 65)

    results = {}

    for name, model in CANDIDATE_MODELS.items():
        with mlflow.start_run(run_name=name):
            use_scaled = name in ["LogisticRegression"]
            Xtr = X_train_s if use_scaled else X_train.values
            Xts = X_test_s if use_scaled else X_test.values

            model.fit(Xtr, y_train)
            y_pred = model.predict(Xts)
            y_proba = model.predict_proba(Xts)[:, 1]
            metrics = evaluate_classifier(y_test, y_pred, y_proba)

            mlflow.log_params({"model": name, "features": len(FEATURE_COLS), "train_size": len(X_train)})
            mlflow.log_metrics(metrics)

            if hasattr(model, "feature_importances_"):
                fi = pd.Series(model.feature_importances_, index=FEATURE_COLS)
                mlflow.log_dict(fi.to_dict(), "feature_importance.json")

            run_id = mlflow.active_run().info.run_id
            exp_id = mlflow.active_run().info.experiment_id

            results[name] = {
                **metrics,
                "run_id": run_id,
                "exp_id": exp_id,
                "model": model,
                "scaler": scaler if use_scaled else None,
            }

            print(
                f"  {'✅' if metrics['f1_macro'] > 0.75 else '➡️ '}  {name:<22}"
                f"  Acc={metrics['accuracy']:.4f}  "
                f"F1={metrics['f1_macro']:.4f}  "
                f"AUC={metrics['roc_auc']:.4f}"
            )

    # ── Best by F1 macro ──
    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    best = results[best_name]

    print(f"\n  🏆  Best Model: {best_name}  (F1={best['f1_macro']:.4f})")

    model_file = MODEL_PATH / f"gender_classifier_v{version}.pkl"
    artifact = {
        "model": best["model"],
        "scaler": best["scaler"],
        "feature_cols": FEATURE_COLS,
        "model_name": best_name,
        "classes": ["female", "male"],
    }
    joblib.dump(artifact, model_file)
    print(f"  💾  Saved: {model_file}")

    save_model_metadata(
        model_name=MODEL_NAME,
        version=version,
        algorithm=best_name,
        metrics={"accuracy": best["accuracy"], "f1_macro": best["f1_macro"], "roc_auc": best["roc_auc"]},
        mlflow_run_id=best["run_id"],
        mlflow_experiment_id=best["exp_id"],
        feature_columns=FEATURE_COLS,
        stage="Production",
    )
    promote_to_production(MODEL_NAME, version)
    print(f"  📋  Registered v{version} → Production")
    print("=" * 65 + "\n")

    return model_file, best


if __name__ == "__main__":
    train_gender_classifier()
