"""
Voyage Analytics 2.0 — SHAP Explainability Module
Generates SHAP explanations for flight price & gender predictions.
Gracefully degrades if SHAP is unavailable (e.g. Python 3.14 incompatibility).
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Safe SHAP import : does NOT crash if incompatible
SHAP_AVAILABLE = False
try:
    import shap as _shap_lib

    SHAP_AVAILABLE = True
except Exception as _shap_err:
    logger.warning(f"SHAP disabled (incompatible environment): {_shap_err}")


def get_shap_explanation(
    artifact: dict,
    input_df: pd.DataFrame,
    max_display: int = 10,
) -> dict[str, Any]:
    """
    Compute SHAP values for a single prediction.
    Returns empty dict if SHAP is unavailable or fails.

    Args:
        artifact: Loaded model artifact dict (contains 'model', 'feature_cols')
        input_df: Single-row DataFrame with feature columns
        max_display: Number of top features to return
    """
    if not SHAP_AVAILABLE:
        return {}

    try:
        model = artifact["model"]
        feature_cols = artifact.get("feature_cols", list(input_df.columns))
        X = input_df[feature_cols].values

        try:
            # Tree-based models (XGBoost, RF, LightGBM)
            explainer = _shap_lib.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)

            # For classifiers, shap_values may be a list (per class)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # Positive class

            shap_arr = shap_vals[0]
            base_val = float(
                explainer.expected_value
                if not isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value[1]
            )

        except Exception:
            # Linear models fallback
            explainer = _shap_lib.LinearExplainer(model, X)
            shap_vals = explainer.shap_values(X)
            shap_arr = shap_vals[0]
            base_val = float(explainer.expected_value)

        contribs = [
            {
                "feature": feature_cols[i],
                "value": float(input_df[feature_cols[i]].iloc[0]),
                "shap_value": float(shap_arr[i]),
            }
            for i in range(len(feature_cols))
        ]
        contribs.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "base_value": base_val,
            "top_features": contribs[:max_display],
            "total_features": len(feature_cols),
        }

    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        return {}


def format_shap_for_api(shap_result: dict) -> list[dict]:
    """Format SHAP result for API response."""
    if not shap_result:
        return []
    return [
        {
            "feature": c["feature"],
            "input_value": round(c["value"], 4),
            "impact": round(c["shap_value"], 4),
            "direction": "increases" if c["shap_value"] > 0 else "decreases",
        }
        for c in shap_result.get("top_features", [])
    ]
