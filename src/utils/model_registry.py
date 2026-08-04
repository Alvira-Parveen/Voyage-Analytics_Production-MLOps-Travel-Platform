"""
Voyage Analytics 2.0 — Model Registry
Manages versioned model metadata (JSON-based).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

REGISTRY_DIR = Path("models/registry")


def save_model_metadata(
    model_name: str,
    version: str,
    algorithm: str,
    metrics: dict[str, float],
    mlflow_run_id: str,
    mlflow_experiment_id: str,
    feature_columns: list[str],
    dataset_version: str = "v1.0",
    stage: str = "Staging",
) -> Path:
    """Save model version metadata to the registry."""
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_name": model_name,
        "version": version,
        "algorithm": algorithm,
        "training_date": datetime.utcnow().isoformat(),
        "dataset_version": dataset_version,
        "metrics": metrics,
        "mlflow_run_id": mlflow_run_id,
        "mlflow_experiment_id": mlflow_experiment_id,
        "feature_columns": feature_columns,
        "deployment_stage": stage,
    }

    filename = REGISTRY_DIR / f"{model_name}_v{version}.json"
    with open(filename, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model metadata saved: {filename}")
    return filename


def get_production_model(model_name: str) -> Optional[dict[str, Any]]:
    """Return metadata for the current Production model."""
    if not REGISTRY_DIR.exists():
        return None

    candidates = []
    for p in REGISTRY_DIR.glob(f"{model_name}_v*.json"):
        with open(p) as f:
            meta = json.load(f)
        if meta.get("deployment_stage") == "Production":
            candidates.append(meta)

    if not candidates:
        return None

    # Return the most recently trained Production model
    return sorted(candidates, key=lambda x: x["training_date"], reverse=True)[0]


def list_all_versions(model_name: str) -> list[dict[str, Any]]:
    """List all registered versions of a model."""
    if not REGISTRY_DIR.exists():
        return []
    versions = []
    for p in REGISTRY_DIR.glob(f"{model_name}_v*.json"):
        with open(p) as f:
            versions.append(json.load(f))
    return sorted(versions, key=lambda x: x["training_date"], reverse=True)


def promote_to_production(model_name: str, version: str) -> bool:
    """Promote a specific version to Production, demote all others."""
    if not REGISTRY_DIR.exists():
        return False

    for p in REGISTRY_DIR.glob(f"{model_name}_v*.json"):
        with open(p) as f:
            meta = json.load(f)

        meta["deployment_stage"] = "Production" if meta["version"] == version else "Archived"

        with open(p, "w") as f:
            json.dump(meta, f, indent=2)

    logger.info(f"Promoted {model_name} v{version} to Production")
    return True
