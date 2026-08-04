"""
Voyage Analytics 2.0 — Data Validation Layer
Validates all 3 datasets before training begins.
Generates a validation report (JSON).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Validation Rules

FLIGHTS_RULES = {
    "required_columns": [
        "travelCode", "userCode", "from", "to",
        "flightType", "price", "time", "distance", "agency", "date"
    ],
    "numeric_columns": ["price", "time", "distance"],
    "categorical_columns": {
        "flightType": ["firstClass", "economyClass", "businessClass", "economic", "premium"],
    },
    "positive_columns": ["price", "time", "distance"],
    "missing_threshold": 0.05,   # Max 5% missing per column
    "min_rows": 1000,
}

HOTELS_RULES = {
    "required_columns": [
        "travelCode", "userCode", "name", "place", "days", "price", "total", "date"
    ],
    "numeric_columns": ["days", "price", "total"],
    "positive_columns": ["price", "total", "days"],
    "missing_threshold": 0.05,
    "min_rows": 100,
}

USERS_RULES = {
    "required_columns": ["code", "company", "name", "gender", "age"],
    "numeric_columns": ["age"],
    "categorical_columns": {
        "gender": ["male", "female"],
    },
    "positive_columns": ["age"],
    "missing_threshold": 0.05,
    "min_rows": 50,
}

# Validator Class

class DataValidator:
    """Validates a DataFrame against a set of rules and generates a report."""

    def __init__(self, df: pd.DataFrame, rules: dict, dataset_name: str):
        self.df = df.copy()
        self.rules = rules
        self.dataset_name = dataset_name
        self.issues: list[dict] = []
        self.passed: list[str] = []

    def _check_required_columns(self):
        missing_cols = [
            c for c in self.rules.get("required_columns", [])
            if c not in self.df.columns
        ]
        if missing_cols:
            self.issues.append({
                "type": "MISSING_COLUMNS",
                "severity": "CRITICAL",
                "detail": f"Missing columns: {missing_cols}"
            })
        else:
            self.passed.append("required_columns ✓")

    def _check_min_rows(self):
        min_rows = self.rules.get("min_rows", 0)
        if len(self.df) < min_rows:
            self.issues.append({
                "type": "INSUFFICIENT_ROWS",
                "severity": "CRITICAL",
                "detail": f"Expected >= {min_rows} rows, got {len(self.df)}"
            })
        else:
            self.passed.append(f"row_count ({len(self.df)} rows) ✓")

    def _check_missing_values(self):
        threshold = self.rules.get("missing_threshold", 0.05)
        for col in self.df.columns:
            miss_rate = self.df[col].isna().mean()
            if miss_rate > threshold:
                self.issues.append({
                    "type": "HIGH_MISSING_RATE",
                    "severity": "WARNING",
                    "column": col,
                    "detail": f"{col} has {miss_rate:.1%} missing (threshold: {threshold:.1%})"
                })
        self.passed.append("missing_value_check ✓")

    def _check_positive_columns(self):
        for col in self.rules.get("positive_columns", []):
            if col not in self.df.columns:
                continue
            neg_count = (self.df[col] <= 0).sum()
            if neg_count > 0:
                self.issues.append({
                    "type": "NEGATIVE_VALUES",
                    "severity": "WARNING",
                    "column": col,
                    "detail": f"{col} has {neg_count} non-positive values"
                })
            else:
                self.passed.append(f"{col}_positive_check ✓")

    def _check_categoricals(self):
        for col, valid_vals in self.rules.get("categorical_columns", {}).items():
            if col not in self.df.columns:
                continue
            # Normalize to lowercase for comparison
            actual = self.df[col].dropna().str.lower().unique().tolist()
            valid_lower = [v.lower() for v in valid_vals]
            unexpected = [v for v in actual if v not in valid_lower]
            if unexpected:
                self.issues.append({
                    "type": "UNEXPECTED_CATEGORIES",
                    "severity": "WARNING",
                    "column": col,
                    "detail": f"{col} has unexpected values: {unexpected[:10]}"
                })
            else:
                self.passed.append(f"{col}_category_check ✓")

    def _check_duplicates(self):
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            self.issues.append({
                "type": "DUPLICATE_ROWS",
                "severity": "INFO",
                "detail": f"{dup_count} duplicate rows found"
            })
        else:
            self.passed.append("duplicate_check ✓")

    def validate(self) -> dict[str, Any]:
        """Run all checks and return a report dict."""
        self._check_required_columns()
        self._check_min_rows()
        self._check_missing_values()
        self._check_positive_columns()
        self._check_categoricals()
        self._check_duplicates()

        critical_issues = [i for i in self.issues if i.get("severity") == "CRITICAL"]
        status = "FAILED" if critical_issues else (
            "WARNING" if self.issues else "PASSED"
        )

        report = {
            "dataset": self.dataset_name,
            "validated_at": datetime.utcnow().isoformat(),
            "shape": list(self.df.shape),
            "status": status,
            "passed_checks": self.passed,
            "issues": self.issues,
            "summary": {
                "total_checks": len(self.passed) + len(self.issues),
                "passed": len(self.passed),
                "warnings": len([i for i in self.issues if i.get("severity") == "WARNING"]),
                "critical": len(critical_issues),
            }
        }
        return report


# Run All Validations

def validate_all_datasets(
    data_dir: str = "data/raw",
    report_dir: str = "data/processed"
) -> dict[str, Any]:
    """Validate all 3 datasets and save a combined report."""
    data_path = Path(data_dir)
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    all_reports = {}
    dataset_map = {
        "flights": ("flights.csv", FLIGHTS_RULES),
        "hotels":  ("hotels.csv",  HOTELS_RULES),
        "users":   ("users.csv",   USERS_RULES),
    }

    print("\n" + "="*60)
    print("  VOYAGE ANALYTICS 2.0 — DATA VALIDATION")
    print("="*60)

    for name, (filename, rules) in dataset_map.items():
        filepath = data_path / filename
        if not filepath.exists():
            print(f"  ❌ {filename} NOT FOUND at {filepath}")
            continue

        df = pd.read_csv(filepath)
        validator = DataValidator(df, rules, name)
        report = validator.validate()
        all_reports[name] = report

        status_icon = "✅" if report["status"] == "PASSED" else (
            "⚠️" if report["status"] == "WARNING" else "❌"
        )
        print(f"\n  {status_icon}  {name.upper()} Dataset")
        print(f"     Shape   : {report['shape'][0]:,} rows × {report['shape'][1]} cols")
        print(f"     Status  : {report['status']}")
        print(f"     Checks  : {report['summary']['passed']} passed, "
              f"{report['summary']['warnings']} warnings, "
              f"{report['summary']['critical']} critical")
        if report["issues"]:
            for issue in report["issues"][:3]:
                print(f"     ⚠  {issue['detail']}")

    # Save combined report
    combined = {
        "validation_run": datetime.utcnow().isoformat(),
        "datasets": all_reports,
        "overall_status": (
            "FAILED" if any(r["status"] == "FAILED" for r in all_reports.values())
            else "WARNING" if any(r["status"] == "WARNING" for r in all_reports.values())
            else "PASSED"
        )
    }

    report_file = report_path / "validation_report.json"
    with open(report_file, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Overall Status : {combined['overall_status']}")
    print(f"  Report saved   : {report_file}")
    print("="*60 + "\n")

    return combined


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    validate_all_datasets()
