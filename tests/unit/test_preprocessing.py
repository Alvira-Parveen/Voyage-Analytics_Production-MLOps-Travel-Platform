"""
Unit Tests — Preprocessing & Feature Engineering
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocess import preprocess_flights, preprocess_hotels, preprocess_users
from src.data.validate import FLIGHTS_RULES, USERS_RULES, DataValidator

# Fixtures


@pytest.fixture
def sample_flights():
    return pd.DataFrame(
        {
            "travelCode": [0, 1, 2],
            "userCode": [0, 1, 2],
            "from": ["Recife (PE)", "Brasilia (DF)", "Sao Paulo (SP)"],
            "to": ["Florianopolis (SC)", "Florianopolis (SC)", "Rio de Janeiro (RJ)"],
            "flightType": ["firstClass", "economyClass", "businessClass"],
            "price": [1434.38, 500.00, 800.00],
            "time": [1.76, 2.00, 1.50],
            "distance": [676.53, 637.56, 350.00],
            "agency": ["FlyingDrops", "CloudFy", "Rainbow"],
            "date": ["09/26/2019", "10/03/2019", "11/15/2019"],
        }
    )


@pytest.fixture
def sample_hotels():
    return pd.DataFrame(
        {
            "travelCode": [0, 2],
            "userCode": [0, 0],
            "name": ["Hotel A", "Hotel K"],
            "place": ["Florianopolis (SC)", "Salvador (BH)"],
            "days": [4, 2],
            "price": [313.02, 263.41],
            "total": [1252.08, 526.82],
            "date": ["09/26/2019", "10/10/2019"],
        }
    )


@pytest.fixture
def sample_users():
    return pd.DataFrame(
        {
            "code": [0, 1, 2, 3],
            "company": ["4You", "4You", "4You", "4You"],
            "name": ["Roy Braun", "Joseph H", "Wilma M", "Paula D"],
            "gender": ["male", "male", "female", "female"],
            "age": [21, 37, 48, 23],
        }
    )


# Preprocessing Tests


class TestFlightsPreprocessing:

    def test_returns_dataframe(self, sample_flights):
        result = preprocess_flights(sample_flights)
        assert isinstance(result, pd.DataFrame)

    def test_removes_duplicates(self, sample_flights):
        df_dup = pd.concat([sample_flights, sample_flights])
        result = preprocess_flights(df_dup)
        assert len(result) == len(sample_flights)

    def test_date_parsed(self, sample_flights):
        result = preprocess_flights(sample_flights)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_no_negative_prices(self, sample_flights):
        df_bad = sample_flights.copy()
        df_bad.loc[0, "price"] = -100
        result = preprocess_flights(df_bad)
        assert (result["price"] > 0).all()

    def test_flighttype_lowercase(self, sample_flights):
        df = sample_flights.copy()
        df.loc[0, "flightType"] = "FirstClass"
        result = preprocess_flights(df)
        assert result["flightType"].str.islower().all()


class TestUsersPreprocessing:

    def test_gender_binary_column(self, sample_users):
        result = preprocess_users(sample_users)
        assert "gender_binary" in result.columns

    def test_gender_binary_values(self, sample_users):
        result = preprocess_users(sample_users)
        assert set(result["gender_binary"].unique()).issubset({0, 1})

    def test_invalid_gender_removed(self, sample_users):
        df = sample_users.copy()
        df.loc[0, "gender"] = "unknown"
        result = preprocess_users(df)
        assert "unknown" not in result["gender"].values

    def test_no_negative_age(self, sample_users):
        df = sample_users.copy()
        df.loc[0, "age"] = -5
        result = preprocess_users(df)
        assert (result["age"] > 0).all()


# Validation Tests


class TestDataValidator:

    def test_passes_valid_data(self, sample_flights):
        v = DataValidator(sample_flights, FLIGHTS_RULES, "flights")
        report = v.validate()
        # Sample has 3 rows so min_rows will fail, but no CRITICAL column issues
        critical_non_row = [
            i for i in report["issues"] if i.get("severity") == "CRITICAL" and i.get("type") != "INSUFFICIENT_ROWS"
        ]
        assert len(critical_non_row) == 0

    def test_detects_missing_columns(self):
        df = pd.DataFrame({"price": [100], "distance": [500]})
        v = DataValidator(df, FLIGHTS_RULES, "flights")
        report = v.validate()
        assert report["status"] == "FAILED"
        assert any(i["type"] == "MISSING_COLUMNS" for i in report["issues"])

    def test_detects_insufficient_rows(self, sample_flights):
        rules = {**FLIGHTS_RULES, "min_rows": 1000}
        v = DataValidator(sample_flights, rules, "flights")
        report = v.validate()
        assert any(i["type"] == "INSUFFICIENT_ROWS" for i in report["issues"])

    def test_report_has_required_keys(self, sample_flights):
        v = DataValidator(sample_flights, FLIGHTS_RULES, "flights")
        report = v.validate()
        for key in ["dataset", "validated_at", "shape", "status", "issues", "summary"]:
            assert key in report
