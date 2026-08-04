"""
Voyage Analytics 2.0 — Data Preprocessing Pipeline
Cleans and prepares all 3 datasets for model training.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Flights Preprocessing

def preprocess_flights(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare flights dataset."""
    logger.info(f"Preprocessing flights: {df.shape}")
    df = df.copy()

    # --- Drop duplicates ---
    df.drop_duplicates(inplace=True)

    # --- Parse date ---
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    # --- Drop rows with missing critical fields ---
    df.dropna(subset=["price", "distance", "time", "flightType", "agency"], inplace=True)

    # --- Remove impossible values ---
    df = df[(df["price"] > 0) & (df["distance"] > 0) & (df["time"] > 0)]

    # --- Normalize text ---
    df["flightType"] = df["flightType"].str.strip().str.lower()
    df["agency"] = df["agency"].str.strip()
    df["from"] = df["from"].str.strip()
    df["to"] = df["to"].str.strip()

    logger.info(f"Flights after cleaning: {df.shape}")
    return df.reset_index(drop=True)

# Hotels Preprocessing

def preprocess_hotels(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare hotels dataset."""
    logger.info(f"Preprocessing hotels: {df.shape}")
    df = df.copy()

    df.drop_duplicates(inplace=True)

    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    df.dropna(subset=["price", "days", "total", "name", "place"], inplace=True)
    df = df[(df["price"] > 0) & (df["days"] > 0) & (df["total"] > 0)]

    df["name"] = df["name"].str.strip()
    df["place"] = df["place"].str.strip()

    logger.info(f"Hotels after cleaning: {df.shape}")
    return df.reset_index(drop=True)

# Users Preprocessing

def preprocess_users(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare users dataset."""
    logger.info(f"Preprocessing users: {df.shape}")
    df = df.copy()

    df.drop_duplicates(inplace=True)
    df.dropna(subset=["gender", "age", "company"], inplace=True)

    df["gender"] = df["gender"].str.strip().str.lower()
    df["company"] = df["company"].str.strip()
    df["name"] = df["name"].str.strip()

    df = df[df["gender"].isin(["male", "female"])]
    df = df[(df["age"] > 0) & (df["age"] < 120)]

    df["gender_binary"] = (df["gender"] == "male").astype(int)

    logger.info(f"Users after cleaning: {df.shape}")
    return df.reset_index(drop=True)

# Run Pipeline

def run_preprocessing_pipeline(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, clean, and save all 3 datasets."""
    raw = Path(raw_dir)
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  VOYAGE ANALYTICS 2.0 — DATA PREPROCESSING")
    print("="*60)

    flights_raw = pd.read_csv(raw / "flights.csv")
    hotels_raw  = pd.read_csv(raw / "hotels.csv")
    users_raw   = pd.read_csv(raw / "users.csv")

    flights = preprocess_flights(flights_raw)
    hotels  = preprocess_hotels(hotels_raw)
    users   = preprocess_users(users_raw)

    flights.to_csv(out / "flights_clean.csv", index=False)
    hotels.to_csv(out / "hotels_clean.csv", index=False)
    users.to_csv(out / "users_clean.csv", index=False)

    print(f"  ✅  flights_clean.csv  — {flights.shape[0]:,} rows × {flights.shape[1]} cols")
    print(f"  ✅  hotels_clean.csv   — {hotels.shape[0]:,} rows × {hotels.shape[1]} cols")
    print(f"  ✅  users_clean.csv    — {users.shape[0]:,} rows × {users.shape[1]} cols")
    print(f"\n  Saved to: {out.resolve()}")
    print("="*60 + "\n")

    return flights, hotels, users


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    run_preprocessing_pipeline()
