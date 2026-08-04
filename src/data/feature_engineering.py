"""
Voyage Analytics 2.0 — Feature Engineering Pipeline
Creates rich, informative features for all 3 ML tasks.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

BRAZIL_HOLIDAYS = [
    "01/01", "04/21", "05/01", "09/07",
    "10/12", "11/02", "11/15", "12/25",
]

# Flights Feature Engineering

def engineer_flights(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for the flight price regression model."""
    logger.info("Engineering flight features...")
    df = df.copy()

    # Date features
    df["year"]    = df["date"].dt.year
    df["month"]   = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Season (Southern Hemisphere — Brazil)
    def get_season(month):
        if month in [12, 1, 2]:  return "summer"
        elif month in [3, 4, 5]:  return "autumn"
        elif month in [6, 7, 8]:  return "winter"
        else:                      return "spring"

    df["season"] = df["month"].apply(get_season)

    # Holiday flag (Brazilian public holidays)
    df["month_day"] = df["date"].dt.strftime("%m/%d")
    df["is_holiday"] = df["month_day"].isin(BRAZIL_HOLIDAYS).astype(int)
    df.drop(columns=["month_day"], inplace=True)

    # Price per km
    df["price_per_km"] = (df["price"] / df["distance"]).round(4)

    # Speed proxy
    df["speed_proxy"] = (df["distance"] / df["time"]).round(4)

    # Agency popularity (frequency encode)
    agency_counts = df["agency"].value_counts()
    df["agency_popularity"] = df["agency"].map(agency_counts)

    # Route encode (from → to)
    df["route"] = df["from"].str[:3] + "_" + df["to"].str[:3]

    # Label-encode categoricals
    for col in ["flightType", "agency", "from", "to", "season", "route"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    logger.info(f"Flights features ready: {df.shape}")
    return df

# Hotels Feature Engineering

def engineer_hotels(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for hotel recommendation."""
    logger.info("Engineering hotel features...")
    df = df.copy()

    df["price_per_day"] = (df["price"]).round(4)
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Stay duration category
    def stay_cat(d):
        if d <= 2:   return "short"
        elif d <= 5: return "medium"
        else:        return "long"
    df["stay_category"] = df["days"].apply(stay_cat)

    # Hotel popularity (how many times booked)
    hotel_counts = df["name"].value_counts()
    df["hotel_popularity"] = df["name"].map(hotel_counts)

    # Place popularity
    place_counts = df["place"].value_counts()
    df["place_popularity"] = df["place"].map(place_counts)

    # Avg spend per user
    user_avg_spend = df.groupby("userCode")["total"].mean()
    df["user_avg_spend"] = df["userCode"].map(user_avg_spend)

    logger.info(f"Hotels features ready: {df.shape}")
    return df

# Users Feature Engineering

def engineer_users(
    users: pd.DataFrame,
    flights: pd.DataFrame,
    hotels: pd.DataFrame
) -> pd.DataFrame:
    """Create behavioral features for the gender classification model."""
    logger.info("Engineering user features...")
    df = users.copy()

    # Travel frequency
    travel_freq = flights.groupby("userCode")["travelCode"].count().rename("travel_frequency")
    df = df.merge(travel_freq, left_on="code", right_on="userCode", how="left")
    df["travel_frequency"] = df["travel_frequency"].fillna(0)

    # Avg flight price per user
    avg_flight_price = flights.groupby("userCode")["price"].mean().rename("avg_flight_price")
    df = df.merge(avg_flight_price, left_on="code", right_on="userCode", how="left")
    df["avg_flight_price"] = df["avg_flight_price"].fillna(flights["price"].median())

    # Total flight spend
    total_flight_spend = flights.groupby("userCode")["price"].sum().rename("total_flight_spend")
    df = df.merge(total_flight_spend, left_on="code", right_on="userCode", how="left")
    df["total_flight_spend"] = df["total_flight_spend"].fillna(0)

    # Preferred flight type (mode)
    preferred_ft = (
        flights.groupby("userCode")["flightType"]
        .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else "economyclass")
        .rename("preferred_flight_type")
    )
    df = df.merge(preferred_ft, left_on="code", right_on="userCode", how="left")
    df["preferred_flight_type"] = df["preferred_flight_type"].fillna("economyclass")

    # Hotel bookings count
    hotel_count = hotels.groupby("userCode")["travelCode"].count().rename("hotel_bookings")
    df = df.merge(hotel_count, left_on="code", right_on="userCode", how="left")
    df["hotel_bookings"] = df["hotel_bookings"].fillna(0)

    # Avg hotel spend
    avg_hotel_spend = hotels.groupby("userCode")["total"].mean().rename("avg_hotel_spend")
    df = df.merge(avg_hotel_spend, left_on="code", right_on="userCode", how="left")
    df["avg_hotel_spend"] = df["avg_hotel_spend"].fillna(hotels["total"].median())

    # Age group
    def age_group(a):
        if a < 25:   return "young"
        elif a < 40: return "adult"
        elif a < 60: return "middle_aged"
        else:        return "senior"
    df["age_group"] = df["age"].apply(age_group)

    # Spending category
    df["spending_category"] = pd.cut(
        df["avg_flight_price"],
        bins=[0, 300, 800, 1500, np.inf],
        labels=["budget", "economy", "premium", "luxury"]
    ).astype(str)

    # Label encode
    for col in ["company", "preferred_flight_type", "age_group", "spending_category"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    logger.info(f"Users features ready: {df.shape}")
    return df

# Run Feature Engineering

def run_feature_engineering(processed_dir: str = "data/processed"):
    """Load clean data, engineer features, save to processed dir."""
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  VOYAGE ANALYTICS 2.0 — FEATURE ENGINEERING")
    print("="*60)

    flights = pd.read_csv(out / "flights_clean.csv", parse_dates=["date"])
    hotels  = pd.read_csv(out / "hotels_clean.csv",  parse_dates=["date"])
    users   = pd.read_csv(out / "users_clean.csv")

    flights_fe = engineer_flights(flights)
    hotels_fe  = engineer_hotels(hotels)
    users_fe   = engineer_users(users, flights, hotels)

    flights_fe.to_csv(out / "flights_features.csv", index=False)
    hotels_fe.to_csv(out / "hotels_features.csv", index=False)
    users_fe.to_csv(out / "users_features.csv", index=False)

    print(f"  ✅  flights_features.csv — {flights_fe.shape[0]:,} rows × {flights_fe.shape[1]} cols")
    print(f"  ✅  hotels_features.csv  — {hotels_fe.shape[0]:,} rows × {hotels_fe.shape[1]} cols")
    print(f"  ✅  users_features.csv   — {users_fe.shape[0]:,} rows × {users_fe.shape[1]} cols")
    print("="*60 + "\n")

    return flights_fe, hotels_fe, users_fe


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    run_feature_engineering()
