"""
Voyage Analytics 2.0 — Hotel Recommendation System
Hybrid Engine: Collaborative Filtering (SVD) + Content-Based.
Defined as a standalone module for proper pickling.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

try:
    from surprise import SVD, Dataset, Reader, accuracy
    from surprise.model_selection import train_test_split as s_split

    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    logger.warning("scikit-surprise not installed — content-based only")


class ContentBasedRecommender:
    """Recommends hotels based on feature similarity."""

    def __init__(self):
        self.hotel_profiles = None
        self.similarity_matrix = None
        self.hotel_names = []

    def fit(self, hotels_df: pd.DataFrame):
        feature_cols = ["price", "days", "hotel_popularity", "place_popularity"]
        available = [c for c in feature_cols if c in hotels_df.columns]

        profiles = hotels_df.groupby("name")[available].mean().reset_index()
        self.hotel_names = profiles["name"].tolist()

        scaler = MinMaxScaler()
        features = scaler.fit_transform(profiles[available])
        self.similarity_matrix = cosine_similarity(features)
        self.hotel_profiles = profiles
        return self

    def get_similar_hotels(self, hotel_name: str, top_n: int = 5):
        if hotel_name not in self.hotel_names:
            return self._get_popular(top_n)
        idx = self.hotel_names.index(hotel_name)
        sims = self.similarity_matrix[idx]
        top_indices = np.argsort(sims)[::-1][1 : top_n + 1]
        return [{"hotel": self.hotel_names[i], "similarity": float(sims[i])} for i in top_indices]

    def _get_popular(self, n: int):
        if self.hotel_profiles is None:
            return []
        col = "hotel_popularity" if "hotel_popularity" in self.hotel_profiles.columns else "price"
        top = self.hotel_profiles.nlargest(n, col)
        return [{"hotel": h, "similarity": 1.0} for h in top["name"].tolist()]


class CollaborativeRecommender:
    """SVD-based collaborative filtering using user–hotel interactions."""

    def __init__(self):
        self.svd_model = None
        self.user_hotel_matrix = None
        self.all_hotels = []
        self.svd_metrics = {}

    def fit(self, hotels_df: pd.DataFrame):
        self.all_hotels = hotels_df["name"].unique().tolist()

        if not SURPRISE_AVAILABLE:
            return self

        interaction = (
            hotels_df.groupby(["userCode", "name"])["total"].sum().reset_index().rename(columns={"total": "rating"})
        )

        mn, mx = interaction["rating"].min(), interaction["rating"].max()
        interaction["rating"] = 1 + 4 * (interaction["rating"] - mn) / (mx - mn + 1e-9)

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(interaction[["userCode", "name", "rating"]], reader)
        trainset, testset = s_split(data, test_size=0.2, random_state=42)

        self.svd_model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
        self.svd_model.fit(trainset)
        predictions = self.svd_model.test(testset)
        self.svd_metrics = {
            "svd_rmse": float(accuracy.rmse(predictions, verbose=False)),
            "svd_mae": float(accuracy.mae(predictions, verbose=False)),
        }
        self.user_hotel_matrix = interaction
        return self

    def recommend(self, user_code: int, top_n: int = 5):
        if self.svd_model is None:
            return []

        seen = set()
        if self.user_hotel_matrix is not None:
            seen = set(self.user_hotel_matrix[self.user_hotel_matrix["userCode"] == user_code]["name"].tolist())

        unseen = [h for h in self.all_hotels if h not in seen]
        if not unseen:
            unseen = self.all_hotels

        preds = [(h, self.svd_model.predict(user_code, h).est) for h in unseen]
        preds.sort(key=lambda x: x[1], reverse=True)
        return [{"hotel": h, "predicted_rating": round(r, 3)} for h, r in preds[:top_n]]


class HybridRecommender:
    """
    Hybrid: SVD collaborative filtering + content-based fallback for cold-start.
    """

    def __init__(self):
        self.cf = CollaborativeRecommender()
        self.cb = ContentBasedRecommender()
        self.hotel_profiles = None

    @property
    def svd_metrics(self):
        return self.cf.svd_metrics

    def fit(self, hotels_df: pd.DataFrame):
        self.hotel_profiles = hotels_df.copy()
        self.cb.fit(hotels_df)
        self.cf.fit(hotels_df)
        return self

    def recommend(self, user_code: int, top_n: int = 5, reason: bool = True):
        cf_recs = self.cf.recommend(user_code, top_n)

        if cf_recs:
            enriched = []
            for rec in cf_recs:
                profile = {}
                if self.hotel_profiles is not None:
                    row = self.hotel_profiles[self.hotel_profiles["name"] == rec["hotel"]]
                    if not row.empty:
                        profile = {
                            "place": row["place"].iloc[0],
                            "avg_price_per_day": round(float(row["price"].mean()), 2),
                            "avg_stay_days": round(float(row["days"].mean()), 1),
                        }
                enriched.append(
                    {
                        **rec,
                        **profile,
                        "source": "collaborative_filtering",
                        "reason": "Based on similar users' bookings" if reason else "",
                    }
                )
            return enriched
        else:
            popular = self.cb._get_popular(top_n)
            return [
                {**h, "source": "content_based", "reason": "Popular hotel recommendation" if reason else ""}
                for h in popular
            ]
