"""
Hybrid Movie Recommendation Engine.
Blends content-based metadata similarity with collaborative filtering scores
using an alpha weighting parameter, cold-start fallback mechanics, and self-exclusion.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender


class HybridRecommender:
    """
    Hybrid Recommender combining Content-Based and Collaborative Filtering:
    Score = alpha * Score_content + (1 - alpha) * Score_collaborative
    """
    def __init__(self, content_engine: ContentBasedRecommender, cf_engine: CollaborativeRecommender):
        self.content_engine = content_engine
        self.cf_engine = cf_engine
        self.movies_df = content_engine.movies_df
        self.ratings_df: Optional[pd.DataFrame] = None

    def set_ratings_data(self, ratings_df: pd.DataFrame):
        """Sets ratings dataframe used to identify already-rated movies."""
        self.ratings_df = ratings_df.copy()

    def recommend(
        self,
        user_id: Optional[int] = None,
        seed_movie_id: Optional[int] = None,
        top_k: int = 5,
        alpha: float = 0.5,
        exclude_rated: bool = True
    ) -> List[Dict]:
        """
        Produces top-K hybrid recommendations.
        - alpha = 1.0 -> 100% Content-Based
        - alpha = 0.0 -> 100% Collaborative Filtering (or popularity fallback)
        - alpha = 0.5 -> Balanced Hybrid

        Guarantees:
        - Self-exclusion (seed_movie_id is never in recommendation results)
        - Excludes already rated items if exclude_rated=True
        - Handles unknown user_id and unknown seed_movie_id gracefully
        - Hybrid scores and subscores strictly clamped to [0.0, 1.0]
        """
        # Clamp alpha to [0.0, 1.0]
        alpha = float(np.clip(alpha, 0.0, 1.0))
        all_movies = self.movies_df.copy()
        movie_ids = all_movies["movie_id"].tolist()

        # Identify rated items
        rated_ids = set()
        if exclude_rated and self.ratings_df is not None and user_id is not None:
            user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
            rated_ids = set(user_ratings["movie_id"].tolist())

        # 1. Collaborative scores in [0.0, 1.0]
        if user_id is not None:
            cf_scores = self.cf_engine.get_collaborative_scores(user_id)
        else:
            cf_scores = dict(self.cf_engine.popularity_scores)

        # 2. Content scores in [0.0, 1.0]
        if seed_movie_id is not None and seed_movie_id in self.content_engine.movie_id_to_idx:
            content_scores = self.content_engine.get_content_scores(seed_movie_id)
        elif user_id is not None and self.ratings_df is not None:
            # Build user profile from user's favorite movies (ratings >= 4.0)
            user_favs = self.ratings_df[
                (self.ratings_df["user_id"] == user_id) & (self.ratings_df["rating"] >= 4.0)
            ]
            if not user_favs.empty:
                fav_mids = user_favs.sort_values(by="rating", ascending=False)["movie_id"].head(3).tolist()
                accum = np.zeros(len(movie_ids))
                for f_mid in fav_mids:
                    f_scores = self.content_engine.get_content_scores(f_mid)
                    for i, mid in enumerate(movie_ids):
                        accum[i] += f_scores.get(mid, 0.0)
                accum /= max(1, len(fav_mids))
                content_scores = {mid: float(accum[i]) for i, mid in enumerate(movie_ids)}
            else:
                content_scores = {mid: 0.5 for mid in movie_ids}
        else:
            # Cold-start neutral content baseline
            content_scores = {mid: 0.5 for mid in movie_ids}

        # 3. Hybrid ranking computation
        scored = []
        is_cf_known = (user_id is not None and self.cf_engine.is_known_user(user_id))

        for mid in movie_ids:
            # Self-exclusion: query seed movie must never be recommended
            if seed_movie_id is not None and mid == seed_movie_id:
                continue

            # Exclude already rated
            if exclude_rated and mid in rated_ids:
                continue

            c_val = float(np.clip(content_scores.get(mid, 0.0), 0.0, 1.0))
            cf_val = float(np.clip(cf_scores.get(mid, 0.5), 0.0, 1.0))
            h_val = float(np.clip((alpha * c_val) + ((1.0 - alpha) * cf_val), 0.0, 1.0))

            meta = all_movies[all_movies["movie_id"] == mid].iloc[0]

            # Explanation rationale
            if alpha >= 0.7:
                reason = f"High content similarity with {meta['genres']} and director {meta['director']}"
            elif alpha <= 0.3:
                if is_cf_known:
                    reason = f"Community preference match (predicted ~{1.0 + cf_val * 4.0:.1f}★)"
                else:
                    reason = "Popularity consensus among viewers with verified reviews"
            else:
                reason = "Balanced thematic alignment & collective recommendation affinity"

            scored.append({
                "movie_id": int(mid),
                "title": meta["title"],
                "genres": meta["genres"],
                "director": meta["director"],
                "release_year": int(meta["release_year"]),
                "vote_average": float(meta["vote_average"]),
                "overview": meta["overview"],
                "content_score": round(c_val, 4),
                "collaborative_score": round(cf_val, 4),
                "predicted_rating": round(1.0 + (cf_val * 4.0), 2),
                "hybrid_score": round(h_val, 4),
                "reason": reason
            })

        # Sort descending by hybrid_score
        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored[:top_k]
