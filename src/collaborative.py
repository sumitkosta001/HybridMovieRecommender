"""
Collaborative Filtering Module with Popularity Fallback.
Computes personalized rating predictions using Matrix Factorization (Truncated SVD),
with Bayesian weighted rating fallback when ratings are sparse or for unknown users (cold-start).
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


class CollaborativeRecommender:
    """
    Matrix-Factorization based collaborative filtering model with fallback mechanics.
    """
    def __init__(self, n_factors: int = 4, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
        
        self.all_movie_ids: List[int] = []
        self.users: List[int] = []
        self.user_to_idx: Dict[int, int] = {}
        self.movie_to_idx: Dict[int, int] = {}
        self.idx_to_movie: Dict[int, int] = {}
        
        self.predicted_matrix: Optional[np.ndarray] = None
        self.global_mean: float = 3.5
        self.popularity_scores: Dict[int, float] = {}

    def fit(self, ratings_df: pd.DataFrame, all_movie_ids: Optional[List[int]] = None) -> "CollaborativeRecommender":
        """
        Fits SVD on User x Movie interaction matrix.
        Precomputes popularity scores for cold-start fallback.
        """
        if ratings_df.empty:
            if all_movie_ids:
                self.all_movie_ids = list(all_movie_ids)
                self.popularity_scores = {mid: 0.5 for mid in self.all_movie_ids}
            return self

        self.global_mean = float(ratings_df["rating"].mean())
        if all_movie_ids is not None:
            self.all_movie_ids = sorted(list(set(all_movie_ids)))
        else:
            self.all_movie_ids = sorted(ratings_df["movie_id"].unique().tolist())

        # Precompute Bayesian-smoothed item popularity scores for fallback
        # Score = (v / (v + m)) * R + (m / (v + m)) * C
        # where v = count of ratings, m = min rating threshold (default 2), R = mean rating, C = global mean
        m_thresh = 2.0
        item_stats = ratings_df.groupby("movie_id")["rating"].agg(["count", "mean"]).to_dict(orient="index")
        
        for mid in self.all_movie_ids:
            if mid in item_stats:
                v = float(item_stats[mid]["count"])
                r = float(item_stats[mid]["mean"])
                bayes_score = (v / (v + m_thresh)) * r + (m_thresh / (v + m_thresh)) * self.global_mean
            else:
                bayes_score = self.global_mean
            # Normalize from [1.0, 5.0] to [0.0, 1.0]
            self.popularity_scores[mid] = float(np.clip((bayes_score - 1.0) / 4.0, 0.0, 1.0))

        # Pivot into User x Movie matrix
        pivot = ratings_df.pivot_table(index="user_id", columns="movie_id", values="rating")
        
        # Ensure all catalog movies are present
        for mid in self.all_movie_ids:
            if mid not in pivot.columns:
                pivot[mid] = np.nan

        pivot = pivot.sort_index(axis=1)
        self.users = list(pivot.index)
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.users)}
        self.movie_to_idx = {mid: idx for idx, mid in enumerate(pivot.columns)}
        self.idx_to_movie = {idx: mid for mid, idx in self.movie_to_idx.items()}

        # Fill missing ratings with item mean or global mean
        item_means = pivot.mean(axis=0).fillna(self.global_mean)
        pivot_filled = pivot.apply(lambda col: col.fillna(item_means[col.name]), axis=0)

        # Handle matrix dimensionality for SVD
        n_samples, n_features = pivot_filled.shape
        effective_k = min(self.n_factors, n_samples - 1, n_features - 1)
        
        if effective_k >= 1:
            self.svd.n_components = effective_k
            user_factors = self.svd.fit_transform(pivot_filled.values)
            item_factors = self.svd.components_
            reconstructed = np.dot(user_factors, item_factors)
            self.predicted_matrix = np.clip(reconstructed, 1.0, 5.0)
        else:
            # Fallback to mean matrix if insufficient users/items to factorize
            self.predicted_matrix = pivot_filled.values

        return self

    def get_collaborative_scores(self, user_id: int) -> Dict[int, float]:
        """
        Returns {movie_id: normalized_score in [0.0, 1.0]}.
        If user_id is unknown or unrated (cold-start), falls back to popularity scores.
        """
        if self.predicted_matrix is None or user_id not in self.user_to_idx:
            # Cold-start fallback: popularity ranking
            return dict(self.popularity_scores)

        u_idx = self.user_to_idx[user_id]
        pred_ratings = self.predicted_matrix[u_idx]
        
        # Normalize 1-5 scale into [0.0, 1.0]
        norm_scores = np.clip((pred_ratings - 1.0) / 4.0, 0.0, 1.0)
        return {
            self.idx_to_movie[i]: float(norm_scores[i]) 
            for i in range(len(norm_scores))
        }

    def is_known_user(self, user_id: int) -> bool:
        """Checks if user has rating history in the training matrix."""
        return user_id in self.user_to_idx
