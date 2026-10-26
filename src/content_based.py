"""
Content-Based Recommendation Engine.
Computes text representations via TF-IDF vectorization and computes pairwise cosine similarity.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def create_metadata_soup(row: pd.Series) -> str:
    """
    Combines director (weighted by repetition), genres, cast, and overview
    into an enriched semantic string.
    """
    director = str(row.get("director", "")).strip()
    # Director given double weight for authorial stylistic signature
    director_rep = f"{director} {director}" if director else ""
    
    genres = str(row.get("genres", "")).replace("|", " ")
    cast = str(row.get("cast", ""))
    overview = str(row.get("overview", ""))

    tokens = [director_rep, genres, cast, overview]
    return " ".join([t for t in tokens if t]).lower()


class ContentBasedRecommender:
    """
    Content-Based filtering model based on TF-IDF cosine similarity.
    """
    def __init__(self, stop_words: str = "english", ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=ngram_range,
            min_df=1
        )
        self.movies_df: Optional[pd.DataFrame] = None
        self.tfidf_matrix = None
        self.cosine_sim_matrix: Optional[np.ndarray] = None
        self.movie_id_to_idx: Dict[int, int] = {}
        self.idx_to_movie_id: Dict[int, int] = {}

    def fit(self, movies_df: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Fits TF-IDF vectorizer and builds cosine similarity matrix.
        """
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.movie_id_to_idx = {
            int(mid): idx for idx, mid in enumerate(self.movies_df["movie_id"])
        }
        self.idx_to_movie_id = {
            idx: int(mid) for idx, mid in enumerate(self.movies_df["movie_id"])
        }

        soups = self.movies_df.apply(create_metadata_soup, axis=1)
        self.tfidf_matrix = self.vectorizer.fit_transform(soups)
        self.cosine_sim_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Ensure cosine bounds [0.0, 1.0]
        np.clip(self.cosine_sim_matrix, 0.0, 1.0, out=self.cosine_sim_matrix)
        return self

    def get_similar_movies(self, movie_id: int, top_k: int = 5) -> List[Dict]:
        """
        Returns top-K similar movies to a given movie_id.
        Self-movie is explicitly excluded.
        Returns empty list if movie_id is not in catalog.
        """
        if self.cosine_sim_matrix is None or movie_id not in self.movie_id_to_idx:
            return []

        idx = self.movie_id_to_idx[movie_id]
        scores = list(enumerate(self.cosine_sim_matrix[idx]))
        
        # Exclude self (index == idx) and sort descending
        scores = [s for s in scores if s[0] != idx]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_k]

        results = []
        source_row = self.movies_df.iloc[idx]
        src_genres = set(str(source_row.get("genres", "")).split("|"))

        for match_idx, sim_score in top_scores:
            matched_row = self.movies_df.iloc[match_idx]
            match_genres = set(str(matched_row.get("genres", "")).split("|"))
            shared_genres = list(src_genres.intersection(match_genres))
            same_dir = source_row.get("director") == matched_row.get("director")

            reasons = []
            if same_dir and source_row.get("director"):
                reasons.append(f"Directed by {matched_row['director']}")
            if shared_genres:
                reasons.append(f"Genres: {', '.join(shared_genres)}")
            reason_str = " • ".join(reasons) if reasons else "Thematic story alignment"

            results.append({
                "movie_id": int(matched_row["movie_id"]),
                "title": matched_row["title"],
                "genres": matched_row["genres"],
                "director": matched_row["director"],
                "release_year": int(matched_row["release_year"]),
                "vote_average": float(matched_row["vote_average"]),
                "similarity_score": round(float(sim_score), 4),
                "reason": reason_str
            })

        return results

    def get_content_scores(self, movie_id: int) -> Dict[int, float]:
        """
        Returns dictionary of {movie_id: content_score} normalized in [0, 1].
        If movie_id not found, returns zeros for all known movies.
        """
        if self.cosine_sim_matrix is None:
            return {}

        if movie_id not in self.movie_id_to_idx:
            return {mid: 0.0 for mid in self.movie_id_to_idx}

        idx = self.movie_id_to_idx[movie_id]
        scores = self.cosine_sim_matrix[idx]
        return {self.idx_to_movie_id[i]: float(scores[i]) for i in range(len(scores))}
