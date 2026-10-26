"""
FastAPI REST Inference Service for Hybrid Movie Recommendation System.
Provides RESTful endpoints for:
- Health check and catalog stats (/health)
- Movie catalog exploration and search (/movies, /movies/{movie_id})
- Content-based 'More Like This' recommendations (/recommend/content/{movie_id})
- Personalized Hybrid recommendations with dynamic alpha (/recommend/hybrid)
- Offline evaluation metrics (/evaluation)
- Dynamic rating submission (/ratings)
"""

from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field
import pandas as pd

from src.data_loader import get_dataset
from src.engine import initialize_engines

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
SUMMARY_PATH = MODELS_DIR / "summary.json"

# Pydantic Schemas
class MovieItem(BaseModel):
    movie_id: int
    title: str
    genres: str
    director: str
    release_year: int
    vote_average: float
    overview: str

class ContentRecommendationItem(BaseModel):
    movie_id: int
    title: str
    genres: str
    director: str
    release_year: int
    vote_average: float
    similarity_score: float
    reason: str

class ContentRecommendationResponse(BaseModel):
    target_movie_id: int
    strategy: str
    recommendations: List[ContentRecommendationItem]

class HybridRecommendationItem(BaseModel):
    movie_id: int
    title: str
    genres: str
    director: str
    release_year: int
    vote_average: float
    overview: str
    content_score: float
    collaborative_score: float
    predicted_rating: float
    hybrid_score: float
    reason: str

class RatingSubmission(BaseModel):
    user_id: int = Field(..., gt=0, description="Positive User ID")
    movie_id: int = Field(..., gt=0, description="Positive Movie ID")
    rating: float = Field(..., ge=0.5, le=5.0, description="Rating between 0.5 and 5.0")

class HealthResponse(BaseModel):
    status: str
    movies_count: int
    users_count: int
    ratings_count: int

# Initialize App
app = FastAPI(
    title="CineMatch Hybrid Recommendation API",
    description="High-performance recommendation API combining Content-Based TF-IDF cosine similarity and SVD collaborative filtering.",
    version="1.0.0"
)

# Global in-memory engine state
movies_df, ratings_df = get_dataset()
cb_engine, cf_engine, hybrid_engine = initialize_engines(movies_df, ratings_df)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Verify service status and current dataset scale."""
    return {
        "status": "healthy",
        "movies_count": len(movies_df),
        "users_count": int(ratings_df["user_id"].nunique()),
        "ratings_count": len(ratings_df)
    }


@app.get("/movies", response_model=List[MovieItem], tags=["Catalog"])
def list_movies(
    genre: Optional[str] = Query(None, description="Filter by genre substring, e.g. 'Sci-Fi'"),
    search: Optional[str] = Query(None, description="Search term matching title or director"),
    limit: int = Query(20, ge=1, le=100)
):
    """Retrieve movie catalog with optional genre or text search."""
    df = movies_df.copy()
    if genre:
        df = df[df["genres"].str.contains(genre, case=False, na=False)]
    if search:
        mask = (df["title"].str.contains(search, case=False, na=False)) | \
               (df["director"].str.contains(search, case=False, na=False))
        df = df[mask]
    return df.head(limit).to_dict(orient="records")


@app.get("/movies/{movie_id}", response_model=MovieItem, tags=["Catalog"])
def get_movie(movie_id: int):
    """Retrieve metadata for a specific movie ID."""
    match = movies_df[movies_df["movie_id"] == movie_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Movie ID {movie_id} not found in catalog.")
    return match.iloc[0].to_dict()


@app.get("/recommend/content/{movie_id}", response_model=ContentRecommendationResponse, tags=["Recommendations"])
def get_content_recommendations(
    movie_id: int,
    top_k: int = Query(5, ge=1, le=20)
):
    """Get TF-IDF cosine lookalikes for a target movie."""
    if movie_id not in cb_engine.movie_id_to_idx:
        raise HTTPException(status_code=404, detail=f"Movie ID {movie_id} not found in catalog.")
    recs = cb_engine.get_similar_movies(movie_id=movie_id, top_k=top_k)
    return {
        "target_movie_id": movie_id,
        "strategy": "Content-Based (TF-IDF + Cosine Similarity)",
        "recommendations": recs
    }


@app.get("/recommend/hybrid", response_model=List[HybridRecommendationItem], tags=["Recommendations"])
def get_hybrid_recommendations(
    user_id: Optional[int] = Query(None, description="Active user ID (omit for cold-start guest)"),
    seed_movie_id: Optional[int] = Query(None, description="Optional seed movie ID for context"),
    top_k: int = Query(6, ge=1, le=20),
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="1.0 = Pure Content, 0.0 = Pure Collaborative"),
    exclude_rated: bool = Query(True, description="Filter out movies already rated by user")
):
    """Generate personalized hybrid recommendations with dynamic alpha balance."""
    recs = hybrid_engine.recommend(
        user_id=user_id,
        seed_movie_id=seed_movie_id,
        top_k=top_k,
        alpha=alpha,
        exclude_rated=exclude_rated
    )
    return recs


@app.get("/evaluation", tags=["Evaluation"])
def get_evaluation():
    """Retrieve offline Precision@K, Recall@K, and NDCG@K comparison benchmarks."""
    if SUMMARY_PATH.is_file():
        import json
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("benchmark", [])
    raise HTTPException(status_code=404, detail="Evaluation benchmark has not been generated yet. Run train_and_cache.py first.")


@app.post("/ratings", status_code=status.HTTP_201_CREATED, tags=["User Actions"])
def submit_rating(payload: RatingSubmission):
    """Record a new user rating interaction and update runtime profile."""
    global ratings_df
    if payload.movie_id not in set(movies_df["movie_id"]):
        raise HTTPException(status_code=400, detail=f"Cannot rate movie_id {payload.movie_id}: not in catalog.")

    new_row = pd.DataFrame([{
        "user_id": payload.user_id,
        "movie_id": payload.movie_id,
        "rating": payload.rating,
        "timestamp": 1700000000
    }])
    ratings_df = pd.concat([ratings_df, new_row], ignore_index=True)
    hybrid_engine.set_ratings_data(ratings_df)

    return {
        "status": "success",
        "message": f"Rating {payload.rating} recorded for User {payload.user_id} on Movie {payload.movie_id}."
    }
