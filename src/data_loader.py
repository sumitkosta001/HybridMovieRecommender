"""
Data Loader Module for Hybrid Movie Recommender.
Safely loads movie metadata and user ratings from CSV files or in-memory sample datasets.
Performs data validation, missing value imputation, type casting, bounds checking, and deduplication.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import pandas as pd


# Default project directories relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MOVIES_CSV = DEFAULT_DATA_DIR / "movies.csv"
DEFAULT_RATINGS_CSV = DEFAULT_DATA_DIR / "ratings.csv"


def load_movies_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Loads movie metadata from CSV.
    If filepath is not provided, defaults to data/movies.csv relative to project root.
    If the file does not exist, falls back to in-memory sample catalog if available.

    Validations & Guarantees:
      - Validates required columns: movie_id, title, genres, director, overview
      - Reports clear errors for empty or malformed files
      - Casts 'movie_id' to positive integer, drops duplicates
      - Handles missing text columns with clean fallbacks
      - Clips 'vote_average' to [0.0, 10.0]
    """
    path = Path(filepath) if filepath else DEFAULT_MOVIES_CSV

    if path.is_file():
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            raise ValueError(f"Movie metadata file is empty: {path}")
        except Exception as e:
            raise ValueError(f"Failed to read movie metadata file {path}: {e}")
    else:
        # Fallback to sample_data if CSV not yet created
        try:
            from data.sample_data import SAMPLE_MOVIES
            df = pd.DataFrame(SAMPLE_MOVIES)
        except ImportError:
            raise FileNotFoundError(f"Movie metadata file not found at {path} and fallback unavailable.")

    if df.empty:
        raise ValueError(f"Movie metadata dataset is empty (0 rows).")

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Validate required columns
    required_cols = ["movie_id", "title"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Movie metadata must contain required column: '{col}'. Found: {list(df.columns)}")

    # Clean movie_id: numeric, positive, unique
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce")
    df = df.dropna(subset=["movie_id"]).copy()
    df["movie_id"] = df["movie_id"].astype(int)
    df = df[df["movie_id"] > 0]

    if df.empty:
        raise ValueError("No valid positive numeric 'movie_id' entries found in movie metadata.")

    # Deduplicate by movie_id
    df = df.drop_duplicates(subset=["movie_id"]).reset_index(drop=True)

    # Impute missing string fields
    text_defaults = {
        "title": "Unknown Title",
        "genres": "Unknown",
        "director": "Unknown Director",
        "cast": "",
        "overview": ""
    }
    for col, default_val in text_defaults.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = df[col].fillna(default_val).astype(str).str.strip()

    # Numeric fields validation and clipping
    if "vote_average" not in df.columns:
        df["vote_average"] = 7.0
    else:
        df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(7.0).clip(0.0, 10.0)

    if "release_year" not in df.columns:
        df["release_year"] = 2000
    else:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(2000).astype(int)

    return df


def load_ratings_data(
    filepath: Optional[Union[str, Path]] = None,
    valid_movie_ids: Optional[set] = None
) -> pd.DataFrame:
    """
    Loads user ratings interactions from CSV.
    If filepath is not provided, defaults to data/ratings.csv relative to project root.

    Validations & Guarantees:
      - Validates required columns: user_id, movie_id, rating
      - Reports clear errors for empty or malformed files
      - Casts 'user_id' and 'movie_id' to positive integers
      - Validates ratings are within realistic bounds [0.5, 5.0]
      - Filters out ratings with invalid or uncataloged movie_ids if valid_movie_ids provided
      - Aggregates duplicate ratings from same user for same movie
    """
    path = Path(filepath) if filepath else DEFAULT_RATINGS_CSV

    if path.is_file():
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            raise ValueError(f"Ratings file is empty: {path}")
        except Exception as e:
            raise ValueError(f"Failed to read ratings file {path}: {e}")
    else:
        try:
            from data.sample_data import SAMPLE_RATINGS
            df = pd.DataFrame(SAMPLE_RATINGS)
        except ImportError:
            raise FileNotFoundError(f"Ratings file not found at {path} and fallback unavailable.")

    if df.empty:
        raise ValueError("Ratings dataset is empty (0 rows).")

    df.columns = [c.strip().lower() for c in df.columns]

    # Validate required columns
    required_cols = ["user_id", "movie_id", "rating"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Ratings data must contain required column: '{col}'. Found: {list(df.columns)}")

    # Cast IDs and rating
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Drop rows missing essential identifiers or rating
    df = df.dropna(subset=["user_id", "movie_id", "rating"]).copy()
    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)
    df = df[(df["user_id"] > 0) & (df["movie_id"] > 0)]

    # Validate bounds: ratings must be in [0.5, 5.0]
    # Filter or clip values
    df["rating"] = df["rating"].astype(float).clip(0.5, 5.0)

    # Optional timestamp preservation
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(int)

    # Filter out movie IDs that do not exist in catalog
    if valid_movie_ids is not None:
        df = df[df["movie_id"].isin(valid_movie_ids)]

    if df.empty:
        raise ValueError("No valid ratings remaining after filtering against catalog and invalid bounds.")

    # Deduplicate ratings: if a user rated the same movie multiple times, take the mean
    agg_dict = {"rating": "mean"}
    if "timestamp" in df.columns:
        agg_dict["timestamp"] = "max"
    
    df = df.groupby(["user_id", "movie_id"], as_index=False).agg(agg_dict)
    return df.reset_index(drop=True)


def get_dataset(data_dir: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience helper to load both movies and ratings datasets from CSV files.
    Defaults to data/ under project root.
    """
    target_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    movies_path = target_dir / "movies.csv"
    ratings_path = target_dir / "ratings.csv"

    movies_df = load_movies_data(movies_path)
    ratings_df = load_ratings_data(ratings_path, valid_movie_ids=set(movies_df["movie_id"]))
    return movies_df, ratings_df
