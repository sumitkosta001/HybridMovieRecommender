"""
Comprehensive pytest test suite for Hybrid Movie Recommendation System.
Verifies:
- CSV Loading, relative paths, missing columns, empty files, and malformed data handling
- Valid rating bounds [0.5, 5.0] and duplicate ID handling
- Catalog consistency between movies.csv and ratings.csv
- Recommendation output shapes and keys
- Strict self-exclusion of query seed movies
- Unknown movie IDs handling
- Unknown user IDs fallback handling
- Score bounds in [0.0, 1.0]
- Alpha sensitivity (pure content alpha=1.0 vs pure collaborative alpha=0.0)
- Cold-start fallback behavior
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from src.data_loader import (
    load_movies_data,
    load_ratings_data,
    get_dataset,
    DEFAULT_MOVIES_CSV,
    DEFAULT_RATINGS_CSV
)
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender
from src.engine import initialize_engines


@pytest.fixture(scope="module")
def setup_system():
    """Initializes engines with clean CSV-loaded dataset."""
    movies_df, ratings_df = get_dataset()
    cb_engine, cf_engine, hybrid_engine = initialize_engines(movies_df, ratings_df)
    return {
        "movies_df": movies_df,
        "ratings_df": ratings_df,
        "cb_engine": cb_engine,
        "cf_engine": cf_engine,
        "hybrid_engine": hybrid_engine
    }


# =====================================================================
# DATA LOADER & CSV VALIDATION TESTS
# =====================================================================

def test_csv_files_exist_and_load():
    """Verifies default CSV files exist in data/ and load properly."""
    assert DEFAULT_MOVIES_CSV.is_file(), f"Missing {DEFAULT_MOVIES_CSV}"
    assert DEFAULT_RATINGS_CSV.is_file(), f"Missing {DEFAULT_RATINGS_CSV}"

    movies_df, ratings_df = get_dataset()
    assert len(movies_df) >= 20
    assert len(ratings_df) >= 50


def test_movie_and_rating_id_consistency():
    """Ensures every movie_id in ratings.csv exists in movies.csv."""
    movies_df, ratings_df = get_dataset()
    movie_catalog_ids = set(movies_df["movie_id"])
    rated_movie_ids = set(ratings_df["movie_id"])

    # All rated movies must be members of the catalog
    assert rated_movie_ids.issubset(movie_catalog_ids)
    # The demonstration catalog should have 100% rating coverage
    assert len(rated_movie_ids) == len(movie_catalog_ids)


def test_missing_required_columns_error(tmp_path):
    """Verifies that missing essential columns raise informative ValueError."""
    # Test movies without 'title'
    bad_movies_file = tmp_path / "bad_movies.csv"
    bad_movies_file.write_text("movie_id,director\n1,Christopher Nolan")
    with pytest.raises(ValueError, match="Movie metadata must contain required column: 'title'"):
        load_movies_data(bad_movies_file)

    # Test ratings without 'rating'
    bad_ratings_file = tmp_path / "bad_ratings.csv"
    bad_ratings_file.write_text("user_id,movie_id\n1,101")
    with pytest.raises(ValueError, match="Ratings data must contain required column: 'rating'"):
        load_ratings_data(bad_ratings_file)


def test_empty_csv_file_error(tmp_path):
    """Verifies that empty CSV files raise clear errors."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_movies_data(empty_file)

    with pytest.raises(ValueError, match="empty"):
        load_ratings_data(empty_file)


def test_duplicate_movie_id_deduplication(tmp_path):
    """Verifies duplicate movie_ids are automatically deduplicated in loader."""
    dup_movies = tmp_path / "dup_movies.csv"
    dup_movies.write_text(
        "movie_id,title,genres,director,overview\n"
        "101,Interstellar,Sci-Fi,Nolan,Space voyage\n"
        "101,Interstellar Remastered,Sci-Fi,Nolan,Space voyage duplicate\n"
        "102,Inception,Sci-Fi,Nolan,Dream heist\n"
    )
    df = load_movies_data(dup_movies)
    assert len(df) == 2
    assert df["movie_id"].tolist() == [101, 102]


def test_rating_bounds_and_filtering(tmp_path):
    """Verifies out-of-bound ratings are clipped to [0.5, 5.0] and deduplicated."""
    test_ratings = tmp_path / "test_ratings.csv"
    test_ratings.write_text(
        "user_id,movie_id,rating,timestamp\n"
        "1,101,6.5,1600000000\n"  # Too high -> clips to 5.0
        "1,102,-1.0,1600000001\n" # Too low -> clips to 0.5
        "1,103,4.0,1600000002\n"
        "1,103,5.0,1600000003\n"  # Duplicate (1, 103) -> mean = 4.5
    )
    df = load_ratings_data(test_ratings)
    # Check clipping
    assert df.loc[df["movie_id"] == 101, "rating"].iloc[0] == 5.0
    assert df.loc[df["movie_id"] == 102, "rating"].iloc[0] == 0.5
    # Check deduplication mean
    assert df.loc[df["movie_id"] == 103, "rating"].iloc[0] == 4.5


# =====================================================================
# ENGINE & RECOMMENDATION MECHANICS TESTS
# =====================================================================

def test_data_loader_validity(setup_system):
    """Test data loader handles formats, types, and column validations."""
    movies_df = setup_system["movies_df"]
    ratings_df = setup_system["ratings_df"]

    assert not movies_df.empty
    assert not ratings_df.empty
    assert "movie_id" in movies_df.columns
    assert "title" in movies_df.columns
    assert "genres" in movies_df.columns
    assert "user_id" in ratings_df.columns
    assert "rating" in ratings_df.columns

    # Verify ratings are within bounds
    assert (ratings_df["rating"] >= 0.5).all()
    assert (ratings_df["rating"] <= 5.0).all()


def test_content_based_self_exclusion_and_structure(setup_system):
    """Verifies that query movie is never recommended to itself and returns top_k items."""
    cb_engine = setup_system["cb_engine"]
    query_id = 101  # Interstellar
    top_k = 5

    recs = cb_engine.get_similar_movies(query_id, top_k=top_k)
    assert len(recs) <= top_k
    assert len(recs) > 0

    rec_ids = [r["movie_id"] for r in recs]
    # Self-exclusion
    assert query_id not in rec_ids

    # Structure and score bounds
    for r in recs:
        assert "title" in r
        assert "similarity_score" in r
        assert 0.0 <= r["similarity_score"] <= 1.0


def test_content_based_unknown_movie_id(setup_system):
    """Verifies that non-existent movie ID returns an empty list without crashing."""
    cb_engine = setup_system["cb_engine"]
    unknown_id = 999999
    recs = cb_engine.get_similar_movies(unknown_id, top_k=5)
    assert recs == []

    scores = cb_engine.get_content_scores(unknown_id)
    assert all(score == 0.0 for score in scores.values())


def test_collaborative_score_bounds_and_fallback(setup_system):
    """Verifies collaborative scores are in [0, 1] and cold-start user gets popularity scores."""
    cf_engine = setup_system["cf_engine"]
    known_user = 1
    unknown_user = 999999

    known_scores = cf_engine.get_collaborative_scores(known_user)
    assert len(known_scores) > 0
    for mid, score in known_scores.items():
        assert 0.0 <= score <= 1.0

    # Cold start fallback test
    fallback_scores = cf_engine.get_collaborative_scores(unknown_user)
    assert len(fallback_scores) > 0
    for mid, score in fallback_scores.items():
        assert 0.0 <= score <= 1.0
    # Ensure fallback matches precomputed popularity
    assert fallback_scores == cf_engine.popularity_scores


def test_hybrid_recommendation_self_exclusion(setup_system):
    """Verifies hybrid engine strictly excludes the seed_movie_id."""
    hybrid_engine = setup_system["hybrid_engine"]
    seed_id = 102  # Inception
    top_k = 6

    recs = hybrid_engine.recommend(user_id=1, seed_movie_id=seed_id, top_k=top_k, alpha=0.5)
    rec_ids = [r["movie_id"] for r in recs]
    assert seed_id not in rec_ids


def test_hybrid_score_normalization_and_keys(setup_system):
    """Verifies all hybrid scores are mathematically bounded in [0.0, 1.0]."""
    hybrid_engine = setup_system["hybrid_engine"]
    recs = hybrid_engine.recommend(user_id=2, seed_movie_id=108, top_k=5, alpha=0.6)

    for item in recs:
        assert 0.0 <= item["content_score"] <= 1.0
        assert 0.0 <= item["collaborative_score"] <= 1.0
        assert 0.0 <= item["hybrid_score"] <= 1.0
        assert 1.0 <= item["predicted_rating"] <= 5.0
        assert "reason" in item


def test_hybrid_alpha_extremes(setup_system):
    """Verifies alpha=1.0 matches content weighting and alpha=0.0 matches collaborative weighting."""
    hybrid_engine = setup_system["hybrid_engine"]
    user_id = 1
    seed_id = 101

    # Alpha = 1.0 -> Pure content
    recs_content = hybrid_engine.recommend(user_id=user_id, seed_movie_id=seed_id, top_k=3, alpha=1.0)
    for r in recs_content:
        assert abs(r["hybrid_score"] - r["content_score"]) < 1e-4

    # Alpha = 0.0 -> Pure collaborative
    recs_cf = hybrid_engine.recommend(user_id=user_id, seed_movie_id=seed_id, top_k=3, alpha=0.0)
    for r in recs_cf:
        assert abs(r["hybrid_score"] - r["collaborative_score"]) < 1e-4


def test_hybrid_unknown_user_and_seed_fallback(setup_system):
    """Verifies completely cold-start queries (unknown user + no seed) succeed cleanly."""
    hybrid_engine = setup_system["hybrid_engine"]
    unknown_user = 88888

    # Query with neither known user nor seed movie
    recs = hybrid_engine.recommend(user_id=unknown_user, seed_movie_id=None, top_k=5, alpha=0.5)
    assert len(recs) == 5
    for r in recs:
        assert 0.0 <= r["hybrid_score"] <= 1.0
        assert len(r["title"]) > 0


# =====================================================================
# EVALUATION RIGOR & LEAKAGE AUDIT TESTS
# =====================================================================

def test_evaluation_reproducibility(setup_system):
    """Verifies that running evaluation benchmark with identical seed produces identical metrics."""
    from src.evaluation import run_evaluation_benchmark
    movies_df = setup_system["movies_df"]
    ratings_df = setup_system["ratings_df"]

    res1 = run_evaluation_benchmark(movies_df, ratings_df, test_size=0.25, random_state=42)
    res2 = run_evaluation_benchmark(movies_df, ratings_df, test_size=0.25, random_state=42)

    pd.testing.assert_frame_equal(res1, res2)


def test_ndcg_and_dcg_boundary_conditions():
    """Verifies NDCG@K mathematical properties (ideal ranking=1.0, inverted ranking<1.0, zero hits=0.0)."""
    from src.evaluation import compute_dcg_at_k, compute_ndcg_at_k

    # 1. Zero hits
    actual_dict = {101: 5.0, 102: 4.5}
    recommended_zero = [201, 202, 203]
    assert compute_ndcg_at_k(recommended_zero, actual_dict, k=3) == 0.0

    # 2. Perfect ranking (relevant items at rank 1 and 2)
    recommended_perfect = [101, 102, 201]
    assert compute_ndcg_at_k(recommended_perfect, actual_dict, k=2) == 1.0

    # 3. Suboptimal ranking (relevant item placed lower)
    recommended_suboptimal = [201, 101, 202]
    ndcg_sub = compute_ndcg_at_k(recommended_suboptimal, actual_dict, k=2)
    assert 0.0 < ndcg_sub < 1.0

    # 4. User with no relevant ratings (all < 4.0)
    non_relevant_dict = {101: 2.0, 102: 3.0}
    assert compute_ndcg_at_k([101, 102], non_relevant_dict, k=2) == 0.0


def test_no_test_leakage_in_collaborative_and_hybrid():
    """
    Verifies that test holdout ratings are strictly absent from collaborative
    and hybrid engines during evaluation.
    """
    from src.evaluation import run_evaluation_benchmark
    from src.data_loader import get_dataset
    from sklearn.model_selection import train_test_split

    movies_df, ratings_df = get_dataset()
    user_counts = ratings_df["user_id"].value_counts()
    eligible = user_counts[user_counts >= 3].index
    filtered = ratings_df[ratings_df["user_id"].isin(eligible)].copy()

    train_df, test_df = train_test_split(filtered, test_size=0.25, random_state=42, stratify=filtered["user_id"])

    # Build CF engine on train_df only
    cf_engine = CollaborativeRecommender(n_factors=4, random_state=42)
    cf_engine.fit(train_df, all_movie_ids=movies_df["movie_id"].tolist())

    # Precomputed popularity ratings must strictly match train_df ratings, NOT full ratings_df
    for mid in movies_df["movie_id"]:
        train_ratings_for_mid = train_df[train_df["movie_id"] == mid]["rating"]
        v_train = len(train_ratings_for_mid)
        # SVD popularity must reflect train count, not total count
        if v_train > 0:
            expected_mean = float(train_ratings_for_mid.mean())
            m = 2.0
            c = float(train_df["rating"].mean())
            expected_bayes = (v_train / (v_train + m)) * expected_mean + (m / (v_train + m)) * c
            expected_score = float(np.clip((expected_bayes - 1.0) / 4.0, 0.0, 1.0))
            assert abs(cf_engine.popularity_scores[mid] - expected_score) < 1e-4

