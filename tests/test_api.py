"""
Pytest integration tests for FastAPI Recommendation Service (api.py).
Verifies:
- /health endpoint returns dataset counts
- /movies catalog listing, search, and 404 behavior
- /recommend/content/{movie_id} returns valid lookalikes with self-exclusion
- /recommend/hybrid returns correct schema, bound clamping, and alpha blending
- /ratings records new ratings and validates input bounds
- /evaluation returns benchmark metrics
"""

import pytest
from fastapi.testclient import TestClient
from api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["movies_count"] >= 20
    assert data["ratings_count"] >= 50


def test_api_list_movies_and_search(client):
    # List all
    res = client.get("/movies")
    assert res.status_code == 200
    movies = res.json()
    assert len(movies) > 0

    # Search filter
    res_search = client.get("/movies?search=Nolan")
    assert res_search.status_code == 200
    nolan_movies = res_search.json()
    assert len(nolan_movies) >= 2
    assert all("Nolan" in m["director"] for m in nolan_movies)


def test_api_get_movie_by_id(client):
    res = client.get("/movies/101")
    assert res.status_code == 200
    movie = res.json()
    assert movie["movie_id"] == 101
    assert movie["title"] == "Interstellar"

    # Non-existent movie
    res_404 = client.get("/movies/999999")
    assert res_404.status_code == 404


def test_api_content_recommendations(client):
    res = client.get("/recommend/content/101?top_k=4")
    assert res.status_code == 200
    data = res.json()
    assert data["target_movie_id"] == 101
    recs = data["recommendations"]
    assert len(recs) <= 4
    # Query movie is excluded from recommendations
    assert 101 not in [r["movie_id"] for r in recs]


def test_api_hybrid_recommendations(client):
    # User 1 with alpha 0.6
    res = client.get("/recommend/hybrid?user_id=1&top_k=5&alpha=0.6")
    assert res.status_code == 200
    recs = res.json()
    assert len(recs) == 5
    for r in recs:
        assert 0.0 <= r["hybrid_score"] <= 1.0
        assert 0.0 <= r["content_score"] <= 1.0
        assert 0.0 <= r["collaborative_score"] <= 1.0
        assert "reason" in r

    # Cold-start guest user
    res_guest = client.get("/recommend/hybrid?top_k=3")
    assert res_guest.status_code == 200
    assert len(res_guest.json()) == 3


def test_api_evaluation_benchmark(client):
    res = client.get("/evaluation")
    assert res.status_code == 200
    benchmarks = res.json()
    assert len(benchmarks) >= 3
    variant_names = [b["Model Variant"] for b in benchmarks]
    assert any("Hybrid" in v for v in variant_names)


def test_api_submit_rating(client):
    payload = {
        "user_id": 99,
        "movie_id": 101,
        "rating": 4.5
    }
    res = client.post("/ratings", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["status"] == "success"

    # Invalid rating (> 5.0)
    bad_payload = {
        "user_id": 99,
        "movie_id": 101,
        "rating": 6.5
    }
    res_bad = client.post("/ratings", json=bad_payload)
    assert res_bad.status_code == 422
