"""
Data Generation Script.
Exports `movies.csv` and generates a synthetic `ratings.csv` with multiple user profiles
and varied ratings across all movies for meaningful SVD matrix factorization demonstration.
"""

from pathlib import Path
import random
import pandas as pd
from data.sample_data import SAMPLE_MOVIES

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def generate_csv_datasets(seed: int = 42):
    random.seed(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Export movies.csv
    movies_df = pd.DataFrame(SAMPLE_MOVIES)
    expected_cols = [
        "movie_id", "title", "genres", "director", "cast",
        "release_year", "vote_average", "overview"
    ]
    movies_df = movies_df[expected_cols]
    movies_path = DATA_DIR / "movies.csv"
    movies_df.to_csv(movies_path, index=False)
    print(f"Exported {len(movies_df)} movies to {movies_path}")

    # 2. Generate Synthetic Ratings with Diverse Archetypes
    # Note: These ratings are synthetic demonstrations and not real-world telemetry.
    # We define 10 user archetypes with affinities to specific genres/directors,
    # ensuring every movie receives multiple varied ratings so Truncated SVD runs stably.
    
    movie_lookup = {m["movie_id"]: m for m in SAMPLE_MOVIES}
    movie_ids = [m["movie_id"] for m in SAMPLE_MOVIES]

    user_profiles = [
        # User 1: Christopher Nolan & Sci-Fi Aficionado
        {"user_id": 1, "pref_directors": ["Christopher Nolan", "Denis Villeneuve"], "pref_genres": ["Sci-Fi", "Thriller"], "base_rating": 4.5},
        # User 2: Quentin Tarantino & Crime Aficionado
        {"user_id": 2, "pref_directors": ["Quentin Tarantino", "Francis Ford Coppola", "Martin Scorsese"], "pref_genres": ["Crime", "Drama"], "base_rating": 4.5},
        # User 3: Animation & Fantasy Art Lover
        {"user_id": 3, "pref_directors": ["Hayao Miyazaki", "Bob Persichetti Peter Ramsey"], "pref_genres": ["Animation", "Fantasy"], "base_rating": 4.5},
        # User 4: Dramatic Arts, Music & Romance Lover
        {"user_id": 4, "pref_directors": ["Damien Chazelle", "Bong Joon Ho"], "pref_genres": ["Drama", "Music", "Romance"], "base_rating": 4.5},
        # User 5: Wes Anderson & Whimsical Comedy Lover
        {"user_id": 5, "pref_directors": ["Wes Anderson", "Bong Joon Ho"], "pref_genres": ["Comedy", "Adventure"], "base_rating": 4.5},
        # User 6: Sci-Fi & Action Fanatic
        {"user_id": 6, "pref_directors": ["Denis Villeneuve", "Lana Wachowski Lilly Wachowski"], "pref_genres": ["Sci-Fi", "Action"], "base_rating": 4.0},
        # User 7: Classic Cinema & Mob Saga Devotee
        {"user_id": 7, "pref_directors": ["Francis Ford Coppola", "Martin Scorsese", "Quentin Tarantino"], "pref_genres": ["Crime", "Biography"], "base_rating": 4.5},
        # User 8: Highbrow Thriller & Dark Action Critic (Hard to please)
        {"user_id": 8, "pref_directors": ["Christopher Nolan", "Bong Joon Ho"], "pref_genres": ["Thriller", "Mystery"], "base_rating": 3.5},
        # User 9: Mainstream Crowd Pleaser (rates most top movies high)
        {"user_id": 9, "pref_directors": ["Christopher Nolan", "Denis Villeneuve", "Hayao Miyazaki"], "pref_genres": ["Adventure", "Action"], "base_rating": 4.0},
        # User 10: Indie & Art-House Enthusiast
        {"user_id": 10, "pref_directors": ["Wes Anderson", "Damien Chazelle"], "pref_genres": ["Music", "Romance", "Comedy"], "base_rating": 4.0},
        # Users 11-16: Diverse casual viewers
        {"user_id": 11, "pref_directors": ["Christopher Nolan"], "pref_genres": ["Sci-Fi"], "base_rating": 3.5},
        {"user_id": 12, "pref_directors": ["Quentin Tarantino"], "pref_genres": ["Action", "Crime"], "base_rating": 4.0},
        {"user_id": 13, "pref_directors": ["Hayao Miyazaki"], "pref_genres": ["Animation"], "base_rating": 4.0},
        {"user_id": 14, "pref_directors": ["Denis Villeneuve"], "pref_genres": ["Sci-Fi", "Adventure"], "base_rating": 4.0},
        {"user_id": 15, "pref_directors": ["Francis Ford Coppola"], "pref_genres": ["Drama"], "base_rating": 3.5},
        {"user_id": 16, "pref_directors": ["Damien Chazelle"], "pref_genres": ["Drama"], "base_rating": 4.0},
    ]

    ratings_records = []
    base_timestamp = 1672531199  # 2023-01-01 epoch

    for profile in user_profiles:
        uid = profile["user_id"]
        pref_dirs = profile["pref_directors"]
        pref_genres = profile["pref_genres"]

        # Each user rates between 8 and 16 movies
        num_ratings = random.randint(8, 16)
        # Weight movie selection toward preferred genres/directors
        weighted_mids = []
        for mid in movie_ids:
            m = movie_lookup[mid]
            weight = 1
            if any(d in m["director"] for d in pref_dirs):
                weight += 4
            if any(g in m["genres"] for g in pref_genres):
                weight += 3
            weighted_mids.extend([mid] * weight)

        selected_mids = set()
        attempts = 0
        while len(selected_mids) < num_ratings and attempts < 100:
            selected_mids.add(random.choice(weighted_mids))
            attempts += 1

        for mid in selected_mids:
            m = movie_lookup[mid]
            is_fav_dir = any(d in m["director"] for d in pref_dirs)
            is_fav_genre = any(g in m["genres"] for g in pref_genres)

            if is_fav_dir and is_fav_genre:
                rating = random.choice([4.5, 5.0])
            elif is_fav_dir or is_fav_genre:
                rating = random.choice([3.5, 4.0, 4.5, 5.0])
            else:
                rating = random.choice([1.5, 2.0, 2.5, 3.0, 3.5])

            ratings_records.append({
                "user_id": int(uid),
                "movie_id": int(mid),
                "rating": float(rating),
                "timestamp": int(base_timestamp + random.randint(1000, 31536000))
            })

    ratings_df = pd.DataFrame(ratings_records)
    # Ensure no duplicate (user_id, movie_id)
    ratings_df = ratings_df.drop_duplicates(subset=["user_id", "movie_id"]).sort_values(by=["user_id", "movie_id"]).reset_index(drop=True)
    
    ratings_path = DATA_DIR / "ratings.csv"
    ratings_df.to_csv(ratings_path, index=False)
    print(f"Exported {len(ratings_df)} synthetic ratings across {ratings_df['user_id'].nunique()} users to {ratings_path}")

    # Verify coverage
    rated_movies = set(ratings_df["movie_id"])
    all_mids = set(movies_df["movie_id"])
    missing_coverage = all_mids - rated_movies
    print(f"Movie catalog coverage: {len(rated_movies)}/{len(all_mids)} movies rated.")
    if missing_coverage:
        print(f"Warning: movies not rated: {missing_coverage}")

if __name__ == "__main__":
    generate_csv_datasets()
