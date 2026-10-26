"""
Train and Initialization Pipeline Script.
Fits content-based, collaborative, and hybrid recommendation engines,
executes leak-free offline evaluation benchmarking, and saves artifacts to models/.
"""

import json
from pathlib import Path
import pandas as pd
from src.data_loader import get_dataset
from src.engine import initialize_engines
from src.evaluation import run_evaluation_benchmark

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"


def run_pipeline():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("STEP 1: Loading Movie Catalog and Rating Interactions from CSV...")
    print("=" * 65)
    movies_df, ratings_df = get_dataset()
    print(f"Loaded {len(movies_df)} movies and {len(ratings_df)} user ratings across {ratings_df['user_id'].nunique()} users.")

    print("\n" + "=" * 65)
    print("STEP 2: Initializing and Fitting Recommender Engines...")
    print("=" * 65)
    cb_engine, cf_engine, hybrid_engine = initialize_engines(movies_df, ratings_df)
    print("Fitted TF-IDF Content Engine, SVD Collaborative Engine, and Hybrid Blending Engine.")

    print("\n" + "=" * 65)
    print("STEP 3: Running Leak-Free Offline Evaluation Benchmark...")
    print("=" * 65)
    benchmark_df = run_evaluation_benchmark(
        movies_df=movies_df,
        ratings_df=ratings_df,
        test_size=0.25,
        random_state=42,
        k_values=(5, 10)
    )
    print("\nMeasured Offline Recommendation Benchmark:")
    print(benchmark_df.to_string(index=False))

    # Save benchmark CSV
    benchmark_csv_path = MODELS_DIR / "evaluation_benchmark.csv"
    benchmark_df.to_csv(benchmark_csv_path, index=False)
    print(f"\nSaved benchmark metrics to {benchmark_csv_path}")

    # Save summary JSON
    summary = {
        "movies_count": len(movies_df),
        "ratings_count": len(ratings_df),
        "users_count": int(ratings_df["user_id"].nunique()),
        "benchmark": benchmark_df.to_dict(orient="records")
    }
    summary_path = MODELS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved pipeline summary to {summary_path}")

    return benchmark_df


if __name__ == "__main__":
    run_pipeline()
