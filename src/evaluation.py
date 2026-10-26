"""
Recommendation System Quality Evaluation Module.
Evaluates model variants (Content-Based, Collaborative Filtering, and Hybrid)
using Precision@K, Recall@K, and NDCG@K metrics on a holdout test split.

Protocol Guarantees:
- Valid train/test split of user ratings: test ratings are STRICTLY isolated from training data.
- User profiles in training only use train_ratings_df.
- Relevant items in test ground truth defined as ratings >= 4.0.
- Safe handling of zero relevance / zero DCG cases.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender


def compute_dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank K using binary or graded relevance."""
    rel = np.asarray(relevance_scores, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    # Standard formula: sum((2^rel - 1) / log2(rank + 1))
    discounts = np.log2(np.arange(2, rel.size + 2))
    return float(np.sum((2.0 ** rel - 1.0) / discounts))


def compute_ndcg_at_k(recommended_ids: List[int], actual_ratings_dict: Dict[int, float], k: int = 5) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at rank K.
    Binary relevance: 1.0 if actual rating >= 4.0, else 0.0.
    IDCG is computed over the top-min(k, total_relevant) ideal ranking.
    """
    top_k = recommended_ids[:k]
    # Actual gains along the recommended sequence
    actual_gains = [1.0 if actual_ratings_dict.get(mid, 0.0) >= 4.0 else 0.0 for mid in top_k]
    dcg = compute_dcg_at_k(actual_gains, k)

    # Ideal gains: up to k relevant items (gains = 1.0)
    relevant_count = sum(1 for r in actual_ratings_dict.values() if r >= 4.0)
    if relevant_count == 0:
        return 0.0
    ideal_gains = [1.0] * min(k, relevant_count)
    idcg = compute_dcg_at_k(ideal_gains, k)

    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def run_evaluation_benchmark(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    k_values: Tuple[int, ...] = (5, 10)
) -> pd.DataFrame:
    """
    Trains models strictly on train split and benchmarks ranking metrics on test holdout.
    Compares:
      1. Popularity Baseline (pure prior community ranking)
      2. Content-Based Filtering (alpha=1.0)
      3. Collaborative Filtering (alpha=0.0)
      4. Hybrid Recommender (alpha=0.5)
    Returns comparison summary DataFrame across all variants.
    """
    # 1. Train / Test Split
    # Filter users with at least 3 ratings to ensure both train and test have presence
    user_counts = ratings_df["user_id"].value_counts()
    eligible_users = user_counts[user_counts >= 3].index
    filtered_ratings = ratings_df[ratings_df["user_id"].isin(eligible_users)].copy()

    train_ratings, test_ratings = train_test_split(
        filtered_ratings,
        test_size=test_size,
        random_state=random_state,
        stratify=filtered_ratings["user_id"]
    )

    # 2. Fit engines strictly on train_ratings
    cb_engine = ContentBasedRecommender()
    cb_engine.fit(movies_df)

    cf_engine = CollaborativeRecommender(n_factors=4, random_state=random_state)
    cf_engine.fit(train_ratings, all_movie_ids=movies_df["movie_id"].tolist())

    hybrid_engine = HybridRecommender(content_engine=cb_engine, cf_engine=cf_engine)
    # Crucial: set ONLY train ratings into hybrid engine so test set is not leaked
    hybrid_engine.set_ratings_data(train_ratings)

    # Precompute popularity ranked movie_ids strictly from train ratings
    pop_sorted_mids = [
        mid for mid, score in sorted(
            cf_engine.popularity_scores.items(), key=lambda x: x[1], reverse=True
        )
    ]

    # 3. Model Variants to Compare
    variants = {
        "Popularity Baseline": None,
        "Content-Based (alpha=1.0)": 1.0,
        "Collaborative Filtering (alpha=0.0)": 0.0,
        "Hybrid Recommender (alpha=0.5)": 0.5
    }

    test_users = sorted(test_ratings["user_id"].unique().tolist())
    max_k = max(k_values)

    summary_rows = []

    for var_name, alpha in variants.items():
        metrics_accum = {f"P@{k}": [] for k in k_values}
        metrics_accum.update({f"R@{k}": [] for k in k_values})
        metrics_accum.update({f"NDCG@{k}": [] for k in k_values})

        for uid in test_users:
            u_test = test_ratings[test_ratings["user_id"] == uid]
            relevant_items = set(u_test[u_test["rating"] >= 4.0]["movie_id"].tolist())
            actual_dict = dict(zip(u_test["movie_id"], u_test["rating"]))

            # Clean evaluation standard: users without relevant items in test holdout are skipped
            if not relevant_items:
                continue

            if var_name == "Popularity Baseline":
                rec_ids = pop_sorted_mids[:max_k]
            else:
                recs = hybrid_engine.recommend(
                    user_id=uid,
                    top_k=max_k,
                    alpha=alpha,
                    exclude_rated=False
                )
                rec_ids = [r["movie_id"] for r in recs]

            for k in k_values:
                top_k_ids = rec_ids[:k]
                hits = len(set(top_k_ids).intersection(relevant_items))

                precision = hits / float(k)
                recall = hits / float(len(relevant_items))
                ndcg = compute_ndcg_at_k(top_k_ids, actual_dict, k=k)

                metrics_accum[f"P@{k}"].append(precision)
                metrics_accum[f"R@{k}"].append(recall)
                metrics_accum[f"NDCG@{k}"].append(ndcg)

        row = {"Model Variant": var_name}
        for metric_name, val_list in metrics_accum.items():
            row[metric_name] = round(float(np.mean(val_list)), 4) if val_list else 0.0
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)
