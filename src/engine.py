"""
Engine Factory Module.
Initializes, trains, and caches recommendation engines from loaded data.
"""

from typing import Optional, Tuple
import pandas as pd

from src.data_loader import get_dataset
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender


def initialize_engines(
    movies_df: Optional[pd.DataFrame] = None,
    ratings_df: Optional[pd.DataFrame] = None
) -> Tuple[ContentBasedRecommender, CollaborativeRecommender, HybridRecommender]:
    """
    Constructs and fits all three engines (content, collaborative, hybrid).
    """
    if movies_df is None or ratings_df is None:
        movies_df, ratings_df = get_dataset()

    cb_engine = ContentBasedRecommender()
    cb_engine.fit(movies_df)

    cf_engine = CollaborativeRecommender(n_factors=4, random_state=42)
    cf_engine.fit(ratings_df, all_movie_ids=movies_df["movie_id"].tolist())

    hybrid_engine = HybridRecommender(content_engine=cb_engine, cf_engine=cf_engine)
    hybrid_engine.set_ratings_data(ratings_df)

    return cb_engine, cf_engine, hybrid_engine
