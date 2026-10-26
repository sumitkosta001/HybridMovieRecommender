"""
Interactive Streamlit Dashboard for Hybrid Movie Recommendation System.
Features:
- Live Hybrid Recommendations with alpha dynamic blending slider
- Interactive Movie Catalog Explorer & Title Search
- Content-Based 'More Like This' similarity discovery
- Glassmorphic, modern responsive UI design
"""

import streamlit as st
import pandas as pd
from src.engine import initialize_engines
from src.data_loader import get_dataset

st.set_page_config(
    page_title="CineMatch | Hybrid Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(79, 70, 229, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        font-weight: 700;
        font-size: 2.4rem;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #ffffff, #a5b4fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin: 0;
    }

    .movie-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.3rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1.2rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }

    .movie-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px -5px rgba(99, 102, 241, 0.15);
        border-color: #c7d2fe;
    }

    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.3rem;
    }

    .genre-tag {
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.3rem;
        margin-bottom: 0.4rem;
    }

    .badge-score {
        display: inline-flex;
        align-items: center;
        padding: 0.3rem 0.7rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.88rem;
    }

    .badge-hybrid {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
    }

    .badge-cf {
        background: #f1f5f9;
        color: #334155;
    }

    .badge-content {
        background: #fdf2f8;
        color: #be185d;
    }

    .reason-box {
        background-color: #f8fafc;
        border-left: 3px solid #6366f1;
        padding: 0.5rem 0.8rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: #475569;
        margin-top: 0.7rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading recommendation engines...")
def load_app_engines():
    movies_df, ratings_df = get_dataset()
    cb_engine, cf_engine, hybrid_engine = initialize_engines(movies_df, ratings_df)
    return movies_df, ratings_df, cb_engine, cf_engine, hybrid_engine


movies_df, ratings_df, cb_engine, cf_engine, hybrid_engine = load_app_engines()

# Header Banner
st.markdown("""
<div class="main-header">
    <h1>🎬 CineMatch Hybrid Recommender</h1>
    <p>Personalized discovery blending TF-IDF Content Similarity with Truncated SVD Matrix Factorization & Bayesian Popularity Fallbacks.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Recommendation Controls")
    
    available_users = sorted(ratings_df["user_id"].unique().tolist())
    selected_user = st.selectbox(
        "👤 Select Active User Profile",
        options=[None] + available_users,
        format_func=lambda x: f"Cold-Start Guest (No Ratings)" if x is None else f"User #{x} (Known Profile)",
        index=1
    )

    st.markdown("---")
    st.subheader("⚖️ Hybrid Blend Weight (α)")
    alpha_val = st.slider(
        "Alpha Balance (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="α = 1.0 is 100% Content-based (genres/director). α = 0.0 is 100% Collaborative filtering (ratings/popularity)."
    )

    # Descriptive badge
    if alpha_val >= 0.75:
        st.info(f"**Content-Dominant ({int(alpha_val*100)}% Content / {int((1-alpha_val)*100)}% CF)**: Maximizes plot, director, and genre similarity.")
    elif alpha_val <= 0.25:
        st.info(f"**Collaborative-Dominant ({int((1-alpha_val)*100)}% CF / {int(alpha_val*100)}% Content)**: Relies primarily on community rating patterns.")
    else:
        st.success(f"**Balanced Hybrid ({int(alpha_val*100)}% / {int((1-alpha_val)*100)}%)**: Synergizes thematic match with rating affinity.")

    top_n = st.slider("Number of Recommendations", min_value=3, max_value=12, value=6)
    exclude_already_rated = st.checkbox("Exclude already rated movies", value=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Personalized Hybrid Feed",
    "🔍 Content Lookalikes ('More Like This')",
    "🎞️ Catalog Explorer & Search",
    "📊 Offline Evaluation Benchmark"
])

# -------------------------------------------------------------
# TAB 1: PERSONALIZED HYBRID FEED
# -------------------------------------------------------------
with tab1:
    st.subheader("🎯 Tailored Recommendations")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption(
            f"Recommendations computed for **{'Guest User' if selected_user is None else f'User #{selected_user}'}** "
            f"with blend factor **α = {alpha_val:.2f}**."
        )
    with col2:
        seed_choice = st.selectbox(
            "Context Seed Movie (Optional)",
            options=[None] + movies_df["movie_id"].tolist(),
            format_func=lambda x: "None (Global Profile)" if x is None else movies_df[movies_df["movie_id"] == x]["title"].iloc[0]
        )

    # Compute recommendations
    recommendations = hybrid_engine.recommend(
        user_id=selected_user,
        seed_movie_id=seed_choice,
        top_k=top_n,
        alpha=alpha_val,
        exclude_rated=exclude_already_rated
    )

    if not recommendations:
        st.warning("No recommendations available under current filters. Try unchecking 'Exclude already rated movies'.")
    else:
        cols_per_row = 3
        for i in range(0, len(recommendations), cols_per_row):
            row_recs = recommendations[i:i + cols_per_row]
            cols = st.columns(cols_per_row)
            for j, item in enumerate(row_recs):
                with cols[j]:
                    genres_list = [g.strip() for g in item["genres"].split("|") if g.strip()]
                    genre_badges = "".join([f'<span class="genre-tag">{g}</span>' for g in genres_list])
                    
                    st.markdown(f"""
                    <div class="movie-card">
                        <div>
                            <div class="card-title">{item['title']}</div>
                            <div style="color: #64748b; font-size: 0.85rem; margin-bottom: 0.4rem;">
                                📅 {item['release_year']} • 🎬 {item['director']} • ⭐ {item['vote_average']} / 10
                            </div>
                            <div>{genre_badges}</div>
                            <p style="color: #334155; font-size: 0.85rem; margin: 0.5rem 0;">{item['overview'][:140]}...</p>
                        </div>
                        <div>
                            <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-top: 0.4rem;">
                                <span class="badge-score badge-hybrid">Hybrid: {item['hybrid_score']:.2f}</span>
                                <span class="badge-score badge-content">Content: {item['content_score']:.2f}</span>
                                <span class="badge-score badge-cf">CF: {item['collaborative_score']:.2f} (~{item['predicted_rating']}★)</span>
                            </div>
                            <div class="reason-box">💡 {item['reason']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: CONTENT-BASED LOOKALIKES
# -------------------------------------------------------------
with tab2:
    st.subheader("🔍 More Like This (TF-IDF Cosine Similarity)")
    st.write("Pick any movie in the catalog to inspect its stylistic, authorial, and genre lookalikes.")

    target_movie = st.selectbox(
        "Select Movie to Find Similar Titles",
        options=movies_df["movie_id"].tolist(),
        format_func=lambda x: f"{movies_df[movies_df['movie_id'] == x]['title'].iloc[0]} ({movies_df[movies_df['movie_id'] == x]['director'].iloc[0]})"
    )

    if target_movie:
        selected_meta = movies_df[movies_df["movie_id"] == target_movie].iloc[0]
        st.markdown(
            f"**Selected:** *{selected_meta['title']}* ({selected_meta['release_year']}) — "
            f"Directed by **{selected_meta['director']}** | Genres: `{selected_meta['genres']}`"
        )
        st.write(f"> *{selected_meta['overview']}*")

        sim_recs = cb_engine.get_similar_movies(target_movie, top_k=6)
        if sim_recs:
            st.markdown("#### Thematic Lookalikes:")
            sim_cols = st.columns(3)
            for idx, s_item in enumerate(sim_recs):
                with sim_cols[idx % 3]:
                    s_genres = "".join([f'<span class="genre-tag">{g}</span>' for g in s_item['genres'].split('|')])
                    st.markdown(f"""
                    <div class="movie-card">
                        <div>
                            <div class="card-title">{s_item['title']}</div>
                            <div style="color: #64748b; font-size: 0.85rem; margin-bottom: 0.4rem;">
                                📅 {s_item['release_year']} • 🎬 {s_item['director']}
                            </div>
                            <div>{s_genres}</div>
                        </div>
                        <div>
                            <span class="badge-score badge-content">Cosine Match: {s_item['similarity_score']:.2f}</span>
                            <div class="reason-box">🔗 {s_item['reason']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 3: CATALOG EXPLORER & SEARCH
# -------------------------------------------------------------
with tab3:
    st.subheader("🎞️ Movie Catalog Explorer")
    
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        search_query = st.text_input("🔍 Search by title, director, or actor...", "")
    with col_s2:
        all_genres = sorted(list(set([g.strip() for row in movies_df["genres"] for g in row.split("|")])))
        genre_filter = st.selectbox("Filter by Genre", options=["All Genres"] + all_genres)

    filtered_df = movies_df.copy()
    if search_query:
        mask = (
            filtered_df["title"].str.contains(search_query, case=False, na=False) |
            filtered_df["director"].str.contains(search_query, case=False, na=False) |
            filtered_df["cast"].str.contains(search_query, case=False, na=False)
        )
        filtered_df = filtered_df[mask]

    if genre_filter != "All Genres":
        filtered_df = filtered_df[filtered_df["genres"].str.contains(genre_filter, case=False, na=False)]

    st.write(f"Showing **{len(filtered_df)}** matching movies:")
    st.dataframe(
        filtered_df[["movie_id", "title", "release_year", "director", "genres", "vote_average", "overview"]],
        width="stretch",
        hide_index=True
    )

# -------------------------------------------------------------
# TAB 4: OFFLINE EVALUATION BENCHMARK
# -------------------------------------------------------------
with tab4:
    st.subheader("📊 Offline Quality & Ranking Benchmark")
    st.markdown(r"""
    Evaluation protocol: **Stratified 75/25 Train/Test rating holdout split**.
    - **Precision@K**: Fraction of top-K recommendations rated $\ge 4.0$ by test users.
    - **Recall@K**: Fraction of all test user favorites retrieved within top-K.
    - **NDCG@K**: Normalized Discounted Cumulative Gain penalizing relevant items placed lower in the ranking.
    """)

    from pathlib import Path
    eval_csv_path = Path(__file__).resolve().parent / "models" / "evaluation_benchmark.csv"
    if eval_csv_path.is_file():
        benchmark_table = pd.read_csv(eval_csv_path)
        st.dataframe(benchmark_table, width="stretch", hide_index=True)
        
        # Display bar chart comparison for P@5 and NDCG@5
        st.bar_chart(benchmark_table.set_index("Model Variant")[["P@5", "R@5", "NDCG@5"]])
    else:
        st.info("Evaluation metrics have not been generated yet. Run `python train_and_cache.py` to produce benchmark numbers.")

