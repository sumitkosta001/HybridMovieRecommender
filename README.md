# CineMatch: Hybrid Movie Recommendation System

An end-to-end, production-grade Hybrid Movie Recommendation Platform combining **Content-Based TF-IDF Cosine Similarity** with **Collaborative Filtering via Matrix Factorization (Truncated SVD)**, exposed through an interactive **Streamlit dashboard** and a high-performance **FastAPI REST inference microservice**.

---

## 📌 Architecture & Mathematical Foundations

```
                         +-----------------------------+
                         |     User Query / Context    |
                         +--------------+--------------+
                                        |
                           +------------+------------+
                           |                         |
                           v                         v
              +------------------------+  +-------------------------+
              |     Content-Based      |  |  Collaborative Filtering |
              |   (TF-IDF + Cosine)    |  |  (Truncated SVD + Bayes) |
              +------------+-----------+  +------------+------------+
                           |                           |
                           | Score_content in [0, 1]   | Score_cf in [0, 1]
                           +------------+--------------+
                                        |
                                        v
                       +----------------------------------+
                       |      Hybrid Linear Blending      |
                       | Score = α*Content + (1-α)*CF     |
                       +----------------+-----------------+
                                        |
                                        v
                       +----------------------------------+
                       | Self-Exclusion & Rank Ordering   |
                       | Top-K Recommendations + Reasons  |
                       +----------------------------------+
```

### 1. Content-Based Filtering (TF-IDF + Cosine Similarity)
- Formulates a weighted **metadata soup** combining director (doubled weight to capture authorial style), genres, cast, and synopsis tokens.
- Vectorized using Scikit-Learn `TfidfVectorizer` (unigrams and bigrams).
- Pairwise Cosine Similarity:
  $$\text{CosineSim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
  Produces dense similarity lookalikes with explainable attribute tags (director and shared genres).

### 2. Collaborative Filtering (Truncated SVD + Bayesian Fallback)
- Transforms user-item rating interactions into a rating pivot table.
- Decomposes the interaction matrix using **Truncated SVD** ($k=4$ latent factor components):
  $$R \approx U_k \Sigma_k V_k^T$$
- **Reconstruction & Clamping**: Predicted star ratings are clipped to $[1.0, 5.0]$ and normalized onto $[0.0, 1.0]$:
  $$\text{Score}_{\text{CF}} = \frac{\hat{r}_{u, i} - 1.0}{4.0}$$
- **Cold-Start Bayesian Fallback**: For guest users or unrated items, the engine applies Bayesian-smoothed item popularity priors:
  $$\text{Score}_{\text{Bayes}} = \frac{v}{v + m} R + \frac{m}{v + m} C$$
  where $v$ is rating count, $m=2.0$ is the Bayesian prior threshold, $R$ is item mean, and $C$ is global mean.

### 3. Hybrid Blending & Guarantees
- Linear interpolation parameterized by $\alpha \in [0.0, 1.0]$:
  $$\text{Score}_{\text{hybrid}} = \alpha \cdot \text{Score}_{\text{content}} + (1 - \alpha) \cdot \text{Score}_{\text{collaborative}}$$
- **Self-Exclusion**: Target query seed movies are strictly filtered out of recommendations.
- **Deduplication & Rated Filtering**: Optionally filters out movies the user has already rated.
- **Cold-Start Resilience**: Unknown users and unseen movie queries seamlessly fall back to popularity priors without runtime exceptions.

---

## 📊 Dataset Schema, Provenance & Limitations

The system operates on CSV files located in `data/`:

### 1. `data/movies.csv`
Contains 20 curated cinematic titles with rich metadata:
- `movie_id` (*int, primary key*): Unique identifier (e.g., `101`)
- `title` (*string*): Title of the film
- `genres` (*string*): Pipe-delimited genres (e.g., `Sci-Fi|Adventure|Drama`)
- `director` (*string*): Film director
- `cast` (*string*): Main actor names
- `release_year` (*int*): Year of theatrical release
- `vote_average` (*float*): Consensus score $[0.0, 10.0]$
- `overview` (*string*): Thematic synopsis used for TF-IDF feature extraction

### 2. `data/ratings.csv`
Contains user-movie interaction logs across 16 user archetypes:
- `user_id` (*int*): User account identifier $[1..16]$
- `movie_id` (*int*): Target movie identifier
- `rating` (*float*): Review rating on a $[0.5, 5.0]$ scale
- `timestamp` (*int*): Epoch timestamp

### 3. Provenance & Limitations Notice
> [!IMPORTANT]
> **Functional Demonstration Dataset Only**:
> The movie catalog and rating profiles in `data/` are curated and synthetically generated demonstration data designed to allow the collaborative-filtering matrix factorization, TF-IDF vectorizer, and hybrid blending mechanics to run deterministically and meaningfully in a local environment.
>
> **These synthetic rating interactions do NOT represent real-world user behavioral distributions.** High scores or rankings achieved in this demonstration setup should be understood as evidence of correct mathematical implementation and algorithmic functioning, not empirical evidence of recommendation accuracy on human cohorts.
>
> **Regenerating Datasets**:
> To regenerate or refresh the CSV files with reproducible random seeds:
> ```bash
> python -m data.generate_data
> ```

---

## 📈 Measured Offline Evaluation Benchmark

The evaluation pipeline ([`src/evaluation.py`](src/evaluation.py)) implements a strict **stratified 75/25 Train/Test holdout split** on user ratings. Holdout ratings $\ge 4.0$ are treated as relevant ground-truth test items. Crucially, **the test split is completely isolated from training data and user profiles to prevent data leakage**.

Actual measured performance across model variants:

| Model Variant | Precision@5 | Precision@10 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.0286 | 0.0571 | 0.0595 | 0.4048 | 0.0430 | 0.1566 |
| **Content-Based ($\alpha=1.0$)** | 0.1429 | 0.1214 | 0.4524 | 0.7143 | 0.2164 | 0.3227 |
| **Collaborative Filtering ($\alpha=0.0$)** | 0.0429 | 0.0643 | 0.1786 | 0.4762 | 0.1441 | 0.2512 |
| **Hybrid Recommender ($\alpha=0.5$)** | 0.0571 | 0.0857 | 0.2143 | 0.6190 | 0.1011 | 0.2389 |

To re-run the benchmark and save artifacts to `models/`:
```bash
python train_and_cache.py
```

---

## 📁 Repository Structure

```
MyHybridMovieRecommender/
├── data/
│   ├── movies.csv               # Movie catalog CSV (loaded by data_loader)
│   ├── ratings.csv              # User ratings interaction matrix CSV
│   ├── sample_data.py           # Catalog metadata definitions
│   └── generate_data.py         # Script to regenerate movies.csv and ratings.csv
├── models/
│   ├── evaluation_benchmark.csv # Precision@K, Recall@K, NDCG@K results
│   └── summary.json             # Pipeline summary stats
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Relative project paths, schema checks, error reporting
│   ├── content_based.py         # TF-IDF vectorization & cosine similarity engine
│   ├── collaborative.py         # SVD matrix factorization & Bayesian popularity fallback
│   ├── hybrid.py                # Normalized alpha-blended hybrid recommender
│   ├── engine.py                # Pipeline factory & initialization from CSV
│   └── evaluation.py            # Leak-free Precision@K, Recall@K, NDCG@K benchmarking
├── tests/
│   ├── test_recommender.py      # Core engine & CSV edge cases (14 tests)
│   └── test_api.py              # FastAPI REST endpoint integration tests (7 tests)
├── train_and_cache.py           # Pipeline script for fitting and benchmarking
├── api.py                       # High-performance FastAPI REST inference service
├── app.py                       # Interactive 4-tab Streamlit web application
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git exclusion rules
└── README.md                    # Project documentation
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites & Virtual Environment
Ensure Python 3.10+ is installed:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline (Optional Benchmark Generation)
```bash
python train_and_cache.py
```

### 4. Run Test Suite
Run the 21 automated pytest tests verifying self-exclusion, score bounds, fallback mechanics, and REST endpoints:
```bash
pytest -v
```

### 5. Launch FastAPI REST Microservice
```bash
uvicorn api:app --reload --port 8000
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Hybrid recommendations: `http://localhost:8000/recommend/hybrid?user_id=1&top_k=5&alpha=0.6`

### 6. Launch Interactive Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).

