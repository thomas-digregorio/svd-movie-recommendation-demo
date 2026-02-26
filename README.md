# SVD Movie Recommendation Demo

This project compares two different "SVD recommender" approaches on MovieLens 100K and exposes the results in a Streamlit app.

1. **Method A: Truncated SVD on a centered sparse user-item matrix**
2. **Method B: Surprise SVD matrix factorization trained on observed ratings only**

The comparison includes RMSE and ranking metrics, latency per sweep point, and top-N recommendations for a selected user.  Shows low-rank structure acts as compression + denoising.

## Dataset

- Source: MovieLens 100K (`ml-100k`) from GroupLens
- Automatically downloaded to `./data/` if missing
- Files used:
  - `u.data`: ratings
  - `u.item`: movie titles

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` installs `scikit-surprise` only on Python `<3.13` due upstream build incompatibility on 3.13.

## Run

```bash
streamlit run app.py
```

## Quick CLI sanity check

```bash
python -m src.sanity_check
```

This fits both methods with `k=20`, prints metrics, and prints sample recommendations.

## Methods

### A) Truncated SVD (centered sparse matrix)

- Build matrix from **train interactions only**:
  - For each user, compute train mean rating.
  - Store centered values `(r_ui - mean_u)` at observed entries.
  - Missing entries are treated as `0` after centering.
- Fit `sklearn.decomposition.TruncatedSVD`.
- Reconstruct full score matrix, add user means back, and clip predictions to `[1, 5]`.

### B) Surprise SVD (MF on observed ratings)

- Fit `surprise.SVD` using train interactions (`user_id`, `item_id`, `rating`) when available.
- On Python 3.13+, `scikit-surprise` is currently skipped in `requirements.txt` and the app uses a compatible NumPy SGD fallback with the same core hyperparameters (`n_factors`, `n_epochs`, `lr_all`, `reg_all`).
- Predict test ratings with `algo.predict(uid, iid).est`.
- For recommendations, score candidate items not seen in train and rank descending.

## Train / test split logic

Split is performed per user and guarantees each modeled user has at least one train interaction:

1. Force one random interaction per user into train.
2. For remaining interactions, assign `floor(test_size * user_interactions)` to test.
3. Put the rest into train.

This guarantees train coverage per user, but does **not** guarantee a test point for every user.

## Evaluation

### Rating prediction metric

- RMSE on held-out test interactions.

### Ranking metrics (averaged over eligible users)

- Precision@K
- Recall@K
- NDCG@K
- MAP@K
- MRR@K

Definitions:

- Relevance: `test_rating >= relevance_threshold` (default `4.0`)
- Eligible users: users with at least one relevant test item
- Candidate set: items not seen by the user in train

## Streamlit behavior and config notes

Default values:

- `seed=42`
- `test_size=0.2`
- `relevance_threshold=4.0`
- `Ranking K=10`
- `N recommendations=10`
- Method A `k_list=[10,20,50,100]`, `n_iter=7`
- Method B `n_factors_list=[10,20,50,100]`, `n_epochs=20`, `lr_all=0.005`, `reg_all=0.02`
- Optional caps: `max_users=0`, `max_interactions=0` (`0` means no cap)

Important implementation choices:

1. `max_users` and `max_interactions` caps are applied **before** split for faster experiments.
2. The user dropdown shows all raw MovieLens user IDs even when caps reduce the modeled subset.
3. While sweeping latent rank, ranking metrics are computed at fixed sidebar `Ranking K` to isolate latent-rank effects from top-K cutoff effects.
4. Sidebar edits are staged until **Recompute** is clicked.
5. Model fitting uses `st.cache_resource` and data uses `st.cache_data`.
6. Changing only `N recommendations` affects display output, not model fitting.

## Caveats

- Offline metrics are proxies; online performance can differ.
- Ranking against unseen-train candidates is sensitive to split design.
- Popularity bias can inflate offline ranking scores.
- Unknown users/items (under heavy caps) may lead to less personalized outputs.

## Project structure

```text
app.py
src/
  __init__.py
  config.py
  data.py
  metrics.py
  svd_truncated.py
  svd_surprise.py
  viz.py
  sanity_check.py
requirements.txt
README.md
.gitignore
```
