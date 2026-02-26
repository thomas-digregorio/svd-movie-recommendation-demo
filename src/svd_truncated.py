"""Method A: TruncatedSVD on a centered sparse ratings matrix."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.metrics import build_relevant_items_map, compute_ranking_metrics, rmse


def fit_truncated(
    train_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    k: int,
    n_iter: int,
    seed: int,
) -> dict[str, Any]:
    """Fit a centered sparse TruncatedSVD model."""
    if train_df.empty:
        raise ValueError("train_df is empty; cannot fit TruncatedSVD model.")
    if n_users <= 0 or n_items <= 0:
        raise ValueError("n_users and n_items must be positive.")

    user_idx = train_df["user_idx"].to_numpy(dtype=np.int32)
    item_idx = train_df["item_idx"].to_numpy(dtype=np.int32)
    ratings = train_df["rating"].to_numpy(dtype=np.float32)

    counts = np.bincount(user_idx, minlength=n_users)
    sums = np.bincount(user_idx, weights=ratings, minlength=n_users).astype(np.float32)
    global_mean = np.float32(ratings.mean())
    user_mean = np.full(n_users, global_mean, dtype=np.float32)
    nonzero = counts > 0
    user_mean[nonzero] = sums[nonzero] / counts[nonzero].astype(np.float32)

    centered = ratings - user_mean[user_idx]
    centered_matrix = csr_matrix(
        (centered, (user_idx, item_idx)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )

    min_dim = min(n_users, n_items)
    max_components = min_dim - 1 if min_dim > 1 else 1
    effective_k = int(max(1, min(k, max_components)))

    svd = TruncatedSVD(
        n_components=effective_k,
        n_iter=n_iter,
        random_state=seed,
    )
    z = svd.fit_transform(centered_matrix).astype(np.float32, copy=False)
    v = svd.components_.astype(np.float32, copy=False)

    reconstructed_centered = (z @ v).astype(np.float32, copy=False)
    reconstructed = reconstructed_centered + user_mean[:, None]
    np.clip(reconstructed, 1.0, 5.0, out=reconstructed)

    train_items_by_user = {
        int(user): set(items.astype(int).tolist())
        for user, items in train_df.groupby("user_idx")["item_idx"]
    }

    return {
        "requested_k": int(k),
        "effective_k": effective_k,
        "n_users": int(n_users),
        "n_items": int(n_items),
        "user_mean": user_mean,
        "pred_matrix": reconstructed,
        "train_items_by_user": train_items_by_user,
        "svd_model": svd,
    }


def predict_truncated(model: dict[str, Any], user_idx: int, item_idx: int) -> float:
    """Predict a user/item rating from the reconstructed matrix."""
    pred_matrix = model["pred_matrix"]
    if user_idx < 0 or user_idx >= pred_matrix.shape[0]:
        raise IndexError(f"user_idx {user_idx} is out of range.")
    if item_idx < 0 or item_idx >= pred_matrix.shape[1]:
        raise IndexError(f"item_idx {item_idx} is out of range.")
    return float(pred_matrix[user_idx, item_idx])


def recommend_truncated(
    model: dict[str, Any],
    user_idx: int,
    n_recommendations: int,
    exclude_train: bool = True,
) -> list[tuple[int, float]]:
    """Top-N recommendations for a user index."""
    if user_idx < 0 or user_idx >= model["n_users"]:
        return []
    n_recommendations = int(max(0, n_recommendations))
    if n_recommendations == 0:
        return []

    scores = model["pred_matrix"][user_idx].copy()
    if exclude_train:
        seen_items = model["train_items_by_user"].get(user_idx, set())
        if seen_items:
            seen_idx = np.fromiter(seen_items, dtype=np.int32, count=len(seen_items))
            scores[seen_idx] = -np.inf

    n_pick = min(n_recommendations, scores.size)
    if n_pick == 0:
        return []

    top_idx = np.argpartition(scores, -n_pick)[-n_pick:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return [(int(i), float(scores[i])) for i in top_idx if np.isfinite(scores[i])]


def evaluate_truncated(
    model: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ranking_k: int,
    relevance_threshold: float,
) -> dict[str, float | int]:
    """Evaluate RMSE and ranking metrics for a fitted TruncatedSVD model."""
    ranking_k = int(max(1, ranking_k))
    if test_df.empty:
        return {
            "rmse": 0.0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "map_at_k": 0.0,
            "mrr_at_k": 0.0,
            "eligible_users": 0,
        }

    user_idx = test_df["user_idx"].to_numpy(dtype=np.int32)
    item_idx = test_df["item_idx"].to_numpy(dtype=np.int32)
    y_true = test_df["rating"].to_numpy(dtype=np.float32)
    y_pred = model["pred_matrix"][user_idx, item_idx]
    rmse_value = rmse(y_true, y_pred)

    relevant_items_by_user = build_relevant_items_map(
        zip(
            test_df["user_idx"].astype(int),
            test_df["item_idx"].astype(int),
            test_df["rating"].astype(float),
        ),
        threshold=relevance_threshold,
    )

    recommendations_by_user: dict[int, list[int]] = {}
    for user_idx_value, relevant_items in relevant_items_by_user.items():
        if not relevant_items:
            continue
        user_recs = recommend_truncated(
            model=model,
            user_idx=int(user_idx_value),
            n_recommendations=ranking_k,
            exclude_train=True,
        )
        recommendations_by_user[int(user_idx_value)] = [item for item, _ in user_recs]

    ranking_metrics = compute_ranking_metrics(
        recommendations_by_user=recommendations_by_user,
        relevant_items_by_user=relevant_items_by_user,
        k=ranking_k,
    )

    return {"rmse": rmse_value, **ranking_metrics}


def run_truncated_sweep(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    k_list: list[int],
    n_iter: int,
    seed: int,
    ranking_k: int,
    relevance_threshold: float,
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    """Sweep TruncatedSVD ranks and return result rows plus fitted models."""
    rows: list[dict[str, float | int | str]] = []
    models_by_k: dict[int, dict[str, Any]] = {}

    for k in k_list:
        start = time.perf_counter()
        model = fit_truncated(
            train_df=train_df,
            n_users=n_users,
            n_items=n_items,
            k=int(k),
            n_iter=n_iter,
            seed=seed,
        )
        metrics = evaluate_truncated(
            model=model,
            train_df=train_df,
            test_df=test_df,
            ranking_k=ranking_k,
            relevance_threshold=relevance_threshold,
        )
        runtime_sec = time.perf_counter() - start

        models_by_k[int(k)] = model
        rows.append(
            {
                "method": "truncated_svd",
                "k": int(model["effective_k"]),
                "rmse": float(metrics["rmse"]),
                "precision_at_k": float(metrics["precision_at_k"]),
                "recall_at_k": float(metrics["recall_at_k"]),
                "ndcg_at_k": float(metrics["ndcg_at_k"]),
                "map_at_k": float(metrics["map_at_k"]),
                "mrr_at_k": float(metrics["mrr_at_k"]),
                "eligible_users": int(metrics["eligible_users"]),
                "runtime_sec": float(runtime_sec),
            }
        )

    return pd.DataFrame(rows), models_by_k

