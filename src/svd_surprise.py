"""Method B: Surprise SVD trained on observed ratings only."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD

from src.metrics import build_relevant_items_map, compute_ranking_metrics, rmse


def fit_surprise(
    train_df: pd.DataFrame,
    n_factors: int,
    n_epochs: int,
    lr_all: float,
    reg_all: float,
    seed: int,
) -> dict[str, Any]:
    """Fit a Surprise SVD model on the train interactions."""
    if train_df.empty:
        raise ValueError("train_df is empty; cannot fit Surprise SVD model.")

    reader = Reader(rating_scale=(1.0, 5.0))
    dataset = Dataset.load_from_df(train_df[["user_id", "item_id", "rating"]], reader)
    trainset = dataset.build_full_trainset()

    algo = SVD(
        n_factors=int(n_factors),
        n_epochs=int(n_epochs),
        lr_all=float(lr_all),
        reg_all=float(reg_all),
        random_state=int(seed),
    )
    algo.fit(trainset)

    train_items_by_user = {
        int(user_id): set(group["item_id"].astype(int).tolist())
        for user_id, group in train_df.groupby("user_id")
    }
    return {
        "algo": algo,
        "trainset": trainset,
        "n_factors": int(n_factors),
        "train_items_by_user": train_items_by_user,
    }


def predict_surprise(model: dict[str, Any], test_df: pd.DataFrame) -> np.ndarray:
    """Predict ratings for each row in test_df."""
    algo: SVD = model["algo"]
    preds = [
        float(algo.predict(int(row.user_id), int(row.item_id)).est)
        for row in test_df.itertuples(index=False)
    ]
    return np.asarray(preds, dtype=np.float32)


def recommend_surprise(
    model: dict[str, Any],
    train_df: pd.DataFrame,
    all_item_ids: list[int],
    raw_user_id: int,
    n_recommendations: int,
) -> list[tuple[int, float]]:
    """Top-N raw-item recommendations for a raw user id."""
    del train_df  # model already stores train user-item history

    n_recommendations = int(max(0, n_recommendations))
    if n_recommendations == 0:
        return []

    raw_user_id = int(raw_user_id)
    seen_items = model["train_items_by_user"].get(raw_user_id, set())
    candidate_items = [int(item_id) for item_id in all_item_ids if int(item_id) not in seen_items]
    if not candidate_items:
        return []

    algo: SVD = model["algo"]
    scored = [
        (item_id, float(algo.predict(raw_user_id, item_id).est))
        for item_id in candidate_items
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n_recommendations]


def evaluate_surprise(
    model: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_item_ids: list[int],
    ranking_k: int,
    relevance_threshold: float,
) -> dict[str, float | int]:
    """Evaluate RMSE and ranking metrics for a fitted Surprise SVD model."""
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

    y_true = test_df["rating"].to_numpy(dtype=np.float32)
    y_pred = predict_surprise(model, test_df)
    rmse_value = rmse(y_true, y_pred)

    relevant_items_by_user = build_relevant_items_map(
        zip(
            test_df["user_id"].astype(int),
            test_df["item_id"].astype(int),
            test_df["rating"].astype(float),
        ),
        threshold=relevance_threshold,
    )

    recommendations_by_user: dict[int, list[int]] = {}
    for raw_user_id, relevant_items in relevant_items_by_user.items():
        if not relevant_items:
            continue
        recs = recommend_surprise(
            model=model,
            train_df=train_df,
            all_item_ids=all_item_ids,
            raw_user_id=int(raw_user_id),
            n_recommendations=ranking_k,
        )
        recommendations_by_user[int(raw_user_id)] = [item_id for item_id, _ in recs]

    ranking_metrics = compute_ranking_metrics(
        recommendations_by_user=recommendations_by_user,
        relevant_items_by_user=relevant_items_by_user,
        k=ranking_k,
    )
    return {"rmse": rmse_value, **ranking_metrics}


def run_surprise_sweep(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_item_ids: list[int],
    n_factors_list: list[int],
    n_epochs: int,
    lr_all: float,
    reg_all: float,
    seed: int,
    ranking_k: int,
    relevance_threshold: float,
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    """Sweep Surprise SVD latent dimensions and return result rows plus fitted models."""
    rows: list[dict[str, float | int | str]] = []
    models_by_factor: dict[int, dict[str, Any]] = {}

    for n_factors in n_factors_list:
        start = time.perf_counter()
        model = fit_surprise(
            train_df=train_df,
            n_factors=int(n_factors),
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            seed=seed,
        )
        metrics = evaluate_surprise(
            model=model,
            train_df=train_df,
            test_df=test_df,
            all_item_ids=all_item_ids,
            ranking_k=ranking_k,
            relevance_threshold=relevance_threshold,
        )
        runtime_sec = time.perf_counter() - start

        models_by_factor[int(n_factors)] = model
        rows.append(
            {
                "method": "surprise_svd",
                "k": int(n_factors),
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

    return pd.DataFrame(rows), models_by_factor

