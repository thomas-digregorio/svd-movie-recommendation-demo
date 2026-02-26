"""Method B: Surprise SVD trained on observed ratings only.

If scikit-surprise is unavailable (notably on Python 3.13+), a NumPy SGD fallback
is used with similar factorization hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import build_relevant_items_map, compute_ranking_metrics, rmse

try:
    from surprise import Dataset, Reader, SVD  # type: ignore

    SURPRISE_BACKEND_AVAILABLE = True
except Exception:
    Dataset = None  # type: ignore[assignment]
    Reader = None  # type: ignore[assignment]
    SVD = None  # type: ignore[assignment]
    SURPRISE_BACKEND_AVAILABLE = False


@dataclass
class _Prediction:
    est: float


class _FallbackSVD:
    """Simple MF model trained by SGD on observed ratings only."""

    def __init__(
        self,
        n_factors: int,
        n_epochs: int,
        lr_all: float,
        reg_all: float,
        random_state: int,
    ) -> None:
        self.n_factors = int(n_factors)
        self.n_epochs = int(n_epochs)
        self.lr_all = float(lr_all)
        self.reg_all = float(reg_all)
        self.random_state = int(random_state)

        self.global_mean = 0.0
        self.user_to_idx: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.user_bias = np.zeros(0, dtype=np.float32)
        self.item_bias = np.zeros(0, dtype=np.float32)
        self.user_factors = np.zeros((0, self.n_factors), dtype=np.float32)
        self.item_factors = np.zeros((0, self.n_factors), dtype=np.float32)

    def fit(self, train_df: pd.DataFrame) -> "_FallbackSVD":
        users = np.sort(train_df["user_id"].astype(int).unique())
        items = np.sort(train_df["item_id"].astype(int).unique())
        self.user_to_idx = {uid: idx for idx, uid in enumerate(users)}
        self.item_to_idx = {iid: idx for idx, iid in enumerate(items)}

        n_users = len(users)
        n_items = len(items)
        rng = np.random.default_rng(self.random_state)

        self.global_mean = float(train_df["rating"].mean())
        self.user_bias = np.zeros(n_users, dtype=np.float32)
        self.item_bias = np.zeros(n_items, dtype=np.float32)
        self.user_factors = rng.normal(
            loc=0.0,
            scale=0.1,
            size=(n_users, self.n_factors),
        ).astype(np.float32)
        self.item_factors = rng.normal(
            loc=0.0,
            scale=0.1,
            size=(n_items, self.n_factors),
        ).astype(np.float32)

        user_idx = train_df["user_id"].map(self.user_to_idx).to_numpy(dtype=np.int32)
        item_idx = train_df["item_id"].map(self.item_to_idx).to_numpy(dtype=np.int32)
        ratings = train_df["rating"].to_numpy(dtype=np.float32)

        for _ in range(self.n_epochs):
            order = rng.permutation(len(ratings))
            for row_idx in order:
                u = int(user_idx[row_idx])
                i = int(item_idx[row_idx])
                rating = float(ratings[row_idx])

                pu = self.user_factors[u]
                qi = self.item_factors[i]
                pred = self.global_mean + self.user_bias[u] + self.item_bias[i] + float(np.dot(pu, qi))
                err = rating - pred

                self.user_bias[u] += self.lr_all * (err - self.reg_all * self.user_bias[u])
                self.item_bias[i] += self.lr_all * (err - self.reg_all * self.item_bias[i])

                pu_prev = pu.copy()
                self.user_factors[u] += self.lr_all * (err * qi - self.reg_all * pu)
                self.item_factors[i] += self.lr_all * (err * pu_prev - self.reg_all * qi)

        return self

    def predict(self, raw_user_id: int, raw_item_id: int) -> _Prediction:
        uid = int(raw_user_id)
        iid = int(raw_item_id)
        user_idx = self.user_to_idx.get(uid)
        item_idx = self.item_to_idx.get(iid)

        estimate = float(self.global_mean)
        if user_idx is not None:
            estimate += float(self.user_bias[user_idx])
        if item_idx is not None:
            estimate += float(self.item_bias[item_idx])
        if user_idx is not None and item_idx is not None:
            estimate += float(np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))

        estimate = float(np.clip(estimate, 1.0, 5.0))
        return _Prediction(est=estimate)


def get_surprise_backend_label(model: dict[str, Any] | None = None) -> str:
    """Return the active Surprise backend label."""
    if model is not None:
        return str(model.get("backend", "unknown"))
    return "scikit-surprise" if SURPRISE_BACKEND_AVAILABLE else "numpy-fallback-sgd"


def fit_surprise(
    train_df: pd.DataFrame,
    n_factors: int,
    n_epochs: int,
    lr_all: float,
    reg_all: float,
    seed: int,
) -> dict[str, Any]:
    """Fit a Surprise-style SVD model on observed train interactions."""
    if train_df.empty:
        raise ValueError("train_df is empty; cannot fit Surprise SVD model.")

    if SURPRISE_BACKEND_AVAILABLE:
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
        backend = "scikit-surprise"
    else:
        algo = _FallbackSVD(
            n_factors=int(n_factors),
            n_epochs=int(n_epochs),
            lr_all=float(lr_all),
            reg_all=float(reg_all),
            random_state=int(seed),
        ).fit(train_df)
        trainset = None
        backend = "numpy-fallback-sgd"

    train_items_by_user = {
        int(user_id): set(group["item_id"].astype(int).tolist())
        for user_id, group in train_df.groupby("user_id")
    }
    return {
        "algo": algo,
        "trainset": trainset,
        "n_factors": int(n_factors),
        "backend": backend,
        "train_items_by_user": train_items_by_user,
    }


def predict_surprise(model: dict[str, Any], test_df: pd.DataFrame) -> np.ndarray:
    """Predict ratings for each row in test_df."""
    algo = model["algo"]
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

    algo = model["algo"]
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
    """Evaluate RMSE and ranking metrics for a fitted Surprise-style SVD model."""
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
    """Sweep Surprise-style latent dimensions and return rows plus fitted models."""
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
