"""Regression and ranking metrics for recommender evaluation."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    if y_true.size == 0:
        return 0.0
    diff = y_true.astype(np.float64) - y_pred.astype(np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def _precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return float(hits / k)


def _recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return float(hits / len(relevant))


def _ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1.0)

    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg > 0.0 else 0.0


def _average_precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0

    hit_count = 0
    precision_sum = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hit_count += 1
            precision_sum += hit_count / rank

    denominator = min(len(relevant), k)
    if denominator == 0:
        return 0.0
    return float(precision_sum / denominator)


def _reciprocal_rank_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return float(1.0 / rank)
    return 0.0


def compute_ranking_metrics(
    recommendations_by_user: dict[int, list[int]],
    relevant_items_by_user: dict[int, set[int]],
    k: int,
) -> dict[str, float | int]:
    """
    Compute Precision/Recall/NDCG/MAP/MRR@K over eligible users.

    Eligible users are those with at least one relevant item in test.
    """
    precision_values: list[float] = []
    recall_values: list[float] = []
    ndcg_values: list[float] = []
    ap_values: list[float] = []
    rr_values: list[float] = []

    eligible_users = 0
    for user_id, relevant_items in relevant_items_by_user.items():
        if not relevant_items:
            continue
        eligible_users += 1
        recommended = recommendations_by_user.get(user_id, [])
        precision_values.append(_precision_at_k(recommended, relevant_items, k))
        recall_values.append(_recall_at_k(recommended, relevant_items, k))
        ndcg_values.append(_ndcg_at_k(recommended, relevant_items, k))
        ap_values.append(_average_precision_at_k(recommended, relevant_items, k))
        rr_values.append(_reciprocal_rank_at_k(recommended, relevant_items, k))

    if eligible_users == 0:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "map_at_k": 0.0,
            "mrr_at_k": 0.0,
            "eligible_users": 0,
        }

    return {
        "precision_at_k": float(np.mean(precision_values)),
        "recall_at_k": float(np.mean(recall_values)),
        "ndcg_at_k": float(np.mean(ndcg_values)),
        "map_at_k": float(np.mean(ap_values)),
        "mrr_at_k": float(np.mean(rr_values)),
        "eligible_users": eligible_users,
    }


def build_relevant_items_map(
    test_rows: Iterable[tuple[int, int, float]],
    threshold: float,
) -> dict[int, set[int]]:
    """Build user -> relevant item-set map from test rows."""
    relevant: dict[int, set[int]] = {}
    for user_id, item_id, rating in test_rows:
        user_relevant = relevant.setdefault(int(user_id), set())
        if float(rating) >= threshold:
            user_relevant.add(int(item_id))
    return relevant

