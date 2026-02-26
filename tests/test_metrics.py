from __future__ import annotations

import math

import numpy as np
import pytest

from src.metrics import build_relevant_items_map, compute_ranking_metrics, rmse


def test_rmse_empty_arrays_return_nan() -> None:
    value = rmse(np.array([], dtype=np.float32), np.array([], dtype=np.float32))
    assert math.isnan(value)


def test_build_relevant_items_map_respects_threshold() -> None:
    rows = [(1, 10, 4.0), (1, 11, 3.5), (2, 20, 5.0)]
    relevant = build_relevant_items_map(rows, threshold=4.0)
    assert relevant == {1: {10}, 2: {20}}


def test_compute_ranking_metrics_expected_values() -> None:
    recommendations_by_user = {
        1: [10, 20, 30],
        2: [40, 50, 60],
    }
    relevant_items_by_user = {
        1: {20, 99},
        2: {70},
        3: set(),
    }

    metrics = compute_ranking_metrics(
        recommendations_by_user=recommendations_by_user,
        relevant_items_by_user=relevant_items_by_user,
        k=2,
    )

    assert metrics["eligible_users"] == 2
    assert metrics["precision_at_k"] == pytest.approx(0.25)
    assert metrics["recall_at_k"] == pytest.approx(0.25)
    assert metrics["ndcg_at_k"] == pytest.approx(0.19342640361727081)
    assert metrics["map_at_k"] == pytest.approx(0.125)
    assert metrics["mrr_at_k"] == pytest.approx(0.25)
