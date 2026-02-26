from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.svd_truncated import evaluate_truncated, fit_truncated, predict_truncated, recommend_truncated


@pytest.fixture
def train_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_idx": 0, "item_idx": 0, "rating": 5.0},
            {"user_idx": 0, "item_idx": 1, "rating": 4.0},
            {"user_idx": 1, "item_idx": 0, "rating": 4.0},
            {"user_idx": 1, "item_idx": 2, "rating": 2.0},
            {"user_idx": 2, "item_idx": 1, "rating": 3.0},
            {"user_idx": 2, "item_idx": 3, "rating": 5.0},
            {"user_idx": 3, "item_idx": 2, "rating": 4.0},
            {"user_idx": 3, "item_idx": 4, "rating": 1.0},
        ]
    )


@pytest.fixture
def test_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_idx": 0, "item_idx": 2, "rating": 4.0},
            {"user_idx": 1, "item_idx": 1, "rating": 3.0},
            {"user_idx": 2, "item_idx": 0, "rating": 2.0},
            {"user_idx": 3, "item_idx": 3, "rating": 4.0},
        ]
    )


def test_fit_truncated_uses_factor_representation(train_df: pd.DataFrame) -> None:
    model = fit_truncated(train_df=train_df, n_users=4, n_items=5, k=10, n_iter=7, seed=42)

    assert "pred_matrix" not in model
    assert model["effective_k"] == 3
    assert model["user_factors"].shape == (4, 3)
    assert model["item_factors"].shape == (5, 3)


def test_predict_truncated_outputs_bounded_scores(train_df: pd.DataFrame) -> None:
    model = fit_truncated(train_df=train_df, n_users=4, n_items=5, k=3, n_iter=7, seed=42)
    pred = predict_truncated(model=model, user_idx=0, item_idx=2)

    assert 1.0 <= pred <= 5.0
    with pytest.raises(IndexError):
        predict_truncated(model=model, user_idx=-1, item_idx=2)
    with pytest.raises(IndexError):
        predict_truncated(model=model, user_idx=0, item_idx=100)


def test_recommend_truncated_excludes_train_items(train_df: pd.DataFrame) -> None:
    model = fit_truncated(train_df=train_df, n_users=4, n_items=5, k=3, n_iter=7, seed=42)
    recs = recommend_truncated(model=model, user_idx=0, n_recommendations=5, exclude_train=True)
    seen_items = {0, 1}

    assert all(item_idx not in seen_items for item_idx, _ in recs)
    assert len(recs) <= 3
    assert recommend_truncated(model=model, user_idx=0, n_recommendations=0) == []


def test_evaluate_truncated_empty_test_returns_nan(train_df: pd.DataFrame) -> None:
    model = fit_truncated(train_df=train_df, n_users=4, n_items=5, k=3, n_iter=7, seed=42)
    empty_test = pd.DataFrame(columns=["user_idx", "item_idx", "rating"])
    metrics = evaluate_truncated(
        model=model,
        train_df=train_df,
        test_df=empty_test,
        ranking_k=10,
        relevance_threshold=4.0,
    )

    assert metrics["eligible_users"] == 0
    assert math.isnan(float(metrics["rmse"]))
    assert math.isnan(float(metrics["precision_at_k"]))


def test_evaluate_truncated_nonempty_returns_finite_rmse(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    model = fit_truncated(train_df=train_df, n_users=4, n_items=5, k=3, n_iter=7, seed=42)
    metrics = evaluate_truncated(
        model=model,
        train_df=train_df,
        test_df=test_df,
        ranking_k=3,
        relevance_threshold=4.0,
    )

    assert np.isfinite(float(metrics["rmse"]))
