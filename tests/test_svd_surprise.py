from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.svd_surprise import evaluate_surprise, fit_surprise, recommend_surprise


@pytest.fixture
def train_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": 1, "item_id": 10, "rating": 5.0},
            {"user_id": 1, "item_id": 11, "rating": 4.0},
            {"user_id": 2, "item_id": 10, "rating": 4.0},
            {"user_id": 2, "item_id": 12, "rating": 2.0},
            {"user_id": 3, "item_id": 11, "rating": 3.0},
            {"user_id": 3, "item_id": 13, "rating": 5.0},
            {"user_id": 4, "item_id": 14, "rating": 3.0},
        ]
    )


@pytest.fixture
def test_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": 1, "item_id": 12, "rating": 4.0},
            {"user_id": 2, "item_id": 11, "rating": 3.0},
            {"user_id": 3, "item_id": 10, "rating": 2.0},
            {"user_id": 4, "item_id": 10, "rating": 5.0},
        ]
    )


def test_fit_surprise_empty_raises() -> None:
    empty_train = pd.DataFrame(columns=["user_id", "item_id", "rating"])
    with pytest.raises(ValueError, match="train_df is empty"):
        fit_surprise(
            train_df=empty_train,
            n_factors=10,
            n_epochs=5,
            lr_all=0.005,
            reg_all=0.02,
            seed=42,
        )


def test_recommend_surprise_excludes_train_items(train_df: pd.DataFrame) -> None:
    model = fit_surprise(
        train_df=train_df,
        n_factors=8,
        n_epochs=3,
        lr_all=0.005,
        reg_all=0.02,
        seed=42,
    )
    all_item_ids = [10, 11, 12, 13, 14]
    recs = recommend_surprise(
        model=model,
        all_item_ids=all_item_ids,
        raw_user_id=1,
        n_recommendations=10,
    )
    seen_items = {10, 11}

    assert all(item_id not in seen_items for item_id, _ in recs)
    assert len(recs) <= len(all_item_ids) - len(seen_items)
    assert recommend_surprise(model=model, all_item_ids=all_item_ids, raw_user_id=1, n_recommendations=0) == []


def test_evaluate_surprise_empty_test_returns_nan(train_df: pd.DataFrame) -> None:
    model = fit_surprise(
        train_df=train_df,
        n_factors=8,
        n_epochs=3,
        lr_all=0.005,
        reg_all=0.02,
        seed=42,
    )
    empty_test = pd.DataFrame(columns=["user_id", "item_id", "rating"])
    metrics = evaluate_surprise(
        model=model,
        test_df=empty_test,
        all_item_ids=[10, 11, 12, 13, 14],
        ranking_k=10,
        relevance_threshold=4.0,
    )

    assert metrics["eligible_users"] == 0
    assert math.isnan(float(metrics["rmse"]))
    assert math.isnan(float(metrics["precision_at_k"]))


def test_evaluate_surprise_nonempty_returns_finite_rmse(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    model = fit_surprise(
        train_df=train_df,
        n_factors=8,
        n_epochs=3,
        lr_all=0.005,
        reg_all=0.02,
        seed=42,
    )
    metrics = evaluate_surprise(
        model=model,
        test_df=test_df,
        all_item_ids=[10, 11, 12, 13, 14],
        ranking_k=3,
        relevance_threshold=4.0,
    )
    assert np.isfinite(float(metrics["rmse"]))
