from __future__ import annotations

import pandas as pd
import pytest

import app
from src.config import AppConfig


def test_choose_best_model_key_ignores_nan_rows() -> None:
    results_df = pd.DataFrame(
        [
            {
                "_model_key": 10,
                "method": "truncated_svd",
                "ndcg_at_k": float("nan"),
                "map_at_k": float("nan"),
                "mrr_at_k": float("nan"),
                "precision_at_k": float("nan"),
                "recall_at_k": float("nan"),
                "rmse": float("nan"),
            },
            {
                "_model_key": 20,
                "method": "truncated_svd",
                "ndcg_at_k": 0.25,
                "map_at_k": 0.10,
                "mrr_at_k": 0.40,
                "precision_at_k": 0.20,
                "recall_at_k": 0.30,
                "rmse": 0.95,
            },
        ]
    )

    assert app._choose_best_model_key(results_df, method="truncated_svd") == 20


def test_choose_best_model_key_returns_none_when_no_valid_rows() -> None:
    results_df = pd.DataFrame(
        [
            {
                "_model_key": 10,
                "method": "surprise_svd",
                "ndcg_at_k": float("nan"),
                "map_at_k": float("nan"),
                "mrr_at_k": float("nan"),
                "precision_at_k": float("nan"),
                "recall_at_k": float("nan"),
                "rmse": float("nan"),
            }
        ]
    )

    assert app._choose_best_model_key(results_df, method="surprise_svd") is None


def test_compute_outputs_raises_on_empty_test_before_fitting(monkeypatch) -> None:
    ratings_df = pd.DataFrame(
        [
            {
                "user_id": 1,
                "item_id": 10,
                "rating": 5.0,
                "timestamp": 1,
                "user_idx": 0,
                "item_idx": 0,
            }
        ]
    )
    train_df = ratings_df.copy()
    empty_test_df = pd.DataFrame(columns=ratings_df.columns)
    stats = {
        "n_interactions": 1,
        "n_users": 1,
        "n_items": 1,
        "sparsity": 0.0,
        "rating_histogram": {5: 1},
    }

    called = {"truncated": False, "surprise": False}

    def _fit_truncated_stub(**kwargs):
        called["truncated"] = True
        return {}

    def _fit_surprise_stub(**kwargs):
        called["surprise"] = True
        return {}

    monkeypatch.setattr(
        app,
        "_load_movies_cached",
        lambda: pd.DataFrame([{"item_id": 10, "title": "Movie"}]),
    )
    monkeypatch.setattr(app, "_load_all_users_cached", lambda: [1])
    monkeypatch.setattr(
        app,
        "_load_ratings_and_split_cached",
        lambda **kwargs: (ratings_df, train_df, empty_test_df, stats),
    )
    monkeypatch.setattr(app, "_fit_truncated_models_cached", _fit_truncated_stub)
    monkeypatch.setattr(app, "_fit_surprise_models_cached", _fit_surprise_stub)

    with pytest.raises(ValueError, match="Test split is empty"):
        app._compute_outputs(config=AppConfig(), model_token=0)

    assert called["truncated"] is False
    assert called["surprise"] is False
