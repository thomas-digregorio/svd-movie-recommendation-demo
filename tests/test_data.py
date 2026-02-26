from __future__ import annotations

import zipfile

import pandas as pd
import pytest

from src.data import _safe_extractall, compute_dataset_stats, make_train_test_split


def _sample_ratings_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"interaction_id": 1, "user_id": 10, "item_id": 100, "rating": 5.0, "timestamp": 1, "user_idx": 0, "item_idx": 0},
            {"interaction_id": 2, "user_id": 10, "item_id": 101, "rating": 4.0, "timestamp": 2, "user_idx": 0, "item_idx": 1},
            {"interaction_id": 3, "user_id": 10, "item_id": 102, "rating": 3.0, "timestamp": 3, "user_idx": 0, "item_idx": 2},
            {"interaction_id": 4, "user_id": 11, "item_id": 100, "rating": 4.0, "timestamp": 4, "user_idx": 1, "item_idx": 0},
            {"interaction_id": 5, "user_id": 11, "item_id": 103, "rating": 2.0, "timestamp": 5, "user_idx": 1, "item_idx": 3},
            {"interaction_id": 6, "user_id": 12, "item_id": 104, "rating": 5.0, "timestamp": 6, "user_idx": 2, "item_idx": 4},
        ]
    )


def test_make_train_test_split_guarantees_train_coverage() -> None:
    ratings_df = _sample_ratings_df()
    train_df, test_df = make_train_test_split(ratings_df, test_size=0.5, seed=7)

    assert set(train_df["user_idx"].unique()) == set(ratings_df["user_idx"].unique())
    assert len(train_df) + len(test_df) == len(ratings_df)

    full_ids = set(ratings_df["interaction_id"].tolist())
    train_ids = set(train_df["interaction_id"].tolist())
    test_ids = set(test_df["interaction_id"].tolist())
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == full_ids


def test_compute_dataset_stats_basic_counts() -> None:
    ratings_df = _sample_ratings_df()
    stats = compute_dataset_stats(ratings_df)

    assert stats["n_interactions"] == 6
    assert stats["n_users"] == 3
    assert stats["n_items"] == 5
    assert stats["rating_histogram"][5] == 2


def test_safe_extractall_allows_normal_zip(tmp_path) -> None:
    zip_path = tmp_path / "valid.zip"
    target_dir = tmp_path / "extract"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("ml-100k/u.data", "1\t1\t5\t1\n")

    with zipfile.ZipFile(zip_path, "r") as archive:
        _safe_extractall(archive, target_dir)

    assert (target_dir / "ml-100k" / "u.data").exists()


def test_safe_extractall_rejects_path_traversal(tmp_path) -> None:
    zip_path = tmp_path / "invalid.zip"
    target_dir = tmp_path / "extract"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../evil.txt", "malicious")

    with zipfile.ZipFile(zip_path, "r") as archive:
        with pytest.raises(ValueError, match="Unsafe zip entry path"):
            _safe_extractall(archive, target_dir)
