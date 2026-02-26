"""MovieLens 100K download, loading, mapping, and split utilities."""

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests


ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML_100K_DIRNAME = "ml-100k"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def download_movielens_100k(data_dir: Path | str = DEFAULT_DATA_DIR) -> Path:
    """Download and extract MovieLens 100K if needed, then return dataset directory."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = data_dir / ML_100K_DIRNAME
    ratings_path = dataset_dir / "u.data"
    if ratings_path.exists():
        return dataset_dir

    zip_path = data_dir / "ml-100k.zip"
    with requests.get(ML_100K_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with zip_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1_048_576):
                if chunk:
                    output.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(data_dir)

    if not ratings_path.exists():
        raise FileNotFoundError("MovieLens 100K extraction failed: u.data not found.")
    return dataset_dir


def load_ratings_df(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    max_users: int = 0,
    max_interactions: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load ratings and add contiguous zero-based user/item indices.

    Optional caps are applied before index mapping:
    - max_users: sample users uniformly without replacement (0 = no cap)
    - max_interactions: sample interactions uniformly without replacement (0 = no cap)
    """
    dataset_dir = download_movielens_100k(data_dir)
    ratings_df = pd.read_csv(
        dataset_dir / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    ratings_df = ratings_df.astype(
        {
            "user_id": "int32",
            "item_id": "int32",
            "rating": "float32",
            "timestamp": "int64",
        }
    )

    rng = np.random.default_rng(seed)
    if max_users > 0:
        unique_users = ratings_df["user_id"].drop_duplicates().to_numpy()
        if max_users < unique_users.size:
            selected_users = rng.choice(unique_users, size=max_users, replace=False)
            ratings_df = ratings_df[ratings_df["user_id"].isin(selected_users)]

    if max_interactions > 0 and max_interactions < len(ratings_df):
        sampled_idx = rng.choice(ratings_df.index.to_numpy(), size=max_interactions, replace=False)
        ratings_df = ratings_df.loc[sampled_idx]

    ratings_df = ratings_df.sort_values(["user_id", "item_id", "timestamp"]).reset_index(drop=True)

    unique_user_ids = np.sort(ratings_df["user_id"].unique())
    unique_item_ids = np.sort(ratings_df["item_id"].unique())
    user_to_idx = {raw_id: idx for idx, raw_id in enumerate(unique_user_ids)}
    item_to_idx = {raw_id: idx for idx, raw_id in enumerate(unique_item_ids)}

    ratings_df["user_idx"] = ratings_df["user_id"].map(user_to_idx).astype("int32")
    ratings_df["item_idx"] = ratings_df["item_id"].map(item_to_idx).astype("int32")
    return ratings_df[
        ["user_id", "item_id", "rating", "timestamp", "user_idx", "item_idx"]
    ].reset_index(drop=True)


def load_movies_df(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load movie IDs and titles from MovieLens 100K."""
    dataset_dir = download_movielens_100k(data_dir)
    movies_df = pd.read_csv(
        dataset_dir / "u.item",
        sep="|",
        names=[
            "item_id",
            "title",
            "release_date",
            "video_release_date",
            "imdb_url",
            "unknown",
            "action",
            "adventure",
            "animation",
            "childrens",
            "comedy",
            "crime",
            "documentary",
            "drama",
            "fantasy",
            "film_noir",
            "horror",
            "musical",
            "mystery",
            "romance",
            "sci_fi",
            "thriller",
            "war",
            "western",
        ],
        usecols=["item_id", "title"],
        encoding="latin-1",
        engine="python",
    )
    return movies_df.astype({"item_id": "int32"})


def load_all_raw_user_ids(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[int]:
    """Load the full user-id list from ML-100K (ignores optional caps)."""
    dataset_dir = download_movielens_100k(data_dir)
    users = pd.read_csv(
        dataset_dir / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        usecols=["user_id"],
        engine="python",
    )["user_id"].drop_duplicates()
    return sorted(users.astype(int).tolist())


def make_train_test_split(
    ratings_df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split interactions per user while guaranteeing each user has >=1 train interaction.

    Logic:
    1) For each user, force one random interaction into train.
    2) For remaining interactions, assign floor(test_size * user_interactions) to test.
    3) Put the rest into train.
    """
    if not 0.0 <= test_size < 1.0:
        raise ValueError("test_size must be in [0.0, 1.0).")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for _, user_group in ratings_df.groupby("user_idx", sort=False):
        user_indices = user_group.index.to_numpy()
        if user_indices.size == 1:
            train_indices.append(int(user_indices[0]))
            continue

        forced_train = int(rng.choice(user_indices))
        train_indices.append(forced_train)
        remaining = user_indices[user_indices != forced_train]

        target_test_count = int(np.floor(test_size * user_indices.size))
        n_test = int(np.clip(target_test_count, 0, remaining.size))

        if n_test == 0:
            train_indices.extend(remaining.astype(int).tolist())
            continue

        sampled_test = rng.choice(remaining, size=n_test, replace=False)
        sampled_test_set = set(sampled_test.astype(int).tolist())
        test_indices.extend(sampled_test_set)
        train_indices.extend([int(i) for i in remaining if int(i) not in sampled_test_set])

    train_df = ratings_df.loc[train_indices].copy().reset_index(drop=True)
    test_df = ratings_df.loc[test_indices].copy().reset_index(drop=True)
    return train_df, test_df


def compute_dataset_stats(ratings_df: pd.DataFrame) -> dict[str, float | int | dict[int, int]]:
    """Compute basic dataset summary statistics."""
    n_interactions = int(len(ratings_df))
    n_users = int(ratings_df["user_idx"].nunique())
    n_items = int(ratings_df["item_idx"].nunique())
    total_cells = n_users * n_items
    sparsity = 1.0 - (n_interactions / total_cells) if total_cells else 0.0

    histogram = (
        ratings_df["rating"]
        .round(0)
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "n_interactions": n_interactions,
        "n_users": n_users,
        "n_items": n_items,
        "sparsity": float(sparsity),
        "rating_histogram": {int(k): int(v) for k, v in histogram.items()},
    }

