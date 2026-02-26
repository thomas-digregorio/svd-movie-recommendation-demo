"""Quick CLI sanity check for both recommender methods."""

from __future__ import annotations

from pprint import pprint

from src.data import load_movies_df, load_ratings_df, make_train_test_split
from src.svd_surprise import (
    evaluate_surprise,
    fit_surprise,
    get_surprise_backend_label,
    recommend_surprise,
)
from src.svd_truncated import evaluate_truncated, fit_truncated, recommend_truncated


def main() -> None:
    seed = 42
    test_size = 0.2
    relevance_threshold = 4.0
    ranking_k = 10
    latent_dim = 20
    n_recommendations = 5

    print("Loading MovieLens 100K...")
    ratings_df = load_ratings_df(seed=seed)
    movies_df = load_movies_df()
    train_df, test_df = make_train_test_split(
        ratings_df=ratings_df,
        test_size=test_size,
        seed=seed,
    )

    n_users = int(ratings_df["user_idx"].nunique())
    n_items = int(ratings_df["item_idx"].nunique())
    all_item_ids = sorted(ratings_df["item_id"].astype(int).unique().tolist())
    title_by_item = movies_df.set_index("item_id")["title"].astype(str).to_dict()

    print(f"Users: {n_users} | Items: {n_items}")
    print(f"Train interactions: {len(train_df)} | Test interactions: {len(test_df)}")

    print("\nFitting TruncatedSVD...")
    truncated_model = fit_truncated(
        train_df=train_df,
        n_users=n_users,
        n_items=n_items,
        k=latent_dim,
        n_iter=7,
        seed=seed,
    )
    truncated_metrics = evaluate_truncated(
        model=truncated_model,
        train_df=train_df,
        test_df=test_df,
        ranking_k=ranking_k,
        relevance_threshold=relevance_threshold,
    )
    print("TruncatedSVD metrics:")
    pprint(truncated_metrics)

    print("\nFitting Surprise SVD...")
    surprise_model = fit_surprise(
        train_df=train_df,
        n_factors=latent_dim,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        seed=seed,
    )
    print(f"Surprise backend: {get_surprise_backend_label(surprise_model)}")
    surprise_metrics = evaluate_surprise(
        model=surprise_model,
        train_df=train_df,
        test_df=test_df,
        all_item_ids=all_item_ids,
        ranking_k=ranking_k,
        relevance_threshold=relevance_threshold,
    )
    print("Surprise SVD metrics:")
    pprint(surprise_metrics)

    sample_raw_user_id = int(ratings_df["user_id"].iloc[0])
    print(f"\nSample recommendations for raw user_id={sample_raw_user_id}")

    raw_to_user_idx = (
        ratings_df[["user_id", "user_idx"]]
        .drop_duplicates()
        .set_index("user_id")["user_idx"]
        .astype(int)
        .to_dict()
    )
    item_idx_to_raw = (
        ratings_df[["item_idx", "item_id"]]
        .drop_duplicates()
        .set_index("item_idx")["item_id"]
        .astype(int)
        .to_dict()
    )

    truncated_user_idx = int(raw_to_user_idx[sample_raw_user_id])
    trunc_recs = recommend_truncated(
        model=truncated_model,
        user_idx=truncated_user_idx,
        n_recommendations=n_recommendations,
        exclude_train=True,
    )
    print("\nTruncatedSVD top recommendations:")
    for item_idx, score in trunc_recs:
        raw_item_id = int(item_idx_to_raw[item_idx])
        print(f"- {title_by_item.get(raw_item_id, f'Item {raw_item_id}')} ({score:.3f})")

    surprise_recs = recommend_surprise(
        model=surprise_model,
        train_df=train_df,
        all_item_ids=all_item_ids,
        raw_user_id=sample_raw_user_id,
        n_recommendations=n_recommendations,
    )
    print("\nSurprise SVD top recommendations:")
    for raw_item_id, score in surprise_recs:
        print(f"- {title_by_item.get(raw_item_id, f'Item {raw_item_id}')} ({score:.3f})")

    print("\nSanity check complete.")


if __name__ == "__main__":
    main()
