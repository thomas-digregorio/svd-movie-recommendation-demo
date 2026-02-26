"""Streamlit app comparing TruncatedSVD vs Surprise SVD on MovieLens 100K."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import streamlit as st

from src.config import AppConfig, SPLIT_LOGIC_DESCRIPTION, config_from_sidebar
from src.data import (
    compute_dataset_stats,
    load_all_raw_user_ids,
    load_movies_df,
    load_ratings_df,
    make_train_test_split,
)
from src.svd_surprise import (
    evaluate_surprise,
    fit_surprise,
    get_surprise_backend_label,
    recommend_surprise,
)
from src.svd_truncated import evaluate_truncated, fit_truncated, recommend_truncated
from src.viz import plot_metric_vs_k, plot_rating_histogram


st.set_page_config(page_title="SVD Movie Recommender Demo", layout="wide")


def _init_ui_state(defaults: AppConfig) -> None:
    widget_defaults = {
        "ui_seed": defaults.seed,
        "ui_test_size": defaults.test_size,
        "ui_relevance_threshold": defaults.relevance_threshold,
        "ui_ranking_k": defaults.ranking_k,
        "ui_n_recommendations": defaults.n_recommendations,
        "ui_max_users": defaults.max_users,
        "ui_max_interactions": defaults.max_interactions,
        "ui_truncated_k_list": ",".join(str(k) for k in defaults.truncated.k_list),
        "ui_truncated_n_iter": defaults.truncated.n_iter,
        "ui_surprise_n_factors_list": ",".join(
            str(k) for k in defaults.surprise.n_factors_list
        ),
        "ui_surprise_n_epochs": defaults.surprise.n_epochs,
        "ui_surprise_lr_all": defaults.surprise.lr_all,
        "ui_surprise_reg_all": defaults.surprise.reg_all,
    }
    for key, value in widget_defaults.items():
        st.session_state.setdefault(key, value)


def _build_sidebar_inputs() -> tuple[dict[str, Any], bool]:
    with st.sidebar:
        st.header("Configuration")
        seed = st.number_input("seed", min_value=0, step=1, key="ui_seed")
        test_size = st.slider("test_size", min_value=0.05, max_value=0.5, step=0.01, key="ui_test_size")
        relevance_threshold = st.slider(
            "relevance_threshold",
            min_value=1.0,
            max_value=5.0,
            step=0.5,
            key="ui_relevance_threshold",
        )
        ranking_k = st.slider("Ranking K", min_value=1, max_value=50, step=1, key="ui_ranking_k")
        n_recommendations = st.slider(
            "N recommendations",
            min_value=1,
            max_value=50,
            step=1,
            key="ui_n_recommendations",
        )
        max_users = st.number_input("max_users (0 = no cap)", min_value=0, step=1, key="ui_max_users")
        max_interactions = st.number_input(
            "max_interactions (0 = no cap)",
            min_value=0,
            step=1,
            key="ui_max_interactions",
        )

        st.subheader("Method A: TruncatedSVD")
        truncated_k_list = st.text_input("k_list", key="ui_truncated_k_list")
        truncated_n_iter = st.slider("n_iter", min_value=3, max_value=20, step=1, key="ui_truncated_n_iter")

        st.subheader("Method B: Surprise SVD")
        surprise_n_factors_list = st.text_input("n_factors_list", key="ui_surprise_n_factors_list")
        surprise_n_epochs = st.slider("n_epochs", min_value=5, max_value=100, step=1, key="ui_surprise_n_epochs")
        surprise_lr_all = st.number_input("lr_all", min_value=0.0001, max_value=0.1, step=0.0005, format="%.4f", key="ui_surprise_lr_all")
        surprise_reg_all = st.number_input("reg_all", min_value=0.0001, max_value=1.0, step=0.001, format="%.3f", key="ui_surprise_reg_all")

        recompute_clicked = st.button("Recompute", type="primary", use_container_width=True)

    inputs = {
        "seed": seed,
        "test_size": test_size,
        "relevance_threshold": relevance_threshold,
        "ranking_k": ranking_k,
        "n_recommendations": n_recommendations,
        "max_users": max_users,
        "max_interactions": max_interactions,
        "truncated_k_list": truncated_k_list,
        "truncated_n_iter": truncated_n_iter,
        "surprise_n_factors_list": surprise_n_factors_list,
        "surprise_n_epochs": surprise_n_epochs,
        "surprise_lr_all": surprise_lr_all,
        "surprise_reg_all": surprise_reg_all,
    }
    return inputs, recompute_clicked


def _model_signature(config: AppConfig) -> tuple[Any, ...]:
    return (
        config.seed,
        config.test_size,
        config.max_users,
        config.max_interactions,
        tuple(config.truncated.k_list),
        config.truncated.n_iter,
        tuple(config.surprise.n_factors_list),
        config.surprise.n_epochs,
        config.surprise.lr_all,
        config.surprise.reg_all,
    )


def _choose_best_model_key(results_df: pd.DataFrame, method: str) -> int | None:
    subset = results_df[results_df["method"] == method]
    if subset.empty:
        return None
    sortable = subset.dropna(
        subset=[
            "ndcg_at_k",
            "map_at_k",
            "mrr_at_k",
            "precision_at_k",
            "recall_at_k",
            "rmse",
        ]
    )
    if sortable.empty:
        return None
    best = sortable.sort_values(
        by=[
            "ndcg_at_k",
            "map_at_k",
            "mrr_at_k",
            "precision_at_k",
            "recall_at_k",
            "rmse",
        ],
        ascending=[False, False, False, False, False, True],
    ).iloc[0]
    return int(best["_model_key"])


@st.cache_data(show_spinner=False)
def _load_movies_cached() -> pd.DataFrame:
    return load_movies_df()


@st.cache_data(show_spinner=False)
def _load_all_users_cached() -> list[int]:
    return load_all_raw_user_ids()


@st.cache_data(show_spinner=False)
def _load_ratings_and_split_cached(
    max_users: int,
    max_interactions: int,
    seed: int,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ratings_df = load_ratings_df(
        max_users=max_users,
        max_interactions=max_interactions,
        seed=seed,
    )
    train_df, test_df = make_train_test_split(
        ratings_df=ratings_df,
        test_size=test_size,
        seed=seed,
    )
    stats = compute_dataset_stats(ratings_df)
    return ratings_df, train_df, test_df, stats


@st.cache_resource(show_spinner=False)
def _fit_truncated_models_cached(
    train_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    k_list: tuple[int, ...],
    n_iter: int,
    seed: int,
    recompute_token: int,
) -> dict[int, dict[str, Any]]:
    del recompute_token
    output: dict[int, dict[str, Any]] = {}
    for requested_k in k_list:
        start = time.perf_counter()
        model = fit_truncated(
            train_df=train_df,
            n_users=n_users,
            n_items=n_items,
            k=int(requested_k),
            n_iter=n_iter,
            seed=seed,
        )
        output[int(requested_k)] = {
            "model": model,
            "fit_runtime_sec": time.perf_counter() - start,
        }
    return output


@st.cache_resource(show_spinner=False)
def _fit_surprise_models_cached(
    train_df: pd.DataFrame,
    n_factors_list: tuple[int, ...],
    n_epochs: int,
    lr_all: float,
    reg_all: float,
    seed: int,
    recompute_token: int,
) -> dict[int, dict[str, Any]]:
    del recompute_token
    output: dict[int, dict[str, Any]] = {}
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
        output[int(n_factors)] = {
            "model": model,
            "fit_runtime_sec": time.perf_counter() - start,
        }
    return output


def _compute_outputs(config: AppConfig, model_token: int) -> dict[str, Any]:
    movies_df = _load_movies_cached()
    all_raw_user_ids = _load_all_users_cached()
    ratings_df, train_df, test_df, dataset_stats = _load_ratings_and_split_cached(
        max_users=config.max_users,
        max_interactions=config.max_interactions,
        seed=config.seed,
        test_size=config.test_size,
    )
    if test_df.empty:
        raise ValueError(
            "Test split is empty under current caps/split settings. "
            "Increase max_interactions or test_size."
        )

    n_users = int(ratings_df["user_idx"].nunique())
    n_items = int(ratings_df["item_idx"].nunique())
    modeled_item_ids = sorted(ratings_df["item_id"].astype(int).unique().tolist())

    truncated_models = _fit_truncated_models_cached(
        train_df=train_df,
        n_users=n_users,
        n_items=n_items,
        k_list=tuple(config.truncated.k_list),
        n_iter=config.truncated.n_iter,
        seed=config.seed,
        recompute_token=model_token,
    )
    surprise_models = _fit_surprise_models_cached(
        train_df=train_df,
        n_factors_list=tuple(config.surprise.n_factors_list),
        n_epochs=config.surprise.n_epochs,
        lr_all=config.surprise.lr_all,
        reg_all=config.surprise.reg_all,
        seed=config.seed,
        recompute_token=model_token,
    )

    rows: list[dict[str, Any]] = []
    for requested_k in sorted(truncated_models):
        payload = truncated_models[requested_k]
        start = time.perf_counter()
        eval_metrics = evaluate_truncated(
            model=payload["model"],
            train_df=train_df,
            test_df=test_df,
            ranking_k=config.ranking_k,
            relevance_threshold=config.relevance_threshold,
        )
        eval_runtime = time.perf_counter() - start
        rows.append(
            {
                "_model_key": int(requested_k),
                "method": "truncated_svd",
                "k": int(payload["model"]["effective_k"]),
                "rmse": float(eval_metrics["rmse"]),
                "precision_at_k": float(eval_metrics["precision_at_k"]),
                "recall_at_k": float(eval_metrics["recall_at_k"]),
                "ndcg_at_k": float(eval_metrics["ndcg_at_k"]),
                "map_at_k": float(eval_metrics["map_at_k"]),
                "mrr_at_k": float(eval_metrics["mrr_at_k"]),
                "eligible_users": int(eval_metrics["eligible_users"]),
                "runtime_sec": float(payload["fit_runtime_sec"] + eval_runtime),
            }
        )

    for requested_k in sorted(surprise_models):
        payload = surprise_models[requested_k]
        start = time.perf_counter()
        eval_metrics = evaluate_surprise(
            model=payload["model"],
            test_df=test_df,
            all_item_ids=modeled_item_ids,
            ranking_k=config.ranking_k,
            relevance_threshold=config.relevance_threshold,
        )
        eval_runtime = time.perf_counter() - start
        rows.append(
            {
                "_model_key": int(requested_k),
                "method": "surprise_svd",
                "k": int(requested_k),
                "rmse": float(eval_metrics["rmse"]),
                "precision_at_k": float(eval_metrics["precision_at_k"]),
                "recall_at_k": float(eval_metrics["recall_at_k"]),
                "ndcg_at_k": float(eval_metrics["ndcg_at_k"]),
                "map_at_k": float(eval_metrics["map_at_k"]),
                "mrr_at_k": float(eval_metrics["mrr_at_k"]),
                "eligible_users": int(eval_metrics["eligible_users"]),
                "runtime_sec": float(payload["fit_runtime_sec"] + eval_runtime),
            }
        )

    results_df = pd.DataFrame(rows)
    best_truncated_key = _choose_best_model_key(results_df, method="truncated_svd")
    best_surprise_key = _choose_best_model_key(results_df, method="surprise_svd")

    raw_user_to_idx = (
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
    movie_title_by_item = (
        movies_df.set_index("item_id")["title"].astype(str).to_dict()
    )

    return {
        "movies_df": movies_df,
        "all_raw_user_ids": all_raw_user_ids,
        "ratings_df": ratings_df,
        "train_df": train_df,
        "test_df": test_df,
        "dataset_stats": dataset_stats,
        "results_df": results_df,
        "truncated_models": truncated_models,
        "surprise_models": surprise_models,
        "surprise_backend": (
            get_surprise_backend_label(next(iter(surprise_models.values()))["model"])
            if surprise_models
            else "unavailable"
        ),
        "best_truncated_key": best_truncated_key,
        "best_surprise_key": best_surprise_key,
        "raw_user_to_idx": raw_user_to_idx,
        "item_idx_to_raw": item_idx_to_raw,
        "modeled_item_ids": modeled_item_ids,
        "movie_title_by_item": movie_title_by_item,
    }


def _display_recommendation_table(
    recommendations: list[tuple[int, float]],
    title_by_item: dict[int, str],
) -> pd.DataFrame:
    rows = []
    for item_id, score in recommendations:
        rows.append(
            {
                "item_id": int(item_id),
                "title": title_by_item.get(int(item_id), f"Item {item_id}"),
                "predicted_score": round(float(score), 4),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.title("SVD Movie Recommendation Demo (MovieLens 100K)")
    st.caption(
        "Compare centered sparse TruncatedSVD and Surprise matrix factorization "
        "on identical train/test splits."
    )

    if "active_config" not in st.session_state:
        st.session_state["active_config"] = AppConfig()
    if "model_signature" not in st.session_state:
        st.session_state["model_signature"] = _model_signature(st.session_state["active_config"])
    if "model_token" not in st.session_state:
        st.session_state["model_token"] = 0
    if "outputs" not in st.session_state:
        st.session_state["outputs"] = None

    _init_ui_state(st.session_state["active_config"])
    sidebar_inputs, recompute_clicked = _build_sidebar_inputs()

    try:
        pending_config = config_from_sidebar(sidebar_inputs)
    except Exception as exc:
        st.sidebar.error(f"Invalid sidebar input: {exc}")
        st.stop()

    should_compute = False
    if st.session_state["outputs"] is None:
        st.session_state["active_config"] = pending_config
        st.session_state["model_signature"] = _model_signature(pending_config)
        should_compute = True
    elif recompute_clicked:
        prior_signature = st.session_state["model_signature"]
        next_signature = _model_signature(pending_config)
        if next_signature != prior_signature:
            st.session_state["model_token"] += 1
            st.session_state["model_signature"] = next_signature
        st.session_state["active_config"] = pending_config
        should_compute = True

    active_config: AppConfig = st.session_state["active_config"]
    pending_changes = pending_config != active_config
    if pending_changes:
        st.warning("Sidebar changes are pending. Click Recompute to apply them.")

    if should_compute:
        with st.spinner("Running data prep, model fits, and evaluations..."):
            try:
                st.session_state["outputs"] = _compute_outputs(
                    config=active_config,
                    model_token=st.session_state["model_token"],
                )
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

    outputs = st.session_state["outputs"]
    if outputs is None:
        st.error("No run outputs are available.")
        st.stop()

    results_df: pd.DataFrame = outputs["results_df"]
    table_cols = [
        "method",
        "k",
        "rmse",
        "precision_at_k",
        "recall_at_k",
        "ndcg_at_k",
        "map_at_k",
        "mrr_at_k",
        "eligible_users",
        "runtime_sec",
    ]
    display_df = results_df[table_cols].sort_values(["method", "k"]).reset_index(drop=True)

    st.subheader("1) Configs / Assumptions")
    config_payload = active_config.as_dict()
    config_payload["ranking_evaluation_note"] = (
        f"All ranking metrics are computed at fixed K={active_config.ranking_k} while sweeping "
        "model rank (k / n_factors), which isolates model-capacity effects from cutoff effects."
    )
    config_payload["caps_note"] = (
        "Optional caps are applied before splitting. "
        "User dropdown still uses all MovieLens 100K user IDs."
    )
    config_payload["surprise_backend"] = outputs["surprise_backend"]
    config_payload["split_logic_description"] = SPLIT_LOGIC_DESCRIPTION
    st.json(config_payload)

    st.subheader("2) Dataset Summary")
    stats = outputs["dataset_stats"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users (modeled)", stats["n_users"])
    c2.metric("Items (modeled)", stats["n_items"])
    c3.metric("Interactions (modeled)", stats["n_interactions"])
    c4.metric("Sparsity", f"{stats['sparsity']:.4f}")
    st.caption(
        f"Train interactions: {len(outputs['train_df'])} | "
        f"Test interactions: {len(outputs['test_df'])}"
    )
    st.pyplot(plot_rating_histogram(stats["rating_histogram"]))

    st.subheader("3) Results Table")
    st.dataframe(display_df, use_container_width=True)

    st.subheader("4) Plots")
    st.pyplot(plot_metric_vs_k(display_df, "rmse", "RMSE", "RMSE vs Latent Rank"))
    st.pyplot(
        plot_metric_vs_k(
            display_df,
            "precision_at_k",
            f"Precision@{active_config.ranking_k}",
            "Precision@K vs Latent Rank",
        )
    )
    st.pyplot(
        plot_metric_vs_k(
            display_df,
            "recall_at_k",
            f"Recall@{active_config.ranking_k}",
            "Recall@K vs Latent Rank",
        )
    )
    st.pyplot(
        plot_metric_vs_k(
            display_df,
            "ndcg_at_k",
            f"NDCG@{active_config.ranking_k}",
            "NDCG@K vs Latent Rank",
        )
    )
    st.pyplot(
        plot_metric_vs_k(
            display_df,
            "map_at_k",
            f"MAP@{active_config.ranking_k}",
            "MAP@K vs Latent Rank",
        )
    )
    st.pyplot(
        plot_metric_vs_k(
            display_df,
            "mrr_at_k",
            f"MRR@{active_config.ranking_k}",
            "MRR@K vs Latent Rank",
        )
    )

    st.subheader("5) Recommendations Explorer")
    all_user_ids = outputs["all_raw_user_ids"]
    selected_user = st.selectbox("Select raw user_id", options=all_user_ids, index=0)

    user_train_df = outputs["train_df"][outputs["train_df"]["user_id"] == int(selected_user)].copy()
    if user_train_df.empty:
        st.info("No modeled train interactions for this user under current caps/split.")
    else:
        top_train = (
            user_train_df.sort_values(["rating", "timestamp"], ascending=[False, False])
            .head(max(active_config.n_recommendations, 10))
            .merge(outputs["movies_df"], on="item_id", how="left")
            [["item_id", "title", "rating"]]
        )
        st.write("Top-rated TRAIN movies for selected user")
        st.dataframe(top_train, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Truncated SVD Recommendations**")
        best_truncated_key = outputs["best_truncated_key"]
        if best_truncated_key is None:
            st.info("No Truncated SVD model available.")
        else:
            st.caption(
                f"Best model selected by NDCG/MAP/MRR tie-breakers: requested k={best_truncated_key}"
            )
            raw_user_to_idx = outputs["raw_user_to_idx"]
            if int(selected_user) not in raw_user_to_idx:
                st.info("Selected user not in modeled subset (likely due max_users cap).")
            else:
                user_idx = int(raw_user_to_idx[int(selected_user)])
                truncated_model = outputs["truncated_models"][best_truncated_key]["model"]
                recs_idx = recommend_truncated(
                    model=truncated_model,
                    user_idx=user_idx,
                    n_recommendations=active_config.n_recommendations,
                    exclude_train=True,
                )
                recs_raw = [
                    (outputs["item_idx_to_raw"].get(item_idx, -1), score)
                    for item_idx, score in recs_idx
                ]
                rec_df = _display_recommendation_table(
                    recs_raw,
                    outputs["movie_title_by_item"],
                )
                st.dataframe(rec_df, use_container_width=True)

    with col_right:
        st.markdown("**Surprise SVD Recommendations**")
        best_surprise_key = outputs["best_surprise_key"]
        if best_surprise_key is None:
            st.info("No Surprise SVD model available.")
        else:
            st.caption(
                f"Best model selected by NDCG/MAP/MRR tie-breakers: n_factors={best_surprise_key}"
            )
            raw_user_to_idx = outputs["raw_user_to_idx"]
            if int(selected_user) not in raw_user_to_idx:
                st.info("Selected user not in modeled subset (likely due max_users cap).")
            else:
                surprise_model = outputs["surprise_models"][best_surprise_key]["model"]
                recs_raw = recommend_surprise(
                    model=surprise_model,
                    all_item_ids=outputs["modeled_item_ids"],
                    raw_user_id=int(selected_user),
                    n_recommendations=active_config.n_recommendations,
                )
                rec_df = _display_recommendation_table(
                    recs_raw,
                    outputs["movie_title_by_item"],
                )
                st.dataframe(rec_df, use_container_width=True)

    st.subheader("6) Notes")
    st.markdown(
        f"""
        - `truncated_svd` builds a centered user-item matrix with missing entries as zeros after centering,
          then applies low-rank approximation.
        - `surprise_svd` learns latent factors directly from observed ratings only.
        - Surprise backend in this run: `{outputs["surprise_backend"]}`.
        - This is offline evaluation: ranking metrics are estimated on held-out interactions and may differ from online behavior.
        - Candidate set for ranking metrics is each user's items not seen in train.
        - Relevance is defined as test rating >= relevance threshold.
        - The low-rank-basis idea mirrors eigenfaces: both methods compress high-dimensional signals into latent components.
        """
    )


if __name__ == "__main__":
    main()
