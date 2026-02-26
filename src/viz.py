"""Matplotlib plotting helpers for Streamlit views."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


METHOD_LABELS = {
    "truncated_svd": "Truncated SVD (Centered Sparse Matrix)",
    "surprise_svd": "Surprise SVD (MF on Observed Ratings)",
}
METHOD_COLORS = {
    "truncated_svd": "#1b9e77",
    "surprise_svd": "#d95f02",
}


def plot_rating_histogram(rating_histogram: dict[int, int]) -> plt.Figure:
    """Create a histogram bar chart from rating-count dictionary."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ratings = sorted(rating_histogram.keys())
    counts = [rating_histogram[r] for r in ratings]
    ax.bar(ratings, counts, color="#4c78a8", width=0.65)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Ratings Distribution")
    ax.set_xticks(ratings)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_metric_vs_k(
    results_df: pd.DataFrame,
    metric_col: str,
    y_label: str,
    title: str,
) -> plt.Figure:
    """Plot a metric-vs-k curve for each method."""
    fig, ax = plt.subplots(figsize=(8, 4))

    for method, method_df in results_df.groupby("method"):
        ordered = method_df.sort_values("k")
        ax.plot(
            ordered["k"],
            ordered[metric_col],
            marker="o",
            linewidth=2,
            color=METHOD_COLORS.get(method),
            label=METHOD_LABELS.get(method, method),
        )

    ax.set_xlabel("Latent Rank (k / n_factors)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig

