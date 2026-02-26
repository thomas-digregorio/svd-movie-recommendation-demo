"""Configuration objects and parsing helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


DEFAULT_K_LIST = [10, 20, 50, 100]
SPLIT_LOGIC_DESCRIPTION = (
    "Per user, one interaction is forced into train first. "
    "The remaining interactions are split by test_size. "
    "This guarantees every modeled user has at least one train interaction."
)


def parse_int_list(text: str, default: list[int]) -> list[int]:
    """Parse a comma-separated integer list with order-preserving deduplication."""
    cleaned = text.strip()
    if not cleaned:
        return list(default)

    values: list[int] = []
    for token in cleaned.split(","):
        piece = token.strip()
        if not piece:
            continue
        value = int(piece)
        if value <= 0:
            raise ValueError("List values must be positive integers.")
        values.append(value)

    if not values:
        raise ValueError("No integer values were provided.")

    return list(dict.fromkeys(values))


@dataclass(frozen=True)
class TruncatedSVDConfig:
    k_list: list[int] = field(default_factory=lambda: list(DEFAULT_K_LIST))
    n_iter: int = 7


@dataclass(frozen=True)
class SurpriseSVDConfig:
    n_factors_list: list[int] = field(default_factory=lambda: list(DEFAULT_K_LIST))
    n_epochs: int = 20
    lr_all: float = 0.005
    reg_all: float = 0.02


@dataclass(frozen=True)
class AppConfig:
    seed: int = 42
    test_size: float = 0.2
    relevance_threshold: float = 4.0
    ranking_k: int = 10
    n_recommendations: int = 10
    max_users: int = 0
    max_interactions: int = 0
    truncated: TruncatedSVDConfig = field(default_factory=TruncatedSVDConfig)
    surprise: SurpriseSVDConfig = field(default_factory=SurpriseSVDConfig)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["split_logic"] = SPLIT_LOGIC_DESCRIPTION
        return payload


def config_from_sidebar(inputs: dict[str, Any]) -> AppConfig:
    """Create a typed config object from sidebar widget values."""
    seed = int(inputs["seed"])
    return AppConfig(
        seed=seed,
        test_size=float(inputs["test_size"]),
        relevance_threshold=float(inputs["relevance_threshold"]),
        ranking_k=int(inputs["ranking_k"]),
        n_recommendations=int(inputs["n_recommendations"]),
        max_users=int(inputs["max_users"]),
        max_interactions=int(inputs["max_interactions"]),
        truncated=TruncatedSVDConfig(
            k_list=parse_int_list(str(inputs["truncated_k_list"]), DEFAULT_K_LIST),
            n_iter=int(inputs["truncated_n_iter"]),
        ),
        surprise=SurpriseSVDConfig(
            n_factors_list=parse_int_list(
                str(inputs["surprise_n_factors_list"]),
                DEFAULT_K_LIST,
            ),
            n_epochs=int(inputs["surprise_n_epochs"]),
            lr_all=float(inputs["surprise_lr_all"]),
            reg_all=float(inputs["surprise_reg_all"]),
        ),
    )

