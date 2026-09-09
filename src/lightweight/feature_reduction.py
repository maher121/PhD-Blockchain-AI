"""Transparent train-only feature reduction for V0.5."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.ai.model_utils import validate_selected_features
from src.security.ground_truth import ATTACK_METADATA_COLUMNS

V05_FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {
        "is_attack",
        "attack_type",
        "attack_severity",
        "original_value",
        "modified_value",
        "experiment_id",
        "attack_rate",
        "target",
        "prediction",
        "anomaly_score",
        *ATTACK_METADATA_COLUMNS,
    }
)


def validate_model_features(feature_names: Iterable[str]) -> None:
    """Fail closed when a forbidden V0.5 field is selected as model input."""
    names = list(feature_names)
    forbidden = sorted(set(names) & V05_FORBIDDEN_FEATURES)
    if forbidden:
        raise ValueError(f"Forbidden V0.5 model input columns: {forbidden}")
    validate_selected_features(names)


class CorrelationFeatureReducer:
    """Build nested subsets by minimizing train-only feature redundancy."""

    method = "train_correlation_redundancy_ranking"

    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = list(feature_names)
        validate_model_features(self.feature_names)
        self.ranking_: list[str] | None = None
        self.mean_absolute_correlations_: dict[str, float] | None = None
        self.max_selected_correlations_: dict[str, float] | None = None

    def fit(self, training_features: pd.DataFrame) -> "CorrelationFeatureReducer":
        missing = [name for name in self.feature_names if name not in training_features]
        if missing:
            raise ValueError(f"Training data is missing candidate features: {missing}")
        values = training_features[self.feature_names].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Feature ranking received missing or infinite training values.")
        original_position = {name: index for index, name in enumerate(self.feature_names)}
        correlations = (
            pd.DataFrame(values, columns=self.feature_names)
            .corr(method="pearson")
            .abs()
            .fillna(0.0)
        )
        for index in range(len(correlations)):
            correlations.iat[index, index] = 0.0
        mean_correlations = correlations.mean(axis=1)
        self.mean_absolute_correlations_ = {
            name: float(mean_correlations[name]) for name in self.feature_names
        }
        remaining = set(self.feature_names)
        first = min(
            remaining,
            key=lambda name: (self.mean_absolute_correlations_[name], original_position[name]),
        )
        ranking = [first]
        remaining.remove(first)
        max_selected = {first: 0.0}
        while remaining:
            candidate = min(
                remaining,
                key=lambda name: (
                    float(correlations.loc[name, ranking].max()),
                    self.mean_absolute_correlations_[name],
                    original_position[name],
                ),
            )
            max_selected[candidate] = float(correlations.loc[candidate, ranking].max())
            ranking.append(candidate)
            remaining.remove(candidate)
        self.ranking_ = ranking
        self.max_selected_correlations_ = max_selected
        return self

    def subsets(self, fractions: Sequence[float]) -> dict[float, list[str]]:
        if self.ranking_ is None:
            raise RuntimeError("Feature reducer is not fitted.")
        if not fractions or any(not 0.0 < float(value) <= 1.0 for value in fractions):
            raise ValueError("Feature fractions must be non-empty and in (0, 1].")
        subsets: dict[float, list[str]] = {}
        for fraction in fractions:
            value = float(fraction)
            count = min(len(self.feature_names), max(1, int(round(len(self.feature_names) * value))))
            selected = set(self.ranking_[:count])
            subsets[value] = [name for name in self.feature_names if name in selected]
        return subsets

    def ranking_frame(self) -> pd.DataFrame:
        if (
            self.ranking_ is None
            or self.mean_absolute_correlations_ is None
            or self.max_selected_correlations_ is None
        ):
            raise RuntimeError("Feature reducer is not fitted.")
        return pd.DataFrame(
            {
                "rank": np.arange(1, len(self.ranking_) + 1),
                "feature": self.ranking_,
                "mean_absolute_training_correlation": [
                    self.mean_absolute_correlations_[name] for name in self.ranking_
                ],
                "max_correlation_with_earlier_features": [
                    self.max_selected_correlations_[name] for name in self.ranking_
                ],
                "selection_data": "clean_training_split_only",
                "labels_used": False,
            }
        )
