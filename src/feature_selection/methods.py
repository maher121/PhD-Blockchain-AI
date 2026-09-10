"""Classical train-only feature selectors for the V0.6 baseline."""

from __future__ import annotations

from numbers import Real
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, f_classif, mutual_info_classif

from src.config import DATACO_CATEGORICAL_FEATURES, GLOBAL_SEED
from src.feature_selection.base import (
    SupervisedFeatureSelector,
    UnsupervisedFeatureSelector,
)
from src.lightweight.feature_reduction import CorrelationFeatureReducer

_MI_DISCRETE_STRATEGY = "v02_configured_onehot_prefixes"
_CORRELATION_RETENTION_RULE = (
    "retain the earliest canonical candidate; remove each later candidate when its "
    "absolute Pearson correlation with any previously retained feature meets the threshold"
)


class _RankingMixin:
    """Expose a consistent best-to-worst ranking table after fitting."""

    _ranking_records_: tuple[dict[str, Any], ...] | None

    def ranking_frame(self) -> pd.DataFrame:
        if self._ranking_records_ is None:
            raise RuntimeError("Feature selector is not fitted.")
        return pd.DataFrame([dict(record) for record in self._ranking_records_])


class VarianceThresholdSelector(UnsupervisedFeatureSelector):
    """Remove exactly constant training features as an unsupervised diagnostic."""

    selector_id = "variance_threshold"
    method = "variance_threshold_zero"

    def __init__(
        self,
        candidate_features: Sequence[str],
        *,
        threshold: float = 0.0,
        random_state: int = GLOBAL_SEED,
    ) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise ValueError("Variance threshold must be the numeric value 0.0.")
        if not np.isfinite(float(threshold)) or float(threshold) != 0.0:
            raise ValueError("V0.6 variance threshold must be exactly 0.0.")
        self.threshold = 0.0
        self.variances_: dict[str, float] | None = None
        self.retained_features_: tuple[str, ...] | None = None
        self.removed_features_: tuple[str, ...] | None = None
        super().__init__(candidate_features, random_state=random_state)

    def _fit_unsupervised(self, training_features: pd.DataFrame) -> np.ndarray:
        estimator = VarianceThreshold(threshold=self.threshold)
        try:
            estimator.fit(training_features)
        except ValueError as exc:
            raise ValueError("Variance threshold would remove every candidate feature.") from exc
        support = estimator.get_support()
        self.variances_ = {
            feature: float(value)
            for feature, value in zip(self.candidate_features, estimator.variances_)
        }
        self.retained_features_ = tuple(
            feature for feature, retained in zip(self.candidate_features, support) if retained
        )
        self.removed_features_ = tuple(
            feature for feature, retained in zip(self.candidate_features, support) if not retained
        )
        return support

    def _get_method_parameters(self) -> Mapping[str, Any]:
        return {"threshold": self.threshold, "count_strategy": "natural"}

    def _get_diagnostics(self) -> Mapping[str, Any]:
        return {
            "variance_by_feature": dict(self.variances_ or {}),
            "retained_features": self.retained_features_ or (),
            "removed_features": self.removed_features_ or (),
            "selection_data": "training_features_only",
        }


class PairwiseCorrelationFilter(UnsupervisedFeatureSelector):
    """Remove later candidates correlated with previously retained features."""

    selector_id = "pairwise_correlation_filter"
    method = "absolute_pearson_correlation_filter"

    def __init__(
        self,
        candidate_features: Sequence[str],
        *,
        threshold: float = 0.95,
        random_state: int = GLOBAL_SEED,
    ) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise ValueError("Correlation threshold must be numeric and in (0, 1].")
        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("Correlation threshold must be finite and in (0, 1].")
        self.threshold = threshold
        self.retained_features_: tuple[str, ...] | None = None
        self.removed_features_: tuple[str, ...] | None = None
        self.removal_details_: tuple[dict[str, Any], ...] | None = None
        self.high_correlation_pairs_: tuple[dict[str, Any], ...] | None = None
        self.constant_features_: tuple[str, ...] | None = None
        super().__init__(candidate_features, random_state=random_state)

    def _fit_unsupervised(self, training_features: pd.DataFrame) -> np.ndarray:
        correlations = training_features.corr(method="pearson").abs()
        constant_features = tuple(
            feature
            for feature in self.candidate_features
            if training_features[feature].nunique(dropna=True) <= 1
        )
        retained_indices: list[int] = []
        removal_details: list[dict[str, Any]] = []
        high_pairs: list[dict[str, Any]] = []

        for later_index in range(len(self.candidate_features)):
            later_feature = self.candidate_features[later_index]
            for earlier_index in range(later_index):
                earlier_feature = self.candidate_features[earlier_index]
                value = correlations.iat[earlier_index, later_index]
                if np.isfinite(value) and float(value) >= self.threshold:
                    high_pairs.append(
                        {
                            "earlier_feature": earlier_feature,
                            "later_feature": later_feature,
                            "absolute_correlation": float(value),
                        }
                    )

            blockers: list[tuple[float, int]] = []
            for retained_index in retained_indices:
                value = correlations.iat[retained_index, later_index]
                if np.isfinite(value) and float(value) >= self.threshold:
                    blockers.append((float(value), retained_index))
            if not blockers:
                retained_indices.append(later_index)
                continue

            correlation, blocker_index = min(
                blockers, key=lambda item: (-item[0], item[1])
            )
            removal_details.append(
                {
                    "removed_feature": later_feature,
                    "retained_blocker": self.candidate_features[blocker_index],
                    "absolute_correlation": correlation,
                }
            )

        support = np.zeros(len(self.candidate_features), dtype=bool)
        support[retained_indices] = True
        self.retained_features_ = tuple(
            self.candidate_features[index] for index in retained_indices
        )
        self.removed_features_ = tuple(
            detail["removed_feature"] for detail in removal_details
        )
        self.removal_details_ = tuple(removal_details)
        self.high_correlation_pairs_ = tuple(high_pairs)
        self.constant_features_ = constant_features
        return support

    def _get_method_parameters(self) -> Mapping[str, Any]:
        return {
            "threshold": self.threshold,
            "correlation": "absolute_pearson",
            "retention_rule": _CORRELATION_RETENTION_RULE,
            "count_strategy": "natural",
        }

    def _get_warnings(self) -> Sequence[str]:
        if self.constant_features_:
            return (
                "Pearson correlation is undefined for constant features; they were retained "
                "for the separate variance diagnostic.",
            )
        return ()

    def _get_diagnostics(self) -> Mapping[str, Any]:
        return {
            "retained_features": self.retained_features_ or (),
            "removed_features": self.removed_features_ or (),
            "removal_details": self.removal_details_ or (),
            "high_correlation_pairs": self.high_correlation_pairs_ or (),
            "constant_features": self.constant_features_ or (),
            "retention_rule": _CORRELATION_RETENTION_RULE,
            "selection_data": "training_features_only",
        }


class CorrelationRedundancySelector(_RankingMixin, UnsupervisedFeatureSelector):
    """V0.6 adapter for the unchanged historical V0.5 redundancy ranking."""

    selector_id = "correlation_redundancy_ranking"
    method = CorrelationFeatureReducer.method

    def __init__(
        self,
        candidate_features: Sequence[str],
        *,
        n_features_to_select: int,
        random_state: int = GLOBAL_SEED,
    ) -> None:
        self._ranking_records_ = None
        self.constant_features_: tuple[str, ...] | None = None
        super().__init__(
            candidate_features,
            n_features_to_select=n_features_to_select,
            random_state=random_state,
        )

    def _fit_unsupervised(self, training_features: pd.DataFrame) -> np.ndarray:
        reducer = CorrelationFeatureReducer(self.candidate_features).fit(training_features)
        self.constant_features_ = tuple(
            feature
            for feature in self.candidate_features
            if training_features[feature].nunique(dropna=True) <= 1
        )
        historical_ranking = reducer.ranking_frame()
        selected = set(
            historical_ranking["feature"].iloc[: self.n_features_to_select].tolist()
        )
        records: list[dict[str, Any]] = []
        for row in historical_ranking.itertuples(index=False):
            records.append(
                {
                    "feature": row.feature,
                    "score": None,
                    "score_status": "sequential_multicriterion",
                    "rank": int(row.rank),
                    "selected": row.feature in selected,
                    "mean_absolute_training_correlation": float(
                        row.mean_absolute_training_correlation
                    ),
                    "max_correlation_with_earlier_features": float(
                        row.max_correlation_with_earlier_features
                    ),
                }
            )
        self._ranking_records_ = tuple(records)
        return np.asarray(
            [feature in selected for feature in self.candidate_features], dtype=bool
        )

    def _get_method_parameters(self) -> Mapping[str, Any]:
        return {
            "historical_stage": "V0.5",
            "ranking_behavior": "unchanged_v05_correlation_feature_reducer",
        }

    def _get_warnings(self) -> Sequence[str]:
        if self.constant_features_:
            return (
                "The historical V0.5 ranking fills undefined constant-feature "
                "correlations with zero and may prioritize constants.",
            )
        return ()

    def _get_diagnostics(self) -> Mapping[str, Any]:
        return {
            "ranking": self._ranking_records_ or (),
            "ranking_order": "best_to_worst",
            "transformed_feature_order": "canonical_candidate_order",
            "score_definition": (
                "No scalar score: the historical method uses sequential maximum "
                "correlation, mean correlation, and canonical-position tie-breaking."
            ),
            "constant_features": self.constant_features_ or (),
            "constant_feature_policy": "preserve_unchanged_historical_v05_behavior",
            "selection_data": "training_features_only",
            "labels_used": False,
        }


class MutualInformationSelector(_RankingMixin, SupervisedFeatureSelector):
    """Select top-K features by training-label mutual information."""

    selector_id = "mutual_information_select_k_best"
    method = "mutual_information_classification"

    def __init__(
        self,
        candidate_features: Sequence[str],
        *,
        n_features_to_select: int,
        random_state: int = GLOBAL_SEED,
    ) -> None:
        self._ranking_records_ = None
        self.scores_: dict[str, float | None] | None = None
        self.discrete_features_: tuple[str, ...] | None = None
        self.discrete_mask_: tuple[bool, ...] | None = None
        self.constant_features_: tuple[str, ...] | None = None
        super().__init__(
            candidate_features,
            n_features_to_select=n_features_to_select,
            random_state=random_state,
        )

    def _fit_supervised(
        self, training_features: pd.DataFrame, labels: pd.Series
    ) -> np.ndarray:
        discrete_mask = np.asarray(
            [
                any(feature.startswith(f"{source}_") for source in DATACO_CATEGORICAL_FEATURES)
                for feature in self.candidate_features
            ],
            dtype=bool,
        )
        discrete_features = tuple(
            feature
            for feature, discrete in zip(self.candidate_features, discrete_mask)
            if discrete
        )
        for feature in discrete_features:
            values = set(training_features[feature].unique())
            if not values.issubset({0, 1, 0.0, 1.0}):
                raise ValueError(
                    f"Configured V0.2 one-hot feature is not binary: {feature!r}"
                )

        scores = mutual_info_classif(
            training_features,
            labels,
            discrete_features=discrete_mask,
            random_state=self.random_state,
        )
        constant_mask = np.asarray(
            [
                training_features[feature].nunique(dropna=True) <= 1
                for feature in self.candidate_features
            ],
            dtype=bool,
        )
        scores = np.asarray(scores, dtype=float)
        scores[constant_mask] = np.nan
        records, support = _rank_scores(
            self.candidate_features, scores, self.n_features_to_select
        )
        self._ranking_records_ = records
        self.scores_ = {
            feature: _finite_or_none(score)
            for feature, score in zip(self.candidate_features, scores)
        }
        self.discrete_features_ = discrete_features
        self.discrete_mask_ = tuple(bool(value) for value in discrete_mask)
        self.constant_features_ = tuple(
            feature
            for feature, constant in zip(self.candidate_features, constant_mask)
            if constant
        )
        return support

    def _get_method_parameters(self) -> Mapping[str, Any]:
        return {
            "score_function": "sklearn.feature_selection.mutual_info_classif",
            "discrete_feature_strategy": _MI_DISCRETE_STRATEGY,
            "continuous_neighbor_count": 3,
        }

    def _get_diagnostics(self) -> Mapping[str, Any]:
        return {
            "ranking": self._ranking_records_ or (),
            "ranking_order": "descending_mutual_information_then_canonical_order",
            "transformed_feature_order": "canonical_candidate_order",
            "discrete_feature_strategy": _MI_DISCRETE_STRATEGY,
            "discrete_feature_sources": DATACO_CATEGORICAL_FEATURES,
            "discrete_features": self.discrete_features_ or (),
            "discrete_mask": self.discrete_mask_ or (),
            "constant_features": self.constant_features_ or (),
            "constant_feature_policy": "undefined_score_and_rank_after_finite_scores",
            "selection_data": "training_features_and_controlled_is_attack_labels_only",
        }

    def _get_warnings(self) -> Sequence[str]:
        if self.constant_features_:
            return (
                "Mutual information scores for constant features are marked undefined "
                "and ranked after finite scores.",
            )
        return ()


class AnovaFSelector(_RankingMixin, SupervisedFeatureSelector):
    """Secondary top-K comparator using univariate training-label ANOVA F-scores."""

    selector_id = "anova_f_select_k_best"
    method = "anova_f_classification"

    def __init__(
        self,
        candidate_features: Sequence[str],
        *,
        n_features_to_select: int,
        random_state: int = GLOBAL_SEED,
    ) -> None:
        self._ranking_records_ = None
        self.scores_: dict[str, float | None] | None = None
        self.p_values_: dict[str, float | None] | None = None
        self.undefined_features_: tuple[str, ...] | None = None
        super().__init__(
            candidate_features,
            n_features_to_select=n_features_to_select,
            random_state=random_state,
        )

    def _fit_supervised(
        self, training_features: pd.DataFrame, labels: pd.Series
    ) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            scores, p_values = f_classif(training_features, labels)
        records, support = _rank_scores(
            self.candidate_features, scores, self.n_features_to_select
        )
        self._ranking_records_ = records
        self.scores_ = {
            feature: _finite_or_none(score)
            for feature, score in zip(self.candidate_features, scores)
        }
        self.p_values_ = {
            feature: _finite_or_none(value)
            for feature, value in zip(self.candidate_features, p_values)
        }
        self.undefined_features_ = tuple(
            feature
            for feature, score in zip(self.candidate_features, scores)
            if not np.isfinite(score)
        )
        return support

    def _get_method_parameters(self) -> Mapping[str, Any]:
        return {
            "score_function": "sklearn.feature_selection.f_classif",
            "scientific_role": "secondary_linear_univariate_comparator",
        }

    def _get_warnings(self) -> Sequence[str]:
        messages = [
            "ANOVA-F may miss bidirectional manipulation patterns with little mean shift."
        ]
        if self.undefined_features_:
            messages.append(
                "ANOVA-F produced undefined scores for degenerate features: "
                + ", ".join(self.undefined_features_)
            )
        return tuple(messages)

    def _get_diagnostics(self) -> Mapping[str, Any]:
        return {
            "ranking": self._ranking_records_ or (),
            "ranking_order": "descending_f_score_then_canonical_order",
            "transformed_feature_order": "canonical_candidate_order",
            "p_values_by_feature": dict(self.p_values_ or {}),
            "undefined_features": self.undefined_features_ or (),
            "undefined_score_policy": "rank_after_finite_scores_in_canonical_order",
            "selection_data": "training_features_and_controlled_is_attack_labels_only",
        }


def _rank_scores(
    candidate_features: Sequence[str],
    scores: Sequence[float] | np.ndarray,
    selected_count: int | None,
) -> tuple[tuple[dict[str, Any], ...], np.ndarray]:
    if selected_count is None:
        raise ValueError("A requested feature count is required for ranking selectors.")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) != len(candidate_features):
        raise ValueError("Feature scores must match the candidate feature count.")

    order = sorted(
        range(len(candidate_features)),
        key=lambda index: _score_sort_key(values[index], index),
    )
    selected_indices = set(order[:selected_count])
    records: list[dict[str, Any]] = []
    for rank, index in enumerate(order, start=1):
        score = values[index]
        records.append(
            {
                "feature": candidate_features[index],
                "score": _finite_or_none(score),
                "score_status": _score_status(score),
                "rank": rank,
                "selected": index in selected_indices,
            }
        )
    support = np.asarray(
        [index in selected_indices for index in range(len(candidate_features))], dtype=bool
    )
    return tuple(records), support


def _score_sort_key(score: float, canonical_index: int) -> tuple[int, float, int]:
    if np.isposinf(score):
        return (0, 0.0, canonical_index)
    if np.isfinite(score):
        return (1, -float(score), canonical_index)
    return (2, 0.0, canonical_index)


def _score_status(score: float) -> str:
    if np.isposinf(score):
        return "positive_infinity"
    if np.isneginf(score):
        return "negative_infinity"
    if np.isnan(score):
        return "undefined"
    return "finite"


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None
