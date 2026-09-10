"""Synthetic-data tests for V0.6 classical feature-selection methods."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from src.feature_selection import (
    CONTROLLED_ATTACK_LABEL_SOURCE,
    AnovaFSelector,
    BaseFeatureSelector,
    CorrelationRedundancySelector,
    MutualInformationSelector,
    PairwiseCorrelationFilter,
    SupervisedFeatureSelector,
    VarianceThresholdSelector,
)
from src.lightweight.feature_reduction import CorrelationFeatureReducer
from src.lightweight.models import create_model


@pytest.fixture()
def unsupervised_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base = np.linspace(-2.0, 2.0, 80)
    return pd.DataFrame(
        {
            "base": base,
            "exact_copy": base.copy(),
            "high_copy": base + rng.normal(0.0, 0.01, len(base)),
            "unrelated": rng.normal(0.0, 1.0, len(base)),
            "constant": np.ones(len(base)),
        },
        index=pd.Index(range(1000, 1080), name="row_id"),
    )


@pytest.fixture()
def supervised_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    labels_array = np.asarray([0, 1] * 120, dtype="int8")
    index = pd.Index(range(2000, 2240), name="row_id")
    features = pd.DataFrame(
        {
            "noise": rng.normal(0.0, 1.0, len(labels_array)),
            "signal": labels_array + rng.normal(0.0, 0.08, len(labels_array)),
            "Type_DEBIT": rng.integers(0, 2, len(labels_array)).astype(float),
            "constant": np.ones(len(labels_array)),
        },
        index=index,
    )
    labels = pd.Series(labels_array, index=index, name="is_attack")
    return features, labels


def test_variance_threshold_removes_only_constant_feature(unsupervised_data) -> None:
    selector = VarianceThresholdSelector(list(unsupervised_data)).fit(unsupervised_data)
    assert selector.result.selected_feature_count == 4
    assert "constant" not in selector.result.selected_features
    assert "unrelated" in selector.result.selected_features
    assert selector.variances_["constant"] == pytest.approx(0.0)
    assert selector.result.parameters["count_strategy"] == "natural"


def test_variance_threshold_is_deterministic(unsupervised_data) -> None:
    first = VarianceThresholdSelector(list(unsupervised_data)).fit(unsupervised_data)
    second = VarianceThresholdSelector(list(unsupervised_data)).fit(
        unsupervised_data.copy()
    )
    assert first.result.to_dict() == second.result.to_dict()


def test_variance_threshold_rejects_labels(unsupervised_data) -> None:
    labels = pd.Series([0, 1] * 40, index=unsupervised_data.index, name="is_attack")
    with pytest.raises(ValueError, match="do not accept labels"):
        VarianceThresholdSelector(list(unsupervised_data)).fit(
            unsupervised_data, labels
        )


@pytest.mark.parametrize("threshold", [-0.1, 0.1, np.inf, np.nan, True])
def test_variance_threshold_rejects_nonzero_configuration(
    unsupervised_data, threshold: object
) -> None:
    with pytest.raises(ValueError, match="threshold"):
        VarianceThresholdSelector(
            list(unsupervised_data), threshold=threshold  # type: ignore[arg-type]
        )


def test_pairwise_filter_removes_exact_and_high_correlations(
    unsupervised_data,
) -> None:
    selector = PairwiseCorrelationFilter(list(unsupervised_data)).fit(
        unsupervised_data
    )
    assert selector.result.selected_features == ("base", "unrelated", "constant")
    assert selector.removed_features_ == ("exact_copy", "high_copy")
    assert selector.result.selected_feature_count == 3
    assert "constant" in selector.constant_features_
    assert selector.result.warnings


def test_pairwise_filter_exposes_duplicate_correlation(unsupervised_data) -> None:
    selector = PairwiseCorrelationFilter(list(unsupervised_data)).fit(
        unsupervised_data
    )
    details = {row["removed_feature"]: row for row in selector.removal_details_}
    assert details["exact_copy"]["retained_blocker"] == "base"
    assert details["exact_copy"]["absolute_correlation"] == pytest.approx(1.0)
    assert details["high_copy"]["retained_blocker"] == "base"
    assert "unrelated" in selector.retained_features_


def test_pairwise_filter_tie_breaking_uses_candidate_order(unsupervised_data) -> None:
    candidates = list(unsupervised_data)
    first = PairwiseCorrelationFilter(candidates).fit(unsupervised_data)
    second = PairwiseCorrelationFilter(candidates).fit(
        unsupervised_data[list(reversed(candidates))]
    )
    assert first.result.selected_features == second.result.selected_features
    assert first.removal_details_ == second.removal_details_


def test_pairwise_filter_rejects_labels(unsupervised_data) -> None:
    labels = pd.Series([0, 1] * 40, index=unsupervised_data.index, name="is_attack")
    with pytest.raises(ValueError, match="do not accept labels"):
        PairwiseCorrelationFilter(list(unsupervised_data)).fit(
            unsupervised_data, labels
        )


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.1, np.inf, np.nan, False])
def test_pairwise_filter_rejects_invalid_threshold(
    unsupervised_data, threshold: object
) -> None:
    with pytest.raises(ValueError, match="threshold"):
        PairwiseCorrelationFilter(
            list(unsupervised_data), threshold=threshold  # type: ignore[arg-type]
        )


def test_correlation_redundancy_matches_historical_v05_ranking(
    unsupervised_data,
) -> None:
    candidates = list(unsupervised_data)
    historical = CorrelationFeatureReducer(candidates).fit(unsupervised_data)
    selector = CorrelationRedundancySelector(
        candidates, n_features_to_select=3
    ).fit(unsupervised_data)
    assert selector.ranking_frame()["feature"].tolist() == historical.ranking_frame()[
        "feature"
    ].tolist()
    assert selector.ranking_frame()["rank"].tolist() == [1, 2, 3, 4, 5]
    assert selector.ranking_frame()["feature"].tolist() == [
        "constant",
        "unrelated",
        "base",
        "high_copy",
        "exact_copy",
    ]
    assert selector.result.method == historical.method
    assert selector.result.labels_used is False
    assert "constant" in selector.result.diagnostics["constant_features"]
    assert selector.result.warnings


def test_correlation_redundancy_returns_exact_top_k(unsupervised_data) -> None:
    selector = CorrelationRedundancySelector(
        list(unsupervised_data), n_features_to_select=2
    ).fit(unsupervised_data)
    ranking = selector.ranking_frame()
    ranked_selected = set(ranking.loc[ranking["rank"] <= 2, "feature"])
    assert selector.result.selected_feature_count == 2
    assert set(selector.result.selected_features) == ranked_selected
    assert ranking["selected"].tolist() == [True, True, False, False, False]


def test_correlation_redundancy_top_k_is_nested(unsupervised_data) -> None:
    candidates = list(unsupervised_data)
    top_two = CorrelationRedundancySelector(
        candidates, n_features_to_select=2
    ).fit(unsupervised_data)
    top_three = CorrelationRedundancySelector(
        candidates, n_features_to_select=3
    ).fit(unsupervised_data)
    assert top_two.ranking_frame()["feature"].tolist() == top_three.ranking_frame()[
        "feature"
    ].tolist()
    assert set(top_two.result.selected_features).issubset(
        top_three.result.selected_features
    )


def test_correlation_redundancy_rejects_labels(unsupervised_data) -> None:
    labels = pd.Series([0, 1] * 40, index=unsupervised_data.index, name="is_attack")
    with pytest.raises(ValueError, match="do not accept labels"):
        CorrelationRedundancySelector(
            list(unsupervised_data), n_features_to_select=2
        ).fit(unsupervised_data, labels)


def test_mutual_information_requires_controlled_labels(supervised_data) -> None:
    features, _ = supervised_data
    selector = MutualInformationSelector(list(features), n_features_to_select=2)
    with pytest.raises(ValueError, match="require training labels"):
        selector.fit(features)


def test_mutual_information_ranks_signal_and_returns_exact_k(supervised_data) -> None:
    features, labels = supervised_data
    selector = MutualInformationSelector(
        list(features), n_features_to_select=2, random_state=42
    ).fit(features, labels)
    ranking = selector.ranking_frame()
    assert ranking.iloc[0]["feature"] == "signal"
    assert ranking["rank"].tolist() == [1, 2, 3, 4]
    assert int(ranking["selected"].sum()) == 2
    assert selector.result.selected_feature_count == 2
    assert set(selector.result.selected_features) == set(
        ranking.loc[ranking["selected"], "feature"]
    )
    assert selector.result.training_label_source == CONTROLLED_ATTACK_LABEL_SOURCE


def test_mutual_information_is_deterministic_with_same_seed(supervised_data) -> None:
    features, labels = supervised_data
    first = MutualInformationSelector(
        list(features), n_features_to_select=2, random_state=42
    ).fit(features, labels)
    second = MutualInformationSelector(
        list(features), n_features_to_select=2, random_state=42
    ).fit(features.copy(), labels.copy())
    pd.testing.assert_frame_equal(first.ranking_frame(), second.ranking_frame())
    assert first.result.to_dict() == second.result.to_dict()


def test_mutual_information_uses_v02_onehot_prefixes(supervised_data) -> None:
    features, labels = supervised_data
    selector = MutualInformationSelector(
        list(features), n_features_to_select=2
    ).fit(features, labels)
    assert selector.discrete_features_ == ("Type_DEBIT",)
    assert selector.discrete_mask_ == (False, False, True, False)
    assert (
        selector.result.diagnostics["discrete_feature_strategy"]
        == "v02_configured_onehot_prefixes"
    )


def test_mutual_information_identifies_all_configured_onehot_prefixes() -> None:
    features = pd.DataFrame(
        {
            "continuous": np.linspace(0.0, 1.0, 20),
            "Type_DEBIT": [0.0, 1.0] * 10,
            "Market_LATAM": [1.0, 0.0] * 10,
            "Shipping Mode_Same Day": [0.0, 1.0] * 10,
            "Customer Segment_Consumer": [1.0, 0.0] * 10,
            "Department Name_Apparel": [0.0, 1.0] * 10,
        }
    )
    labels = pd.Series([0, 1] * 10, name="is_attack")
    selector = MutualInformationSelector(
        list(features), n_features_to_select=2
    ).fit(features, labels)
    assert selector.discrete_features_ == (
        "Type_DEBIT",
        "Market_LATAM",
        "Shipping Mode_Same Day",
        "Customer Segment_Consumer",
        "Department Name_Apparel",
    )


def test_mutual_information_ranks_constant_feature_last(supervised_data) -> None:
    features, labels = supervised_data
    selector = MutualInformationSelector(
        list(features), n_features_to_select=2
    ).fit(features, labels)
    constant = selector.ranking_frame().iloc[-1]
    assert constant["feature"] == "constant"
    assert constant["score"] is None or pd.isna(constant["score"])
    assert constant["score_status"] == "undefined"
    assert selector.scores_["constant"] is None
    assert selector.result.warnings


def test_mutual_information_rejects_nonbinary_configured_onehot(supervised_data) -> None:
    features, labels = supervised_data
    invalid = features.copy()
    invalid.loc[invalid.index[0], "Type_DEBIT"] = 2.0
    with pytest.raises(ValueError, match="one-hot feature is not binary"):
        MutualInformationSelector(
            list(features), n_features_to_select=2
        ).fit(invalid, labels)


def test_anova_ranks_signal_and_handles_constant_feature(supervised_data) -> None:
    features, labels = supervised_data
    selector = AnovaFSelector(list(features), n_features_to_select=2).fit(
        features, labels
    )
    ranking = selector.ranking_frame()
    constant = ranking.loc[ranking["feature"] == "constant"].iloc[0]
    assert ranking.iloc[0]["feature"] == "signal"
    assert selector.result.selected_feature_count == 2
    assert constant["score"] is None or pd.isna(constant["score"])
    assert constant["score_status"] == "undefined"
    assert "constant" in selector.undefined_features_
    assert any("bidirectional" in message for message in selector.result.warnings)


def test_anova_is_deterministic(supervised_data) -> None:
    features, labels = supervised_data
    first = AnovaFSelector(list(features), n_features_to_select=2).fit(
        features, labels
    )
    second = AnovaFSelector(list(features), n_features_to_select=2).fit(
        features.copy(), labels.copy()
    )
    pd.testing.assert_frame_equal(first.ranking_frame(), second.ranking_frame())
    assert first.result.selected_features == second.result.selected_features


def test_anova_requires_controlled_labels(supervised_data) -> None:
    features, _ = supervised_data
    with pytest.raises(ValueError, match="require training labels"):
        AnovaFSelector(list(features), n_features_to_select=2).fit(features)


def test_ranking_ties_use_canonical_order() -> None:
    features = pd.DataFrame(
        {"first": [1.0] * 8, "second": [1.0] * 8, "third": [1.0] * 8}
    )
    labels = pd.Series([0, 1] * 4, name="is_attack")
    selector = AnovaFSelector(list(features), n_features_to_select=1).fit(
        features, labels
    )
    ranking = selector.ranking_frame()
    assert ranking["feature"].tolist() == ["first", "second", "third"]
    assert selector.result.selected_features == ("first",)


@pytest.mark.parametrize(
    "selector_factory",
    [
        lambda names: CorrelationRedundancySelector(
            names, n_features_to_select=2
        ),
        lambda names: MutualInformationSelector(names, n_features_to_select=2),
        lambda names: AnovaFSelector(names, n_features_to_select=2),
    ],
)
def test_ranking_selectors_reject_invalid_k(
    selector_factory: Callable[[list[str]], BaseFeatureSelector],
) -> None:
    with pytest.raises(ValueError, match="n_features_to_select"):
        selector_factory(["only"])


@pytest.mark.parametrize(
    "selector_factory",
    [
        lambda names: VarianceThresholdSelector(names),
        lambda names: PairwiseCorrelationFilter(names),
        lambda names: CorrelationRedundancySelector(
            names, n_features_to_select=1
        ),
        lambda names: MutualInformationSelector(names, n_features_to_select=1),
        lambda names: AnovaFSelector(names, n_features_to_select=1),
    ],
)
def test_forbidden_features_are_rejected_by_every_method(
    selector_factory: Callable[[list[str]], BaseFeatureSelector],
) -> None:
    with pytest.raises(ValueError, match="Forbidden V0.5"):
        selector_factory(["safe", "attack_type"])


def test_transformed_ranking_features_remain_in_canonical_order(supervised_data) -> None:
    features, labels = supervised_data
    selector = MutualInformationSelector(
        list(features), n_features_to_select=2
    ).fit(features, labels)
    transformed = selector.transform(features)
    expected = [
        feature for feature in features.columns if feature in selector.result.selected_features
    ]
    assert list(transformed) == expected
    assert selector.ranking_frame().iloc[0]["feature"] == "signal"


def _method_selectors(
    features: pd.DataFrame, labels: pd.Series
) -> list[BaseFeatureSelector]:
    selectors: list[BaseFeatureSelector] = [
        VarianceThresholdSelector(list(features)).fit(features),
        PairwiseCorrelationFilter(list(features)).fit(features),
        CorrelationRedundancySelector(
            list(features), n_features_to_select=2
        ).fit(features),
        MutualInformationSelector(list(features), n_features_to_select=2).fit(
            features, labels
        ),
        AnovaFSelector(list(features), n_features_to_select=2).fit(features, labels),
    ]
    return selectors


@pytest.mark.parametrize("model_name", ["decision_tree", "logistic_regression"])
def test_all_selectors_are_compatible_with_v05_supervised_models(
    supervised_data, model_name: str
) -> None:
    features, labels = supervised_data
    parameters = (
        {"max_depth": 3}
        if model_name == "decision_tree"
        else {"solver": "liblinear", "max_iter": 100}
    )
    for selector in _method_selectors(features, labels):
        transformed = selector.transform(features)
        model = create_model(
            model_name, selector.get_feature_names_out().tolist(), parameters
        ).fit(transformed, labels)
        assert len(model.predict(transformed)) == len(features)
        assert model.feature_names == list(transformed)


def test_natural_and_ranked_selectors_declare_supervision(supervised_data) -> None:
    features, labels = supervised_data
    selectors = _method_selectors(features, labels)
    observed = {selector.selector_id: selector.result.labels_used for selector in selectors}
    assert observed == {
        "variance_threshold": False,
        "pairwise_correlation_filter": False,
        "correlation_redundancy_ranking": False,
        "mutual_information_select_k_best": True,
        "anova_f_select_k_best": True,
    }
    assert all(
        isinstance(selector, SupervisedFeatureSelector) == selector.result.labels_used
        for selector in selectors
    )
