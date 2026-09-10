"""Synthetic-data tests for the V0.6 feature-selector contract."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.feature_selection import (
    CONTROLLED_ATTACK_LABEL_SOURCE,
    FEATURE_SELECTOR_CONTRACT_VERSION,
    BaseFeatureSelector,
    SupervisedFeatureSelector,
    UnsupervisedFeatureSelector,
)
from src.lightweight.models import create_model


class FixedUnsupervisedSelector(UnsupervisedFeatureSelector):
    selector_id = "fixed_unsupervised"
    method = "synthetic_fixed_mask"

    def _fit_unsupervised(self, training_features: pd.DataFrame) -> np.ndarray:
        self.received_columns_ = tuple(training_features.columns)
        return np.asarray([False, True, False, True], dtype=bool)

    def _get_method_parameters(self) -> dict[str, str]:
        return {"fixture": "fixed"}


class FixedSupervisedSelector(SupervisedFeatureSelector):
    selector_id = "fixed_supervised"
    method = "synthetic_label_aware_mask"

    def _fit_supervised(
        self, training_features: pd.DataFrame, labels: pd.Series
    ) -> np.ndarray:
        self.received_labels_ = labels.copy()
        return np.asarray([True, False, True, False], dtype=bool)

    def _get_warnings(self) -> tuple[str, ...]:
        return ("synthetic test selector",)

    def _get_diagnostics(self) -> dict[str, int]:
        return {"observed_positive_labels": int(self.received_labels_.sum())}


@pytest.fixture()
def synthetic_training_data() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.Index([101, 104, 108, 111, 120, 133, 145, 160], name="row_id")
    features = pd.DataFrame(
        {
            "signal": [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3],
            "signal_copy": [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3],
            "noise": [0.4, -0.2, 0.7, -0.5, 0.1, 0.6, -0.3, 0.2],
            "constant": [1.0] * 8,
        },
        index=index,
    )
    labels = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], index=index, name="is_attack")
    return features, labels


def _unsupervised(features: pd.DataFrame) -> FixedUnsupervisedSelector:
    return FixedUnsupervisedSelector(list(features), n_features_to_select=2)


def _supervised(features: pd.DataFrame) -> FixedSupervisedSelector:
    return FixedSupervisedSelector(list(features), n_features_to_select=2)


def test_selector_contract_has_concrete_implementations(synthetic_training_data) -> None:
    features, _ = synthetic_training_data
    assert inspect.isabstract(BaseFeatureSelector)
    assert inspect.isabstract(UnsupervisedFeatureSelector)
    assert inspect.isabstract(SupervisedFeatureSelector)
    assert isinstance(_unsupervised(features), BaseFeatureSelector)


def test_unsupervised_fit_and_transform(synthetic_training_data) -> None:
    features, _ = synthetic_training_data
    selector = _unsupervised(features)
    assert selector.fit(features) is selector
    transformed = selector.transform(features)
    assert selector.is_fitted
    assert list(transformed) == ["signal_copy", "constant"]
    pd.testing.assert_frame_equal(transformed, features[["signal_copy", "constant"]])


def test_fit_transform_matches_separate_operations(synthetic_training_data) -> None:
    features, _ = synthetic_training_data
    combined = _unsupervised(features).fit_transform(features)
    separate_selector = _unsupervised(features).fit(features)
    pd.testing.assert_frame_equal(combined, separate_selector.transform(features))


def test_selected_names_mask_and_indices_are_consistent(synthetic_training_data) -> None:
    features, _ = synthetic_training_data
    selector = _unsupervised(features).fit(features)
    assert selector.get_feature_names_out().tolist() == ["signal_copy", "constant"]
    assert selector.get_support().tolist() == [False, True, False, True]
    assert selector.get_support(indices=True).tolist() == [1, 3]
    assert selector.result.selected_features == ("signal_copy", "constant")
    assert selector.result.selected_indices == (1, 3)


def test_transform_uses_deterministic_candidate_order(synthetic_training_data) -> None:
    features, _ = synthetic_training_data
    selector = _unsupervised(features).fit(features)
    reordered = features[["constant", "noise", "signal_copy", "signal"]]
    transformed = selector.transform(reordered)
    assert list(transformed) == ["signal_copy", "constant"]


def test_transform_before_fit_fails(synthetic_training_data) -> None:
    features, _ = synthetic_training_data
    with pytest.raises(RuntimeError, match="not fitted"):
        _unsupervised(features).transform(features)


@pytest.mark.parametrize("change", ["missing", "unexpected"])
def test_incompatible_feature_schema_fails(synthetic_training_data, change: str) -> None:
    features, _ = synthetic_training_data
    selector = _unsupervised(features).fit(features)
    incompatible = features.drop(columns="constant")
    if change == "unexpected":
        incompatible = features.assign(is_attack=0)
    with pytest.raises(ValueError, match="missing|unexpected"):
        selector.transform(incompatible)


def test_duplicate_feature_names_are_rejected(synthetic_training_data) -> None:
    features, _ = synthetic_training_data
    with pytest.raises(ValueError, match="non-empty and unique"):
        FixedUnsupervisedSelector(
            ["signal", "signal"], n_features_to_select=1
        )

    duplicated = features.copy()
    duplicated.columns = ["signal", "signal_copy", "noise", "noise"]
    with pytest.raises(ValueError, match="duplicate columns"):
        _unsupervised(features).fit(duplicated)


@pytest.mark.parametrize("invalid_count", [0, 5, -1, 1.5, True])
def test_invalid_feature_count_is_rejected(
    synthetic_training_data, invalid_count: object
) -> None:
    features, _ = synthetic_training_data
    with pytest.raises(ValueError, match="n_features_to_select"):
        FixedUnsupervisedSelector(
            list(features), n_features_to_select=invalid_count  # type: ignore[arg-type]
        )


def test_supervised_selector_requires_labels(synthetic_training_data) -> None:
    features, _ = synthetic_training_data
    with pytest.raises(ValueError, match="require training labels"):
        _supervised(features).fit(features)


def test_supervised_fit_transform_records_label_use(synthetic_training_data) -> None:
    features, labels = synthetic_training_data
    selector = _supervised(features)
    transformed = selector.fit_transform(features, labels)
    assert list(transformed) == ["signal", "noise"]
    assert selector.result.labels_used is True
    assert selector.result.training_labels_sha256 is not None
    assert selector.result.training_label_name == "is_attack"
    assert selector.result.training_label_source == CONTROLLED_ATTACK_LABEL_SOURCE
    assert selector.result.diagnostics == {"observed_positive_labels": 4}
    assert selector.result.warnings == ("synthetic test selector",)
    pd.testing.assert_series_equal(selector.received_labels_, labels.astype("int8"))


def test_supervised_labels_must_align_with_training_rows(synthetic_training_data) -> None:
    features, labels = synthetic_training_data
    misaligned = labels.reset_index(drop=True)
    with pytest.raises(ValueError, match="align exactly"):
        _supervised(features).fit(features, misaligned)


def test_supervised_selector_rejects_operational_target_labels(
    synthetic_training_data,
) -> None:
    features, labels = synthetic_training_data
    operational_target = labels.rename("Late_delivery_risk")
    with pytest.raises(ValueError, match="controlled training label source"):
        _supervised(features).fit(features, operational_target)


def test_numpy_labels_require_explicit_controlled_source(
    synthetic_training_data,
) -> None:
    features, labels = synthetic_training_data
    values = labels.to_numpy()
    with pytest.raises(ValueError, match="explicit controlled training label source"):
        _supervised(features).fit(features, values)
    selector = _supervised(features).fit(
        features, values, label_source=CONTROLLED_ATTACK_LABEL_SOURCE
    )
    assert selector.result.training_label_source == CONTROLLED_ATTACK_LABEL_SOURCE


def test_unsupervised_selector_rejects_labels(synthetic_training_data) -> None:
    features, labels = synthetic_training_data
    selector = _unsupervised(features)
    with pytest.raises(ValueError, match="do not accept labels"):
        selector.fit(features, labels)
    assert selector.is_fitted is False


def test_repeated_equivalent_fits_are_deterministic(synthetic_training_data) -> None:
    features, _ = synthetic_training_data
    first = _unsupervised(features).fit(features)
    second = _unsupervised(features.copy()).fit(features.copy())
    assert first.result.to_dict() == second.result.to_dict()
    assert len(first.result.training_rows_sha256) == 64
    assert len(first.result.training_features_sha256) == 64


def test_fitted_selector_cannot_be_refit_on_another_split(synthetic_training_data) -> None:
    features, _ = synthetic_training_data
    selector = _unsupervised(features).fit(features)
    with pytest.raises(RuntimeError, match="already fitted"):
        selector.fit(features)


def test_metadata_records_unsupervised_provenance(synthetic_training_data) -> None:
    features, _ = synthetic_training_data
    result = _unsupervised(features).fit(features).result
    assert result.contract_version == FEATURE_SELECTOR_CONTRACT_VERSION
    assert result.selector_id == "fixed_unsupervised"
    assert result.selector_type == "unsupervised"
    assert result.method == "synthetic_fixed_mask"
    assert result.labels_used is False
    assert result.training_labels_sha256 is None
    assert result.training_label_name is None
    assert result.training_label_source is None
    assert result.training_row_source == "dataframe_index"
    assert result.training_row_count == len(features)
    assert len(result.candidate_features_sha256) == 64
    assert result.candidate_feature_count == 4
    assert result.selected_feature_count == 2
    assert result.parameters == {
        "n_features_to_select": 2,
        "random_state": 42,
        "fixture": "fixed",
    }


def test_row_provenance_changes_when_training_rows_change(synthetic_training_data) -> None:
    features, _ = synthetic_training_data
    changed = features.copy()
    changed.index = pd.Index(range(len(changed)), name="row_id")
    original = _unsupervised(features).fit(features).result
    modified = _unsupervised(changed).fit(changed).result
    assert original.training_rows_sha256 != modified.training_rows_sha256
    assert original.training_features_sha256 != modified.training_features_sha256


def test_processed_row_id_is_fingerprinted_but_never_selected(
    synthetic_training_data,
) -> None:
    features, _ = synthetic_training_data
    with_trace = features.reset_index()
    selector = _unsupervised(features).fit(with_trace)
    transformed = selector.transform(with_trace)
    assert selector.result.training_row_source == "row_id_column"
    assert "row_id" not in selector.received_columns_
    assert "row_id" not in transformed
    assert list(transformed) == ["signal_copy", "constant"]


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_non_finite_training_features_are_rejected(
    synthetic_training_data, invalid_value: float
) -> None:
    features, _ = synthetic_training_data
    invalid = features.copy()
    invalid.iloc[0, 0] = invalid_value
    with pytest.raises(ValueError, match="missing or infinite"):
        _unsupervised(features).fit(invalid)


def test_complex_training_features_are_rejected(synthetic_training_data) -> None:
    features, _ = synthetic_training_data
    complex_features = features.astype(complex)
    with pytest.raises(ValueError, match="numeric dtypes"):
        _unsupervised(features).fit(complex_features)


def test_result_parameter_mappings_cannot_be_mutated(synthetic_training_data) -> None:
    features, _ = synthetic_training_data
    result = _unsupervised(features).fit(features).result
    with pytest.raises(TypeError):
        result.parameters["random_state"] = 7  # type: ignore[index]
    with pytest.raises(TypeError):
        result.diagnostics["changed"] = True  # type: ignore[index]


def test_leakage_prone_candidate_feature_is_rejected(synthetic_training_data) -> None:
    features, _ = synthetic_training_data
    with pytest.raises(ValueError, match="Forbidden V0.5"):
        FixedUnsupervisedSelector(
            [*features.columns, "is_attack"], n_features_to_select=2
        )


def test_invalid_support_mask_is_rejected(synthetic_training_data) -> None:
    features, _ = synthetic_training_data

    class InvalidFutureSelector(UnsupervisedFeatureSelector):
        selector_id = "invalid_future"
        method = "invalid_integer_mask"

        def _fit_unsupervised(self, training_features: pd.DataFrame) -> np.ndarray:
            return np.asarray([1, 0, 1, 0])

    with pytest.raises(ValueError, match="boolean values"):
        InvalidFutureSelector(list(features), n_features_to_select=2).fit(features)


def test_support_mask_must_match_requested_feature_count(synthetic_training_data) -> None:
    features, _ = synthetic_training_data

    class WrongCountSelector(UnsupervisedFeatureSelector):
        selector_id = "wrong_count"
        method = "wrong_count_mask"

        def _fit_unsupervised(self, training_features: pd.DataFrame) -> np.ndarray:
            return np.asarray([True, False, False, False], dtype=bool)

    with pytest.raises(ValueError, match="does not match"):
        WrongCountSelector(list(features), n_features_to_select=2).fit(features)


def test_future_selector_output_is_compatible_with_existing_model_factory(
    synthetic_training_data,
) -> None:
    features, labels = synthetic_training_data
    selector: BaseFeatureSelector = _supervised(features).fit(features, labels)
    selected = selector.transform(features)
    model = create_model(
        "decision_tree", selector.get_feature_names_out().tolist(), {"max_depth": 2}
    )
    model.fit(selected, labels)
    assert model.feature_names == ["signal", "noise"]
    assert len(model.predict(selected)) == len(features)

    linear = create_model(
        "logistic_regression",
        selector.get_feature_names_out().tolist(),
        {"solver": "liblinear"},
    ).fit(selected, labels)
    assert linear.feature_names == ["signal", "noise"]
