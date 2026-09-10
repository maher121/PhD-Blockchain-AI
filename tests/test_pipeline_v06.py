"""Synthetic integration tests for the V0.6-C experiment protocol."""

from __future__ import annotations

from dataclasses import replace
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from src.ai.model_utils import ProcessedSplit
from src.feature_selection import (
    BaseFeatureSelector,
    MutualInformationSelector,
    UnsupervisedFeatureSelector,
    VarianceThresholdSelector,
)
from src.pipeline_v06 import (
    V06_DECISION_TREE_PARAMETERS,
    V06_PREDICTION_THRESHOLD,
    V06ProtocolConfig,
    audit_development_data,
    audit_locked_test_data,
    create_v06_selector,
    load_v06_development_data,
    load_v06_protocol_config,
    lock_validation_configuration,
    prepare_v06_locked_test_data,
    run_locked_test_experiment,
    run_validation_experiment,
)
from src.security.experiment_data import (
    DevelopmentExperimentData,
    prepare_attack_split,
    prepare_development_experiment_data,
    prepare_test_experiment_data,
)
from src.security.ground_truth import ATTACK_METADATA_COLUMNS, MANIFEST_COLUMNS


class RecordingUnsupervisedSelector(UnsupervisedFeatureSelector):
    selector_id = "recording_unsupervised"
    method = "record_training_fit"

    def __init__(self, candidate_features: list[str]) -> None:
        self.fit_calls = 0
        self.fit_index: tuple[int, ...] | None = None
        super().__init__(candidate_features)

    def _fit_unsupervised(self, training_features: pd.DataFrame) -> np.ndarray:
        self.fit_calls += 1
        self.fit_index = tuple(training_features.index)
        return np.ones(len(self.candidate_features), dtype=bool)


@pytest.fixture()
def synthetic_protocol_data() -> tuple[
    ProcessedSplit,
    ProcessedSplit,
    ProcessedSplit,
    tuple[str, ...],
    dict,
]:
    candidates = (
        "order_item_quantity",
        "order_item_total",
        "days_schedule",
        "day",
        "noise",
    )
    train = _synthetic_split("train", 1000, candidates)
    validation = _synthetic_split("validation", 2000, candidates)
    test = _synthetic_split("test", 3000, candidates)
    metadata = {
        "features": {"columns": list(candidates)},
        "preprocessing": {
            "fitted_on_rows": len(train.features),
            "ml_feature_count": len(candidates),
            "ml_feature_columns": list(candidates),
        },
        "split": {
            "seed": 42,
            "strategy": "synthetic_grouped",
            "sizes": {
                "train": len(train.features),
                "validation": len(validation.features),
                "test": len(test.features),
            },
        },
    }
    return train, validation, test, candidates, metadata


@pytest.fixture()
def protocol_config() -> V06ProtocolConfig:
    return V06ProtocolConfig(
        experiment_id="v0_6_synthetic_s42",
        candidate_feature_counts=(3, 2, 1),
    )


@pytest.fixture()
def development_data(
    synthetic_protocol_data, protocol_config
) -> DevelopmentExperimentData:
    train, validation, _, candidates, metadata = synthetic_protocol_data
    return prepare_development_experiment_data(
        train=train,
        validation=validation,
        candidate_features=candidates,
        dataset_metadata=metadata,
        attack_type=protocol_config.attack_type,
        attack_rate=protocol_config.attack_rate,
        severity=protocol_config.attack_severity,
        random_seed=protocol_config.seed,
        experiment_id=protocol_config.experiment_id,
    )


def _synthetic_split(
    name: str, row_start: int, candidates: tuple[str, ...], rows: int = 140
) -> ProcessedSplit:
    positions = np.arange(rows)
    row_ids = np.arange(row_start, row_start + rows)
    index = pd.Index(row_ids, name="source_index")
    quantity = (positions % 5 + 1).astype(float)
    total = (25.0 + positions * 1.5 + quantity * 4.0).astype(float)
    scheduled = (positions % 4 + 1).astype(float)
    dates = pd.Timestamp("2017-01-02") + pd.to_timedelta(positions % 20, unit="D")
    metadata = pd.DataFrame(
        {
            "row_id": row_ids,
            "Order Item Quantity": quantity,
            "Order Item Total": total,
            "Days for shipment (scheduled)": scheduled,
            "Order Status": np.asarray(["A", "B", "C", "D"])[positions % 4],
            "order date (DateOrders)": dates.astype(str),
            "Order Region": np.asarray(["North", "South", "East", "West"])[
                positions % 4
            ],
        },
        index=index,
    )
    features = pd.DataFrame(
        {
            "row_id": row_ids,
            "order_item_quantity": quantity,
            "order_item_total": total,
            "days_schedule": scheduled,
            "day": pd.Series(dates, index=index).dt.day.astype(float),
            "noise": np.sin(positions / 7.0),
        },
        index=index,
    )
    assert list(features.columns) == ["row_id", *candidates]
    return ProcessedSplit(name, features, metadata, None)


def _prepare_test(
    synthetic_protocol_data, protocol_config, lock
):
    train, _, test, candidates, _ = synthetic_protocol_data
    assert candidates == lock.candidate_features
    return prepare_v06_locked_test_data(
        lock,
        test=test,
        training_reference=train,
    )


def test_attack_split_builder_uses_existing_v04_schemas(
    synthetic_protocol_data, protocol_config
) -> None:
    train, validation, _, candidates, _ = synthetic_protocol_data
    prepared = prepare_attack_split(
        split=validation,
        training_reference=train,
        candidate_features=candidates,
        attack_type="mixed",
        attack_rate=0.05,
        severity="MEDIUM",
        random_seed=42,
        experiment_id="synthetic_validation",
    )
    assert len(prepared.manifest) == 7
    assert int(prepared.labels.sum()) == 7
    assert list(prepared.manifest) == list(MANIFEST_COLUMNS)
    assert prepared.labels.name == "is_attack"
    assert prepared.labels.index.equals(prepared.features.index)
    assert prepared.features["row_id"].tolist() == validation.features["row_id"].tolist()
    assert not set(ATTACK_METADATA_COLUMNS) & set(prepared.features)


def test_attack_split_builder_is_reproducible(
    synthetic_protocol_data, protocol_config
) -> None:
    train, validation, _, candidates, _ = synthetic_protocol_data
    kwargs = {
        "split": validation,
        "training_reference": train,
        "candidate_features": candidates,
        "attack_type": protocol_config.attack_type,
        "attack_rate": protocol_config.attack_rate,
        "severity": protocol_config.attack_severity,
        "random_seed": protocol_config.seed,
        "experiment_id": "same_id",
    }
    first = prepare_attack_split(**kwargs)
    second = prepare_attack_split(**kwargs)
    pd.testing.assert_frame_equal(first.features, second.features)
    pd.testing.assert_frame_equal(first.manifest, second.manifest)
    pd.testing.assert_series_equal(first.labels, second.labels)
    assert first.attacked_features_sha256 == second.attacked_features_sha256


def test_development_data_has_no_test_member(development_data) -> None:
    assert development_data.train.split_name == "train"
    assert development_data.validation.split_name == "validation"
    assert not hasattr(development_data, "test")
    assert "test" not in inspect.signature(
        prepare_development_experiment_data
    ).parameters


def test_no_selection_validation_baseline_uses_frozen_v05_tree(
    development_data, protocol_config
) -> None:
    outcome = run_validation_experiment(
        development_data,
        selector=None,
        config=protocol_config,
        run_id="baseline_validation",
    )
    assert outcome.record.phase == "development_validation"
    assert outcome.record.evaluation_split == "validation"
    assert outcome.record.selected_features == development_data.candidate_features
    assert outcome.record.selector["selector_id"] == "none"
    assert outcome.model.parameters == V06_DECISION_TREE_PARAMETERS
    assert outcome.model.estimator.max_depth == 5
    assert outcome.model.estimator.min_samples_leaf == 20
    assert outcome.model.estimator.class_weight == "balanced"
    assert outcome.model.estimator.random_state == 42
    assert outcome.record.prediction_threshold == V06_PREDICTION_THRESHOLD


def test_unsupervised_selector_fits_clean_training_only(
    development_data, protocol_config
) -> None:
    selector = RecordingUnsupervisedSelector(list(development_data.candidate_features))
    outcome = run_validation_experiment(
        development_data,
        selector=selector,
        config=protocol_config,
        run_id="unsupervised_validation",
    )
    assert selector.fit_calls == 1
    assert selector.fit_index == tuple(development_data.train.clean_features.index)
    assert selector.result.labels_used is False
    assert (
        selector.result.training_features_sha256
        == development_data.train.clean_features_sha256
    )
    assert outcome.record.leakage_audit.status == "PASS"


def test_supervised_selector_fits_attacked_training_labels_only(
    development_data, protocol_config
) -> None:
    selector = MutualInformationSelector(
        list(development_data.candidate_features), n_features_to_select=2
    )
    outcome = run_validation_experiment(
        development_data,
        selector=selector,
        config=protocol_config,
        run_id="mi_validation",
    )
    assert selector.result.labels_used is True
    assert (
        selector.result.training_features_sha256
        == development_data.train.attacked_features_sha256
    )
    assert selector.result.training_labels_sha256 == development_data.train.labels_sha256
    assert outcome.record.selected_feature_count == 2
    assert outcome.record.selector_provenance["training_label_source"] == (
        "controlled_training_is_attack"
    )


def test_validation_transform_does_not_refit_selector(
    development_data, protocol_config
) -> None:
    selector = RecordingUnsupervisedSelector(list(development_data.candidate_features))
    run_validation_experiment(
        development_data,
        selector=selector,
        config=protocol_config,
        run_id="single_fit_validation",
    )
    assert selector.fit_calls == 1


def test_poisoned_validation_cannot_change_fitted_selection(
    synthetic_protocol_data, protocol_config
) -> None:
    train, validation, _, candidates, metadata = synthetic_protocol_data
    normal = prepare_development_experiment_data(
        train=train,
        validation=validation,
        candidate_features=candidates,
        dataset_metadata=metadata,
        experiment_id="normal",
    )
    poisoned_validation = ProcessedSplit(
        "validation",
        validation.features.assign(noise=validation.features["noise"] * 1_000_000),
        validation.metadata.copy(),
        None,
    )
    poisoned = prepare_development_experiment_data(
        train=train,
        validation=poisoned_validation,
        candidate_features=candidates,
        dataset_metadata=metadata,
        experiment_id="poisoned",
    )
    first_selector = MutualInformationSelector(list(candidates), n_features_to_select=2)
    second_selector = MutualInformationSelector(list(candidates), n_features_to_select=2)
    run_validation_experiment(
        normal, selector=first_selector, config=protocol_config, run_id="normal_mi"
    )
    run_validation_experiment(
        poisoned,
        selector=second_selector,
        config=protocol_config,
        run_id="poisoned_mi",
    )
    assert first_selector.result.selected_features == second_selector.result.selected_features
    assert (
        first_selector.result.training_features_sha256
        == second_selector.result.training_features_sha256
    )


def test_filter_natural_count_and_ranking_exact_k(
    development_data, protocol_config
) -> None:
    natural = create_v06_selector(
        "variance_threshold",
        development_data.candidate_features,
        protocol_config,
    )
    ranked = create_v06_selector(
        "mutual_information_select_k_best",
        development_data.candidate_features,
        protocol_config,
        n_features_to_select=2,
    )
    natural_run = run_validation_experiment(
        development_data,
        selector=natural,
        config=protocol_config,
        run_id="natural_validation",
    )
    ranked_run = run_validation_experiment(
        development_data,
        selector=ranked,
        config=protocol_config,
        run_id="ranked_validation",
    )
    assert natural_run.record.selected_feature_count == len(
        natural.result.selected_features
    )
    assert ranked_run.record.selected_feature_count == 2


@pytest.mark.parametrize(
    ("selector_id", "feature_count"),
    [
        ("none", None),
        ("variance_threshold", None),
        ("pairwise_correlation_filter", None),
        ("correlation_redundancy_ranking", 2),
        ("mutual_information_select_k_best", 2),
        ("anova_f_select_k_best", 2),
    ],
)
def test_selector_factory_supports_every_v06_c_path(
    development_data,
    protocol_config,
    selector_id: str,
    feature_count: int | None,
) -> None:
    selector = create_v06_selector(
        selector_id,
        development_data.candidate_features,
        protocol_config,
        n_features_to_select=feature_count,
    )
    outcome = run_validation_experiment(
        development_data,
        selector=selector,
        config=protocol_config,
        run_id=f"{selector_id}_validation",
    )
    assert outcome.record.selector["selector_id"] == selector_id
    if feature_count is not None:
        assert outcome.record.selected_feature_count == feature_count


def test_selected_order_is_identical_across_development_splits(
    development_data, protocol_config
) -> None:
    selector = MutualInformationSelector(
        list(development_data.candidate_features), n_features_to_select=2
    )
    outcome = run_validation_experiment(
        development_data,
        selector=selector,
        config=protocol_config,
        run_id="ordered_validation",
    )
    expected = tuple(
        feature
        for feature in development_data.candidate_features
        if feature in selector.result.selected_features
    )
    assert outcome.record.selected_features == expected
    assert tuple(outcome.model.feature_names) == expected


def test_validation_run_is_reproducible(development_data, protocol_config) -> None:
    first = run_validation_experiment(
        development_data,
        selector=MutualInformationSelector(
            list(development_data.candidate_features), n_features_to_select=2
        ),
        config=protocol_config,
        run_id="reproducible",
    )
    second = run_validation_experiment(
        development_data,
        selector=MutualInformationSelector(
            list(development_data.candidate_features), n_features_to_select=2
        ),
        config=protocol_config,
        run_id="reproducible",
    )
    assert first.record.selected_features == second.record.selected_features
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    assert first.record.metrics == second.record.metrics
    assert first.record.prediction_sha256 == second.record.prediction_sha256


def test_result_record_separates_average_precision_and_trapezoidal_pr_auc(
    development_data, protocol_config
) -> None:
    record = run_validation_experiment(
        development_data,
        selector=None,
        config=protocol_config,
        run_id="metrics_validation",
    ).record
    assert "average_precision" in record.metrics
    assert "pr_auc_trapezoidal" in record.metrics
    assert record.metrics["pr_auc_trapezoidal_method"] == (
        "trapezoidal_precision_recall_curve"
    )
    assert record.metrics["pr_auc_status"] == "valid"
    assert record.metrics["roc_auc_status"] == "valid"
    assert "pr_auc" not in record.metrics
    assert record.metrics["attack_prevalence"] == pytest.approx(0.05)


def test_result_record_contains_provenance_and_resource_seams(
    development_data, protocol_config
) -> None:
    record = run_validation_experiment(
        development_data,
        selector=VarianceThresholdSelector(list(development_data.candidate_features)),
        config=protocol_config,
        run_id="provenance_validation",
    ).record
    payload = record.to_dict()
    assert payload["training_provenance"]["row_ids_sha256"]
    assert payload["selector_provenance"]["training_features_sha256"]
    assert payload["split_fingerprints"]["evaluation_rows_sha256"]
    assert payload["leakage_audit"]["status"] == "PASS"
    assert payload["resource_measurement"]["performed"] is False
    assert payload["resource_measurement"]["deferred_stage"] == "V0.6-D"


def test_development_audit_detects_train_validation_contamination(
    development_data, protocol_config
) -> None:
    contaminated = DevelopmentExperimentData(
        candidate_features=development_data.candidate_features,
        train=development_data.train,
        validation=development_data.validation,
        dataset_metadata=development_data.dataset_metadata,
    )
    contaminated.validation.features.loc[
        contaminated.validation.features.index[0], "row_id"
    ] = contaminated.train.features["row_id"].iloc[0]
    audit = audit_development_data(contaminated, protocol_config)
    assert audit.status == "FAIL"
    assert any(
        check.name == "train_validation_rows_disjoint" and not check.passed
        for check in audit.checks
    )


def test_final_test_requires_lock_and_does_not_refit_selector(
    development_data, synthetic_protocol_data, protocol_config
) -> None:
    selector = RecordingUnsupervisedSelector(list(development_data.candidate_features))
    validation = run_validation_experiment(
        development_data,
        selector=selector,
        config=protocol_config,
        run_id="lock_source_validation",
    )
    with pytest.raises(ValueError, match="requires a validation lock"):
        run_locked_test_experiment(None, None, run_id="invalid_test")  # type: ignore[arg-type]
    lock = lock_validation_configuration(validation)
    test_data = _prepare_test(synthetic_protocol_data, protocol_config, lock)
    final = run_locked_test_experiment(lock, test_data, run_id="locked_final_test")
    assert selector.fit_calls == 1
    assert final.record.phase == "final_test"
    assert final.record.evaluation_split == "test"
    assert final.lock_id == lock.lock_id
    assert final.record.training_provenance["selector_refitted_for_test"] is False
    assert final.record.training_provenance["model_retrained_for_test"] is False


def test_locked_test_audit_detects_train_test_contamination(
    development_data, synthetic_protocol_data, protocol_config
) -> None:
    validation = run_validation_experiment(
        development_data,
        selector=None,
        config=protocol_config,
        run_id="contamination_source_validation",
    )
    lock = lock_validation_configuration(validation)
    train, _, _, candidates, _ = synthetic_protocol_data
    overlapping_test = ProcessedSplit(
        "test", train.features.copy(), train.metadata.copy(), None
    )
    test_data = prepare_v06_locked_test_data(
        lock,
        test=overlapping_test,
        training_reference=train,
    )
    audit = audit_locked_test_data(lock, test_data)
    assert audit.status == "FAIL"
    assert any(
        check.name == "train_test_rows_disjoint" and not check.passed
        for check in audit.checks
    )


def test_locked_test_reuses_identical_selected_subset(
    development_data, synthetic_protocol_data, protocol_config
) -> None:
    selector = MutualInformationSelector(
        list(development_data.candidate_features), n_features_to_select=2
    )
    validation = run_validation_experiment(
        development_data,
        selector=selector,
        config=protocol_config,
        run_id="subset_source_validation",
    )
    lock = lock_validation_configuration(validation)
    test_data = _prepare_test(synthetic_protocol_data, protocol_config, lock)
    final = run_locked_test_experiment(lock, test_data, run_id="subset_final_test")
    assert final.record.selected_features == validation.record.selected_features
    assert final.record.selected_features_sha256 == (
        validation.record.selected_features_sha256
    )
    assert tuple(lock.model.feature_names) == final.record.selected_features


def test_locked_test_rejects_post_lock_model_retraining(
    development_data, synthetic_protocol_data, protocol_config
) -> None:
    validation = run_validation_experiment(
        development_data,
        selector=None,
        config=protocol_config,
        run_id="model_integrity_validation",
    )
    lock = lock_validation_configuration(validation)
    test_data = _prepare_test(synthetic_protocol_data, protocol_config, lock)
    shifted_labels = pd.Series(
        np.roll(test_data.labels.to_numpy(), 19),
        index=test_data.labels.index,
        name="is_attack",
    )
    lock.model.fit(test_data.features, shifted_labels)
    with pytest.raises(ValueError, match="integrity check failed before test access"):
        _prepare_test(synthetic_protocol_data, protocol_config, lock)
    audit = audit_locked_test_data(lock, test_data)
    assert audit.status == "FAIL"
    assert any(
        check.name == "model_state_matches_lock" and not check.passed
        for check in audit.checks
    )


def test_lock_rejects_model_mutated_after_validation(
    development_data, protocol_config
) -> None:
    validation = run_validation_experiment(
        development_data,
        selector=None,
        config=protocol_config,
        run_id="pre_lock_integrity_validation",
    )
    validation.model.fit(
        development_data.validation.features,
        pd.Series(
            np.roll(development_data.validation.labels.to_numpy(), 17),
            index=development_data.validation.labels.index,
            name="is_attack",
        ),
    )
    with pytest.raises(ValueError, match="changed before locking"):
        lock_validation_configuration(validation)


def test_locked_test_rejects_mutated_ground_truth(
    development_data, synthetic_protocol_data, protocol_config
) -> None:
    validation = run_validation_experiment(
        development_data,
        selector=None,
        config=protocol_config,
        run_id="truth_integrity_validation",
    )
    lock = lock_validation_configuration(validation)
    test_data = _prepare_test(synthetic_protocol_data, protocol_config, lock)
    test_data.ground_truth.loc[0, "is_attack"] = 1 - int(
        test_data.ground_truth.loc[0, "is_attack"]
    )
    audit = audit_locked_test_data(lock, test_data)
    assert audit.status == "FAIL"
    assert any(
        check.name == "prepared_test_fingerprints_intact" and not check.passed
        for check in audit.checks
    )


def test_test_preparation_requires_matching_lock_authorization(
    development_data, synthetic_protocol_data, protocol_config
) -> None:
    validation = run_validation_experiment(
        development_data,
        selector=None,
        config=protocol_config,
        run_id="authorization_validation",
    )
    lock = lock_validation_configuration(validation)
    train, _, test, candidates, _ = synthetic_protocol_data
    unauthorized = prepare_test_experiment_data(
        test=test,
        training_reference=train,
        candidate_features=candidates,
        experiment_id="unauthorized_test",
        validation_lock_id="wrong_lock",
        authorization_capability=object(),
    )
    audit = audit_locked_test_data(lock, unauthorized)
    assert audit.status == "FAIL"
    assert any(
        check.name == "test_prepared_after_explicit_lock" and not check.passed
        for check in audit.checks
    )


def test_forbidden_and_attack_metadata_features_are_rejected(protocol_config) -> None:
    with pytest.raises(ValueError, match="Forbidden V0.5"):
        create_v06_selector("none", ["safe", "attack_type"], protocol_config)
    with pytest.raises(ValueError, match="Forbidden V0.5"):
        create_v06_selector("none", ["safe", "anomaly_score"], protocol_config)


def test_protocol_rejects_model_or_threshold_changes(protocol_config) -> None:
    with pytest.raises(ValueError, match="frozen V0.5 baseline"):
        replace(protocol_config, model_parameters={"max_depth": 6})
    with pytest.raises(ValueError, match="frozen at 0.5"):
        replace(protocol_config, prediction_threshold=0.6)
    with pytest.raises(ValueError, match="frozen V0.5 baseline"):
        replace(
            protocol_config,
            model_parameters={**V06_DECISION_TREE_PARAMETERS, "random_state": 7},
        )


def test_default_protocol_configuration_is_valid() -> None:
    config = load_v06_protocol_config()
    assert config.experiment_id == "v0_6_development_s42"
    assert config.seed == 42
    assert config.configured_seeds == (42, 43, 44, 45, 46)
    assert config.candidate_feature_counts == (32, 22, 11)
    assert config.attack_type == "mixed"
    assert config.attack_rate == pytest.approx(0.05)
    assert config.attack_severity == "MEDIUM"
    assert dict(config.model_parameters) == V06_DECISION_TREE_PARAMETERS


def test_protocol_configuration_uses_seed_specific_experiment_id() -> None:
    config = load_v06_protocol_config(seed=43)
    assert config.seed == 43
    assert config.experiment_id == "v0_6_development_s43"


@pytest.mark.parametrize("seed", [True, 42.9, 999])
def test_protocol_rejects_invalid_or_unregistered_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        load_v06_protocol_config(seed=seed)  # type: ignore[arg-type]


def test_development_loader_does_not_require_test_files(
    tmp_path, synthetic_protocol_data, protocol_config
) -> None:
    train, validation, _, candidates, metadata = synthetic_protocol_data
    for split in (train, validation):
        split_dir = tmp_path / split.name
        split_dir.mkdir()
        split.features.to_csv(split_dir / "features.csv", index=False)
        split.metadata.to_csv(split_dir / "metadata.csv", index=False)
    (tmp_path / "dataset_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    loaded = load_v06_development_data(protocol_config, processed_dir=tmp_path)
    assert loaded.train.split_name == "train"
    assert loaded.validation.split_name == "validation"
    assert not (tmp_path / "test").exists()
