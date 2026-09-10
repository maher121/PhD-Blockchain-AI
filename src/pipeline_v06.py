"""Leakage-safe single-candidate orchestration for V0.6 feature selection."""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, field
import hashlib
from numbers import Integral, Real
from pathlib import Path
import pickle
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.ai.model_utils import ProcessedSplit, load_processed_dataco_splits
from src.config import (
    FEATURE_SELECTION_CONFIG_FILE,
    GLOBAL_SEED,
    PROCESSED_DATA_DIR,
)
from src.feature_selection import (
    AnovaFSelector,
    BaseFeatureSelector,
    CorrelationRedundancySelector,
    MutualInformationSelector,
    PairwiseCorrelationFilter,
    SupervisedFeatureSelector,
    UnsupervisedFeatureSelector,
    VarianceThresholdSelector,
)
from src.lightweight.feature_reduction import validate_model_features
from src.lightweight.models import LightweightDetector, create_model
from src.security.attack_generator import DEFAULT_ATTACK_CONFIG, load_attack_config
from src.security.evaluation import evaluate_detection
from src.security.experiment_data import (
    DevelopmentExperimentData,
    PreparedAttackSplit,
    fingerprint_attack_labels,
    fingerprint_frame,
    fingerprint_feature_matrix,
    fingerprint_feature_names,
    fingerprint_mapping,
    fingerprint_row_ids,
    prepare_development_experiment_data,
    prepare_test_experiment_data,
)
from src.security.ground_truth import assert_no_attack_metadata

V06_PROTOCOL_VERSION = "v0.6-c"
V06_MODEL_NAME = "decision_tree"
V06_DECISION_TREE_PARAMETERS: dict[str, Any] = {
    "max_depth": 5,
    "min_samples_leaf": 20,
    "class_weight": "balanced",
}
V06_PREDICTION_THRESHOLD = 0.5
V06_THRESHOLD_POLICY = (
    "sklearn DecisionTreeClassifier.predict; binary predict_proba decision boundary 0.5; "
    "no threshold tuning"
)
V06_RESOURCE_INTEGRATION_POINTS: tuple[str, ...] = (
    "src.lightweight.resource_monitor.measure_call",
    "src.lightweight.training.train_model",
    "src.lightweight.training.run_inference",
    "src.lightweight.resource_monitor.measure_model_size",
)


@dataclass(frozen=True)
class V06ProtocolConfig:
    """Resolved settings for one V0.6-C development candidate."""

    experiment_id: str = "v0_6_development_s42"
    seed: int = GLOBAL_SEED
    configured_seeds: tuple[int, ...] = (GLOBAL_SEED,)
    candidate_feature_counts: tuple[int, ...] = (32, 22, 11)
    variance_threshold: float = 0.0
    correlation_threshold: float = 0.95
    attack_type: str = "mixed"
    attack_rate: float = 0.05
    attack_severity: str = "MEDIUM"
    attack_config_path: Path = DEFAULT_ATTACK_CONFIG
    model_name: str = V06_MODEL_NAME
    model_parameters: Mapping[str, Any] = field(
        default_factory=lambda: dict(V06_DECISION_TREE_PARAMETERS)
    )
    prediction_threshold: float = V06_PREDICTION_THRESHOLD

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, str) or not self.experiment_id.strip():
            raise ValueError("V0.6 experiment_id must be a non-empty string.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("V0.6 seed must be an integer.")
        seeds = tuple(self.configured_seeds)
        if not seeds or any(isinstance(seed, bool) or not isinstance(seed, Integral) for seed in seeds):
            raise ValueError("V0.6 configured seeds must be non-empty integers.")
        if len(set(int(seed) for seed in seeds)) != len(seeds):
            raise ValueError("V0.6 configured seeds must be unique.")
        if int(self.seed) not in {int(seed) for seed in seeds}:
            raise ValueError("The active V0.6 seed must be registered in configured_seeds.")
        counts = tuple(self.candidate_feature_counts)
        if not counts or any(
            isinstance(count, bool) or not isinstance(count, Integral) or int(count) < 1
            for count in counts
        ):
            raise ValueError("V0.6 candidate feature counts must be positive integers.")
        if len(set(int(count) for count in counts)) != len(counts):
            raise ValueError("V0.6 candidate feature counts must be unique.")
        if float(self.variance_threshold) != 0.0:
            raise ValueError("V0.6 variance threshold is frozen at 0.0.")
        if (
            isinstance(self.correlation_threshold, bool)
            or not isinstance(self.correlation_threshold, Real)
            or not 0.0 < float(self.correlation_threshold) <= 1.0
        ):
            raise ValueError("V0.6 correlation threshold must be in (0, 1].")
        if (
            isinstance(self.attack_rate, bool)
            or not isinstance(self.attack_rate, Real)
            or not 0.0 <= float(self.attack_rate) <= 1.0
        ):
            raise ValueError("V0.6 attack rate must be in [0, 1].")
        if self.model_name != V06_MODEL_NAME:
            raise ValueError("V0.6 primary model is frozen as decision_tree.")
        parameters = dict(self.model_parameters)
        if parameters != V06_DECISION_TREE_PARAMETERS:
            raise ValueError("V0.6 Decision Tree parameters must match the frozen V0.5 baseline.")
        if "random_state" in parameters:
            raise ValueError("Decision Tree random_state must be supplied only by the V0.6 seed.")
        if float(self.prediction_threshold) != V06_PREDICTION_THRESHOLD:
            raise ValueError("V0.6 prediction threshold is frozen at 0.5.")

        attack_config = load_attack_config(self.attack_config_path)
        severity = str(self.attack_severity).upper()
        if severity not in attack_config["severity_levels"]:
            raise ValueError(f"Unsupported V0.6 attack severity: {severity!r}")
        if self.attack_type != "mixed" and self.attack_type not in attack_config["scenarios"]:
            raise ValueError(f"Unsupported V0.6 attack type: {self.attack_type!r}")

        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "configured_seeds", tuple(int(seed) for seed in seeds))
        object.__setattr__(
            self, "candidate_feature_counts", tuple(int(count) for count in counts)
        )
        object.__setattr__(self, "attack_severity", severity)
        object.__setattr__(self, "attack_config_path", Path(self.attack_config_path))
        object.__setattr__(
            self, "model_parameters", MappingProxyType(deepcopy(parameters))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": V06_PROTOCOL_VERSION,
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "configured_seeds": list(self.configured_seeds),
            "candidate_feature_counts": list(self.candidate_feature_counts),
            "variance_threshold": self.variance_threshold,
            "correlation_threshold": self.correlation_threshold,
            "attack": {
                "type": self.attack_type,
                "rate": self.attack_rate,
                "severity": self.attack_severity,
                "config_path": str(self.attack_config_path),
            },
            "model": {
                "name": self.model_name,
                "parameters": dict(self.model_parameters),
                "random_state": self.seed,
                "prediction_threshold": self.prediction_threshold,
                "threshold_policy": V06_THRESHOLD_POLICY,
            },
        }


@dataclass(frozen=True)
class LeakageCheck:
    name: str
    passed: bool
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "evidence": deepcopy(dict(self.evidence)),
        }


@dataclass(frozen=True)
class LeakageAudit:
    checks: tuple[LeakageCheck, ...]

    @property
    def status(self) -> str:
        return "PASS" if all(check.passed for check in self.checks) else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": V06_PROTOCOL_VERSION,
            "status": self.status,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class ExperimentRunRecord:
    run_id: str
    phase: str
    seed: int
    evaluation_split: str
    attack_configuration: Mapping[str, Any]
    selector: Mapping[str, Any]
    candidate_feature_count: int
    selected_feature_count: int
    selected_features: tuple[str, ...]
    selected_features_sha256: str
    model_id: str
    model_parameters: Mapping[str, Any]
    prediction_threshold: float
    threshold_policy: str
    metrics: Mapping[str, Any]
    split_fingerprints: Mapping[str, Any]
    training_provenance: Mapping[str, Any]
    selector_provenance: Mapping[str, Any]
    prediction_sha256: str
    model_state_sha256: str
    leakage_audit: LeakageAudit
    warnings: tuple[str, ...]
    resource_measurement: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": V06_PROTOCOL_VERSION,
            "run_id": self.run_id,
            "phase": self.phase,
            "seed": self.seed,
            "evaluation_split": self.evaluation_split,
            "attack_configuration": deepcopy(dict(self.attack_configuration)),
            "selector": deepcopy(dict(self.selector)),
            "candidate_feature_count": self.candidate_feature_count,
            "selected_feature_count": self.selected_feature_count,
            "selected_features": list(self.selected_features),
            "selected_features_sha256": self.selected_features_sha256,
            "model_id": self.model_id,
            "model_parameters": deepcopy(dict(self.model_parameters)),
            "prediction_threshold": self.prediction_threshold,
            "threshold_policy": self.threshold_policy,
            "metrics": deepcopy(dict(self.metrics)),
            "split_fingerprints": deepcopy(dict(self.split_fingerprints)),
            "training_provenance": deepcopy(dict(self.training_provenance)),
            "selector_provenance": deepcopy(dict(self.selector_provenance)),
            "prediction_sha256": self.prediction_sha256,
            "model_state_sha256": self.model_state_sha256,
            "leakage_audit": self.leakage_audit.to_dict(),
            "warnings": list(self.warnings),
            "resource_measurement": deepcopy(dict(self.resource_measurement)),
        }


@dataclass(frozen=True)
class ValidationExperiment:
    """One fitted candidate evaluated only on validation data."""

    record: ExperimentRunRecord
    model: LightweightDetector
    selector: BaseFeatureSelector | None
    predictions: pd.DataFrame
    training_row_ids: tuple[Any, ...]
    validation_row_ids: tuple[Any, ...]
    selector_result_snapshot: dict[str, Any] | None
    config: V06ProtocolConfig


@dataclass(frozen=True)
class LockedValidationConfiguration:
    """In-memory V0.6-C gate; persisted candidate locking belongs to V0.6-E."""

    lock_id: str
    validation_run_id: str
    candidate_features: tuple[str, ...]
    selected_features: tuple[str, ...]
    selected_features_sha256: str
    model: LightweightDetector
    model_state_sha256: str
    selector: BaseFeatureSelector | None
    selector_result_snapshot: Mapping[str, Any] | None
    training_row_ids: tuple[Any, ...]
    validation_row_ids: tuple[Any, ...]
    config: V06ProtocolConfig
    test_authorization_capability: object = field(repr=False, compare=False)


@dataclass
class FinalTestExperiment:
    """One explicitly locked candidate evaluated on final test data."""

    record: ExperimentRunRecord
    predictions: pd.DataFrame
    lock_id: str


def load_v06_protocol_config(
    path: Path | str = FEATURE_SELECTION_CONFIG_FILE,
    *,
    seed: int | None = None,
) -> V06ProtocolConfig:
    """Load and validate the non-matrix V0.6-C protocol configuration."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"V0.6 configuration not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != V06_PROTOCOL_VERSION:
        raise ValueError(f"V0.6 configuration version must be {V06_PROTOCOL_VERSION!r}.")
    feature_config = payload.get("feature_selection", {})
    attack_config = payload.get("attack", {})
    model_config = payload.get("primary_model", {})
    seeds = tuple(payload.get("seeds", ()))
    if not seeds:
        raise ValueError("V0.6 configuration must define at least one seed.")
    active_seed = seeds[0] if seed is None else seed
    experiment_base = str(payload.get("experiment_id", ""))
    return V06ProtocolConfig(
        experiment_id=f"{experiment_base}_s{active_seed}",
        seed=active_seed,
        configured_seeds=seeds,
        candidate_feature_counts=tuple(feature_config.get("candidate_k", ())),
        variance_threshold=float(feature_config.get("variance_threshold", np.nan)),
        correlation_threshold=float(feature_config.get("correlation_threshold", np.nan)),
        attack_type=str(attack_config.get("type", "")),
        attack_rate=float(attack_config.get("rate", np.nan)),
        attack_severity=str(attack_config.get("severity", "")),
        model_name=str(model_config.get("name", "")),
        model_parameters=model_config.get("parameters", {}),
        prediction_threshold=float(model_config.get("prediction_threshold", np.nan)),
    )


def create_v06_selector(
    selector_id: str,
    candidate_features: Sequence[str],
    config: V06ProtocolConfig,
    *,
    n_features_to_select: int | None = None,
) -> BaseFeatureSelector | None:
    """Create one requested selector without running an experiment matrix."""
    if selector_id == "none":
        if n_features_to_select is not None:
            raise ValueError("The no-selection baseline does not accept K.")
        validate_model_features(candidate_features)
        return None
    if selector_id == VarianceThresholdSelector.selector_id:
        _reject_natural_count(selector_id, n_features_to_select)
        return VarianceThresholdSelector(
            candidate_features,
            threshold=config.variance_threshold,
            random_state=config.seed,
        )
    if selector_id == PairwiseCorrelationFilter.selector_id:
        _reject_natural_count(selector_id, n_features_to_select)
        return PairwiseCorrelationFilter(
            candidate_features,
            threshold=config.correlation_threshold,
            random_state=config.seed,
        )
    if n_features_to_select is None:
        raise ValueError(f"Ranking selector {selector_id!r} requires K.")
    selector_types = {
        CorrelationRedundancySelector.selector_id: CorrelationRedundancySelector,
        MutualInformationSelector.selector_id: MutualInformationSelector,
        AnovaFSelector.selector_id: AnovaFSelector,
    }
    selector_class = selector_types.get(selector_id)
    if selector_class is None:
        raise ValueError(f"Unsupported V0.6 selector: {selector_id!r}")
    return selector_class(
        candidate_features,
        n_features_to_select=n_features_to_select,
        random_state=config.seed,
    )


def load_v06_development_data(
    config: V06ProtocolConfig,
    *,
    processed_dir: Path | str = PROCESSED_DATA_DIR,
) -> DevelopmentExperimentData:
    """Load only train/validation V0.2 splits and prepare controlled attacks."""
    bundle = load_processed_dataco_splits(
        ("train", "validation"), processed_dir=processed_dir
    )
    return prepare_development_experiment_data(
        train=bundle.splits["train"],
        validation=bundle.splits["validation"],
        candidate_features=bundle.selected_features,
        dataset_metadata=bundle.dataset_metadata,
        attack_type=config.attack_type,
        attack_rate=config.attack_rate,
        severity=config.attack_severity,
        random_seed=config.seed,
        experiment_id=config.experiment_id,
        config_path=config.attack_config_path,
    )


def run_validation_experiment(
    data: DevelopmentExperimentData,
    *,
    selector: BaseFeatureSelector | None,
    config: V06ProtocolConfig,
    run_id: str,
) -> ValidationExperiment:
    """Fit one candidate on training and evaluate validation only."""
    _validate_run_id(run_id)
    preliminary_audit = audit_development_data(data, config)
    _require_passing_audit(preliminary_audit)
    _validate_selector_candidates(selector, data.candidate_features)

    selector_snapshot: dict[str, Any] | None = None
    if isinstance(selector, UnsupervisedFeatureSelector):
        selector.fit(data.train.clean_features)
        selector_snapshot = selector.result.to_dict()
        selected_train = selector.transform(data.train.features)
        selected_validation = selector.transform(data.validation.features)
        selector_details = selector.result.to_dict()
    elif isinstance(selector, SupervisedFeatureSelector):
        selector.fit(data.train.features, data.train.labels)
        selector_snapshot = selector.result.to_dict()
        selected_train = selector.transform(data.train.features)
        selected_validation = selector.transform(data.validation.features)
        selector_details = selector.result.to_dict()
    elif selector is None:
        selected_train = data.train.features.loc[:, list(data.candidate_features)].copy()
        selected_validation = data.validation.features.loc[
            :, list(data.candidate_features)
        ].copy()
        selector_details = _no_selection_details(data.candidate_features)
    else:
        raise ValueError("selector must implement the V0.6 feature-selector contract.")

    selected_features = tuple(selected_train.columns)
    if tuple(selected_validation.columns) != selected_features:
        raise RuntimeError("Selected feature order differs between train and validation.")
    selector_unchanged = (
        selector is None or selector.result.to_dict() == selector_snapshot
    )
    model = _create_frozen_decision_tree(selected_features, config)
    model.fit(selected_train, data.train.labels)
    predictions = _prediction_frame(model, selected_validation, data.validation.features)
    metrics = _normalized_metrics(
        evaluate_detection(data.validation.ground_truth, predictions)
    )
    audit = _audit_validation_execution(
        data=data,
        config=config,
        selector=selector,
        selected_features=selected_features,
        model=model,
        preliminary=preliminary_audit,
        selector_unchanged=selector_unchanged,
    )
    _require_passing_audit(audit)
    record = _build_run_record(
        run_id=run_id,
        phase="development_validation",
        evaluation=data.validation,
        training=data.train,
        config=config,
        selector_details=selector_details,
        selected_features=selected_features,
        model=model,
        predictions=predictions,
        metrics=metrics,
        audit=audit,
    )
    return ValidationExperiment(
        record=record,
        model=model,
        selector=selector,
        predictions=predictions,
        training_row_ids=tuple(data.train.features["row_id"]),
        validation_row_ids=tuple(data.validation.features["row_id"]),
        selector_result_snapshot=selector_snapshot,
        config=config,
    )


def lock_validation_configuration(
    validation: ValidationExperiment,
) -> LockedValidationConfiguration:
    """Create the explicit in-memory test gate for one validated candidate."""
    if validation.record.phase != "development_validation":
        raise ValueError("Only a development validation result can be locked.")
    if validation.record.leakage_audit.status != "PASS":
        raise ValueError("A failed validation leakage audit cannot be locked.")
    _validate_validation_record(validation)
    if _model_state_sha256(validation.model) != validation.record.model_state_sha256:
        raise ValueError("Validation model state changed before locking.")
    locked_model = deepcopy(validation.model)
    locked_selector = copy(validation.selector)
    return LockedValidationConfiguration(
        lock_id=f"{validation.record.run_id}__locked",
        validation_run_id=validation.record.run_id,
        candidate_features=tuple(validation.record.selector["candidate_features"]),
        selected_features=validation.record.selected_features,
        selected_features_sha256=validation.record.selected_features_sha256,
        model=locked_model,
        model_state_sha256=_model_state_sha256(locked_model),
        selector=locked_selector,
        selector_result_snapshot=deepcopy(validation.selector_result_snapshot),
        training_row_ids=validation.training_row_ids,
        validation_row_ids=validation.validation_row_ids,
        config=validation.config,
        test_authorization_capability=object(),
    )


def load_v06_locked_test_data(
    lock: LockedValidationConfiguration,
    *,
    processed_dir: Path | str = PROCESSED_DATA_DIR,
) -> PreparedAttackSplit:
    """Load test only after receiving an explicit validation lock."""
    if not isinstance(lock, LockedValidationConfiguration):
        raise ValueError("Final-test data loading requires a validation lock.")
    _validate_lock_integrity(lock)
    bundle = load_processed_dataco_splits(("train", "test"), processed_dir=processed_dir)
    if tuple(bundle.selected_features) != lock.candidate_features:
        raise ValueError("Locked candidates differ from the frozen V0.2 feature manifest.")
    return prepare_v06_locked_test_data(
        lock,
        test=bundle.splits["test"],
        training_reference=bundle.splits["train"],
    )


def prepare_v06_locked_test_data(
    lock: LockedValidationConfiguration,
    *,
    test: ProcessedSplit,
    training_reference: ProcessedSplit,
) -> PreparedAttackSplit:
    """Prepare official test data using an issued in-memory lock capability."""
    if not isinstance(lock, LockedValidationConfiguration):
        raise ValueError("Final-test preparation requires a validation lock.")
    _validate_lock_integrity(lock)
    return prepare_test_experiment_data(
        test=test,
        training_reference=training_reference,
        candidate_features=lock.candidate_features,
        attack_type=lock.config.attack_type,
        attack_rate=lock.config.attack_rate,
        severity=lock.config.attack_severity,
        random_seed=lock.config.seed,
        experiment_id=f"{lock.config.experiment_id}_test",
        config_path=lock.config.attack_config_path,
        validation_lock_id=lock.lock_id,
        authorization_capability=lock.test_authorization_capability,
    )


def run_locked_test_experiment(
    lock: LockedValidationConfiguration,
    test_data: PreparedAttackSplit,
    *,
    run_id: str,
) -> FinalTestExperiment:
    """Evaluate one locked validation candidate without fitting or reselection."""
    if not isinstance(lock, LockedValidationConfiguration):
        raise ValueError("Final-test evaluation requires a validation lock.")
    _validate_run_id(run_id)
    preliminary = audit_locked_test_data(lock, test_data)
    _require_passing_audit(preliminary)

    if lock.selector is None:
        selected_test = test_data.features.loc[:, list(lock.selected_features)].copy()
    else:
        selected_test = lock.selector.transform(test_data.features)
    if tuple(selected_test.columns) != lock.selected_features:
        raise RuntimeError("Final-test feature order differs from the validation lock.")
    if (
        lock.selector is not None
        and lock.selector.result.to_dict() != lock.selector_result_snapshot
    ):
        raise RuntimeError("Fitted selector state changed after validation locking.")

    predictions = _prediction_frame(lock.model, selected_test, test_data.features)
    metrics = _normalized_metrics(evaluate_detection(test_data.ground_truth, predictions))
    audit = audit_locked_test_data(lock, test_data)
    _require_passing_audit(audit)
    record = _build_run_record(
        run_id=run_id,
        phase="final_test",
        evaluation=test_data,
        training=None,
        config=lock.config,
        selector_details=(
            _no_selection_details(lock.candidate_features)
            if lock.selector is None
            else lock.selector.result.to_dict()
        ),
        selected_features=lock.selected_features,
        model=lock.model,
        predictions=predictions,
        metrics=metrics,
        audit=audit,
        training_provenance_override={
            "validation_run_id": lock.validation_run_id,
            "lock_id": lock.lock_id,
            "training_rows_sha256": (
                None
                if lock.selector is None
                else lock.selector.result.training_rows_sha256
            ),
            "model_retrained_for_test": False,
            "selector_refitted_for_test": False,
        },
    )
    return FinalTestExperiment(record=record, predictions=predictions, lock_id=lock.lock_id)


def audit_development_data(
    data: DevelopmentExperimentData, config: V06ProtocolConfig
) -> LeakageAudit:
    """Audit split isolation and inherited V0.2/V0.4 data governance."""
    train_ids = tuple(data.train.features["row_id"])
    validation_ids = tuple(data.validation.features["row_id"])
    metadata = data.dataset_metadata
    fitted_rows = metadata.get("preprocessing", {}).get("fitted_on_rows")
    declared_features = metadata.get("preprocessing", {}).get("ml_feature_columns")
    declared_sizes = metadata.get("split", {}).get("sizes", {})
    checks = [
        _check(
            "development_exposes_train_and_validation_only",
            not hasattr(data, "test"),
            available_splits=[data.train.split_name, data.validation.split_name],
        ),
        _check(
            "split_identities",
            data.train.split_name == "train"
            and data.validation.split_name == "validation",
            train=data.train.split_name,
            validation=data.validation.split_name,
        ),
        _check(
            "train_validation_rows_disjoint",
            set(train_ids).isdisjoint(validation_ids),
            train_count=len(train_ids),
            validation_count=len(validation_ids),
        ),
        _check(
            "prepared_split_fingerprints_intact",
            _prepared_split_is_intact(data.train)
            and _prepared_split_is_intact(data.validation),
            train_rows_sha256=data.train.row_ids_sha256,
            validation_rows_sha256=data.validation.row_ids_sha256,
        ),
        _check(
            "v02_preprocessing_fitted_on_training_rows",
            int(fitted_rows) == len(train_ids) if fitted_rows is not None else False,
            declared_fitted_rows=fitted_rows,
            observed_train_rows=len(train_ids),
        ),
        _check(
            "v02_candidate_manifest_preserved",
            tuple(declared_features or ()) == data.candidate_features,
            declared_count=len(declared_features or ()),
            observed_count=len(data.candidate_features),
        ),
        _check(
            "declared_split_sizes_preserved",
            (
                int(declared_sizes.get("train", -1)) == len(train_ids)
                and int(declared_sizes.get("validation", -1)) == len(validation_ids)
            ),
            declared=declared_sizes,
        ),
        _check(
            "controlled_training_labels",
            data.train.labels.name == "is_attack"
            and set(data.train.labels.unique()) == {0, 1},
            label_name=data.train.labels.name,
        ),
        _check(
            "attack_configuration_matches_protocol",
            _attack_matches(data.train, config)
            and _attack_matches(data.validation, config),
            configured=config.to_dict()["attack"],
        ),
        _check(
            "attack_metadata_absent_from_feature_matrices",
            _features_are_governed(data.train.features, data.candidate_features)
            and _features_are_governed(
                data.validation.features, data.candidate_features
            ),
            candidate_count=len(data.candidate_features),
        ),
    ]
    return LeakageAudit(tuple(checks))


def audit_locked_test_data(
    lock: LockedValidationConfiguration, test_data: PreparedAttackSplit
) -> LeakageAudit:
    """Audit a locked test split before inference is permitted."""
    test_ids = tuple(test_data.features["row_id"])
    selector_matches = (
        lock.selector is None
        or lock.selector.result.to_dict() == lock.selector_result_snapshot
    )
    checks = [
        _check("explicit_validation_lock", bool(lock.lock_id), lock_id=lock.lock_id),
        _check(
            "test_split_identity",
            test_data.split_name == "test",
            split=test_data.split_name,
        ),
        _check(
            "test_prepared_after_explicit_lock",
            test_data.test_authorization_id == lock.lock_id
            and test_data.test_authorization_capability
            is lock.test_authorization_capability,
            authorization_id=test_data.test_authorization_id,
        ),
        _check(
            "prepared_test_fingerprints_intact",
            _prepared_split_is_intact(test_data),
            test_rows_sha256=test_data.row_ids_sha256,
        ),
        _check(
            "train_test_rows_disjoint",
            set(lock.training_row_ids).isdisjoint(test_ids),
            test_count=len(test_ids),
        ),
        _check(
            "validation_test_rows_disjoint",
            set(lock.validation_row_ids).isdisjoint(test_ids),
            test_count=len(test_ids),
        ),
        _check(
            "candidate_manifest_matches_lock",
            test_data.candidate_features == lock.candidate_features,
            locked_count=len(lock.candidate_features),
            test_count=len(test_data.candidate_features),
        ),
        _check(
            "selected_feature_fingerprint_matches_lock",
            fingerprint_feature_names(lock.selected_features)
            == lock.selected_features_sha256,
            selected_count=len(lock.selected_features),
        ),
        _check(
            "selector_state_unchanged",
            selector_matches,
            selector_id=("none" if lock.selector is None else lock.selector.selector_id),
        ),
        _check(
            "model_matches_lock",
            tuple(lock.model.feature_names) == lock.selected_features
            and lock.model.model_name == V06_MODEL_NAME
            and lock.model.random_state == lock.config.seed
            and lock.model.parameters == V06_DECISION_TREE_PARAMETERS,
            model=lock.model.model_name,
        ),
        _check(
            "model_state_matches_lock",
            _model_state_sha256(lock.model) == lock.model_state_sha256,
            model_state_sha256=lock.model_state_sha256,
        ),
        _check(
            "test_attack_configuration_matches_lock",
            _attack_matches(test_data, lock.config),
            configured=lock.config.to_dict()["attack"],
        ),
        _check(
            "attack_metadata_absent_from_test_features",
            _features_are_governed(test_data.features, lock.candidate_features),
            candidate_count=len(lock.candidate_features),
        ),
    ]
    return LeakageAudit(tuple(checks))


def _audit_validation_execution(
    *,
    data: DevelopmentExperimentData,
    config: V06ProtocolConfig,
    selector: BaseFeatureSelector | None,
    selected_features: tuple[str, ...],
    model: LightweightDetector,
    preliminary: LeakageAudit,
    selector_unchanged: bool,
) -> LeakageAudit:
    checks = list(preliminary.checks)
    canonical_selected = tuple(
        feature for feature in data.candidate_features if feature in selected_features
    )
    if selector is None:
        selector_rows_match = True
        selector_features_match = selected_features == data.candidate_features
        selector_labels_match = True
        selector_candidate_match = True
        labels_used = False
    else:
        expected_features_sha256 = (
            data.train.attacked_features_sha256
            if isinstance(selector, SupervisedFeatureSelector)
            else data.train.clean_features_sha256
        )
        selector_rows_match = (
            selector.result.training_rows_sha256 == data.train.row_ids_sha256
        )
        selector_features_match = (
            selector.result.training_features_sha256 == expected_features_sha256
        )
        selector_labels_match = (
            selector.result.training_labels_sha256 == data.train.labels_sha256
            if isinstance(selector, SupervisedFeatureSelector)
            else selector.result.training_labels_sha256 is None
        )
        selector_candidate_match = (
            selector.result.candidate_features_sha256
            == fingerprint_feature_names(data.candidate_features)
        )
        labels_used = selector.result.labels_used
    checks.extend(
        [
            _check(
                "selector_fitted_on_training_rows_only",
                selector_rows_match
                and (
                    selector is None
                    or selector.result.training_rows_sha256
                    != data.validation.row_ids_sha256
                ),
                training_rows_sha256=data.train.row_ids_sha256,
                validation_rows_sha256=data.validation.row_ids_sha256,
            ),
            _check(
                "selector_training_representation",
                selector_features_match,
                selector_type=("none" if selector is None else selector.selector_type),
            ),
            _check(
                "selector_label_governance",
                selector_labels_match
                and (
                    labels_used
                    if isinstance(selector, SupervisedFeatureSelector)
                    else not labels_used
                ),
                labels_used=labels_used,
            ),
            _check(
                "selector_candidate_manifest",
                selector_candidate_match,
                candidate_count=len(data.candidate_features),
            ),
            _check(
                "selected_features_preserve_canonical_order",
                selected_features == canonical_selected,
                selected_features=list(selected_features),
            ),
            _check(
                "validation_transform_did_not_refit_selector",
                selector_unchanged,
                selector_id=("none" if selector is None else selector.selector_id),
            ),
            _check(
                "fixed_v05_decision_tree",
                model.model_name == V06_MODEL_NAME
                and model.parameters == V06_DECISION_TREE_PARAMETERS
                and model.random_state == config.seed
                and tuple(model.feature_names) == selected_features,
                parameters=model.parameters,
            ),
            _check(
                "fixed_prediction_threshold",
                config.prediction_threshold == V06_PREDICTION_THRESHOLD,
                threshold=config.prediction_threshold,
                policy=V06_THRESHOLD_POLICY,
            ),
        ]
    )
    return LeakageAudit(tuple(checks))


def _build_run_record(
    *,
    run_id: str,
    phase: str,
    evaluation: PreparedAttackSplit,
    training: PreparedAttackSplit | None,
    config: V06ProtocolConfig,
    selector_details: Mapping[str, Any],
    selected_features: tuple[str, ...],
    model: LightweightDetector,
    predictions: pd.DataFrame,
    metrics: Mapping[str, Any],
    audit: LeakageAudit,
    training_provenance_override: Mapping[str, Any] | None = None,
) -> ExperimentRunRecord:
    selector_provenance = {
        "selector_id": selector_details["selector_id"],
        "selector_type": selector_details["selector_type"],
        "labels_used": selector_details["labels_used"],
        "training_rows_sha256": selector_details.get("training_rows_sha256"),
        "training_features_sha256": selector_details.get("training_features_sha256"),
        "training_labels_sha256": selector_details.get("training_labels_sha256"),
        "training_label_source": selector_details.get("training_label_source"),
    }
    training_provenance = (
        {
            "split": training.split_name,
            "row_ids_sha256": training.row_ids_sha256,
            "clean_features_sha256": training.clean_features_sha256,
            "attacked_features_sha256": training.attacked_features_sha256,
            "labels_sha256": training.labels_sha256,
        }
        if training is not None
        else dict(training_provenance_override or {})
    )
    return ExperimentRunRecord(
        run_id=run_id,
        phase=phase,
        seed=config.seed,
        evaluation_split=evaluation.split_name,
        attack_configuration={
            **config.to_dict()["attack"],
            "generator_metadata": deepcopy(evaluation.attack_metadata),
        },
        selector=deepcopy(dict(selector_details)),
        candidate_feature_count=len(selector_details["candidate_features"]),
        selected_feature_count=len(selected_features),
        selected_features=selected_features,
        selected_features_sha256=fingerprint_feature_names(selected_features),
        model_id=V06_MODEL_NAME,
        model_parameters=deepcopy(model.parameters),
        prediction_threshold=V06_PREDICTION_THRESHOLD,
        threshold_policy=V06_THRESHOLD_POLICY,
        metrics=deepcopy(dict(metrics)),
        split_fingerprints={
            "evaluation_split": evaluation.split_name,
            "evaluation_rows_sha256": evaluation.row_ids_sha256,
            "evaluation_features_sha256": evaluation.attacked_features_sha256,
            "evaluation_labels_sha256": evaluation.labels_sha256,
        },
        training_provenance=training_provenance,
        selector_provenance=selector_provenance,
        prediction_sha256=sha256_predictions(predictions),
        model_state_sha256=_model_state_sha256(model),
        leakage_audit=audit,
        warnings=tuple(selector_details.get("warnings", ())),
        resource_measurement={
            "performed": False,
            "deferred_stage": "V0.6-D",
            "integration_points": V06_RESOURCE_INTEGRATION_POINTS,
            "energy_claim": False,
        },
    )


def sha256_predictions(predictions: pd.DataFrame) -> str:
    from src.data.versioning import sha256_frame

    return sha256_frame(predictions.reset_index(drop=True))


def _create_frozen_decision_tree(
    selected_features: Sequence[str], config: V06ProtocolConfig
) -> LightweightDetector:
    model = create_model(
        V06_MODEL_NAME,
        selected_features,
        dict(config.model_parameters),
        random_state=config.seed,
    )
    estimator = model.estimator
    if (
        estimator.max_depth != 5
        or estimator.min_samples_leaf != 20
        or estimator.class_weight != "balanced"
        or estimator.random_state != config.seed
    ):
        raise RuntimeError("Constructed Decision Tree differs from the frozen V0.5 baseline.")
    return model


def _prediction_frame(
    model: LightweightDetector,
    selected_features: pd.DataFrame,
    traced_features: pd.DataFrame,
) -> pd.DataFrame:
    predicted = model.predict_frame(selected_features)
    return pd.DataFrame(
        {
            "record_id": traced_features["row_id"].to_numpy(copy=True),
            "anomaly_score": predicted["anomaly_score"].to_numpy(copy=True),
            "anomaly_label": predicted["anomaly_label"].to_numpy(copy=True),
        }
    )


def _normalized_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    evaluated = int(metrics["evaluated_records"])
    attacked = int(metrics["attacked_records"])
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "average_precision": metrics["average_precision"],
        "pr_auc_trapezoidal": metrics["pr_auc"],
        "pr_auc_trapezoidal_method": metrics["pr_auc_method"],
        "pr_auc_status": (
            "valid" if metrics["pr_auc"] is not None else "undefined_single_class"
        ),
        "roc_auc": metrics["roc_auc"],
        "roc_auc_status": (
            "valid" if metrics["roc_auc"] is not None else "undefined_single_class"
        ),
        "true_positives": metrics["true_positives"],
        "true_negatives": metrics["true_negatives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "confusion_matrix": metrics["confusion_matrix"],
        "evaluated_records": evaluated,
        "attacked_records": attacked,
        "attack_prevalence": float(attacked / evaluated) if evaluated else None,
    }


def _no_selection_details(candidate_features: Sequence[str]) -> dict[str, Any]:
    candidates = tuple(candidate_features)
    return {
        "contract_version": V06_PROTOCOL_VERSION,
        "selector_id": "none",
        "selector_type": "none",
        "method": "no_feature_selection",
        "parameters": {},
        "labels_used": False,
        "candidate_feature_count": len(candidates),
        "selected_feature_count": len(candidates),
        "candidate_features": candidates,
        "selected_features": candidates,
        "selected_indices": tuple(range(len(candidates))),
        "support_mask": tuple(True for _ in candidates),
        "random_state": None,
        "training_rows_sha256": None,
        "training_features_sha256": None,
        "training_labels_sha256": None,
        "training_label_source": None,
        "warnings": (),
        "diagnostics": {"comparison_anchor": "frozen_v05_decision_tree_f43"},
    }


def _attack_matches(split: PreparedAttackSplit, config: V06ProtocolConfig) -> bool:
    metadata = split.attack_metadata
    mode_matches = (
        metadata.get("attack_mode") == "mixed"
        if config.attack_type == "mixed"
        else metadata.get("attack_types") == [config.attack_type]
    )
    return bool(
        mode_matches
        and np.isclose(metadata.get("configured_attack_rate", np.nan), config.attack_rate)
        and metadata.get("severity") == config.attack_severity
        and metadata.get("random_seed") == config.seed
    )


def _features_are_governed(
    features: pd.DataFrame, candidate_features: Sequence[str]
) -> bool:
    try:
        validate_model_features(candidate_features)
        assert_no_attack_metadata(features)
    except ValueError:
        return False
    return list(features.columns) == ["row_id", *candidate_features]


def _prepared_split_is_intact(split: PreparedAttackSplit) -> bool:
    try:
        aligned_truth = (
            split.ground_truth.set_index("record_id")
            .loc[split.features["row_id"], "is_attack"]
            .astype("int8")
        )
        aligned_truth.index = split.features.index
        aligned_truth.name = "is_attack"
        return bool(
            fingerprint_row_ids(split.features) == split.row_ids_sha256
            and fingerprint_feature_matrix(
                split.clean_features, split.candidate_features
            )
            == split.clean_features_sha256
            and fingerprint_feature_matrix(split.features, split.candidate_features)
            == split.attacked_features_sha256
            and fingerprint_attack_labels(split.features, split.labels)
            == split.labels_sha256
            and split.labels.equals(aligned_truth)
            and fingerprint_frame(split.ground_truth) == split.ground_truth_sha256
            and fingerprint_frame(split.manifest) == split.manifest_sha256
            and fingerprint_mapping(split.attack_metadata)
            == split.attack_metadata_sha256
        )
    except (KeyError, TypeError, ValueError):
        return False


def _check(name: str, passed: bool, **evidence: Any) -> LeakageCheck:
    return LeakageCheck(name=name, passed=bool(passed), evidence=evidence)


def _require_passing_audit(audit: LeakageAudit) -> None:
    if audit.status != "PASS":
        failed = [check.name for check in audit.checks if not check.passed]
        raise ValueError(f"V0.6 leakage audit failed: {failed}")


def _validate_selector_candidates(
    selector: BaseFeatureSelector | None, candidate_features: Sequence[str]
) -> None:
    if selector is not None and selector.candidate_features != tuple(candidate_features):
        raise ValueError("Selector candidates must match the frozen feature manifest.")


def _reject_natural_count(selector_id: str, count: int | None) -> None:
    if count is not None:
        raise ValueError(f"Natural-count selector {selector_id!r} does not accept K.")


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string.")


def _validate_validation_record(validation: ValidationExperiment) -> None:
    record = validation.record
    config = validation.config
    configured_attack = record.attack_configuration
    if (
        record.seed != config.seed
        or record.model_id != V06_MODEL_NAME
        or dict(record.model_parameters) != V06_DECISION_TREE_PARAMETERS
        or record.prediction_threshold != V06_PREDICTION_THRESHOLD
        or configured_attack.get("type") != config.attack_type
        or not np.isclose(configured_attack.get("rate", np.nan), config.attack_rate)
        or configured_attack.get("severity") != config.attack_severity
        or record.selected_features_sha256
        != fingerprint_feature_names(record.selected_features)
    ):
        raise ValueError("Validation record and protocol configuration are inconsistent.")


def _validate_lock_integrity(lock: LockedValidationConfiguration) -> None:
    selector_unchanged = (
        lock.selector is None
        or lock.selector.result.to_dict() == lock.selector_result_snapshot
    )
    if (
        not selector_unchanged
        or fingerprint_feature_names(lock.selected_features)
        != lock.selected_features_sha256
        or _model_state_sha256(lock.model) != lock.model_state_sha256
        or lock.model.model_name != V06_MODEL_NAME
        or lock.model.parameters != V06_DECISION_TREE_PARAMETERS
        or lock.model.random_state != lock.config.seed
        or tuple(lock.model.feature_names) != lock.selected_features
    ):
        raise ValueError("Validation lock integrity check failed before test access.")


def _model_state_sha256(model: LightweightDetector) -> str:
    payload = pickle.dumps(model.estimator, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(payload).hexdigest()
