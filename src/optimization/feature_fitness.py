"""Leakage-safe constrained feature fitness for V0.8-B development data.

This module evaluates generic 43-bit masks against already prepared training and
development-validation workloads. It cannot load or prepare a final-test split.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.ai.model_utils import MODEL_GENERATED_COLUMNS
from src.lightweight.feature_reduction import validate_model_features
from src.lightweight.models import create_model
from src.security.evaluation import detector_visible_mask, evaluate_detection
from src.security.experiment_data import (
    DevelopmentExperimentData,
    fingerprint_feature_names,
)
from src.security.ground_truth import ATTACK_METADATA_COLUMNS, assert_no_attack_metadata


V08B_FITNESS_SCHEMA_VERSION = "v0.8-b-real-fitness-1"
EXPECTED_DIMENSIONS = 43
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_MODEL_NAME = "decision_tree"
EXPECTED_MODEL_PARAMETERS = MappingProxyType(
    {"max_depth": 5, "min_samples_leaf": 20, "class_weight": "balanced"}
)
EXPECTED_THRESHOLD = 0.5
MARGINS = MappingProxyType(
    {"average_precision": 0.05, "f1": 0.05, "recall": 0.10}
)
NUMERICAL_BOUNDARY_TOLERANCE = 1e-12

_VISIBILITY_COLUMNS = frozenset(
    {
        "visibility",
        "visibility_rate",
        "visible_mask",
        "visible_attack",
        "visible_attacks",
        "visible_only_recall",
        "selected_feature_visibility_rate",
    }
)
FORBIDDEN_MODEL_COLUMNS = frozenset(
    {
        "row_id",
        "record_id",
        "Late_delivery_risk",
        "target",
        "Order Status",
        "SUSPECTED_FRAUD",
        "prediction",
        *ATTACK_METADATA_COLUMNS,
        *MODEL_GENERATED_COLUMNS,
        *_VISIBILITY_COLUMNS,
    }
)


@dataclass(frozen=True)
class BaselineMetrics:
    average_precision: float
    f1: float
    recall: float

    def __post_init__(self) -> None:
        values = (self.average_precision, self.f1, self.recall)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Fitness baseline metrics must be finite and positive.")

    def to_dict(self) -> dict[str, float]:
        return {
            "average_precision": self.average_precision,
            "f1": self.f1,
            "recall": self.recall,
        }


@dataclass(frozen=True)
class FitnessContext:
    candidate_features: tuple[str, ...]
    candidate_manifest_sha256: str
    workloads: tuple[DevelopmentExperimentData, ...]
    seeds: tuple[int, ...]
    baseline: BaselineMetrics
    dataset_split_seed: int
    expected_train_rows: int
    expected_validation_rows: int
    model_name: str = EXPECTED_MODEL_NAME
    model_parameters: Mapping[str, Any] = EXPECTED_MODEL_PARAMETERS
    prediction_threshold: float = EXPECTED_THRESHOLD
    attack_type: str = "mixed"
    attack_rate: float = 0.05
    attack_severity: str = "MEDIUM"
    test_accessed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "model_parameters", MappingProxyType(dict(self.model_parameters))
        )


@dataclass(frozen=True)
class AuditCheck:
    name: str
    passed: bool
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class FitnessLeakageAudit:
    checks: tuple[AuditCheck, ...]

    @property
    def status(self) -> str:
        return "PASS" if all(check.passed for check in self.checks) else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": V08B_FITNESS_SCHEMA_VERSION,
            "stage": "V0.8-B",
            "status": self.status,
            "scope": "training_and_development_validation_only",
            "test_accessed": False,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class SeedFitnessRecord:
    seed: int
    selected_feature_count: int
    metrics: Mapping[str, float | int]
    training_rows_sha256: str
    validation_rows_sha256: str
    training_labels_sha256: str
    validation_labels_sha256: str
    validation_features_sha256: str
    visible_attack_count: int
    invisible_attack_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "selected_feature_count": self.selected_feature_count,
            "metrics": dict(self.metrics),
            "training_rows_sha256": self.training_rows_sha256,
            "validation_rows_sha256": self.validation_rows_sha256,
            "training_labels_sha256": self.training_labels_sha256,
            "validation_labels_sha256": self.validation_labels_sha256,
            "validation_features_sha256": self.validation_features_sha256,
            "visible_attack_count": self.visible_attack_count,
            "invisible_attack_count": self.invisible_attack_count,
        }


@dataclass(frozen=True)
class FeatureFitnessEvaluation:
    mask: tuple[int, ...]
    mask_sha256: str
    selected_features: tuple[str, ...]
    selected_features_sha256: str
    selected_feature_count: int
    feasible: bool
    normalized_violation: float
    relative_losses: Mapping[str, float]
    mean_metrics: Mapping[str, float]
    confusion_totals: Mapping[str, int]
    per_seed: tuple[SeedFitnessRecord, ...]
    decision_tree_fit_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "relative_losses", MappingProxyType(dict(self.relative_losses))
        )
        object.__setattr__(self, "mean_metrics", MappingProxyType(dict(self.mean_metrics)))
        object.__setattr__(
            self, "confusion_totals", MappingProxyType(dict(self.confusion_totals))
        )

    @property
    def average_precision(self) -> float:
        return self.mean_metrics["average_precision"]

    @property
    def f1(self) -> float:
        return self.mean_metrics["f1"]

    @property
    def recall(self) -> float:
        return self.mean_metrics["recall"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": V08B_FITNESS_SCHEMA_VERSION,
            "mask": list(self.mask),
            "mask_sha256": self.mask_sha256,
            "selected_features": list(self.selected_features),
            "selected_features_sha256": self.selected_features_sha256,
            "selected_feature_count": self.selected_feature_count,
            "feasible": self.feasible,
            "normalized_violation": self.normalized_violation,
            "relative_losses": dict(self.relative_losses),
            "mean_metrics": dict(self.mean_metrics),
            "confusion_totals": dict(self.confusion_totals),
            "per_seed": [record.to_dict() for record in self.per_seed],
            "decision_tree_fit_count": self.decision_tree_fit_count,
            "fitness_objective": "constrained_minimum_cardinality",
            "weighted_objective_used": False,
            "pareto_objective_used": False,
            "resource_or_energy_objective_used": False,
        }


def relative_loss(baseline: float, candidate: float) -> float:
    """Return nonnegative relative degradation from a positive baseline."""
    if not math.isfinite(baseline) or baseline <= 0.0:
        raise ValueError("Relative loss requires a finite positive baseline.")
    if not math.isfinite(candidate):
        raise ValueError("Relative loss requires a finite candidate metric.")
    return max(0.0, (baseline - candidate) / baseline)


def constraint_outcome(
    baseline: BaselineMetrics,
    mean_metrics: Mapping[str, float],
) -> tuple[dict[str, float], bool, float]:
    """Apply the frozen AP/F1/recall margins and normalized violation."""
    losses = {
        metric: relative_loss(getattr(baseline, metric), float(mean_metrics[metric]))
        for metric in MARGINS
    }
    feasible = all(
        loss <= MARGINS[metric]
        or math.isclose(
            loss,
            MARGINS[metric],
            rel_tol=0.0,
            abs_tol=NUMERICAL_BOUNDARY_TOLERANCE,
        )
        for metric, loss in losses.items()
    )
    violation = sum(
        max(0.0, (loss - MARGINS[metric]) / MARGINS[metric])
        for metric, loss in losses.items()
    )
    if feasible and violation <= NUMERICAL_BOUNDARY_TOLERANCE:
        violation = 0.0
    return losses, feasible, float(violation)


def feature_fitness_is_better(
    left: FeatureFitnessEvaluation,
    right: FeatureFitnessEvaluation,
) -> bool:
    """Apply the approved feasible-first deterministic constrained ranking."""
    if left.feasible != right.feasible:
        return left.feasible
    if left.feasible:
        left_key = (
            left.selected_feature_count,
            -left.average_precision,
            -left.f1,
            -left.recall,
            left.mask,
        )
        right_key = (
            right.selected_feature_count,
            -right.average_precision,
            -right.f1,
            -right.recall,
            right.mask,
        )
    else:
        left_key = (
            left.normalized_violation,
            -left.average_precision,
            -left.f1,
            -left.recall,
            left.selected_feature_count,
            left.mask,
        )
        right_key = (
            right.normalized_violation,
            -right.average_precision,
            -right.f1,
            -right.recall,
            right.selected_feature_count,
            right.mask,
        )
    return left_key < right_key


def audit_fitness_context(context: FitnessContext) -> FitnessLeakageAudit:
    """Return machine-readable evidence that fitness can use development data only."""
    candidates = context.candidate_features
    forbidden = sorted(set(candidates) & FORBIDDEN_MODEL_COLUMNS)
    workload_seeds = tuple(
        int(workload.train.attack_metadata.get("random_seed", -1))
        for workload in context.workloads
    )
    expected_columns = ["row_id", *candidates]
    split_checks: list[bool] = []
    fingerprint_checks: list[bool] = []
    label_checks: list[bool] = []
    attack_checks: list[bool] = []
    preprocessing_checks: list[bool] = []
    for seed, workload in zip(context.seeds, context.workloads):
        train = workload.train
        validation = workload.validation
        split_checks.append(
            train.split_name == "train"
            and validation.split_name == "validation"
            and len(train.features) == context.expected_train_rows
            and len(validation.features) == context.expected_validation_rows
            and set(train.features["row_id"]).isdisjoint(validation.features["row_id"])
            and list(train.features.columns) == expected_columns
            and list(validation.features.columns) == expected_columns
        )
        fingerprint_checks.append(
            train.candidate_features == candidates
            and validation.candidate_features == candidates
            and fingerprint_feature_names(train.candidate_features)
            == context.candidate_manifest_sha256
            and fingerprint_feature_names(validation.candidate_features)
            == context.candidate_manifest_sha256
        )
        label_checks.append(
            train.labels.name == "is_attack"
            and validation.labels.name == "is_attack"
            and set(train.labels.unique()) == {0, 1}
            and set(validation.labels.unique()) == {0, 1}
            and "is_attack" not in train.features
            and "is_attack" not in validation.features
        )
        attack_checks.append(
            train.attack_metadata.get("attack_mode") == context.attack_type
            and validation.attack_metadata.get("attack_mode") == context.attack_type
            and math.isclose(
                float(train.attack_metadata.get("configured_attack_rate", math.nan)),
                context.attack_rate,
            )
            and math.isclose(
                float(validation.attack_metadata.get("configured_attack_rate", math.nan)),
                context.attack_rate,
            )
            and train.attack_metadata.get("severity") == context.attack_severity
            and validation.attack_metadata.get("severity") == context.attack_severity
            and train.attack_metadata.get("random_seed") == seed
            and validation.attack_metadata.get("random_seed") == seed
        )
        metadata = workload.dataset_metadata
        preprocessing_checks.append(
            metadata.get("split", {}).get("seed") == context.dataset_split_seed
            and metadata.get("split", {}).get("strategy") == "order_grouped"
            and metadata.get("preprocessing", {}).get("fitted_on_rows")
            == context.expected_train_rows
            and tuple(metadata.get("preprocessing", {}).get("ml_feature_columns", ()))
            == candidates
        )
    checks = (
        _check(
            "exact_43_feature_manifest",
            len(candidates) == EXPECTED_DIMENSIONS
            and len(set(candidates)) == EXPECTED_DIMENSIONS
            and fingerprint_feature_names(candidates) == context.candidate_manifest_sha256,
            feature_count=len(candidates),
            manifest_sha256=context.candidate_manifest_sha256,
        ),
        _check(
            "forbidden_columns_absent",
            not forbidden,
            forbidden_columns=forbidden,
        ),
        _check(
            "candidate_features_pass_existing_governance",
            _model_features_valid(candidates),
        ),
        _check(
            "only_train_and_development_validation_exposed",
            len(context.workloads) == len(context.seeds)
            and all(not hasattr(workload, "test") for workload in context.workloads),
            workload_count=len(context.workloads),
        ),
        _check(
            "test_access_prohibited",
            context.test_accessed is False,
            test_accessed=context.test_accessed,
            allowed_splits=["train", "validation"],
        ),
        _check(
            "exact_attack_model_seeds",
            context.seeds == EXPECTED_SEEDS and workload_seeds == EXPECTED_SEEDS,
            configured=list(context.seeds),
            workloads=list(workload_seeds),
        ),
        _check(
            "train_validation_split_identity_and_separation",
            bool(split_checks) and all(split_checks),
        ),
        _check(
            "feature_order_and_fingerprint_preserved",
            bool(fingerprint_checks) and all(fingerprint_checks),
        ),
        _check(
            "controlled_is_attack_is_target_only",
            bool(label_checks) and all(label_checks),
            ground_truth="controlled experiment-generated is_attack",
        ),
        _check(
            "late_delivery_risk_not_cyber_ground_truth",
            "Late_delivery_risk" not in candidates,
        ),
        _check(
            "suspected_fraud_not_cyber_ground_truth",
            "SUSPECTED_FRAUD" not in candidates and "Order Status" not in candidates,
        ),
        _check(
            "attack_metadata_predictions_visibility_and_row_ids_excluded",
            not forbidden
            and all(list(workload.train.features.columns) == expected_columns for workload in context.workloads)
            and all(list(workload.validation.features.columns) == expected_columns for workload in context.workloads),
        ),
        _check(
            "preprocessing_frozen_and_training_derived",
            bool(preprocessing_checks) and all(preprocessing_checks),
            split_seed=context.dataset_split_seed,
        ),
        _check(
            "controlled_attack_protocol_frozen",
            bool(attack_checks) and all(attack_checks),
            attack_type=context.attack_type,
            attack_rate=context.attack_rate,
            severity=context.attack_severity,
        ),
        _check(
            "frozen_decision_tree_and_threshold",
            context.model_name == EXPECTED_MODEL_NAME
            and dict(context.model_parameters) == dict(EXPECTED_MODEL_PARAMETERS)
            and context.prediction_threshold == EXPECTED_THRESHOLD,
            model=context.model_name,
            parameters=dict(context.model_parameters),
            threshold=context.prediction_threshold,
        ),
    )
    return FitnessLeakageAudit(checks)


class FeatureFitnessEvaluator:
    """Evaluate one mask across the five frozen development workloads."""

    def __init__(
        self,
        context: FitnessContext,
        *,
        model_factory: Callable[..., Any] = create_model,
        metric_evaluator: Callable[[pd.DataFrame, pd.DataFrame], Mapping[str, Any]] = evaluate_detection,
    ) -> None:
        audit = audit_fitness_context(context)
        if audit.status != "PASS":
            failed = [check.name for check in audit.checks if not check.passed]
            raise ValueError(f"V0.8-B fitness leakage audit failed: {failed}")
        self.context = context
        self.model_factory = model_factory
        self.metric_evaluator = metric_evaluator
        self.evaluation_count = 0
        self.decision_tree_fit_count = 0
        self.evaluation_wall_times: list[float] = []
        self.evaluation_cpu_times: list[float] = []

    def __call__(self, mask: np.ndarray | Sequence[int]) -> FeatureFitnessEvaluation:
        canonical_mask = validate_candidate_mask(mask, len(self.context.candidate_features))
        selected = tuple(
            feature
            for feature, active in zip(self.context.candidate_features, canonical_mask)
            if active
        )
        validate_model_features(selected)
        assert_no_attack_metadata(selected)
        if set(selected) & FORBIDDEN_MODEL_COLUMNS:
            raise ValueError("Candidate mask selected a forbidden model column.")
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        records: list[SeedFitnessRecord] = []
        for seed, workload in zip(self.context.seeds, self.context.workloads):
            train_x = workload.train.features.loc[:, list(selected)].copy()
            validation_x = workload.validation.features.loc[:, list(selected)].copy()
            if list(train_x.columns) != list(selected) or list(validation_x.columns) != list(selected):
                raise RuntimeError("Selected feature order changed during projection.")
            if set(train_x.columns) & FORBIDDEN_MODEL_COLUMNS or set(validation_x.columns) & FORBIDDEN_MODEL_COLUMNS:
                raise RuntimeError("Forbidden columns entered a projected model matrix.")
            model = self.model_factory(
                self.context.model_name,
                selected,
                dict(self.context.model_parameters),
                random_state=seed,
            )
            self._verify_model(model, selected, seed)
            model.fit(train_x, workload.train.labels)
            self.decision_tree_fit_count += 1
            predicted = model.predict_frame(validation_x)
            predictions = pd.DataFrame(
                {
                    "record_id": workload.validation.features["row_id"].to_numpy(copy=True),
                    "anomaly_score": predicted["anomaly_score"].to_numpy(copy=True),
                    "anomaly_label": predicted["anomaly_label"].to_numpy(copy=True),
                }
            )
            raw_metrics = self.metric_evaluator(workload.validation.ground_truth, predictions)
            metrics = _complete_metrics(raw_metrics)
            visible = detector_visible_mask(
                workload.validation.clean_features.loc[:, list(selected)],
                workload.validation.features.loc[:, list(selected)],
            )
            attacked = workload.validation.ground_truth["is_attack"].eq(1).to_numpy()
            visible_count = int((attacked & visible).sum())
            invisible_count = int((attacked & ~visible).sum())
            metrics["selected_feature_visibility_rate"] = (
                float(visible_count / int(attacked.sum())) if attacked.any() else 0.0
            )
            records.append(
                SeedFitnessRecord(
                    seed=seed,
                    selected_feature_count=len(selected),
                    metrics=metrics,
                    training_rows_sha256=workload.train.row_ids_sha256,
                    validation_rows_sha256=workload.validation.row_ids_sha256,
                    training_labels_sha256=workload.train.labels_sha256,
                    validation_labels_sha256=workload.validation.labels_sha256,
                    validation_features_sha256=workload.validation.attacked_features_sha256,
                    visible_attack_count=visible_count,
                    invisible_attack_count=invisible_count,
                )
            )
        mean_fields = (
            "average_precision",
            "f1",
            "recall",
            "precision",
            "roc_auc",
            "accuracy",
            "false_positive_rate",
            "false_negative_rate",
            "attack_prevalence",
            "selected_feature_visibility_rate",
        )
        means = {
            field: float(np.mean([float(record.metrics[field]) for record in records]))
            for field in mean_fields
        }
        losses, feasible, violation = constraint_outcome(self.context.baseline, means)
        confusion = {
            field: int(sum(int(record.metrics[field]) for record in records))
            for field in (
                "true_positives",
                "true_negatives",
                "false_positives",
                "false_negatives",
            )
        }
        evaluation = FeatureFitnessEvaluation(
            mask=tuple(int(value) for value in canonical_mask),
            mask_sha256=hashlib.sha256(canonical_mask.tobytes()).hexdigest(),
            selected_features=selected,
            selected_features_sha256=fingerprint_feature_names(selected),
            selected_feature_count=len(selected),
            feasible=feasible,
            normalized_violation=violation,
            relative_losses=losses,
            mean_metrics=means,
            confusion_totals=confusion,
            per_seed=tuple(records),
            decision_tree_fit_count=len(records),
        )
        self.evaluation_count += 1
        self.evaluation_wall_times.append(time.perf_counter() - wall_start)
        self.evaluation_cpu_times.append(time.process_time() - cpu_start)
        return evaluation

    def _verify_model(self, model: Any, selected: Sequence[str], seed: int) -> None:
        if (
            getattr(model, "model_name", None) != EXPECTED_MODEL_NAME
            or dict(getattr(model, "parameters", {})) != dict(EXPECTED_MODEL_PARAMETERS)
            or getattr(model, "random_state", None) != seed
            or tuple(getattr(model, "feature_names", ())) != tuple(selected)
        ):
            raise RuntimeError("Fitness model differs from the frozen Decision Tree protocol.")

    def instrumentation(self) -> dict[str, Any]:
        wall = self.evaluation_wall_times
        cpu = self.evaluation_cpu_times
        return {
            "objective_evaluations": self.evaluation_count,
            "decision_tree_fits": self.decision_tree_fit_count,
            "mean_evaluation_wall_time_sec": float(np.mean(wall)) if wall else None,
            "min_evaluation_wall_time_sec": float(np.min(wall)) if wall else None,
            "max_evaluation_wall_time_sec": float(np.max(wall)) if wall else None,
            "total_evaluation_wall_time_sec": float(np.sum(wall)) if wall else 0.0,
            "mean_evaluation_cpu_time_sec": float(np.mean(cpu)) if cpu else None,
            "total_evaluation_cpu_time_sec": float(np.sum(cpu)) if cpu else 0.0,
        }


def validate_candidate_mask(mask: np.ndarray | Sequence[int], dimensions: int = 43) -> np.ndarray:
    """Return a canonical uint8 mask after strict shape/binary/nonempty checks."""
    array = np.asarray(mask)
    if array.ndim != 1 or len(array) != dimensions:
        raise ValueError(f"Candidate mask must be one-dimensional with length {dimensions}.")
    if array.dtype.kind not in "biu" or not np.isin(array, (0, 1)).all():
        raise ValueError("Candidate mask must contain binary integer values only.")
    canonical = array.astype(np.uint8, copy=True)
    if int(canonical.sum()) == 0:
        raise ValueError("Candidate mask must select at least one feature.")
    return canonical


def _complete_metrics(raw: Mapping[str, Any]) -> dict[str, float | int]:
    required = (
        "average_precision",
        "f1",
        "recall",
        "precision",
        "roc_auc",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "evaluated_records",
        "attacked_records",
    )
    if any(raw.get(field) is None for field in required):
        raise ValueError("Fitness metric evaluation returned an undefined required metric.")
    tp = int(raw["true_positives"])
    tn = int(raw["true_negatives"])
    fp = int(raw["false_positives"])
    fn = int(raw["false_negatives"])
    evaluated = int(raw["evaluated_records"])
    attacked = int(raw["attacked_records"])
    values: dict[str, float | int] = {
        "average_precision": float(raw["average_precision"]),
        "f1": float(raw["f1"]),
        "recall": float(raw["recall"]),
        "precision": float(raw["precision"]),
        "roc_auc": float(raw["roc_auc"]),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": float((tp + tn) / evaluated) if evaluated else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "attack_prevalence": float(attacked / evaluated) if evaluated else 0.0,
        "evaluated_records": evaluated,
        "attacked_records": attacked,
    }
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError("Fitness metrics must be finite.")
    return values


def _model_features_valid(features: Sequence[str]) -> bool:
    try:
        validate_model_features(features)
        assert_no_attack_metadata(tuple(features))
    except ValueError:
        return False
    return True


def _check(name: str, passed: bool, **evidence: Any) -> AuditCheck:
    return AuditCheck(name, bool(passed), MappingProxyType(dict(evidence)))
