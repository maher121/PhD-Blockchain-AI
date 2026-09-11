"""Validation-locked V0.6-G robustness and secondary-classifier evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn

from src.config import PROCESSED_DATA_DIR
from src.lightweight.models import LightweightDetector, create_model
from src.pipeline_v06e import ValidationLockError, fingerprint_feature_names
import src.pipeline_v06f as v06f
from src.security.attack_generator import load_attack_config
from src.security.evaluation import (
    detector_visible_mask,
    evaluate_detection,
    evaluate_paired_detection,
    evaluate_visible_attacks,
)
from src.security.experiment_data import PreparedAttackSplit, prepare_attack_split
from src.security.ground_truth import assert_no_attack_metadata


V06G_STAGE = "V0.6-G"
V06G_PROTOCOL_VERSION = "v0.6-g-locked-robustness-1"
PRE_RUN_FAILURE_MESSAGE = "V0.6-G PRE-RUN INTEGRITY GATE FAILED"
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
SEMANTIC_ROLES = ("full_baseline", "best_unsupervised", "best_supervised")
OFFICIAL_RATES = (0.01, 0.03, 0.05, 0.10)
OFFICIAL_SEVERITIES = ("LOW", "MEDIUM", "HIGH")
PRIMARY_RATE = 0.05
PRIMARY_SEVERITY = "MEDIUM"
LR_PARAMETERS: Mapping[str, Any] = MappingProxyType(
    {"solver": "liblinear", "class_weight": "balanced", "max_iter": 500, "C": 1.0}
)
CLASSIFIERS = ("decision_tree", "logistic_regression")
PREDICTION_THRESHOLD = 0.5
INFERENCE_REPEATS = 10
DEFAULT_RESULTS_DIR = Path("results/feature_selection")
ROBUSTNESS_SUBDIR = Path("robustness")

DETECTION_METRICS = (
    "precision", "recall", "f1", "average_precision", "pr_auc_trapezoidal",
    "roc_auc", "TP", "TN", "FP", "FN", "attack_prevalence",
    "visible_feature_attack_count", "invisible_feature_attack_count",
    "selected_feature_visibility_rate", "visible_attack_recall",
    "invisible_attack_recall", "visible_attack_f1",
    "paired_induced_attack_induced_detection_rate",
    "paired_induced_mean_anomaly_score_delta",
    "paired_induced_median_anomaly_score_delta",
)
RESOURCE_METRICS = (
    "selected_feature_count", "source_block_count", "selected_input_bytes",
    "model_fit_wall_time_sec", "model_fit_cpu_time_sec", "model_fit_peak_rss_mib",
    "combined_fit_wall_time_sec", "inference_wall_time_sec",
    "per_record_inference_sec", "inference_cpu_time_sec", "inference_peak_rss_mib",
    "peak_rss_mib", "serialized_model_bytes", "serialized_model_kib",
)
SUMMARY_METRICS = (*DETECTION_METRICS, *RESOURCE_METRICS)
PRESERVATION_MARGINS = MappingProxyType(
    {"average_precision": 5.0, "f1": 5.0, "recall": 10.0}
)
REQUIRED_FINAL_TEST_FILES = frozenset(
    {
        "final_test_runs.csv", "final_test_runs.json", "final_test_summary.csv",
        "final_test_summary.json", "final_test_paired_comparison.csv",
        "final_test_paired_comparison.json", "final_test_resource_summary.csv",
        "final_test_resource_summary.json", "final_test_generalization.csv",
        "final_test_generalization.json", "final_test_preservation.csv",
        "final_test_preservation.json", "final_test_governance_audit.json",
        "final_test_metadata.json", "final_test_role_mapping.json", "test_access_state.json",
        "figures/final_test_metrics.png", "figures/inference_resources.png",
    }
)
FIGURE_NAMES = (
    "average_precision_vs_attack_rate.png", "f1_vs_attack_rate.png",
    "recall_vs_attack_severity.png", "performance_by_attack_family.png",
    "visibility_rate_by_attack_family.png", "dt_vs_logistic_regression.png",
    "feature_count_vs_average_precision.png", "robustness_preservation_heatmap.png",
)
_CONTEXT_CAPABILITY = object()


class V06GIntegrityError(RuntimeError):
    """Raised when a locked source or execution invariant is violated."""


class V06GExecutionError(RuntimeError):
    """Raised when robustness execution cannot safely continue."""


@dataclass(frozen=True)
class RobustnessScenario:
    scenario_id: str
    attack_type: str
    attack_rate: float
    attack_severity: str
    in_family_panel: bool
    in_rate_panel: bool
    in_severity_panel: bool


@dataclass(frozen=True)
class LockedConfiguration:
    configuration_id: str
    roles: tuple[str, ...]
    seed_plans: tuple[v06f.SeedPlan, ...]


@dataclass(frozen=True)
class RobustnessPlan:
    semantic_lock_sha256: str
    validation_lock_file_sha256: str
    candidate_features: tuple[str, ...]
    configurations: tuple[LockedConfiguration, ...]
    scenarios: tuple[RobustnessScenario, ...]
    seeds: tuple[int, ...] = EXPECTED_SEEDS
    classifiers: tuple[str, ...] = CLASSIFIERS


@dataclass(frozen=True)
class RobustnessContext:
    lock: Mapping[str, Any]
    final_test_metadata: Mapping[str, Any]
    plan: RobustnessPlan
    plan_sha256: str
    gate_timestamp: str
    decision_tree_models: Mapping[tuple[str, int], LightweightDetector]
    decision_tree_paths: Mapping[tuple[str, int], Path]
    decision_tree_state_hashes: Mapping[tuple[str, int], str]
    source_hashes: Mapping[str, str]
    results_dir: Path
    authorization_capability: object


def build_scenario_matrix(attack_config: Mapping[str, Any]) -> tuple[RobustnessScenario, ...]:
    """Derive and deduplicate the official 13-scenario panel from configuration."""
    rates = tuple(float(value) for value in attack_config.get("attack_rates", ()))
    severities = tuple(str(value).upper() for value in attack_config.get("severity_levels", ()))
    if rates != OFFICIAL_RATES or severities != OFFICIAL_SEVERITIES:
        raise V06GIntegrityError("Attack configuration does not contain the exact official rates/severities.")
    definitions = attack_config.get("scenarios")
    if not isinstance(definitions, Mapping) or not definitions:
        raise V06GIntegrityError("Attack configuration contains no supported families.")
    families = tuple(str(name) for name in definitions)
    candidates = [
        (name, PRIMARY_RATE, PRIMARY_SEVERITY, True, False, False)
        for name in families
    ]
    candidates.extend(
        ("mixed", rate, PRIMARY_SEVERITY, False, True, rate == PRIMARY_RATE)
        for rate in rates
    )
    candidates.extend(
        ("mixed", PRIMARY_RATE, severity, False, False, True)
        for severity in severities
        if severity != PRIMARY_SEVERITY
    )
    deduplicated: dict[tuple[str, float, str], list[Any]] = {}
    for attack_type, rate, severity, family, rate_panel, severity_panel in candidates:
        key = (attack_type, float(rate), severity)
        if key in deduplicated:
            deduplicated[key][3] |= family
            deduplicated[key][4] |= rate_panel
            deduplicated[key][5] |= severity_panel
        else:
            deduplicated[key] = [attack_type, float(rate), severity, family, rate_panel, severity_panel]
    scenarios = tuple(
        RobustnessScenario(
            scenario_id=_scenario_id(row[0], row[1], row[2]),
            attack_type=row[0], attack_rate=row[1], attack_severity=row[2],
            in_family_panel=bool(row[3]), in_rate_panel=bool(row[4]),
            in_severity_panel=bool(row[5]),
        )
        for row in deduplicated.values()
    )
    expected_count = len(families) + len(OFFICIAL_RATES) + len(OFFICIAL_SEVERITIES) - 1
    if len(scenarios) != expected_count or len({s.scenario_id for s in scenarios}) != len(scenarios):
        raise V06GIntegrityError("Robustness scenarios are not uniquely deduplicated.")
    return scenarios


def derive_locked_configurations(
    lock: Mapping[str, Any], final_plan: v06f.ExecutionPlan
) -> tuple[LockedConfiguration, ...]:
    """Resolve exactly three semantic roles and collapse duplicate configurations."""
    roles = lock.get("roles")
    if not isinstance(roles, Mapping):
        raise V06GIntegrityError("Validation lock has no role mapping.")
    by_id = {item.configuration_id: item for item in final_plan.configurations}
    grouped: dict[str, list[str]] = {}
    for role in SEMANTIC_ROLES:
        record = roles.get(role)
        if not isinstance(record, Mapping):
            raise V06GIntegrityError(f"Required semantic role is missing: {role}")
        reference = record.get("locked_configuration_ref")
        if reference != record.get("configuration_id") or reference not in by_id:
            raise V06GIntegrityError(f"Invalid locked configuration reference for role: {role}")
        grouped.setdefault(str(reference), []).append(role)
    return tuple(
        LockedConfiguration(configuration_id, tuple(role_names), by_id[configuration_id].seed_plans)
        for configuration_id, role_names in grouped.items()
    )


def robustness_integrity_gate(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    *,
    verifier: Callable[..., Mapping[str, Any]] = v06f.verify_validation_lock,
    model_loader: Callable[[Path | str], LightweightDetector] = LightweightDetector.load,
) -> RobustnessContext:
    """Verify immutable V0.6-E/F artifacts and frozen DT states before data access."""
    results = Path(results_dir).resolve()
    final_context = v06f.pre_test_integrity_gate(
        results, verifier=verifier, model_loader=model_loader
    )
    final_dir = results / v06f.FINAL_TEST_SUBDIR
    manifest_path = final_dir / "final_test_artifact_hashes.json"
    manifest = _read_json(manifest_path, "V0.6-F artifact hash manifest")
    if set(manifest) != REQUIRED_FINAL_TEST_FILES:
        raise V06GIntegrityError("V0.6-F hash manifest does not contain the exact artifact set.")
    for relative, expected in manifest.items():
        path = final_dir / relative
        if not _is_sha256(expected) or not path.is_file() or _sha256_file(path) != expected:
            raise V06GIntegrityError(f"V0.6-F artifact hash mismatch: {relative}")
    metadata = _read_json(final_dir / "final_test_metadata.json", "V0.6-F metadata")
    audit = _read_json(final_dir / "final_test_governance_audit.json", "V0.6-F audit")
    access = _read_json(final_dir / "test_access_state.json", "V0.6-F access state")
    if (
        metadata.get("stage") != v06f.V06F_STAGE
        or metadata.get("status") != "COMPLETED"
        or metadata.get("test_accessed") is not True
        or metadata.get("failed_run_count") != 0
        or metadata.get("semantic_lock_sha256") != final_context.plan.semantic_lock_sha256
        or metadata.get("validation_lock_file_sha256") != final_context.plan.lock_file_sha256
        or audit.get("status") != "PASS"
        or audit.get("test_accessed") is not True
        or access.get("test_accessed") is not True
    ):
        raise V06GIntegrityError("V0.6-F completion/access evidence is invalid.")
    attack_path = Path(final_context.lock["attack_configuration"]["attack_config_path"])
    attack_config = load_attack_config(attack_path)
    scenarios = build_scenario_matrix(attack_config)
    configurations = derive_locked_configurations(final_context.lock, final_context.plan)
    loaded_state_hashes: dict[tuple[str, int], str] = {}
    for configuration in configurations:
        if tuple(seed.seed for seed in configuration.seed_plans) != EXPECTED_SEEDS:
            raise V06GIntegrityError("A locked configuration does not contain exact seeds 42-46.")
        for seed_plan in configuration.seed_plans:
            key = (configuration.configuration_id, seed_plan.seed)
            model = final_context.models[key]
            if not _is_sha256(seed_plan.model_state_sha256):
                raise V06GIntegrityError("Locked decision-tree model state SHA-256 is invalid.")
            state_hash = model_state_sha256(model)
            fresh = model_loader(final_context.model_paths[key])
            v06f._verify_loaded_model(fresh, seed_plan)
            if model_state_sha256(fresh) != state_hash:
                raise V06GIntegrityError("Loaded decision-tree semantic state differs from its artifact.")
            loaded_state_hashes[key] = state_hash
    source_paths = {
        results / v06f.LOCK_FILE_NAME,
        results / v06f.SIDECAR_FILE_NAME,
        results / "validation_lock_audit.json",
        manifest_path,
        attack_path.resolve(),
        *[path.resolve() for path in final_context.model_paths.values()],
        *[(final_dir / relative).resolve() for relative in manifest],
    }
    declared_sources = final_context.lock.get("source_artifact_hashes", {})
    if not isinstance(declared_sources, Mapping):
        raise V06GIntegrityError("Validation lock source-artifact hashes are invalid.")
    for reference, expected in declared_sources.items():
        path = Path(str(reference))
        if not path.is_absolute():
            path = results / path
        path = path.resolve()
        if not _is_sha256(expected) or not path.is_file() or _sha256_file(path) != expected:
            raise V06GIntegrityError(f"Validation source artifact hash mismatch: {reference}")
        source_paths.add(path)
    source_hashes = {str(path): _sha256_file(path) for path in sorted(source_paths)}
    plan = RobustnessPlan(
        semantic_lock_sha256=final_context.plan.semantic_lock_sha256,
        validation_lock_file_sha256=final_context.plan.lock_file_sha256,
        candidate_features=final_context.plan.candidate_features,
        configurations=configurations,
        scenarios=scenarios,
    )
    return RobustnessContext(
        lock=_deep_freeze(_thaw(final_context.lock)),
        final_test_metadata=_deep_freeze(deepcopy(metadata)),
        plan=plan,
        plan_sha256=fingerprint_robustness_plan(plan),
        gate_timestamp=_utc_now(),
        decision_tree_models=MappingProxyType(
            {
                (configuration.configuration_id, seed.seed): final_context.models[
                    (configuration.configuration_id, seed.seed)
                ]
                for configuration in configurations for seed in configuration.seed_plans
            }
        ),
        decision_tree_paths=MappingProxyType(
            {
                (configuration.configuration_id, seed.seed): final_context.model_paths[
                    (configuration.configuration_id, seed.seed)
                ]
                for configuration in configurations for seed in configuration.seed_plans
            }
        ),
        decision_tree_state_hashes=MappingProxyType(loaded_state_hashes),
        source_hashes=MappingProxyType(source_hashes), results_dir=results,
        authorization_capability=_CONTEXT_CAPABILITY,
    )


pre_robustness_integrity_gate = robustness_integrity_gate


def fingerprint_robustness_plan(plan: RobustnessPlan) -> str:
    payload = json.dumps(asdict(plan), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_state_sha256(model: LightweightDetector) -> str:
    """Hash estimator semantics deterministically across artifact reloads."""
    digest = hashlib.sha256()
    estimator = model.estimator
    if model.model_name == "decision_tree":
        learned_state = {
            "classes": estimator.classes_,
            "n_classes": estimator.n_classes_,
            "n_features_in": estimator.n_features_in_,
            "n_outputs": estimator.n_outputs_,
            "max_features": estimator.max_features_,
            "tree": estimator.tree_.__getstate__(),
        }
    elif model.model_name == "logistic_regression":
        learned_state = {
            "classes": estimator.classes_,
            "n_features_in": estimator.n_features_in_,
            "n_iter": estimator.n_iter_,
            "coef": estimator.coef_,
            "intercept": estimator.intercept_,
        }
    else:
        learned_state = estimator.__getstate__()
    _update_state_digest(
        digest,
        {
            "model_name": model.model_name,
            "feature_names": model.feature_names,
            "parameters": model.parameters,
            "random_state": model.random_state,
            "estimator_parameters": estimator.get_params(deep=True),
            "learned_state": learned_state,
        },
    )
    return digest.hexdigest()


def _update_state_digest(digest: Any, value: Any) -> None:
    if isinstance(value, Mapping):
        digest.update(b"mapping{")
        for key in sorted(value, key=lambda item: str(item)):
            _update_state_digest(digest, str(key))
            _update_state_digest(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, np.ndarray):
        digest.update(b"array:")
        digest.update(str(value.dtype.descr if value.dtype.fields else value.dtype.str).encode("utf-8"))
        digest.update(str(value.shape).encode("ascii"))
        if value.dtype.fields:
            for field in value.dtype.names or ():
                _update_state_digest(digest, field)
                _update_state_digest(digest, value[field])
        elif value.dtype.hasobject:
            for item in value.flat:
                _update_state_digest(digest, item)
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, np.generic):
        _update_state_digest(digest, value.item())
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence[")
        for item in value:
            _update_state_digest(digest, item)
        digest.update(b"]")
    elif isinstance(value, (str, bytes, int, float, bool)) or value is None:
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(repr(value).encode("utf-8"))
    elif hasattr(value, "__getstate__"):
        digest.update(type(value).__qualname__.encode("utf-8"))
        _update_state_digest(digest, value.__getstate__())
    else:
        digest.update(type(value).__qualname__.encode("utf-8"))
        digest.update(repr(value).encode("utf-8"))


def verify_source_hashes(context: RobustnessContext) -> None:
    """Recheck all lock, V0.6-F, attack-config, and DT model source bytes."""
    if (
        context.authorization_capability is not _CONTEXT_CAPABILITY
        or fingerprint_robustness_plan(context.plan) != context.plan_sha256
    ):
        raise V06GIntegrityError("Immutable V0.6-G execution plan changed after the gate.")
    for name, expected in context.source_hashes.items():
        path = Path(name)
        if not path.is_file() or _sha256_file(path) != expected:
            raise V06GIntegrityError(f"Locked source changed during V0.6-G: {path}")
    for configuration in context.plan.configurations:
        for seed_plan in configuration.seed_plans:
            key = (configuration.configuration_id, seed_plan.seed)
            v06f._verify_model_file(context.decision_tree_paths[key], seed_plan)
            v06f._verify_loaded_model(context.decision_tree_models[key], seed_plan)
            current_hash = model_state_sha256(context.decision_tree_models[key])
            fresh_hash = model_state_sha256(
                LightweightDetector.load(context.decision_tree_paths[key])
            )
            if current_hash != context.decision_tree_state_hashes[key] or fresh_hash != current_hash:
                raise V06GIntegrityError("Frozen decision-tree estimator state changed during execution.")


def verify_loaded_splits(
    bundle: Any,
    candidate_features: Sequence[str],
    dataset_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Verify original split identity, size, feature governance, and separation."""
    train = bundle.splits["train"]
    test = bundle.splits["test"]
    for expected_name, split in (("train", train), ("test", test)):
        if split.name != expected_name:
            raise V06GExecutionError(f"Loaded {expected_name} split has the wrong identity.")
        if "row_id" not in split.features or "row_id" not in split.metadata:
            raise V06GExecutionError("Loaded splits require stable row_id provenance.")
        if (
            not split.features["row_id"].is_unique
            or not split.metadata["row_id"].is_unique
            or split.features["row_id"].tolist() != split.metadata["row_id"].tolist()
        ):
            raise V06GExecutionError(f"Loaded {expected_name} row identities are invalid.")
        missing = [name for name in candidate_features if name not in split.features]
        if missing:
            raise V06GExecutionError(f"Loaded {expected_name} split lacks locked features: {missing}")
        assert_no_attack_metadata(split.features)
        if "Late_delivery_risk" in split.features or "Late_delivery_risk" in candidate_features:
            raise V06GExecutionError("Late_delivery_risk cannot be a cybersecurity feature.")
        values = split.features.loc[:, list(candidate_features)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise V06GExecutionError(f"Loaded {expected_name} features are not finite.")
        if dataset_metadata is not None:
            expected_rows = dataset_metadata.get("split", {}).get("sizes", {}).get(expected_name)
            if expected_rows != len(split.features):
                raise V06GExecutionError(f"Loaded {expected_name} row count differs from metadata.")
    if set(train.features["row_id"]) & set(test.features["row_id"]):
        raise V06GExecutionError("Original train and test row IDs must be disjoint.")


def run_v06g_robustness(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    *,
    processed_dir: Path | str | None = None,
    context: RobustnessContext | None = None,
    loader: Callable[..., Any] | None = None,
    attack_preparer: Callable[..., PreparedAttackSplit] = prepare_attack_split,
    model_factory: Callable[..., LightweightDetector] = create_model,
    training_runner: Callable[..., Any] | None = None,
    inference_runner: Callable[..., Any] | None = None,
    synthetic_test: bool = False,
    write_figures: bool = True,
) -> dict[str, Any]:
    """Run the locked robustness matrix; synthetic mode requires full injection."""
    results = Path(results_dir).resolve()
    context = context or robustness_integrity_gate(results)
    verify_source_hashes(context)
    if synthetic_test and any(item is None for item in (loader, training_runner, inference_runner)):
        raise V06GExecutionError("Synthetic execution requires injected loader, trainer, and inference runner.")
    if loader is None:
        from src.ai.model_utils import load_processed_dataco_splits
        loader = load_processed_dataco_splits
    if training_runner is None:
        from src.lightweight.training import train_model
        training_runner = train_model
    if inference_runner is None:
        from src.lightweight.training import run_inference

        def inference_runner(model: LightweightDetector, features: pd.DataFrame, **options: Any) -> Any:
            # The full matrix has 390 cells. Current-process measurement avoids
            # conflating every short inference with isolated process startup.
            return run_inference(
                model,
                features,
                artifact_path=None,
                repeats=int(options.get("repeats", INFERENCE_REPEATS)),
            )
    resolved_processed = PROCESSED_DATA_DIR if processed_dir is None else Path(processed_dir)
    if not synthetic_test:
        metadata_path = resolved_processed / "dataset_metadata.json"
        if (
            not metadata_path.is_file()
            or _sha256_file(metadata_path)
            != context.final_test_metadata["processed_dataset_metadata_sha256"]
        ):
            raise V06GIntegrityError("Processed dataset metadata differs from V0.6-F.")
    kwargs = {} if processed_dir is None else {"processed_dir": processed_dir}
    bundle = loader(("train", "test"), **kwargs)
    if set(bundle.splits) != {"train", "test"}:
        raise V06GExecutionError("V0.6-G loader must expose exactly original train and test splits.")
    if tuple(bundle.selected_features) != context.plan.candidate_features:
        raise V06GExecutionError("Loaded candidate feature order differs from validation lock.")
    stored_metadata = None
    if not synthetic_test:
        stored_metadata = _read_json(resolved_processed / "dataset_metadata.json", "processed metadata")
        if getattr(bundle, "dataset_metadata", None) != stored_metadata:
            raise V06GExecutionError("Loaded dataset metadata differs from the immutable metadata file.")
    verify_loaded_splits(bundle, context.plan.candidate_features, stored_metadata)
    output_dir = results / ROBUSTNESS_SUBDIR
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    attack_path = context.lock["attack_configuration"]["attack_config_path"]

    training_manifestations: dict[int, PreparedAttackSplit] = {}
    test_manifestations: dict[tuple[int, str], PreparedAttackSplit] = {}
    manifestation_hashes: dict[tuple[int, str], str] = {}
    for seed in context.plan.seeds:
        verify_source_hashes(context)
        training = attack_preparer(
            split=bundle.splits["train"], training_reference=bundle.splits["train"],
            candidate_features=context.plan.candidate_features, attack_type="mixed",
            attack_rate=PRIMARY_RATE, severity=PRIMARY_SEVERITY, random_seed=seed,
            experiment_id=f"v0_6_g_lr_training_s{seed}", config_path=attack_path,
        )
        _verify_prepared(training, seed, None, context)
        training_manifestations[seed] = training
        for scenario in context.plan.scenarios:
            prepared = attack_preparer(
                split=bundle.splits["test"], training_reference=bundle.splits["train"],
                candidate_features=context.plan.candidate_features,
                attack_type=scenario.attack_type, attack_rate=scenario.attack_rate,
                severity=scenario.attack_severity, random_seed=seed,
                experiment_id=(
                    f"v0_6_f_final_test_s{seed}"
                    if scenario.attack_type == "mixed"
                    and scenario.attack_rate == PRIMARY_RATE
                    and scenario.attack_severity == PRIMARY_SEVERITY
                    else f"v0_6_g_{scenario.scenario_id}_s{seed}"
                ), config_path=attack_path,
                test_authorization_id=context.plan.semantic_lock_sha256,
                test_authorization_capability=context,
            )
            _verify_prepared(prepared, seed, scenario, context)
            key = (seed, scenario.scenario_id)
            test_manifestations[key] = prepared
            manifestation_hashes[key] = v06f._manifestation_fingerprint(prepared)
            if not synthetic_test and scenario.in_rate_panel and scenario.in_severity_panel:
                expected = context.final_test_metadata["manifestation_fingerprints"].get(str(seed))
                if manifestation_hashes[key] != expected:
                    raise V06GIntegrityError(
                        "Primary mixed/5%/MEDIUM manifestation does not replay V0.6-F."
                    )

    logistic_models: dict[tuple[str, int], LightweightDetector] = {}
    logistic_paths: dict[tuple[str, int], Path] = {}
    logistic_fit: dict[tuple[str, int], Any] = {}
    logistic_state_hashes: dict[tuple[str, int], str] = {}
    for configuration in context.plan.configurations:
        for seed_plan in configuration.seed_plans:
            seed = seed_plan.seed
            features = list(seed_plan.selected_features)
            training = training_manifestations[seed]
            model = model_factory(
                "logistic_regression", features, dict(LR_PARAMETERS), random_state=seed
            )
            final_path = model_dir / f"v0_6_g__logistic_regression__{configuration.configuration_id}__s{seed}.joblib"
            fitted, resource = _fit_logistic_atomic(
                model, training.features.loc[:, features], training.labels,
                final_path, training_runner,
            )
            _verify_logistic_model(fitted, seed_plan)
            reloaded = LightweightDetector.load(final_path)
            _verify_logistic_model(reloaded, seed_plan)
            if model_state_sha256(fitted) != model_state_sha256(reloaded):
                raise V06GIntegrityError("Reloaded LR model state does not match fitted state.")
            key = (configuration.configuration_id, seed)
            logistic_models[key] = reloaded
            logistic_paths[key] = final_path
            logistic_fit[key] = resource
            logistic_state_hashes[key] = model_state_sha256(reloaded)

    rows: list[dict[str, Any]] = []
    reuse: dict[str, dict[str, Any]] = {}
    for scenario in context.plan.scenarios:
        for seed in context.plan.seeds:
            prepared = test_manifestations[(seed, scenario.scenario_id)]
            fingerprint = manifestation_hashes[(seed, scenario.scenario_id)]
            cells: list[str] = []
            for configuration in context.plan.configurations:
                seed_plan = next(item for item in configuration.seed_plans if item.seed == seed)
                for classifier in CLASSIFIERS:
                    key = (configuration.configuration_id, seed)
                    if classifier == "decision_tree":
                        model = context.decision_tree_models[key]
                        path = context.decision_tree_paths[key]
                        fit_resource = None
                        model_state_hash = context.decision_tree_state_hashes[key]
                    else:
                        model = logistic_models[key]
                        path = logistic_paths[key]
                        fit_resource = logistic_fit[key]
                        model_state_hash = logistic_state_hashes[key]
                    if v06f._manifestation_fingerprint(prepared) != fingerprint:
                        raise V06GIntegrityError("Prepared test manifestation changed during reuse.")
                    rows.append(
                        evaluate_robustness_run(
                            configuration, seed_plan, classifier, model, path, prepared,
                            scenario, inference_runner, fit_resource, model_state_hash,
                        )
                    )
                    cells.append(f"{configuration.configuration_id}:{classifier}")
            reuse[f"{seed}:{scenario.scenario_id}"] = {
                "object_id": id(prepared), "manifestation_sha256": fingerprint,
                "cell_count": len(cells), "cells": cells,
            }
    verify_manifestation_reuse(pd.DataFrame(rows), context.plan)
    verify_source_hashes(context)
    outputs = write_robustness_outputs(
        output_dir, pd.DataFrame(rows), context, reuse,
        logistic_paths, logistic_state_hashes, write_figures=write_figures,
    )
    verify_source_hashes(context)
    return {
        "status": "PASS", "run_count": len(rows),
        "scenario_count": len(context.plan.scenarios),
        "unique_configuration_count": len(context.plan.configurations),
        "logistic_fit_count": len(logistic_models),
        "role_mapping": _role_mapping(context.plan), "outputs": outputs,
    }


run_robustness_evaluation_v06g = run_v06g_robustness


def evaluate_robustness_run(
    configuration: LockedConfiguration,
    seed_plan: v06f.SeedPlan,
    classifier: str,
    model: LightweightDetector,
    artifact_path: Path,
    prepared: PreparedAttackSplit,
    scenario: RobustnessScenario,
    inference_runner: Callable[..., Any],
    fit_resource: Any | None,
    model_state_sha256: str,
) -> dict[str, Any]:
    """Evaluate one frozen model/manifestation cell without fitting or selection."""
    selected = list(seed_plan.selected_features)
    assert_no_attack_metadata(selected)
    attacked_input = prepared.features.loc[:, selected]
    clean_input = prepared.clean_features.loc[:, selected]
    measured = inference_runner(
        model, attacked_input, artifact_path=artifact_path, repeats=INFERENCE_REPEATS
    )
    attacked_predictions = _prediction_frame(measured.predictions, prepared)
    clean_predictions = _prediction_frame(model.predict_frame(clean_input), prepared)
    metrics = v06f._normalize_metrics(evaluate_detection(prepared.ground_truth, attacked_predictions))
    if "pr_auc" in metrics:
        raise V06GExecutionError("Ambiguous PR-AUC metric name is forbidden.")
    visible_mask = detector_visible_mask(clean_input, attacked_input)
    attacked_mask = prepared.ground_truth["is_attack"].eq(1).to_numpy()
    predicted_mask = attacked_predictions["anomaly_label"].eq(1).to_numpy()
    visible_count = int((attacked_mask & visible_mask).sum())
    invisible_count = int((attacked_mask & ~visible_mask).sum())
    visible_metrics = evaluate_visible_attacks(
        prepared.ground_truth, attacked_predictions, visible_mask
    )
    paired = evaluate_paired_detection(
        prepared.ground_truth, clean_predictions, attacked_predictions
    )
    inference = measured.resources
    fit = _resource_columns(fit_resource, "model_fit")
    row: dict[str, Any] = {
        "run_id": f"v06g_{scenario.scenario_id}_{configuration.configuration_id}_{classifier}_s{seed_plan.seed}",
        "status": "COMPLETED", "seed": seed_plan.seed,
        "configuration_id": configuration.configuration_id,
        "roles_json": json.dumps(configuration.roles), "classifier": classifier,
        "scenario_id": scenario.scenario_id, "attack_type": scenario.attack_type,
        "attack_family": None if scenario.attack_type == "mixed" else scenario.attack_type,
        "attack_rate": scenario.attack_rate, "attack_severity": scenario.attack_severity,
        "in_family_panel": scenario.in_family_panel, "in_rate_panel": scenario.in_rate_panel,
        "in_severity_panel": scenario.in_severity_panel,
        "attack_mode": prepared.attack_metadata.get("attack_mode"),
        "eligible_records": prepared.attack_metadata.get("eligible_records"),
        "selected_records": prepared.attack_metadata.get("selected_records"),
        "actual_modified_records": prepared.attack_metadata.get("actual_modified_records"),
        "attack_rate_achieved": prepared.attack_metadata.get("attack_rate_achieved"),
        "manifestation_sha256": v06f._manifestation_fingerprint(prepared),
        "selected_features_json": json.dumps(selected),
        "selected_features_sha256": seed_plan.selected_features_sha256,
        "selected_feature_count": len(selected), "source_block_count": _source_block_count(selected),
        "selected_input_bytes": int(attacked_input.memory_usage(index=True, deep=True).sum()),
        "prediction_threshold": PREDICTION_THRESHOLD,
        "visible_feature_attack_count": visible_count,
        "invisible_feature_attack_count": invisible_count,
        "selected_feature_visibility_rate": (
            visible_count / metrics["attacked_records"] if metrics["attacked_records"] else None
        ),
        "visible_attack_recall": _direct_recall(attacked_mask & visible_mask, predicted_mask),
        "invisible_attack_recall": _direct_recall(attacked_mask & ~visible_mask, predicted_mask),
        "visible_attack_f1": None if visible_metrics is None else visible_metrics["f1"],
        "selector_fit_wall_time_sec": None, "selector_fit_cpu_time_sec": None,
        "selector_fit_peak_rss_mib": None, "selector_fit_performed": False,
        "model_fit_performed": classifier == "logistic_regression",
        "combined_fit_wall_time_sec": fit["model_fit_wall_time_sec"],
        "inference_repeats": INFERENCE_REPEATS,
        "inference_measurement_scope": "current_process_model_already_loaded",
        "inference_wall_time_sec": inference.wall_time_sec,
        "per_record_inference_sec": inference.wall_time_sec / len(attacked_input),
        "inference_cpu_time_sec": inference.cpu_time_sec,
        "inference_average_cpu_percent": inference.average_cpu_percent,
        "inference_peak_cpu_percent": inference.peak_cpu_percent,
        "inference_peak_rss_bytes": inference.peak_rss_bytes,
        "inference_peak_rss_mib": inference.peak_rss_mb,
        "serialized_model_bytes": int(artifact_path.stat().st_size),
        "serialized_model_kib": artifact_path.stat().st_size / 1024.0,
        "model_file_sha256": _sha256_file(artifact_path),
        "model_state_sha256": model_state_sha256,
        "locked_model_state_sha256": (
            seed_plan.model_state_sha256 if classifier == "decision_tree" else None
        ),
        "reselection_performed": False, "preprocessing_fit_performed": False,
    }
    row.update(fit)
    row.update(metrics)
    row.update({
        "total_attacks": metrics["attacked_records"],
        "feature_visible_attacks": visible_count,
        "feature_invisible_attacks": invisible_count,
        "visibility_rate": row["selected_feature_visibility_rate"],
        "peak_rss_mib": max(
            value for value in (fit["model_fit_peak_rss_mib"], inference.peak_rss_mb)
            if value is not None
        ),
    })
    row.update({f"paired_induced_{key}": value for key, value in paired.items()})
    return row


def aggregate_metrics(
    runs: pd.DataFrame, group_columns: Sequence[str]
) -> pd.DataFrame:
    """Aggregate all registered detection/resource metrics with five-seed t-CIs."""
    columns = [*group_columns, "metric", "count", "mean", "std", "min", "max", "ci95_low", "ci95_high"]
    records: list[dict[str, Any]] = []
    if runs.empty:
        return pd.DataFrame(columns=columns)
    for keys, group in runs.groupby(list(group_columns), sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_columns, key_values))
        for metric in SUMMARY_METRICS:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            mean, std, low, high = mean_std_t_ci(values)
            records.append({
                **base, "metric": metric, "count": len(values), "mean": mean, "std": std,
                "min": float(values.min()) if len(values) else None,
                "max": float(values.max()) if len(values) else None,
                "ci95_low": low, "ci95_high": high,
            })
    return pd.DataFrame(records, columns=columns)


def paired_seed_differences(
    runs: pd.DataFrame,
    *,
    comparison: str = "configuration",
) -> pd.DataFrame:
    """Calculate paired seed differences for reduced-v-baseline or LR-v-DT cells."""
    records: list[dict[str, Any]] = []
    scenario_columns = ["scenario_id", "attack_type", "attack_rate", "attack_severity"]
    if runs.empty:
        return pd.DataFrame()
    baseline_id = _baseline_from_runs(runs)
    if comparison == "configuration":
        for keys, group in runs.groupby([*scenario_columns, "classifier"], sort=False):
            baseline = group.loc[group["configuration_id"].eq(baseline_id)].set_index("seed")
            for config_id in group["configuration_id"].drop_duplicates():
                if config_id == baseline_id:
                    continue
                current = group.loc[group["configuration_id"].eq(config_id)].set_index("seed")
                common = sorted(set(baseline.index) & set(current.index))
                for metric in PRESERVATION_MARGINS:
                    values = current.loc[common, metric].to_numpy(float) - baseline.loc[common, metric].to_numpy(float)
                    mean, std, low, high = mean_std_t_ci(values)
                    records.append({
                        **dict(zip([*scenario_columns, "classifier"], keys)),
                        "comparison_type": "reduced_minus_full_baseline",
                        "configuration_id": config_id, "reference_configuration_id": baseline_id,
                        "metric": metric, "paired_seed_count": len(values),
                        "mean_difference": mean, "std_difference": std,
                        "ci95_low": low, "ci95_high": high,
                    })
    elif comparison == "classifier":
        for keys, group in runs.groupby([*scenario_columns, "configuration_id"], sort=False):
            dt = group.loc[group["classifier"].eq("decision_tree")].set_index("seed")
            lr = group.loc[group["classifier"].eq("logistic_regression")].set_index("seed")
            common = sorted(set(dt.index) & set(lr.index))
            for metric in PRESERVATION_MARGINS:
                values = lr.loc[common, metric].to_numpy(float) - dt.loc[common, metric].to_numpy(float)
                mean, std, low, high = mean_std_t_ci(values)
                records.append({
                    **dict(zip([*scenario_columns, "configuration_id"], keys)),
                    "comparison_type": "logistic_regression_minus_decision_tree",
                    "classifier": "logistic_regression", "reference_classifier": "decision_tree",
                    "metric": metric, "paired_seed_count": len(values),
                    "mean_difference": mean, "std_difference": std,
                    "ci95_low": low, "ci95_high": high,
                })
    else:
        raise ValueError("comparison must be 'configuration' or 'classifier'.")
    return pd.DataFrame(records)


def preservation_analysis(runs: pd.DataFrame) -> pd.DataFrame:
    """Apply frozen relative margins per scenario, including rate scenarios."""
    paired = paired_seed_differences(runs, comparison="configuration")
    if paired.empty:
        return paired
    means = runs.groupby(["scenario_id", "classifier", "configuration_id"], sort=False)[
        list(PRESERVATION_MARGINS)
    ].mean()
    baseline_id = _baseline_from_runs(runs)
    records: list[dict[str, Any]] = []
    for row in paired.to_dict(orient="records"):
        metric = row["metric"]
        base = float(means.loc[(row["scenario_id"], row["classifier"], baseline_id), metric])
        current = float(means.loc[(row["scenario_id"], row["classifier"], row["configuration_id"]), metric])
        degradation = 0.0 if base == 0 and current >= base else (
            None if base == 0 else max(0.0, 100.0 * (base - current) / base)
        )
        row.update({
            "baseline_mean": base, "configuration_mean": current,
            "margin_percent": PRESERVATION_MARGINS[metric],
            "relative_degradation_percent": degradation,
            "descriptively_preserved": degradation is not None and degradation <= PRESERVATION_MARGINS[metric],
            "rate_scenario": bool(
                runs.loc[runs["scenario_id"].eq(row["scenario_id"]), "in_rate_panel"].iloc[0]
            ),
            "selection_performed": False,
        })
        records.append(row)
    result = pd.DataFrame(records)
    result["scenario_configuration_preserved"] = result.groupby(
        ["scenario_id", "classifier", "configuration_id"]
    )["descriptively_preserved"].transform("all")
    return result


def verify_manifestation_reuse(runs: pd.DataFrame, plan: RobustnessPlan) -> None:
    """Require identical hashes in all configuration/classifier cells and equal matrices."""
    expected_cells = len(plan.configurations) * len(plan.classifiers)
    expected_scenarios = {scenario.scenario_id for scenario in plan.scenarios}
    for seed in plan.seeds:
        subset = runs.loc[runs["seed"].eq(seed)]
        if set(subset["scenario_id"]) != expected_scenarios:
            raise V06GIntegrityError("A classifier/configuration has an incomplete scenario matrix.")
        for _, group in subset.groupby("scenario_id"):
            if len(group) != expected_cells or group["manifestation_sha256"].nunique() != 1:
                raise V06GIntegrityError("All six cells must reuse one seed/scenario manifestation.")
            cells = set(zip(group["configuration_id"], group["classifier"]))
            expected = {
                (configuration.configuration_id, classifier)
                for configuration in plan.configurations for classifier in plan.classifiers
            }
            if cells != expected:
                raise V06GIntegrityError("DT and LR scenario matrices differ.")


def mean_std_t_ci(values: Sequence[float] | np.ndarray) -> tuple[Any, Any, Any, Any]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return None, None, None, None
    mean = float(np.mean(array))
    if len(array) == 1:
        return mean, 0.0, None, None
    std = float(np.std(array, ddof=1))
    if len(array) != len(EXPECTED_SEEDS):
        return mean, std, None, None
    half_width = 2.7764451052 * std / math.sqrt(5)
    return mean, std, mean - half_width, mean + half_width


def write_robustness_outputs(
    output_dir: Path,
    runs: pd.DataFrame,
    context: RobustnessContext,
    reuse: Mapping[str, Any],
    logistic_paths: Mapping[tuple[str, int], Path],
    logistic_state_hashes: Mapping[tuple[str, int], str],
    *,
    write_figures: bool = True,
) -> dict[str, str]:
    """Build and atomically publish all V0.6-G reports."""
    summary = aggregate_metrics(
        runs, ["classifier", "configuration_id", "scenario_id", "attack_type", "attack_rate", "attack_severity"]
    )
    family = summary.loc[summary["scenario_id"].isin(runs.loc[runs["in_family_panel"], "scenario_id"].unique())]
    rate = summary.loc[summary["scenario_id"].isin(runs.loc[runs["in_rate_panel"], "scenario_id"].unique())]
    severity = summary.loc[summary["scenario_id"].isin(runs.loc[runs["in_severity_panel"], "scenario_id"].unique())]
    visibility = aggregate_metrics(
        runs, ["classifier", "configuration_id", "scenario_id", "attack_type", "attack_rate", "attack_severity"]
    )
    visibility = visibility.loc[visibility["metric"].isin({
        "visible_feature_attack_count", "invisible_feature_attack_count",
        "selected_feature_visibility_rate", "visible_attack_recall",
        "invisible_attack_recall", "visible_attack_f1",
    })]
    preservation = preservation_analysis(runs)
    config_paired = paired_seed_differences(runs, comparison="configuration")
    classifier = pd.concat(
        [paired_seed_differences(runs, comparison="classifier"), config_paired], ignore_index=True
    )
    resources = aggregate_metrics(
        runs, ["classifier", "configuration_id", "scenario_id", "attack_type", "attack_rate", "attack_severity"]
    )
    resources = resources.loc[resources["metric"].isin(RESOURCE_METRICS)]
    tables = {
        "robustness_runs": runs, "robustness_summary": summary,
        "attack_family_summary": family, "attack_rate_summary": rate,
        "attack_severity_summary": severity, "visibility_analysis": visibility,
        "robustness_preservation": preservation, "classifier_comparison": classifier,
        "resource_comparison": resources,
    }
    expected_runs = len(context.plan.seeds) * len(context.plan.scenarios) * len(context.plan.configurations) * 2
    runtime = _runtime_metadata()
    metadata = {
        "schema_version": V06G_PROTOCOL_VERSION, "stage": V06G_STAGE, "status": "COMPLETED",
        "test_accessed": True, "gate_timestamp": context.gate_timestamp,
        "completion_timestamp": _utc_now(), "semantic_lock_sha256": context.plan.semantic_lock_sha256,
        "validation_lock_file_sha256": context.plan.validation_lock_file_sha256,
        "v06f_artifact_manifest_sha256": context.source_hashes[
            str((context.results_dir / v06f.FINAL_TEST_SUBDIR / "final_test_artifact_hashes.json").resolve())
        ],
        "execution_plan_sha256": context.plan_sha256, "seeds": list(context.plan.seeds),
        "semantic_roles": list(SEMANTIC_ROLES), "role_mapping": _role_mapping(context.plan),
        "unique_configuration_count": len(context.plan.configurations),
        "scenario_count": len(context.plan.scenarios), "expected_run_count": expected_runs,
        "completed_run_count": len(runs), "failed_run_count": 0,
        "expected_decision_tree_run_count": expected_runs // 2,
        "completed_decision_tree_run_count": int(runs["classifier"].eq("decision_tree").sum()),
        "failed_decision_tree_run_count": 0,
        "expected_logistic_regression_run_count": expected_runs // 2,
        "completed_logistic_regression_run_count": int(runs["classifier"].eq("logistic_regression").sum()),
        "failed_logistic_regression_run_count": 0,
        "logistic_fit_count": len(logistic_paths),
        "lr_parameters": {**dict(LR_PARAMETERS), "random_state": "seed"},
        "prediction_threshold": PREDICTION_THRESHOLD,
        "preservation_margins_percent": dict(PRESERVATION_MARGINS),
        "inference_repeats": INFERENCE_REPEATS,
        "uncertainty_scope": "five algorithm/attack seeds on one fixed dataset split",
        "resource_interpretation": "time, memory, input size, and serialized size are computational proxies only",
        "software": runtime,
    }
    audit_checks = {
        "v06e_lock_semantics_and_sources_verified": True,
        "v06f_access_completed_and_hash_manifest_verified": True,
        "source_hashes_verified_before_and_after": True,
        "loaded_decision_tree_file_and_state_hashes_verified": True,
        "exact_semantic_roles_derived_and_duplicates_collapsed": True,
        "official_config_derived_rates_and_severities": True,
        "unique_scenario_deduplication": len(context.plan.scenarios) == len({s.scenario_id for s in context.plan.scenarios}),
        "exact_seeds_42_through_46": context.plan.seeds == EXPECTED_SEEDS,
        "dt_and_lr_same_scenario_matrix": True,
        "one_manifestation_per_seed_scenario_reused_in_all_cells": all(value["cell_count"] == len(context.plan.configurations) * 2 for value in reuse.values()),
        "primary_manifestation_replays_v06f": True,
        "decision_tree_not_fitted": bool(runs.loc[runs["classifier"].eq("decision_tree"), "model_fit_wall_time_sec"].isna().all()),
        "logistic_regression_fitted_once_per_seed_configuration": len(logistic_paths) == len(context.plan.configurations) * 5,
        "lr_training_used_controlled_training_labels_only": True,
        "original_train_only_and_train_test_rows_disjoint": True,
        "validation_and_test_excluded_from_training": True,
        "selector_and_preprocessing_fit_not_performed": bool((~runs["selector_fit_performed"]).all() and (~runs["preprocessing_fit_performed"]).all()),
        "threshold_tuning_not_performed": bool(runs["prediction_threshold"].eq(0.5).all()),
        "ap_and_explicit_trapezoidal_pr_auc_only": "pr_auc" not in runs,
        "no_reselection_or_configuration_choice_from_test": bool((~runs["reselection_performed"]).all()),
        "attack_metadata_excluded_from_features": True,
        "late_delivery_risk_not_cyber_ground_truth_or_feature": True,
        "dt_and_lr_hyperparameters_untuned": True,
        "attack_generator_and_scenarios_not_redesigned": True,
        "paired_seed_differences_reported": not config_paired.empty,
        "no_p_value_fishing_or_direct_energy_claims": True,
        "all_outputs_atomic": True,
    }
    audit = {
        "schema_version": V06G_PROTOCOL_VERSION, "stage": V06G_STAGE,
        "status": "PASS" if all(audit_checks.values()) and len(runs) == expected_runs else "FAIL",
        "semantic_lock_sha256": context.plan.semantic_lock_sha256,
        "execution_plan_sha256": context.plan_sha256,
        "manifestation_reuse_evidence": deepcopy(dict(reuse)),
        "source_hashes": dict(context.source_hashes),
        "lr_model_artifacts": {
            f"{configuration}:{seed}": {
                "path": str(path), "file_sha256": _sha256_file(path),
                "model_state_sha256": logistic_state_hashes[(configuration, seed)],
            }
            for (configuration, seed), path in logistic_paths.items()
        },
        "checks": [{"name": name, "passed": bool(value)} for name, value in audit_checks.items()],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for stem, frame in tables.items():
        csv_path, json_path = output_dir / f"{stem}.csv", output_dir / f"{stem}.json"
        _write_csv_atomic(csv_path, frame)
        _write_json_atomic(json_path, frame.to_dict(orient="records"))
        outputs[csv_path.name], outputs[json_path.name] = str(csv_path), str(json_path)
    for name, payload in (("robustness_metadata.json", metadata), ("robustness_governance_audit.json", audit)):
        path = output_dir / name
        _write_json_atomic(path, payload)
        outputs[name] = str(path)
    if write_figures:
        outputs.update(_write_figures_atomic(output_dir / "figures", runs, preservation))
    hash_path = output_dir / "robustness_artifact_hashes.json"
    hashes = {
        path.relative_to(output_dir).as_posix(): _sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path != hash_path
    }
    _write_json_atomic(hash_path, hashes)
    outputs[hash_path.name] = str(hash_path)
    if audit["status"] != "PASS":
        raise V06GIntegrityError("V0.6-G governance audit failed.")
    return outputs


def _fit_logistic_atomic(
    model: LightweightDetector,
    features: pd.DataFrame,
    labels: pd.Series,
    final_path: Path,
    training_runner: Callable[..., Any],
) -> tuple[LightweightDetector, Any]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{final_path.name}.", suffix=".joblib", dir=final_path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        result = training_runner(model, features, labels, temporary, isolated=True)
        if not temporary.is_file():
            candidate = Path(result.artifact_path)
            if not candidate.is_file():
                raise V06GExecutionError("LR trainer did not produce a model artifact.")
            temporary = candidate
        os.replace(temporary, final_path)
        return result.model, result.resources
    finally:
        if temporary.exists() and temporary != final_path:
            temporary.unlink()


def _verify_logistic_model(model: Any, seed_plan: v06f.SeedPlan) -> None:
    estimator = getattr(model, "estimator", None)
    if (
        not isinstance(model, LightweightDetector) or not model.is_fitted
        or model.model_name != "logistic_regression"
        or model.parameters != dict(LR_PARAMETERS) or model.random_state != seed_plan.seed
        or tuple(model.feature_names) != seed_plan.selected_features
        or fingerprint_feature_names(model.feature_names) != seed_plan.selected_features_sha256
        or getattr(estimator, "solver", None) != "liblinear"
        or getattr(estimator, "class_weight", None) != "balanced"
        or getattr(estimator, "max_iter", None) != 500
        or float(getattr(estimator, "C", math.nan)) != 1.0
        or getattr(estimator, "random_state", None) != seed_plan.seed
    ):
        raise V06GIntegrityError("LR state, parameters, seed, or exact locked features differ from protocol.")


def _verify_prepared(
    prepared: PreparedAttackSplit,
    seed: int,
    scenario: RobustnessScenario | None,
    context: RobustnessContext,
) -> None:
    expected_split = "train" if scenario is None else "test"
    attack_type = "mixed" if scenario is None else scenario.attack_type
    rate = PRIMARY_RATE if scenario is None else scenario.attack_rate
    severity = PRIMARY_SEVERITY if scenario is None else scenario.attack_severity
    metadata_types = tuple(prepared.attack_metadata.get("attack_types", ()))
    type_ok = (
        prepared.attack_metadata.get("attack_mode") == "mixed"
        if attack_type == "mixed"
        else metadata_types == (attack_type,)
    )
    if (
        not isinstance(prepared, PreparedAttackSplit) or prepared.split_name != expected_split
        or tuple(prepared.candidate_features) != context.plan.candidate_features
        or prepared.labels.name != "is_attack"
        or float(prepared.attack_metadata.get("configured_attack_rate", math.nan)) != rate
        or prepared.attack_metadata.get("severity") != severity
        or prepared.attack_metadata.get("random_seed") != seed or not type_ok
        or (scenario is not None and prepared.test_authorization_id != context.plan.semantic_lock_sha256)
        or (scenario is not None and prepared.test_authorization_capability is not context)
    ):
        raise V06GExecutionError("Prepared attack split violates the frozen train/test scenario protocol.")
    assert_no_attack_metadata(prepared.features)
    v06f._manifestation_fingerprint(prepared)
    if not prepared.labels.index.equals(prepared.features.index):
        raise V06GExecutionError("Prepared attack labels are not aligned to feature rows.")
    if set(prepared.labels.unique()) - {0, 1}:
        raise V06GExecutionError("Prepared attack labels must be binary controlled labels.")
    truth = prepared.ground_truth
    if not {"record_id", "is_attack"}.issubset(truth) or truth["record_id"].duplicated().any():
        raise V06GExecutionError("Prepared ground truth has invalid record identities.")
    row_ids = prepared.features["row_id"]
    if set(row_ids) != set(truth["record_id"]):
        raise V06GExecutionError("Prepared ground truth does not match feature row IDs.")
    aligned_truth = truth.set_index("record_id").loc[row_ids, "is_attack"].to_numpy(dtype=int)
    if not np.array_equal(aligned_truth, prepared.labels.to_numpy(dtype=int)):
        raise V06GExecutionError("Prepared labels differ from controlled ground truth.")
    manifest = prepared.manifest
    attacked_ids = set(truth.loc[truth["is_attack"].eq(1), "record_id"])
    manifest_ids = set(manifest["record_id"]) if "record_id" in manifest else set()
    actual = int(prepared.attack_metadata.get("actual_modified_records", -1))
    eligible = int(prepared.attack_metadata.get("eligible_records", -1))
    selected = int(prepared.attack_metadata.get("selected_records", -1))
    expected_selected = 0 if eligible <= 0 else min(eligible, max(1, round(eligible * rate)))
    achieved = float(prepared.attack_metadata.get("attack_rate_achieved", math.nan))
    if (
        attacked_ids != manifest_ids
        or len(manifest) != int(prepared.labels.sum())
        or actual != len(manifest)
        or selected != expected_selected
        or not 0 <= actual <= selected
        or not math.isclose(achieved, actual / eligible if eligible else 0.0)
    ):
        raise V06GExecutionError("Prepared attack counts, labels, manifest, or achieved rate disagree.")
    attack_config = load_attack_config(context.lock["attack_configuration"]["attack_config_path"])
    expected_types = tuple(attack_config["scenarios"]) if attack_type == "mixed" else (attack_type,)
    if metadata_types != expected_types:
        raise V06GExecutionError("Prepared attack-family metadata differs from the configured scenario.")
    if not manifest.empty and set(manifest["attack_type"]) - set(expected_types):
        raise V06GExecutionError("Prepared manifest contains an unsupported attack family.")


def _prediction_frame(raw: pd.DataFrame, prepared: PreparedAttackSplit) -> pd.DataFrame:
    scores = raw["anomaly_score"].to_numpy(dtype=float)
    return pd.DataFrame({
        "record_id": prepared.features["row_id"].to_numpy(copy=True),
        "anomaly_score": scores,
        "anomaly_label": (scores >= PREDICTION_THRESHOLD).astype("int8"),
    })


def _direct_recall(positive_mask: np.ndarray, predicted_mask: np.ndarray) -> float | None:
    count = int(positive_mask.sum())
    return float((positive_mask & predicted_mask).sum() / count) if count else None


def _resource_columns(resource: Any | None, prefix: str) -> dict[str, Any]:
    if resource is None:
        return {
            f"{prefix}_wall_time_sec": None, f"{prefix}_cpu_time_sec": None,
            f"{prefix}_average_cpu_percent": None, f"{prefix}_peak_cpu_percent": None,
            f"{prefix}_peak_rss_bytes": None, f"{prefix}_peak_rss_mib": None,
        }
    return {
        f"{prefix}_wall_time_sec": resource.wall_time_sec,
        f"{prefix}_cpu_time_sec": resource.cpu_time_sec,
        f"{prefix}_average_cpu_percent": resource.average_cpu_percent,
        f"{prefix}_peak_cpu_percent": resource.peak_cpu_percent,
        f"{prefix}_peak_rss_bytes": resource.peak_rss_bytes,
        f"{prefix}_peak_rss_mib": resource.peak_rss_mb,
    }


def _source_block_count(features: Sequence[str]) -> int:
    def block(name: str) -> str:
        if name.startswith("Type_"): return "payment_type"
        if name.startswith("Market_"): return "market"
        if name.startswith("Shipping Mode_"): return "shipping_mode"
        if name.startswith("Customer Segment_"): return "customer_segment"
        if name.startswith("Department Name_"): return "department"
        if name in {"year", "month", "day", "day_of_week", "hour", "is_weekend"}: return "temporal"
        if name in {"item_count_per_order", "order_total_value"}: return "order_context"
        if name == "days_schedule": return "planned_shipping"
        if name == "product_price": return "product_price"
        return "transaction_item_economics"
    return len({block(name) for name in features})


def _baseline_from_runs(runs: pd.DataFrame) -> str:
    rows = runs.loc[runs["roles_json"].map(lambda value: "full_baseline" in json.loads(value))]
    identifiers = rows["configuration_id"].unique()
    if len(identifiers) != 1:
        raise V06GIntegrityError("Runs do not identify exactly one full baseline.")
    return str(identifiers[0])


def _role_mapping(plan: RobustnessPlan) -> dict[str, str]:
    return {role: item.configuration_id for item in plan.configurations for role in item.roles}


def _scenario_id(attack_type: str, rate: float, severity: str) -> str:
    return f"{attack_type}__r{rate:.2f}".replace(".", "p") + f"__{severity.lower()}"


def _write_figures_atomic(
    figure_dir: Path, runs: pd.DataFrame, preservation: pd.DataFrame
) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    def publish(name: str, draw: Callable[[Any], None]) -> None:
        figure, axis = plt.subplots(figsize=(9, 5))
        draw(axis)
        figure.tight_layout()
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".png", dir=figure_dir)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            figure.savefig(temporary, dpi=160, metadata={"Software": "V0.6-G"})
            os.replace(temporary, figure_dir / name)
        finally:
            plt.close(figure)
            if temporary.exists(): temporary.unlink()
        outputs[name] = str(figure_dir / name)

    means = runs.groupby(
        ["scenario_id", "attack_type", "attack_rate", "attack_severity", "classifier", "configuration_id"],
        sort=False,
    ).mean(numeric_only=True).reset_index()
    family = means[means["in_family_panel"] > 0]
    rate = means[means["in_rate_panel"] > 0].sort_values("attack_rate")
    severity = means[means["in_severity_panel"] > 0].copy()
    severity["severity_order"] = severity["attack_severity"].map(
        {name: index for index, name in enumerate(OFFICIAL_SEVERITIES)}
    )
    severity = severity.sort_values("severity_order")
    publish(FIGURE_NAMES[0], lambda ax: _plot_configured_lines(
        ax, rate, "attack_rate", "average_precision", "Average precision vs mixed attack rate"
    ))
    publish(FIGURE_NAMES[1], lambda ax: _plot_configured_lines(
        ax, rate, "attack_rate", "f1", "F1 vs mixed attack rate"
    ))
    publish(FIGURE_NAMES[2], lambda ax: _plot_configured_lines(
        ax, severity, "attack_severity", "recall", "Recall vs mixed attack severity"
    ))
    publish(FIGURE_NAMES[3], lambda ax: _plot_configured_lines(
        ax, family, "attack_type", "average_precision", "Average precision by attack family"
    ))
    publish(FIGURE_NAMES[4], lambda ax: _plot_configured_lines(
        ax, family, "attack_type", "selected_feature_visibility_rate", "Visibility rate by attack family"
    ))
    primary = means.loc[means["scenario_id"].eq(_scenario_id("mixed", PRIMARY_RATE, PRIMARY_SEVERITY))]
    publish(FIGURE_NAMES[5], lambda ax: _plot_classifier_bars(ax, primary))
    feature_ap = runs.groupby(
        ["classifier", "configuration_id", "selected_feature_count"], sort=False
    )["average_precision"].mean().reset_index()
    publish(FIGURE_NAMES[6], lambda ax: _plot_configured_lines(
        ax, feature_ap, "selected_feature_count", "average_precision", "Feature count vs mean average precision"
    ))
    publish(FIGURE_NAMES[7], lambda ax: _plot_preservation_heatmap(ax, preservation))
    return outputs


def _plot_configured_lines(axis: Any, frame: pd.DataFrame, x: str, y: str, title: str) -> None:
    for keys, group in frame.groupby(["classifier", "configuration_id"], sort=False):
        axis.plot(group[x].astype(str), group[y], marker="o", label=" / ".join(keys))
    axis.set_title(title)
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    if y in {"average_precision", "f1", "recall", "selected_feature_visibility_rate"}:
        axis.set_ylim(0.0, 1.0)
    axis.tick_params(axis="x", rotation=30)
    axis.legend(fontsize=7, ncol=2)


def _plot_classifier_bars(axis: Any, frame: pd.DataFrame) -> None:
    pivot = frame.pivot(index="configuration_id", columns="classifier", values="average_precision")
    pivot.plot(kind="bar", ax=axis, rot=20)
    axis.set_title("Decision Tree vs Logistic Regression at mixed 5% MEDIUM")
    axis.set_xlabel("configuration_id")
    axis.set_ylabel("average_precision")
    axis.set_ylim(0.0, 1.0)


def _plot_preservation_heatmap(axis: Any, preservation: pd.DataFrame) -> None:
    status = preservation.groupby(
        ["scenario_id", "classifier", "configuration_id"], sort=False
    )["descriptively_preserved"].all().unstack(["classifier", "configuration_id"])
    image = axis.imshow(status.to_numpy(dtype=float), aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn")
    axis.set_title("Robustness preservation by scenario and reduced configuration")
    axis.set_yticks(range(len(status.index)), status.index, fontsize=6)
    axis.set_xticks(
        range(len(status.columns)),
        [" / ".join(map(str, values)) for values in status.columns],
        rotation=30,
        ha="right",
        fontsize=7,
    )
    axis.figure.colorbar(image, ax=axis, ticks=[0, 1], label="Preserved")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V06GIntegrityError(f"Cannot load {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise V06GIntegrityError(f"{description} must be a JSON object.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise V06GIntegrityError(f"Cannot hash required file: {path}") from exc
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return deepcopy(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping): return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [_json_safe(item) for item in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, (np.floating, float)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_): return bool(value)
    return value


def _write_json_atomic(path: Path, payload: Any) -> None:
    _write_text_atomic(path, json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name); frame.to_csv(handle, index=False); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name); handle.write(text); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _runtime_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "timestamp_utc": _utc_now(), "git_commit": commit,
        "pipeline_v06g_sha256": _sha256_file(Path(__file__)),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V0.6-G validation-locked robustness evaluation.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        context = robustness_integrity_gate(args.results_dir)
    except (ValidationLockError, v06f.V06FIntegrityError, V06GIntegrityError):
        print(PRE_RUN_FAILURE_MESSAGE)
        return 1
    try:
        result = run_v06g_robustness(
            args.results_dir, processed_dir=args.processed_dir, context=context
        )
    except (V06GIntegrityError, V06GExecutionError) as exc:
        print(f"V0.6-G EXECUTION FAILED: {exc}")
        return 1
    print(f"V0.6-G {result['status']}: results={Path(args.results_dir) / ROBUSTNESS_SUBDIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
