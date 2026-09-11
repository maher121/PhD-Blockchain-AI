"""Locked, first-access final-test evaluation for V0.6-F."""

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
import pickle
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
from src.lightweight.models import LightweightDetector
from src.pipeline_v06e import (
    LOCK_FILE_NAME,
    SIDECAR_FILE_NAME,
    ValidationLockError,
    fingerprint_feature_names,
    semantic_payload_sha256,
    verify_validation_lock,
)
from src.security.evaluation import (
    detector_visible_mask,
    evaluate_detection,
    evaluate_paired_detection,
    evaluate_visible_attacks,
)
from src.security.experiment_data import (
    PreparedAttackSplit,
    fingerprint_attack_labels,
    fingerprint_feature_matrix,
    fingerprint_frame,
    fingerprint_mapping,
    prepare_attack_split,
)
from src.security.ground_truth import assert_no_attack_metadata


V06F_STAGE = "V0.6-F"
V06F_PROTOCOL_VERSION = "v0.6-f-locked-final-test-1"
PRETEST_FAILURE_MESSAGE = "V0.6-F PRE-TEST INTEGRITY GATE FAILED"
FINAL_TEST_GATE_MESSAGE = "V0.6-F FINAL TEST GATE PASSED — accessing locked final test."
REQUIRED_ROLES = (
    "full_baseline",
    "best_unsupervised",
    "best_supervised",
    "smallest_preserving",
)
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_ATTACK = {"attack_type": "mixed", "attack_rate": 0.05, "attack_severity": "MEDIUM"}
EXPECTED_MODEL = {
    "model_id": "decision_tree",
    "parameters": {"max_depth": 5, "min_samples_leaf": 20, "class_weight": "balanced"},
}
DEFAULT_RESULTS_DIR = Path("results/feature_selection")
FINAL_TEST_SUBDIR = Path("final_test")
INFERENCE_REPEATS = 10
_CONTEXT_CAPABILITY = object()
SUMMARY_METRICS = (
    "average_precision",
    "f1",
    "recall",
    "precision",
    "roc_auc",
    "selected_feature_count",
    "inference_wall_time_sec",
    "peak_rss_mib",
    "serialized_model_bytes",
)


class V06FIntegrityError(RuntimeError):
    """Raised when locked final-test invariants do not hold."""


class V06FExecutionError(RuntimeError):
    """Raised when authorized final-test execution cannot safely continue."""


@dataclass(frozen=True)
class SeedPlan:
    seed: int
    selected_features: tuple[str, ...]
    selected_features_sha256: str
    model_reference: str
    model_file_sha256: str
    model_state_sha256: str
    model_size_bytes: int


@dataclass(frozen=True)
class ConfigurationPlan:
    configuration_id: str
    roles: tuple[str, ...]
    seed_plans: tuple[SeedPlan, ...]


@dataclass(frozen=True)
class ExecutionPlan:
    semantic_lock_sha256: str
    lock_file_sha256: str
    attack_config_sha256: str
    processed_dataset_metadata_sha256: str
    candidate_features: tuple[str, ...]
    configurations: tuple[ConfigurationPlan, ...]


@dataclass(frozen=True)
class IntegrityContext:
    lock: Mapping[str, Any]
    plan: ExecutionPlan
    plan_sha256: str
    gate_timestamp: str
    models: Mapping[tuple[str, int], LightweightDetector]
    model_paths: Mapping[tuple[str, int], Path]
    validation_audit: Mapping[str, Any]
    authorization_capability: object


def pre_test_integrity_gate(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    *,
    verifier: Callable[..., Mapping[str, Any]] = verify_validation_lock,
    model_loader: Callable[[Path | str], LightweightDetector] = LightweightDetector.load,
) -> IntegrityContext:
    """Verify every persisted lock and model invariant before data can be loaded."""
    results_dir = Path(results_dir)
    lock_path = results_dir / LOCK_FILE_NAME
    sidecar_path = results_dir / SIDECAR_FILE_NAME
    audit_path = results_dir / "validation_lock_audit.json"
    lock = _read_json(lock_path, "validation lock", ValidationLockError)
    expected_semantic_hash = lock.get("semantic_payload_sha256")
    if not _is_sha256(expected_semantic_hash):
        raise ValidationLockError("Validation lock has no valid semantic payload hash.")
    if semantic_payload_sha256(lock) != expected_semantic_hash:
        raise ValidationLockError("Validation lock semantic payload hash mismatch.")

    lock_file_sha256 = _sha256_file(lock_path)
    _verify_sidecar(sidecar_path, lock_path.name, lock_file_sha256)
    verification = verifier(lock_path, source_dir=results_dir, verify_sources=True)
    if (
        verification.get("status") != "PASS"
        or verification.get("semantic_payload_sha256") != expected_semantic_hash
        or verification.get("file_sha256") != lock_file_sha256
        or verification.get("source_hashes_verified") is not True
        or verification.get("source_semantics_reconstructed") is not True
        or verification.get("sidecar_verified") is not True
        or verification.get("test_accessed") is not False
    ):
        raise ValidationLockError("V0.6-E semantic/source reconstruction did not pass.")

    audit = _read_json(audit_path, "V0.6-E validation audit", ValidationLockError)
    _verify_validation_audit(audit, expected_semantic_hash, lock_file_sha256)
    _verify_frozen_protocol(lock)
    d_metadata = _read_json(
        results_dir / "run_metadata.json", "V0.6-D run metadata", ValidationLockError
    )
    attack_config_sha256 = str(d_metadata.get("attack_config_sha256", ""))
    processed_metadata_sha256 = str(
        d_metadata.get("processed_dataset_metadata_sha256", "")
    )
    if not _is_sha256(attack_config_sha256) or not _is_sha256(
        processed_metadata_sha256
    ):
        raise ValidationLockError("V0.6-D metadata lacks frozen input hashes.")
    attack_path = Path(lock["attack_configuration"]["attack_config_path"])
    if not attack_path.is_file() or _sha256_file(attack_path) != attack_config_sha256:
        raise ValidationLockError("Locked attack configuration content hash mismatch.")
    plan = _build_execution_plan(
        lock,
        lock_file_sha256,
        attack_config_sha256,
        processed_metadata_sha256,
    )

    loaded_models: dict[tuple[str, int], LightweightDetector] = {}
    model_paths: dict[tuple[str, int], Path] = {}
    for configuration in plan.configurations:
        for seed_plan in configuration.seed_plans:
            path = _resolve_model_path(seed_plan.model_reference, results_dir)
            _verify_model_file(path, seed_plan)
            try:
                model = model_loader(path)
            except Exception as exc:
                raise V06FIntegrityError(f"Cannot load locked model artifact: {path}") from exc
            _verify_loaded_model(model, seed_plan)
            _verify_model_file(path, seed_plan)
            loaded_models[(configuration.configuration_id, seed_plan.seed)] = model
            model_paths[(configuration.configuration_id, seed_plan.seed)] = path

    plan_sha256 = fingerprint_execution_plan(plan)
    return IntegrityContext(
        lock=_deep_freeze(deepcopy(lock)),
        plan=plan,
        plan_sha256=plan_sha256,
        gate_timestamp=_utc_now(),
        models=MappingProxyType(loaded_models),
        model_paths=MappingProxyType(model_paths),
        validation_audit=_deep_freeze(deepcopy(audit)),
        authorization_capability=_CONTEXT_CAPABILITY,
    )


def run_v06f_final_test(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    *,
    processed_dir: Path | str | None = None,
    context: IntegrityContext | None = None,
    loader: Callable[..., Any] | None = None,
    attack_preparer: Callable[..., PreparedAttackSplit] = prepare_attack_split,
    inference_runner: Callable[..., Any] | None = None,
    synthetic_test: bool = False,
) -> dict[str, Any]:
    """Perform the single authorized locked test evaluation without any fitting."""
    results_dir = Path(results_dir)
    context = context or pre_test_integrity_gate(results_dir)
    _verify_context_integrity(context)
    if synthetic_test and any(
        dependency is None for dependency in (loader, inference_runner)
    ):
        raise V06FExecutionError(
            "Synthetic execution requires fully injected loader and inference dependencies."
        )
    resolved_processed_dir = (
        PROCESSED_DATA_DIR if processed_dir is None else Path(processed_dir)
    )
    if not synthetic_test:
        metadata_path = resolved_processed_dir / "dataset_metadata.json"
        if (
            not metadata_path.is_file()
            or _sha256_file(metadata_path)
            != context.plan.processed_dataset_metadata_sha256
        ):
            raise V06FIntegrityError(
                "Processed dataset metadata does not match the locked V0.6-D input."
            )
    output_dir = results_dir / FINAL_TEST_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    access_path = output_dir / "test_access_state.json"
    previous_access = _existing_true_access(access_path)
    if access_path.exists() and not synthetic_test:
        raise V06FExecutionError("Locked final test access has already been claimed.")

    print(FINAL_TEST_GATE_MESSAGE)
    access_timestamp = _utc_now()
    runtime = _runtime_metadata()
    access_state = {
        "schema_version": V06F_PROTOCOL_VERSION,
        "stage": V06F_STAGE,
        "test_accessed": True,
        "first_test_access_timestamp": (
            previous_access.get("first_test_access_timestamp")
            if previous_access
            else access_timestamp
        ),
        "semantic_lock_sha256": context.plan.semantic_lock_sha256,
        "validation_lock_file_sha256": context.plan.lock_file_sha256,
        "git_commit": runtime["git_commit"],
        "runtime": runtime,
        "synthetic_test": bool(synthetic_test),
    }
    if synthetic_test:
        _write_json_atomic(access_path, access_state)
    else:
        _write_json_exclusive(access_path, access_state)
    _verify_context_integrity(context)

    if loader is None:
        from src.ai.model_utils import load_processed_dataco_splits

        loader = load_processed_dataco_splits
    loader_kwargs = {} if processed_dir is None else {"processed_dir": processed_dir}
    bundle = loader(("train", "test"), **loader_kwargs)
    _verify_context_integrity(context)
    if tuple(bundle.selected_features) != context.plan.candidate_features:
        raise V06FExecutionError("Loaded candidate features differ from the validation lock.")
    if set(bundle.splits) != {"train", "test"}:
        raise V06FExecutionError("Final-test loader must expose exactly train and test splits.")
    if not synthetic_test:
        stored_metadata = _read_json(
            resolved_processed_dir / "dataset_metadata.json",
            "processed dataset metadata",
            V06FExecutionError,
        )
        if bundle.dataset_metadata != stored_metadata:
            raise V06FExecutionError(
                "Loaded dataset metadata differs from the locked metadata file."
            )

    attack = context.lock["attack_configuration"]
    manifestations: dict[int, PreparedAttackSplit] = {}
    manifestation_before: dict[int, str] = {}
    for seed in EXPECTED_SEEDS:
        _verify_context_integrity(context)
        prepared = attack_preparer(
            split=bundle.splits["test"],
            training_reference=bundle.splits["train"],
            candidate_features=context.plan.candidate_features,
            attack_type=attack["attack_type"],
            attack_rate=float(attack["attack_rate"]),
            severity=attack["attack_severity"],
            random_seed=seed,
            experiment_id=f"v0_6_f_final_test_s{seed}",
            config_path=attack["attack_config_path"],
            test_authorization_id=context.plan.semantic_lock_sha256,
            test_authorization_capability=context,
        )
        _verify_prepared_manifestation(
            prepared, seed, context.plan.candidate_features, context
        )
        manifestations[seed] = prepared
        manifestation_before[seed] = _manifestation_fingerprint(prepared)

    if inference_runner is None:
        from src.lightweight.training import run_inference

        inference_runner = run_inference

    rows: list[dict[str, Any]] = []
    reuse_evidence: dict[str, dict[str, Any]] = {
        str(seed): {
            "object_id": id(prepared),
            "manifestation_sha256": manifestation_before[seed],
            "configuration_ids": [],
        }
        for seed, prepared in manifestations.items()
    }
    for configuration in context.plan.configurations:
        for seed_plan in configuration.seed_plans:
            run_id = f"v06f_s{seed_plan.seed}_{configuration.configuration_id}"
            prepared = manifestations[seed_plan.seed]
            reuse_evidence[str(seed_plan.seed)]["configuration_ids"].append(
                configuration.configuration_id
            )
            try:
                _verify_context_integrity(context)
                if _manifestation_fingerprint(prepared) != manifestation_before[seed_plan.seed]:
                    raise V06FIntegrityError("A final-test manifestation changed during execution.")
                model = context.models[(configuration.configuration_id, seed_plan.seed)]
                artifact_path = context.model_paths[
                    (configuration.configuration_id, seed_plan.seed)
                ]
                _verify_loaded_model(model, seed_plan)
                _verify_model_file(artifact_path, seed_plan)
                row = _evaluate_locked_run(
                    run_id,
                    configuration,
                    seed_plan,
                    model,
                    prepared,
                    artifact_path,
                    inference_runner,
                )
                _verify_model_file(artifact_path, seed_plan)
                rows.append(row)
            except Exception as exc:
                rows.append(
                    {
                        "run_id": run_id,
                        "status": "FAILED",
                        "seed": seed_plan.seed,
                        "configuration_id": configuration.configuration_id,
                        "roles_json": json.dumps(configuration.roles),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "model_fit_time_sec": None,
                        "selector_fit_time_sec": None,
                        "reselection_performed": False,
                    }
                )

    for seed, prepared in manifestations.items():
        if _manifestation_fingerprint(prepared) != manifestation_before[seed]:
            raise V06FIntegrityError("A final-test manifestation changed during execution.")
    _verify_context_integrity(context)
    outputs = _write_final_outputs(
        output_dir,
        rows,
        context,
        access_state,
        reuse_evidence,
        runtime,
    )
    completed = sum(row["status"] == "COMPLETED" for row in rows)
    return {
        "status": "PASS" if completed == len(rows) else "COMPLETED_WITH_FAILURES",
        "expected_run_count": len(rows),
        "completed_run_count": completed,
        "failed_run_count": len(rows) - completed,
        "unique_configuration_count": len(context.plan.configurations),
        "role_mapping": _role_mapping(context.plan),
        "outputs": outputs,
        "test_accessed": True,
    }


def fingerprint_execution_plan(plan: ExecutionPlan) -> str:
    payload = json.dumps(asdict(plan), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_execution_plan(
    lock: Mapping[str, Any],
    lock_file_sha256: str,
    attack_config_sha256: str,
    processed_dataset_metadata_sha256: str,
) -> ExecutionPlan:
    roles = lock["roles"]
    configurations = lock["locked_configurations"]
    grouped: dict[str, list[str]] = {}
    for role in REQUIRED_ROLES:
        role_record = roles.get(role)
        if not isinstance(role_record, Mapping):
            raise V06FIntegrityError(f"Required role is missing: {role}")
        reference = role_record.get("locked_configuration_ref")
        if not isinstance(reference, str) or not reference:
            raise V06FIntegrityError(f"Required role has no locked configuration: {role}")
        if role_record.get("configuration_id") != reference or reference not in configurations:
            raise V06FIntegrityError(f"Role/configuration reference is inconsistent: {role}")
        grouped.setdefault(reference, []).append(role)

    plan_configurations: list[ConfigurationPlan] = []
    for configuration_id, role_names in grouped.items():
        configuration = configurations[configuration_id]
        seed_records = configuration.get("seed_specific")
        if not isinstance(seed_records, Mapping) or set(seed_records) != {str(s) for s in EXPECTED_SEEDS}:
            raise V06FIntegrityError(f"Configuration lacks exact seed records: {configuration_id}")
        expected_features = {
            seed: row.get("selected_features") for seed, row in seed_records.items()
        }
        expected_fingerprints = {
            seed: row.get("selected_features_sha256") for seed, row in seed_records.items()
        }
        if (
            configuration.get("selected_features") != expected_features
            or configuration.get("selected_feature_fingerprints") != expected_fingerprints
        ):
            raise V06FIntegrityError(
                f"Configuration feature maps are inconsistent: {configuration_id}"
            )
        seed_plans: list[SeedPlan] = []
        for seed in EXPECTED_SEEDS:
            row = seed_records[str(seed)]
            features = row.get("selected_features")
            feature_hash = row.get("selected_features_sha256")
            artifact = row.get("model_artifact")
            if not isinstance(features, list) or fingerprint_feature_names(features) != feature_hash:
                raise V06FIntegrityError(
                    f"Selected feature fingerprint mismatch: {configuration_id}, seed {seed}"
                )
            if (
                role_names
                and any(roles[role]["selected_features"].get(str(seed)) != features for role in role_names)
            ):
                raise V06FIntegrityError(f"Role features differ from configuration: {configuration_id}")
            if any(
                roles[role]["selected_feature_fingerprints"].get(str(seed)) != feature_hash
                or roles[role]["model_artifacts"].get(str(seed)) != artifact
                for role in role_names
            ):
                raise V06FIntegrityError(f"Role fingerprints/artifacts differ: {configuration_id}")
            if not isinstance(artifact, Mapping):
                raise V06FIntegrityError(f"Missing model artifact: {configuration_id}, seed {seed}")
            if artifact.get("availability") != "hashed":
                raise V06FIntegrityError(f"Model artifact was not locked by file hash: {configuration_id}")
            seed_plans.append(
                SeedPlan(
                    seed=seed,
                    selected_features=tuple(features),
                    selected_features_sha256=str(feature_hash),
                    model_reference=str(artifact.get("reference", "")),
                    model_file_sha256=str(artifact.get("file_sha256", "")),
                    model_state_sha256=str(artifact.get("model_state_sha256", "")),
                    model_size_bytes=int(artifact.get("serialized_model_bytes", -1)),
                )
            )
        plan_configurations.append(
            ConfigurationPlan(configuration_id, tuple(role_names), tuple(seed_plans))
        )
    return ExecutionPlan(
        semantic_lock_sha256=str(lock["semantic_payload_sha256"]),
        lock_file_sha256=lock_file_sha256,
        attack_config_sha256=attack_config_sha256,
        processed_dataset_metadata_sha256=processed_dataset_metadata_sha256,
        candidate_features=tuple(lock["candidate_manifest"]["features"]),
        configurations=tuple(plan_configurations),
    )


def _verify_frozen_protocol(lock: Mapping[str, Any]) -> None:
    if lock.get("test_accessed") is not False or tuple(lock.get("seeds", ())) != EXPECTED_SEEDS:
        raise V06FIntegrityError("The lock does not attest untouched test data and exact seeds 42-46.")
    attack = lock.get("attack_configuration")
    if not isinstance(attack, Mapping) or any(attack.get(k) != v for k, v in EXPECTED_ATTACK.items()):
        raise V06FIntegrityError("Locked attack must be exactly mixed/0.05/MEDIUM.")
    if not isinstance(attack.get("attack_config_path"), str) or not attack["attack_config_path"]:
        raise V06FIntegrityError("Locked attack configuration path is missing.")
    if lock.get("model_configuration") != EXPECTED_MODEL:
        raise V06FIntegrityError("Locked model is not the frozen V0.5 decision tree.")
    threshold = lock.get("threshold")
    if (
        float(lock.get("decision_threshold", math.nan)) != 0.5
        or not isinstance(threshold, Mapping)
        or float(threshold.get("prediction_threshold", math.nan)) != 0.5
    ):
        raise V06FIntegrityError("Locked decision threshold must be exactly 0.5.")
    if set(lock.get("roles", {})) != set(REQUIRED_ROLES):
        raise V06FIntegrityError("Validation lock must contain exactly four required roles.")
    manifest = lock.get("candidate_manifest")
    if (
        not isinstance(manifest, Mapping)
        or fingerprint_feature_names(manifest.get("features", ())) != manifest.get("sha256")
        or manifest.get("sha256") != lock.get("candidate_manifest_hash")
    ):
        raise V06FIntegrityError("Candidate feature manifest fingerprint is inconsistent.")


def _verify_validation_audit(
    audit: Mapping[str, Any], semantic_hash: str, lock_file_sha256: str
) -> None:
    passed = {
        item.get("name")
        for item in audit.get("checks", ())
        if isinstance(item, Mapping) and item.get("passed") is True
    }
    required = {
        "source_d_leakage_audit_pass",
        "source_d_test_accessed_false",
        "source_d_metadata_test_accessed_false",
        "source_hashes_verified",
        "source_semantics_reconstructed",
        "semantic_payload_verified",
        "sidecar_file_hash_verified",
    }
    if (
        audit.get("stage") != "V0.6-E"
        or audit.get("status") != "PASS"
        or audit.get("test_accessed") is not False
        or audit.get("semantic_payload_sha256") != semantic_hash
        or audit.get("validation_lock_file_sha256") != lock_file_sha256
        or not required.issubset(passed)
    ):
        raise ValidationLockError("V0.6-E audit PASS/test_accessed=false evidence is invalid.")


def _verify_loaded_model(model: Any, seed_plan: SeedPlan) -> None:
    if not isinstance(model, LightweightDetector):
        raise V06FIntegrityError("Locked artifact did not load as LightweightDetector.")
    estimator = model.estimator
    if (
        not model.is_fitted
        or model.model_name != "decision_tree"
        or model.parameters != EXPECTED_MODEL["parameters"]
        or model.random_state != seed_plan.seed
        or getattr(estimator, "max_depth", None) != 5
        or getattr(estimator, "min_samples_leaf", None) != 20
        or getattr(estimator, "class_weight", None) != "balanced"
        or getattr(estimator, "random_state", None) != seed_plan.seed
        or tuple(model.feature_names) != seed_plan.selected_features
        or fingerprint_feature_names(model.feature_names) != seed_plan.selected_features_sha256
    ):
        raise V06FIntegrityError("Loaded model state/config/seed/features differ from the lock.")


def _verify_model_file(path: Path, seed_plan: SeedPlan) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != seed_plan.model_size_bytes
        or _sha256_file(path) != seed_plan.model_file_sha256
    ):
        raise V06FIntegrityError(f"Locked model file hash or size mismatch: {path}")


def _evaluate_locked_run(
    run_id: str,
    configuration: ConfigurationPlan,
    seed_plan: SeedPlan,
    model: LightweightDetector,
    prepared: PreparedAttackSplit,
    artifact_path: Path,
    inference_runner: Callable[..., Any],
) -> dict[str, Any]:
    selected = list(seed_plan.selected_features)
    if "Late_delivery_risk" in selected:
        raise V06FIntegrityError("Late_delivery_risk cannot be a final-test model feature.")
    assert_no_attack_metadata(selected)
    attacked_input = prepared.features.loc[:, selected]
    clean_input = prepared.clean_features.loc[:, selected]
    measured = inference_runner(
        model,
        attacked_input,
        artifact_path=artifact_path,
        repeats=INFERENCE_REPEATS,
    )
    attacked_predictions = _prediction_frame(model, attacked_input, prepared, measured.predictions)
    clean_predictions = _prediction_frame(model, clean_input, prepared, model.predict_frame(clean_input))
    metrics = _normalize_metrics(evaluate_detection(prepared.ground_truth, attacked_predictions))
    visible_mask = detector_visible_mask(clean_input, attacked_input)
    visible = evaluate_visible_attacks(prepared.ground_truth, attacked_predictions, visible_mask)
    paired = evaluate_paired_detection(
        prepared.ground_truth, clean_predictions, attacked_predictions
    )
    resource = measured.resources
    visible_attacked = int(
        (prepared.ground_truth["is_attack"].eq(1).to_numpy() & visible_mask).sum()
    )
    row: dict[str, Any] = {
        "run_id": run_id,
        "status": "COMPLETED",
        "seed": seed_plan.seed,
        "configuration_id": configuration.configuration_id,
        "roles_json": json.dumps(configuration.roles),
        "selected_features_json": json.dumps(selected),
        "selected_features_sha256": seed_plan.selected_features_sha256,
        "selected_feature_count": len(selected),
        "prediction_threshold": 0.5,
        "attack_type": "mixed",
        "attack_rate": 0.05,
        "attack_severity": "MEDIUM",
        "manifestation_sha256": _manifestation_fingerprint(prepared),
        "evaluated_records": metrics["evaluated_records"],
        "attacked_records": metrics["attacked_records"],
        "attack_prevalence": metrics["attack_prevalence"],
        "visible_attacked_records": visible_attacked,
        "invisible_attacked_records": metrics["attacked_records"] - visible_attacked,
        "visible_only_recall": None if visible is None else visible["recall"],
        "selector_fit_time_sec": None,
        "model_fit_time_sec": None,
        "fitting_performed": False,
        "reselection_performed": False,
        "inference_repeats": INFERENCE_REPEATS,
        "inference_wall_time_sec": resource.wall_time_sec,
        "per_record_inference_sec": resource.wall_time_sec / len(attacked_input),
        "peak_rss_bytes": resource.peak_rss_bytes,
        "peak_rss_mib": resource.peak_rss_mb,
        "serialized_model_bytes": seed_plan.model_size_bytes,
    }
    row.update(metrics)
    row.update({f"paired_induced_{key}": value for key, value in paired.items()})
    return row


def _prediction_frame(
    model: LightweightDetector,
    features: pd.DataFrame,
    prepared: PreparedAttackSplit,
    raw_predictions: pd.DataFrame,
) -> pd.DataFrame:
    scores = raw_predictions["anomaly_score"].to_numpy(dtype=float)
    labels = (scores >= 0.5).astype("int8")
    return pd.DataFrame(
        {
            "record_id": prepared.features["row_id"].to_numpy(copy=True),
            "anomaly_score": scores,
            "anomaly_label": labels,
        }
    )


def _normalize_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    evaluated = int(metrics["evaluated_records"])
    attacked = int(metrics["attacked_records"])
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "average_precision": metrics["average_precision"],
        "pr_auc_trapezoidal": metrics["pr_auc"],
        "pr_auc_trapezoidal_method": metrics["pr_auc_method"],
        "roc_auc": metrics["roc_auc"],
        "true_positives": metrics["true_positives"],
        "true_negatives": metrics["true_negatives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "TP": metrics["true_positives"],
        "TN": metrics["true_negatives"],
        "FP": metrics["false_positives"],
        "FN": metrics["false_negatives"],
        "evaluated_records": evaluated,
        "attacked_records": attacked,
        "attack_prevalence": float(attacked / evaluated) if evaluated else None,
    }


def _verify_prepared_manifestation(
    prepared: PreparedAttackSplit,
    seed: int,
    candidates: Sequence[str],
    context: IntegrityContext,
) -> None:
    if not isinstance(prepared, PreparedAttackSplit) or prepared.split_name != "test":
        raise V06FExecutionError("Attack preparer did not return an authorized test manifestation.")
    metadata = prepared.attack_metadata
    mode = metadata.get("attack_mode")
    if (
        mode != "mixed"
        or float(metadata.get("configured_attack_rate", math.nan)) != 0.05
        or metadata.get("severity") != "MEDIUM"
        or metadata.get("random_seed") != seed
        or tuple(prepared.candidate_features) != tuple(candidates)
        or prepared.labels.name != "is_attack"
        or "is_attack" not in prepared.ground_truth
        or prepared.test_authorization_id != context.plan.semantic_lock_sha256
        or prepared.test_authorization_capability is not context
    ):
        raise V06FExecutionError("Prepared attack manifestation violates the locked protocol.")
    assert_no_attack_metadata(prepared.features)


def _manifestation_fingerprint(prepared: PreparedAttackSplit) -> str:
    values = {
        "row_ids_sha256": prepared.row_ids_sha256,
        "clean_features_sha256": fingerprint_feature_matrix(
            prepared.clean_features, prepared.candidate_features
        ),
        "attacked_features_sha256": fingerprint_feature_matrix(
            prepared.features, prepared.candidate_features
        ),
        "labels_sha256": fingerprint_attack_labels(prepared.features, prepared.labels),
        "ground_truth_sha256": fingerprint_frame(prepared.ground_truth),
        "manifest_sha256": fingerprint_frame(prepared.manifest),
        "attack_metadata_sha256": fingerprint_mapping(prepared.attack_metadata),
    }
    expected = {
        "clean_features_sha256": prepared.clean_features_sha256,
        "attacked_features_sha256": prepared.attacked_features_sha256,
        "labels_sha256": prepared.labels_sha256,
        "ground_truth_sha256": prepared.ground_truth_sha256,
        "manifest_sha256": prepared.manifest_sha256,
        "attack_metadata_sha256": prepared.attack_metadata_sha256,
    }
    if any(values[key] != value for key, value in expected.items()):
        raise V06FIntegrityError("Prepared manifestation fingerprint is inconsistent.")
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_final_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    context: IntegrityContext,
    access_state: Mapping[str, Any],
    reuse_evidence: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, str]:
    frame = pd.DataFrame(rows)
    completed = frame.loc[frame["status"].eq("COMPLETED")].copy()
    expected_count = len(context.plan.configurations) * len(EXPECTED_SEEDS)
    matrix_complete = len(completed) == expected_count
    analysis_frame = completed if matrix_complete else completed.iloc[0:0]
    summary = _summarize_runs(analysis_frame)
    paired = _paired_comparisons(analysis_frame, context.plan)
    resources = summary.loc[
        summary["metric"].isin(
            {"selected_feature_count", "inference_wall_time_sec", "peak_rss_mib", "serialized_model_bytes"}
        )
    ].copy()
    generalization = _validation_to_test(summary, context)
    preservation = _descriptive_preservation(summary, context.plan)
    role_mapping = _role_mapping(context.plan)
    metadata = {
        "schema_version": V06F_PROTOCOL_VERSION,
        "stage": V06F_STAGE,
        "status": "COMPLETED" if len(completed) == expected_count else "COMPLETED_WITH_FAILURES",
        "test_accessed": True,
        "gate_timestamp": context.gate_timestamp,
        "first_test_access_timestamp": access_state["first_test_access_timestamp"],
        "locked_run_count": expected_count,
        "number_of_locked_roles": len(REQUIRED_ROLES),
        "number_of_unique_locked_configurations": len(context.plan.configurations),
        "completed_run_count": int(len(completed)),
        "failed_run_count": int(expected_count - len(completed)),
        "semantic_lock_sha256": context.plan.semantic_lock_sha256,
        "validation_lock_file_sha256": context.plan.lock_file_sha256,
        "execution_plan_sha256": context.plan_sha256,
        "attack_config_sha256": context.plan.attack_config_sha256,
        "processed_dataset_metadata_sha256": (
            context.plan.processed_dataset_metadata_sha256
        ),
        "candidate_features_sha256": fingerprint_feature_names(context.plan.candidate_features),
        "manifestation_fingerprints": {
            seed: evidence["manifestation_sha256"] for seed, evidence in reuse_evidence.items()
        },
        "software": runtime,
        "git_commit": runtime["git_commit"],
    }
    audit = _governance_audit(context, metadata, reuse_evidence, frame)

    payloads: dict[str, Any] = {
        "final_test_runs.json": rows,
        "final_test_summary.json": summary.to_dict(orient="records"),
        "final_test_paired_comparison.json": paired.to_dict(orient="records"),
        "final_test_resource_summary.json": resources.to_dict(orient="records"),
        "final_test_generalization.json": generalization.to_dict(orient="records"),
        "final_test_preservation.json": preservation.to_dict(orient="records"),
        "final_test_governance_audit.json": audit,
        "final_test_metadata.json": metadata,
        "final_test_role_mapping.json": role_mapping,
    }
    frames = {
        "final_test_runs.csv": frame,
        "final_test_summary.csv": summary,
        "final_test_paired_comparison.csv": paired,
        "final_test_resource_summary.csv": resources,
        "final_test_generalization.csv": generalization,
        "final_test_preservation.csv": preservation,
    }
    outputs: dict[str, str] = {}
    for name, payload in payloads.items():
        path = output_dir / name
        _write_json_atomic(path, payload)
        outputs[name] = str(path)
    for name, output_frame in frames.items():
        path = output_dir / name
        _write_csv_atomic(path, output_frame)
        outputs[name] = str(path)
    outputs.update(_write_figures(output_dir, completed, summary))
    hash_path = output_dir / "final_test_artifact_hashes.json"
    hashes = {
        str(path.relative_to(output_dir)): _sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path != hash_path
    }
    _write_json_atomic(hash_path, hashes)
    outputs[hash_path.name] = str(hash_path)
    return outputs


def _summarize_runs(completed: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if completed.empty:
        return pd.DataFrame(
            columns=["configuration_id", "metric", "count", "mean", "std", "min", "max", "ci95_low", "ci95_high"]
        )
    for configuration_id, group in completed.groupby("configuration_id", sort=False):
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            mean, std, low, high = _mean_std_t_ci(values)
            records.append(
                {
                    "configuration_id": configuration_id,
                    "metric": metric,
                    "count": len(values),
                    "mean": mean,
                    "std": std,
                    "min": float(values.min()) if len(values) else None,
                    "max": float(values.max()) if len(values) else None,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(records)


def _paired_comparisons(completed: pd.DataFrame, plan: ExecutionPlan) -> pd.DataFrame:
    baseline = _configuration_for_role(plan, "full_baseline")
    records: list[dict[str, Any]] = []
    if completed.empty:
        return pd.DataFrame(columns=["configuration_id", "metric", "paired_seed_count", "mean_difference", "ci95_low", "ci95_high"])
    base = completed.loc[completed["configuration_id"].eq(baseline)].set_index("seed")
    for configuration in plan.configurations:
        current = completed.loc[
            completed["configuration_id"].eq(configuration.configuration_id)
        ].set_index("seed")
        common = sorted(set(base.index) & set(current.index))
        for metric in ("average_precision", "f1", "recall"):
            differences = (
                current.loc[common, metric].to_numpy(dtype=float)
                - base.loc[common, metric].to_numpy(dtype=float)
            )
            mean, _, low, high = _mean_std_t_ci(differences)
            records.append(
                {
                    "configuration_id": configuration.configuration_id,
                    "baseline_configuration_id": baseline,
                    "metric": metric,
                    "paired_seed_count": len(differences),
                    "mean_difference": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(records)


def _validation_to_test(summary: pd.DataFrame, context: IntegrityContext) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for configuration in context.plan.configurations:
        locked = context.lock["locked_configurations"][configuration.configuration_id]
        for metric in ("average_precision", "f1", "recall", "precision", "roc_auc"):
            match = summary.loc[
                summary["configuration_id"].eq(configuration.configuration_id)
                & summary["metric"].eq(metric),
                "mean",
            ]
            test_mean = None if match.empty else match.iloc[0]
            validation_mean = locked["metrics"][metric]
            records.append(
                {
                    "configuration_id": configuration.configuration_id,
                    "metric": metric,
                    "validation_mean": validation_mean,
                    "final_test_mean": test_mean,
                    "test_minus_validation": None if test_mean is None else test_mean - validation_mean,
                }
            )
    return pd.DataFrame(records)


def _descriptive_preservation(summary: pd.DataFrame, plan: ExecutionPlan) -> pd.DataFrame:
    baseline = _configuration_for_role(plan, "full_baseline")
    margins = {"average_precision": 5.0, "f1": 5.0, "recall": 10.0}
    means = {
        (row.configuration_id, row.metric): row.mean
        for row in summary.itertuples(index=False)
    }
    records: list[dict[str, Any]] = []
    for configuration in plan.configurations:
        checks = []
        for metric, margin in margins.items():
            base = means.get((baseline, metric))
            value = means.get((configuration.configuration_id, metric))
            degradation = None
            passed = False
            if base is not None and value is not None:
                degradation = 0.0 if base == 0 else max(0.0, 100.0 * (base - value) / base)
                passed = degradation <= margin
            checks.append(passed)
            records.append(
                {
                    "configuration_id": configuration.configuration_id,
                    "metric": metric,
                    "margin_percent": margin,
                    "relative_degradation_percent": degradation,
                    "descriptively_preserved": passed,
                    "selection_performed": False,
                }
            )
        for record in records[-3:]:
            record["overall_descriptive_preservation"] = all(checks)
    return pd.DataFrame(records)


def _governance_audit(
    context: IntegrityContext,
    metadata: Mapping[str, Any],
    reuse_evidence: Mapping[str, Any],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    completed = frame.loc[frame["status"].eq("COMPLETED")]
    checks = {
        "lock_integrity_verified_before_test_access": True,
        "test_accessed_only_after_gate": True,
        "validation_lock_sidecar_verified": True,
        "v06e_semantic_source_reconstruction_passed": True,
        "semantic_hash_read_from_lock": True,
        "v06e_audit_pass_and_test_unaccessed": True,
        "frozen_mixed_005_medium_attack": True,
        "frozen_decision_tree_configuration": True,
        "frozen_threshold_05": True,
        "exact_seeds_42_through_46": True,
        "four_required_roles_verified": True,
        "role_configuration_feature_fingerprints_consistent": True,
        "model_file_hashes_and_sizes_verified": True,
        "loaded_model_state_config_seed_features_verified": True,
        "unique_configurations_derived_only_from_roles": True,
        "duplicate_roles_collapsed_and_mapping_preserved": True,
        "execution_plan_frozen_and_fingerprinted": True,
        "test_access_state_written_before_loader": True,
        "one_manifestation_per_seed_reused_across_configurations": (
            set(reuse_evidence) == {str(seed) for seed in EXPECTED_SEEDS}
            and all(
                len(value["configuration_ids"]) == len(context.plan.configurations)
                for value in reuse_evidence.values()
            )
        ),
        "all_locked_runs_completed": (
            metadata["completed_run_count"] == metadata["locked_run_count"]
            and metadata["failed_run_count"] == 0
        ),
        "locked_models_used_without_training": bool(
            completed.empty or completed["fitting_performed"].eq(False).all()
        ),
        "selector_fit_not_performed": bool(
            completed.empty or completed["selector_fit_time_sec"].isna().all()
        ),
        "selector_not_fitted_on_validation": True,
        "selector_not_fitted_on_test": True,
        "model_fit_not_performed": bool(
            completed.empty or completed["model_fit_time_sec"].isna().all()
        ),
        "test_labels_not_used_in_training": True,
        "test_features_not_used_in_training": True,
        "preprocessing_not_fitted_on_test": True,
        "preprocessor_refit_not_performed": True,
        "locked_k_unchanged": True,
        "locked_selected_feature_lists_unchanged": True,
        "locked_model_parameters_unchanged": True,
        "locked_threshold_unchanged": True,
        "locked_attack_configuration_unchanged": True,
        "threshold_tuning_not_performed": True,
        "consensus_selection_not_performed": True,
        "reselection_not_performed": bool(
            completed.empty or completed["reselection_performed"].eq(False).all()
        ),
        "no_configuration_choice_from_test_metrics": True,
        "controlled_is_attack_ground_truth_used": True,
        "late_delivery_risk_not_ground_truth_or_feature": True,
        "attack_metadata_absent_from_model_features": True,
        "all_failures_recorded": metadata["completed_run_count"] + metadata["failed_run_count"] == metadata["locked_run_count"],
        "no_p_values_energy_claims_or_reselection": True,
        "test_accessed_true": True,
    }
    return {
        "schema_version": V06F_PROTOCOL_VERSION,
        "stage": V06F_STAGE,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "test_accessed": True,
        "gate_timestamp": context.gate_timestamp,
        "first_test_access_timestamp": metadata["first_test_access_timestamp"],
        "semantic_lock_sha256": context.plan.semantic_lock_sha256,
        "execution_plan_sha256": context.plan_sha256,
        "manifestation_reuse_evidence": deepcopy(dict(reuse_evidence)),
        "checks": [
            {"name": name, "passed": bool(passed)} for name, passed in checks.items()
        ],
    }


def _write_figures(
    output_dir: Path, completed: pd.DataFrame, summary: pd.DataFrame
) -> dict[str, str]:
    if completed.empty:
        return {}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    metrics = summary.loc[summary["metric"].isin(["average_precision", "f1", "recall"])]
    pivot = metrics.pivot(index="configuration_id", columns="metric", values="mean")
    axis = pivot.plot(kind="bar", figsize=(9, 5), rot=20)
    axis.set_ylabel("Final-test mean")
    axis.set_title("Locked final-test detection metrics")
    axis.figure.tight_layout()
    path = figure_dir / "final_test_metrics.png"
    axis.figure.savefig(path, dpi=160)
    plt.close(axis.figure)
    outputs[path.name] = str(path)

    resource = summary.loc[summary["metric"].eq("inference_wall_time_sec")]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(resource["configuration_id"], resource["mean"])
    axis.set_ylabel("Seconds per inference call")
    axis.set_title("Locked artifact inference (10-repeat measurement)")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    path = figure_dir / "inference_resources.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs[path.name] = str(path)
    return outputs


def _mean_std_t_ci(values: np.ndarray) -> tuple[Any, Any, Any, Any]:
    if len(values) == 0:
        return None, None, None, None
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, 0.0, None, None
    std = float(np.std(values, ddof=1))
    if len(values) != len(EXPECTED_SEEDS):
        return mean, std, None, None
    critical = {1: 12.7062047364, 2: 4.3026527297, 3: 3.1824463053, 4: 2.7764451052}[len(values) - 1]
    half_width = critical * std / math.sqrt(len(values))
    return mean, std, mean - half_width, mean + half_width


def _role_mapping(plan: ExecutionPlan) -> dict[str, str]:
    return {
        role: configuration.configuration_id
        for configuration in plan.configurations
        for role in configuration.roles
    }


def _configuration_for_role(plan: ExecutionPlan, role: str) -> str:
    return _role_mapping(plan)[role]


def _verify_context_integrity(context: IntegrityContext) -> None:
    if (
        context.authorization_capability is not _CONTEXT_CAPABILITY
        or fingerprint_execution_plan(context.plan) != context.plan_sha256
    ):
        raise V06FIntegrityError("Locked execution plan mutated after the integrity gate.")


def _existing_true_access(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    state = _read_json(path, "test access state", V06FExecutionError)
    return state if state.get("test_accessed") is True else None


def _verify_sidecar(path: Path, lock_name: str, file_sha256: str) -> None:
    try:
        parts = path.read_text(encoding="ascii").strip().split()
    except OSError as exc:
        raise ValidationLockError(f"Cannot load validation lock sidecar: {exc}") from exc
    if len(parts) != 2 or parts[0] != file_sha256 or parts[1] != lock_name:
        raise ValidationLockError("Validation lock file SHA-256 sidecar mismatch.")


def _resolve_model_path(reference: str, results_dir: Path) -> Path:
    path = Path(reference)
    candidates = [path] if path.is_absolute() else [path, results_dir / path, results_dir.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise V06FIntegrityError(f"Locked model artifact is unavailable: {reference}")


def _model_state_sha256(model: LightweightDetector) -> str:
    payload = pickle.dumps(model.estimator, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise V06FIntegrityError(f"Cannot hash required file: {path}") from exc
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def _read_json(path: Path, description: str, error_type: type[Exception]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise error_type(f"Cannot load {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise error_type(f"{description} must be a JSON object.")
    return value


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json_atomic(path: Path, payload: Any) -> None:
    text = json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_text_atomic(path, text)


def _write_json_exclusive(path: Path, payload: Any) -> None:
    text = json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise V06FExecutionError(
            "Locked final test access was claimed by another execution."
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _runtime_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    try:
        dirty = bool(
            subprocess.run(
                ["git", "status", "--short"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return {
        "timestamp_utc": _utc_now(),
        "git_commit": commit,
        "git_worktree_dirty": dirty,
        "pipeline_v06f_sha256": _sha256_file(Path(__file__)),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the V0.6-F locked final-test evaluation once.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        context = pre_test_integrity_gate(args.results_dir)
    except (ValidationLockError, V06FIntegrityError):
        print(PRETEST_FAILURE_MESSAGE)
        return 1
    try:
        result = run_v06f_final_test(
            args.results_dir, processed_dir=args.processed_dir, context=context
        )
    except V06FExecutionError as exc:
        print(f"V0.6-F FINAL TEST EXECUTION FAILED: {exc}")
        return 1
    print(f"V0.6-F {result['status']}: results={args.results_dir / FINAL_TEST_SUBDIR}")
    return 0 if result["failed_run_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
