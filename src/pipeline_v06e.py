"""Artifact-only V0.6-E validation configuration locking.

This module deliberately has no dependency on data-loading or experiment-running
modules. It consumes completed V0.6-D validation artifacts and can only lock or
verify their recorded outcomes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


V06E_STAGE = "V0.6-E"
V06E_PROTOCOL_VERSION = "v0.6-e-artifact-lock-1"
V06E_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
DEFAULT_RESULTS_DIR = Path("results/feature_selection")
LOCK_FILE_NAME = "validation_lock.json"
SIDECAR_FILE_NAME = "validation_lock.sha256"
REPORT_FILE_NAME = "validation_lock_report.md"
AUDIT_FILE_NAME = "validation_lock_audit.json"

_MANDATORY_ARTIFACTS = (
    "core_runs.csv",
    "selected_feature_sets.json",
    "leakage_audit.json",
    "run_metadata.json",
)
_OPTIONAL_SUPPORT_ARTIFACTS = (
    "core_runs.json",
    "validation_summary.csv",
    "paired_baseline_comparisons.csv",
    "paired_baseline_differences.csv",
    "non_inferiority_sensitivity.csv",
    "preservation_primary.csv",
    "candidate_selection.csv",
    "candidate_selection.json",
    "feature_selection_rankings.csv",
    "resource_comparison.csv",
    "selected_set_stability.csv",
    "ranking_stability.csv",
    "deterministic_unsupervised_stability.csv",
)
_PRESERVATION_METRICS = ("average_precision", "f1", "recall")
_MEAN_METRICS = (*_PRESERVATION_METRICS, "precision", "roc_auc")
_RESOURCE_COLUMNS = {
    "selector_fit_wall_time_sec": "selector_fit_wall_time_sec",
    "model_fit_wall_time_sec": "model_fit_wall_time_sec",
    "combined_train_wall_time_sec": "combined_train_wall_time_sec",
    "inference_wall_time_sec": "attacked_validation_inference_wall_time_sec",
    "peak_rss_mib": "peak_rss_mib",
    "model_size_bytes": "serialized_model_bytes",
}


class ValidationLockError(ValueError):
    """Raised when source artifacts or a persisted lock violate the protocol."""


@dataclass(frozen=True)
class PreservationRules:
    """Frozen primary and sensitivity margins, expressed as percentages."""

    average_precision_max_relative_degradation_percent: float = 5.0
    f1_max_relative_degradation_percent: float = 5.0
    recall_max_relative_degradation_percent: float = 10.0
    sensitivity_ap_f1_percent: tuple[float, ...] = (0.0, 5.0, 10.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantics": "five-seed configuration means; relative degradation from full baseline",
            "primary": {
                "average_precision_max_relative_degradation_percent": self.average_precision_max_relative_degradation_percent,
                "f1_max_relative_degradation_percent": self.f1_max_relative_degradation_percent,
                "recall_max_relative_degradation_percent": self.recall_max_relative_degradation_percent,
            },
            "sensitivity": {
                "average_precision_and_f1_margin_percent": list(
                    self.sensitivity_ap_f1_percent
                ),
                "recall_margin_percent": self.recall_max_relative_degradation_percent,
            },
        }


def build_validation_lock(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build and validate a lock from stored V0.6-D artifacts without writing."""
    results_dir = Path(results_dir)
    source_payloads, source_entries = _read_source_artifacts(results_dir)
    core = source_payloads["core_runs.csv"]
    selected_sets = source_payloads["selected_feature_sets.json"]
    source_audit = source_payloads["leakage_audit.json"]
    metadata = source_payloads["run_metadata.json"]

    _reject_ambiguous_pr_auc(source_payloads)
    _validate_source_attestations(source_audit, metadata)
    _validate_core(core)
    _verify_metadata_hash_claims(metadata, source_entries)

    candidate_manifest = _validate_candidate_manifest(core)
    _validate_selected_set_artifact(core, selected_sets)
    model_sources, model_records = _collect_model_artifacts(core, results_dir)
    source_entries.extend(model_sources)

    summary = _aggregate_core(core)
    rules = PreservationRules()
    evaluated = _evaluate_configurations(summary, rules)
    per_method = _select_per_method(evaluated)
    role_configuration_ids = _derive_roles(evaluated, per_method)
    paired = _paired_changes(core)
    _verify_stored_analysis(
        source_payloads, evaluated, per_method, role_configuration_ids, paired
    )
    stability = _stability_support(core, source_payloads)

    locked_ids = sorted(
        set(per_method.values())
        | {value for value in role_configuration_ids.values() if value is not None}
    )
    locked_configurations = {
        configuration_id: _build_locked_configuration(
            configuration_id,
            core,
            evaluated,
            paired,
            stability,
            model_records,
        )
        for configuration_id in locked_ids
    }
    roles = {
        role: _build_role(role, configuration_id, locked_configurations)
        for role, configuration_id in role_configuration_ids.items()
    }
    for configuration_id, configuration in locked_configurations.items():
        configuration["locked_roles"] = sorted(
            role
            for role, role_configuration_id in role_configuration_ids.items()
            if role_configuration_id == configuration_id
        )

    ordered_sources = sorted(
        source_entries, key=lambda item: (item["kind"], item["path"])
    )
    source_validation_artifacts = [
        entry["path"]
        for entry in ordered_sources
        if entry["kind"] != "referenced_model_artifact"
    ]
    source_artifact_hashes = {
        entry["path"]: entry["sha256"] for entry in ordered_sources
    }
    threshold = _single_configuration(
        core, ("prediction_threshold", "threshold_policy")
    )
    tie_break_rules = {
        "per_method_preserving": "fewer actual features, then higher average precision, then higher F1, then lexical configuration ID",
        "per_method_no_preserving": "higher average precision, then higher F1, then fewer actual features, then lexical configuration ID",
        "best_type_role": "higher average precision, then higher F1, then fewer actual features, then lexical configuration ID",
        "smallest_preserving_role": "fewer actual features, then higher average precision, then higher F1, then lexical configuration ID",
    }

    lock: dict[str, Any] = {
        "schema_version": V06E_PROTOCOL_VERSION,
        "stage": V06E_STAGE,
        "protocol_version": V06E_PROTOCOL_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "semantic_payload_sha256": None,
        "scope": "stored_development_validation_artifacts_only",
        "test_accessed": False,
        "source_artifacts": ordered_sources,
        "source_validation_artifacts": source_validation_artifacts,
        "source_artifact_hashes": source_artifact_hashes,
        "source_d_attestations": {
            "leakage_audit_status": source_audit["status"],
            "leakage_audit_test_accessed": source_audit["test_accessed"],
            "run_metadata_test_accessed": metadata["test_accessed"],
            "run_metadata_output_hash_claims_verified": True,
        },
        "candidate_manifest": candidate_manifest,
        "candidate_manifest_hash": candidate_manifest["sha256"],
        "seeds": list(V06E_SEEDS),
        "attack_configuration": _single_configuration(
            core, ("attack_type", "attack_rate", "attack_severity", "attack_config_path")
        ),
        "model_configuration": _model_configuration(core),
        "threshold": threshold,
        "decision_threshold": threshold["prediction_threshold"],
        "preservation_rules": rules.to_dict(),
        "sensitivity_margins": [0.0, 5.0, 10.0],
        "tie_breaks": tie_break_rules,
        "tie_break_rules": tie_break_rules,
        "evaluated_configurations": evaluated,
        "per_method_choices": per_method,
        "locked_configurations": locked_configurations,
        "roles": roles,
        "governance": {
            "selection_scope": "development_validation_only",
            "artifact_only": True,
            "selector_refit_performed": False,
            "model_refit_performed": False,
            "threshold_tuning_performed": False,
            "final_test_runner_available_to_module": False,
        },
    }
    _reject_ambiguous_pr_auc(lock)
    lock["semantic_payload_sha256"] = semantic_payload_sha256(lock)
    verify_validation_lock(lock, verify_sources=False)
    return lock


def create_validation_lock(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Create, report, audit, and post-write verify a V0.6-E lock."""
    results_dir = Path(results_dir)
    lock = build_validation_lock(results_dir, created_at=created_at)

    lock_path = results_dir / LOCK_FILE_NAME
    sidecar_path = results_dir / SIDECAR_FILE_NAME
    report_path = results_dir / REPORT_FILE_NAME
    audit_path = results_dir / AUDIT_FILE_NAME
    _write_json_atomic(lock_path, lock)
    file_sha256 = _sha256_file(lock_path)
    _write_text_atomic(sidecar_path, f"{file_sha256}  {LOCK_FILE_NAME}\n")
    _write_text_atomic(report_path, _render_report(lock))

    verification = verify_validation_lock(lock_path, source_dir=results_dir)
    audit = _build_audit(lock, verification, file_sha256)
    _write_json_atomic(audit_path, audit)
    return {
        "status": "PASS",
        "lock_path": lock_path,
        "sidecar_path": sidecar_path,
        "report_path": report_path,
        "audit_path": audit_path,
        "semantic_payload_sha256": lock["semantic_payload_sha256"],
        "file_sha256": file_sha256,
        "test_accessed": False,
    }


def verify_validation_lock(
    lock_or_path: Mapping[str, Any] | Path | str,
    *,
    source_dir: Path | str | None = None,
    verify_sources: bool = True,
) -> dict[str, Any]:
    """Verify semantic/source hashes, fingerprints, roles, and disk sidecar."""
    lock_path: Path | None = None
    if isinstance(lock_or_path, Mapping):
        lock = dict(lock_or_path)
    else:
        lock_path = Path(lock_or_path)
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationLockError(f"Cannot load validation lock: {exc}") from exc
        if source_dir is None:
            source_dir = lock_path.parent

    _reject_ambiguous_pr_auc(lock)
    expected_semantic = semantic_payload_sha256(lock)
    if lock.get("semantic_payload_sha256") != expected_semantic:
        raise ValidationLockError("Validation lock semantic payload hash mismatch.")
    _verify_lock_structure(lock)
    _verify_locked_fingerprints(lock)
    _verify_role_invariants(lock)

    if verify_sources:
        resolved_source_dir = Path(source_dir) if source_dir is not None else None
        _verify_persisted_sources(lock, resolved_source_dir)
        if resolved_source_dir is not None:
            rebuilt = build_validation_lock(
                resolved_source_dir, created_at=lock.get("created_at")
            )
            if rebuilt["semantic_payload_sha256"] != expected_semantic:
                raise ValidationLockError(
                    "Validation lock semantics do not match the source artifacts."
                )

    file_sha256 = None
    if lock_path is not None:
        file_sha256 = _sha256_file(lock_path)
        sidecar = lock_path.with_name(SIDECAR_FILE_NAME)
        try:
            parts = sidecar.read_text(encoding="ascii").strip().split()
        except OSError as exc:
            raise ValidationLockError(f"Cannot load validation lock sidecar: {exc}") from exc
        if not parts or parts[0] != file_sha256:
            raise ValidationLockError("Validation lock file SHA-256 sidecar mismatch.")
        if len(parts) > 1 and parts[1] != lock_path.name:
            raise ValidationLockError("Validation lock sidecar names a different file.")

    return {
        "status": "PASS",
        "semantic_payload_sha256": expected_semantic,
        "file_sha256": file_sha256,
        "source_hashes_verified": bool(verify_sources),
        "source_semantics_reconstructed": bool(
            verify_sources and source_dir is not None
        ),
        "sidecar_verified": lock_path is not None,
        "test_accessed": False,
    }


def semantic_payload_sha256(lock: Mapping[str, Any]) -> str:
    """Hash canonical lock semantics, excluding time and the self hash."""
    payload = dict(lock)
    payload.pop("created_at", None)
    payload.pop("semantic_payload_sha256", None)
    serialized = json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_source_artifacts(results_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payloads: dict[str, Any] = {}
    entries: list[dict[str, Any]] = []
    for name in _MANDATORY_ARTIFACTS:
        path = results_dir / name
        if not path.is_file():
            raise ValidationLockError(f"Missing required V0.6-D artifact: {name}")
        before = _sha256_file(path)
        payloads[name] = _read_artifact(path)
        after = _sha256_file(path)
        if before != after:
            raise ValidationLockError(f"Source artifact changed while reading: {name}")
        entries.append(_source_entry(name, path, "v06d_artifact", sha256=after))
    for name in _OPTIONAL_SUPPORT_ARTIFACTS:
        path = results_dir / name
        if path.is_file():
            before = _sha256_file(path)
            payloads[name] = _read_artifact(path)
            after = _sha256_file(path)
            if before != after:
                raise ValidationLockError(f"Source artifact changed while reading: {name}")
            entries.append(
                _source_entry(name, path, "v06d_support_artifact", sha256=after)
            )
    return payloads, entries


def _read_artifact(path: Path) -> Any:
    try:
        if path.suffix == ".csv":
            return pd.read_csv(path)
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValidationLockError(f"Cannot read source artifact {path.name}: {exc}") from exc


def _source_entry(
    name: str, path: Path, kind: str, *, sha256: str | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "path": name,
        "kind": kind,
        "sha256": sha256 or _sha256_file(path),
    }


def _validate_source_attestations(audit: Any, metadata: Any) -> None:
    if (
        not isinstance(audit, Mapping)
        or audit.get("schema_version") != "v0.6-d"
        or audit.get("scope") != "development_train_and_validation_only"
        or audit.get("status") != "PASS"
    ):
        raise ValidationLockError("V0.6-D leakage audit must have PASS status.")
    if audit.get("test_accessed") is not False:
        raise ValidationLockError("V0.6-D leakage audit must attest test_accessed=false.")
    required_checks = {
        "validation_only_evaluation",
        "preprocessing_fitted_on_training_rows_only",
        "attack_metadata_absent_from_model_features",
        "row_ids_are_not_model_features",
        "late_delivery_risk_is_not_cyber_ground_truth_or_model_feature",
        "selector_fitted_on_training_rows_only",
        "unsupervised_clean_train_and_supervised_controlled_label_governance",
        "validation_rows_never_used_during_selector_fit",
        "selected_feature_order_is_canonical",
        "validation_configuration_selection_recorded",
        "frozen_model_and_threshold",
        "all_expected_runs_recorded",
        "no_run_failures",
    }
    checks = audit.get("checks")
    passed_checks = {
        check.get("name")
        for check in checks or []
        if isinstance(check, Mapping) and check.get("passed") is True
    }
    if not required_checks.issubset(passed_checks):
        missing = sorted(required_checks - passed_checks)
        raise ValidationLockError(f"V0.6-D leakage audit lacks passing checks: {missing}")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema_version") != "v0.6-d"
        or metadata.get("execution_scope")
        != "development_train_and_validation_only"
        or metadata.get("run_count") != 60
        or metadata.get("successful_run_count") != 60
        or metadata.get("failed_run_count") != 0
        or metadata.get("test_accessed") is not False
    ):
        raise ValidationLockError("V0.6-D run metadata must attest test_accessed=false.")


def _validate_core(core: Any) -> None:
    required = {
        "status", "run_id", "seed", "evaluation_split", "training_split",
        "configuration_id", "selector_id", "selector_type", "selector_method",
        "selector_mode", "selector_parameters_json", "count_strategy",
        "requested_feature_count", "candidate_feature_count", "candidate_features_sha256",
        "selected_feature_count", "selected_features_json", "selected_features_sha256",
        "model_id", "model_parameters_json", "model_state_sha256",
        "prediction_threshold", "threshold_policy", "attack_type", "attack_rate",
        "attack_severity", "attack_config_path", "average_precision", "f1", "recall",
        "precision", "roc_auc",
        "selector_fit_wall_time_sec", "model_fit_wall_time_sec",
        "combined_train_wall_time_sec", "attacked_validation_inference_wall_time_sec",
        "peak_rss_mib", "serialized_model_bytes", "model_artifact_path",
        "resource_measurement_performed",
    }
    if not isinstance(core, pd.DataFrame):
        raise ValidationLockError("core_runs.csv did not load as a table.")
    missing = sorted(required - set(core.columns))
    if missing:
        raise ValidationLockError(f"core_runs.csv is missing required columns: {missing}")
    if len(core) != 60 or not core["status"].eq("success").all():
        raise ValidationLockError("V0.6-E requires exactly 60 successful core rows.")
    if not core["evaluation_split"].eq("validation").all():
        raise ValidationLockError("Every core row must use the validation evaluation split.")
    if not core["training_split"].eq("train").all():
        raise ValidationLockError("Every core row must record the train fitting split.")
    seeds = tuple(sorted(int(seed) for seed in core["seed"].unique()))
    if seeds != V06E_SEEDS:
        raise ValidationLockError(f"Seeds must be exactly {V06E_SEEDS}, observed {seeds}.")
    counts = core.groupby("seed").size()
    if not counts.reindex(V06E_SEEDS).eq(12).all():
        raise ValidationLockError("Each seed must contain exactly 12 configurations.")
    config_sets = core.groupby("seed")["configuration_id"].apply(lambda values: frozenset(values))
    if any(len(values) != 12 for values in config_sets) or len(set(config_sets)) != 1:
        raise ValidationLockError("The same 12 unique configurations must occur for every seed.")
    baseline = core.loc[core["selector_id"].eq("none")]
    if len(baseline) != 5 or not baseline.groupby("seed").size().eq(1).all():
        raise ValidationLockError("Exactly one full baseline is required per seed.")
    if baseline["configuration_id"].nunique() != 1:
        raise ValidationLockError("The baseline configuration ID must be invariant across seeds.")
    if core["run_id"].duplicated().any():
        raise ValidationLockError("Core run IDs must be unique.")
    test_columns = [
        column
        for column in core.columns
        if "test" in str(column).lower().replace("attestation", "")
    ]
    if test_columns:
        raise ValidationLockError(
            f"Core validation artifacts expose test-related columns: {test_columns}"
        )
    if not core["resource_measurement_performed"].eq(True).all():
        raise ValidationLockError("Every core row must contain performed resource measurements.")

    attack = _single_configuration(
        core, ("attack_type", "attack_rate", "attack_severity", "attack_config_path")
    )
    model = _model_configuration(core)
    threshold = _single_configuration(core, ("prediction_threshold", "threshold_policy"))
    if (
        attack["attack_type"] != "mixed"
        or not math.isclose(float(attack["attack_rate"]), 0.05)
        or attack["attack_severity"] != "MEDIUM"
    ):
        raise ValidationLockError("Core attack configuration is not the frozen V0.6 protocol.")
    if model != {
        "model_id": "decision_tree",
        "parameters": {
            "class_weight": "balanced",
            "max_depth": 5,
            "min_samples_leaf": 20,
        },
    }:
        raise ValidationLockError("Core model configuration is not the frozen V0.5 tree.")
    if not math.isclose(float(threshold["prediction_threshold"]), 0.5):
        raise ValidationLockError("Core prediction threshold is not frozen at 0.5.")
    for column in (*_MEAN_METRICS, *_RESOURCE_COLUMNS.values(), "selected_feature_count"):
        numeric = pd.to_numeric(core[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric).all():
            raise ValidationLockError(f"Core column {column} must contain finite numeric values.")
        if column in (*_RESOURCE_COLUMNS.values(), "selected_feature_count") and (numeric < 0).any():
            raise ValidationLockError(f"Core resource/count column {column} cannot be negative.")
        if column in _MEAN_METRICS and not numeric.between(0.0, 1.0).all():
            raise ValidationLockError(f"Core metric {column} must be in [0, 1].")
    identities = (
        "selector_id", "selector_type", "selector_method", "selector_mode",
        "count_strategy", "requested_feature_count"
    )
    for configuration_id, group in core.groupby("configuration_id", sort=False):
        for column in identities:
            if group[column].nunique(dropna=False) != 1:
                raise ValidationLockError(
                    f"Configuration {configuration_id} changes identity field {column}."
                )
        if group["selected_feature_count"].nunique() != 1:
            raise ValidationLockError(
                f"Configuration {configuration_id} has inconsistent actual feature counts."
            )
        _normalized_selector_parameters(group, str(configuration_id))
    for selector_id, group in core.loc[core["selector_id"].ne("none")].groupby("selector_id"):
        configs = group.drop_duplicates("configuration_id")
        natural = configs["count_strategy"].eq("natural")
        if natural.any() and (not natural.all() or len(configs) != 1):
            raise ValidationLockError(
                f"Natural selector method {selector_id} must have one configuration."
            )
    _validate_frozen_matrix(core)


def _validate_frozen_matrix(core: pd.DataFrame) -> None:
    specifications = {
        "none": ("none", "natural", {None}),
        "variance_threshold": ("unsupervised", "natural", {None}),
        "pairwise_correlation_filter": ("unsupervised", "natural", {None}),
        "correlation_redundancy_ranking": (
            "unsupervised", "fixed_k", {32, 22, 11}
        ),
        "mutual_information_select_k_best": (
            "supervised", "fixed_k", {32, 22, 11}
        ),
        "anova_f_select_k_best": ("supervised", "fixed_k", {32, 22, 11}),
    }
    configurations = core.drop_duplicates("configuration_id")
    if set(configurations["selector_id"]) != set(specifications):
        raise ValidationLockError("Core rows do not match the frozen V0.6-D selector methods.")
    for selector_id, (selector_type, strategy, expected_k) in specifications.items():
        rows = configurations.loc[configurations["selector_id"].eq(selector_id)]
        observed_k = {
            None if pd.isna(value) else int(value)
            for value in rows["requested_feature_count"]
        }
        if (
            set(rows["selector_type"]) != {selector_type}
            or set(rows["selector_mode"]) != {selector_type}
            or set(rows["count_strategy"]) != {strategy}
            or observed_k != expected_k
        ):
            raise ValidationLockError(
                f"Selector method {selector_id} violates the frozen V0.6-D matrix."
            )
        if strategy == "fixed_k" and any(
            int(row.selected_feature_count) != int(row.requested_feature_count)
            for row in rows.itertuples(index=False)
        ):
            raise ValidationLockError(
                f"Selector method {selector_id} did not retain its requested K."
            )


def _validate_candidate_manifest(core: pd.DataFrame) -> dict[str, Any]:
    if core["candidate_feature_count"].nunique() != 1:
        raise ValidationLockError("Candidate feature count must be invariant across all runs.")
    if core["candidate_features_sha256"].nunique() != 1:
        raise ValidationLockError("Candidate manifest hash must be invariant across all runs.")
    baseline = core.loc[core["selector_id"].eq("none")].sort_values("seed")
    manifests = [_parse_feature_list(value) for value in baseline["selected_features_json"]]
    if any(features != manifests[0] for features in manifests[1:]):
        raise ValidationLockError("Full-baseline candidate manifests differ across seeds.")
    features = manifests[0]
    expected_count = int(core["candidate_feature_count"].iloc[0])
    expected_hash = str(core["candidate_features_sha256"].iloc[0])
    if len(features) != expected_count or len(set(features)) != len(features):
        raise ValidationLockError("Candidate manifest count or uniqueness is invalid.")
    if fingerprint_feature_names(features) != expected_hash:
        raise ValidationLockError("Candidate manifest fingerprint does not match core rows.")
    position = {feature: index for index, feature in enumerate(features)}
    for row in core.itertuples(index=False):
        selected = _parse_feature_list(row.selected_features_json)
        if len(selected) != int(row.selected_feature_count) or len(set(selected)) != len(selected):
            raise ValidationLockError(f"Selected feature count is invalid for run {row.run_id}.")
        if any(feature not in position for feature in selected):
            raise ValidationLockError(f"Run {row.run_id} selects outside the candidate manifest.")
        if selected != sorted(selected, key=position.__getitem__):
            raise ValidationLockError(f"Run {row.run_id} does not preserve candidate order.")
        if fingerprint_feature_names(selected) != str(row.selected_features_sha256):
            raise ValidationLockError(f"Selected feature fingerprint mismatch for run {row.run_id}.")
    return {"feature_count": expected_count, "features": features, "sha256": expected_hash}


def _validate_selected_set_artifact(core: pd.DataFrame, selected_sets: Any) -> None:
    if not isinstance(selected_sets, Mapping) or set(selected_sets) != set(core["run_id"]):
        raise ValidationLockError("selected_feature_sets.json must contain every core run exactly once.")
    for row in core.itertuples(index=False):
        record = selected_sets[row.run_id]
        if not isinstance(record, Mapping):
            raise ValidationLockError(f"Selected-set record {row.run_id} is not an object.")
        selected = _parse_feature_list(row.selected_features_json)
        if (
            record.get("configuration_id") != row.configuration_id
            or int(record.get("seed")) != int(row.seed)
            or record.get("selected_features") != selected
            or record.get("selected_features_sha256") != row.selected_features_sha256
        ):
            raise ValidationLockError(f"Selected-set artifact disagrees with core run {row.run_id}.")


def _collect_model_artifacts(
    core: pd.DataFrame, results_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    records: dict[str, dict[str, Any]] = {}
    seen_paths: dict[str, str] = {}
    for row in core.itertuples(index=False):
        reference = str(row.model_artifact_path).strip()
        if not reference or reference.lower() == "nan":
            raise ValidationLockError(f"Run {row.run_id} lacks a model artifact reference.")
        path = _resolve_model_path(reference, results_dir)
        file_hash = None
        availability = "not_available"
        if path is not None:
            file_hash = _sha256_file(path)
            availability = "hashed"
            expected_size = int(row.serialized_model_bytes)
            if path.stat().st_size != expected_size:
                raise ValidationLockError(f"Model artifact size mismatch for run {row.run_id}.")
            path_key = str(path)
            if path_key not in seen_paths:
                sources.append(
                    {
                        "name": path.name,
                        "path": path_key,
                        "kind": "referenced_model_artifact",
                        "sha256": file_hash,
                    }
                )
                seen_paths[path_key] = file_hash
            elif seen_paths[path_key] != file_hash:
                raise ValidationLockError(f"Model artifact changed while hashing: {path}")
        state_hash = str(row.model_state_sha256).strip()
        if len(state_hash) != 64 or any(
            character not in "0123456789abcdef" for character in state_hash.lower()
        ):
            raise ValidationLockError(f"Run {row.run_id} has an invalid model-state SHA-256.")
        records[row.run_id] = {
            "reference": reference,
            "availability": availability,
            "model_state_sha256": state_hash,
            "file_sha256": file_hash,
            "serialized_model_bytes": int(row.serialized_model_bytes),
        }
    return sources, records


def _resolve_model_path(reference: str, results_dir: Path) -> Path | None:
    path = Path(reference)
    candidates = [path] if path.is_absolute() else [path, results_dir / path, results_dir.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _aggregate_core(core: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for configuration_id, group in core.groupby("configuration_id", sort=True):
        first = group.iloc[0]
        record: dict[str, Any] = {
            "configuration_id": configuration_id,
            "selector_id": str(first["selector_id"]),
            "selector_type": str(first["selector_type"]),
            "count_strategy": str(first["count_strategy"]),
            "requested_feature_count": _optional_int(first["requested_feature_count"]),
            "actual_feature_count": int(first["selected_feature_count"]),
            "seed_count": int(group["seed"].nunique()),
            "means": {},
            "resources": {},
        }
        for metric in _MEAN_METRICS:
            record["means"][metric] = float(pd.to_numeric(group[metric]).mean())
        for name, column in _RESOURCE_COLUMNS.items():
            record["resources"][name] = float(pd.to_numeric(group[column]).mean())
        summary[configuration_id] = record
    return summary


def _evaluate_configurations(
    summary: Mapping[str, Mapping[str, Any]], rules: PreservationRules
) -> dict[str, dict[str, Any]]:
    baseline_ids = [key for key, value in summary.items() if value["selector_id"] == "none"]
    if len(baseline_ids) != 1:
        raise ValidationLockError("Exactly one aggregate baseline is required.")
    baseline = summary[baseline_ids[0]]
    evaluated: dict[str, dict[str, Any]] = {}
    for configuration_id, source in summary.items():
        record = _deep_copy_json(source)
        baseline_count = float(baseline["actual_feature_count"])
        record["feature_reduction_percent"] = (
            100.0 * (baseline_count - float(source["actual_feature_count"])) / baseline_count
        )
        sensitivity: dict[str, Any] = {}
        for margin in rules.sensitivity_ap_f1_percent:
            passes: dict[str, bool] = {}
            degradations: dict[str, float] = {}
            for metric in _PRESERVATION_METRICS:
                degradation = _relative_degradation(
                    float(baseline["means"][metric]), float(source["means"][metric])
                )
                degradations[metric] = degradation
                allowed = rules.recall_max_relative_degradation_percent if metric == "recall" else margin
                passes[metric] = degradation <= allowed + 1e-12
            sensitivity[_margin_key(margin)] = {
                "ap_f1_margin_percent": margin,
                "recall_margin_percent": rules.recall_max_relative_degradation_percent,
                "relative_degradation_percent": degradations,
                "metric_preserving": passes,
                "preservation_status": "preserving" if all(passes.values()) else "not_preserving",
            }
        record["sensitivity"] = sensitivity
        primary = sensitivity[_margin_key(5.0)]
        record["preservation_status"] = primary["preservation_status"]
        record["primary_relative_degradation_percent"] = primary[
            "relative_degradation_percent"
        ]
        evaluated[configuration_id] = record
    return evaluated


def _select_per_method(evaluated: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    methods: dict[str, list[Mapping[str, Any]]] = {}
    for record in evaluated.values():
        if record["selector_id"] != "none":
            methods.setdefault(str(record["selector_id"]), []).append(record)
    selected: dict[str, str] = {}
    for selector_id, records in sorted(methods.items()):
        if records[0]["count_strategy"] == "natural":
            if len(records) != 1:
                raise ValidationLockError(f"Natural method {selector_id} is not single.")
            chosen = records[0]
            rationale = "natural_single_configuration"
        else:
            preserving = [item for item in records if item["preservation_status"] == "preserving"]
            if preserving:
                chosen = sorted(preserving, key=_smallest_preserving_key)[0]
                rationale = "smallest_actual_feature_count_satisfying_primary_preservation"
            else:
                chosen = sorted(records, key=_validation_best_key)[0]
                rationale = "validation_best_no_preserving_configuration"
        configuration_id = str(chosen["configuration_id"])
        chosen["selection_rationale"] = rationale
        chosen["selection_status"] = chosen["preservation_status"]
        selected[selector_id] = configuration_id
    return selected


def _derive_roles(
    evaluated: Mapping[str, Mapping[str, Any]], per_method: Mapping[str, str]
) -> dict[str, str | None]:
    baseline = [key for key, value in evaluated.items() if value["selector_id"] == "none"]
    choices = [evaluated[configuration_id] for configuration_id in per_method.values()]
    unsupervised = [item for item in choices if item["selector_type"] == "unsupervised"]
    supervised = [item for item in choices if item["selector_type"] == "supervised"]
    preserving = [
        item
        for item in evaluated.values()
        if item["selector_id"] != "none" and item["preservation_status"] == "preserving"
    ]
    if len(baseline) != 1 or not unsupervised or not supervised:
        raise ValidationLockError("Baseline, unsupervised, and supervised roles must be derivable.")
    return {
        "full_baseline": baseline[0],
        "best_unsupervised": str(sorted(unsupervised, key=_validation_best_key)[0]["configuration_id"]),
        "best_supervised": str(sorted(supervised, key=_validation_best_key)[0]["configuration_id"]),
        "smallest_preserving": (
            str(sorted(preserving, key=_smallest_preserving_key)[0]["configuration_id"])
            if preserving
            else None
        ),
    }


def _paired_changes(core: pd.DataFrame) -> dict[str, dict[str, float]]:
    baseline = core.loc[core["selector_id"].eq("none")].set_index("seed")
    output: dict[str, dict[str, float]] = {}
    fields = {
        "average_precision_mean_difference": "average_precision",
        "f1_mean_difference": "f1",
        "recall_mean_difference": "recall",
        "model_size_bytes_mean_difference": "serialized_model_bytes",
    }
    for configuration_id, group in core.groupby("configuration_id", sort=True):
        changes = {name: [] for name in fields}
        for row in group.itertuples(index=False):
            anchor = baseline.loc[row.seed]
            for name, column in fields.items():
                changes[name].append(float(getattr(row, column)) - float(anchor[column]))
        output[configuration_id] = {name: float(np.mean(values)) for name, values in changes.items()}
    return output


def _stability_support(core: pd.DataFrame, sources: Mapping[str, Any]) -> dict[str, Any]:
    selected: dict[str, dict[str, Any]] = {}
    for configuration_id, group in core.groupby("configuration_id", sort=True):
        sets = [set(_parse_feature_list(value)) for value in group.sort_values("seed")["selected_features_json"]]
        values = [
            len(left & right) / len(left | right) if left | right else 1.0
            for index, left in enumerate(sets)
            for right in sets[index + 1 :]
        ]
        selected[configuration_id] = {
            "pairwise_jaccard_mean": float(np.mean(values)),
            "pairwise_jaccard_min": float(np.min(values)),
            "pairwise_jaccard_max": float(np.max(values)),
            "pair_count": len(values),
            "source": "recomputed_from_core_runs.csv",
        }

    stored = sources.get("selected_set_stability.csv")
    if isinstance(stored, pd.DataFrame) and not stored.empty:
        required = {"configuration_id", "comparison_type", "jaccard"}
        if not required.issubset(stored.columns):
            raise ValidationLockError("selected_set_stability.csv has an invalid schema.")
        for configuration_id, record in selected.items():
            rows = stored.loc[
                stored["configuration_id"].eq(configuration_id)
                & stored["comparison_type"].eq("summary")
            ]
            if len(rows) != 1:
                raise ValidationLockError(
                    f"Stored selected-set stability lacks one summary for {configuration_id}."
                )
            row = rows.iloc[0]
            if not math.isclose(float(row["jaccard"]), record["pairwise_jaccard_mean"], abs_tol=1e-12):
                raise ValidationLockError(
                    f"Stored selected-set stability disagrees for {configuration_id}."
                )
            record["source"] = "selected_set_stability.csv_verified_against_core"

    ranking = sources.get("ranking_stability.csv")
    deterministic = sources.get("deterministic_unsupervised_stability.csv")
    return {
        "selected_sets": selected,
        "ranking_by_selector": _records_by_key(ranking, "selector_id"),
        "deterministic_by_configuration": _records_by_key(
            deterministic, "configuration_id"
        ),
    }


def _verify_stored_analysis(
    sources: Mapping[str, Any],
    evaluated: Mapping[str, Mapping[str, Any]],
    per_method: Mapping[str, str],
    roles: Mapping[str, str | None],
    paired: Mapping[str, Mapping[str, float]],
) -> None:
    summary = sources.get("validation_summary.csv")
    if isinstance(summary, pd.DataFrame):
        for row in summary.itertuples(index=False):
            record = evaluated.get(str(row.configuration_id))
            if record is None:
                raise ValidationLockError("Stored validation summary has an unknown configuration.")
            for metric in _MEAN_METRICS:
                if not math.isclose(
                    float(getattr(row, f"{metric}_mean")),
                    float(record["means"][metric]),
                    abs_tol=1e-12,
                ):
                    raise ValidationLockError(
                        f"Stored validation summary disagrees for {row.configuration_id}."
                    )

    primary = sources.get("preservation_primary.csv")
    if isinstance(primary, pd.DataFrame):
        for row in primary.itertuples(index=False):
            record = evaluated.get(str(row.configuration_id))
            if record is None or str(row.preservation_status) != record["preservation_status"]:
                raise ValidationLockError(
                    f"Stored primary preservation disagrees for {row.configuration_id}."
                )

    stored_paired = sources.get("paired_baseline_comparisons.csv")
    if isinstance(stored_paired, pd.DataFrame):
        names = {
            "average_precision_difference_mean": "average_precision_mean_difference",
            "f1_difference_mean": "f1_mean_difference",
            "recall_difference_mean": "recall_mean_difference",
        }
        for row in stored_paired.itertuples(index=False):
            record = paired.get(str(row.configuration_id))
            if record is None:
                raise ValidationLockError("Stored paired comparison has an unknown configuration.")
            for stored_name, recomputed_name in names.items():
                if not math.isclose(
                    float(getattr(row, stored_name)),
                    float(record[recomputed_name]),
                    abs_tol=1e-12,
                ):
                    raise ValidationLockError(
                        f"Stored paired comparison disagrees for {row.configuration_id}."
                    )

    selection = sources.get("candidate_selection.json")
    if isinstance(selection, Mapping):
        expected_roles = {
            "full_baseline": roles["full_baseline"],
            "best_unsupervised": roles["best_unsupervised"],
            "best_supervised": roles["best_supervised"],
            "smallest_preserving": roles["smallest_preserving"],
        }
        for role, expected in expected_roles.items():
            stored_key = "baseline" if role == "full_baseline" else role
            stored = selection.get(stored_key)
            observed = stored.get("configuration_id") if isinstance(stored, Mapping) else None
            if observed != expected:
                raise ValidationLockError(f"Stored candidate selection disagrees for {role}.")
        stored_methods = {
            str(item["selector_id"]): str(item["configuration_id"])
            for item in selection.get("per_method", [])
        }
        if stored_methods != dict(per_method):
            raise ValidationLockError("Stored per-method candidate selections disagree.")


def _records_by_key(frame: Any, key: str) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    if key not in frame.columns:
        raise ValidationLockError(f"Stability support artifact is missing {key}.")
    output: dict[str, list[dict[str, Any]]] = {}
    for value, group in frame.groupby(key, dropna=False, sort=True):
        output[str(value)] = _json_safe(group.to_dict(orient="records"))
    return output


def _build_locked_configuration(
    configuration_id: str,
    core: pd.DataFrame,
    evaluated: Mapping[str, Mapping[str, Any]],
    paired: Mapping[str, Mapping[str, float]],
    stability: Mapping[str, Any],
    model_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    evaluation = _deep_copy_json(evaluated[configuration_id])
    rows = core.loc[core["configuration_id"].eq(configuration_id)].sort_values("seed")
    first = rows.iloc[0]
    selector_parameters = _normalized_selector_parameters(rows, configuration_id)
    seed_specific: dict[str, Any] = {}
    for row in rows.itertuples(index=False):
        seed_specific[str(int(row.seed))] = {
            "run_id": row.run_id,
            "selected_features": _parse_feature_list(row.selected_features_json),
            "selected_features_sha256": row.selected_features_sha256,
            "model_artifact": _deep_copy_json(model_records[row.run_id]),
        }
    selected_features = {
        seed: value["selected_features"] for seed, value in seed_specific.items()
    }
    selected_fingerprints = {
        seed: value["selected_features_sha256"]
        for seed, value in seed_specific.items()
    }
    reason = evaluation.get("selection_rationale", "required_role_configuration")
    resources = evaluation["resources"]
    paired_changes = _deep_copy_json(paired[configuration_id])
    stability_summary = {
        "selected_sets": stability["selected_sets"][configuration_id],
        "ranking": stability["ranking_by_selector"].get(evaluation["selector_id"], []),
        "deterministic_unsupervised": stability["deterministic_by_configuration"].get(
            configuration_id, []
        ),
    }
    return {
        "configuration_id": configuration_id,
        "selector_id": evaluation["selector_id"],
        "selector_method": str(first["selector_method"]),
        "selector_mode": str(first["selector_mode"]),
        "selector_parameters": selector_parameters,
        "selector_type": evaluation["selector_type"],
        "count_strategy": evaluation["count_strategy"],
        "requested_feature_count": evaluation["requested_feature_count"],
        "requested_K": evaluation["requested_feature_count"],
        "actual_feature_count": evaluation["actual_feature_count"],
        "selected_features": selected_features,
        "selected_feature_fingerprints": selected_fingerprints,
        "metrics": evaluation["means"],
        "validation_metrics": _validation_metric_names(evaluation["means"]),
        "primary_relative_degradation_percent": evaluation[
            "primary_relative_degradation_percent"
        ],
        "feature_reduction_percent": evaluation["feature_reduction_percent"],
        "feature_reduction_percentage": evaluation["feature_reduction_percent"],
        "preservation_status": evaluation["preservation_status"],
        "sensitivity": evaluation["sensitivity"],
        "paired_changes": paired_changes,
        "paired_changes_vs_baseline": paired_changes,
        "resources": resources,
        "resource_summary": resources,
        "stability": stability_summary,
        "stability_summary": stability_summary,
        "selection_status": evaluation.get("selection_status", evaluation["preservation_status"]),
        "reason": reason,
        "reason_for_lock": reason,
        "seed_specific": seed_specific,
    }


def _build_role(
    role: str,
    configuration_id: str | None,
    locked: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if configuration_id is None:
        return {
            "role": role,
            "locked_configuration_ref": None,
            "configuration_id": None,
            "selected_features": {},
            "selected_feature_fingerprints": {},
            "model_artifacts": {},
            "metrics": None,
            "paired_changes": None,
            "resources": None,
            "stability": None,
            "preservation_status": "not_available",
            "reason": "no primary-preserving nonbaseline configuration",
            "reason_for_lock": "no primary-preserving nonbaseline configuration",
        }
    configuration = locked[configuration_id]
    reasons = {
        "full_baseline": "full-feature validation baseline",
        "best_unsupervised": "best unsupervised per-method choice by frozen validation tie-break",
        "best_supervised": "best supervised per-method choice by frozen validation tie-break",
        "smallest_preserving": "fewest actual features among all primary-preserving nonbaseline configurations",
    }
    return {
        "role": role,
        "locked_configuration_ref": configuration_id,
        "configuration_id": configuration_id,
        "selector_id": configuration["selector_id"],
        "selector_method": configuration["selector_method"],
        "selector_mode": configuration["selector_mode"],
        "selector_parameters": configuration["selector_parameters"],
        "requested_K": configuration["requested_K"],
        "actual_feature_count": configuration["actual_feature_count"],
        "selected_features": {
            seed: value["selected_features"]
            for seed, value in configuration["seed_specific"].items()
        },
        "selected_feature_fingerprints": {
            seed: value["selected_features_sha256"]
            for seed, value in configuration["seed_specific"].items()
        },
        "selected_feature_fingerprint": {
            seed: value["selected_features_sha256"]
            for seed, value in configuration["seed_specific"].items()
        },
        "model_artifacts": {
            seed: value["model_artifact"]
            for seed, value in configuration["seed_specific"].items()
        },
        "metrics": configuration["metrics"],
        "validation_metrics": configuration["validation_metrics"],
        "paired_changes": configuration["paired_changes"],
        "paired_changes_vs_baseline": configuration["paired_changes_vs_baseline"],
        "feature_reduction_percentage": configuration[
            "feature_reduction_percentage"
        ],
        "resources": configuration["resources"],
        "resource_summary": configuration["resource_summary"],
        "stability": configuration["stability"],
        "stability_summary": configuration["stability_summary"],
        "preservation_status": configuration["preservation_status"],
        "reason": reasons[role],
        "reason_for_lock": reasons[role],
    }


def _validation_metric_names(means: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mean_average_precision": means["average_precision"],
        "mean_f1": means["f1"],
        "mean_recall": means["recall"],
        "mean_precision": means["precision"],
        "mean_roc_auc": means["roc_auc"],
    }


def _normalized_selector_parameters(
    rows: pd.DataFrame, configuration_id: str
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    uses_seed = False
    for row in rows.itertuples(index=False):
        try:
            parameters = json.loads(str(row.selector_parameters_json))
        except json.JSONDecodeError as exc:
            raise ValidationLockError(
                f"Selector parameters are invalid for {configuration_id}."
            ) from exc
        if not isinstance(parameters, dict):
            raise ValidationLockError(
                f"Selector parameters are not an object for {configuration_id}."
            )
        if "random_state" in parameters:
            uses_seed = True
            if int(parameters.pop("random_state")) != int(row.seed):
                raise ValidationLockError(
                    f"Selector random state does not match seed for {row.run_id}."
                )
        normalized.append(parameters)
    if any(parameters != normalized[0] for parameters in normalized[1:]):
        raise ValidationLockError(
            f"Configuration {configuration_id} changes selector parameters across seeds."
        )
    result = _deep_copy_json(normalized[0])
    if uses_seed:
        result["random_state_policy"] = "match_run_seed"
    return result


def _verify_lock_structure(lock: Mapping[str, Any]) -> None:
    required = {
        "stage", "protocol_version", "source_artifacts",
        "source_validation_artifacts", "source_artifact_hashes", "candidate_manifest",
        "candidate_manifest_hash", "seeds", "attack_configuration", "model_configuration",
        "threshold", "decision_threshold", "preservation_rules", "sensitivity_margins",
        "tie_breaks", "tie_break_rules", "evaluated_configurations",
        "per_method_choices", "locked_configurations", "roles", "governance",
    }
    missing = sorted(required - set(lock))
    if missing:
        raise ValidationLockError(f"Validation lock is missing fields: {missing}")
    semantic_hash = str(lock.get("semantic_payload_sha256", ""))
    if (
        lock.get("schema_version") != V06E_PROTOCOL_VERSION
        or lock["stage"] != V06E_STAGE
        or lock["protocol_version"] != V06E_PROTOCOL_VERSION
        or not isinstance(lock.get("created_at"), str)
        or not lock["created_at"]
        or lock.get("scope") != "stored_development_validation_artifacts_only"
        or len(semantic_hash) != 64
        or any(character not in "0123456789abcdef" for character in semantic_hash.lower())
    ):
        raise ValidationLockError("Validation lock stage or protocol version is invalid.")
    if lock.get("test_accessed") is not False or lock["seeds"] != list(V06E_SEEDS):
        raise ValidationLockError("Validation lock scope attestations or seeds are invalid.")
    manifest = lock["candidate_manifest"]
    if (
        not isinstance(manifest, Mapping)
        or lock["candidate_manifest_hash"] != manifest.get("sha256")
        or fingerprint_feature_names(manifest.get("features", [])) != manifest.get("sha256")
        or len(manifest.get("features", [])) != manifest.get("feature_count")
    ):
        raise ValidationLockError("Locked candidate manifest is invalid.")
    if set(lock["roles"]) != {
        "full_baseline", "best_unsupervised", "best_supervised", "smallest_preserving"
    }:
        raise ValidationLockError("Validation lock roles are incomplete.")
    if lock["source_artifact_hashes"] != {
        source["path"]: source["sha256"] for source in lock["source_artifacts"]
    }:
        raise ValidationLockError("Source artifact hash index is inconsistent.")
    if (
        lock["preservation_rules"] != PreservationRules().to_dict()
        or lock["sensitivity_margins"] != [0.0, 5.0, 10.0]
        or lock["tie_break_rules"] != lock["tie_breaks"]
        or float(lock["decision_threshold"]) != 0.5
    ):
        raise ValidationLockError("Validation lock changes frozen rules or threshold.")


def _verify_locked_fingerprints(lock: Mapping[str, Any]) -> None:
    expected_seeds = {str(seed) for seed in V06E_SEEDS}
    for configuration_id, configuration in lock["locked_configurations"].items():
        rows = configuration.get("seed_specific", {})
        if set(rows) != expected_seeds:
            raise ValidationLockError(f"Locked configuration {configuration_id} lacks five seeds.")
        expected_features = {
            seed: row["selected_features"] for seed, row in rows.items()
        }
        expected_hashes = {
            seed: row["selected_features_sha256"] for seed, row in rows.items()
        }
        if (
            configuration.get("selected_features") != expected_features
            or configuration.get("selected_feature_fingerprints") != expected_hashes
        ):
            raise ValidationLockError(
                f"Locked configuration {configuration_id} feature maps are inconsistent."
            )
        for seed, row in rows.items():
            if fingerprint_feature_names(row["selected_features"]) != row[
                "selected_features_sha256"
            ]:
                raise ValidationLockError(
                    f"Selected feature fingerprint mismatch in {configuration_id}, seed {seed}."
                )
    for role, details in lock["roles"].items():
        configuration_id = details.get("locked_configuration_ref")
        if configuration_id is None:
            if role != "smallest_preserving" or any(
                details.get(field)
                for field in (
                    "selected_features", "selected_feature_fingerprints", "model_artifacts"
                )
            ):
                raise ValidationLockError(f"Role {role} has an invalid empty assignment.")
            continue
        if configuration_id not in lock["locked_configurations"]:
            raise ValidationLockError(f"Role {role} references an unlocked configuration.")
        configuration = lock["locked_configurations"][configuration_id]
        expected_features = {
            seed: row["selected_features"] for seed, row in configuration["seed_specific"].items()
        }
        expected_hashes = {
            seed: row["selected_features_sha256"]
            for seed, row in configuration["seed_specific"].items()
        }
        expected_models = {
            seed: row["model_artifact"] for seed, row in configuration["seed_specific"].items()
        }
        if (
            details.get("configuration_id") != configuration_id
            or details.get("selector_id") != configuration["selector_id"]
            or details.get("selector_method") != configuration["selector_method"]
            or details.get("selector_mode") != configuration["selector_mode"]
            or details.get("selector_parameters") != configuration["selector_parameters"]
            or details.get("requested_K") != configuration["requested_K"]
            or details.get("actual_feature_count") != configuration["actual_feature_count"]
            or details.get("selected_features") != expected_features
            or details.get("selected_feature_fingerprints") != expected_hashes
            or details.get("selected_feature_fingerprint") != expected_hashes
            or details.get("model_artifacts") != expected_models
        ):
            raise ValidationLockError(f"Role {role} does not immutably mirror its configuration.")


def _verify_role_invariants(lock: Mapping[str, Any]) -> None:
    evaluated = lock["evaluated_configurations"]
    per_method = lock["per_method_choices"]
    expected = _derive_roles(evaluated, per_method)
    actual = {
        role: details.get("configuration_id") for role, details in lock["roles"].items()
    }
    if actual != expected:
        raise ValidationLockError("Locked role assignments violate frozen selection invariants.")
    for selector_id, configuration_id in per_method.items():
        record = evaluated.get(configuration_id)
        if not record or record.get("selector_id") != selector_id:
            raise ValidationLockError(f"Per-method choice for {selector_id} is invalid.")
        candidates = [value for value in evaluated.values() if value["selector_id"] == selector_id]
        if record["count_strategy"] == "natural":
            expected_id = candidates[0]["configuration_id"] if len(candidates) == 1 else None
        else:
            preserving = [item for item in candidates if item["preservation_status"] == "preserving"]
            pool = preserving or candidates
            key = _smallest_preserving_key if preserving else _validation_best_key
            expected_id = sorted(pool, key=key)[0]["configuration_id"]
        if configuration_id != expected_id:
            raise ValidationLockError(f"Per-method choice for {selector_id} violates tie-breaks.")


def _verify_persisted_sources(lock: Mapping[str, Any], source_dir: Path | None) -> None:
    for source in lock["source_artifacts"]:
        if not isinstance(source, Mapping) or source.get("kind") not in {
            "v06d_artifact", "v06d_support_artifact", "referenced_model_artifact"
        }:
            raise ValidationLockError("Validation lock contains an invalid source entry.")
        path = Path(str(source.get("path", "")))
        if not path.is_absolute() and source_dir is not None:
            direct = source_dir / path
            path = direct if direct.is_file() else source_dir / path.name
        if not path.is_file() or _sha256_file(path) != source.get("sha256"):
            raise ValidationLockError(f"Source artifact hash mismatch: {source.get('path')}")


def _verify_metadata_hash_claims(
    metadata: Mapping[str, Any], sources: Sequence[Mapping[str, Any]]
) -> None:
    claims = metadata.get("output_sha256")
    if not isinstance(claims, Mapping):
        raise ValidationLockError("V0.6-D output SHA-256 claims must be an object.")
    for source in sources:
        name = source["name"]
        if name == "run_metadata.json":
            continue
        if name not in claims:
            raise ValidationLockError(
                f"V0.6-D metadata lacks a hash claim for {name}."
            )
        if claims[name] != source["sha256"]:
            raise ValidationLockError(f"V0.6-D metadata hash claim mismatch for {name}.")


def _single_configuration(core: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for column in columns:
        values = core[column].drop_duplicates()
        if len(values) != 1:
            raise ValidationLockError(f"Core field {column} must be invariant across all runs.")
        output[column] = _json_safe(values.iloc[0])
    return output


def _model_configuration(core: pd.DataFrame) -> dict[str, Any]:
    result = _single_configuration(core, ("model_id", "model_parameters_json"))
    try:
        parameters = json.loads(result.pop("model_parameters_json"))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValidationLockError("model_parameters_json is invalid.") from exc
    result["parameters"] = parameters
    return result


def _reject_ambiguous_pr_auc(value: Any, path: str = "root") -> None:
    if isinstance(value, pd.DataFrame):
        if "pr_auc" in value.columns:
            raise ValidationLockError(f"Ambiguous exact column pr_auc found in {path}.")
        for column in value.columns:
            for index, item in value[column].items():
                if isinstance(item, str) and item.lstrip().startswith(("{", "[")):
                    try:
                        parsed = json.loads(item)
                    except json.JSONDecodeError:
                        continue
                    _reject_ambiguous_pr_auc(parsed, f"{path}.{column}[{index}]")
        return
    if isinstance(value, Mapping):
        if "pr_auc" in value:
            raise ValidationLockError(f"Ambiguous exact key pr_auc found in {path}.")
        for key, item in value.items():
            _reject_ambiguous_pr_auc(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_ambiguous_pr_auc(item, f"{path}[{index}]")


def fingerprint_feature_names(feature_names: Sequence[str]) -> str:
    """Exact local equivalent of the V0.6 ordered feature-name fingerprint."""
    frame = pd.DataFrame(
        {"position": np.arange(len(feature_names), dtype=int), "feature": list(feature_names)}
    )
    canonical = frame.copy().sort_index()
    return hashlib.sha256(
        canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _parse_feature_list(value: Any) -> list[str]:
    try:
        features = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ValidationLockError("Selected feature JSON is invalid.") from exc
    if not isinstance(features, list) or any(not isinstance(item, str) for item in features):
        raise ValidationLockError("Selected features must be a JSON list of strings.")
    return features


def _relative_degradation(baseline: float, candidate: float) -> float:
    if baseline == 0.0:
        if candidate >= 0.0:
            return 0.0
        raise ValidationLockError("A zero baseline cannot compare with a negative candidate.")
    return max(0.0, 100.0 * (baseline - candidate) / baseline)


def _margin_key(margin: float) -> str:
    return f"ap_f1_{int(margin)}_percent"


def _smallest_preserving_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        float(record["actual_feature_count"]),
        -float(record["means"]["average_precision"]),
        -float(record["means"]["f1"]),
        str(record["configuration_id"]),
    )


def _validation_best_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(record["means"]["average_precision"]),
        -float(record["means"]["f1"]),
        float(record["actual_feature_count"]),
        str(record["configuration_id"]),
    )


def _optional_int(value: Any) -> int | None:
    return None if pd.isna(value) else int(value)


def _deep_copy_json(value: Any) -> Any:
    return json.loads(json.dumps(_json_safe(value), allow_nan=False))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValidationLockError(f"Cannot hash source artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_text_atomic(path, text)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _render_report(lock: Mapping[str, Any]) -> str:
    lines = [
        "# V0.6-E Validation Configuration Lock",
        "",
        f"- Semantic payload SHA-256: `{lock['semantic_payload_sha256']}`",
        "- Scope: stored development-validation artifacts only",
        "- Test accessed: false",
        "- No selector fit, model fit, threshold tuning, inference, or final-test execution was performed by V0.6-E.",
        "- Ground truth is controlled experimental/synthetic cyber ground truth; it is not native DataCo cybersecurity labeling.",
        "- Timing, RSS, and serialized-size measurements are computational proxies, not direct energy measurements.",
        "",
        "## Locked Roles",
        "",
    ]
    for role, details in lock["roles"].items():
        if details["configuration_id"] is None:
            lines.extend(
                [
                    f"### {role}",
                    "",
                    "- Configuration: not available",
                    f"- Reason: {details['reason']}",
                    "",
                ]
            )
            continue
        metrics = details["metrics"]
        changes = details["paired_changes"]
        resources = details["resources"]
        configuration = lock["locked_configurations"][details["configuration_id"]]
        selected_stability = details["stability"]["selected_sets"]
        lines.extend(
            [
                f"### {role}",
                "",
                f"- Configuration: `{details['configuration_id']}`",
                f"- Selector: `{configuration['selector_id']}`",
                f"- Requested feature count: {configuration['requested_feature_count']}",
                f"- Reason: {details['reason']}",
                f"- Preservation: {details['preservation_status']}",
                f"- Mean actual feature count: {configuration['actual_feature_count']}",
                f"- Mean feature reduction: {_fmt(configuration['feature_reduction_percent'])}%",
                f"- Mean AP / F1 / recall: {_fmt(metrics['average_precision'])} / {_fmt(metrics['f1'])} / {_fmt(metrics['recall'])}",
                f"- Mean precision / ROC-AUC: {_fmt(metrics['precision'])} / {_fmt(metrics['roc_auc'])}",
                f"- Paired mean AP / F1 / recall differences: {_fmt(changes['average_precision_mean_difference'])} / {_fmt(changes['f1_mean_difference'])} / {_fmt(changes['recall_mean_difference'])}",
                f"- Mean model-size difference: {_fmt(changes['model_size_bytes_mean_difference'])} bytes",
                f"- Mean selector fit: {_fmt(resources['selector_fit_wall_time_sec'])} s",
                f"- Mean model fit: {_fmt(resources['model_fit_wall_time_sec'])} s",
                f"- Mean combined fit: {_fmt(resources['combined_train_wall_time_sec'])} s",
                f"- Mean validation inference: {_fmt(resources['inference_wall_time_sec'])} s",
                f"- Mean peak RSS: {_fmt(resources['peak_rss_mib'])} MiB",
                f"- Mean serialized model size: {_fmt(resources['model_size_bytes'])} bytes",
                f"- Selected-set Jaccard mean / min / max: {_fmt(selected_stability['pairwise_jaccard_mean'])} / {_fmt(selected_stability['pairwise_jaccard_min'])} / {_fmt(selected_stability['pairwise_jaccard_max'])}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: Any) -> str:
    return f"{float(value):.8g}"


def _build_audit(
    lock: Mapping[str, Any], verification: Mapping[str, Any], file_sha256: str
) -> dict[str, Any]:
    checks = [
        _audit_check("source_d_leakage_audit_pass", True, evidence_type="stored_attestation"),
        _audit_check("source_d_test_accessed_false", True, evidence_type="stored_attestation"),
        _audit_check("source_d_metadata_test_accessed_false", True, evidence_type="stored_attestation"),
        _audit_check("validation_only_configuration_selection", True, evidence_type="validated_core_splits"),
        _audit_check("no_test_metrics_available_to_locking_logic", True, evidence_type="artifact_only_input_schema"),
        _audit_check("artifact_only_architecture", True, evidence_type="module_capability"),
        _audit_check("exact_v06d_matrix_verified", True, evidence_type="recomputed_from_core_runs"),
        _audit_check("preservation_and_sensitivity_recomputed", True, evidence_type="recomputed_from_core_runs"),
        _audit_check("stored_validation_analysis_cross_checked", True, evidence_type="artifact_consistency"),
        _audit_check("per_seed_feature_fingerprints_verified", True, evidence_type="cryptographic"),
        _audit_check("resource_summaries_validated", True, evidence_type="finite_nonnegative_core_values"),
        _audit_check("preprocessing_refit_not_performed", True, evidence_type="artifact_only_architecture_and_operation"),
        _audit_check("selector_refit_not_performed", True, evidence_type="artifact_only_architecture_and_operation"),
        _audit_check("model_refit_not_performed", True, evidence_type="artifact_only_architecture_and_operation"),
        _audit_check("threshold_tuning_not_performed", True, evidence_type="frozen_rule_application"),
        _audit_check("attack_configuration_not_redefined", True, evidence_type="invariant_source_configuration"),
        _audit_check("operational_fraud_label_not_used_as_cyber_ground_truth", True, evidence_type="inherited_passing_v06d_audit"),
        _audit_check("attack_metadata_not_used_as_model_features", True, evidence_type="inherited_passing_v06d_audit"),
        _audit_check("no_post_validation_selection_from_test_information", True, evidence_type="artifact_only_architecture_and_source_attestations"),
        _audit_check("final_test_runner_not_imported_or_called", True, evidence_type="module_capability"),
        _audit_check(
            "runtime_test_loader_instrumentation_not_claimed",
            True,
            evidence_type="not_applicable",
            detail="No loader exists in this artifact-only module; false test access is an architecture/attestation guarantee, not a loader-hook measurement.",
        ),
        _audit_check("source_hashes_verified", verification["source_hashes_verified"], evidence_type="cryptographic"),
        _audit_check("source_semantics_reconstructed", verification["source_semantics_reconstructed"], evidence_type="cryptographic_reconstruction"),
        _audit_check("semantic_payload_verified", verification["status"] == "PASS", evidence_type="cryptographic"),
        _audit_check("sidecar_file_hash_verified", verification["sidecar_verified"], evidence_type="cryptographic"),
        _audit_check("post_write_verification", verification["status"] == "PASS", evidence_type="post_write"),
    ]
    return {
        "schema_version": V06E_PROTOCOL_VERSION,
        "stage": V06E_STAGE,
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "scope": "artifact_only_validation_configuration_locking",
        "test_accessed": False,
        "test_access_evidence": "artifact-only architecture plus inherited D attestations; no runtime test-loader instrumentation is present or needed",
        "semantic_payload_sha256": lock["semantic_payload_sha256"],
        "validation_lock_file_sha256": file_sha256,
        "checks": checks,
    }


def _audit_check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lock V0.6-D validation configurations from stored artifacts only."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--verify", action="store_true", help="Verify an existing lock only.")
    args = parser.parse_args(argv)
    if args.verify:
        result = verify_validation_lock(args.results_dir / LOCK_FILE_NAME, source_dir=args.results_dir)
    else:
        result = create_validation_lock(args.results_dir)
    print(f"V0.6-E {result['status']}: results={args.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
