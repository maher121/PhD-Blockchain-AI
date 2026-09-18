"""V0.9-F computational-resource efficiency evaluation of the locked BGWO winner.

V0.9-F reuses the accepted V0.8-E computational-resource protocol to measure the
frozen configurations K43, K42, MI-K11, the V0.8-C locked BPSO-K10 subset, and the
V0.9-D locked BGWO subset. It measures training and inference separately, records
fine-grained per-observation timestamps, compares the BGWO winner against the
references, imports frozen optimizer overhead and predictive evidence, and locks
its own results.

It never runs an optimizer or feature selector, never tunes DT/LR or thresholds,
never accesses the final test set, never estimates energy from CPU/wall time, and
never modifies the V0.9-D or V0.9-E locks.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from src.config import PROJECT_ROOT
from src.green.measurement import (
    CanonicalUnit,
    MeasurementPhase,
    MeasurementProvenance,
    WorkerStatus,
    collect_environment_metadata,
    dataframe_deep_memory_bytes,
    numpy_dense_nbytes,
    run_fresh_worker_protocol,
)
from src.lightweight.models import LightweightDetector, create_model
from src.security.experiment_data import fingerprint_feature_names
from src.security.ground_truth import assert_no_attack_metadata
import src.pipeline_v06g as v06g
import src.pipeline_v07b as v07b
import src.pipeline_v08b as v08b
import src.pipeline_v08d as v08d
import src.pipeline_v08e as v08e
import src.pipeline_v09d as v09d
import src.pipeline_v09e as v09e


V09F_STAGE = "V0.9-F"
V09F_SCHEMA_VERSION = "v0.9-f-resource-efficiency-1"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "resource_benchmark_v09f.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "bgwo" / "v09f"
DEFAULT_MODEL_DIR = DEFAULT_OUTPUT_DIR / "v09f_model_artifacts"
DEFAULT_CHECKPOINT_DIR = DEFAULT_OUTPUT_DIR / ".v09f_checkpoints"

CONFIGURATION_IDS = ("K43", "K42", "MI-K11", "BPSO-K10", "BGWO")
CONFIGURATION_SOURCE_IDS = {
    "K43": "none_natural",
    "K42": "pairwise_correlation_filter_natural",
    "MI-K11": "mutual_information_select_k_best_k11",
    "BPSO-K10": "v08c_locked_bpso_k10",
    "BGWO": "v09d_locked_bgwo_winner",
}
CLASSIFIERS = ("decision_tree", "logistic_regression")
SEEDS = (42, 43, 44, 45, 46)
PHASES = (MeasurementPhase.TRAINING.value, MeasurementPhase.INFERENCE.value)
REPETITIONS = tuple(range(1, 11))

TRAINING_INNER_OPERATION_COUNT = 1
INFERENCE_INNER_OPERATION_COUNT = 512
OUTER_REPETITIONS = 10
WARMUP_CALLS = 3

EXPECTED_TRAINING_OBSERVATIONS = len(CONFIGURATION_IDS) * len(CLASSIFIERS) * len(SEEDS) * OUTER_REPETITIONS
EXPECTED_INFERENCE_OBSERVATIONS = EXPECTED_TRAINING_OBSERVATIONS
EXPECTED_TOTAL_OBSERVATIONS = EXPECTED_TRAINING_OBSERVATIONS + EXPECTED_INFERENCE_OBSERVATIONS
EXPECTED_CELL_COUNT = len(CONFIGURATION_IDS) * len(CLASSIFIERS) * len(SEEDS) * len(PHASES)
EXPECTED_MODEL_ARTIFACT_COUNT = len(CONFIGURATION_IDS) * len(CLASSIFIERS) * len(SEEDS)

EXPECTED_STARTING_HEAD = "9c1b276"
EXPECTED_BGWO_WINNER = v09e.EXPECTED_BGWO_WINNER
EXPECTED_BPSO_WINNER = v09e.EXPECTED_BPSO_WINNER
EXPECTED_BGWO_FEATURES = v09e.EXPECTED_BGWO_FEATURES
EXPECTED_BPSO_FEATURES = v09e.EXPECTED_BPSO_FEATURES
EXPECTED_V09E_RESULT_LOCK_SHA256 = "8398f51ad4a17e559f741c1bc04690809903b7f688ab65fe4146fb1bc25d0556"
EXPECTED_V08E_RESULT_LOCK_SHA256 = "7307fda1cdb5f100cb6268dc0f4eff05b65bfcefadeb06707c8a3e13fd6888d5"
EXPECTED_V08D_RESULT_LOCK_SHA256 = v08e.EXPECTED_V08D_RESULT_LOCK_SHA256
T_CRITICAL_95_DF4 = v09e.T_CRITICAL_95_DF4

STANDBY_GAP_THRESHOLD_SEC = 1.0
DIRECT_ENERGY_STATUS = "DIRECT_ENERGY_UNAVAILABLE"

RESOURCE_PROTOCOL_PATH = PROJECT_ROOT / "config" / "resource_benchmark_v08e.yaml"
RESOURCE_ENGINE_PATH = PROJECT_ROOT / "src" / "pipeline_v08e.py"
V08E_RESULT_LOCK_PATH = PROJECT_ROOT / "results" / "bpso" / "v08e_e4_analysis" / "v08e_scientific_result_lock.json"
V09E_RESULT_LOCK_PATH = v09e.RESULT_LOCK_PATH
V09E_SUMMARY_PATH = v09e.SUMMARY_PATH
V09D_EXECUTION_PATH = PROJECT_ROOT / "results" / "bgwo" / "v09d" / "v09d_execution_summary.json"

V09F_IMMUTABLE_PATHS = tuple(
    dict.fromkeys(
        (
            *v09e.IMMUTABLE_PATHS,
            RESOURCE_PROTOCOL_PATH,
            RESOURCE_ENGINE_PATH,
            PROJECT_ROOT / "src" / "pipeline_v09e.py",
            V08E_RESULT_LOCK_PATH,
            V09E_RESULT_LOCK_PATH,
            V09E_SUMMARY_PATH,
            PROJECT_ROOT / "results" / "bgwo" / "v09e" / "v09e_test_access_audit.json",
        )
    )
)

PREFLIGHT_PATH = DEFAULT_OUTPUT_DIR / "v09f_preflight.json"
MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "v09f_run_manifest.json"
ENVIRONMENT_PATH = DEFAULT_OUTPUT_DIR / "v09f_environment.json"
SLEEP_PATH = DEFAULT_OUTPUT_DIR / "v09f_sleep_governance.json"
MODEL_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "v09f_model_artifact_manifest.json"
RAW_JSON_PATH = DEFAULT_OUTPUT_DIR / "v09f_raw_resource_observations.json"
RAW_CSV_PATH = DEFAULT_OUTPUT_DIR / "v09f_raw_resource_observations.csv"
FAILURES_JSON_PATH = DEFAULT_OUTPUT_DIR / "v09f_measurement_failures.json"
FAILURES_CSV_PATH = DEFAULT_OUTPUT_DIR / "v09f_measurement_failures.csv"
SUMMARY_JSON_PATH = DEFAULT_OUTPUT_DIR / "v09f_resource_summary.json"
SUMMARY_CSV_PATH = DEFAULT_OUTPUT_DIR / "v09f_resource_summary.csv"
COMPARISONS_JSON_PATH = DEFAULT_OUTPUT_DIR / "v09f_resource_comparisons.json"
COMPARISONS_CSV_PATH = DEFAULT_OUTPUT_DIR / "v09f_resource_comparisons.csv"
MODEL_SIZE_PATH = DEFAULT_OUTPUT_DIR / "v09f_model_size_comparison.json"
FEATURE_REDUCTION_PATH = DEFAULT_OUTPUT_DIR / "v09f_feature_reduction.json"
TIMING_QUALITY_PATH = DEFAULT_OUTPUT_DIR / "v09f_timing_quality.json"
OPTIMIZER_OVERHEAD_PATH = DEFAULT_OUTPUT_DIR / "v09f_optimizer_overhead.json"
BREAK_EVEN_PATH = DEFAULT_OUTPUT_DIR / "v09f_break_even.json"
TRADEOFF_PATH = DEFAULT_OUTPUT_DIR / "v09f_tradeoff.json"
EXECUTION_PATH = DEFAULT_OUTPUT_DIR / "v09f_execution_summary.json"
RESULT_LOCK_PATH = DEFAULT_OUTPUT_DIR / "v09f_result_lock.json"

SCIENTIFIC_METRICS = (
    "wall_time_sec",
    "process_cpu_time_sec",
    "absolute_peak_rss_mib",
    "incremental_peak_rss_mib",
    "inference_latency_sec",
    "per_record_latency_sec",
    "throughput_records_sec",
)

METRIC_UNITS = {
    "wall_time_sec": "seconds",
    "process_cpu_time_sec": "seconds",
    "absolute_peak_rss_mib": "MiB",
    "incremental_peak_rss_mib": "MiB",
    "inference_latency_sec": "seconds",
    "per_record_latency_sec": "seconds",
    "throughput_records_sec": "records/second",
    "serialized_model_bytes": "bytes",
    "input_dataframe_memory_bytes": "bytes",
    "numpy_dense_nbytes": "bytes",
    "feature_count": "features",
    "wall_clock_elapsed_sec": "seconds",
    "monotonic_elapsed_sec": "seconds",
    "standby_gap_sec": "seconds",
}

PRIMARY_PAIRS = (("BGWO", "K43"), ("BGWO", "BPSO-K10"))
SECONDARY_PAIRS = (("BGWO", "K42"), ("BGWO", "MI-K11"))

OBSERVATION_FIELDS = (
    "observation_id",
    "schema_version",
    "stage",
    "environment_id",
    "classifier",
    "configuration_id",
    "source_configuration_id",
    "seed",
    "phase",
    "outer_repetition",
    "inner_operation_count",
    "measured_operation_count",
    "record_count",
    "measured_record_count",
    "feature_count",
    "warmup_calls",
    "worker_pid",
    "wall_time_sec",
    "process_cpu_time_sec",
    "start_rss_mib",
    "end_rss_mib",
    "absolute_peak_rss_mib",
    "incremental_peak_rss_mib",
    "input_dataframe_memory_bytes",
    "numpy_dense_nbytes",
    "serialized_model_bytes",
    "inference_latency_sec",
    "per_record_latency_sec",
    "throughput_records_sec",
    "time_unit",
    "rss_unit",
    "size_unit",
    "throughput_unit",
    "measurement_scope",
    "measurement_provenance",
    "derived_measurement_provenance",
    "workload_sha256",
    "model_sha256",
    "configuration_features_sha256",
    "v06_lock_sha256",
    "v08c_semantic_lock_sha256",
    "v09d_winner_semantic_lock_sha256",
    "v09e_result_semantic_lock_sha256",
    "status",
    "failure_type",
    "failure_stage",
    "failure_reason",
    "observation_started_at_utc",
    "observation_finished_at_utc",
    "wall_clock_elapsed_sec",
    "monotonic_elapsed_sec",
    "standby_gap_sec",
)

FAILURE_FIELDS = v08e.FAILURE_FIELDS

TIMESTAMP_FIELDS = (
    "observation_started_at_utc",
    "observation_finished_at_utc",
    "wall_clock_elapsed_sec",
    "monotonic_elapsed_sec",
    "standby_gap_sec",
)


class V09FError(RuntimeError):
    """Raised when V0.9-F resource-governance or integrity checks fail closed."""


class V09FCheckpointError(V09FError):
    """Raised when a benchmark checkpoint is incompatible with the protocol."""


def current_head_short(root: Path = PROJECT_ROOT) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _utc_now() -> str:
    return v08e._utc_now()


def _json_sha256(payload: Any) -> str:
    return v08e._json_sha256(payload)


def _read_json(path: Path | str) -> dict[str, Any]:
    return v08e._read_json(path)


def _atomic_write_json(path: Path | str, payload: Any) -> None:
    v08e._atomic_write_json(path, payload)


def _atomic_write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str]) -> None:
    v08e._atomic_write_csv(path, rows, fieldnames=fieldnames)


def sha256_file(path: Path | str) -> str:
    return v08e.sha256_file(path)


def canonical_lf_sha256(path: Path | str) -> str:
    return v08e.canonical_lf_sha256(path)


def load_v09f_protocol(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the frozen V0.9-F resource protocol configuration."""

    config_path = Path(path)
    try:
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, Exception) as exc:  # noqa: BLE001 - fail closed on any config error
        if isinstance(exc, V09FError):
            raise
        raise V09FError(f"Cannot load V0.9-F config: {config_path}") from exc
    if not isinstance(payload, dict):
        raise V09FError("V0.9-F config must be a YAML mapping.")
    measurement = payload.get("measurement", {})
    required = {
        "version": (payload.get("version"), V09F_SCHEMA_VERSION),
        "stage": (payload.get("stage"), V09F_STAGE),
        "configurations": (tuple(payload.get("configurations", ())), CONFIGURATION_IDS),
        "classifiers": (tuple(payload.get("classifiers", ())), CLASSIFIERS),
        "seeds": (tuple(payload.get("seeds", ())), SEEDS),
        "phases": (tuple(payload.get("phases", ())), PHASES),
        "outer_repetitions": (int(measurement.get("outer_repetitions", -1)), OUTER_REPETITIONS),
        "warmup_calls": (int(measurement.get("warmup_calls", -1)), WARMUP_CALLS),
        "training_inner_operation_count": (
            int(measurement.get("training_inner_operation_count", -1)),
            TRAINING_INNER_OPERATION_COUNT,
        ),
        "inference_inner_operation_count": (
            int(measurement.get("inference_inner_operation_count", -1)),
            INFERENCE_INNER_OPERATION_COUNT,
        ),
        "worker_policy": (measurement.get("worker_policy"), "fresh_process_per_outer_repetition"),
        "adaptive_calibration": (measurement.get("adaptive_calibration"), False),
        "outlier_deletion": (measurement.get("outlier_deletion"), "prohibited"),
        "per_observation_timestamps": (measurement.get("per_observation_timestamps"), "required"),
    }
    drift = {
        key: {"observed": observed, "expected": expected}
        for key, (observed, expected) in required.items()
        if observed != expected
    }
    if drift:
        raise V09FError(f"V0.9-F protocol drift detected: {drift}")
    return payload


def snapshot_immutable_paths(paths: Sequence[Path] = V09F_IMMUTABLE_PATHS) -> dict[str, str]:
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise V09FError(f"Missing frozen V0.6-V0.9-E prerequisite artifacts: {missing}")
    return {str(Path(path).resolve()): sha256_file(path) for path in paths}


def verify_v09f_preflight(
    *,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    expected_head: str | None = EXPECTED_STARTING_HEAD,
    basis: v08b.FrozenBasis | None = None,
    bgwo_lock: Mapping[str, Any] | None = None,
    bpso_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify every frozen scientific identity before any resource measurement."""

    observed_head = current_head_short(PROJECT_ROOT)
    if expected_head and observed_head != expected_head:
        raise V09FError(f"V09F_NO_GO: expected HEAD {expected_head}, observed {observed_head}.")

    protocol = load_v09f_protocol(config_path)
    resolved_basis = basis if basis is not None else v08b.load_frozen_basis()
    resolved_bgwo = dict(bgwo_lock) if bgwo_lock is not None else v09e.load_verified_bgwo_winner_lock()
    resolved_bpso = dict(bpso_lock) if bpso_lock is not None else v09e.load_verified_bpso_winner_lock()
    v09d.verify_winner_lock(resolved_bgwo)
    v08d.verify_winner_lock(resolved_bpso)

    roles = resolved_basis.lock.get("roles", {})
    k11_by_seed = roles.get("best_supervised", {}).get("selected_features", {})
    baseline_metrics = roles.get("full_baseline", {}).get("metrics", {})
    preprocessing = resolved_basis.dataset_metadata.get("preprocessing", {})
    dataset_split = resolved_basis.dataset_metadata.get("split", {})

    checks = {
        "starting_checkpoint": observed_head == EXPECTED_STARTING_HEAD,
        "bgwo_winner_lock_verified": resolved_bgwo.get("status") == "VALIDATION_LOCKED"
        and resolved_bgwo.get("eligible_for_v09e") is True,
        "bgwo_winner_exact": resolved_bgwo.get("mask_sha256") == EXPECTED_BGWO_WINNER["mask_sha256"]
        and resolved_bgwo.get("selected_features_sha256") == EXPECTED_BGWO_WINNER["selected_features_sha256"]
        and int(resolved_bgwo.get("selected_feature_count", -1)) == EXPECTED_BGWO_WINNER["selected_feature_count"],
        "bgwo_winner_validation_only_provenance": resolved_bgwo.get("selection_scope")
        == "TRAIN_AND_DEVELOPMENT_VALIDATION_ONLY"
        and resolved_bgwo.get("final_test_accessed") is False,
        "bgwo_winner_unchanged_by_v09e": True,
        "bpso_winner_lock_verified": resolved_bpso.get("status") == "VALIDATION_LOCKED",
        "bpso_winner_exact": resolved_bpso.get("mask_sha256") == EXPECTED_BPSO_WINNER["mask_sha256"]
        and resolved_bpso.get("selected_features_sha256") == EXPECTED_BPSO_WINNER["selected_features_sha256"],
        "v06_validation_lock_verified": resolved_basis.lock.get("test_accessed") is False,
        "mi_k11_governed_subset": int(roles.get("best_supervised", {}).get("actual_feature_count", -1)) == 11
        and set(k11_by_seed) == {str(seed) for seed in SEEDS},
        "k42_governed_subset": len(resolved_basis.k42_features) == 42 and int(resolved_basis.k42_mask.sum()) == 42,
        "k43_baseline": int(roles.get("full_baseline", {}).get("actual_feature_count", -1)) == 43
        and set(baseline_metrics) >= {"average_precision", "f1", "precision", "recall", "roc_auc"},
        "dataset_identity": dataset_split.get("seed") == 42
        and dataset_split.get("strategy") == "order_grouped"
        and dataset_split.get("sizes") == v09e.EXPECTED_TEST_SPLIT,
        "feature_ordering": tuple(preprocessing.get("ml_feature_columns", ()))
        == tuple(resolved_basis.candidate_features)
        == tuple(resolved_basis.lock.get("candidate_manifest", {}).get("features", ())),
        "preprocessing_artifacts": preprocessing.get("fitted_on_rows") == 28000
        and resolved_basis.dataset_metadata.get("features", {}).get("cybersecurity_labels") is False,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise V09FError(f"V0.9-F preflight failed: {failed}")

    configurations = v09e.build_configuration_plan(resolved_basis, resolved_bgwo, resolved_bpso)
    classifiers = v09e.classifier_plan()

    v09e_lock = _read_json(V09E_RESULT_LOCK_PATH)
    v09e.verify_result_lock(v09e_lock)
    v08e_lock = _read_json(V08E_RESULT_LOCK_PATH)
    if v08e_lock.get("semantic_result_lock_sha256") != EXPECTED_V08E_RESULT_LOCK_SHA256:
        raise V09FError("V08E_NO_GO: frozen V0.8-E resource result lock hash differs.")
    if v09e_lock.get("semantic_result_lock_sha256") != EXPECTED_V09E_RESULT_LOCK_SHA256:
        raise V09FError("V09F_NO_GO: frozen V0.9-E final-test result lock hash differs.")

    energy = v08e.resolve_energy_capability()
    if energy["decision"] != "DIRECT_ENERGY_UNAVAILABLE":
        raise V09FError(
            "V09F_NO_GO: a direct energy backend is reported available, but V0.9-F only "
            "implements computational proxies and must not silently ignore physical energy."
        )

    immutable = snapshot_immutable_paths()
    return {
        "schema_version": V09F_SCHEMA_VERSION,
        "stage": V09F_STAGE,
        "status": "PASS",
        "starting_head": observed_head,
        "checks": checks,
        "protocol_config_path": str(Path(config_path).resolve()),
        "protocol_config_lf_sha256": canonical_lf_sha256(config_path),
        "resource_protocol_identity": {
            "source_stage": "V0.8-E2",
            "protocol_config_path": str(RESOURCE_PROTOCOL_PATH),
            "protocol_config_lf_sha256": canonical_lf_sha256(RESOURCE_PROTOCOL_PATH),
            "resource_engine_path": str(RESOURCE_ENGINE_PATH),
            "resource_engine_sha256": sha256_file(RESOURCE_ENGINE_PATH),
            "v08e_semantic_result_lock_sha256": EXPECTED_V08E_RESULT_LOCK_SHA256,
        },
        "bgwo_winner_lock_path": str(v09e.BGWO_WINNER_LOCK_PATH),
        "bgwo_winner_lock_sha256": sha256_file(v09e.BGWO_WINNER_LOCK_PATH),
        "bgwo_winner_semantic_lock_sha256": resolved_bgwo["semantic_lock_sha256"],
        "bgwo_winner_mask_sha256": resolved_bgwo["mask_sha256"],
        "bgwo_winner_selected_features_sha256": resolved_bgwo["selected_features_sha256"],
        "bgwo_winner_selected_feature_count": int(resolved_bgwo["selected_feature_count"]),
        "bpso_winner_semantic_lock_sha256": resolved_bpso["semantic_lock_sha256"],
        "bpso_winner_mask_sha256": resolved_bpso["mask_sha256"],
        "bpso_winner_selected_features_sha256": resolved_bpso["selected_features_sha256"],
        "bpso_winner_selected_feature_count": int(resolved_bpso["selected_feature_count"]),
        "v09e_result_lock_sha256": sha256_file(V09E_RESULT_LOCK_PATH),
        "v09e_result_semantic_lock_sha256": v09e_lock["semantic_result_lock_sha256"],
        "dataset_identity": {
            "name": "DataCo SMART Supply Chain",
            "split_seed": dataset_split.get("seed"),
            "split_sizes": dict(dataset_split.get("sizes", {})),
            "test_rows": int(dataset_split.get("sizes", {}).get("test", 0)),
            "dimensions": int(len(resolved_basis.candidate_features)),
            "feature_manifest_sha256": resolved_basis.candidate_manifest_sha256,
        },
        "direct_energy_status": energy["decision"],
        "energy_capability": energy,
        "configuration_plan": list(configurations),
        "classifier_plan": classifiers,
        "immutable_hashes": immutable,
    }


def _protocol_payload(preconditions: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": V09F_STAGE,
        "schema_version": V09F_SCHEMA_VERSION,
        "configurations": list(CONFIGURATION_IDS),
        "classifiers": list(CLASSIFIERS),
        "seeds": list(SEEDS),
        "phases": list(PHASES),
        "outer_repetitions": OUTER_REPETITIONS,
        "warmup_calls": WARMUP_CALLS,
        "training_inner_operation_count": TRAINING_INNER_OPERATION_COUNT,
        "inference_inner_operation_count": INFERENCE_INNER_OPERATION_COUNT,
        "adaptive_calibration": False,
        "worker_policy": "fresh_process_per_outer_repetition",
        "training_boundary": "model.fit_only_data_loading_selection_serialization_excluded",
        "inference_boundary": "prediction_only_model_loading_warmup_excluded",
        "per_observation_timestamps": True,
        "standby_gap_threshold_sec": STANDBY_GAP_THRESHOLD_SEC,
        "direct_energy_decision": DIRECT_ENERGY_STATUS,
        "resource_protocol_source": "V0.8-E2",
        "v09d_semantic_lock_sha256": preconditions["bgwo_winner_semantic_lock_sha256"],
        "v09e_semantic_result_lock_sha256": preconditions["v09e_result_semantic_lock_sha256"],
    }


def protocol_sha256(preconditions: Mapping[str, Any]) -> str:
    return _json_sha256(_protocol_payload(preconditions))


def build_run_manifest(preconditions: Mapping[str, Any], environment_id: str) -> dict[str, Any]:
    manifest = {
        "schema_version": V09F_SCHEMA_VERSION,
        "stage": V09F_STAGE,
        "status": "PLANNED",
        "configurations": list(CONFIGURATION_IDS),
        "classifiers": list(CLASSIFIERS),
        "seeds": list(SEEDS),
        "phases": list(PHASES),
        "outer_repetitions": OUTER_REPETITIONS,
        "warmup_calls": WARMUP_CALLS,
        "training_inner_operation_count": TRAINING_INNER_OPERATION_COUNT,
        "inference_inner_operation_count": INFERENCE_INNER_OPERATION_COUNT,
        "adaptive_calibration": False,
        "expected_training_observations": EXPECTED_TRAINING_OBSERVATIONS,
        "expected_inference_observations": EXPECTED_INFERENCE_OBSERVATIONS,
        "expected_total_observations": EXPECTED_TOTAL_OBSERVATIONS,
        "expected_cell_count": EXPECTED_CELL_COUNT,
        "expected_model_artifact_count": EXPECTED_MODEL_ARTIFACT_COUNT,
        "environment_id": environment_id,
        "protocol_sha256": protocol_sha256(preconditions),
        "provenance_labels": [
            MeasurementProvenance.DIRECT_COMPUTATIONAL.value,
            MeasurementProvenance.DERIVED.value,
            MeasurementProvenance.IMPORTED_HISTORICAL.value,
            MeasurementProvenance.DIRECT_ENERGY.value,
            MeasurementProvenance.ESTIMATED.value,
        ],
    }
    validate_run_manifest(manifest)
    return manifest


def validate_run_manifest(manifest: Mapping[str, Any]) -> None:
    if (
        tuple(manifest.get("configurations", ())) != CONFIGURATION_IDS
        or tuple(manifest.get("classifiers", ())) != CLASSIFIERS
        or tuple(manifest.get("seeds", ())) != SEEDS
        or tuple(manifest.get("phases", ())) != PHASES
        or int(manifest.get("outer_repetitions", -1)) != OUTER_REPETITIONS
        or int(manifest.get("warmup_calls", -1)) != WARMUP_CALLS
        or int(manifest.get("training_inner_operation_count", -1)) != TRAINING_INNER_OPERATION_COUNT
        or int(manifest.get("inference_inner_operation_count", -1)) != INFERENCE_INNER_OPERATION_COUNT
        or manifest.get("adaptive_calibration") is not False
        or int(manifest.get("expected_total_observations", -1)) != EXPECTED_TOTAL_OBSERVATIONS
    ):
        raise V09FError("Run manifest does not match the frozen 1000-observation design.")


def build_observation_schedule(
    *,
    configurations: Sequence[str] = CONFIGURATION_IDS,
    classifiers: Sequence[str] = CLASSIFIERS,
    seeds: Sequence[int] = SEEDS,
    phases: Sequence[str] = PHASES,
    outer_repetitions: int = OUTER_REPETITIONS,
) -> tuple[v08e.PlannedObservation, ...]:
    if tuple(configurations) != CONFIGURATION_IDS:
        raise V09FError("Schedule configurations must be exactly K43/K42/MI-K11/BPSO-K10/BGWO.")
    if tuple(classifiers) != CLASSIFIERS or tuple(seeds) != SEEDS:
        raise V09FError("Schedule classifiers/seeds drift from the frozen protocol.")
    if tuple(phases) != PHASES or outer_repetitions != OUTER_REPETITIONS:
        raise V09FError("Schedule phases/repetitions drift from the frozen protocol.")

    rows: list[v08e.PlannedObservation] = []
    for phase_index, phase in enumerate(phases):
        for seed_index, seed in enumerate(seeds):
            for classifier_index, classifier in enumerate(classifiers):
                rotation = (phase_index + seed_index + classifier_index) % len(configurations)
                ordered = tuple(configurations[rotation:]) + tuple(configurations[:rotation])
                for configuration_id in ordered:
                    inner_count = (
                        TRAINING_INNER_OPERATION_COUNT
                        if phase == MeasurementPhase.TRAINING.value
                        else INFERENCE_INNER_OPERATION_COUNT
                    )
                    for repetition in REPETITIONS:
                        rows.append(
                            v08e.PlannedObservation(
                                observation_id=(
                                    f"v09f__{phase}__{classifier}__{configuration_id}"
                                    f"__s{seed}__r{repetition:02d}"
                                ),
                                classifier=classifier,
                                configuration_id=configuration_id,
                                source_configuration_id=CONFIGURATION_SOURCE_IDS[configuration_id],
                                seed=seed,
                                phase=phase,
                                outer_repetition=repetition,
                                inner_operation_count=inner_count,
                            )
                        )
    _validate_schedule(rows)
    return tuple(rows)


def _validate_schedule(schedule: Sequence[v08e.PlannedObservation]) -> None:
    training = sum(item.phase == MeasurementPhase.TRAINING.value for item in schedule)
    inference = sum(item.phase == MeasurementPhase.INFERENCE.value for item in schedule)
    if (training, inference, len(schedule)) != (
        EXPECTED_TRAINING_OBSERVATIONS,
        EXPECTED_INFERENCE_OBSERVATIONS,
        EXPECTED_TOTAL_OBSERVATIONS,
    ):
        raise V09FError("Schedule must contain exactly 500 training and 500 inference observations.")
    ids = [item.observation_id for item in schedule]
    if len(set(ids)) != len(ids):
        raise V09FError("Planned observation IDs must be unique.")


def schedule_cells(schedule: Sequence[v08e.PlannedObservation]) -> tuple[v08e.BenchmarkCell, ...]:
    cells: list[v08e.BenchmarkCell] = []
    for index in range(0, len(schedule), OUTER_REPETITIONS):
        block = schedule[index : index + OUTER_REPETITIONS]
        if len(block) != OUTER_REPETITIONS:
            raise V09FError("Schedule includes an incomplete benchmark cell block.")
        identity = {
            (
                row.classifier,
                row.configuration_id,
                row.source_configuration_id,
                row.seed,
                row.phase,
                row.inner_operation_count,
            )
            for row in block
        }
        repetitions = tuple(row.outer_repetition for row in block)
        if len(identity) != 1 or repetitions != REPETITIONS:
            raise V09FError("Each benchmark cell must contain repetitions 1-10 once.")
        cells.append(v08e.BenchmarkCell(*next(iter(identity))))
    if len(cells) != EXPECTED_CELL_COUNT:
        raise V09FError(f"V0.9-F requires exactly {EXPECTED_CELL_COUNT} benchmark cells.")
    return tuple(cells)


def build_features_only_inference_workloads(
    workloads: v07b.PreparedBenchmarkWorkloads,
    *,
    candidate_features: Sequence[str],
) -> dict[int, v08e.InferenceFeatureWorkload]:
    candidates = tuple(candidate_features)
    output: dict[int, v08e.InferenceFeatureWorkload] = {}
    for seed in SEEDS:
        manifestation = workloads.get(seed, MeasurementPhase.INFERENCE.value)
        features = manifestation.features.loc[:, ["row_id", *candidates]].copy()
        assert_no_attack_metadata(features)
        output[seed] = v08e.InferenceFeatureWorkload(
            seed=seed,
            features=features,
            workload_sha256=workloads.workload_hashes[(seed, MeasurementPhase.INFERENCE.value)],
            row_ids_sha256=manifestation.row_ids_sha256,
            feature_matrix_sha256=manifestation.attacked_features_sha256,
        )
    return output


def prepare_inference_model_artifacts(
    *,
    output_dir: Path | str,
    configuration_plan: Sequence[Mapping[str, Any]],
    workloads: v07b.PreparedBenchmarkWorkloads,
    model_factory: Callable[..., LightweightDetector] = create_model,
) -> dict[tuple[str, str, int], v08e.PreparedModelArtifact]:
    """Prepare 50 train-only model artifacts outside benchmark timing boundaries."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    by_key: dict[tuple[str, str, int], v08e.PreparedModelArtifact] = {}
    for plan in configuration_plan:
        configuration_id = str(plan["configuration_id"])
        source_configuration_id = str(plan["source_configuration_id"])
        for classifier in CLASSIFIERS:
            parameters = dict(v07b.DT_PARAMETERS if classifier == "decision_tree" else v07b.LR_PARAMETERS)
            for seed in SEEDS:
                features = tuple(plan["features_by_seed"][str(seed)])
                assert_no_attack_metadata(features)
                training = workloads.get(seed, MeasurementPhase.TRAINING.value)
                selected_train = training.features.loc[:, list(features)].copy()
                model_path = destination / f"{configuration_id}__{classifier}__s{seed}.joblib"
                if model_path.is_file():
                    model = LightweightDetector.load(model_path)
                else:
                    model = model_factory(classifier, features, parameters, random_state=seed)
                    model.fit(selected_train, training.labels.copy())
                    model.save(model_path)
                if (
                    model.model_name != classifier
                    or model.random_state != seed
                    or tuple(model.feature_names) != features
                    or dict(model.parameters) != parameters
                ):
                    raise V09FError("Prepared inference model artifact identity verification failed.")
                by_key[(configuration_id, classifier, seed)] = v08e.PreparedModelArtifact(
                    classifier=classifier,
                    configuration_id=configuration_id,
                    source_configuration_id=source_configuration_id,
                    seed=seed,
                    path=model_path,
                    file_sha256=sha256_file(model_path),
                    model_state_sha256=v06g.model_state_sha256(model),
                    serialized_model_bytes=model_path.stat().st_size,
                    feature_names=features,
                    feature_manifest_sha256=fingerprint_feature_names(features),
                )
    if len(by_key) != EXPECTED_MODEL_ARTIFACT_COUNT:
        raise V09FError(
            f"Model-artifact preparation must produce exactly {EXPECTED_MODEL_ARTIFACT_COUNT} artifacts."
        )
    return by_key


def _elapsed_and_gap(wall_start: float, wall_end: float, mono_start: float, mono_end: float) -> tuple[float, float, float]:
    wall_elapsed = max(0.0, float(wall_end) - float(wall_start))
    mono_elapsed = max(0.0, float(mono_end) - float(mono_start))
    return wall_elapsed, mono_elapsed, max(0.0, wall_elapsed - mono_elapsed)


def _run_cell_measurement(
    *,
    context: v07b.V07BContext,
    cell: v08e.BenchmarkCell,
    configuration_plan: Mapping[str, Any],
    model_artifact: v08e.PreparedModelArtifact,
    workloads: v07b.PreparedBenchmarkWorkloads,
    inference_workload: v08e.InferenceFeatureWorkload,
    environment_id: str,
    v08c_semantic_lock_sha256: str,
    v09d_semantic_lock_sha256: str,
    v09e_semantic_result_lock_sha256: str,
    runner: Callable[..., Any] = run_fresh_worker_protocol,
    clock: Callable[[], float] = time.time,
    perf_clock: Callable[[], float] = time.perf_counter,
    timestamp_fn: Callable[[], str] = _utc_now,
    start_method: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Measure one complete benchmark cell (10 timestamped repetitions)."""

    features = tuple(configuration_plan["features_by_seed"][str(cell.seed)])
    assert_no_attack_metadata(features)
    if cell.phase == MeasurementPhase.TRAINING.value:
        manifestation = workloads.get(cell.seed, MeasurementPhase.TRAINING.value)
        selected = manifestation.features.loc[:, list(features)].copy()
        records = len(selected)
        prepare = v07b._TrainingPreparation(
            cell.classifier,
            features,
            dict(v07b.DT_PARAMETERS if cell.classifier == "decision_tree" else v07b.LR_PARAMETERS),
            cell.seed,
            selected,
            manifestation.labels.copy(),
        )
        operation = v07b._training_fit_only
        scope = "model.fit_only_data_loading_selection_serialization_excluded"
        workload_sha256 = workloads.workload_hashes[(cell.seed, MeasurementPhase.TRAINING.value)]
    else:
        selected = inference_workload.features.loc[:, list(features)].copy()
        records = len(selected)
        prepare = v07b._InferencePreparation(model_artifact, selected)
        operation = v07b._prediction_only
        scope = "prediction_only_model_loading_warmup_excluded"
        workload_sha256 = inference_workload.workload_sha256
    selected_bytes = dataframe_deep_memory_bytes(selected, include_index=True)
    dense_bytes = numpy_dense_nbytes(selected.to_numpy(dtype=float))

    rows: list[dict[str, Any]] = []
    for repetition in REPETITIONS:
        started_at = timestamp_fn()
        wall_start = clock()
        mono_start = perf_clock()
        run = runner(
            operation,
            prepare=prepare,
            inner_operation_count=cell.inner_operation_count,
            phase=cell.phase,
            warmup_calls=WARMUP_CALLS,
            outer_repetitions=1,
            records_per_operation=records,
            timeout_seconds=300.0,
            measurement_scope=scope,
            start_method=start_method,
        )
        mono_end = perf_clock()
        wall_end = clock()
        finished_at = timestamp_fn()
        results = tuple(getattr(run, "results", ()))
        if len(results) != 1:
            raise V09FError("Each fresh-worker repetition must return exactly one result.")
        result = results[0]
        wall_elapsed, mono_elapsed, standby_gap = _elapsed_and_gap(
            wall_start, wall_end, mono_start, mono_end
        )
        success = result.status is WorkerStatus.COMPLETED and result.observation is not None
        observation = result.observation
        row = {
            "observation_id": (
                f"v09f__{cell.phase}__{cell.classifier}__{cell.configuration_id}"
                f"__s{cell.seed}__r{repetition:02d}"
            ),
            "schema_version": V09F_SCHEMA_VERSION,
            "stage": V09F_STAGE,
            "environment_id": environment_id,
            "classifier": cell.classifier,
            "configuration_id": cell.configuration_id,
            "source_configuration_id": cell.source_configuration_id,
            "seed": cell.seed,
            "phase": cell.phase,
            "outer_repetition": repetition,
            "inner_operation_count": cell.inner_operation_count,
            "measured_operation_count": result.measured_operation_count,
            "record_count": records,
            "measured_record_count": result.measured_operation_count * records,
            "feature_count": len(features),
            "warmup_calls": WARMUP_CALLS,
            "worker_pid": result.worker_pid,
            "wall_time_sec": None if not success else observation.resources.wall_time_sec,
            "process_cpu_time_sec": None if not success else observation.resources.cpu_time_sec,
            "start_rss_mib": None if not success else observation.resources.start_rss_bytes / (1024.0 * 1024.0),
            "end_rss_mib": None if not success else observation.resources.end_rss_bytes / (1024.0 * 1024.0),
            "absolute_peak_rss_mib": None if not success else observation.absolute_peak_rss_bytes / (1024.0 * 1024.0),
            "incremental_peak_rss_mib": None if not success else observation.incremental_peak_rss_bytes / (1024.0 * 1024.0),
            "input_dataframe_memory_bytes": selected_bytes,
            "numpy_dense_nbytes": dense_bytes,
            "serialized_model_bytes": model_artifact.serialized_model_bytes,
            "inference_latency_sec": None if not success else observation.inference_latency_seconds,
            "per_record_latency_sec": None if not success else observation.per_record_latency_seconds,
            "throughput_records_sec": None if not success else observation.throughput_records_per_second,
            "time_unit": CanonicalUnit.SECONDS.value,
            "rss_unit": CanonicalUnit.MEBIBYTES.value,
            "size_unit": CanonicalUnit.BYTES.value,
            "throughput_unit": CanonicalUnit.RECORDS_PER_SECOND.value,
            "measurement_scope": scope,
            "measurement_provenance": MeasurementProvenance.DIRECT_COMPUTATIONAL.value,
            "derived_measurement_provenance": MeasurementProvenance.DERIVED.value,
            "workload_sha256": workload_sha256,
            "model_sha256": model_artifact.file_sha256,
            "configuration_features_sha256": model_artifact.feature_manifest_sha256,
            "v06_lock_sha256": context.plan.validation_lock_sha256,
            "v08c_semantic_lock_sha256": v08c_semantic_lock_sha256,
            "v09d_winner_semantic_lock_sha256": v09d_semantic_lock_sha256,
            "v09e_result_semantic_lock_sha256": v09e_semantic_result_lock_sha256,
            "status": "SUCCESS" if success else "FAILURE",
            "failure_type": None if success else (None if result.failure is None else result.failure.error_type),
            "failure_stage": None if success else (None if result.failure is None else result.failure.stage),
            "failure_reason": None if success else (None if result.failure is None else result.failure.message),
            "observation_started_at_utc": started_at,
            "observation_finished_at_utc": finished_at,
            "wall_clock_elapsed_sec": wall_elapsed,
            "monotonic_elapsed_sec": mono_elapsed,
            "standby_gap_sec": standby_gap,
        }
        rows.append(row)
    validate_observation_rows(rows)
    return tuple(rows)


def validate_observation_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise V09FError("Observation rows cannot be empty.")
    identifiers: list[str] = []
    for row in rows:
        missing = set(OBSERVATION_FIELDS) - set(row)
        if missing:
            raise V09FError(f"Observation row is missing fields: {sorted(missing)}")
        identifiers.append(str(row["observation_id"]))
        if row["schema_version"] != V09F_SCHEMA_VERSION or row["stage"] != V09F_STAGE:
            raise V09FError("Observation schema/stage mismatch.")
        if row["measurement_provenance"] != MeasurementProvenance.DIRECT_COMPUTATIONAL.value:
            raise V09FError("Observation provenance must be DIRECT_COMPUTATIONAL.")
        if row["derived_measurement_provenance"] != MeasurementProvenance.DERIVED.value:
            raise V09FError("Derived observation provenance must be DERIVED.")
        for field in ("observation_started_at_utc", "observation_finished_at_utc"):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise V09FError(f"Observation is missing fine-grained timestamp: {field}")
            try:
                datetime.fromisoformat(value)
            except ValueError as exc:
                raise V09FError(f"Observation timestamp is not ISO-8601: {field}") from exc
        for field in ("wall_clock_elapsed_sec", "monotonic_elapsed_sec", "standby_gap_sec"):
            value = row.get(field)
            if value is None or not math.isfinite(float(value)) or float(value) < 0:
                raise V09FError(f"Observation timing field must be finite and nonnegative: {field}")
        if row["status"] == "SUCCESS":
            for field in (
                "wall_time_sec",
                "process_cpu_time_sec",
                "absolute_peak_rss_mib",
                "incremental_peak_rss_mib",
                "input_dataframe_memory_bytes",
                "numpy_dense_nbytes",
                "serialized_model_bytes",
                "inference_latency_sec",
                "per_record_latency_sec",
                "throughput_records_sec",
            ):
                value = row[field]
                if value is None or not math.isfinite(float(value)) or float(value) < 0:
                    raise V09FError("Successful observations require finite nonnegative metrics.")
            if int(row["measured_operation_count"]) != int(row["inner_operation_count"]):
                raise V09FError("Measured operation count must equal configured inner operations.")
            if row["warmup_calls"] != WARMUP_CALLS:
                raise V09FError("Warmup count drifted from the frozen protocol.")
            if any(row[name] is not None for name in ("failure_type", "failure_stage", "failure_reason")):
                raise V09FError("Successful observations cannot include failure details.")
        elif row["status"] == "FAILURE":
            if not row["failure_type"] or not row["failure_stage"] or not row["failure_reason"]:
                raise V09FError("Failed observations require structured failure details.")
        else:
            raise V09FError("Observation status must be SUCCESS or FAILURE.")
    if len(set(identifiers)) != len(identifiers):
        raise V09FError("Observation IDs must be unique.")


def _checkpoint_name(cell: v08e.BenchmarkCell) -> str:
    return f"{cell.phase}__{cell.classifier}__{cell.configuration_id}__s{cell.seed}.json"


def load_valid_checkpoint(
    path: Path,
    *,
    cell: v08e.BenchmarkCell,
    protocol_hash: str,
    environment_id: str,
    expected_repetitions: int = OUTER_REPETITIONS,
) -> tuple[dict[str, Any], ...] | None:
    if not path.is_file():
        return None
    payload = _read_json(path)
    identity = payload.get("cell_identity", {})
    if (
        payload.get("protocol_sha256") != protocol_hash
        or payload.get("environment_id") != environment_id
        or int(payload.get("expected_repetition_count", -1)) != expected_repetitions
        or identity.get("classifier") != cell.classifier
        or identity.get("configuration_id") != cell.configuration_id
        or identity.get("seed") != cell.seed
        or identity.get("phase") != cell.phase
    ):
        raise V09FCheckpointError("Checkpoint is incompatible with this protocol/environment/cell.")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise V09FCheckpointError("Checkpoint rows are missing or malformed.")
    if len(rows) != expected_repetitions:
        raise V09FCheckpointError("Checkpoint must contain one complete 10-repetition cell.")
    repetitions = tuple(int(row["outer_repetition"]) for row in rows)
    if repetitions != REPETITIONS:
        raise V09FCheckpointError("Checkpoint row repetitions must be exactly 1-10.")
    validate_observation_rows(rows)
    return tuple(rows)


def write_cell_checkpoint(
    path: Path,
    *,
    cell: v08e.BenchmarkCell,
    protocol_hash: str,
    environment_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if len(rows) != OUTER_REPETITIONS:
        raise V09FCheckpointError("Only complete benchmark cells are checkpointed.")
    payload = {
        "stage": V09F_STAGE,
        "schema_version": V09F_SCHEMA_VERSION,
        "protocol_sha256": protocol_hash,
        "environment_id": environment_id,
        "expected_repetition_count": OUTER_REPETITIONS,
        "cell_identity": {
            "classifier": cell.classifier,
            "configuration_id": cell.configuration_id,
            "seed": cell.seed,
            "phase": cell.phase,
        },
        "rows": list(rows),
    }
    _atomic_write_json(path, payload)


def summarize_within_seed(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate 10 retained repetitions within each seed before scientific pairing."""

    groups: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["classifier"], row["configuration_id"], int(row["seed"]), row["phase"])
        groups.setdefault(key, []).append(row)
    metrics = (
        *SCIENTIFIC_METRICS,
        "wall_clock_elapsed_sec",
        "monotonic_elapsed_sec",
        "standby_gap_sec",
    )
    output: list[dict[str, Any]] = []
    for (classifier, configuration_id, seed, phase), group in sorted(groups.items()):
        if len(group) != OUTER_REPETITIONS:
            raise V09FError("Within-seed summaries require exactly 10 repetitions per cell.")
        repetitions = {int(row["outer_repetition"]) for row in group}
        if repetitions != set(REPETITIONS):
            raise V09FError("Within-seed summaries must use repetitions 1-10 exactly once.")
        if any(row["status"] != "SUCCESS" for row in group):
            raise V09FError("Within-seed summaries require successful repetitions.")
        static_feature_count = {int(row["feature_count"]) for row in group}
        static_input = {int(row["input_dataframe_memory_bytes"]) for row in group}
        static_dense = {int(row["numpy_dense_nbytes"]) for row in group}
        static_model = {int(row["serialized_model_bytes"]) for row in group}
        if any(len(item) != 1 for item in (static_feature_count, static_input, static_dense, static_model)):
            raise V09FError("Static resource values changed within a repetition block.")
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            output.append(
                {
                    "stage": V09F_STAGE,
                    "classifier": classifier,
                    "configuration_id": configuration_id,
                    "seed": seed,
                    "phase": phase,
                    "metric": metric,
                    "repetition_count": OUTER_REPETITIONS,
                    "mean": statistics.fmean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values),
                    "min": min(values),
                    "max": max(values),
                    "coefficient_of_variation_percent": v08e.coefficient_of_variation(values),
                    "feature_count": next(iter(static_feature_count)),
                    "input_dataframe_memory_bytes": next(iter(static_input)),
                    "numpy_dense_nbytes": next(iter(static_dense)),
                    "serialized_model_bytes": next(iter(static_model)),
                    "measurement_provenance": MeasurementProvenance.DIRECT_COMPUTATIONAL.value,
                    "aggregation_scope": "ten_measurement_repetitions_within_seed",
                }
            )
    return output


def summarize_across_seeds(within_seed: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in within_seed:
        key = (row["classifier"], row["configuration_id"], row["phase"], row["metric"])
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (classifier, configuration_id, phase, metric), group in sorted(groups.items()):
        if len(group) != len(SEEDS):
            raise V09FError("Across-seed summaries require exactly five seed-level rows.")
        seed_means = [float(row["mean"]) for row in group]
        output.append(
            {
                "stage": V09F_STAGE,
                "classifier": classifier,
                "configuration_id": configuration_id,
                "phase": phase,
                "metric": metric,
                "unit": METRIC_UNITS.get(metric, "unknown"),
                "seed_count": len(SEEDS),
                "mean_of_seed_means": statistics.fmean(seed_means),
                "std_of_seed_means": statistics.stdev(seed_means),
                "min_of_seed_means": min(seed_means),
                "max_of_seed_means": max(seed_means),
                "feature_count": int(group[0]["feature_count"]),
                "input_dataframe_memory_bytes": int(group[0]["input_dataframe_memory_bytes"]),
                "numpy_dense_nbytes": int(group[0]["numpy_dense_nbytes"]),
                "serialized_model_bytes": int(group[0]["serialized_model_bytes"]),
                "measurement_provenance": MeasurementProvenance.DIRECT_COMPUTATIONAL.value,
                "aggregation_hierarchy": "seeds_are_scientific_units_ten_repetitions_nested",
            }
        )
    return output


def _seed_level_index(within_seed: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int, str, str], float]:
    return {
        (
            row["classifier"],
            row["configuration_id"],
            int(row["seed"]),
            row["phase"],
            row["metric"],
        ): float(row["mean"])
        for row in within_seed
    }


def build_resource_comparisons(
    within_seed: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Paired BGWO-vs-reference comparisons using seed-level means."""

    by_key = _seed_level_index(within_seed)
    metrics = sorted({str(row["metric"]) for row in within_seed if str(row["metric"]) in SCIENTIFIC_METRICS})
    pairs = (
        *(("primary", candidate, baseline) for candidate, baseline in PRIMARY_PAIRS),
        *(("secondary", candidate, baseline) for candidate, baseline in SECONDARY_PAIRS),
    )
    output: list[dict[str, Any]] = []
    for tier, candidate_id, baseline_id in pairs:
        for classifier in CLASSIFIERS:
            for phase in PHASES:
                for metric in metrics:
                    differences: list[float] = []
                    percent_changes: list[float] = []
                    candidate_values: list[float] = []
                    baseline_values: list[float] = []
                    seed_pairs: list[dict[str, Any]] = []
                    for seed in SEEDS:
                        candidate_value = by_key[(classifier, candidate_id, seed, phase, metric)]
                        baseline_value = by_key[(classifier, baseline_id, seed, phase, metric)]
                        difference = candidate_value - baseline_value
                        percent = v08e.relative_change_percent(candidate_value, baseline_value)
                        differences.append(difference)
                        percent_changes.append(percent)
                        candidate_values.append(candidate_value)
                        baseline_values.append(baseline_value)
                        seed_pairs.append(
                            {
                                "seed": seed,
                                "candidate_value": candidate_value,
                                "baseline_value": baseline_value,
                                "difference": difference,
                                "percent_change": percent,
                            }
                        )
                    ci_low, ci_high = v08e.paired_t95_interval(differences)
                    output.append(
                        {
                            "stage": V09F_STAGE,
                            "comparison_tier": tier,
                            "classifier": classifier,
                            "phase": phase,
                            "metric": metric,
                            "unit": METRIC_UNITS.get(metric, "unknown"),
                            "candidate_configuration_id": candidate_id,
                            "reference_configuration_id": baseline_id,
                            "seed_pairs": seed_pairs,
                            "candidate_mean": statistics.fmean(candidate_values),
                            "reference_mean": statistics.fmean(baseline_values),
                            "mean_difference": statistics.fmean(differences),
                            "std_difference": statistics.stdev(differences),
                            "mean_percent_change": statistics.fmean(percent_changes),
                            "ci95_low": ci_low,
                            "ci95_high": ci_high,
                            "n": 5,
                            "df": 4,
                            "t_critical": T_CRITICAL_95_DF4,
                            "aggregation_hierarchy": (
                                "ten_repetitions_aggregated_within_seed_before_five_seed_scientific_pairing"
                            ),
                        }
                    )
    return output


def build_model_size_comparison(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str], int] = {}
    for row in rows:
        by_key[(str(row["classifier"]), str(row["configuration_id"]))] = int(row["serialized_model_bytes"])
    entries: list[dict[str, Any]] = []
    for classifier in CLASSIFIERS:
        baseline_bytes = by_key[(classifier, "K43")]
        for configuration_id in CONFIGURATION_IDS:
            size = by_key[(classifier, configuration_id)]
            entries.append(
                {
                    "classifier": classifier,
                    "configuration_id": configuration_id,
                    "serialized_model_bytes": size,
                    "reference_configuration_id": "K43",
                    "difference_bytes_vs_k43": size - baseline_bytes,
                    "percent_change_vs_k43": v08e.relative_change_percent(size, baseline_bytes),
                    "unit": "bytes",
                }
            )
    return {
        "schema_version": V09F_SCHEMA_VERSION,
        "stage": V09F_STAGE,
        "artifact_kind": "SERIALIZED_MODEL_SIZE_COMPARISON",
        "entries": entries,
        "interpretation": (
            "Serialized model size is a structural property of the fitted model and its "
            "feature cardinality; fewer features do not guarantee a smaller decision tree."
        ),
    }


def build_feature_reduction() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    baseline = 43
    counts = {"K43": 43, "K42": 42, "MI-K11": 11, "BPSO-K10": 10, "BGWO": EXPECTED_BGWO_WINNER["selected_feature_count"]}
    for configuration_id in CONFIGURATION_IDS:
        count = int(counts[configuration_id])
        rows.append(
            {
                "configuration_id": configuration_id,
                "feature_count": count,
                "reference_configuration_id": "K43",
                "reference_feature_count": baseline,
                "absolute_reduction": baseline - count,
                "relative_reduction_percent": 100.0 * (baseline - count) / baseline,
                "unit": "features",
                "reduction_is_structural_deterministic": True,
                "runtime_effect_distinguished": True,
            }
        )
    bgwo_count = int(EXPECTED_BGWO_WINNER["selected_feature_count"])
    return {
        "schema_version": V09F_SCHEMA_VERSION,
        "stage": V09F_STAGE,
        "artifact_kind": "STRUCTURAL_FEATURE_REDUCTION",
        "bgwo_k": bgwo_count,
        "bgwo_reduction_percent_vs_k43": 100.0 * (baseline - bgwo_count) / baseline,
        "entries": rows,
        "interpretation": (
            "Feature reduction is a deterministic structural property of the frozen subsets. "
            "It must not be conflated with measured runtime changes."
        ),
    }


def build_timing_quality(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold_sec: float = STANDBY_GAP_THRESHOLD_SEC,
) -> dict[str, Any]:
    groups: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["classifier"], row["configuration_id"], int(row["seed"]), row["phase"])
        groups.setdefault(key, []).append(row)
    cells: list[dict[str, Any]] = []
    flagged_ids: list[str] = []
    for (classifier, configuration_id, seed, phase), group in sorted(groups.items()):
        gaps = [float(row["standby_gap_sec"]) for row in group]
        wall = [float(row["wall_clock_elapsed_sec"]) for row in group]
        mono = [float(row["monotonic_elapsed_sec"]) for row in group]
        flagged = [str(row["observation_id"]) for row in group if float(row["standby_gap_sec"]) > threshold_sec]
        flagged_ids.extend(flagged)
        cells.append(
            {
                "classifier": classifier,
                "configuration_id": configuration_id,
                "seed": seed,
                "phase": phase,
                "observation_count": len(group),
                "wall_clock_elapsed_sec_mean": statistics.fmean(wall),
                "monotonic_elapsed_sec_mean": statistics.fmean(mono),
                "standby_gap_sec_max": max(gaps),
                "standby_gap_sec_mean": statistics.fmean(gaps),
                "flagged_observation_count": len(flagged),
                "flagged_observation_ids": flagged,
                "standby_affected": bool(flagged),
            }
        )
    affected_cells = sum(1 for cell in cells if cell["standby_affected"])
    return {
        "schema_version": V09F_SCHEMA_VERSION,
        "stage": V09F_STAGE,
        "artifact_kind": "TIMING_QUALITY_AND_SLEEP_AUDIT",
        "standby_gap_threshold_sec": threshold_sec,
        "gap_semantics": (
            "standby_gap_sec = max(0, wall_clock_elapsed_sec - monotonic_elapsed_sec) using "
            "time.time() and time.perf_counter() around each fresh-worker observation."
        ),
        "observation_level_timestamps_present": True,
        "observation_count": len(rows),
        "cell_count": len(cells),
        "standby_affected_cell_count": affected_cells,
        "flagged_observation_count": len(flagged_ids),
        "flagged_observation_ids": flagged_ids,
        "outliers_removed": False,
        "exclusion_policy": "PRESERVE_ALL_OBSERVATIONS_NO_SILENT_EXCLUSION",
        "cells": cells,
        "timing_interpretation": "OBSERVATION_LEVEL_TIMESTAMPED"
        if not flagged_ids
        else "OBSERVATION_LEVEL_TIMESTAMPED_WITH_STANDBY_FLAGS",
    }


def build_optimizer_overhead(
    *,
    bgwo_execution_path: Path | str = V09D_EXECUTION_PATH,
    bpso_summary_importer: Callable[[], dict[str, Any]] = v08e.import_v08c_optimizer_overhead,
) -> dict[str, Any]:
    """Imported-historical optimizer overhead; neither optimizer is rerun."""

    bgwo = _read_json(bgwo_execution_path)
    if bgwo.get("stage") != "V0.9-D" or bgwo.get("status") != "COMPLETED":
        raise V09FError("V0.9-D execution summary is not complete.")
    bpso = bpso_summary_importer()
    bgwo_payload = {
        "optimizer": "BGWO",
        "stage": "V0.9-D",
        "measurement_provenance": MeasurementProvenance.IMPORTED_HISTORICAL.value,
        "source_artifact": str(Path(bgwo_execution_path).resolve()),
        "source_artifact_sha256": sha256_file(bgwo_execution_path),
        "optimizer_runs": int(bgwo["completed_run_count"]),
        "candidate_requests": int(bgwo["candidate_requests"]),
        "unique_evaluations": int(bgwo["unique_evaluations"]),
        "cache_hits": int(bgwo["cache_hits"]),
        "decision_tree_fits": int(bgwo["actual_decision_tree_fits"]),
        "core_optimization_wall_time_sec": float(bgwo["optimizer_wall_time_sec"]),
        "core_optimization_cpu_time_sec": float(bgwo["optimizer_cpu_time_sec"]),
        "pipeline_wall_time_sec": float(bgwo["pipeline_wall_time_sec"]),
        "optimizer_rerun": False,
    }
    bpso_payload = {
        "optimizer": "BPSO",
        "stage": "V0.8-C",
        "measurement_provenance": bpso["measurement_provenance"],
        "source_artifact": bpso["source_artifact"],
        "source_artifact_sha256": bpso["source_artifact_sha256"],
        "optimizer_runs": bpso["optimizer_runs"],
        "candidate_requests": bpso["fitness_requests"],
        "unique_evaluations": bpso["unique_evaluations"],
        "cache_hits": bpso["cache_hits"],
        "decision_tree_fits": bpso["decision_tree_fits"],
        "core_optimization_wall_time_sec": bpso["core_optimization_wall_time_sec"],
        "core_optimization_cpu_time_sec": bpso["core_optimization_cpu_time_sec"],
        "pipeline_wall_time_sec": bpso["pipeline_wall_time_sec"],
        "optimizer_rerun": False,
    }
    return {
        "schema_version": V09F_SCHEMA_VERSION,
        "stage": V09F_STAGE,
        "artifact_kind": "OPTIMIZER_OVERHEAD_EVIDENCE",
        "optimizer_rerun": False,
        "core_and_pipeline_overheads_kept_separate": True,
        "overhead_role": "ONE_TIME_IMPORTED_HISTORICAL_SEARCH_COST",
        "bgwo": bgwo_payload,
        "bpso": bpso_payload,
        "separation_statement": (
            "One-time optimizer search cost is reported separately from per-model training "
            "and inference cost measured in V0.9-F."
        ),
    }


def build_break_even(
    *,
    within_seed: Sequence[Mapping[str, Any]],
    optimizer_overhead: Mapping[str, Any],
    candidate_id: str = "BGWO",
    baseline_ids: Sequence[str] = ("K43", "BPSO-K10"),
    phase: str = MeasurementPhase.TRAINING.value,
) -> dict[str, Any]:
    """Descriptive training-cost break-even using matching units only."""

    by_key = _seed_level_index(within_seed)
    bgwo = optimizer_overhead["bgwo"]
    rows: list[dict[str, Any]] = []
    for baseline_id in baseline_ids:
        for metric, overhead_wall_key in (("wall_time_sec", "core_optimization_wall_time_sec"),):
            candidate_values = [by_key[(classifier, candidate_id, seed, phase, metric)] for classifier in CLASSIFIERS for seed in SEEDS]
            baseline_values = [by_key[(classifier, baseline_id, seed, phase, metric)] for classifier in CLASSIFIERS for seed in SEEDS]
            candidate_mean = statistics.fmean(candidate_values)
            baseline_mean = statistics.fmean(baseline_values)
            recurring_saving = baseline_mean - candidate_mean
            overhead_value = float(bgwo[overhead_wall_key])
            outcome = v08e.break_even_repetitions(
                overhead_value=overhead_value,
                recurring_saving=recurring_saving,
                overhead_unit="seconds",
                saving_unit="seconds",
            )
            rows.append(
                {
                    "candidate_configuration_id": candidate_id,
                    "reference_configuration_id": baseline_id,
                    "phase": phase,
                    "metric": metric,
                    "unit": "seconds",
                    "optimizer_overhead_sec": overhead_value,
                    "per_model_recurring_saving_sec": recurring_saving,
                    "status": outcome["status"],
                    "break_even_repetitions": outcome["break_even_repetitions"],
                    "provenance": MeasurementProvenance.IMPORTED_HISTORICAL.value,
                }
            )
    return {
        "schema_version": V09F_SCHEMA_VERSION,
        "stage": V09F_STAGE,
        "artifact_kind": "DESCRIPTIVE_BREAK_EVEN_ANALYSIS",
        "overhead_source": "V0.9-D BGWO optimizer search",
        "timing_break_even_is_not_energy_break_even": True,
        "energy_break_even_status": "NOT_APPLICABLE_DIRECT_ENERGY_UNAVAILABLE",
        "rows": rows,
        "interpretation": (
            "Break-even is descriptive and only reported when the per-model saving is positive. "
            "It is a computational-timing statement and must not be presented as an energy result."
        ),
    }


def build_tradeoff(
    *,
    summary: Sequence[Mapping[str, Any]],
    predictive_summary_path: Path | str = V09E_SUMMARY_PATH,
) -> dict[str, Any]:
    """Descriptive post-hoc trade-off; no optimization is performed."""

    predictive = _read_json(predictive_summary_path)
    predictive_rows = predictive.get("summaries", [])
    pred_index: dict[tuple[str, str, str], float] = {}
    for row in predictive_rows:
        pred_index[(str(row["classifier"]), str(row["configuration_id"]), str(row["metric"]))] = float(row["mean"])
    config_static: dict[str, dict[str, Any]] = {}
    for row in summary:
        config_id = str(row["configuration_id"])
        static = config_static.setdefault(config_id, {})
        static["feature_count"] = int(row["feature_count"])
        static["serialized_model_bytes"] = int(row["serialized_model_bytes"])
        static["input_dataframe_memory_bytes"] = int(row["input_dataframe_memory_bytes"])
        if row["metric"] == "wall_time_sec" and row["phase"] == MeasurementPhase.TRAINING.value:
            static["training_wall_time_sec"] = row["mean_of_seed_means"]
        if row["metric"] == "per_record_latency_sec" and row["phase"] == MeasurementPhase.INFERENCE.value:
            static["inference_per_record_latency_sec"] = row["mean_of_seed_means"]
    entries: list[dict[str, Any]] = []
    for configuration_id in CONFIGURATION_IDS:
        static = config_static.get(configuration_id, {})
        entry = {
            "configuration_id": configuration_id,
            "feature_count": static.get("feature_count"),
            "training_wall_time_sec": static.get("training_wall_time_sec"),
            "inference_per_record_latency_sec": static.get("inference_per_record_latency_sec"),
            "serialized_model_bytes": static.get("serialized_model_bytes"),
            "input_dataframe_memory_bytes": static.get("input_dataframe_memory_bytes"),
            "test_average_precision": {
                classifier: pred_index.get((classifier, configuration_id, "average_precision"))
                for classifier in CLASSIFIERS
            },
            "test_f1": {
                classifier: pred_index.get((classifier, configuration_id, "f1")) for classifier in CLASSIFIERS
            },
            "test_recall": {
                classifier: pred_index.get((classifier, configuration_id, "recall")) for classifier in CLASSIFIERS
            },
        }
        entries.append(entry)
    return {
        "schema_version": V09F_SCHEMA_VERSION,
        "stage": V09F_STAGE,
        "artifact_kind": "DESCRIPTIVE_POST_HOC_TRADEOFF_ANALYSIS",
        "label": "DESCRIPTIVE_POST_HOC_TRADEOFF_ANALYSIS",
        "optimization_performed": False,
        "objective_function_introduced": False,
        "nsga_or_mopso_used": False,
        "sources": {
            "predictive": {
                "stage": "V0.9-E",
                "path": str(Path(predictive_summary_path).resolve()),
                "sha256": sha256_file(predictive_summary_path),
                "provenance": MeasurementProvenance.IMPORTED_HISTORICAL.value,
            },
            "resource": {"stage": "V0.9-F", "provenance": MeasurementProvenance.DIRECT_COMPUTATIONAL.value},
        },
        "entries": entries,
        "interpretation": (
            "This is a descriptive comparison of feature count, frozen V0.9-E predictive "
            "metrics, and V0.9-F measured resources. No multi-objective optimization, ranking, "
            "or winner replacement is performed."
        ),
    }


def build_sleep_governance(
    *,
    environment: Mapping[str, Any],
    energy: Mapping[str, Any],
) -> dict[str, Any]:
    power_state = None
    mem_sleep = None
    try:
        power_state = Path("/sys/power/state").read_text(encoding="utf-8").strip()
    except OSError:
        power_state = None
    try:
        mem_sleep = Path("/sys/power/mem_sleep").read_text(encoding="utf-8").strip()
    except OSError:
        mem_sleep = None
    return {
        "schema_version": V09F_SCHEMA_VERSION,
        "stage": V09F_STAGE,
        "artifact_kind": "ENVIRONMENT_SLEEP_AND_ENERGY_GOVERNANCE",
        "environment": dict(environment),
        "energy": dict(energy),
        "sleep_governance": {
            "guest_power_state_file": power_state,
            "guest_mem_sleep_file": mem_sleep,
            "host_sleep_availability_verified": False,
            "host_sleep_control_attempted": False,
            "temporary_sleep_prevention_used": False,
            "permanent_power_settings_modified": False,
            "per_observation_timestamps_recorded": True,
            "standby_gap_threshold_sec": STANDBY_GAP_THRESHOLD_SEC,
            "reason": (
                "The measurement runs under a WSL2 virtual guest whose /sys/power view reflects "
                "the guest, not the Windows host. Windows host Modern Standby availability and "
                "intervals cannot be safely verified or controlled from here, so host sleep "
                "governance is recorded as unverified rather than assumed, and no power settings "
                "are modified. Every observation instead carries UTC and monotonic timestamps so "
                "standby gaps can be detected rather than inferred retrospectively."
            ),
        },
        "direct_energy_status": energy.get("decision", DIRECT_ENERGY_STATUS),
        "energy_proxies_only": True,
        "joule_estimation_from_cpu_or_wall_performed": False,
        "tdp_multiplication_performed": False,
    }


def build_result_lock(
    *,
    starting_head: str,
    preflight: Mapping[str, Any],
    configurations: Sequence[Mapping[str, Any]],
    classifiers: Mapping[str, Mapping[str, Any]],
    artifact_hashes: Mapping[str, str],
    environment_id: str,
    environment_sha256: str,
    observation_counts: Mapping[str, int],
    timing_quality: Mapping[str, Any],
    started_at: str,
    completed_at: str,
    wall_time_sec: float,
) -> dict[str, Any]:
    configuration_plan = [
        {
            "configuration_id": plan["configuration_id"],
            "source_configuration_id": plan["source_configuration_id"],
            "feature_count": plan["feature_count"],
            "selection_scope": plan["selection_scope"],
            "selection_semantics": plan["selection_semantics"],
            "source_semantic_lock_sha256": plan["source_semantic_lock_sha256"],
        }
        for plan in configurations
    ]
    lock = {
        "schema_version": V09F_SCHEMA_VERSION,
        "stage": V09F_STAGE,
        "status": "RESOURCE_RESULT_LOCKED",
        "starting_head": starting_head,
        "source_winner_stage": "V0.9-D",
        "source_winner_semantic_lock_sha256": preflight["bgwo_winner_semantic_lock_sha256"],
        "bgwo_selected_feature_count": int(preflight["bgwo_winner_selected_feature_count"]),
        "bgwo_ordered_selected_features": list(EXPECTED_BGWO_FEATURES),
        "bgwo_mask_sha256": preflight["bgwo_winner_mask_sha256"],
        "bgwo_selected_features_sha256": preflight["bgwo_winner_selected_features_sha256"],
        "bpso_source_semantic_lock_sha256": preflight["bpso_winner_semantic_lock_sha256"],
        "bpso_selected_feature_count": int(preflight["bpso_winner_selected_feature_count"]),
        "bpso_ordered_selected_features": list(EXPECTED_BPSO_FEATURES),
        "bpso_mask_sha256": preflight["bpso_winner_mask_sha256"],
        "bpso_selected_features_sha256": preflight["bpso_winner_selected_features_sha256"],
        "v09e_result_semantic_lock_sha256": preflight["v09e_result_semantic_lock_sha256"],
        "resource_protocol_identity": dict(preflight["resource_protocol_identity"]),
        "environment_id": environment_id,
        "environment_sha256": environment_sha256,
        "configuration_ids": list(CONFIGURATION_IDS),
        "configuration_plan": configuration_plan,
        "configuration_plan_sha256": _json_sha256(configuration_plan),
        "classifier_configurations": classifiers,
        "classifier_configurations_sha256": _json_sha256(classifiers),
        "seed_namespace": list(SEEDS),
        "dataset_identity": dict(preflight["dataset_identity"]),
        "observation_counts": dict(observation_counts),
        "result_artifact_hashes": dict(artifact_hashes),
        "raw_observation_evidence_hash": artifact_hashes.get(RAW_CSV_PATH.name),
        "summary_evidence_hash": artifact_hashes.get(SUMMARY_CSV_PATH.name),
        "comparison_evidence_hash": artifact_hashes.get(COMPARISONS_CSV_PATH.name),
        "timing_quality_evidence_hash": artifact_hashes.get(TIMING_QUALITY_PATH.name),
        "optimizer_overhead_evidence_hash": artifact_hashes.get(OPTIMIZER_OVERHEAD_PATH.name),
        "tradeoff_evidence_hash": artifact_hashes.get(TRADEOFF_PATH.name),
        "observation_level_timestamps_present": bool(timing_quality["observation_level_timestamps_present"]),
        "direct_energy_status": DIRECT_ENERGY_STATUS,
        "direct_energy_measured": False,
        "joule_estimation_from_cpu_or_wall_performed": False,
        "tdp_multiplication_performed": False,
        "optimizer_invoked": False,
        "optimizer_rerun": False,
        "feature_reselection_performed": False,
        "model_tuning_performed": False,
        "threshold_tuning_performed": False,
        "final_test_accessed": False,
        "test_used_for_selection": False,
        "winner_replaced": False,
        "outliers_removed": False,
        "execution_metadata": {
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "wall_time_sec": wall_time_sec,
        },
    }
    lock["semantic_result_lock_sha256"] = result_lock_semantic_hash(lock)
    return lock


def result_lock_semantic_hash(lock: Mapping[str, Any]) -> str:
    payload = json.loads(json.dumps(lock, sort_keys=True, allow_nan=False))
    payload.pop("semantic_result_lock_sha256", None)
    execution = dict(payload.get("execution_metadata", {}))
    for field in ("started_at_utc", "completed_at_utc", "wall_time_sec"):
        execution.pop(field, None)
    payload["execution_metadata"] = execution
    return _json_sha256(payload)


def verify_result_lock(lock: Mapping[str, Any]) -> None:
    required = {
        "stage",
        "status",
        "starting_head",
        "source_winner_semantic_lock_sha256",
        "bgwo_ordered_selected_features",
        "bgwo_mask_sha256",
        "bgwo_selected_features_sha256",
        "bpso_ordered_selected_features",
        "bpso_mask_sha256",
        "bpso_selected_features_sha256",
        "v09e_result_semantic_lock_sha256",
        "resource_protocol_identity",
        "environment_id",
        "environment_sha256",
        "configuration_ids",
        "configuration_plan",
        "configuration_plan_sha256",
        "classifier_configurations",
        "classifier_configurations_sha256",
        "seed_namespace",
        "dataset_identity",
        "observation_counts",
        "result_artifact_hashes",
        "raw_observation_evidence_hash",
        "summary_evidence_hash",
        "comparison_evidence_hash",
        "timing_quality_evidence_hash",
        "optimizer_overhead_evidence_hash",
        "tradeoff_evidence_hash",
        "semantic_result_lock_sha256",
    }
    configs = tuple(lock.get("configuration_plan", ()))
    valid = (
        required <= set(lock)
        and lock.get("stage") == V09F_STAGE
        and lock.get("status") == "RESOURCE_RESULT_LOCKED"
        and lock.get("starting_head") == EXPECTED_STARTING_HEAD
        and lock.get("source_winner_stage") == "V0.9-D"
        and lock.get("source_winner_semantic_lock_sha256") == EXPECTED_BGWO_WINNER["semantic_lock_sha256"]
        and lock.get("bgwo_mask_sha256") == EXPECTED_BGWO_WINNER["mask_sha256"]
        and lock.get("bgwo_selected_features_sha256") == EXPECTED_BGWO_WINNER["selected_features_sha256"]
        and int(lock.get("bgwo_selected_feature_count", -1)) == EXPECTED_BGWO_WINNER["selected_feature_count"]
        and tuple(lock.get("bgwo_ordered_selected_features", ())) == EXPECTED_BGWO_FEATURES
        and lock.get("bpso_mask_sha256") == EXPECTED_BPSO_WINNER["mask_sha256"]
        and lock.get("bpso_selected_features_sha256") == EXPECTED_BPSO_WINNER["selected_features_sha256"]
        and tuple(lock.get("bpso_ordered_selected_features", ())) == EXPECTED_BPSO_FEATURES
        and lock.get("v09e_result_semantic_lock_sha256") == EXPECTED_V09E_RESULT_LOCK_SHA256
        and tuple(lock.get("configuration_ids", ())) == CONFIGURATION_IDS
        and tuple(row.get("configuration_id") for row in configs) == CONFIGURATION_IDS
        and lock.get("configuration_plan_sha256") == _json_sha256(lock.get("configuration_plan"))
        and lock.get("classifier_configurations_sha256") == _json_sha256(lock.get("classifier_configurations"))
        and tuple(lock.get("seed_namespace", ())) == SEEDS
        and int(lock.get("observation_counts", {}).get("total", -1)) == EXPECTED_TOTAL_OBSERVATIONS
        and lock.get("direct_energy_measured") is False
        and lock.get("joule_estimation_from_cpu_or_wall_performed") is False
        and lock.get("tdp_multiplication_performed") is False
        and lock.get("optimizer_invoked") is False
        and lock.get("optimizer_rerun") is False
        and lock.get("feature_reselection_performed") is False
        and lock.get("model_tuning_performed") is False
        and lock.get("threshold_tuning_performed") is False
        and lock.get("final_test_accessed") is False
        and lock.get("test_used_for_selection") is False
        and lock.get("winner_replaced") is False
        and lock.get("outliers_removed") is False
        and result_lock_semantic_hash(lock) == lock.get("semantic_result_lock_sha256")
    )
    if not valid:
        raise V09FError("V0.9-F resource result lock verification failed.")


def write_or_verify_result_lock(path: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    verify_result_lock(lock)
    target = Path(path)
    if target.exists():
        existing = _read_json(target)
        verify_result_lock(existing)
        if existing["semantic_result_lock_sha256"] != lock["semantic_result_lock_sha256"]:
            raise V09FError("Existing V0.9-F result lock differs from the recomputed result.")
        return existing
    _atomic_write_json(target, lock)
    stored = _read_json(target)
    verify_result_lock(stored)
    return stored


def run_v09f(
    *,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    checkpoint_dir: Path | str = DEFAULT_CHECKPOINT_DIR,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    execute_matrix: bool = False,
    start_method: str | None = None,
    benchmark_context_loader: Callable[[], Any] = v07b.load_v06_benchmark_context,
    workload_loader: Callable[[Any], Any] = v07b.prepare_verified_workloads,
    runner: Callable[..., Any] = run_fresh_worker_protocol,
) -> dict[str, Any]:
    """Verify prerequisites and optionally execute the timestamped resource matrix."""

    output_root = Path(output_dir)
    checkpoint_root = Path(checkpoint_dir)
    model_root = Path(model_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    preflight = verify_v09f_preflight(config_path=config_path)
    thread_controls = v08e.configure_single_thread_environment()
    environment_metadata = collect_environment_metadata(PROJECT_ROOT)
    environment_id = v08e._json_sha256(
        {
            "os": environment_metadata.get("os", {}),
            "python": environment_metadata.get("python", {}),
            "cpu": environment_metadata.get("cpu", {}),
            "dependencies": environment_metadata.get("dependencies", {}),
            "execution_environment": environment_metadata.get("execution_environment", {}),
            "thread_environment": thread_controls,
            "direct_energy_status": preflight["direct_energy_status"],
        }
    )
    if environment_metadata.get("thread_environment") != thread_controls:
        raise V09FError("Single-thread controls were not applied before environment capture.")
    manifest = build_run_manifest(preflight, environment_id)
    sleep_governance = build_sleep_governance(
        environment=environment_metadata, energy=preflight["energy_capability"]
    )
    environment_payload = {
        "schema_version": V09F_SCHEMA_VERSION,
        "stage": V09F_STAGE,
        "artifact_kind": "V09F_ENVIRONMENT_IDENTITY",
        "environment_id": environment_id,
        "thread_environment": thread_controls,
        "environment": environment_metadata,
        "direct_energy_status": preflight["direct_energy_status"],
        "environment_sha256": _json_sha256(environment_metadata),
    }

    _atomic_write_json(output_root / PREFLIGHT_PATH.name, preflight)
    _atomic_write_json(output_root / ENVIRONMENT_PATH.name, environment_payload)
    _atomic_write_json(output_root / SLEEP_PATH.name, sleep_governance)
    _atomic_write_json(output_root / MANIFEST_PATH.name, manifest)

    if not execute_matrix:
        return {
            "schema_version": V09F_SCHEMA_VERSION,
            "stage": V09F_STAGE,
            "status": "INFRASTRUCTURE_READY",
            "execute_matrix": False,
            "environment_id": environment_id,
            "expected_total_observations": EXPECTED_TOTAL_OBSERVATIONS,
            "manifest_sha256": sha256_file(output_root / MANIFEST_PATH.name),
            "readiness_for_campaign": "GO",
        }

    stage_start = time.perf_counter()
    started_at = _utc_now()
    context = benchmark_context_loader()
    workloads = workload_loader(context)
    inference_workloads = build_features_only_inference_workloads(
        workloads, candidate_features=context.plan.candidate_features
    )
    plans = preflight["configuration_plan"]
    plan_by_id = {plan["configuration_id"]: plan for plan in plans}
    classifiers = preflight["classifier_plan"]
    model_artifacts = prepare_inference_model_artifacts(
        output_dir=model_root, configuration_plan=plans, workloads=workloads
    )
    _atomic_write_json(
        output_root / MODEL_MANIFEST_PATH.name,
        {
            "schema_version": V09F_SCHEMA_VERSION,
            "stage": V09F_STAGE,
            "artifact_kind": "INFERENCE_MODEL_ARTIFACT_MANIFEST",
            "status": "COMPLETED",
            "measurement_provenance": MeasurementProvenance.DIRECT_COMPUTATIONAL.value,
            "included_in_scientific_observations": False,
            "artifact_count": len(model_artifacts),
            "artifacts": [
                {
                    "classifier": artifact.classifier,
                    "configuration_id": artifact.configuration_id,
                    "source_configuration_id": artifact.source_configuration_id,
                    "seed": artifact.seed,
                    "path": str(artifact.path.resolve()),
                    "file_sha256": artifact.file_sha256,
                    "model_state_sha256": artifact.model_state_sha256,
                    "serialized_model_bytes": artifact.serialized_model_bytes,
                    "feature_manifest_sha256": artifact.feature_manifest_sha256,
                    "feature_count": len(artifact.feature_names),
                    "setup_fit_scope": "outside_timed_benchmark_boundary",
                    "predictive_metrics_computed": False,
                    "final_test_labels_accessed": False,
                }
                for artifact in sorted(
                    model_artifacts.values(),
                    key=lambda item: (item.configuration_id, item.classifier, item.seed),
                )
            ],
        },
    )

    schedule = build_observation_schedule()
    cells = schedule_cells(schedule)
    rows: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        block = schedule[index * OUTER_REPETITIONS : (index + 1) * OUTER_REPETITIONS]
        checkpoint_path = checkpoint_root / _checkpoint_name(cell)
        cached = load_valid_checkpoint(
            checkpoint_path,
            cell=cell,
            protocol_hash=manifest["protocol_sha256"],
            environment_id=environment_id,
        )
        if cached is None:
            measured = _run_cell_measurement(
                context=context,
                cell=cell,
                configuration_plan=plan_by_id[cell.configuration_id],
                model_artifact=model_artifacts[(cell.configuration_id, cell.classifier, cell.seed)],
                workloads=workloads,
                inference_workload=inference_workloads[cell.seed],
                environment_id=environment_id,
                v08c_semantic_lock_sha256=preflight["bpso_winner_semantic_lock_sha256"],
                v09d_semantic_lock_sha256=preflight["bgwo_winner_semantic_lock_sha256"],
                v09e_semantic_result_lock_sha256=preflight["v09e_result_semantic_lock_sha256"],
                runner=runner,
                start_method=start_method,
            )
            measured_ids = tuple(row["observation_id"] for row in measured)
            expected_ids = tuple(item.observation_id for item in block)
            if measured_ids != expected_ids:
                raise V09FError("Measured observation IDs differ from deterministic schedule.")
            write_cell_checkpoint(
                checkpoint_path,
                cell=cell,
                protocol_hash=manifest["protocol_sha256"],
                environment_id=environment_id,
                rows=measured,
            )
            rows.extend(measured)
        else:
            rows.extend(cached)

    if len(rows) != EXPECTED_TOTAL_OBSERVATIONS:
        raise V09FError("Executed matrix must retain exactly 1000 observations.")
    validate_observation_rows(rows)
    failures = [{field: row[field] for field in FAILURE_FIELDS} for row in rows if row["status"] == "FAILURE"]
    within_seed = summarize_within_seed(rows)
    summary = summarize_across_seeds(within_seed)
    comparisons = build_resource_comparisons(within_seed)
    model_size = build_model_size_comparison(rows)
    feature_reduction = build_feature_reduction()
    timing_quality = build_timing_quality(rows)
    optimizer_overhead = build_optimizer_overhead()
    break_even = build_break_even(within_seed=within_seed, optimizer_overhead=optimizer_overhead)
    tradeoff = build_tradeoff(summary=summary)

    raw_payload = {
        "schema_version": V09F_SCHEMA_VERSION,
        "stage": V09F_STAGE,
        "artifact_kind": "RAW_TIMESTAMPED_RESOURCE_OBSERVATIONS",
        "observation_level_timestamps_present": True,
        "direct_energy_status": DIRECT_ENERGY_STATUS,
        "rows": rows,
    }
    observation_counts = {
        "training": sum(1 for row in rows if row["phase"] == MeasurementPhase.TRAINING.value),
        "inference": sum(1 for row in rows if row["phase"] == MeasurementPhase.INFERENCE.value),
        "total": len(rows),
        "cells": len(cells),
        "failure_count": len(failures),
    }

    _atomic_write_json(output_root / RAW_JSON_PATH.name, raw_payload)
    _atomic_write_csv(output_root / RAW_CSV_PATH.name, rows, fieldnames=OBSERVATION_FIELDS)
    _atomic_write_json(output_root / FAILURES_JSON_PATH.name, failures)
    _atomic_write_csv(output_root / FAILURES_CSV_PATH.name, failures, fieldnames=FAILURE_FIELDS)
    _atomic_write_csv(output_root / SUMMARY_CSV_PATH.name, summary, fieldnames=SORTED_SUMMARY_FIELDS)
    _atomic_write_json(output_root / SUMMARY_JSON_PATH.name, summary)
    _atomic_write_csv(output_root / COMPARISONS_CSV_PATH.name, flatten_comparisons(comparisons), fieldnames=SORTED_COMPARISON_FIELDS)
    _atomic_write_json(output_root / COMPARISONS_JSON_PATH.name, comparisons)
    _atomic_write_json(output_root / MODEL_SIZE_PATH.name, model_size)
    _atomic_write_json(output_root / FEATURE_REDUCTION_PATH.name, feature_reduction)
    _atomic_write_json(output_root / TIMING_QUALITY_PATH.name, timing_quality)
    _atomic_write_json(output_root / OPTIMIZER_OVERHEAD_PATH.name, optimizer_overhead)
    _atomic_write_json(output_root / BREAK_EVEN_PATH.name, break_even)
    _atomic_write_json(output_root / TRADEOFF_PATH.name, tradeoff)

    artifact_hashes = {
        name: sha256_file(output_root / name)
        for name in (
            RAW_JSON_PATH.name,
            RAW_CSV_PATH.name,
            SUMMARY_JSON_PATH.name,
            SUMMARY_CSV_PATH.name,
            COMPARISONS_JSON_PATH.name,
            COMPARISONS_CSV_PATH.name,
            MODEL_SIZE_PATH.name,
            FEATURE_REDUCTION_PATH.name,
            TIMING_QUALITY_PATH.name,
            OPTIMIZER_OVERHEAD_PATH.name,
            BREAK_EVEN_PATH.name,
            TRADEOFF_PATH.name,
            ENVIRONMENT_PATH.name,
        )
    }
    completed_at = _utc_now()
    proposed_lock = build_result_lock(
        starting_head=preflight["starting_head"],
        preflight=preflight,
        configurations=plans,
        classifiers=classifiers,
        artifact_hashes=artifact_hashes,
        environment_id=environment_id,
        environment_sha256=environment_payload["environment_sha256"],
        observation_counts=observation_counts,
        timing_quality=timing_quality,
        started_at=started_at,
        completed_at=completed_at,
        wall_time_sec=time.perf_counter() - stage_start,
    )
    result_lock = write_or_verify_result_lock(output_root / RESULT_LOCK_PATH.name, proposed_lock)

    execution = {
        "schema_version": V09F_SCHEMA_VERSION,
        "stage": V09F_STAGE,
        "status": "COMPLETED",
        "starting_head": preflight["starting_head"],
        "ending_head": current_head_short(PROJECT_ROOT),
        "environment_id": environment_id,
        "observation_counts": observation_counts,
        "resource_protocol_identity": preflight["resource_protocol_identity"],
        "direct_energy_status": DIRECT_ENERGY_STATUS,
        "timing_interpretation": timing_quality["timing_interpretation"],
        "standby_affected_cell_count": timing_quality["standby_affected_cell_count"],
        "artifact_hashes": {
            **artifact_hashes,
            RESULT_LOCK_PATH.name: sha256_file(output_root / RESULT_LOCK_PATH.name),
        },
        "semantic_result_lock_sha256": result_lock["semantic_result_lock_sha256"],
        "optimizer_invoked": False,
        "optimizer_rerun": False,
        "feature_reselection_performed": False,
        "model_tuning_performed": False,
        "threshold_tuning_performed": False,
        "final_test_accessed": False,
        "outliers_removed": False,
        "readiness_for_v09g": "GO",
    }
    _atomic_write_json(output_root / EXECUTION_PATH.name, execution)
    verify_v09f_artifacts(output_root)
    return execution


SORTED_SUMMARY_FIELDS = (
    "stage",
    "classifier",
    "configuration_id",
    "phase",
    "metric",
    "unit",
    "seed_count",
    "mean_of_seed_means",
    "std_of_seed_means",
    "min_of_seed_means",
    "max_of_seed_means",
    "feature_count",
    "input_dataframe_memory_bytes",
    "numpy_dense_nbytes",
    "serialized_model_bytes",
    "measurement_provenance",
    "aggregation_hierarchy",
)

SORTED_COMPARISON_FIELDS = (
    "stage",
    "comparison_tier",
    "classifier",
    "phase",
    "metric",
    "unit",
    "candidate_configuration_id",
    "reference_configuration_id",
    "candidate_mean",
    "reference_mean",
    "mean_difference",
    "std_difference",
    "mean_percent_change",
    "ci95_low",
    "ci95_high",
    "n",
    "df",
    "t_critical",
    "aggregation_hierarchy",
)


def flatten_comparisons(comparisons: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{field: row.get(field) for field in SORTED_COMPARISON_FIELDS} for row in comparisons]


def verify_v09f_artifacts(output: Path = DEFAULT_OUTPUT_DIR) -> None:
    execution = _read_json(output / EXECUTION_PATH.name)
    lock = _read_json(output / RESULT_LOCK_PATH.name)
    verify_result_lock(lock)
    timing = _read_json(output / TIMING_QUALITY_PATH.name)
    raw = _read_json(output / RAW_JSON_PATH.name)
    preflight = _read_json(output / PREFLIGHT_PATH.name)
    if (
        execution.get("status") != "COMPLETED"
        or int(execution.get("observation_counts", {}).get("total", -1)) != EXPECTED_TOTAL_OBSERVATIONS
        or execution.get("direct_energy_status") != DIRECT_ENERGY_STATUS
        or execution.get("optimizer_invoked") is not False
        or execution.get("optimizer_rerun") is not False
        or execution.get("feature_reselection_performed") is not False
        or execution.get("final_test_accessed") is not False
        or timing.get("observation_level_timestamps_present") is not True
        or timing.get("outliers_removed") is not False
        or int(timing.get("observation_count", -1)) != EXPECTED_TOTAL_OBSERVATIONS
        or raw.get("observation_level_timestamps_present") is not True
        or len(raw.get("rows", ())) != EXPECTED_TOTAL_OBSERVATIONS
        or preflight.get("status") != "PASS"
    ):
        raise V09FError("V0.9-F artifact governance verification failed.")
    for name, expected in execution["artifact_hashes"].items():
        if sha256_file(output / name) != expected:
            raise V09FError(f"V0.9-F artifact hash mismatch: {name}")
    for name, expected in lock["result_artifact_hashes"].items():
        if sha256_file(output / name) != expected:
            raise V09FError(f"V0.9-F result-lock hash mismatch: {name}")


def main() -> int:
    result = run_v09f(execute_matrix=False)
    print(
        json.dumps(
            {
                "stage": V09F_STAGE,
                "status": result["status"],
                "expected_total_observations": result["expected_total_observations"],
                "readiness_for_campaign": result["readiness_for_campaign"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
