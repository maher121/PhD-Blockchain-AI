"""V0.8-E2 resource-benchmark infrastructure locked to the approved E1 protocol.

This module implements infrastructure only:
- frozen protocol/configuration loading
- prerequisite and integrity verification
- model-artifact preparation outside timed boundaries
- benchmark schedule/checkpoint/resume plumbing
- statistical and break-even utilities
- read-only imports for V0.8-C and V0.8-D historical evidence

It does not execute BPSO, predictive final-test evaluation, or adaptive feature
selection/tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import yaml

from src.config import PROJECT_ROOT
from src.green.measurement import (
    CanonicalUnit,
    DEFAULT_WARMUP_CALLS,
    MeasurementPhase,
    MeasurementProvenance,
    WorkerStatus,
    collect_environment_metadata,
    dataframe_deep_memory_bytes,
    numpy_dense_nbytes,
    run_fresh_worker_protocol,
)
from src.lightweight.models import LightweightDetector, create_model
import src.pipeline_v06g as v06g
import src.pipeline_v07b as v07b
import src.pipeline_v07d as v07d
import src.pipeline_v08b as v08b
import src.pipeline_v08d as v08d
from src.security.ground_truth import assert_no_attack_metadata


V08E_STAGE = "V0.8-E2"
V08E_SCHEMA_VERSION = "v0.8-e2-resource-benchmark-1"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "resource_benchmark_v08e.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "bpso"
DEFAULT_MODEL_DIR = DEFAULT_OUTPUT_DIR / "v08e_model_artifacts"
DEFAULT_CHECKPOINT_DIR = DEFAULT_OUTPUT_DIR / ".v08e_checkpoints"

CONFIGURATION_IDS = ("K43", "K42", "K11", "BPSO-K10")
CONFIGURATION_SOURCE_IDS = {
    "K43": "none_natural",
    "K42": "pairwise_correlation_filter_natural",
    "K11": "mutual_information_select_k_best_k11",
    "BPSO-K10": "v08c_locked_bpso_k10",
}
CLASSIFIERS = ("decision_tree", "logistic_regression")
SEEDS = (42, 43, 44, 45, 46)
PHASES = (MeasurementPhase.TRAINING.value, MeasurementPhase.INFERENCE.value)
REPETITIONS = tuple(range(1, 11))

TRAINING_INNER_OPERATION_COUNT = 1
INFERENCE_INNER_OPERATION_COUNT = 512
OUTER_REPETITIONS = 10
WARMUP_CALLS = DEFAULT_WARMUP_CALLS

EXPECTED_TRAINING_OBSERVATIONS = 400
EXPECTED_INFERENCE_OBSERVATIONS = 400
EXPECTED_TOTAL_OBSERVATIONS = 800

EXPECTED_K10 = {
    "selected_feature_count": 10,
    "mask_sha256": "5da981b5b87db97338ecdde9ca8a8b87db3a62771d03dc6a4ad901f6548a3299",
    "selected_features_sha256": "5157d5bb6b17dc6c92b790671dac029e2b0d2b4454db9824870db1290a6c4121",
    "semantic_lock_sha256": "0f356ccab422774b128ab82f5da8881f24816902d4ccc6a7cecc9760c6202ca8",
}
EXPECTED_V08D_RESULT_LOCK_SHA256 = "ac19e5a0dba5d058f88485e6ab8ab9a96f97c30f697e95a744ccaa6563d2b657"
T_CRITICAL_95_DF4 = 2.776445

IMMUTABLE_PATHS = tuple(
    dict.fromkeys(
        (
            *v08d.IMMUTABLE_PATHS,
            PROJECT_ROOT / "src" / "pipeline_v07d.py",
            PROJECT_ROOT / "src" / "pipeline_v07e.py",
            PROJECT_ROOT / "tests" / "test_pipeline_v07d.py",
            PROJECT_ROOT / "tests" / "test_pipeline_v07e.py",
            PROJECT_ROOT / "src" / "pipeline_v08d.py",
            PROJECT_ROOT / "tests" / "test_pipeline_v08d.py",
            PROJECT_ROOT / "results" / "bpso" / "v08d_execution_summary.json",
            PROJECT_ROOT / "results" / "bpso" / "v08d_final_test_lock.json",
            PROJECT_ROOT / "results" / "bpso" / "v08d_final_test_raw.json",
            PROJECT_ROOT / "results" / "bpso" / "v08d_final_test_summary.json",
            PROJECT_ROOT / "results" / "bpso" / "v08d_paired_comparisons.json",
            PROJECT_ROOT / "results" / "bpso" / "v08d_preservation.json",
            PROJECT_ROOT / "results" / "bpso" / "v08d_test_access_audit.json",
        )
    )
)

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
    "v08d_semantic_result_lock_sha256",
    "status",
    "failure_type",
    "failure_stage",
    "failure_reason",
)

FAILURE_FIELDS = (
    "observation_id",
    "environment_id",
    "classifier",
    "configuration_id",
    "seed",
    "phase",
    "outer_repetition",
    "failure_type",
    "failure_stage",
    "failure_reason",
)


class V08EError(RuntimeError):
    """Base error for the V0.8-E2 benchmark infrastructure."""


class V08EIntegrityError(V08EError):
    """Raised when frozen evidence or protocol lock checks fail."""


class V08ECheckpointError(V08EError):
    """Raised when checkpoint/resume compatibility checks fail."""


@dataclass(frozen=True)
class PlannedObservation:
    observation_id: str
    classifier: str
    configuration_id: str
    source_configuration_id: str
    seed: int
    phase: str
    outer_repetition: int
    inner_operation_count: int


@dataclass(frozen=True)
class BenchmarkCell:
    classifier: str
    configuration_id: str
    source_configuration_id: str
    seed: int
    phase: str
    inner_operation_count: int


@dataclass(frozen=True)
class InferenceFeatureWorkload:
    """Feature-only inference workload (labels and ground truth intentionally absent)."""

    seed: int
    features: pd.DataFrame
    workload_sha256: str
    row_ids_sha256: str
    feature_matrix_sha256: str


@dataclass(frozen=True)
class PreparedModelArtifact:
    classifier: str
    configuration_id: str
    source_configuration_id: str
    seed: int
    path: Path
    file_sha256: str
    model_state_sha256: str
    serialized_model_bytes: int
    feature_names: tuple[str, ...]
    feature_manifest_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical_lf_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return normalized if normalized.endswith("\n") else normalized + "\n"


def canonical_lf_sha256(path: Path | str) -> str:
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return sha256_file(target)
    except OSError as exc:
        raise V08EIntegrityError(f"Cannot read required artifact for hashing: {target}") from exc
    return hashlib.sha256(_canonical_lf_text(text).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    target = Path(path)
    try:
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise V08EIntegrityError(f"Cannot hash required artifact: {target}") from exc
    return digest.hexdigest()


def _read_json(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V08EIntegrityError(f"Cannot read JSON artifact {target}: {exc}") from exc
    if not isinstance(payload, dict):
        raise V08EIntegrityError(f"JSON artifact must be an object: {target}")
    return payload


def _atomic_write_json(path: Path | str, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _atomic_write_csv(
    path: Path | str,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    v07d.atomic_write_csv(path, rows, fieldnames=fieldnames)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def load_v08e_protocol(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the frozen V0.8-E2 benchmark protocol configuration."""

    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise V08EIntegrityError(f"Cannot load V0.8-E2 config: {config_path}") from exc
    if not isinstance(payload, dict):
        raise V08EIntegrityError("V0.8-E2 config must be a YAML mapping.")
    measurement = payload.get("measurement", {})
    required = {
        "version": (payload.get("version"), V08E_SCHEMA_VERSION),
        "stage": (payload.get("stage"), V08E_STAGE),
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
        "worker_policy": (
            measurement.get("worker_policy"),
            "fresh_process_per_outer_repetition",
        ),
        "adaptive_calibration": (measurement.get("adaptive_calibration"), False),
    }
    drift = {
        key: {"observed": observed, "expected": expected}
        for key, (observed, expected) in required.items()
        if observed != expected
    }
    if drift:
        raise V08EIntegrityError(f"V0.8-E2 protocol drift detected: {drift}")
    return payload


def snapshot_immutable_paths(paths: Sequence[Path] = IMMUTABLE_PATHS) -> dict[str, str]:
    """Hash frozen prerequisites using canonical LF hashing where possible."""

    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise V08EIntegrityError(f"Missing frozen prerequisite artifacts: {missing}")
    return {str(Path(path).resolve()): canonical_lf_sha256(path) for path in paths}


def _verify_k43_k42_k11_identity(
    basis: v08b.FrozenBasis,
    plans: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {plan["configuration_id"]: plan for plan in plans}
    if tuple(by_id) != CONFIGURATION_IDS:
        raise V08EIntegrityError("Configuration identity order must be K43/K42/K11/BPSO-K10.")

    k43 = by_id["K43"]
    k42 = by_id["K42"]
    k11 = by_id["K11"]
    k43_sets = {tuple(k43["features_by_seed"][str(seed)]) for seed in SEEDS}
    k42_sets = {tuple(k42["features_by_seed"][str(seed)]) for seed in SEEDS}
    k11_sets = {tuple(k11["features_by_seed"][str(seed)]) for seed in SEEDS}
    if len(k43_sets) != 1 or len(next(iter(k43_sets))) != 43:
        raise V08EIntegrityError("K43 identity failed (feature count/order mismatch).")
    if len(k42_sets) != 1 or len(next(iter(k42_sets))) != 42:
        raise V08EIntegrityError("K42 identity failed (deterministic 42-feature set required).")
    if len(k11_sets) <= 1 or any(len(item) != 11 for item in k11_sets):
        raise V08EIntegrityError("MI-K11 identity failed (seed-specific 11-feature semantics required).")

    k43_features = next(iter(k43_sets))
    k42_features = next(iter(k42_sets))
    return {
        "k43": {
            "feature_count": 43,
            "features_sha256": v08d.fingerprint_feature_names(k43_features),
            "candidate_manifest_sha256": basis.candidate_manifest_sha256,
        },
        "k42": {
            "feature_count": 42,
            "features_sha256": v08d.fingerprint_feature_names(k42_features),
            "mask_sha256": basis.k42_mask_sha256,
        },
        "k11": {
            "feature_count": 11,
            "seed_specific": True,
            "seed_feature_hashes": {
                str(seed): v08d.fingerprint_feature_names(k11["features_by_seed"][str(seed)])
                for seed in SEEDS
            },
        },
    }


def verify_v08e_prerequisites(
    *,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    winner_lock_path: Path | str = v08d.WINNER_LOCK_PATH,
    result_lock_path: Path | str = v08d.RESULT_LOCK_PATH,
) -> dict[str, Any]:
    """Verify frozen V0.6/V0.7/V0.8-C/V0.8-D integrity and lock identities."""

    protocol = load_v08e_protocol(config_path)
    immutable_hashes = snapshot_immutable_paths()
    basis = v08b.load_frozen_basis()
    winner_lock = v08d.load_verified_winner_lock(Path(winner_lock_path))
    result_lock = _read_json(result_lock_path)
    v08d.verify_final_test_lock(result_lock)

    if winner_lock.get("semantic_lock_sha256") != EXPECTED_K10["semantic_lock_sha256"]:
        raise V08EIntegrityError("V0.8-C semantic lock hash is not the approved frozen value.")
    if result_lock.get("semantic_result_lock_sha256") != EXPECTED_V08D_RESULT_LOCK_SHA256:
        raise V08EIntegrityError("V0.8-D semantic result lock hash is not the approved frozen value.")
    for key, expected in EXPECTED_K10.items():
        if winner_lock.get(key) != expected:
            raise V08EIntegrityError(f"Approved K10 identity drift: {key}")

    plans = v08d.build_configuration_plan(basis, winner_lock)
    identity = _verify_k43_k42_k11_identity(basis, plans)

    execution_c = _read_json(PROJECT_ROOT / "results" / "bpso" / "v08c_execution_summary.json")
    execution_d = _read_json(PROJECT_ROOT / "results" / "bpso" / "v08d_execution_summary.json")
    if execution_c.get("status") != "COMPLETED" or execution_d.get("status") != "COMPLETED":
        raise V08EIntegrityError("Frozen V0.8-C/V0.8-D execution summaries are incomplete.")
    if execution_d.get("readiness_for_v08e") != "GO":
        raise V08EIntegrityError("V0.8-D readiness for V0.8-E must be GO.")

    return {
        "schema_version": V08E_SCHEMA_VERSION,
        "stage": V08E_STAGE,
        "status": "PASS",
        "protocol_config": protocol,
        "protocol_config_path": str(Path(config_path).resolve()),
        "protocol_config_lf_sha256": canonical_lf_sha256(config_path),
        "immutable_hashes": immutable_hashes,
        "winner_lock_sha256": sha256_file(winner_lock_path),
        "winner_semantic_lock_sha256": winner_lock["semantic_lock_sha256"],
        "result_lock_sha256": sha256_file(result_lock_path),
        "result_semantic_lock_sha256": result_lock["semantic_result_lock_sha256"],
        "k10_identity": {
            "feature_count": winner_lock["selected_feature_count"],
            "mask_sha256": winner_lock["mask_sha256"],
            "selected_features_sha256": winner_lock["selected_features_sha256"],
        },
        "configuration_identity": identity,
        "basis": basis,
        "winner_lock": winner_lock,
        "configuration_plan": list(plans),
    }


def _protocol_payload(preconditions: Mapping[str, Any]) -> dict[str, Any]:
    config = preconditions["protocol_config"]
    measurement = config["measurement"]
    return {
        "stage": V08E_STAGE,
        "schema_version": V08E_SCHEMA_VERSION,
        "configurations": list(CONFIGURATION_IDS),
        "classifiers": list(CLASSIFIERS),
        "seeds": list(SEEDS),
        "phases": list(PHASES),
        "outer_repetitions": OUTER_REPETITIONS,
        "warmup_calls": WARMUP_CALLS,
        "training_inner_operation_count": TRAINING_INNER_OPERATION_COUNT,
        "inference_inner_operation_count": INFERENCE_INNER_OPERATION_COUNT,
        "adaptive_calibration": False,
        "worker_policy": measurement["worker_policy"],
        "training_boundary": measurement["training_boundary"],
        "inference_boundary": measurement["inference_boundary"],
        "direct_energy_decision": "DIRECT_ENERGY_UNAVAILABLE",
        "v08c_semantic_lock_sha256": preconditions["winner_semantic_lock_sha256"],
        "v08d_semantic_result_lock_sha256": preconditions["result_semantic_lock_sha256"],
    }


def protocol_sha256(preconditions: Mapping[str, Any]) -> str:
    payload = _protocol_payload(preconditions)
    return _json_sha256(payload)


def build_run_manifest(preconditions: Mapping[str, Any], environment_id: str) -> dict[str, Any]:
    """Build the exact future 800-observation manifest without executing it."""

    expected = (
        len(CONFIGURATION_IDS)
        * len(CLASSIFIERS)
        * len(SEEDS)
        * len(PHASES)
        * OUTER_REPETITIONS
    )
    manifest = {
        "schema_version": V08E_SCHEMA_VERSION,
        "stage": V08E_STAGE,
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
        "expected_total_observations": expected,
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
        or int(manifest.get("training_inner_operation_count", -1))
        != TRAINING_INNER_OPERATION_COUNT
        or int(manifest.get("inference_inner_operation_count", -1))
        != INFERENCE_INNER_OPERATION_COUNT
        or manifest.get("adaptive_calibration") is not False
        or int(manifest.get("expected_total_observations", -1))
        != EXPECTED_TOTAL_OBSERVATIONS
    ):
        raise V08EIntegrityError("Run manifest does not match the frozen 800-observation design.")


def configure_single_thread_environment() -> dict[str, str]:
    return v07d.configure_single_thread_environment()


def resolve_energy_capability(
    capability_path: Path | str = PROJECT_ROOT / "results" / "green_evaluation" / "energy_capability.json",
) -> dict[str, Any]:
    payload = _read_json(capability_path)
    decision = payload.get("direct_energy_decision")
    if decision not in {"DIRECT_ENERGY_AVAILABLE", "DIRECT_ENERGY_UNAVAILABLE"}:
        raise V08EIntegrityError("V0.7-C capability decision is invalid.")
    return {
        "decision": decision,
        "accepted_backend": payload.get("accepted_direct_energy_backend"),
        "measurement_backend": payload.get("classification_policy", {}),
        "provenance": MeasurementProvenance.DIRECT_ENERGY.value
        if decision == "DIRECT_ENERGY_AVAILABLE"
        else "DIRECT_ENERGY_UNAVAILABLE",
    }


def build_environment_identity(
    *,
    repository_path: Path | str = PROJECT_ROOT,
    capability_path: Path | str = PROJECT_ROOT / "results" / "green_evaluation" / "energy_capability.json",
) -> tuple[dict[str, Any], str]:
    """Collect deterministic environment metadata and compute a compatibility ID."""

    metadata = collect_environment_metadata(repository_path)
    capability = resolve_energy_capability(capability_path)
    dependencies = metadata.get("dependencies", {})
    payload = {
        "os": metadata.get("os", {}),
        "python": metadata.get("python", {}),
        "numpy": dependencies.get("numpy"),
        "pandas": dependencies.get("pandas"),
        "scikit_learn": dependencies.get("scikit-learn"),
        "cpu": metadata.get("cpu", {}),
        "execution_environment": metadata.get("execution_environment", {}),
        "thread_environment": metadata.get("thread_environment", {}),
        "measurement_backend": capability,
    }
    return payload, _json_sha256(payload)


def build_observation_schedule(
    *,
    configurations: Sequence[str] = CONFIGURATION_IDS,
    classifiers: Sequence[str] = CLASSIFIERS,
    seeds: Sequence[int] = SEEDS,
    phases: Sequence[str] = PHASES,
    outer_repetitions: int = OUTER_REPETITIONS,
) -> tuple[PlannedObservation, ...]:
    """Build a deterministic, cyclically balanced 800-observation schedule."""

    if tuple(configurations) != CONFIGURATION_IDS:
        raise V08EIntegrityError("Schedule configurations must be exactly K43/K42/K11/BPSO-K10.")
    if tuple(classifiers) != CLASSIFIERS or tuple(seeds) != SEEDS:
        raise V08EIntegrityError("Schedule classifiers/seeds drift from the frozen protocol.")
    if tuple(phases) != PHASES or outer_repetitions != OUTER_REPETITIONS:
        raise V08EIntegrityError("Schedule phases/repetitions drift from the frozen protocol.")

    rows: list[PlannedObservation] = []
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
                            PlannedObservation(
                                observation_id=(
                                    f"v08e__{phase}__{classifier}__{configuration_id}"
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


def _validate_schedule(schedule: Sequence[PlannedObservation]) -> None:
    training = sum(item.phase == MeasurementPhase.TRAINING.value for item in schedule)
    inference = sum(item.phase == MeasurementPhase.INFERENCE.value for item in schedule)
    if (training, inference, len(schedule)) != (
        EXPECTED_TRAINING_OBSERVATIONS,
        EXPECTED_INFERENCE_OBSERVATIONS,
        EXPECTED_TOTAL_OBSERVATIONS,
    ):
        raise V08EIntegrityError("Schedule must contain exactly 400 training and 400 inference observations.")
    ids = [item.observation_id for item in schedule]
    if len(set(ids)) != len(ids):
        raise V08EIntegrityError("Planned observation IDs must be unique.")


def schedule_cells(schedule: Sequence[PlannedObservation]) -> tuple[BenchmarkCell, ...]:
    cells: list[BenchmarkCell] = []
    for index in range(0, len(schedule), OUTER_REPETITIONS):
        block = schedule[index : index + OUTER_REPETITIONS]
        if len(block) != OUTER_REPETITIONS:
            raise V08EIntegrityError("Schedule includes an incomplete benchmark cell block.")
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
            raise V08EIntegrityError("Each benchmark cell must contain repetitions 1-10 once.")
        cells.append(BenchmarkCell(*next(iter(identity))))
    if len(cells) != 80:
        raise V08EIntegrityError("V0.8-E2 requires exactly 80 benchmark cells.")
    return tuple(cells)


def build_features_only_inference_workloads(
    workloads: v07b.PreparedBenchmarkWorkloads,
    *,
    candidate_features: Sequence[str],
) -> dict[int, InferenceFeatureWorkload]:
    """Extract governed feature-only inference matrices from prepared workloads."""

    candidates = tuple(candidate_features)
    output: dict[int, InferenceFeatureWorkload] = {}
    for seed in SEEDS:
        manifestation = workloads.get(seed, MeasurementPhase.INFERENCE.value)
        features = manifestation.features.loc[:, ["row_id", *candidates]].copy()
        assert_no_attack_metadata(features)
        output[seed] = InferenceFeatureWorkload(
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
) -> dict[tuple[str, str, int], PreparedModelArtifact]:
    """Prepare 40 train-only model artifacts outside benchmark timing boundaries."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    by_key: dict[tuple[str, str, int], PreparedModelArtifact] = {}
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
                    model = model_factory(
                        classifier,
                        features,
                        parameters,
                        random_state=seed,
                    )
                    model.fit(selected_train, training.labels.copy())
                    model.save(model_path)
                if (
                    model.model_name != classifier
                    or model.random_state != seed
                    or tuple(model.feature_names) != features
                    or dict(model.parameters) != parameters
                ):
                    raise V08EIntegrityError("Prepared inference model artifact identity verification failed.")
                by_key[(configuration_id, classifier, seed)] = PreparedModelArtifact(
                    classifier=classifier,
                    configuration_id=configuration_id,
                    source_configuration_id=source_configuration_id,
                    seed=seed,
                    path=model_path,
                    file_sha256=sha256_file(model_path),
                    model_state_sha256=v06g.model_state_sha256(model),
                    serialized_model_bytes=model_path.stat().st_size,
                    feature_names=features,
                    feature_manifest_sha256=v08d.fingerprint_feature_names(features),
                )
    if len(by_key) != 40:
        raise V08EIntegrityError("Model-artifact preparation must produce exactly 40 artifacts.")
    return by_key


def _run_cell_measurement(
    *,
    context: v07b.V07BContext,
    cell: BenchmarkCell,
    configuration_plan: Mapping[str, Any],
    model_artifact: PreparedModelArtifact,
    workloads: v07b.PreparedBenchmarkWorkloads,
    inference_workload: InferenceFeatureWorkload,
    environment_id: str,
    v08c_semantic_lock_sha256: str,
    v08d_semantic_result_lock_sha256: str,
    runner: Callable[..., Any] = run_fresh_worker_protocol,
    start_method: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Measure one complete benchmark cell (10 retained repetitions)."""

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

    run = runner(
        operation,
        prepare=prepare,
        inner_operation_count=cell.inner_operation_count,
        phase=cell.phase,
        warmup_calls=WARMUP_CALLS,
        outer_repetitions=OUTER_REPETITIONS,
        records_per_operation=records,
        timeout_seconds=300.0,
        measurement_scope=scope,
        start_method=start_method,
    )
    rows: list[dict[str, Any]] = []
    for result in run.results:
        success = result.status is WorkerStatus.COMPLETED and result.observation is not None
        observation = result.observation
        row = {
            "observation_id": (
                f"v08e__{cell.phase}__{cell.classifier}__{cell.configuration_id}"
                f"__s{cell.seed}__r{result.outer_repetition:02d}"
            ),
            "schema_version": V08E_SCHEMA_VERSION,
            "stage": V08E_STAGE,
            "environment_id": environment_id,
            "classifier": cell.classifier,
            "configuration_id": cell.configuration_id,
            "source_configuration_id": cell.source_configuration_id,
            "seed": cell.seed,
            "phase": cell.phase,
            "outer_repetition": result.outer_repetition,
            "inner_operation_count": cell.inner_operation_count,
            "measured_operation_count": result.measured_operation_count,
            "record_count": records,
            "measured_record_count": result.measured_operation_count * records,
            "feature_count": len(features),
            "warmup_calls": run.protocol.warmup_calls,
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
            "v08d_semantic_result_lock_sha256": v08d_semantic_result_lock_sha256,
            "status": "SUCCESS" if success else "FAILURE",
            "failure_type": None if success else (None if result.failure is None else result.failure.error_type),
            "failure_stage": None if success else (None if result.failure is None else result.failure.stage),
            "failure_reason": None if success else (None if result.failure is None else result.failure.message),
        }
        rows.append(row)
    validate_observation_rows(rows)
    return tuple(rows)


def validate_observation_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise V08EIntegrityError("Observation rows cannot be empty.")
    identifiers: list[str] = []
    for row in rows:
        missing = set(OBSERVATION_FIELDS) - set(row)
        if missing:
            raise V08EIntegrityError(f"Observation row is missing fields: {sorted(missing)}")
        identifiers.append(str(row["observation_id"]))
        if row["schema_version"] != V08E_SCHEMA_VERSION or row["stage"] != V08E_STAGE:
            raise V08EIntegrityError("Observation schema/stage mismatch.")
        if row["measurement_provenance"] != MeasurementProvenance.DIRECT_COMPUTATIONAL.value:
            raise V08EIntegrityError("Observation provenance must be DIRECT_COMPUTATIONAL.")
        if row["derived_measurement_provenance"] != MeasurementProvenance.DERIVED.value:
            raise V08EIntegrityError("Derived observation provenance must be DERIVED.")
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
                    raise V08EIntegrityError("Successful observations require finite nonnegative metrics.")
            if int(row["measured_operation_count"]) != int(row["inner_operation_count"]):
                raise V08EIntegrityError("Measured operation count must equal configured inner operations.")
            if row["warmup_calls"] != WARMUP_CALLS:
                raise V08EIntegrityError("Warmup count drifted from the frozen protocol.")
            if any(row[name] is not None for name in ("failure_type", "failure_stage", "failure_reason")):
                raise V08EIntegrityError("Successful observations cannot include failure details.")
        elif row["status"] == "FAILURE":
            if not row["failure_type"] or not row["failure_stage"] or not row["failure_reason"]:
                raise V08EIntegrityError("Failed observations require structured failure details.")
        else:
            raise V08EIntegrityError("Observation status must be SUCCESS or FAILURE.")
    if len(set(identifiers)) != len(identifiers):
        raise V08EIntegrityError("Observation IDs must be unique.")


def _checkpoint_name(cell: BenchmarkCell) -> str:
    return f"{cell.phase}__{cell.classifier}__{cell.configuration_id}__s{cell.seed}.json"


def load_valid_checkpoint(
    path: Path,
    *,
    cell: BenchmarkCell,
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
        raise V08ECheckpointError("Checkpoint is incompatible with this protocol/environment/cell.")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise V08ECheckpointError("Checkpoint rows are missing or malformed.")
    if len(rows) != expected_repetitions:
        raise V08ECheckpointError("Checkpoint must contain one complete 10-repetition cell.")
    repetitions = tuple(int(row["outer_repetition"]) for row in rows)
    if repetitions != REPETITIONS:
        raise V08ECheckpointError("Checkpoint row repetitions must be exactly 1-10.")
    validate_observation_rows(rows)
    return tuple(rows)


def write_cell_checkpoint(
    path: Path,
    *,
    cell: BenchmarkCell,
    protocol_hash: str,
    environment_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if len(rows) != OUTER_REPETITIONS:
        raise V08ECheckpointError("Only complete benchmark cells are checkpointed.")
    payload = {
        "stage": V08E_STAGE,
        "schema_version": V08E_SCHEMA_VERSION,
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
    output: list[dict[str, Any]] = []
    metrics = (
        "wall_time_sec",
        "process_cpu_time_sec",
        "absolute_peak_rss_mib",
        "incremental_peak_rss_mib",
        "inference_latency_sec",
        "per_record_latency_sec",
        "throughput_records_sec",
    )
    for (classifier, configuration_id, seed, phase), group in sorted(groups.items()):
        if len(group) != OUTER_REPETITIONS:
            raise V08EIntegrityError("Within-seed summaries require exactly 10 repetitions per cell.")
        repetitions = {int(row["outer_repetition"]) for row in group}
        if repetitions != set(REPETITIONS):
            raise V08EIntegrityError("Within-seed summaries must use repetitions 1-10 exactly once.")
        if any(row["status"] != "SUCCESS" for row in group):
            raise V08EIntegrityError("Within-seed summaries require successful repetitions.")
        static_feature_count = {int(row["feature_count"]) for row in group}
        static_input = {int(row["input_dataframe_memory_bytes"]) for row in group}
        static_dense = {int(row["numpy_dense_nbytes"]) for row in group}
        static_model = {int(row["serialized_model_bytes"]) for row in group}
        if any(len(item) != 1 for item in (static_feature_count, static_input, static_dense, static_model)):
            raise V08EIntegrityError("Static resource values changed within a repetition block.")
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            output.append(
                {
                    "stage": V08E_STAGE,
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
                    "coefficient_of_variation_percent": coefficient_of_variation(values),
                    "feature_count": next(iter(static_feature_count)),
                    "input_dataframe_memory_bytes": next(iter(static_input)),
                    "numpy_dense_nbytes": next(iter(static_dense)),
                    "serialized_model_bytes": next(iter(static_model)),
                    "measurement_provenance": MeasurementProvenance.DIRECT_COMPUTATIONAL.value,
                    "aggregation_scope": "ten_measurement_repetitions_within_seed",
                }
            )
    return output


def coefficient_of_variation(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise V08EIntegrityError("Coefficient of variation requires at least two values.")
    mean = statistics.fmean(values)
    if math.isclose(mean, 0.0, abs_tol=1e-15):
        return 0.0 if all(math.isclose(value, 0.0, abs_tol=1e-15) for value in values) else math.inf
    return 100.0 * statistics.stdev(values) / abs(mean)


def paired_t95_interval(values: Sequence[float]) -> tuple[float, float]:
    if len(values) != 5:
        raise V08EIntegrityError("Paired t95 interval requires exactly five seed-level values.")
    mean = statistics.fmean(values)
    margin = T_CRITICAL_95_DF4 * statistics.stdev(values) / math.sqrt(5.0)
    return mean - margin, mean + margin


def relative_change_percent(candidate: float, baseline: float) -> float:
    if math.isclose(baseline, 0.0, abs_tol=1e-15):
        raise V08EIntegrityError("Relative change is undefined for zero baseline.")
    return 100.0 * (candidate - baseline) / baseline


def paired_seed_differences(
    seed_summaries: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str = "BPSO-K10",
    baseline_ids: Sequence[str] = ("K43", "K42", "K11"),
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, int, str, str], float] = {}
    for row in seed_summaries:
        by_key[(
            row["classifier"],
            row["configuration_id"],
            int(row["seed"]),
            row["phase"],
            row["metric"],
        )] = float(row["mean"])
    metrics = sorted({str(row["metric"]) for row in seed_summaries})
    output: list[dict[str, Any]] = []
    for classifier in CLASSIFIERS:
        for phase in PHASES:
            for baseline in baseline_ids:
                for metric in metrics:
                    differences: list[float] = []
                    percent_changes: list[float] = []
                    seed_pairs: list[dict[str, Any]] = []
                    for seed in SEEDS:
                        candidate = by_key[(classifier, candidate_id, seed, phase, metric)]
                        baseline_value = by_key[(classifier, baseline, seed, phase, metric)]
                        difference = candidate - baseline_value
                        percent = relative_change_percent(candidate, baseline_value)
                        differences.append(difference)
                        percent_changes.append(percent)
                        seed_pairs.append(
                            {
                                "seed": seed,
                                "candidate_value": candidate,
                                "baseline_value": baseline_value,
                                "difference": difference,
                                "percent_change": percent,
                            }
                        )
                    ci_low, ci_high = paired_t95_interval(differences)
                    output.append(
                        {
                            "stage": V08E_STAGE,
                            "classifier": classifier,
                            "phase": phase,
                            "metric": metric,
                            "configuration_id": candidate_id,
                            "reference_configuration_id": baseline,
                            "seed_pairs": seed_pairs,
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


def break_even_repetitions(
    *,
    overhead_value: float,
    recurring_saving: float,
    overhead_unit: str,
    saving_unit: str,
) -> dict[str, Any]:
    """Return descriptive break-even repetitions; only positive savings are applicable."""

    if overhead_unit != saving_unit:
        raise V08EIntegrityError("Break-even calculation requires matching units (no wall/CPU mixing).")
    if recurring_saving <= 0.0:
        return {
            "status": "NOT_APPLICABLE",
            "break_even_repetitions": None,
            "overhead": overhead_value,
            "recurring_saving": recurring_saving,
            "unit": overhead_unit,
        }
    return {
        "status": "APPLICABLE",
        "break_even_repetitions": overhead_value / recurring_saving,
        "overhead": overhead_value,
        "recurring_saving": recurring_saving,
        "unit": overhead_unit,
    }


def import_v08c_optimizer_overhead(
    path: Path | str = PROJECT_ROOT / "results" / "bpso" / "v08c_execution_summary.json",
) -> dict[str, Any]:
    """Read-only import of frozen V0.8-C optimization overhead evidence."""

    payload = _read_json(path)
    if payload.get("stage") != "V0.8-C" or payload.get("status") != "COMPLETED":
        raise V08EIntegrityError("V0.8-C optimization summary is not complete.")
    imported = {
        "stage": V08E_STAGE,
        "measurement_provenance": MeasurementProvenance.IMPORTED_HISTORICAL.value,
        "source_stage": "V0.8-C",
        "source_artifact": str(Path(path).resolve()),
        "source_artifact_sha256": sha256_file(path),
        "optimizer_runs": int(payload["completed_run_count"]),
        "fitness_requests": int(payload["fitness_requests"]),
        "unique_evaluations": int(payload["unique_evaluations"]),
        "cache_hits": int(payload["cache_hits"]),
        "decision_tree_fits": int(payload["actual_decision_tree_fits"]),
        "core_optimization_wall_time_sec": float(payload["optimization_wall_time_sec"]),
        "core_optimization_cpu_time_sec": float(payload["optimization_cpu_time_sec"]),
        "pipeline_wall_time_sec": float(payload["pipeline_wall_time_sec"]),
    }
    expected = {
        "optimizer_runs": 5,
        "fitness_requests": 972,
        "unique_evaluations": 972,
        "cache_hits": 0,
        "decision_tree_fits": 4860,
    }
    for key, value in expected.items():
        if imported[key] != value:
            raise V08EIntegrityError(f"Frozen V0.8-C overhead mismatch for {key}: {imported[key]} != {value}")
    return imported


def import_v08d_predictive_evidence(
    *,
    summary_path: Path | str = PROJECT_ROOT / "results" / "bpso" / "v08d_final_test_summary.json",
    lock_path: Path | str = PROJECT_ROOT / "results" / "bpso" / "v08d_final_test_lock.json",
) -> dict[str, Any]:
    """Read-only import of frozen V0.8-D predictive evidence."""

    summary = _read_json(summary_path)
    lock = _read_json(lock_path)
    v08d.verify_final_test_lock(lock)
    if (
        summary.get("stage") != "V0.8-D"
        or summary.get("artifact_kind") != "FINAL_TEST_AGGREGATION"
        or lock.get("semantic_result_lock_sha256") != EXPECTED_V08D_RESULT_LOCK_SHA256
    ):
        raise V08EIntegrityError("Frozen V0.8-D predictive evidence is invalid.")
    rows = summary.get("summaries")
    if not isinstance(rows, list) or not rows:
        raise V08EIntegrityError("V0.8-D summary rows are missing.")
    if any(int(row.get("n", -1)) != 5 for row in rows):
        raise V08EIntegrityError("V0.8-D predictive evidence must preserve n=5 seed summaries.")
    return {
        "stage": V08E_STAGE,
        "measurement_provenance": MeasurementProvenance.IMPORTED_HISTORICAL.value,
        "source_stage": "V0.8-D",
        "source_summary_artifact": str(Path(summary_path).resolve()),
        "source_summary_sha256": sha256_file(summary_path),
        "source_result_lock_artifact": str(Path(lock_path).resolve()),
        "source_result_lock_sha256": sha256_file(lock_path),
        "source_semantic_result_lock_sha256": lock["semantic_result_lock_sha256"],
        "summaries": rows,
    }


def run_v08e(
    *,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    checkpoint_dir: Path | str = DEFAULT_CHECKPOINT_DIR,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    execute_matrix: bool = False,
    start_method: str | None = None,
) -> dict[str, Any]:
    """Run V0.8-E2 infrastructure checks and optionally execute the full matrix.

    The default `execute_matrix=False` prevents accidental scientific execution
    before external review.
    """

    output_root = Path(output_dir)
    checkpoint_root = Path(checkpoint_dir)
    model_root = Path(model_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    preconditions = verify_v08e_prerequisites(config_path=config_path)
    thread_controls = configure_single_thread_environment()
    environment_metadata, environment_id = build_environment_identity()
    if environment_metadata.get("thread_environment") != thread_controls:
        raise V08EIntegrityError("Single-thread controls were not applied before environment capture.")
    manifest = build_run_manifest(preconditions, environment_id)
    preflight = {
        "schema_version": V08E_SCHEMA_VERSION,
        "stage": V08E_STAGE,
        "status": "PREFLIGHT_PASS",
        "timestamp_utc": _utc_now(),
        "environment_id": environment_id,
        "environment": environment_metadata,
        "protocol": _protocol_payload(preconditions),
        "protocol_sha256": manifest["protocol_sha256"],
        "config_lf_sha256": preconditions["protocol_config_lf_sha256"],
        "winner_semantic_lock_sha256": preconditions["winner_semantic_lock_sha256"],
        "result_semantic_lock_sha256": preconditions["result_semantic_lock_sha256"],
        "immutable_hashes": preconditions["immutable_hashes"],
        "checks": [
            {"name": "k10_identity_locked", "passed": True},
            {"name": "k43_k42_k11_identities_locked", "passed": True},
            {"name": "v08c_v08d_semantic_locks_verified", "passed": True},
            {"name": "canonical_lf_scientific_hashing_enabled", "passed": True},
            {"name": "fresh_worker_protocol_locked", "passed": True},
            {"name": "final_test_predictive_rerun_prohibited", "passed": True},
            {"name": "bpso_rerun_prohibited", "passed": True},
        ],
    }
    _atomic_write_json(output_root / "v08e_preflight.json", preflight)
    _atomic_write_json(output_root / "v08e_environment_metadata.json", environment_metadata)
    _atomic_write_json(output_root / "v08e_run_manifest.json", manifest)

    if not execute_matrix:
        return {
            "schema_version": V08E_SCHEMA_VERSION,
            "stage": V08E_STAGE,
            "status": "INFRASTRUCTURE_READY",
            "execute_matrix": False,
            "environment_id": environment_id,
            "expected_total_observations": EXPECTED_TOTAL_OBSERVATIONS,
            "manifest_sha256": sha256_file(output_root / "v08e_run_manifest.json"),
            "readiness_for_v08e3": "GO",
        }

    context = v07b.load_v06_benchmark_context()
    workloads = v07b.prepare_verified_workloads(context)
    inference_workloads = build_features_only_inference_workloads(
        workloads,
        candidate_features=context.plan.candidate_features,
    )
    plans = preconditions["configuration_plan"]
    plan_by_id = {plan["configuration_id"]: plan for plan in plans}
    model_artifacts = prepare_inference_model_artifacts(
        output_dir=model_root,
        configuration_plan=plans,
        workloads=workloads,
    )
    _atomic_write_json(
        output_root / "v08e_model_artifact_manifest.json",
        {
            "schema_version": V08E_SCHEMA_VERSION,
            "stage": V08E_STAGE,
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
                v08c_semantic_lock_sha256=preconditions["winner_semantic_lock_sha256"],
                v08d_semantic_result_lock_sha256=preconditions["result_semantic_lock_sha256"],
                start_method=start_method,
            )
            measured_ids = tuple(row["observation_id"] for row in measured)
            expected_ids = tuple(item.observation_id for item in block)
            if measured_ids != expected_ids:
                raise V08EIntegrityError("Measured observation IDs differ from deterministic schedule.")
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
        raise V08EIntegrityError("Executed matrix must retain exactly 800 observations.")
    validate_observation_rows(rows)
    failures = [
        {field: row[field] for field in FAILURE_FIELDS}
        for row in rows
        if row["status"] == "FAILURE"
    ]
    _atomic_write_json(output_root / "v08e_computational_observations.json", rows)
    _atomic_write_csv(
        output_root / "v08e_computational_observations.csv",
        rows,
        fieldnames=OBSERVATION_FIELDS,
    )
    _atomic_write_json(output_root / "v08e_measurement_failures.json", failures)
    _atomic_write_csv(
        output_root / "v08e_measurement_failures.csv",
        failures,
        fieldnames=FAILURE_FIELDS,
    )
    return {
        "schema_version": V08E_SCHEMA_VERSION,
        "stage": V08E_STAGE,
        "status": "COMPLETED",
        "environment_id": environment_id,
        "retained_observations": len(rows),
        "failure_count": len(failures),
        "readiness_for_v08e3": "GO" if not failures else "REVISE",
    }


def run_engineering_smoke_only(
    *,
    output_dir: Path | str = PROJECT_ROOT / "results" / "bpso" / "v08e_engineering_smoke",
) -> dict[str, Any]:
    """Record a non-scientific smoke marker without generating benchmark evidence."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": V08E_SCHEMA_VERSION,
        "stage": V08E_STAGE,
        "label": "ENGINEERING_SMOKE_ONLY",
        "status": "COMPLETED",
        "timestamp_utc": _utc_now(),
        "merged_into_scientific_observations": False,
        "predictive_labels_accessed": False,
    }
    _atomic_write_json(destination / "v08e_engineering_smoke_only.json", payload)
    return payload


def main() -> int:
    result = run_v08e(execute_matrix=False)
    print(
        json.dumps(
            {
                "stage": V08E_STAGE,
                "status": result["status"],
                "expected_total_observations": result["expected_total_observations"],
                "readiness_for_v08e3": result["readiness_for_v08e3"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
