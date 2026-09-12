"""Controlled Tier-1 computational-efficiency experiments for V0.7-D.

The pipeline consumes the immutable V0.6 plans verified by V0.7-B and the
DIRECT_ENERGY_UNAVAILABLE decision from V0.7-C. It emits raw computational
observations and descriptive summaries only; no energy, carbon, inferential,
or Pareto result is calculated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import traceback
from typing import Any, Callable, Mapping, Sequence

from src.config import PROJECT_ROOT
from src.green.measurement import (
    CanonicalUnit,
    DEFAULT_OUTER_REPETITIONS,
    DEFAULT_WARMUP_CALLS,
    MeasurementPhase,
    MeasurementProvenance,
    WorkerStatus,
    collect_environment_metadata,
)
import src.pipeline_v07b as v07b
import src.pipeline_v07c as v07c


V07D_STAGE = "V0.7-D"
V07D_SCHEMA_VERSION = "v0.7-d-tier1-computational-1"
SUCCESS = "SUCCESS"
FAILURE = "FAILURE"
MEBIBYTE = 1024.0 * 1024.0
PLANNED_TRAINING_OBSERVATIONS = 300
PLANNED_INFERENCE_OBSERVATIONS = 300
PLANNED_TOTAL_OBSERVATIONS = 600
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "green_evaluation"
DEFAULT_CAPABILITY_PATH = DEFAULT_RESULTS_DIR / "energy_capability.json"
STAGING_DIRECTORY_NAME = ".v07d_staging"
CHECKPOINT_DIRECTORY_NAME = ".v07d_checkpoints"
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
V07B_IMMUTABLE_PATHS = (
    PROJECT_ROOT / "src" / "pipeline_v07b.py",
    PROJECT_ROOT / "tests" / "test_pipeline_v07b.py",
    v07b.DEFAULT_CONFIG_PATH,
)
V07C_IMMUTABLE_PATHS = (
    PROJECT_ROOT / "src" / "pipeline_v07c.py",
    PROJECT_ROOT / "tests" / "test_pipeline_v07c.py",
    DEFAULT_CAPABILITY_PATH,
)
OBSERVATION_FIELDS = (
    "observation_id",
    "schema_version",
    "stage",
    "environment_id",
    "classifier",
    "configuration_id",
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
    "selected_input_bytes",
    "numpy_dense_nbytes",
    "serialized_model_bytes",
    "per_operation_latency_sec",
    "per_record_latency_sec",
    "throughput_records_sec",
    "time_unit",
    "rss_unit",
    "size_unit",
    "throughput_unit",
    "measurement_kind",
    "measurement_scope",
    "measurement_provenance",
    "derived_measurement_provenance",
    "workload_sha256",
    "model_sha256",
    "feature_manifest_sha256",
    "v06_lock_sha256",
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
SUMMARY_FIELDS = (
    "phase",
    "classifier",
    "configuration_id",
    "feature_count",
    "planned_observations",
    "successful_observations",
    "failed_observations",
    "mean_wall_time_sec",
    "std_wall_time_sec",
    "mean_process_cpu_time_sec",
    "mean_absolute_peak_rss_mib",
    "mean_incremental_peak_rss_mib",
    "mean_selected_input_bytes",
    "mean_serialized_model_bytes",
    "mean_per_operation_latency_sec",
    "std_per_operation_latency_sec",
    "mean_per_record_latency_sec",
    "mean_throughput_records_sec",
    "summary_scope",
)


class V07DError(RuntimeError):
    """Base class for V0.7-D failures."""


class V07DIntegrityError(V07DError):
    """Raised when preflight, immutable evidence, or output coverage fails."""


@dataclass(frozen=True)
class PlannedObservation:
    observation_id: str
    classifier: str
    configuration_id: str
    seed: int
    phase: str
    outer_repetition: int
    inner_operation_count: int


@dataclass(frozen=True)
class CalibrationAnchor:
    classifier: str
    configuration_id: str
    seed: int
    workload_sha256: str
    model_sha256: str
    result: v07b.CalibrationResult


@dataclass(frozen=True)
class FrozenCalibration:
    target_duration_sec: float
    maximum_inner_operation_count: int
    selected_inner_operation_count: int
    anchors: tuple[CalibrationAnchor, ...]
    pairing_scope: str = "all_feature_configurations_and_classifiers"

    def __post_init__(self) -> None:
        if len(self.anchors) != 6:
            raise V07DIntegrityError("Calibration requires all six classifier/configuration anchors.")
        if self.selected_inner_operation_count != max(
            anchor.result.inner_operation_count for anchor in self.anchors
        ):
            raise V07DIntegrityError("Frozen inner count must be the maximum qualifying anchor count.")
        if any(
            anchor.result.target_duration_sec != self.target_duration_sec
            or not anchor.result.target_reached
            for anchor in self.anchors
        ):
            raise V07DIntegrityError("Every calibration anchor must reach the same target.")


def configure_single_thread_environment() -> dict[str, str]:
    """Constrain common native thread pools before any benchmark worker starts."""

    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    return {name: os.environ[name] for name in THREAD_VARIABLES}


def load_v07c_tier1_decision(
    path: Path | str = DEFAULT_CAPABILITY_PATH,
) -> tuple[dict[str, Any], str]:
    """Require the immutable V0.7-C unavailable decision for Tier-1 execution."""

    capability_path = Path(path)
    try:
        payload = json.loads(capability_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V07DIntegrityError(f"Cannot load V0.7-C capability record: {exc}") from exc
    if not isinstance(payload, dict):
        raise V07DIntegrityError("V0.7-C capability record must be a JSON object.")
    if (
        payload.get("stage") != v07c.V07C_STAGE
        or payload.get("schema_version") != v07c.V07C_SCHEMA_VERSION
        or payload.get("direct_energy_decision") != v07c.DIRECT_ENERGY_UNAVAILABLE
        or payload.get("accepted_direct_energy_backend") is not None
        or payload.get("experiments_executed") is not False
        or payload.get("final_observations_executed") != 0
        or payload.get("carbon", {}).get("claim_reported") is not False
    ):
        raise V07DIntegrityError(
            "V0.7-D requires the complete DIRECT_ENERGY_UNAVAILABLE V0.7-C decision."
        )
    return payload, sha256_file(capability_path)


def build_observation_schedule(
    plan: v07b.BenchmarkPlan,
    *,
    inference_inner_operation_count: int,
    outer_repetitions: int = DEFAULT_OUTER_REPETITIONS,
) -> tuple[PlannedObservation, ...]:
    """Build a deterministic, cyclically balanced 600-observation schedule."""

    if tuple(item.configuration_id for item in plan.configurations) != v07b.EXPECTED_CONFIGURATION_IDS:
        raise V07DIntegrityError("Schedule requires exactly the frozen three configurations.")
    if plan.classifiers != v07b.CLASSIFIERS or plan.seeds != v07b.EXPECTED_SEEDS:
        raise V07DIntegrityError("Schedule requires two frozen classifiers and seeds 42-46.")
    if outer_repetitions != DEFAULT_OUTER_REPETITIONS:
        raise V07DIntegrityError("V0.7-D requires exactly ten outer repetitions.")
    if inference_inner_operation_count < 1:
        raise V07DIntegrityError("Inference inner count must be positive and calibrated.")

    configuration_ids = tuple(item.configuration_id for item in plan.configurations)
    rows: list[PlannedObservation] = []
    for phase_index, phase in enumerate(
        (MeasurementPhase.TRAINING.value, MeasurementPhase.INFERENCE.value)
    ):
        for seed_index, seed in enumerate(plan.seeds):
            for classifier_index, classifier in enumerate(plan.classifiers):
                rotation = (phase_index + seed_index + classifier_index) % len(configuration_ids)
                ordered = configuration_ids[rotation:] + configuration_ids[:rotation]
                for configuration_id in ordered:
                    inner_count = (
                        1
                        if phase == MeasurementPhase.TRAINING.value
                        else inference_inner_operation_count
                    )
                    for repetition in range(1, outer_repetitions + 1):
                        observation_id = (
                            f"v07d__{phase}__{classifier}__{configuration_id}"
                            f"__s{seed}__r{repetition:02d}"
                        )
                        rows.append(
                            PlannedObservation(
                                observation_id,
                                classifier,
                                configuration_id,
                                seed,
                                phase,
                                repetition,
                                inner_count,
                            )
                        )
    _validate_schedule_counts(rows)
    return tuple(rows)


def schedule_cells(schedule: Sequence[PlannedObservation]) -> tuple[v07b.BenchmarkCell, ...]:
    """Collapse consecutive ten-repetition plans into sequential benchmark cells."""

    cells: list[v07b.BenchmarkCell] = []
    for index in range(0, len(schedule), DEFAULT_OUTER_REPETITIONS):
        block = schedule[index : index + DEFAULT_OUTER_REPETITIONS]
        if len(block) != DEFAULT_OUTER_REPETITIONS:
            raise V07DIntegrityError("Schedule contains an incomplete repetition block.")
        keys = {
            (item.classifier, item.configuration_id, item.seed, item.phase, item.inner_operation_count)
            for item in block
        }
        if len(keys) != 1 or tuple(item.outer_repetition for item in block) != tuple(
            range(1, DEFAULT_OUTER_REPETITIONS + 1)
        ):
            raise V07DIntegrityError("Each benchmark cell must contain repetitions 1-10.")
        cells.append(v07b.BenchmarkCell(*next(iter(keys))))
    if len(cells) != 60:
        raise V07DIntegrityError("V0.7-D requires exactly 60 sequential benchmark cells.")
    return tuple(cells)


def _validate_schedule_counts(schedule: Sequence[PlannedObservation]) -> None:
    training = sum(item.phase == MeasurementPhase.TRAINING.value for item in schedule)
    inference = sum(item.phase == MeasurementPhase.INFERENCE.value for item in schedule)
    if (training, inference, len(schedule)) != (
        PLANNED_TRAINING_OBSERVATIONS,
        PLANNED_INFERENCE_OBSERVATIONS,
        PLANNED_TOTAL_OBSERVATIONS,
    ):
        raise V07DIntegrityError("Planned matrix must contain exactly 300 training and 300 inference observations.")
    identifiers = [item.observation_id for item in schedule]
    if len(set(identifiers)) != len(identifiers):
        raise V07DIntegrityError("Planned observation IDs must be unique.")


def run_calibration_matrix(
    context: v07b.V07BContext,
    workloads: v07b.PreparedBenchmarkWorkloads,
    config: Mapping[str, Any],
    *,
    seed: int = 42,
    calibrator: Callable[..., v07b.CalibrationResult] = v07b.calibrate_inference_inner_count,
    runner: Callable[..., Any] = v07b.run_fresh_worker_protocol,
    start_method: str | None = None,
) -> FrozenCalibration:
    """Calibrate six anchors and freeze one count valid for paired comparisons."""

    calibration_config = config["measurement"]["inference_calibration"]
    target = float(calibration_config["target_duration_sec"])
    maximum = int(calibration_config["maximum_inner_operation_count"])
    if calibration_config.get("geometric_growth_factor") != 2:
        raise V07DIntegrityError("Calibration growth factor must remain exactly two.")
    manifestation = workloads.get(seed, MeasurementPhase.INFERENCE.value)
    anchors: list[CalibrationAnchor] = []
    for classifier in context.plan.classifiers:
        for configuration in context.plan.configurations:
            feature_names = configuration.features_for_seed(seed)
            selected = manifestation.features.loc[:, list(feature_names)].copy()
            artifact = context.plan.model(classifier, configuration.configuration_id, seed)
            prepare = v07b._InferencePreparation(artifact, selected)
            result = calibrator(
                v07b._prediction_only,
                prepare=prepare,
                target_duration_sec=target,
                maximum_inner_operation_count=maximum,
                warmup_calls=DEFAULT_WARMUP_CALLS,
                timeout_seconds=300.0,
                start_method=start_method,
                runner=runner,
            )
            _verify_geometric_attempts(result, maximum)
            anchors.append(
                CalibrationAnchor(
                    classifier,
                    configuration.configuration_id,
                    seed,
                    workloads.workload_hashes[(seed, MeasurementPhase.INFERENCE.value)],
                    artifact.file_sha256,
                    result,
                )
            )
    return FrozenCalibration(
        target,
        maximum,
        max(anchor.result.inner_operation_count for anchor in anchors),
        tuple(anchors),
    )


def _verify_geometric_attempts(result: v07b.CalibrationResult, maximum: int) -> None:
    counts = tuple(attempt.inner_operation_count for attempt in result.attempts)
    if counts != tuple(2**index for index in range(len(counts))):
        raise V07DIntegrityError("Calibration candidate counts must follow 1, 2, 4, 8, ...")
    if result.inner_operation_count > maximum:
        raise V07DIntegrityError("Calibration exceeded the configured safety maximum.")


def calibration_rows(
    calibration: FrozenCalibration, environment_id: str
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for anchor in calibration.anchors:
        for order, attempt in enumerate(anchor.result.attempts, start=1):
            rows.append(
                {
                    "calibration_id": (
                        f"v07d_calibration__{anchor.classifier}__{anchor.configuration_id}"
                        f"__s{anchor.seed}__n{attempt.inner_operation_count}"
                    ),
                    "environment_id": environment_id,
                    "classifier": anchor.classifier,
                    "configuration_id": anchor.configuration_id,
                    "seed": anchor.seed,
                    "phase": MeasurementPhase.INFERENCE.value,
                    "candidate_order": order,
                    "inner_operation_count": attempt.inner_operation_count,
                    "wall_time_sec": attempt.wall_time_sec,
                    "target_duration_sec": calibration.target_duration_sec,
                    "target_reached": attempt.wall_time_sec >= calibration.target_duration_sec,
                    "selected_for_anchor": attempt.inner_operation_count
                    == anchor.result.inner_operation_count,
                    "frozen_for_main_matrix": attempt.inner_operation_count
                    == calibration.selected_inner_operation_count,
                    "worker_pid": attempt.worker_pid,
                    "warmup_calls": DEFAULT_WARMUP_CALLS,
                    "measurement_kind": MeasurementProvenance.DIRECT_COMPUTATIONAL.value,
                    "measurement_scope": "prediction_only_calibration",
                    "time_unit": CanonicalUnit.SECONDS.value,
                    "workload_sha256": anchor.workload_sha256,
                    "model_sha256": anchor.model_sha256,
                    "excluded_from_main_observations": True,
                }
            )
    return tuple(rows)


def load_frozen_calibration(
    path: Path | str,
    *,
    environment_id: str,
    config: Mapping[str, Any],
) -> FrozenCalibration:
    """Restore the original calibration when resuming an interrupted matrix."""

    calibration_path = Path(path)
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        rows = payload["rows"]
        calibration_config = config["measurement"]["inference_calibration"]
        target = float(payload["target_duration_sec"])
        maximum = int(payload["maximum_inner_operation_count"])
        selected = int(payload["selected_inner_operation_count"])
        pairing_scope = str(payload["pairing_scope"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise V07DIntegrityError(f"Cannot restore frozen calibration: {calibration_path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("stage") != V07D_STAGE
        or payload.get("schema_version") != V07D_SCHEMA_VERSION
        or payload.get("excluded_from_main_observations") is not True
        or not isinstance(rows, list)
        or target != float(calibration_config["target_duration_sec"])
        or maximum != int(calibration_config["maximum_inner_operation_count"])
        or pairing_scope != calibration_config["pairing_scope"]
    ):
        raise V07DIntegrityError("Stored calibration does not match the frozen V0.7-D protocol.")

    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("environment_id") != environment_id
            or row.get("phase") != MeasurementPhase.INFERENCE.value
            or row.get("excluded_from_main_observations") is not True
            or row.get("measurement_kind") != MeasurementProvenance.DIRECT_COMPUTATIONAL.value
        ):
            raise V07DIntegrityError("Stored calibration row has invalid provenance or environment.")
        try:
            key = (str(row["classifier"]), str(row["configuration_id"]), int(row["seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise V07DIntegrityError("Stored calibration row is incomplete.") from exc
        grouped.setdefault(key, []).append(row)

    anchors: list[CalibrationAnchor] = []
    try:
        expected_keys = {
            (classifier, configuration_id, 42)
            for classifier in v07b.CLASSIFIERS
            for configuration_id in v07b.EXPECTED_CONFIGURATION_IDS
        }
        if set(grouped) != expected_keys:
            raise V07DIntegrityError("Stored calibration does not contain the six frozen anchors.")
        for classifier in v07b.CLASSIFIERS:
            for configuration_id in v07b.EXPECTED_CONFIGURATION_IDS:
                seed = 42
                group = grouped[(classifier, configuration_id, seed)]
                ordered = sorted(group, key=lambda row: int(row["candidate_order"]))
                if tuple(int(row["candidate_order"]) for row in ordered) != tuple(
                    range(1, len(ordered) + 1)
                ):
                    raise V07DIntegrityError("Stored calibration candidate ordering is invalid.")
                selected_rows = [row for row in ordered if row["selected_for_anchor"] is True]
                if len(selected_rows) != 1 or selected_rows[0] is not ordered[-1]:
                    raise V07DIntegrityError("Stored calibration must retain one final selected attempt.")
                attempts = tuple(
                    v07b.CalibrationAttempt(
                        int(row["inner_operation_count"]),
                        float(row["wall_time_sec"]),
                        None if row["worker_pid"] is None else int(row["worker_pid"]),
                    )
                    for row in ordered
                )
                result = v07b.CalibrationResult(
                    MeasurementPhase.INFERENCE.value,
                    target,
                    int(selected_rows[0]["inner_operation_count"]),
                    attempts,
                )
                _verify_geometric_attempts(result, maximum)
                workload_hashes = {str(row["workload_sha256"]) for row in ordered}
                model_hashes = {str(row["model_sha256"]) for row in ordered}
                if len(workload_hashes) != 1 or len(model_hashes) != 1:
                    raise V07DIntegrityError("Stored calibration anchor hashes are inconsistent.")
                anchors.append(
                    CalibrationAnchor(
                        classifier,
                        configuration_id,
                        seed,
                        workload_hashes.pop(),
                        model_hashes.pop(),
                        result,
                    )
                )
    except (KeyError, TypeError, ValueError, v07b.V07BCalibrationError) as exc:
        raise V07DIntegrityError("Stored calibration rows are invalid.") from exc

    frozen = FrozenCalibration(target, maximum, selected, tuple(anchors), pairing_scope)
    if selected > maximum:
        raise V07DIntegrityError("Stored calibration exceeds its safety maximum.")
    return frozen


def transform_v07b_rows(
    rows: Sequence[Mapping[str, Any]],
    plans: Sequence[PlannedObservation],
) -> tuple[dict[str, Any], ...]:
    """Convert one complete V0.7-B cell into the canonical V0.7-D schema."""

    if len(rows) != len(plans):
        raise V07DIntegrityError("Every planned repetition must retain one result row.")
    transformed: list[dict[str, Any]] = []
    for raw, plan in zip(rows, plans):
        if (
            raw.get("classifier") != plan.classifier
            or raw.get("configuration_id") != plan.configuration_id
            or raw.get("seed") != plan.seed
            or raw.get("phase") != plan.phase
            or raw.get("outer_repetition") != plan.outer_repetition
            or raw.get("inner_operation_count") != plan.inner_operation_count
        ):
            raise V07DIntegrityError("Worker row does not match its deterministic observation plan.")
        success = raw.get("status") == WorkerStatus.COMPLETED.value
        row = {
            "observation_id": plan.observation_id,
            "schema_version": V07D_SCHEMA_VERSION,
            "stage": V07D_STAGE,
            "environment_id": raw["environment_id"],
            "classifier": plan.classifier,
            "configuration_id": plan.configuration_id,
            "seed": plan.seed,
            "phase": plan.phase,
            "outer_repetition": plan.outer_repetition,
            "inner_operation_count": plan.inner_operation_count,
            "measured_operation_count": raw["measured_operation_count"],
            "record_count": raw["record_count"],
            "measured_record_count": (
                raw["measured_operation_count"] * raw["record_count"]
            ),
            "feature_count": raw["feature_count"],
            "warmup_calls": raw["warmup_calls"],
            "worker_pid": raw["worker_pid"],
            "wall_time_sec": raw["wall_time_sec"],
            "process_cpu_time_sec": raw["process_cpu_time_sec"],
            "start_rss_mib": _bytes_to_mib(raw["start_rss_bytes"]),
            "end_rss_mib": _bytes_to_mib(raw["end_rss_bytes"]),
            "absolute_peak_rss_mib": _bytes_to_mib(raw["absolute_peak_rss_bytes"]),
            "incremental_peak_rss_mib": _bytes_to_mib(raw["incremental_peak_rss_bytes"]),
            "selected_input_bytes": raw["selected_input_bytes"],
            "numpy_dense_nbytes": raw["numpy_dense_nbytes"],
            "serialized_model_bytes": raw["serialized_model_bytes"],
            "per_operation_latency_sec": raw["inference_latency_sec"],
            "per_record_latency_sec": raw["per_record_latency_sec"],
            "throughput_records_sec": raw["throughput_records_sec"],
            "time_unit": CanonicalUnit.SECONDS.value,
            "rss_unit": CanonicalUnit.MEBIBYTES.value,
            "size_unit": CanonicalUnit.BYTES.value,
            "throughput_unit": CanonicalUnit.RECORDS_PER_SECOND.value,
            "measurement_kind": MeasurementProvenance.DIRECT_COMPUTATIONAL.value,
            "measurement_scope": raw["measurement_scope"],
            "measurement_provenance": MeasurementProvenance.DIRECT_COMPUTATIONAL.value,
            "derived_measurement_provenance": MeasurementProvenance.DERIVED.value,
            "workload_sha256": raw["workload_sha256"],
            "model_sha256": raw["model_sha256"],
            "feature_manifest_sha256": raw["feature_manifest_sha256"],
            "v06_lock_sha256": raw["v06_lock_sha256"],
            "status": SUCCESS if success else FAILURE,
            "failure_type": None if success else raw["error_type"],
            "failure_stage": None if success else raw["error_stage"],
            "failure_reason": None if success else raw["error_message"],
        }
        transformed.append(row)
    validate_observation_rows(transformed, expected_schedule=plans)
    return tuple(transformed)


def _bytes_to_mib(value: Any) -> float | None:
    return None if value is None else float(value) / MEBIBYTE


def validate_observation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_schedule: Sequence[PlannedObservation] | None = None,
) -> None:
    """Fail closed on schema, provenance, units, failures, or missing observations."""

    if not rows:
        raise V07DIntegrityError("Observation output cannot be empty.")
    if expected_schedule is not None:
        expected_ids = [item.observation_id for item in expected_schedule]
        observed_ids = [str(row.get("observation_id")) for row in rows]
        if observed_ids != expected_ids:
            raise V07DIntegrityError("Observation order or coverage differs from the frozen schedule.")
    identifiers: list[str] = []
    for row in rows:
        missing = set(OBSERVATION_FIELDS) - set(row)
        if missing:
            raise V07DIntegrityError(f"Observation row is missing fields: {sorted(missing)}")
        identifiers.append(str(row["observation_id"]))
        if row["stage"] != V07D_STAGE or row["schema_version"] != V07D_SCHEMA_VERSION:
            raise V07DIntegrityError("Observation stage/schema is invalid.")
        if row["measurement_kind"] != MeasurementProvenance.DIRECT_COMPUTATIONAL.value:
            raise V07DIntegrityError("V0.7-D permits DIRECT_COMPUTATIONAL rows only.")
        if row["measurement_provenance"] != MeasurementProvenance.DIRECT_COMPUTATIONAL.value:
            raise V07DIntegrityError("Computational provenance is invalid.")
        if (row["time_unit"], row["rss_unit"], row["size_unit"], row["throughput_unit"]) != (
            CanonicalUnit.SECONDS.value,
            CanonicalUnit.MEBIBYTES.value,
            CanonicalUnit.BYTES.value,
            CanonicalUnit.RECORDS_PER_SECOND.value,
        ):
            raise V07DIntegrityError("Observation units are not canonical.")
        if any("energy" in str(key).lower() or "carbon" in str(key).lower() for key in row):
            raise V07DIntegrityError("Tier-1 observation rows cannot contain energy or carbon fields.")
        if row["status"] == SUCCESS:
            numeric = (
                "wall_time_sec",
                "process_cpu_time_sec",
                "start_rss_mib",
                "end_rss_mib",
                "absolute_peak_rss_mib",
                "incremental_peak_rss_mib",
                "selected_input_bytes",
                "serialized_model_bytes",
                "per_operation_latency_sec",
                "per_record_latency_sec",
                "throughput_records_sec",
            )
            if any(row[name] is None or not math.isfinite(float(row[name])) or float(row[name]) < 0 for name in numeric):
                raise V07DIntegrityError("Successful observations require finite nonnegative metrics.")
            if float(row["wall_time_sec"]) <= 0 or int(row["measured_operation_count"]) != int(
                row["inner_operation_count"]
            ):
                raise V07DIntegrityError("Successful observation operation accounting is invalid.")
            if row["warmup_calls"] != DEFAULT_WARMUP_CALLS or row["worker_pid"] is None:
                raise V07DIntegrityError("Successful observations require warm-up and a worker PID.")
            if any(row[name] is not None for name in ("failure_type", "failure_stage", "failure_reason")):
                raise V07DIntegrityError("Successful observations cannot contain failure details.")
        elif row["status"] == FAILURE:
            if not row["failure_type"] or not row["failure_stage"] or not row["failure_reason"]:
                raise V07DIntegrityError("Failed observations require complete structured failure details.")
        else:
            raise V07DIntegrityError("Observation status must be SUCCESS or FAILURE.")
    if len(set(identifiers)) != len(identifiers):
        raise V07DIntegrityError("Observation IDs must be unique; no duplicate run is allowed.")


def validate_complete_matrix(
    rows: Sequence[Mapping[str, Any]],
    schedule: Sequence[PlannedObservation],
    context: v07b.V07BContext,
) -> dict[str, Any]:
    """Verify exact coverage, pairing, immutable references, and accounting."""

    validate_observation_rows(rows, expected_schedule=schedule)
    if len(rows) != PLANNED_TOTAL_OBSERVATIONS:
        raise V07DIntegrityError("Completed output must retain all 600 planned observations.")
    expected_features = {
        (configuration.configuration_id, seed): (
            configuration.feature_count,
            configuration.feature_hash_for_seed(seed),
        )
        for configuration in context.plan.configurations
        for seed in context.plan.seeds
    }
    expected_models = {
        (model.classifier, model.configuration_id, model.seed): model.file_sha256
        for model in context.plan.models
    }
    workload_by_pair: dict[tuple[int, str], set[str]] = {}
    for row in rows:
        feature_count, feature_hash = expected_features[(row["configuration_id"], int(row["seed"]))]
        if row["feature_count"] != feature_count or row["feature_manifest_sha256"] != feature_hash:
            raise V07DIntegrityError("Feature count/order fingerprint changed during execution.")
        model_key = (row["classifier"], row["configuration_id"], int(row["seed"]))
        if row["model_sha256"] != expected_models[model_key]:
            raise V07DIntegrityError("Frozen model hash changed during execution.")
        expected_workload = context.plan.workload(row["phase"], int(row["seed"])).workload_sha256
        if row["workload_sha256"] != expected_workload:
            raise V07DIntegrityError("Frozen workload hash changed during execution.")
        workload_by_pair.setdefault((int(row["seed"]), row["phase"]), set()).add(
            row["workload_sha256"]
        )
    if any(len(values) != 1 for values in workload_by_pair.values()):
        raise V07DIntegrityError("Paired classifier/configuration rows did not reuse one workload hash.")
    training = [row for row in rows if row["phase"] == MeasurementPhase.TRAINING.value]
    inference = [row for row in rows if row["phase"] == MeasurementPhase.INFERENCE.value]
    return {
        "planned_training_observations": PLANNED_TRAINING_OBSERVATIONS,
        "successful_training_observations": sum(row["status"] == SUCCESS for row in training),
        "failed_training_observations": sum(row["status"] == FAILURE for row in training),
        "planned_inference_observations": PLANNED_INFERENCE_OBSERVATIONS,
        "successful_inference_observations": sum(row["status"] == SUCCESS for row in inference),
        "failed_inference_observations": sum(row["status"] == FAILURE for row in inference),
        "planned_total_observations": PLANNED_TOTAL_OBSERVATIONS,
        "successful_total_observations": sum(row["status"] == SUCCESS for row in rows),
        "failed_total_observations": sum(row["status"] == FAILURE for row in rows),
        "retained_total_observations": len(rows),
        "run_complete": len(rows) == PLANNED_TOTAL_OBSERVATIONS,
    }


def build_descriptive_summary(
    rows: Sequence[Mapping[str, Any]], phase: str
) -> tuple[dict[str, Any], ...]:
    """Produce measurement-level descriptions without scientific inference or ranking."""

    phase_rows = [row for row in rows if row["phase"] == phase]
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in phase_rows:
        groups.setdefault((row["classifier"], row["configuration_id"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for classifier, configuration_id in sorted(groups):
        group = groups[(classifier, configuration_id)]
        successes = [row for row in group if row["status"] == SUCCESS]
        summary.append(
            {
                "phase": phase,
                "classifier": classifier,
                "configuration_id": configuration_id,
                "feature_count": group[0]["feature_count"],
                "planned_observations": len(group),
                "successful_observations": len(successes),
                "failed_observations": len(group) - len(successes),
                "mean_wall_time_sec": _mean(successes, "wall_time_sec"),
                "std_wall_time_sec": _sample_std(successes, "wall_time_sec"),
                "mean_process_cpu_time_sec": _mean(successes, "process_cpu_time_sec"),
                "mean_absolute_peak_rss_mib": _mean(successes, "absolute_peak_rss_mib"),
                "mean_incremental_peak_rss_mib": _mean(successes, "incremental_peak_rss_mib"),
                "mean_selected_input_bytes": _mean(successes, "selected_input_bytes"),
                "mean_serialized_model_bytes": _mean(successes, "serialized_model_bytes"),
                "mean_per_operation_latency_sec": _mean(successes, "per_operation_latency_sec"),
                "std_per_operation_latency_sec": _sample_std(successes, "per_operation_latency_sec"),
                "mean_per_record_latency_sec": _mean(successes, "per_record_latency_sec"),
                "mean_throughput_records_sec": _mean(successes, "throughput_records_sec"),
                "summary_scope": "descriptive_measurement_repetitions_not_independent_scientific_samples",
            }
        )
    return tuple(summary)


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    return None if not rows else statistics.fmean(float(row[field]) for row in rows)


def _sample_std(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    return None if len(rows) < 2 else statistics.stdev(float(row[field]) for row in rows)


def build_preflight_record(
    context: v07b.V07BContext,
    capability_hash: str,
    environment: Mapping[str, Any],
    environment_id: str,
    immutable_hashes: Mapping[str, Mapping[str, str]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = {
        "configurations": [item.configuration_id for item in context.plan.configurations],
        "feature_counts": [item.feature_count for item in context.plan.configurations],
        "classifiers": list(context.plan.classifiers),
        "seeds": list(context.plan.seeds),
        "outer_repetitions": DEFAULT_OUTER_REPETITIONS,
        "warmup_calls": DEFAULT_WARMUP_CALLS,
        "training_inner_operation_count": 1,
        "inference_calibration": dict(config["measurement"]["inference_calibration"]),
        "training_boundary": "model.fit_only_data_loading_selection_serialization_excluded",
        "inference_boundary": "prediction_only_model_loading_warmup_excluded",
        "worker_policy": "fresh_process_per_outer_repetition",
        "execution_policy": "sequential_cells",
        "direct_energy_decision": v07c.DIRECT_ENERGY_UNAVAILABLE,
    }
    protocol_hash = _json_sha256(protocol)
    return {
        "schema_version": V07D_SCHEMA_VERSION,
        "stage": V07D_STAGE,
        "status": "PREFLIGHT_PASS",
        "authoritative_prd": "docs/PhD_PRD.md",
        "timestamp_utc": _utc_now(),
        "environment_id": environment_id,
        "environment": dict(environment),
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "benchmark_plan_sha256": v07b.benchmark_plan_sha256(context.plan),
        "v06_lock_sha256": context.plan.validation_lock_sha256,
        "v06_semantic_lock_sha256": context.plan.semantic_lock_sha256,
        "v07c_capability_sha256": capability_hash,
        "immutable_hashes": {name: dict(values) for name, values in immutable_hashes.items()},
        "checks": [
            {"name": "v06_lock_models_features_workloads_verified", "passed": True},
            {"name": "v07b_hashes_verified", "passed": True},
            {"name": "v07c_hashes_and_decision_verified", "passed": True},
            {"name": "direct_energy_unavailable_tier1_only", "passed": True},
            {"name": "selector_fitting_prohibited", "passed": True},
            {"name": "model_and_threshold_tuning_prohibited", "passed": True},
            {"name": "thread_pools_constrained_to_one", "passed": True},
        ],
    }


def run_v07d_experiments(
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    capability_path: Path | str = DEFAULT_CAPABILITY_PATH,
    context: v07b.V07BContext | None = None,
    workloads: v07b.PreparedBenchmarkWorkloads | None = None,
    calibrator: Callable[..., v07b.CalibrationResult] = v07b.calibrate_inference_inner_count,
    calibration_runner: Callable[..., Any] = v07b.run_fresh_worker_protocol,
    cell_runner: Callable[..., Sequence[Mapping[str, Any]]] = v07b.run_benchmark_cell,
    start_method: str | None = None,
    allow_completed: bool = False,
) -> dict[str, Any]:
    """Execute or safely resume the complete sequential Tier-1 matrix once."""

    output_root = Path(results_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    completed_manifest = output_root / "v07d_run_manifest.json"
    if completed_manifest.is_file() and not allow_completed:
        raise V07DError("A completed V0.7-D manifest already exists; refusing to rerun the experiment.")

    started_at = _utc_now()
    thread_controls = configure_single_thread_environment()
    config = v07b.load_green_evaluation_config()
    capability, capability_hash = load_v07c_tier1_decision(capability_path)
    benchmark_context = v07b.load_v06_benchmark_context() if context is None else context
    v07b.verify_v06_source_hashes(benchmark_context.plan)
    immutable_before = {
        "v06": dict(benchmark_context.plan.source_hashes),
        "v07b": v07c.snapshot_file_hashes(V07B_IMMUTABLE_PATHS),
        "v07c": v07c.snapshot_file_hashes(V07C_IMMUTABLE_PATHS),
    }
    environment = collect_environment_metadata(PROJECT_ROOT)
    if environment["thread_environment"] != thread_controls:
        raise V07DIntegrityError("Thread environment was not constrained before measurement.")
    environment_id = _json_sha256(environment)
    prepared = v07b.prepare_verified_workloads(benchmark_context) if workloads is None else workloads
    assignments = [
        (v07b.BenchmarkCell(classifier, configuration.configuration_id, seed, phase, 1), prepared.get(seed, phase))
        for phase in (MeasurementPhase.TRAINING.value, MeasurementPhase.INFERENCE.value)
        for seed in benchmark_context.plan.seeds
        for classifier in benchmark_context.plan.classifiers
        for configuration in benchmark_context.plan.configurations
    ]
    v07b.verify_same_workload_reused(assignments)
    preflight = build_preflight_record(
        benchmark_context,
        capability_hash,
        environment,
        environment_id,
        immutable_before,
        config,
    )

    staging = output_root / STAGING_DIRECTORY_NAME
    checkpoints = output_root / CHECKPOINT_DIRECTORY_NAME
    staging.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    atomic_write_json(staging / "environment_metadata.json", environment)
    atomic_write_json(staging / "v07d_preflight.json", preflight)

    calibration_path = staging / "calibration_results.json"
    if calibration_path.is_file():
        frozen_calibration = load_frozen_calibration(
            calibration_path, environment_id=environment_id, config=config
        )
        calibration_output_rows = calibration_rows(frozen_calibration, environment_id)
        if not (staging / "calibration_results.csv").is_file():
            raise V07DIntegrityError("Stored calibration CSV is missing during recovery.")
    else:
        if any(checkpoints.iterdir()):
            raise V07DIntegrityError("Cannot resume checkpoints without the original calibration.")
        frozen_calibration = run_calibration_matrix(
            benchmark_context,
            prepared,
            config,
            calibrator=calibrator,
            runner=calibration_runner,
            start_method=start_method,
        )
        calibration_output_rows = calibration_rows(frozen_calibration, environment_id)
        atomic_write_json(
            calibration_path,
            {
                "schema_version": V07D_SCHEMA_VERSION,
                "stage": V07D_STAGE,
                "excluded_from_main_observations": True,
                "target_duration_sec": frozen_calibration.target_duration_sec,
                "maximum_inner_operation_count": frozen_calibration.maximum_inner_operation_count,
                "selected_inner_operation_count": frozen_calibration.selected_inner_operation_count,
                "pairing_scope": frozen_calibration.pairing_scope,
                "rows": list(calibration_output_rows),
            },
        )
        atomic_write_csv(staging / "calibration_results.csv", calibration_output_rows)

    schedule = build_observation_schedule(
        benchmark_context.plan,
        inference_inner_operation_count=frozen_calibration.selected_inner_operation_count,
    )
    cells = schedule_cells(schedule)
    rows: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(cells):
        block = schedule[
            cell_index * DEFAULT_OUTER_REPETITIONS : (cell_index + 1) * DEFAULT_OUTER_REPETITIONS
        ]
        checkpoint = checkpoints / _checkpoint_name(cell)
        cached = _load_valid_checkpoint(checkpoint, block, preflight["protocol_sha256"], environment_id)
        if cached is not None:
            cell_rows = cached
        else:
            raw_rows = cell_runner(
                benchmark_context,
                prepared,
                cell,
                environment_id=environment_id,
                warmup_calls=DEFAULT_WARMUP_CALLS,
                outer_repetitions=DEFAULT_OUTER_REPETITIONS,
                timeout_seconds=300.0,
                start_method=start_method,
            )
            cell_rows = transform_v07b_rows(raw_rows, block)
            atomic_write_json(
                checkpoint,
                {
                    "protocol_sha256": preflight["protocol_sha256"],
                    "environment_id": environment_id,
                    "rows": list(cell_rows),
                },
            )
        rows.extend(cell_rows)
        print(f"V0.7-D cell {cell_index + 1}/60 complete", flush=True)

    accounting = validate_complete_matrix(rows, schedule, benchmark_context)
    training_rows = tuple(row for row in rows if row["phase"] == MeasurementPhase.TRAINING.value)
    inference_rows = tuple(row for row in rows if row["phase"] == MeasurementPhase.INFERENCE.value)
    failure_rows = tuple(
        {name: row[name] for name in FAILURE_FIELDS} for row in rows if row["status"] == FAILURE
    )
    training_summary = build_descriptive_summary(rows, MeasurementPhase.TRAINING.value)
    inference_summary = build_descriptive_summary(rows, MeasurementPhase.INFERENCE.value)

    immutable_after = {
        "v06": v07c.snapshot_file_hashes(tuple(Path(path) for path in immutable_before["v06"])),
        "v07b": v07c.snapshot_file_hashes(V07B_IMMUTABLE_PATHS),
        "v07c": v07c.snapshot_file_hashes(V07C_IMMUTABLE_PATHS),
    }
    if immutable_after != immutable_before:
        raise V07DIntegrityError("Immutable V0.6 or V0.7-B/C evidence changed during execution.")
    v07b.verify_v06_source_hashes(benchmark_context.plan)

    governance = {
        "schema_version": V07D_SCHEMA_VERSION,
        "stage": V07D_STAGE,
        "status": "PASS",
        "checks": [
            {"name": "exact_600_observations_retained", "passed": accounting["run_complete"]},
            {"name": "calibration_excluded_from_main_rows", "passed": True},
            {"name": "one_frozen_inference_inner_count", "passed": True},
            {"name": "fresh_worker_per_outer_observation", "passed": True},
            {"name": "warmup_and_preparation_outside_timed_boundary", "passed": True},
            {"name": "same_seed_phase_workload_hash_reused", "passed": True},
            {"name": "selector_fit_and_reselection_not_performed", "passed": True},
            {"name": "hyperparameters_and_threshold_not_tuned", "passed": True},
            {"name": "failed_observations_retained", "passed": True},
            {"name": "outliers_not_removed", "passed": True},
            {"name": "direct_energy_rows_absent", "passed": True},
            {"name": "carbon_rows_absent", "passed": True},
            {"name": "v06_v07b_v07c_hashes_unchanged", "passed": True},
            {"name": "no_hypothesis_test_or_pareto_analysis", "passed": True},
        ],
        "scientific_unit_policy": (
            "ten outer repetitions are measurement repetitions; V0.7-E must summarize by paired seed"
        ),
        "outlier_policy": "all observations retained; no automatic outlier deletion",
        "interpretation_boundary": "computational efficiency only; no electrical-energy or carbon claim",
        "immutable_hashes_after": immutable_after,
    }
    completed_at = _utc_now()
    manifest = {
        "schema_version": V07D_SCHEMA_VERSION,
        "stage": V07D_STAGE,
        "status": "COMPLETED",
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "environment_id": environment_id,
        "protocol_sha256": preflight["protocol_sha256"],
        "direct_energy_decision": capability["direct_energy_decision"],
        "selected_inner_operation_count": frozen_calibration.selected_inner_operation_count,
        "accounting": accounting,
        "run_completeness": "COMPLETE" if accounting["run_complete"] else "INCOMPLETE",
        "configuration_ids": list(v07b.EXPECTED_CONFIGURATION_IDS),
        "classifiers": list(v07b.CLASSIFIERS),
        "seeds": list(v07b.EXPECTED_SEEDS),
        "outer_repetitions": DEFAULT_OUTER_REPETITIONS,
        "workload": {"attack_type": "mixed", "attack_rate": 0.05, "severity": "MEDIUM"},
        "measurement_kind": MeasurementProvenance.DIRECT_COMPUTATIONAL.value,
        "electrical_energy_reported": False,
        "carbon_reported": False,
        "statistical_inference_performed": False,
        "pareto_analysis_performed": False,
    }

    artifacts: dict[str, Any] = {
        "training_observations.json": list(training_rows),
        "inference_observations.json": list(inference_rows),
        "computational_observations.json": list(rows),
        "measurement_failures.json": list(failure_rows),
        "training_descriptive_summary.json": list(training_summary),
        "inference_descriptive_summary.json": list(inference_summary),
        "v07d_run_manifest.json": manifest,
        "v07d_governance_audit.json": governance,
    }
    csv_artifacts: dict[str, tuple[Sequence[Mapping[str, Any]], Sequence[str] | None]] = {
        "training_observations.csv": (training_rows, OBSERVATION_FIELDS),
        "inference_observations.csv": (inference_rows, OBSERVATION_FIELDS),
        "computational_observations.csv": (rows, OBSERVATION_FIELDS),
        "measurement_failures.csv": (failure_rows, FAILURE_FIELDS),
        "training_descriptive_summary.csv": (training_summary, SUMMARY_FIELDS),
        "inference_descriptive_summary.csv": (inference_summary, SUMMARY_FIELDS),
    }
    for name, payload in artifacts.items():
        atomic_write_json(staging / name, payload)
    for name, (payload, fields) in csv_artifacts.items():
        atomic_write_csv(staging / name, payload, fieldnames=fields)

    artifact_names = (
        "environment_metadata.json",
        "v07d_preflight.json",
        "calibration_results.csv",
        "calibration_results.json",
        *artifacts.keys(),
        *csv_artifacts.keys(),
    )
    artifact_hashes = {name: sha256_file(staging / name) for name in sorted(set(artifact_names))}
    atomic_write_json(staging / "v07d_artifact_hashes.json", artifact_hashes)
    publish_order = [name for name in artifact_hashes if name != "v07d_run_manifest.json"]
    publish_order.extend(("v07d_run_manifest.json", "v07d_artifact_hashes.json"))
    for name in publish_order:
        (staging / name).replace(output_root / name)
    _remove_empty_directory(staging)
    return manifest


def _checkpoint_name(cell: v07b.BenchmarkCell) -> str:
    return (
        f"{cell.phase}__{cell.classifier}__{cell.configuration_id}__s{cell.seed}.json"
    )


def _load_valid_checkpoint(
    path: Path,
    plans: Sequence[PlannedObservation],
    protocol_hash: str,
    environment_id: str,
) -> tuple[dict[str, Any], ...] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V07DIntegrityError(f"Existing checkpoint is unreadable: {path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("protocol_sha256") != protocol_hash
        or payload.get("environment_id") != environment_id
        or not isinstance(payload.get("rows"), list)
    ):
        raise V07DIntegrityError("Existing checkpoint does not match this protocol/environment.")
    rows = tuple(payload["rows"])
    validate_observation_rows(rows, expected_schedule=plans)
    return rows


def snapshot_immutable_inputs(context: v07b.V07BContext) -> dict[str, dict[str, str]]:
    return {
        "v06": dict(context.plan.source_hashes),
        "v07b": v07c.snapshot_file_hashes(V07B_IMMUTABLE_PATHS),
        "v07c": v07c.snapshot_file_hashes(V07C_IMMUTABLE_PATHS),
    }


def atomic_write_json(path: Path | str, payload: Any) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    atomic_write_text(Path(path), text)


def atomic_write_csv(
    path: Path | str,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or (tuple(rows[0]) if rows else ()))
    if not fields:
        raise V07DError("CSV output requires explicit fields when there are no rows.")
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(destination)
    except (OSError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise V07DError(f"Cannot atomically write CSV artifact: {destination}") from exc


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise V07DError(f"Cannot atomically write artifact: {path}") from exc


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise V07DIntegrityError(f"Cannot hash required file: {path}") from exc
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _remove_empty_directory(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


def write_failure_diagnostic(results_dir: Path | str, exc: BaseException) -> None:
    atomic_write_json(
        Path(results_dir) / "v07d_failure_diagnostic.json",
        {
            "schema_version": V07D_SCHEMA_VERSION,
            "stage": V07D_STAGE,
            "status": "CRITICAL_FAILURE",
            "timestamp_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "completed_manifest_published": False,
        },
    )


def main() -> int:
    try:
        manifest = run_v07d_experiments()
    except BaseException as exc:
        write_failure_diagnostic(DEFAULT_RESULTS_DIR, exc)
        raise
    print(json.dumps(manifest["accounting"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
