"""Immutable V0.6 loading and Tier-1 computational benchmarking for V0.7-B.

This module prepares benchmark plans and cell-level runners only. Importing it
does not access data, fit selectors, run the final matrix, or claim electrical
energy measurements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.ai.model_utils import load_processed_dataco_splits
from src.config import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.green.measurement import (
    DEFAULT_OUTER_REPETITIONS,
    DEFAULT_WARMUP_CALLS,
    FreshWorkerRun,
    MeasurementPhase,
    MeasurementProvenance,
    WorkerStatus,
    dataframe_deep_memory_bytes,
    numpy_dense_nbytes,
    run_fresh_worker_protocol,
)
from src.lightweight.models import LightweightDetector, create_model
from src.pipeline_v06e import (
    LOCK_FILE_NAME,
    SIDECAR_FILE_NAME,
    fingerprint_feature_names,
    semantic_payload_sha256,
    verify_validation_lock,
)
import src.pipeline_v06f as v06f
import src.pipeline_v06g as v06g
from src.security.experiment_data import PreparedAttackSplit, prepare_attack_split


V07B_STAGE = "V0.7-B"
V07B_SCHEMA_VERSION = "v0.7-b-computational-benchmark-1"
AUTHORITATIVE_PRD_PATH = PROJECT_ROOT / "docs" / "PhD_PRD.md"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "green_evaluation.yaml"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "feature_selection"
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
SEMANTIC_ROLES = ("full_baseline", "best_unsupervised", "best_supervised")
EXPECTED_CONFIGURATION_IDS = (
    "none_natural",
    "pairwise_correlation_filter_natural",
    "mutual_information_select_k_best_k11",
)
EXPECTED_FEATURE_COUNTS = MappingProxyType(
    {"full_baseline": 43, "best_unsupervised": 42, "best_supervised": 11}
)
CLASSIFIERS = ("decision_tree", "logistic_regression")
DT_PARAMETERS = MappingProxyType(
    {"max_depth": 5, "min_samples_leaf": 20, "class_weight": "balanced"}
)
LR_PARAMETERS = MappingProxyType(
    {"solver": "liblinear", "class_weight": "balanced", "max_iter": 500, "C": 1.0}
)
PREDICTION_THRESHOLD = 0.5
ATTACK_TYPE = "mixed"
ATTACK_RATE = 0.05
ATTACK_SEVERITY = "MEDIUM"
PRIMARY_SCENARIO_ID = "mixed__r0p05__medium"
OUTPUT_REQUIRED_FIELDS = frozenset(
    {
        "environment_id",
        "classifier",
        "configuration_id",
        "seed",
        "phase",
        "feature_count",
        "record_count",
        "outer_repetition",
        "inner_operation_count",
        "measured_operation_count",
        "wall_time_sec",
        "process_cpu_time_sec",
        "start_rss_bytes",
        "end_rss_bytes",
        "absolute_peak_rss_bytes",
        "incremental_peak_rss_bytes",
        "selected_input_bytes",
        "serialized_model_bytes",
        "per_record_latency_sec",
        "throughput_records_sec",
        "measurement_kind",
        "workload_sha256",
        "model_sha256",
        "feature_manifest_sha256",
        "v06_lock_sha256",
    }
)
_WORKLOAD_COMPONENT_COLUMNS = MappingProxyType(
    {
        "row_ids_sha256": "training_row_ids_sha256",
        "clean_features_sha256": "training_clean_features_sha256",
        "attacked_features_sha256": "training_attacked_features_sha256",
        "labels_sha256": "training_labels_sha256",
        "ground_truth_sha256": "training_ground_truth_sha256",
        "manifest_sha256": "training_manifest_sha256",
        "attack_metadata_sha256": "training_attack_metadata_sha256",
    }
)


class V07BError(RuntimeError):
    """Base class for structured V0.7-B failures."""


class V07BConfigurationError(V07BError):
    """Raised when the V0.7-B protocol configuration is invalid."""


class V07BIntegrityError(V07BError):
    """Raised when immutable V0.6 evidence fails closed."""


class V07BCalibrationError(V07BError):
    """Raised when calibration cannot safely reach its duration target."""


@dataclass(frozen=True)
class ModelArtifactPlan:
    classifier: str
    configuration_id: str
    seed: int
    path: Path
    file_sha256: str
    model_state_sha256: str
    serialized_model_bytes: int
    feature_manifest_sha256: str
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class LockedConfigurationPlan:
    role: str
    configuration_id: str
    feature_count: int
    feature_names_by_seed: tuple[tuple[int, tuple[str, ...]], ...]
    feature_hashes_by_seed: tuple[tuple[int, str], ...]

    def features_for_seed(self, seed: int) -> tuple[str, ...]:
        return dict(self.feature_names_by_seed)[seed]

    def feature_hash_for_seed(self, seed: int) -> str:
        return dict(self.feature_hashes_by_seed)[seed]


@dataclass(frozen=True)
class WorkloadHashPlan:
    seed: int
    phase: str
    workload_sha256: str
    components: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class BenchmarkPlan:
    validation_lock_path: Path
    validation_lock_sha256: str
    semantic_lock_sha256: str
    candidate_features: tuple[str, ...]
    configurations: tuple[LockedConfigurationPlan, ...]
    models: tuple[ModelArtifactPlan, ...]
    workloads: tuple[WorkloadHashPlan, ...]
    source_hashes: tuple[tuple[str, str], ...]
    attack_config_path: Path
    attack_config_sha256: str
    processed_dataset_metadata_sha256: str
    seeds: tuple[int, ...] = EXPECTED_SEEDS
    classifiers: tuple[str, ...] = CLASSIFIERS
    prediction_threshold: float = PREDICTION_THRESHOLD

    def configuration(self, configuration_id: str) -> LockedConfigurationPlan:
        matches = [item for item in self.configurations if item.configuration_id == configuration_id]
        if len(matches) != 1:
            raise V07BIntegrityError(f"Unknown or duplicate configuration: {configuration_id}")
        return matches[0]

    def model(self, classifier: str, configuration_id: str, seed: int) -> ModelArtifactPlan:
        matches = [
            item
            for item in self.models
            if (item.classifier, item.configuration_id, item.seed)
            == (classifier, configuration_id, seed)
        ]
        if len(matches) != 1:
            raise V07BIntegrityError("Missing or duplicate frozen model plan.")
        return matches[0]

    def workload(self, phase: str, seed: int) -> WorkloadHashPlan:
        matches = [item for item in self.workloads if (item.phase, item.seed) == (phase, seed)]
        if len(matches) != 1:
            raise V07BIntegrityError("Missing or duplicate frozen workload plan.")
        return matches[0]


@dataclass(frozen=True)
class V07BContext:
    lock: Mapping[str, Any]
    plan: BenchmarkPlan


@dataclass(frozen=True)
class PreparedBenchmarkWorkloads:
    by_seed_phase: Mapping[tuple[int, str], PreparedAttackSplit]
    workload_hashes: Mapping[tuple[int, str], str]

    def get(self, seed: int, phase: str) -> PreparedAttackSplit:
        return self.by_seed_phase[(seed, phase)]


@dataclass(frozen=True)
class CalibrationAttempt:
    inner_operation_count: int
    wall_time_sec: float
    worker_pid: int | None


@dataclass(frozen=True)
class CalibrationResult:
    phase: str
    target_duration_sec: float
    inner_operation_count: int
    attempts: tuple[CalibrationAttempt, ...]
    target_reached: bool = True
    pairing_scope: str = "all_feature_configurations_and_classifiers"

    def __post_init__(self) -> None:
        if self.phase != MeasurementPhase.INFERENCE.value:
            raise V07BCalibrationError("V0.7-B calibrates the timed inference block only.")
        if not self.target_reached or not self.attempts:
            raise V07BCalibrationError("A frozen calibration must have reached its target.")
        if self.attempts[-1].inner_operation_count != self.inner_operation_count:
            raise V07BCalibrationError("Frozen inner count must equal the final calibration attempt.")
        if self.attempts[-1].wall_time_sec < self.target_duration_sec:
            raise V07BCalibrationError("Calibration target was not reached.")


@dataclass(frozen=True)
class BenchmarkCell:
    classifier: str
    configuration_id: str
    seed: int
    phase: str
    inner_operation_count: int


def load_green_evaluation_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the V0.7-B protocol and reject scientific drift."""

    config_path = Path(path)
    if not config_path.is_file():
        raise V07BConfigurationError(f"Green evaluation configuration is missing: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise V07BConfigurationError(f"Cannot load green evaluation configuration: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != V07B_SCHEMA_VERSION:
        raise V07BConfigurationError(f"Configuration version must be {V07B_SCHEMA_VERSION}.")
    expected = {
        "seeds": list(EXPECTED_SEEDS),
        "classifiers": list(CLASSIFIERS),
        "semantic_roles": list(SEMANTIC_ROLES),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise V07BConfigurationError(f"Configuration {key} must be exactly {value}.")
    workload = payload.get("workload", {})
    if (
        workload.get("attack_type") != ATTACK_TYPE
        or float(workload.get("attack_rate", math.nan)) != ATTACK_RATE
        or workload.get("severity") != ATTACK_SEVERITY
    ):
        raise V07BConfigurationError("Workload must be exactly mixed/0.05/MEDIUM.")
    measurement = payload.get("measurement", {})
    if (
        measurement.get("warmup_calls") != DEFAULT_WARMUP_CALLS
        or measurement.get("outer_repetitions") != DEFAULT_OUTER_REPETITIONS
        or measurement.get("worker_policy") != "fresh_process_per_outer_repetition"
    ):
        raise V07BConfigurationError("Measurement defaults or fresh-worker policy changed.")
    calibration = measurement.get("inference_calibration", {})
    if (
        float(calibration.get("target_duration_sec", math.nan)) <= 0
        or int(calibration.get("geometric_growth_factor", 0)) != 2
        or int(calibration.get("maximum_inner_operation_count", 0)) < 1
    ):
        raise V07BConfigurationError("Inference calibration must define a positive target and safe geometric bound.")
    energy = payload.get("energy_governance", {})
    if energy.get("tier") != 1 or energy.get("direct_energy_enabled") is not False:
        raise V07BConfigurationError("V0.7-B is Tier-1 computational benchmarking only.")
    prd = _resolve_repository_path(payload.get("authoritative_prd"), PROJECT_ROOT)
    if prd.resolve() != AUTHORITATIVE_PRD_PATH.resolve() or not prd.is_file():
        raise V07BConfigurationError("docs/PhD_PRD.md is the required authoritative PRD.")
    return payload


def load_v06_benchmark_context(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    *,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    verifier: Callable[..., Mapping[str, Any]] = verify_validation_lock,
    model_loader: Callable[[Path | str], LightweightDetector] = LightweightDetector.load,
    verify_sources: bool = True,
) -> V07BContext:
    """Fail-closed loading gate for all immutable V0.6 benchmark evidence."""

    config = load_green_evaluation_config(config_path)
    results = Path(results_dir).resolve()
    lock_path = results / LOCK_FILE_NAME
    sidecar_path = results / SIDECAR_FILE_NAME
    lock = _read_json(lock_path, "validation lock")
    lock_hash = _sha256_file(lock_path)
    _verify_sidecar(sidecar_path, lock_path.name, lock_hash)
    semantic_hash = lock.get("semantic_payload_sha256")
    if not _is_sha256(semantic_hash) or semantic_payload_sha256(lock) != semantic_hash:
        raise V07BIntegrityError("Validation lock semantic payload hash mismatch.")
    try:
        verification = verifier(lock_path, source_dir=results, verify_sources=verify_sources)
    except Exception as exc:
        raise V07BIntegrityError(f"Validation lock verification failed: {exc}") from exc
    if (
        verification.get("status") != "PASS"
        or verification.get("file_sha256") != lock_hash
        or verification.get("semantic_payload_sha256") != semantic_hash
        or verification.get("sidecar_verified") is not True
        or (verify_sources and verification.get("source_hashes_verified") is not True)
    ):
        raise V07BIntegrityError("Validation lock did not pass complete verification.")

    _verify_frozen_lock_protocol(lock)
    configurations = build_locked_configuration_plans(lock)
    source_hashes: dict[str, str] = {
        str(lock_path): lock_hash,
        str(sidecar_path): _sha256_file(sidecar_path),
    }
    _verify_lock_source_hashes(lock, results, source_hashes)

    validation_audit_path = results / "validation_lock_audit.json"
    run_metadata_path = results / "run_metadata.json"
    core_runs_path = results / "core_runs.csv"
    final_dir = results / "final_test"
    robustness_dir = results / "robustness"
    final_metadata_path = final_dir / "final_test_metadata.json"
    robustness_metadata_path = robustness_dir / "robustness_metadata.json"
    robustness_audit_path = robustness_dir / "robustness_governance_audit.json"
    robustness_runs_path = robustness_dir / "robustness_runs.csv"
    validation_audit = _read_json(validation_audit_path, "V0.6-E validation audit")
    _verify_validation_audit(validation_audit, str(semantic_hash), lock_hash)
    run_metadata = _read_json(run_metadata_path, "V0.6-D run metadata")
    final_metadata = _read_json(final_metadata_path, "V0.6-F metadata")
    robustness_metadata = _read_json(robustness_metadata_path, "V0.6-G metadata")
    robustness_audit = _read_json(robustness_audit_path, "V0.6-G governance audit")
    robustness_runs = _read_csv(robustness_runs_path, "V0.6-G robustness runs")
    final_manifest_path = final_dir / "final_test_artifact_hashes.json"
    robustness_manifest_path = robustness_dir / "robustness_artifact_hashes.json"
    final_manifest = _verify_artifact_manifest(final_dir, final_manifest_path, source_hashes)
    if set(final_manifest) != v06g.REQUIRED_FINAL_TEST_FILES:
        raise V07BIntegrityError("V0.6-F artifact manifest is incomplete or unexpected.")
    robustness_manifest = _verify_artifact_manifest(
        robustness_dir, robustness_manifest_path, source_hashes
    )
    for path in (
        validation_audit_path,
        run_metadata_path,
        core_runs_path,
        final_metadata_path,
        robustness_metadata_path,
        robustness_audit_path,
    ):
        source_hashes[str(path.resolve())] = _sha256_file(path)

    attack_path = _resolve_locked_path(lock["attack_configuration"]["attack_config_path"], results)
    attack_hash = str(run_metadata.get("attack_config_sha256", ""))
    processed_hash = str(run_metadata.get("processed_dataset_metadata_sha256", ""))
    _require_file_hash(attack_path, attack_hash, "attack configuration")
    processed_path = PROCESSED_DATA_DIR / "dataset_metadata.json"
    _require_file_hash(processed_path, processed_hash, "processed dataset metadata")
    source_hashes[str(attack_path)] = attack_hash
    source_hashes[str(processed_path.resolve())] = processed_hash
    _verify_recorded_source_hash(
        PROJECT_ROOT / "src" / "pipeline_v06f.py",
        final_metadata.get("software", {}).get("pipeline_v06f_sha256"),
        "V0.6-F source",
        source_hashes,
    )
    _verify_recorded_source_hash(
        PROJECT_ROOT / "src" / "pipeline_v06g.py",
        robustness_metadata.get("software", {}).get("pipeline_v06g_sha256"),
        "V0.6-G source",
        source_hashes,
    )
    _verify_v06_completion_metadata(
        semantic_hash, lock_hash, configurations, final_metadata, robustness_metadata
    )

    dt_models = _build_decision_tree_model_plans(
        lock, configurations, results, robustness_runs, model_loader
    )
    lr_models = _build_logistic_model_plans(
        configurations,
        robustness_dir,
        robustness_manifest,
        robustness_audit,
        model_loader,
    )
    models = (*dt_models, *lr_models)
    if len(models) != len(CLASSIFIERS) * len(configurations) * len(EXPECTED_SEEDS):
        raise V07BIntegrityError("Frozen model matrix is incomplete.")
    for model in models:
        source_hashes[str(model.path)] = model.file_sha256

    core_runs = _read_csv(core_runs_path, "V0.6-D core runs")
    workloads = _build_workload_hash_plans(core_runs, configurations, final_metadata)
    plan = BenchmarkPlan(
        validation_lock_path=lock_path,
        validation_lock_sha256=lock_hash,
        semantic_lock_sha256=str(semantic_hash),
        candidate_features=tuple(lock["candidate_manifest"]["features"]),
        configurations=configurations,
        models=tuple(models),
        workloads=workloads,
        source_hashes=tuple(sorted(source_hashes.items())),
        attack_config_path=attack_path,
        attack_config_sha256=attack_hash,
        processed_dataset_metadata_sha256=processed_hash,
    )
    if config.get("validation_lock") != "results/feature_selection/validation_lock.json":
        raise V07BConfigurationError("Configured validation lock path is not the frozen V0.6 lock.")
    verify_v06_source_hashes(plan)
    return V07BContext(lock=MappingProxyType(lock), plan=plan)


def build_locked_configuration_plans(
    lock: Mapping[str, Any],
) -> tuple[LockedConfigurationPlan, ...]:
    """Derive exactly the three benchmark configurations from semantic lock roles."""

    roles = lock.get("roles")
    locked = lock.get("locked_configurations")
    if not isinstance(roles, Mapping) or not isinstance(locked, Mapping):
        raise V07BIntegrityError("Validation lock lacks roles or locked configurations.")
    plans: list[LockedConfigurationPlan] = []
    for role in SEMANTIC_ROLES:
        role_record = roles.get(role)
        if not isinstance(role_record, Mapping):
            raise V07BIntegrityError(f"Required semantic role is missing: {role}")
        configuration_id = role_record.get("locked_configuration_ref")
        if configuration_id != role_record.get("configuration_id") or configuration_id not in locked:
            raise V07BIntegrityError(f"Invalid lock reference for role: {role}")
        configuration = locked[configuration_id]
        seed_specific = configuration.get("seed_specific")
        if not isinstance(seed_specific, Mapping) or set(seed_specific) != {
            str(seed) for seed in EXPECTED_SEEDS
        }:
            raise V07BIntegrityError("Each configuration must contain exactly seeds 42-46.")
        feature_rows: list[tuple[int, tuple[str, ...]]] = []
        hash_rows: list[tuple[int, str]] = []
        expected_count = EXPECTED_FEATURE_COUNTS[role]
        for seed in EXPECTED_SEEDS:
            row = seed_specific[str(seed)]
            features = row.get("selected_features")
            feature_hash = row.get("selected_features_sha256")
            if (
                not isinstance(features, list)
                or any(not isinstance(item, str) for item in features)
                or len(features) != expected_count
                or len(set(features)) != len(features)
                or fingerprint_feature_names(features) != feature_hash
            ):
                raise V07BIntegrityError(
                    f"Feature count/order/hash mismatch: {configuration_id}, seed {seed}"
                )
            if configuration.get("selected_features", {}).get(str(seed)) != features:
                raise V07BIntegrityError("Locked feature map differs from seed-specific order.")
            if configuration.get("selected_feature_fingerprints", {}).get(str(seed)) != feature_hash:
                raise V07BIntegrityError("Locked feature fingerprint map is inconsistent.")
            feature_rows.append((seed, tuple(features)))
            hash_rows.append((seed, str(feature_hash)))
        if int(configuration.get("actual_feature_count", -1)) != expected_count:
            raise V07BIntegrityError(f"Unexpected feature count for {configuration_id}.")
        plans.append(
            LockedConfigurationPlan(
                role,
                str(configuration_id),
                expected_count,
                tuple(feature_rows),
                tuple(hash_rows),
            )
        )
    ids = tuple(item.configuration_id for item in plans)
    if len(set(ids)) != 3 or ids != EXPECTED_CONFIGURATION_IDS:
        raise V07BIntegrityError("The lock must resolve exactly the frozen 43/42/11 configurations.")
    return tuple(plans)


def verify_v06_source_hashes(plan: BenchmarkPlan) -> None:
    """Recheck every source captured by the immutable V0.7-B loading gate."""

    for name, expected in plan.source_hashes:
        _require_file_hash(Path(name), expected, "V0.6 source")


def prepare_verified_workloads(
    context: V07BContext,
    *,
    processed_dir: Path | str = PROCESSED_DATA_DIR,
    loader: Callable[..., Any] = load_processed_dataco_splits,
    attack_preparer: Callable[..., PreparedAttackSplit] = prepare_attack_split,
) -> PreparedBenchmarkWorkloads:
    """Prepare one train and test manifestation per seed and verify V0.6 hashes.

    Each returned object is shared by every feature configuration and classifier
    in its seed/phase pairing block. No validation split is loaded.
    """

    verify_v06_source_hashes(context.plan)
    processed = Path(processed_dir)
    metadata_path = processed / "dataset_metadata.json"
    _require_file_hash(
        metadata_path,
        context.plan.processed_dataset_metadata_sha256,
        "processed dataset metadata",
    )
    bundle = loader(("train", "test"), processed_dir=processed)
    if set(bundle.splits) != {"train", "test"}:
        raise V07BIntegrityError("Workload loader must expose exactly train and test.")
    if tuple(bundle.selected_features) != context.plan.candidate_features:
        raise V07BIntegrityError("Loaded candidate feature order differs from the lock.")
    v06g.verify_loaded_splits(bundle, context.plan.candidate_features, bundle.dataset_metadata)
    attack = context.lock["attack_configuration"]
    prepared_by_key: dict[tuple[int, str], PreparedAttackSplit] = {}
    hashes: dict[tuple[int, str], str] = {}
    for seed in context.plan.seeds:
        train = attack_preparer(
            split=bundle.splits["train"],
            training_reference=bundle.splits["train"],
            candidate_features=context.plan.candidate_features,
            attack_type=ATTACK_TYPE,
            attack_rate=ATTACK_RATE,
            severity=ATTACK_SEVERITY,
            random_seed=seed,
            experiment_id=f"v0_6_development_s{seed}_train",
            config_path=attack["attack_config_path"],
        )
        test = attack_preparer(
            split=bundle.splits["test"],
            training_reference=bundle.splits["train"],
            candidate_features=context.plan.candidate_features,
            attack_type=ATTACK_TYPE,
            attack_rate=ATTACK_RATE,
            severity=ATTACK_SEVERITY,
            random_seed=seed,
            experiment_id=f"v0_6_f_final_test_s{seed}",
            config_path=attack["attack_config_path"],
            test_authorization_id=context.plan.semantic_lock_sha256,
            test_authorization_capability=context,
        )
        for phase, manifestation in (
            (MeasurementPhase.TRAINING.value, train),
            (MeasurementPhase.INFERENCE.value, test),
        ):
            _verify_prepared_workload(manifestation, phase, seed, context.plan)
            fingerprint = v06f._manifestation_fingerprint(manifestation)
            expected = context.plan.workload(phase, seed)
            if fingerprint != expected.workload_sha256:
                raise V07BIntegrityError(f"{phase} workload fingerprint mismatch for seed {seed}.")
            prepared_by_key[(seed, phase)] = manifestation
            hashes[(seed, phase)] = fingerprint
    verify_v06_source_hashes(context.plan)
    return PreparedBenchmarkWorkloads(
        MappingProxyType(prepared_by_key), MappingProxyType(hashes)
    )


def calibrate_inference_inner_count(
    operation: Callable[[Any], Any],
    *,
    prepare: Callable[[], Any] | None = None,
    target_duration_sec: float = 1.0,
    maximum_inner_operation_count: int = 16_384,
    warmup_calls: int = DEFAULT_WARMUP_CALLS,
    timeout_seconds: float = 300.0,
    start_method: str | None = None,
    runner: Callable[..., FreshWorkerRun] = run_fresh_worker_protocol,
) -> CalibrationResult:
    """Geometrically calibrate one inference block in fresh workers."""

    if not math.isfinite(target_duration_sec) or target_duration_sec <= 0:
        raise V07BCalibrationError("Calibration target must be finite and positive.")
    if maximum_inner_operation_count < 1:
        raise V07BCalibrationError("Calibration safety maximum must be positive.")
    attempts: list[CalibrationAttempt] = []
    count = 1
    while count <= maximum_inner_operation_count:
        run = runner(
            operation,
            prepare=prepare,
            inner_operation_count=count,
            phase=MeasurementPhase.INFERENCE,
            warmup_calls=warmup_calls,
            outer_repetitions=1,
            records_per_operation=1,
            timeout_seconds=timeout_seconds,
            measurement_scope="prediction_only_calibration",
            start_method=start_method,
        )
        result = run.results[0]
        if result.status is not WorkerStatus.COMPLETED or result.observation is None:
            detail = result.failure.message if result.failure is not None else result.status.value
            raise V07BCalibrationError(f"Calibration worker failed at count {count}: {detail}")
        wall = result.observation.resources.wall_time_sec
        attempts.append(CalibrationAttempt(count, wall, result.worker_pid))
        if wall >= target_duration_sec:
            return CalibrationResult(
                MeasurementPhase.INFERENCE.value,
                target_duration_sec,
                count,
                tuple(attempts),
            )
        count *= 2
    raise V07BCalibrationError(
        "Calibration target was not reached before maximum_inner_operation_count."
    )


def build_benchmark_schedule(
    plan: BenchmarkPlan,
    calibration: CalibrationResult,
    *,
    training_inner_operation_count: int = 1,
) -> tuple[BenchmarkCell, ...]:
    """Build, but never execute, the paired 60-cell computational schedule."""

    if training_inner_operation_count != 1:
        raise V07BConfigurationError("Each training observation must measure one model.fit call.")
    cells = tuple(
        BenchmarkCell(
            classifier,
            configuration.configuration_id,
            seed,
            phase,
            (
                calibration.inner_operation_count
                if phase == MeasurementPhase.INFERENCE.value
                else training_inner_operation_count
            ),
        )
        for phase in (MeasurementPhase.TRAINING.value, MeasurementPhase.INFERENCE.value)
        for seed in plan.seeds
        for classifier in plan.classifiers
        for configuration in plan.configurations
    )
    inference_counts = {
        cell.inner_operation_count
        for cell in cells
        if cell.phase == MeasurementPhase.INFERENCE.value
    }
    if inference_counts != {calibration.inner_operation_count}:
        raise V07BIntegrityError("Calibrated inner count was not reused across paired cells.")
    return cells


def verify_same_workload_reused(
    assignments: Sequence[tuple[BenchmarkCell, PreparedAttackSplit]],
) -> None:
    """Require object-identical workload reuse within every seed/phase block."""

    grouped: dict[tuple[int, str], set[int]] = {}
    for cell, workload in assignments:
        grouped.setdefault((cell.seed, cell.phase), set()).add(id(workload))
    if any(len(object_ids) != 1 for object_ids in grouped.values()):
        raise V07BIntegrityError("Paired configurations/classifiers did not reuse one workload object.")


@dataclass
class _TrainingState:
    model: LightweightDetector
    features: pd.DataFrame
    labels: pd.Series


@dataclass(frozen=True)
class _TrainingPreparation:
    classifier: str
    feature_names: tuple[str, ...]
    parameters: Mapping[str, Any]
    seed: int
    features: pd.DataFrame
    labels: pd.Series

    def __call__(self) -> _TrainingState:
        model = create_model(
            self.classifier,
            self.feature_names,
            dict(self.parameters),
            random_state=self.seed,
        )
        return _TrainingState(model, self.features, self.labels)


def _training_fit_only(state: _TrainingState) -> None:
    state.model.fit(state.features, state.labels)


@dataclass(frozen=True)
class _InferenceState:
    model: LightweightDetector
    features: pd.DataFrame


@dataclass(frozen=True)
class _InferencePreparation:
    artifact: ModelArtifactPlan
    features: pd.DataFrame

    def __call__(self) -> _InferenceState:
        _require_file_hash(self.artifact.path, self.artifact.file_sha256, "frozen model")
        model = LightweightDetector.load(self.artifact.path)
        _verify_model_against_plan(model, self.artifact)
        return _InferenceState(model, self.features)


def _prediction_only(state: _InferenceState) -> None:
    state.model.predict_frame(state.features)


def run_benchmark_cell(
    context: V07BContext,
    workloads: PreparedBenchmarkWorkloads,
    cell: BenchmarkCell,
    *,
    environment_id: str,
    warmup_calls: int = DEFAULT_WARMUP_CALLS,
    outer_repetitions: int = DEFAULT_OUTER_REPETITIONS,
    timeout_seconds: float = 300.0,
    start_method: str | None = None,
    runner: Callable[..., FreshWorkerRun] = run_fresh_worker_protocol,
) -> tuple[dict[str, Any], ...]:
    """Run one explicitly requested benchmark cell using V0.7-A fresh workers."""

    if not environment_id.strip():
        raise V07BConfigurationError("environment_id must not be empty.")
    if cell.classifier not in context.plan.classifiers or cell.seed not in context.plan.seeds:
        raise V07BIntegrityError("Unexpected classifier or seed in benchmark cell.")
    if cell.phase not in {MeasurementPhase.TRAINING.value, MeasurementPhase.INFERENCE.value}:
        raise V07BIntegrityError("Benchmark phase must be training or inference.")
    configuration = context.plan.configuration(cell.configuration_id)
    artifact = context.plan.model(cell.classifier, cell.configuration_id, cell.seed)
    manifestation = workloads.get(cell.seed, cell.phase)
    expected_workload_hash = context.plan.workload(cell.phase, cell.seed).workload_sha256
    if workloads.workload_hashes[(cell.seed, cell.phase)] != expected_workload_hash:
        raise V07BIntegrityError("Prepared workload hash changed before measurement.")
    feature_names = configuration.features_for_seed(cell.seed)
    selected = manifestation.features.loc[:, list(feature_names)].copy()
    selected_bytes = dataframe_deep_memory_bytes(selected)
    dense_bytes = numpy_dense_nbytes(selected.to_numpy(dtype=float))
    if cell.phase == MeasurementPhase.TRAINING.value:
        parameters = DT_PARAMETERS if cell.classifier == "decision_tree" else LR_PARAMETERS
        prepare = _TrainingPreparation(
            cell.classifier,
            feature_names,
            dict(parameters),
            cell.seed,
            selected,
            manifestation.labels.copy(),
        )
        operation = _training_fit_only
        scope = "model.fit_only_data_loading_selection_serialization_excluded"
    else:
        prepare = _InferencePreparation(artifact, selected)
        operation = _prediction_only
        scope = "prediction_only_model_loading_warmup_excluded"
    verify_v06_source_hashes(context.plan)
    run = runner(
        operation,
        prepare=prepare,
        inner_operation_count=cell.inner_operation_count,
        phase=cell.phase,
        warmup_calls=warmup_calls,
        outer_repetitions=outer_repetitions,
        records_per_operation=len(selected),
        timeout_seconds=timeout_seconds,
        measurement_scope=scope,
        start_method=start_method,
    )
    rows = _worker_rows(
        run,
        cell,
        environment_id,
        configuration,
        artifact,
        expected_workload_hash,
        selected_bytes,
        dense_bytes,
        context.plan.validation_lock_sha256,
        len(selected),
    )
    verify_output_schema(rows)
    verify_v06_source_hashes(context.plan)
    return rows


def verify_output_schema(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate manifest-compatible computational rows and provenance."""

    if not rows:
        raise V07BIntegrityError("Benchmark output must retain at least one worker result.")
    for row in rows:
        missing = OUTPUT_REQUIRED_FIELDS - set(row)
        if missing:
            raise V07BIntegrityError(f"Benchmark output is missing fields: {sorted(missing)}")
        if row.get("measurement_kind") != MeasurementProvenance.DIRECT_COMPUTATIONAL.value:
            raise V07BIntegrityError("Tier-1 observations must be DIRECT_COMPUTATIONAL, not energy.")
        if row.get("direct_energy_joules") is not None:
            raise V07BIntegrityError("V0.7-B cannot emit direct-energy observations.")
        if row.get("status") == WorkerStatus.COMPLETED.value:
            numeric_nonnegative = (
                "process_cpu_time_sec",
                "start_rss_bytes",
                "end_rss_bytes",
                "absolute_peak_rss_bytes",
                "incremental_peak_rss_bytes",
                "selected_input_bytes",
                "serialized_model_bytes",
                "per_record_latency_sec",
                "throughput_records_sec",
            )
            if not math.isfinite(float(row["wall_time_sec"])) or row["wall_time_sec"] <= 0:
                raise V07BIntegrityError("Completed wall time must be finite and positive.")
            if any(row[name] is None or float(row[name]) < 0 for name in numeric_nonnegative):
                raise V07BIntegrityError("Completed resource metrics must be nonnegative.")
            if row["absolute_peak_rss_bytes"] < max(
                row["start_rss_bytes"], row["end_rss_bytes"]
            ):
                raise V07BIntegrityError("RSS fields violate the absolute peak invariant.")
        elif not row.get("error_type") or not row.get("error_stage"):
            raise V07BIntegrityError("Failed workers require structured error fields.")


def _worker_rows(
    run: FreshWorkerRun,
    cell: BenchmarkCell,
    environment_id: str,
    configuration: LockedConfigurationPlan,
    artifact: ModelArtifactPlan,
    workload_hash: str,
    selected_bytes: int,
    dense_bytes: int,
    lock_hash: str,
    record_count: int,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for result in run.results:
        base: dict[str, Any] = {
            "schema_version": V07B_SCHEMA_VERSION,
            "stage": V07B_STAGE,
            "status": result.status.value,
            "environment_id": environment_id,
            "classifier": cell.classifier,
            "configuration_id": cell.configuration_id,
            "seed": cell.seed,
            "phase": cell.phase,
            "feature_count": configuration.feature_count,
            "record_count": record_count,
            "outer_repetition": result.outer_repetition,
            "inner_operation_count": cell.inner_operation_count,
            "measured_operation_count": result.measured_operation_count,
            "warmup_calls": run.protocol.warmup_calls,
            "worker_pid": result.worker_pid,
            "wall_time_sec": None,
            "process_cpu_time_sec": None,
            "start_rss_bytes": None,
            "end_rss_bytes": None,
            "absolute_peak_rss_bytes": None,
            "incremental_peak_rss_bytes": None,
            "selected_input_bytes": selected_bytes,
            "numpy_dense_nbytes": dense_bytes,
            "serialized_model_bytes": artifact.serialized_model_bytes,
            "inference_latency_sec": None,
            "per_record_latency_sec": None,
            "throughput_records_sec": None,
            "measurement_kind": MeasurementProvenance.DIRECT_COMPUTATIONAL.value,
            "derived_measurement_kind": MeasurementProvenance.DERIVED.value,
            "direct_energy_joules": None,
            "energy_directly_measured": False,
            "workload_sha256": workload_hash,
            "model_sha256": artifact.file_sha256,
            "feature_manifest_sha256": artifact.feature_manifest_sha256,
            "v06_lock_sha256": lock_hash,
            "measurement_scope": run.protocol.measurement_scope,
            "error_type": None,
            "error_message": None,
            "error_stage": None,
        }
        if result.observation is not None:
            observation = result.observation
            resources = observation.resources
            base.update(
                {
                    "wall_time_sec": resources.wall_time_sec,
                    "process_cpu_time_sec": resources.cpu_time_sec,
                    "start_rss_bytes": resources.start_rss_bytes,
                    "end_rss_bytes": resources.end_rss_bytes,
                    "absolute_peak_rss_bytes": observation.absolute_peak_rss_bytes,
                    "incremental_peak_rss_bytes": observation.incremental_peak_rss_bytes,
                    "inference_latency_sec": observation.inference_latency_seconds,
                    "per_record_latency_sec": observation.per_record_latency_seconds,
                    "throughput_records_sec": observation.throughput_records_per_second,
                }
            )
        else:
            failure = result.failure
            base.update(
                {
                    "error_type": None if failure is None else failure.error_type,
                    "error_message": None if failure is None else failure.message,
                    "error_stage": result.status.value.lower() if failure is None else failure.stage,
                }
            )
        rows.append(base)
    return tuple(rows)


def _verify_frozen_lock_protocol(lock: Mapping[str, Any]) -> None:
    attack = lock.get("attack_configuration")
    model = lock.get("model_configuration")
    threshold = lock.get("threshold")
    if tuple(lock.get("seeds", ())) != EXPECTED_SEEDS:
        raise V07BIntegrityError("Seeds must be exactly 42-46.")
    if not isinstance(attack, Mapping) or (
        attack.get("attack_type"), attack.get("attack_rate"), attack.get("attack_severity")
    ) != (ATTACK_TYPE, ATTACK_RATE, ATTACK_SEVERITY):
        raise V07BIntegrityError("Locked workload must be exactly mixed/0.05/MEDIUM.")
    if model != {"model_id": "decision_tree", "parameters": dict(DT_PARAMETERS)}:
        raise V07BIntegrityError("Decision Tree configuration differs from frozen V0.6.")
    if (
        not isinstance(threshold, Mapping)
        or threshold.get("prediction_threshold") != PREDICTION_THRESHOLD
        or lock.get("decision_threshold") != PREDICTION_THRESHOLD
    ):
        raise V07BIntegrityError("Prediction threshold must remain frozen at 0.5.")
    candidate = lock.get("candidate_manifest")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("feature_count") != 43
        or fingerprint_feature_names(candidate.get("features", ())) != candidate.get("sha256")
        or candidate.get("sha256") != lock.get("candidate_manifest_hash")
    ):
        raise V07BIntegrityError("Candidate feature manifest is invalid.")


def _verify_lock_source_hashes(
    lock: Mapping[str, Any], results: Path, collected: dict[str, str]
) -> None:
    declared = lock.get("source_artifact_hashes")
    if not isinstance(declared, Mapping) or not declared:
        raise V07BIntegrityError("Validation lock source hashes are missing.")
    for reference, expected in declared.items():
        path = _resolve_locked_path(str(reference), results)
        _require_file_hash(path, expected, "validation-lock source")
        collected[str(path)] = str(expected)


def _verify_artifact_manifest(
    root: Path, manifest_path: Path, collected: dict[str, str]
) -> dict[str, str]:
    manifest = _read_json(manifest_path, f"artifact hash manifest {manifest_path.name}")
    if not manifest or any(not isinstance(key, str) or not _is_sha256(value) for key, value in manifest.items()):
        raise V07BIntegrityError(f"Corrupted artifact hash manifest: {manifest_path}")
    resolved_root = root.resolve()
    for relative, expected in manifest.items():
        path = (root / relative).resolve()
        if resolved_root not in path.parents:
            raise V07BIntegrityError("Artifact manifest path escapes its immutable directory.")
        _require_file_hash(path, expected, "declared V0.6 artifact")
        collected[str(path)] = expected
    collected[str(manifest_path.resolve())] = _sha256_file(manifest_path)
    return manifest


def _verify_v06_completion_metadata(
    semantic_hash: str,
    lock_hash: str,
    configurations: Sequence[LockedConfigurationPlan],
    final: Mapping[str, Any],
    robustness: Mapping[str, Any],
) -> None:
    role_mapping = {item.role: item.configuration_id for item in configurations}
    if (
        final.get("stage") != "V0.6-F"
        or final.get("status") != "COMPLETED"
        or final.get("failed_run_count") != 0
        or final.get("semantic_lock_sha256") != semantic_hash
        or final.get("validation_lock_file_sha256") != lock_hash
        or final.get("number_of_unique_locked_configurations") != 3
    ):
        raise V07BIntegrityError("V0.6-F completion metadata is invalid.")
    if (
        robustness.get("stage") != "V0.6-G"
        or robustness.get("status") != "COMPLETED"
        or robustness.get("failed_run_count") != 0
        or robustness.get("completed_run_count") != 390
        or robustness.get("logistic_fit_count") != 15
        or robustness.get("unique_configuration_count") != 3
        or tuple(robustness.get("seeds", ())) != EXPECTED_SEEDS
        or robustness.get("semantic_lock_sha256") != semantic_hash
        or robustness.get("validation_lock_file_sha256") != lock_hash
        or robustness.get("role_mapping") != role_mapping
        or robustness.get("lr_parameters")
        != {**dict(LR_PARAMETERS), "random_state": "seed"}
        or robustness.get("prediction_threshold") != PREDICTION_THRESHOLD
    ):
        raise V07BIntegrityError("V0.6-G frozen classifier metadata is invalid.")


def _verify_validation_audit(
    audit: Mapping[str, Any], semantic_hash: str, lock_hash: str
) -> None:
    required = {
        "source_d_leakage_audit_pass",
        "source_d_test_accessed_false",
        "source_d_metadata_test_accessed_false",
        "source_hashes_verified",
        "source_semantics_reconstructed",
        "semantic_payload_verified",
        "sidecar_file_hash_verified",
    }
    passed = {
        item.get("name")
        for item in audit.get("checks", ())
        if isinstance(item, Mapping) and item.get("passed") is True
    }
    if (
        audit.get("stage") != "V0.6-E"
        or audit.get("status") != "PASS"
        or audit.get("test_accessed") is not False
        or audit.get("semantic_payload_sha256") != semantic_hash
        or audit.get("validation_lock_file_sha256") != lock_hash
        or not required.issubset(passed)
    ):
        raise V07BIntegrityError("V0.6-E validation-lock audit is invalid.")


def _build_decision_tree_model_plans(
    lock: Mapping[str, Any],
    configurations: Sequence[LockedConfigurationPlan],
    results: Path,
    robustness_runs: pd.DataFrame,
    model_loader: Callable[[Path | str], LightweightDetector],
) -> tuple[ModelArtifactPlan, ...]:
    records: list[ModelArtifactPlan] = []
    for configuration in configurations:
        source = lock["locked_configurations"][configuration.configuration_id]["seed_specific"]
        for seed in EXPECTED_SEEDS:
            artifact = source[str(seed)].get("model_artifact")
            if not isinstance(artifact, Mapping) or artifact.get("availability") != "hashed":
                raise V07BIntegrityError("Missing hashed Decision Tree artifact.")
            historical = robustness_runs.loc[
                robustness_runs["classifier"].eq("decision_tree")
                & robustness_runs["configuration_id"].eq(configuration.configuration_id)
                & robustness_runs["seed"].eq(seed)
            ]
            state_hashes = tuple(historical["model_state_sha256"].drop_duplicates())
            locked_state_hashes = tuple(
                historical["locked_model_state_sha256"].drop_duplicates()
            )
            if (
                len(state_hashes) != 1
                or not _is_sha256(state_hashes[0])
                or locked_state_hashes != (artifact.get("model_state_sha256"),)
            ):
                raise V07BIntegrityError("V0.6-G Decision Tree state evidence is inconsistent.")
            path = _resolve_locked_path(str(artifact.get("reference", "")), results)
            plan = ModelArtifactPlan(
                "decision_tree",
                configuration.configuration_id,
                seed,
                path,
                str(artifact.get("file_sha256", "")),
                str(state_hashes[0]),
                int(artifact.get("serialized_model_bytes", -1)),
                configuration.feature_hash_for_seed(seed),
                configuration.features_for_seed(seed),
            )
            _verify_model_file_and_state(plan, model_loader, state_hasher=v06g.model_state_sha256)
            records.append(plan)
    return tuple(records)


def _build_logistic_model_plans(
    configurations: Sequence[LockedConfigurationPlan],
    robustness_dir: Path,
    artifact_manifest: Mapping[str, str],
    audit: Mapping[str, Any],
    model_loader: Callable[[Path | str], LightweightDetector],
) -> tuple[ModelArtifactPlan, ...]:
    if audit.get("stage") != "V0.6-G" or audit.get("status") != "PASS":
        raise V07BIntegrityError("V0.6-G governance audit did not pass.")
    declared = audit.get("lr_model_artifacts")
    if not isinstance(declared, Mapping) or len(declared) != 15:
        raise V07BIntegrityError("V0.6-G must declare exactly 15 frozen LR models.")
    records: list[ModelArtifactPlan] = []
    for configuration in configurations:
        for seed in EXPECTED_SEEDS:
            key = f"{configuration.configuration_id}:{seed}"
            raw = declared.get(key)
            if not isinstance(raw, Mapping):
                raise V07BIntegrityError(f"Missing Logistic Regression model: {key}")
            path = _resolve_locked_path(str(raw.get("path", "")), robustness_dir)
            relative = path.relative_to(robustness_dir.resolve()).as_posix()
            if artifact_manifest.get(relative) != raw.get("file_sha256"):
                raise V07BIntegrityError("LR model hash differs between V0.6-G manifests.")
            plan = ModelArtifactPlan(
                "logistic_regression",
                configuration.configuration_id,
                seed,
                path,
                str(raw.get("file_sha256", "")),
                str(raw.get("model_state_sha256", "")),
                path.stat().st_size,
                configuration.feature_hash_for_seed(seed),
                configuration.features_for_seed(seed),
            )
            _verify_model_file_and_state(plan, model_loader, state_hasher=v06g.model_state_sha256)
            records.append(plan)
    return tuple(records)


def _verify_model_file_and_state(
    plan: ModelArtifactPlan,
    loader: Callable[[Path | str], LightweightDetector],
    *,
    state_hasher: Callable[[LightweightDetector], str],
) -> None:
    _require_file_hash(plan.path, plan.file_sha256, "frozen model")
    if plan.path.stat().st_size != plan.serialized_model_bytes:
        raise V07BIntegrityError(f"Frozen model size mismatch: {plan.path}")
    try:
        model = loader(plan.path)
    except Exception as exc:
        raise V07BIntegrityError(f"Cannot load frozen model: {plan.path}") from exc
    _verify_model_against_plan(model, plan)
    if state_hasher(model) != plan.model_state_sha256:
        raise V07BIntegrityError(f"Frozen model semantic state hash mismatch: {plan.path}")
    _require_file_hash(plan.path, plan.file_sha256, "frozen model")


def _verify_model_against_plan(model: Any, plan: ModelArtifactPlan) -> None:
    estimator = getattr(model, "estimator", None)
    if (
        not isinstance(model, LightweightDetector)
        or not model.is_fitted
        or model.model_name != plan.classifier
        or model.random_state != plan.seed
        or tuple(model.feature_names) != plan.feature_names
        or fingerprint_feature_names(model.feature_names) != plan.feature_manifest_sha256
    ):
        raise V07BIntegrityError("Frozen model classifier/seed/features are inconsistent.")
    if plan.classifier == "decision_tree":
        valid = (
            model.parameters == dict(DT_PARAMETERS)
            and getattr(estimator, "max_depth", None) == 5
            and getattr(estimator, "min_samples_leaf", None) == 20
            and getattr(estimator, "class_weight", None) == "balanced"
            and getattr(estimator, "random_state", None) == plan.seed
        )
    elif plan.classifier == "logistic_regression":
        valid = (
            model.parameters == dict(LR_PARAMETERS)
            and getattr(estimator, "solver", None) == "liblinear"
            and getattr(estimator, "class_weight", None) == "balanced"
            and getattr(estimator, "max_iter", None) == 500
            and getattr(estimator, "C", None) == 1.0
            and getattr(estimator, "random_state", None) == plan.seed
        )
    else:
        valid = False
    if not valid:
        raise V07BIntegrityError("Unexpected frozen model configuration.")


def _build_workload_hash_plans(
    core: pd.DataFrame,
    configurations: Sequence[LockedConfigurationPlan],
    final_metadata: Mapping[str, Any],
) -> tuple[WorkloadHashPlan, ...]:
    required = {"seed", "configuration_id", *_WORKLOAD_COMPONENT_COLUMNS.values()}
    if not required.issubset(core.columns):
        raise V07BIntegrityError("V0.6-D core runs lack workload hashes.")
    selected_ids = {item.configuration_id for item in configurations}
    selected = core.loc[core["configuration_id"].isin(selected_ids)]
    final_hashes = final_metadata.get("manifestation_fingerprints")
    if not isinstance(final_hashes, Mapping) or set(final_hashes) != {
        str(seed) for seed in EXPECTED_SEEDS
    }:
        raise V07BIntegrityError("V0.6-F workload hashes are incomplete.")
    plans: list[WorkloadHashPlan] = []
    for seed in EXPECTED_SEEDS:
        rows = selected.loc[selected["seed"].eq(seed)]
        if set(rows["configuration_id"]) != selected_ids:
            raise V07BIntegrityError("V0.6-D training workload matrix is incomplete.")
        components: dict[str, str] = {}
        for key, column in _WORKLOAD_COMPONENT_COLUMNS.items():
            values = tuple(rows[column].drop_duplicates())
            if len(values) != 1 or not _is_sha256(values[0]):
                raise V07BIntegrityError("Training workload was not identical across configurations.")
            components[key] = str(values[0])
        plans.append(
            WorkloadHashPlan(
                seed,
                MeasurementPhase.TRAINING.value,
                _composite_workload_hash(components),
                tuple(components.items()),
            )
        )
        inference_hash = final_hashes[str(seed)]
        if not _is_sha256(inference_hash):
            raise V07BIntegrityError("Invalid V0.6-F manifestation SHA-256.")
        plans.append(
            WorkloadHashPlan(
                seed,
                MeasurementPhase.INFERENCE.value,
                str(inference_hash),
                (),
            )
        )
    return tuple(plans)


def _verify_prepared_workload(
    prepared: PreparedAttackSplit, phase: str, seed: int, plan: BenchmarkPlan
) -> None:
    expected_split = "train" if phase == MeasurementPhase.TRAINING.value else "test"
    if (
        not isinstance(prepared, PreparedAttackSplit)
        or prepared.split_name != expected_split
        or tuple(prepared.candidate_features) != plan.candidate_features
        or prepared.labels.name != "is_attack"
        or prepared.attack_metadata.get("attack_mode") != ATTACK_TYPE
        or prepared.attack_metadata.get("configured_attack_rate") != ATTACK_RATE
        or prepared.attack_metadata.get("severity") != ATTACK_SEVERITY
        or prepared.attack_metadata.get("random_seed") != seed
    ):
        raise V07BIntegrityError("Prepared workload violates mixed/0.05/MEDIUM governance.")
    expected = plan.workload(phase, seed)
    if expected.components:
        observed = _prepared_components(prepared)
        if observed != dict(expected.components):
            raise V07BIntegrityError("Prepared training workload component hash mismatch.")


def _prepared_components(prepared: PreparedAttackSplit) -> dict[str, str]:
    v06f._manifestation_fingerprint(prepared)
    return {
        "row_ids_sha256": prepared.row_ids_sha256,
        "clean_features_sha256": prepared.clean_features_sha256,
        "attacked_features_sha256": prepared.attacked_features_sha256,
        "labels_sha256": prepared.labels_sha256,
        "ground_truth_sha256": prepared.ground_truth_sha256,
        "manifest_sha256": prepared.manifest_sha256,
        "attack_metadata_sha256": prepared.attack_metadata_sha256,
    }


def _composite_workload_hash(components: Mapping[str, str]) -> str:
    payload = json.dumps(dict(components), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verify_recorded_source_hash(
    path: Path,
    expected: Any,
    description: str,
    collected: dict[str, str],
) -> None:
    _require_file_hash(path, expected, description)
    collected[str(path.resolve())] = str(expected)


def _require_file_hash(path: Path, expected: Any, description: str) -> None:
    if not _is_sha256(expected):
        raise V07BIntegrityError(f"{description} has no valid SHA-256 declaration.")
    if not path.is_file():
        raise V07BIntegrityError(f"Missing {description}: {path}")
    if _sha256_file(path) != expected:
        raise V07BIntegrityError(f"{description} hash mismatch: {path}")


def _resolve_locked_path(reference: str, base: Path) -> Path:
    if not reference:
        raise V07BIntegrityError("Locked artifact path is empty.")
    path = Path(reference)
    candidates = [path] if path.is_absolute() else [base / path, base.parent / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise V07BIntegrityError(f"Missing locked artifact: {reference}")


def _resolve_repository_path(reference: Any, base: Path) -> Path:
    if not isinstance(reference, str) or not reference:
        raise V07BConfigurationError("Required repository path is missing.")
    path = Path(reference)
    return path if path.is_absolute() else base / path


def _verify_sidecar(path: Path, lock_name: str, lock_hash: str) -> None:
    try:
        parts = path.read_text(encoding="ascii").strip().split()
    except OSError as exc:
        raise V07BIntegrityError(f"Validation lock sidecar is required: {exc}") from exc
    if len(parts) != 2 or parts != [lock_hash, lock_name]:
        raise V07BIntegrityError("Validation lock file SHA-256 sidecar mismatch.")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V07BIntegrityError(f"Cannot load {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise V07BIntegrityError(f"{description} must be a JSON object.")
    return value


def _read_csv(path: Path, description: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise V07BIntegrityError(f"Cannot load {description}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise V07BIntegrityError(f"Cannot hash required file: {path}") from exc
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def benchmark_plan_sha256(plan: BenchmarkPlan) -> str:
    """Fingerprint the frozen benchmark plan for a later non-result manifest."""

    payload = json.dumps(asdict(plan), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Stable aliases for downstream stage naming.
immutable_v06_loading_gate = load_v06_benchmark_context
calibrate_inner_operation_count = calibrate_inference_inner_count
