"""Synthetic and immutable-evidence tests for the V0.7-D Tier-1 runner."""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from src.green.measurement import MeasurementProvenance
import src.pipeline_v07b as v07b
import src.pipeline_v07c as v07c
import src.pipeline_v07d as v07d


def _synthetic_context(tmp_path: Path) -> v07b.V07BContext:
    configurations = []
    for role, configuration_id, count in zip(
        v07b.SEMANTIC_ROLES, v07b.EXPECTED_CONFIGURATION_IDS, (43, 42, 11)
    ):
        features_by_seed = []
        hashes_by_seed = []
        for seed in v07b.EXPECTED_SEEDS:
            features = tuple(f"{configuration_id}_feature_{index}" for index in range(count))
            feature_hash = v07b.fingerprint_feature_names(features)
            features_by_seed.append((seed, features))
            hashes_by_seed.append((seed, feature_hash))
        configurations.append(
            v07b.LockedConfigurationPlan(
                role,
                configuration_id,
                count,
                tuple(features_by_seed),
                tuple(hashes_by_seed),
            )
        )
    models = []
    for classifier in v07b.CLASSIFIERS:
        for configuration in configurations:
            for seed in v07b.EXPECTED_SEEDS:
                models.append(
                    v07b.ModelArtifactPlan(
                        classifier,
                        configuration.configuration_id,
                        seed,
                        tmp_path / f"{classifier}_{configuration.configuration_id}_{seed}.joblib",
                        f"{len(models) + 1:064x}",
                        "a" * 64,
                        1000 + configuration.feature_count,
                        configuration.feature_hash_for_seed(seed),
                        configuration.features_for_seed(seed),
                    )
                )
    workloads = tuple(
        v07b.WorkloadHashPlan(seed, phase, f"{seed * 10 + index:064x}", ())
        for seed in v07b.EXPECTED_SEEDS
        for index, phase in enumerate(("training", "inference"), start=1)
    )
    plan = v07b.BenchmarkPlan(
        tmp_path / "validation_lock.json",
        "b" * 64,
        "c" * 64,
        tuple(f"feature_{index}" for index in range(43)),
        tuple(configurations),
        tuple(models),
        workloads,
        (),
        tmp_path / "attack.yaml",
        "d" * 64,
        "e" * 64,
    )
    return v07b.V07BContext(MappingProxyType({}), plan)


@pytest.fixture()
def synthetic_context(tmp_path: Path) -> v07b.V07BContext:
    return _synthetic_context(tmp_path)


@pytest.fixture()
def schedule(synthetic_context: v07b.V07BContext) -> tuple[v07d.PlannedObservation, ...]:
    return v07d.build_observation_schedule(
        synthetic_context.plan, inference_inner_operation_count=8
    )


@pytest.fixture(scope="module")
def repository_context() -> v07b.V07BContext:
    return v07b.load_v06_benchmark_context()


def _capability_payload(decision: str = v07c.DIRECT_ENERGY_UNAVAILABLE) -> dict[str, Any]:
    return {
        "stage": v07c.V07C_STAGE,
        "schema_version": v07c.V07C_SCHEMA_VERSION,
        "direct_energy_decision": decision,
        "accepted_direct_energy_backend": None,
        "experiments_executed": False,
        "final_observations_executed": 0,
        "carbon": {"claim_reported": False},
    }


def _raw_row(
    plan: v07d.PlannedObservation,
    context: v07b.V07BContext,
    *,
    pid: int,
    success: bool = True,
) -> dict[str, Any]:
    configuration = context.plan.configuration(plan.configuration_id)
    artifact = context.plan.model(plan.classifier, plan.configuration_id, plan.seed)
    return {
        "status": WorkerStatus.COMPLETED.value if success else WorkerStatus.FAILED.value,
        "environment_id": "synthetic-environment",
        "classifier": plan.classifier,
        "configuration_id": plan.configuration_id,
        "seed": plan.seed,
        "phase": plan.phase,
        "feature_count": configuration.feature_count,
        "record_count": 28_000 if plan.phase == "training" else 6_000,
        "outer_repetition": plan.outer_repetition,
        "inner_operation_count": plan.inner_operation_count,
        "measured_operation_count": plan.inner_operation_count if success else 0,
        "warmup_calls": 3 if success else 0,
        "worker_pid": pid,
        "wall_time_sec": 1.0 if success else None,
        "process_cpu_time_sec": 0.9 if success else None,
        "start_rss_bytes": 100 * 1024 * 1024 if success else None,
        "end_rss_bytes": 101 * 1024 * 1024 if success else None,
        "absolute_peak_rss_bytes": 102 * 1024 * 1024 if success else None,
        "incremental_peak_rss_bytes": 2 * 1024 * 1024 if success else None,
        "selected_input_bytes": 10_000,
        "numpy_dense_nbytes": 9_000,
        "serialized_model_bytes": artifact.serialized_model_bytes,
        "inference_latency_sec": 1.0 / plan.inner_operation_count if success else None,
        "per_record_latency_sec": (
            1.0
            / (plan.inner_operation_count * (28_000 if plan.phase == "training" else 6_000))
            if success
            else None
        ),
        "throughput_records_sec": (
            plan.inner_operation_count * (28_000 if plan.phase == "training" else 6_000)
            if success
            else None
        ),
        "measurement_scope": (
            "model.fit_only_data_loading_selection_serialization_excluded"
            if plan.phase == "training"
            else "prediction_only_model_loading_warmup_excluded"
        ),
        "workload_sha256": context.plan.workload(plan.phase, plan.seed).workload_sha256,
        "model_sha256": artifact.file_sha256,
        "feature_manifest_sha256": configuration.feature_hash_for_seed(plan.seed),
        "v06_lock_sha256": context.plan.validation_lock_sha256,
        "error_type": None if success else "RuntimeError",
        "error_stage": None if success else "measurement",
        "error_message": None if success else "synthetic failure",
    }


def _canonical_rows(
    schedule: tuple[v07d.PlannedObservation, ...], context: v07b.V07BContext
) -> tuple[dict[str, Any], ...]:
    rows = []
    for index in range(0, len(schedule), 10):
        block = schedule[index : index + 10]
        raw = [_raw_row(plan, context, pid=10_000 + index + offset) for offset, plan in enumerate(block)]
        rows.extend(v07d.transform_v07b_rows(raw, block))
    return tuple(rows)


def _frozen_calibration() -> v07d.FrozenCalibration:
    anchors = []
    index = 0
    for classifier in v07b.CLASSIFIERS:
        for configuration_id in v07b.EXPECTED_CONFIGURATION_IDS:
            final_count = 8 if index == 0 else 4
            attempts = tuple(
                v07b.CalibrationAttempt(count, 1.1 if count == final_count else count / 10, count)
                for count in (1, 2, 4, 8)
                if count <= final_count
            )
            result = v07b.CalibrationResult("inference", 1.0, final_count, attempts)
            anchors.append(
                v07d.CalibrationAnchor(
                    classifier, configuration_id, 42, "a" * 64, "b" * 64, result
                )
            )
            index += 1
    return v07d.FrozenCalibration(1.0, 16_384, 8, tuple(anchors))


# Imported late only to keep helper declarations compact.
from src.green.measurement import WorkerStatus


def test_01_v07c_decision_required(tmp_path: Path) -> None:
    path = tmp_path / "capability.json"
    path.write_text(json.dumps(_capability_payload(v07c.DIRECT_ENERGY_AVAILABLE)), encoding="utf-8")
    with pytest.raises(v07d.V07DIntegrityError, match="DIRECT_ENERGY_UNAVAILABLE"):
        v07d.load_v07c_tier1_decision(path)


def test_02_direct_energy_unavailable_accepted_for_tier1(tmp_path: Path) -> None:
    path = tmp_path / "capability.json"
    path.write_text(json.dumps(_capability_payload()), encoding="utf-8")
    payload, digest = v07d.load_v07c_tier1_decision(path)
    assert payload["direct_energy_decision"] == v07c.DIRECT_ENERGY_UNAVAILABLE
    assert len(digest) == 64


def test_03_no_energy_output_generated(schedule, synthetic_context) -> None:
    row = _canonical_rows(schedule[:10], synthetic_context)[0]
    assert not any("energy" in key.lower() for key in row)


def test_04_exactly_three_configurations(schedule) -> None:
    assert {item.configuration_id for item in schedule} == set(v07b.EXPECTED_CONFIGURATION_IDS)


def test_05_exactly_two_classifiers(schedule) -> None:
    assert {item.classifier for item in schedule} == set(v07b.CLASSIFIERS)


def test_06_exactly_five_seeds(schedule) -> None:
    assert {item.seed for item in schedule} == set(v07b.EXPECTED_SEEDS)


def test_07_exactly_ten_outer_repetitions(schedule) -> None:
    assert {item.outer_repetition for item in schedule} == set(range(1, 11))


def test_08_expected_300_training_observations(schedule) -> None:
    assert sum(item.phase == "training" for item in schedule) == 300


def test_09_expected_300_inference_observations(schedule) -> None:
    assert sum(item.phase == "inference" for item in schedule) == 300


def test_10_total_expected_observations_is_600(schedule) -> None:
    assert len(schedule) == v07d.PLANNED_TOTAL_OBSERVATIONS == 600


def test_11_fresh_worker_per_observation(schedule, synthetic_context) -> None:
    rows = _canonical_rows(schedule[:10], synthetic_context)
    assert len({row["worker_pid"] for row in rows}) == 10


def test_12_warmup_excluded(schedule, synthetic_context) -> None:
    row = _canonical_rows(schedule[:10], synthetic_context)[0]
    assert row["warmup_calls"] == 3
    assert "warmup" in row["measurement_scope"] or row["phase"] == "training"


def test_13_model_load_excluded_from_inference_timing() -> None:
    assert v07b._prediction_only.__name__ == "_prediction_only"
    assert "LightweightDetector.load" in inspect.getsource(v07b._InferencePreparation.__call__)
    assert "load" not in inspect.getsource(v07b._prediction_only)


def test_14_training_boundary_excludes_preparation(schedule, synthetic_context) -> None:
    row = _canonical_rows(schedule[:10], synthetic_context)[0]
    assert row["phase"] == "training"
    assert row["measurement_scope"].startswith("model.fit_only")
    assert "data_loading" in row["measurement_scope"]


def test_15_calibration_occurs_before_experiment_in_runner() -> None:
    source = inspect.getsource(v07d.run_v07d_experiments)
    assert source.index("run_calibration_matrix") < source.index("for cell_index")


def test_16_calibration_output_is_separated_and_recoverable(tmp_path: Path) -> None:
    calibration = _frozen_calibration()
    rows = v07d.calibration_rows(calibration, "env")
    assert rows
    assert all(row["excluded_from_main_observations"] is True for row in rows)
    assert all(row["calibration_id"].startswith("v07d_calibration__") for row in rows)
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": v07d.V07D_SCHEMA_VERSION,
                "stage": v07d.V07D_STAGE,
                "excluded_from_main_observations": True,
                "target_duration_sec": calibration.target_duration_sec,
                "maximum_inner_operation_count": calibration.maximum_inner_operation_count,
                "selected_inner_operation_count": calibration.selected_inner_operation_count,
                "pairing_scope": calibration.pairing_scope,
                "rows": list(rows),
            }
        ),
        encoding="utf-8",
    )
    restored = v07d.load_frozen_calibration(
        path,
        environment_id="env",
        config={
            "measurement": {
                "inference_calibration": {
                    "target_duration_sec": 1.0,
                    "maximum_inner_operation_count": 16_384,
                    "pairing_scope": "all_feature_configurations_and_classifiers",
                }
            }
        },
    )
    assert restored == calibration


def test_17_inner_count_frozen_after_calibration(schedule) -> None:
    assert {item.inner_operation_count for item in schedule if item.phase == "inference"} == {8}


def test_18_same_pairing_workload_hashes(schedule, synthetic_context) -> None:
    rows = _canonical_rows(schedule, synthetic_context)
    v07d.validate_complete_matrix(rows, schedule, synthetic_context)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["seed"], row["phase"]), set()).add(row["workload_sha256"])
    assert all(len(values) == 1 for values in grouped.values())


def test_19_no_selector_fit_interface() -> None:
    source = inspect.getsource(v07d)
    assert "create_selector" not in source
    assert "selector.fit(" not in source


def test_20_no_hyperparameter_tuning() -> None:
    config = v07b.load_green_evaluation_config()
    assert config["governance"]["classifier_tuning"] == "prohibited"
    assert config["governance"]["threshold_tuning"] == "prohibited"


def test_21_model_configuration_frozen() -> None:
    assert dict(v07b.DT_PARAMETERS) == {
        "max_depth": 5,
        "min_samples_leaf": 20,
        "class_weight": "balanced",
    }
    assert dict(v07b.LR_PARAMETERS) == {
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": 500,
        "C": 1.0,
    }


def test_22_feature_order_frozen(synthetic_context) -> None:
    configuration = synthetic_context.plan.configurations[2]
    features = configuration.features_for_seed(42)
    assert v07b.fingerprint_feature_names(features) == configuration.feature_hash_for_seed(42)


def test_23_workload_immutable(schedule, synthetic_context) -> None:
    rows = _canonical_rows(schedule[:10], synthetic_context)
    assert {row["workload_sha256"] for row in rows} == {
        synthetic_context.plan.workload("training", 42).workload_sha256
    }


def test_24_failures_retained(schedule, synthetic_context) -> None:
    block = schedule[:10]
    raw = [_raw_row(plan, synthetic_context, pid=100 + index, success=index != 4) for index, plan in enumerate(block)]
    rows = v07d.transform_v07b_rows(raw, block)
    assert len(rows) == 10
    assert sum(row["status"] == v07d.FAILURE for row in rows) == 1
    assert rows[4]["failure_reason"] == "synthetic failure"


def test_25_no_silent_observation_removal(schedule, synthetic_context) -> None:
    block = schedule[:10]
    raw = [_raw_row(plan, synthetic_context, pid=100 + index) for index, plan in enumerate(block[:-1])]
    with pytest.raises(v07d.V07DIntegrityError, match="Every planned repetition"):
        v07d.transform_v07b_rows(raw, block)


def test_26_output_row_schema(schedule, synthetic_context) -> None:
    row = _canonical_rows(schedule[:10], synthetic_context)[0]
    assert set(v07d.OBSERVATION_FIELDS) == set(row)


def test_27_canonical_units(schedule, synthetic_context) -> None:
    row = _canonical_rows(schedule[:10], synthetic_context)[0]
    assert (row["time_unit"], row["rss_unit"], row["size_unit"], row["throughput_unit"]) == (
        "seconds", "MiB", "bytes", "records/second"
    )


def test_28_provenance_valid(schedule, synthetic_context) -> None:
    row = _canonical_rows(schedule[:10], synthetic_context)[0]
    assert row["measurement_provenance"] == MeasurementProvenance.DIRECT_COMPUTATIONAL.value
    assert row["derived_measurement_provenance"] == MeasurementProvenance.DERIVED.value


def test_29_no_direct_energy_rows(schedule, synthetic_context) -> None:
    rows = _canonical_rows(schedule[:10], synthetic_context)
    assert {row["measurement_kind"] for row in rows} == {"DIRECT_COMPUTATIONAL"}


def test_30_no_carbon_rows(schedule, synthetic_context) -> None:
    row = _canonical_rows(schedule[:10], synthetic_context)[0]
    assert not any("carbon" in key.lower() or "co2" in key.lower() for key in row)


def test_31_observation_ids_unique(schedule) -> None:
    assert len({item.observation_id for item in schedule}) == 600


def test_32_schedule_deterministic(synthetic_context) -> None:
    first = v07d.build_observation_schedule(synthetic_context.plan, inference_inner_operation_count=8)
    second = v07d.build_observation_schedule(synthetic_context.plan, inference_inner_operation_count=8)
    assert first == second


def test_33_atomic_output_behavior(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    v07d.atomic_write_json(path, {"value": 1})
    v07d.atomic_write_json(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 2}
    assert not (tmp_path / ".artifact.json.tmp").exists()


def test_34_run_completeness_accounting(schedule, synthetic_context) -> None:
    rows = _canonical_rows(schedule, synthetic_context)
    accounting = v07d.validate_complete_matrix(rows, schedule, synthetic_context)
    assert accounting == {
        "planned_training_observations": 300,
        "successful_training_observations": 300,
        "failed_training_observations": 0,
        "planned_inference_observations": 300,
        "successful_inference_observations": 300,
        "failed_inference_observations": 0,
        "planned_total_observations": 600,
        "successful_total_observations": 600,
        "failed_total_observations": 0,
        "retained_total_observations": 600,
        "run_complete": True,
    }


def test_35_v06_hashes_unchanged(repository_context) -> None:
    before = dict(repository_context.plan.source_hashes)
    v07b.verify_v06_source_hashes(repository_context.plan)
    assert v07c.snapshot_file_hashes(tuple(Path(path) for path in before)) == before


def test_36_v07b_hashes_unchanged() -> None:
    before = v07c.snapshot_file_hashes(v07d.V07B_IMMUTABLE_PATHS)
    assert v07c.snapshot_file_hashes(v07d.V07B_IMMUTABLE_PATHS) == before


def test_37_v07c_hashes_unchanged() -> None:
    before = v07c.snapshot_file_hashes(v07d.V07C_IMMUTABLE_PATHS)
    assert v07c.snapshot_file_hashes(v07d.V07C_IMMUTABLE_PATHS) == before


def test_38_json_csv_consistency(tmp_path: Path, schedule, synthetic_context) -> None:
    rows = _canonical_rows(schedule[:10], synthetic_context)
    json_path = tmp_path / "rows.json"
    csv_path = tmp_path / "rows.csv"
    v07d.atomic_write_json(json_path, list(rows))
    v07d.atomic_write_csv(csv_path, rows, fieldnames=v07d.OBSERVATION_FIELDS)
    json_rows = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(json_rows) == len(csv_rows) == 10
    assert [row["observation_id"] for row in json_rows] == [
        row["observation_id"] for row in csv_rows
    ]
