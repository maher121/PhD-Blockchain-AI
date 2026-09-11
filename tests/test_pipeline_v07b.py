"""Focused synthetic and artifact-loading tests for V0.7-B infrastructure."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.green.measurement import (
    DEFAULT_OUTER_REPETITIONS,
    DEFAULT_WARMUP_CALLS,
    ComputationalObservation,
    FreshWorkerProtocol,
    FreshWorkerRun,
    MeasurementPhase,
    WorkerFailure,
    WorkerResult,
    WorkerStatus,
)
from src.lightweight.models import LightweightDetector
from src.lightweight.resource_monitor import ResourceMeasurement
import src.pipeline_v07b as v07b


def _features(count: int) -> list[str]:
    return [f"feature_{index:02d}" for index in range(count)]


def _minimal_lock() -> dict[str, Any]:
    roles = dict(zip(v07b.SEMANTIC_ROLES, v07b.EXPECTED_CONFIGURATION_IDS))
    configurations: dict[str, Any] = {}
    for role, configuration_id in roles.items():
        count = v07b.EXPECTED_FEATURE_COUNTS[role]
        seed_specific: dict[str, Any] = {}
        selected: dict[str, list[str]] = {}
        hashes: dict[str, str] = {}
        for seed in v07b.EXPECTED_SEEDS:
            names = _features(count)
            digest = v07b.fingerprint_feature_names(names)
            seed_specific[str(seed)] = {
                "selected_features": names,
                "selected_features_sha256": digest,
            }
            selected[str(seed)] = names
            hashes[str(seed)] = digest
        configurations[configuration_id] = {
            "actual_feature_count": count,
            "seed_specific": seed_specific,
            "selected_features": selected,
            "selected_feature_fingerprints": hashes,
        }
    candidates = _features(43)
    return {
        "seeds": list(v07b.EXPECTED_SEEDS),
        "attack_configuration": {
            "attack_type": "mixed",
            "attack_rate": 0.05,
            "attack_severity": "MEDIUM",
            "attack_config_path": "config/attack_scenarios.yaml",
        },
        "model_configuration": {
            "model_id": "decision_tree",
            "parameters": dict(v07b.DT_PARAMETERS),
        },
        "threshold": {"prediction_threshold": 0.5},
        "decision_threshold": 0.5,
        "candidate_manifest": {
            "features": candidates,
            "feature_count": 43,
            "sha256": v07b.fingerprint_feature_names(candidates),
        },
        "candidate_manifest_hash": v07b.fingerprint_feature_names(candidates),
        "roles": {
            role: {
                "configuration_id": configuration_id,
                "locked_configuration_ref": configuration_id,
            }
            for role, configuration_id in roles.items()
        },
        "locked_configurations": configurations,
    }


@pytest.fixture(scope="module")
def repository_context() -> v07b.V07BContext:
    """Loading and hashing only; this does not prepare attacks or run benchmarks."""

    return v07b.load_v06_benchmark_context()


def _resource() -> ResourceMeasurement:
    return ResourceMeasurement(2.0, 1.0, 50.0, None, 300, 100, 200)


def _completed_run(kwargs: dict[str, Any], wall_time: float = 2.0) -> FreshWorkerRun:
    protocol = FreshWorkerProtocol(
        inner_operation_count=kwargs["inner_operation_count"],
        phase=kwargs["phase"],
        warmup_calls=kwargs["warmup_calls"],
        outer_repetitions=kwargs["outer_repetitions"],
        records_per_operation=kwargs["records_per_operation"],
        timeout_seconds=kwargs["timeout_seconds"],
        measurement_scope=kwargs["measurement_scope"],
        start_method=kwargs.get("start_method"),
    )
    results = []
    for repetition in range(1, protocol.outer_repetitions + 1):
        resources = ResourceMeasurement(wall_time, 1.0, 50.0, None, 300, 100, 200)
        observation = ComputationalObservation(
            protocol.phase,
            repetition,
            protocol.inner_operation_count,
            protocol.inner_operation_count,
            protocol.warmup_calls,
            resources,
            records_per_operation=protocol.records_per_operation,
            measurement_scope=protocol.measurement_scope,
        )
        results.append(
            WorkerResult(
                repetition,
                WorkerStatus.COMPLETED,
                1000 + repetition,
                protocol.warmup_calls,
                protocol.inner_operation_count,
                0.1,
                ("preparation_start", "preparation_end", "measurement_start", "measurement_end"),
                observation=observation,
            )
        )
    return FreshWorkerRun(protocol, tuple(results))


@pytest.fixture()
def synthetic_cell(tmp_path: Path) -> SimpleNamespace:
    names = ("feature_a", "feature_b")
    frame = pd.DataFrame(
        {"feature_a": np.linspace(-1, 1, 20), "feature_b": np.tile([0.0, 1.0], 10)}
    )
    labels = pd.Series(([0] * 10) + ([1] * 10), name="is_attack", dtype="int8")
    model = LightweightDetector(
        "decision_tree", names, dict(v07b.DT_PARAMETERS), random_state=42
    ).fit(frame, labels)
    model_path = model.save(tmp_path / "model.joblib").resolve()
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    feature_hash = v07b.fingerprint_feature_names(names)
    artifact = v07b.ModelArtifactPlan(
        "decision_tree",
        "synthetic",
        42,
        model_path,
        model_hash,
        v07b.v06g.model_state_sha256(model),
        model_path.stat().st_size,
        feature_hash,
        names,
    )
    configuration = v07b.LockedConfigurationPlan(
        "full_baseline", "synthetic", 2, ((42, names),), ((42, feature_hash),)
    )
    workload_hash = "a" * 64
    workload_plan = v07b.WorkloadHashPlan(42, "inference", workload_hash, ())
    plan = v07b.BenchmarkPlan(
        validation_lock_path=tmp_path / "lock.json",
        validation_lock_sha256="b" * 64,
        semantic_lock_sha256="c" * 64,
        candidate_features=names,
        configurations=(configuration,),
        models=(artifact,),
        workloads=(workload_plan,),
        source_hashes=((str(model_path), model_hash),),
        attack_config_path=tmp_path / "attack.yaml",
        attack_config_sha256="d" * 64,
        processed_dataset_metadata_sha256="e" * 64,
        seeds=(42,),
        classifiers=("decision_tree",),
    )
    prepared = SimpleNamespace(features=frame, labels=labels)
    workloads = v07b.PreparedBenchmarkWorkloads(
        MappingProxyType({(42, "inference"): prepared}),
        MappingProxyType({(42, "inference"): workload_hash}),
    )
    context = v07b.V07BContext(MappingProxyType({}), plan)
    cell = v07b.BenchmarkCell("decision_tree", "synthetic", 42, "inference", 4)
    return SimpleNamespace(
        context=context,
        cell=cell,
        workloads=workloads,
        model=model,
        artifact=artifact,
        frame=frame,
        labels=labels,
    )


def test_01_authoritative_prd_path_available() -> None:
    config = v07b.load_green_evaluation_config()
    assert v07b.AUTHORITATIVE_PRD_PATH.is_file()
    assert config["authoritative_prd"] == "docs/PhD_PRD.md"
    assert "README" not in config["authoritative_prd"]


def test_02_validation_lock_is_required(tmp_path: Path) -> None:
    with pytest.raises(v07b.V07BIntegrityError, match="validation lock"):
        v07b.load_v06_benchmark_context(tmp_path)


def test_03_validation_lock_file_hash_and_sidecar_are_verified(repository_context) -> None:
    plan = repository_context.plan
    assert hashlib.sha256(plan.validation_lock_path.read_bytes()).hexdigest() == plan.validation_lock_sha256
    assert plan.validation_lock_sha256 == "71dc92b8a34db1bb3c2f55a69e56f43733544329128ec60c5cdbbce2445d1604"


def test_04_exactly_three_locked_configurations(repository_context) -> None:
    assert tuple(item.configuration_id for item in repository_context.plan.configurations) == v07b.EXPECTED_CONFIGURATION_IDS


def test_05_feature_counts_43_42_11_are_verified(repository_context) -> None:
    assert tuple(item.feature_count for item in repository_context.plan.configurations) == (43, 42, 11)


def test_06_feature_order_is_preserved() -> None:
    lock = _minimal_lock()
    plans = v07b.build_locked_configuration_plans(lock)
    assert plans[0].features_for_seed(42) == tuple(_features(43))
    lock["locked_configurations"][v07b.EXPECTED_CONFIGURATION_IDS[0]]["seed_specific"]["42"]["selected_features"].reverse()
    with pytest.raises(v07b.V07BIntegrityError, match="count/order/hash"):
        v07b.build_locked_configuration_plans(lock)


def test_07_selector_fitting_is_prohibited() -> None:
    config = v07b.load_green_evaluation_config()
    assert config["governance"]["selector_fitting"] == "prohibited"
    assert "selector" not in v07b.run_benchmark_cell.__annotations__


def test_08_seeds_are_exactly_42_to_46(repository_context) -> None:
    assert repository_context.plan.seeds == (42, 43, 44, 45, 46)
    lock = _minimal_lock()
    lock["seeds"] = [42]
    with pytest.raises(v07b.V07BIntegrityError, match="Seeds"):
        v07b._verify_frozen_lock_protocol(lock)


def test_09_decision_tree_is_frozen(synthetic_cell) -> None:
    v07b._verify_model_against_plan(synthetic_cell.model, synthetic_cell.artifact)
    synthetic_cell.model.estimator.max_depth = 6
    with pytest.raises(v07b.V07BIntegrityError, match="Unexpected frozen"):
        v07b._verify_model_against_plan(synthetic_cell.model, synthetic_cell.artifact)


def test_10_logistic_regression_is_frozen(tmp_path: Path) -> None:
    names = ("a", "b")
    frame = pd.DataFrame({"a": [-2, -1, 1, 2], "b": [0, 1, 0, 1]})
    labels = pd.Series([0, 0, 1, 1])
    model = LightweightDetector(
        "logistic_regression", names, dict(v07b.LR_PARAMETERS), random_state=42
    ).fit(frame, labels)
    path = model.save(tmp_path / "lr.joblib").resolve()
    plan = v07b.ModelArtifactPlan(
        "logistic_regression", "x", 42, path, hashlib.sha256(path.read_bytes()).hexdigest(),
        v07b.v06g.model_state_sha256(model), path.stat().st_size,
        v07b.fingerprint_feature_names(names), names,
    )
    v07b._verify_model_against_plan(model, plan)
    model.estimator.C = 2.0
    with pytest.raises(v07b.V07BIntegrityError, match="Unexpected frozen"):
        v07b._verify_model_against_plan(model, plan)


def test_11_threshold_is_frozen() -> None:
    lock = _minimal_lock()
    lock["decision_threshold"] = 0.4
    with pytest.raises(v07b.V07BIntegrityError, match="threshold"):
        v07b._verify_frozen_lock_protocol(lock)


def test_12_workload_is_mixed_only() -> None:
    lock = _minimal_lock()
    lock["attack_configuration"]["attack_type"] = "quantity_manipulation"
    with pytest.raises(v07b.V07BIntegrityError, match="mixed/0.05/MEDIUM"):
        v07b._verify_frozen_lock_protocol(lock)


def test_13_attack_rate_is_exactly_point_zero_five() -> None:
    lock = _minimal_lock()
    lock["attack_configuration"]["attack_rate"] = 0.1
    with pytest.raises(v07b.V07BIntegrityError, match="mixed/0.05/MEDIUM"):
        v07b._verify_frozen_lock_protocol(lock)


def test_14_attack_severity_is_medium() -> None:
    lock = _minimal_lock()
    lock["attack_configuration"]["attack_severity"] = "HIGH"
    with pytest.raises(v07b.V07BIntegrityError, match="mixed/0.05/MEDIUM"):
        v07b._verify_frozen_lock_protocol(lock)


def test_15_same_workload_object_is_reused() -> None:
    workload = object()
    cells = [v07b.BenchmarkCell(classifier, config, 42, "inference", 4) for classifier in v07b.CLASSIFIERS for config in v07b.EXPECTED_CONFIGURATION_IDS]
    v07b.verify_same_workload_reused([(cell, workload) for cell in cells])
    with pytest.raises(v07b.V07BIntegrityError, match="reuse"):
        v07b.verify_same_workload_reused([(cells[0], object()), (cells[1], object())])


def test_16_missing_model_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(v07b.V07BIntegrityError, match="Missing frozen model"):
        v07b._require_file_hash(tmp_path / "missing.joblib", "a" * 64, "frozen model")


def test_17_model_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"
    path.write_bytes(b"model")
    with pytest.raises(v07b.V07BIntegrityError, match="hash mismatch"):
        v07b._require_file_hash(path, "a" * 64, "frozen model")


def test_18_feature_hash_mismatch_is_rejected() -> None:
    lock = _minimal_lock()
    config = v07b.EXPECTED_CONFIGURATION_IDS[2]
    lock["locked_configurations"][config]["seed_specific"]["42"]["selected_features_sha256"] = "0" * 64
    with pytest.raises(v07b.V07BIntegrityError, match="count/order/hash"):
        v07b.build_locked_configuration_plans(lock)


def test_19_warmup_count_default_is_three() -> None:
    config = v07b.load_green_evaluation_config()
    assert DEFAULT_WARMUP_CALLS == config["measurement"]["warmup_calls"] == 3


def test_20_outer_repetition_default_is_ten() -> None:
    config = v07b.load_green_evaluation_config()
    assert DEFAULT_OUTER_REPETITIONS == config["measurement"]["outer_repetitions"] == 10


def test_21_cell_runner_delegates_to_fresh_worker_api(synthetic_cell) -> None:
    captured: dict[str, Any] = {}

    def runner(operation: Any, **kwargs: Any) -> FreshWorkerRun:
        captured.update(kwargs)
        captured["operation"] = operation
        return _completed_run(kwargs)

    rows = v07b.run_benchmark_cell(
        synthetic_cell.context,
        synthetic_cell.workloads,
        synthetic_cell.cell,
        environment_id="synthetic-env",
        outer_repetitions=2,
        runner=runner,
    )
    assert len(rows) == 2
    assert captured["outer_repetitions"] == 2
    assert captured["measurement_scope"] == "prediction_only_model_loading_warmup_excluded"


def _calibration_runner(operation: Any, **kwargs: Any) -> FreshWorkerRun:
    del operation
    return _completed_run(kwargs, wall_time=kwargs["inner_operation_count"] * 0.1)


def test_22_calibration_uses_geometric_growth() -> None:
    result = v07b.calibrate_inference_inner_count(
        lambda _: None, target_duration_sec=0.35, runner=_calibration_runner
    )
    assert [item.inner_operation_count for item in result.attempts] == [1, 2, 4]


def test_23_calibration_reaches_duration_target() -> None:
    result = v07b.calibrate_inference_inner_count(
        lambda _: None, target_duration_sec=0.35, runner=_calibration_runner
    )
    assert result.target_reached
    assert result.attempts[-1].wall_time_sec >= result.target_duration_sec


def test_24_calibration_obeys_safety_bound() -> None:
    with pytest.raises(v07b.V07BCalibrationError, match="maximum_inner_operation_count"):
        v07b.calibrate_inference_inner_count(
            lambda _: None,
            target_duration_sec=1.0,
            maximum_inner_operation_count=4,
            runner=_calibration_runner,
        )


def test_25_same_calibrated_inner_count_is_reused(repository_context) -> None:
    calibration = v07b.CalibrationResult(
        "inference", 1.0, 8, (v07b.CalibrationAttempt(8, 1.1, 1),)
    )
    cells = v07b.build_benchmark_schedule(repository_context.plan, calibration)
    inference = [cell for cell in cells if cell.phase == "inference"]
    assert len(cells) == 60
    assert {cell.inner_operation_count for cell in inference} == {8}


def test_26_training_boundary_is_model_fit_only() -> None:
    calls: list[tuple[Any, Any]] = []
    model = SimpleNamespace(fit=lambda features, labels: calls.append((features, labels)))
    state = v07b._TrainingState(model, "features", "labels")
    v07b._training_fit_only(state)
    assert calls == [("features", "labels")]


def test_27_inference_boundary_is_prediction_only() -> None:
    calls: list[Any] = []
    model = SimpleNamespace(predict_frame=lambda features: calls.append(features))
    v07b._prediction_only(v07b._InferenceState(model, "features"))
    assert calls == ["features"]


def test_28_model_loading_is_in_prepare_not_timed_operation(synthetic_cell) -> None:
    captured: dict[str, Any] = {}

    def runner(operation: Any, **kwargs: Any) -> FreshWorkerRun:
        captured["operation"] = operation
        captured["prepare"] = kwargs["prepare"]
        return _completed_run(kwargs)

    v07b.run_benchmark_cell(
        synthetic_cell.context, synthetic_cell.workloads, synthetic_cell.cell,
        environment_id="env", outer_repetitions=1, runner=runner,
    )
    assert captured["operation"] is v07b._prediction_only
    assert isinstance(captured["prepare"], v07b._InferencePreparation)


def test_29_timing_metrics_are_valid(synthetic_cell) -> None:
    rows = v07b.run_benchmark_cell(
        synthetic_cell.context, synthetic_cell.workloads, synthetic_cell.cell,
        environment_id="env", outer_repetitions=1,
        runner=lambda operation, **kwargs: _completed_run(kwargs),
    )
    assert rows[0]["wall_time_sec"] == 2.0
    assert rows[0]["process_cpu_time_sec"] == 1.0


def test_30_rss_metrics_are_valid(synthetic_cell) -> None:
    row = v07b.run_benchmark_cell(
        synthetic_cell.context, synthetic_cell.workloads, synthetic_cell.cell,
        environment_id="env", outer_repetitions=1,
        runner=lambda operation, **kwargs: _completed_run(kwargs),
    )[0]
    assert (row["start_rss_bytes"], row["end_rss_bytes"], row["absolute_peak_rss_bytes"]) == (100, 200, 300)
    assert row["incremental_peak_rss_bytes"] == 200


def test_31_selected_input_memory_metric_is_actual(synthetic_cell) -> None:
    row = v07b.run_benchmark_cell(
        synthetic_cell.context, synthetic_cell.workloads, synthetic_cell.cell,
        environment_id="env", outer_repetitions=1,
        runner=lambda operation, **kwargs: _completed_run(kwargs),
    )[0]
    expected = int(synthetic_cell.frame.memory_usage(index=True, deep=True).sum())
    assert row["selected_input_bytes"] == expected
    assert row["numpy_dense_nbytes"] == synthetic_cell.frame.to_numpy(dtype=float).nbytes


def test_32_model_size_metric_uses_serialized_file(synthetic_cell) -> None:
    row = v07b.run_benchmark_cell(
        synthetic_cell.context, synthetic_cell.workloads, synthetic_cell.cell,
        environment_id="env", outer_repetitions=1,
        runner=lambda operation, **kwargs: _completed_run(kwargs),
    )[0]
    assert row["serialized_model_bytes"] == synthetic_cell.artifact.path.stat().st_size


def test_33_throughput_derivation_is_valid(synthetic_cell) -> None:
    row = v07b.run_benchmark_cell(
        synthetic_cell.context, synthetic_cell.workloads, synthetic_cell.cell,
        environment_id="env", outer_repetitions=1,
        runner=lambda operation, **kwargs: _completed_run(kwargs),
    )[0]
    assert row["throughput_records_sec"] == pytest.approx(40.0)


def test_34_latency_derivation_is_valid(synthetic_cell) -> None:
    row = v07b.run_benchmark_cell(
        synthetic_cell.context, synthetic_cell.workloads, synthetic_cell.cell,
        environment_id="env", outer_repetitions=1,
        runner=lambda operation, **kwargs: _completed_run(kwargs),
    )[0]
    assert row["inference_latency_sec"] == pytest.approx(0.5)
    assert row["per_record_latency_sec"] == pytest.approx(0.025)


def test_35_direct_energy_is_never_fabricated(synthetic_cell) -> None:
    row = v07b.run_benchmark_cell(
        synthetic_cell.context, synthetic_cell.workloads, synthetic_cell.cell,
        environment_id="env", outer_repetitions=1,
        runner=lambda operation, **kwargs: _completed_run(kwargs),
    )[0]
    assert row["measurement_kind"] == "DIRECT_COMPUTATIONAL"
    assert row["direct_energy_joules"] is None
    assert row["energy_directly_measured"] is False


def test_36_v06_source_hashes_are_unchanged(repository_context) -> None:
    before = dict(repository_context.plan.source_hashes)
    v07b.verify_v06_source_hashes(repository_context.plan)
    after = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in before}
    assert after == before


def test_37_output_schema_is_valid(synthetic_cell) -> None:
    rows = v07b.run_benchmark_cell(
        synthetic_cell.context, synthetic_cell.workloads, synthetic_cell.cell,
        environment_id="env", outer_repetitions=1,
        runner=lambda operation, **kwargs: _completed_run(kwargs),
    )
    v07b.verify_output_schema(rows)
    assert v07b.OUTPUT_REQUIRED_FIELDS.issubset(rows[0])


def test_38_worker_failure_is_structured_and_retained(synthetic_cell) -> None:
    def failing_runner(operation: Any, **kwargs: Any) -> FreshWorkerRun:
        del operation
        protocol = FreshWorkerProtocol(
            inner_operation_count=kwargs["inner_operation_count"],
            phase=kwargs["phase"],
            warmup_calls=kwargs["warmup_calls"],
            outer_repetitions=1,
            records_per_operation=kwargs["records_per_operation"],
            measurement_scope=kwargs["measurement_scope"],
        )
        failure = WorkerResult(
            1, WorkerStatus.FAILED, 10, 0, 0, 0.1, (),
            failure=WorkerFailure("RuntimeError", "synthetic failure", "measurement"),
        )
        return FreshWorkerRun(protocol, (failure,))

    row = v07b.run_benchmark_cell(
        synthetic_cell.context, synthetic_cell.workloads, synthetic_cell.cell,
        environment_id="env", outer_repetitions=1, runner=failing_runner,
    )[0]
    assert row["status"] == "FAILED"
    assert row["error_type"] == "RuntimeError"
    assert row["error_stage"] == "measurement"
