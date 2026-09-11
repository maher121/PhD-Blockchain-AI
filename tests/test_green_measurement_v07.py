"""Synthetic, hardware-independent validation of the V0.7-A infrastructure."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.green.energy import (
    ExternalMeterConfig,
    aggregate_selected_rapl_delta_joules,
    discover_rapl_domains,
    estimate_tdp_energy_record,
    instantaneous_watts_to_joules,
    probe_external_meter,
    probe_historical_estimate,
    probe_linux_rapl,
    probe_nvml,
    probe_windows_emi,
    rapl_energy_delta_joules,
)
from src.green.measurement import (
    DEFAULT_OUTER_REPETITIONS,
    DEFAULT_WARMUP_CALLS,
    CanonicalUnit,
    ComputationalObservation,
    FreshWorkerProtocol,
    MeasurementMetric,
    MeasurementPhase,
    MeasurementProvenance,
    MeasurementRecord,
    WorkerStatus,
    collect_environment_metadata,
    collect_git_metadata,
    dataframe_deep_memory_bytes,
    is_wsl_environment,
    latency_and_throughput,
    numpy_dense_nbytes,
    run_fresh_worker_protocol,
    serialized_model_size_bytes,
)
from src.lightweight.resource_monitor import ResourceMeasurement


_TEST_START_METHOD = "fork" if "fork" in mp.get_all_start_methods() else None


def _synthetic_operation(_: Any) -> None:
    time.sleep(0.001)


def _failing_operation(_: Any) -> None:
    raise RuntimeError("synthetic worker failure")


def _slow_preparation() -> str:
    time.sleep(0.05)
    return "loaded-model"


def _loaded_model_operation(state: Any) -> None:
    if state != "loaded-model":
        raise RuntimeError("model was not prepared")
    time.sleep(0.001)


@pytest.fixture(scope="module")
def fresh_run():
    return run_fresh_worker_protocol(
        _synthetic_operation,
        inner_operation_count=4,
        warmup_calls=2,
        outer_repetitions=3,
        records_per_operation=5,
        timeout_seconds=60,
        start_method=_TEST_START_METHOD,
    )


def _resources(**changes: Any) -> ResourceMeasurement:
    values = {
        "wall_time_sec": 2.0,
        "cpu_time_sec": 1.0,
        "average_cpu_percent": 50.0,
        "peak_cpu_percent": None,
        "peak_rss_bytes": 300,
        "start_rss_bytes": 100,
        "end_rss_bytes": 200,
    }
    values.update(changes)
    return ResourceMeasurement(**values)


def _make_rapl_domain(
    root: Path,
    relative: str,
    *,
    name: str,
    energy_uj: int,
    max_energy_range_uj: int = 1_000,
) -> None:
    domain = root / relative
    domain.mkdir(parents=True)
    (domain / "name").write_text(name, encoding="utf-8")
    (domain / "energy_uj").write_text(str(energy_uj), encoding="ascii")
    (domain / "max_energy_range_uj").write_text(
        str(max_energy_range_uj), encoding="ascii"
    )


def test_01_schema_validation_and_serialization() -> None:
    record = MeasurementRecord(
        MeasurementMetric.WALL_CLOCK_TIME,
        1.25,
        CanonicalUnit.SECONDS,
        MeasurementProvenance.DIRECT_COMPUTATIONAL,
        MeasurementPhase.TRAINING,
        "model_fit_only",
    )
    assert record.to_dict() == {
        "metric": "wall_clock_time",
        "value": 1.25,
        "unit": "seconds",
        "provenance": "DIRECT_COMPUTATIONAL",
        "phase": "training",
        "scope": "model_fit_only",
        "metadata": {},
    }


def test_02_all_provenance_categories_exist() -> None:
    assert {item.value for item in MeasurementProvenance} == {
        "DIRECT_COMPUTATIONAL",
        "DIRECT_ENERGY",
        "DERIVED",
        "IMPORTED_HISTORICAL",
        "ESTIMATED",
    }


def test_03_unit_validation_rejects_noncanonical_pair() -> None:
    with pytest.raises(ValueError, match="wall_clock_time"):
        MeasurementRecord(
            MeasurementMetric.WALL_CLOCK_TIME,
            1,
            CanonicalUnit.BYTES,
            MeasurementProvenance.DIRECT_COMPUTATIONAL,
            MeasurementPhase.EVALUATION,
            "test",
        )


def test_04_estimated_cannot_masquerade_as_direct_energy() -> None:
    with pytest.raises(ValueError, match="Estimated energy"):
        MeasurementRecord(
            MeasurementMetric.ENERGY,
            2,
            CanonicalUnit.JOULES,
            MeasurementProvenance.DIRECT_ENERGY,
            MeasurementPhase.TRAINING,
            "tdp-model",
            metadata={"estimated": True, "method": "TDP_x_wall_time"},
        )
    estimate = estimate_tdp_energy_record(
        10, 2, phase=MeasurementPhase.TRAINING, scope="synthetic"
    )
    assert estimate.value == 20
    assert estimate.provenance is MeasurementProvenance.ESTIMATED


def test_05_cpu_time_cannot_be_called_energy() -> None:
    with pytest.raises(ValueError, match="computational measurement"):
        MeasurementRecord(
            MeasurementMetric.PROCESS_CPU_TIME,
            1,
            CanonicalUnit.SECONDS,
            MeasurementProvenance.DIRECT_ENERGY,
            MeasurementPhase.TRAINING,
            "fit",
        )


def test_06_memory_cannot_be_called_energy() -> None:
    with pytest.raises(ValueError, match="computational measurement"):
        MeasurementRecord(
            MeasurementMetric.ABSOLUTE_PEAK_RSS,
            1,
            CanonicalUnit.BYTES,
            MeasurementProvenance.DIRECT_ENERGY,
            MeasurementPhase.TRAINING,
            "fit",
        )


def test_07_positive_wall_and_nonnegative_cpu_time(fresh_run) -> None:
    for result in fresh_run.results:
        assert result.observation is not None
        assert result.observation.resources.wall_time_sec > 0
        assert result.observation.resources.cpu_time_sec >= 0


def test_08_rss_invariants_and_incremental_peak() -> None:
    observation = ComputationalObservation(
        MeasurementPhase.EVALUATION, 1, 1, 1, 0, _resources()
    )
    assert observation.absolute_peak_rss_bytes == 300
    assert observation.incremental_peak_rss_bytes == 200
    with pytest.raises(ValueError, match="Absolute peak RSS"):
        ComputationalObservation(
            MeasurementPhase.EVALUATION,
            1,
            1,
            1,
            0,
            _resources(peak_rss_bytes=150, end_rss_bytes=200),
        )


def test_09_dataframe_deep_memory_matches_pandas() -> None:
    frame = pd.DataFrame({"label": ["alpha", "beta"], "value": [1, 2]})
    assert dataframe_deep_memory_bytes(frame) == int(
        frame.memory_usage(index=True, deep=True).sum()
    )


def test_10_numpy_dense_nbytes() -> None:
    array = np.zeros((3, 4), dtype=np.float64)
    assert numpy_dense_nbytes(array) == 3 * 4 * 8
    with pytest.raises(TypeError, match="dense NumPy"):
        numpy_dense_nbytes([[0.0]])  # type: ignore[arg-type]


def test_11_serialized_model_size_uses_actual_file(tmp_path: Path) -> None:
    artifact = tmp_path / "synthetic.joblib"
    artifact.write_bytes(b"model-bytes")
    assert serialized_model_size_bytes(artifact) == len(b"model-bytes")


def test_12_per_record_latency() -> None:
    metrics = latency_and_throughput(
        2.0, measured_operation_count=4, records_per_operation=10
    )
    assert metrics["per_record_latency_seconds"] == pytest.approx(0.05)
    assert metrics["inference_latency_seconds"] == pytest.approx(0.5)


def test_13_throughput() -> None:
    metrics = latency_and_throughput(
        2.0, measured_operation_count=4, records_per_operation=10
    )
    assert metrics["throughput_records_per_second"] == pytest.approx(20.0)


def test_14_warmup_occurs_before_timing(fresh_run) -> None:
    events = fresh_run.results[0].lifecycle_events
    assert events.index("warmup:2") < events.index("measurement_start")
    assert events.index("measurement_start") < events.index("measured:1")


def test_15_warmup_count_and_defaults(fresh_run) -> None:
    assert DEFAULT_WARMUP_CALLS == 3
    assert FreshWorkerProtocol(inner_operation_count=1).warmup_calls == 3
    assert {result.warmup_calls_completed for result in fresh_run.results} == {2}


def test_16_outer_repetition_count_and_defaults(fresh_run) -> None:
    assert DEFAULT_OUTER_REPETITIONS == 10
    assert FreshWorkerProtocol(inner_operation_count=1).outer_repetitions == 10
    assert len(fresh_run.results) == 3
    assert [result.outer_repetition for result in fresh_run.results] == [1, 2, 3]


def test_17_explicit_inner_operation_count(fresh_run) -> None:
    assert fresh_run.protocol.inner_operation_count == 4
    assert {result.measured_operation_count for result in fresh_run.results} == {4}
    assert {result.observation.measured_record_count for result in fresh_run.results} == {20}
    with pytest.raises(TypeError):
        FreshWorkerProtocol()  # type: ignore[call-arg]


def test_18_each_outer_observation_has_fresh_worker(fresh_run) -> None:
    pids = [result.worker_pid for result in fresh_run.results]
    assert None not in pids
    assert len(set(pids)) == len(pids)


def test_19_worker_failures_are_structured_and_retained() -> None:
    run = run_fresh_worker_protocol(
        _failing_operation,
        inner_operation_count=1,
        warmup_calls=0,
        outer_repetitions=2,
        timeout_seconds=60,
        start_method=_TEST_START_METHOD,
    )
    assert len(run.results) == 2
    assert run.failed_count == 2
    assert all(result.status is WorkerStatus.FAILED for result in run.results)
    assert all(result.failure.error_type == "RuntimeError" for result in run.results)
    assert all(result.failure.stage == "measurement" for result in run.results)


def test_20_model_loading_is_outside_inference_boundary() -> None:
    run = run_fresh_worker_protocol(
        _loaded_model_operation,
        prepare=_slow_preparation,
        inner_operation_count=1,
        warmup_calls=0,
        outer_repetitions=1,
        timeout_seconds=60,
        start_method=_TEST_START_METHOD,
    )
    result = run.results[0]
    assert result.status is WorkerStatus.COMPLETED
    assert result.preparation_wall_time_seconds >= 0.045
    assert result.observation.resources.wall_time_sec < result.preparation_wall_time_seconds
    assert result.lifecycle_events.index("preparation_end") < result.lifecycle_events.index(
        "measurement_start"
    )


def test_21_rapl_unavailable_returns_structured_capability(tmp_path: Path) -> None:
    capability = probe_linux_rapl(tmp_path / "absent")
    assert capability.available is False
    assert capability.reason_code == "POWERCAP_PATH_UNAVAILABLE"
    assert capability.metadata["automatic_aggregation"] is False


def test_22_synthetic_rapl_delta_and_units() -> None:
    assert rapl_energy_delta_joules(1_000_000, 3_500_000) == pytest.approx(2.5)


def test_23_rapl_counter_wraparound() -> None:
    assert rapl_energy_delta_joules(900, 100, 1_000) == pytest.approx(0.0002)
    with pytest.raises(ValueError, match="wraparound"):
        rapl_energy_delta_joules(900, 100)


def test_24_rapl_domains_are_detected_but_not_automatically_summed(
    tmp_path: Path,
) -> None:
    _make_rapl_domain(tmp_path, "intel-rapl:0", name="package-0", energy_uj=900)
    _make_rapl_domain(
        tmp_path,
        "intel-rapl:0/intel-rapl:0:0",
        name="core",
        energy_uj=200,
    )
    capability = probe_linux_rapl(tmp_path)
    assert capability.available is True
    assert len(capability.metadata["domains"]) == 2
    assert capability.metadata["automatic_aggregation"] is False
    assert "total_joules" not in capability.metadata


def test_25_overlapping_rapl_domains_cannot_be_summed(tmp_path: Path) -> None:
    _make_rapl_domain(tmp_path, "intel-rapl:0", name="package-0", energy_uj=900)
    _make_rapl_domain(
        tmp_path,
        "intel-rapl:0/intel-rapl:0:0",
        name="core",
        energy_uj=200,
    )
    domains = discover_rapl_domains(tmp_path)
    ids = [domain.domain_id for domain in domains]
    with pytest.raises(ValueError, match="Overlapping RAPL"):
        aggregate_selected_rapl_delta_joules(
            domains,
            ids,
            {ids[0]: 100, ids[1]: 100},
            {ids[0]: 200, ids[1]: 200},
        )


def test_26_emi_unavailable_behavior() -> None:
    capability = probe_windows_emi(system_name="Linux", wsl=False)
    assert capability.available is False
    assert capability.reason_code == "UNSUPPORTED_OS"


def test_27_wsl_is_not_native_emi() -> None:
    capability = probe_windows_emi(system_name="Linux", wsl=True)
    assert capability.available is False
    assert capability.reason_code == "WSL_NOT_NATIVE_WINDOWS"
    assert is_wsl_environment(release="microsoft-standard-WSL2", version="", environ={})


def test_28_nvml_missing_optional_dependency_is_safe() -> None:
    def missing(_: str) -> Any:
        raise ModuleNotFoundError("synthetic missing pynvml")

    capability = probe_nvml(module_loader=missing)
    assert capability.available is False
    assert capability.reason_code == "OPTIONAL_DEPENDENCY_MISSING"
    assert capability.supports_cumulative_energy is False


def test_29_nvml_watts_are_not_cumulative_joules() -> None:
    class PowerOnlyNvml:
        def nvmlInit(self) -> None:
            return None

        def nvmlShutdown(self) -> None:
            return None

        def nvmlDeviceGetCount(self) -> int:
            return 1

        def nvmlDeviceGetHandleByIndex(self, _: int) -> int:
            return 0

        def nvmlDeviceGetName(self, _: int) -> str:
            return "synthetic GPU"

        def nvmlDeviceGetPowerUsage(self, _: int) -> int:
            return 5_000

    capability = probe_nvml(module_loader=lambda _: PowerOnlyNvml())
    assert capability.available is True
    assert capability.supports_instantaneous_power is True
    assert capability.supports_cumulative_energy is False
    assert capability.workload_relevant is False
    assert capability.metadata["instantaneous_power_is_energy"] is False
    with pytest.raises(ValueError, match="not cumulative joules"):
        instantaneous_watts_to_joules(5.0)


def test_30_external_meter_is_an_unavailable_placeholder() -> None:
    capability = probe_external_meter(
        ExternalMeterConfig("future wattmeter", connection="serial-placeholder")
    )
    assert capability.available is False
    assert capability.reason_code == "INTERFACE_ONLY"
    with pytest.raises(ValueError, match="cannot mark"):
        ExternalMeterConfig("fabricated", verified_attached=True)


def test_31_historical_tdp_capability_is_estimate_only() -> None:
    capability = probe_historical_estimate()
    assert capability.available is True
    assert capability.supports_cumulative_energy is False
    assert capability.metadata["provenance"] == "ESTIMATED"
    assert capability.metadata["direct_energy"] is False


def test_32_environment_metadata_schema() -> None:
    metadata = collect_environment_metadata()
    assert {
        "os",
        "kernel",
        "python",
        "cpu",
        "ram",
        "virtualization",
        "execution_environment",
        "gpu",
        "power_source",
        "thread_environment",
        "dependencies",
        "git",
    } <= set(metadata)
    assert metadata["ram"]["total_bytes"] > 0
    assert metadata["cpu"]["logical_count"] >= 1


def test_33_git_metadata_is_structured_inside_and_outside_repository(
    tmp_path: Path,
) -> None:
    repository = collect_git_metadata(Path(__file__).parents[1])
    outside = collect_git_metadata(tmp_path)
    assert repository["available"] is True
    assert len(repository["commit"]) == 40
    assert isinstance(repository["dirty"], bool)
    assert outside["available"] is False
    assert outside["commit"] is None


def test_34_future_optimizer_search_phase_is_supported() -> None:
    record = MeasurementRecord(
        MeasurementMetric.WALL_CLOCK_TIME,
        1,
        CanonicalUnit.SECONDS,
        MeasurementProvenance.DIRECT_COMPUTATIONAL,
        MeasurementPhase.OPTIMIZER_SEARCH,
        "complete_optimizer_search",
    )
    assert record.phase.value == "optimizer_search"


def test_35_observation_exports_raw_and_derived_provenance() -> None:
    observation = ComputationalObservation(
        MeasurementPhase.INFERENCE,
        1,
        2,
        2,
        3,
        _resources(),
        records_per_operation=10,
        dataframe_deep_memory_bytes=500,
        numpy_dense_nbytes=400,
        serialized_model_size_bytes=300,
    )
    records = {record.metric: record for record in observation.measurement_records()}
    assert records[MeasurementMetric.WALL_CLOCK_TIME].provenance is MeasurementProvenance.DIRECT_COMPUTATIONAL
    assert records[MeasurementMetric.PER_RECORD_LATENCY].provenance is MeasurementProvenance.DERIVED
    assert records[MeasurementMetric.THROUGHPUT].unit is CanonicalUnit.RECORDS_PER_SECOND
    assert records[MeasurementMetric.SERIALIZED_MODEL_SIZE].value == 300
