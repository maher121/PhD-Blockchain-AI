"""Validated computational measurements and fresh-worker orchestration for V0.7.

The resource boundary deliberately reuses :mod:`src.lightweight.resource_monitor`.
This module adds provenance, canonical units, repeated fresh-process execution,
and reproducibility metadata; it does not infer electrical energy from CPU time
or memory use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import importlib.metadata
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
from queue import Empty
import shutil
import subprocess
import time
import traceback
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil

from src.lightweight.resource_monitor import (
    ResourceMeasurement,
    ResourceMonitor,
    measure_model_size,
)


DEFAULT_WARMUP_CALLS = 3
DEFAULT_OUTER_REPETITIONS = 10


class MeasurementProvenance(str, Enum):
    """Scientific origin of a reported quantity."""

    DIRECT_COMPUTATIONAL = "DIRECT_COMPUTATIONAL"
    DIRECT_ENERGY = "DIRECT_ENERGY"
    DERIVED = "DERIVED"
    IMPORTED_HISTORICAL = "IMPORTED_HISTORICAL"
    ESTIMATED = "ESTIMATED"


class MeasurementPhase(str, Enum):
    """Reusable phases, including the future optimizer boundary."""

    TRAINING = "training"
    INFERENCE = "inference"
    SELECTOR_FIT = "selector_fit"
    OPTIMIZER_SEARCH = "optimizer_search"
    EVALUATION = "evaluation"


class MeasurementMetric(str, Enum):
    WALL_CLOCK_TIME = "wall_clock_time"
    PROCESS_CPU_TIME = "process_cpu_time"
    START_RSS = "start_rss"
    END_RSS = "end_rss"
    ABSOLUTE_PEAK_RSS = "absolute_peak_rss"
    INCREMENTAL_PEAK_RSS = "incremental_peak_rss"
    DATAFRAME_DEEP_MEMORY = "dataframe_deep_memory"
    NUMPY_DENSE_NBYTES = "numpy_dense_nbytes"
    SERIALIZED_MODEL_SIZE = "serialized_model_size"
    INFERENCE_LATENCY = "inference_latency"
    PER_RECORD_LATENCY = "per_record_latency"
    THROUGHPUT = "throughput"
    ENERGY = "energy"
    POWER = "power"


class CanonicalUnit(str, Enum):
    SECONDS = "seconds"
    BYTES = "bytes"
    MEBIBYTES = "MiB"
    JOULES = "joules"
    WATTS = "watts"
    RECORDS_PER_SECOND = "records/second"


_METRIC_UNITS: dict[MeasurementMetric, frozenset[CanonicalUnit]] = {
    MeasurementMetric.WALL_CLOCK_TIME: frozenset({CanonicalUnit.SECONDS}),
    MeasurementMetric.PROCESS_CPU_TIME: frozenset({CanonicalUnit.SECONDS}),
    MeasurementMetric.START_RSS: frozenset({CanonicalUnit.BYTES, CanonicalUnit.MEBIBYTES}),
    MeasurementMetric.END_RSS: frozenset({CanonicalUnit.BYTES, CanonicalUnit.MEBIBYTES}),
    MeasurementMetric.ABSOLUTE_PEAK_RSS: frozenset(
        {CanonicalUnit.BYTES, CanonicalUnit.MEBIBYTES}
    ),
    MeasurementMetric.INCREMENTAL_PEAK_RSS: frozenset(
        {CanonicalUnit.BYTES, CanonicalUnit.MEBIBYTES}
    ),
    MeasurementMetric.DATAFRAME_DEEP_MEMORY: frozenset({CanonicalUnit.BYTES}),
    MeasurementMetric.NUMPY_DENSE_NBYTES: frozenset({CanonicalUnit.BYTES}),
    MeasurementMetric.SERIALIZED_MODEL_SIZE: frozenset({CanonicalUnit.BYTES}),
    MeasurementMetric.INFERENCE_LATENCY: frozenset({CanonicalUnit.SECONDS}),
    MeasurementMetric.PER_RECORD_LATENCY: frozenset({CanonicalUnit.SECONDS}),
    MeasurementMetric.THROUGHPUT: frozenset({CanonicalUnit.RECORDS_PER_SECOND}),
    MeasurementMetric.ENERGY: frozenset({CanonicalUnit.JOULES}),
    MeasurementMetric.POWER: frozenset({CanonicalUnit.WATTS}),
}

_DIRECT_COMPUTATIONAL_METRICS = frozenset(
    {
        MeasurementMetric.WALL_CLOCK_TIME,
        MeasurementMetric.PROCESS_CPU_TIME,
        MeasurementMetric.START_RSS,
        MeasurementMetric.END_RSS,
        MeasurementMetric.ABSOLUTE_PEAK_RSS,
        MeasurementMetric.DATAFRAME_DEEP_MEMORY,
        MeasurementMetric.NUMPY_DENSE_NBYTES,
        MeasurementMetric.SERIALIZED_MODEL_SIZE,
    }
)
_DERIVED_METRICS = frozenset(
    {
        MeasurementMetric.INCREMENTAL_PEAK_RSS,
        MeasurementMetric.INFERENCE_LATENCY,
        MeasurementMetric.PER_RECORD_LATENCY,
        MeasurementMetric.THROUGHPUT,
    }
)


@dataclass(frozen=True)
class MeasurementRecord:
    """One schema-validated quantity with explicit provenance and canonical unit."""

    metric: MeasurementMetric | str
    value: float | int
    unit: CanonicalUnit | str
    provenance: MeasurementProvenance | str
    phase: MeasurementPhase | str
    scope: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        metric = MeasurementMetric(self.metric)
        unit = CanonicalUnit(self.unit)
        provenance = MeasurementProvenance(self.provenance)
        phase = MeasurementPhase(self.phase)
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("Measurement value must be numeric, not boolean.")
        if not math.isfinite(float(self.value)) or float(self.value) < 0.0:
            raise ValueError("Measurement value must be finite and nonnegative.")
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("Measurement scope must be a non-empty string.")
        if unit not in _METRIC_UNITS[metric]:
            allowed = ", ".join(sorted(item.value for item in _METRIC_UNITS[metric]))
            raise ValueError(f"{metric.value} must use one of: {allowed}.")
        if metric in _DIRECT_COMPUTATIONAL_METRICS and provenance not in {
            MeasurementProvenance.DIRECT_COMPUTATIONAL,
            MeasurementProvenance.IMPORTED_HISTORICAL,
        }:
            raise ValueError(f"{metric.value} is a computational measurement, not energy.")
        if metric in _DERIVED_METRICS and provenance not in {
            MeasurementProvenance.DERIVED,
            MeasurementProvenance.IMPORTED_HISTORICAL,
        }:
            raise ValueError(f"{metric.value} must be labelled as derived or historical.")
        if provenance is MeasurementProvenance.DIRECT_ENERGY and metric not in {
            MeasurementMetric.ENERGY,
            MeasurementMetric.POWER,
        }:
            raise ValueError("CPU timing and memory cannot be labelled DIRECT_ENERGY.")
        if metric is MeasurementMetric.ENERGY and provenance is MeasurementProvenance.DIRECT_COMPUTATIONAL:
            raise ValueError("Energy cannot be labelled DIRECT_COMPUTATIONAL.")
        if provenance is MeasurementProvenance.ESTIMATED and metric is not MeasurementMetric.ENERGY:
            raise ValueError("ESTIMATED is reserved here for explicitly estimated energy.")
        if provenance is MeasurementProvenance.DIRECT_ENERGY and bool(
            self.metadata.get("estimated", False)
        ):
            raise ValueError("Estimated energy cannot masquerade as DIRECT_ENERGY.")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "scope", self.scope.strip())
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric.value,
            "value": self.value,
            "unit": self.unit.value,
            "provenance": self.provenance.value,
            "phase": self.phase.value,
            "scope": self.scope,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ComputationalObservation:
    """Raw and derived computational quantities for one measured block."""

    phase: MeasurementPhase | str
    outer_repetition: int
    inner_operation_count: int
    measured_operation_count: int
    warmup_calls: int
    resources: ResourceMeasurement
    records_per_operation: int = 1
    dataframe_deep_memory_bytes: int | None = None
    numpy_dense_nbytes: int | None = None
    serialized_model_size_bytes: int | None = None
    measurement_scope: str = "operation_only"

    def __post_init__(self) -> None:
        phase = MeasurementPhase(self.phase)
        if self.outer_repetition < 1:
            raise ValueError("outer_repetition must be one-based and positive.")
        if self.inner_operation_count < 1:
            raise ValueError("inner_operation_count must be positive.")
        if self.measured_operation_count != self.inner_operation_count:
            raise ValueError("A successful observation must record every configured operation.")
        if self.warmup_calls < 0:
            raise ValueError("warmup_calls must be nonnegative.")
        if self.records_per_operation < 1:
            raise ValueError("records_per_operation must be positive.")
        if self.resources.wall_time_sec <= 0.0:
            raise ValueError("wall-clock time must be positive.")
        if self.resources.cpu_time_sec < 0.0:
            raise ValueError("process CPU time must be nonnegative.")
        if self.resources.peak_rss_bytes < max(
            self.resources.start_rss_bytes, self.resources.end_rss_bytes
        ):
            raise ValueError("Absolute peak RSS must be at least start and end RSS.")
        for value in (
            self.dataframe_deep_memory_bytes,
            self.numpy_dense_nbytes,
            self.serialized_model_size_bytes,
        ):
            if value is not None and value < 0:
                raise ValueError("Byte measurements must be nonnegative.")
        if not self.measurement_scope.strip():
            raise ValueError("measurement_scope must not be empty.")
        object.__setattr__(self, "phase", phase)

    @property
    def absolute_peak_rss_bytes(self) -> int:
        return self.resources.peak_rss_bytes

    @property
    def incremental_peak_rss_bytes(self) -> int:
        return self.resources.peak_rss_bytes - self.resources.start_rss_bytes

    @property
    def inference_latency_seconds(self) -> float:
        return self.resources.wall_time_sec / self.measured_operation_count

    @property
    def per_record_latency_seconds(self) -> float:
        return self.resources.wall_time_sec / self.measured_record_count

    @property
    def measured_record_count(self) -> int:
        return self.measured_operation_count * self.records_per_operation

    @property
    def throughput_records_per_second(self) -> float:
        return self.measured_record_count / self.resources.wall_time_sec

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "outer_repetition": self.outer_repetition,
            "inner_operation_count": self.inner_operation_count,
            "measured_operation_count": self.measured_operation_count,
            "measured_record_count": self.measured_record_count,
            "records_per_operation": self.records_per_operation,
            "warmup_calls": self.warmup_calls,
            "measurement_scope": self.measurement_scope,
            "wall_clock_seconds": self.resources.wall_time_sec,
            "process_cpu_seconds": self.resources.cpu_time_sec,
            "start_rss_bytes": self.resources.start_rss_bytes,
            "end_rss_bytes": self.resources.end_rss_bytes,
            "absolute_peak_rss_bytes": self.absolute_peak_rss_bytes,
            "incremental_peak_rss_bytes": self.incremental_peak_rss_bytes,
            "dataframe_deep_memory_bytes": self.dataframe_deep_memory_bytes,
            "numpy_dense_nbytes": self.numpy_dense_nbytes,
            "serialized_model_size_bytes": self.serialized_model_size_bytes,
            "inference_latency_seconds": self.inference_latency_seconds,
            "per_record_latency_seconds": self.per_record_latency_seconds,
            "throughput_records_per_second": self.throughput_records_per_second,
        }

    def measurement_records(self) -> tuple[MeasurementRecord, ...]:
        scope = self.measurement_scope
        records = [
            MeasurementRecord(MeasurementMetric.WALL_CLOCK_TIME, self.resources.wall_time_sec, CanonicalUnit.SECONDS, MeasurementProvenance.DIRECT_COMPUTATIONAL, self.phase, scope),
            MeasurementRecord(MeasurementMetric.PROCESS_CPU_TIME, self.resources.cpu_time_sec, CanonicalUnit.SECONDS, MeasurementProvenance.DIRECT_COMPUTATIONAL, self.phase, scope),
            MeasurementRecord(MeasurementMetric.START_RSS, self.resources.start_rss_bytes, CanonicalUnit.BYTES, MeasurementProvenance.DIRECT_COMPUTATIONAL, self.phase, scope),
            MeasurementRecord(MeasurementMetric.END_RSS, self.resources.end_rss_bytes, CanonicalUnit.BYTES, MeasurementProvenance.DIRECT_COMPUTATIONAL, self.phase, scope),
            MeasurementRecord(MeasurementMetric.ABSOLUTE_PEAK_RSS, self.resources.peak_rss_bytes, CanonicalUnit.BYTES, MeasurementProvenance.DIRECT_COMPUTATIONAL, self.phase, scope),
            MeasurementRecord(MeasurementMetric.INCREMENTAL_PEAK_RSS, self.incremental_peak_rss_bytes, CanonicalUnit.BYTES, MeasurementProvenance.DERIVED, self.phase, scope),
            MeasurementRecord(MeasurementMetric.INFERENCE_LATENCY, self.inference_latency_seconds, CanonicalUnit.SECONDS, MeasurementProvenance.DERIVED, self.phase, scope),
            MeasurementRecord(MeasurementMetric.PER_RECORD_LATENCY, self.per_record_latency_seconds, CanonicalUnit.SECONDS, MeasurementProvenance.DERIVED, self.phase, scope),
            MeasurementRecord(MeasurementMetric.THROUGHPUT, self.throughput_records_per_second, CanonicalUnit.RECORDS_PER_SECOND, MeasurementProvenance.DERIVED, self.phase, scope),
        ]
        optional = (
            (MeasurementMetric.DATAFRAME_DEEP_MEMORY, self.dataframe_deep_memory_bytes),
            (MeasurementMetric.NUMPY_DENSE_NBYTES, self.numpy_dense_nbytes),
            (MeasurementMetric.SERIALIZED_MODEL_SIZE, self.serialized_model_size_bytes),
        )
        records.extend(
            MeasurementRecord(metric, value, CanonicalUnit.BYTES, MeasurementProvenance.DIRECT_COMPUTATIONAL, self.phase, scope)
            for metric, value in optional
            if value is not None
        )
        return tuple(records)


def dataframe_deep_memory_bytes(frame: pd.DataFrame, *, include_index: bool = True) -> int:
    """Return pandas deep memory, including object payloads and optionally the index."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    return int(frame.memory_usage(index=include_index, deep=True).sum())


def numpy_dense_nbytes(array: np.ndarray) -> int:
    """Return actual dense NumPy storage bytes without estimating from shape."""

    if not isinstance(array, np.ndarray):
        raise TypeError("array must be a dense NumPy ndarray.")
    return int(array.nbytes)


def serialized_model_size_bytes(path: Path | str) -> int:
    """Reuse V0.5's filesystem-backed serialized model size measurement."""

    return int(measure_model_size(path)["model_size_bytes"])


def latency_and_throughput(
    wall_time_seconds: float,
    *,
    measured_operation_count: int,
    records_per_operation: int,
) -> dict[str, float | int]:
    """Derive per-operation/per-record latency and throughput from one block."""

    if not math.isfinite(wall_time_seconds) or wall_time_seconds <= 0.0:
        raise ValueError("wall_time_seconds must be finite and positive.")
    if measured_operation_count < 1 or records_per_operation < 1:
        raise ValueError("Operation and record counts must be positive.")
    records = measured_operation_count * records_per_operation
    return {
        "measured_operation_count": measured_operation_count,
        "measured_record_count": records,
        "inference_latency_seconds": wall_time_seconds / measured_operation_count,
        "per_record_latency_seconds": wall_time_seconds / records,
        "throughput_records_per_second": records / wall_time_seconds,
    }


@dataclass(frozen=True)
class FreshWorkerProtocol:
    """Configuration for repeated isolated measurements.

    ``inner_operation_count`` intentionally has no default. V0.7-B or later must
    supply the value selected by calibration rather than inheriting a hidden
    final experiment count.
    """

    inner_operation_count: int
    phase: MeasurementPhase | str = MeasurementPhase.INFERENCE
    warmup_calls: int = DEFAULT_WARMUP_CALLS
    outer_repetitions: int = DEFAULT_OUTER_REPETITIONS
    records_per_operation: int = 1
    timeout_seconds: float = 300.0
    measurement_scope: str = "operation_only_preparation_excluded"
    start_method: str | None = None

    def __post_init__(self) -> None:
        phase = MeasurementPhase(self.phase)
        if self.inner_operation_count < 1:
            raise ValueError("inner_operation_count must be positive.")
        if self.warmup_calls < 0:
            raise ValueError("warmup_calls must be nonnegative.")
        if self.outer_repetitions < 1:
            raise ValueError("outer_repetitions must be positive.")
        if self.records_per_operation < 1:
            raise ValueError("records_per_operation must be positive.")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be finite and positive.")
        if not self.measurement_scope.strip():
            raise ValueError("measurement_scope must not be empty.")
        if self.start_method is not None and self.start_method not in mp.get_all_start_methods():
            raise ValueError(f"Unsupported multiprocessing start method: {self.start_method}")
        object.__setattr__(self, "phase", phase)


class WorkerStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    NO_RESULT = "NO_RESULT"


@dataclass(frozen=True)
class WorkerFailure:
    error_type: str
    message: str
    stage: str
    traceback_text: str | None = None
    exit_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "stage": self.stage,
            "traceback": self.traceback_text,
            "exit_code": self.exit_code,
        }


@dataclass(frozen=True)
class WorkerResult:
    outer_repetition: int
    status: WorkerStatus | str
    worker_pid: int | None
    warmup_calls_completed: int
    measured_operation_count: int
    preparation_wall_time_seconds: float | None
    lifecycle_events: tuple[str, ...]
    observation: ComputationalObservation | None = None
    failure: WorkerFailure | None = None

    def __post_init__(self) -> None:
        status = WorkerStatus(self.status)
        if (status is WorkerStatus.COMPLETED) != (self.observation is not None):
            raise ValueError("Only completed workers may contain an observation.")
        if (status is WorkerStatus.COMPLETED) == (self.failure is not None):
            raise ValueError("Completed workers cannot contain failures; failed workers must.")
        object.__setattr__(self, "status", status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outer_repetition": self.outer_repetition,
            "status": self.status.value,
            "worker_pid": self.worker_pid,
            "warmup_calls_completed": self.warmup_calls_completed,
            "measured_operation_count": self.measured_operation_count,
            "preparation_wall_time_seconds": self.preparation_wall_time_seconds,
            "lifecycle_events": list(self.lifecycle_events),
            "observation": None if self.observation is None else self.observation.to_dict(),
            "failure": None if self.failure is None else self.failure.to_dict(),
        }


@dataclass(frozen=True)
class FreshWorkerRun:
    protocol: FreshWorkerProtocol
    results: tuple[WorkerResult, ...]

    def __post_init__(self) -> None:
        if len(self.results) != self.protocol.outer_repetitions:
            raise ValueError("Every outer repetition, including failures, must be retained.")
        if tuple(result.outer_repetition for result in self.results) != tuple(
            range(1, self.protocol.outer_repetitions + 1)
        ):
            raise ValueError("Worker results must preserve outer-repetition order.")

    @property
    def completed_count(self) -> int:
        return sum(result.status is WorkerStatus.COMPLETED for result in self.results)

    @property
    def failed_count(self) -> int:
        return len(self.results) - self.completed_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": {
                "phase": self.protocol.phase.value,
                "warmup_calls": self.protocol.warmup_calls,
                "outer_repetitions": self.protocol.outer_repetitions,
                "inner_operation_count": self.protocol.inner_operation_count,
                "records_per_operation": self.protocol.records_per_operation,
                "measurement_scope": self.protocol.measurement_scope,
            },
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "results": [result.to_dict() for result in self.results],
        }


def run_fresh_worker_protocol(
    operation: Callable[[Any], Any],
    *,
    inner_operation_count: int,
    prepare: Callable[[], Any] | None = None,
    phase: MeasurementPhase | str = MeasurementPhase.INFERENCE,
    warmup_calls: int = DEFAULT_WARMUP_CALLS,
    outer_repetitions: int = DEFAULT_OUTER_REPETITIONS,
    records_per_operation: int = 1,
    timeout_seconds: float = 300.0,
    measurement_scope: str = "operation_only_preparation_excluded",
    start_method: str | None = None,
) -> FreshWorkerRun:
    """Run each outer observation in a new process and retain every outcome.

    ``prepare`` executes before warm-up and before the resource boundary. Model
    loading can therefore be excluded from inference timing without being
    omitted from the worker lifecycle. Callables must be multiprocessing-safe
    (normally module-level functions or picklable callable objects).
    """

    protocol = FreshWorkerProtocol(
        inner_operation_count=inner_operation_count,
        phase=phase,
        warmup_calls=warmup_calls,
        outer_repetitions=outer_repetitions,
        records_per_operation=records_per_operation,
        timeout_seconds=timeout_seconds,
        measurement_scope=measurement_scope,
        start_method=start_method,
    )
    available_methods = mp.get_all_start_methods()
    method = start_method or ("forkserver" if "forkserver" in available_methods else "spawn")
    context = mp.get_context(method)
    results = tuple(
        _run_one_fresh_worker(context, operation, prepare, protocol, repetition)
        for repetition in range(1, protocol.outer_repetitions + 1)
    )
    return FreshWorkerRun(protocol=protocol, results=results)


def _run_one_fresh_worker(
    context: Any,
    operation: Callable[[Any], Any],
    prepare: Callable[[], Any] | None,
    protocol: FreshWorkerProtocol,
    repetition: int,
) -> WorkerResult:
    queue = context.Queue()
    process = context.Process(
        target=_fresh_worker_entry,
        args=(queue, operation, prepare, protocol, repetition),
    )
    try:
        process.start()
    except BaseException as exc:
        queue.close()
        queue.join_thread()
        return WorkerResult(
            repetition,
            WorkerStatus.FAILED,
            None,
            0,
            0,
            None,
            (),
            failure=WorkerFailure(type(exc).__name__, str(exc), "worker_start"),
        )
    try:
        payload = queue.get(timeout=protocol.timeout_seconds)
    except Empty:
        was_alive = process.is_alive()
        if was_alive:
            process.terminate()
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)
        if was_alive:
            result = WorkerResult(
                repetition,
                WorkerStatus.TIMED_OUT,
                process.pid,
                0,
                0,
                None,
                (),
                failure=WorkerFailure(
                    "TimeoutError",
                    f"Worker exceeded {protocol.timeout_seconds} seconds.",
                    "worker_timeout",
                    exit_code=process.exitcode,
                ),
            )
        else:
            result = WorkerResult(
                repetition,
                WorkerStatus.NO_RESULT,
                process.pid,
                0,
                0,
                None,
                (),
                failure=WorkerFailure(
                    "WorkerNoResult",
                    "Worker exited without returning a structured result.",
                    "result_transport",
                    exit_code=process.exitcode,
                ),
            )
    else:
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        result = _worker_result_from_payload(payload, process.exitcode)
    queue.close()
    queue.join_thread()
    return result


def _fresh_worker_entry(
    queue: Any,
    operation: Callable[[Any], Any],
    prepare: Callable[[], Any] | None,
    protocol: FreshWorkerProtocol,
    repetition: int,
) -> None:
    events: list[str] = []
    warmups = 0
    measured = 0
    preparation_time: float | None = None
    stage = "preparation"
    try:
        events.append("preparation_start")
        preparation_start = time.perf_counter()
        state = None if prepare is None else prepare()
        preparation_time = time.perf_counter() - preparation_start
        events.append("preparation_end")
        stage = "warmup"
        for call_number in range(1, protocol.warmup_calls + 1):
            operation(state)
            warmups += 1
            events.append(f"warmup:{call_number}")
        stage = "measurement"
        events.append("measurement_start")
        with ResourceMonitor() as monitor:
            for operation_number in range(1, protocol.inner_operation_count + 1):
                operation(state)
                measured += 1
                events.append(f"measured:{operation_number}")
        events.append("measurement_end")
        if monitor.measurement is None:
            raise RuntimeError("Resource monitor did not produce a measurement.")
        observation = ComputationalObservation(
            phase=protocol.phase,
            outer_repetition=repetition,
            inner_operation_count=protocol.inner_operation_count,
            measured_operation_count=measured,
            warmup_calls=warmups,
            resources=monitor.measurement,
            records_per_operation=protocol.records_per_operation,
            measurement_scope=protocol.measurement_scope,
        )
        queue.put(
            {
                "status": WorkerStatus.COMPLETED.value,
                "outer_repetition": repetition,
                "worker_pid": os.getpid(),
                "warmup_calls_completed": warmups,
                "measured_operation_count": measured,
                "preparation_wall_time_seconds": preparation_time,
                "lifecycle_events": events,
                "resources": monitor.measurement.to_dict(),
                "observation": observation.to_dict(),
            }
        )
    except BaseException as exc:
        queue.put(
            {
                "status": WorkerStatus.FAILED.value,
                "outer_repetition": repetition,
                "worker_pid": os.getpid(),
                "warmup_calls_completed": warmups,
                "measured_operation_count": measured,
                "preparation_wall_time_seconds": preparation_time,
                "lifecycle_events": events,
                "failure": {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "stage": stage,
                    "traceback": traceback.format_exc(),
                },
            }
        )


def _worker_result_from_payload(payload: Mapping[str, Any], exit_code: int | None) -> WorkerResult:
    status = WorkerStatus(payload["status"])
    if status is WorkerStatus.COMPLETED:
        raw = payload["observation"]
        observation = ComputationalObservation(
            phase=raw["phase"],
            outer_repetition=int(raw["outer_repetition"]),
            inner_operation_count=int(raw["inner_operation_count"]),
            measured_operation_count=int(raw["measured_operation_count"]),
            warmup_calls=int(raw["warmup_calls"]),
            resources=ResourceMeasurement.from_dict(dict(payload["resources"])),
            records_per_operation=int(raw["records_per_operation"]),
            measurement_scope=str(raw["measurement_scope"]),
        )
        return WorkerResult(
            int(payload["outer_repetition"]),
            status,
            int(payload["worker_pid"]),
            int(payload["warmup_calls_completed"]),
            int(payload["measured_operation_count"]),
            float(payload["preparation_wall_time_seconds"]),
            tuple(payload["lifecycle_events"]),
            observation=observation,
        )
    raw_failure = payload["failure"]
    return WorkerResult(
        int(payload["outer_repetition"]),
        status,
        int(payload["worker_pid"]),
        int(payload["warmup_calls_completed"]),
        int(payload["measured_operation_count"]),
        payload["preparation_wall_time_seconds"],
        tuple(payload["lifecycle_events"]),
        failure=WorkerFailure(
            str(raw_failure["error_type"]),
            str(raw_failure["message"]),
            str(raw_failure["stage"]),
            raw_failure.get("traceback"),
            exit_code,
        ),
    )


def is_wsl_environment(
    *, release: str | None = None, version: str | None = None, environ: Mapping[str, str] | None = None
) -> bool:
    """Return whether the current Linux environment is Windows Subsystem for Linux."""

    release_text = platform.release() if release is None else release
    version_text = platform.version() if version is None else version
    environment = os.environ if environ is None else environ
    combined = f"{release_text} {version_text}".lower()
    return "microsoft" in combined or "wsl" in combined or bool(environment.get("WSL_DISTRO_NAME"))


def collect_git_metadata(repository_path: Path | str | None = None) -> dict[str, Any]:
    """Return structured Git commit/dirty metadata without requiring a repository."""

    path = Path.cwd() if repository_path is None else Path(repository_path)
    git = shutil.which("git")
    if git is None:
        return {"available": False, "commit": None, "dirty": None, "reason": "git_not_installed"}
    try:
        commit = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            [git, "status", "--porcelain"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "commit": None,
            "dirty": None,
            "reason": f"git_metadata_unavailable: {type(exc).__name__}",
        }
    return {"available": True, "commit": commit, "dirty": bool(status.strip()), "reason": None}


def collect_environment_metadata(
    repository_path: Path | str | None = None,
    *,
    dependency_names: Sequence[str] = (
        "numpy",
        "pandas",
        "scikit-learn",
        "psutil",
        "PyYAML",
        "pytest",
    ),
) -> dict[str, Any]:
    """Collect reproducible host, runtime, dependency, and source-control context."""

    wsl = is_wsl_environment()
    memory = psutil.virtual_memory()
    battery = _safe_battery_status(wsl)
    return {
        "os": {"system": platform.system(), "release": platform.release(), "version": platform.version()},
        "kernel": platform.release(),
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "cpu": {
            "model": _cpu_model(),
            "physical_count": psutil.cpu_count(logical=False),
            "logical_count": psutil.cpu_count(logical=True),
        },
        "ram": {"total_bytes": int(memory.total)},
        "virtualization": _virtualization_metadata(wsl),
        "execution_environment": {
            "is_wsl": wsl,
            "is_native": not wsl,
            "kind": "wsl" if wsl else "native",
        },
        "gpu": _gpu_metadata(),
        "power_source": battery,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        },
        "dependencies": {name: _dependency_version(name) for name in dependency_names},
        "git": collect_git_metadata(repository_path),
    }


def _dependency_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cpu_model() -> str | None:
    candidates = [platform.processor(), os.environ.get("PROCESSOR_IDENTIFIER", "")]
    cpuinfo = Path("/proc/cpuinfo")
    try:
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                candidates.insert(0, line.split(":", 1)[1].strip())
                break
    except OSError:
        pass
    return next((value.strip() for value in candidates if value and value.strip()), None)


def _virtualization_metadata(wsl: bool) -> dict[str, Any]:
    if wsl:
        return {"detected": True, "type": "WSL", "evidence": "kernel_or_environment"}
    evidence: list[str] = []
    for path in (Path("/sys/class/dmi/id/product_name"), Path("/sys/class/dmi/id/sys_vendor")):
        try:
            value = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if value:
            evidence.append(value)
    markers = ("virtual", "vmware", "kvm", "hyper-v", "qemu", "xen")
    detected = any(marker in " ".join(evidence).lower() for marker in markers)
    return {
        "detected": detected,
        "type": "virtual_machine" if detected else "none_detected",
        "evidence": evidence,
    }


def _gpu_metadata() -> dict[str, Any]:
    command = shutil.which("nvidia-smi")
    if command is None:
        return {"present": False, "devices": [], "detection": "nvidia_smi_not_found"}
    try:
        output = subprocess.run(
            [command, "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return {"present": None, "devices": [], "detection": f"probe_failed: {type(exc).__name__}"}
    devices = [line.strip() for line in output.splitlines() if line.strip()]
    return {"present": bool(devices), "devices": devices, "detection": "nvidia_smi"}


def _safe_battery_status(wsl: bool) -> dict[str, Any]:
    try:
        battery = psutil.sensors_battery()
    except (AttributeError, OSError, psutil.Error):
        battery = None
    if battery is None:
        return {"available": False, "reliable": False, "source": None, "reason": "sensor_unavailable"}
    if wsl:
        return {
            "available": True,
            "reliable": False,
            "source": "virtual_battery",
            "reason": "WSL battery telemetry is not accepted as wall-power instrumentation.",
        }
    return {
        "available": True,
        "reliable": True,
        "source": "AC" if battery.power_plugged else "battery",
        "percent": float(battery.percent),
        "reason": None,
    }


# Concise aliases for report code and downstream stages.
Provenance = MeasurementProvenance
Phase = MeasurementPhase
