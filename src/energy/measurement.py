"""System-level measurement: wall/CPU time, CPU utilization and memory.

All metrics returned by this module are *measured* on the executing machine
via ``time`` and ``psutil``. They are never called "energy".
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class SystemSnapshot:
    """A point-in-time sample of process resource usage."""

    cpu_percent: float
    memory_rss_bytes: int
    memory_vms_bytes: int


@dataclass
class Measurement:
    """Aggregated measurement results for one instrumented execution."""

    wall_time_seconds: float = 0.0
    cpu_time_seconds: float = 0.0
    avg_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    peak_memory_rss_bytes: int = 0
    start_memory_rss_bytes: int = 0
    end_memory_rss_bytes: int = 0
    samples: list[SystemSnapshot] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "wall_time_seconds": round(self.wall_time_seconds, 6),
            "cpu_time_seconds": round(self.cpu_time_seconds, 6),
            "avg_cpu_percent": round(self.avg_cpu_percent, 2),
            "peak_cpu_percent": round(self.peak_cpu_percent, 2),
            "peak_memory_rss_bytes": self.peak_memory_rss_bytes,
            "start_memory_rss_bytes": self.start_memory_rss_bytes,
            "end_memory_rss_bytes": self.end_memory_rss_bytes,
            "sample_count": len(self.samples),
        }


def _process_handle() -> psutil.Process:
    """Return a psutil handle for the current process."""
    return psutil.Process()


def read_system_snapshot(process: psutil.Process) -> SystemSnapshot:
    """Sample current CPU and memory usage of ``process``."""
    try:
        cpu = process.cpu_percent(interval=None)
    except psutil.Error as exc:
        logger.warning("Could not read CPU usage: %s", exc)
        cpu = 0.0
    try:
        memory_info = process.memory_info()
        rss = int(memory_info.rss)
        vms = int(memory_info.vms)
    except psutil.Error as exc:
        logger.warning("Could not read memory usage: %s", exc)
        rss, vms = 0, 0
    return SystemSnapshot(cpu_percent=cpu, memory_rss_bytes=rss, memory_vms_bytes=vms)


def report_system_info() -> dict[str, Any]:
    """Return platform CPU/memory information for the current machine.

    All values are *measured* on this machine; they are informational context,
    not physics-level instrumentation.
    """
    import platform

    process = _process_handle()
    snapshot = read_system_snapshot(process)
    memory = psutil.virtual_memory()
    cpu_freq = psutil.cpu_freq()

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "cpu_freq_max_mhz": round(float(cpu_freq.max)) if cpu_freq else None,
        "system_cpu_percent": round(psutil.cpu_percent(interval=0.2), 2),
        "process_cpu_percent": round(snapshot.cpu_percent, 2),
        "process_rss_bytes": snapshot.memory_rss_bytes,
        "memory_total_bytes": int(memory.total),
        "memory_available_bytes": int(memory.available),
        "memory_percent": round(float(memory.percent), 2),
    }


class ExecutionTimer:
    """Context manager that measures time, CPU and memory of a code block.

    Example
    -------
    >>> with ExecutionTimer() as timer:
    ...     "<code under measurement>"
    >>> print(timer.measurement.to_dict())

    The fraction of CPU utilised is relative to a single core and is sampled
    on a best-effort basis; for quick blocks it may read 0.0.
    """

    def __init__(self, sample_interval_seconds: float = 0.02, label: str = "block") -> None:
        self._sample_interval_seconds = max(sample_interval_seconds, 0.005)
        self.label = label
        self._cpu_percent_samples: list[float] = []
        self._snapshots: list[SystemSnapshot] = []
        self.measurement = Measurement()
        self._process = _process_handle()
        self._entered = False

    def __enter__(self) -> "ExecutionTimer":
        self._process = _process_handle()
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()
        self._entered = True
        self._snapshots.append(read_system_snapshot(self._process))
        self.measurement.start_memory_rss_bytes = self._snapshots[-1].memory_rss_bytes

        def _sample() -> None:
            while self._entered:
                snapshot = read_system_snapshot(self._process)
                self._snapshots.append(snapshot)
                self._cpu_percent_samples.append(snapshot.cpu_percent)
                time.sleep(self._sample_interval_seconds)

        self._sample_thread_handle = threading.Thread(
            target=_sample, daemon=True, name=f"timer-{self.label}"
        )
        self._sample_thread_handle.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._entered = False
        self._sample_thread_handle.join(timeout=1.0)
        self._snapshots.append(read_system_snapshot(self._process))
        end_rss = self._snapshots[-1].memory_rss_bytes
        wall = time.perf_counter() - self._wall_start
        cpu = time.process_time() - self._cpu_start

        peak_rss = max(snapshot.memory_rss_bytes for snapshot in self._snapshots)
        peak_cpu = max(self._cpu_percent_samples) if self._cpu_percent_samples else 0.0
        avg_cpu = sum(self._cpu_percent_samples) / len(self._cpu_percent_samples) if self._cpu_percent_samples else 0.0

        self.measurement = Measurement(
            wall_time_seconds=wall,
            cpu_time_seconds=cpu,
            avg_cpu_percent=avg_cpu,
            peak_cpu_percent=peak_cpu,
            peak_memory_rss_bytes=peak_rss,
            start_memory_rss_bytes=self.measurement.start_memory_rss_bytes,
            end_memory_rss_bytes=end_rss,
            samples=self._snapshots,
        )
        logger.info(
            "Measured %s: wall=%.4fs cpu=%.4fs peak_rss=%d bytes",
            self.label,
            wall,
            cpu,
            peak_rss,
        )


def measure_function(measurement_fn: Any) -> tuple[Any, Measurement]:
    """Measure a zero-argument callable and return (result, measurement)."""
    with ExecutionTimer(label=measurement_fn.__name__) as timer:
        result = measurement_fn()
    return result, timer.measurement