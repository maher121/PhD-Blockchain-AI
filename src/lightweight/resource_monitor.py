"""Reusable process-resource monitoring for V0.5 computational proxies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Callable

import psutil


@dataclass(frozen=True)
class ResourceMeasurement:
    wall_time_sec: float
    cpu_time_sec: float
    average_cpu_percent: float
    peak_cpu_percent: float | None
    peak_rss_bytes: int
    start_rss_bytes: int
    end_rss_bytes: int

    @property
    def peak_rss_mb(self) -> float:
        return self.peak_rss_bytes / (1024.0 * 1024.0)

    @property
    def rss_delta_mb(self) -> float:
        return (self.peak_rss_bytes - self.start_rss_bytes) / (1024.0 * 1024.0)

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "wall_time_sec": self.wall_time_sec,
            "cpu_time_sec": self.cpu_time_sec,
            "average_cpu_percent": self.average_cpu_percent,
            "peak_cpu_percent": self.peak_cpu_percent,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_rss_mb": self.peak_rss_mb,
            "start_rss_bytes": self.start_rss_bytes,
            "end_rss_bytes": self.end_rss_bytes,
            "rss_delta_mb": self.rss_delta_mb,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ResourceMeasurement":
        return cls(
            wall_time_sec=float(values["wall_time_sec"]),
            cpu_time_sec=float(values["cpu_time_sec"]),
            average_cpu_percent=float(values["average_cpu_percent"]),
            peak_cpu_percent=(
                None
                if values.get("peak_cpu_percent") is None
                else float(values["peak_cpu_percent"])
            ),
            peak_rss_bytes=int(values["peak_rss_bytes"]),
            start_rss_bytes=int(values["start_rss_bytes"]),
            end_rss_bytes=int(values["end_rss_bytes"]),
        )

    def with_per_call_time(self, repeats: int) -> "ResourceMeasurement":
        if repeats < 1:
            raise ValueError("repeats must be positive.")
        return ResourceMeasurement(
            wall_time_sec=self.wall_time_sec / repeats,
            cpu_time_sec=self.cpu_time_sec / repeats,
            average_cpu_percent=self.average_cpu_percent,
            peak_cpu_percent=self.peak_cpu_percent,
            peak_rss_bytes=self.peak_rss_bytes,
            start_rss_bytes=self.start_rss_bytes,
            end_rss_bytes=self.end_rss_bytes,
        )


class ResourceMonitor:
    """Measure a block without adding a fixed sampler-sleep delay to timing."""

    def __init__(self, *, sample_interval_sec: float = 0.005) -> None:
        self.sample_interval_sec = max(float(sample_interval_sec), 0.001)
        self.measurement: ResourceMeasurement | None = None

    def __enter__(self) -> "ResourceMonitor":
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._rss_samples = [int(self._process.memory_info().rss)]
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()

        def sample() -> None:
            while not self._stop.wait(self.sample_interval_sec):
                try:
                    self._rss_samples.append(int(self._process.memory_info().rss))
                except psutil.Error:
                    pass

        self._thread = threading.Thread(target=sample, daemon=True, name="v0.5-rss-monitor")
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        wall_time = time.perf_counter() - self._wall_start
        cpu_time = time.process_time() - self._cpu_start
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            end_rss = int(self._process.memory_info().rss)
        except psutil.Error:
            end_rss = self._rss_samples[-1]
        self._rss_samples.append(end_rss)
        self.measurement = ResourceMeasurement(
            wall_time_sec=wall_time,
            cpu_time_sec=cpu_time,
            average_cpu_percent=(100.0 * cpu_time / wall_time if wall_time else 0.0),
            peak_cpu_percent=None,
            peak_rss_bytes=max(self._rss_samples),
            start_rss_bytes=self._rss_samples[0],
            end_rss_bytes=end_rss,
        )


def measure_call(
    function: Callable[[], Any], *, label: str
) -> tuple[Any, ResourceMeasurement]:
    """Execute a zero-argument callable and measure time, CPU, and process RSS."""
    del label
    with ResourceMonitor() as monitor:
        result = function()
    if monitor.measurement is None:
        raise RuntimeError("Resource monitor did not produce a measurement.")
    return result, monitor.measurement


def measure_model_size(path: Path | str) -> dict[str, float | int]:
    """Read the actual serialized artifact size from disk."""
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Model artifact not found: {artifact}")
    size = int(artifact.stat().st_size)
    return {"model_size_bytes": size, "model_size_kb": size / 1024.0}
