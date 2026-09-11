"""Energy capability probing and scientifically constrained counter utilities.

V0.7-A performs capability discovery only. No function in this module runs a
real energy experiment, and unavailable optional hardware never causes a probe
to crash merely because it is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import importlib
import math
from pathlib import Path
import platform
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from src.green.measurement import (
    CanonicalUnit,
    MeasurementMetric,
    MeasurementPhase,
    MeasurementProvenance,
    MeasurementRecord,
    is_wsl_environment,
)


MICROJOULES_PER_JOULE = 1_000_000.0
MILLIJOULES_PER_JOULE = 1_000.0
DEFAULT_RAPL_ROOT = Path("/sys/class/powercap")


class EnergyBackend(str, Enum):
    LINUX_RAPL = "linux_rapl"
    WINDOWS_EMI = "windows_emi"
    NVML = "nvml"
    EXTERNAL_METER = "external_meter"
    HISTORICAL_ESTIMATE = "historical_estimate"


@dataclass(frozen=True)
class CapabilityResult:
    backend: EnergyBackend | str
    available: bool
    reason_code: str
    reason: str
    supports_cumulative_energy: bool = False
    supports_instantaneous_power: bool = False
    cumulative_energy_unit: str | None = None
    instantaneous_power_unit: str | None = None
    workload_relevant: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        backend = EnergyBackend(self.backend)
        if not self.reason_code.strip() or not self.reason.strip():
            raise ValueError("Capability results require a structured reason code and reason.")
        if self.supports_cumulative_energy and self.cumulative_energy_unit is None:
            raise ValueError("Cumulative-energy support requires an explicit unit.")
        if self.supports_instantaneous_power and self.instantaneous_power_unit is None:
            raise ValueError("Instantaneous-power support requires an explicit unit.")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend.value,
            "available": self.available,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "supports_cumulative_energy": self.supports_cumulative_energy,
            "supports_instantaneous_power": self.supports_instantaneous_power,
            "cumulative_energy_unit": self.cumulative_energy_unit,
            "instantaneous_power_unit": self.instantaneous_power_unit,
            "workload_relevant": self.workload_relevant,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RaplDomain:
    domain_id: str
    path: Path
    name: str | None
    energy_uj_path: Path
    max_energy_range_uj: int | None
    readable: bool
    readability_reason: str
    parent_domain_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "path": str(self.path),
            "name": self.name,
            "energy_uj": str(self.energy_uj_path),
            "max_energy_range_uj": self.max_energy_range_uj,
            "readable": self.readable,
            "readability_reason": self.readability_reason,
            "parent_domain_id": self.parent_domain_id,
        }


def probe_linux_rapl(root: Path | str = DEFAULT_RAPL_ROOT) -> CapabilityResult:
    """Safely enumerate RAPL domains without aggregating overlapping counters."""

    powercap = Path(root)
    if not powercap.exists() or not powercap.is_dir():
        return CapabilityResult(
            EnergyBackend.LINUX_RAPL,
            False,
            "POWERCAP_PATH_UNAVAILABLE",
            f"Linux powercap directory is unavailable: {powercap}",
            workload_relevant=True,
            metadata={"root": str(powercap), "domains": [], "automatic_aggregation": False},
        )
    domains = _discover_rapl_domains(powercap)
    if not domains:
        return CapabilityResult(
            EnergyBackend.LINUX_RAPL,
            False,
            "NO_ENERGY_COUNTERS",
            "No energy_uj counters were found under Linux powercap.",
            workload_relevant=True,
            metadata={"root": str(powercap), "domains": [], "automatic_aggregation": False},
        )
    readable = [domain for domain in domains if domain.readable]
    return CapabilityResult(
        EnergyBackend.LINUX_RAPL,
        bool(readable),
        "AVAILABLE" if readable else "COUNTERS_UNREADABLE",
        (
            "Readable RAPL counters are available; explicit non-overlapping domain selection is required."
            if readable
            else "RAPL energy counters exist but none are readable."
        ),
        supports_cumulative_energy=bool(readable),
        cumulative_energy_unit="microjoules" if readable else None,
        workload_relevant=True,
        metadata={
            "root": str(powercap),
            "domains": [domain.to_dict() for domain in domains],
            "automatic_aggregation": False,
            "aggregation_policy": "explicit_non_overlapping_domain_selection_required",
        },
    )


def discover_rapl_domains(root: Path | str = DEFAULT_RAPL_ROOT) -> tuple[RaplDomain, ...]:
    """Public domain discovery helper used by preflight and synthetic validation."""

    powercap = Path(root)
    if not powercap.is_dir():
        return ()
    return _discover_rapl_domains(powercap)


def _discover_rapl_domains(root: Path) -> tuple[RaplDomain, ...]:
    preliminary: list[RaplDomain] = []
    try:
        counters = sorted(root.rglob("energy_uj"), key=lambda item: str(item))
    except OSError:
        counters = []
    for counter in counters:
        directory = counter.parent
        domain_id = str(directory.relative_to(root)) or "."
        readable, readability_reason = _counter_readability(counter)
        preliminary.append(
            RaplDomain(
                domain_id=domain_id,
                path=directory,
                name=_read_text(directory / "name"),
                energy_uj_path=counter,
                max_energy_range_uj=_read_nonnegative_int(directory / "max_energy_range_uj"),
                readable=readable,
                readability_reason=readability_reason,
            )
        )
    paths = {domain.path: domain.domain_id for domain in preliminary}
    complete: list[RaplDomain] = []
    for domain in preliminary:
        parent_id = next(
            (paths[parent] for parent in domain.path.parents if parent in paths),
            None,
        )
        complete.append(replace(domain, parent_domain_id=parent_id))
    return tuple(complete)


def rapl_energy_delta_joules(
    start_energy_uj: int,
    end_energy_uj: int,
    max_energy_range_uj: int | None = None,
) -> float:
    """Calculate one RAPL counter delta, including a single counter wrap."""

    if isinstance(start_energy_uj, bool) or isinstance(end_energy_uj, bool):
        raise TypeError("RAPL counter values must be integer microjoules.")
    if start_energy_uj < 0 or end_energy_uj < 0:
        raise ValueError("RAPL counter values must be nonnegative.")
    if end_energy_uj >= start_energy_uj:
        delta_uj = end_energy_uj - start_energy_uj
    else:
        if max_energy_range_uj is None:
            raise ValueError("Counter decreased; max_energy_range_uj is required for wraparound.")
        if max_energy_range_uj <= 0 or start_energy_uj > max_energy_range_uj or end_energy_uj > max_energy_range_uj:
            raise ValueError("Counter values must fall within max_energy_range_uj.")
        delta_uj = (max_energy_range_uj - start_energy_uj) + end_energy_uj
    return delta_uj / MICROJOULES_PER_JOULE


def read_selected_rapl_counters(
    domains: Sequence[RaplDomain], selected_domain_ids: Sequence[str]
) -> dict[str, int]:
    """Read only an explicit, validated, non-overlapping RAPL domain set."""

    selected = _validate_rapl_selection(domains, selected_domain_ids)
    return {
        domain.domain_id: int(domain.energy_uj_path.read_text(encoding="ascii").strip())
        for domain in selected
    }


def aggregate_selected_rapl_delta_joules(
    domains: Sequence[RaplDomain],
    selected_domain_ids: Sequence[str],
    start_readings_uj: Mapping[str, int],
    end_readings_uj: Mapping[str, int],
) -> dict[str, Any]:
    """Aggregate only an explicitly selected set proven non-overlapping by hierarchy."""

    selected = _validate_rapl_selection(domains, selected_domain_ids)
    deltas: dict[str, float] = {}
    for domain in selected:
        if domain.domain_id not in start_readings_uj or domain.domain_id not in end_readings_uj:
            raise ValueError(f"Missing start/end reading for RAPL domain {domain.domain_id}.")
        deltas[domain.domain_id] = rapl_energy_delta_joules(
            start_readings_uj[domain.domain_id],
            end_readings_uj[domain.domain_id],
            domain.max_energy_range_uj,
        )
    return {
        "selected_domain_ids": list(selected_domain_ids),
        "domain_deltas_joules": deltas,
        "total_joules": sum(deltas.values()),
        "unit": CanonicalUnit.JOULES.value,
        "provenance": MeasurementProvenance.DIRECT_ENERGY.value,
        "automatic_aggregation": False,
    }


def _validate_rapl_selection(
    domains: Sequence[RaplDomain], selected_domain_ids: Sequence[str]
) -> tuple[RaplDomain, ...]:
    if not selected_domain_ids:
        raise ValueError("RAPL aggregation requires explicit domain selection.")
    if len(set(selected_domain_ids)) != len(selected_domain_ids):
        raise ValueError("RAPL domain selection contains duplicates.")
    by_id = {domain.domain_id: domain for domain in domains}
    unknown = [domain_id for domain_id in selected_domain_ids if domain_id not in by_id]
    if unknown:
        raise ValueError(f"Unknown RAPL domains: {unknown}")
    selected = tuple(by_id[domain_id] for domain_id in selected_domain_ids)
    if any(not domain.readable for domain in selected):
        raise ValueError("Selected RAPL domains must all be readable.")
    selected_ids = set(selected_domain_ids)
    for domain in selected:
        parent = domain.parent_domain_id
        while parent is not None:
            if parent in selected_ids:
                raise ValueError(
                    f"Overlapping RAPL domains cannot be summed: {parent} and {domain.domain_id}."
                )
            parent = by_id[parent].parent_domain_id if parent in by_id else None
    return selected


def probe_windows_emi(
    *, system_name: str | None = None, wsl: bool | None = None
) -> CapabilityResult:
    """Report EMI unavailable unless native Windows support can be verified.

    V0.7-A intentionally contains no native EMI counter reader, so even native
    Windows is reported as unverified rather than fabricated as available.
    """

    system = platform.system() if system_name is None else system_name
    running_in_wsl = is_wsl_environment() if wsl is None else wsl
    if running_in_wsl:
        return CapabilityResult(
            EnergyBackend.WINDOWS_EMI,
            False,
            "WSL_NOT_NATIVE_WINDOWS",
            "WSL is not equivalent to native Windows EMI access.",
            workload_relevant=True,
            metadata={"system": system, "is_wsl": True, "verification": "not_verified"},
        )
    if system.lower() != "windows":
        return CapabilityResult(
            EnergyBackend.WINDOWS_EMI,
            False,
            "UNSUPPORTED_OS",
            "Windows EMI requires verified native Windows access.",
            workload_relevant=True,
            metadata={"system": system, "is_wsl": False, "verification": "not_verified"},
        )
    return CapabilityResult(
        EnergyBackend.WINDOWS_EMI,
        False,
        "NATIVE_EMI_UNVERIFIED",
        "Native Windows detected, but V0.7-A has no verified EMI counter adapter.",
        workload_relevant=True,
        metadata={"system": system, "is_wsl": False, "verification": "not_verified"},
    )


def probe_nvml(
    *,
    workload_uses_gpu: bool = False,
    module_loader: Callable[[str], Any] | None = None,
) -> CapabilityResult:
    """Probe optional NVML power and cumulative-energy interfaces independently."""

    loader = importlib.import_module if module_loader is None else module_loader
    try:
        nvml = loader("pynvml")
    except (ImportError, ModuleNotFoundError) as exc:
        return CapabilityResult(
            EnergyBackend.NVML,
            False,
            "OPTIONAL_DEPENDENCY_MISSING",
            f"Optional pynvml dependency is unavailable: {type(exc).__name__}",
            workload_relevant=workload_uses_gpu,
            metadata={"workload": "gpu" if workload_uses_gpu else "cpu_only"},
        )
    try:
        nvml.nvmlInit()
        count = int(nvml.nvmlDeviceGetCount())
        devices: list[dict[str, Any]] = []
        for index in range(count):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            name = nvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            power_supported = _nvml_call_supported(nvml, "nvmlDeviceGetPowerUsage", handle)
            energy_supported = _nvml_call_supported(
                nvml, "nvmlDeviceGetTotalEnergyConsumption", handle
            )
            devices.append(
                {
                    "index": index,
                    "name": str(name),
                    "instantaneous_power_supported": power_supported,
                    "cumulative_energy_supported": energy_supported,
                }
            )
    except BaseException as exc:
        return CapabilityResult(
            EnergyBackend.NVML,
            False,
            "NVML_INITIALIZATION_FAILED",
            f"NVML could not be queried safely: {type(exc).__name__}: {exc}",
            workload_relevant=workload_uses_gpu,
            metadata={"workload": "gpu" if workload_uses_gpu else "cpu_only"},
        )
    finally:
        shutdown = getattr(nvml, "nvmlShutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except BaseException:
                pass
    power = any(device["instantaneous_power_supported"] for device in devices)
    energy = any(device["cumulative_energy_supported"] for device in devices)
    reason_code = "AVAILABLE" if energy else "INSTANTANEOUS_POWER_ONLY" if power else "NO_ENERGY_TELEMETRY"
    reason = (
        "NVML is available. Cumulative millijoule counters and instantaneous milliwatt readings are distinct."
        if energy
        else "NVML exposes instantaneous power only; watts are not cumulative energy."
        if power
        else "NVML initialized but exposed no supported energy or power telemetry."
    )
    if not workload_uses_gpu:
        reason += " GPU telemetry is not a total-workload energy measure for CPU-only experiments."
    return CapabilityResult(
        EnergyBackend.NVML,
        bool(devices),
        reason_code,
        reason,
        supports_cumulative_energy=energy,
        supports_instantaneous_power=power,
        cumulative_energy_unit="millijoules" if energy else None,
        instantaneous_power_unit="milliwatts" if power else None,
        workload_relevant=workload_uses_gpu,
        metadata={
            "devices": devices,
            "workload": "gpu" if workload_uses_gpu else "cpu_only",
            "instantaneous_power_is_energy": False,
        },
    )


def _nvml_call_supported(nvml: Any, function_name: str, handle: Any) -> bool:
    function = getattr(nvml, function_name, None)
    if not callable(function):
        return False
    try:
        function(handle)
    except BaseException:
        return False
    return True


@dataclass(frozen=True)
class ExternalMeterConfig:
    """Future adapter configuration; it does not imply physical attachment."""

    meter_name: str
    connection: str | None = None
    sampling_interval_seconds: float | None = None
    verified_attached: bool = False

    def __post_init__(self) -> None:
        if not self.meter_name.strip():
            raise ValueError("meter_name must not be empty.")
        if self.sampling_interval_seconds is not None and (
            not math.isfinite(self.sampling_interval_seconds)
            or self.sampling_interval_seconds <= 0.0
        ):
            raise ValueError("sampling_interval_seconds must be finite and positive.")
        if self.verified_attached:
            raise ValueError("V0.7-A cannot mark an external meter as verified attached.")


class ExternalMeterAdapter(Protocol):
    """Interface contract for a future calibrated cumulative-energy meter."""

    def probe(self) -> CapabilityResult: ...

    def read_cumulative_energy_joules(self) -> float: ...


def probe_external_meter(config: ExternalMeterConfig | None = None) -> CapabilityResult:
    """Return an honest unavailable placeholder for future wattmeter adapters."""

    return CapabilityResult(
        EnergyBackend.EXTERNAL_METER,
        False,
        "NOT_CONFIGURED" if config is None else "INTERFACE_ONLY",
        (
            "No external power meter is configured or claimed to be attached."
            if config is None
            else "External-meter configuration exists, but V0.7-A provides only an unverified interface placeholder."
        ),
        workload_relevant=True,
        metadata={"configuration": None if config is None else config.__dict__},
    )


def probe_historical_estimate() -> CapabilityResult:
    """Describe historical TDP x time support as estimation, never direct energy."""

    return CapabilityResult(
        EnergyBackend.HISTORICAL_ESTIMATE,
        True,
        "ESTIMATE_ONLY",
        "Historical TDP x time is available only as an ESTIMATED quantity.",
        workload_relevant=True,
        metadata={
            "provenance": MeasurementProvenance.ESTIMATED.value,
            "direct_energy": False,
            "method": "TDP_x_wall_time",
        },
    )


def estimate_tdp_energy_record(
    power_watts: float,
    wall_time_seconds: float,
    *,
    phase: MeasurementPhase | str,
    scope: str,
) -> MeasurementRecord:
    """Create an explicitly estimated TDP x time energy record."""

    if not math.isfinite(power_watts) or power_watts < 0.0:
        raise ValueError("power_watts must be finite and nonnegative.")
    if not math.isfinite(wall_time_seconds) or wall_time_seconds < 0.0:
        raise ValueError("wall_time_seconds must be finite and nonnegative.")
    return MeasurementRecord(
        MeasurementMetric.ENERGY,
        power_watts * wall_time_seconds,
        CanonicalUnit.JOULES,
        MeasurementProvenance.ESTIMATED,
        phase,
        scope,
        metadata={
            "estimated": True,
            "method": "TDP_x_wall_time",
            "assumed_power_watts": power_watts,
            "measured_wall_time_seconds": wall_time_seconds,
        },
    )


def instantaneous_watts_to_joules(_: float) -> float:
    """Reject conversion without a sampled integration protocol and timestamps."""

    raise ValueError(
        "Instantaneous watts are not cumulative joules; a validated time-series integration protocol is required."
    )


def probe_energy_capabilities(
    *,
    rapl_root: Path | str = DEFAULT_RAPL_ROOT,
    workload_uses_gpu: bool = False,
    external_meter_config: ExternalMeterConfig | None = None,
) -> tuple[CapabilityResult, ...]:
    """Probe every V0.7-A backend without selecting one or running a workload."""

    return (
        probe_linux_rapl(rapl_root),
        probe_windows_emi(),
        probe_nvml(workload_uses_gpu=workload_uses_gpu),
        probe_external_meter(external_meter_config),
        probe_historical_estimate(),
    )


def _counter_readability(path: Path) -> tuple[bool, str]:
    try:
        value = int(path.read_text(encoding="ascii").strip())
    except PermissionError:
        return False, "permission_denied"
    except (OSError, ValueError):
        return False, "not_readable_integer"
    return (True, "readable") if value >= 0 else (False, "negative_counter")


def _read_text(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    return value or None


def _read_nonnegative_int(path: Path) -> int | None:
    try:
        value = int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None
    return value if value >= 0 else None


# Naming aliases used in some instrumentation literature.
calculate_rapl_delta_joules = rapl_energy_delta_joules
