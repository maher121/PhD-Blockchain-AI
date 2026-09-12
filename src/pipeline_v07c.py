"""Direct-energy backend validation and scientific feasibility gate for V0.7-C.

This stage probes capabilities and validates supplied preflight evidence. It does
not execute benchmark workloads, estimate direct energy, or produce energy-run
results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from src.config import PROJECT_ROOT
from src.green.energy import (
    CapabilityResult,
    EnergyBackend,
    ExternalMeterConfig,
    probe_energy_capabilities,
)
from src.green.measurement import (
    CanonicalUnit,
    MeasurementProvenance,
    collect_environment_metadata,
)
from src.pipeline_v07b import (
    AUTHORITATIVE_PRD_PATH,
    DEFAULT_CONFIG_PATH,
    V07BContext,
    load_green_evaluation_config,
    load_v06_benchmark_context,
    verify_v06_source_hashes,
)


V07C_STAGE = "V0.7-C"
V07C_SCHEMA_VERSION = "v0.7-c-energy-capability-1"
DIRECT_ENERGY_AVAILABLE = "DIRECT_ENERGY_AVAILABLE"
DIRECT_ENERGY_UNAVAILABLE = "DIRECT_ENERGY_UNAVAILABLE"
CPU_ONLY_WORKLOAD = "scikit-learn Decision Tree and Logistic Regression on CPU"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "green_evaluation" / "energy_capability.json"
DIRECT_BACKEND_PRIORITY = (
    EnergyBackend.LINUX_RAPL,
    EnergyBackend.WINDOWS_EMI,
    EnergyBackend.EXTERNAL_METER,
    EnergyBackend.NVML,
)
V07B_IMMUTABLE_PATHS = (
    PROJECT_ROOT / "src" / "pipeline_v07b.py",
    PROJECT_ROOT / "tests" / "test_pipeline_v07b.py",
    DEFAULT_CONFIG_PATH,
)
REQUIRED_PROVENANCE_FIELDS = frozenset(
    {
        "backend",
        "backend_version",
        "adapter",
        "device",
        "domain",
        "scope",
        "raw_unit",
        "canonical_unit",
        "permission_status",
        "runtime",
        "validation_timestamp_utc",
    }
)


class V07CError(RuntimeError):
    """Base class for V0.7-C validation failures."""


class V07CIntegrityError(V07CError):
    """Raised when an immutable input changes during capability validation."""


@dataclass(frozen=True)
class DirectEnergyEvidence:
    """Non-experimental preflight evidence required for direct-energy acceptance."""

    measurement_method: str
    device: str
    domain: str
    scope: str
    cumulative_energy: bool
    scientifically_valid_integration: bool
    canonical_energy_unit: str
    counter_stable: bool
    counter_resolution_joules: float
    resolution_sufficient: bool
    idle_energy_joules: float
    load_energy_joules: float
    idle_load_distinguishable: bool
    workload_relevant: bool
    fallback_estimation_used: bool
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        numeric = (
            self.counter_resolution_joules,
            self.idle_energy_joules,
            self.load_energy_joules,
        )
        if any(not math.isfinite(value) or value < 0 for value in numeric):
            raise ValueError("Energy preflight values must be finite and nonnegative.")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurement_method": self.measurement_method,
            "device": self.device,
            "domain": self.domain,
            "scope": self.scope,
            "cumulative_energy": self.cumulative_energy,
            "scientifically_valid_integration": self.scientifically_valid_integration,
            "canonical_energy_unit": self.canonical_energy_unit,
            "counter_stable": self.counter_stable,
            "counter_resolution_joules": self.counter_resolution_joules,
            "resolution_sufficient": self.resolution_sufficient,
            "idle_energy_joules": self.idle_energy_joules,
            "load_energy_joules": self.load_energy_joules,
            "idle_load_distinguishable": self.idle_load_distinguishable,
            "workload_relevant": self.workload_relevant,
            "fallback_estimation_used": self.fallback_estimation_used,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class BackendAssessment:
    backend: EnergyBackend
    capability: CapabilityResult
    evidence: DirectEnergyEvidence | None
    accepted_as_direct_energy: bool
    rejection_reasons: tuple[str, ...]
    reported_measurement_classification: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend.value,
            "capability": self.capability.to_dict(),
            "evidence": None if self.evidence is None else self.evidence.to_dict(),
            "accepted_as_direct_energy": self.accepted_as_direct_energy,
            "rejection_reasons": list(self.rejection_reasons),
            "reported_measurement_classification": self.reported_measurement_classification,
        }


def assess_backend(
    capability: CapabilityResult,
    evidence: DirectEnergyEvidence | None = None,
) -> BackendAssessment:
    """Apply the V0.7-C direct-energy acceptance rule to one capability."""

    backend = capability.backend
    if backend is EnergyBackend.HISTORICAL_ESTIMATE:
        return BackendAssessment(
            backend,
            capability,
            evidence,
            False,
            ("ESTIMATED_ONLY",),
            MeasurementProvenance.ESTIMATED.value,
        )

    reasons: list[str] = []
    if not capability.available:
        reasons.append(f"CAPABILITY_{capability.reason_code}")
    if capability.workload_relevant is not True:
        reasons.append("WORKLOAD_IRRELEVANT")
    if evidence is None:
        if capability.available:
            reasons.append("VALIDATION_EVIDENCE_MISSING")
    else:
        _validate_evidence(capability, evidence, reasons)
    accepted = not reasons
    return BackendAssessment(
        backend,
        capability,
        evidence,
        accepted,
        tuple(dict.fromkeys(reasons)),
        MeasurementProvenance.DIRECT_ENERGY.value if accepted else None,
    )


def _validate_evidence(
    capability: CapabilityResult,
    evidence: DirectEnergyEvidence,
    reasons: list[str],
) -> None:
    if not evidence.measurement_method.strip():
        reasons.append("MEASUREMENT_METHOD_MISSING")
    if not evidence.device.strip():
        reasons.append("DEVICE_MISSING")
    if not evidence.domain.strip():
        reasons.append("DOMAIN_MISSING")
    if not evidence.scope.strip():
        reasons.append("SCOPE_MISSING")
    if evidence.canonical_energy_unit != CanonicalUnit.JOULES.value:
        reasons.append("NONCANONICAL_ENERGY_UNIT")
    if not (
        capability.supports_cumulative_energy
        and evidence.cumulative_energy
        or capability.supports_instantaneous_power
        and evidence.scientifically_valid_integration
    ):
        reasons.append("NO_CUMULATIVE_OR_VALID_INTEGRATION")
    if not evidence.counter_stable:
        reasons.append("COUNTER_BEHAVIOR_UNSTABLE")
    if evidence.counter_resolution_joules <= 0 or not evidence.resolution_sufficient:
        reasons.append("COUNTER_RESOLUTION_INSUFFICIENT")
    if (
        not evidence.idle_load_distinguishable
        or evidence.load_energy_joules
        <= evidence.idle_energy_joules + evidence.counter_resolution_joules
    ):
        reasons.append("IDLE_LOAD_NOT_DISTINGUISHABLE")
    if capability.workload_relevant is not True or not evidence.workload_relevant:
        reasons.append("WORKLOAD_IRRELEVANT")
    if evidence.fallback_estimation_used:
        reasons.append("FALLBACK_ESTIMATION_USED")
    missing = sorted(
        key
        for key in REQUIRED_PROVENANCE_FIELDS
        if key not in evidence.provenance
        or evidence.provenance[key] is None
        or not str(evidence.provenance[key]).strip()
    )
    if missing:
        reasons.append("PROVENANCE_INCOMPLETE:" + ",".join(missing))
    else:
        expected = {
            "backend": capability.backend.value,
            "device": evidence.device,
            "domain": evidence.domain,
            "scope": evidence.scope,
            "canonical_unit": CanonicalUnit.JOULES.value,
        }
        if any(str(evidence.provenance[key]) != value for key, value in expected.items()):
            reasons.append("PROVENANCE_INCONSISTENT")


def evaluate_capability_gate(
    capabilities: Sequence[CapabilityResult],
    evidence_by_backend: Mapping[EnergyBackend | str, DirectEnergyEvidence] | None = None,
) -> tuple[str, tuple[BackendAssessment, ...], EnergyBackend | None]:
    """Evaluate every required backend and return one feasibility decision."""

    by_backend = {capability.backend: capability for capability in capabilities}
    required = set(EnergyBackend)
    if set(by_backend) != required or len(capabilities) != len(required):
        raise V07CError("Capability gate requires each backend exactly once.")
    evidence = {
        EnergyBackend(key): value for key, value in (evidence_by_backend or {}).items()
    }
    assessments = tuple(
        assess_backend(by_backend[backend], evidence.get(backend)) for backend in EnergyBackend
    )
    accepted = {item.backend for item in assessments if item.accepted_as_direct_energy}
    selected = next((backend for backend in DIRECT_BACKEND_PRIORITY if backend in accepted), None)
    decision = DIRECT_ENERGY_AVAILABLE if selected is not None else DIRECT_ENERGY_UNAVAILABLE
    return decision, assessments, selected


def run_v07c_feasibility_gate(
    *,
    output_path: Path | str | None = DEFAULT_OUTPUT_PATH,
    rapl_root: Path | str = Path("/sys/class/powercap"),
    external_meter_config: ExternalMeterConfig | None = None,
    evidence_by_backend: Mapping[EnergyBackend | str, DirectEnergyEvidence] | None = None,
    v06_context: V07BContext | None = None,
) -> dict[str, Any]:
    """Probe the current runtime, verify immutable inputs, and record the decision."""

    config = load_green_evaluation_config(DEFAULT_CONFIG_PATH)
    if Path(config["authoritative_prd"]) != Path("docs/PhD_PRD.md"):
        raise V07CIntegrityError("The authoritative PRD must remain docs/PhD_PRD.md.")
    context = load_v06_benchmark_context() if v06_context is None else v06_context
    v06_before = dict(context.plan.source_hashes)
    v07b_before = snapshot_file_hashes(V07B_IMMUTABLE_PATHS)

    environment = collect_environment_metadata(PROJECT_ROOT)
    capabilities = probe_energy_capabilities(
        rapl_root=rapl_root,
        workload_uses_gpu=False,
        external_meter_config=external_meter_config,
    )
    decision, assessments, selected = evaluate_capability_gate(
        capabilities, evidence_by_backend
    )

    verify_v06_source_hashes(context.plan)
    v06_after = snapshot_file_hashes(tuple(Path(path) for path in v06_before))
    if v06_after != v06_before:
        raise V07CIntegrityError("V0.6 immutable source or artifact hashes changed.")
    v07b_after = snapshot_file_hashes(V07B_IMMUTABLE_PATHS)
    if v07b_after != v07b_before:
        raise V07CIntegrityError("V0.7-B implementation artifacts changed.")

    report = {
        "schema_version": V07C_SCHEMA_VERSION,
        "stage": V07C_STAGE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_prd": str(AUTHORITATIVE_PRD_PATH.relative_to(PROJECT_ROOT)),
        "current_runtime": environment,
        "workload": {
            "description": CPU_ONLY_WORKLOAD,
            "uses_cpu": True,
            "uses_gpu": False,
            "gpu_energy_is_total_workload_energy": False,
        },
        "classification_policy": {
            "electrical_energy": MeasurementProvenance.DIRECT_ENERGY.value,
            "historical_tdp_time": MeasurementProvenance.ESTIMATED.value,
            "timing_cpu_rss_and_bytes": MeasurementProvenance.DIRECT_COMPUTATIONAL.value,
            "estimated_energy_may_be_direct": False,
            "instantaneous_watts_are_joules": False,
        },
        "backends": [assessment.to_dict() for assessment in assessments],
        "accepted_direct_energy_backend": None if selected is None else selected.value,
        "direct_energy_decision": decision,
        "carbon": {
            "claim_reported": False,
            "claim_supported": False,
            "reason": (
                "Direct energy is unavailable; no carbon claim is scientifically supported."
                if selected is None
                else "V0.7-C validates energy capability only and has no grid-intensity evidence."
            ),
        },
        "integrity": {
            "v06_hashes_unchanged": True,
            "v06_verified_file_count": len(v06_before),
            "v07b_artifacts_unchanged": True,
            "v07b_artifact_hashes": _display_hashes(v07b_after),
        },
        "experiments_executed": False,
        "final_observations_executed": 0,
        "recommendation_for_v07d": (
            "Proceed with Tier 1 DIRECT_COMPUTATIONAL efficiency evaluation only."
            if selected is None
            else f"Tier 2 may use only the accepted {selected.value} backend under its validated scope."
        ),
    }
    if output_path is not None:
        write_capability_report(report, output_path)
    return report


def snapshot_file_hashes(paths: Sequence[Path | str]) -> dict[str, str]:
    """Hash required files without modifying them."""

    hashes: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise V07CIntegrityError(f"Required immutable file is missing: {path}")
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise V07CIntegrityError(f"Cannot hash immutable file: {path}") from exc
        hashes[str(path)] = digest.hexdigest()
    return hashes


def _display_hashes(hashes: Mapping[str, str]) -> dict[str, str]:
    displayed: dict[str, str] = {}
    for raw_path, digest in hashes.items():
        path = Path(raw_path)
        try:
            name = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            name = str(path)
        displayed[name] = digest
    return displayed


def write_capability_report(report: Mapping[str, Any], path: Path | str) -> None:
    """Atomically write the non-experimental capability record."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    payload = json.dumps(dict(report), indent=2, sort_keys=True) + "\n"
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(destination)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise V07CError(f"Cannot write capability report: {destination}") from exc


def main() -> int:
    report = run_v07c_feasibility_gate()
    print(report["direct_energy_decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
