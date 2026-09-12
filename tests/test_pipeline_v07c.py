"""Focused, non-experimental tests for the V0.7-C energy feasibility gate."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from src.green.energy import (
    CapabilityResult,
    EnergyBackend,
    ExternalMeterConfig,
    aggregate_selected_rapl_delta_joules,
    discover_rapl_domains,
    instantaneous_watts_to_joules,
    probe_external_meter,
    probe_historical_estimate,
    probe_linux_rapl,
    probe_nvml,
    probe_windows_emi,
    rapl_energy_delta_joules,
)
from src.green.measurement import CanonicalUnit, MeasurementProvenance
import src.pipeline_v07b as v07b
import src.pipeline_v07c as v07c


def _make_rapl_domain(
    root: Path,
    relative: str,
    *,
    name: str,
    energy_uj: int,
    max_energy_range_uj: int = 1_000_000,
) -> None:
    domain = root / relative
    domain.mkdir(parents=True)
    (domain / "name").write_text(name, encoding="utf-8")
    (domain / "energy_uj").write_text(str(energy_uj), encoding="ascii")
    (domain / "max_energy_range_uj").write_text(
        str(max_energy_range_uj), encoding="ascii"
    )


def _unavailable(backend: EnergyBackend, reason: str = "SYNTHETIC_UNAVAILABLE") -> CapabilityResult:
    return CapabilityResult(backend, False, reason, "Synthetic unavailable capability.")


def _direct_capability(
    backend: EnergyBackend,
    *,
    cumulative: bool = True,
    instantaneous: bool = False,
    relevant: bool = True,
) -> CapabilityResult:
    return CapabilityResult(
        backend,
        True,
        "AVAILABLE",
        "Synthetic direct-energy capability.",
        supports_cumulative_energy=cumulative,
        supports_instantaneous_power=instantaneous,
        cumulative_energy_unit="microjoules" if cumulative else None,
        instantaneous_power_unit="milliwatts" if instantaneous else None,
        workload_relevant=relevant,
    )


def _evidence(
    backend: EnergyBackend,
    *,
    device: str = "synthetic-device",
    domain: str = "package-0",
    scope: str = "cpu_package",
    cumulative: bool = True,
    integration: bool = False,
    unit: str = "joules",
    stable: bool = True,
    resolution: float = 0.001,
    resolution_sufficient: bool = True,
    idle: float = 1.0,
    load: float = 2.0,
    distinguishable: bool = True,
    relevant: bool = True,
    fallback: bool = False,
    provenance_changes: dict[str, Any] | None = None,
) -> v07c.DirectEnergyEvidence:
    provenance = {
        "backend": backend.value,
        "backend_version": "synthetic-1",
        "adapter": "synthetic-adapter",
        "device": device,
        "domain": domain,
        "scope": scope,
        "raw_unit": "microjoules",
        "canonical_unit": "joules",
        "permission_status": "readable",
        "runtime": "synthetic-native",
        "validation_timestamp_utc": "2026-09-12T00:00:00+00:00",
    }
    if provenance_changes:
        provenance.update(provenance_changes)
    return v07c.DirectEnergyEvidence(
        measurement_method="cumulative_hardware_counter",
        device=device,
        domain=domain,
        scope=scope,
        cumulative_energy=cumulative,
        scientifically_valid_integration=integration,
        canonical_energy_unit=unit,
        counter_stable=stable,
        counter_resolution_joules=resolution,
        resolution_sufficient=resolution_sufficient,
        idle_energy_joules=idle,
        load_energy_joules=load,
        idle_load_distinguishable=distinguishable,
        workload_relevant=relevant,
        fallback_estimation_used=fallback,
        provenance=provenance,
    )


def _complete_capabilities(accepted: CapabilityResult | None = None) -> tuple[CapabilityResult, ...]:
    capabilities = {backend: _unavailable(backend) for backend in EnergyBackend}
    capabilities[EnergyBackend.HISTORICAL_ESTIMATE] = probe_historical_estimate()
    if accepted is not None:
        capabilities[accepted.backend] = accepted
    return tuple(capabilities[backend] for backend in EnergyBackend)


@pytest.fixture(scope="module")
def repository_context() -> v07b.V07BContext:
    return v07b.load_v06_benchmark_context()


@pytest.fixture(scope="module")
def repository_report(repository_context: v07b.V07BContext) -> dict[str, Any]:
    return v07c.run_v07c_feasibility_gate(
        output_path=None,
        v06_context=repository_context,
    )


def test_01_rapl_unavailable_behavior(tmp_path: Path) -> None:
    capability = probe_linux_rapl(tmp_path / "missing")
    assessment = v07c.assess_backend(capability)
    assert capability.available is False
    assert capability.reason_code == "POWERCAP_PATH_UNAVAILABLE"
    assert assessment.accepted_as_direct_energy is False
    assert "CAPABILITY_POWERCAP_PATH_UNAVAILABLE" in assessment.rejection_reasons


def test_02_rapl_valid_capability_with_synthetic_counter(tmp_path: Path) -> None:
    _make_rapl_domain(
        tmp_path, "intel-rapl:0", name="package-0", energy_uj=100_000
    )
    capability = probe_linux_rapl(tmp_path)
    domains = discover_rapl_domains(tmp_path)
    delta = aggregate_selected_rapl_delta_joules(
        domains,
        ("intel-rapl:0",),
        {"intel-rapl:0": 100_000},
        {"intel-rapl:0": 350_000},
    )
    assessment = v07c.assess_backend(
        capability,
        _evidence(
            EnergyBackend.LINUX_RAPL,
            device="cpu-0",
            domain="intel-rapl:0",
        ),
    )
    assert capability.available is True
    assert capability.metadata["domains"][0]["name"] == "package-0"
    assert capability.metadata["domains"][0]["max_energy_range_uj"] == 1_000_000
    assert delta["total_joules"] == pytest.approx(0.25)
    assert assessment.accepted_as_direct_energy is True


def test_03_rapl_wraparound() -> None:
    assert rapl_energy_delta_joules(900_000, 100_000, 1_000_000) == pytest.approx(0.2)


def test_04_overlapping_domains_rejected(tmp_path: Path) -> None:
    _make_rapl_domain(tmp_path, "intel-rapl:0", name="package-0", energy_uj=10)
    _make_rapl_domain(
        tmp_path,
        "intel-rapl:0/intel-rapl:0:0",
        name="core",
        energy_uj=5,
    )
    domains = discover_rapl_domains(tmp_path)
    ids = tuple(domain.domain_id for domain in domains)
    with pytest.raises(ValueError, match="Overlapping RAPL"):
        aggregate_selected_rapl_delta_joules(
            domains,
            ids,
            {domain_id: 1 for domain_id in ids},
            {domain_id: 2 for domain_id in ids},
        )


def test_05_wsl_is_not_native_emi() -> None:
    capability = probe_windows_emi(system_name="Linux", wsl=True)
    assert capability.available is False
    assert capability.reason_code == "WSL_NOT_NATIVE_WINDOWS"


def test_06_emi_unavailable_behavior() -> None:
    capability = probe_windows_emi(system_name="Windows", wsl=False)
    assert capability.available is False
    assert capability.reason_code == "NATIVE_EMI_UNVERIFIED"


def test_07_nvml_unavailable_behavior() -> None:
    def missing(_: str) -> Any:
        raise ModuleNotFoundError("synthetic missing pynvml")

    capability = probe_nvml(module_loader=missing)
    assert capability.available is False
    assert capability.reason_code == "OPTIONAL_DEPENDENCY_MISSING"


def test_08_watts_are_not_treated_as_joules() -> None:
    with pytest.raises(ValueError, match="not cumulative joules"):
        instantaneous_watts_to_joules(5.0)
    assessment = v07c.assess_backend(
        _direct_capability(EnergyBackend.NVML, cumulative=False, instantaneous=True),
        _evidence(EnergyBackend.NVML, cumulative=False, integration=False),
    )
    assert "NO_CUMULATIVE_OR_VALID_INTEGRATION" in assessment.rejection_reasons


def test_09_cumulative_energy_distinguished_from_instantaneous_power() -> None:
    class SyntheticNvml:
        def nvmlInit(self) -> None: pass
        def nvmlShutdown(self) -> None: pass
        def nvmlDeviceGetCount(self) -> int: return 1
        def nvmlDeviceGetHandleByIndex(self, _: int) -> int: return 0
        def nvmlDeviceGetName(self, _: int) -> str: return "synthetic GPU"
        def nvmlDeviceGetPowerUsage(self, _: int) -> int: return 5_000
        def nvmlDeviceGetTotalEnergyConsumption(self, _: int) -> int: return 12_000

    capability = probe_nvml(
        workload_uses_gpu=True, module_loader=lambda _: SyntheticNvml()
    )
    assert capability.supports_instantaneous_power is True
    assert capability.instantaneous_power_unit == "milliwatts"
    assert capability.supports_cumulative_energy is True
    assert capability.cumulative_energy_unit == "millijoules"
    assert capability.metadata["instantaneous_power_is_energy"] is False


def test_10_cpu_only_workload_rejects_gpu_only_total_energy() -> None:
    capability = _direct_capability(EnergyBackend.NVML, relevant=False)
    assessment = v07c.assess_backend(
        capability, _evidence(EnergyBackend.NVML, relevant=False)
    )
    assert assessment.accepted_as_direct_energy is False
    assert "WORKLOAD_IRRELEVANT" in assessment.rejection_reasons


def test_11_external_meter_unavailable() -> None:
    capability = probe_external_meter(ExternalMeterConfig("unverified meter"))
    assert capability.available is False
    assert capability.reason_code == "INTERFACE_ONLY"


def test_12_estimated_energy_remains_estimated() -> None:
    assessment = v07c.assess_backend(probe_historical_estimate())
    assert assessment.reported_measurement_classification == "ESTIMATED"
    assert assessment.accepted_as_direct_energy is False


def test_13_estimated_energy_cannot_become_direct_energy() -> None:
    assessment = v07c.assess_backend(
        probe_historical_estimate(),
        _evidence(EnergyBackend.HISTORICAL_ESTIMATE),
    )
    assert assessment.rejection_reasons == ("ESTIMATED_ONLY",)
    assert assessment.reported_measurement_classification != "DIRECT_ENERGY"


def test_14_backend_acceptance_gate() -> None:
    rapl = _direct_capability(EnergyBackend.LINUX_RAPL)
    decision, assessments, selected = v07c.evaluate_capability_gate(
        _complete_capabilities(rapl),
        {EnergyBackend.LINUX_RAPL: _evidence(EnergyBackend.LINUX_RAPL)},
    )
    assert decision == v07c.DIRECT_ENERGY_AVAILABLE
    assert selected is EnergyBackend.LINUX_RAPL
    assert sum(item.accepted_as_direct_energy for item in assessments) == 1


def test_15_backend_rejection_reasons_are_structured() -> None:
    evidence = _evidence(
        EnergyBackend.LINUX_RAPL,
        stable=False,
        resolution_sufficient=False,
        distinguishable=False,
        fallback=True,
    )
    assessment = v07c.assess_backend(
        _direct_capability(EnergyBackend.LINUX_RAPL), evidence
    )
    assert {
        "COUNTER_BEHAVIOR_UNSTABLE",
        "COUNTER_RESOLUTION_INSUFFICIENT",
        "IDLE_LOAD_NOT_DISTINGUISHABLE",
        "FALLBACK_ESTIMATION_USED",
    }.issubset(assessment.rejection_reasons)


def test_16_complete_provenance_required() -> None:
    evidence = _evidence(
        EnergyBackend.LINUX_RAPL,
        provenance_changes={"backend_version": ""},
    )
    assessment = v07c.assess_backend(
        _direct_capability(EnergyBackend.LINUX_RAPL), evidence
    )
    assert any(
        reason.startswith("PROVENANCE_INCOMPLETE:backend_version")
        for reason in assessment.rejection_reasons
    )


def test_17_canonical_energy_units_required() -> None:
    assessment = v07c.assess_backend(
        _direct_capability(EnergyBackend.LINUX_RAPL),
        _evidence(EnergyBackend.LINUX_RAPL, unit=CanonicalUnit.WATTS.value),
    )
    assert "NONCANONICAL_ENERGY_UNIT" in assessment.rejection_reasons
    assert MeasurementProvenance.DIRECT_ENERGY.value != MeasurementProvenance.DIRECT_COMPUTATIONAL.value


def test_18_no_carbon_claim_in_capability_stage(repository_report: dict[str, Any]) -> None:
    assert repository_report["carbon"]["claim_reported"] is False
    if repository_report["direct_energy_decision"] == v07c.DIRECT_ENERGY_UNAVAILABLE:
        assert repository_report["carbon"]["claim_supported"] is False


def test_19_v06_hashes_unchanged(repository_context: v07b.V07BContext) -> None:
    expected = dict(repository_context.plan.source_hashes)
    v07b.verify_v06_source_hashes(repository_context.plan)
    observed = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in expected
    }
    assert observed == expected


def test_20_v07b_artifacts_unchanged(repository_report: dict[str, Any]) -> None:
    integrity = repository_report["integrity"]
    assert integrity["v07b_artifacts_unchanged"] is True
    assert set(integrity["v07b_artifact_hashes"]) == {
        "src/pipeline_v07b.py",
        "tests/test_pipeline_v07b.py",
        "config/green_evaluation.yaml",
    }


def test_21_current_gate_has_exactly_one_decision(repository_report: dict[str, Any]) -> None:
    assert repository_report["direct_energy_decision"] in {
        v07c.DIRECT_ENERGY_AVAILABLE,
        v07c.DIRECT_ENERGY_UNAVAILABLE,
    }
    assert repository_report["experiments_executed"] is False
    assert repository_report["final_observations_executed"] == 0
