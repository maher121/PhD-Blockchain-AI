"""Artifact and claim-boundary tests for the V0.7-F research outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "v07_green_evaluation.ipynb"
REPORT_PATH = ROOT / "docs" / "v07_green_evaluation_report.md"
RESULTS = ROOT / "results" / "green_evaluation"


@pytest.fixture(scope="module")
def notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report() -> str:
    return REPORT_PATH.read_text(encoding="utf-8")


def test_01_v07f_artifacts_exist() -> None:
    assert NOTEBOOK_PATH.is_file()
    assert REPORT_PATH.is_file()


def test_02_notebook_is_valid_nbformat_v4(notebook) -> None:
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"


def test_03_every_code_cell_was_executed_without_error(notebook) -> None:
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) >= 8
    execution_counts = [cell["execution_count"] for cell in code_cells]
    assert all(isinstance(count, int) for count in execution_counts)
    assert execution_counts == sorted(execution_counts)
    assert all(
        output.get("output_type") != "error"
        for cell in code_cells
        for output in cell["outputs"]
    )


def test_04_notebook_completed_and_verified_hashes(notebook) -> None:
    output_text = "\n".join(
        str(value)
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        for value in output.get("text", [])
    )
    assert "V0.7-E manifest entries verified: 21" in output_text
    assert "V0.7-F artifact-only notebook completed without rerunning experiments." in output_text


def test_05_notebook_code_is_artifact_only(notebook) -> None:
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    prohibited = (
        ".fit(",
        "run_v07",
        "prepare_verified_workloads",
        "train_model",
        "attack_generator",
        "data/raw",
        ".to_csv(",
        ".to_json(",
        ".write_text(",
        ".write_bytes(",
        ".savefig(",
    )
    assert not [token for token in prohibited if token in code]


def test_06_notebook_reads_required_frozen_evidence(notebook) -> None:
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    required_names = {
        "energy_capability.json",
        "environment_metadata.json",
        "v07d_run_manifest.json",
        "v07d_governance_audit.json",
        "computational_observations.csv",
        "v07e_seed_summaries.csv",
        "v07e_paired_comparisons.csv",
        "v07e_efficiency_summary.csv",
        "v07e_performance_efficiency.csv",
        "v07e_variability_analysis.csv",
        "v07e_pareto_analysis.csv",
        "v07e_governance_audit.json",
        "v07e_artifact_hashes.json",
    }
    assert not [name for name in required_names if name not in source]


def test_07_v07e_hash_manifest_is_intact() -> None:
    manifest = json.loads((RESULTS / "v07e_artifact_hashes.json").read_text(encoding="utf-8"))
    assert len(manifest) == 21
    for relative_path, expected_digest in manifest.items():
        artifact = RESULTS / relative_path
        assert artifact.is_file(), relative_path
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected_digest


def test_08_governance_and_observation_accounting_are_complete() -> None:
    run_manifest = json.loads((RESULTS / "v07d_run_manifest.json").read_text(encoding="utf-8"))
    governance_d = json.loads((RESULTS / "v07d_governance_audit.json").read_text(encoding="utf-8"))
    governance_e = json.loads((RESULTS / "v07e_governance_audit.json").read_text(encoding="utf-8"))
    with (RESULTS / "computational_observations.csv").open(encoding="utf-8", newline="") as handle:
        observations = list(csv.DictReader(handle))

    assert len(observations) == 600
    assert run_manifest["accounting"]["successful_total_observations"] == 600
    assert run_manifest["accounting"]["failed_total_observations"] == 0
    assert run_manifest["accounting"]["run_complete"] is True
    assert governance_d["status"] == governance_e["status"] == "PASS"
    assert all(check["passed"] for audit in (governance_d, governance_e) for check in audit["checks"])


def test_09_report_has_required_research_sections(report) -> None:
    required_sections = (
        "## Abstract",
        "## 1. Research Question and Scope",
        "## 3. Frozen Experimental Design",
        "## 4. Measurement and Statistical Protocol",
        "## 5. Direct-Energy Feasibility",
        "## 6. Frozen Predictive Performance",
        "## 7. MI K=11 Computational Results",
        "## 9. Measurement Variability",
        "## 10. Performance-Efficiency and Pareto Analysis",
        "## 11. Hypothesis Assessment",
        "## 15. Limitations",
        "## 16. Conclusions",
        "## 18. Reproducibility and Artifact Provenance",
    )
    assert not [section for section in required_sections if section not in report]


def test_10_report_records_core_frozen_facts(report) -> None:
    required_facts = (
        "600/600",
        "74.4186%",
        "81.81%",
        "33 of 36",
        "DIRECT_ENERGY_UNAVAILABLE",
        "zero failures",
        "five paired seed means",
        "No objective weights or composite score were used",
    )
    assert not [fact for fact in required_facts if fact not in report]


def test_11_report_preserves_energy_and_claim_boundaries(report) -> None:
    assert "No direct electrical-energy backend passed preflight" in report
    assert "not measured joules" in report
    assert "No accepted backend, joule result, watt result, carbon result, or CO2e result exists" in report
    assert "They are not claims of statistical significance" in report
    assert "does not support production readiness" in report


def test_12_report_values_match_frozen_tables(report) -> None:
    with (RESULTS / "v07e_performance_efficiency.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mi_rows = [row for row in rows if row["configuration_id"] == "mutual_information_select_k_best_k11"]
    assert len(mi_rows) == 2
    assert {round(float(row["feature_reduction_percent"]), 4) for row in mi_rows} == {74.4186}
    assert all(row["predictive_performance_preserved"] == "True" for row in mi_rows)

    efficiency = json.loads((RESULTS / "v07e_efficiency_summary.json").read_text(encoding="utf-8"))
    lr_training = next(
        row
        for row in efficiency
        if row["classifier"] == "logistic_regression"
        and row["configuration_id"] == "mutual_information_select_k_best_k11"
        and row["phase"] == "training"
        and row["metric"] == "wall_time_sec"
    )
    assert lr_training["beneficial_change_percent"] == pytest.approx(81.812302, abs=1e-6)
    assert lr_training["interpretation"] == "clear_reduction"
    assert "81.81%" in report


def test_13_all_report_figure_links_resolve(report) -> None:
    links = re.findall(r"!\[[^]]*\]\(([^)]+)\)", report)
    assert len(links) == 8
    for link in links:
        assert (REPORT_PATH.parent / link).resolve().is_file(), link


def test_14_report_and_notebook_reference_each_other(report, notebook) -> None:
    notebook_source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "notebooks/v07_green_evaluation.ipynb" in report
    assert "docs/v07_green_evaluation_report.md" in notebook_source
