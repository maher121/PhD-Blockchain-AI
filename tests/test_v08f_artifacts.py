"""Artifact checks for V0.8-F notebook and scientific synthesis outputs."""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "v08_bpso_feature_selection_evaluation.ipynb"
REPORT_PATH = ROOT / "docs" / "v08_bpso_evaluation_report.md"
TABLE_DIR = ROOT / "results" / "bpso" / "v08f_reporting" / "tables"
FIGURE_DIR = ROOT / "results" / "bpso" / "v08f_reporting" / "figures"


@pytest.fixture(scope="module")
def notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report_text() -> str:
    return REPORT_PATH.read_text(encoding="utf-8")


def test_01_v08f_artifacts_exist() -> None:
    assert NOTEBOOK_PATH.is_file()
    assert REPORT_PATH.is_file()
    assert TABLE_DIR.is_dir()
    assert FIGURE_DIR.is_dir()


def test_02_notebook_is_valid_and_executed(notebook: dict[str, object]) -> None:
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) >= 10
    execution_counts = [cell["execution_count"] for cell in code_cells]
    assert all(isinstance(count, int) for count in execution_counts)
    assert execution_counts == sorted(execution_counts)
    assert all(
        output.get("output_type") != "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )


def test_03_notebook_is_artifact_only(notebook: dict[str, object]) -> None:
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    prohibited = (
        ".fit(",
        "run_v08c(",
        "run_v08d(",
        "run_e4_analysis(",
        "prepare_verified_workloads",
        "resource_benchmark_executed = True",
    )
    assert not [token for token in prohibited if token in source]


def test_04_notebook_outputs_include_completion_marker(notebook: dict[str, object]) -> None:
    output_text = "\n".join(
        str(value)
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        for value in output.get("text", [])
    )
    assert "V0.8-F artifact-only notebook completed without rerunning experiments." in output_text


def test_05_generated_table_and_figure_counts() -> None:
    assert len(list(TABLE_DIR.glob("*.csv"))) == 7
    assert len(list(FIGURE_DIR.glob("*.png"))) == 10
    assert len(list(FIGURE_DIR.glob("*.pdf"))) == 10


def test_06_report_contains_required_claim_boundaries(report_text: str) -> None:
    assert "STRUCTURAL_GAIN_TIMING_UNCERTAIN" in report_text
    assert "DIRECT_ENERGY_UNAVAILABLE" in report_text
    assert "indicative rather than definitive" in report_text
    assert "notebooks/v08_bpso_feature_selection_evaluation.ipynb" in report_text


def test_07_report_figure_links_resolve(report_text: str) -> None:
    links = re.findall(r"!\[[^]]*\]\(([^)]+)\)", report_text)
    assert len(links) == 10
    for link in links:
        assert (REPORT_PATH.parent / link).resolve().is_file(), link
