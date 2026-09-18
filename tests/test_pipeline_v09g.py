"""Focused V0.9-G tests: artifact-only reporting, notebook, report, and reporting lock.

Design: V0.9-G performs presentation/synthesis only. The suite proves preflight
governance, import/execution guards, notebook top-to-bottom execution
(byte-stable), exports, report disclaimers, prior locks and reporting-lock
create/verify. No optimizer, final-test or resource campaign is ever launched.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re

import nbformat
from nbclient import NotebookClient
import pandas as pd
import pytest

import src.pipeline_v09g as v09g


ROOT = Path(__file__).resolve().parents[1]
REPORT_TEXT = (ROOT / "docs" / "v09_bgwo_evaluation_report.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def artifacts() -> dict[str, object]:
    return v09g.load_v09g_artifacts(ROOT, expected_head=v09g.EXPECTED_HEAD_SHORT)


def test_preflight_import_guards() -> None:
    """V0.9-G must not import optimizer/measurement/final-test/resource engines."""
    source = (ROOT / "src" / "pipeline_v09g.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = {
        "src.optimization",
        "src.measurement",
        "battery",
        "jtop",
        "psutil",
        "nbformat",
        "nbclient",
        "jupyter_client",
    }
    for name in imported:
        assert not name.startswith("src.pipeline_v09"), name
        for banned in forbidden:
            assert not name.startswith(banned), f"forbidden import in pipeline_v09g: {name}"


def test_preflight_passes_with_expected_head() -> None:
    result = v09g.verify_v09g_preflight(root=ROOT, expected_head=v09g.EXPECTED_HEAD_SHORT)
    assert result["status"] == "GO"
    assert result["head_short"] == v09g.EXPECTED_HEAD_SHORT
    assert result["frozen_evidence_file_count"] == len(v09g.SOURCE_EVIDENCE_PATHS)
    assert result["direct_energy_status"] == "DIRECT_ENERGY_UNAVAILABLE"
    assert result["v09d_winner_semantic_lock_sha256"] == v09g.EXPECTED_BGWO_SEMANTIC
    assert result["v09e_result_semantic_lock_sha256"] == v09g.EXPECTED_V09E_RESULT_SEMANTIC
    assert result["v09f_result_semantic_lock_sha256"] == v09g.EXPECTED_V09F_RESULT_SEMANTIC
    assert result["bpso_winner_semantic_lock_sha256"] == v09g.EXPECTED_BPSO_SEMANTIC
    assert all(result["checks"].values())


def test_preflight_rejects_wrong_head() -> None:
    with pytest.raises(v09g.V09GError):
        v09g.verify_v09g_preflight(root=ROOT, expected_head="deadbee")


def test_prior_stage_locks_have_no_drift() -> None:
    imports = {
        "results/feature_selection/validation_lock.json": None,
        "results/bpso/v08c_winner_lock.json": None,
        "results/bgwo/v09d/v09d_winner_lock.json": None,
        "results/bgwo/v09e/v09e_result_lock.json": None,
        "results/bgwo/v09f/v09f_result_lock.json": None,
    }
    for relative in imports:
        assert (ROOT / relative).is_file(), relative
    bgwo = (ROOT / "results/bgwo/v09d/v09d_winner_lock.json").read_bytes()
    assert hashlib.sha256(bgwo).hexdigest() == v09g.snapshot_evidence(ROOT)["results/bgwo/v09d/v09d_winner_lock.json"]


def test_reporting_tables_built_from_frozen_evidence(artifacts: dict[str, object]) -> None:
    tables = v09g.build_reporting_tables(artifacts)
    assert set(tables) == set(v09g.TABLE_FILENAMES)

    win = tables["bgwo_validation_runs"]
    assert len(win) == 6
    row = win.loc[win["optimizer_seed"].astype(str) == "2042"].iloc[0]
    assert int(row["selected_feature_count"]) == 14
    assert float(row["validation_average_precision"]) == pytest.approx(0.2346177845086236)
    assert float(row["validation_f1"]) == pytest.approx(0.1896439632931008)
    assert float(row["validation_recall"]) == pytest.approx(0.3226666666666666)
    agg = win.loc[win["optimizer_seed"].astype(str) == "ALL"].iloc[0]
    assert int(agg["candidate_requests"]) == 768
    assert int(agg["decision_tree_fits"]) == 3840
    assert agg["stop_reason"] == "AGGREGATE_WINNER"

    full_ap = tables["feature_configurations"]
    assert len(full_ap) == 5
    bgwo_cfg = full_ap.loc[full_ap["configuration_id"] == "BGWO"].iloc[0]
    assert int(bgwo_cfg["feature_count"]) == 14

    dt = tables["final_test_dt"]
    bg = dt.loc[dt["configuration_id"] == "BGWO"].iloc[0]
    assert float(bg["mean_average_precision"]) == pytest.approx(0.2260346729225158)
    assert float(bg["mean_f1"]) == pytest.approx(0.179716812321526)
    assert float(bg["mean_recall"]) == pytest.approx(0.308)

    lr = tables["final_test_lr"]
    bg = lr.loc[lr["configuration_id"] == "BGWO"].iloc[0]
    assert float(bg["mean_average_precision"]) == pytest.approx(0.0508101550047126)
    assert float(bg["mean_recall"]) == pytest.approx(0.4679999999999999)

    paired = tables["paired_statistics"]
    assert len(paired) == 30
    row = paired[
        (paired["classifier"] == "decision_tree")
        & (paired["metric"] == "average_precision")
        & (paired["reference_configuration_id"] == "BPSO-K10")
    ].iloc[0]
    assert int(row["n"]) == 5 and int(row["df"]) == 4
    assert float(row["mean_difference"]) == pytest.approx(0.0064231574514364)
    assert float(row["ci95_low"]) == pytest.approx(0.0013098820602024)
    assert float(row["ci95_high"]) == pytest.approx(0.0115364328426705)
    assert row["direction_uncertain"] is not True and bool(row["direction_uncertain"]) is False

    overlap = tables["feature_overlap"]
    assert len(overlap) == 2
    assert float(overlap.iloc[0]["jaccard"]) == pytest.approx(0.1428571428571428)
    assert float(overlap.iloc[1]["jaccard"]) == pytest.approx(0.3478260869565217)

    resource = tables["resource_efficiency"]
    assert len(resource) == 10
    assert "feature_count" in resource.columns
    assert "serialized_model_bytes" in resource.columns
    row = resource[
        (resource["classifier"] == "decision_tree") & (resource["configuration_id"] == "BGWO")
    ].iloc[0]
    assert row["feature_count"] == 14
    assert float(row["serialized_model_bytes"]) == 3812.0

    overhead = tables["optimizer_overhead"]
    assert len(overhead) == 2
    bgwo_o = overhead.loc[overhead["optimizer"] == "BGWO"].iloc[0]
    assert float(bgwo_o["optimizer_wall_time_sec"]) == pytest.approx(325.42544312802784)
    assert int(bgwo_o["candidate_requests"]) == 768

    tradeoff = tables["bpso_vs_bgwo"]
    assert len(tradeoff) == 18


def test_write_tables_and_render_figures(artifacts: dict[str, object], tmp_path: Path) -> None:
    tables = v09g.build_reporting_tables(artifacts)
    written = v09g.write_reporting_tables(tables, table_dir=tmp_path / "tables")
    assert len(written) == 10
    assert all(path.is_file() for path in written.values())

    rendered = v09g.render_reporting_figures(artifacts, figure_dir=tmp_path / "figures")
    assert set(rendered) == set(v09g.FIGURE_NAMES)
    for outputs in rendered.values():
        assert outputs["png"].is_file()
        assert outputs["pdf"].is_file()


def test_build_outputs_end_to_end(tmp_path: Path) -> None:
    output = v09g.build_v09g_outputs(
        root=ROOT,
        expected_head=v09g.EXPECTED_HEAD_SHORT,
        output_dir=tmp_path / "v09g_reporting",
    )
    assert output["status"] == "COMPLETED"
    assert output["table_count"] == 10
    assert output["figure_count"] == 12
    table_dir = Path(output["table_dir"])
    figure_dir = Path(output["figure_dir"])
    assert len(list(table_dir.glob("*.csv"))) == 10
    assert len(list(figure_dir.glob("*.png"))) == 12
    assert len(list(figure_dir.glob("*.pdf"))) == 12


def test_reporting_outputs_exist_at_lock_path() -> None:
    table_dir = v09g.DEFAULT_OUTPUT_DIR / "tables"
    figure_dir = v09g.DEFAULT_OUTPUT_DIR / "figures"
    assert table_dir.is_dir() and figure_dir.is_dir()
    assert len(list(table_dir.glob("*.csv"))) == 10
    assert len(list(figure_dir.glob("*.png"))) == 12
    assert len(list(figure_dir.glob("*.pdf"))) == 12
    assert v09g.LOCK_PATH.is_file()


def test_notebook_exists_with_18_sections() -> None:
    nb = nbformat.read(v09g.NOTEBOOK_PATH, as_version=4)
    sections = [
        cell
        for cell in nb.cells
        if cell.cell_type == "markdown" and re.match(r"^##\s+\d+\.", cell.source.strip())
    ]
    assert len(sections) == 18
    assert sum(1 for c in nb.cells if c.cell_type == "code") >= 18
    combined = "\n".join(c.source for c in nb.cells)
    assert "pipeline_v09g" in combined
    for forbidden in ("pipeline_v09d", "pipeline_v09e", "pipeline_v09f",
                      "pipeline_v08d", "pipeline_v08e", "nbclient"):
        assert forbidden not in combined, forbidden
    assert "import src.optimization" not in combined
    assert "import src.measurement" not in combined


def test_notebook_executes_top_to_bottom_byte_stable() -> None:
    """Executing the committed notebook is byte-stable: sha256 is unchanged."""
    path = v09g.NOTEBOOK_PATH
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=1500,
        kernel_name="python3",
        record_timing=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.metadata.pop("execution", None)
            for out in cell.get("outputs", []):
                out.pop("transient", None)
    nbformat.write(nb, str(path))
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after


def test_report_disclaimers_and_neutral_language() -> None:
    lowered = REPORT_TEXT.lower()
    for required in (
        "direct_energy_unavailable",
        "no tdp",
        "no single overall winner",
        "direction-uncertain",
        "is not cyber ground truth",
        "transparency note",
    ):
        assert required in lowered, required
    assert "2043" in REPORT_TEXT and "2044" in REPORT_TEXT
    assert "v09g_reporting_lock.json" in REPORT_TEXT


def test_evidence_snapshot_unchanged_after_notebook() -> None:
    import json

    lock = json.loads(v09g.LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["preflight_status"] == "GO"
    assert lock["source_evidence_hashes"] == v09g.snapshot_evidence(ROOT)


def test_reporting_lock_create_and_verify(tmp_path: Path) -> None:
    verified = v09g.create_reporting_lock(
        root=ROOT,
        output_dir=v09g.DEFAULT_OUTPUT_DIR,
        notebook_path=v09g.NOTEBOOK_PATH,
        report_path=v09g.REPORT_PATH,
    )
    assert verified["status"] == "VERIFIED"
    assert verified["stage"] == v09g.STAGE
    assert verified["schema_version"] == v09g.SCHEMA_VERSION
    re_verified = v09g.verify_reporting_lock(
        v09g.LOCK_PATH, root=ROOT, output_dir=v09g.DEFAULT_OUTPUT_DIR
    )
    assert re_verified["status"] == "VERIFIED"
    assert re_verified["notebook_hash"] == hashlib.sha256(
        v09g.NOTEBOOK_PATH.read_bytes()
    ).hexdigest()


def test_reporting_lock_rejects_tampered_table() -> None:
    lock_path = v09g.LOCK_PATH
    table = v09g.DEFAULT_TABLE_DIR / "dataset_governance.csv"
    original = table.read_bytes()
    table.write_bytes(original + b"\n")
    try:
        with pytest.raises(v09g.V09GError):
            v09g.verify_reporting_lock(
                lock_path, root=ROOT, output_dir=v09g.DEFAULT_OUTPUT_DIR
            )
    finally:
        table.write_bytes(original)