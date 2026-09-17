"""Focused tests for V0.8-F artifact-only reporting utilities."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import pytest

import src.pipeline_v08f as v08f


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifacts() -> dict[str, object]:
    return v08f.load_v08f_artifacts(ROOT)


def test_preflight_passes_with_expected_head() -> None:
    result = v08f.verify_v08f_preflight(root=ROOT, expected_head=None)
    assert result["status"] == "GO"
    assert re.fullmatch(r"[0-9a-f]{7}", result["head_short"]) is not None
    assert result["v08c_lock_verified"] is True
    assert result["v08d_lock_verified"] is True
    assert result["v08e_lock_verified"] is True
    assert result["v08e_semantic_result_lock_sha256"] == v08f.EXPECTED_E4_SEMANTIC_HASH
    assert result["resource_tradeoff_classification"] == v08f.EXPECTED_CLASSIFICATION


def test_preflight_rejects_wrong_head() -> None:
    with pytest.raises(v08f.V08FError):
        v08f.verify_v08f_preflight(root=ROOT, expected_head="deadbee")


def test_publication_tables_include_governed_values(artifacts: dict[str, object]) -> None:
    tables = v08f.build_publication_tables(artifacts)
    assert set(tables) == set(v08f.TABLE_FILENAMES)

    table_a = tables["table_a_bpso_run_summary"]
    assert len(table_a) == 5
    row_1042 = table_a.loc[table_a["optimizer_seed"] == 1042].iloc[0]
    assert int(row_1042["selected_feature_count"]) == 10
    assert float(row_1042["average_precision"]) == pytest.approx(0.2302848102260538)
    assert float(row_1042["f1"]) == pytest.approx(0.20330805088258272)
    assert float(row_1042["recall"]) == pytest.approx(0.31866666666666665)

    table_b = tables["table_b_locked_k10_features"]
    assert list(table_b["feature"]) == list(v08f.LOCKED_K10_FEATURES)

    table_c = tables["table_c_predictive_results"]
    dt_k10 = table_c[
        (table_c["classifier"] == "decision_tree")
        & (table_c["configuration_id"] == "BPSO-K10")
    ].iloc[0]
    assert float(dt_k10["average_precision"]) == pytest.approx(0.21961151547107932)
    assert float(dt_k10["f1"]) == pytest.approx(0.18725821411669835)
    assert float(dt_k10["recall"]) == pytest.approx(0.2986666666666667)

    table_d = tables["table_d_structural_comparison"]
    dt_k43 = table_d[
        (table_d["classifier"] == "decision_tree") & (table_d["configuration_id"] == "K43")
    ].iloc[0]
    dt_k10_struct = table_d[
        (table_d["classifier"] == "decision_tree") & (table_d["configuration_id"] == "BPSO-K10")
    ].iloc[0]
    assert float(dt_k43["train_input_memory_bytes"]) == pytest.approx(9632132.0)
    assert float(dt_k10_struct["train_input_memory_bytes"]) == pytest.approx(2240132.0)

    table_g = tables["table_g_limitations_claim_boundaries"]
    assert "DIRECT_ENERGY_UNAVAILABLE" in set(table_g["status"])


def test_write_tables_and_render_figures(artifacts: dict[str, object], tmp_path: Path) -> None:
    tables = v08f.build_publication_tables(artifacts)
    written = v08f.write_publication_tables(tables, table_dir=tmp_path / "tables")
    assert len(written) == 7
    assert all(path.is_file() for path in written.values())

    rendered = v08f.render_publication_figures(artifacts, figure_dir=tmp_path / "figures")
    assert set(rendered) == set(v08f.FIGURE_NAMES)
    assert len(rendered) == 10
    for outputs in rendered.values():
        assert outputs["png"].is_file()
        assert outputs["pdf"].is_file()


def test_build_outputs_end_to_end(tmp_path: Path) -> None:
    output = v08f.build_v08f_outputs(
        root=ROOT,
        expected_head=None,
        output_dir=tmp_path / "v08f_reporting",
    )
    assert output["status"] == "COMPLETED"
    assert output["table_count"] == 7
    assert output["figure_count"] == 10

    table_dir = Path(output["table_dir"])
    figure_dir = Path(output["figure_dir"])
    assert table_dir.is_dir()
    assert figure_dir.is_dir()
    assert len(list(table_dir.glob("*.csv"))) == 7
    assert len(list(figure_dir.glob("*.png"))) == 10
    assert len(list(figure_dir.glob("*.pdf"))) == 10


def test_table_roundtrip_csv_schema(artifacts: dict[str, object], tmp_path: Path) -> None:
    tables = v08f.build_publication_tables(artifacts)
    written = v08f.write_publication_tables(tables, table_dir=tmp_path / "tables")
    loaded = pd.read_csv(written["table_e_paired_computational"])
    assert "ci_includes_zero" in loaded.columns
    assert "timing_integrity_limited" in loaded.columns
    assert set(loaded["reference_configuration_id"]) == {"K43", "K42", "K11"}
