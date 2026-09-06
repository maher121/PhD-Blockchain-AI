"""Tests for the reusable dataset audit (V0.2)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.audit import (
    audit_dataframe,
    column_feature_summary,
    leakage_candidates,
)
from tests.conftest import make_dataco_like_frame


def test_audit_shape_and_artefacts(tmp_path) -> None:
    df = make_dataco_like_frame(n_orders=6, rows_per_order=3)
    report = audit_dataframe(df, source_name="fixture", output_dir=tmp_path, save_figures=False)

    assert report["n_rows"] == len(df)
    assert report["n_columns"] == df.shape[1]
    assert set(report["column_names"]) == set(df.columns)

    for name in (
        "dataset_summary.json",
        "missing_values.csv",
        "duplicate_report.json",
        "feature_summary.csv",
        "leakage_candidates.json",
    ):
        assert (tmp_path / name).exists(), name

    summary = json.loads((tmp_path / "dataset_summary.json").read_text())
    assert summary["n_rows"] == len(df)

    missing = pd.read_csv(tmp_path / "missing_values.csv")
    assert set(missing.columns) == {"column", "missing_count", "missing_percentage"}
    assert (missing["missing_count"] >= 0).all()

    feature_summary = pd.read_csv(tmp_path / "feature_summary.csv")
    assert len(feature_summary) == df.shape[1]
    assert {
        "column",
        "dtype",
        "role",
        "n_missing",
        "missing_pct",
        "n_unique",
        "is_constant",
        "is_identifier",
        "is_datetime",
        "is_high_cardinality",
    }.issubset(set(feature_summary.columns))


def test_audit_detects_missing_and_type_candidates(tmp_path) -> None:
    df = make_dataco_like_frame()
    report = audit_dataframe(df, source_name="fixture", output_dir=tmp_path, save_figures=False)

    assert report["missing_summary"]["total_missing_cells"] > 0
    assert "order date (DateOrders)" in report["potential_datetime_columns"]
    assert any(str(col) == "Product Price" for col in report["numeric_columns"])


def test_leakage_candidates_flag_outcome_and_keys() -> None:
    df = make_dataco_like_frame()
    flags = leakage_candidates(df)
    reasons = {f["column"]: f["reason"] for f in flags}
    assert reasons["Delivery Status"] == "outcome"
    assert reasons["Days for shipping (real)"] == "outcome_adjacent"
    assert "Order Id" in reasons


def test_audit_raises_on_empty() -> None:
    with pytest.raises(ValueError):
        audit_dataframe(pd.DataFrame(), output_dir="/tmp/audit", save_figures=False)


def test_audit_figures_rendered(tmp_path) -> None:
    df = make_dataco_like_frame()
    audit_dataframe(df, source_name="fixture", output_dir=tmp_path, save_figures=True)
    figures = list((tmp_path / "figures").glob("*.png"))
    names = {p.name for p in figures}
    assert "target_distribution.png" in names
    assert "numeric_distributions.png" in names


def test_column_feature_summary_constant_flag() -> None:
    df = make_dataco_like_frame()
    df["Product Status"] = 0
    summary = column_feature_summary(df)
    row = summary[summary["column"] == "Product Status"].iloc[0]
    assert bool(row["is_constant"]) is True
    assert bool(row["is_near_constant"]) is True