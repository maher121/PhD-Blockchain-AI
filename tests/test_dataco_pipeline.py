"""End-to-end tests for the V0.2 DataCo pipeline (fixture-based)."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.pipeline_v02 import (
    DataCoDatasetMissingError,
    load_dataco_frame,
    run_dataco_pipeline_v02,
)
from tests.conftest import make_dataco_like_frame, write_dataco_like_csv


@pytest.fixture()
def workspace(tmp_path):
    raw = tmp_path / "raw"
    write_dataco_like_csv(raw, n_orders=24, rows_per_order=4)
    return {
        "raw": raw,
        "audit": tmp_path / "audit",
        "processed": tmp_path / "processed",
        "tmp": tmp_path,
    }


def _run(workspace, **kwargs):
    return run_dataco_pipeline_v02(
        raw_dir=workspace["raw"],
        audit_dir=workspace["audit"],
        processed_dir=workspace["processed"],
        dictionary_path=workspace["tmp"] / "data_dictionary.md",
        metadata_path=workspace["tmp"] / "dataset_metadata.json",
        report_path=workspace["tmp"] / "report.json",
        max_rows=None,
        seed=42,
        split_strategy="order_grouped",
        **kwargs,
    )


def test_pipeline_creates_all_outputs(workspace) -> None:
    bundle = _run(workspace)
    assert bundle["resolution"].found_in_raw
    assert bundle["resolution"].provenance == "raw_dir"

    for split_name in ("train", "validation", "test"):
        for suffix in ("metadata.csv", "features.csv", "target.csv"):
            path = workspace["processed"] / split_name / suffix
            assert path.exists(), f"{split_name}/{suffix}"

    assert (workspace["audit"] / "dataset_summary.json").exists()
    assert (workspace["audit"] / "preprocessing_summary.json").exists()
    assert (workspace["tmp"] / "dataset_metadata.json").exists()
    assert (workspace["tmp"] / "data_dictionary.md").exists()
    assert (workspace["tmp"] / "report.json").exists()


def test_pipeline_split_sizes_and_leakage(workspace) -> None:
    bundle = _run(workspace)
    report = bundle["split_result"].report
    assert report["identity_overlap_checks"]["Order Id"]["train_vs_test"] == 0
    assert not report["leakage_warning"]
    sizes = report["sizes"]
    total = sum(sizes.values())
    n_raw = bundle["work"].shape[0]
    assert total == n_raw


def test_pipeline_raw_data_unchanged(workspace) -> None:
    raw_path = workspace["raw"] / "DataCoSupplyChainDataset.csv"
    before = raw_path.read_bytes()
    _run(workspace)
    assert raw_path.read_bytes() == before


def test_pipeline_deterministic(workspace) -> None:
    a = _run(workspace)
    b = _run(workspace)
    for split_name in ("train", "validation", "test"):
        pa = workspace["processed"] / split_name / "features.csv"
        pb = workspace["processed"] / split_name / "features.csv"
        assert pa.read_bytes() == pb.read_bytes()
        assert (
            a["processed"][split_name].ml_features.shape
            == b["processed"][split_name].ml_features.shape
        )
    assert (
        a["preprocessor"].fit_report["ml_feature_columns"]
        == b["preprocessor"].fit_report["ml_feature_columns"]
    )


def test_pipeline_features_clean(workspace) -> None:
    bundle = _run(workspace)
    for name, frame in bundle["processed"].items():
        assert not frame.ml_features.isna().any().any()
        assert np.isfinite(frame.ml_features.values).all()
        assert "Order Id" not in frame.ml_features.columns
        assert "Order Id" in frame.metadata.columns


def test_pipeline_metadata_records_have_expected_fields(workspace) -> None:
    bundle = _run(workspace)
    metadata = json.loads(
        (workspace["tmp"] / "dataset_metadata.json").read_text()
    )
    assert metadata["dataset"]["file_sha256"]
    assert metadata["dataset"]["n_raw_rows"] == len(bundle["raw_df"])
    assert metadata["split"]["strategy"] == "order_grouped"
    assert metadata["features"]["target_column"] == "Late_delivery_risk"
    assert metadata["features"]["cybersecurity_labels"] is False


def test_pipeline_report_states_no_cyber_labels(workspace) -> None:
    _run(workspace)
    report = json.loads((workspace["tmp"] / "report.json").read_text())
    assert "does not provide native cybersecurity attack labels" in report["payload"]["notes"]["cybersecurity_labels"]
    assert report["payload"]["notes"]["synthetic_attacks"] == "Not created in V0.2."


def test_pipeline_missing_dataset_raises_clear_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.data.dataset_discovery.find_kagglehub_cache_candidates", lambda: []
    )
    with pytest.raises(DataCoDatasetMissingError) as exc_info:
        run_dataco_pipeline_v02(
            raw_dir=tmp_path / "raw",
            audit_dir=tmp_path / "audit",
            processed_dir=tmp_path / "processed",
            max_rows=None,
        )
    assert "DataCoSupplyChainDataset" in str(exc_info.value)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_DATACO_INTEGRATION") != "1",
    reason="Requires the real 180k-row DataCo dataset in data/raw "
    "(set RUN_DATACO_INTEGRATION=1 to execute).",
)
def test_integration_real_dataco_pipeline() -> None:
    """Full pipeline on the real dataset (optional, marked integration)."""
    from src.pipeline_v02 import run_dataco_pipeline_v02

    bundle = run_dataco_pipeline_v02(
        audit_dir="/tmp/dataco_v02_int_audit",
        processed_dir="/tmp/dataco_v02_int_processed",
        dictionary_path="/tmp/dataco_v02_int_data_dictionary.md",
        metadata_path="/tmp/dataco_v02_int_metadata.json",
        report_path="/tmp/dataco_v02_int_report.json",
        max_rows=5000,
        seed=42,
        split_strategy="order_grouped",
    )
    assert bundle["audit_report"]["n_rows"] > 100000
    assert len(bundle["processed"]["train"].ml_features) > 3000
    assert not bundle["processed"]["train"].ml_features.isna().any().any()
    assert (
        bundle["split_result"].report["identity_overlap_checks"]["Order Id"][
            "train_vs_test"
        ]
        == 0
    )


def test_load_dataco_frame_assigns_row_id(workspace) -> None:
    from src.data.dataset_discovery import resolve_dataco_dataset

    resolution = resolve_dataco_dataset(raw_dir=workspace["raw"])
    df = load_dataco_frame(resolution)
    assert df["row_id"].is_monotonic_increasing
    assert df["row_id"].nunique() == len(df)