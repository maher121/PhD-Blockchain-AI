"""Tests for DataCo dataset discovery (signature detection, ambiguity, cache)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.dataset_discovery import (
    _matches_signature,
    find_dataco_candidates,
    find_kagglehub_cache_candidates,
    resolve_dataco_dataset,
)
from tests.conftest import write_dataco_like_csv


def _v01_normalised_csv(tmp_path):
    """A V0.1 canonical-schema CSV (NOT a raw DataCo table)."""
    path = tmp_path / "supply_chain_raw.csv"
    pd.DataFrame(
        {
            "transaction_id": ["TXN-1"],
            "participant_id": ["supplier_a"],
            "product_id": ["PROD-1"],
            "order_id": ["ORD-1"],
            "timestamp": ["2024-01-01"],
            "quantity": [1.0],
            "unit_price": [10.0],
            "total_amount": [10.0],
            "shipping_days": [3.0],
            "location_region": ["EMEA"],
            "transaction_status": ["completed"],
        }
    ).to_csv(path, index=False)
    return path


def test_signature_match() -> None:
    proposed = [
        "Order Id",
        "Order Item Id",
        "Order Customer Id",
        "order date (DateOrders)",
        "Late_delivery_risk",
    ]
    ok, missing = _matches_signature(proposed)
    assert ok and missing == []


def test_signature_rejects_incomplete_header() -> None:
    ok, missing = _matches_signature(["Order Id", "Late_delivery_risk"])
    assert not ok
    assert len(missing) == 3


def test_finds_unambiguous_dataco_in_raw(tmp_path) -> None:
    path = write_dataco_like_csv(tmp_path)
    resolution = resolve_dataco_dataset(raw_dir=tmp_path, stage_to=tmp_path / "staged.csv")
    assert resolution.found_in_raw
    assert resolution.path == path
    assert resolution.provenance == "raw_dir"
    assert not resolution.errors


def test_v01_normalised_csv_is_not_selected(tmp_path) -> None:
    _v01_normalised_csv(tmp_path)
    candidates = find_dataco_candidates(tmp_path)
    assert candidates == []


def test_missing_dataset_reports_expected_format(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.data.dataset_discovery.find_kagglehub_cache_candidates",
        lambda: [],
    )
    resolution = resolve_dataco_dataset(
        raw_dir=tmp_path, stage_to=tmp_path / "DataCoSupplyChainDataset.csv"
    )
    assert resolution.path is None
    assert resolution.provenance == "none"
    assert "Order Item Id" in " ".join(resolution.errors)


def test_ambiguous_raw_candidates_are_reported(tmp_path) -> None:
    write_dataco_like_csv(tmp_path)
    # a second file with the same signature -> ambiguous
    (tmp_path / "second.csv").write_text(
        (tmp_path / "DataCoSupplyChainDataset.csv").read_text()
    )
    resolution = resolve_dataco_dataset(raw_dir=tmp_path)
    assert resolution.path is None
    assert resolution.provenance == "ambiguous_raw"
    assert len(resolution.candidates_raw) == 2


def test_stages_from_kagglehub_cache_when_raw_missing(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    cache_file = write_dataco_like_csv(cache_dir)

    monkeypatch.setattr(
        "src.data.dataset_discovery.find_kagglehub_cache_candidates",
        lambda: [cache_file],
    )
    raw_dir = tmp_path / "raw"
    resolution = resolve_dataco_dataset(raw_dir=raw_dir, stage_to=raw_dir / "DataCoSupplyChainDataset.csv")
    assert resolution.path is not None
    assert resolution.provenance == "kagglehub_cache"
    assert resolution.path.read_bytes() == cache_file.read_bytes()
    assert resolution.staged_to is not None


def test_cache_scanner_returns_genuine_files(tmp_path) -> None:
    write_dataco_like_csv(tmp_path)
    found = find_dataco_candidates(tmp_path)
    assert len(found) == 1


@pytest.mark.skipif(
    find_kagglehub_cache_candidates() == [],
    reason="requires a local kagglehub cache of the DataCo dataset",
)
def test_real_cache_present_on_this_machine() -> None:
    matches = find_kagglehub_cache_candidates()
    assert len(matches) >= 1