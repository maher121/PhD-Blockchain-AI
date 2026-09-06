"""Offline tests for source-aware data loading (kagglehub is mocked)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ai.anomaly_detection import report_anomaly_metrics
from src.data.generator import generate_supply_chain_dataset, write_supply_chain_dataset
from src.data.loading import (
    load_or_download_supply_chain_data,
    normalize_kaggle_supply_chain,
)
from src.pipeline import prepare_features


def _kaggle_style_frame(n: int = 20) -> pd.DataFrame:
    """Hand-built table mirroring the DataCo Kaggle columns used by the map."""
    segments = ["Consumer", "Corporate", "Home Office"]
    return pd.DataFrame(
        {
            "Order Item Id": [57480000000 + i for i in range(n)],
            "Customer Segment": [segments[i % 3] for i in range(n)],
            "Product Card Id": [1029 + (i % 5) for i in range(n)],
            "Order Id": [57300000000 + i for i in range(n)],
            "order date (DateOrders)": pd.date_range("2015-01-01", periods=n, freq="D"),
            "Order Item Quantity": [10 + i for i in range(n)],
            "Order Item Product Price": [25.0 + i for i in range(n)],
            "Sales": [(10 + i) * (25.0 + i) for i in range(n)],
            "Days for shipping (real)": [4.0] * n,
            "Market": ["Europe"] * n,
            "Order Status": ["COMPLETE"] * n,
            "Late_delivery_risk": [0, 1] * (n // 2),
        }
    )


# --- normalization --------------------------------------------------------


def test_normalize_kaggle_columns() -> None:
    canonical = normalize_kaggle_supply_chain(_kaggle_style_frame())
    assert set(
        [
            "transaction_id",
            "participant_id",
            "product_id",
            "order_id",
            "timestamp",
            "quantity",
            "unit_price",
            "total_amount",
            "shipping_days",
            "location_region",
            "transaction_status",
            "natural_label",
        ]
    ).issubset(set(canonical.columns))
    assert canonical["natural_label"].isna().sum() == 0
    assert canonical["quantity"].dtype == "float64"
    assert (canonical["quantity"] > 0).all()


def test_normalize_kaggle_missing_columns_raises() -> None:
    with pytest.raises(ValueError):
        normalize_kaggle_supply_chain(pd.DataFrame({"unrelated": [1, 2]}))


# --- source resolution -----------------------------------------------------


def test_local_file_is_used_without_kaggle(tmp_path, monkeypatch) -> None:
    local_csv = tmp_path / "raw.csv"
    write_supply_chain_dataset(generate_supply_chain_dataset(n_rows=50), local_csv)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("kaggle download should not be called when local exists")

    monkeypatch.setattr("src.data.loading.download_kaggle_dataset", _fail_if_called)
    df, source = load_or_download_supply_chain_data(local_csv, source="auto")
    assert source == "local"
    assert len(df) == 50


def test_kaggle_download_is_cached_locally(tmp_path, monkeypatch) -> None:
    download_target = tmp_path / "downloaded.csv"

    monkeypatch.setattr(
        "src.data.loading.download_kaggle_dataset",
        lambda *a, **k: _kaggle_style_frame(),
    )

    df, source = load_or_download_supply_chain_data(download_target, source="kaggle")
    assert source == "kaggle"
    assert download_target.exists()
    cached = pd.read_csv(download_target, low_memory=False)
    assert len(cached) == len(df)
    assert "natural_label" in cached.columns


def test_fallback_to_synthetic_on_download_failure(tmp_path, monkeypatch) -> None:
    def _explode(*args, **kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr("src.data.loading.download_kaggle_dataset", _explode)
    target = tmp_path / "raw.csv"
    df, source = load_or_download_supply_chain_data(target, source="auto")
    assert source == "synthetic"
    assert "known_anomaly" in df.columns
    assert target.exists()


def test_kaggle_source_fails_hard_when_requested(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.data.loading.download_kaggle_dataset",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network")),
    )
    with pytest.raises(RuntimeError):
        load_or_download_supply_chain_data(tmp_path / "x.csv", source="kaggle")


def test_unknown_source_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        load_or_download_supply_chain_data(tmp_path / "x.csv", source="mars")


# --- label provenance through the pipeline ---------------------------------


def test_prepare_features_natural_labels(tmp_path) -> None:
    raw = normalize_kaggle_supply_chain(_kaggle_style_frame(n=40))
    processed, features, labels, label_kind = prepare_features(
        raw, processed_path=tmp_path / "processed.csv", max_rows=100
    )
    assert label_kind == "natural_late_delivery_risk"
    assert labels is not None
    assert set(labels.unique()).issubset({0, 1})
    assert len(features) == len(labels)


def test_prepare_features_controlled_labels(tmp_path) -> None:
    raw = generate_supply_chain_dataset(n_rows=100, seed=11)
    processed, features, labels, label_kind = prepare_features(
        raw, processed_path=tmp_path / "processed.csv", max_rows=100
    )
    assert label_kind == "controlled_synthetic"
    assert labels is not None
    assert features.shape[0] == len(labels)


# --- optional-label metrics ------------------------------------------------


def test_metrics_without_labels_is_empty() -> None:
    scores = pd.Series([0.1, 0.9, 0.3, 0.7])
    flags = pd.Series([False, True, False, True])
    assert report_anomaly_metrics(None, flags, scores) == {}


def test_metrics_with_single_class_labels_is_empty() -> None:
    scores = pd.Series([0.1, 0.9, 0.3, 0.7])
    flags = pd.Series([False, True, False, True])
    single_class = pd.Series([1, 1, 1, 1])
    assert report_anomaly_metrics(single_class, flags, scores) == {}