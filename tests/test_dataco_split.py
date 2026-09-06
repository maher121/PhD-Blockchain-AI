"""Tests for deterministic, leakage-aware splitting (V0.2)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing.leakage import (
    check_identity_overlap,
    split_dataset,
)
from tests.conftest import make_dataco_like_frame


def test_random_split_is_reproducible() -> None:
    df = make_dataco_like_frame(n_orders=10, rows_per_order=4)
    a = split_dataset(df, strategy="random", seed=7)
    b = split_dataset(df, strategy="random", seed=7)
    pd.testing.assert_frame_equal(a.train, b.train)
    pd.testing.assert_frame_equal(a.validation, b.validation)
    pd.testing.assert_frame_equal(a.test, b.test)


def test_random_split_sizes_match_ratios() -> None:
    df = make_dataco_like_frame(n_orders=50, rows_per_order=4)
    res = split_dataset(df, strategy="random", seed=1)
    n = len(df)
    assert abs(res.report["size_percentages"]["train"] - 0.70) < 0.03
    assert abs(res.report["size_percentages"]["validation"] - 0.15) < 0.03
    assert abs(res.report["size_percentages"]["test"] - 0.15) < 0.03
    assert res.report["sizes"]["train"] + res.report["sizes"]["validation"] + res.report["sizes"]["test"] == n


def test_order_grouped_has_no_train_test_overlap() -> None:
    df = make_dataco_like_frame(n_orders=20, rows_per_order=4)
    res = split_dataset(df, strategy="order_grouped", seed=1)
    overlap = res.report["identity_overlap_checks"]["Order Id"]
    assert overlap["train_vs_validation"] == 0
    assert overlap["train_vs_test"] == 0
    assert overlap["validation_vs_test"] == 0
    assert res.report["leakage_warning"] is False


def test_order_grouped_preserves_order_rows() -> None:
    """Every row of one Order Id must stay in the same split."""
    df = make_dataco_like_frame(n_orders=15, rows_per_order=3)
    res = split_dataset(df, strategy="order_grouped", seed=3)
    for order_id in df["Order Id"].unique():
        present = [
            name
            for name, frame in res.frames().items()
            if (frame["Order Id"].astype(str) == order_id).any()
        ]
        assert len(present) == 1, order_id


def test_order_grouped_sizes_still_honour_ratio() -> None:
    df = make_dataco_like_frame(n_orders=60, rows_per_order=5)
    res = split_dataset(df, strategy="order_grouped", seed=5)
    for split in ("train", "validation", "test"):
        assert 0.1 < res.report["size_percentages"][split] < 0.9


def test_chronological_split_is_ordered() -> None:
    df = make_dataco_like_frame(n_orders=30, rows_per_order=4)
    res = split_dataset(df, strategy="chronological")
    assert res.train["order date (DateOrders)"].is_monotonic_increasing
    assert (
        res.train["order date (DateOrders)"].min()
        <= res.test["order date (DateOrders)"].max()
    )


def test_check_identity_overlap_detects_sharing() -> None:
    df = make_dataco_like_frame(n_orders=4, rows_per_order=2)
    train = df.iloc[:4]
    test = df.iloc[2:]
    report = check_identity_overlap(
        {"train": train, "test": test}, keys=("Order Id",)
    )
    assert report["Order Id"]["train_vs_test"] >= 1


def test_invalid_strategy_raises() -> None:
    df = make_dataco_like_frame()
    with pytest.raises(ValueError):
        split_dataset(df, strategy="bogus")


def test_invalid_ratios_raise() -> None:
    df = make_dataco_like_frame()
    with pytest.raises(ValueError):
        split_dataset(df, ratios={"train": 0.5, "validation": 0.2, "test": 0.2})
    with pytest.raises(ValueError):
        split_dataset(df, ratios={"train": 0.7, "validation": 0.3})


def test_empty_frame_raises() -> None:
    with pytest.raises(ValueError):
        split_dataset(pd.DataFrame())