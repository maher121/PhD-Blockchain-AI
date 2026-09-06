"""Tests for the leakage-safe preprocessing pipeline (V0.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.dataco_pipeline import (
    DataCoPreprocessor,
    PreprocessorConfig,
)
from src.preprocessing.leakage import split_dataset
from tests.conftest import make_dataco_like_frame


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_dataco_like_frame(n_orders=20, rows_per_order=4, seed=9)


def test_transform_requires_fit(frame: pd.DataFrame) -> None:
    preprocessor = DataCoPreprocessor()
    with pytest.raises(RuntimeError):
        preprocessor.transform(frame, "train")


def test_fit_transform_no_missing_features(frame: pd.DataFrame) -> None:
    preprocessor = DataCoPreprocessor().fit(frame)
    out = preprocessor.transform(frame, "train")
    assert not out.ml_features.isna().any().any()
    assert out.ml_features.shape[1] == len(preprocessor.fit_report["ml_feature_columns"])


def test_fit_on_train_transform_val_test_invariant_shape(frame: pd.DataFrame) -> None:
    res = split_dataset(frame, strategy="order_grouped", seed=3)
    preprocessor = DataCoPreprocessor().fit(res.train)
    n_train_cols = preprocessor.fit_report["ml_feature_count"]

    for name, split_frame in res.frames().items():
        out = preprocessor.transform(split_frame, name)
        assert out.ml_features.shape[1] == n_train_cols
        assert not out.ml_features.isna().any().any()
        # no NaN in the target alignment
        if out.target is not None:
            assert len(out.target) == len(out.ml_features)


def test_column_layout_is_deterministic(frame: pd.DataFrame) -> None:
    a = DataCoPreprocessor().fit(frame)
    b = DataCoPreprocessor().fit(frame)
    assert a.fit_report["ml_feature_columns"] == b.fit_report["ml_feature_columns"]


def test_order_aggregates_are_train_derived(frame: pd.DataFrame) -> None:
    res = split_dataset(frame, strategy="order_grouped", seed=3)
    train = res.train
    preprocessor = DataCoPreprocessor(
        PreprocessorConfig(scale_numeric=False)
    ).fit(train)
    out_train = preprocessor.transform(train, "train")

    order = train["Order Id"].astype(str).iloc[0]
    # aggregates are fitted on the training rows that survive transform
    # (missing-target rows are dropped), so mirror that in the expectation.
    mask = (train["Order Id"].astype(str) == order) & train["Late_delivery_risk"].notna()
    expected_total = float(train.loc[mask, "Order Item Total"].sum())
    row_mask = out_train.metadata["Order Id"].astype(str) == order
    observed = out_train.ml_features.loc[
        row_mask.index[row_mask], "order_total_value"
    ]
    assert not observed.empty
    assert np.allclose(observed, expected_total, atol=1e-6)


def test_unseen_categorical_level_is_handled(frame: pd.DataFrame) -> None:
    train = frame[frame["Market"] != "LATAM"]
    prod = DataCoPreprocessor().fit(train)
    # transform the full frame (contains a Market level unseen in training)
    out = prod.transform(frame, "full")
    assert not out.ml_features.isna().any().any()
    assert out.ml_features.shape[1] == prod.fit_report["ml_feature_count"]


def test_negative_values_are_reported_and_imputed(frame: pd.DataFrame) -> None:
    frame2 = frame.copy()
    # introduce more negatives so the warning path triggers
    frame2.loc[0, "Order Item Quantity"] = -99.0
    frame2.loc[1, "Order Item Product Price"] = -1.0
    prod = DataCoPreprocessor().fit(frame2)
    out = prod.transform(frame2, "neg")
    assert not out.ml_features.isna().any().any()
    assert np.isfinite(out.ml_features.values).all()


def test_missing_target_rows_dropped(frame: pd.DataFrame) -> None:
    prod = DataCoPreprocessor().fit(frame)
    with_target = frame[frame["Late_delivery_risk"].notna()]
    out = prod.transform(frame, "with_target")
    assert out.target is not None
    assert np.isfinite(out.target).all()
    assert len(out.ml_features) == (frame["Late_delivery_risk"].notna().sum())


def test_missing_required_column_raises(frame: pd.DataFrame) -> None:
    bad = frame.drop(columns=["Order Item Quantity"])
    prod = DataCoPreprocessor()
    with pytest.raises(ValueError, match="required columns"):
        prod.fit(bad)


def test_metadata_and_features_kept_separate(frame: pd.DataFrame) -> None:
    prod = DataCoPreprocessor().fit(frame)
    out = prod.transform(frame, "all")
    assert "Order Id" in out.metadata.columns
    assert "Order Id" not in out.ml_features.columns
    assert "Customer Segment" not in out.ml_features.columns
    assert any(c.startswith("Customer Segment_") for c in out.ml_features.columns)


def test_config_snapshot_is_recorded(frame: pd.DataFrame) -> None:
    config = PreprocessorConfig(seed=11, scale_numeric=False)
    prod = DataCoPreprocessor(config).fit(frame)
    snap = prod.config_snapshot()
    assert snap["seed"] == 11
    assert snap["scale_numeric"] is False