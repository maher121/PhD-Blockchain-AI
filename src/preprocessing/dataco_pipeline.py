"""Modular, leakage-safe preprocessing pipeline for the DataCo table (V0.2).

Design
------
Leakage-prevention contract (PhD requirement):

    Raw dataset
        -> Train / Validation / Test split          (src.preprocessing.leakage)
        -> fit preprocessing on TRAIN only          (this module: .fit())
        -> transform VALIDATION (same parameters)
        -> transform TEST      (same parameters)

Every statistic learned from data (numeric imputation medians, one-hot
levels and their order, per-order aggregates, the standard scaler, dropped
constant columns) is fitted on the training split and only *applied* to
validation/test. The original raw dataset is never mutated.

The preprocessor separates two outputs per split:

* ``ml_features`` -- columns suitable for supervised ML (scaled numerics,
  one-hot categoricals, documented engineered features). No identifiers.
* ``metadata``    -- traceability / blockchain metadata (Order/Customer/
  Product keys, dates, statuses, raw monetary values, target label). Never
  used as ML features by default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATACO_CATEGORICAL_FEATURES,
    DATACO_TARGET_COLUMN,
    DATA_QUALITY_DROP_EMPTY_TARGET,
    GLOBAL_SEED,
    MAX_ONEHOT_CARDINALITY,
    NEGATIVE_VALUE_COLUMNS,
    SPLIT_GROUP_KEY,
)
from src.preprocessing.feature_engineering_v2 import (
    FEATURE_CATALOG,
    build_raw_ml_features,
    compute_order_level,
)

logger = logging.getLogger(__name__)

#: Traceability / blockchain metadata carried alongside (NOT into) ML features.
METADATA_COLUMNS: tuple[str, ...] = (
    "row_id",
    "Order Id",
    "Order Item Id",
    "Order Customer Id",
    "Product Card Id",
    "order date (DateOrders)",
    "shipping date (DateOrders)",
    "Order Status",
    "Delivery Status",
    "Order Region",
    "Category Name",
    "Product Name",
    "Days for shipping (real)",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Type",
    "Market",
    "Shipping Mode",
    "Customer Segment",
    "Department Name",
)

_OTHER_BUCKET: str = "__OTHER__"

#: Raw columns whose negatives are impossible and translated to feature names.
_NEGATIVE_FEATURE_MAP: dict[str, str] = {
    "Order Item Quantity": "order_item_quantity",
    "Order Item Product Price": "order_item_product_price",
    "Product Price": "product_price",
    "Order Item Total": "order_item_total",
}


def _mask_impossible_negatives(numeric: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Replace negative values of the impossible-negative features with NaN."""
    negative_feature_columns = list(_NEGATIVE_FEATURE_MAP.values())
    present = [c for c in negative_feature_columns if c in numeric.columns]
    mask = numeric[present].lt(0)
    count = int(mask.sum().sum())
    if count:
        numeric = numeric.copy()
        numeric[present] = numeric[present].mask(mask)
    return numeric, count


@dataclass
class PreprocessorConfig:
    """Tunable preprocessing options (every value enters the metadata record)."""

    impute_numeric_strategy: str = "median"
    impute_categorical_strategy: str = "mode"
    scale_numeric: bool = True
    onehot_max_cardinality: int = MAX_ONEHOT_CARDINALITY
    drop_empty_target: bool = DATA_QUALITY_DROP_EMPTY_TARGET
    negative_value_columns: tuple[str, ...] = NEGATIVE_VALUE_COLUMNS
    seed: int = GLOBAL_SEED

    def to_dict(self) -> dict[str, Any]:
        return {
            "impute_numeric_strategy": self.impute_numeric_strategy,
            "impute_categorical_strategy": self.impute_categorical_strategy,
            "scale_numeric": self.scale_numeric,
            "onehot_max_cardinality": int(self.onehot_max_cardinality),
            "drop_empty_target": self.drop_empty_target,
            "negative_value_columns": list(self.negative_value_columns),
            "seed": int(self.seed),
        }


@dataclass
class ProcessedFrame:
    """One split's preprocessed outputs."""

    split_name: str
    ml_features: pd.DataFrame
    metadata: pd.DataFrame
    target: pd.Series | None

    def row_count(self) -> int:
        return int(len(self.ml_features))


class DataCoPreprocessor:
    """Fit-on-train / transform-any preprocessing transformer."""

    def __init__(self, config: PreprocessorConfig | None = None) -> None:
        self.config = config or PreprocessorConfig()
        self._fitted = False
        self._numeric_medians: pd.Series | None = None
        self._scaler: StandardScaler | None = None
        self._onehot_levels: dict[str, list[str]] = {}
        self._onehot_columns: list[str] = []
        self._order_aggregates: pd.DataFrame | None = None
        self._ml_columns: list[str] = []
        self.fit_report: dict[str, Any] = {}

    # -- public API ----------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "DataCoPreprocessor":
        """Fit all learnt statistics on the TRAINING split only."""
        if "row_id" not in train_df.columns:
            train_df = train_df.reset_index().rename(columns={"index": "row_id"})
        _require_columns(train_df, _required_sources())
        if self.config.drop_empty_target and DATACO_TARGET_COLUMN in train_df.columns:
            train_df = train_df[train_df[DATACO_TARGET_COLUMN].notna()]

        self._order_aggregates = compute_order_level(train_df).sort_index()

        raw_features = build_raw_ml_features(train_df, self._order_aggregates)
        numeric = raw_features.apply(pd.to_numeric, errors="coerce")
        numeric, negative_count = _mask_impossible_negatives(numeric)
        if negative_count:
            logger.info("Training fit: masked %d impossible negative values", negative_count)
        self._numeric_medians = numeric.median()

        imputed = numeric.fillna(self._numeric_medians)
        self._scaler = StandardScaler().fit(imputed) if self.config.scale_numeric else None
        scaled = (
            pd.DataFrame(
                self._scaler.transform(imputed),
                columns=imputed.columns,
                index=imputed.index,
            )
            if self._scaler is not None
            else imputed
        )

        self._onehot_levels = _fit_onehot_levels(train_df)
        dummies = _make_dummies(train_df, self._onehot_levels)
        self._onehot_columns = list(dummies.columns)

        combined = pd.concat([scaled, dummies], axis=1)
        combined = _drop_constant_or_allna_columns(combined)
        self._ml_columns = list(combined.columns)

        self._fitted = True
        self.fit_report = {
            "fitted_on_rows": int(len(train_df)),
            "ml_feature_count": len(self._ml_columns),
            "numeric_feature_count": len(numeric.columns),
            "onehot_column_count": len(self._onehot_columns),
            "ml_feature_columns": self._ml_columns,
            "numeric_medians": {
                col: (None if pd.isna(v) else float(v))
                for col, v in self._numeric_medians.items()
            },
            "onehot_levels": self._onehot_levels,
            "order_aggregate_n": int(len(self._order_aggregates)),
            "constant_or_allna_columns_dropped": int(
                len(combined.columns) - len(self._ml_columns)
            ),
        }
        logger.info(
            "Preprocessor fitted on %d train rows; %d ML features",
            len(train_df),
            len(self._ml_columns),
        )
        return self

    def transform(self, df: pd.DataFrame, split_name: str = "unknown") -> ProcessedFrame:
        """Apply the fitted preprocessing to any DataFrame (val/test/train)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        _require_columns(df, _required_sources())

        work = df.copy()
        target = (
            pd.Series(
                pd.to_numeric(work.get(DATACO_TARGET_COLUMN), errors="coerce"),
                index=work.index,
                name="target",
            )
            if DATACO_TARGET_COLUMN in work.columns
            else None
        )

        if self.config.drop_empty_target and target is not None:
            keep = target.notna()
            work = work[keep]
            if target is not None:
                target = target[keep]

        if "row_id" in work.columns:
            row_ids = work["row_id"].astype("int64")
        else:
            row_ids = pd.Series(work.index, index=work.index, name="row_id")

        raw_features = build_raw_ml_features(work, self._order_aggregates)
        numeric = raw_features.apply(pd.to_numeric, errors="coerce")

        numeric, negative_before = _mask_impossible_negatives(numeric)
        if negative_before:
            logger.warning(
                "%s: replaced %d impossible negative numeric values with NaN",
                split_name,
                negative_before,
            )

        imputed = numeric.fillna(self._numeric_medians)
        scaled = (
            pd.DataFrame(
                self._scaler.transform(imputed),
                columns=imputed.columns,
                index=imputed.index,
            )
            if self._scaler is not None
            else imputed
        )

        dummies = _make_dummies(work, self._onehot_levels)
        dummies = dummies.reindex(columns=self._onehot_columns, fill_value=0)

        combined = pd.concat([scaled, dummies], axis=1)
        combined = combined.reindex(columns=self._ml_columns, fill_value=0.0)

        ml_features = combined.copy()
        metadata = _build_metadata(work, row_ids)
        return ProcessedFrame(
            split_name=split_name,
            ml_features=ml_features,
            metadata=metadata,
            target=target.to_numpy(dtype=float) if target is not None else None,
        )

    def transform_all(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
    ) -> dict[str, ProcessedFrame]:
        """Convenience: fit on train and transform all three splits."""
        self.fit(train)
        return {
            name: self.transform(frame, split_name=name)
            for name, frame in {
                "train": train,
                "validation": validation,
                "test": test,
            }.items()
        }

    # -- helpers -------------------------------------------------------------

    def config_snapshot(self) -> dict[str, Any]:
        return self.config.to_dict()


# ---------------------------------------------------------------------------
# module-level helpers
# ---------------------------------------------------------------------------


def _required_sources() -> list[str]:
    return list(
        {
            "Order Item Quantity",
            "Order Item Product Price",
            "Product Price",
            "Order Item Discount",
            "Order Item Discount Rate",
            "Order Item Profit Ratio",
            "Order Item Total",
            "Days for shipment (scheduled)",
            "order date (DateOrders)",
            SPLIT_GROUP_KEY,
        }
        | set(DATACO_CATEGORICAL_FEATURES)
    )


def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"DataCo preprocessor missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )


def _fit_onehot_levels(df: pd.DataFrame) -> dict[str, list[str]]:
    """Fit (from df) the ordered top-k levels per categorical column."""
    levels: dict[str, list[str]] = {}
    for column in DATACO_CATEGORICAL_FEATURES:
        series = df[column].astype("string").fillna("").str.strip()
        counts = series.value_counts(dropna=True)
        levels[column] = [str(v) for v in counts.head(MAX_ONEHOT_CARDINALITY).index]
    return levels


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.DataFrame(index=df.index)
    for column in DATACO_CATEGORICAL_FEATURES:
        encoded[column] = df[column].astype("string").fillna("").str.strip()
    return encoded


def _make_dummies(
    df: pd.DataFrame,
    levels: dict[str, list[str]],
) -> pd.DataFrame:
    encoded = _encode_categoricals(df)
    blocks: list[pd.DataFrame] = []
    for column in encoded.columns:
        fitted_levels = levels[column]
        series = encoded[column]
        placeholder_series = series.where(series.isin(fitted_levels), other=_OTHER_BUCKET)
        if _OTHER_BUCKET not in fitted_levels:
            fitted_levels = fitted_levels + [_OTHER_BUCKET]
        dummies = pd.get_dummies(placeholder_series.astype(str), prefix=column)
        level_columns = [f"{column}_{level}" for level in fitted_levels]
        dummies = dummies.reindex(columns=level_columns, fill_value=0)
        blocks.append(dummies.astype("float64"))
    return pd.concat(blocks, axis=1)


def _drop_constant_or_allna_columns(frame: pd.DataFrame) -> pd.DataFrame:
    allna = set(frame.columns[frame.isna().all()])
    constant = {c for c in frame.columns if frame[c].nunique(dropna=True) <= 1}
    dropped = sorted(allna | constant)
    if dropped:
        logger.info("Dropped constant/all-NaN columns from ML features: %s", dropped)
    return frame.drop(columns=dropped)


def _build_metadata(work: pd.DataFrame, row_ids: pd.Series) -> pd.DataFrame:
    columns = [c for c in METADATA_COLUMNS if c in work.columns]
    metadata = work[columns].copy()
    metadata["row_id"] = row_ids.reindex(metadata.index).astype("int64")
    return metadata


def feature_catalogue() -> list[dict[str, str]]:
    return FEATURE_CATALOG