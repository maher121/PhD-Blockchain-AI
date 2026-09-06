"""Feature engineering producing the numeric matrix consumed by the AI module.

All features are computed from the raw order fields so the notebook and the
pipeline never expose the raw dataframe directly to the anomaly detector.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_FEATURES: tuple[str, ...] = (
    "quantity",
    "unit_price",
    "total_amount",
    "shipping_days",
)


def build_feature_matrix(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...] = BASE_FEATURES,
) -> pd.DataFrame:
    """Build a numeric feature matrix from a processed supply-chain frame.

    Derived features computed on top of the base order columns:

    * ``amount_per_unit``          - total_amount / quantity
    * ``log_total_amount``         - log1p(total_amount)
    * ``hour_of_day``              - cyclic-aware hour of the transaction
    * ``day_of_week``              - day of week (0 = Monday)
    * ``is_weekend``               - 1 if Saturday/Sunday
    * ``avg_quantity_by_product``  - product-level quantity mean (context)
    * ``quantity_ratio_to_mean``   - quantity / product-level mean

    Parameters
    ----------
    df : Processed supply-chain DataFrame.
    feature_columns : Base numeric columns to include.

    Returns
    -------
    pandas.DataFrame with index aligned to ``df`` (same order).
    """
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing base feature columns: {missing}")

    features = pd.DataFrame(index=df.index, dtype=float)
    for column in feature_columns:
        features[column] = pd.to_numeric(df[column], errors="coerce").astype(float)

    features["amount_per_unit"] = (
        features["total_amount"] / features["quantity"]
    ).replace([np.inf, -np.inf], np.nan)

    features["log_total_amount"] = np.log1p(features["total_amount"])

    timestamp = pd.to_datetime(df["timestamp"], errors="coerce")
    features["hour_of_day"] = timestamp.dt.hour.astype("float64")
    features["day_of_week"] = timestamp.dt.dayofweek.astype("float64")
    features["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype("float64")

    product_mean = df.groupby("product_id")["quantity"].transform("mean")
    features["quantity_ratio_to_mean"] = (
        features["quantity"] / product_mean.astype("float64")
    ).replace([np.inf, -np.inf], np.nan)

    features = features.replace([np.inf, -np.inf], np.nan)

    row_threshold = max(1, int(0.5 * features.shape[1]))
    features = features.dropna(axis=0, thresh=row_threshold)
    remaining = features.dropna(axis=1)
    if remaining.shape[1] < features.shape[1]:
        dropped = [column for column in features.columns if column not in remaining.columns]
        logger.warning("Dropped feature columns with excessive missing values: %s", dropped)
        features = remaining

    if features.isna().any().any():
        features = features.fillna(features.median(numeric_only=True))

    return features


def select_features(
    feature_matrix: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Select a feature subset (used for light feature-trials in later phases)."""
    if columns is None:
        return feature_matrix
    missing = [column for column in columns if column not in feature_matrix.columns]
    if missing:
        raise ValueError(f"Unknown feature columns: {missing}")
    return feature_matrix[columns].copy()