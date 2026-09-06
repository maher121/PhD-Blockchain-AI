"""Controlled feature engineering for the DataCo table (Prototype V0.2).

Every engineered feature is defined once in ``FEATURE_CATALOG`` with a clear
name, the exact source columns, a definition, and a documented rationale. The
catalog is the single source of truth used to (a) build the ML feature matrix
and (b) generate ``docs/data_dictionary.md``.

Leakage rules enforced here
---------------------------
* Order-level features (``item_count_per_order``, ``order_total_value``) are
  computed only from the TRAINING split and merged by ``Order Id`` onto
  validation/test frames -- never re-computed on the union of all data.
* Outcome-adjacent columns (``Delivery Status``, ``Days for shipping (real)``,
  ``Order Status``) are excluded from the ML feature set.
* Temporal features derive from the ORDER date, which is available at
  prediction time (unlike the shipping date).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DATACO_CATEGORICAL_FEATURES,
    DATACO_DATETIME_COLUMNS,
    SPLIT_GROUP_KEY,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature catalogue structure:
#   name          : engineered feature name (column)
#   source        : original DataCo column(s) it is derived from
#   definition    : precise formula / computation
#   rationale     : why it is kept for ML
#   category      : one of transaction/order/product/customer/shipping/temporal
#   role          : "ml" (used as an ML feature) or "metadata"
# ---------------------------------------------------------------------------

FEATURE_CATALOG: list[dict[str, str]] = [
    {"name": "order_item_quantity", "source": "Order Item Quantity", "definition": "Quantity of the product ordered on the item line.", "rationale": "Core transaction scale signal.", "category": "transaction", "role": "ml"},
    {"name": "order_item_product_price", "source": "Order Item Product Price", "definition": "Unit price paid for the item on this order line.", "rationale": "Price-point information for value-based risk.", "category": "transaction", "role": "ml"},
    {"name": "product_price", "source": "Product Price", "definition": "Catalogue/list unit price of the product.", "rationale": "Basis for price deviations.", "category": "product", "role": "ml"},
    {"name": "price_delta", "source": "Order Item Product Price, Product Price", "definition": "order_item_product_price - product_price.", "rationale": "Discount/deal spread between catalogue and paid price.", "category": "transaction", "role": "ml"},
    {"name": "order_item_discount", "source": "Order Item Discount", "definition": "Absolute discount applied on the item line.", "rationale": "Deal magnitude correlates with margin pressure.", "category": "transaction", "role": "ml"},
    {"name": "order_item_discount_rate", "source": "Order Item Discount Rate", "definition": "Relative discount rate applied on the item line.", "rationale": "Relative price concession.", "category": "transaction", "role": "ml"},
    {"name": "order_item_profit_ratio", "source": "Order Item Profit Ratio", "definition": "Profit ratio of the item line.", "rationale": "Margin-level signal.", "category": "transaction", "role": "ml"},
    {"name": "order_item_total", "source": "Order Item Total", "definition": "Line total (quantity x product price, pre-discount).", "rationale": "Monetary size of the line.", "category": "transaction", "role": "ml"},
    {"name": "days_schedule", "source": "Days for shipment (scheduled)", "definition": "Scheduled fulfilment time in days (known at order time).", "rationale": "Planned service-level; is NOT the realised delay.", "category": "shipping", "role": "ml"},
    {"name": "item_count_per_order", "source": "Order Id", "definition": "Number of item lines grouped under the same Order Id (train-derived).", "rationale": "Order complexity signal.", "category": "order", "role": "ml"},
    {"name": "order_total_value", "source": "Order Item Total, Order Id", "definition": "Sum of line totals per Order Id (train-derived).", "rationale": "Order-level monetary size.", "category": "order", "role": "ml"},
    {"name": "year", "source": "order date (DateOrders)", "definition": "Calendar year of the order date.", "rationale": "Temporal trend (2015-2017).", "category": "temporal", "role": "ml"},
    {"name": "month", "source": "order date (DateOrders)", "definition": "Calendar month (1-12) of the order date.", "rationale": "Seasonality.", "category": "temporal", "role": "ml"},
    {"name": "day", "source": "order date (DateOrders)", "definition": "Day of month (1-31) of the order date.", "rationale": "Day-level periodicity.", "category": "temporal", "role": "ml"},
    {"name": "day_of_week", "source": "order date (DateOrders)", "definition": "Day of week (0=Monday .. 6=Sunday).", "rationale": "Weekly cadence.", "category": "temporal", "role": "ml"},
    {"name": "hour", "source": "order date (DateOrders)", "definition": "Hour of day (0-23) of the order date.", "rationale": "Intra-day ordering behaviour.", "category": "temporal", "role": "ml"},
    {"name": "is_weekend", "source": "order date (DateOrders)", "definition": "1 if the order date falls on Saturday/Sunday.", "rationale": "Weekend vs weekday behaviour.", "category": "temporal", "role": "ml"},
]

NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "order_item_quantity",
    "order_item_product_price",
    "product_price",
    "price_delta",
    "order_item_discount",
    "order_item_discount_rate",
    "order_item_profit_ratio",
    "order_item_total",
    "days_schedule",
)

_RAW_TO_FEATURE: dict[str, str] = {
    "Order Item Quantity": "order_item_quantity",
    "Order Item Product Price": "order_item_product_price",
    "Product Price": "product_price",
    "Order Item Discount": "order_item_discount",
    "Order Item Discount Rate": "order_item_discount_rate",
    "Order Item Profit Ratio": "order_item_profit_ratio",
    "Order Item Total": "order_item_total",
    "Days for shipment (scheduled)": "days_schedule",
}

ORDER_LEVEL_COLUMNS: tuple[str, ...] = ("item_count_per_order", "order_total_value")


def compute_temporal_features(df: pd.DataFrame, date_column: str = DATACO_DATETIME_COLUMNS[0]) -> pd.DataFrame:
    """Add documented temporal features based on the order date."""
    dates = pd.to_datetime(df[date_column], errors="coerce")
    out = pd.DataFrame(index=df.index)
    out["year"] = dates.dt.year.astype("float64")
    out["month"] = dates.dt.month.astype("float64")
    out["day"] = dates.dt.day.astype("float64")
    out["day_of_week"] = dates.dt.dayofweek.astype("float64")
    out["hour"] = dates.dt.hour.astype("float64")
    out["is_weekend"] = (dates.dt.dayofweek >= 5).astype("float64")
    return out


def compute_order_level(train: pd.DataFrame, order_key: str = SPLIT_GROUP_KEY) -> pd.DataFrame:
    """Compute order-level aggregates from TRAIN rows only.

    Returns a DataFrame indexed by ``order_key`` with the columns
    ``item_count_per_order`` and ``order_total_value`` that can be merged
    onto validation/test frames for leakage-free reuse.
    """
    item_count = train.groupby(order_key).size()
    order_total = train.groupby(order_key)["Order Item Total"].sum()
    aggregates = pd.DataFrame(
        {
            "item_count_per_order": item_count.astype("float64"),
            "order_total_value": order_total.astype("float64"),
        }
    )
    aggregates.index.name = order_key
    return aggregates


def append_order_level(
    features: pd.DataFrame,
    order_aggregates: pd.DataFrame,
    order_key_values: pd.Series,
) -> pd.DataFrame:
    """Attach train-fitted order aggregates, aligned by the order-key values.

    ``order_key_values`` must be a Series (same index as ``features``) holding
    the order key of each row. Keys are matched preserving dtype, so both
    integer (real DataCo) and string (fixture) order ids work. Orders unseen
    in training get NaN, which the preprocessor imputes with train-derived
    values -- exactly the behaviour a fresh order would face at inference time.
    """
    features = features.copy()
    features["item_count_per_order"] = (
        order_key_values.map(order_aggregates["item_count_per_order"]).astype("float64")
    )
    features["order_total_value"] = (
        order_key_values.map(order_aggregates["order_total_value"]).astype("float64")
    )
    return features


def build_raw_ml_features(
    frame: pd.DataFrame,
    order_aggregates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the raw (unencoded, unscaled) numerical ML feature matrix.

    ``order_aggregates`` must come from ``compute_order_level(train)`` to
    respect the fit-on-TRAIN-only leakage contract.
    """
    missing = [
        column for column in _RAW_TO_FEATURE
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing source columns {missing} for feature engineering.")

    features = pd.DataFrame(index=frame.index)
    for raw_column, feature_name in _RAW_TO_FEATURE.items():
        features[feature_name] = pd.to_numeric(frame[raw_column], errors="coerce").astype("float64")

    features["price_delta"] = (
        features["order_item_product_price"] - features["product_price"]
    ).replace([np.inf, -np.inf], np.nan)

    temporal = compute_temporal_features(frame)
    features = features.join(temporal)

    if order_aggregates is None:
        order_aggregates = compute_order_level(frame)
    order_key = SPLIT_GROUP_KEY
    order_key_values = pd.Series(
        frame[order_key].to_numpy(), index=frame.index
    )
    features = append_order_level(features, order_aggregates, order_key_values)

    return features



def categorical_source_columns() -> list[str]:
    """Original DataCo columns of the categorical features (one-hot encoded)."""
    return list(DATACO_CATEGORICAL_FEATURES)


def feature_catalog_df() -> pd.DataFrame:
    """Return the feature catalogue as a DataFrame (for docs/notebooks)."""
    return pd.DataFrame(FEATURE_CATALOG)