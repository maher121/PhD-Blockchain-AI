"""Basic data cleaning and preprocessing for the supply-chain dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_SCHEMA

logger = logging.getLogger(__name__)


def preprocess_supply_chain_data(
    df: pd.DataFrame,
    drop_nan_threshold: float = 0.5,
) -> pd.DataFrame:
    """Perform deterministic, basic cleaning of the raw dataset.

    Steps
    -----
    * Drop duplicate transaction ids (keeping the first occurrence).
    * Drop rows where more than ``drop_nan_threshold`` fraction of values
      are missing.
    * Coerce ID columns to string and forward-fill durable ID-like strings.
    * Drop rows with non-positive quantity, unit_price or total_amount.

    Parameters
    ----------
    df : Raw supply-chain DataFrame (`RAW_SCHEMA`).
    drop_nan_threshold : Max allowed fraction of missing values per row.

    Returns
    -------
    A cleaned pandas.DataFrame (a copy; the input is never mutated).
    """
    if df is None or df.empty:
        raise ValueError("Empty DataFrame provided to preprocessing.")

    work = df.copy()

    work = work.drop_duplicates(subset=["transaction_id"], keep="first")

    previous_count = len(work)
    work = work.dropna(thresh=int(drop_nan_threshold * work.shape[1]))
    if len(work) < previous_count:
        logger.info("Dropped %d rows with excessive missing values", previous_count - len(work))

    id_columns = ["participant_id", "product_id", "order_id"]
    for column in id_columns:
        work[column] = work[column].astype("string").ffill()

    numeric_columns = ["quantity", "unit_price", "total_amount"]
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    previous_count = len(work)
    positive_mask = (work["quantity"] > 0) & (work["unit_price"] > 0) & (
        work["total_amount"] > 0
    )
    work = work[positive_mask].copy()
    if len(work) < previous_count:
        logger.info("Dropped %d rows with non-positive numeric values", previous_count - len(work))

    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"]).copy()

    for column, dtype in RAW_SCHEMA.items():
        if column in work.columns:
            work[column] = work[column].astype(dtype)

    logger.info("Preprocessing complete: %d rows", len(work))
    return work


def save_processed_data(
    df: pd.DataFrame,
    output_path: Path | str = PROCESSED_DATA_DIR / "supply_chain_processed.csv",
) -> None:
    """Persist the processed DataFrame to CSV under ``data/processed``."""
    df.to_csv(output_path, index=False)
    logger.info("Saved processed dataset (%d rows) to %s", len(df), output_path)