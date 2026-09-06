"""Synthetic supply-chain dataset generation for Prototype V0.1.

The generator produces a realistic, fully reproducible CSV that mimics a
pharmaceutical / retail supply chain order ledger.

The ``known_anomaly`` column is a *synthetic/controlled anomaly* label
injected by this generator for evaluation purposes only. It does NOT
represent a genuine cybersecurity attack label.

Anomaly types injected
-----------------------
* quantity_outlier   - extreme order quantities far from the product norm.
* price_outlier      - unit prices far from the product norm.
* delayed_shipment   - shipping time far beyond the expected window.

No network/cyber-attack scenarios are simulated in V0.1.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_SYNTHETIC_DATA_FILE,
    GLOBAL_SEED,
    RAW_SCHEMA,
)

logger = logging.getLogger(__name__)

ANOMALY_TYPES: tuple[str, ...] = ("quantity_outlier", "price_outlier", "delayed_shipment")

PARTICIPANTS: tuple[str, ...] = (
    "supplier_a",
    "supplier_b",
    "manufacturer_1",
    "distributor_1",
    "wholesaler_1",
    "retailer_1",
)

REGIONS: tuple[str, ...] = ("EMEA", "APAC", "AMER", "LATAM", "MENA")

STATUSES: tuple[str, ...] = (
    "completed",
    "in_transit",
    "pending",
    "delayed",
    "cancelled",
)


def _product_base_price() -> tuple[np.ndarray, np.ndarray]:
    """Return fixed (product_id, base_unit_price) pairs used repeatedly."""
    rng = np.random.default_rng(GLOBAL_SEED + 1)
    product_count = 60
    product_ids = np.array([f"PROD-{i:03d}" for i in range(product_count)], dtype=object)
    base_prices = rng.uniform(5.0, 500.0, size=product_count)
    return product_ids, base_prices


def _describe_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Add a human-readable anomaly description column."""
    df["anomaly_type"] = df["known_anomaly"].map(
        {i: label for i, label in enumerate(ANOMALY_TYPES, start=1)}
    )
    df["anomaly_type"] = df["anomaly_type"].fillna("none")
    return df


def generate_supply_chain_dataset(
    n_rows: int = 2000,
    seed: int = GLOBAL_SEED,
    anomaly_fraction: float = 0.03,
) -> pd.DataFrame:
    """Generate a synthetic supply-chain order dataset.

    Parameters
    ----------
    n_rows : Number of order records to generate.
    seed : Random seed for reproducibility.
    anomaly_fraction : Fraction of rows flagged as controlled anomalies.

    Returns
    -------
    pandas.DataFrame with columns defined in ``RAW_SCHEMA`` plus the
    ``known_anomaly`` ground-truth flag.
    """
    if not 0.0 <= anomaly_fraction <= 0.5:
        raise ValueError("anomaly_fraction must be in [0, 0.5].")

    rng = np.random.default_rng(seed)
    product_ids, base_prices = _product_base_price()

    transaction_ids = [f"TXN-{i:06d}" for i in range(1, n_rows + 1)]
    order_ids = [f"ORD-{i:06d}" for i in range(1, n_rows + 1)]
    participant_ids = rng.choice(PARTICIPANTS, size=n_rows)
    product_index = rng.integers(0, len(product_ids), size=n_rows)
    products = pd.Series(product_ids[product_index], dtype=object).to_numpy()
    base_price = base_prices[product_index]

    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-12-31")
    timestamps_ns = rng.integers(start.value // 1_000, end.value // 1_000, size=n_rows)
    timestamps = pd.to_datetime(timestamps_ns * 1_000, unit="ns")

    base_quantity = rng.choice([10, 20, 25, 50, 100, 200, 500], size=n_rows)
    quantity = base_quantity.astype(float)
    unit_price = base_price * rng.uniform(0.9, 1.1, size=n_rows)

    shipping_days = np.clip(rng.normal(5.0, 1.5, size=n_rows), 0.5, 15.0)
    regions = rng.choice(REGIONS, size=n_rows)
    statuses = rng.choice(STATUSES, size=n_rows, p=[0.6, 0.1, 0.12, 0.08, 0.1])

    anomaly_flags = rng.random(n_rows) < anomaly_fraction
    n_anomalies = int(anomaly_flags.sum())
    anomaly_types = rng.integers(1, len(ANOMALY_TYPES) + 1, size=n_anomalies)

    quantity[anomaly_flags] *= rng.choice([3.0, 5.0, 8.0, 12.0], size=n_anomalies)
    unit_price[anomaly_flags] *= rng.choice([1.5, 2.0, 3.0], size=n_anomalies)
    shipping_days[anomaly_flags] = rng.uniform(25.0, 60.0, size=n_anomalies)

    known_anomaly = np.zeros(n_rows, dtype=int)
    known_anomaly[anomaly_flags] = anomaly_types

    df = pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "participant_id": participant_ids,
            "product_id": products,
            "order_id": order_ids,
            "timestamp": timestamps,
            "quantity": quantity,
            "unit_price": unit_price,
            "total_amount": quantity * unit_price,
            "shipping_days": shipping_days,
            "location_region": regions,
            "transaction_status": statuses,
            "known_anomaly": known_anomaly,
        }
    )
    df = df.astype({key: value for key, value in RAW_SCHEMA.items()})
    return _describe_anomalies(df)


def write_supply_chain_dataset(
    df: pd.DataFrame,
    output_path: Path | str = DEFAULT_SYNTHETIC_DATA_FILE,
) -> None:
    """Write the synthetic dataset to ``output_path`` as CSV."""
    df.to_csv(output_path, index=False)
    logger.info("Wrote synthetic dataset (%d rows) to %s", len(df), output_path)


if __name__ == "__main__":
    from src.logging_config import setup_logging

    setup_logging()
    dataset = generate_supply_chain_dataset()
    write_supply_chain_dataset(dataset)
    print(f"Generated {len(dataset)} rows -> {DEFAULT_SYNTHETIC_DATA_FILE}")