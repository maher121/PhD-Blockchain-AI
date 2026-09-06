"""Leakage-aware, deterministic train/validation/test splitting (V0.2).

Leakage-prevention contract
---------------------------
Preprocessing statistics (imputation values, scaling parameters, one-hot
levels, and any train-derived engineering like order-level aggregates) must
be fitted on the TRAINING split only and then *transformed* onto validation
and test. Splitting is therefore the FIRST data-transformation step.

This module enforces the split and verifies that no identity key shared by
``LEAKAGE_IDENTITY_KEYS`` (e.g. the same Order Id) straddles two splits when
the configured strategy is group-aware.

Strategies
----------
* ``order_grouped`` (default) : rows of the same ``Order Id`` are kept in the
  same split (group-aware). Unique orders are shuffled with a fixed seed and
  greedily assigned to buckets so the final ROW ratio matches ``ratios``.
  Prevents order-level identity leakage.
* ``random``                  : plain row-level shuffle with a fixed seed.
  Faster, but the same Order Id can land in multiple splits.
* ``chronological``           : rows sorted by ``order date (DateOrders)`` and
  split by row boundaries (no shuffling). Appropriate for strict
  time-ordered evaluation; no random component.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DATACO_TARGET_COLUMN,
    GLOBAL_SEED,
    LEAKAGE_IDENTITY_KEYS,
    SPLIT_GROUP_KEY,
    SPLIT_RATIOS,
    SPLIT_STRATEGY,
)

logger = logging.getLogger(__name__)

STRATEGIES: tuple[str, ...] = ("random", "order_grouped", "chronological")


@dataclass
class SplitResult:
    """Container for the split outputs and their leakage report."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    report: dict[str, Any] = field(default_factory=dict)

    def frames(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def _validate_ratios(ratios: dict[str, float]) -> None:
    required = {"train", "validation", "test"}
    if set(ratios) != required:
        raise ValueError(f"ratios must contain exactly {sorted(required)}, got {sorted(ratios)}")
    total = sum(ratios.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def check_identity_overlap(
    frames: dict[str, pd.DataFrame],
    keys: tuple[str, ...] = LEAKAGE_IDENTITY_KEYS,
) -> dict[str, dict[str, int]]:
    """Count how many values of each identity key appear in >1 split.

    Returns ``{key: {pair: overlap_count}}`` where pair ranges over
    ``train<->validation``, ``train<->test``, ``validation<->test``.
    A nonzero overlap for ``Order Id`` is identity leakage: the same
    `order` is used both to fit and to evaluate.
    """
    names = list(frames)
    report: dict[str, dict[str, int]] = {}
    for key in keys:
        present = [name for name in names if key in frames[name].columns]
        if len(present) < 2:
            continue
        overlaps: dict[str, int] = {}
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = present[i], present[j]
                set_a = set(frames[a][key].dropna().astype(str).unique())
                set_b = set(frames[b][key].dropna().astype(str).unique())
                overlaps[f"{a}_vs_{b}"] = int(len(set_a & set_b))
        report[key] = overlaps
    return report


def _chronological_split(
    df: pd.DataFrame,
    ratios: dict[str, float],
    time_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if time_column not in df.columns:
        raise ValueError(
            f"Chronological split requires column {time_column!r} in the dataset."
        )
    work = df.sort_values(time_column).reset_index(drop=True)
    n = len(work)
    n_train = int(round(n * ratios["train"]))
    n_val = int(round(n * ratios["validation"]))
    train = work.iloc[:n_train]
    validation = work.iloc[n_train : n_train + n_val]
    test = work.iloc[n_train + n_val :]
    return train, validation, test


def _capped_order_group_split(
    df: pd.DataFrame,
    ratios: dict[str, float],
    seed: int,
    group_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group-aware split: same group never straddles two splits.

    Order keys are shuffled with the seeded RNG and greedily assigned to the
    bucket with the fewest accumulated rows so the final row counts honour
    ``ratios`` as closely as possible.
    """
    if group_key not in df.columns:
        raise ValueError(f"Group-aware split requires group key {group_key!r}.")

    key_series = df[group_key].astype(str)
    if key_series.isna().any():
        raise ValueError(
            f"Group key {group_key!r} contains missing values; cannot split "
            "leakage-free by an incomplete key."
        )

    rng = np.random.default_rng(seed)
    unique_groups = key_series.unique()
    shuffled = rng.choice(unique_groups, size=len(unique_groups), replace=False)

    group_sizes = key_series.value_counts(dropna=False, sort=False).to_dict()
    buckets: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    inflated: dict[str, float] = {k: 0.0 for k in buckets}
    total_n = len(df)

    for order_id in shuffled:
        size = float(group_sizes[order_id])
        # Assign to the bucket whose current fraction is the farthest below
        # its target fraction.
        target_idx = min(
            buckets,
            key=lambda b: (inflated[b] / total_n) - ratios[b],
        )
        buckets[target_idx].append(order_id)
        inflated[target_idx] += size

    group_set = {k: set(v) for k, v in buckets.items()}
    masks = {
        k: key_series.isin(group_set[k])
        for k in buckets
    }
    return df[masks["train"]], df[masks["validation"]], df[masks["test"]]


def split_dataset(
    df: pd.DataFrame,
    ratios: dict[str, float] = SPLIT_RATIOS,
    strategy: str = SPLIT_STRATEGY,
    seed: int = GLOBAL_SEED,
    group_key: str = SPLIT_GROUP_KEY,
    time_column: str = "order date (DateOrders)",
) -> SplitResult:
    """Deterministically split a DataFrame into train/validation/test.

    Parameters
    ----------
    df : Raw DataCo DataFrame (all rows; splitting happens BEFORE any
        preprocessing statistic is computed).
    ratios : ``{"train": 0.70, "validation": 0.15, "test": 0.15}``.
    strategy : one of "random", "order_grouped" (default), "chronological".
    seed : Fixed RNG seed for strategies with a random component.
    group_key : Identity key used by the "order_grouped" strategy.
    time_column : Sort key for the "chronological" strategy.

    Returns
    -------
    SplitResult carrying ``train``/``validation``/``test`` frames and a
    ``report`` describing sizes, strategy and identity-overlap checks.
    """
    if df is None or df.empty:
        raise ValueError("Cannot split an empty dataset.")
    _validate_ratios(ratios)
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown split strategy {strategy!r}; expected {STRATEGIES}.")

    if strategy == "chronological":
        train, validation, test = _chronological_split(df, ratios, time_column)
    elif strategy == "order_grouped":
        train, validation, test = _capped_order_group_split(df, ratios, seed, group_key)
    else:
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(shuffled)
        n_train = int(round(n * ratios["train"]))
        n_val = int(round(n * ratios["validation"]))
        train = shuffled.iloc[:n_train]
        validation = shuffled.iloc[n_train : n_train + n_val]
        test = shuffled.iloc[n_train + n_val :]

    frames = {"train": train, "validation": validation, "test": test}
    overlap = check_identity_overlap(frames)

    sizes = {name: int(len(frame)) for name, frame in frames.items()}
    report: dict[str, Any] = {
        "strategy": strategy,
        "seed": int(seed),
        "ratios": dict(ratios),
        "sizes": sizes,
        "size_percentages": {
            name: round(size / len(df), 6) for name, size in sizes.items()
        },
        "identity_overlap_checks": overlap,
        "leakage_warning": bool(
            overlap.get("Order Id", {}).get("train_vs_test", 0) > 0
        ),
        "target_balance": _target_balance(frames),
    }

    if report["leakage_warning"]:
        logger.warning(
            "Train/test identity overlap detected for 'Order Id' "
            "(strategy=%s): %s",
            strategy,
            overlap["Order Id"],
        )

    logger.info(
        "Split complete (strategy=%s, seed=%s): train=%d, val=%d, test=%d",
        strategy,
        seed,
        sizes["train"],
        sizes["validation"],
        sizes["test"],
    )
    return SplitResult(train=train, validation=validation, test=test, report=report)


def _target_balance(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    if DATACO_TARGET_COLUMN not in next(iter(frames.values())).columns:
        return {}
    return {
        name: {
            "total": int(frame[DATACO_TARGET_COLUMN].notna().sum()),
            "positive": int(
                pd.to_numeric(frame[DATACO_TARGET_COLUMN], errors="coerce").fillna(0).eq(1).sum()
            ),
            "positive_rate": float(
                round(
                    pd.to_numeric(frame[DATACO_TARGET_COLUMN], errors="coerce")
                    .fillna(0)
                    .eq(1)
                    .mean(),
                    6,
                )
            ),
        }
        for name, frame in frames.items()
    }