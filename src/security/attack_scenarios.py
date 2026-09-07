"""Independent controlled data-tampering transformations for V0.4."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def eligible_mask(dataframe: pd.DataFrame, definition: dict[str, Any]) -> pd.Series:
    """Identify rows for which at least one configured source field is usable."""
    masks = [
        _field_eligible(dataframe[column], definition)
        for column in definition.get("candidates", [])
        if column in dataframe
    ]
    if not masks:
        return pd.Series(False, index=dataframe.index, dtype=bool)
    combined = masks[0].copy()
    for mask in masks[1:]:
        combined |= mask
    return combined


def apply_scenario(
    dataframe: pd.DataFrame,
    row_index: Any,
    definition: dict[str, Any],
    severity: str,
    rng: np.random.Generator,
) -> tuple[str, Any, Any]:
    """Apply one scenario to one row and return field, original, modified."""
    candidates = [
        column
        for column in definition.get("candidates", [])
        if column in dataframe
        and bool(_field_eligible(dataframe[column], definition).loc[row_index])
    ]
    if not candidates:
        raise ValueError("Selected row is not eligible for this attack scenario.")
    if definition.get("choose_field_per_record"):
        field = candidates[int(rng.integers(0, len(candidates)))]
    else:
        field = candidates[0]

    original = dataframe.at[row_index, field]
    parameters = definition["severity"][severity]
    transformation = definition["transformation"]
    if transformation in {"multiplicative", "generic_numeric"}:
        factors = parameters["factors"]
        factor = float(factors[int(rng.integers(0, len(factors)))])
        modified = float(original) * factor
        modified = round(modified, 8)
    elif transformation == "additive":
        deltas = parameters["deltas"]
        delta = float(deltas[int(rng.integers(0, len(deltas)))])
        modified = float(original) + delta
        if definition.get("nonnegative_result"):
            modified = max(0.0, modified)
        if np.isclose(float(original), float(modified)):
            modified = float(original) + abs(delta)
    elif transformation == "categorical":
        levels = sorted(dataframe[field].dropna().astype(str).unique().tolist())
        if len(levels) < 2:
            raise ValueError(f"Categorical field {field!r} has fewer than two states.")
        original_text = str(original)
        current = levels.index(original_text)
        step = max(1, int(parameters.get("steps", 1)))
        modified = levels[(current + step) % len(levels)]
        if modified == original_text:
            modified = levels[(current + 1) % len(levels)]
    elif transformation == "timestamp":
        timestamp = pd.to_datetime(original)
        offsets = parameters["offset_hours"]
        offset = int(offsets[int(rng.integers(0, len(offsets)))])
        modified = timestamp + pd.Timedelta(hours=offset)
        if isinstance(original, str):
            modified = modified.isoformat(sep=" ")
    else:
        raise ValueError(f"Unsupported attack transformation: {transformation!r}")

    if pd.api.types.is_integer_dtype(dataframe[field].dtype) and isinstance(
        modified, (float, np.floating)
    ):
        dataframe[field] = dataframe[field].astype(float)
    dataframe.at[row_index, field] = modified
    return field, original, modified


def _field_eligible(series: pd.Series, definition: dict[str, Any]) -> pd.Series:
    transformation = definition.get("transformation")
    if transformation in {"multiplicative", "generic_numeric", "additive"}:
        numeric = pd.to_numeric(series, errors="coerce")
        mask = numeric.notna() & np.isfinite(numeric)
        if definition.get("positive_only"):
            mask &= numeric > 0
        return mask
    if transformation == "timestamp":
        return pd.to_datetime(series, errors="coerce").notna()
    if transformation == "categorical":
        non_null = series.notna()
        return non_null & (series[non_null].astype(str).nunique() >= 2)
    return pd.Series(False, index=series.index, dtype=bool)
