"""Reusable dataset audit for the DataCo Smart Supply Chain table.

The audit produces a machine-readable report (JSON + CSV) under
``results/data_audit/`` describing schema, missing values, duplicates,
categorical cardinality, numeric statistics, identifier/datetime candidates,
constant/near-constant columns and potential leakage columns. It also emits a
small set of diagnostic figures under ``results/data_audit/figures/``.

The audit is *inspection only*: it never modifies or deletes data. Decisions
from the audit are consumed by the preprocessing pipeline, which then decides
whether a filter/transform is justified.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    AUDIT_FIGURES_DIR,
    DATA_AUDIT_DIR,
    DATACO_OUTCOME_COLUMNS,
    DATACO_TARGET_COLUMN,
    NEGATIVE_VALUE_COLUMNS,
)

logger = logging.getLogger(__name__)

# Thresholds used to classify columns (documented in the feature summary).
HIGH_CARDINALITY_NUNIQUE: int = 100
HIGH_CARDINALITY_RATIO: float = 0.05
CONSTANT_TOLERANCE: float = 0.005  # near-constant: <=0.5% distinct values

IDENTIFIER_HINTS: tuple[str, ...] = (
    "id",
    "email",
    "password",
    "zipcode",
    "zip code",
    "lat",
    "long",
    "image",
    "description",
    "street",
    "fname",
    "lname",
    "city",
    "country",
    "state",
)

AGGREGATE_HINTS: tuple[str, ...] = (
    "per order",
    "per customer",
    "sales per",
    "benefit per",
)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_string(series: pd.Series) -> bool:
    dtype = series.dtype
    return dtype == object or isinstance(dtype, pd.StringDtype) or str(dtype) == "string"


def _is_datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


def _inheritable_float(series: pd.Series) -> pd.Series:
    """Coerce a numeric column to float without raising on object dtypes."""
    if not _is_numeric(series):
        return pd.to_numeric(series, errors="coerce").astype("float64")
    return series.astype("float64")


def _classify_role(column: str, series: pd.Series, n_rows: int) -> str:
    """Heuristic column-role classification used in the audit summary only."""
    lowered = column.lower()
    if _is_datetime(series):
        return "datetime"
    if column in DATACO_OUTCOME_COLUMNS:
        return "outcome"
    if any(hint in lowered for hint in IDENTIFIER_HINTS) or series.nunique() == n_rows:
        if any(hint in lowered for hint in ("per order", "per customer", "sales per")):
            return "aggregate"
        return "identifier"
    if any(hint in lowered for hint in AGGREGATE_HINTS):
        return "aggregate"
    if _is_numeric(series):
        return "numeric"
    return "categorical"


def _potential_datetime_columns(df: pd.DataFrame) -> dict[str, str]:
    """Return columns whose name or content suggests a datetime field."""
    candidates: dict[str, str] = {}
    for column in df.columns:
        if _is_datetime(df[column]):
            candidates[column] = "dtype_is_datetime"
            continue
        if "date" in column.lower():
            parsed = pd.to_datetime(df[column].head(500), errors="coerce")
            if parsed.notna().mean() > 0.9:
                candidates[column] = "name_hint_and_parses"
            else:
                candidates[column] = "name_hint_only"
    return candidates


def _quality_checks(
    df: pd.DataFrame,
    negative_columns: tuple[str, ...] = NEGATIVE_VALUE_COLUMNS,
) -> dict[str, Any]:
    """Detect classification issues: invalid numbers/dates and impossible values."""
    invalid_numeric: dict[str, int] = {}
    for column in df.columns:
        if _is_numeric(df[column]):
            continue
        if "date" not in column.lower():
            numeric_coerced = pd.to_numeric(df[column], errors="coerce")
            invalid_numeric[column] = int(df[column].notna().sum() - numeric_coerced.notna().sum())

    invalid_dates: dict[str, int] = {}
    for column, _why in _potential_datetime_columns(df).items():
        if not _is_datetime(df[column]):
            invalid_dates[column] = int(df[column].notna().sum() - pd.to_datetime(df[column], errors="coerce").notna().sum())

    negatives: dict[str, int] = {}
    for column in negative_columns:
        if column in df.columns:
            numeric = _inheritable_float(df[column])
            negatives[column] = int((numeric < 0).sum())

    return {
        "invalid_numeric_counts": {k: v for k, v in invalid_numeric.items() if v},
        "invalid_date_counts": {k: v for k, v in invalid_dates.items() if v},
        "negative_value_counts": {k: v for k, v in negatives.items() if v},
        "checks": {
            "negative_value_columns_checked": list(negative_columns),
        },
    }


def leakage_candidates(df: pd.DataFrame) -> list[dict[str, str]]:
    """Flag columns that could leak information into an ML evaluation.

    Reasons covered
    ---------------
    * ``outcome``        : the column is/encodes the target outcome.
    * ``outcome_adjacent``: derived from the realised outcome (known only
      after the event, e.g. real shipping days).
    * ``key_shared``     : identifier repeated across rows (same Order Id /
      customer / product in train AND test would leak identity).
    * ``aggregate``      : owner-computed per-order/per-customer aggregates
      that may incorporate future information.
    """
    flags: list[dict[str, str]] = []
    n_rows = len(df)

    for column in df.columns:
        lowered = column.lower()
        if column in DATACO_OUTCOME_COLUMNS:
            flags.append(
                {
                    "column": column,
                    "reason": "outcome_adjacent"
                    if column != "Delivery Status"
                    else "outcome",
                    "suggestion": "exclude from ML features; traceability metadata only",
                }
            )
            continue
        if column == DATACO_TARGET_COLUMN:
            flags.append(
                {
                    "column": column,
                    "reason": "target",
                    "suggestion": "used as supervised target, never as a feature",
                }
            )
            continue
        nunique = df[column].nunique()
        if nunique > 0 and 0 < nunique < n_rows and any(
            hint in lowered for hint in ("order id", "customer id", "product card id")
        ):
            flags.append(
                {
                    "column": column,
                    "reason": "key_shared",
                    "suggestion": "split by this key or drop from features",
                }
            )
            continue
        if any(hint in lowered for hint in AGGREGATE_HINTS) or "per order" in lowered:
            flags.append(
                {
                    "column": column,
                    "reason": "aggregate",
                    "suggestion": "verify aggregation window; prefer train-derived aggregates",
                }
            )
    return flags


def column_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column summary row (one output row per dataset column)."""
    n_rows = len(df)
    rows: list[dict[str, Any]] = []

    datetime_columns = {
        column: reason
        for column, reason in _potential_datetime_columns(df).items()
        if reason == "dtype_is_datetime"
    }

    for column in df.columns:
        series = df[column]
        numeric = _inheritable_float(series) if _is_numeric(series) else None
        role = _classify_role(column, series, n_rows)

        row: dict[str, Any] = {
            "column": column,
            "dtype": str(series.dtype),
            "role": role,
            "n_non_null": int(series.notna().sum()),
            "n_missing": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean()), 6),
            "n_unique": int(series.nunique()),
            "pct_unique": round(float(series.nunique() / n_rows) if n_rows else 0.0, 6),
            "is_constant": bool(series.nunique() <= 1),
            "is_near_constant": bool(
                n_rows > 0 and series.nunique() <= max(1, int(CONSTANT_TOLERANCE * n_rows))
            ),
            "is_identifier": role == "identifier",
            "is_datetime": column in datetime_columns,
            "is_high_cardinality": bool(
                series.nunique() > HIGH_CARDINALITY_NUNIQUE
                and (series.nunique() / n_rows) > HIGH_CARDINALITY_RATIO
            ),
        }

        if numeric is not None and numeric.notna().any():
            row.update(
                {
                    "mean": float(numeric.mean()),
                    "std": float(numeric.std()),
                    "min": float(numeric.min()),
                    "q1": float(numeric.quantile(0.25)),
                    "median": float(numeric.median()),
                    "q3": float(numeric.quantile(0.75)),
                    "max": float(numeric.max()),
                }
            )
        if _is_string(series) and series.nunique() > 0:
            top = series.value_counts(dropna=True, ascending=False)
            if not top.empty:
                row["top_value"] = str(top.index[0])
                row["top_value_frequency"] = int(top.iloc[0])
        rows.append(row)

    return pd.DataFrame(rows)


def audit_dataframe(
    df: pd.DataFrame,
    source_name: str = "dataset",
    output_dir: Path | str = DATA_AUDIT_DIR,
    save_figures: bool = True,
) -> dict[str, Any]:
    """Run the full V0.2 audit and persist machine-readable artefacts.

    Returns a nested summary dict that is also the content of
    ``dataset_summary.json``.
    """
    if df is None or df.empty:
        raise ValueError("Cannot audit an empty DataFrame.")

    df = df.copy()
    n_rows, n_columns = df.shape
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"

    feature_summary = column_feature_summary(df)
    missing = feature_summary[["column", "n_missing", "missing_pct"]].rename(
        columns={"n_missing": "missing_count", "missing_pct": "missing_percentage"}
    )

    duplicate_rows = int(df.duplicated().sum())
    duplicated_sample = df[df.duplicated(keep=False)].head(5)

    datetime_potential = _potential_datetime_columns(df)
    leakage = leakage_candidates(df)
    quality = _quality_checks(df)

    summary: dict[str, Any] = {
        "source_name": source_name,
        "n_rows": n_rows,
        "n_columns": n_columns,
        "column_names": list(df.columns),
        "data_types": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_summary": {
            "columns_with_missing": int((feature_summary["n_missing"] > 0).sum()),
            "total_cells": int(n_rows * n_columns),
            "total_missing_cells": int(feature_summary["n_missing"].sum()),
            "overall_missing_pct": round(float(feature_summary["n_missing"].sum() / max(1, n_rows * n_columns)), 6),
        },
        "duplicates": {
            "fully_duplicate_rows": duplicate_rows,
            "duplicate_rows_by_key_Order_Item_Id": int(
                df["Order Item Id"].duplicated().sum() if "Order Item Id" in df.columns else 0
            ),
            "duplicate_rows_by_key_Order_Id": int(
                df["Order Id"].duplicated().sum() if "Order Id" in df.columns else 0
            ),
            "duplicate_sample_n": int(len(duplicated_sample)),
        },
        "categorical_columns": [
            column for column in df.columns if _is_string(df[column])
        ],
        "numeric_columns": [column for column in df.columns if _is_numeric(df[column])],
        "potential_datetime_columns": datetime_potential,
        "potential_identifier_columns": [
            column
            for column, role in zip(feature_summary["column"], feature_summary["role"])
            if role == "identifier"
        ],
        "high_cardinality_categorical_columns": [
            column
            for column in feature_summary["column"]
            if feature_summary.loc[feature_summary["column"] == column, "is_high_cardinality"].iloc[0]
        ],
        "constant_columns": [
            column
            for column in feature_summary["column"]
            if feature_summary.loc[feature_summary["column"] == column, "is_constant"].iloc[0]
        ],
        "near_constant_columns": [
            column
            for column in feature_summary["column"]
            if feature_summary.loc[feature_summary["column"] == column, "is_near_constant"].iloc[0]
        ],
        "leakage_candidates": leakage,
        "quality_checks": quality,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, default=_default_json), encoding="utf-8"
    )
    missing.to_csv(output_dir / "missing_values.csv", index=False)
    feature_summary.to_csv(output_dir / "feature_summary.csv", index=False)
    (output_dir / "duplicate_report.json").write_text(
        json.dumps(
            {
                **summary["duplicates"],
                "sample_duplicate_indices": [int(i) for i in duplicated_sample.index],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "leakage_candidates.json").write_text(
        json.dumps(leakage, indent=2), encoding="utf-8"
    )

    if save_figures:
        try:
            save_audit_figures(df, summary, figures_dir)
        except Exception as exc:  # pragma: no cover - figures are best-effort
            logger.warning("Figure generation failed (continuing): %s", exc)

    logger.info("Audit of %r finished: %d rows x %d cols", source_name, n_rows, n_columns)
    return summary


def _default_json(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def _tight_figure(fig: Any) -> None:
    """Best-effort tight layout that must never abort figure generation."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001 - layout heuristics can fail harmlessly
            pass


def save_audit_figures(
    df: pd.DataFrame,
    summary: dict[str, Any],
    figures_dir: Path | str = AUDIT_FIGURES_DIR,
) -> list[Path]:
    """Render the (small, curated) set of audit figures to PNG files."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    missing = pd.read_csv(figures_dir.parent / "missing_values.csv")
    nonzero = missing[missing["missing_count"] > 0].sort_values("missing_percentage", ascending=False)
    if not nonzero.empty and len(nonzero) <= 80:
        fig, ax = plt.subplots(figsize=(11, max(3, 0.35 * len(nonzero))))
        ax.barh(nonzero["column"], nonzero["missing_percentage"] * 100)
        ax.set_xlabel("Missing %")
        ax.set_title("Columns with missing values")
        _tight_figure(fig)
        path = figures_dir / "missing_values.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)

    numeric_columns = summary["numeric_columns"]
    numeric = df[numeric_columns].apply(_inheritable_float)
    if numeric.shape[1]:
        ncols = min(numeric.shape[1], 9)
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 9))
        for ax, column in zip(axes.ravel(), numeric.columns[:ncols]):
            series = numeric[column].dropna()
            ax.hist(series, bins=40, color="steelblue")
            ax.set_title(column, fontsize=8)
            ax.tick_params(axis="x", rotation=45)
        fig.suptitle("Numerical distributions (clipped at 99th percentile)")
        for ax in axes.ravel():
            ax.set_visible(True)
        _tight_figure(fig)
        path = figures_dir / "numeric_distributions.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)

        if numeric.shape[1] > 1 and len(numeric) > 3:
            fig, ax = plt.subplots(figsize=(11, 9))
            corr = numeric.corr(numeric_only=True)
            im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
            ax.set_yticklabels(corr.columns, fontsize=7)
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title("Numerical correlation matrix")
            _tight_figure(fig)
            path = figures_dir / "correlation_matrix.png"
            fig.savefig(path, dpi=110)
            plt.close(fig)
            written.append(path)

    categorical_columns = [
        column
        for column in df.columns
        if _is_string(df[column]) and df[column].nunique() <= 12
    ][:4]
    if categorical_columns:
        fig, axes = plt.subplots(nrows=1, ncols=len(categorical_columns), figsize=(4.6 * len(categorical_columns), 4))
        if len(categorical_columns) == 1:
            axes = [axes]
        for ax, column in zip(axes, categorical_columns):
            counts = df[column].value_counts(dropna=True).head(12)
            ax.barh([str(v) for v in counts.index][::-1], counts.values[::-1])
            ax.set_title(column, fontsize=9)
            ax.tick_params(axis="y", labelsize=7)
        fig.suptitle("Categorical frequencies (top levels)")
        _tight_figure(fig)
        path = figures_dir / "categorical_frequencies.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)

    if DATACO_TARGET_COLUMN in df.columns:
        target = df[DATACO_TARGET_COLUMN].astype(str)
        counts = target.value_counts(dropna=True).sort_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar([str(v) for v in counts.index], counts.values, color=["#4C72B0", "#DD8452"])
        for i, (k, v) in enumerate(counts.items()):
            ax.text(i, v + 500, str(int(v)), ha="center", fontsize=8)
        ax.set_title(f"Target distribution: {DATACO_TARGET_COLUMN}")
        ax.set_ylabel("Count")
        _tight_figure(fig)
        path = figures_dir / "target_distribution.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)

    logger.info("Saved %d audit figures to %s", len(written), figures_dir)
    return written