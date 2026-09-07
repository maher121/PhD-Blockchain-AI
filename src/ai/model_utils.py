"""Data loading, feature governance, and JSON helpers for V0.3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.config import (
    DATACO_CATEGORICAL_FEATURES,
    DATACO_DATETIME_COLUMNS,
    DATACO_IDENTIFIER_COLUMNS,
    DATACO_NUMERIC_FEATURES,
    DATACO_OUTCOME_COLUMNS,
    DATACO_TARGET_COLUMN,
    DATASET_METADATA_FILE,
    DATA_AUDIT_DIR,
    PROCESSED_DATA_DIR,
)

SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")
MODEL_GENERATED_COLUMNS: tuple[str, ...] = (
    "raw_isolation_score",
    "anomaly_score",
    "normalized_anomaly_score",
    "anomaly_label",
    "predicted_anomaly",
)


@dataclass
class ProcessedSplit:
    name: str
    features: pd.DataFrame
    metadata: pd.DataFrame
    operational_target: pd.DataFrame | None


@dataclass
class ProcessedDataBundle:
    splits: dict[str, ProcessedSplit]
    selected_features: list[str]
    dataset_metadata: dict[str, Any]
    audit_summary: dict[str, Any]


def load_processed_dataco(
    processed_dir: Path | str = PROCESSED_DATA_DIR,
    metadata_path: Path | str | None = None,
    audit_path: Path | str | None = None,
) -> ProcessedDataBundle:
    """Load V0.2 outputs and verify row/feature alignment across all splits."""
    processed_dir = Path(processed_dir)
    metadata_path = Path(metadata_path) if metadata_path else processed_dir / DATASET_METADATA_FILE.name
    audit_path = Path(audit_path) if audit_path else DATA_AUDIT_DIR / "dataset_summary.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"V0.2 dataset metadata not found: {metadata_path}")
    dataset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected_features = list(dataset_metadata["features"]["columns"])
    validate_selected_features(selected_features)

    splits: dict[str, ProcessedSplit] = {}
    for name in SPLIT_NAMES:
        split_dir = processed_dir / name
        feature_path = split_dir / "features.csv"
        metadata_csv = split_dir / "metadata.csv"
        if not feature_path.exists() or not metadata_csv.exists():
            raise FileNotFoundError(f"Missing V0.2 processed files under {split_dir}")

        features = pd.read_csv(feature_path)
        trace = pd.read_csv(metadata_csv, low_memory=False)
        target_path = split_dir / "target.csv"
        target = pd.read_csv(target_path) if target_path.exists() else None

        _validate_split_alignment(name, features, trace, target)
        missing = [column for column in selected_features if column not in features.columns]
        if missing:
            raise ValueError(f"{name} split is missing selected V0.2 features: {missing}")
        if not np.isfinite(features[selected_features].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} split contains non-finite processed features.")
        splits[name] = ProcessedSplit(name, features, trace, target)

    reference_columns = list(splits["train"].features.columns)
    for name in SPLIT_NAMES[1:]:
        if list(splits[name].features.columns) != reference_columns:
            raise ValueError(f"Processed feature layout differs between train and {name}.")

    audit_summary = (
        json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    )
    return ProcessedDataBundle(splits, selected_features, dataset_metadata, audit_summary)


def determine_label_situation(bundle: ProcessedDataBundle) -> dict[str, Any]:
    """Classify the observed label situation without changing label semantics."""
    feature_metadata = bundle.dataset_metadata.get("features", {})
    cyber_labels = bool(feature_metadata.get("cybersecurity_labels", False))
    target_present = all(
        bundle.splits[name].operational_target is not None for name in SPLIT_NAMES
    )
    target_column = feature_metadata.get("target_column")

    if cyber_labels and target_present:
        return {
            "case": "A",
            "kind": "native_anomaly_or_cybersecurity_label",
            "target": target_column,
            "supervised_metrics_allowed": True,
        }
    if target_present and target_column:
        return {
            "case": "C",
            "kind": "operational_outcome_only",
            "target": target_column,
            "source": "DataCo Late_delivery_risk operational delivery outcome",
            "supervised_metrics_allowed": False,
            "reason": (
                "The available target describes late delivery, not an anomaly or "
                "cybersecurity attack. It is not used to fit or evaluate V0.3."
            ),
        }
    return {
        "case": "B",
        "kind": "no_ground_truth_anomaly_label",
        "target": None,
        "supervised_metrics_allowed": False,
    }


def validate_selected_features(feature_names: Iterable[str]) -> None:
    """Reject identifiers, outcomes, targets, and model outputs as inputs."""
    names = list(feature_names)
    forbidden = {
        "row_id",
        "target",
        DATACO_TARGET_COLUMN,
        *DATACO_IDENTIFIER_COLUMNS,
        *DATACO_OUTCOME_COLUMNS,
        *MODEL_GENERATED_COLUMNS,
    }
    invalid = sorted(set(names) & forbidden)
    if invalid:
        raise ValueError(f"Leakage-prone or identifier features selected: {invalid}")
    if not names or len(set(names)) != len(names):
        raise ValueError("Selected feature names must be non-empty and unique.")


def build_feature_manifest(bundle: ProcessedDataBundle) -> dict[str, Any]:
    """Document every selected processed feature and excluded raw/input field."""
    selected = bundle.selected_features
    validate_selected_features(selected)
    raw_columns = bundle.audit_summary.get("column_names", [])
    excluded: list[dict[str, str]] = []
    for column in raw_columns:
        excluded.append({"column": column, "reason": _raw_exclusion_reason(column)})
    for column in MODEL_GENERATED_COLUMNS:
        excluded.append(
            {
                "column": column,
                "reason": "Model-generated output; never permitted as an input feature.",
            }
        )
    return {
        "stage": "V0.3",
        "selection_strategy": (
            "Transparent reuse of the exact leakage-safe processed feature list "
            "recorded by V0.2; no optimization-based selection."
        ),
        "preprocessing": (
            "V0.2 fitted imputation, scaling, and categorical levels on training data only. "
            "V0.3 applies no second fitted transformation."
        ),
        "selected_feature_count": len(selected),
        "selected_features": selected,
        "excluded_features": excluded,
    }


def write_json(payload: dict[str, Any] | list[Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def _validate_split_alignment(
    name: str,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    target: pd.DataFrame | None,
) -> None:
    if features.empty or "row_id" not in features or "row_id" not in metadata:
        raise ValueError(f"{name} split is empty or lacks row_id traceability.")
    if not features["row_id"].is_unique or not metadata["row_id"].is_unique:
        raise ValueError(f"{name} split has duplicate row_id values.")
    if features["row_id"].tolist() != metadata["row_id"].tolist():
        raise ValueError(f"{name} feature and metadata row_id values are misaligned.")
    if target is not None and features["row_id"].tolist() != target["row_id"].tolist():
        raise ValueError(f"{name} feature and operational-target rows are misaligned.")


def _raw_exclusion_reason(column: str) -> str:
    if column in (DATACO_TARGET_COLUMN, "target"):
        return "Operational delivery target, not a cybersecurity/anomaly label; excluded."
    if column in DATACO_OUTCOME_COLUMNS:
        return "Realized or post-outcome field; excluded to prevent future/outcome leakage."
    if column == "shipping date (DateOrders)":
        return "Shipping-time information occurs after order placement; excluded as future information."
    if column == "row_id" or column in DATACO_IDENTIFIER_COLUMNS:
        return "Identifier, high-cardinality traceability field, geography, or PII; metadata only."
    if column in DATACO_NUMERIC_FEATURES or column == DATACO_DATETIME_COLUMNS[0]:
        return "Raw source replaced by its controlled V0.2 engineered/scaled representation."
    if column in DATACO_CATEGORICAL_FEATURES:
        return "Raw category replaced by train-fitted V0.2 one-hot features."
    if column in ("Benefit per order", "Sales per customer", "Order Profit Per Order"):
        return "Owner-computed aggregate with unclear time window; excluded as leakage-prone."
    return "Not selected for the transparent V0.3 baseline; retained only in source metadata/audit."


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)
