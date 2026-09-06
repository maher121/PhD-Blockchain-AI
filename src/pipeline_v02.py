"""V0.2 orchestrator: DataCo dataset pipeline (discover -> audit -> split ->
preprocess -> verify -> persist).

Flow (leakage-safe by construction):

    resolve DataCo dataset           (src.data.dataset_discovery)
    load raw (row_id assigned)       (this module)
    audit raw                        (src.data.audit)
    sample cap (deterministic)       BEFORE the split (honest probabilities)
    split train/validation/test      (src.preprocessing.leakage)
    fit preprocessing on train       (src.preprocessing.dataco_pipeline)
    transform validation/test        (same statistics)
    save processed splits, metadata
    generate docs/data_dictionary.md

The raw dataset under ``data/raw`` is never modified.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    AUDIT_FIGURES_DIR,
    DATA_AUDIT_DIR,
    DATACO_MAX_ROWS,
    DATACO_TARGET_COLUMN,
    DATA_DICTIONARY_FILE,
    DATASET_METADATA_FILE,
    GLOBAL_SEED,
    PIPELINE_REPORT_FILE,
    PROCESSED_SPLIT_DIRS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SPLIT_RATIOS,
    SPLIT_STRATEGY,
)
from src.data.audit import audit_dataframe
from src.data.dataset_discovery import DataCoResolution, resolve_dataco_dataset
from src.data.versioning import build_dataset_metadata, save_dataset_metadata
from src.evaluation.reporting import save_experiment_results
from src.preprocessing.dataco_pipeline import (
    DataCoPreprocessor,
    PreprocessorConfig,
    ProcessedFrame,
    feature_catalogue,
)
from src.preprocessing.leakage import split_dataset

logger = logging.getLogger(__name__)


class DataCoDatasetMissingError(RuntimeError):
    """Raised when no genuine DataCo dataset is available locally."""


#: Human-readable meaning per raw DataCo column (curated for the dictionary).
_COLUMN_MEANINGS: dict[str, str] = {
    "Type": "Payment method type.",
    "Days for shipping (real)": "Realised shipping duration in days (outcome-adjacent).",
    "Days for shipment (scheduled)": "Planned fulfilment duration in days (known at order time).",
    "Benefit per order": "Owner-computed benefit aggregated per order.",
    "Sales per customer": "Owner-computed aggregate sales per customer.",
    "Delivery Status": "Derived outcome: late / on-time / canceled / advance delivery.",
    "Late_delivery_risk": "Real 0/1 flag := order ship date exceeded the scheduled date.",
    "Category Id": "Numeric category identifier.",
    "Category Name": "Product category name.",
    "Customer City": "Customer city (traceability / geo).",
    "Customer Country": "Customer country (traceability / geo).",
    "Customer Email": "Customer email (PII).",
    "Customer Fname": "Customer first name (PII).",
    "Customer Id": "Customer identifier.",
    "Customer Lname": "Customer last name (PII).",
    "Customer Password": "Customer record password (PII).",
    "Customer Segment": "Customer segment (Consumer / Corporate / Home Office).",
    "Customer State": "Customer state (traceability / geo).",
    "Customer Street": "Customer street address (PII).",
    "Customer Zipcode": "Customer postal code.",
    "Department Id": "Numeric department identifier.",
    "Department Name": "Sales department name.",
    "Latitude": "Customer latitude (geo).",
    "Longitude": "Customer longitude (geo).",
    "Market": "Sales market / geography code.",
    "Order City": "Order destination city.",
    "Order Country": "Order destination country.",
    "Order Customer Id": "Customer identifier attached to the order.",
    "order date (DateOrders)": "Date/time the order was placed.",
    "Order Id": "Order identifier (several item lines share one order).",
    "Order Item Cardprod Id": "Card product identifier on the item line.",
    "Order Item Discount": "Absolute discount applied on the item line.",
    "Order Item Discount Rate": "Relative discount rate on the item line.",
    "Order Item Id": "Unique identifier of the order item line.",
    "Order Item Product Price": "Unit price paid on the item line.",
    "Order Item Profit Ratio": "Profit ratio of the item line.",
    "Order Item Quantity": "Quantity ordered on the item line.",
    "Sales": "Net sales value of the item line.",
    "Order Item Total": "Line total (quantity x price).",
    "Order Profit Per Order": "Owner-computed profit aggregated per order.",
    "Order Region": "Order destination region code.",
    "Order State": "Order destination state.",
    "Order Status": "Order lifecycle status (COMPLETE / PENDING / ... ).",
    "Order Zipcode": "Order destination postal code.",
    "Product Card Id": "Product (card) identifier.",
    "Product Category Id": "Numeric product category identifier.",
    "Product Description": "Free-text product description.",
    "Product Image": "Product image URL/path.",
    "Product Name": "Product display name.",
    "Product Price": "Catalogue (list) unit price of the product.",
    "Product Status": "Product stock status code.",
    "shipping date (DateOrders)": "Date/time the order was shipped.",
    "Shipping Mode": "Selected shipping class (First Class / Same Day / ...).",
}

#: Columns reserved as blockchain / traceability metadata for later phases.
_BLOCKCHAIN_COLUMNS: tuple[str, ...] = (
    "Order Id",
    "Order Item Id",
    "Order Customer Id",
    "Customer Id",
    "Product Card Id",
    "order date (DateOrders)",
    "shipping date (DateOrders)",
    "Order Status",
    "Delivery Status",
    "Shipping Mode",
)


def load_dataco_frame(resolved: DataCoResolution) -> pd.DataFrame:
    """Read the resolved DataCo table and attach a stable integer ``row_id``."""
    if resolved.path is None:
        raise DataCoDatasetMissingError(
            "; ".join(resolved.errors) or "No DataCo dataset available."
        )
    path = Path(resolved.path)
    if path.suffix.lower() == ".xlsx":
        try:
            import openpyxl  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("XLSX support requires the optional 'openpyxl' package.") from exc
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
    if df.empty:
        raise DataCoDatasetMissingError(f"Dataset {path} is empty.")
    df = df.reset_index(drop=True)
    df.insert(0, "row_id", range(len(df)))
    return df


def cap_before_split(df: pd.DataFrame, n_rows: int = DATACO_MAX_ROWS, seed: int = GLOBAL_SEED) -> pd.DataFrame:
    """Deterministic row cap. Applied BEFORE the split only."""
    if n_rows is None or len(df) <= n_rows:
        return df
    return df.sample(n=n_rows, random_state=seed).reset_index(drop=True)


def _audit_after(df: pd.DataFrame) -> dict[str, Any]:
    """Reuse the audit to also report on the capped/processed frames."""
    return audit_dataframe(
        df,
        source_name=f"processed_{len(df)}_rows",
        output_dir=DATA_AUDIT_DIR,
        save_figures=False,
    )


def save_processed_splits(
    processed: dict[str, ProcessedFrame],
    split_dirs: dict[str, Path] = PROCESSED_SPLIT_DIRS,
) -> dict[str, Path]:
    """Persist each split (metadata.csv, features.csv, target.csv)."""
    written: dict[str, Path] = {}
    for name, frame in processed.items():
        out_dir = Path(split_dirs[name])
        out_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = out_dir / "metadata.csv"
        frame.metadata.to_csv(metadata_path, index=False)

        features = frame.ml_features.copy()
        features.insert(0, "row_id", frame.metadata["row_id"].values)
        features_path = out_dir / "features.csv"
        features.to_csv(features_path, index=False)

        written[f"{name}_metadata"] = metadata_path
        written[f"{name}_features"] = features_path

        if frame.target is not None:
            target_path = out_dir / "target.csv"
            pd.DataFrame(
                {
                    "row_id": frame.metadata["row_id"].values,
                    "target": frame.target,
                }
            ).to_csv(target_path, index=False)
            written[f"{name}_target"] = target_path
    return written


def preprocessing_summary(
    preprocessor: DataCoPreprocessor,
    processed: dict[str, ProcessedFrame],
    split_report: dict[str, Any],
) -> dict[str, Any]:
    """Machine-readable preprocessing report stored under results/data_audit/."""
    plan = copy.deepcopy(preprocessor.fit_report)
    plan["config"] = preprocessor.config_snapshot()
    plan["split"] = {
        "strategy": split_report.get("strategy"),
        "ratios": split_report.get("ratios"),
    }
    plan["outputs"] = {
        name: {
            "rows": frame.row_count(),
            "ml_feature_columns": int(frame.ml_features.shape[1]),
            "has_target": frame.target is not None,
        }
        for name, frame in processed.items()
    }
    plan["target_column"] = DATACO_TARGET_COLUMN
    plan["cybersecurity_labels_available"] = False
    return plan


def generate_data_dictionary(
    raw_df: pd.DataFrame,
    output_path: Path | str = DATA_DICTIONARY_FILE,
) -> Path:
    """Write ``docs/data_dictionary.md`` from the real columns + feature catalog."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    catalog = feature_catalogue()
    ml_feature_names = {item["name"] for item in catalog if item["role"] == "ml"}
    source_columns = {item["source"] for item in catalog}

    lines: list[str] = []
    lines.append("# Data Dictionary - DataCo Smart Supply Chain Pipeline (V0.2)")
    lines.append("")
    lines.append(
        "Auto-generated by `src.pipeline_v02.generate_data_dictionary` from the "
        "real DataCo columns and the controlled feature catalogue."
    )
    lines.append("")
    lines.append("## Target variables")
    lines.append("")
    lines.append(
        "The dataset provides **no native cybersecurity attack labels**. "
        "`Late_delivery_risk` is a real operational delivery-risk outcome "
        "(0/1) and is the candidate supervised target for later phases. It "
        "must NOT be presented as a security label."
    )
    lines.append("")
    lines.append(
        "| Category | Columns | Notes |"
    )
    lines.append("|---|---|---|")
    lines.append(
        "| Existing labels/outcomes | `Late_delivery_risk`, `Delivery Status`, `Order Status` | Operational outcomes, not cyber labels. |"
    )
    lines.append(
        "| Operational prediction targets | `Late_delivery_risk` (default) | Candidate supervised target for V0.3+. |"
    )
    lines.append(
        "| Cybersecurity/anomaly labels | (none) | Explicitly absent in the dataset. |"
    )
    lines.append(
        "| Synthetic attack labels | (none in V0.2) | Deferred to the later cybersecurity phase. |"
    )
    lines.append("")

    lines.append("## Raw columns")
    lines.append("")
    lines.append(
        "| Original column | Meaning | Data type | Role | Preprocessing | Used for ML | Used for Blockchain | Excluded | Reason |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for column in raw_df.columns:
        meaning = _COLUMN_MEANINGS.get(column, "Raw DataCo attribute.")
        dtype = str(raw_df[column].dtype)
        missing = int(raw_df[column].isna().sum())
        preproc = (
            f"Missing {missing}; "
            if missing
            else ""
        )
        if column in source_columns or column == "order date (DateOrders)":
            preproc += "coerced to numeric / datetime; scaled or one-hot fitted on train"
            role = "ml_source"
        elif column == DATACO_TARGET_COLUMN:
            preproc += "kept as supervised target"
            role = "target"
        elif column in ("Delivery Status", "Days for shipping (real)", "Order Status"):
            preproc += "kept in metadata only (outcome-adjacent)"
            role = "outcome"
        elif _is_identifierish(column):
            preproc += "kept as traceability metadata; not a model feature"
            role = "identifier"
        elif column in ("Latitude", "Longitude"):
            preproc += "kept as traceability metadata (geo); not a model feature in V0.2"
            role = "geo"
        else:
            preproc += "normalised to string; metadata"
            role = "categorical"
        used_ml = "yes" if column in source_columns or column == "order date (DateOrders)" else "no"
        used_bc = "yes" if column in _BLOCKCHAIN_COLUMNS else "no"
        excluded = "yes" if used_ml == "no" else "no"
        reason = _exclusion_reason(role)
        lines.append(
            f"| {column} | {meaning} | {dtype} | {role} | {preproc} | {used_ml} | {used_bc} | {excluded} | {reason} |"
        )
    lines.append("")

    lines.append("## Engineered features (V0.2 feature catalogue)")
    lines.append("")
    lines.append(
        "| Feature | Source columns | Definition | Rationale | Category |"
    )
    lines.append("|---|---|---|---|---|")
    for item in catalog:
        lines.append(
            f"| {item['name']} | {item['source']} | {item['definition']} | "
            f"{item['rationale']} | {item['category']} |"
        )
    lines.append("")

    lines.append(
        f"## Leakage contract\n\nOrder-level features "
        f"(`item_count_per_order`, `order_total_value`) are computed from the "
        f"training split only. Outcome-adjacent columns ({', '.join(('Delivery Status', 'Days for shipping (real)', 'Order Status'))}) "
        f"are excluded from ML features. The default split strategy "
        f"``{SPLIT_STRATEGY}`` keeps each `Order Id` in one split."
    )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote data dictionary to %s", output_path)
    return output_path


def _is_identifierish(column: str) -> bool:
    lowered = column.lower()
    hints = ("id", "email", "password", "fname", "lname", "zipcode", "street", "name")
    return any(h in lowered for h in hints)


def _exclusion_reason(role: str) -> str:
    reasons = {
        "identifier": "High-cardinality / PII key; meaningless as a predictive feature; useful for traceability.",
        "geo": "Geographic coordinates act as identifiers here; excluded from model features in V0.2.",
        "outcome": "Encodes the realised outcome; would leak the target into features.",
        "target": "Target column, not a feature.",
        "categorical": "Metadata only in V0.2.",
    }
    return reasons.get(role, "Not selected as an ML feature / block-chain field in V0.2.")


def run_dataco_pipeline_v02(
    raw_dir: Path | str = RAW_DATA_DIR,
    audit_dir: Path | str = DATA_AUDIT_DIR,
    processed_dir: Path | str = PROCESSED_DATA_DIR,
    split_dirs: dict[str, Path] | None = None,
    dictionary_path: Path | str = DATA_DICTIONARY_FILE,
    metadata_path: Path | str = DATASET_METADATA_FILE,
    report_path: Path | str = PIPELINE_REPORT_FILE,
    split_strategy: str = SPLIT_STRATEGY,
    split_ratios: dict[str, float] = SPLIT_RATIOS,
    max_rows: int | None = DATACO_MAX_ROWS,
    seed: int = GLOBAL_SEED,
) -> dict[str, Any]:
    """Execute the whole V0.2 pipeline and persist every artefact.

    Returns a bundle of frames, reports and written paths for the notebook
    and integration tests.
    """
    audit_dir = Path(audit_dir)
    processed_dir = Path(processed_dir)
    split_dirs = split_dirs or {
        name: processed_dir / name for name in ("train", "validation", "test")
    }
    dictionary_path = Path(dictionary_path)
    metadata_path = Path(metadata_path)
    report_path = Path(report_path)

    resolution = resolve_dataco_dataset(raw_dir=raw_dir)
    if resolution.path is None:
        raise DataCoDatasetMissingError("; ".join(resolution.errors))

    raw_df = load_dataco_frame(resolution)
    audit_report = audit_dataframe(raw_df, source_name=resolution.path.name, output_dir=audit_dir)

    work = cap_before_split(raw_df, n_rows=max_rows, seed=seed)
    logger.info("Pipeline input rows after cap: %d (of %d)", len(work), len(raw_df))

    split_result = split_dataset(
        work,
        ratios=split_ratios,
        strategy=split_strategy,
        seed=seed,
    )
    train, validation, test = split_result.train, split_result.validation, split_result.test

    preprocessor = DataCoPreprocessor(
        PreprocessorConfig(seed=seed, drop_empty_target=True)
    )
    processed = preprocessor.transform_all(train, validation, test)

    written = save_processed_splits(processed, split_dirs)

    plan = preprocessing_summary(preprocessor, processed, split_result.report)
    (audit_dir / "preprocessing_summary.json").write_text(
        json.dumps(plan, indent=2, default=str), encoding="utf-8"
    )

    dictionary_path = generate_data_dictionary(raw_df, dictionary_path)

    metadata = build_dataset_metadata(
        raw_path=resolution.path,
        raw_df=raw_df,
        source_provenance=("capped:" + str(max_rows) if max_rows else "full") + "|provenance:" + resolution.provenance,
        split_report=split_result.report,
        preprocessor_config=plan,
        seed=seed,
        feature_columns=list(processed["train"].ml_features.columns),
        target_column=DATACO_TARGET_COLUMN,
        extra={
            "dataset_used_rows": int(len(work)),
            "resolution": {
                "found_in_raw": resolution.found_in_raw,
                "provenance": resolution.provenance,
                "staged_to": str(resolution.staged_to) if resolution.staged_to else None,
                "messages": resolution.messages,
            },
            "audit": {
                "n_rows": audit_report["n_rows"],
                "n_columns": audit_report["n_columns"],
                "missing_total_cells": audit_report["missing_summary"]["total_missing_cells"],
                "duplicate_rows": audit_report["duplicates"]["fully_duplicate_rows"],
                "leakage_candidates": audit_report["leakage_candidates"],
            },
        },
    )
    save_dataset_metadata(metadata, metadata_path)

    payload = {
        "stage": "V0.2",
        "dataset": metadata["dataset"],
        "processing": metadata["processing"],
        "split": metadata["split"],
        "preprocessing_config": preprocessor.config_snapshot(),
        "fit_report": plan,
        "output_files": {k: str(v) for k, v in written.items()},
        "data_dictionary": str(dictionary_path),
        "audit_files": {
            "dataset_summary.json": str(audit_dir / "dataset_summary.json"),
            "missing_values.csv": str(audit_dir / "missing_values.csv"),
            "duplicate_report.json": str(audit_dir / "duplicate_report.json"),
            "feature_summary.csv": str(audit_dir / "feature_summary.csv"),
            "figures": str(audit_dir / "figures"),
        },
        "notes": {
            "cybersecurity_labels": (
                "The dataset does not provide native cybersecurity attack labels."
            ),
            "synthetic_attacks": "Not created in V0.2.",
            "no_models_trained": True,
        },
    }
    save_experiment_results(payload, output_path=report_path)

    return {
        "resolution": resolution,
        "raw_df": raw_df,
        "audit_report": audit_report,
        "work": work,
        "split_result": split_result,
        "preprocessor": preprocessor,
        "processed": processed,
        "metadata": metadata,
        "written_files": written,
        "payload": payload,
    }


if __name__ == "__main__":
    from src.logging_config import setup_logging

    setup_logging()
    result = run_dataco_pipeline_v02()
    print("=== V0.2 DataCo pipeline (real dataset) ===")
    print("Dataset      :", result["resolution"].path)
    print("Raw rows     :", result["audit_report"]["n_rows"])
    print("Raw columns  :", result["audit_report"]["n_columns"])
    print("Used rows    :", len(result["work"]))
    print("Split        :", result["split_result"].report["strategy"],
          result["split_result"].report["sizes"])
    print("ML features  :", len(result["processed"]["train"].ml_features.columns))
    print("Processed    : data/processed/{train,validation,test}/")
    print("Metadata     :", result["metadata"]["dataset"]["file_sha256"][:12], "...")
    print("Data dict    : docs/data_dictionary.md")