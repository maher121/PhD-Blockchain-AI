"""Dataset metadata / reproducibility record for Prototype V0.2.

Produces a self-describing metadata record covering the dataset file
(filename + SHA-256), the processing timestamp, runtime versions, the random
seed, the split configuration, and the preprocessing configuration so every
V0.2 run can be reproduced and audited later.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DATASET_METADATA_FILE

logger = logging.getLogger(__name__)


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 hex digest of a file (streamed, memory-friendly)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_frame(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 of a DataFrame's canonical CSV serialisation."""
    canonical = df.copy()
    canonical = canonical.sort_index()
    return hashlib.sha256(canonical.to_csv(index=False).encode("utf-8")).hexdigest()


def _version(package_name: str) -> str:
    try:
        import importlib

        return importlib.import_module(package_name).__version__
    except Exception:
        return "unknown"


def build_dataset_metadata(
    *,
    raw_path: Path | str,
    raw_df: pd.DataFrame,
    source_provenance: str,
    split_report: dict[str, Any],
    preprocessor_config: dict[str, Any],
    seed: int,
    feature_columns: list[str],
    target_column: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose the full reproducibility metadata record."""
    import sklearn

    raw_path = Path(raw_path)
    record: dict[str, Any] = {
        "dataset": {
            "name": "DataCo SMART Supply Chain for Big Data Analysis",
            "filename": raw_path.name,
            "path": str(raw_path),
            "file_sha256": sha256_file(raw_path),
            "n_raw_rows": int(len(raw_df)),
            "n_raw_columns": int(raw_df.shape[1]),
            "provenance": source_provenance,
        },
        "processing": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "python_implementation": platform.python_implementation(),
            "pandas_version": _version("pandas"),
            "numpy_version": _version("numpy"),
            "scikit_learn_version": sklearn.__version__,
        },
        "split": {
            "seed": int(seed),
            "strategy": split_report.get("strategy"),
            "ratios": split_report.get("ratios"),
            "sizes": split_report.get("sizes"),
            "identity_overlap_checks": split_report.get("identity_overlap_checks"),
        },
        "preprocessing": preprocessor_config,
        "features": {
            "columns": list(feature_columns),
            "target_column": target_column,
            "cybersecurity_labels": False,
        },
    }
    if extra:
        record["extra"] = extra
    return record


def _default_json(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def save_dataset_metadata(
    metadata: dict[str, Any],
    output_path: Path | str = DATASET_METADATA_FILE,
) -> Path:
    """Persist the metadata record as pretty JSON under data/processed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, indent=2, default=_default_json), encoding="utf-8"
    )
    logger.info("Saved dataset metadata (%d keys) to %s", len(metadata), output_path)
    return output_path