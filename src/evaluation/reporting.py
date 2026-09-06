"""Persist experiment artifacts (metrics, measurements, reports) to JSON/CSV."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import DEFAULT_RESULTS_FILE

logger = logging.getLogger(__name__)


def _default_json_serializer(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _default_json_serializer(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_default_json_serializer(item) for item in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def save_experiment_results(
    payload: dict[str, Any],
    output_path: Path | str = DEFAULT_RESULTS_FILE,
) -> Path:
    """Write a JSON report of the prototype run under ``results/``.

    The payload is enriched with the run timestamp and the platform info so
    each results file is self-describing for reproducibility audits.
    """
    import platform

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "payload": payload,
    }

    path.write_text(
        json.dumps(record, indent=2, default=_default_json_serializer), encoding="utf-8"
    )
    logger.info("Saved experiment results to %s", path)
    return path


def save_anomaly_predictions(
    predictions: Any,
    columns: list[str] | None = None,
    output_path: Any = None,
) -> None:
    """Persist anomaly predictions/results to CSV (best-effort helper)."""
    import pandas as pd

    frame = predictions if isinstance(predictions, pd.DataFrame) else pd.DataFrame(predictions)
    if columns is not None:
        frame.columns = columns
    target = (
        Path(output_path)
        if output_path is not None
        else Path("results/anomaly_predictions.csv")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    logger.info("Saved %d anomaly result rows to %s", len(frame), target)