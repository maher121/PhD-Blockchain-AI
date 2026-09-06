"""Baseline anomaly detection with Isolation Forest (Prototype V0.1).

Labelling caveat
----------------
Two label kinds exist in V0.1:

* ``controlled_synthetic``          - anomalies injected by the synthetic
  generator ``src/data/generator`` (extreme quantity/price, late shipment).
* ``natural_late_delivery_risk``    - the DataCo Kaggle dataset's real
  ``Late_delivery_risk`` flag (0/1), preserved as ``natural_label`` by
  ``src.data.loading.normalize_kaggle_supply_chain``.

NEITHER is a cybersecurity attack label. Reported metrics describe how well
the unsupervised baseline recovers the corresponding label source only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import DEFAULT_MODEL_FILE, GLOBAL_SEED, ISOLATION_FOREST_PARAMS

logger = logging.getLogger(__name__)


class IsolationForestAnomalyDetector:
    """Thin wrapper around scikit-learn's Isolation Forest.

    The wrapper owns the model lifecycle (fit, predict, score, save, load)
    and keeps the pipeline independent from the specific sklearn estimator.
    """

    def __init__(self, **hyperparameters: Any) -> None:
        if "random_state" not in hyperparameters:
            hyperparameters["random_state"] = GLOBAL_SEED
        self._model: IsolationForest | None = None
        self._hyperparameters: dict[str, Any] = dict(hyperparameters)

    def fit(self, features: pd.DataFrame, sample_labels: pd.Series | None = None) -> "IsolationForestAnomalyDetector":
        """Train the detector on the feature matrix.

        ``sample_labels`` is accepted so call sites can report which label
        source was used; the Isolation Forest algorithm itself is
        unsupervised and ignores the labels during fitting.
        """
        if features.empty:
            raise ValueError("Cannot fit on an empty feature matrix.")
        self._model = IsolationForest(**self._hyperparameters)
        self._model.fit(features.to_numpy())
        logger.info("Trained Isolation Forest on %d samples", len(features))
        return self

    @property
    def is_trained(self) -> bool:
        """True after a successful fit."""
        return self._model is not None

    def predict_anomalies(self, features: pd.DataFrame) -> pd.Series:
        """Return a boolean series: True for samples flagged as anomalies."""
        if self._model is None:
            raise RuntimeError("Model not trained - call fit() first.")
        predictions = self._model.predict(features.to_numpy())
        return pd.Series(predictions == -1, index=features.index, dtype=bool)

    def anomaly_scores(self, features: pd.DataFrame) -> pd.Series:
        """Return Isolation Forest anomaly scores (higher = more anomalous)."""
        if self._model is None:
            raise RuntimeError("Model not trained - call fit() first.")
        scores = self._model.score_samples(features.to_numpy())
        return pd.Series(-1.0 * scores, index=features.index, dtype=float)

    def save_model(self, output_path: Path | str = DEFAULT_MODEL_FILE) -> Path:
        """Persist the fitted estimator to ``output_path`` (joblib)."""
        if self._model is None:
            raise RuntimeError("Cannot save an untrained model.")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "hyperparameters": self._hyperparameters}, path)
        logger.info("Saved model to %s", path)
        return path

    def load_model(self, input_path: Path | str) -> "IsolationForestAnomalyDetector":
        """Load a previously saved estimator into this wrapper."""
        payload = joblib.load(input_path)
        self._model = payload["model"]
        self._hyperparameters = payload["hyperparameters"]
        logger.info("Loaded model from %s", input_path)
        return self


def report_anomaly_metrics(
    true_labels: pd.Series | None,
    predicted_flags: pd.Series,
    scores: pd.Series,
) -> dict[str, float]:
    """Compute classification metrics using the true/false binary convention.

    ``true_labels`` is an optional binary mask (1 = known anomaly). When it is
    None or contains a single class, label-based metrics are skipped and an
    empty dict is returned -- the caller must not interpret missing metrics as
    zeros.

    Returns
    -------
    dict of metric name -> value. ``roc_auc`` is only present when both classes
    occur in ``true_labels``.
    """
    if true_labels is None:
        return {}

    true_binary = (true_labels.astype(bool)).astype(int)
    if true_binary.nunique() < 2:
        return {}

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    aligned_true, aligned_pred = true_binary.align(predicted_flags.astype(int), fill_value=0)
    aligned_true, aligned_score = true_binary.align(scores, fill_value=0.0)

    return {
        "precision": float(precision_score(aligned_true, aligned_pred, zero_division=0)),
        "recall": float(recall_score(aligned_true, aligned_pred, zero_division=0)),
        "f1_score": float(f1_score(aligned_true, aligned_pred, zero_division=0)),
        "accuracy": float(accuracy_score(aligned_true, aligned_pred)),
        "roc_auc": float(roc_auc_score(aligned_true, aligned_score)),
    }