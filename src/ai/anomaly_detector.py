"""Reusable V0.3 Isolation Forest baseline for processed DataCo features.

The input is the leakage-safe, train-fitted feature representation produced by
V0.2. The sklearn pipeline therefore performs exact feature selection only; it
does not duplicate V0.2 imputation, encoding, or scaling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from src.config import GLOBAL_SEED, V03_ISOLATION_FOREST_PARAMS, V03_MODEL_FILE

logger = logging.getLogger(__name__)


class IsolationForestBaseline:
    """Isolation Forest with an embedded exact-column feature selector."""

    artifact_version = "v0.3"

    def __init__(
        self,
        feature_names: Sequence[str],
        **model_parameters: Any,
    ) -> None:
        self.feature_names = list(feature_names)
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be a non-empty list of unique columns.")

        parameters = dict(V03_ISOLATION_FOREST_PARAMS)
        parameters.update(model_parameters)
        parameters.setdefault("random_state", GLOBAL_SEED)
        self.model_parameters = parameters

        selector = ColumnTransformer(
            [("v02_processed_features", "passthrough", self.feature_names)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.pipeline = Pipeline(
            [
                ("feature_selector", selector),
                ("isolation_forest", IsolationForest(**parameters)),
            ]
        )
        self._fitted = False
        self._train_anomaly_min: float | None = None
        self._train_anomaly_max: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, features: pd.DataFrame) -> "IsolationForestBaseline":
        """Fit only on the caller-provided training frame."""
        self._validate_features(features)
        self.pipeline.fit(features)
        self._fitted = True
        anomaly = self.anomaly_scores(features)
        self._train_anomaly_min = float(anomaly.min())
        self._train_anomaly_max = float(anomaly.max())
        logger.info(
            "Fitted V0.3 Isolation Forest on %d rows and %d selected features",
            len(features),
            len(self.feature_names),
        )
        return self

    def raw_scores(self, features: pd.DataFrame) -> pd.Series:
        """Return sklearn ``score_samples`` values; lower means more anomalous."""
        self._require_fitted()
        self._validate_features(features)
        values = self.pipeline.score_samples(features)
        return pd.Series(values, index=features.index, name="raw_isolation_score", dtype=float)

    def anomaly_scores(self, features: pd.DataFrame) -> pd.Series:
        """Return ``-score_samples``; higher means more anomalous."""
        scores = -self.raw_scores(features)
        scores.name = "anomaly_score"
        return scores

    def normalized_anomaly_scores(self, features: pd.DataFrame) -> pd.Series:
        """Min-max normalize using training-score bounds; this is not a probability."""
        scores = self.anomaly_scores(features)
        return self._normalize_anomaly_scores(scores)

    def _normalize_anomaly_scores(self, scores: pd.Series) -> pd.Series:
        low = self._train_anomaly_min
        high = self._train_anomaly_max
        if low is None or high is None:
            raise RuntimeError("Training score bounds are unavailable.")
        if np.isclose(high, low):
            normalized = np.zeros(len(scores), dtype=float)
        else:
            normalized = np.clip((scores.to_numpy() - low) / (high - low), 0.0, 1.0)
        return pd.Series(
            normalized,
            index=scores.index,
            name="normalized_anomaly_score",
            dtype=float,
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return integer labels using 1 for anomaly and 0 for normal."""
        self._require_fitted()
        self._validate_features(features)
        predicted = self.pipeline.predict(features)
        return pd.Series(
            (predicted == -1).astype("int8"),
            index=features.index,
            name="anomaly_label",
        )

    def predict_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return all score representations and model labels for each row."""
        raw = self.raw_scores(features)
        anomaly = -raw
        anomaly.name = "anomaly_score"
        normalized = self._normalize_anomaly_scores(anomaly)
        labels = self.predict(features)
        return pd.DataFrame(
            {
                "raw_isolation_score": raw,
                "anomaly_score": anomaly,
                "normalized_anomaly_score": normalized,
                "anomaly_label": labels,
                "isolation_forest_prediction": np.where(labels.to_numpy() == 1, -1, 1),
            },
            index=features.index,
        )

    def save(self, output_path: Path | str = V03_MODEL_FILE) -> Path:
        """Save the selector, fitted estimator, configuration, and score bounds."""
        self._require_fitted()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "artifact_version": self.artifact_version,
                "sklearn_version": sklearn.__version__,
                "pipeline": self.pipeline,
                "feature_names": self.feature_names,
                "model_parameters": self.model_parameters,
                "train_anomaly_min": self._train_anomaly_min,
                "train_anomaly_max": self._train_anomaly_max,
            },
            path,
        )
        logger.info("Saved V0.3 model artifact to %s", path)
        return path

    @classmethod
    def load(cls, input_path: Path | str) -> "IsolationForestBaseline":
        """Load a V0.3 artifact ready for inference."""
        payload = joblib.load(input_path)
        if payload.get("artifact_version") != cls.artifact_version:
            raise ValueError(
                f"Unsupported model artifact version: {payload.get('artifact_version')!r}"
            )
        instance = cls(payload["feature_names"], **payload["model_parameters"])
        instance.pipeline = payload["pipeline"]
        instance._train_anomaly_min = float(payload["train_anomaly_min"])
        instance._train_anomaly_max = float(payload["train_anomaly_max"])
        instance._fitted = True
        return instance

    def _validate_features(self, features: pd.DataFrame) -> None:
        if not isinstance(features, pd.DataFrame) or features.empty:
            raise ValueError("features must be a non-empty pandas DataFrame.")
        missing = [name for name in self.feature_names if name not in features.columns]
        if missing:
            raise ValueError(f"Missing model features: {missing}")
        selected = features[self.feature_names].to_numpy(dtype=float)
        if not np.isfinite(selected).all():
            raise ValueError("Selected model features contain missing or infinite values.")

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model is not fitted; call fit() first.")
