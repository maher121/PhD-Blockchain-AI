"""Model factory and common detector interface for V0.5."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.config import GLOBAL_SEED
from src.lightweight.feature_reduction import validate_model_features

SUPPORTED_MODELS: tuple[str, ...] = (
    "isolation_forest",
    "logistic_regression",
    "decision_tree",
    "random_forest",
)
SUPERVISED_MODELS: frozenset[str] = frozenset(
    {"logistic_regression", "decision_tree", "random_forest"}
)


class LightweightDetector:
    """A leakage-safe interface over the deliberately small V0.5 estimators."""

    artifact_version = "v0.5"

    def __init__(
        self,
        model_name: str,
        feature_names: Sequence[str],
        parameters: dict[str, Any] | None = None,
        *,
        random_state: int = GLOBAL_SEED,
    ) -> None:
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported lightweight model: {model_name!r}")
        self.model_name = model_name
        self.feature_names = list(feature_names)
        validate_model_features(self.feature_names)
        self.random_state = int(random_state)
        self.parameters = dict(parameters or {})
        self.estimator = _build_estimator(model_name, self.parameters, self.random_state)
        self._fitted = False

    @property
    def is_supervised(self) -> bool:
        return self.model_name in SUPERVISED_MODELS

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self, features: pd.DataFrame, labels: pd.Series | np.ndarray | None = None
    ) -> "LightweightDetector":
        matrix = self._matrix(features)
        if self.is_supervised:
            if labels is None:
                raise ValueError(f"{self.model_name} requires binary training labels.")
            target = np.asarray(labels, dtype=int)
            if len(target) != len(matrix) or set(np.unique(target)) != {0, 1}:
                raise ValueError("Supervised training labels must align and contain both 0 and 1.")
            self.estimator.fit(matrix, target)
        else:
            self.estimator.fit(matrix)
        self._fitted = True
        return self

    def anomaly_scores(self, features: pd.DataFrame) -> pd.Series:
        self._require_fitted()
        matrix = self._matrix(features)
        if self.model_name == "isolation_forest":
            values = -self.estimator.score_samples(matrix)
        else:
            values = self.estimator.predict_proba(matrix)[:, 1]
        return pd.Series(values, index=features.index, name="anomaly_score", dtype=float)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        self._require_fitted()
        matrix = self._matrix(features)
        predicted = self.estimator.predict(matrix)
        if self.model_name == "isolation_forest":
            predicted = (predicted == -1).astype("int8")
        return pd.Series(predicted, index=features.index, name="anomaly_label", dtype="int8")

    def predict_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "anomaly_score": self.anomaly_scores(features),
                "anomaly_label": self.predict(features),
            },
            index=features.index,
        )

    def complexity(self) -> dict[str, Any]:
        """Return descriptive indicators; these are not energy measurements."""
        self._require_fitted()
        result: dict[str, Any] = {
            "model": self.model_name,
            "number_of_input_features": len(self.feature_names),
        }
        if self.model_name == "logistic_regression":
            result.update(
                {
                    "number_of_coefficients": int(self.estimator.coef_.size),
                    "number_of_intercepts": int(self.estimator.intercept_.size),
                }
            )
            return result

        estimators = (
            list(self.estimator.estimators_)
            if self.model_name in {"isolation_forest", "random_forest"}
            else [self.estimator]
        )
        trees = [estimator.tree_ for estimator in estimators]
        result.update(
            {
                "number_of_trees": len(trees),
                "configured_max_depth": self.parameters.get("max_depth"),
                "maximum_observed_depth": int(max(tree.max_depth for tree in trees)),
                "number_of_nodes": int(sum(tree.node_count for tree in trees)),
                "number_of_leaves": int(sum(tree.n_leaves for tree in trees)),
            }
        )
        if self.model_name == "isolation_forest":
            result.update(
                {
                    "number_of_estimators": int(self.estimator.n_estimators),
                    "configured_maximum_samples": self.estimator.max_samples,
                    "maximum_samples": int(self.estimator.max_samples_),
                    "configured_maximum_features": self.estimator.max_features,
                    "maximum_features": int(len(self.estimator.estimators_features_[0])),
                }
            )
        return result

    def save(self, path: Path | str) -> Path:
        self._require_fitted()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "artifact_version": self.artifact_version,
                "sklearn_version": sklearn.__version__,
                "model_name": self.model_name,
                "feature_names": self.feature_names,
                "parameters": self.parameters,
                "random_state": self.random_state,
                "estimator": self.estimator,
            },
            output,
        )
        return output

    @classmethod
    def load(cls, path: Path | str) -> "LightweightDetector":
        payload = joblib.load(path)
        if payload.get("artifact_version") != cls.artifact_version:
            raise ValueError(f"Unsupported model artifact: {payload.get('artifact_version')!r}")
        instance = cls(
            payload["model_name"],
            payload["feature_names"],
            payload["parameters"],
            random_state=payload["random_state"],
        )
        instance.estimator = payload["estimator"]
        instance._fitted = True
        return instance

    def _matrix(self, features: pd.DataFrame) -> np.ndarray:
        if not isinstance(features, pd.DataFrame) or features.empty:
            raise ValueError("features must be a non-empty pandas DataFrame.")
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(f"Missing model features: {missing}")
        matrix = features[self.feature_names].to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("Model input contains missing or infinite values.")
        return matrix

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model is not fitted; call fit() first.")


def create_model(
    model_name: str,
    feature_names: Sequence[str],
    parameters: dict[str, Any] | None = None,
    *,
    random_state: int = GLOBAL_SEED,
) -> LightweightDetector:
    """Create one supported V0.5 model with a common reproducible interface."""
    return LightweightDetector(
        model_name, feature_names, parameters, random_state=random_state
    )


def _build_estimator(
    model_name: str, parameters: dict[str, Any], random_state: int
) -> Any:
    resolved = dict(parameters)
    resolved.setdefault("random_state", random_state)
    if model_name == "isolation_forest":
        return IsolationForest(**resolved)
    if model_name == "logistic_regression":
        return LogisticRegression(**resolved)
    if model_name == "decision_tree":
        return DecisionTreeClassifier(**resolved)
    if model_name == "random_forest":
        return RandomForestClassifier(**resolved)
    raise ValueError(f"Unsupported lightweight model: {model_name!r}")
