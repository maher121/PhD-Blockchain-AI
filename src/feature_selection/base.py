"""Common leakage-safe feature-selector contract for V0.6 and future methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, fields
from numbers import Integral
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Mapping, Sequence, Self

import numpy as np
import pandas as pd
from pandas.api.types import is_complex_dtype, is_numeric_dtype

from src.config import GLOBAL_SEED
from src.data.versioning import sha256_frame
from src.lightweight.feature_reduction import validate_model_features

SelectionMode = Literal["unsupervised", "supervised"]
FEATURE_SELECTOR_CONTRACT_VERSION = "v0.6-a"
CONTROLLED_ATTACK_LABEL_SOURCE = "controlled_training_is_attack"


@dataclass(frozen=True)
class SelectionResult:
    """Frozen structured summary of a selector and its training inputs."""

    contract_version: str
    selector_id: str
    selector_type: SelectionMode
    method: str
    parameters: Mapping[str, Any]
    labels_used: bool
    candidate_feature_count: int
    selected_feature_count: int
    candidate_features: tuple[str, ...]
    candidate_features_sha256: str
    selected_features: tuple[str, ...]
    selected_indices: tuple[int, ...]
    support_mask: tuple[bool, ...]
    random_state: int
    training_row_count: int
    training_row_source: str
    training_rows_sha256: str
    training_features_sha256: str
    training_labels_sha256: str | None
    training_label_name: str | None
    training_label_source: str | None
    warnings: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parameters", MappingProxyType(deepcopy(dict(self.parameters)))
        )
        object.__setattr__(
            self, "diagnostics", MappingProxyType(deepcopy(dict(self.diagnostics)))
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deep-copied plain dictionary of the result metadata."""
        payload = {item.name: getattr(self, item.name) for item in fields(self)}
        payload["parameters"] = deepcopy(dict(self.parameters))
        payload["diagnostics"] = deepcopy(dict(self.diagnostics))
        return payload


class BaseFeatureSelector(ABC):
    """Template contract that owns validation, state, and output ordering."""

    selector_id: ClassVar[str] = ""
    selector_type: ClassVar[SelectionMode]
    method: ClassVar[str] = ""

    def __init__(
        self,
        candidate_features: Sequence[str],
        *,
        n_features_to_select: int | None = None,
        random_state: int = GLOBAL_SEED,
    ) -> None:
        self.candidate_features = tuple(candidate_features)
        if any(not isinstance(name, str) for name in self.candidate_features):
            raise ValueError("Candidate feature names must be strings.")
        validate_model_features(self.candidate_features)
        self._validate_identity()
        self.n_features_to_select = self._validate_feature_count(n_features_to_select)
        if isinstance(random_state, bool) or not isinstance(random_state, Integral):
            raise ValueError("random_state must be an integer.")
        self.random_state = int(random_state)
        self.result_: SelectionResult | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether this selector has completed one successful fit."""
        return self.result_ is not None

    @property
    def result(self) -> SelectionResult:
        """Return fitted selector metadata, failing clearly before fit."""
        if self.result_ is None:
            raise RuntimeError("Feature selector is not fitted.")
        return self.result_

    def fit(
        self,
        training_features: pd.DataFrame,
        labels: pd.Series | np.ndarray | None = None,
        *,
        label_source: str | None = None,
    ) -> Self:
        """Fit once on a governed training matrix and optional training labels."""
        if self.is_fitted:
            raise RuntimeError("Feature selector is already fitted; create a new instance.")

        features = self._validate_feature_frame(training_features, context="Training")
        row_provenance, row_source = _row_provenance(training_features)
        training_rows_sha256 = sha256_frame(row_provenance)
        training_features_sha256 = _feature_fingerprint(row_provenance, features)
        prepared_labels, resolved_label_source = self._prepare_labels(
            features, labels, label_source
        )
        training_labels_sha256 = (
            _label_fingerprint(row_provenance, prepared_labels)
            if prepared_labels is not None
            else None
        )
        parameters = self.get_params()
        raw_support = self._fit_support(features, prepared_labels)
        support = self._validate_support(raw_support)
        selected_indices = tuple(int(index) for index in np.flatnonzero(support))
        selected_features = tuple(self.candidate_features[index] for index in selected_indices)
        warnings = tuple(self._get_warnings())
        if any(not isinstance(message, str) or not message for message in warnings):
            raise ValueError("Selector warnings must be non-empty strings.")

        self.result_ = SelectionResult(
            contract_version=FEATURE_SELECTOR_CONTRACT_VERSION,
            selector_id=self.selector_id,
            selector_type=self.selector_type,
            method=self.method,
            parameters=parameters,
            labels_used=prepared_labels is not None,
            candidate_feature_count=len(self.candidate_features),
            selected_feature_count=len(selected_features),
            candidate_features=self.candidate_features,
            candidate_features_sha256=_candidate_fingerprint(self.candidate_features),
            selected_features=selected_features,
            selected_indices=selected_indices,
            support_mask=tuple(bool(value) for value in support),
            random_state=self.random_state,
            training_row_count=len(features),
            training_row_source=row_source,
            training_rows_sha256=training_rows_sha256,
            training_features_sha256=training_features_sha256,
            training_labels_sha256=training_labels_sha256,
            training_label_name=(
                str(prepared_labels.name)
                if prepared_labels is not None and prepared_labels.name is not None
                else None
            ),
            training_label_source=resolved_label_source,
            warnings=warnings,
            diagnostics=dict(self._get_diagnostics()),
        )
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Project a compatible matrix onto fitted features in canonical order."""
        result = self.result
        validated = self._validate_feature_frame(features, context="Transform")
        return validated.loc[:, list(result.selected_features)].copy()

    def fit_transform(
        self,
        training_features: pd.DataFrame,
        labels: pd.Series | np.ndarray | None = None,
        *,
        label_source: str | None = None,
    ) -> pd.DataFrame:
        """Fit on training data and return its selected representation."""
        return self.fit(
            training_features, labels, label_source=label_source
        ).transform(training_features)

    def get_support(self, *, indices: bool = False) -> np.ndarray:
        """Return a copy of the fitted support mask or canonical indices."""
        result = self.result
        if indices:
            return np.asarray(result.selected_indices, dtype=int)
        return np.asarray(result.support_mask, dtype=bool)

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        """Return selected names and optionally verify the candidate ordering."""
        result = self.result
        if input_features is not None and tuple(input_features) != self.candidate_features:
            raise ValueError("input_features must match the candidate feature ordering.")
        return np.asarray(result.selected_features, dtype=object)

    def get_params(self) -> dict[str, Any]:
        """Return resolved base and method-specific selector parameters."""
        parameters = {
            "n_features_to_select": self.n_features_to_select,
            "random_state": self.random_state,
        }
        method_parameters = dict(self._get_method_parameters())
        overlap = set(parameters) & set(method_parameters)
        if overlap:
            raise ValueError(f"Method parameters redefine base parameters: {sorted(overlap)}")
        parameters.update(method_parameters)
        return parameters

    def _validate_identity(self) -> None:
        if not isinstance(self.selector_id, str) or not self.selector_id.strip():
            raise ValueError("selector_id must be a non-empty string.")
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("method must be a non-empty string.")
        if self.selector_type not in ("unsupervised", "supervised"):
            raise ValueError("selector_type must be 'unsupervised' or 'supervised'.")

    def _validate_feature_count(self, value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("n_features_to_select must be an integer or None.")
        count = int(value)
        if not 1 <= count <= len(self.candidate_features):
            raise ValueError(
                "n_features_to_select must be between 1 and the candidate feature count."
            )
        return count

    def _validate_feature_frame(
        self, features: pd.DataFrame, *, context: str
    ) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame):
            raise ValueError(f"{context} features must be a pandas DataFrame.")
        if features.empty:
            raise ValueError(f"{context} features must not be empty.")
        duplicate_columns = features.columns[features.columns.duplicated()].tolist()
        if duplicate_columns:
            raise ValueError(f"{context} features contain duplicate columns: {duplicate_columns}")
        actual = set(features.columns)
        expected = set(self.candidate_features)
        missing = [name for name in self.candidate_features if name not in actual]
        unexpected = [
            name for name in features.columns if name not in expected and name != "row_id"
        ]
        if missing:
            raise ValueError(f"{context} data is missing candidate features: {missing}")
        if unexpected:
            raise ValueError(f"{context} data contains unexpected features: {unexpected}")
        if not features.index.is_unique:
            raise ValueError(f"{context} feature rows must have a unique index.")
        if "row_id" in features:
            if features["row_id"].isna().any() or not features["row_id"].is_unique:
                raise ValueError(f"{context} row_id values must be non-missing and unique.")

        canonical = features.loc[:, list(self.candidate_features)].copy()
        if any(
            not is_numeric_dtype(dtype) or is_complex_dtype(dtype)
            for dtype in canonical.dtypes
        ):
            raise ValueError(f"{context} features must have numeric dtypes.")
        try:
            values = canonical.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context} features must be numeric.") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"{context} features contain missing or infinite values.")
        return canonical

    def _validate_support(self, support: Sequence[bool] | np.ndarray) -> np.ndarray:
        values = np.asarray(support)
        if values.ndim != 1 or len(values) != len(self.candidate_features):
            raise ValueError("Selector support mask must match the candidate feature count.")
        if values.dtype.kind != "b":
            raise ValueError("Selector support mask must contain boolean values.")
        selected_count = int(values.sum())
        if selected_count == 0:
            raise ValueError("Selector must retain at least one feature.")
        if (
            self.n_features_to_select is not None
            and selected_count != self.n_features_to_select
        ):
            raise ValueError(
                "Selector support mask does not match n_features_to_select."
            )
        return values.astype(bool, copy=True)

    @abstractmethod
    def _prepare_labels(
        self,
        training_features: pd.DataFrame,
        labels: pd.Series | np.ndarray | None,
        label_source: str | None,
    ) -> tuple[pd.Series | None, str | None]:
        """Apply the supervision policy and normalize labels."""

    @abstractmethod
    def _fit_support(
        self, training_features: pd.DataFrame, labels: pd.Series | None
    ) -> Sequence[bool] | np.ndarray:
        """Return a boolean support mask in canonical candidate order."""

    def _get_method_parameters(self) -> Mapping[str, Any]:
        return {}

    def _get_warnings(self) -> Sequence[str]:
        return ()

    def _get_diagnostics(self) -> Mapping[str, Any]:
        return {}


class UnsupervisedFeatureSelector(BaseFeatureSelector, ABC):
    """Base for selectors that must fit without labels."""

    selector_type: ClassVar[SelectionMode] = "unsupervised"

    def _prepare_labels(
        self,
        training_features: pd.DataFrame,
        labels: pd.Series | np.ndarray | None,
        label_source: str | None,
    ) -> tuple[None, None]:
        if labels is not None or label_source is not None:
            raise ValueError("Unsupervised selectors do not accept labels or label sources.")
        return None, None

    def _fit_support(
        self, training_features: pd.DataFrame, labels: pd.Series | None
    ) -> Sequence[bool] | np.ndarray:
        if labels is not None:  # Defensive invariant; public fit prevents this path.
            raise RuntimeError("Unsupervised selector received normalized labels.")
        return self._fit_unsupervised(training_features)

    @abstractmethod
    def _fit_unsupervised(
        self, training_features: pd.DataFrame
    ) -> Sequence[bool] | np.ndarray:
        """Fit without labels and return a support mask."""


class SupervisedFeatureSelector(BaseFeatureSelector, ABC):
    """Base for selectors that require aligned binary training labels."""

    selector_type: ClassVar[SelectionMode] = "supervised"

    def _prepare_labels(
        self,
        training_features: pd.DataFrame,
        labels: pd.Series | np.ndarray | None,
        label_source: str | None,
    ) -> tuple[pd.Series, str]:
        if labels is None:
            raise ValueError("Supervised selectors require training labels.")
        if label_source is None:
            if isinstance(labels, pd.Series) and labels.name == "is_attack":
                label_source = CONTROLLED_ATTACK_LABEL_SOURCE
            else:
                raise ValueError(
                    "Supervised selectors require an explicit controlled training label source."
                )
        if label_source != CONTROLLED_ATTACK_LABEL_SOURCE:
            raise ValueError(
                "Supervised selectors require controlled training is_attack labels."
            )
        if isinstance(labels, pd.Series):
            if labels.name not in (None, "is_attack"):
                raise ValueError(
                    "Supervised selector labels must be named is_attack when named."
                )
            if not labels.index.equals(training_features.index):
                raise ValueError("Training labels must align exactly with feature rows.")
            prepared = labels.copy()
        elif isinstance(labels, np.ndarray):
            if labels.ndim != 1:
                raise ValueError("Training labels must be one-dimensional.")
            if len(labels) != len(training_features):
                raise ValueError("Training labels must match the feature row count.")
            prepared = pd.Series(labels, index=training_features.index)
        else:
            raise ValueError("Training labels must be a pandas Series or NumPy array.")
        if len(prepared) != len(training_features):
            raise ValueError("Training labels must match the feature row count.")
        if prepared.isna().any():
            raise ValueError("Training labels must not contain missing values.")
        try:
            values = prepared.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Training labels must be numeric binary values.") from exc
        if not np.isfinite(values).all() or set(np.unique(values)) != {0.0, 1.0}:
            raise ValueError("Training labels must contain both binary classes 0 and 1.")
        normalized = pd.Series(
            values.astype(np.int8), index=training_features.index, name=prepared.name
        )
        return normalized, label_source

    def _fit_support(
        self, training_features: pd.DataFrame, labels: pd.Series | None
    ) -> Sequence[bool] | np.ndarray:
        if labels is None:  # Defensive invariant; public fit prevents this path.
            raise RuntimeError("Supervised selector did not receive normalized labels.")
        return self._fit_supervised(training_features, labels)

    @abstractmethod
    def _fit_supervised(
        self, training_features: pd.DataFrame, labels: pd.Series
    ) -> Sequence[bool] | np.ndarray:
        """Fit with training labels and return a support mask."""


def _index_frame(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.index.to_frame(index=False)
    frame.columns = [f"__row_index_{index}__" for index in range(frame.shape[1])]
    return frame


def _row_provenance(features: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if "row_id" in features:
        frame = features[["row_id"]].reset_index(drop=True).copy()
        frame.columns = ["__row_id__"]
        return frame, "row_id_column"
    return _index_frame(features), "dataframe_index"


def _candidate_fingerprint(candidate_features: Sequence[str]) -> str:
    return sha256_frame(
        pd.DataFrame(
            {
                "position": np.arange(len(candidate_features), dtype=int),
                "feature": list(candidate_features),
            }
        )
    )


def _feature_fingerprint(
    row_provenance: pd.DataFrame, features: pd.DataFrame
) -> str:
    fingerprint_frame = pd.concat(
        [row_provenance, features.reset_index(drop=True)], axis=1
    )
    return sha256_frame(fingerprint_frame)


def _label_fingerprint(
    row_provenance: pd.DataFrame, labels: pd.Series
) -> str:
    fingerprint_frame = pd.concat(
        [
            row_provenance,
            labels.rename("__training_label__").reset_index(drop=True),
        ],
        axis=1,
    )
    return sha256_frame(fingerprint_frame)
