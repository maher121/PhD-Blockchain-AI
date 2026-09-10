"""Controlled attacked-split composition for V0.6 experiments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.versioning import sha256_frame
from src.security.attack_generator import DEFAULT_ATTACK_CONFIG, generate_attack
from src.security.evaluation import project_attacks_to_feature_space
from src.security.ground_truth import MANIFEST_COLUMNS, assert_no_attack_metadata

if TYPE_CHECKING:
    from src.ai.model_utils import ProcessedSplit

VALID_EXPERIMENT_SPLITS: tuple[str, ...] = ("train", "validation", "test")


@dataclass(frozen=True)
class PreparedAttackSplit:
    """One split with clean inputs, controlled attacks, and separate labels."""

    split_name: str
    candidate_features: tuple[str, ...]
    clean_features: pd.DataFrame
    features: pd.DataFrame
    clean_metadata: pd.DataFrame
    metadata: pd.DataFrame
    labels: pd.Series
    ground_truth: pd.DataFrame
    manifest: pd.DataFrame
    attack_metadata: dict[str, Any]
    row_ids_sha256: str
    clean_features_sha256: str
    attacked_features_sha256: str
    labels_sha256: str
    ground_truth_sha256: str
    manifest_sha256: str
    attack_metadata_sha256: str
    test_authorization_id: str | None
    test_authorization_capability: object | None


@dataclass
class DevelopmentExperimentData:
    """Train and validation data exposed to development-phase evaluation."""

    candidate_features: tuple[str, ...]
    train: PreparedAttackSplit
    validation: PreparedAttackSplit
    dataset_metadata: dict[str, Any]


def prepare_attack_split(
    *,
    split: ProcessedSplit,
    training_reference: ProcessedSplit,
    candidate_features: Sequence[str],
    attack_type: str | Sequence[str],
    attack_rate: float,
    severity: str,
    random_seed: int,
    experiment_id: str,
    config_path: Path | str = DEFAULT_ATTACK_CONFIG,
    test_authorization_id: str | None = None,
    test_authorization_capability: object | None = None,
) -> PreparedAttackSplit:
    """Generate and project one split using clean training references only."""
    if split.name not in VALID_EXPERIMENT_SPLITS:
        raise ValueError(f"Unsupported experiment split: {split.name!r}")
    if training_reference.name != "train":
        raise ValueError("Attack projection requires the clean training split as reference.")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be a non-empty string.")
    _validate_attack_arguments(attack_rate, severity, random_seed)

    candidates = tuple(candidate_features)
    from src.lightweight.feature_reduction import validate_model_features

    validate_model_features(candidates)
    clean_features = split.features.copy(deep=True)
    clean_metadata = split.metadata.copy(deep=True)
    reference_features = training_reference.features.copy(deep=True)
    reference_metadata = training_reference.metadata.copy(deep=True)
    _validate_processed_pair(clean_features, clean_metadata, candidates, split.name)
    _validate_processed_pair(
        reference_features, reference_metadata, candidates, "training reference"
    )

    attacked_metadata, attack_metadata, ground_truth = generate_attack(
        clean_metadata,
        attack_type,
        float(attack_rate),
        str(severity).upper(),
        int(random_seed),
        experiment_id=experiment_id,
        config_path=config_path,
    )
    manifest = attack_metadata.pop("manifest")
    if list(manifest.columns) != list(MANIFEST_COLUMNS):
        raise ValueError("Controlled attack manifest does not match the V0.4 schema.")

    attacked_features = project_attacks_to_feature_space(
        clean_features,
        clean_metadata,
        attacked_metadata,
        manifest,
        reference_features=reference_features,
        reference_metadata=reference_metadata,
    )
    _validate_processed_pair(attacked_features, attacked_metadata, candidates, split.name)
    assert_no_attack_metadata(attacked_features)
    labels = _aligned_attack_labels(attacked_features, ground_truth)

    return PreparedAttackSplit(
        split_name=split.name,
        candidate_features=candidates,
        clean_features=clean_features,
        features=attacked_features,
        clean_metadata=clean_metadata,
        metadata=attacked_metadata,
        labels=labels,
        ground_truth=ground_truth.copy(deep=True),
        manifest=manifest.copy(deep=True),
        attack_metadata=deepcopy(attack_metadata),
        row_ids_sha256=fingerprint_row_ids(attacked_features),
        clean_features_sha256=fingerprint_feature_matrix(clean_features, candidates),
        attacked_features_sha256=fingerprint_feature_matrix(
            attacked_features, candidates
        ),
        labels_sha256=fingerprint_attack_labels(attacked_features, labels),
        ground_truth_sha256=fingerprint_frame(ground_truth),
        manifest_sha256=fingerprint_frame(manifest),
        attack_metadata_sha256=fingerprint_mapping(attack_metadata),
        test_authorization_id=test_authorization_id,
        test_authorization_capability=test_authorization_capability,
    )


def prepare_development_experiment_data(
    *,
    train: ProcessedSplit,
    validation: ProcessedSplit,
    candidate_features: Sequence[str],
    dataset_metadata: Mapping[str, Any],
    attack_type: str | Sequence[str] = "mixed",
    attack_rate: float = 0.05,
    severity: str = "MEDIUM",
    random_seed: int = 42,
    experiment_id: str = "v0_6_development_s42",
    config_path: Path | str = DEFAULT_ATTACK_CONFIG,
) -> DevelopmentExperimentData:
    """Prepare train and validation only; this API has no test input."""
    if train.name != "train" or validation.name != "validation":
        raise ValueError("Development data requires train and validation splits.")
    candidates = tuple(candidate_features)
    prepared_train = prepare_attack_split(
        split=train,
        training_reference=train,
        candidate_features=candidates,
        attack_type=attack_type,
        attack_rate=attack_rate,
        severity=severity,
        random_seed=random_seed,
        experiment_id=f"{experiment_id}_train",
        config_path=config_path,
    )
    prepared_validation = prepare_attack_split(
        split=validation,
        training_reference=train,
        candidate_features=candidates,
        attack_type=attack_type,
        attack_rate=attack_rate,
        severity=severity,
        random_seed=random_seed,
        experiment_id=f"{experiment_id}_validation",
        config_path=config_path,
    )
    return DevelopmentExperimentData(
        candidate_features=candidates,
        train=prepared_train,
        validation=prepared_validation,
        dataset_metadata=deepcopy(dict(dataset_metadata)),
    )


def prepare_test_experiment_data(
    *,
    test: ProcessedSplit,
    training_reference: ProcessedSplit,
    candidate_features: Sequence[str],
    attack_type: str | Sequence[str] = "mixed",
    attack_rate: float = 0.05,
    severity: str = "MEDIUM",
    random_seed: int = 42,
    experiment_id: str = "v0_6_final_test_s42",
    config_path: Path | str = DEFAULT_ATTACK_CONFIG,
    validation_lock_id: str,
    authorization_capability: object,
) -> PreparedAttackSplit:
    """Prepare test data explicitly and separately from development data."""
    if test.name != "test":
        raise ValueError("Final-test preparation requires a test split.")
    if not isinstance(validation_lock_id, str) or not validation_lock_id.strip():
        raise ValueError("Final-test preparation requires a validation lock ID.")
    if authorization_capability is None:
        raise ValueError("Final-test preparation requires a lock capability.")
    return prepare_attack_split(
        split=test,
        training_reference=training_reference,
        candidate_features=candidate_features,
        attack_type=attack_type,
        attack_rate=attack_rate,
        severity=severity,
        random_seed=random_seed,
        experiment_id=experiment_id,
        config_path=config_path,
        test_authorization_id=validation_lock_id,
        test_authorization_capability=authorization_capability,
    )


def fingerprint_row_ids(features: pd.DataFrame) -> str:
    """Fingerprint ordered stable row IDs using the V0.6 selector convention."""
    if "row_id" not in features:
        raise ValueError("Experiment features require row_id provenance.")
    rows = features[["row_id"]].reset_index(drop=True).copy()
    rows.columns = ["__row_id__"]
    return sha256_frame(rows)


def fingerprint_feature_matrix(
    features: pd.DataFrame, candidate_features: Sequence[str]
) -> str:
    """Fingerprint ordered row IDs and the canonical model feature matrix."""
    candidates = tuple(candidate_features)
    missing = [feature for feature in candidates if feature not in features]
    if missing:
        raise ValueError(f"Feature fingerprint is missing candidates: {missing}")
    rows = features[["row_id"]].reset_index(drop=True).copy()
    rows.columns = ["__row_id__"]
    matrix = features.loc[:, list(candidates)].reset_index(drop=True)
    return sha256_frame(pd.concat([rows, matrix], axis=1))


def fingerprint_attack_labels(features: pd.DataFrame, labels: pd.Series) -> str:
    """Fingerprint ordered row IDs and aligned controlled labels."""
    if not labels.index.equals(features.index):
        raise ValueError("Attack labels must align exactly with feature rows.")
    rows = features[["row_id"]].reset_index(drop=True).copy()
    rows.columns = ["__row_id__"]
    values = labels.rename("__training_label__").reset_index(drop=True)
    return sha256_frame(pd.concat([rows, values], axis=1))


def fingerprint_feature_names(feature_names: Sequence[str]) -> str:
    """Fingerprint an ordered selected-feature manifest."""
    frame = pd.DataFrame(
        {
            "position": np.arange(len(feature_names), dtype=int),
            "feature": list(feature_names),
        }
    )
    return sha256_frame(frame)


def fingerprint_frame(frame: pd.DataFrame) -> str:
    """Fingerprint a complete structured experiment table."""
    return sha256_frame(frame.reset_index(drop=True))


def fingerprint_mapping(values: Mapping[str, Any]) -> str:
    """Fingerprint JSON-compatible attack metadata deterministically."""
    payload = json.dumps(
        dict(values), sort_keys=True, separators=(",", ":"), default=_json_default
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _aligned_attack_labels(
    features: pd.DataFrame, ground_truth: pd.DataFrame
) -> pd.Series:
    if ground_truth["record_id"].duplicated().any():
        raise ValueError("Ground-truth record IDs must be unique.")
    row_ids = features["row_id"]
    if set(row_ids) != set(ground_truth["record_id"]):
        raise ValueError("Ground truth does not match processed feature row IDs.")
    lookup = ground_truth.set_index("record_id")["is_attack"]
    labels = lookup.loc[row_ids].astype("int8")
    labels.index = features.index
    labels.name = "is_attack"
    return labels


def _validate_attack_arguments(
    attack_rate: float, severity: str, random_seed: int
) -> None:
    if isinstance(attack_rate, bool) or not isinstance(attack_rate, Real):
        raise ValueError("attack_rate must be numeric and in [0, 1].")
    if not np.isfinite(float(attack_rate)) or not 0.0 <= float(attack_rate) <= 1.0:
        raise ValueError("attack_rate must be finite and in [0, 1].")
    if not isinstance(severity, str) or not severity.strip():
        raise ValueError("severity must be a non-empty string.")
    if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
        raise ValueError("random_seed must be an integer.")


def _validate_processed_pair(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    candidate_features: Sequence[str],
    split_name: str,
) -> None:
    if "row_id" not in features or "row_id" not in metadata:
        raise ValueError(f"{split_name} features and metadata require row_id.")
    if not features["row_id"].is_unique or not metadata["row_id"].is_unique:
        raise ValueError(f"{split_name} row_id values must be unique.")
    if features["row_id"].tolist() != metadata["row_id"].tolist():
        raise ValueError(f"{split_name} feature and metadata row IDs are misaligned.")
    missing = [feature for feature in candidate_features if feature not in features]
    if missing:
        raise ValueError(f"{split_name} is missing candidate features: {missing}")
    assert_no_attack_metadata(features)
    values = features.loc[:, list(candidate_features)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{split_name} candidate features must be finite.")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported metadata value for fingerprinting: {type(value)!r}")
