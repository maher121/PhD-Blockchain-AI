"""Ground-truth and leakage contracts for controlled V0.4 experiments."""

from __future__ import annotations

from typing import Any

import pandas as pd


ATTACK_METADATA_COLUMNS: tuple[str, ...] = (
    "is_attack",
    "attack_type",
    "attack_severity",
    "original_field",
    "original_dtype",
    "original_value",
    "modified_value",
    "experiment_id",
    "attack_rate",
    "random_seed",
)

MANIFEST_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "record_id",
    "attack_type",
    "attack_severity",
    "original_field",
    "original_dtype",
    "original_value",
    "modified_value",
    "attack_rate",
    "random_seed",
)


def build_ground_truth(
    dataframe: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    record_id_column: str,
    attack_rate: float,
    random_seed: int,
    experiment_id: str,
) -> pd.DataFrame:
    """Return one separate experimental label row for every input record."""
    if record_id_column not in dataframe:
        raise ValueError(f"Missing record identifier column: {record_id_column}")
    identifiers = dataframe[record_id_column]
    if identifiers.isna().any() or not identifiers.is_unique:
        raise ValueError("Record identifiers must be non-null and unique.")

    ground_truth = pd.DataFrame(
        {
            "record_id": identifiers.to_numpy(copy=True),
            "is_attack": 0,
            "attack_type": pd.Series([None] * len(dataframe), dtype="object"),
            "attack_severity": pd.Series([None] * len(dataframe), dtype="object"),
            "original_field": pd.Series([None] * len(dataframe), dtype="object"),
            "original_value": pd.Series([None] * len(dataframe), dtype="object"),
            "modified_value": pd.Series([None] * len(dataframe), dtype="object"),
            "attack_rate": float(attack_rate),
            "random_seed": int(random_seed),
            "experiment_id": str(experiment_id),
        }
    )
    if manifest.empty:
        ground_truth["is_attack"] = ground_truth["is_attack"].astype("int8")
        return ground_truth

    if manifest["record_id"].duplicated().any():
        raise ValueError("A record may have at most one attack in an experiment.")
    details = manifest.set_index("record_id")
    attacked = ground_truth["record_id"].isin(details.index)
    ground_truth.loc[attacked, "is_attack"] = 1
    for column in (
        "attack_type",
        "attack_severity",
        "original_field",
        "original_value",
        "modified_value",
    ):
        ground_truth.loc[attacked, column] = ground_truth.loc[attacked, "record_id"].map(
            details[column]
        )
    ground_truth["is_attack"] = ground_truth["is_attack"].astype("int8")
    return ground_truth


def assert_no_attack_metadata(features: pd.DataFrame | list[str] | tuple[str, ...]) -> None:
    """Fail closed if experimental labels or trace data enter model inputs."""
    columns = features.columns if isinstance(features, pd.DataFrame) else features
    forbidden = sorted(set(columns) & set(ATTACK_METADATA_COLUMNS))
    if forbidden:
        raise ValueError(f"Attack metadata must not enter the AI feature matrix: {forbidden}")


def empty_manifest() -> pd.DataFrame:
    """Return an empty manifest with a stable serialization schema."""
    return pd.DataFrame(columns=list(MANIFEST_COLUMNS))


def manifest_record(**values: Any) -> dict[str, Any]:
    """Build one manifest row while enforcing the public schema."""
    return {column: values.get(column) for column in MANIFEST_COLUMNS}
