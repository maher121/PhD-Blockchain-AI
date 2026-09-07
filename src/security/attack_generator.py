"""Deterministic orchestration of controlled V0.4 attack scenarios."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from src.config import GLOBAL_SEED, PROJECT_ROOT
from src.security.attack_scenarios import apply_scenario, eligible_mask
from src.security.ground_truth import (
    MANIFEST_COLUMNS,
    build_ground_truth,
    empty_manifest,
    manifest_record,
)


DEFAULT_ATTACK_CONFIG = PROJECT_ROOT / "config" / "attack_scenarios.yaml"
RECORD_ID_CANDIDATES: tuple[str, ...] = (
    "row_id",
    "record_id",
    "Order Item Id",
    "transaction_id",
)


def load_attack_config(path: Path | str = DEFAULT_ATTACK_CONFIG) -> dict[str, Any]:
    """Load and minimally validate the versioned YAML scenario configuration."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Attack configuration not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("scenarios"):
        raise ValueError("Attack configuration must define scenarios.")
    rates = payload.get("attack_rates")
    severities = payload.get("severity_levels")
    if (
        not isinstance(rates, list)
        or not rates
        or any(not 0.0 < float(rate) <= 1.0 for rate in rates)
    ):
        raise ValueError("Attack configuration must define rates in (0, 1].")
    if not isinstance(severities, list) or not severities:
        raise ValueError("Attack configuration must define severity levels.")
    for name, definition in payload["scenarios"].items():
        if not definition.get("candidates") or not definition.get("transformation"):
            raise ValueError(f"Scenario {name!r} lacks candidates or transformation.")
        missing = set(severities) - set(definition.get("severity", {}))
        if missing:
            raise ValueError(f"Scenario {name!r} lacks severities: {sorted(missing)}")
    return payload


def generate_attack(
    dataframe: pd.DataFrame,
    attack_type: str | Iterable[str],
    attack_rate: float,
    severity: str,
    random_state: int | None = None,
    *,
    experiment_id: str | None = None,
    record_id_column: str | None = None,
    config_path: Path | str = DEFAULT_ATTACK_CONFIG,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Generate attacks before detection and return data, metadata, ground truth."""
    config = load_attack_config(config_path)
    random_state = int(
        config.get("default_random_state", GLOBAL_SEED)
        if random_state is None
        else random_state
    )
    scenarios = config["scenarios"]
    severity = str(severity).upper()
    if severity not in config["severity_levels"]:
        raise ValueError(f"Unsupported severity {severity!r}.")
    if not 0.0 <= float(attack_rate) <= 1.0:
        raise ValueError("attack_rate must be between 0 and 1 inclusive.")
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise ValueError("dataframe must be a non-empty pandas DataFrame.")
    if not dataframe.index.is_unique:
        raise ValueError("dataframe index must be unique.")

    selected_types = _resolve_attack_types(attack_type, scenarios)
    for name in selected_types:
        if severity not in scenarios[name].get("severity", {}):
            raise ValueError(f"Severity {severity} is not configured for {name}.")
    identifier = record_id_column or _resolve_record_id_column(dataframe)
    if dataframe[identifier].isna().any() or not dataframe[identifier].is_unique:
        raise ValueError("The record identifier column must be non-null and unique.")
    experiment_id = experiment_id or _default_experiment_id(
        selected_types, attack_rate, severity, random_state
    )

    masks = {name: eligible_mask(dataframe, scenarios[name]) for name in selected_types}
    eligible_union = pd.Series(False, index=dataframe.index, dtype=bool)
    for mask in masks.values():
        eligible_union |= mask
    eligible_indices = dataframe.index[eligible_union].tolist()
    target_count = _selection_count(len(eligible_indices), float(attack_rate))
    rng = np.random.default_rng(random_state)
    assignments = _assign_records(masks, target_count, rng)

    attacked = dataframe.copy(deep=True)
    records: list[dict[str, Any]] = []
    selected_by_type = {name: 0 for name in selected_types}
    for row_index, name in assignments:
        field, original, modified = apply_scenario(
            attacked, row_index, scenarios[name], severity, rng
        )
        if _same_value(original, modified):
            continue
        selected_by_type[name] += 1
        records.append(
            manifest_record(
                experiment_id=experiment_id,
                record_id=dataframe.at[row_index, identifier],
                attack_type=name,
                attack_severity=severity,
                original_field=field,
                original_dtype=str(dataframe[field].dtype),
                original_value=original,
                modified_value=modified,
                attack_rate=float(attack_rate),
                random_seed=random_state,
            )
        )

    manifest = (
        pd.DataFrame(records, columns=list(MANIFEST_COLUMNS)) if records else empty_manifest()
    )
    ground_truth = build_ground_truth(
        dataframe,
        manifest,
        record_id_column=identifier,
        attack_rate=float(attack_rate),
        random_seed=random_state,
        experiment_id=experiment_id,
    )
    actual_count = int(len(manifest))
    eligible_count = int(len(eligible_indices))
    metadata = {
        "stage": "V0.4",
        "experiment_id": experiment_id,
        "attack_mode": "mixed" if len(selected_types) > 1 else "single",
        "attack_types": selected_types,
        "severity": severity,
        "configured_attack_rate": float(attack_rate),
        "eligible_records": eligible_count,
        "selected_records": int(len(assignments)),
        "actual_modified_records": actual_count,
        "attack_rate_achieved": (
            float(actual_count / eligible_count) if eligible_count else 0.0
        ),
        "random_seed": random_state,
        "record_id_column": identifier,
        "per_attack_type": {
            name: {
                "eligible_records": int(masks[name].sum()),
                "actual_modified_records": int(selected_by_type[name]),
            }
            for name in selected_types
        },
        "manifest": manifest,
    }
    return attacked, metadata, ground_truth


def apply_attack(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Public alias matching the apply/restore research interface."""
    return generate_attack(*args, **kwargs)


def restore_original(
    attacked_dataframe: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    record_id_column: str | None = None,
) -> pd.DataFrame:
    """Restore source values from an in-memory attack manifest."""
    restored = attacked_dataframe.copy(deep=True)
    if manifest.empty:
        return restored
    identifier = record_id_column or _resolve_record_id_column(restored)
    if restored[identifier].isna().any() or not restored[identifier].is_unique:
        raise ValueError("The record identifier column must be non-null and unique.")
    positions = pd.Series(restored.index, index=restored[identifier]).to_dict()
    original_dtypes: dict[str, str] = {}
    for row in manifest.itertuples(index=False):
        if row.record_id not in positions:
            raise ValueError(f"Manifest record {row.record_id!r} is absent from dataframe.")
        if row.original_field not in restored:
            raise ValueError(f"Manifest field {row.original_field!r} is absent from dataframe.")
        row_index = positions[row.record_id]
        current = restored.at[row_index, row.original_field]
        if not _same_serialized_value(current, row.modified_value):
            raise ValueError(
                f"Manifest modified value does not match record {row.record_id!r}, "
                f"field {row.original_field!r}."
            )
        restored.at[row_index, row.original_field] = _coerce_restored_value(
            row.original_value, restored[row.original_field].dtype
        )
        if hasattr(row, "original_dtype"):
            original_dtypes[row.original_field] = str(row.original_dtype)
    for field, dtype in original_dtypes.items():
        restored[field] = restored[field].astype(dtype)
    return restored


def _resolve_attack_types(
    attack_type: str | Iterable[str], scenarios: dict[str, Any]
) -> list[str]:
    if isinstance(attack_type, str):
        names = list(scenarios) if attack_type.lower() == "mixed" else [attack_type]
    else:
        requested_names = list(attack_type)
        unknown = sorted(set(requested_names) - set(scenarios))
        if unknown:
            raise ValueError(f"Unknown attack types: {unknown}")
        requested = set(requested_names)
        names = [name for name in scenarios if name in requested]
    if not names:
        raise ValueError("At least one attack type is required.")
    unknown = sorted(set(names) - set(scenarios))
    if unknown:
        raise ValueError(f"Unknown attack types: {unknown}")
    return names


def _resolve_record_id_column(dataframe: pd.DataFrame) -> str:
    for column in RECORD_ID_CANDIDATES:
        if column in dataframe:
            return column
    raise ValueError(
        "No stable record identifier found; provide record_id_column explicitly."
    )


def _selection_count(eligible_count: int, attack_rate: float) -> int:
    if eligible_count == 0 or attack_rate == 0.0:
        return 0
    return min(eligible_count, max(1, int(round(eligible_count * attack_rate))))


def _assign_records(
    masks: dict[str, pd.Series], target_count: int, rng: np.random.Generator
) -> list[tuple[Any, str]]:
    if target_count == 0:
        return []
    available = {
        name: list(rng.permutation(mask.index[mask].to_numpy()))
        for name, mask in masks.items()
    }
    names = list(masks)
    assigned: list[tuple[Any, str]] = []
    used: set[Any] = set()
    while len(assigned) < target_count:
        progress = False
        for name in names:
            while available[name] and available[name][-1] in used:
                available[name].pop()
            if available[name]:
                row_index = available[name].pop()
                used.add(row_index)
                assigned.append((row_index, name))
                progress = True
                if len(assigned) == target_count:
                    break
        if not progress:
            break
    return assigned


def _default_experiment_id(
    attack_types: list[str], attack_rate: float, severity: str, random_state: int
) -> str:
    scenario = "mixed" if len(attack_types) > 1 else attack_types[0]
    if len(attack_types) > 1:
        digest = hashlib.sha256("|".join(attack_types).encode("utf-8")).hexdigest()[:8]
        scenario = f"mixed_{digest}"
    rate = f"{attack_rate:.4f}".replace(".", "p")
    return f"v0_4_{scenario}_r{rate}_{severity.lower()}_s{int(random_state)}"


def _same_value(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(left == right)


def _same_serialized_value(left: Any, right: Any) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-9))
    except (TypeError, ValueError):
        return str(left) == str(right)


def _coerce_restored_value(value: Any, dtype: Any) -> Any:
    if pd.api.types.is_integer_dtype(dtype):
        return int(float(value))
    if pd.api.types.is_float_dtype(dtype):
        return float(value)
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return pd.to_datetime(value)
    return value
