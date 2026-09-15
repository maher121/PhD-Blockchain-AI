"""Synthetic-only protocol validation for the pure V0.8-A BPSO engine.

This module does not load datasets, fit classifiers, select real features, or
read prior scientific results. Its only executable objectives are deterministic
synthetic functions used to validate optimizer mechanics and governance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from src.optimization.bpso import (
    BPSOConfig,
    BinaryParticleSwarmOptimizer,
    STOP_EARLY_NO_IMPROVEMENT,
    to_json_compatible,
)


V08A_STAGE = "V0.8-A"
V08A_SCHEMA_VERSION = "v0.8-a-bpso-protocol-1"
V08A_VALIDATION_KIND = "SYNTHETIC_PROTOCOL_VALIDATION"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "bpso.yaml"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "bpso" / "v08a_protocol_validation.json"
APPROVED_OPTIMIZER_SEEDS = (1042, 1043, 1044, 1045, 1046)
APPROVED_CARDINALITIES = (4, 8, 11, 14, 18, 22, 26, 30, 34, 38)


class V08AProtocolError(ValueError):
    """Raised when the frozen synthetic protocol configuration drifts."""


@dataclass(frozen=True)
class V08AProtocol:
    config: BPSOConfig
    optimizer_seeds: tuple[int, ...]
    synthetic_k42_inactive_index: int
    artifact_path: Path
    config_path: Path
    config_sha256: str
    raw_snapshot: Mapping[str, Any]


@dataclass(frozen=True)
class SyntheticEvaluation:
    """Synthetic constrained-style evaluation with explicit comparison fields."""

    feasible: bool
    violation: float
    cardinality: int
    primary_score: float
    secondary_score: float
    tie_break_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "violation": self.violation,
            "cardinality": self.cardinality,
            "primary_score": self.primary_score,
            "secondary_score": self.secondary_score,
            "tie_break_key": self.tie_break_key,
        }


def constrained_evaluation_is_better(
    left: SyntheticEvaluation, right: SyntheticEvaluation
) -> bool:
    """Apply feasible-first, minimum-cardinality synthetic ranking."""
    if left.feasible != right.feasible:
        return left.feasible
    if not left.feasible:
        left_key = (
            left.violation,
            -left.primary_score,
            -left.secondary_score,
            left.cardinality,
            left.tie_break_key,
        )
        right_key = (
            right.violation,
            -right.primary_score,
            -right.secondary_score,
            right.cardinality,
            right.tie_break_key,
        )
        return left_key < right_key
    left_key = (
        left.cardinality,
        -left.primary_score,
        -left.secondary_score,
        left.tie_break_key,
    )
    right_key = (
        right.cardinality,
        -right.primary_score,
        -right.secondary_score,
        right.tie_break_key,
    )
    return left_key < right_key


def score_evaluation_is_better(
    left: SyntheticEvaluation, right: SyntheticEvaluation
) -> bool:
    """Rank deterministic target-mask cases by synthetic score only."""
    return (
        -left.primary_score,
        -left.secondary_score,
        left.cardinality,
        left.tie_break_key,
    ) < (
        -right.primary_score,
        -right.secondary_score,
        right.cardinality,
        right.tie_break_key,
    )


def load_v08a_protocol(path: Path | str = DEFAULT_CONFIG_PATH) -> V08AProtocol:
    """Load the V0.8-A YAML and fail closed on any approved-protocol drift."""
    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise V08AProtocolError(f"Cannot load V0.8-A protocol: {exc}") from exc
    if not isinstance(payload, dict):
        raise V08AProtocolError("V0.8-A configuration must be a YAML mapping.")
    optimizer = payload.get("optimizer")
    governance = payload.get("governance")
    artifact = payload.get("artifact")
    if not all(isinstance(item, dict) for item in (optimizer, governance, artifact)):
        raise V08AProtocolError("V0.8-A optimizer, governance, and artifact sections are required.")
    assert isinstance(optimizer, dict)
    assert isinstance(governance, dict)
    assert isinstance(artifact, dict)
    initialization = optimizer.get("initialization")
    velocity = optimizer.get("velocity")
    inertia = optimizer.get("inertia")
    early = optimizer.get("early_stopping")
    cache = optimizer.get("cache")
    if not all(isinstance(item, dict) for item in (initialization, velocity, inertia, early, cache)):
        raise V08AProtocolError("V0.8-A optimizer sub-sections are incomplete.")
    assert isinstance(initialization, dict)
    assert isinstance(velocity, dict)
    assert isinstance(inertia, dict)
    assert isinstance(early, dict)
    assert isinstance(cache, dict)
    k42 = initialization.get("k42_anchor")
    if not isinstance(k42, dict):
        raise V08AProtocolError("V0.8-A requires a synthetic K42 anchor declaration.")

    expected_values = {
        "version": (payload.get("version"), V08A_SCHEMA_VERSION),
        "stage": (payload.get("stage"), V08A_STAGE),
        "validation_kind": (payload.get("validation_kind"), V08A_VALIDATION_KIND),
        "scientific_experiment": (payload.get("scientific_experiment"), False),
        "name": (optimizer.get("name"), "BPSO"),
        "dimensions": (optimizer.get("dimensions"), 43),
        "particles": (optimizer.get("particles"), 12),
        "evaluated_generations": (optimizer.get("evaluated_generations"), 20),
        "generation_index_origin": (optimizer.get("generation_index_origin"), 0),
        "maximum_fitness_requests_per_run": (
            optimizer.get("maximum_fitness_requests_per_run"),
            240,
        ),
        "optimizer_seeds": (
            tuple(optimizer.get("optimizer_seeds", ())),
            APPROVED_OPTIMIZER_SEEDS,
        ),
        "random_cardinalities": (
            tuple(initialization.get("random_cardinalities", ())),
            APPROVED_CARDINALITIES,
        ),
        "velocity_distribution": (
            velocity.get("initialization_distribution"),
            "uniform",
        ),
        "velocity_initialization": (
            tuple(velocity.get("initialization_bounds", ())),
            (-1.0, 1.0),
        ),
        "velocity_clamp": (tuple(velocity.get("clamp", ())), (-6.0, 6.0)),
        "transfer_function": (
            optimizer.get("transfer_function"),
            "standard_logistic_sigmoid",
        ),
        "inertia_schedule": (inertia.get("schedule"), "linear_decay"),
        "inertia_start": (inertia.get("start"), 0.9),
        "inertia_end": (inertia.get("end"), 0.4),
        "cognitive_coefficient": (optimizer.get("cognitive_coefficient"), 2.0),
        "social_coefficient": (optimizer.get("social_coefficient"), 2.0),
        "minimum_selected_features": (optimizer.get("minimum_selected_features"), 1),
        "early_stopping_patience": (early.get("patience_evaluated_generations"), 7),
        "early_stopping_min_generation": (early.get("not_before_generation_index"), 10),
        "cache_enabled": (cache.get("enabled"), True),
        "cache_scope": (cache.get("scope"), "one optimizer run"),
        "weighted_multi_objective": (governance.get("weighted_multi_objective"), False),
        "pareto_optimization": (governance.get("pareto_optimization"), False),
        "dataset_access": (governance.get("dataset_access"), "PROHIBITED"),
        "classifier_fitting": (governance.get("classifier_fitting"), "PROHIBITED"),
        "real_feature_selection": (governance.get("real_feature_selection"), "PROHIBITED"),
        "validation_artifact_label": (artifact.get("label"), V08A_VALIDATION_KIND),
        "artifact_scientific_experiment": (artifact.get("scientific_experiment"), False),
    }
    drift = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in expected_values.items()
        if observed != expected
    }
    if drift:
        raise V08AProtocolError(f"Frozen V0.8-A protocol drift: {drift}")

    config = BPSOConfig(
        dimensions=int(optimizer["dimensions"]),
        particle_count=int(optimizer["particles"]),
        evaluated_generations=int(optimizer["evaluated_generations"]),
        random_cardinalities=tuple(int(value) for value in initialization["random_cardinalities"]),
        velocity_initial_min=float(velocity["initialization_bounds"][0]),
        velocity_initial_max=float(velocity["initialization_bounds"][1]),
        velocity_clamp_min=float(velocity["clamp"][0]),
        velocity_clamp_max=float(velocity["clamp"][1]),
        inertia_start=float(inertia["start"]),
        inertia_end=float(inertia["end"]),
        cognitive_coefficient=float(optimizer["cognitive_coefficient"]),
        social_coefficient=float(optimizer["social_coefficient"]),
        minimum_selected_features=int(optimizer["minimum_selected_features"]),
        early_stopping_patience=int(early["patience_evaluated_generations"]),
        early_stopping_min_generation=int(early["not_before_generation_index"]),
        cache_enabled=bool(cache["enabled"]),
        transfer_function=str(optimizer["transfer_function"]),
        optimizer_name=str(optimizer["name"]),
    )
    if config.maximum_fitness_requests != 240:
        raise V08AProtocolError("V0.8-A request ceiling must be exactly 240 per run.")
    artifact_reference = Path(str(artifact.get("path", "")))
    if artifact_reference != Path("results/bpso/v08a_protocol_validation.json"):
        raise V08AProtocolError("V0.8-A validation artifact path changed.")
    inactive = int(k42.get("synthetic_inactive_index", -1))
    if inactive != 42:
        raise V08AProtocolError("Synthetic K42 anchor must use canonical inactive index 42.")
    return V08AProtocol(
        config=config,
        optimizer_seeds=APPROVED_OPTIMIZER_SEEDS,
        synthetic_k42_inactive_index=inactive,
        artifact_path=PROJECT_ROOT / artifact_reference,
        config_path=config_path.resolve(),
        config_sha256=hashlib.sha256(raw).hexdigest(),
        raw_snapshot=to_json_compatible(payload),
    )


def run_synthetic_protocol_validation(
    *,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_path: Path | str | None = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Execute deterministic synthetic checks and optionally publish one artifact."""
    protocol = load_v08a_protocol(config_path)
    config = protocol.config
    target = np.zeros(config.dimensions, dtype=np.uint8)
    target[[0, 5, 10, 20]] = 1

    def target_objective(mask: np.ndarray) -> SyntheticEvaluation:
        distance = int(np.count_nonzero(mask != target))
        return SyntheticEvaluation(
            feasible=True,
            violation=0.0,
            cardinality=int(mask.sum()),
            primary_score=float(-distance),
            secondary_score=float(-abs(int(mask.sum()) - int(target.sum()))),
            tie_break_key=_mask_text(mask),
        )

    target_optimizer = BinaryParticleSwarmOptimizer(
        config, target_objective, score_evaluation_is_better
    )
    first = target_optimizer.optimize(
        protocol.optimizer_seeds[0],
        k42_inactive_index=protocol.synthetic_k42_inactive_index,
    )
    replay = target_optimizer.optimize(
        protocol.optimizer_seeds[0],
        k42_inactive_index=protocol.synthetic_k42_inactive_index,
    )
    replay_identical = first.to_json() == replay.to_json()
    other_population, other_velocities = target_optimizer.initialize(
        protocol.optimizer_seeds[1],
        k42_inactive_index=protocol.synthetic_k42_inactive_index,
    )
    different_seed_changes_state = not (
        np.array_equal(first.initial_population, other_population)
        and np.array_equal(first.initial_velocities, other_velocities)
    )

    def constrained_objective(mask: np.ndarray) -> SyntheticEvaluation:
        feasible = bool(mask[0] == 1 and mask[1] == 1)
        return SyntheticEvaluation(
            feasible=feasible,
            violation=0.0 if feasible else float(2 - int(mask[0]) - int(mask[1])),
            cardinality=int(mask.sum()),
            primary_score=float(mask[2:6].sum()),
            secondary_score=float(-np.count_nonzero(mask[6:])),
            tie_break_key=_mask_text(mask),
        )

    constrained = BinaryParticleSwarmOptimizer(
        config, constrained_objective, constrained_evaluation_is_better
    ).optimize(
        protocol.optimizer_seeds[1],
        k42_inactive_index=protocol.synthetic_k42_inactive_index,
    )

    duplicate_population = np.zeros(
        (config.particle_count, config.dimensions), dtype=np.uint8
    )
    duplicate_population[:, 0] = 1
    duplicate_velocities = np.zeros_like(duplicate_population, dtype=float)
    invocation_count = 0

    def constant_objective(mask: np.ndarray) -> SyntheticEvaluation:
        nonlocal invocation_count
        invocation_count += 1
        return SyntheticEvaluation(True, 0.0, 0, 0.0, 0.0, "constant")

    cached = BinaryParticleSwarmOptimizer(
        config, constant_objective, score_evaluation_is_better
    ).optimize(
        protocol.optimizer_seeds[2],
        initial_population=duplicate_population,
        initial_velocities=duplicate_velocities,
    )
    uncached_count = 0

    def uncached_objective(mask: np.ndarray) -> SyntheticEvaluation:
        nonlocal uncached_count
        uncached_count += 1
        return SyntheticEvaluation(True, 0.0, 0, 0.0, 0.0, "constant")

    uncached_config = replace(config, cache_enabled=False)
    uncached = BinaryParticleSwarmOptimizer(
        uncached_config, uncached_objective, score_evaluation_is_better
    ).optimize(
        protocol.optimizer_seeds[2],
        initial_population=duplicate_population,
        initial_velocities=duplicate_velocities,
    )
    cache_decision_invariant = (
        np.array_equal(cached.best_mask, uncached.best_mask)
        and cached.best_evaluation == uncached.best_evaluation
        and [record.best_mask.tolist() for record in cached.convergence_history]
        == [record.best_mask.tolist() for record in uncached.convergence_history]
    )

    def anchor_dominant_objective(mask: np.ndarray) -> SyntheticEvaluation:
        return SyntheticEvaluation(
            True,
            0.0,
            int(mask.sum()),
            float(np.all(mask == 1)),
            0.0,
            _mask_text(mask),
        )

    early = BinaryParticleSwarmOptimizer(
        config, anchor_dominant_objective, score_evaluation_is_better
    ).optimize(
        protocol.optimizer_seeds[3],
        k42_inactive_index=protocol.synthetic_k42_inactive_index,
    )
    initial_cardinalities = first.initial_population.sum(axis=1).astype(int).tolist()
    expected_cardinalities = [43, 42, *APPROVED_CARDINALITIES]
    checks = {
        "configuration_matches_frozen_protocol": True,
        "generation_zero_is_inside_budget": first.convergence_history[0].generation_index == 0,
        "maximum_request_budget_is_240": config.maximum_fitness_requests == 240,
        "observed_requests_within_budget": all(
            result.total_fitness_requests <= 240
            for result in (first, replay, constrained, cached, early)
        ),
        "same_seed_complete_replay_identical": replay_identical,
        "different_seed_changes_stochastic_state": different_seed_changes_state,
        "approved_initial_cardinalities": initial_cardinalities == expected_cardinalities,
        "constrained_comparator_returns_feasible": constrained.best_evaluation.feasible,
        "cache_hit_recorded": cached.cache_hits > 0,
        "cache_invocations_equal_unique_evaluations": invocation_count == cached.unique_evaluations,
        "cache_accounting_valid": (
            cached.total_fitness_requests == cached.unique_evaluations + cached.cache_hits
        ),
        "cache_does_not_change_decisions": cache_decision_invariant,
        "uncached_invocations_equal_requests": uncached_count == uncached.total_fitness_requests,
        "early_stop_reason_valid": early.stop_reason == STOP_EARLY_NO_IMPROVEMENT,
        "early_stop_boundary_valid": (
            early.convergence_history[-1].generation_index == 10
            and early.evaluated_generation_count == 11
        ),
        "synthetic_validation_only": True,
        "scientific_experiment_is_false": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": V08A_SCHEMA_VERSION,
        "stage": V08A_STAGE,
        "status": status,
        "validation_kind": V08A_VALIDATION_KIND,
        "scientific_experiment": False,
        "scientific_claims_supported": False,
        "protocol_config_sha256": protocol.config_sha256,
        "optimizer_seeds": list(protocol.optimizer_seeds),
        "configuration": config.to_dict(),
        "generation_semantics": {
            "evaluated_indices": [0, config.evaluated_generations - 1],
            "initialization_generation": 0,
            "initialization_counts_toward_budget": True,
            "maximum_requests_per_run": config.maximum_fitness_requests,
        },
        "initialization": {
            "cardinalities": initial_cardinalities,
            "population_sha256": _json_sha256(first.initial_population.tolist()),
            "velocities_sha256": _json_sha256(first.initial_velocities.tolist()),
            "real_feature_identities_bound": False,
        },
        "synthetic_cases": {
            "target_mask": {
                "result_sha256": hashlib.sha256(first.to_json().encode("utf-8")).hexdigest(),
                "best_synthetic_score": first.best_evaluation.primary_score,
                "stop_reason": first.stop_reason,
            },
            "constrained_minimum_cardinality": {
                "best_evaluation": constrained.best_evaluation.to_dict(),
                "stop_reason": constrained.stop_reason,
            },
            "cache": {
                "requests": cached.total_fitness_requests,
                "unique_evaluations": cached.unique_evaluations,
                "cache_hits": cached.cache_hits,
            },
            "early_stopping": {
                "stop_reason": early.stop_reason,
                "evaluated_generation_count": early.evaluated_generation_count,
                "last_generation_index": early.convergence_history[-1].generation_index,
            },
        },
        "checks": checks,
        "governance": {
            "dataset_accessed": False,
            "classifier_fitted": False,
            "real_feature_subset_selected": False,
            "resource_benchmark_executed": False,
            "direct_energy_measured": False,
            "next_stage_started": False,
        },
    }
    summary = to_json_compatible(summary)
    if output_path is not None:
        _atomic_write_json(Path(output_path), summary)
    if status != "PASS":
        raise RuntimeError("V0.8-A synthetic protocol validation failed.")
    return summary


def _mask_text(mask: np.ndarray) -> str:
    return "".join(str(int(value)) for value in mask)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        to_json_compatible(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    summary = run_synthetic_protocol_validation()
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "status": summary["status"],
                "validation_kind": summary["validation_kind"],
                "scientific_experiment": summary["scientific_experiment"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
