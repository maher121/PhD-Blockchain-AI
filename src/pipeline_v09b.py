"""V0.9-B synthetic protocol validation for the pure BGWO engine.

This stage validates deterministic BGWO mechanics and protocol lock compliance
without loading DataCo, fitting governed production models, or accessing final
test data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import yaml

import src.optimization.bgwo as bgwo
import src.pipeline_v08d as v08d
import src.pipeline_v08e as v08e
import src.pipeline_v08e4 as v08e4


V09B_STAGE = "V0.9-B"
V09B_SCHEMA_VERSION = "v0.9-b-bgwo-protocol-validation-1"
V09B_VALIDATION_KIND = "SYNTHETIC_BGWO_IMPLEMENTATION_VALIDATION"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "bgwo_v09.yaml"
DEFAULT_PROTOCOL_DOC_PATH = PROJECT_ROOT / "docs" / "v09_bgwo_protocol.md"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "bpso" / "v09b_protocol_validation.json"

APPROVED_MODEL_ATTACK_SEEDS = (42, 43, 44, 45, 46)
APPROVED_OPTIMIZER_SEEDS = (2042, 2043, 2044, 2045, 2046)
APPROVED_CARDINALITIES = (4, 8, 11, 14, 18, 22, 26, 30, 34, 38)


class V09BProtocolError(ValueError):
    """Raised when the frozen V0.9-A protocol configuration drifts."""


@dataclass(frozen=True)
class V09BProtocol:
    config: bgwo.BGWOConfig
    optimizer_seeds: tuple[int, ...]
    split_seed: int
    model_attack_seeds: tuple[int, ...]
    config_path: Path
    protocol_doc_path: Path
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
    left: SyntheticEvaluation,
    right: SyntheticEvaluation,
) -> bool:
    """Apply frozen feasible/infeasible deterministic ranking semantics."""
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
    left: SyntheticEvaluation,
    right: SyntheticEvaluation,
) -> bool:
    """Rank deterministic target-mask objectives by score."""
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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V09BProtocolError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise V09BProtocolError(f"JSON payload must be an object: {path}")
    return payload


def current_head_short(root: Path = PROJECT_ROOT) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_v08_frozen_integrity() -> dict[str, Any]:
    """Re-verify governed V0.8 lock identities and immutable hash set."""
    preflight = _read_json(PROJECT_ROOT / "results" / "bpso" / "v08e_preflight.json")
    expected_hashes = preflight.get("immutable_hashes", {})
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise V09BProtocolError("Immutable hash map missing in v08e_preflight.json.")
    mismatches: dict[str, dict[str, str]] = {}
    for path, expected in expected_hashes.items():
        observed = v08e.canonical_lf_sha256(path)
        if observed != expected:
            mismatches[str(path)] = {"expected": str(expected), "observed": observed}

    v08d_lock = _read_json(PROJECT_ROOT / "results" / "bpso" / "v08d_final_test_lock.json")
    v08e_lock = _read_json(
        PROJECT_ROOT / "results" / "bpso" / "v08e_e4_analysis" / "v08e_scientific_result_lock.json"
    )
    v08d.verify_final_test_lock(v08d_lock)
    v08e4.verify_result_lock(v08e_lock)

    if mismatches:
        raise V09BProtocolError(
            f"Frozen V0.8 integrity mismatch count: {len(mismatches)}"
        )

    return {
        "status": "PASS",
        "immutable_hash_count": len(expected_hashes),
        "v08d_semantic_result_lock_sha256": v08d_lock["semantic_result_lock_sha256"],
        "v08e_semantic_result_lock_sha256": v08e_lock["semantic_result_lock_sha256"],
    }


def _assert_doc_yaml_consistency(doc_text: str, payload: Mapping[str, Any]) -> None:
    checks = {
        "population_12": "wolves (population) = `12`" in doc_text,
        "iterations_20": "evaluated iterations including initialization = `20`" in doc_text,
        "max_run_budget_240": "max fitness requests per run = `12 * 20 = 240`" in doc_text,
        "max_five_budget_1200": "maximum total requests across five runs = `1200`" in doc_text,
        "seed_family": "BGWO optimizer seeds frozen to: `2042, 2043, 2044, 2045, 2046`" in doc_text,
        "transfer": "S(x) = 1 / (1 + exp(-x))" in doc_text,
        "clamp": "clamp sigmoid input to `[-6, 6]`" in doc_text,
        "ap_threshold": "AP relative loss <= `5%`" in doc_text,
        "f1_threshold": "F1 relative loss <= `5%`" in doc_text,
        "recall_threshold": "Recall relative loss <= `10%`" in doc_text,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise V09BProtocolError(
            f"Protocol document and YAML consistency check failed: {failed}"
        )

    optimizer = payload.get("optimizer", {})
    budget = optimizer.get("budget", {})
    transfer = optimizer.get("binary_transfer", {})
    constraints = payload.get("predictive_baseline", {}).get("preservation_constraints", {})
    if (
        optimizer.get("wolves") != 12
        or optimizer.get("evaluated_iterations", {}).get("count_including_initialization") != 20
        or budget.get("maximum_fitness_requests_per_run") != 240
        or budget.get("maximum_fitness_requests_five_runs") != 1200
        or tuple(optimizer.get("optimizer_seeds", ())) != APPROVED_OPTIMIZER_SEEDS
        or transfer.get("function") != "standard_logistic_sigmoid"
        or tuple(transfer.get("input_clamp", ())) != (-6.0, 6.0)
        or constraints.get("average_precision_relative_loss_max") != 0.05
        or constraints.get("f1_relative_loss_max") != 0.05
        or constraints.get("recall_relative_loss_max") != 0.10
    ):
        raise V09BProtocolError("Protocol YAML values disagree with V0.9-A lock expectations.")


def load_v09b_protocol(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    protocol_doc_path: Path | str = DEFAULT_PROTOCOL_DOC_PATH,
) -> V09BProtocol:
    """Load the locked V0.9-A protocol and fail closed on drift."""
    config_target = Path(config_path)
    doc_target = Path(protocol_doc_path)
    try:
        raw_config = config_target.read_bytes()
        payload = yaml.safe_load(raw_config.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise V09BProtocolError(f"Cannot load V0.9-A YAML protocol: {exc}") from exc
    if not isinstance(payload, dict):
        raise V09BProtocolError("V0.9-A YAML configuration must be a mapping.")
    try:
        doc_text = doc_target.read_text(encoding="utf-8")
    except OSError as exc:
        raise V09BProtocolError(f"Cannot load V0.9-A protocol document: {exc}") from exc

    expected_values = {
        "version": (payload.get("version"), "v0.9-a-bgwo-protocol-1"),
        "stage": (payload.get("stage"), "V0.9-A"),
        "kind": (payload.get("kind"), "SCIENTIFIC_PROTOCOL_LOCK"),
        "scientific_experiment": (payload.get("scientific_experiment"), False),
        "dataset_row_cap": (payload.get("data_governance", {}).get("row_cap"), 40000),
        "feature_space": (payload.get("data_governance", {}).get("feature_space_count"), 43),
        "ground_truth": (payload.get("data_governance", {}).get("ground_truth"), "is_attack"),
        "split_seed": (payload.get("data_governance", {}).get("split", {}).get("seed"), 42),
        "search_classifier": (payload.get("search_classifier", {}).get("name"), "decision_tree"),
        "search_threshold": (
            payload.get("search_classifier", {}).get("prediction_threshold"),
            0.5,
        ),
        "optimizer_name": (payload.get("optimizer", {}).get("name"), "BGWO"),
        "dimensions": (payload.get("optimizer", {}).get("dimensions"), 43),
        "wolves": (payload.get("optimizer", {}).get("wolves"), 12),
        "independent_runs": (payload.get("optimizer", {}).get("independent_runs"), 5),
        "optimizer_seeds": (
            tuple(payload.get("optimizer", {}).get("optimizer_seeds", ())),
            APPROVED_OPTIMIZER_SEEDS,
        ),
        "evaluated_iterations": (
            payload.get("optimizer", {}).get("evaluated_iterations", {}).get("count_including_initialization"),
            20,
        ),
        "iterative_updates": (
            payload.get("optimizer", {}).get("evaluated_iterations", {}).get("iterative_updates_after_initialization"),
            19,
        ),
        "run_budget": (
            payload.get("optimizer", {}).get("budget", {}).get("maximum_fitness_requests_per_run"),
            240,
        ),
        "five_budget": (
            payload.get("optimizer", {}).get("budget", {}).get("maximum_fitness_requests_five_runs"),
            1200,
        ),
        "random_cardinalities": (
            tuple(payload.get("optimizer", {}).get("initialization", {}).get("random_exact_cardinalities", ())),
            APPROVED_CARDINALITIES,
        ),
        "transfer_function": (
            payload.get("optimizer", {}).get("binary_transfer", {}).get("function"),
            "standard_logistic_sigmoid",
        ),
        "transfer_clamp": (
            tuple(payload.get("optimizer", {}).get("binary_transfer", {}).get("input_clamp", ())),
            (-6.0, 6.0),
        ),
        "minimum_selected_features": (
            payload.get("optimizer", {}).get("repair", {}).get("minimum_selected_features"),
            1,
        ),
        "early_stopping_min_iteration": (
            payload.get("optimizer", {}).get("stopping", {}).get("not_before_evaluated_iteration_index"),
            10,
        ),
        "early_stopping_patience": (
            payload.get("optimizer", {}).get("stopping", {}).get("patience_evaluated_iterations"),
            7,
        ),
    }
    drift = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in expected_values.items()
        if observed != expected
    }
    if drift:
        raise V09BProtocolError(f"Frozen V0.9-A protocol drift: {drift}")

    _assert_doc_yaml_consistency(doc_text, payload)

    optimizer = payload["optimizer"]
    config = bgwo.BGWOConfig(
        dimensions=int(optimizer["dimensions"]),
        wolf_count=int(optimizer["wolves"]),
        evaluated_iterations=int(
            optimizer["evaluated_iterations"]["count_including_initialization"]
        ),
        random_cardinalities=tuple(
            int(value) for value in optimizer["initialization"]["random_exact_cardinalities"]
        ),
        sigmoid_clamp_min=float(optimizer["binary_transfer"]["input_clamp"][0]),
        sigmoid_clamp_max=float(optimizer["binary_transfer"]["input_clamp"][1]),
        control_parameter_start=float(optimizer["canonical_binary_gwo"]["control_parameter_a"]["start"]),
        control_parameter_end=float(optimizer["canonical_binary_gwo"]["control_parameter_a"]["end"]),
        minimum_selected_features=int(optimizer["repair"]["minimum_selected_features"]),
        early_stopping_patience=int(optimizer["stopping"]["patience_evaluated_iterations"]),
        early_stopping_min_iteration=int(
            optimizer["stopping"]["not_before_evaluated_iteration_index"]
        ),
        cache_enabled=bool(optimizer["cache"]["enabled"]),
        transfer_function=str(optimizer["binary_transfer"]["function"]),
        optimizer_name=str(optimizer["name"]),
    )
    if config.maximum_candidate_requests != 240:
        raise V09BProtocolError("V0.9-A per-run budget must be exactly 240 requests.")

    return V09BProtocol(
        config=config,
        optimizer_seeds=APPROVED_OPTIMIZER_SEEDS,
        split_seed=42,
        model_attack_seeds=APPROVED_MODEL_ATTACK_SEEDS,
        config_path=config_target.resolve(),
        protocol_doc_path=doc_target.resolve(),
        config_sha256=hashlib.sha256(raw_config).hexdigest(),
        raw_snapshot=bgwo.to_json_compatible(payload),
    )


def run_v09b_synthetic_validation(
    *,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    protocol_doc_path: Path | str = DEFAULT_PROTOCOL_DOC_PATH,
    output_path: Path | str | None = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Execute deterministic synthetic checks for BGWO mechanics only."""
    protocol = load_v09b_protocol(config_path, protocol_doc_path)
    frozen = verify_v08_frozen_integrity()
    config = protocol.config

    synthetic_seed_primary = 9901
    synthetic_seed_secondary = 9902
    synthetic_seed_tertiary = 9903

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

    target_optimizer = bgwo.BinaryGreyWolfOptimizer(
        config,
        target_objective,
        score_evaluation_is_better,
    )
    first = target_optimizer.optimize(synthetic_seed_primary)
    replay = target_optimizer.optimize(synthetic_seed_primary)
    deterministic_replay_identical = first.to_json() == replay.to_json()
    other_population, other_latent = target_optimizer.initialize(synthetic_seed_secondary)
    different_seed_changes_state = not (
        np.array_equal(first.initial_population, other_population)
        and np.array_equal(first.initial_latent_positions, other_latent)
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

    constrained = bgwo.BinaryGreyWolfOptimizer(
        config,
        constrained_objective,
        constrained_evaluation_is_better,
    ).optimize(synthetic_seed_secondary)

    duplicate_population = np.zeros((config.wolf_count, config.dimensions), dtype=np.uint8)
    duplicate_population[:, 0] = 1
    duplicate_latent = duplicate_population.astype(float)
    invocation_count = 0

    def constant_objective(mask: np.ndarray) -> SyntheticEvaluation:
        nonlocal invocation_count
        invocation_count += 1
        return SyntheticEvaluation(True, 0.0, 1, 0.0, 0.0, "constant")

    cached = bgwo.BinaryGreyWolfOptimizer(
        config,
        constant_objective,
        constrained_evaluation_is_better,
    ).optimize(
        synthetic_seed_tertiary,
        initial_population=duplicate_population,
        initial_latent_positions=duplicate_latent,
    )

    uncached_calls = 0

    def uncached_objective(mask: np.ndarray) -> SyntheticEvaluation:
        nonlocal uncached_calls
        uncached_calls += 1
        return SyntheticEvaluation(True, 0.0, 1, 0.0, 0.0, "constant")

    uncached = bgwo.BinaryGreyWolfOptimizer(
        replace(config, cache_enabled=False),
        uncached_objective,
        constrained_evaluation_is_better,
    ).optimize(
        synthetic_seed_tertiary,
        initial_population=duplicate_population,
        initial_latent_positions=duplicate_latent,
    )
    cache_decision_invariant = (
        np.array_equal(cached.best_mask, uncached.best_mask)
        and cached.best_evaluation == uncached.best_evaluation
        and [row.best_mask.tolist() for row in cached.convergence_history]
        == [row.best_mask.tolist() for row in uncached.convergence_history]
    )

    def anchor_dominant_objective(mask: np.ndarray) -> SyntheticEvaluation:
        return SyntheticEvaluation(
            feasible=True,
            violation=0.0,
            cardinality=int(mask.sum()),
            primary_score=float(np.all(mask == 1)),
            secondary_score=0.0,
            tie_break_key=_mask_text(mask),
        )

    early = bgwo.BinaryGreyWolfOptimizer(
        config,
        anchor_dominant_objective,
        score_evaluation_is_better,
    ).optimize(synthetic_seed_primary)

    full_budget = bgwo.BinaryGreyWolfOptimizer(
        replace(config, early_stopping_min_iteration=config.evaluated_iterations),
        constrained_objective,
        constrained_evaluation_is_better,
    ).optimize(synthetic_seed_secondary)

    checks = {
        "protocol_loaded_and_locked": True,
        "v08_frozen_integrity_pass": frozen["status"] == "PASS",
        "dimensions_43": config.dimensions == 43,
        "wolves_12": config.wolf_count == 12,
        "evaluated_iterations_20": config.evaluated_iterations == 20,
        "maximum_candidate_requests_240": config.maximum_candidate_requests == 240,
        "five_run_maximum_requests_1200": config.maximum_candidate_requests * 5 == 1200,
        "deterministic_replay_identical": deterministic_replay_identical,
        "different_seed_changes_state": different_seed_changes_state,
        "initial_cardinalities_match_lock": (
            first.initial_population.sum(axis=1).astype(int).tolist()
            == [43, 42, *APPROVED_CARDINALITIES]
        ),
        "constrained_comparator_returns_feasible": constrained.best_evaluation.feasible,
        "cache_hit_recorded": cached.cache_hits > 0,
        "cache_invocations_equal_unique_evaluations": invocation_count == cached.unique_evaluations,
        "cache_accounting_valid": (
            cached.total_candidate_requests == cached.unique_evaluations + cached.cache_hits
        ),
        "cache_does_not_change_decisions": cache_decision_invariant,
        "uncached_invocations_equal_requests": (
            uncached_calls == uncached.total_candidate_requests
        ),
        "early_stop_reason_valid": early.stop_reason == bgwo.STOP_EARLY_NO_IMPROVEMENT,
        "early_stop_boundary_valid": (
            early.convergence_history[-1].iteration_index == 10
            and early.evaluated_iteration_count == 11
        ),
        "full_budget_reaches_240_requests": (
            full_budget.total_candidate_requests == 240
            and full_budget.evaluated_iteration_count == 20
        ),
        "synthetic_only_validation": True,
        "production_optimizer_runs_not_executed": True,
        "dataco_not_accessed": True,
        "validation_not_accessed": True,
        "final_test_not_accessed": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "schema_version": V09B_SCHEMA_VERSION,
        "stage": V09B_STAGE,
        "status": status,
        "validation_kind": V09B_VALIDATION_KIND,
        "scientific_experiment": False,
        "scientific_claims_supported": False,
        "starting_head": current_head_short(PROJECT_ROOT),
        "protocol_config_sha256": protocol.config_sha256,
        "protocol_document": str(protocol.protocol_doc_path.relative_to(PROJECT_ROOT)),
        "config_path": str(protocol.config_path.relative_to(PROJECT_ROOT)),
        "governed_model_attack_seeds": list(protocol.model_attack_seeds),
        "governed_optimizer_seeds_for_later_stages": list(protocol.optimizer_seeds),
        "synthetic_optimizer_seeds_used": [
            synthetic_seed_primary,
            synthetic_seed_secondary,
            synthetic_seed_tertiary,
        ],
        "configuration": config.to_dict(),
        "frozen_v08_integrity": frozen,
        "synthetic_cases": {
            "target_mask": {
                "result_sha256": hashlib.sha256(first.to_json().encode("utf-8")).hexdigest(),
                "best_primary_score": first.best_evaluation.primary_score,
                "stop_reason": first.stop_reason,
            },
            "constrained_minimum_cardinality": {
                "best_evaluation": constrained.best_evaluation.to_dict(),
                "stop_reason": constrained.stop_reason,
            },
            "cache": {
                "requests": cached.total_candidate_requests,
                "unique_evaluations": cached.unique_evaluations,
                "cache_hits": cached.cache_hits,
            },
            "early_stopping": {
                "stop_reason": early.stop_reason,
                "evaluated_iteration_count": early.evaluated_iteration_count,
                "last_iteration_index": early.convergence_history[-1].iteration_index,
            },
            "full_budget": {
                "stop_reason": full_budget.stop_reason,
                "requests": full_budget.total_candidate_requests,
                "evaluated_iteration_count": full_budget.evaluated_iteration_count,
            },
        },
        "checks": checks,
        "governance": {
            "dataco_accessed": False,
            "governed_dt_trained": False,
            "final_test_accessed": False,
            "bpso_rerun": False,
            "production_bgwo_search_executed": False,
            "next_stage_started": False,
        },
    }
    summary = bgwo.to_json_compatible(summary)
    if output_path is not None:
        _atomic_write_json(Path(output_path), summary)
    if status != "PASS":
        raise RuntimeError("V0.9-B synthetic BGWO validation failed.")
    return summary


def _mask_text(mask: np.ndarray) -> str:
    return "".join(str(int(value)) for value in mask)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    summary = run_v09b_synthetic_validation()
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
