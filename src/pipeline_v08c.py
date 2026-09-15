"""Five-run V0.8-C BPSO development-validation optimization campaign."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.optimization.bpso import (
    BPSOConfig,
    BinaryParticleSwarmOptimizer,
    STOP_EARLY_NO_IMPROVEMENT,
    STOP_MAX_GENERATIONS,
)
from src.optimization.feature_fitness import (
    EXPECTED_MODEL_PARAMETERS,
    EXPECTED_SEEDS,
    EXPECTED_THRESHOLD,
    FeatureFitnessEvaluation,
    FeatureFitnessEvaluator,
    FitnessContext,
    MARGINS,
    SeedFitnessRecord,
    audit_fitness_context,
    constraint_outcome,
    feature_fitness_is_better,
)
from src.pipeline_v08a import APPROVED_CARDINALITIES, APPROVED_OPTIMIZER_SEEDS, load_v08a_protocol
from src.pipeline_v08b import (
    DEFAULT_OUTPUT_DIR,
    IMMUTABLE_PATHS as V08B_INHERITED_IMMUTABLE_PATHS,
    FrozenBasis,
    build_fitness_context,
    load_frozen_basis,
    load_frozen_development_workloads,
    sha256_file,
)
from src.security.experiment_data import DevelopmentExperimentData, fingerprint_feature_names


V08C_STAGE = "V0.8-C"
V08C_SCHEMA_VERSION = "v0.8-c-five-run-validation-1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
OPTIMIZER_SEEDS = APPROVED_OPTIMIZER_SEEDS
MODEL_ATTACK_SEEDS = EXPECTED_SEEDS
RUN_FILE_TEMPLATE = "v08c_run_{seed}.json"
RUN_ARTIFACT_KIND = "OPTIMIZER_RUN_EVIDENCE"
VALIDATION_SCOPE = "TRAIN_AND_DEVELOPMENT_VALIDATION_ONLY"
V08B_TRACKED_PATHS = (
    "src/optimization/feature_fitness.py",
    "src/pipeline_v08b.py",
    "tests/test_bpso_fitness.py",
    "tests/test_pipeline_v08b.py",
)
V08B_EVIDENCE_PATHS = (
    OUTPUT_DIR / "v08b_preflight.json",
    OUTPUT_DIR / "v08b_leakage_audit.json",
    OUTPUT_DIR / "v08b_pilot.json",
    OUTPUT_DIR / "v08b_budget_projection.json",
)
IMMUTABLE_PATHS = tuple(
    dict.fromkeys(
        (
            *V08B_INHERITED_IMMUTABLE_PATHS,
            *(PROJECT_ROOT / path for path in V08B_TRACKED_PATHS),
            *V08B_EVIDENCE_PATHS,
        )
    )
)


class V08CError(RuntimeError):
    """Raised when V0.8-C cannot preserve its frozen scientific protocol."""


class V08CCheckpointError(V08CError):
    """Raised when a completed-run checkpoint is incompatible or corrupted."""


@dataclass(frozen=True)
class ProtocolIdentity:
    optimizer_config_sha256: str
    fitness_protocol_sha256: str
    comparator_sha256: str
    production_protocol_sha256: str
    feature_manifest_sha256: str
    real_k42_mask_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "optimizer_config_sha256": self.optimizer_config_sha256,
            "fitness_protocol_sha256": self.fitness_protocol_sha256,
            "comparator_sha256": self.comparator_sha256,
            "production_protocol_sha256": self.production_protocol_sha256,
            "feature_manifest_sha256": self.feature_manifest_sha256,
            "real_k42_mask_sha256": self.real_k42_mask_sha256,
        }


def snapshot_immutable_paths(paths: Sequence[Path] = IMMUTABLE_PATHS) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise V08CError(f"Missing frozen V0.6/V0.7/V0.8 prerequisite: {missing}")
    return {str(path.resolve()): sha256_file(path) for path in paths}


def verify_v08b_prerequisites() -> dict[str, Any]:
    """Require committed V0.8-B code and all passing machine-readable gates."""
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *V08B_TRACKED_PATHS],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *V08B_TRACKED_PATHS],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if tracked.returncode != 0 or clean.returncode != 0:
        raise V08CError("V0.8-B source/tests must be committed and unmodified.")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    source_commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *V08B_TRACKED_PATHS],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    preflight = _read_json(V08B_EVIDENCE_PATHS[0])
    leakage = _read_json(V08B_EVIDENCE_PATHS[1])
    pilot = _read_json(V08B_EVIDENCE_PATHS[2])
    budget = _read_json(V08B_EVIDENCE_PATHS[3])
    generated_hashes = preflight.get("generated_artifact_hashes", {})
    evidence_valid = (
        preflight.get("status") == "PASS"
        and preflight.get("checks", {}).get("aggregate_primary_metrics_match") is True
        and preflight.get("checks", {}).get("deterministic_repeat_identical") is True
        and preflight.get("checks", {}).get("test_not_accessed") is True
        and leakage.get("status") == "PASS"
        and leakage.get("test_accessed") is False
        and pilot.get("label") == "QUARANTINED_PILOT"
        and pilot.get("eligibility") == "NOT_ELIGIBLE_FOR_FINAL_SELECTION"
        and pilot.get("test_accessed") is False
        and budget.get("full_search_executed") is False
        and generated_hashes.get("v08b_leakage_audit.json")
        == sha256_file(V08B_EVIDENCE_PATHS[1])
        and generated_hashes.get("v08b_pilot.json") == sha256_file(V08B_EVIDENCE_PATHS[2])
        and generated_hashes.get("v08b_budget_projection.json")
        == sha256_file(V08B_EVIDENCE_PATHS[3])
    )
    if not head or not source_commit or not evidence_valid:
        raise V08CError("V0.8-B prerequisite evidence did not pass.")
    return {
        "status": "PASS",
        "starting_head": head,
        "v08b_source_commit": source_commit,
        "tracked_and_unmodified": True,
        "integrity_pass": True,
        "leakage_pass": True,
        "k43_reproduction_pass": True,
        "deterministic_repeat_pass": True,
        "final_test_prohibition_pass": True,
        "pilot_label": "QUARANTINED_PILOT",
        "evidence_hashes": {
            path.name: sha256_file(path) for path in V08B_EVIDENCE_PATHS
        },
    }


def production_protocol_identity(
    config: BPSOConfig, basis: FrozenBasis
) -> ProtocolIdentity:
    optimizer_hash = sha256_file(PROJECT_ROOT / "config" / "bpso.yaml")
    comparator_hash = hashlib.sha256(
        inspect.getsource(feature_fitness_is_better).encode("utf-8")
    ).hexdigest()
    fitness_payload = {
        "feature_fitness_source_sha256": sha256_file(
            PROJECT_ROOT / "src" / "optimization" / "feature_fitness.py"
        ),
        "baseline": basis.baseline.to_dict(),
        "margins": dict(MARGINS),
        "model": "decision_tree",
        "model_parameters": dict(EXPECTED_MODEL_PARAMETERS),
        "prediction_threshold": EXPECTED_THRESHOLD,
        "model_attack_seeds": list(MODEL_ATTACK_SEEDS),
        "attack": {"type": "mixed", "rate": 0.05, "severity": "MEDIUM"},
        "selection_scope": VALIDATION_SCOPE,
    }
    fitness_hash = _json_sha256(fitness_payload)
    production_payload = {
        "optimizer": config.to_dict(),
        "optimizer_seeds": list(OPTIMIZER_SEEDS),
        "optimizer_config_sha256": optimizer_hash,
        "fitness_protocol_sha256": fitness_hash,
        "comparator_sha256": comparator_hash,
        "feature_manifest_sha256": basis.candidate_manifest_sha256,
        "real_k42_mask_sha256": basis.k42_mask_sha256,
    }
    return ProtocolIdentity(
        optimizer_config_sha256=optimizer_hash,
        fitness_protocol_sha256=fitness_hash,
        comparator_sha256=comparator_hash,
        production_protocol_sha256=_json_sha256(production_payload),
        feature_manifest_sha256=basis.candidate_manifest_sha256,
        real_k42_mask_sha256=basis.k42_mask_sha256,
    )


def verify_production_config(config: BPSOConfig) -> None:
    expected = {
        "dimensions": 43,
        "particle_count": 12,
        "evaluated_generations": 20,
        "random_cardinalities": tuple(APPROVED_CARDINALITIES),
        "velocity_initial_min": -1.0,
        "velocity_initial_max": 1.0,
        "velocity_clamp_min": -6.0,
        "velocity_clamp_max": 6.0,
        "inertia_start": 0.9,
        "inertia_end": 0.4,
        "cognitive_coefficient": 2.0,
        "social_coefficient": 2.0,
        "minimum_selected_features": 1,
        "early_stopping_patience": 7,
        "early_stopping_min_generation": 10,
        "cache_enabled": True,
        "transfer_function": "standard_logistic_sigmoid",
    }
    observed = {name: getattr(config, name) for name in expected}
    if observed != expected or config.maximum_fitness_requests != 240:
        raise V08CError(f"Production BPSO configuration drift: {observed}")


def production_dry_validation(
    config: BPSOConfig, basis: FrozenBasis
) -> dict[str, Any]:
    """Validate all five initial populations without evaluating fitness."""
    verify_production_config(config)
    expected_cardinalities = [43, 42, *APPROVED_CARDINALITIES]
    rows = []
    dummy = lambda mask: None
    comparator = lambda left, right: False
    optimizer = BinaryParticleSwarmOptimizer(config, dummy, comparator)
    for seed in OPTIMIZER_SEEDS:
        population, velocities = optimizer.initialize(seed, k42_mask=basis.k42_mask)
        cardinalities = population.sum(axis=1).astype(int).tolist()
        if (
            cardinalities != expected_cardinalities
            or not np.array_equal(population[0], np.ones(43, dtype=np.uint8))
            or not np.array_equal(population[1], basis.k42_mask)
            or not np.isin(population, (0, 1)).all()
            or np.any(population.sum(axis=1) < 1)
            or np.any(velocities < -1.0)
            or np.any(velocities > 1.0)
        ):
            raise V08CError(f"Production initialization failed for optimizer seed {seed}.")
        rows.append(
            {
                "optimizer_seed": seed,
                "cardinalities": cardinalities,
                "initial_population_sha256": _json_sha256(population.tolist()),
                "initial_velocities_sha256": _json_sha256(velocities.tolist()),
            }
        )
    return {
        "status": "PASS",
        "fitness_evaluations": 0,
        "final_test_accessed": False,
        "initializations": rows,
    }


def execute_production_run(
    seed: int,
    config: BPSOConfig,
    context: FitnessContext,
    basis: FrozenBasis,
    identity: ProtocolIdentity,
) -> dict[str, Any]:
    """Execute one fresh optimizer run and return an uncheckpointed record."""
    if seed not in OPTIMIZER_SEEDS:
        raise V08CError(f"Unapproved optimizer seed: {seed}")
    evaluator = FeatureFitnessEvaluator(context)
    optimizer = BinaryParticleSwarmOptimizer(config, evaluator, feature_fitness_is_better)
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = optimizer.optimize(seed, k42_mask=basis.k42_mask)
    cpu_time = time.process_time() - cpu_start
    wall_time = time.perf_counter() - wall_start
    best = result.best_evaluation
    history = [record.to_dict() for record in result.convergence_history]
    return {
        "schema_version": V08C_SCHEMA_VERSION,
        "stage": V08C_STAGE,
        "artifact_kind": RUN_ARTIFACT_KIND,
        "status": "COMPLETED_AND_VALIDATED_PENDING_CHECKPOINT",
        "execution_mode": "FRESH",
        "optimizer_seed": seed,
        "model_attack_seeds": list(MODEL_ATTACK_SEEDS),
        "selection_scope": VALIDATION_SCOPE,
        "test_accessed": False,
        "logistic_regression_evaluated": False,
        "resource_benchmark_executed": False,
        "direct_energy_measured": False,
        "protocol_identity": identity.to_dict(),
        "configuration": config.to_dict(),
        "feature_manifest_sha256": basis.candidate_manifest_sha256,
        "real_k42_mask_sha256": basis.k42_mask_sha256,
        "initial_population": result.initial_population.tolist(),
        "initial_population_sha256": _json_sha256(result.initial_population.tolist()),
        "initial_cardinalities": result.initial_population.sum(axis=1).astype(int).tolist(),
        "initial_velocities_sha256": _json_sha256(result.initial_velocities.tolist()),
        "evaluated_generation_count": result.evaluated_generation_count,
        "stop_reason": result.stop_reason,
        "best_generation": result.best_generation,
        "best_mask": result.best_mask.tolist(),
        "best_mask_sha256": best.mask_sha256,
        "selected_features": list(best.selected_features),
        "selected_features_sha256": best.selected_features_sha256,
        "selected_feature_count": best.selected_feature_count,
        "best_evaluation": best.to_dict(),
        "fitness_requests": result.total_fitness_requests,
        "unique_evaluations": result.unique_evaluations,
        "cache_hits": result.cache_hits,
        "actual_decision_tree_fits": evaluator.decision_tree_fit_count,
        "repairs": result.repair_count,
        "convergence_history": history,
        "wall_time_sec": wall_time,
        "process_cpu_time_sec": cpu_time,
        "evaluation_timing": evaluator.instrumentation(),
    }


def validate_run_artifact(
    record: Mapping[str, Any],
    seed: int,
    config: BPSOConfig,
    context: FitnessContext,
    basis: FrozenBasis,
    identity: ProtocolIdentity,
) -> dict[str, bool]:
    """Recompute all run-level invariants without fitting another model."""
    try:
        mask = np.asarray(record["best_mask"], dtype=np.uint8)
        evaluation = evaluation_from_dict(record["best_evaluation"])
        selected = tuple(
            feature
            for feature, active in zip(basis.candidate_features, mask)
            if active
        )
        losses, feasible, violation = constraint_outcome(
            context.baseline, evaluation.mean_metrics
        )
        history = record["convergence_history"]
        initial = np.asarray(record["initial_population"], dtype=np.uint8)
        history_evaluations = [evaluation_from_dict(row["best_evaluation"]) for row in history]
        checks = {
            "stage_and_status": record.get("stage") == V08C_STAGE
            and record.get("artifact_kind") == RUN_ARTIFACT_KIND,
            "approved_optimizer_seed": record.get("optimizer_seed") == seed
            and seed in OPTIMIZER_SEEDS,
            "protocol_identity": record.get("protocol_identity") == identity.to_dict(),
            "production_configuration": record.get("configuration") == config.to_dict(),
            "mask_length_and_nonempty": mask.shape == (43,) and int(mask.sum()) >= 1,
            "mask_binary": bool(np.isin(mask, (0, 1)).all()),
            "mask_hash": record.get("best_mask_sha256")
            == hashlib.sha256(mask.tobytes()).hexdigest()
            == evaluation.mask_sha256,
            "selected_feature_count": record.get("selected_feature_count")
            == int(mask.sum())
            == evaluation.selected_feature_count,
            "selected_feature_order": tuple(record.get("selected_features", ()))
            == selected
            == evaluation.selected_features,
            "selected_feature_hash": record.get("selected_features_sha256")
            == fingerprint_feature_names(selected)
            == evaluation.selected_features_sha256,
            "feature_manifest": record.get("feature_manifest_sha256")
            == basis.candidate_manifest_sha256,
            "real_k42_binding": record.get("real_k42_mask_sha256")
            == basis.k42_mask_sha256
            and np.array_equal(initial[1], basis.k42_mask),
            "initial_population": initial.shape == (12, 43)
            and record.get("initial_cardinalities")
            == [43, 42, *APPROVED_CARDINALITIES]
            and record.get("initial_population_sha256")
            == _json_sha256(initial.tolist()),
            "development_only": record.get("selection_scope") == VALIDATION_SCOPE
            and record.get("test_accessed") is False,
            "decision_tree_only": record.get("logistic_regression_evaluated") is False,
            "five_model_attack_seeds": tuple(record.get("model_attack_seeds", ()))
            == MODEL_ATTACK_SEEDS
            and tuple(seed_row.seed for seed_row in evaluation.per_seed)
            == MODEL_ATTACK_SEEDS,
            "feasibility_recomputed": evaluation.feasible is feasible,
            "losses_recomputed": all(
                math.isclose(
                    evaluation.relative_losses[name], value, rel_tol=0.0, abs_tol=1e-12
                )
                for name, value in losses.items()
            ),
            "violation_recomputed": math.isclose(
                evaluation.normalized_violation,
                violation,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "request_cache_accounting": record.get("fitness_requests")
            == record.get("unique_evaluations") + record.get("cache_hits"),
            "request_generation_accounting": record.get("fitness_requests")
            == record.get("evaluated_generation_count") * 12
            and all(row.get("request_count") == 12 for row in history)
            and history[-1].get("cumulative_unique_evaluations")
            == record.get("unique_evaluations")
            and history[-1].get("cumulative_cache_hits") == record.get("cache_hits"),
            "dt_fit_accounting": record.get("actual_decision_tree_fits")
            == record.get("unique_evaluations") * 5,
            "generation_budget": 1 <= record.get("evaluated_generation_count", 0) <= 20
            and record.get("fitness_requests", 0) <= 240,
            "stop_reason": record.get("stop_reason")
            in {STOP_MAX_GENERATIONS, STOP_EARLY_NO_IMPROVEMENT},
            "history_coverage": len(history) == record.get("evaluated_generation_count")
            and [row["generation_index"] for row in history] == list(range(len(history))),
            "history_final_best": history[-1]["best_mask"] == record.get("best_mask"),
            "comparator_consistency": not any(
                feature_fitness_is_better(item, evaluation)
                for item in history_evaluations
            ),
            "metrics_finite": _all_finite(evaluation.mean_metrics.values())
            and math.isfinite(evaluation.normalized_violation),
            "repair_accounting": record.get("repairs")
            == sum(int(row["repair_count"]) for row in history),
            "no_resource_or_energy_stage": record.get("resource_benchmark_executed")
            is False
            and record.get("direct_energy_measured") is False,
        }
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise V08CCheckpointError(f"Malformed run artifact for seed {seed}: {exc}") from exc
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise V08CCheckpointError(f"Run artifact validation failed for seed {seed}: {failed}")
    return checks


def execute_or_resume_run(
    seed: int,
    output_dir: Path,
    config: BPSOConfig,
    context: FitnessContext,
    basis: FrozenBasis,
    identity: ProtocolIdentity,
    *,
    executor: Callable[
        [int, BPSOConfig, FitnessContext, FrozenBasis, ProtocolIdentity], dict[str, Any]
    ] = execute_production_run,
) -> tuple[dict[str, Any], str, str]:
    """Reuse one accepted complete run or atomically checkpoint one fresh run."""
    path = output_dir / RUN_FILE_TEMPLATE.format(seed=seed)
    if path.exists():
        record = _read_json(path)
        validate_run_artifact(record, seed, config, context, basis, identity)
        return record, "RESUMED", sha256_file(path)
    record = executor(seed, config, context, basis, identity)
    validate_run_artifact(record, seed, config, context, basis, identity)
    record["status"] = "COMPLETED_AND_VALIDATED"
    _atomic_write_json(path, record)
    stored = _read_json(path)
    validate_run_artifact(stored, seed, config, context, basis, identity)
    return stored, "FRESH", sha256_file(path)


def evaluation_from_dict(payload: Mapping[str, Any]) -> FeatureFitnessEvaluation:
    per_seed = tuple(
        SeedFitnessRecord(
            seed=int(row["seed"]),
            selected_feature_count=int(row["selected_feature_count"]),
            metrics=row["metrics"],
            training_rows_sha256=str(row["training_rows_sha256"]),
            validation_rows_sha256=str(row["validation_rows_sha256"]),
            training_labels_sha256=str(row["training_labels_sha256"]),
            validation_labels_sha256=str(row["validation_labels_sha256"]),
            validation_features_sha256=str(row["validation_features_sha256"]),
            visible_attack_count=int(row["visible_attack_count"]),
            invisible_attack_count=int(row["invisible_attack_count"]),
        )
        for row in payload.get("per_seed", ())
    )
    return FeatureFitnessEvaluation(
        mask=tuple(int(value) for value in payload["mask"]),
        mask_sha256=str(payload["mask_sha256"]),
        selected_features=tuple(payload["selected_features"]),
        selected_features_sha256=str(payload["selected_features_sha256"]),
        selected_feature_count=int(payload["selected_feature_count"]),
        feasible=bool(payload["feasible"]),
        normalized_violation=float(payload["normalized_violation"]),
        relative_losses=payload["relative_losses"],
        mean_metrics=payload["mean_metrics"],
        confusion_totals=payload["confusion_totals"],
        per_seed=per_seed,
        decision_tree_fit_count=int(payload["decision_tree_fit_count"]),
    )


def build_stability_analysis(
    records: Sequence[Mapping[str, Any]], candidate_features: Sequence[str]
) -> dict[str, Any]:
    masks = [np.asarray(record["best_mask"], dtype=np.uint8) for record in records]
    seeds = [int(record["optimizer_seed"]) for record in records]
    cardinalities = [int(mask.sum()) for mask in masks]
    pairs = []
    for left in range(len(masks)):
        for right in range(left + 1, len(masks)):
            intersection = int(np.logical_and(masks[left], masks[right]).sum())
            union = int(np.logical_or(masks[left], masks[right]).sum())
            jaccard = float(intersection / union) if union else 1.0
            hamming = float(np.count_nonzero(masks[left] != masks[right]) / len(candidate_features))
            pairs.append(
                {
                    "seed_left": seeds[left],
                    "seed_right": seeds[right],
                    "intersection": intersection,
                    "union": union,
                    "jaccard": jaccard,
                    "normalized_hamming": hamming,
                }
            )
    frequencies = []
    counts = np.sum(np.stack(masks), axis=0).astype(int)
    for index, (feature, count) in enumerate(zip(candidate_features, counts)):
        frequencies.append(
            {
                "feature_index": index,
                "feature": feature,
                "selection_count": int(count),
                "selection_frequency": float(count / len(masks)),
            }
        )
    evaluations = [evaluation_from_dict(record["best_evaluation"]) for record in records]
    variation_values = {
        "average_precision": [item.average_precision for item in evaluations],
        "f1": [item.f1 for item in evaluations],
        "recall": [item.recall for item in evaluations],
        "normalized_violation": [item.normalized_violation for item in evaluations],
        "cardinality": cardinalities,
    }
    return {
        "schema_version": V08C_SCHEMA_VERSION,
        "stage": V08C_STAGE,
        "artifact_kind": "DERIVED_OPTIMIZER_STABILITY",
        "optimizer_run_count": len(records),
        "cardinality": _summary(cardinalities),
        "pairwise": pairs,
        "jaccard": _summary([row["jaccard"] for row in pairs]),
        "normalized_hamming": _summary(
            [row["normalized_hamming"] for row in pairs]
        ),
        "feature_frequency": frequencies,
        "strict_consensus_features": [
            row["feature"] for row in frequencies if row["selection_count"] == 5
        ],
        "majority_features": [
            row["feature"] for row in frequencies if row["selection_count"] >= 4
        ],
        "consensus_used_for_selection": False,
        "winner_variation": {
            name: _summary(values) for name, values in variation_values.items()
        },
        "statistical_significance_claimed": False,
        "test_accessed": False,
    }


def build_convergence_analysis(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    run_summaries = []
    plot_rows = []
    for record in records:
        history = record["convergence_history"]
        for row in history:
            evaluation = row["best_evaluation"]
            plot_rows.append(
                {
                    "optimizer_seed": record["optimizer_seed"],
                    "generation_index": row["generation_index"],
                    "best_selected_feature_count": row["best_selected_feature_count"],
                    "average_precision": evaluation["mean_metrics"]["average_precision"],
                    "f1": evaluation["mean_metrics"]["f1"],
                    "recall": evaluation["mean_metrics"]["recall"],
                    "feasible": evaluation["feasible"],
                    "normalized_violation": evaluation["normalized_violation"],
                    "population_diversity": row["population_diversity"],
                    "cumulative_unique_evaluations": row[
                        "cumulative_unique_evaluations"
                    ],
                    "cumulative_cache_hits": row["cumulative_cache_hits"],
                }
            )
        diversities = [float(row["population_diversity"]) for row in history]
        run_summaries.append(
            {
                "optimizer_seed": record["optimizer_seed"],
                "initial_best": history[0]["best_evaluation"],
                "final_best": record["best_evaluation"],
                "best_generation": record["best_generation"],
                "evaluated_generations": record["evaluated_generation_count"],
                "early_stopped": record["stop_reason"]
                == STOP_EARLY_NO_IMPROVEMENT,
                "stop_reason": record["stop_reason"],
                "comparator_improvements_after_initialization": max(
                    0,
                    sum(bool(row["global_best_improved"]) for row in history) - 1,
                ),
                "unique_evaluations": record["unique_evaluations"],
                "cache_hit_rate": float(
                    record["cache_hits"] / record["fitness_requests"]
                ),
                "repairs": record["repairs"],
                "initial_diversity": diversities[0],
                "final_diversity": diversities[-1],
                "minimum_diversity": min(diversities),
                "maximum_diversity": max(diversities),
            }
        )
    aggregate = []
    for generation in range(20):
        rows = [row for row in plot_rows if row["generation_index"] == generation]
        if not rows:
            continue
        aggregate.append(
            {
                "generation_index": generation,
                "run_count": len(rows),
                "mean_best_selected_feature_count": statistics.fmean(
                    row["best_selected_feature_count"] for row in rows
                ),
                "median_best_selected_feature_count": statistics.median(
                    row["best_selected_feature_count"] for row in rows
                ),
                "mean_average_precision": statistics.fmean(
                    row["average_precision"] for row in rows
                ),
                "mean_f1": statistics.fmean(row["f1"] for row in rows),
                "mean_recall": statistics.fmean(row["recall"] for row in rows),
                "mean_population_diversity": statistics.fmean(
                    row["population_diversity"] for row in rows
                ),
                "feasible_run_count": sum(bool(row["feasible"]) for row in rows),
            }
        )
    return {
        "schema_version": V08C_SCHEMA_VERSION,
        "stage": V08C_STAGE,
        "artifact_kind": "DERIVED_CONVERGENCE_ANALYSIS",
        "run_summaries": run_summaries,
        "plot_data": plot_rows,
        "aggregate_by_generation": aggregate,
        "test_accessed": False,
    }


def build_validation_comparison(
    records: Sequence[Mapping[str, Any]], basis: FrozenBasis
) -> dict[str, Any]:
    roles = basis.lock["roles"]
    k43 = roles["full_baseline"]["metrics"]
    k42 = roles["best_unsupervised"]["metrics"]
    k11 = roles["best_supervised"]["metrics"]
    rows = []
    for record in records:
        evaluation = evaluation_from_dict(record["best_evaluation"])
        rows.append(
            {
                "optimizer_seed": record["optimizer_seed"],
                "selected_feature_count": evaluation.selected_feature_count,
                "feature_reduction_vs_k43_percent": float(
                    100.0 * (43 - evaluation.selected_feature_count) / 43
                ),
                "feature_count_difference_vs_k11": evaluation.selected_feature_count
                - 11,
                "average_precision": evaluation.average_precision,
                "average_precision_relative_change_vs_k43_percent": _relative_change(
                    evaluation.average_precision, k43["average_precision"]
                ),
                "average_precision_relative_loss_vs_k43": evaluation.relative_losses[
                    "average_precision"
                ],
                "f1": evaluation.f1,
                "f1_relative_change_vs_k43_percent": _relative_change(
                    evaluation.f1, k43["f1"]
                ),
                "f1_relative_loss_vs_k43": evaluation.relative_losses["f1"],
                "recall": evaluation.recall,
                "recall_relative_change_vs_k43_percent": _relative_change(
                    evaluation.recall, k43["recall"]
                ),
                "recall_relative_loss_vs_k43": evaluation.relative_losses["recall"],
                "feasible": evaluation.feasible,
                "normalized_violation": evaluation.normalized_violation,
                "interpretation": {
                    "substantial_reduction_k_le_21": evaluation.selected_feature_count
                    <= 21,
                    "competitive_with_k11_cardinality": evaluation.selected_feature_count
                    <= 11,
                    "compactness_advantage_k_le_10": evaluation.selected_feature_count
                    <= 10,
                },
            }
        )
    return {
        "schema_version": V08C_SCHEMA_VERSION,
        "stage": V08C_STAGE,
        "artifact_kind": "VALIDATION_COMPARISON",
        "bpso_run_winners": rows,
        "historical_validation_anchors": {
            "K43": {"feature_count": 43, "metrics": k43},
            "K42": {"feature_count": 42, "metrics": k42},
            "K11": {
                "feature_count": 11,
                "metrics": k11,
                "selection_semantics": "seed-specific historical MI rule",
            },
        },
        "bpso_selection_semantics": "one universal subset candidate per optimizer run",
        "final_test_comparison_performed": False,
        "statistical_significance_claimed": False,
    }


def select_validation_winner(
    records: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], FeatureFitnessEvaluation]:
    if len(records) != 5:
        raise V08CError("Winner selection requires exactly five run records.")
    best_record = records[0]
    best_evaluation = evaluation_from_dict(best_record["best_evaluation"])
    for record in records[1:]:
        candidate = evaluation_from_dict(record["best_evaluation"])
        if feature_fitness_is_better(candidate, best_evaluation):
            best_record = record
            best_evaluation = candidate
    return best_record, best_evaluation


def build_winner_lock(
    winner_record: Mapping[str, Any],
    evaluation: FeatureFitnessEvaluation,
    identity: ProtocolIdentity,
    run_hashes: Mapping[int, str],
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    rationale = (
        "feasible candidates ranked by fewer features, higher AP, higher F1, "
        "higher recall, then canonical mask"
        if evaluation.feasible
        else "infeasible candidates ranked by lower normalized violation, higher AP, "
        "higher F1, higher recall, fewer features, then canonical mask"
    )
    lock = {
        "schema_version": V08C_SCHEMA_VERSION,
        "stage": V08C_STAGE,
        "status": "VALIDATION_LOCKED",
        "eligible_for_v08d": True,
        "selection_scope": VALIDATION_SCOPE,
        "source_optimizer_seed": winner_record["optimizer_seed"],
        "mask": list(evaluation.mask),
        "ordered_selected_features": list(evaluation.selected_features),
        "selected_feature_count": evaluation.selected_feature_count,
        "feature_manifest_sha256": identity.feature_manifest_sha256,
        "mask_sha256": evaluation.mask_sha256,
        "selected_features_sha256": evaluation.selected_features_sha256,
        "fitness_protocol_sha256": identity.fitness_protocol_sha256,
        "optimizer_config_sha256": identity.optimizer_config_sha256,
        "production_protocol_sha256": identity.production_protocol_sha256,
        "validation_metrics": dict(evaluation.mean_metrics),
        "relative_losses": dict(evaluation.relative_losses),
        "feasible": evaluation.feasible,
        "normalized_violation": evaluation.normalized_violation,
        "comparator": {
            "definition": rationale,
            "sha256": identity.comparator_sha256,
            "resource_metrics_used": False,
            "test_metrics_used": False,
        },
        "selection_rationale": rationale,
        "five_run_evidence_sha256": {
            str(seed): run_hashes[seed] for seed in OPTIMIZER_SEEDS
        },
        "creation_metadata": {
            "created_at_utc": created_at or datetime.now(timezone.utc).isoformat(),
            "optimizer_run_count": 5,
            "optimizer_seeds": list(OPTIMIZER_SEEDS),
            "model_attack_seeds": list(MODEL_ATTACK_SEEDS),
        },
        "winner_statement": "Winner selected without V0.8 final-test access.",
        "historical_test_statement": (
            "Final test was historically accessed in V0.6; it remained untouched by "
            "V0.8 search and selection."
        ),
        "final_test_accessed": False,
        "logistic_regression_evaluated": False,
        "resource_benchmark_executed": False,
        "direct_energy_measured": False,
    }
    lock["semantic_lock_sha256"] = winner_lock_semantic_hash(lock)
    return lock


def winner_lock_semantic_hash(lock: Mapping[str, Any]) -> str:
    payload = dict(lock)
    payload.pop("semantic_lock_sha256", None)
    creation = dict(payload.get("creation_metadata", {}))
    creation.pop("created_at_utc", None)
    payload["creation_metadata"] = creation
    return _json_sha256(payload)


def write_or_verify_winner_lock(path: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    verify_winner_lock(lock)
    if path.exists():
        existing = _read_json(path)
        verify_winner_lock(existing)
        if existing["semantic_lock_sha256"] != lock["semantic_lock_sha256"]:
            raise V08CError("Existing winner lock differs from the recomputed winner.")
        return existing
    _atomic_write_json(path, lock)
    stored = _read_json(path)
    verify_winner_lock(stored)
    return stored


def verify_winner_lock(lock: Mapping[str, Any]) -> None:
    required = {
        "stage",
        "status",
        "eligible_for_v08d",
        "selection_scope",
        "source_optimizer_seed",
        "mask",
        "ordered_selected_features",
        "selected_feature_count",
        "feature_manifest_sha256",
        "mask_sha256",
        "selected_features_sha256",
        "fitness_protocol_sha256",
        "optimizer_config_sha256",
        "validation_metrics",
        "feasible",
        "normalized_violation",
        "comparator",
        "five_run_evidence_sha256",
        "semantic_lock_sha256",
    }
    mask = np.asarray(lock.get("mask", ()), dtype=np.uint8)
    selected = tuple(lock.get("ordered_selected_features", ()))
    valid = (
        required <= set(lock)
        and lock.get("stage") == V08C_STAGE
        and lock.get("status") == "VALIDATION_LOCKED"
        and lock.get("eligible_for_v08d") is True
        and lock.get("selection_scope") == VALIDATION_SCOPE
        and lock.get("source_optimizer_seed") in OPTIMIZER_SEEDS
        and mask.shape == (43,)
        and np.isin(mask, (0, 1)).all()
        and int(mask.sum()) == lock.get("selected_feature_count") == len(selected)
        and hashlib.sha256(mask.tobytes()).hexdigest() == lock.get("mask_sha256")
        and fingerprint_feature_names(selected) == lock.get("selected_features_sha256")
        and set(lock.get("five_run_evidence_sha256", {}))
        == {str(seed) for seed in OPTIMIZER_SEEDS}
        and lock.get("final_test_accessed") is False
        and lock.get("logistic_regression_evaluated") is False
        and lock.get("comparator", {}).get("resource_metrics_used") is False
        and lock.get("comparator", {}).get("test_metrics_used") is False
        and winner_lock_semantic_hash(lock) == lock.get("semantic_lock_sha256")
    )
    if not valid:
        raise V08CError("Winner validation lock verification failed.")


def run_v08c(
    *,
    output_dir: Path | str = OUTPUT_DIR,
    basis_loader: Callable[[], FrozenBasis] = load_frozen_basis,
    workload_loader: Callable[[FrozenBasis], tuple[DevelopmentExperimentData, ...]] = load_frozen_development_workloads,
    run_executor: Callable[
        [int, BPSOConfig, FitnessContext, FrozenBasis, ProtocolIdentity], dict[str, Any]
    ] = execute_production_run,
) -> dict[str, Any]:
    """Execute or safely resume the complete five-run V0.8-C campaign."""
    stage_start = time.perf_counter()
    prerequisites = verify_v08b_prerequisites()
    immutable_before = snapshot_immutable_paths()
    basis = basis_loader()
    workloads = workload_loader(basis)
    context = build_fitness_context(basis, workloads)
    audit = audit_fitness_context(context)
    if audit.status != "PASS" or context.test_accessed:
        raise V08CError("V0.8-B governed fitness context failed before optimization.")
    config = load_v08a_protocol().config
    verify_production_config(config)
    identity = production_protocol_identity(config, basis)
    dry_validation = production_dry_validation(config, basis)
    output = Path(output_dir)
    records = []
    modes: dict[int, str] = {}
    run_hashes: dict[int, str] = {}
    for seed in OPTIMIZER_SEEDS:
        record, mode, run_hash = execute_or_resume_run(
            seed,
            output,
            config,
            context,
            basis,
            identity,
            executor=run_executor,
        )
        validate_run_artifact(record, seed, config, context, basis, identity)
        if snapshot_immutable_paths() != immutable_before:
            raise V08CError(f"Frozen artifacts changed during optimizer run {seed}.")
        records.append(record)
        modes[seed] = mode
        run_hashes[seed] = run_hash
    if tuple(record["optimizer_seed"] for record in records) != OPTIMIZER_SEEDS:
        raise V08CError("Five-run optimizer coverage is incomplete.")

    stability = build_stability_analysis(records, basis.candidate_features)
    convergence = build_convergence_analysis(records)
    comparison = build_validation_comparison(records, basis)
    winner_record, winner_evaluation = select_validation_winner(records)
    proposed_lock = build_winner_lock(
        winner_record, winner_evaluation, identity, run_hashes
    )
    winner_lock = write_or_verify_winner_lock(
        output / "v08c_winner_lock.json", proposed_lock
    )
    verify_winner_lock(winner_lock)
    _atomic_write_json(output / "v08c_stability.json", stability)
    _atomic_write_json(output / "v08c_convergence.json", convergence)
    _atomic_write_json(output / "v08c_validation_comparison.json", comparison)
    immutable_after = snapshot_immutable_paths()
    if immutable_after != immutable_before:
        raise V08CError("Frozen artifacts changed during V0.8-C analysis or locking.")
    total_requests = sum(int(record["fitness_requests"]) for record in records)
    total_unique = sum(int(record["unique_evaluations"]) for record in records)
    total_hits = sum(int(record["cache_hits"]) for record in records)
    total_fits = sum(int(record["actual_decision_tree_fits"]) for record in records)
    summary = {
        "schema_version": V08C_SCHEMA_VERSION,
        "stage": V08C_STAGE,
        "status": "COMPLETED",
        "readiness_for_v08d": "GO",
        "starting_head": prerequisites["starting_head"],
        "prerequisites": prerequisites,
        "protocol_identity": identity.to_dict(),
        "dry_validation": dry_validation,
        "execution_mode": "FRESH" if set(modes.values()) == {"FRESH"} else "RESUMED",
        "run_execution_modes": {str(seed): modes[seed] for seed in OPTIMIZER_SEEDS},
        "run_artifact_sha256": {str(seed): run_hashes[seed] for seed in OPTIMIZER_SEEDS},
        "optimizer_seeds": list(OPTIMIZER_SEEDS),
        "model_attack_seeds": list(MODEL_ATTACK_SEEDS),
        "completed_run_count": len(records),
        "fitness_requests": total_requests,
        "unique_evaluations": total_unique,
        "cache_hits": total_hits,
        "actual_decision_tree_fits": total_fits,
        "optimization_wall_time_sec": sum(float(record["wall_time_sec"]) for record in records),
        "optimization_cpu_time_sec": sum(
            float(record["process_cpu_time_sec"]) for record in records
        ),
        "pipeline_wall_time_sec": time.perf_counter() - stage_start,
        "stop_reasons": {
            str(seed): record["stop_reason"]
            for seed, record in zip(OPTIMIZER_SEEDS, records)
        },
        "total_repairs": sum(int(record["repairs"]) for record in records),
        "winner": {
            "source_optimizer_seed": winner_lock["source_optimizer_seed"],
            "selected_feature_count": winner_lock["selected_feature_count"],
            "mask_sha256": winner_lock["mask_sha256"],
            "selected_features_sha256": winner_lock["selected_features_sha256"],
            "semantic_lock_sha256": winner_lock["semantic_lock_sha256"],
            "feasible": winner_lock["feasible"],
            "normalized_violation": winner_lock["normalized_violation"],
        },
        "artifact_hashes": {
            name: sha256_file(output / name)
            for name in (
                "v08c_stability.json",
                "v08c_convergence.json",
                "v08c_validation_comparison.json",
                "v08c_winner_lock.json",
            )
        },
        "immutable_hashes_before": immutable_before,
        "immutable_hashes_after": immutable_after,
        "selection_scope": VALIDATION_SCOPE,
        "final_test_accessed": False,
        "logistic_regression_evaluated": False,
        "resource_benchmark_executed": False,
        "direct_energy_measured": False,
        "final_scientific_claims_supported": False,
    }
    _atomic_write_json(output / "v08c_execution_summary.json", summary)
    _verify_v08c_artifacts(output, config, context, basis, identity)
    return summary


def _verify_v08c_artifacts(
    output: Path,
    config: BPSOConfig,
    context: FitnessContext,
    basis: FrozenBasis,
    identity: ProtocolIdentity,
) -> None:
    for seed in OPTIMIZER_SEEDS:
        validate_run_artifact(
            _read_json(output / RUN_FILE_TEMPLATE.format(seed=seed)),
            seed,
            config,
            context,
            basis,
            identity,
        )
    verify_winner_lock(_read_json(output / "v08c_winner_lock.json"))
    for name, kind in (
        ("v08c_stability.json", "DERIVED_OPTIMIZER_STABILITY"),
        ("v08c_convergence.json", "DERIVED_CONVERGENCE_ANALYSIS"),
        ("v08c_validation_comparison.json", "VALIDATION_COMPARISON"),
    ):
        artifact = _read_json(output / name)
        if artifact.get("stage") != V08C_STAGE or artifact.get("artifact_kind") != kind:
            raise V08CError(f"Invalid derived artifact: {name}")


def _summary(values: Sequence[float | int]) -> dict[str, float]:
    numeric = [float(value) for value in values]
    if not numeric:
        raise V08CError("Cannot summarize an empty value sequence.")
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
        "min": min(numeric),
        "max": max(numeric),
        "median": statistics.median(numeric),
    }


def _relative_change(candidate: float, baseline: float) -> float:
    if baseline == 0.0:
        raise V08CError("Relative validation comparison requires nonzero baseline.")
    return float(100.0 * (candidate - baseline) / baseline)


def _all_finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V08CCheckpointError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V08CCheckpointError(f"JSON artifact must be an object: {path}")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    summary = run_v08c()
    print(
        json.dumps(
            {
                "stage": V08C_STAGE,
                "status": summary["status"],
                "execution_mode": summary["execution_mode"],
                "completed_run_count": summary["completed_run_count"],
                "winner": summary["winner"],
                "final_test_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
