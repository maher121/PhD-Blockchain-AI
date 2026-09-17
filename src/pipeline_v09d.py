"""V0.9-D first production BGWO search across five governed validation runs.

This stage executes the approved five-seed BGWO search (seeds 2042-2046) on
the train + development-validation scope using the shared governed fitness
evaluator, computes stability and convergence analyses, selects exactly one
BGWO winner with the frozen validation-only comparator, writes and verifies a
semantic winner lock, and audits that the final test split stays untouched.
It does not run BPSO, does not tune BGWO, does not access final test data, and
produces no V0.9-E artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.optimization.bgwo import (
    BGWOConfig,
    BinaryGreyWolfOptimizer,
    STOP_EARLY_NO_IMPROVEMENT,
    STOP_MAX_ITERATIONS,
)
from src.optimization.feature_fitness import (
    EXPECTED_MODEL_PARAMETERS,
    EXPECTED_THRESHOLD,
    FeatureFitnessEvaluation,
    FeatureFitnessEvaluator,
    FitnessContext,
    MARGINS,
    SeedFitnessRecord,
    constraint_outcome,
    feature_fitness_is_better,
)
from src.pipeline_v08b import (
    DEFAULT_OUTPUT_DIR as V08B_DEFAULT_OUTPUT_DIR,
    IMMUTABLE_PATHS as V08B_IMMUTABLE_PATHS,
    FrozenBasis,
    build_fitness_context,
    load_frozen_basis,
    load_frozen_development_workloads,
    sha256_file,
)
import src.pipeline_v08d as v08d
import src.pipeline_v09b as v09b
import src.pipeline_v09c as v09c
from src.security.experiment_data import (
    DevelopmentExperimentData,
    fingerprint_feature_names,
)


V09D_STAGE = "V0.9-D"
V09D_SCHEMA_VERSION = "v0.9-d-five-run-bgwo-search-1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "bgwo" / "v09d"
WINNER_LOCK_PATH = OUTPUT_DIR / "v09d_winner_lock.json"
RUN_FILE_TEMPLATE = "v09d_run_{seed}.json"
RUN_ARTIFACT_KIND = "OPTIMIZER_RUN_EVIDENCE"
VALIDATION_SCOPE = "TRAIN_AND_DEVELOPMENT_VALIDATION_ONLY"
OPTIMIZER_SEEDS = v09b.APPROVED_OPTIMIZER_SEEDS
MODEL_ATTACK_SEEDS = v09b.APPROVED_MODEL_ATTACK_SEEDS
CARDINALITIES = v09b.APPROVED_CARDINALITIES

EXPECTED_DATASET = "DataCo SMART Supply Chain"
EXPECTED_ROW_CAP = 40000
EXPECTED_SPLIT = {"train": 28000, "validation": 6000, "test": 6000}
EXPECTED_DIMENSIONS = 43
EXPECTED_WOLF_COUNT = 12
EXPECTED_EVALUATED_ITERATIONS = 20
EXPECTED_BUDGET_PER_RUN = EXPECTED_WOLF_COUNT * EXPECTED_EVALUATED_ITERATIONS
EXPECTED_BUDGET_ACROSS_RUNS = EXPECTED_BUDGET_PER_RUN * len(OPTIMIZER_SEEDS)
EXPECTED_INITIAL_CARDINALITIES = (43, 42, *CARDINALITIES)
EARLIEST_EARLY_STOP_EVALUATED_ITERATIONS = 11

IMMUTABLE_PATHS = tuple(
    dict.fromkeys(
        (
            *V08B_IMMUTABLE_PATHS,
            PROJECT_ROOT / "config" / "bgwo_v09.yaml",
            PROJECT_ROOT / "docs" / "v09_bgwo_protocol.md",
            PROJECT_ROOT / "src" / "optimization" / "bgwo.py",
            PROJECT_ROOT / "src" / "pipeline_v09b.py",
            PROJECT_ROOT / "src" / "pipeline_v09c.py",
        )
    )
)
V09C_EVIDENCE_PATHS = (
    PROJECT_ROOT / "results" / "bgwo" / "v09c_pilot" / "v09c_preflight.json",
    PROJECT_ROOT / "results" / "bgwo" / "v09c_pilot" / "v09c_leakage_audit.json",
    PROJECT_ROOT / "results" / "bgwo" / "v09c_pilot" / "v09c_baseline_reproduction.json",
    PROJECT_ROOT / "results" / "bgwo" / "v09c_pilot" / "v09c_comparator_consistency.json",
    PROJECT_ROOT / "results" / "bgwo" / "v09c_pilot" / "v09c_pilot_summary.json",
    PROJECT_ROOT / "results" / "bgwo" / "v09c_pilot" / "v09c_execution_summary.json",
    PROJECT_ROOT / "results" / "bgwo" / "v09c_pilot" / "v09c_artifact_hashes.json",
)


class V09DError(RuntimeError):
    """Raised when V0.9-D cannot preserve its frozen scientific protocol."""


class V09DCheckpointError(V09DError):
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
        raise V09DError(f"Missing frozen V0.8/V0.9 prerequisite: {missing}")
    return {str(path.resolve()): sha256_file(path) for path in paths}


def production_protocol_identity(
    config: BGWOConfig, basis: FrozenBasis, protocol: v09b.V09BProtocol
) -> ProtocolIdentity:
    optimizer_hash = sha256_file(PROJECT_ROOT / "config" / "bgwo_v09.yaml")
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
        "ground_truth": "is_attack",
    }
    fitness_hash = _json_sha256(fitness_payload)
    production_payload = {
        "optimizer": config.to_dict(),
        "optimizer_name": "BGWO",
        "optimizer_seeds": list(OPTIMIZER_SEEDS),
        "optimizer_config_sha256": optimizer_hash,
        "fitness_protocol_sha256": fitness_hash,
        "comparator_sha256": comparator_hash,
        "feature_manifest_sha256": basis.candidate_manifest_sha256,
        "real_k42_mask_sha256": basis.k42_mask_sha256,
        "dataset": {
            "name": EXPECTED_DATASET,
            "row_cap": EXPECTED_ROW_CAP,
            "split_seed": protocol.split_seed,
            "split_sizes": EXPECTED_SPLIT,
            "dimensions": EXPECTED_DIMENSIONS,
        },
    }
    return ProtocolIdentity(
        optimizer_config_sha256=optimizer_hash,
        fitness_protocol_sha256=fitness_hash,
        comparator_sha256=comparator_hash,
        production_protocol_sha256=_json_sha256(production_payload),
        feature_manifest_sha256=basis.candidate_manifest_sha256,
        real_k42_mask_sha256=basis.k42_mask_sha256,
    )


def verify_production_config(config: BGWOConfig) -> None:
    expected = {
        "optimizer_name": "BGWO",
        "dimensions": EXPECTED_DIMENSIONS,
        "wolf_count": EXPECTED_WOLF_COUNT,
        "evaluated_iterations": EXPECTED_EVALUATED_ITERATIONS,
        "maximum_candidate_requests": EXPECTED_BUDGET_PER_RUN,
        "iterative_update_count": EXPECTED_EVALUATED_ITERATIONS - 1,
        "random_cardinalities": list(CARDINALITIES),
        "sigmoid_clamp": [-6.0, 6.0],
        "transfer_function": "standard_logistic_sigmoid",
        "control_parameter_schedule": {
            "kind": "linear_decay",
            "start": 2.0,
            "end": 0.0,
        },
        "minimum_selected_features": 1,
        "early_stopping": {
            "patience_evaluated_iterations": 7,
            "not_before_iteration_index": 10,
        },
        "cache_enabled": True,
        "iteration_index_origin": 0,
    }
    observed = config.to_dict()
    for name, value in expected.items():
        if observed.get(name) != value:
            raise V09DError(f"Production BGWO configuration drift at {name}: {observed}")
    if config.maximum_candidate_requests != EXPECTED_BUDGET_PER_RUN:
        raise V09DError("Production BGWO request budget drift.")


def verify_v09c_artifacts() -> dict[str, Any]:
    missing = [str(path) for path in V09C_EVIDENCE_PATHS if not path.is_file()]
    if missing:
        raise V09DError(f"Missing V0.9-C evidence artifacts: {missing}")
    execution = _read_json(V09C_EVIDENCE_PATHS[5])
    pilot = _read_json(V09C_EVIDENCE_PATHS[4])
    hashes_artifact = _read_json(V09C_EVIDENCE_PATHS[6])
    expected_hashes = {
        path.name: sha256_file(path) for path in V09C_EVIDENCE_PATHS[:-1]
    }
    checks = {
        "v09c_execution_passed": (
            execution.get("stage") == v09c.V09C_STAGE
            and execution.get("status") == "PASS"
        ),
        "v09c_pilot_quarantined": (
            pilot.get("label") == v09c.V09C_PILOT_LABEL
            and pilot.get("eligibility")
            == v09c.V09C_PILOT_ELIGIBILITY
        ),
        "v09c_no_winner_or_search": (
            execution.get("governance", {}).get("production_winner_selected") is False
            and execution.get("governance", {}).get("production_bgwo_search_executed") is False
            and execution.get("governance", {}).get("winner_lock_created") is False
        ),
        "v09c_no_test_access": (
            execution.get("governance", {}).get("final_test_accessed") is False
            and pilot.get("governance", {}).get("final_test_accessed") is False
        ),
        "v09c_artifact_hashes_intact": sorted(hashes_artifact) == sorted(expected_hashes)
        and all(
            hashes_artifact.get(name) == expected_hashes.get(name)
            for name in expected_hashes
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise V09DError(f"V0.9-C artifact verification failed: {failed}")
    return {
        "status": "PASS",
        "checks": checks,
        "v09c_artifact_hashes": {name: hashes_artifact[name] for name in sorted(hashes_artifact)},
    }


def verify_no_v09e_artifacts() -> None:
    root = PROJECT_ROOT / "results" / "bgwo"
    if not root.is_dir():
        return
    forbidden = sorted(path for path in root.rglob("*") if "v09e" in path.name.lower())
    if forbidden:
        raise V09DError(f"V0.9-E artifacts must not exist before V0.9-D: {forbidden}")


def verify_v09d_preflight(protocol: v09b.V09BProtocol | None = None) -> dict[str, Any]:
    loaded = protocol or v09b.load_v09b_protocol()
    v09c_preflight = v09c.verify_v09c_preflight(loaded)
    v09c_artifacts = verify_v09c_artifacts()
    verify_no_v09e_artifacts()
    checks = dict(v09c_preflight["checks"])
    checks.update(
        {
            "v09d_protocol_seeds": tuple(loaded.optimizer_seeds) == OPTIMIZER_SEEDS
            and tuple(loaded.model_attack_seeds) == MODEL_ATTACK_SEEDS,
            "v09d_budget_limits": loaded.config.wolf_count == EXPECTED_WOLF_COUNT
            and loaded.config.evaluated_iterations == EXPECTED_EVALUATED_ITERATIONS
            and loaded.config.maximum_candidate_requests == EXPECTED_BUDGET_PER_RUN,
        }
    )
    checks.update(v09c_artifacts["checks"])
    checks["v09e_artifacts_absent"] = True
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise V09DError(f"V0.9-D preflight failed: {failed}")
    return {
        "schema_version": V09D_SCHEMA_VERSION,
        "stage": V09D_STAGE,
        "status": "PASS",
        "starting_head": v09c.current_head_short(PROJECT_ROOT),
        "protocol_config_sha256": loaded.config_sha256,
        "protocol_document": str(loaded.protocol_doc_path.relative_to(PROJECT_ROOT)),
        "protocol_config": str(loaded.config_path.relative_to(PROJECT_ROOT)),
        "frozen_v08_integrity": v09c_preflight["frozen_v08_integrity"],
        "v09c_preflight": v09c_preflight,
        "v09c_artifacts": v09c_artifacts,
        "checks": checks,
    }


def production_dry_validation(
    config: BGWOConfig,
    basis: FrozenBasis,
    winner_lock: Mapping[str, Any],
) -> dict[str, Any]:
    verify_production_config(config)
    expected_cardinalities = list(EXPECTED_INITIAL_CARDINALITIES)
    dummy = lambda mask: None
    optimizer = BinaryGreyWolfOptimizer(config, dummy, lambda left, right: False)
    rows = []
    for seed in OPTIMIZER_SEEDS:
        population, latent = optimizer.initialize(seed, k42_mask=basis.k42_mask)
        cardinalities = population.sum(axis=1).astype(int).tolist()
        if (
            cardinalities != expected_cardinalities
            or not np.array_equal(population[0], np.ones(43, dtype=np.uint8))
            or not np.array_equal(population[1], basis.k42_mask)
            or not np.isin(population, (0, 1)).all()
            or np.any(population.sum(axis=1) < 1)
            or np.any(latent < -6.0)
            or np.any(latent > 6.0)
            or not np.array_equal(latent, population.astype(float))
        ):
            raise V09DError(f"Production initialization failed for optimizer seed {seed}.")
        independence = v09c.audit_initialization_independence(
            population, basis.candidate_features, basis, winner_lock
        )
        if independence["status"] != "PASS":
            raise V09DError(f"Initialization independence failed for optimizer seed {seed}.")
        rows.append(
            {
                "optimizer_seed": seed,
                "cardinalities": cardinalities,
                "initial_population_sha256": _json_sha256(population.tolist()),
                "initial_latent_positions_sha256": _json_sha256(latent.tolist()),
                "initialization_independence": independence,
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
    config: BGWOConfig,
    context: FitnessContext,
    basis: FrozenBasis,
    identity: ProtocolIdentity,
) -> dict[str, Any]:
    if seed not in OPTIMIZER_SEEDS:
        raise V09DError(f"Unapproved optimizer seed: {seed}")
    evaluator = FeatureFitnessEvaluator(context)
    optimizer = BinaryGreyWolfOptimizer(config, evaluator, feature_fitness_is_better)
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = optimizer.optimize(seed, k42_mask=basis.k42_mask)
    cpu_time = time.process_time() - cpu_start
    wall_time = time.perf_counter() - wall_start
    best = result.best_evaluation
    history = [record.to_dict() for record in result.convergence_history]
    return {
        "schema_version": V09D_SCHEMA_VERSION,
        "stage": V09D_STAGE,
        "artifact_kind": RUN_ARTIFACT_KIND,
        "status": "COMPLETED_AND_VALIDATED_PENDING_CHECKPOINT",
        "execution_mode": "FRESH",
        "optimizer_name": result.optimizer_name,
        "optimizer_seed": seed,
        "model_attack_seeds": list(MODEL_ATTACK_SEEDS),
        "selection_scope": VALIDATION_SCOPE,
        "test_accessed": False,
        "test_authorized": False,
        "test_used_for_winner_selection": False,
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
        "initial_latent_positions": result.initial_latent_positions.tolist(),
        "initial_latent_positions_sha256": _json_sha256(
            result.initial_latent_positions.tolist()
        ),
        "evaluated_iteration_count": result.evaluated_iteration_count,
        "stop_reason": result.stop_reason,
        "best_iteration": result.best_iteration,
        "best_mask": result.best_mask.tolist(),
        "best_mask_sha256": best.mask_sha256,
        "selected_features": list(best.selected_features),
        "selected_features_sha256": best.selected_features_sha256,
        "selected_feature_count": best.selected_feature_count,
        "best_evaluation": best.to_dict(),
        "candidate_requests": result.total_candidate_requests,
        "unique_evaluations": result.unique_evaluations,
        "cache_hits": result.cache_hits,
        "actual_decision_tree_fits": evaluator.decision_tree_fit_count,
        "repairs": result.repair_count,
        "convergence_history": history,
        "wall_time_sec": wall_time,
        "process_cpu_time_sec": cpu_time,
        "evaluation_timing": evaluator.instrumentation(),
    }


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


def validate_run_artifact(
    record: Mapping[str, Any],
    seed: int,
    config: BGWOConfig,
    context: FitnessContext,
    basis: FrozenBasis,
    identity: ProtocolIdentity,
) -> dict[str, bool]:
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
        latent = np.asarray(record["initial_latent_positions"], dtype=float)
        history_evaluations = [evaluation_from_dict(row["best_evaluation"]) for row in history]
        checks = {
            "stage_and_status": record.get("stage") == V09D_STAGE
            and record.get("artifact_kind") == RUN_ARTIFACT_KIND,
            "optimizer_name_bgwo": record.get("optimizer_name") == "BGWO",
            "approved_optimizer_seed": record.get("optimizer_seed") == seed
            and seed in OPTIMIZER_SEEDS,
            "protocol_identity": record.get("protocol_identity") == identity.to_dict(),
            "production_configuration": record.get("configuration") == config.to_dict(),
            "mask_length_and_nonempty": mask.shape == (EXPECTED_DIMENSIONS,)
            and int(mask.sum()) >= 1,
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
            "initial_population": initial.shape == (EXPECTED_WOLF_COUNT, EXPECTED_DIMENSIONS)
            and record.get("initial_cardinalities") == list(EXPECTED_INITIAL_CARDINALITIES)
            and record.get("initial_population_sha256")
            == _json_sha256(initial.tolist()),
            "initial_latent_state": latent.shape
            == (EXPECTED_WOLF_COUNT, EXPECTED_DIMENSIONS)
            and np.all(latent >= -6.0)
            and np.all(latent <= 6.0)
            and np.array_equal(latent, initial.astype(float))
            and record.get("initial_latent_positions_sha256")
            == _json_sha256(latent.tolist()),
            "development_only": record.get("selection_scope") == VALIDATION_SCOPE
            and record.get("test_accessed") is False
            and record.get("test_authorized") is False
            and record.get("test_used_for_winner_selection") is False,
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
            "request_cache_accounting": record.get("candidate_requests")
            == record.get("unique_evaluations") + record.get("cache_hits"),
            "request_iteration_accounting": record.get("candidate_requests")
            == record.get("evaluated_iteration_count") * EXPECTED_WOLF_COUNT
            and all(row.get("request_count") == EXPECTED_WOLF_COUNT for row in history)
            and history[-1].get("cumulative_unique_evaluations")
            == record.get("unique_evaluations")
            and history[-1].get("cumulative_cache_hits") == record.get("cache_hits"),
            "dt_fit_accounting": record.get("actual_decision_tree_fits")
            == record.get("unique_evaluations") * len(MODEL_ATTACK_SEEDS),
            "iteration_budget": 1 <= record.get("evaluated_iteration_count", 0) <= 20
            and record.get("candidate_requests", 0) <= 240,
            "stop_reason": record.get("stop_reason")
            in {STOP_MAX_ITERATIONS, STOP_EARLY_NO_IMPROVEMENT},
            "early_stop_eligibility": record.get("stop_reason") != STOP_EARLY_NO_IMPROVEMENT
            or record.get("evaluated_iteration_count", 0)
            >= EARLIEST_EARLY_STOP_EVALUATED_ITERATIONS,
            "history_coverage": len(history) == record.get("evaluated_iteration_count")
            and [row["iteration_index"] for row in history] == list(range(len(history))),
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
        raise V09DCheckpointError(f"Malformed run artifact for seed {seed}: {exc}") from exc
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise V09DCheckpointError(f"Run artifact validation failed for seed {seed}: {failed}")
    return checks


def execute_or_resume_run(
    seed: int,
    output_dir: Path,
    config: BGWOConfig,
    context: FitnessContext,
    basis: FrozenBasis,
    identity: ProtocolIdentity,
    *,
    executor: Callable[
        [int, BGWOConfig, FitnessContext, FrozenBasis, ProtocolIdentity], dict[str, Any]
    ] = execute_production_run,
) -> tuple[dict[str, Any], str, str]:
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
            absolute = int(np.count_nonzero(masks[left] != masks[right]))
            pairs.append(
                {
                    "seed_left": seeds[left],
                    "seed_right": seeds[right],
                    "intersection": intersection,
                    "union": union,
                    "jaccard": jaccard,
                    "normalized_hamming": hamming,
                    "absolute_hamming_distance": absolute,
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
        "schema_version": V09D_SCHEMA_VERSION,
        "stage": V09D_STAGE,
        "artifact_kind": "DERIVED_OPTIMIZER_STABILITY",
        "optimizer_run_count": len(records),
        "cardinality": _summary(cardinalities),
        "pairwise": pairs,
        "jaccard": _summary([row["jaccard"] for row in pairs]),
        "normalized_hamming": _summary(
            [row["normalized_hamming"] for row in pairs]
        ),
        "absolute_hamming_distance": _summary(
            [row["absolute_hamming_distance"] for row in pairs]
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
                    "iteration_index": row["iteration_index"],
                    "best_selected_feature_count": row["best_selected_feature_count"],
                    "average_precision": evaluation["mean_metrics"]["average_precision"],
                    "f1": evaluation["mean_metrics"]["f1"],
                    "recall": evaluation["mean_metrics"]["recall"],
                    "feasible": evaluation["feasible"],
                    "normalized_violation": evaluation["normalized_violation"],
                    "population_diversity": row["population_diversity"],
                    "control_parameter_a": row["control_parameter_a"],
                    "best_rank_improved": row["best_rank_improved"],
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
                "best_iteration": record["best_iteration"],
                "evaluated_iterations": record["evaluated_iteration_count"],
                "early_stopped": record["stop_reason"]
                == STOP_EARLY_NO_IMPROVEMENT,
                "stop_reason": record["stop_reason"],
                "comparator_improvements_after_initialization": max(
                    0,
                    sum(bool(row["best_rank_improved"]) for row in history) - 1,
                ),
                "unique_evaluations": record["unique_evaluations"],
                "cache_hit_rate": float(
                    record["cache_hits"] / record["candidate_requests"]
                ),
                "repairs": record["repairs"],
                "initial_diversity": diversities[0],
                "final_diversity": diversities[-1],
                "minimum_diversity": min(diversities),
                "maximum_diversity": max(diversities),
            }
        )
    aggregate = []
    for iteration in range(EXPECTED_EVALUATED_ITERATIONS):
        rows = [row for row in plot_rows if row["iteration_index"] == iteration]
        if not rows:
            continue
        aggregate.append(
            {
                "iteration_index": iteration,
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
        "schema_version": V09D_SCHEMA_VERSION,
        "stage": V09D_STAGE,
        "artifact_kind": "DERIVED_CONVERGENCE_ANALYSIS",
        "run_summaries": run_summaries,
        "plot_data": plot_rows,
        "aggregate_by_iteration": aggregate,
        "test_accessed": False,
    }


def build_validation_comparison(
    records: Sequence[Mapping[str, Any]],
    basis: FrozenBasis,
    winner_lock: Mapping[str, Any],
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
                    100.0 * (EXPECTED_DIMENSIONS - evaluation.selected_feature_count)
                    / EXPECTED_DIMENSIONS
                ),
                "feature_count_difference_vs_k11": evaluation.selected_feature_count - 11,
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
    bpso_validation = winner_lock.get("validation_metrics", {})
    return {
        "schema_version": V09D_SCHEMA_VERSION,
        "stage": V09D_STAGE,
        "artifact_kind": "VALIDATION_COMPARISON",
        "bgwo_run_winners": rows,
        "historical_validation_anchors": {
            "K43": {"feature_count": EXPECTED_DIMENSIONS, "metrics": k43},
            "K42": {"feature_count": 42, "metrics": k42},
            "K11": {
                "feature_count": 11,
                "metrics": k11,
                "selection_semantics": "seed-specific historical MI rule",
            },
        },
        "bpso_descriptive_context": {
            "description": (
                "Descriptive only. The governed V0.8 BPSO winner subset is shown "
                "for context; no superiority claim is made and BPSO is not rerun."
            ),
            "source": "v08c_winner_lock",
            "selected_feature_count": int(winner_lock.get("selected_feature_count", 0)),
            "validation_metrics": {
                name: bpso_validation.get(name)
                for name in ("average_precision", "f1", "recall")
            },
            "no_superiority_declared": True,
            "bpso_rerun": False,
            "used_for_winner_selection": False,
        },
        "bgwo_selection_semantics": "one universal subset candidate per optimizer run",
        "final_test_comparison_performed": False,
        "statistical_significance_claimed": False,
        "test_accessed": False,
    }


def select_search_winner(
    records: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], FeatureFitnessEvaluation]:
    if len(records) != len(OPTIMIZER_SEEDS):
        raise V09DError("Winner selection requires exactly five run records.")
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
    protocol: v09b.V09BProtocol,
    budget: Mapping[str, Any],
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
        "schema_version": V09D_SCHEMA_VERSION,
        "stage": V09D_STAGE,
        "status": "VALIDATION_LOCKED",
        "eligible_for_v09e": True,
        "selection_scope": VALIDATION_SCOPE,
        "search_scope": "PRODUCTION_BGWO_SEARCH_TRAIN_AND_DEVELOPMENT_VALIDATION_ONLY",
        "optimizer": "BGWO",
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
            "optimizer": "BGWO",
            "resource_metrics_used": False,
            "test_metrics_used": False,
        },
        "selection_rationale": rationale,
        "five_run_evidence_sha256": {
            str(seed): run_hashes[seed] for seed in OPTIMIZER_SEEDS
        },
        "optimization_budget": {
            "optimizer": "BGWO",
            "wolf_count": EXPECTED_WOLF_COUNT,
            "configured_evaluated_iterations": EXPECTED_EVALUATED_ITERATIONS,
            "maximum_candidate_requests_per_run": EXPECTED_BUDGET_PER_RUN,
            "maximum_candidate_requests_across_runs": EXPECTED_BUDGET_ACROSS_RUNS,
            "five_run_total_candidate_requests": int(budget["candidate_requests"]),
        },
        "dataset_identity": {
            "name": EXPECTED_DATASET,
            "row_cap": EXPECTED_ROW_CAP,
            "split_seed": protocol.split_seed,
            "split_sizes": EXPECTED_SPLIT,
            "dimensions": EXPECTED_DIMENSIONS,
            "feature_manifest_sha256": identity.feature_manifest_sha256,
        },
        "creation_metadata": {
            "created_at_utc": created_at or datetime.now(timezone.utc).isoformat(),
            "optimizer_run_count": len(OPTIMIZER_SEEDS),
            "optimizer_seeds": list(OPTIMIZER_SEEDS),
            "model_attack_seeds": list(MODEL_ATTACK_SEEDS),
            "optimizer": "BGWO",
        },
        "winner_statement": "Winner selected without V0.9 final-test access.",
        "historical_test_statement": (
            "Final test remains locked for all V0.9 stages; it was not touched by "
            "V0.9-D search and selection."
        ),
        "final_test_accessed": False,
        "test_authorized": False,
        "test_used_for_winner_selection": False,
        "logistic_regression_evaluated": False,
        "resource_benchmark_executed": False,
        "direct_energy_measured": False,
        "bpso_rerun": False,
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
            raise V09DError("Existing winner lock differs from the recomputed winner.")
        return existing
    _atomic_write_json(path, lock)
    stored = _read_json(path)
    verify_winner_lock(stored)
    return stored


def verify_winner_lock(lock: Mapping[str, Any]) -> None:
    required = {
        "stage",
        "status",
        "eligible_for_v09e",
        "selection_scope",
        "search_scope",
        "optimizer",
        "source_optimizer_seed",
        "mask",
        "ordered_selected_features",
        "selected_feature_count",
        "feature_manifest_sha256",
        "mask_sha256",
        "selected_features_sha256",
        "fitness_protocol_sha256",
        "optimizer_config_sha256",
        "production_protocol_sha256",
        "validation_metrics",
        "feasible",
        "normalized_violation",
        "comparator",
        "five_run_evidence_sha256",
        "optimization_budget",
        "dataset_identity",
        "test_authorized",
        "test_used_for_winner_selection",
        "semantic_lock_sha256",
    }
    mask = np.asarray(lock.get("mask", ()), dtype=np.uint8)
    selected = tuple(lock.get("ordered_selected_features", ()))
    budget = lock.get("optimization_budget", {})
    dataset = lock.get("dataset_identity", {})
    valid = (
        required <= set(lock)
        and lock.get("stage") == V09D_STAGE
        and lock.get("status") == "VALIDATION_LOCKED"
        and lock.get("eligible_for_v09e") is True
        and lock.get("selection_scope") == VALIDATION_SCOPE
        and lock.get("optimizer") == "BGWO"
        and lock.get("source_optimizer_seed") in OPTIMIZER_SEEDS
        and mask.shape == (EXPECTED_DIMENSIONS,)
        and np.isin(mask, (0, 1)).all()
        and int(mask.sum()) == lock.get("selected_feature_count") == len(selected)
        and hashlib.sha256(mask.tobytes()).hexdigest() == lock.get("mask_sha256")
        and fingerprint_feature_names(selected) == lock.get("selected_features_sha256")
        and set(lock.get("five_run_evidence_sha256", {}))
        == {str(seed) for seed in OPTIMIZER_SEEDS}
        and lock.get("final_test_accessed") is False
        and lock.get("test_authorized") is False
        and lock.get("test_used_for_winner_selection") is False
        and lock.get("logistic_regression_evaluated") is False
        and lock.get("bpso_rerun") is False
        and lock.get("comparator", {}).get("resource_metrics_used") is False
        and lock.get("comparator", {}).get("test_metrics_used") is False
        and budget.get("optimizer") == "BGWO"
        and budget.get("maximum_candidate_requests_across_runs") == EXPECTED_BUDGET_ACROSS_RUNS
        and 0 <= int(budget.get("five_run_total_candidate_requests", 0))
        <= EXPECTED_BUDGET_ACROSS_RUNS
        and dataset.get("name") == EXPECTED_DATASET
        and dataset.get("row_cap") == EXPECTED_ROW_CAP
        and dataset.get("split_sizes") == EXPECTED_SPLIT
        and dataset.get("dimensions") == EXPECTED_DIMENSIONS
        and winner_lock_semantic_hash(lock) == lock.get("semantic_lock_sha256")
    )
    if not valid:
        raise V09DError("Winner validation lock verification failed.")


def build_test_access_audit(
    *,
    winner_lock: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    leakage_audit: Mapping[str, Any],
) -> dict[str, Any]:
    root = PROJECT_ROOT / "results" / "bgwo"
    v09e_entries = sorted(path for path in root.rglob("*") if "v09e" in path.name.lower())
    return {
        "schema_version": V09D_SCHEMA_VERSION,
        "stage": V09D_STAGE,
        "status": "PASS",
        "final_test_accessed": False,
        "test_authorized": False,
        "test_used_for_winner_selection": False,
        "winner_lock_test_flags": {
            "final_test_accessed": winner_lock.get("final_test_accessed"),
            "test_authorized": winner_lock.get("test_authorized"),
            "test_used_for_winner_selection": winner_lock.get(
                "test_used_for_winner_selection"
            ),
        },
        "run_test_flags": {
            str(record["optimizer_seed"]): {
                "test_accessed": record.get("test_accessed"),
                "test_authorized": record.get("test_authorized"),
                "test_used_for_winner_selection": record.get(
                    "test_used_for_winner_selection"
                ),
            }
            for record in records
        },
        "leakage_audit_status": leakage_audit.get("status"),
        "leakage_test_accessed": leakage_audit.get("test_accessed"),
        "no_v09e_artifacts": not v09e_entries,
        "bpso_rerun": False,
        "test_loader_referenced": False,
    }


def run_v09d(
    *,
    output_dir: Path | str = OUTPUT_DIR,
    basis_loader: Callable[[], FrozenBasis] = load_frozen_basis,
    workload_loader: Callable[[FrozenBasis], tuple[DevelopmentExperimentData, ...]] = load_frozen_development_workloads,
) -> dict[str, Any]:
    stage_start = time.perf_counter()
    protocol = v09b.load_v09b_protocol()
    preflight = verify_v09d_preflight(protocol)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "v09d_winner_lock.json").exists():
        existing_lock = _read_json(output / "v09d_winner_lock.json")
        try:
            verify_winner_lock(existing_lock)
        except V09DError as exc:
            raise V09DError("Existing V0.9-D winner lock is invalid; refusing to resume.") from exc

    basis = basis_loader()
    workloads = workload_loader(basis)
    context = v09c.build_v09c_fitness_context(basis, workloads)
    leakage = v09c.build_v09c_leakage_audit(context)
    if leakage["status"] != "PASS":
        raise V09DError("V0.9-D leakage audit failed.")

    config = protocol.config
    verify_production_config(config)
    identity = production_protocol_identity(config, basis, protocol)
    winner_lock_bpso = v08d.load_verified_winner_lock()
    dry_validation = production_dry_validation(config, basis, winner_lock_bpso)

    immutable_before = snapshot_immutable_paths()
    records: list[dict[str, Any]] = []
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
        )
        validate_run_artifact(record, seed, config, context, basis, identity)
        if snapshot_immutable_paths() != immutable_before:
            raise V09DError(f"Frozen artifacts changed during optimizer run {seed}.")
        records.append(record)
        modes[seed] = mode
        run_hashes[seed] = run_hash
    if tuple(record["optimizer_seed"] for record in records) != OPTIMIZER_SEEDS:
        raise V09DError("Five-run optimizer coverage is incomplete.")

    stability = build_stability_analysis(records, basis.candidate_features)
    convergence = build_convergence_analysis(records)
    comparison = build_validation_comparison(records, basis, winner_lock_bpso)
    winner_record, winner_evaluation = select_search_winner(records)
    total_requests = sum(int(record["candidate_requests"]) for record in records)
    proposed_lock = build_winner_lock(
        winner_record,
        winner_evaluation,
        identity,
        run_hashes,
        protocol=protocol,
        budget={"candidate_requests": total_requests},
    )
    winner_lock = write_or_verify_winner_lock(
        output / "v09d_winner_lock.json", proposed_lock
    )
    verify_winner_lock(winner_lock)

    _atomic_write_json(output / "v09d_stability.json", stability)
    _atomic_write_json(output / "v09d_convergence.json", convergence)
    _atomic_write_json(output / "v09d_validation_comparison.json", comparison)
    audit = build_test_access_audit(
        winner_lock=winner_lock,
        records=records,
        leakage_audit=leakage,
    )
    _atomic_write_json(output / "v09d_test_access_audit.json", audit)

    immutable_after = snapshot_immutable_paths()
    if immutable_after != immutable_before:
        raise V09DError("Frozen artifacts changed during V0.9-D analysis or locking.")

    total_unique = sum(int(record["unique_evaluations"]) for record in records)
    total_hits = sum(int(record["cache_hits"]) for record in records)
    total_fits = sum(int(record["actual_decision_tree_fits"]) for record in records)
    total_repairs = sum(int(record["repairs"]) for record in records)
    summary = {
        "schema_version": V09D_SCHEMA_VERSION,
        "stage": V09D_STAGE,
        "status": "COMPLETED",
        "readiness_for_v09e": "GO",
        "starting_head": preflight["starting_head"],
        "preflight": preflight,
        "protocol_identity": identity.to_dict(),
        "dataset_identity": {
            "name": EXPECTED_DATASET,
            "row_cap": EXPECTED_ROW_CAP,
            "split_seed": protocol.split_seed,
            "split_sizes": EXPECTED_SPLIT,
            "dimensions": EXPECTED_DIMENSIONS,
        },
        "dry_validation": dry_validation,
        "execution_mode": "FRESH" if set(modes.values()) == {"FRESH"} else "RESUMED",
        "run_execution_modes": {str(seed): modes[seed] for seed in OPTIMIZER_SEEDS},
        "run_artifact_sha256": {str(seed): run_hashes[seed] for seed in OPTIMIZER_SEEDS},
        "optimizer_seeds": list(OPTIMIZER_SEEDS),
        "model_attack_seeds": list(MODEL_ATTACK_SEEDS),
        "completed_run_count": len(records),
        "candidate_requests": total_requests,
        "unique_evaluations": total_unique,
        "cache_hits": total_hits,
        "actual_decision_tree_fits": total_fits,
        "optimizer_wall_time_sec": sum(float(record["wall_time_sec"]) for record in records),
        "optimizer_cpu_time_sec": sum(
            float(record["process_cpu_time_sec"]) for record in records
        ),
        "pipeline_wall_time_sec": time.perf_counter() - stage_start,
        "stop_reasons": {
            str(seed): record["stop_reason"]
            for seed, record in zip(OPTIMIZER_SEEDS, records)
        },
        "total_repairs": total_repairs,
        "winner": {
            "source_optimizer_seed": winner_lock["source_optimizer_seed"],
            "optimizer": "BGWO",
            "selected_feature_count": winner_lock["selected_feature_count"],
            "mask_sha256": winner_lock["mask_sha256"],
            "selected_features_sha256": winner_lock["selected_features_sha256"],
            "semantic_lock_sha256": winner_lock["semantic_lock_sha256"],
            "feasible": winner_lock["feasible"],
            "normalized_violation": winner_lock["normalized_violation"],
            "validation_metrics": {
                "average_precision": winner_lock["validation_metrics"]["average_precision"],
                "f1": winner_lock["validation_metrics"]["f1"],
                "recall": winner_lock["validation_metrics"]["recall"],
            },
        },
        "artifact_hashes": {
            name: sha256_file(output / name)
            for name in (
                "v09d_stability.json",
                "v09d_convergence.json",
                "v09d_validation_comparison.json",
                "v09d_winner_lock.json",
                "v09d_test_access_audit.json",
            )
        },
        "immutable_hashes_before": immutable_before,
        "immutable_hashes_after": immutable_after,
        "selection_scope": VALIDATION_SCOPE,
        "final_test_accessed": False,
        "test_authorized": False,
        "test_used_for_winner_selection": False,
        "logistic_regression_evaluated": False,
        "resource_benchmark_executed": False,
        "direct_energy_measured": False,
        "bpso_rerun": False,
        "final_scientific_claims_supported": False,
    }
    _atomic_write_json(output / "v09d_execution_summary.json", summary)
    verify_v09d_artifacts(output, config, context, basis, identity)
    return summary


def verify_v09d_artifacts(
    output: Path,
    config: BGWOConfig,
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
    verify_winner_lock(_read_json(output / "v09d_winner_lock.json"))
    for name, kind in (
        ("v09d_stability.json", "DERIVED_OPTIMIZER_STABILITY"),
        ("v09d_convergence.json", "DERIVED_CONVERGENCE_ANALYSIS"),
        ("v09d_validation_comparison.json", "VALIDATION_COMPARISON"),
    ):
        artifact = _read_json(output / name)
        if artifact.get("stage") != V09D_STAGE or artifact.get("artifact_kind") != kind:
            raise V09DError(f"Invalid derived artifact: {name}")
    audit = _read_json(output / "v09d_test_access_audit.json")
    if audit.get("status") != "PASS" or audit.get("final_test_accessed") is not False:
        raise V09DError("V0.9-D test access audit artifact is invalid.")


def _summary(values: Sequence[float | int]) -> dict[str, float]:
    numeric = [float(value) for value in values]
    if not numeric:
        raise V09DError("Cannot summarize an empty value sequence.")
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
        raise V09DError("Relative validation comparison requires nonzero baseline.")
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
        raise V09DCheckpointError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V09DCheckpointError(f"JSON artifact must be an object: {path}")
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
    summary = run_v09d()
    print(
        json.dumps(
            {
                "stage": V09D_STAGE,
                "status": summary["status"],
                "execution_mode": summary["execution_mode"],
                "completed_run_count": summary["completed_run_count"],
                "winner": summary["winner"],
                "final_test_accessed": False,
                "test_authorized": False,
                "test_used_for_winner_selection": False,
                "bpso_rerun": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())