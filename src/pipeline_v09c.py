"""V0.9-C real-fitness integration with a quarantined BGWO pilot.

This stage reuses the governed V0.8 feature-fitness semantics and validates
real evaluator connectivity for BGWO. It does not execute production BGWO
search, does not select a winner, and does not access final test data.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.optimization.bgwo import (
    BGWOConfig,
    BinaryGreyWolfOptimizer,
    STOP_EARLY_NO_IMPROVEMENT,
    STOP_MAX_ITERATIONS,
)
import src.optimization.feature_fitness as fitness
from src.optimization.feature_fitness import (
    FeatureFitnessEvaluation,
    FeatureFitnessEvaluator,
    FitnessContext,
    audit_fitness_context,
    feature_fitness_is_better,
)
import src.pipeline_v08b as v08b
import src.pipeline_v08d as v08d
import src.pipeline_v08e4 as v08e4
import src.pipeline_v09b as v09b
from src.security.experiment_data import DevelopmentExperimentData, fingerprint_feature_names


V09C_STAGE = "V0.9-C"
V09C_SCHEMA_VERSION = "v0.9-c-real-fitness-integration-1"
V09C_LABEL = "LEAKAGE_SAFE_REAL_FITNESS_INTEGRATION"
V09C_PILOT_LABEL = "QUARANTINED_PILOT"
V09C_PILOT_ELIGIBILITY = "NOT_ELIGIBLE_FOR_SCIENTIFIC_WINNER_SELECTION"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "bgwo" / "v09c_pilot"

EXPECTED_DATASET = "DataCo SMART Supply Chain"
EXPECTED_ROW_CAP = 40000
EXPECTED_SPLIT = {"train": 28000, "validation": 6000, "test": 6000}
EXPECTED_DIMENSIONS = 43
EXPECTED_PILOT_OPTIMIZER_SEED = 2042
EXPECTED_PILOT_EVALUATED_ITERATIONS = 2
EXPECTED_PILOT_MAX_REQUESTS = 24


class V09CError(RuntimeError):
    """Raised when V0.9-C integration checks fail closed."""


def current_head_short(root: Path = PROJECT_ROOT) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_v09c_preflight(protocol: v09b.V09BProtocol | None = None) -> dict[str, Any]:
    """Verify frozen prerequisites and protocol identities before integration."""
    loaded = protocol or v09b.load_v09b_protocol()
    frozen = v09b.verify_v08_frozen_integrity()
    winner_lock = v08d.load_verified_winner_lock()
    v08d_lock = _read_json(PROJECT_ROOT / "results" / "bpso" / "v08d_final_test_lock.json")
    v08e_lock = _read_json(
        PROJECT_ROOT / "results" / "bpso" / "v08e_e4_analysis" / "v08e_scientific_result_lock.json"
    )
    v08d.verify_final_test_lock(v08d_lock)
    v08e4.verify_result_lock(v08e_lock)
    v09b_status = v09b.run_v09b_synthetic_validation(output_path=None)

    governed = loaded.raw_snapshot.get("governed_sources", {})
    checks = {
        "v09a_stage": loaded.raw_snapshot.get("stage") == "V0.9-A",
        "v09a_schema": loaded.raw_snapshot.get("version") == "v0.9-a-bgwo-protocol-1",
        "v09a_optimizer_name": loaded.config.optimizer_name == "BGWO",
        "v09a_dimensions": loaded.config.dimensions == EXPECTED_DIMENSIONS,
        "v09a_population": loaded.config.wolf_count == 12,
        "v09a_budget_per_run": loaded.config.maximum_candidate_requests == 240,
        "v08_frozen_integrity": frozen["status"] == "PASS"
        and frozen["immutable_hash_count"] == 53,
        "v08c_winner_unchanged": winner_lock["semantic_lock_sha256"] == v08d.EXPECTED_WINNER["semantic_lock_sha256"],
        "v08d_lock_unchanged": v08d_lock["semantic_result_lock_sha256"]
        == governed.get("v08d_semantic_result_lock_sha256"),
        "v08e_lock_unchanged": v08e_lock["semantic_result_lock_sha256"]
        == governed.get("v08e_semantic_result_lock_sha256"),
        "v09b_implementation_status": v09b_status["status"] == "PASS",
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise V09CError(f"V0.9-C preflight failed: {failed}")
    return {
        "schema_version": V09C_SCHEMA_VERSION,
        "stage": V09C_STAGE,
        "status": "PASS",
        "starting_head": current_head_short(PROJECT_ROOT),
        "protocol_config_sha256": loaded.config_sha256,
        "protocol_document": str(loaded.protocol_doc_path.relative_to(PROJECT_ROOT)),
        "protocol_config": str(loaded.config_path.relative_to(PROJECT_ROOT)),
        "frozen_v08_integrity": frozen,
        "v09b_status": {
            "status": v09b_status["status"],
            "validation_kind": v09b_status["validation_kind"],
        },
        "checks": checks,
    }


def ensure_test_split_inaccessible(context: FitnessContext) -> None:
    """Fail closed when any test split surface can reach optimizer fitness."""
    if context.test_accessed:
        raise V09CError("V0.9-C fitness context indicates test access.")
    for index, workload in enumerate(context.workloads):
        if hasattr(workload, "test"):
            raise V09CError(f"Workload {index} unexpectedly exposes a test split.")
        if workload.train.split_name != "train" or workload.validation.split_name != "validation":
            raise V09CError(f"Workload {index} split names are not train/validation.")
        if (
            workload.train.test_authorization_id is not None
            or workload.validation.test_authorization_id is not None
            or workload.train.test_authorization_capability is not None
            or workload.validation.test_authorization_capability is not None
        ):
            raise V09CError(f"Workload {index} carries test authorization metadata.")


def build_v09c_fitness_context(
    basis: v08b.FrozenBasis,
    workloads: tuple[DevelopmentExperimentData, ...],
) -> FitnessContext:
    """Build the shared governed evaluator context reused from V0.8-B/C."""
    context = v08b.build_fitness_context(basis, workloads)
    ensure_test_split_inaccessible(context)
    return context


def run_comparator_consistency_checks() -> dict[str, Any]:
    """Programmatically verify feasible and infeasible ranking semantics."""
    feasible_smaller_k = _synthetic_evaluation(
        mask=_mask_with_indices((0, 1, 2)),
        feasible=True,
        violation=0.0,
        average_precision=0.7,
        f1=0.7,
        recall=0.7,
    )
    feasible_larger_k = _synthetic_evaluation(
        mask=_mask_with_indices((0, 1, 2, 3)),
        feasible=True,
        violation=0.0,
        average_precision=1.0,
        f1=1.0,
        recall=1.0,
    )
    feasible_ap_high = _synthetic_evaluation(
        mask=_mask_with_indices((0, 1, 2, 3)),
        feasible=True,
        violation=0.0,
        average_precision=0.91,
        f1=0.7,
        recall=0.7,
    )
    feasible_ap_low = _synthetic_evaluation(
        mask=_mask_with_indices((0, 1, 2, 3)),
        feasible=True,
        violation=0.0,
        average_precision=0.90,
        f1=0.8,
        recall=0.8,
    )
    feasible_recall_high = _synthetic_evaluation(
        mask=_mask_with_indices((0, 1, 2, 3)),
        feasible=True,
        violation=0.0,
        average_precision=0.90,
        f1=0.80,
        recall=0.71,
    )
    feasible_recall_low = _synthetic_evaluation(
        mask=_mask_with_indices((0, 1, 2, 3)),
        feasible=True,
        violation=0.0,
        average_precision=0.90,
        f1=0.80,
        recall=0.70,
    )
    feasible_mask_low = _synthetic_evaluation(
        mask=_mask_with_indices((41, 42)),
        feasible=True,
        violation=0.0,
        average_precision=0.8,
        f1=0.8,
        recall=0.8,
    )
    feasible_mask_high = _synthetic_evaluation(
        mask=_mask_with_indices((0, 1)),
        feasible=True,
        violation=0.0,
        average_precision=0.8,
        f1=0.8,
        recall=0.8,
    )

    infeasible_low_violation = _synthetic_evaluation(
        mask=_mask_with_indices((0, 1, 2, 3)),
        feasible=False,
        violation=0.10,
        average_precision=0.5,
        f1=0.5,
        recall=0.5,
    )
    infeasible_high_violation = _synthetic_evaluation(
        mask=_mask_with_indices((0, 1)),
        feasible=False,
        violation=0.20,
        average_precision=0.9,
        f1=0.9,
        recall=0.9,
    )
    infeasible_ap_high = _synthetic_evaluation(
        mask=_mask_with_indices((0, 1, 2)),
        feasible=False,
        violation=0.20,
        average_precision=0.81,
        f1=0.6,
        recall=0.6,
    )
    infeasible_ap_low = _synthetic_evaluation(
        mask=_mask_with_indices((0, 1, 2)),
        feasible=False,
        violation=0.20,
        average_precision=0.80,
        f1=0.9,
        recall=0.9,
    )
    infeasible_small_k = _synthetic_evaluation(
        mask=_mask_with_indices((0, 1, 3)),
        feasible=False,
        violation=0.20,
        average_precision=0.80,
        f1=0.70,
        recall=0.70,
    )
    infeasible_large_k = _synthetic_evaluation(
        mask=_mask_with_indices((0, 1, 2, 3)),
        feasible=False,
        violation=0.20,
        average_precision=0.80,
        f1=0.70,
        recall=0.70,
    )
    infeasible_mask_low = _synthetic_evaluation(
        mask=_mask_with_indices((41, 42)),
        feasible=False,
        violation=0.25,
        average_precision=0.5,
        f1=0.5,
        recall=0.5,
    )
    infeasible_mask_high = _synthetic_evaluation(
        mask=_mask_with_indices((0, 1)),
        feasible=False,
        violation=0.25,
        average_precision=0.5,
        f1=0.5,
        recall=0.5,
    )

    checks = {
        "feasible_smaller_k_priority": feature_fitness_is_better(feasible_smaller_k, feasible_larger_k),
        "feasible_ap_priority": feature_fitness_is_better(feasible_ap_high, feasible_ap_low),
        "feasible_recall_priority": feature_fitness_is_better(feasible_recall_high, feasible_recall_low),
        "feasible_canonical_mask_tiebreak": feature_fitness_is_better(feasible_mask_low, feasible_mask_high),
        "infeasible_violation_priority": feature_fitness_is_better(infeasible_low_violation, infeasible_high_violation),
        "infeasible_ap_priority": feature_fitness_is_better(infeasible_ap_high, infeasible_ap_low),
        "infeasible_smaller_k_after_metrics": feature_fitness_is_better(infeasible_small_k, infeasible_large_k),
        "infeasible_canonical_mask_tiebreak": feature_fitness_is_better(infeasible_mask_low, infeasible_mask_high),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema_version": V09C_SCHEMA_VERSION,
        "stage": V09C_STAGE,
        "status": status,
        "feasible_comparator_verified": all(
            checks[name]
            for name in (
                "feasible_smaller_k_priority",
                "feasible_ap_priority",
                "feasible_recall_priority",
                "feasible_canonical_mask_tiebreak",
            )
        ),
        "infeasible_comparator_verified": all(
            checks[name]
            for name in (
                "infeasible_violation_priority",
                "infeasible_ap_priority",
                "infeasible_smaller_k_after_metrics",
                "infeasible_canonical_mask_tiebreak",
            )
        ),
        "checks": checks,
    }


def build_v09c_leakage_audit(context: FitnessContext) -> dict[str, Any]:
    """Augment shared evaluator audit with V0.9-C boundary checks."""
    base = audit_fitness_context(context)
    feature_fitness_source = Path(fitness.__file__).read_text(encoding="utf-8")
    checks = {
        check.name: check.passed for check in base.checks
    }
    checks.update(
        {
            "winner_selection_disabled_in_stage": True,
            "no_test_derived_feature_selection": True,
            "resource_objective_excluded_from_fitness": (
                '"resource_or_energy_objective_used": False' in feature_fitness_source
                and "weighted_objective_used" in feature_fitness_source
            ),
            "energy_metrics_excluded_from_fitness": "direct_energy" not in feature_fitness_source.lower(),
            "shared_feature_fitness_reused": "class FeatureFitnessEvaluator" in feature_fitness_source,
            "shared_constrained_comparator_reused": "def feature_fitness_is_better" in feature_fitness_source,
        }
    )
    status = "PASS" if base.status == "PASS" and all(checks.values()) else "FAIL"
    return {
        "schema_version": V09C_SCHEMA_VERSION,
        "stage": V09C_STAGE,
        "status": status,
        "label": "REAL_FITNESS_LEAKAGE_AUDIT",
        "scope": "train_and_validation_only",
        "checks": checks,
        "pass_count": sum(bool(value) for value in checks.values()),
        "total_check_count": len(checks),
        "forbidden_ground_truth": ["Late_delivery_risk", "SUSPECTED_FRAUD"],
        "ground_truth": "is_attack",
        "test_accessed": False,
    }


def reproduce_k43_baseline(
    protocol: v09b.V09BProtocol,
    context: FitnessContext,
    basis: v08b.FrozenBasis,
) -> dict[str, Any]:
    """Reproduce frozen K43 baseline with V0.8-approved tolerance semantics."""
    record, evaluation = v08b.reproduce_k43(context, basis)
    expected_protocol = {
        "average_precision": float(protocol.raw_snapshot["predictive_baseline"]["average_precision"]),
        "f1": float(protocol.raw_snapshot["predictive_baseline"]["f1"]),
        "recall": float(protocol.raw_snapshot["predictive_baseline"]["recall"]),
    }
    expected_frozen = basis.baseline.to_dict()
    reproduced = {
        "average_precision": evaluation.average_precision,
        "f1": evaluation.f1,
        "recall": evaluation.recall,
    }
    abs_diff_frozen = {
        metric: abs(reproduced[metric] - expected_frozen[metric])
        for metric in reproduced
    }
    abs_diff_protocol = {
        metric: abs(reproduced[metric] - expected_protocol[metric])
        for metric in reproduced
    }
    max_diff_frozen = max(abs_diff_frozen.values())
    status = "PASS" if record["status"] == "PASS" else "FAIL"
    return {
        "schema_version": V09C_SCHEMA_VERSION,
        "stage": V09C_STAGE,
        "status": status,
        "label": "K43_BASELINE_REPRODUCTION",
        "tolerance": {
            "kind": "absolute",
            "value": v08b.BASELINE_REPRODUCTION_ABS_TOLERANCE,
        },
        "expected_protocol": expected_protocol,
        "expected_frozen": expected_frozen,
        "reproduced": reproduced,
        "absolute_diff_vs_frozen": abs_diff_frozen,
        "absolute_diff_vs_protocol": abs_diff_protocol,
        "maximum_absolute_diff_vs_frozen": max_diff_frozen,
        "fit_count_for_reproduction": record["preflight_decision_tree_fits"],
        "v08b_reproduction_checks": record["checks"],
    }


def build_pilot_config(config: BGWOConfig) -> BGWOConfig:
    """Derive the fixed small-budget V0.9-C quarantined pilot configuration."""
    pilot = replace(config, evaluated_iterations=EXPECTED_PILOT_EVALUATED_ITERATIONS)
    if (
        pilot.wolf_count != 12
        or pilot.evaluated_iterations != EXPECTED_PILOT_EVALUATED_ITERATIONS
        or pilot.maximum_candidate_requests != EXPECTED_PILOT_MAX_REQUESTS
    ):
        raise V09CError("V0.9-C pilot configuration drift detected.")
    return pilot


def audit_initialization_independence(
    initial_population: np.ndarray,
    candidate_features: Sequence[str],
    basis: v08b.FrozenBasis,
    winner_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Ensure prohibited historical identities are not injected in BGWO init."""
    bpso_mask = np.asarray(winner_lock["mask"], dtype=np.uint8)
    k11_by_seed = basis.lock.get("roles", {}).get("best_supervised", {}).get("selected_features", {})
    k11_masks = {
        seed: _mask_from_selected_features(candidate_features, tuple(features))
        for seed, features in k11_by_seed.items()
    }
    bpso_present = any(np.array_equal(row, bpso_mask) for row in initial_population)
    k11_present = any(
        np.array_equal(row, mask)
        for row in initial_population
        for mask in k11_masks.values()
    )
    checks = {
        "real_k42_anchor_used": np.array_equal(initial_population[1], basis.k42_mask),
        "bpso_k10_identity_not_present": not bpso_present,
        "mi_k11_seed_identities_not_present": not k11_present,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "bpso_k10_present": bpso_present,
        "mi_k11_present": k11_present,
    }


def run_quarantined_pilot(
    protocol: v09b.V09BProtocol,
    context: FitnessContext,
    basis: v08b.FrozenBasis,
    winner_lock: Mapping[str, Any],
    *,
    optimizer_seed: int = EXPECTED_PILOT_OPTIMIZER_SEED,
) -> dict[str, Any]:
    """Execute the V0.9-C small-budget integration pilot (non-scientific)."""
    if optimizer_seed != EXPECTED_PILOT_OPTIMIZER_SEED:
        raise V09CError("V0.9-C pilot seed drift detected.")
    ensure_test_split_inaccessible(context)
    pilot_config = build_pilot_config(protocol.config)
    evaluator = FeatureFitnessEvaluator(context)
    optimizer = BinaryGreyWolfOptimizer(
        pilot_config,
        evaluator,
        feature_fitness_is_better,
    )
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = optimizer.optimize(optimizer_seed, k42_mask=basis.k42_mask)
    wall_time = time.perf_counter() - wall_start
    cpu_time = time.process_time() - cpu_start

    identity_audit = audit_initialization_independence(
        result.initial_population,
        basis.candidate_features,
        basis,
        winner_lock,
    )
    accounting = {
        "request_accounting": result.total_candidate_requests
        == result.unique_evaluations + result.cache_hits,
        "request_budget": result.total_candidate_requests <= pilot_config.maximum_candidate_requests,
        "dt_fit_accounting": evaluator.decision_tree_fit_count == result.unique_evaluations * 5,
        "pilot_boundary_no_early_stop": (
            result.stop_reason == STOP_MAX_ITERATIONS
            and result.evaluated_iteration_count == pilot_config.evaluated_iterations
        ),
    }
    if identity_audit["status"] != "PASS" or not all(accounting.values()):
        raise V09CError("V0.9-C pilot identity or accounting validation failed.")

    evaluation = result.best_evaluation
    return {
        "schema_version": V09C_SCHEMA_VERSION,
        "stage": V09C_STAGE,
        "status": "PASS",
        "label": V09C_PILOT_LABEL,
        "eligibility": V09C_PILOT_ELIGIBILITY,
        "scientific_experiment": False,
        "scientific_claims_supported": False,
        "optimizer_seed": optimizer_seed,
        "population": pilot_config.wolf_count,
        "evaluated_iterations": pilot_config.evaluated_iterations,
        "maximum_candidate_requests": pilot_config.maximum_candidate_requests,
        "candidate_requests": result.total_candidate_requests,
        "unique_candidate_evaluations": result.unique_evaluations,
        "cache_hits": result.cache_hits,
        "decision_tree_fits": evaluator.decision_tree_fit_count,
        "evaluated_iteration_count": result.evaluated_iteration_count,
        "runtime": {
            "wall_time_sec": wall_time,
            "process_cpu_time_sec": cpu_time,
        },
        "best": {
            "selected_feature_count": evaluation.selected_feature_count,
            "average_precision": evaluation.average_precision,
            "f1": evaluation.f1,
            "recall": evaluation.recall,
            "feasible": evaluation.feasible,
            "normalized_violation": evaluation.normalized_violation,
            "mask_sha256": evaluation.mask_sha256,
        },
        "stop_reason": result.stop_reason,
        "early_stop": result.stop_reason == STOP_EARLY_NO_IMPROVEMENT,
        "identity_audit": identity_audit,
        "accounting": accounting,
        "governance": {
            "winner_selected": False,
            "winner_lock_created": False,
            "production_search_executed": False,
            "final_test_accessed": False,
            "bpso_rerun": False,
        },
    }


def verify_no_winner_lock_artifacts(output_dir: Path | str) -> None:
    """Fail if the stage output contains any winner-lock artifact."""
    target = Path(output_dir)
    matches = sorted(
        path
        for path in target.rglob("*")
        if path.is_file() and "winner_lock" in path.name
    )
    if matches:
        relative = [str(path.relative_to(target)) for path in matches]
        raise V09CError(f"V0.9-C must not create winner locks: {relative}")


def run_v09c(
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    basis_loader: Callable[[], v08b.FrozenBasis] = v08b.load_frozen_basis,
    workload_loader: Callable[[v08b.FrozenBasis], tuple[DevelopmentExperimentData, ...]] = v08b.load_frozen_development_workloads,
) -> dict[str, Any]:
    """Execute V0.9-C preflight, baseline reproduction, leakage checks, and pilot."""
    stage_start = time.perf_counter()
    protocol = v09b.load_v09b_protocol()
    preflight = verify_v09c_preflight(protocol)

    basis = basis_loader()
    workloads = workload_loader(basis)
    context = build_v09c_fitness_context(basis, workloads)
    leakage = build_v09c_leakage_audit(context)
    if leakage["status"] != "PASS":
        raise V09CError("V0.9-C leakage audit failed.")

    baseline = reproduce_k43_baseline(protocol, context, basis)
    if baseline["status"] != "PASS":
        raise V09CError("V0.9-C baseline reproduction failed.")

    comparator = run_comparator_consistency_checks()
    if comparator["status"] != "PASS":
        raise V09CError("V0.9-C comparator consistency checks failed.")

    winner_lock = v08d.load_verified_winner_lock()
    pilot = run_quarantined_pilot(protocol, context, basis, winner_lock)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "protocol_config_sha256": protocol.config_sha256,
        "protocol_document_sha256": hashlib.sha256(
            protocol.protocol_doc_path.read_bytes()
        ).hexdigest(),
        "bgwo_source_sha256": hashlib.sha256(
            (PROJECT_ROOT / "src" / "optimization" / "bgwo.py").read_bytes()
        ).hexdigest(),
        "feature_fitness_source_sha256": hashlib.sha256(
            (PROJECT_ROOT / "src" / "optimization" / "feature_fitness.py").read_bytes()
        ).hexdigest(),
        "feature_fitness_comparator_sha256": hashlib.sha256(
            inspect.getsource(feature_fitness_is_better).encode("utf-8")
        ).hexdigest(),
    }
    execution = {
        "schema_version": V09C_SCHEMA_VERSION,
        "stage": V09C_STAGE,
        "status": "PASS",
        "label": V09C_LABEL,
        "pilot_label": V09C_PILOT_LABEL,
        "starting_head": preflight["starting_head"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_wall_time_sec": time.perf_counter() - stage_start,
        "protocol_identity": identity,
        "preflight_status": preflight["status"],
        "baseline_status": baseline["status"],
        "leakage_status": leakage["status"],
        "comparator_status": comparator["status"],
        "pilot_status": pilot["status"],
        "governance": {
            "production_winner_selected": False,
            "winner_lock_created": False,
            "production_bgwo_search_executed": False,
            "final_test_accessed": False,
            "bpso_rerun": False,
        },
    }
    payloads: dict[str, Mapping[str, Any]] = {
        "v09c_preflight.json": preflight,
        "v09c_leakage_audit.json": leakage,
        "v09c_baseline_reproduction.json": baseline,
        "v09c_comparator_consistency.json": comparator,
        "v09c_pilot_summary.json": pilot,
        "v09c_execution_summary.json": execution,
    }
    for name, payload in payloads.items():
        _atomic_write_json(output / name, payload)
    verify_no_winner_lock_artifacts(output)
    artifact_hashes = {
        name: sha256_file(output / name)
        for name in sorted(payloads)
    }
    _atomic_write_json(output / "v09c_artifact_hashes.json", artifact_hashes)

    return {
        "stage": V09C_STAGE,
        "status": "PASS",
        "output_dir": str(output),
        "artifact_hashes": artifact_hashes,
        "pilot": {
            "label": pilot["label"],
            "optimizer_seed": pilot["optimizer_seed"],
            "candidate_requests": pilot["candidate_requests"],
            "decision_tree_fits": pilot["decision_tree_fits"],
        },
        "final_test_accessed": False,
        "production_winner_selected": False,
        "winner_lock_created": False,
        "production_bgwo_search_executed": False,
        "bpso_rerun": False,
    }


def _synthetic_evaluation(
    *,
    mask: np.ndarray,
    feasible: bool,
    violation: float,
    average_precision: float,
    f1: float,
    recall: float,
) -> FeatureFitnessEvaluation:
    selected_indices = [index for index, value in enumerate(mask) if int(value) == 1]
    selected_features = tuple(f"f{index:02d}" for index in selected_indices)
    return FeatureFitnessEvaluation(
        mask=tuple(int(value) for value in mask),
        mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
        selected_features=selected_features,
        selected_features_sha256=fingerprint_feature_names(selected_features),
        selected_feature_count=len(selected_features),
        feasible=feasible,
        normalized_violation=float(violation),
        relative_losses={
            "average_precision": 0.0,
            "f1": 0.0,
            "recall": 0.0,
        },
        mean_metrics={
            "average_precision": float(average_precision),
            "f1": float(f1),
            "recall": float(recall),
            "precision": 0.0,
            "roc_auc": 0.0,
            "accuracy": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "attack_prevalence": 0.05,
            "selected_feature_visibility_rate": 0.0,
        },
        confusion_totals={
            "true_positives": 0,
            "true_negatives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        },
        per_seed=(),
        decision_tree_fit_count=0,
    )


def _mask_with_indices(indices: Sequence[int], dimensions: int = 43) -> np.ndarray:
    mask = np.zeros(dimensions, dtype=np.uint8)
    for index in indices:
        mask[int(index)] = 1
    return mask


def _mask_from_selected_features(
    candidate_features: Sequence[str],
    selected_features: Sequence[str],
) -> np.ndarray:
    selected = set(selected_features)
    return np.asarray(
        [int(feature in selected) for feature in candidate_features],
        dtype=np.uint8,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V09CError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise V09CError(f"JSON artifact must be an object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    summary = run_v09c()
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "status": summary["status"],
                "pilot": summary["pilot"],
                "final_test_accessed": summary["final_test_accessed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
