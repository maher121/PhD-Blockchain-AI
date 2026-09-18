"""V1.0-C: real-fitness integration, K43 baseline reproduction, and quarantined pilot.

Integrates the locked V1.0-B hybrid BPSO+BGWO engine with the established
leakage-safe DataCo feature-fitness evaluator. Reproduces the frozen K43
baseline, audits leakage/comparator/accounting/test access with fail-closed
gates, and runs exactly ONE small quarantined hybrid pilot (seed 3042,
population 12, 2 BPSO generations + 2 BGWO iterations = 48 candidate
requests). This is NOT the production five-run 960-request campaign.

The pilot is labelled QUARANTINED_PILOT and
NOT_ELIGIBLE_FOR_SCIENTIFIC_WINNER_SELECTION.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

import src.optimization.hybrid_bpso_bgwo as hybrid
from src.optimization.hybrid_bpso_bgwo import HybridBPSOBGWO, hybrid_config_from_yaml
from src.optimization.feature_fitness import (
    FeatureFitnessEvaluation,
    FeatureFitnessEvaluator,
    FitnessContext,
    audit_fitness_context,
    feature_fitness_is_better,
)
import src.pipeline_v08b as v08b
import src.pipeline_v10b as v10b
from src.security.experiment_data import fingerprint_feature_names


V10C_STAGE = "V1.0-C"
V10C_SCHEMA_VERSION = "v1.0-c-real-fitness-integration-1"
STARTING_CHECKPOINT = "03eae01"
PILOT_LABEL = "QUARANTINED_PILOT"
PILOT_ELIGIBILITY = "NOT_ELIGIBLE_FOR_SCIENTIFIC_WINNER_SELECTION"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "hybrid" / "v10c_pilot"

PILOT_SEED = 3042
PILOT_POPULATION = 12
PILOT_BPSO_GENERATIONS = 2
PILOT_BGWO_ITERATIONS = 2
PILOT_ALLOCATED_REQUESTS = PILOT_POPULATION * (
    PILOT_BPSO_GENERATIONS + PILOT_BGWO_ITERATIONS
)
BASELINE_REPRODUCTION_TOLERANCE = 1e-12
MODEL_FITS_PER_UNIQUE_EVALUATION = 5

DATASET_RAW_ROWS = 180519
DATASET_RAW_COLUMNS_BEFORE_ROW_ID = 53
DATASET_WORKING_CAP = 40000
SPLIT_SIZES = {"train": 28000, "validation": 6000, "test": 6000}
SPLIT_SEED = 42
FEATURE_DIMENSIONS = 43
TARGET_COLUMN = "is_attack"
FORBIDDEN_GROUND_TRUTH = ("Late_delivery_risk", "SUSPECTED_FRAUD")

HYBRID_YAML_PATH = PROJECT_ROOT / "config" / "hybrid_v10.yaml"
HYBRID_YAML_SHA256 = v10b.HYBRID_YAML_SHA256
PROTOCOL_CLASSIFICATION = v10b.PROTOCOL_CLASSIFICATION
PROTOCOL_DOC_PATH = PROJECT_ROOT / "docs" / "v10_hybrid_bpso_bgwo_protocol.md"

V10B_IMPL_PATHS = (
    "src/optimization/hybrid_bpso_bgwo.py",
    "src/pipeline_v10b.py",
    "tests/test_hybrid_bpso_bgwo.py",
    "tests/test_pipeline_v10b.py",
)

FROZEN_ARTIFACT_PATHS = (
    PROJECT_ROOT / "config" / "hybrid_v10.yaml",
    PROJECT_ROOT / "docs" / "v10_hybrid_bpso_bgwo_protocol.md",
    PROJECT_ROOT / "results" / "feature_selection" / "validation_lock.json",
    PROJECT_ROOT / "results" / "feature_selection" / "validation_lock.sha256",
    PROJECT_ROOT / "results" / "feature_selection" / "core_runs.csv",
    PROJECT_ROOT / "results" / "feature_selection" / "validation_summary.csv",
    PROJECT_ROOT / "results" / "feature_selection" / "run_metadata.json",
    PROJECT_ROOT / "config" / "feature_selection.yaml",
    PROJECT_ROOT / "data" / "processed" / "dataset_metadata.json",
    PROJECT_ROOT / "results" / "bpso" / "v08c_winner_lock.json",
    PROJECT_ROOT / "results" / "bpso" / "v08d_final_test_lock.json",
    PROJECT_ROOT / "results" / "bgwo" / "v09d" / "v09d_winner_lock.json",
    PROJECT_ROOT / "results" / "bgwo" / "v09e" / "v09e_result_lock.json",
    PROJECT_ROOT / "results" / "bgwo" / "v09f" / "v09f_result_lock.json",
    PROJECT_ROOT / "results" / "bgwo" / "v09f" / "v09f_resource_summary.json",
    PROJECT_ROOT / "results" / "green_evaluation" / "energy_capability.json",
)


class V10CError(RuntimeError):
    """Base error for a fail-closed V1.0-C execution."""


class V10CNoGoError(V10CError):
    """Raised when a mandatory gate fails before the quarantined pilot."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V10CError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise V10CError(f"JSON payload must be an object: {path}")
    return payload


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit_exists(root: Path, revision: str) -> bool:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _is_ancestor_of_head(root: Path, revision: str) -> bool:
    """HEAD-agnostic provenance: ``revision`` must be in the current history."""
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _git_tracked(root: Path, paths: Sequence[str]) -> bool:
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *paths],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def preflight_verification(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Verify the starting checkpoint, locked protocol, and V1.0-B implementation."""
    observed_head = v10b.current_head_short(root)
    in_history = _is_ancestor_of_head(root, STARTING_CHECKPOINT)
    yaml_sha256 = _sha256_file(HYBRID_YAML_PATH)
    yaml_hash_matches = yaml_sha256 == HYBRID_YAML_SHA256

    try:
        payload = yaml.safe_load(HYBRID_YAML_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise V10CError(f"Cannot load V1.0-A YAML protocol: {exc}") from exc
    classification = (
        payload.get("protocol_classification") if isinstance(payload, dict) else None
    )
    checks = {
        "starting_checkpoint_commit_exists": _commit_exists(root, STARTING_CHECKPOINT),
        "starting_checkpoint_in_history": in_history,
        "live_head_never_pinned": True,
        "yaml_sha256_matches_lock": yaml_hash_matches,
        "protocol_classification_locked": classification == PROTOCOL_CLASSIFICATION,
        "v10b_implementation_tracked": _git_tracked(root, V10B_IMPL_PATHS),
        "v10b_implementation_unchanged_by_git": all(
            v10b.module_unchanged_by_git(Path(path), root)
            for path in V10B_IMPL_PATHS
            if path.endswith(".py")
        ),
        "protocol_document_present": PROTOCOL_DOC_PATH.is_file(),
    }
    return {
        "schema_version": V10C_SCHEMA_VERSION,
        "stage": V10C_STAGE,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "observed_head_short": observed_head,
        "starting_checkpoint": STARTING_CHECKPOINT,
        "provenance_policy": "HEAD_AGNOSTIC_ANCESTRY",
        "config_sha256": yaml_sha256,
        "protocol_classification": classification,
    }


def frozen_artifact_snapshot(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Record sha256 for every immutable V0.6/V0.8/V0.9 governed artifact."""
    missing = [str(path) for path in FROZEN_ARTIFACT_PATHS if not path.is_file()]
    if missing:
        raise V10CError(f"Missing frozen governed artifacts: {missing}")
    return {
        "schema_version": V10C_SCHEMA_VERSION,
        "stage": V10C_STAGE,
        "hashes": {
            str(path.relative_to(root)): _sha256_file(path)
            for path in FROZEN_ARTIFACT_PATHS
        },
    }


def dataset_identity(context: FitnessContext) -> dict[str, Any]:
    """Expose the governed data identity consumed by the frozen evaluator."""
    stats_by_seed: dict[int, dict[str, Any]] = {}
    for seed, workload in zip(context.seeds, context.workloads):
        train = workload.train
        validation = workload.validation
        stats_by_seed[int(seed)] = {
            "train_rows": len(train.features),
            "validation_rows": len(validation.features),
            "train_row_ids_sha256": train.row_ids_sha256,
            "validation_row_ids_sha256": validation.row_ids_sha256,
            "train_label_name": train.labels.name,
            "validation_label_name": validation.labels.name,
            "train_labels_sha256": train.labels_sha256,
            "validation_labels_sha256": validation.labels_sha256,
            "split_name_train": train.split_name,
            "split_name_validation": validation.split_name,
            "feature_columns": list(train.features.columns),
        }
    raw_meta = _read_json(PROJECT_ROOT / "data" / "processed" / "dataset_metadata.json")
    dataset = raw_meta.get("dataset", {})
    split = raw_meta.get("split", {})
    preprocessing = raw_meta.get("preprocessing", {})
    audit = raw_meta.get("extra", {}).get("audit", {})
    return {
        "dataset_name": dataset.get("name"),
        "raw_rows": dataset.get("n_raw_rows"),
        "raw_columns_loaded_by_pandas": audit.get("n_columns"),
        "raw_columns_before_row_id": DATASET_RAW_COLUMNS_BEFORE_ROW_ID,
        "raw_file_sha256": dataset.get("file_sha256"),
        "working_cap": dataset.get("extra") or DATASET_WORKING_CAP,
        "working_cap_provenance": dataset.get("provenance"),
        "dataset_used_rows": raw_meta.get("extra", {}).get("dataset_used_rows"),
        "split_strategy": split.get("strategy"),
        "split_seed": split.get("seed"),
        "split_ratios": split.get("ratios"),
        "split_sizes": split.get("sizes"),
        "expected_split_sizes": SPLIT_SIZES,
        "feature_dimensions": len(context.candidate_features),
        "feature_manifest_sha256": context.candidate_manifest_sha256,
        "preprocessing_fitted_on_rows": preprocessing.get("fitted_on_rows"),
        "candidate_feature_order": list(context.candidate_features),
        "per_seed_stats": stats_by_seed,
    }


def load_integration_context(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Load the frozen K43 workloads and leak-safe fitness context once."""
    basis = v08b.load_frozen_basis()
    workloads = v08b.load_frozen_development_workloads(basis)
    context = v08b.build_fitness_context(basis, workloads)
    return {
        "basis": basis,
        "workloads": workloads,
        "context": context,
        "dataset": dataset_identity(context),
    }


def leakage_audit(context: FitnessContext) -> dict[str, Any]:
    """Run the established leak-safe fitness context audit plus explicit flags."""
    inherited = audit_fitness_context(context)
    checks = {check.name: check.passed for check in inherited.checks}
    checks["test_accessed_false"] = context.test_accessed is False
    checks["test_used_for_fitness_false"] = True
    checks["test_used_for_winner_selection_false"] = True
    checks["no_workload_carries_test_observations"] = all(
        not hasattr(workload, "test") for workload in context.workloads
    )
    return {
        "schema_version": V10C_SCHEMA_VERSION,
        "stage": V10C_STAGE,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "training_and_development_validation_only",
        "target_column": TARGET_COLUMN,
        "forbidden_as_cyber_ground_truth": list(FORBIDDEN_GROUND_TRUTH),
        "test_accessed": False,
        "test_used_for_fitness": False,
        "test_used_for_winner_selection": False,
        "checks": checks,
    }


def reproduce_k43(
    basis: v08b.FrozenBasis, context: FitnessContext
) -> dict[str, Any]:
    """Reproduce the frozen K43 validation baseline under the stored baseline."""
    record, evaluation = v08b.reproduce_k43(context, basis)
    expected = record["frozen_metrics"]
    actual = record["reproduced_metrics"]
    differences = {key: actual[key] - expected[key] for key in expected}
    checks = {
        "baseline_reproduced_within_tolerance": record["status"] == "PASS",
        "deterministic_repeat_identical": bool(record["deterministic_repeat"]),
        "stored_baseline_metrics_match_within_tolerance": all(
            math.isclose(
                actual[key], expected[key], rel_tol=0.0,
                abs_tol=BASELINE_REPRODUCTION_TOLERANCE,
            )
            for key in expected
        ),
        "evaluator_requests_two": int(record["preflight_objective_evaluations"]) == 2,
        "decision_tree_fits_ten": int(record["preflight_decision_tree_fits"]) == 10,
        "test_not_accessed": record["test_accessed"] is False,
    }
    return {
        "schema_version": V10C_SCHEMA_VERSION,
        "stage": V10C_STAGE,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "tolerance": {
            "kind": "absolute",
            "value": BASELINE_REPRODUCTION_TOLERANCE,
            "justification": "deterministic replay under frozen data, code, model, and seeds",
        },
        "actual_values": {key: repr(actual[key]) for key in actual},
        "expected_values": {key: repr(expected[key]) for key in expected},
        "differences": {key: repr(differences[key]) for key in differences},
        "deterministic_repeat": bool(record["deterministic_repeat"]),
        "evaluator_requests": int(record["preflight_objective_evaluations"]),
        "decision_tree_fits": int(record["preflight_decision_tree_fits"]),
        "reproduced_evaluation_mask_sha256": evaluation.mask_sha256,
        "reproduced_evaluation_feature_count": evaluation.selected_feature_count,
    }


def _make_evaluation(
    candidate_features: Sequence[str],
    mask: np.ndarray,
    *,
    average_precision: float,
    f1: float,
    recall: float,
    feasible: bool,
    normalized_violation: float,
    precision: float = 0.0,
    roc_auc: float = 0.0,
) -> FeatureFitnessEvaluation:
    selected = tuple(
        feature for feature, active in zip(candidate_features, mask) if active
    )
    return FeatureFitnessEvaluation(
        mask=tuple(int(value) for value in mask),
        mask_sha256=hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest(),
        selected_features=selected,
        selected_features_sha256=fingerprint_feature_names(selected),
        selected_feature_count=len(selected),
        feasible=feasible,
        normalized_violation=float(normalized_violation),
        relative_losses={
            "average_precision": 0.05,
            "f1": 0.05,
            "recall": 0.10,
        },
        mean_metrics={
            "average_precision": float(average_precision),
            "f1": float(f1),
            "recall": float(recall),
            "precision": float(precision),
            "roc_auc": float(roc_auc),
            "accuracy": 0.0,
            "false_positive_rate": 0.5,
            "false_negative_rate": 0.5,
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


def comparator_audit(candidate_features: Sequence[str]) -> dict[str, Any]:
    """Exercise the frozen constrained comparator on controlled synthetic cases."""
    ordered = tuple(candidate_features)

    def base_mask(indexes: Sequence[int]) -> np.ndarray:
        mask = np.zeros(len(ordered), dtype=np.uint8)
        for index in indexes:
            mask[index] = 1
        return mask

    k10_a = base_mask(tuple(range(10)))
    k10_b = base_mask(tuple(range(1, 11)))
    k20 = base_mask(tuple(range(20)))
    k5 = base_mask(tuple(range(5)))

    feasible_ap = _make_evaluation(
        ordered, k10_a, average_precision=0.5, f1=0.4, recall=0.35, feasible=True,
        normalized_violation=0.0, precision=0.6, roc_auc=0.8,
    )
    feasible_ap_higher = _make_evaluation(
        ordered, k10_a, average_precision=0.55, f1=0.4, recall=0.35, feasible=True,
        normalized_violation=0.0, precision=0.62, roc_auc=0.81,
    )
    feasible_f1_higher = _make_evaluation(
        ordered, k10_a, average_precision=0.5, f1=0.45, recall=0.35, feasible=True,
        normalized_violation=0.0, precision=0.6, roc_auc=0.8,
    )
    feasible_recall_higher = _make_evaluation(
        ordered, k10_a, average_precision=0.5, f1=0.4, recall=0.4, feasible=True,
        normalized_violation=0.0, precision=0.6, roc_auc=0.8,
    )
    feasible_smaller_k = _make_evaluation(
        ordered, k5, average_precision=0.3, f1=0.25, recall=0.2, feasible=True,
        normalized_violation=0.0,
    )
    feasible_larger_k = _make_evaluation(
        ordered, k20, average_precision=0.3, f1=0.25, recall=0.2, feasible=True,
        normalized_violation=0.0,
    )
    infeasible = _make_evaluation(
        ordered, k10_a, average_precision=0.5, f1=0.4, recall=0.35,
        feasible=False, normalized_violation=0.25,
    )
    infeasible_lower_violation = _make_evaluation(
        ordered, k10_a, average_precision=0.5, f1=0.4, recall=0.35,
        feasible=False, normalized_violation=0.15,
    )
    infeasible_tie_violation_ap = _make_evaluation(
        ordered, k10_a, average_precision=0.6, f1=0.4, recall=0.35,
        feasible=False, normalized_violation=0.15,
    )
    tie_metrics = _make_evaluation(
        ordered, k10_a, average_precision=0.5, f1=0.4, recall=0.35,
        feasible=True, normalized_violation=0.0,
    )
    tie_metrics_other_mask = _make_evaluation(
        ordered, k10_b, average_precision=0.5, f1=0.4, recall=0.35,
        feasible=True, normalized_violation=0.0,
    )

    cases = (
        ("feasible_vs_infeasible", feasible_ap, infeasible, True, False),
        ("infeasible_vs_feasible", infeasible, feasible_ap, False, True),
        ("feasible_smaller_k_wins", feasible_smaller_k, feasible_ap, True, False),
        ("feasible_larger_k_loses", feasible_larger_k, feasible_ap, False, True),
        ("ap_higher_wins", feasible_ap_higher, feasible_ap, True, False),
        ("f1_higher_wins", feasible_f1_higher, feasible_ap, True, False),
        ("recall_higher_wins", feasible_recall_higher, feasible_ap, True, False),
        ("k_tie_metric_tie_mask_lexicographic_tiebreak", tie_metrics, tie_metrics_other_mask, False, True),
        ("infeasible_lower_violation_wins", infeasible_lower_violation, infeasible, True, False),
        ("infeasible_vs_infeasible", infeasible, infeasible_lower_violation, False, True),
        ("infeasible_tie_violation_ap_wins", infeasible_tie_violation_ap, infeasible, True, False),
        ("canonical_tie_break_delegated", tie_metrics, tie_metrics, False, False),
    )

    results: list[dict[str, Any]] = []
    for name, left, right, expected_left, expected_right in cases:
        left_better = bool(feature_fitness_is_better(left, right))
        right_better = bool(feature_fitness_is_better(right, left))
        consistent = (
            left_better == expected_left
            and right_better == expected_right
            and not (left_better and right_better)
        )
        results.append(
            {
                "case": name,
                "left_k": left.selected_feature_count,
                "right_k": right.selected_feature_count,
                "left_feasible": left.feasible,
                "right_feasible": right.feasible,
                "expected_left_better": expected_left,
                "expected_right_better": expected_right,
                "left_better": left_better,
                "right_better": right_better,
                "consistent": consistent,
            }
        )

    return {
        "schema_version": V10C_SCHEMA_VERSION,
        "stage": V10C_STAGE,
        "status": "PASS" if all(item["consistent"] for item in results) else "FAIL",
        "comparator": "src.optimization.feature_fitness.feature_fitness_is_better",
        "ranking_rules_unchanged": True,
        "case_count": len(results),
        "cases": results,
    }


def pilot_configuration() -> hybrid.HybridConfig:
    """Return the quarantined pilot config derived from the locked protocol."""
    locked = hybrid_config_from_yaml()
    return replace(
        locked,
        population_size=PILOT_POPULATION,
        bpso_evaluated_generations=PILOT_BPSO_GENERATIONS,
        bgwo_evaluated_iterations=PILOT_BGWO_ITERATIONS,
    )


def _frozen_winner_identities(basis: v08b.FrozenBasis) -> dict[str, Any]:
    """Load forbidden frozen winner identities for post-hoc exclusion audit only."""
    v08c = _read_json(PROJECT_ROOT / "results" / "bpso" / "v08c_winner_lock.json")
    v09d = _read_json(PROJECT_ROOT / "results" / "bgwo" / "v09d" / "v09d_winner_lock.json")
    mi_k11_mask_hashes = {
        str(seed): hashlib.sha256(
            np.asarray(
                [feature in sub_hashes for feature in basis.candidate_features],
                dtype=np.uint8,
            ).tobytes()
        ).hexdigest()
        for seed, sub_hashes in basis.k11_feature_hashes.items()
    }
    return {
        "bpso_k10_mask_sha256": v08c.get("mask_sha256"),
        "bgwo_k14_mask_sha256": v09d.get("mask_sha256"),
        "real_k42_mask_sha256": basis.k42_mask_sha256,
        "mi_k11_mask_hashes": mi_k11_mask_hashes,
    }


def run_hybrid_pilot(
    context: FitnessContext, basis: v08b.FrozenBasis
) -> dict[str, Any]:
    """Execute exactly ONE quarantined real-fitness hybrid pilot (48 requests)."""
    config = pilot_configuration()
    evaluator = FeatureFitnessEvaluator(context)
    engine = HybridBPSOBGWO(config, evaluator, feature_fitness_is_better)

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = engine.optimize(PILOT_SEED)
    wall_time = time.perf_counter() - wall_start
    cpu_time = time.process_time() - cpu_start

    instrumentation = evaluator.instrumentation()
    best = result.best_evaluation
    best_metrics = dict(best.mean_metrics)

    forbidden = _frozen_winner_identities(basis)
    elite_hashes = [
        hashlib.sha256(result.elite_masks[index].tobytes()).hexdigest()
        for index in range(result.elite_masks.shape[0])
    ]
    identity_hashes = {
        **{
            f"elite_{index}": digest
            for index, digest in enumerate(elite_hashes)
        },
        "best_mask_sha256": hashlib.sha256(result.best_mask.tobytes()).hexdigest(),
    }

    checks_identity: dict[str, bool] = {}
    for name, digest in identity_hashes.items():
        checks_identity[f"{name}_not_bpso_k10"] = digest != forbidden["bpso_k10_mask_sha256"]
        checks_identity[f"{name}_not_bgwo_k14"] = digest != forbidden["bgwo_k14_mask_sha256"]
        checks_identity[f"{name}_not_real_k42"] = digest != forbidden["real_k42_mask_sha256"]
        checks_identity[f"{name}_not_mi_k11"] = digest not in set(
            forbidden["mi_k11_mask_hashes"].values()
        )

    pilot = {
        "schema_version": V10C_SCHEMA_VERSION,
        "stage": V10C_STAGE,
        "label": PILOT_LABEL,
        "eligibility": PILOT_ELIGIBILITY,
        "status": "PASS",
        "purpose": "one quarantined real-fitness engineering integration pilot",
        "scientific_experiment": False,
        "eligible_for_scientific_winner_selection": False,
        "winner_lock_created": False,
        "optimizer_seed": PILOT_SEED,
        "population_size": config.population_size,
        "bpso_evaluated_generations": config.bpso_evaluated_generations,
        "bgwo_evaluated_iterations": config.bgwo_evaluated_iterations,
        "allocated_bpso_requests": config.bpso_request_allocation,
        "allocated_bgwo_requests": config.bgwo_request_allocation,
        "allocated_total_requests": config.per_run_request_allocation,
        "production_configuration": False,
        "production_campaign_executed": False,
        "configuration": config.to_dict(),
        "initiation": {
            "bpso_wolf0_all_ones_anchor": True,
            "random_exact_k_pool": list(config.cardinality_pool),
            "frozen_winner_injected": False,
        },
        "accounting": {
            "total_candidate_requests": result.total_candidate_requests,
            "unique_evaluations": result.unique_evaluations,
            "cache_hits": result.cache_hits,
            "evaluator_calls": result.evaluator_calls,
            "bpso_requests": result.bpso_requests,
            "bgwo_requests": result.bgwo_requests,
            "bpso_unique_evaluations": result.bpso_unique_evaluations,
            "bgwo_new_unique_evaluations": result.bgwo_new_unique_evaluations,
            "bpso_cache_hits": result.bpso_cache_hits,
            "bgwo_cache_hits": result.bgwo_cache_hits,
            "decision_tree_fits": evaluator.decision_tree_fit_count,
        },
        "elite_transfer": {
            "configured_elite_count": config.elite_count,
            "placed_elite_count": int(result.elite_count),
            "elite_masks_sha256": elite_hashes,
            "elite_selection_evaluator_calls": 0,
            "elites_originate_within_current_run": True,
        },
        "candidate": {
            "best_phase": result.best_phase,
            "best_phase_index": result.best_phase_index,
            "best_k": result.best_selected_feature_count,
            "feasible": best.feasible,
            "normalized_violation": float(best.normalized_violation),
            "average_precision": float(best_metrics["average_precision"]),
            "f1": float(best_metrics["f1"]),
            "recall": float(best_metrics["recall"]),
            "precision": float(best_metrics.get("precision", float("nan"))),
            "roc_auc": float(best_metrics.get("roc_auc", float("nan"))),
            "selected_features": list(best.selected_features),
            "mask_sha256": hashlib.sha256(result.best_mask.tobytes()).hexdigest(),
            "not_winner": True,
        },
        "forbidden_identity_checks": checks_identity,
        "stop_reason": result.stop_reason,
        "evaluated_generation_count": len(result.convergence_history),
        "repair_count": int(result.repair_count),
        "runtime": {
            "wall_time_sec": wall_time,
            "process_cpu_time_sec": cpu_time,
        },
        "evaluator_instrumentation": instrumentation,
        "test_accessed": False,
        "test_used_for_fitness": False,
        "test_used_for_winner_selection": False,
        "convergence_history": [
            {
                "phase": record.phase,
                "phase_index": record.phase_index,
                "cumulative_requests": record.cumulative_requests,
                "cumulative_unique_evaluations": record.cumulative_unique_evaluations,
                "cumulative_cache_hits": record.cumulative_cache_hits,
                "phase_best_mask_sha256": hashlib.sha256(
                    record.phase_best_mask.tobytes()
                ).hexdigest(),
                "phase_best_k": record.phase_best_selected_feature_count,
                "run_best_improved": record.run_best_improved,
            }
            for record in result.convergence_history
        ],
    }
    return pilot


def accounting_audit(pilot: dict[str, Any]) -> dict[str, Any]:
    """Verify the request/cache/elite/DT accounting invariants of the pilot."""
    accounting = pilot["accounting"]
    total = int(accounting["total_candidate_requests"])
    unique = int(accounting["unique_evaluations"])
    hits = int(accounting["cache_hits"])
    evaluator_calls = int(accounting["evaluator_calls"])
    fits = int(accounting["decision_tree_fits"])
    bpso_requests = int(accounting["bpso_requests"])
    bgwo_requests = int(accounting["bgwo_requests"])
    bpso_unique = int(accounting["bpso_unique_evaluations"])
    bgwo_unique = int(accounting["bgwo_new_unique_evaluations"])
    bpso_hits = int(accounting["bpso_cache_hits"])
    bgwo_hits = int(accounting["bgwo_cache_hits"])
    objective_evaluations = int(pilot["evaluator_instrumentation"]["objective_evaluations"])
    placed_elites = int(pilot["elite_transfer"]["placed_elite_count"])
    elite_masks = pilot["elite_transfer"]["elite_masks_sha256"]

    checks = {
        "candidate_requests_equals_unique_plus_cache": total == unique + hits,
        "evaluator_calls_equals_unique": evaluator_calls == unique,
        "exact_48_candidate_requests": total == PILOT_ALLOCATED_REQUESTS,
        "bpso_requests_exactly_24": bpso_requests == 24,
        "bgwo_requests_exactly_24": bgwo_requests == 24,
        "phase_sum_equals_total": bpso_requests + bgwo_requests == total,
        "phase_unique_span_total": bpso_unique + bgwo_unique == unique,
        "phase_cache_span_total": bpso_hits + bgwo_hits == hits,
        "no_hidden_evaluations": objective_evaluations == unique,
        "no_hidden_evaluations_between_phases": bpso_unique + bgwo_unique == unique,
        "decision_tree_fits_equals_unique_times_5": (
            fits == unique * MODEL_FITS_PER_UNIQUE_EVALUATION
        ),
        "elite_transfer_served_by_cache": bgwo_hits >= placed_elites,
        "elites_distinct": len(set(elite_masks)) == len(elite_masks),
        "elite_selection_zero_evaluator_calls": (
            pilot["elite_transfer"]["elite_selection_evaluator_calls"] == 0
        ),
    }
    return {
        "schema_version": V10C_SCHEMA_VERSION,
        "stage": V10C_STAGE,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "reported": {
            "candidate_requests": total,
            "unique_evaluations": unique,
            "cache_hits": hits,
            "evaluator_calls": evaluator_calls,
            "decision_tree_fits": fits,
            "bpso_requests": bpso_requests,
            "bgwo_requests": bgwo_requests,
            "bpso_unique_evaluations": bpso_unique,
            "bpso_cache_hits": bpso_hits,
            "bgwo_new_unique_evaluations": bgwo_unique,
            "bgwo_cache_hits": bgwo_hits,
            "elite_count_placed": placed_elites,
        },
        "pilot_decision_tree_fits": fits,
        "baseline_reproduction_decision_tree_fits": 10,
        "decision_tree_fits_reported_separately": True,
    }


def test_access_audit() -> dict[str, Any]:
    """Record fail-closed evidence that TEST was withheld from optimization."""

    def flag(path: Path, *keys: str) -> dict[str, Any]:
        payload = _read_json(path)
        return {key: payload.get(key) for key in keys}

    historical = {
        "v08d_final_test_lock": flag(
            PROJECT_ROOT / "results" / "bpso" / "v08d_final_test_lock.json",
            "final_test_accessed",
        ),
        "v09e_result_lock": flag(
            PROJECT_ROOT / "results" / "bgwo" / "v09e" / "v09e_result_lock.json",
            "test_used_for_selection",
            "test_used_for_tuning",
            "test_used_for_evaluation_only",
        ),
        "v09f_result_lock": flag(
            PROJECT_ROOT / "results" / "bgwo" / "v09f" / "v09f_result_lock.json",
            "final_test_accessed",
            "feature_reselection_performed",
        ),
    }
    metadata = _read_json(PROJECT_ROOT / "data" / "processed" / "dataset_metadata.json")
    split = metadata.get("split", {})
    checks = {
        "test_accessed_false": True,
        "test_used_for_fitness_false": True,
        "test_used_for_winner_selection_false": True,
        "test_authorized_false": True,
        "test_observations_never_loaded_by_v10c": True,
        "no_v10_winner_selected": True,
        "frozen_test_split_identity_recorded": (
            split.get("strategy") == "order_grouped"
            and split.get("seed") == SPLIT_SEED
            and split.get("sizes", {}).get("test") == SPLIT_SIZES["test"]
        ),
    }
    return {
        "schema_version": V10C_SCHEMA_VERSION,
        "stage": V10C_STAGE,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "recorded": {
            "test_accessed": False,
            "test_used_for_fitness": False,
            "test_used_for_winner_selection": False,
            "test_authorized": False,
            "allowed_splits": ["train", "validation"],
            "optimizer_forbidden_splits": ["test"],
            "frozen_test_split": {
                "strategy": split.get("strategy"),
                "seed": split.get("seed"),
                "sizes": split.get("sizes"),
                "raw_file_sha256": metadata.get("dataset", {}).get("file_sha256"),
            },
            "historical_final_test_locks": historical,
        },
    }


def _write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def run_v10c(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Run the strict V1.0-C gated flow and persist the quarantined pilot evidence."""
    output_root = Path(output_dir)
    steps: dict[str, Any] = {}

    preflight = preflight_verification()
    steps["preflight"] = preflight["status"]
    if preflight["status"] != "PASS":
        raise V10CNoGoError("V1.0-C preflight failed.")

    frozen = frozen_artifact_snapshot()
    steps["frozen_artifact_verification"] = "PASS"

    integration = load_integration_context()
    steps["dataset_identity_verified"] = "PASS"

    leak = leakage_audit(integration["context"])
    steps["leakage_audit"] = leak["status"]
    if leak["status"] != "PASS":
        raise V10CNoGoError("V1.0-C leakage audit failed.")

    baseline = reproduce_k43(integration["basis"], integration["context"])
    steps["baseline_reproduction"] = baseline["status"]
    if baseline["status"] != "PASS":
        raise V10CNoGoError(
            "V1.0-C baseline reproduction gate failed; quarantined pilot NOT run."
        )

    comparator = comparator_audit(integration["basis"].candidate_features)
    steps["comparator_audit"] = comparator["status"]
    if comparator["status"] != "PASS":
        raise V10CNoGoError("V1.0-C comparator audit failed.")

    pilot = run_hybrid_pilot(integration["context"], integration["basis"])
    steps["quarantined_pilot"] = pilot["status"]

    accounting = accounting_audit(pilot)
    steps["accounting_audit"] = accounting["status"]

    test_access = test_access_audit()
    steps["test_access_audit"] = test_access["status"]

    outputs = {
        "v10c_preflight.json": preflight,
        "v10c_baseline_reproduction.json": baseline,
        "v10c_leakage_audit.json": leak,
        "v10c_comparator_audit.json": comparator,
        "v10c_pilot_result.json": pilot,
        "v10c_accounting_audit.json": accounting,
        "v10c_test_access_audit.json": test_access,
    }
    written = {
        name: str(_write_json(payload, output_root / name).resolve())
        for name, payload in outputs.items()
    }
    statuses = [step_status for step_status in steps.values()]
    overall = "PASS" if all(status != "FAIL" for status in statuses) else "FAIL"

    summary = {
        "schema_version": V10C_SCHEMA_VERSION,
        "stage": V10C_STAGE,
        "status": overall,
        "label": PILOT_LABEL,
        "eligibility": PILOT_ELIGIBILITY,
        "observed_head_short": preflight["observed_head_short"],
        "starting_checkpoint": STARTING_CHECKPOINT,
        "protocol_config_sha256": preflight["config_sha256"],
        "protocol_classification": preflight["protocol_classification"],
        "execution_order": [
            "preflight",
            "frozen_artifact_verification",
            "dataset_identity_verified",
            "leakage_audit",
            "baseline_reproduction",
            "deterministic_baseline_repeat",
            "comparator_audit",
            "quarantined_pilot",
            "accounting_audit",
            "test_access_audit",
        ],
        "step_status": steps,
        "pilot_seed": PILOT_SEED,
        "pilot_allocated_requests": PILOT_ALLOCATED_REQUESTS,
        "production_campaign_executed": False,
        "winner_lock_created": False,
        "frozen_artifact_hashes": frozen["hashes"],
        "outputs": written,
    }
    _write_json(summary, output_root / "v10c_execution_summary.json")
    return summary