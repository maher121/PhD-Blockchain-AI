"""V0.8-B leakage-safe real fitness preflight and quarantined pilot.

The pipeline loads only the frozen training and development-validation protocol.
It reproduces K43 before running one small, non-selectable engineering pilot. It
does not expose a final-test loader or execute the approved full-search budget.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.optimization.bpso import BPSOConfig, BinaryParticleSwarmOptimizer
from src.optimization.feature_fitness import (
    BaselineMetrics,
    EXPECTED_MODEL_PARAMETERS,
    EXPECTED_SEEDS,
    EXPECTED_THRESHOLD,
    FeatureFitnessEvaluation,
    FeatureFitnessEvaluator,
    FitnessContext,
    audit_fitness_context,
    feature_fitness_is_better,
)
from src.pipeline_v06 import (
    V06_DECISION_TREE_PARAMETERS,
    V06_MODEL_NAME,
    V06_PREDICTION_THRESHOLD,
    audit_development_data,
    load_v06_development_data,
    load_v06_protocol_config,
)
from src.pipeline_v06e import verify_validation_lock
from src.pipeline_v08a import load_v08a_protocol
from src.security.experiment_data import DevelopmentExperimentData, fingerprint_feature_names


V08B_STAGE = "V0.8-B"
V08B_SCHEMA_VERSION = "v0.8-b-real-fitness-preflight-1"
PREFLIGHT_LABEL = "REAL_FITNESS_PREFLIGHT"
PILOT_LABEL = "QUARANTINED_PILOT"
PILOT_ELIGIBILITY = "NOT_ELIGIBLE_FOR_FINAL_SELECTION"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_RESULTS_DIR = PROJECT_ROOT / "results" / "feature_selection"
GREEN_RESULTS_DIR = PROJECT_ROOT / "results" / "green_evaluation"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "bpso"
VALIDATION_LOCK_PATH = FEATURE_RESULTS_DIR / "validation_lock.json"
VALIDATION_SUMMARY_PATH = FEATURE_RESULTS_DIR / "validation_summary.csv"
CORE_RUNS_PATH = FEATURE_RESULTS_DIR / "core_runs.csv"
RUN_METADATA_PATH = FEATURE_RESULTS_DIR / "run_metadata.json"
DATASET_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_metadata.json"
FEATURE_CONFIG_PATH = PROJECT_ROOT / "config" / "feature_selection.yaml"
BASELINE_REPRODUCTION_ABS_TOLERANCE = 1e-12
FULL_SEARCH_FIT_UPPER_BOUND = 6_000
FULL_SEARCH_REQUEST_UPPER_BOUND = 1_200
PILOT_OPTIMIZER_SEED = 1042

V08A_TRACKED_PATHS = (
    "config/bpso.yaml",
    "src/optimization/__init__.py",
    "src/optimization/bpso.py",
    "src/pipeline_v08a.py",
    "tests/test_bpso.py",
    "tests/test_pipeline_v08a.py",
)
IMMUTABLE_PATHS = (
    VALIDATION_LOCK_PATH,
    FEATURE_RESULTS_DIR / "validation_lock.sha256",
    FEATURE_RESULTS_DIR / "leakage_audit.json",
    CORE_RUNS_PATH,
    VALIDATION_SUMMARY_PATH,
    RUN_METADATA_PATH,
    FEATURE_CONFIG_PATH,
    DATASET_METADATA_PATH,
    GREEN_RESULTS_DIR / "energy_capability.json",
    GREEN_RESULTS_DIR / "v07d_artifact_hashes.json",
    GREEN_RESULTS_DIR / "v07e_artifact_hashes.json",
    GREEN_RESULTS_DIR / "v07d_governance_audit.json",
    GREEN_RESULTS_DIR / "v07e_governance_audit.json",
    *(PROJECT_ROOT / path for path in V08A_TRACKED_PATHS),
    DEFAULT_OUTPUT_DIR / "v08a_protocol_validation.json",
)


class V08BError(RuntimeError):
    """Base error for a fail-closed V0.8-B execution."""


class V08BPrecheckError(V08BError):
    """Raised before pilot execution when a mandatory gate fails."""


@dataclass(frozen=True)
class FrozenBasis:
    candidate_features: tuple[str, ...]
    candidate_manifest_sha256: str
    k42_features: tuple[str, ...]
    k42_mask: np.ndarray
    k42_features_sha256: str
    k42_mask_sha256: str
    k11_seed_specific: bool
    k11_feature_hashes: Mapping[int, str]
    baseline: BaselineMetrics
    baseline_seed_metrics: Mapping[int, Mapping[str, float]]
    dataset_metadata: Mapping[str, Any]
    lock: Mapping[str, Any]
    source_hashes: Mapping[str, str]


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_immutable_paths(paths: Sequence[Path] = IMMUTABLE_PATHS) -> dict[str, str]:
    """Hash the frozen V0.6, V0.7, and V0.8-A evidence used by this stage."""
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise V08BPrecheckError(f"Missing immutable prerequisite artifacts: {missing}")
    return {str(path.resolve()): sha256_file(path) for path in paths}


def verify_v08a_committed() -> dict[str, Any]:
    """Require all V0.8-A source/config/tests to be tracked and clean at HEAD."""
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *V08A_TRACKED_PATHS],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *V08A_TRACKED_PATHS],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if tracked.returncode != 0 or unchanged.returncode != 0:
        raise V08BPrecheckError("V0.8-A must be tracked, committed, and unmodified.")
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *V08A_TRACKED_PATHS],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    validation = _read_json(DEFAULT_OUTPUT_DIR / "v08a_protocol_validation.json")
    if (
        not commit
        or validation.get("stage") != "V0.8-A"
        or validation.get("status") != "PASS"
        or validation.get("validation_kind") != "SYNTHETIC_PROTOCOL_VALIDATION"
        or validation.get("scientific_experiment") is not False
        or validation.get("protocol_config_sha256")
        != sha256_file(PROJECT_ROOT / "config" / "bpso.yaml")
    ):
        raise V08BPrecheckError("V0.8-A validation evidence is invalid.")
    return {
        "status": "PASS",
        "commit": commit,
        "tracked_paths": list(V08A_TRACKED_PATHS),
        "working_tree_changes_for_v08a": False,
        "validation_artifact_sha256": sha256_file(
            DEFAULT_OUTPUT_DIR / "v08a_protocol_validation.json"
        ),
    }


def load_frozen_basis() -> FrozenBasis:
    """Verify and load only V0.6 development-validation evidence."""
    verification = verify_validation_lock(
        VALIDATION_LOCK_PATH,
        source_dir=FEATURE_RESULTS_DIR,
        verify_sources=True,
    )
    if verification.get("status") != "PASS" or verification.get("test_accessed") is not False:
        raise V08BPrecheckError("V0.6 validation lock verification failed.")
    lock = _read_json(VALIDATION_LOCK_PATH)
    run_metadata = _read_json(RUN_METADATA_PATH)
    dataset_metadata = _read_json(DATASET_METADATA_PATH)
    summary = pd.read_csv(VALIDATION_SUMMARY_PATH)
    core = pd.read_csv(CORE_RUNS_PATH)
    candidate = lock.get("candidate_manifest", {})
    candidates = tuple(candidate.get("features", ()))
    candidate_hash = str(candidate.get("sha256", ""))
    if (
        lock.get("scope") != "stored_development_validation_artifacts_only"
        or lock.get("test_accessed") is not False
        or run_metadata.get("test_accessed") is not False
        or len(candidates) != 43
        or candidate.get("feature_count") != 43
        or fingerprint_feature_names(candidates) != candidate_hash
        or candidate_hash != lock.get("candidate_manifest_hash")
        or run_metadata.get("output_sha256", {}).get("validation_summary.csv")
        != sha256_file(VALIDATION_SUMMARY_PATH)
        or run_metadata.get("output_sha256", {}).get("core_runs.csv")
        != sha256_file(CORE_RUNS_PATH)
        or run_metadata.get("processed_dataset_metadata_sha256")
        != sha256_file(DATASET_METADATA_PATH)
    ):
        raise V08BPrecheckError("Frozen V0.6 development evidence is inconsistent.")
    split = dataset_metadata.get("split", {})
    preprocessing = dataset_metadata.get("preprocessing", {})
    if (
        split.get("seed") != 42
        or split.get("strategy") != "order_grouped"
        or split.get("sizes") != {"train": 28000, "validation": 6000, "test": 6000}
        or preprocessing.get("fitted_on_rows") != 28000
        or tuple(preprocessing.get("ml_feature_columns", ())) != candidates
        or dataset_metadata.get("features", {}).get("cybersecurity_labels") is not False
    ):
        raise V08BPrecheckError("Frozen split or preprocessing metadata changed.")
    protocol = load_v06_protocol_config(FEATURE_CONFIG_PATH)
    if (
        protocol.configured_seeds != EXPECTED_SEEDS
        or protocol.attack_type != "mixed"
        or protocol.attack_rate != 0.05
        or protocol.attack_severity != "MEDIUM"
        or protocol.model_name != V06_MODEL_NAME
        or dict(protocol.model_parameters) != V06_DECISION_TREE_PARAMETERS
        or protocol.prediction_threshold != V06_PREDICTION_THRESHOLD
    ):
        raise V08BPrecheckError("Frozen V0.6 model, seed, or attack protocol changed.")
    if (
        lock.get("model_configuration")
        != {"model_id": "decision_tree", "parameters": dict(EXPECTED_MODEL_PARAMETERS)}
        or lock.get("threshold", {}).get("prediction_threshold") != EXPECTED_THRESHOLD
        or tuple(lock.get("seeds", ())) != EXPECTED_SEEDS
        or lock.get("attack_configuration", {}).get("attack_type") != "mixed"
        or lock.get("attack_configuration", {}).get("attack_rate") != 0.05
        or lock.get("attack_configuration", {}).get("attack_severity") != "MEDIUM"
    ):
        raise V08BPrecheckError("Validation lock protocol fields do not match V0.8-B.")

    roles = lock.get("roles", {})
    k42_role = roles.get("best_unsupervised", {})
    if (
        k42_role.get("configuration_id") != "pairwise_correlation_filter_natural"
        or k42_role.get("locked_configuration_ref")
        != "pairwise_correlation_filter_natural"
        or k42_role.get("actual_feature_count") != 42
    ):
        raise V08BPrecheckError("Frozen K42 semantic role is invalid.")
    k42_by_seed = k42_role.get("selected_features", {})
    if set(k42_by_seed) != {str(seed) for seed in EXPECTED_SEEDS}:
        raise V08BPrecheckError("K42 binding does not contain exact seeds 42-46.")
    k42_sets = {tuple(k42_by_seed[str(seed)]) for seed in EXPECTED_SEEDS}
    if len(k42_sets) != 1:
        raise V08BPrecheckError("K42 must remain deterministic across seeds.")
    k42_features = next(iter(k42_sets))
    k42_hash = fingerprint_feature_names(k42_features)
    recorded_k42_hashes = k42_role.get("selected_feature_fingerprints", {})
    if (
        len(k42_features) != 42
        or not all(recorded_k42_hashes.get(str(seed)) == k42_hash for seed in EXPECTED_SEEDS)
        or any(feature not in candidates for feature in k42_features)
        or tuple(feature for feature in candidates if feature in k42_features) != k42_features
    ):
        raise V08BPrecheckError("K42 selected-feature order or fingerprint is invalid.")
    k42_mask = np.asarray([feature in k42_features for feature in candidates], dtype=np.uint8)
    if int(k42_mask.sum()) != 42:
        raise V08BPrecheckError("K42 mask binding did not retain exactly 42 features.")

    k11_role = roles.get("best_supervised", {})
    k11_by_seed = k11_role.get("selected_feature_fingerprints", {})
    if (
        k11_role.get("configuration_id") != "mutual_information_select_k_best_k11"
        or k11_role.get("actual_feature_count") != 11
        or set(k11_by_seed) != {str(seed) for seed in EXPECTED_SEEDS}
    ):
        raise V08BPrecheckError("Frozen K11 semantic role is invalid.")
    k11_hashes = {seed: str(k11_by_seed[str(seed)]) for seed in EXPECTED_SEEDS}

    baseline_role = roles.get("full_baseline", {})
    baseline = BaselineMetrics(
        average_precision=float(baseline_role.get("metrics", {}).get("average_precision")),
        f1=float(baseline_role.get("metrics", {}).get("f1")),
        recall=float(baseline_role.get("metrics", {}).get("recall")),
    )
    baseline_summary = summary.loc[summary["configuration_id"].eq("none_natural")]
    baseline_core = core.loc[core["configuration_id"].eq("none_natural")].sort_values("seed")
    if len(baseline_summary) != 1 or len(baseline_core) != 5:
        raise V08BPrecheckError("Frozen K43 validation evidence is incomplete.")
    summary_row = baseline_summary.iloc[0]
    if any(
        not math.isclose(
            getattr(baseline, metric),
            float(summary_row[f"{metric}_mean"]),
            rel_tol=0.0,
            abs_tol=BASELINE_REPRODUCTION_ABS_TOLERANCE,
        )
        for metric in ("average_precision", "f1", "recall")
    ):
        raise V08BPrecheckError("K43 baseline metrics disagree across frozen artifacts.")
    per_seed = {
        int(row["seed"]): {
            metric: float(row[metric])
            for metric in ("average_precision", "f1", "recall", "precision", "roc_auc")
        }
        for row in baseline_core.to_dict(orient="records")
    }
    source_hashes = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in (
            VALIDATION_LOCK_PATH,
            VALIDATION_SUMMARY_PATH,
            CORE_RUNS_PATH,
            RUN_METADATA_PATH,
            DATASET_METADATA_PATH,
            FEATURE_CONFIG_PATH,
        )
    }
    return FrozenBasis(
        candidate_features=candidates,
        candidate_manifest_sha256=candidate_hash,
        k42_features=k42_features,
        k42_mask=k42_mask,
        k42_features_sha256=k42_hash,
        k42_mask_sha256=hashlib.sha256(k42_mask.tobytes()).hexdigest(),
        k11_seed_specific=len(set(k11_hashes.values())) > 1,
        k11_feature_hashes=k11_hashes,
        baseline=baseline,
        baseline_seed_metrics=per_seed,
        dataset_metadata=dataset_metadata,
        lock=lock,
        source_hashes=source_hashes,
    )


def load_frozen_development_workloads(basis: FrozenBasis) -> tuple[DevelopmentExperimentData, ...]:
    """Load and verify only the frozen train/validation manifestations."""
    core = pd.read_csv(CORE_RUNS_PATH)
    baseline_rows = core.loc[core["configuration_id"].eq("none_natural")].set_index("seed")
    workloads: list[DevelopmentExperimentData] = []
    for seed in EXPECTED_SEEDS:
        config = load_v06_protocol_config(FEATURE_CONFIG_PATH, seed=seed)
        data = load_v06_development_data(config)
        inherited_audit = audit_development_data(data, config)
        if inherited_audit.status != "PASS":
            raise V08BPrecheckError(f"Inherited V0.6 development audit failed for seed {seed}.")
        row = baseline_rows.loc[seed]
        fingerprints_match = (
            data.candidate_features == basis.candidate_features
            and data.train.row_ids_sha256 == row["training_row_ids_sha256"]
            and data.train.clean_features_sha256 == row["training_clean_features_sha256"]
            and data.train.attacked_features_sha256 == row["training_attacked_features_sha256"]
            and data.train.labels_sha256 == row["training_labels_sha256"]
            and data.validation.row_ids_sha256 == row["evaluation_rows_sha256"]
            and data.validation.attacked_features_sha256 == row["evaluation_features_sha256"]
            and data.validation.labels_sha256 == row["evaluation_labels_sha256"]
        )
        if not fingerprints_match:
            raise V08BPrecheckError(f"Frozen development workload mismatch for seed {seed}.")
        workloads.append(data)
    return tuple(workloads)


def build_fitness_context(
    basis: FrozenBasis, workloads: tuple[DevelopmentExperimentData, ...]
) -> FitnessContext:
    return FitnessContext(
        candidate_features=basis.candidate_features,
        candidate_manifest_sha256=basis.candidate_manifest_sha256,
        workloads=workloads,
        seeds=EXPECTED_SEEDS,
        baseline=basis.baseline,
        dataset_split_seed=42,
        expected_train_rows=28000,
        expected_validation_rows=6000,
        model_name=V06_MODEL_NAME,
        model_parameters=V06_DECISION_TREE_PARAMETERS,
        prediction_threshold=V06_PREDICTION_THRESHOLD,
        attack_type="mixed",
        attack_rate=0.05,
        attack_severity="MEDIUM",
        test_accessed=False,
    )


def reproduce_k43(
    context: FitnessContext,
    basis: FrozenBasis,
) -> tuple[dict[str, Any], FeatureFitnessEvaluation]:
    """Evaluate K43 twice and require exact frozen deterministic reproduction."""
    evaluator = FeatureFitnessEvaluator(context)
    mask = np.ones(43, dtype=np.uint8)
    first = evaluator(mask)
    second = evaluator(mask)
    primary = ("average_precision", "f1", "recall")
    aggregate_differences = {
        metric: first.mean_metrics[metric] - getattr(basis.baseline, metric)
        for metric in primary
    }
    aggregate_match = all(
        math.isclose(
            first.mean_metrics[metric],
            getattr(basis.baseline, metric),
            rel_tol=0.0,
            abs_tol=BASELINE_REPRODUCTION_ABS_TOLERANCE,
        )
        for metric in primary
    )
    seed_match = all(
        math.isclose(
            float(record.metrics[metric]),
            basis.baseline_seed_metrics[record.seed][metric],
            rel_tol=0.0,
            abs_tol=BASELINE_REPRODUCTION_ABS_TOLERANCE,
        )
        for record in first.per_seed
        for metric in ("average_precision", "f1", "recall", "precision", "roc_auc")
    )
    deterministic = first.to_dict() == second.to_dict()
    checks = {
        "exact_five_seeds": tuple(record.seed for record in first.per_seed) == EXPECTED_SEEDS,
        "exact_43_features": first.selected_feature_count == 43,
        "frozen_model_parameters": dict(context.model_parameters)
        == dict(EXPECTED_MODEL_PARAMETERS),
        "threshold_is_point_five": context.prediction_threshold == EXPECTED_THRESHOLD,
        "aggregate_primary_metrics_match": aggregate_match,
        "per_seed_metrics_match": seed_match,
        "deterministic_repeat_identical": deterministic,
        "test_not_accessed": context.test_accessed is False,
        "fit_count_is_ten_for_two_checks": evaluator.decision_tree_fit_count == 10,
    }
    record = {
        "schema_version": V08B_SCHEMA_VERSION,
        "stage": V08B_STAGE,
        "label": PREFLIGHT_LABEL,
        "status": "PASS" if all(checks.values()) else "PRECHECK_FAIL",
        "checks": checks,
        "tolerance": {
            "kind": "absolute",
            "value": BASELINE_REPRODUCTION_ABS_TOLERANCE,
            "justification": "deterministic replay under frozen data, code, model, and seeds",
        },
        "frozen_metrics": basis.baseline.to_dict(),
        "reproduced_metrics": {metric: first.mean_metrics[metric] for metric in primary},
        "differences": aggregate_differences,
        "deterministic_repeat": deterministic,
        "preflight_objective_evaluations": 2,
        "preflight_decision_tree_fits": evaluator.decision_tree_fit_count,
        "test_accessed": False,
        "pilot_executed": False,
        "full_search_executed": False,
    }
    return record, first


def run_quarantined_pilot(
    context: FitnessContext,
    basis: FrozenBasis,
) -> dict[str, Any]:
    """Run the fixed 4-particle, 3-generation non-selectable engineering pilot."""
    full_protocol = load_v08a_protocol().config
    pilot_config: BPSOConfig = replace(
        full_protocol,
        particle_count=4,
        evaluated_generations=3,
        random_cardinalities=(4, 11),
    )
    evaluator = FeatureFitnessEvaluator(context)
    optimizer = BinaryParticleSwarmOptimizer(
        pilot_config, evaluator, feature_fitness_is_better
    )
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = optimizer.optimize(PILOT_OPTIMIZER_SEED, k42_mask=basis.k42_mask)
    cpu_time = time.process_time() - cpu_start
    wall_time = time.perf_counter() - wall_start
    instrumentation = evaluator.instrumentation()
    fit_accounting = evaluator.decision_tree_fit_count == result.unique_evaluations * 5
    if (
        result.total_fitness_requests > 12
        or not fit_accounting
        or result.initial_population.sum(axis=1).astype(int).tolist() != [43, 42, 4, 11]
    ):
        raise V08BError("Quarantined pilot accounting or initialization failed.")
    history = [
        {
            "generation_index": row.generation_index,
            "best_mask_sha256": hashlib.sha256(row.best_mask.tobytes()).hexdigest(),
            "best_selected_feature_count": row.best_selected_feature_count,
            "best_feasible": row.best_evaluation.feasible,
            "cumulative_unique_evaluations": row.cumulative_unique_evaluations,
            "cumulative_cache_hits": row.cumulative_cache_hits,
            "population_diversity": row.population_diversity,
            "repair_count": row.repair_count,
            "inertia": row.inertia,
        }
        for row in result.convergence_history
    ]
    pilot = {
        "schema_version": V08B_SCHEMA_VERSION,
        "stage": V08B_STAGE,
        "label": PILOT_LABEL,
        "eligibility": PILOT_ELIGIBILITY,
        "status": "PASS",
        "purpose": "engineering validation of real evaluator connectivity only",
        "optimizer_seed": PILOT_OPTIMIZER_SEED,
        "configuration": pilot_config.to_dict(),
        "full_search_configuration_modified": False,
        "initial_cardinalities": [43, 42, 4, 11],
        "initial_population_sha256": _json_sha256(result.initial_population.tolist()),
        "real_k42_anchor_mask_sha256": basis.k42_mask_sha256,
        "fitness_requests": result.total_fitness_requests,
        "unique_masks": result.unique_evaluations,
        "cache_hits": result.cache_hits,
        "actual_decision_tree_fits": evaluator.decision_tree_fit_count,
        "fit_accounting_valid": fit_accounting,
        "wall_time_sec": wall_time,
        "process_cpu_time_sec": cpu_time,
        "evaluation_timing": instrumentation,
        "stop_reason": result.stop_reason,
        "evaluated_generations": result.evaluated_generation_count,
        "repair_count": result.repair_count,
        "convergence_history": history,
        "quarantined_candidate": {
            "mask_sha256": result.best_evaluation.mask_sha256,
            "selected_feature_count": result.best_evaluation.selected_feature_count,
            "feasible": result.best_evaluation.feasible,
            "feature_names_published": False,
        },
        "final_selection_performed": False,
        "test_accessed": False,
        "scientific_experiment": False,
        "scientific_superiority_claim_supported": False,
        "resource_benchmark": False,
        "direct_energy_measured": False,
        "rerun_count": 0,
    }
    return pilot


def build_budget_projection(pilot: Mapping[str, Any]) -> dict[str, Any]:
    timing = pilot["evaluation_timing"]
    mean_time = float(timing["mean_evaluation_wall_time_sec"])
    max_time = float(timing["max_evaluation_wall_time_sec"])
    cache_hits = int(pilot["cache_hits"])
    requests = int(pilot["fitness_requests"])
    naive_expected = mean_time * FULL_SEARCH_REQUEST_UPPER_BOUND
    observed_max_extrapolation = max_time * FULL_SEARCH_REQUEST_UPPER_BOUND
    return {
        "schema_version": V08B_SCHEMA_VERSION,
        "stage": V08B_STAGE,
        "label": "FULL_SEARCH_BUDGET_PROJECTION_ONLY",
        "full_search_executed": False,
        "approved_full_search": {
            "particles": 12,
            "evaluated_generations": 20,
            "optimizer_runs": 5,
            "maximum_fitness_requests": FULL_SEARCH_REQUEST_UPPER_BOUND,
            "maximum_decision_tree_fits": FULL_SEARCH_FIT_UPPER_BOUND,
        },
        "pilot_mean_unique_evaluation_wall_time_sec": mean_time,
        "pilot_max_unique_evaluation_wall_time_sec": max_time,
        "naive_mean_cost_projection_sec": naive_expected,
        "observed_max_cost_projection_sec": observed_max_extrapolation,
        "cache_adjusted_projection_sec": None,
        "cache_adjusted_projection_defensible": False,
        "cache_projection_reason": (
            "three-generation pilot cache behavior is too small to extrapolate"
            if cache_hits or requests
            else "no pilot request evidence"
        ),
        "pilot_cache_hit_rate": float(cache_hits / requests) if requests else 0.0,
        "runtime_review_threshold_sec": 7200.0,
        "runtime_review_required": observed_max_extrapolation > 7200.0,
        "approved_budget_changed": False,
        "planning_estimate_only": True,
        "direct_energy_claim": False,
    }


def run_v08b(
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    basis_loader: Callable[[], FrozenBasis] = load_frozen_basis,
    workload_loader: Callable[[FrozenBasis], tuple[DevelopmentExperimentData, ...]] = load_frozen_development_workloads,
) -> dict[str, Any]:
    """Execute all V0.8-B gates in order and fail closed before any pilot."""
    output = Path(output_dir)
    immutable_before = snapshot_immutable_paths()
    v08a_integrity = verify_v08a_committed()
    basis = basis_loader()
    workloads = workload_loader(basis)
    context = build_fitness_context(basis, workloads)
    leakage = audit_fitness_context(context)
    leakage_artifact = {
        **leakage.to_dict(),
        "label": "REAL_FITNESS_LEAKAGE_AUDIT",
        "ground_truth": "controlled experiment-generated is_attack only",
        "final_test_statement": "untouched by V0.8 search and selection",
    }
    _atomic_write_json(output / "v08b_leakage_audit.json", leakage_artifact)
    if leakage.status != "PASS":
        raise V08BPrecheckError("V0.8-B leakage audit failed; pilot prohibited.")
    preflight, _ = reproduce_k43(context, basis)
    preflight.update(
        {
            "integrity": v08a_integrity,
            "candidate_manifest": {
                "feature_count": len(basis.candidate_features),
                "sha256": basis.candidate_manifest_sha256,
            },
            "real_k42_binding": {
                "configuration_id": "pairwise_correlation_filter_natural",
                "feature_count": len(basis.k42_features),
                "selected_features_sha256": basis.k42_features_sha256,
                "mask_sha256": basis.k42_mask_sha256,
                "deterministic_across_seeds": True,
            },
            "k11_governance": {
                "configuration_id": "mutual_information_select_k_best_k11",
                "seed_specific": basis.k11_seed_specific,
                "feature_hashes_by_seed": dict(basis.k11_feature_hashes),
                "converted_to_universal_subset": False,
            },
            "source_hashes": dict(basis.source_hashes),
        }
    )
    if preflight["status"] != "PASS":
        preflight.update(
            {
                "pilot_executed": False,
                "full_search_executed": False,
                "final_selection_performed": False,
                "final_test_accessed": False,
                "readiness_for_v08c": "NO-GO",
            }
        )
        _atomic_write_json(output / "v08b_preflight.json", preflight)
        raise V08BPrecheckError("PRECHECK_FAIL: K43 reproduction did not match V0.6.")
    pilot = run_quarantined_pilot(context, basis)
    budget = build_budget_projection(pilot)
    _atomic_write_json(output / "v08b_pilot.json", pilot)
    _atomic_write_json(output / "v08b_budget_projection.json", budget)
    immutable_after = snapshot_immutable_paths()
    if immutable_after != immutable_before:
        raise V08BError("Frozen V0.6, V0.7, or V0.8-A artifacts changed during V0.8-B.")
    generated = {
        "v08b_leakage_audit.json": sha256_file(output / "v08b_leakage_audit.json"),
        "v08b_pilot.json": sha256_file(output / "v08b_pilot.json"),
        "v08b_budget_projection.json": sha256_file(output / "v08b_budget_projection.json"),
    }
    preflight.update(
        {
            "status": "PASS",
            "pilot_executed": True,
            "pilot_label": PILOT_LABEL,
            "pilot_eligibility": PILOT_ELIGIBILITY,
            "full_search_executed": False,
            "final_selection_performed": False,
            "final_test_accessed": False,
            "resource_benchmark_executed": False,
            "direct_energy_measured": False,
            "immutable_hashes_before": immutable_before,
            "immutable_hashes_after": immutable_after,
            "generated_artifact_hashes": generated,
            "readiness_for_v08c": "GO",
        }
    )
    _atomic_write_json(output / "v08b_preflight.json", preflight)
    _verify_generated_artifacts(output)
    return {
        "stage": V08B_STAGE,
        "status": "PASS",
        "pilot": PILOT_LABEL,
        "eligibility": PILOT_ELIGIBILITY,
        "full_search_executed": False,
        "final_selection_performed": False,
        "test_accessed": False,
        "readiness_for_v08c": "GO",
        "artifact_hashes": {
            **generated,
            "v08b_preflight.json": sha256_file(output / "v08b_preflight.json"),
        },
    }


def _verify_generated_artifacts(output: Path) -> None:
    expected = {
        "v08b_preflight.json": PREFLIGHT_LABEL,
        "v08b_leakage_audit.json": "REAL_FITNESS_LEAKAGE_AUDIT",
        "v08b_pilot.json": PILOT_LABEL,
        "v08b_budget_projection.json": "FULL_SEARCH_BUDGET_PROJECTION_ONLY",
    }
    for name, label in expected.items():
        payload = _read_json(output / name)
        if payload.get("stage") != V08B_STAGE or payload.get("label") != label:
            raise V08BError(f"Generated V0.8-B artifact failed verification: {name}")
    pilot = _read_json(output / "v08b_pilot.json")
    if (
        pilot.get("eligibility") != PILOT_ELIGIBILITY
        or pilot.get("final_selection_performed") is not False
        or pilot.get("test_accessed") is not False
    ):
        raise V08BError("Pilot quarantine metadata is invalid.")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V08BPrecheckError(f"Cannot read required JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise V08BPrecheckError(f"Required JSON artifact must be an object: {path}")
    return value


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def main() -> int:
    result = run_v08b()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
