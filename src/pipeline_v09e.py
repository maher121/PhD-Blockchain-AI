"""V0.9-E governed one-time final-test evaluation of the locked BGWO winner.

This stage loads the already-frozen V0.9-D BGWO winner lock (and the frozen
V0.8-C BPSO K10 lock), authorizes final-test access only for evaluation, fits
the frozen logistic-regression and decision-tree configurations on the training
split, scores the held-out test split, and reports paired statistics, feature
overlap, and preservation outcomes. It never runs an optimizer or feature
selector, never tunes a classifier or threshold, never reselects the winner, and
never accesses final test before the winner lock is verified and access is
explicitly authorized.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.lightweight.models import create_model
from src.pipeline_v08b import FrozenBasis, load_frozen_basis, sha256_file
import src.pipeline_v08d as v08d
import src.pipeline_v09d as v09d
import src.pipeline_v07b as v07b
from src.security.evaluation import detector_visible_mask, evaluate_detection
from src.security.experiment_data import fingerprint_feature_names
from src.security.ground_truth import assert_no_attack_metadata


V09E_STAGE = "V0.9-E"
V09E_SCHEMA_VERSION = "v0.9-e-governed-final-test-1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "bgwo" / "v09e"
BGWO_WINNER_LOCK_PATH = PROJECT_ROOT / "results" / "bgwo" / "v09d" / "v09d_winner_lock.json"
BPSO_WINNER_LOCK_PATH = PROJECT_ROOT / "results" / "bpso" / "v08c_winner_lock.json"

EXPECTED_HEAD_SHORT = "1007d3f"
EXPECTED_DATASET = "DataCo SMART Supply Chain"
EXPECTED_DIMENSIONS = 43
EXPECTED_TEST_ROWS = 6000
EXPECTED_TEST_SPLIT = {"train": 28000, "validation": 6000, "test": 6000}

MODEL_ATTACK_SEEDS = (42, 43, 44, 45, 46)
CLASSIFIERS = ("decision_tree", "logistic_regression")
CONFIGURATION_IDS = ("K43", "K42", "MI-K11", "BPSO-K10", "BGWO")
CANDIDATE_COMPARISON_IDS = ("BGWO", "BPSO-K10")
PRIMARY_METRICS = ("average_precision", "f1", "recall", "precision", "roc_auc")
PRESERVATION_METRICS = ("average_precision", "f1", "recall")
PRESERVATION_MARGINS = {"average_precision": 0.05, "f1": 0.05, "recall": 0.10}
T_CRITICAL_95_DF4 = 2.7764451051977987
EXPECTED_EVALUATION_COUNT = len(CONFIGURATION_IDS) * len(CLASSIFIERS) * len(MODEL_ATTACK_SEEDS)

EXPECTED_BGWO_WINNER = {
    "source_optimizer_seed": 2042,
    "selected_feature_count": 14,
    "mask_sha256": "7ebb823374255f4f10c737c62a2111604aa50193a8f0d91b8483f3864cac7ad6",
    "selected_features_sha256": "0d5219835f3dc96acb3c82f94d286813b834eceefc4156aa383234a842b9c359",
    "semantic_lock_sha256": "e228f4c619de7e2028e4089723c6ae066fe8c2ab9ed1d94d98e0439a63df2744",
}
EXPECTED_BGWO_FEATURES = (
    "order_item_quantity",
    "product_price",
    "order_item_discount",
    "order_item_discount_rate",
    "order_item_total",
    "days_schedule",
    "is_weekend",
    "item_count_per_order",
    "Market_LATAM",
    "Market_Pacific Asia",
    "Market_USCA",
    "Department Name_Outdoors",
    "Department Name_Discs Shop",
    "Department Name_Book Shop",
)
EXPECTED_BPSO_WINNER = {
    "source_optimizer_seed": 1042,
    "selected_feature_count": 10,
    "mask_sha256": "5da981b5b87db97338ecdde9ca8a8b87db3a62771d03dc6a4ad901f6548a3299",
    "selected_features_sha256": "5157d5bb6b17dc6c92b790671dac029e2b0d2b4454db9824870db1290a6c4121",
    "semantic_lock_sha256": "0f356ccab422774b128ab82f5da8881f24816902d4ccc6a7cecc9760c6202ca8",
}
EXPECTED_BPSO_FEATURES = (
    "order_item_quantity",
    "order_item_profit_ratio",
    "order_item_total",
    "hour",
    "Type_PAYMENT",
    "Market_LATAM",
    "Shipping Mode_Same Day",
    "Customer Segment_Corporate",
    "Department Name_Apparel",
    "Department Name_Health and Beauty",
)

V09E_TRACKED_PATHS = (
    "config/bgwo_v09.yaml",
    "docs/v09_bgwo_protocol.md",
    "src/optimization/bgwo.py",
    "src/pipeline_v09b.py",
    "src/pipeline_v09c.py",
    "src/pipeline_v09d.py",
    "results/bgwo/v09d/v09d_winner_lock.json",
    "results/bgwo/v09d/v09d_run_2042.json",
    "results/bgwo/v09d/v09d_run_2043.json",
    "results/bgwo/v09d/v09d_run_2044.json",
    "results/bgwo/v09d/v09d_run_2045.json",
    "results/bgwo/v09d/v09d_run_2046.json",
    "results/bgwo/v09d/v09d_stability.json",
    "results/bgwo/v09d/v09d_convergence.json",
    "results/bgwo/v09d/v09d_validation_comparison.json",
    "results/bgwo/v09d/v09d_test_access_audit.json",
    "results/bgwo/v09d/v09d_execution_summary.json",
    "results/bpso/v08c_winner_lock.json",
)
IMMUTABLE_PATHS = tuple(
    dict.fromkeys(
        (
            *v08d.IMMUTABLE_PATHS,
            *(PROJECT_ROOT / path for path in V09E_TRACKED_PATHS),
        )
    )
)

RAW_PATH = OUTPUT_DIR / "v09e_raw_final_test.json"
METRICS_CSV_PATH = OUTPUT_DIR / "v09e_test_metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "v09e_test_summary.json"
PAIRED_CSV_PATH = OUTPUT_DIR / "v09e_paired_statistics.csv"
OVERLAP_PATH = OUTPUT_DIR / "v09e_feature_overlap.json"
PRESERVATION_PATH = OUTPUT_DIR / "v09e_preservation_analysis.json"
AUDIT_PATH = OUTPUT_DIR / "v09e_test_access_audit.json"
EXECUTION_PATH = OUTPUT_DIR / "v09e_execution_summary.json"
RESULT_LOCK_PATH = OUTPUT_DIR / "v09e_result_lock.json"

METRIC_COLUMNS = (
    "average_precision",
    "pr_auc_trapezoidal",
    "f1",
    "recall",
    "precision",
    "roc_auc",
    "accuracy",
    "false_positive_rate",
    "false_negative_rate",
    "attack_prevalence",
    "true_positives",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "evaluated_records",
    "attacked_records",
    "visible_attack_count",
    "invisible_attack_count",
    "selected_feature_visibility_rate",
)


class V09EError(RuntimeError):
    """Raised when V0.9-E final-test governance or evidence fails closed."""


def current_head_short(root: Path = PROJECT_ROOT) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def snapshot_immutable_paths(paths: Sequence[Path] = IMMUTABLE_PATHS) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise V09EError(f"Missing frozen V0.6-V0.9-D prerequisite: {missing}")
    return {str(path.resolve()): sha256_file(path) for path in paths}


def load_verified_bgwo_winner_lock(path: Path = BGWO_WINNER_LOCK_PATH) -> dict[str, Any]:
    lock = _read_json(path)
    v09d.verify_winner_lock(lock)
    exact = all(lock.get(name) == value for name, value in EXPECTED_BGWO_WINNER.items())
    mask = np.asarray(lock.get("mask", ()), dtype=np.uint8)
    features = tuple(lock.get("ordered_selected_features", ()))
    if (
        not exact
        or lock.get("stage") != "V0.9-D"
        or lock.get("status") != "VALIDATION_LOCKED"
        or lock.get("eligible_for_v09e") is not True
        or lock.get("selection_scope") != "TRAIN_AND_DEVELOPMENT_VALIDATION_ONLY"
        or lock.get("optimizer") != "BGWO"
        or lock.get("final_test_accessed") is not False
        or lock.get("test_authorized") is not False
        or lock.get("test_used_for_winner_selection") is not False
        or lock.get("bpso_rerun") is not False
        or mask.shape != (EXPECTED_DIMENSIONS,)
        or int(mask.sum()) != EXPECTED_BGWO_WINNER["selected_feature_count"]
        or hashlib.sha256(mask.tobytes()).hexdigest() != EXPECTED_BGWO_WINNER["mask_sha256"]
        or features != EXPECTED_BGWO_FEATURES
        or fingerprint_feature_names(features)
        != EXPECTED_BGWO_WINNER["selected_features_sha256"]
    ):
        raise V09EError("V0.9-D BGWO winner lock does not match the authorized winner.")
    return lock


def load_verified_bpso_winner_lock(path: Path = BPSO_WINNER_LOCK_PATH) -> dict[str, Any]:
    lock = _read_json(path)
    v08d.verify_winner_lock(lock)
    exact = all(lock.get(name) == value for name, value in EXPECTED_BPSO_WINNER.items())
    features = tuple(lock.get("ordered_selected_features", ()))
    if (
        not exact
        or lock.get("stage") != "V0.8-C"
        or lock.get("eligible_for_v08d") is not True
        or features != EXPECTED_BPSO_FEATURES
    ):
        raise V09EError("V0.8-C BPSO winner lock does not match the authorized K10 winner.")
    return lock


def verify_v09e_preflight(
    *,
    expected_head: str | None = EXPECTED_HEAD_SHORT,
    basis: FrozenBasis | None = None,
    bgwo_lock: Mapping[str, Any] | None = None,
    bpso_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify every frozen prerequisite before any final-test access."""

    observed_head = current_head_short(PROJECT_ROOT)
    if expected_head and observed_head != expected_head:
        raise V09EError(
            f"V09E_NO_GO: expected HEAD {expected_head}, observed {observed_head}."
        )

    resolved_bgwo = dict(bgwo_lock) if bgwo_lock is not None else load_verified_bgwo_winner_lock()
    resolved_bpso = dict(bpso_lock) if bpso_lock is not None else load_verified_bpso_winner_lock()
    if bgwo_lock is not None:
        v09d.verify_winner_lock(resolved_bgwo)
    if bpso_lock is not None:
        v08d.verify_winner_lock(resolved_bpso)
    if resolved_bgwo.get("semantic_lock_sha256") != EXPECTED_BGWO_WINNER["semantic_lock_sha256"]:
        raise V09EError("V09E_NO_GO: BGWO winner semantic lock hash differs from governed value.")
    if resolved_bpso.get("semantic_lock_sha256") != EXPECTED_BPSO_WINNER["semantic_lock_sha256"]:
        raise V09EError("V09E_NO_GO: BPSO winner semantic lock hash differs from governed value.")

    resolved_basis = basis if basis is not None else load_frozen_basis()
    roles = resolved_basis.lock.get("roles", {})
    k11_by_seed = roles.get("best_supervised", {}).get("selected_features", {})
    baseline_metrics = roles.get("full_baseline", {}).get("metrics", {})
    preprocessing = resolved_basis.dataset_metadata.get("preprocessing", {})
    dataset_split = resolved_basis.dataset_metadata.get("split", {})

    checks = {
        "starting_checkpoint": observed_head == EXPECTED_HEAD_SHORT,
        "bgwo_winner_lock_verified": resolved_bgwo.get("status") == "VALIDATION_LOCKED"
        and resolved_bgwo.get("eligible_for_v09e") is True,
        "bgwo_winner_exact": resolved_bgwo.get("mask_sha256")
        == EXPECTED_BGWO_WINNER["mask_sha256"]
        and resolved_bgwo.get("selected_features_sha256")
        == EXPECTED_BGWO_WINNER["selected_features_sha256"],
        "bgwo_winner_validation_only_provenance": resolved_bgwo.get("selection_scope")
        == "TRAIN_AND_DEVELOPMENT_VALIDATION_ONLY"
        and resolved_bgwo.get("final_test_accessed") is False,
        "bpso_winner_lock_verified": resolved_bpso.get("status") == "VALIDATION_LOCKED",
        "bpso_winner_exact": resolved_bpso.get("mask_sha256")
        == EXPECTED_BPSO_WINNER["mask_sha256"]
        and resolved_bpso.get("selected_features_sha256")
        == EXPECTED_BPSO_WINNER["selected_features_sha256"],
        "v06_validation_lock_verified": resolved_basis.lock.get("test_accessed") is False,
        "mi_k11_governed_subset": int(
            roles.get("best_supervised", {}).get("actual_feature_count", -1)
        )
        == 11
        and set(k11_by_seed) == {str(seed) for seed in MODEL_ATTACK_SEEDS},
        "k42_governed_subset": len(resolved_basis.k42_features) == 42
        and int(resolved_basis.k42_mask.sum()) == 42,
        "k43_baseline": int(roles.get("full_baseline", {}).get("actual_feature_count", -1))
        == 43
        and set(baseline_metrics)
        >= {"average_precision", "f1", "precision", "recall", "roc_auc"},
        "dataset_identity": dataset_split.get("seed") == 42
        and dataset_split.get("strategy") == "order_grouped"
        and dataset_split.get("sizes") == EXPECTED_TEST_SPLIT,
        "feature_ordering": tuple(preprocessing.get("ml_feature_columns", ()))
        == tuple(resolved_basis.candidate_features)
        == tuple(resolved_basis.lock.get("candidate_manifest", {}).get("features", ())),
        "preprocessing_artifacts": preprocessing.get("fitted_on_rows") == 28000
        and resolved_basis.dataset_metadata.get("features", {}).get(
            "cybersecurity_labels"
        )
        is False,
        "test_not_accessed_by_preflight": True,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise V09EError(f"V0.9-E preflight failed: {failed}")

    frozen_snapshot = snapshot_immutable_paths()
    return {
        "schema_version": V09E_SCHEMA_VERSION,
        "stage": V09E_STAGE,
        "status": "PASS",
        "starting_head": observed_head,
        "bgwo_winner_lock_sha256": sha256_file(BGWO_WINNER_LOCK_PATH),
        "bgwo_winner_semantic_lock_sha256": resolved_bgwo["semantic_lock_sha256"],
        "bgwo_winner_mask_sha256": resolved_bgwo["mask_sha256"],
        "bgwo_winner_selected_features_sha256": resolved_bgwo["selected_features_sha256"],
        "bgwo_winner_selected_feature_count": int(resolved_bgwo["selected_feature_count"]),
        "bpso_winner_semantic_lock_sha256": resolved_bpso["semantic_lock_sha256"],
        "bpso_winner_mask_sha256": resolved_bpso["mask_sha256"],
        "bpso_winner_selected_features_sha256": resolved_bpso["selected_features_sha256"],
        "bpso_winner_selected_feature_count": int(resolved_bpso["selected_feature_count"]),
        "dataset_identity": {
            "name": EXPECTED_DATASET,
            "split_seed": dataset_split.get("seed"),
            "split_sizes": dict(dataset_split.get("sizes", {})),
            "test_rows": int(dataset_split.get("sizes", {}).get("test", 0)),
            "dimensions": EXPECTED_DIMENSIONS,
            "feature_manifest_sha256": resolved_basis.candidate_manifest_sha256,
        },
        "checks": checks,
        "frozen_immutable_hash_count": len(frozen_snapshot),
    }


def build_configuration_plan(
    basis: FrozenBasis,
    bgwo_lock: Mapping[str, Any],
    bpso_lock: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    candidates = basis.candidate_features
    k11_by_seed = basis.lock["roles"]["best_supervised"]["selected_features"]
    if set(k11_by_seed) != {str(seed) for seed in MODEL_ATTACK_SEEDS}:
        raise V09EError("Historical MI-K11 features do not cover exact seeds 42-46.")
    bpso_features = tuple(bpso_lock["ordered_selected_features"])
    bgwo_features = tuple(bgwo_lock["ordered_selected_features"])
    plans = (
        {
            "configuration_id": "K43",
            "source_configuration_id": "none_natural",
            "feature_count": 43,
            "selection_scope": "frozen V0.6 full baseline",
            "selection_semantics": "universal full candidate manifest",
            "source_semantic_lock_sha256": None,
            "features_by_seed": {str(seed): list(candidates) for seed in MODEL_ATTACK_SEEDS},
        },
        {
            "configuration_id": "K42",
            "source_configuration_id": "pairwise_correlation_filter_natural",
            "feature_count": 42,
            "selection_scope": "frozen V0.6 pairwise-correlation baseline",
            "selection_semantics": "universal deterministic training-derived subset",
            "source_semantic_lock_sha256": None,
            "features_by_seed": {
                str(seed): list(basis.k42_features) for seed in MODEL_ATTACK_SEEDS
            },
        },
        {
            "configuration_id": "MI-K11",
            "source_configuration_id": "mutual_information_select_k_best_k11",
            "feature_count": 11,
            "selection_scope": "frozen V0.6 historical MI rule",
            "selection_semantics": "seed-specific supervised training-derived subsets",
            "source_semantic_lock_sha256": None,
            "features_by_seed": {
                str(seed): list(k11_by_seed[str(seed)]) for seed in MODEL_ATTACK_SEEDS
            },
        },
        {
            "configuration_id": "BPSO-K10",
            "source_configuration_id": "v08c_locked_bpso_k10",
            "feature_count": 10,
            "selection_scope": "V0.8-C train and development-validation lock",
            "selection_semantics": "one universal locked BPSO subset across all seeds",
            "source_semantic_lock_sha256": bpso_lock["semantic_lock_sha256"],
            "features_by_seed": {
                str(seed): list(bpso_features) for seed in MODEL_ATTACK_SEEDS
            },
        },
        {
            "configuration_id": "BGWO",
            "source_configuration_id": "v09d_locked_bgwo_winner",
            "feature_count": 14,
            "selection_scope": "V0.9-D train and development-validation lock",
            "selection_semantics": "one universal locked BGWO subset across all seeds",
            "source_semantic_lock_sha256": bgwo_lock["semantic_lock_sha256"],
            "features_by_seed": {
                str(seed): list(bgwo_features) for seed in MODEL_ATTACK_SEEDS
            },
        },
    )
    if tuple(plan["configuration_id"] for plan in plans) != CONFIGURATION_IDS:
        raise V09EError(
            "Final-test configuration plan must contain exactly K43/K42/MI-K11/BPSO-K10/BGWO."
        )
    for plan in plans:
        if plan["feature_count"] != len(plan["features_by_seed"]["42"]):
            raise V09EError(f"Configuration feature count mismatch: {plan['configuration_id']}")
        for seed in MODEL_ATTACK_SEEDS:
            features = tuple(plan["features_by_seed"][str(seed)])
            if (
                len(features) != plan["feature_count"]
                or len(set(features)) != len(features)
                or tuple(feature for feature in candidates if feature in features) != features
            ):
                raise V09EError(
                    f"Configuration feature identity/order mismatch: "
                    f"{plan['configuration_id']} seed {seed}"
                )
    if len({tuple(k11_by_seed[str(seed)]) for seed in MODEL_ATTACK_SEEDS}) <= 1:
        raise V09EError("MI-K11 must preserve its historical seed-specific semantics.")
    if tuple(plans[3]["features_by_seed"]["42"]) != bpso_features:
        raise V09EError("BPSO-K10 plan does not match the locked BPSO subset.")
    if tuple(plans[4]["features_by_seed"]["42"]) != bgwo_features:
        raise V09EError("BGWO plan does not match the locked BGWO subset.")
    return plans


def classifier_plan() -> dict[str, dict[str, Any]]:
    plan = v08d.classifier_plan()
    expected = {
        "decision_tree": {
            "parameters": {
                "max_depth": 5,
                "min_samples_leaf": 20,
                "class_weight": "balanced",
            },
            "prediction_threshold": 0.5,
        },
        "logistic_regression": {
            "parameters": {
                "solver": "liblinear",
                "class_weight": "balanced",
                "max_iter": 500,
                "C": 1.0,
            },
            "prediction_threshold": 0.5,
        },
    }
    if plan != expected:
        raise V09EError("Frozen DT/LR classifier configuration changed.")
    return plan


def activate_test_access_audit(
    output_path: Path | str,
    preflight: Mapping[str, Any],
    bgwo_lock: Mapping[str, Any],
    bpso_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist authorization before any test workload is loaded."""
    path = Path(output_path)
    if path.with_name(RESULT_LOCK_PATH.name).exists():
        raise V09EError("V0.9-E result lock already exists; campaign rerun prohibited.")
    rerun_count = 0
    prior_events: list[dict[str, Any]] = []
    if path.exists():
        prior = _read_json(path)
        if prior.get("status") != "TEST_ACCESS_ACTIVATED":
            raise V09EError("Existing final-test audit is not a recoverable interrupted state.")
        rerun_count = int(prior.get("rerun_count", 0)) + 1
        prior_events = list(prior.get("events", ()))
    lock_verified_at = _utc_now()
    activated_at = _utc_now()
    audit = {
        "schema_version": V09E_SCHEMA_VERSION,
        "stage": V09E_STAGE,
        "status": "TEST_ACCESS_ACTIVATED",
        "starting_head": preflight.get("starting_head"),
        "winner_lock_verified_before_test_access": True,
        "winner_lock_verified_at_utc": lock_verified_at,
        "test_access_activated_at_utc": activated_at,
        "source_winner_stage": "V0.9-D",
        "source_winner_semantic_lock_sha256": bgwo_lock["semantic_lock_sha256"],
        "bgwo_winner_lock_sha256": preflight.get("bgwo_winner_lock_sha256"),
        "locked_mask_sha256": bgwo_lock["mask_sha256"],
        "locked_selected_features_sha256": bgwo_lock["selected_features_sha256"],
        "bpso_source_semantic_lock_sha256": bpso_lock["semantic_lock_sha256"],
        "test_used_for_evaluation_only": True,
        "test_used_for_selection": False,
        "test_used_for_tuning": False,
        "test_used_for_feature_selection": False,
        "checks": {
            "winner_lock_verified_before_test_access": True,
            "optimizer_invoked": False,
            "feature_selection_invoked": False,
            "winner_replacement_allowed": False,
            "hyperparameter_search_invoked": False,
            "threshold_tuning_invoked": False,
            "bgwo_rerun": False,
            "bpso_rerun": False,
            "test_driven_configuration_change": False,
            "test_driven_threshold_change": False,
            "test_driven_classifier_tuning": False,
            "final_test_used_only_for_evaluation": True,
        },
        "events": [
            *prior_events,
            {"event": "WINNER_LOCK_VERIFIED", "timestamp_utc": lock_verified_at},
            {"event": "TEST_ACCESS_ACTIVATED", "timestamp_utc": activated_at},
        ],
        "final_test_accessed": True,
        "final_test_opened": False,
        "campaign_completed": False,
        "rerun_count": rerun_count,
        "partial_metrics_used_for_configuration": False,
    }
    _atomic_write_json(path, audit)
    return audit


def assert_test_access_authorized(audit: Mapping[str, Any]) -> None:
    """Fail closed unless final-test access was explicitly authorized."""
    authorized = (
        audit.get("status") in {"TEST_ACCESS_ACTIVATED", "PASS"}
        and audit.get("winner_lock_verified_before_test_access") is True
        and audit.get("final_test_accessed") is True
        and audit.get("test_used_for_evaluation_only") is True
        and audit.get("test_used_for_selection") is False
        and audit.get("test_used_for_tuning") is False
        and audit.get("test_used_for_feature_selection") is False
    )
    if not authorized:
        raise V09EError("Final test is inaccessible without explicit evaluation-only authorization.")


def load_authorized_test_workloads(
    context: Any,
    audit: Mapping[str, Any],
    *,
    loader: Callable[[Any], Any] = v07b.prepare_verified_workloads,
) -> Any:
    """Load the frozen test workloads only after authorization is proven."""
    assert_test_access_authorized(audit)
    return loader(context)


def complete_test_access_audit(
    output_path: Path | str,
    audit: Mapping[str, Any],
    *,
    evaluation_count: int,
) -> dict[str, Any]:
    path = Path(output_path)
    completed = dict(audit)
    completed_at = _utc_now()
    completed.update(
        {
            "status": "PASS",
            "final_test_opened": True,
            "campaign_completed": True,
            "governed_evaluation_count": evaluation_count,
            "campaign_completed_at_utc": completed_at,
            "winner_modified_after_access": False,
            "events": [
                *audit["events"],
                {
                    "event": "FINAL_TEST_OPENED_FOR_EVALUATION",
                    "timestamp_utc": audit["test_access_activated_at_utc"],
                },
                {"event": "CAMPAIGN_COMPLETED", "timestamp_utc": completed_at},
            ],
        }
    )
    _atomic_write_json(path, completed)
    return completed


def evaluate_final_test_campaign(
    configuration_plans: Sequence[Mapping[str, Any]],
    classifiers: Mapping[str, Mapping[str, Any]],
    workloads: Any,
    *,
    model_factory: Callable[..., Any] = create_model,
    metric_evaluator: Callable[[pd.DataFrame, pd.DataFrame], Mapping[str, Any]] = evaluate_detection,
) -> list[dict[str, Any]]:
    """Fit 50 governed train-only models and evaluate the fixed test workload."""
    rows: list[dict[str, Any]] = []
    for plan in configuration_plans:
        for classifier in CLASSIFIERS:
            model_spec = classifiers[classifier]
            for seed in MODEL_ATTACK_SEEDS:
                training = workloads.get(seed, "training")
                test = workloads.get(seed, "inference")
                features = tuple(plan["features_by_seed"][str(seed)])
                assert_no_attack_metadata(features)
                train_x = training.features.loc[:, list(features)].copy()
                test_x = test.features.loc[:, list(features)].copy()
                if list(train_x.columns) != list(features) or list(test_x.columns) != list(features):
                    raise V09EError("Feature order changed during governed projection.")
                model = model_factory(
                    classifier,
                    features,
                    dict(model_spec["parameters"]),
                    random_state=seed,
                )
                v08d._verify_model(model, classifier, features, seed, model_spec)
                model.fit(train_x, training.labels)
                prediction = model.predict_frame(test_x)
                prediction_frame = pd.DataFrame(
                    {
                        "record_id": test.features["row_id"].to_numpy(copy=True),
                        "anomaly_score": prediction["anomaly_score"].to_numpy(copy=True),
                        "anomaly_label": prediction["anomaly_label"].to_numpy(copy=True),
                    }
                )
                raw_metrics = metric_evaluator(test.ground_truth, prediction_frame)
                metrics = v08d._complete_metrics(raw_metrics)
                visible = detector_visible_mask(
                    test.clean_features.loc[:, list(features)],
                    test.features.loc[:, list(features)],
                )
                attacked = test.ground_truth["is_attack"].eq(1).to_numpy()
                visible_count = int((visible & attacked).sum())
                metrics["visible_attack_count"] = visible_count
                metrics["invisible_attack_count"] = int(attacked.sum()) - visible_count
                metrics["selected_feature_visibility_rate"] = float(
                    visible_count / int(attacked.sum())
                )
                rows.append(
                    {
                        "schema_version": V09E_SCHEMA_VERSION,
                        "stage": V09E_STAGE,
                        "configuration_id": plan["configuration_id"],
                        "source_configuration_id": plan["source_configuration_id"],
                        "selection_scope": plan["selection_scope"],
                        "selection_semantics": plan["selection_semantics"],
                        "source_semantic_lock_sha256": plan["source_semantic_lock_sha256"],
                        "classifier": classifier,
                        "seed": seed,
                        "selected_feature_count": len(features),
                        "selected_features": list(features),
                        "selected_features_sha256": fingerprint_feature_names(features),
                        "model_parameters": dict(model_spec["parameters"]),
                        "prediction_threshold": model_spec["prediction_threshold"],
                        "training_split": "train",
                        "evaluation_split": "test",
                        "training_rows_sha256": training.row_ids_sha256,
                        "training_labels_sha256": training.labels_sha256,
                        "test_rows_sha256": test.row_ids_sha256,
                        "test_labels_sha256": test.labels_sha256,
                        "test_features_sha256": test.attacked_features_sha256,
                        "test_row_count": int(len(test.ground_truth)),
                        "metrics": metrics,
                        "model_fit_on_test": False,
                        "preprocessing_fit_on_test": False,
                        "feature_selection_on_test": False,
                        "threshold_tuned_on_test": False,
                        "test_access_authorized": True,
                    }
                )
    validate_raw_campaign(rows, configuration_plans, classifiers)
    return rows


def validate_raw_campaign(
    rows: Sequence[Mapping[str, Any]],
    plans: Sequence[Mapping[str, Any]],
    classifiers: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = {
        (configuration, classifier, seed)
        for configuration in CONFIGURATION_IDS
        for classifier in CLASSIFIERS
        for seed in MODEL_ATTACK_SEEDS
    }
    observed = {
        (row["configuration_id"], row["classifier"], int(row["seed"])) for row in rows
    }
    plan_by_id = {plan["configuration_id"]: plan for plan in plans}
    valid = (
        len(rows) == EXPECTED_EVALUATION_COUNT
        and len(observed) == EXPECTED_EVALUATION_COUNT
        and observed == expected
        and tuple(plan_by_id) == CONFIGURATION_IDS
        and set(classifiers) == set(CLASSIFIERS)
        and all(row["stage"] == V09E_STAGE for row in rows)
        and all(row["training_split"] == "train" for row in rows)
        and all(row["evaluation_split"] == "test" for row in rows)
        and all(row["model_fit_on_test"] is False for row in rows)
        and all(row["preprocessing_fit_on_test"] is False for row in rows)
        and all(row["feature_selection_on_test"] is False for row in rows)
        and all(row["threshold_tuned_on_test"] is False for row in rows)
        and all(row["prediction_threshold"] == 0.5 for row in rows)
        and all(row["test_access_authorized"] is True for row in rows)
        and all(int(row["test_row_count"]) == int(row["metrics"]["evaluated_records"]) for row in rows)
        and all(v08d._all_finite(row["metrics"].values()) for row in rows)
    )
    if not valid:
        raise V09EError("The governed final-test campaign is incomplete or invalid.")
    for configuration in ("BPSO-K10", "BGWO"):
        plan = plan_by_id[configuration]
        locked = tuple(plan["features_by_seed"]["42"])
        subset = [row for row in rows if row["configuration_id"] == configuration]
        if any(
            tuple(row["selected_features"]) != locked
            or row["selected_features_sha256"] != fingerprint_feature_names(locked)
            for row in subset
        ):
            raise V09EError(
                f"The final-test campaign did not use the exact locked {configuration} subset."
            )


def aggregate_final_test(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = []
    for configuration in CONFIGURATION_IDS:
        for classifier in CLASSIFIERS:
            group = [
                row
                for row in rows
                if row["configuration_id"] == configuration
                and row["classifier"] == classifier
            ]
            if len(group) != 5 or {row["seed"] for row in group} != set(MODEL_ATTACK_SEEDS):
                raise V09EError("Aggregation requires exactly five paired seeds per cell.")
            for metric in METRIC_COLUMNS:
                values = [float(row["metrics"][metric]) for row in group]
                summaries.append(
                    {
                        "configuration_id": configuration,
                        "classifier": classifier,
                        "metric": metric,
                        "n": 5,
                        "mean": statistics.fmean(values),
                        "std": statistics.stdev(values),
                        "min": min(values),
                        "max": max(values),
                    }
                )
    return {
        "schema_version": V09E_SCHEMA_VERSION,
        "stage": V09E_STAGE,
        "artifact_kind": "FINAL_TEST_AGGREGATION",
        "scientific_unit": "paired model/attack seed",
        "summaries": summaries,
        "test_driven_selection": False,
    }


def paired_comparisons(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = []
    for classifier in CLASSIFIERS:
        seen: set[frozenset[str]] = set()
        by_configuration = {
            configuration: {
                int(row["seed"]): row
                for row in rows
                if row["classifier"] == classifier
                and row["configuration_id"] == configuration
            }
            for configuration in CONFIGURATION_IDS
        }
        for candidate_id in CANDIDATE_COMPARISON_IDS:
            candidate = by_configuration[candidate_id]
            if set(candidate) != set(MODEL_ATTACK_SEEDS):
                raise V09EError("Paired comparison candidate seed coverage failed.")
            for reference_id in CONFIGURATION_IDS:
                if reference_id == candidate_id:
                    continue
                pair_key = frozenset((candidate_id, reference_id))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                reference = by_configuration[reference_id]
                if set(reference) != set(MODEL_ATTACK_SEEDS):
                    raise V09EError("Paired comparison reference seed coverage failed.")
                for metric in PRIMARY_METRICS:
                    pairs = [
                        {
                            "seed": seed,
                            "candidate_value": float(candidate[seed]["metrics"][metric]),
                            "reference_value": float(reference[seed]["metrics"][metric]),
                            "difference_candidate_minus_reference": float(
                                candidate[seed]["metrics"][metric]
                                - reference[seed]["metrics"][metric]
                            ),
                        }
                        for seed in MODEL_ATTACK_SEEDS
                    ]
                    differences = [row["difference_candidate_minus_reference"] for row in pairs]
                    mean = statistics.fmean(differences)
                    std = statistics.stdev(differences)
                    half_width = T_CRITICAL_95_DF4 * std / math.sqrt(5)
                    low, high = mean - half_width, mean + half_width
                    output.append(
                        {
                            "classifier": classifier,
                            "candidate_configuration_id": candidate_id,
                            "reference_configuration_id": reference_id,
                            "metric": metric,
                            "pairs": pairs,
                            "n": 5,
                            "df": 4,
                            "mean_difference": mean,
                            "std_difference": std,
                            "ci95_low": low,
                            "ci95_high": high,
                            "direction_uncertain": low <= 0.0 <= high,
                            "p_value_reported": False,
                        }
                    )
    return {
        "schema_version": V09E_SCHEMA_VERSION,
        "stage": V09E_STAGE,
        "artifact_kind": "PAIRED_FINAL_TEST_COMPARISONS",
        "difference_definition": "candidate minus reference within matched model/attack seed",
        "confidence_interval": "two-sided 95% Student-t interval, n=5, df=4",
        "candidate_configurations": list(CANDIDATE_COMPARISON_IDS),
        "primary_metrics": list(PRIMARY_METRICS),
        "comparisons": output,
    }


def preservation_analysis(summary: Mapping[str, Any]) -> dict[str, Any]:
    lookup = {
        (row["configuration_id"], row["classifier"], row["metric"]): row["mean"]
        for row in summary["summaries"]
    }
    results = []
    for classifier in CLASSIFIERS:
        checks = []
        for metric, margin in PRESERVATION_MARGINS.items():
            baseline = float(lookup[("K43", classifier, metric)])
            candidate = float(lookup[("BGWO", classifier, metric)])
            relative_change = (candidate - baseline) / baseline
            loss = max(0.0, -relative_change)
            checks.append(
                {
                    "metric": metric,
                    "k43_mean": baseline,
                    "bgwo_mean": candidate,
                    "relative_change": relative_change,
                    "relative_loss": loss,
                    "margin": margin,
                    "preserved": loss <= margin
                    or math.isclose(loss, margin, rel_tol=0.0, abs_tol=1e-12),
                }
            )
        results.append(
            {
                "classifier": classifier,
                "metric_checks": checks,
                "overall_preserved": all(check["preserved"] for check in checks),
            }
        )
    status = {row["classifier"]: row["overall_preserved"] for row in results}
    if status == {"decision_tree": True, "logistic_regression": True}:
        interpretation = "PRESERVATION_HOLDS_FOR_BOTH_CLASSIFIERS"
    elif status == {"decision_tree": True, "logistic_regression": False}:
        interpretation = "PRESERVATION_HOLDS_FOR_DT_ONLY_SEARCH_CLASSIFIER_DEPENDENCE_POSSIBLE"
    else:
        interpretation = "PRESERVATION_DOES_NOT_HOLD_FOR_BOTH_CLASSIFIERS"
    return {
        "schema_version": V09E_SCHEMA_VERSION,
        "stage": V09E_STAGE,
        "artifact_kind": "FINAL_TEST_PRESERVATION",
        "margins": dict(PRESERVATION_MARGINS),
        "margin_semantics": "preregistered descriptive engineering preservation criteria",
        "domain_validated_margins": False,
        "results": results,
        "cross_classifier_interpretation": interpretation,
        "winner_reselection_performed": False,
    }


def feature_overlap_analysis(
    bgwo_lock: Mapping[str, Any],
    bpso_lock: Mapping[str, Any],
    basis: FrozenBasis,
) -> dict[str, Any]:
    candidates = tuple(basis.candidate_features)
    order = {feature: index for index, feature in enumerate(candidates)}
    bgwo = tuple(bgwo_lock["ordered_selected_features"])
    bpso = tuple(bpso_lock["ordered_selected_features"])

    def _set_block(left: Sequence[str], right: Sequence[str]) -> dict[str, Any]:
        left_set, right_set = set(left), set(right)
        intersection = left_set & right_set
        union = left_set | right_set
        ordered = lambda values: sorted(values, key=lambda feature: order[feature])
        return {
            "intersection_size": len(intersection),
            "union_size": len(union),
            "jaccard": float(len(intersection) / len(union)) if union else 1.0,
            "shared_features": ordered(intersection),
            "left_only_features": ordered(left_set - right_set),
            "right_only_features": ordered(right_set - left_set),
        }

    bgwo_vs_bpso = _set_block(bgwo, bpso)
    mi_k11_by_seed = basis.lock["roles"]["best_supervised"]["selected_features"]
    per_seed = []
    for seed in MODEL_ATTACK_SEEDS:
        mi_features = tuple(mi_k11_by_seed[str(seed)])
        block = _set_block(bgwo, mi_features)
        per_seed.append({"seed": seed, "mi_k11_feature_count": len(mi_features), **block})
    mi_union = set()
    mi_intersection: set[str] | None = None
    for seed in MODEL_ATTACK_SEEDS:
        seed_features = set(mi_k11_by_seed[str(seed)])
        mi_union |= seed_features
        mi_intersection = seed_features if mi_intersection is None else (mi_intersection & seed_features)
    return {
        "schema_version": V09E_SCHEMA_VERSION,
        "stage": V09E_STAGE,
        "artifact_kind": "FEATURE_OVERLAP_ANALYSIS",
        "descriptive_only": True,
        "mutated_any_subset": False,
        "bgwo_feature_count": len(bgwo),
        "bpso_feature_count": len(bpso),
        "bgwo_vs_bpso": {
            **bgwo_vs_bpso,
            "shared_features": bgwo_vs_bpso["shared_features"],
            "bgwo_only_features": bgwo_vs_bpso["left_only_features"],
            "bpso_only_features": bgwo_vs_bpso["right_only_features"],
        },
        "bgwo_vs_mi_k11": {
            "per_seed": per_seed,
            "mean_seed_jaccard": statistics.fmean(row["jaccard"] for row in per_seed),
            "union_jaccard": float(
                len(set(bgwo) & mi_union) / len(set(bgwo) | mi_union)
            )
            if mi_union
            else 1.0,
            "intersection_jaccard": float(
                len(set(bgwo) & mi_intersection) / len(set(bgwo) | mi_intersection)
            )
            if mi_intersection
            else 1.0,
        },
    }


def build_test_leakage_audit(
    *,
    bgwo_lock_before: Mapping[str, Any],
    bgwo_lock_after: Mapping[str, Any],
    bgwo_lock_sha256_before: str,
    bgwo_lock_sha256_after: str,
    bpso_lock_sha256_before: str,
    bpso_lock_sha256_after: str,
    audit: Mapping[str, Any],
    starting_head: str,
    ending_head: str,
    evaluation_count: int,
) -> dict[str, Any]:
    recomputed_mask_hash = hashlib.sha256(
        np.asarray(bgwo_lock_after["mask"], dtype=np.uint8).tobytes()
    ).hexdigest()
    mask_unchanged = list(bgwo_lock_before["mask"]) == list(bgwo_lock_after["mask"])
    features_unchanged = tuple(bgwo_lock_before["ordered_selected_features"]) == tuple(
        bgwo_lock_after["ordered_selected_features"]
    ) and bgwo_lock_before["selected_features_sha256"] == bgwo_lock_after[
        "selected_features_sha256"
    ]
    semantic_unchanged = (
        bgwo_lock_before["semantic_lock_sha256"] == bgwo_lock_after["semantic_lock_sha256"]
    )
    checks = {
        "test_accessed_only_after_winner_lock_verification": audit.get(
            "winner_lock_verified_before_test_access"
        )
        is True,
        "winner_lock_verified_before_test_access": audit.get(
            "winner_lock_verified_before_test_access"
        )
        is True,
        "bgwo_feature_mask_unchanged_before_after_test": mask_unchanged,
        "bgwo_feature_hash_unchanged": features_unchanged,
        "bgwo_semantic_lock_unchanged": semantic_unchanged,
        "bgwo_winner_lock_file_unchanged": bgwo_lock_sha256_before
        == bgwo_lock_sha256_after,
        "bgwo_mask_hash_recomputed_matches": recomputed_mask_hash
        == bgwo_lock_after["mask_sha256"],
        "no_optimizer_executed": True,
        "no_feature_selector_executed": True,
        "no_hyperparameter_search_executed": True,
        "no_threshold_tuning_executed": True,
        "no_winner_replacement_executed": True,
        "bpso_not_rerun": True,
        "bgwo_not_rerun": True,
        "bpso_winner_lock_file_unchanged": bpso_lock_sha256_before
        == bpso_lock_sha256_after,
        "starting_head_unchanged": starting_head == ending_head,
        "governed_evaluation_count_is_fifty": evaluation_count == EXPECTED_EVALUATION_COUNT,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise V09EError(f"V0.9-E post-evaluation leakage audit failed: {failed}")
    return {
        "schema_version": V09E_SCHEMA_VERSION,
        "stage": V09E_STAGE,
        "status": "PASS",
        "artifact_kind": "POST_EVALUATION_TEST_ACCESS_AUDIT",
        "starting_head": starting_head,
        "ending_head": ending_head,
        "source_winner_stage": "V0.9-D",
        "winner_lock_verified_before_test_access": True,
        "winner_lock_verified_at_utc": audit.get("winner_lock_verified_at_utc"),
        "test_access_activated_at_utc": audit.get("test_access_activated_at_utc"),
        "campaign_completed_at_utc": audit.get("campaign_completed_at_utc"),
        "bgwo_winner_semantic_lock_sha256": bgwo_lock_after["semantic_lock_sha256"],
        "bgwo_winner_mask_sha256": bgwo_lock_after["mask_sha256"],
        "bgwo_winner_selected_features_sha256": bgwo_lock_after["selected_features_sha256"],
        "bpso_winner_semantic_lock_sha256": audit.get("bpso_source_semantic_lock_sha256"),
        "governed_evaluation_count": evaluation_count,
        "final_test_accessed": True,
        "final_test_opened": True,
        "campaign_completed": True,
        "rerun_count": int(audit.get("rerun_count", 0)),
        "test_used_for_evaluation_only": True,
        "test_used_for_selection": False,
        "test_used_for_tuning": False,
        "test_used_for_feature_selection": False,
        "checks": checks,
        "bgwo_winner_lock_sha256_before": bgwo_lock_sha256_before,
        "bgwo_winner_lock_sha256_after": bgwo_lock_sha256_after,
        "bpso_winner_lock_sha256_before": bpso_lock_sha256_before,
        "bpso_winner_lock_sha256_after": bpso_lock_sha256_after,
    }


def build_result_lock(
    *,
    starting_head: str,
    bgwo_lock: Mapping[str, Any],
    bpso_lock: Mapping[str, Any],
    configurations: Sequence[Mapping[str, Any]],
    classifiers: Mapping[str, Mapping[str, Any]],
    test_dataset_identity: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    preservation: Mapping[str, Any],
    governed_evaluation_count: int,
    started_at: str,
    completed_at: str,
    wall_time_sec: float,
    rerun_count: int,
) -> dict[str, Any]:
    configuration_plan = [
        {
            "configuration_id": plan["configuration_id"],
            "source_configuration_id": plan["source_configuration_id"],
            "feature_count": plan["feature_count"],
            "selection_scope": plan["selection_scope"],
            "selection_semantics": plan["selection_semantics"],
            "source_semantic_lock_sha256": plan["source_semantic_lock_sha256"],
        }
        for plan in configurations
    ]
    preservation_status = {
        row["classifier"]: row["overall_preserved"] for row in preservation["results"]
    }
    lock = {
        "schema_version": V09E_SCHEMA_VERSION,
        "stage": V09E_STAGE,
        "status": "FINAL_TEST_EVALUATED",
        "starting_head": starting_head,
        "source_winner_stage": "V0.9-D",
        "source_winner_semantic_lock_sha256": bgwo_lock["semantic_lock_sha256"],
        "bgwo_selected_feature_count": int(bgwo_lock["selected_feature_count"]),
        "bgwo_ordered_selected_features": list(bgwo_lock["ordered_selected_features"]),
        "bgwo_mask_sha256": bgwo_lock["mask_sha256"],
        "bgwo_selected_features_sha256": bgwo_lock["selected_features_sha256"],
        "bgwo_feature_manifest_sha256": bgwo_lock["feature_manifest_sha256"],
        "bpso_source_semantic_lock_sha256": bpso_lock["semantic_lock_sha256"],
        "bpso_selected_feature_count": int(bpso_lock["selected_feature_count"]),
        "bpso_ordered_selected_features": list(bpso_lock["ordered_selected_features"]),
        "bpso_mask_sha256": bpso_lock["mask_sha256"],
        "bpso_selected_features_sha256": bpso_lock["selected_features_sha256"],
        "configuration_ids": list(CONFIGURATION_IDS),
        "configuration_plan": configuration_plan,
        "configuration_plan_sha256": _json_sha256(configuration_plan),
        "classifier_configurations": classifiers,
        "classifier_configurations_sha256": _json_sha256(classifiers),
        "seed_namespace": list(MODEL_ATTACK_SEEDS),
        "test_dataset_identity": dict(test_dataset_identity),
        "governed_evaluation_count": governed_evaluation_count,
        "result_artifact_hashes": dict(artifact_hashes),
        "preservation_result": preservation_status,
        "test_access_audit_sha256": artifact_hashes[AUDIT_PATH.name],
        "test_used_for_evaluation_only": True,
        "test_used_for_selection": False,
        "test_used_for_tuning": False,
        "test_used_for_feature_selection": False,
        "optimizer_invoked": False,
        "feature_reselection_performed": False,
        "test_driven_tuning_performed": False,
        "threshold_tuning_performed": False,
        "winner_replaced_after_test": False,
        "bpso_rerun": False,
        "bgwo_rerun": False,
        "resource_benchmark_executed": False,
        "direct_energy_measured": False,
        "winner_unchanged_statement": (
            "The V0.9-D BGWO winner was not modified after final-test access."
        ),
        "historical_test_statement": (
            "The final test was historically accessed in V0.6 but remained untouched by "
            "V0.9-D search and selection; V0.9-E accesses it once for governed evaluation only."
        ),
        "execution_metadata": {
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "wall_time_sec": wall_time_sec,
            "governed_evaluation_count": governed_evaluation_count,
            "rerun_count": rerun_count,
        },
    }
    lock["semantic_result_lock_sha256"] = result_lock_semantic_hash(lock)
    return lock


def result_lock_semantic_hash(lock: Mapping[str, Any]) -> str:
    payload = _deepcopy_json(lock)
    payload.pop("semantic_result_lock_sha256", None)
    execution = dict(payload.get("execution_metadata", {}))
    for field in ("started_at_utc", "completed_at_utc", "wall_time_sec"):
        execution.pop(field, None)
    payload["execution_metadata"] = execution
    return _json_sha256(payload)


def verify_result_lock(lock: Mapping[str, Any]) -> None:
    required = {
        "stage",
        "status",
        "starting_head",
        "source_winner_semantic_lock_sha256",
        "bgwo_selected_feature_count",
        "bgwo_ordered_selected_features",
        "bgwo_mask_sha256",
        "bgwo_selected_features_sha256",
        "bpso_source_semantic_lock_sha256",
        "bpso_selected_feature_count",
        "bpso_ordered_selected_features",
        "bpso_mask_sha256",
        "bpso_selected_features_sha256",
        "configuration_ids",
        "configuration_plan",
        "configuration_plan_sha256",
        "classifier_configurations",
        "classifier_configurations_sha256",
        "seed_namespace",
        "test_dataset_identity",
        "governed_evaluation_count",
        "result_artifact_hashes",
        "preservation_result",
        "test_access_audit_sha256",
        "semantic_result_lock_sha256",
    }
    configs = tuple(lock.get("configuration_plan", ()))
    valid = (
        required <= set(lock)
        and lock.get("stage") == V09E_STAGE
        and lock.get("status") == "FINAL_TEST_EVALUATED"
        and lock.get("source_winner_stage") == "V0.9-D"
        and lock.get("source_winner_semantic_lock_sha256")
        == EXPECTED_BGWO_WINNER["semantic_lock_sha256"]
        and lock.get("bgwo_mask_sha256") == EXPECTED_BGWO_WINNER["mask_sha256"]
        and lock.get("bgwo_selected_features_sha256")
        == EXPECTED_BGWO_WINNER["selected_features_sha256"]
        and lock.get("bgwo_selected_feature_count")
        == EXPECTED_BGWO_WINNER["selected_feature_count"]
        and tuple(lock.get("bgwo_ordered_selected_features", ())) == EXPECTED_BGWO_FEATURES
        and lock.get("bpso_source_semantic_lock_sha256")
        == EXPECTED_BPSO_WINNER["semantic_lock_sha256"]
        and lock.get("bpso_mask_sha256") == EXPECTED_BPSO_WINNER["mask_sha256"]
        and lock.get("bpso_selected_features_sha256")
        == EXPECTED_BPSO_WINNER["selected_features_sha256"]
        and tuple(lock.get("bpso_ordered_selected_features", ())) == EXPECTED_BPSO_FEATURES
        and tuple(lock.get("configuration_ids", ())) == CONFIGURATION_IDS
        and tuple(row.get("configuration_id") for row in configs) == CONFIGURATION_IDS
        and lock.get("configuration_plan_sha256") == _json_sha256(lock.get("configuration_plan"))
        and lock.get("classifier_configurations_sha256")
        == _json_sha256(lock.get("classifier_configurations"))
        and tuple(lock.get("seed_namespace", ())) == MODEL_ATTACK_SEEDS
        and int(lock.get("test_dataset_identity", {}).get("test_rows", -1))
        == EXPECTED_TEST_ROWS
        and int(lock.get("governed_evaluation_count", -1)) == EXPECTED_EVALUATION_COUNT
        and lock.get("test_used_for_selection") is False
        and lock.get("test_used_for_tuning") is False
        and lock.get("test_used_for_feature_selection") is False
        and lock.get("optimizer_invoked") is False
        and lock.get("feature_reselection_performed") is False
        and lock.get("test_driven_tuning_performed") is False
        and lock.get("threshold_tuning_performed") is False
        and lock.get("winner_replaced_after_test") is False
        and lock.get("bpso_rerun") is False
        and lock.get("bgwo_rerun") is False
        and lock.get("resource_benchmark_executed") is False
        and lock.get("direct_energy_measured") is False
        and result_lock_semantic_hash(lock) == lock.get("semantic_result_lock_sha256")
    )
    if not valid:
        raise V09EError("V0.9-E final-test result lock verification failed.")


def write_or_verify_result_lock(path: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    verify_result_lock(lock)
    target = Path(path)
    if target.exists():
        existing = _read_json(target)
        verify_result_lock(existing)
        if existing["semantic_result_lock_sha256"] != lock["semantic_result_lock_sha256"]:
            raise V09EError("Existing V0.9-E result lock differs from the recomputed result.")
        return existing
    _atomic_write_json(target, lock)
    stored = _read_json(target)
    verify_result_lock(stored)
    return stored


def _write_metrics_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    frame_rows = []
    for row in rows:
        entry = {
            "configuration_id": row["configuration_id"],
            "source_configuration_id": row["source_configuration_id"],
            "classifier": row["classifier"],
            "seed": row["seed"],
            "selected_feature_count": row["selected_feature_count"],
            "selected_features_sha256": row["selected_features_sha256"],
            "source_semantic_lock_sha256": row["source_semantic_lock_sha256"],
        }
        entry.update({metric: row["metrics"][metric] for metric in METRIC_COLUMNS})
        frame_rows.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frame_rows).to_csv(path, index=False)


def _write_paired_csv(paired: Mapping[str, Any], path: Path) -> None:
    rows = [
        {
            "classifier": row["classifier"],
            "candidate_configuration_id": row["candidate_configuration_id"],
            "reference_configuration_id": row["reference_configuration_id"],
            "metric": row["metric"],
            "n": row["n"],
            "df": row["df"],
            "mean_difference": row["mean_difference"],
            "std_difference": row["std_difference"],
            "ci95_low": row["ci95_low"],
            "ci95_high": row["ci95_high"],
            "direction_uncertain": row["direction_uncertain"],
        }
        for row in paired["comparisons"]
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_v09e(
    *,
    output_dir: Path | str = OUTPUT_DIR,
    basis_loader: Callable[[], FrozenBasis] = load_frozen_basis,
    benchmark_context_loader: Callable[[], Any] = v07b.load_v06_benchmark_context,
    workload_loader: Callable[[Any], Any] = v07b.prepare_verified_workloads,
) -> dict[str, Any]:
    """Run the single governed final-test campaign and lock its evidence."""
    output = Path(output_dir)
    result_lock_path = output / RESULT_LOCK_PATH.name
    if result_lock_path.exists():
        raise V09EError("V0.9-E final-test campaign is already locked; rerun prohibited.")

    stage_start = time.perf_counter()
    started_at = _utc_now()
    preflight = verify_v09e_preflight()
    starting_head = preflight["starting_head"]

    bgwo_lock = load_verified_bgwo_winner_lock()
    bpso_lock = load_verified_bpso_winner_lock()
    basis = basis_loader()
    configurations = build_configuration_plan(basis, bgwo_lock, bpso_lock)
    classifiers = classifier_plan()

    output.mkdir(parents=True, exist_ok=True)
    immutable_before = snapshot_immutable_paths()
    bgwo_lock_sha_before = sha256_file(BGWO_WINNER_LOCK_PATH)
    bpso_lock_sha_before = sha256_file(BPSO_WINNER_LOCK_PATH)
    bgwo_lock_before = _deepcopy_json(bgwo_lock)

    audit_path = output / AUDIT_PATH.name
    access_audit = activate_test_access_audit(
        audit_path, preflight, bgwo_lock, bpso_lock
    )

    benchmark_context = benchmark_context_loader()
    workloads = load_authorized_test_workloads(
        benchmark_context, access_audit, loader=workload_loader
    )
    rows = evaluate_final_test_campaign(configurations, classifiers, workloads)
    if len(rows) != EXPECTED_EVALUATION_COUNT:
        raise V09EError("Governed evaluation count is not exactly fifty.")
    access_audit = complete_test_access_audit(
        audit_path, access_audit, evaluation_count=len(rows)
    )

    summary = aggregate_final_test(rows)
    paired = paired_comparisons(rows)
    preservation = preservation_analysis(summary)
    overlap = feature_overlap_analysis(bgwo_lock, bpso_lock, basis)

    raw = {
        "schema_version": V09E_SCHEMA_VERSION,
        "stage": V09E_STAGE,
        "artifact_kind": "RAW_GOVERNED_FINAL_TEST_EVIDENCE",
        "selection_preceded_test_access": True,
        "source_winner_semantic_lock_sha256": bgwo_lock["semantic_lock_sha256"],
        "bpso_source_semantic_lock_sha256": bpso_lock["semantic_lock_sha256"],
        "configuration_plan": list(configurations),
        "classifier_plan": classifiers,
        "rows": [dict(row) for row in rows],
        "test_workload_hashes": {
            str(seed): getattr(workloads, "workload_hashes", {}).get((seed, "inference"))
            for seed in MODEL_ATTACK_SEEDS
        },
        "governed_evaluation_count": len(rows),
        "test_driven_selection": False,
    }

    _atomic_write_json(output / RAW_PATH.name, raw)
    _write_metrics_csv(rows, output / METRICS_CSV_PATH.name)
    _atomic_write_json(output / SUMMARY_PATH.name, summary)
    _write_paired_csv(paired, output / PAIRED_CSV_PATH.name)
    _atomic_write_json(output / OVERLAP_PATH.name, overlap)
    _atomic_write_json(output / PRESERVATION_PATH.name, preservation)

    bgwo_lock_after = _read_json(BGWO_WINNER_LOCK_PATH)
    v09d.verify_winner_lock(bgwo_lock_after)
    bgwo_lock_sha_after = sha256_file(BGWO_WINNER_LOCK_PATH)
    bpso_lock_sha_after = sha256_file(BPSO_WINNER_LOCK_PATH)
    ending_head = current_head_short(PROJECT_ROOT)
    leakage = build_test_leakage_audit(
        bgwo_lock_before=bgwo_lock_before,
        bgwo_lock_after=bgwo_lock_after,
        bgwo_lock_sha256_before=bgwo_lock_sha_before,
        bgwo_lock_sha256_after=bgwo_lock_sha_after,
        bpso_lock_sha256_before=bpso_lock_sha_before,
        bpso_lock_sha256_after=bpso_lock_sha_after,
        audit=access_audit,
        starting_head=starting_head,
        ending_head=ending_head,
        evaluation_count=len(rows),
    )
    _atomic_write_json(output / AUDIT_PATH.name, leakage)

    artifact_hashes = {
        name: sha256_file(output / name)
        for name in (
            RAW_PATH.name,
            METRICS_CSV_PATH.name,
            SUMMARY_PATH.name,
            PAIRED_CSV_PATH.name,
            OVERLAP_PATH.name,
            PRESERVATION_PATH.name,
            AUDIT_PATH.name,
        )
    }
    test_dataset_identity = {
        "name": EXPECTED_DATASET,
        "split_seed": preflight["dataset_identity"]["split_seed"],
        "split_sizes": dict(preflight["dataset_identity"]["split_sizes"]),
        "test_rows": EXPECTED_TEST_ROWS,
        "dimensions": EXPECTED_DIMENSIONS,
        "feature_manifest_sha256": basis.candidate_manifest_sha256,
    }
    completed_at = _utc_now()
    proposed_lock = build_result_lock(
        starting_head=starting_head,
        bgwo_lock=bgwo_lock,
        bpso_lock=bpso_lock,
        configurations=configurations,
        classifiers=classifiers,
        test_dataset_identity=test_dataset_identity,
        artifact_hashes=artifact_hashes,
        preservation=preservation,
        governed_evaluation_count=len(rows),
        started_at=started_at,
        completed_at=completed_at,
        wall_time_sec=time.perf_counter() - stage_start,
        rerun_count=int(access_audit["rerun_count"]),
    )
    result_lock = write_or_verify_result_lock(result_lock_path, proposed_lock)

    immutable_after = snapshot_immutable_paths()
    if immutable_after != immutable_before:
        raise V09EError("Frozen V0.6-V0.9-D artifacts changed during final-test evaluation.")
    if bgwo_lock_sha_after != bgwo_lock_sha_before:
        raise V09EError("The V0.9-D winner lock changed during final-test evaluation.")

    execution = {
        "schema_version": V09E_SCHEMA_VERSION,
        "stage": V09E_STAGE,
        "status": "COMPLETED",
        "readiness_for_v09f": "GO",
        "starting_head": starting_head,
        "ending_head": ending_head,
        "preflight": preflight,
        "dataset_identity": test_dataset_identity,
        "winner": {
            "optimizer": "BGWO",
            "source_optimizer_seed": bgwo_lock["source_optimizer_seed"],
            "selected_feature_count": bgwo_lock["selected_feature_count"],
            "mask_sha256": bgwo_lock["mask_sha256"],
            "selected_features_sha256": bgwo_lock["selected_features_sha256"],
            "semantic_lock_sha256": bgwo_lock["semantic_lock_sha256"],
        },
        "bpso_reference": {
            "selected_feature_count": bpso_lock["selected_feature_count"],
            "mask_sha256": bpso_lock["mask_sha256"],
            "selected_features_sha256": bpso_lock["selected_features_sha256"],
            "semantic_lock_sha256": bpso_lock["semantic_lock_sha256"],
        },
        "configuration_ids": list(CONFIGURATION_IDS),
        "classifiers": list(CLASSIFIERS),
        "seeds": list(MODEL_ATTACK_SEEDS),
        "governed_evaluation_count": len(rows),
        "campaign_wall_time_sec": time.perf_counter() - stage_start,
        "campaign_rerun_count": access_audit["rerun_count"],
        "preservation": {
            row["classifier"]: row["overall_preserved"] for row in preservation["results"]
        },
        "cross_classifier_interpretation": preservation["cross_classifier_interpretation"],
        "artifact_hashes": {
            **artifact_hashes,
            RESULT_LOCK_PATH.name: sha256_file(result_lock_path),
        },
        "semantic_result_lock_sha256": result_lock["semantic_result_lock_sha256"],
        "immutable_hashes_before": immutable_before,
        "immutable_hashes_after": immutable_after,
        "final_test_accessed": True,
        "test_used_for_evaluation_only": True,
        "test_used_for_selection": False,
        "test_used_for_tuning": False,
        "test_used_for_feature_selection": False,
        "winner_modified_after_test_access": False,
        "optimizer_invoked": False,
        "feature_reselection_performed": False,
        "test_driven_tuning_performed": False,
        "threshold_tuning_performed": False,
        "bgwo_rerun": False,
        "bpso_rerun": False,
        "resource_benchmark_executed": False,
        "direct_energy_measured": False,
    }
    _atomic_write_json(output / EXECUTION_PATH.name, execution)
    verify_v09e_artifacts(output)
    return execution


def verify_v09e_artifacts(output: Path = OUTPUT_DIR) -> None:
    execution = _read_json(output / EXECUTION_PATH.name)
    lock = _read_json(output / RESULT_LOCK_PATH.name)
    verify_result_lock(lock)
    audit = _read_json(output / AUDIT_PATH.name)
    raw = _read_json(output / RAW_PATH.name)
    if (
        execution.get("status") != "COMPLETED"
        or execution.get("governed_evaluation_count") != EXPECTED_EVALUATION_COUNT
        or execution.get("winner_modified_after_test_access") is not False
        or audit.get("status") != "PASS"
        or audit.get("winner_lock_verified_before_test_access") is not True
        or audit.get("governed_evaluation_count") != EXPECTED_EVALUATION_COUNT
        or raw.get("governed_evaluation_count") != EXPECTED_EVALUATION_COUNT
        or raw.get("test_driven_selection") is not False
    ):
        raise V09EError("V0.9-E artifact governance verification failed.")
    for name, expected in execution["artifact_hashes"].items():
        if sha256_file(output / name) != expected:
            raise V09EError(f"V0.9-E artifact hash mismatch: {name}")
    for name, expected in lock["result_artifact_hashes"].items():
        if sha256_file(output / name) != expected:
            raise V09EError(f"V0.9-E result-lock hash mismatch: {name}")


def _deepcopy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V09EError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V09EError(f"JSON artifact must be an object: {path}")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    result = run_v09e()
    print(
        json.dumps(
            {
                "stage": V09E_STAGE,
                "status": result["status"],
                "governed_evaluation_count": result["governed_evaluation_count"],
                "preservation": result["preservation"],
                "cross_classifier_interpretation": result["cross_classifier_interpretation"],
                "semantic_result_lock_sha256": result["semantic_result_lock_sha256"],
                "readiness_for_v09f": result["readiness_for_v09f"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
