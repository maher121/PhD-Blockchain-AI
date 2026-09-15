"""Governed V0.8-D final-test evaluation of the locked BPSO winner."""

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
from src.pipeline_v08c import (
    IMMUTABLE_PATHS as V08C_INHERITED_IMMUTABLE_PATHS,
    V08C_STAGE,
    verify_winner_lock,
)
import src.pipeline_v07b as v07b
from src.security.evaluation import detector_visible_mask, evaluate_detection
from src.security.experiment_data import fingerprint_feature_names
from src.security.ground_truth import assert_no_attack_metadata


V08D_STAGE = "V0.8-D"
V08D_SCHEMA_VERSION = "v0.8-d-governed-final-test-1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "bpso"
WINNER_LOCK_PATH = OUTPUT_DIR / "v08c_winner_lock.json"
MODEL_ATTACK_SEEDS = (42, 43, 44, 45, 46)
CONFIGURATION_IDS = ("K43", "K42", "K11", "BPSO-K10")
CLASSIFIERS = ("decision_tree", "logistic_regression")
T_CRITICAL_95_DF4 = 2.7764451051977987
EXPECTED_WINNER = {
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
V08C_TRACKED_PATHS = (
    "src/pipeline_v08c.py",
    "tests/test_pipeline_v08c.py",
    "results/bpso/v08c_run_1042.json",
    "results/bpso/v08c_run_1043.json",
    "results/bpso/v08c_run_1044.json",
    "results/bpso/v08c_run_1045.json",
    "results/bpso/v08c_run_1046.json",
    "results/bpso/v08c_stability.json",
    "results/bpso/v08c_convergence.json",
    "results/bpso/v08c_validation_comparison.json",
    "results/bpso/v08c_winner_lock.json",
    "results/bpso/v08c_execution_summary.json",
)
IMMUTABLE_PATHS = tuple(
    dict.fromkeys(
        (
            *V08C_INHERITED_IMMUTABLE_PATHS,
            *(PROJECT_ROOT / path for path in V08C_TRACKED_PATHS),
        )
    )
)
RAW_PATH = OUTPUT_DIR / "v08d_final_test_raw.json"
SUMMARY_PATH = OUTPUT_DIR / "v08d_final_test_summary.json"
PAIRED_PATH = OUTPUT_DIR / "v08d_paired_comparisons.json"
PRESERVATION_PATH = OUTPUT_DIR / "v08d_preservation.json"
AUDIT_PATH = OUTPUT_DIR / "v08d_test_access_audit.json"
EXECUTION_PATH = OUTPUT_DIR / "v08d_execution_summary.json"
RESULT_LOCK_PATH = OUTPUT_DIR / "v08d_final_test_lock.json"


class V08DError(RuntimeError):
    """Raised when final-test governance or evidence fails closed."""


def snapshot_immutable_paths(paths: Sequence[Path] = IMMUTABLE_PATHS) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise V08DError(f"Missing frozen prerequisite: {missing}")
    return {str(path.resolve()): sha256_file(path) for path in paths}


def verify_v08c_prerequisites() -> dict[str, Any]:
    """Require committed, pushed V0.8-C code, evidence, and exact winner lock."""
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *V08C_TRACKED_PATHS],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *V08C_TRACKED_PATHS],
        cwd=PROJECT_ROOT,
        check=False,
    )
    upstream = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode or clean.returncode or upstream.returncode:
        raise V08DError("V0.8-C must be committed, clean, and have an upstream branch.")
    divergence = subprocess.run(
        ["git", "rev-list", "--left-right", "--count", "HEAD...@{u}"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if divergence != "0\t0":
        raise V08DError(f"V0.8-C HEAD is not pushed exactly to upstream: {divergence}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    summary = _read_json(OUTPUT_DIR / "v08c_execution_summary.json")
    lock = load_verified_winner_lock()
    if (
        summary.get("status") != "COMPLETED"
        or summary.get("readiness_for_v08d") != "GO"
        or summary.get("final_test_accessed") is not False
        or summary.get("winner", {}).get("semantic_lock_sha256")
        != lock["semantic_lock_sha256"]
        or summary.get("artifact_hashes", {}).get("v08c_winner_lock.json")
        != sha256_file(WINNER_LOCK_PATH)
    ):
        raise V08DError("V0.8-C completion evidence is invalid.")
    return {
        "status": "PASS",
        "starting_head": head,
        "upstream": upstream.stdout.strip(),
        "ahead": 0,
        "behind": 0,
        "v08c_committed_and_pushed": True,
        "winner_lock_sha256": sha256_file(WINNER_LOCK_PATH),
        "winner_semantic_lock_sha256": lock["semantic_lock_sha256"],
    }


def load_verified_winner_lock(path: Path = WINNER_LOCK_PATH) -> dict[str, Any]:
    lock = _read_json(path)
    verify_winner_lock(lock)
    exact = all(lock.get(name) == value for name, value in EXPECTED_WINNER.items())
    mask = np.asarray(lock.get("mask", ()), dtype=np.uint8)
    features = tuple(lock.get("ordered_selected_features", ()))
    if (
        not exact
        or lock.get("stage") != V08C_STAGE
        or lock.get("status") != "VALIDATION_LOCKED"
        or lock.get("eligible_for_v08d") is not True
        or lock.get("selection_scope") != "TRAIN_AND_DEVELOPMENT_VALIDATION_ONLY"
        or lock.get("final_test_accessed") is not False
        or mask.shape != (43,)
        or int(mask.sum()) != 10
        or hashlib.sha256(mask.tobytes()).hexdigest() != EXPECTED_WINNER["mask_sha256"]
        or features != EXPECTED_BPSO_FEATURES
        or fingerprint_feature_names(features)
        != EXPECTED_WINNER["selected_features_sha256"]
    ):
        raise V08DError("V0.8-C winner lock does not match the authorized K10 winner.")
    return lock


def build_configuration_plan(
    basis: FrozenBasis, winner_lock: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    candidates = basis.candidate_features
    k11_by_seed = basis.lock["roles"]["best_supervised"]["selected_features"]
    expected_seed_keys = {str(seed) for seed in MODEL_ATTACK_SEEDS}
    if set(k11_by_seed) != expected_seed_keys:
        raise V08DError("Historical MI-K11 features do not cover exact seeds 42-46.")
    bpso_features = tuple(winner_lock["ordered_selected_features"])
    plans = (
        {
            "configuration_id": "K43",
            "source_configuration_id": "none_natural",
            "feature_count": 43,
            "selection_scope": "frozen V0.6 full baseline",
            "selection_semantics": "universal full candidate manifest",
            "features_by_seed": {str(seed): list(candidates) for seed in MODEL_ATTACK_SEEDS},
        },
        {
            "configuration_id": "K42",
            "source_configuration_id": "pairwise_correlation_filter_natural",
            "feature_count": 42,
            "selection_scope": "frozen V0.6 pairwise-correlation baseline",
            "selection_semantics": "universal deterministic training-derived subset",
            "features_by_seed": {
                str(seed): list(basis.k42_features) for seed in MODEL_ATTACK_SEEDS
            },
        },
        {
            "configuration_id": "K11",
            "source_configuration_id": "mutual_information_select_k_best_k11",
            "feature_count": 11,
            "selection_scope": "frozen V0.6 historical MI rule",
            "selection_semantics": "seed-specific supervised training-derived subsets",
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
            "features_by_seed": {
                str(seed): list(bpso_features) for seed in MODEL_ATTACK_SEEDS
            },
        },
    )
    if tuple(plan["configuration_id"] for plan in plans) != CONFIGURATION_IDS:
        raise V08DError("Final-test configuration plan must contain exactly K43/K42/K11/BPSO-K10.")
    for plan in plans:
        if plan["feature_count"] != len(plan["features_by_seed"]["42"]):
            raise V08DError(f"Configuration feature count mismatch: {plan['configuration_id']}")
        for seed in MODEL_ATTACK_SEEDS:
            features = tuple(plan["features_by_seed"][str(seed)])
            if (
                len(features) != plan["feature_count"]
                or len(set(features)) != len(features)
                or tuple(feature for feature in candidates if feature in features) != features
            ):
                raise V08DError(
                    f"Configuration feature identity/order mismatch: {plan['configuration_id']} seed {seed}"
                )
    if len({tuple(k11_by_seed[str(seed)]) for seed in MODEL_ATTACK_SEEDS}) <= 1:
        raise V08DError("MI-K11 must preserve its historical seed-specific semantics.")
    return plans


def classifier_plan() -> dict[str, dict[str, Any]]:
    plan = {
        "decision_tree": {
            "parameters": dict(v07b.DT_PARAMETERS),
            "prediction_threshold": 0.5,
        },
        "logistic_regression": {
            "parameters": dict(v07b.LR_PARAMETERS),
            "prediction_threshold": 0.5,
        },
    }
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
        raise V08DError("Frozen DT/LR classifier configuration changed.")
    return plan


def activate_test_access_audit(
    output_path: Path,
    prerequisites: Mapping[str, Any],
    winner_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist authorization before any test workload is loaded."""
    if output_path.with_name(RESULT_LOCK_PATH.name).exists():
        raise V08DError("V0.8-D final result lock already exists; campaign rerun prohibited.")
    rerun_count = 0
    prior_events: list[dict[str, Any]] = []
    if output_path.exists():
        prior = _read_json(output_path)
        if prior.get("status") != "TEST_ACCESS_ACTIVATED":
            raise V08DError("Existing final-test audit is not a recoverable interrupted state.")
        rerun_count = int(prior.get("rerun_count", 0)) + 1
        prior_events = list(prior.get("events", ()))
    lock_verified_at = _utc_now()
    activated_at = _utc_now()
    audit = {
        "schema_version": V08D_SCHEMA_VERSION,
        "stage": V08D_STAGE,
        "status": "TEST_ACCESS_ACTIVATED",
        "winner_lock_verified_before_test_access": True,
        "winner_lock_verified_at_utc": lock_verified_at,
        "test_access_activated_at_utc": activated_at,
        "source_winner_semantic_lock_sha256": winner_lock["semantic_lock_sha256"],
        "locked_mask_sha256": winner_lock["mask_sha256"],
        "locked_selected_features_sha256": winner_lock["selected_features_sha256"],
        "checks": {
            "v08c_committed_and_pushed": prerequisites["v08c_committed_and_pushed"],
            "winner_lock_verified": True,
            "winner_selected_before_test_access": True,
            "optimizer_invoked": False,
            "feature_selection_invoked": False,
            "winner_replacement_allowed": False,
            "test_driven_configuration_change": False,
            "test_driven_threshold_change": False,
            "test_driven_classifier_tuning": False,
            "final_test_used_only_for_evaluation": True,
        },
        "events": [
            *prior_events,
            {
                "event": "WINNER_LOCK_VERIFIED",
                "timestamp_utc": lock_verified_at,
            },
            {
                "event": "TEST_ACCESS_ACTIVATED",
                "timestamp_utc": activated_at,
            },
        ],
        "final_test_opened": False,
        "campaign_completed": False,
        "rerun_count": rerun_count,
        "partial_metrics_used_for_configuration": False,
    }
    _atomic_write_json(output_path, audit)
    return audit


def complete_test_access_audit(
    output_path: Path,
    audit: Mapping[str, Any],
    *,
    evaluation_count: int,
) -> dict[str, Any]:
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
                {"event": "FINAL_TEST_OPENED_FOR_EVALUATION", "timestamp_utc": audit["test_access_activated_at_utc"]},
                {"event": "CAMPAIGN_COMPLETED", "timestamp_utc": completed_at},
            ],
        }
    )
    _atomic_write_json(output_path, completed)
    return completed


def evaluate_final_test_campaign(
    configuration_plans: Sequence[Mapping[str, Any]],
    classifiers: Mapping[str, Mapping[str, Any]],
    workloads: Any,
    winner_lock: Mapping[str, Any],
    *,
    model_factory: Callable[..., Any] = create_model,
    metric_evaluator: Callable[[pd.DataFrame, pd.DataFrame], Mapping[str, Any]] = evaluate_detection,
) -> list[dict[str, Any]]:
    """Fit 40 governed train-only models and evaluate the fixed test workload."""
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
                    raise V08DError("Feature order changed during governed projection.")
                model = model_factory(
                    classifier,
                    features,
                    dict(model_spec["parameters"]),
                    random_state=seed,
                )
                _verify_model(model, classifier, features, seed, model_spec)
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
                metrics = _complete_metrics(raw_metrics)
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
                        "schema_version": V08D_SCHEMA_VERSION,
                        "stage": V08D_STAGE,
                        "configuration_id": plan["configuration_id"],
                        "source_configuration_id": plan["source_configuration_id"],
                        "selection_scope": plan["selection_scope"],
                        "selection_semantics": plan["selection_semantics"],
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
                        "source_winner_semantic_lock_sha256": winner_lock[
                            "semantic_lock_sha256"
                        ],
                        "metrics": metrics,
                        "model_fit_on_test": False,
                        "preprocessing_fit_on_test": False,
                        "feature_selection_on_test": False,
                        "threshold_tuned_on_test": False,
                        "test_access_authorized": True,
                    }
                )
    validate_raw_campaign(rows, configuration_plans, classifiers, winner_lock)
    return rows


def validate_raw_campaign(
    rows: Sequence[Mapping[str, Any]],
    plans: Sequence[Mapping[str, Any]],
    classifiers: Mapping[str, Mapping[str, Any]],
    winner_lock: Mapping[str, Any],
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
        len(rows) == 40
        and len(observed) == 40
        and observed == expected
        and tuple(plan_by_id) == CONFIGURATION_IDS
        and set(classifiers) == set(CLASSIFIERS)
        and all(row["stage"] == V08D_STAGE for row in rows)
        and all(row["training_split"] == "train" for row in rows)
        and all(row["evaluation_split"] == "test" for row in rows)
        and all(row["model_fit_on_test"] is False for row in rows)
        and all(row["preprocessing_fit_on_test"] is False for row in rows)
        and all(row["feature_selection_on_test"] is False for row in rows)
        and all(row["threshold_tuned_on_test"] is False for row in rows)
        and all(row["prediction_threshold"] == 0.5 for row in rows)
        and all(
            row["source_winner_semantic_lock_sha256"]
            == winner_lock["semantic_lock_sha256"]
            for row in rows
        )
        and all(_all_finite(row["metrics"].values()) for row in rows)
    )
    if not valid:
        raise V08DError("The governed final-test campaign is incomplete or invalid.")
    bpso = [row for row in rows if row["configuration_id"] == "BPSO-K10"]
    if any(
        tuple(row["selected_features"]) != tuple(winner_lock["ordered_selected_features"])
        or row["selected_features_sha256"] != winner_lock["selected_features_sha256"]
        for row in bpso
    ):
        raise V08DError("The final-test campaign did not use the exact locked BPSO subset.")


def aggregate_final_test(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = (
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
        "selected_feature_visibility_rate",
    )
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
                raise V08DError("Aggregation requires exactly five paired seeds per cell.")
            for metric in metrics:
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
        "schema_version": V08D_SCHEMA_VERSION,
        "stage": V08D_STAGE,
        "artifact_kind": "FINAL_TEST_AGGREGATION",
        "scientific_unit": "paired model/attack seed",
        "summaries": summaries,
        "test_driven_selection": False,
    }


def paired_comparisons(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = []
    for classifier in CLASSIFIERS:
        bpso = {
            row["seed"]: row
            for row in rows
            if row["classifier"] == classifier
            and row["configuration_id"] == "BPSO-K10"
        }
        for baseline in ("K43", "K42", "K11"):
            reference = {
                row["seed"]: row
                for row in rows
                if row["classifier"] == classifier
                and row["configuration_id"] == baseline
            }
            if set(bpso) != set(reference) or set(bpso) != set(MODEL_ATTACK_SEEDS):
                raise V08DError("Paired comparison seed alignment failed.")
            for metric in ("average_precision", "f1", "recall"):
                pairs = [
                    {
                        "seed": seed,
                        "bpso_value": float(bpso[seed]["metrics"][metric]),
                        "baseline_value": float(reference[seed]["metrics"][metric]),
                        "difference_bpso_minus_baseline": float(
                            bpso[seed]["metrics"][metric]
                            - reference[seed]["metrics"][metric]
                        ),
                    }
                    for seed in MODEL_ATTACK_SEEDS
                ]
                differences = [row["difference_bpso_minus_baseline"] for row in pairs]
                mean = statistics.fmean(differences)
                std = statistics.stdev(differences)
                half_width = T_CRITICAL_95_DF4 * std / math.sqrt(5)
                low, high = mean - half_width, mean + half_width
                output.append(
                    {
                        "classifier": classifier,
                        "configuration_id": "BPSO-K10",
                        "reference_configuration_id": baseline,
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
        "schema_version": V08D_SCHEMA_VERSION,
        "stage": V08D_STAGE,
        "artifact_kind": "PAIRED_FINAL_TEST_COMPARISONS",
        "difference_definition": "BPSO-K10 minus baseline within matched seed",
        "confidence_interval": "two-sided 95% Student-t interval, n=5, df=4",
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
        for metric, margin in (
            ("average_precision", 0.05),
            ("f1", 0.05),
            ("recall", 0.10),
        ):
            baseline = float(lookup[("K43", classifier, metric)])
            candidate = float(lookup[("BPSO-K10", classifier, metric)])
            loss = max(0.0, (baseline - candidate) / baseline)
            checks.append(
                {
                    "metric": metric,
                    "k43_mean": baseline,
                    "bpso_k10_mean": candidate,
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
        "schema_version": V08D_SCHEMA_VERSION,
        "stage": V08D_STAGE,
        "artifact_kind": "FINAL_TEST_PRESERVATION",
        "margins": {"average_precision": 0.05, "f1": 0.05, "recall": 0.10},
        "margin_semantics": "preregistered descriptive engineering preservation criteria",
        "domain_validated_margins": False,
        "results": results,
        "cross_classifier_interpretation": interpretation,
        "winner_reselection_performed": False,
    }


def build_final_test_lock(
    winner_lock: Mapping[str, Any],
    classifiers: Mapping[str, Mapping[str, Any]],
    artifact_hashes: Mapping[str, str],
    preservation: Mapping[str, Any],
    *,
    started_at: str,
    completed_at: str,
    wall_time_sec: float,
    rerun_count: int,
) -> dict[str, Any]:
    preservation_status = {
        row["classifier"]: row["overall_preserved"] for row in preservation["results"]
    }
    lock = {
        "schema_version": V08D_SCHEMA_VERSION,
        "stage": V08D_STAGE,
        "status": "FINAL_TEST_EVALUATED",
        "source_winner_semantic_lock_sha256": winner_lock["semantic_lock_sha256"],
        "bpso_mask_sha256": winner_lock["mask_sha256"],
        "bpso_selected_features_sha256": winner_lock["selected_features_sha256"],
        "feature_manifest_sha256": winner_lock["feature_manifest_sha256"],
        "classifier_configurations": classifiers,
        "classifier_configurations_sha256": _json_sha256(classifiers),
        "seed_namespace": list(MODEL_ATTACK_SEEDS),
        "result_artifact_hashes": dict(artifact_hashes),
        "preservation_result": preservation_status,
        "test_access_audit_sha256": artifact_hashes["v08d_test_access_audit.json"],
        "execution_metadata": {
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "wall_time_sec": wall_time_sec,
            "governed_evaluation_count": 40,
            "rerun_count": rerun_count,
        },
        "winner_unchanged_statement": (
            "The V0.8-C winner was not modified after final-test access."
        ),
        "historical_test_statement": (
            "The final test was historically accessed in V0.6 but remained untouched by "
            "V0.8 search and selection."
        ),
        "optimizer_invoked": False,
        "feature_reselection_performed": False,
        "test_driven_tuning_performed": False,
        "resource_benchmark_executed": False,
        "direct_energy_measured": False,
    }
    lock["semantic_result_lock_sha256"] = final_test_lock_semantic_hash(lock)
    return lock


def final_test_lock_semantic_hash(lock: Mapping[str, Any]) -> str:
    payload = deepcopy_json(lock)
    payload.pop("semantic_result_lock_sha256", None)
    execution = dict(payload.get("execution_metadata", {}))
    for field in ("started_at_utc", "completed_at_utc", "wall_time_sec"):
        execution.pop(field, None)
    payload["execution_metadata"] = execution
    return _json_sha256(payload)


def verify_final_test_lock(lock: Mapping[str, Any]) -> None:
    required = {
        "stage",
        "status",
        "source_winner_semantic_lock_sha256",
        "bpso_mask_sha256",
        "bpso_selected_features_sha256",
        "feature_manifest_sha256",
        "classifier_configurations",
        "classifier_configurations_sha256",
        "seed_namespace",
        "result_artifact_hashes",
        "preservation_result",
        "test_access_audit_sha256",
        "semantic_result_lock_sha256",
    }
    valid = (
        required <= set(lock)
        and lock.get("stage") == V08D_STAGE
        and lock.get("status") == "FINAL_TEST_EVALUATED"
        and lock.get("source_winner_semantic_lock_sha256")
        == EXPECTED_WINNER["semantic_lock_sha256"]
        and lock.get("bpso_mask_sha256") == EXPECTED_WINNER["mask_sha256"]
        and lock.get("bpso_selected_features_sha256")
        == EXPECTED_WINNER["selected_features_sha256"]
        and tuple(lock.get("seed_namespace", ())) == MODEL_ATTACK_SEEDS
        and lock.get("classifier_configurations_sha256")
        == _json_sha256(lock.get("classifier_configurations"))
        and lock.get("optimizer_invoked") is False
        and lock.get("feature_reselection_performed") is False
        and lock.get("test_driven_tuning_performed") is False
        and lock.get("resource_benchmark_executed") is False
        and lock.get("direct_energy_measured") is False
        and final_test_lock_semantic_hash(lock)
        == lock.get("semantic_result_lock_sha256")
    )
    if not valid:
        raise V08DError("V0.8-D final-test result lock verification failed.")


def run_v08d(
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
        raise V08DError("V0.8-D final-test campaign is already locked; rerun prohibited.")
    started_at = _utc_now()
    wall_start = time.perf_counter()
    prerequisites = verify_v08c_prerequisites()
    immutable_before = snapshot_immutable_paths()
    winner_lock = load_verified_winner_lock()
    basis = basis_loader()
    configurations = build_configuration_plan(basis, winner_lock)
    classifiers = classifier_plan()
    audit_path = output / AUDIT_PATH.name
    access_audit = activate_test_access_audit(audit_path, prerequisites, winner_lock)

    benchmark_context = benchmark_context_loader()
    workloads = workload_loader(benchmark_context)
    rows = evaluate_final_test_campaign(
        configurations, classifiers, workloads, winner_lock
    )
    access_audit = complete_test_access_audit(
        audit_path, access_audit, evaluation_count=len(rows)
    )
    raw = {
        "schema_version": V08D_SCHEMA_VERSION,
        "stage": V08D_STAGE,
        "artifact_kind": "RAW_GOVERNED_FINAL_TEST_EVIDENCE",
        "selection_preceded_test_access": True,
        "source_winner_semantic_lock_sha256": winner_lock["semantic_lock_sha256"],
        "configuration_plan": list(configurations),
        "classifier_plan": classifiers,
        "rows": rows,
        "governed_evaluation_count": len(rows),
        "test_driven_selection": False,
    }
    summary = aggregate_final_test(rows)
    paired = paired_comparisons(rows)
    preservation = preservation_analysis(summary)
    _atomic_write_json(output / RAW_PATH.name, raw)
    _atomic_write_json(output / SUMMARY_PATH.name, summary)
    _atomic_write_json(output / PAIRED_PATH.name, paired)
    _atomic_write_json(output / PRESERVATION_PATH.name, preservation)
    scientific_artifact_hashes = {
        RAW_PATH.name: sha256_file(output / RAW_PATH.name),
        SUMMARY_PATH.name: sha256_file(output / SUMMARY_PATH.name),
        PAIRED_PATH.name: sha256_file(output / PAIRED_PATH.name),
        PRESERVATION_PATH.name: sha256_file(output / PRESERVATION_PATH.name),
        AUDIT_PATH.name: sha256_file(audit_path),
    }
    completed_at = _utc_now()
    result_lock = build_final_test_lock(
        winner_lock,
        classifiers,
        scientific_artifact_hashes,
        preservation,
        started_at=started_at,
        completed_at=completed_at,
        wall_time_sec=time.perf_counter() - wall_start,
        rerun_count=int(access_audit["rerun_count"]),
    )
    _atomic_write_json(result_lock_path, result_lock)
    verify_final_test_lock(_read_json(result_lock_path))
    immutable_after = snapshot_immutable_paths()
    if immutable_after != immutable_before:
        raise V08DError("Frozen V0.6-V0.8-C artifacts changed during final-test evaluation.")
    execution = {
        "schema_version": V08D_SCHEMA_VERSION,
        "stage": V08D_STAGE,
        "status": "COMPLETED",
        "readiness_for_v08e": "GO",
        "starting_head": prerequisites["starting_head"],
        "prerequisites": prerequisites,
        "winner": {
            "source_optimizer_seed": winner_lock["source_optimizer_seed"],
            "selected_feature_count": winner_lock["selected_feature_count"],
            "mask_sha256": winner_lock["mask_sha256"],
            "selected_features_sha256": winner_lock["selected_features_sha256"],
            "semantic_lock_sha256": winner_lock["semantic_lock_sha256"],
        },
        "configuration_ids": list(CONFIGURATION_IDS),
        "classifiers": list(CLASSIFIERS),
        "seeds": list(MODEL_ATTACK_SEEDS),
        "governed_evaluation_count": len(rows),
        "campaign_wall_time_sec": time.perf_counter() - wall_start,
        "campaign_rerun_count": access_audit["rerun_count"],
        "preservation": {
            row["classifier"]: row["overall_preserved"]
            for row in preservation["results"]
        },
        "cross_classifier_interpretation": preservation[
            "cross_classifier_interpretation"
        ],
        "artifact_hashes": {
            **scientific_artifact_hashes,
            RESULT_LOCK_PATH.name: sha256_file(result_lock_path),
        },
        "semantic_result_lock_sha256": result_lock["semantic_result_lock_sha256"],
        "immutable_hashes_before": immutable_before,
        "immutable_hashes_after": immutable_after,
        "winner_modified_after_test_access": False,
        "optimizer_invoked": False,
        "feature_reselection_performed": False,
        "test_driven_tuning_performed": False,
        "resource_benchmark_executed": False,
        "direct_energy_measured": False,
    }
    _atomic_write_json(output / EXECUTION_PATH.name, execution)
    verify_v08d_artifacts(output)
    return execution


def verify_v08d_artifacts(output: Path = OUTPUT_DIR) -> None:
    execution = _read_json(output / EXECUTION_PATH.name)
    lock = _read_json(output / RESULT_LOCK_PATH.name)
    verify_final_test_lock(lock)
    audit = _read_json(output / AUDIT_PATH.name)
    raw = _read_json(output / RAW_PATH.name)
    if (
        execution.get("status") != "COMPLETED"
        or execution.get("governed_evaluation_count") != 40
        or execution.get("winner_modified_after_test_access") is not False
        or audit.get("status") != "PASS"
        or audit.get("winner_lock_verified_before_test_access") is not True
        or audit.get("governed_evaluation_count") != 40
        or raw.get("governed_evaluation_count") != 40
        or raw.get("test_driven_selection") is not False
    ):
        raise V08DError("V0.8-D artifact governance verification failed.")
    for name, expected in execution["artifact_hashes"].items():
        if sha256_file(output / name) != expected:
            raise V08DError(f"V0.8-D artifact hash mismatch: {name}")
    for name, expected in lock["result_artifact_hashes"].items():
        if sha256_file(output / name) != expected:
            raise V08DError(f"V0.8-D result-lock hash mismatch: {name}")


def _verify_model(
    model: Any,
    classifier: str,
    features: Sequence[str],
    seed: int,
    specification: Mapping[str, Any],
) -> None:
    if (
        getattr(model, "model_name", None) != classifier
        or tuple(getattr(model, "feature_names", ())) != tuple(features)
        or dict(getattr(model, "parameters", {})) != dict(specification["parameters"])
        or getattr(model, "random_state", None) != seed
        or specification["prediction_threshold"] != 0.5
    ):
        raise V08DError("Constructed classifier differs from its frozen configuration.")


def _complete_metrics(raw: Mapping[str, Any]) -> dict[str, float | int]:
    required = (
        "average_precision",
        "pr_auc",
        "f1",
        "recall",
        "precision",
        "roc_auc",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "evaluated_records",
        "attacked_records",
    )
    if any(raw.get(name) is None for name in required):
        raise V08DError("A required final-test metric is undefined.")
    tp = int(raw["true_positives"])
    tn = int(raw["true_negatives"])
    fp = int(raw["false_positives"])
    fn = int(raw["false_negatives"])
    evaluated = int(raw["evaluated_records"])
    attacked = int(raw["attacked_records"])
    return {
        "average_precision": float(raw["average_precision"]),
        "pr_auc_trapezoidal": float(raw["pr_auc"]),
        "pr_auc_trapezoidal_method": str(raw["pr_auc_method"]),
        "f1": float(raw["f1"]),
        "recall": float(raw["recall"]),
        "precision": float(raw["precision"]),
        "roc_auc": float(raw["roc_auc"]),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": float((tp + tn) / evaluated),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "attack_prevalence": float(attacked / evaluated),
        "evaluated_records": evaluated,
        "attacked_records": attacked,
    }


def _all_finite(values: Sequence[Any]) -> bool:
    return all(
        isinstance(value, str) or math.isfinite(float(value))
        for value in values
    )


def deepcopy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


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
        raise V08DError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V08DError(f"JSON artifact must be an object: {path}")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
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
    result = run_v08d()
    print(
        json.dumps(
            {
                "stage": V08D_STAGE,
                "status": result["status"],
                "governed_evaluation_count": result["governed_evaluation_count"],
                "preservation": result["preservation"],
                "semantic_result_lock_sha256": result[
                    "semantic_result_lock_sha256"
                ],
                "readiness_for_v08e": result["readiness_for_v08e"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
