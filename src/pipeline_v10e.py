"""V1.0-E governed one-time final-test evaluation of the locked Hybrid winner.

This stage loads the already-frozen V1.0-D hybrid BPSO+BGWO winner lock (and the
frozen V0.8-C BPSO K10 and V0.9-D BGWO K14 reference locks), authorizes final-test
access **only** for evaluation, fits the frozen decision-tree and
logistic-regression configurations on the training split, scores the held-out
test split, and reports paired statistics, preservation, feature reduction, and
feature overlap. It never runs an optimizer or feature selector, never tunes a
classifier or threshold, never reselects the winner, and never accesses final
test before the V1.0-D winner lock is verified and access is explicitly
authorized.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.lightweight.models import create_model
from src.pipeline_v08b import FrozenBasis, load_frozen_basis
import src.pipeline_v08d as v08d
import src.pipeline_v09e as v09e
import src.pipeline_v10b as v10b
import src.pipeline_v10c as v10c
import src.pipeline_v10d as v10d
import src.pipeline_v07b as v07b
from src.security.evaluation import detector_visible_mask, evaluate_detection
from src.security.experiment_data import fingerprint_feature_names
from src.security.ground_truth import assert_no_attack_metadata


V10E_STAGE = "V1.0-E"
V10E_SCHEMA_VERSION = "v1.0-e-governed-final-test-1"
STARTING_CHECKPOINT = "e7465e9"
STARTING_COMMIT = "e7465e91fa36b8d1496eb55901666b5e5ee0531e"
PROVENANCE_POLICY = "HEAD_AGNOSTIC_ANCESTRY"
TEST_ACCESS_CLASSIFICATION = "TEST_USED_ONLY_AFTER_LOCK_FOR_FINAL_EVALUATION"
TEST_AUTHORIZATION_REASON = "POST_WINNER_LOCK_FINAL_TEST_EVALUATION_ONLY"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "hybrid" / "v10e"
V10D_OUTPUT_DIR = PROJECT_ROOT / "results" / "hybrid" / "v10d"
HYBRID_WINNER_LOCK_PATH = V10D_OUTPUT_DIR / "v10d_winner_lock.json"
BPSO_WINNER_LOCK_PATH = PROJECT_ROOT / "results" / "bpso" / "v08c_winner_lock.json"
BGWO_WINNER_LOCK_PATH = PROJECT_ROOT / "results" / "bgwo" / "v09d" / "v09d_winner_lock.json"

EXPECTED_DATASET = "DataCo SMART Supply Chain"
EXPECTED_DIMENSIONS = 43
EXPECTED_TEST_ROWS = 6000
EXPECTED_TEST_SPLIT = {"train": 28000, "validation": 6000, "test": 6000}

MODEL_ATTACK_SEEDS = (42, 43, 44, 45, 46)
CLASSIFIERS = ("decision_tree", "logistic_regression")
CONFIGURATION_IDS = ("K43", "K42", "MI-K11", "BPSO-K10", "BGWO-K14", "HYBRID-K13")
CANDIDATE_COMPARISON_IDS = ("HYBRID-K13",)
PRIMARY_METRICS = ("average_precision", "f1", "recall", "precision", "roc_auc")
PRESERVATION_METRICS = ("average_precision", "f1", "recall")
PRESERVATION_MARGINS = {"average_precision": 0.05, "f1": 0.05, "recall": 0.10}
T_CRITICAL_95_DF4 = 2.7764451051977987
CONFIGURATION_FEATURE_COUNTS = (43, 42, 11, 10, 14, 13)
EXPECTED_EVALUATION_COUNT = len(CONFIGURATION_IDS) * len(CLASSIFIERS) * len(MODEL_ATTACK_SEEDS)
EXPECTED_PAIRED_COMPARISON_COUNT = (
    len(CANDIDATE_COMPARISON_IDS)
    * (len(CONFIGURATION_IDS) - 1)
    * len(CLASSIFIERS)
    * len(PRIMARY_METRICS)
)

EXPECTED_HYBRID_WINNER = {
    "optimizer_seed": 3045,
    "selected_feature_count": 13,
    "mask_sha256": "d1cc8c8b8643ce5d5daff3bdb6ff4c74dab6b1d78069a7f0052bcda10cb68c62",
    "feature_list_sha256": "8be4b0818f6ca59a057ef91d47738fa692548a6ddbed983b4496c644296b906a",
    "feature_manifest_sha256": "5146fd08fe766979adaf443bf9f4f7d32ee3c94cfdaf0fd46ec0efd10e92a10d",
    "semantic_lock_sha256": "1c258dc1a90ad78197d25e62bbc9fca24904cbcb77edafd24b12726059e49d84",
}
EXPECTED_HYBRID_FEATURES = (
    "order_item_quantity",
    "product_price",
    "order_item_total",
    "is_weekend",
    "Type_DEBIT",
    "Type_TRANSFER",
    "Type_CASH",
    "Market_LATAM",
    "Market_Pacific Asia",
    "Shipping Mode_First Class",
    "Department Name_Golf",
    "Department Name_Fitness",
    "Department Name_Health and Beauty",
)

V10E_TRACKED_PATHS = (
    "src/pipeline_v10b.py",
    "src/pipeline_v10c.py",
    "src/pipeline_v10d.py",
    "tests/test_pipeline_v10d.py",
    "results/hybrid/v10d/v10d_run_3042.json",
    "results/hybrid/v10d/v10d_run_3043.json",
    "results/hybrid/v10d/v10d_run_3044.json",
    "results/hybrid/v10d/v10d_run_3045.json",
    "results/hybrid/v10d/v10d_run_3046.json",
    "results/hybrid/v10d/v10d_stability_summary.json",
    "results/hybrid/v10d/v10d_winner_lock.json",
    "results/hybrid/v10d/v10d_test_access_audit.json",
    "results/hybrid/v10d/v10d_validation_comparison.json",
    "results/hybrid/v10d/v10d_execution_summary.json",
)
IMMUTABLE_PATHS = tuple(
    dict.fromkeys(
        (
            *v10c.FROZEN_ARTIFACT_PATHS,
            *(PROJECT_ROOT / path for path in V10E_TRACKED_PATHS),
        )
    )
)

PREFLIGHT_PATH = OUTPUT_DIR / "v10e_preflight.json"
AUTHORIZATION_PATH = OUTPUT_DIR / "v10e_test_authorization.json"
RAW_CSV_PATH = OUTPUT_DIR / "v10e_final_test_raw.csv"
SUMMARY_CSV_PATH = OUTPUT_DIR / "v10e_final_test_summary.csv"
PAIRED_CSV_PATH = OUTPUT_DIR / "v10e_paired_comparisons.csv"
PRESERVATION_PATH = OUTPUT_DIR / "v10e_preservation_analysis.json"
REDUCTION_PATH = OUTPUT_DIR / "v10e_feature_reduction.json"
OVERLAP_PATH = OUTPUT_DIR / "v10e_feature_overlap.json"
IMMUTABILITY_BEFORE_PATH = OUTPUT_DIR / "v10e_immutability_before.json"
IMMUTABILITY_AFTER_PATH = OUTPUT_DIR / "v10e_immutability_after.json"
AUDIT_PATH = OUTPUT_DIR / "v10e_test_access_audit.json"
EXECUTION_PATH = OUTPUT_DIR / "v10e_execution_summary.json"
RESULT_LOCK_PATH = OUTPUT_DIR / "v10e_result_lock.json"

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


class V10EError(RuntimeError):
    """Raised when V1.0-E final-test governance or evidence fails closed."""


def _read_json(path: Path | str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V10EError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V10EError(f"JSON artifact must be an object: {path}")
    return value


def _atomic_write_json(path: Path | str, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _deepcopy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _json_sha256(value: Any) -> str:
    return v10d._sha256_bytes(v10d._canonical_json(value).encode("utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_head_short(root: Path = PROJECT_ROOT) -> str:
    return v10b.current_head_short(root)


def snapshot_immutable_paths(
    paths: Sequence[Path] = IMMUTABLE_PATHS,
) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise V10EError(f"Missing frozen V0.6-V1.0-D prerequisite: {missing}")
    return {str(path.resolve()): v10d._sha256_file(path) for path in paths}


# ---------------------------------------------------------------------------
# Frozen lock loading
# ---------------------------------------------------------------------------


def load_verified_hybrid_winner_lock(
    path: Path = HYBRID_WINNER_LOCK_PATH,
) -> dict[str, Any]:
    lock = _read_json(path)
    semantic = v10d.verify_semantic_lock(lock, V10D_OUTPUT_DIR)
    if semantic.get("status") != "PASS":
        raise V10EError("V1.0-D semantic lock verification failed.")
    exact = all(lock.get(name) == value for name, value in EXPECTED_HYBRID_WINNER.items())
    mask = np.asarray(lock.get("mask", ()), dtype=np.uint8)
    features = tuple(lock.get("ordered_selected_features", ()))
    if (
        not exact
        or lock.get("stage") != "V1.0-D"
        or lock.get("selection_scope") != "TRAIN_AND_DEVELOPMENT_VALIDATION_ONLY"
        or lock.get("test_accessed") is not False
        or lock.get("test_used_for_winner_selection") is not False
        or lock.get("final_test_evaluated") is not False
        or mask.shape != (EXPECTED_DIMENSIONS,)
        or int(mask.sum()) != EXPECTED_HYBRID_WINNER["selected_feature_count"]
        or v10d._mask_sha256(mask) != EXPECTED_HYBRID_WINNER["mask_sha256"]
        or features != EXPECTED_HYBRID_FEATURES
        or fingerprint_feature_names(features) != EXPECTED_HYBRID_WINNER["feature_list_sha256"]
        or lock.get("feature_manifest_sha256")
        != EXPECTED_HYBRID_WINNER["feature_manifest_sha256"]
    ):
        raise V10EError("V1.0-D hybrid winner lock does not match the authorized winner.")
    return lock


def load_verified_bpso_winner_lock(
    path: Path = BPSO_WINNER_LOCK_PATH,
) -> dict[str, Any]:
    return v09e.load_verified_bpso_winner_lock(path)


def load_verified_bgwo_winner_lock(
    path: Path = BGWO_WINNER_LOCK_PATH,
) -> dict[str, Any]:
    return v09e.load_verified_bgwo_winner_lock(path)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def verify_v10d_evidence(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Validate the committed V1.0-D completion evidence required as a gate."""
    try:
        summary = _read_json(V10D_OUTPUT_DIR / "v10d_execution_summary.json")
        winner = _read_json(HYBRID_WINNER_LOCK_PATH)
        access = _read_json(V10D_OUTPUT_DIR / "v10d_test_access_audit.json")
    except V10EError:
        return {"status": "FAIL", "reason": "V1.0-D evidence artifacts missing"}
    checks = {
        "v10d_summary_status_pass": summary.get("status") == "PASS",
        "v10d_final_test_not_evaluated": summary.get("final_test_evaluated") is False,
        "v10d_winner_seed_3045": summary.get("winner_seed") == 3045,
        "v10d_winner_k_13": summary.get("winner_k") == 13,
        "v10d_winner_semantic_matches_lock": summary.get("winner_semantic_lock_sha256")
        == winner.get("semantic_lock_sha256"),
        "v10d_winner_semantic_expected": winner.get("semantic_lock_sha256")
        == EXPECTED_HYBRID_WINNER["semantic_lock_sha256"],
        "v10d_test_access_classification": access.get("classification")
        == v10d.TEST_ACCESS_CLASSIFICATION,
        "v10d_test_access_pass": access.get("status") == "PASS",
        "v10d_test_used_for_winner_selection_false": winner.get(
            "test_used_for_winner_selection"
        )
        is False,
    }
    return {
        "schema_version": V10E_SCHEMA_VERSION,
        "stage": V10E_STAGE,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def verify_v10e_preflight(
    *,
    expected_checkpoint: str | None = STARTING_CHECKPOINT,
    basis: FrozenBasis | None = None,
    hybrid_lock: Mapping[str, Any] | None = None,
    bpso_lock: Mapping[str, Any] | None = None,
    bgwo_lock: Mapping[str, Any] | None = None,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Verify every frozen prerequisite before any final-test access."""
    observed_head = current_head_short(root)

    checkpoint = expected_checkpoint or STARTING_CHECKPOINT
    checkpoint_commit_exists = v10d._commit_exists(root, checkpoint)
    checkpoint_is_ancestor = v10d._is_ancestor_of_head(root, checkpoint)
    if expected_checkpoint is None:
        head_check = True
    else:
        head_check = expected_checkpoint == observed_head
        if not (checkpoint_commit_exists and checkpoint_is_ancestor):
            raise V10EError(
                f"V10E_NO_GO: starting checkpoint {checkpoint} is not in HEAD history."
            )

    resolved_hybrid = (
        dict(hybrid_lock)
        if hybrid_lock is not None
        else load_verified_hybrid_winner_lock()
    )
    resolved_bpso = (
        dict(bpso_lock) if bpso_lock is not None else load_verified_bpso_winner_lock()
    )
    resolved_bgwo = (
        dict(bgwo_lock) if bgwo_lock is not None else load_verified_bgwo_winner_lock()
    )
    if hybrid_lock is not None:
        semantic = v10d.verify_semantic_lock(resolved_hybrid, V10D_OUTPUT_DIR)
        if semantic.get("status") != "PASS":
            raise V10EError("V10E_NO_GO: injected hybrid winner lock is not self-consistent.")
    if bpso_lock is not None:
        v08d.verify_winner_lock(resolved_bpso)
    if bgwo_lock is not None:
        v09e.v09d.verify_winner_lock(resolved_bgwo)
    if (
        resolved_hybrid.get("semantic_lock_sha256")
        != EXPECTED_HYBRID_WINNER["semantic_lock_sha256"]
    ):
        raise V10EError("V10E_NO_GO: hybrid winner semantic lock hash differs.")

    resolved_basis = basis if basis is not None else load_frozen_basis()
    roles = resolved_basis.lock.get("roles", {})
    k11_by_seed = roles.get("best_supervised", {}).get("selected_features", {})
    baseline_metrics = roles.get("full_baseline", {}).get("metrics", {})
    preprocessing = resolved_basis.dataset_metadata.get("preprocessing", {})
    dataset_split = resolved_basis.dataset_metadata.get("split", {})

    v10d_evidence = verify_v10d_evidence(root)

    hybrid_mask = np.asarray(resolved_hybrid.get("mask", ()), dtype=np.uint8)
    hybrid_features = tuple(resolved_hybrid.get("ordered_selected_features", ()))

    checks = {
        "hybrid_winner_mask_recomputed": hybrid_mask.shape == (EXPECTED_DIMENSIONS,)
        and int(hybrid_mask.sum()) == EXPECTED_HYBRID_WINNER["selected_feature_count"]
        and v10d._mask_sha256(hybrid_mask) == EXPECTED_HYBRID_WINNER["mask_sha256"]
        and resolved_hybrid.get("mask_sha256") == EXPECTED_HYBRID_WINNER["mask_sha256"],
        "hybrid_winner_features_recomputed": hybrid_features == EXPECTED_HYBRID_FEATURES
        and fingerprint_feature_names(hybrid_features)
        == EXPECTED_HYBRID_WINNER["feature_list_sha256"],
        "hybrid_winner_feature_manifest": resolved_hybrid.get("feature_manifest_sha256")
        == EXPECTED_HYBRID_WINNER["feature_manifest_sha256"],
        "starting_checkpoint": head_check,
        "starting_checkpoint_commit_exists": checkpoint_commit_exists,
        "starting_checkpoint_in_history": checkpoint_is_ancestor,
        "live_head_never_pinned": True,
        "hybrid_winner_lock_verified": resolved_hybrid.get("stage") == "V1.0-D"
        and resolved_hybrid.get("selection_scope") == "TRAIN_AND_DEVELOPMENT_VALIDATION_ONLY",
        "hybrid_winner_exact": resolved_hybrid.get("mask_sha256")
        == EXPECTED_HYBRID_WINNER["mask_sha256"]
        and resolved_hybrid.get("feature_list_sha256")
        == EXPECTED_HYBRID_WINNER["feature_list_sha256"],
        "hybrid_winner_validation_only_provenance": resolved_hybrid.get("test_accessed")
        is False
        and resolved_hybrid.get("test_used_for_winner_selection") is False
        and resolved_hybrid.get("final_test_evaluated") is False,
        "bpso_winner_lock_verified": resolved_bpso.get("stage") == "V0.8-C"
        and resolved_bpso.get("eligible_for_v08d") is True,
        "bpso_winner_exact": resolved_bpso.get("mask_sha256")
        == v09e.EXPECTED_BPSO_WINNER["mask_sha256"],
        "bgwo_winner_lock_verified": resolved_bgwo.get("stage") == "V0.9-D"
        and resolved_bgwo.get("eligible_for_v09e") is True,
        "bgwo_winner_exact": resolved_bgwo.get("mask_sha256")
        == v09e.EXPECTED_BGWO_WINNER["mask_sha256"],
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
        "v10d_implementation_tracked": v10d._git_tracked(root, V10E_TRACKED_PATHS),
        "v10d_evidence_valid": v10d_evidence.get("status") == "PASS",
        "test_not_accessed_by_preflight": True,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise V10EError(f"V1.0-E preflight failed: {failed}")

    frozen_snapshot = snapshot_immutable_paths()
    return {
        "schema_version": V10E_SCHEMA_VERSION,
        "stage": V10E_STAGE,
        "status": "PASS",
        "starting_head": observed_head,
        "starting_checkpoint": STARTING_CHECKPOINT,
        "provenance_policy": PROVENANCE_POLICY,
        "hybrid_winner_lock_sha256": v10d._sha256_file(HYBRID_WINNER_LOCK_PATH),
        "hybrid_winner_semantic_lock_sha256": resolved_hybrid["semantic_lock_sha256"],
        "hybrid_winner_mask_sha256": resolved_hybrid["mask_sha256"],
        "hybrid_winner_feature_list_sha256": resolved_hybrid["feature_list_sha256"],
        "hybrid_winner_feature_manifest_sha256": resolved_hybrid["feature_manifest_sha256"],
        "hybrid_winner_selected_feature_count": int(
            resolved_hybrid["selected_feature_count"]
        ),
        "bpso_winner_semantic_lock_sha256": resolved_bpso["semantic_lock_sha256"],
        "bpso_winner_mask_sha256": resolved_bpso["mask_sha256"],
        "bgwo_winner_semantic_lock_sha256": resolved_bgwo["semantic_lock_sha256"],
        "bgwo_winner_mask_sha256": resolved_bgwo["mask_sha256"],
        "dataset_identity": {
            "name": EXPECTED_DATASET,
            "split_seed": dataset_split.get("seed"),
            "split_sizes": dict(dataset_split.get("sizes", {})),
            "test_rows": int(dataset_split.get("sizes", {}).get("test", 0)),
            "dimensions": EXPECTED_DIMENSIONS,
            "feature_manifest_sha256": resolved_basis.candidate_manifest_sha256,
        },
        "v10d_evidence": v10d_evidence,
        "checks": checks,
        "frozen_immutable_hash_count": len(frozen_snapshot),
    }


# ---------------------------------------------------------------------------
# Configuration + classifier plans
# ---------------------------------------------------------------------------


def build_configuration_plan(
    basis: FrozenBasis,
    hybrid_lock: Mapping[str, Any],
    bpso_lock: Mapping[str, Any],
    bgwo_lock: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    candidates = basis.candidate_features
    k11_by_seed = basis.lock["roles"]["best_supervised"]["selected_features"]
    if set(k11_by_seed) != {str(seed) for seed in MODEL_ATTACK_SEEDS}:
        raise V10EError("Historical MI-K11 features do not cover exact seeds 42-46.")
    bpso_features = tuple(bpso_lock["ordered_selected_features"])
    bgwo_features = tuple(bgwo_lock["ordered_selected_features"])
    hybrid_features = tuple(hybrid_lock["ordered_selected_features"])
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
            "configuration_id": "BGWO-K14",
            "source_configuration_id": "v09d_locked_bgwo_winner",
            "feature_count": 14,
            "selection_scope": "V0.9-D train and development-validation lock",
            "selection_semantics": "one universal locked BGWO subset across all seeds",
            "source_semantic_lock_sha256": bgwo_lock["semantic_lock_sha256"],
            "features_by_seed": {
                str(seed): list(bgwo_features) for seed in MODEL_ATTACK_SEEDS
            },
        },
        {
            "configuration_id": "HYBRID-K13",
            "source_configuration_id": "v10d_locked_hybrid_winner",
            "feature_count": 13,
            "selection_scope": "V1.0-D train and development-validation lock",
            "selection_semantics": "one universal locked Hybrid subset across all seeds",
            "source_semantic_lock_sha256": hybrid_lock["semantic_lock_sha256"],
            "features_by_seed": {
                str(seed): list(hybrid_features) for seed in MODEL_ATTACK_SEEDS
            },
        },
    )
    if tuple(plan["configuration_id"] for plan in plans) != CONFIGURATION_IDS:
        raise V10EError(
            "Final-test configuration plan must contain exactly "
            "K43/K42/MI-K11/BPSO-K10/BGWO-K14/HYBRID-K13."
        )
    for plan in plans:
        if plan["feature_count"] != len(plan["features_by_seed"]["42"]):
            raise V10EError(f"Configuration feature count mismatch: {plan['configuration_id']}")
        for seed in MODEL_ATTACK_SEEDS:
            features = tuple(plan["features_by_seed"][str(seed)])
            if (
                len(features) != plan["feature_count"]
                or len(set(features)) != len(features)
                or tuple(feature for feature in candidates if feature in features) != features
            ):
                raise V10EError(
                    f"Configuration feature identity/order mismatch: "
                    f"{plan['configuration_id']} seed {seed}"
                )
    if len({tuple(k11_by_seed[str(seed)]) for seed in MODEL_ATTACK_SEEDS}) <= 1:
        raise V10EError("MI-K11 must preserve its historical seed-specific semantics.")
    if tuple(plans[3]["features_by_seed"]["42"]) != bpso_features:
        raise V10EError("BPSO-K10 plan does not match the locked BPSO subset.")
    if tuple(plans[4]["features_by_seed"]["42"]) != bgwo_features:
        raise V10EError("BGWO-K14 plan does not match the locked BGWO subset.")
    if tuple(plans[5]["features_by_seed"]["42"]) != hybrid_features:
        raise V10EError("HYBRID-K13 plan does not match the locked Hybrid subset.")
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
        raise V10EError("Frozen DT/LR classifier configuration changed.")
    return plan


# ---------------------------------------------------------------------------
# Test authorization gate
# ---------------------------------------------------------------------------


def activate_test_authorization(
    output_path: Path | str,
    preflight: Mapping[str, Any],
    hybrid_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist the evaluation-only authorization before any test access."""
    path = Path(output_path)
    if path.with_name(RESULT_LOCK_PATH.name).exists():
        raise V10EError("V1.0-E result lock already exists; campaign rerun prohibited.")
    authorized_at = _utc_now()
    authorization = {
        "schema_version": V10E_SCHEMA_VERSION,
        "stage": V10E_STAGE,
        "status": "TEST_AUTHORIZED",
        "artifact_kind": "FINAL_TEST_EVALUATION_AUTHORIZATION",
        "starting_head": preflight.get("starting_head"),
        "starting_checkpoint": STARTING_CHECKPOINT,
        "provenance_policy": PROVENANCE_POLICY,
        "test_authorized_before": False,
        "test_authorized": True,
        "test_authorization_reason": TEST_AUTHORIZATION_REASON,
        "classification": TEST_ACCESS_CLASSIFICATION,
        "winner_lock_verified_before_test_access": True,
        "winner_lock_verified_at_utc": authorized_at,
        "test_access_activated_at_utc": authorized_at,
        "source_winner_stage": "V1.0-D",
        "source_winner_semantic_lock_sha256": hybrid_lock["semantic_lock_sha256"],
        "hybrid_winner_lock_sha256": preflight.get("hybrid_winner_lock_sha256"),
        "locked_mask_sha256": hybrid_lock["mask_sha256"],
        "locked_feature_list_sha256": hybrid_lock["feature_list_sha256"],
        "locked_feature_manifest_sha256": hybrid_lock["feature_manifest_sha256"],
        "checks": {
            "winner_lock_verified_before_test_access": True,
            "optimizer_invoked": False,
            "feature_selection_invoked": False,
            "winner_replacement_allowed": False,
            "hyperparameter_search_invoked": False,
            "threshold_tuning_invoked": False,
            "bpso_rerun": False,
            "bgwo_rerun": False,
            "hybrid_optimizer_rerun": False,
            "test_driven_configuration_change": False,
            "test_driven_threshold_change": False,
            "test_driven_classifier_tuning": False,
            "final_test_used_only_for_evaluation": True,
        },
        "events": [
            {"event": "WINNER_LOCK_VERIFIED", "timestamp_utc": authorized_at},
            {"event": "FINAL_TEST_EVALUATION_AUTHORIZED", "timestamp_utc": authorized_at},
        ],
        "test_used_for_evaluation_only": True,
        "test_used_for_selection": False,
        "test_used_for_tuning": False,
        "test_used_for_feature_selection": False,
        "test_used_for_winner_selection": False,
        "final_test_accessed": False,
        "campaign_completed": False,
        "governed_evaluation_count": 0,
        "rerun_count": 0,
        "partial_metrics_used_for_configuration": False,
    }
    _atomic_write_json(path, authorization)
    return authorization


def assert_test_access_authorized(authorization: Mapping[str, Any]) -> None:
    """Fail closed unless final-test access was explicitly authorized."""
    authorized = (
        authorization.get("status") in {"TEST_AUTHORIZED", "PASS"}
        and authorization.get("test_authorized") is True
        and authorization.get("winner_lock_verified_before_test_access") is True
        and authorization.get("test_used_for_evaluation_only") is True
        and authorization.get("test_used_for_selection") is False
        and authorization.get("test_used_for_tuning") is False
        and authorization.get("test_used_for_feature_selection") is False
        and authorization.get("test_used_for_winner_selection") is False
    )
    if not authorized:
        raise V10EError(
            "Final test is inaccessible without explicit evaluation-only authorization."
        )


def load_authorized_test_workloads(
    context: Any,
    authorization: Mapping[str, Any],
    *,
    loader: Callable[[Any], Any] = v07b.prepare_verified_workloads,
) -> Any:
    """Load the frozen test workloads only after authorization is proven."""
    assert_test_access_authorized(authorization)
    return loader(context)


def complete_test_authorization(
    output_path: Path | str,
    authorization: Mapping[str, Any],
    *,
    evaluation_count: int,
) -> dict[str, Any]:
    path = Path(output_path)
    completed = dict(authorization)
    completed_at = _utc_now()
    completed.update(
        {
            "status": "PASS",
            "final_test_accessed": True,
            "campaign_completed": True,
            "governed_evaluation_count": evaluation_count,
            "campaign_completed_at_utc": completed_at,
            "events": [
                *authorization["events"],
                {
                    "event": "FINAL_TEST_OPENED_FOR_EVALUATION",
                    "timestamp_utc": authorization["test_access_activated_at_utc"],
                },
                {"event": "CAMPAIGN_COMPLETED", "timestamp_utc": completed_at},
            ],
        }
    )
    _atomic_write_json(path, completed)
    return completed


# ---------------------------------------------------------------------------
# Governed evaluation
# ---------------------------------------------------------------------------


def evaluate_final_test_campaign(
    configuration_plans: Sequence[Mapping[str, Any]],
    classifiers: Mapping[str, Mapping[str, Any]],
    workloads: Any,
    *,
    model_factory: Callable[..., Any] = create_model,
    metric_evaluator: Callable[
        [pd.DataFrame, pd.DataFrame], Mapping[str, Any]
    ] = evaluate_detection,
) -> list[dict[str, Any]]:
    """Fit exactly 60 governed train-only models and score the fixed test split."""
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
                if list(train_x.columns) != list(features) or list(test_x.columns) != list(
                    features
                ):
                    raise V10EError("Feature order changed during governed projection.")
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
                        "schema_version": V10E_SCHEMA_VERSION,
                        "stage": V10E_STAGE,
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
        and all(row["stage"] == V10E_STAGE for row in rows)
        and all(row["training_split"] == "train" for row in rows)
        and all(row["evaluation_split"] == "test" for row in rows)
        and all(row["model_fit_on_test"] is False for row in rows)
        and all(row["preprocessing_fit_on_test"] is False for row in rows)
        and all(row["feature_selection_on_test"] is False for row in rows)
        and all(row["threshold_tuned_on_test"] is False for row in rows)
        and all(row["prediction_threshold"] == 0.5 for row in rows)
        and all(row["test_access_authorized"] is True for row in rows)
        and all(
            int(row["test_row_count"]) == int(row["metrics"]["evaluated_records"])
            for row in rows
        )
        and all(v08d._all_finite(row["metrics"].values()) for row in rows)
    )
    if not valid:
        raise V10EError("The governed final-test campaign is incomplete or invalid.")
    for configuration in ("BPSO-K10", "BGWO-K14", "HYBRID-K13"):
        plan = plan_by_id[configuration]
        locked = tuple(plan["features_by_seed"]["42"])
        subset = [row for row in rows if row["configuration_id"] == configuration]
        if any(
            tuple(row["selected_features"]) != locked
            or row["selected_features_sha256"] != fingerprint_feature_names(locked)
            for row in subset
        ):
            raise V10EError(
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
                raise V10EError("Aggregation requires exactly five paired seeds per cell.")
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
        "schema_version": V10E_SCHEMA_VERSION,
        "stage": V10E_STAGE,
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
                raise V10EError("Paired comparison candidate seed coverage failed.")
            for reference_id in CONFIGURATION_IDS:
                if reference_id == candidate_id:
                    continue
                pair_key = frozenset((candidate_id, reference_id))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                reference = by_configuration[reference_id]
                if set(reference) != set(MODEL_ATTACK_SEEDS):
                    raise V10EError("Paired comparison reference seed coverage failed.")
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
                    differences = [
                        row["difference_candidate_minus_reference"] for row in pairs
                    ]
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
    if len(output) != EXPECTED_PAIRED_COMPARISON_COUNT:
        raise V10EError("Paired comparison count does not match the governed plan.")
    return {
        "schema_version": V10E_SCHEMA_VERSION,
        "stage": V10E_STAGE,
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
            candidate = float(lookup[("HYBRID-K13", classifier, metric)])
            relative_change = (candidate - baseline) / baseline
            loss = max(0.0, -relative_change)
            checks.append(
                {
                    "metric": metric,
                    "k43_mean": baseline,
                    "hybrid_k13_mean": candidate,
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
        "schema_version": V10E_SCHEMA_VERSION,
        "stage": V10E_STAGE,
        "artifact_kind": "FINAL_TEST_PRESERVATION",
        "reference_configuration_id": "K43",
        "candidate_configuration_id": "HYBRID-K13",
        "margins": dict(PRESERVATION_MARGINS),
        "margin_semantics": "preregistered descriptive engineering preservation criteria",
        "domain_validated_margins": False,
        "results": results,
        "cross_classifier_interpretation": interpretation,
        "winner_reselection_performed": False,
    }


def feature_reduction_analysis(
    configurations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total = len(configurations[0]["features_by_seed"]["42"])
    rows = []
    for plan in configurations:
        count = int(plan["feature_count"])
        reduced = total - count
        rows.append(
            {
                "configuration_id": plan["configuration_id"],
                "feature_count": count,
                "total_feature_count": total,
                "reduced_feature_count": reduced,
                "reduction_fraction": float(reduced / total),
                "reduction_percent": float(100.0 * reduced / total),
                "retained_fraction": float(count / total),
            }
        )
    return {
        "schema_version": V10E_SCHEMA_VERSION,
        "stage": V10E_STAGE,
        "artifact_kind": "FEATURE_REDUCTION_ANALYSIS",
        "descriptive_only": True,
        "total_feature_count": total,
        "configurations": rows,
    }


def feature_overlap_analysis(
    basis: FrozenBasis,
    hybrid_lock: Mapping[str, Any],
    bpso_lock: Mapping[str, Any],
    bgwo_lock: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = tuple(basis.candidate_features)
    order = {feature: index for index, feature in enumerate(candidates)}
    hybrid = tuple(hybrid_lock["ordered_selected_features"])
    bpso = tuple(bpso_lock["ordered_selected_features"])
    bgwo = tuple(bgwo_lock["ordered_selected_features"])
    mi_k11_by_seed = basis.lock["roles"]["best_supervised"]["selected_features"]

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

    def _pair(right: Sequence[str], right_label: str) -> dict[str, Any]:
        block = _set_block(hybrid, right)
        return {
            **block,
            "hybrid_only_features": block["left_only_features"],
            f"{right_label}_only_features": block["right_only_features"],
        }

    per_seed = []
    for seed in MODEL_ATTACK_SEEDS:
        mi_features = tuple(mi_k11_by_seed[str(seed)])
        block = _set_block(hybrid, mi_features)
        per_seed.append(
            {"seed": seed, "mi_k11_feature_count": len(mi_features), **block}
        )
    mi_union: set[str] = set()
    mi_intersection: set[str] | None = None
    for seed in MODEL_ATTACK_SEEDS:
        seed_features = set(mi_k11_by_seed[str(seed)])
        mi_union |= seed_features
        mi_intersection = (
            seed_features if mi_intersection is None else (mi_intersection & seed_features)
        )
    return {
        "schema_version": V10E_SCHEMA_VERSION,
        "stage": V10E_STAGE,
        "artifact_kind": "FEATURE_OVERLAP_ANALYSIS",
        "descriptive_only": True,
        "mutated_any_subset": False,
        "hybrid_feature_count": len(hybrid),
        "bpso_feature_count": len(bpso),
        "bgwo_feature_count": len(bgwo),
        "hybrid_vs_bpso": _pair(bpso, "bpso"),
        "hybrid_vs_bgwo": _pair(bgwo, "bgwo"),
        "hybrid_vs_mi_k11": {
            "per_seed": per_seed,
            "mean_seed_jaccard": statistics.fmean(row["jaccard"] for row in per_seed),
            "union_jaccard": float(
                len(set(hybrid) & mi_union) / len(set(hybrid) | mi_union)
            )
            if mi_union
            else 1.0,
            "intersection_jaccard": float(
                len(set(hybrid) & mi_intersection) / len(set(hybrid) | mi_intersection)
            )
            if mi_intersection
            else 1.0,
        },
    }


# ---------------------------------------------------------------------------
# Immutability + post-evaluation audit
# ---------------------------------------------------------------------------


def build_immutability_snapshot(
    phase: str,
    snapshot: Mapping[str, str],
) -> dict[str, Any]:
    if phase not in {"BEFORE", "AFTER"}:
        raise V10EError("Immutability snapshot phase must be BEFORE or AFTER.")
    return {
        "schema_version": V10E_SCHEMA_VERSION,
        "stage": V10E_STAGE,
        "artifact_kind": "IMMUTABILITY_SNAPSHOT",
        "snapshot_phase": phase,
        "status": "PASS",
        "hash_count": len(snapshot),
        "hashes": dict(snapshot),
    }


def build_test_access_audit(
    *,
    authorization: Mapping[str, Any],
    hybrid_lock_before: Mapping[str, Any],
    hybrid_lock_after: Mapping[str, Any],
    hybrid_lock_sha256_before: str,
    hybrid_lock_sha256_after: str,
    immutability_before: Mapping[str, Any],
    immutability_after: Mapping[str, Any],
    starting_head: str,
    ending_head: str,
    evaluation_count: int,
) -> dict[str, Any]:
    recomputed_mask_hash = v10d._mask_sha256(
        np.asarray(hybrid_lock_after["mask"], dtype=np.uint8)
    )
    mask_unchanged = list(hybrid_lock_before["mask"]) == list(hybrid_lock_after["mask"])
    features_unchanged = tuple(
        hybrid_lock_before["ordered_selected_features"]
    ) == tuple(hybrid_lock_after["ordered_selected_features"]) and hybrid_lock_before[
        "feature_list_sha256"
    ] == hybrid_lock_after["feature_list_sha256"]
    semantic_unchanged = (
        hybrid_lock_before["semantic_lock_sha256"]
        == hybrid_lock_after["semantic_lock_sha256"]
    )
    immutability_unchanged = (
        dict(immutability_before.get("hashes", {}))
        == dict(immutability_after.get("hashes", {}))
    )
    checks = {
        "test_accessed_only_after_winner_lock_verification": authorization.get(
            "winner_lock_verified_before_test_access"
        )
        is True,
        "test_authorized": authorization.get("test_authorized") is True,
        "winner_lock_verified_before_test_access": authorization.get(
            "winner_lock_verified_before_test_access"
        )
        is True,
        "hybrid_feature_mask_unchanged_before_after_test": mask_unchanged,
        "hybrid_feature_hash_unchanged": features_unchanged,
        "hybrid_semantic_lock_unchanged": semantic_unchanged,
        "hybrid_winner_lock_file_unchanged": hybrid_lock_sha256_before
        == hybrid_lock_sha256_after,
        "hybrid_mask_hash_recomputed_matches": recomputed_mask_hash
        == hybrid_lock_after["mask_sha256"],
        "immutable_artifacts_unchanged": immutability_unchanged,
        "no_optimizer_executed": True,
        "no_feature_selector_executed": True,
        "no_hyperparameter_search_executed": True,
        "no_threshold_tuning_executed": True,
        "no_winner_replacement_executed": True,
        "bpso_not_rerun": True,
        "bgwo_not_rerun": True,
        "hybrid_optimizer_not_rerun": True,
        "starting_head_unchanged": starting_head == ending_head,
        "governed_evaluation_count_is_sixty": evaluation_count
        == EXPECTED_EVALUATION_COUNT,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise V10EError(f"V1.0-E post-evaluation test-access audit failed: {failed}")
    return {
        "schema_version": V10E_SCHEMA_VERSION,
        "stage": V10E_STAGE,
        "status": "PASS",
        "artifact_kind": "POST_EVALUATION_TEST_ACCESS_AUDIT",
        "classification": TEST_ACCESS_CLASSIFICATION,
        "starting_head": starting_head,
        "ending_head": ending_head,
        "source_winner_stage": "V1.0-D",
        "winner_lock_verified_before_test_access": True,
        "winner_lock_verified_at_utc": authorization.get("winner_lock_verified_at_utc"),
        "test_access_activated_at_utc": authorization.get("test_access_activated_at_utc"),
        "campaign_completed_at_utc": authorization.get("campaign_completed_at_utc"),
        "hybrid_winner_semantic_lock_sha256": hybrid_lock_after["semantic_lock_sha256"],
        "hybrid_winner_mask_sha256": hybrid_lock_after["mask_sha256"],
        "hybrid_winner_feature_list_sha256": hybrid_lock_after["feature_list_sha256"],
        "test_authorized": True,
        "test_accessed": True,
        "test_used_for_evaluation": True,
        "test_used_for_evaluation_only": True,
        "test_used_for_selection": False,
        "test_used_for_tuning": False,
        "test_used_for_feature_selection": False,
        "test_used_for_winner_selection": False,
        "governed_evaluation_count": evaluation_count,
        "final_test_accessed": True,
        "final_test_opened": True,
        "campaign_completed": True,
        "rerun_count": int(authorization.get("rerun_count", 0)),
        "checks": checks,
        "hybrid_winner_lock_sha256_before": hybrid_lock_sha256_before,
        "hybrid_winner_lock_sha256_after": hybrid_lock_sha256_after,
        "immutability_before_hash_count": int(immutability_before.get("hash_count", 0)),
        "immutability_after_hash_count": int(immutability_after.get("hash_count", 0)),
    }


# ---------------------------------------------------------------------------
# Result lock
# ---------------------------------------------------------------------------


def build_result_lock(
    *,
    starting_head: str,
    hybrid_lock: Mapping[str, Any],
    bpso_lock: Mapping[str, Any],
    bgwo_lock: Mapping[str, Any],
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
        "schema_version": V10E_SCHEMA_VERSION,
        "stage": V10E_STAGE,
        "status": "FINAL_TEST_EVALUATED",
        "starting_head": starting_head,
        "starting_checkpoint": STARTING_CHECKPOINT,
        "provenance_policy": PROVENANCE_POLICY,
        "source_winner_stage": "V1.0-D",
        "classification": TEST_ACCESS_CLASSIFICATION,
        "hybrid_source_semantic_lock_sha256": hybrid_lock["semantic_lock_sha256"],
        "hybrid_selected_feature_count": int(hybrid_lock["selected_feature_count"]),
        "hybrid_ordered_selected_features": list(hybrid_lock["ordered_selected_features"]),
        "hybrid_mask_sha256": hybrid_lock["mask_sha256"],
        "hybrid_feature_list_sha256": hybrid_lock["feature_list_sha256"],
        "hybrid_feature_manifest_sha256": hybrid_lock["feature_manifest_sha256"],
        "bpso_source_semantic_lock_sha256": bpso_lock["semantic_lock_sha256"],
        "bpso_mask_sha256": bpso_lock["mask_sha256"],
        "bgwo_source_semantic_lock_sha256": bgwo_lock["semantic_lock_sha256"],
        "bgwo_mask_sha256": bgwo_lock["mask_sha256"],
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
        "test_used_for_winner_selection": False,
        "optimizer_invoked": False,
        "feature_reselection_performed": False,
        "test_driven_tuning_performed": False,
        "threshold_tuning_performed": False,
        "winner_replaced_after_test": False,
        "hybrid_optimizer_rerun": False,
        "bpso_rerun": False,
        "bgwo_rerun": False,
        "resource_benchmark_executed": False,
        "direct_energy_measured": False,
        "winner_unchanged_statement": (
            "The V1.0-D Hybrid winner was not modified after final-test access."
        ),
        "historical_test_statement": (
            "The final test was historically accessed in V0.6 but remained untouched by "
            "V1.0-D search and selection; V1.0-E accesses it once for governed evaluation only."
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
        "source_winner_stage",
        "hybrid_source_semantic_lock_sha256",
        "hybrid_selected_feature_count",
        "hybrid_ordered_selected_features",
        "hybrid_mask_sha256",
        "hybrid_feature_list_sha256",
        "bpso_source_semantic_lock_sha256",
        "bgwo_source_semantic_lock_sha256",
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
        and lock.get("stage") == V10E_STAGE
        and lock.get("status") == "FINAL_TEST_EVALUATED"
        and lock.get("source_winner_stage") == "V1.0-D"
        and lock.get("hybrid_source_semantic_lock_sha256")
        == EXPECTED_HYBRID_WINNER["semantic_lock_sha256"]
        and lock.get("hybrid_mask_sha256") == EXPECTED_HYBRID_WINNER["mask_sha256"]
        and lock.get("hybrid_feature_list_sha256")
        == EXPECTED_HYBRID_WINNER["feature_list_sha256"]
        and lock.get("hybrid_selected_feature_count")
        == EXPECTED_HYBRID_WINNER["selected_feature_count"]
        and tuple(lock.get("hybrid_ordered_selected_features", ()))
        == EXPECTED_HYBRID_FEATURES
        and lock.get("bpso_source_semantic_lock_sha256")
        == v09e.EXPECTED_BPSO_WINNER["semantic_lock_sha256"]
        and lock.get("bgwo_source_semantic_lock_sha256")
        == v09e.EXPECTED_BGWO_WINNER["semantic_lock_sha256"]
        and tuple(lock.get("configuration_ids", ())) == CONFIGURATION_IDS
        and tuple(row.get("configuration_id") for row in configs) == CONFIGURATION_IDS
        and lock.get("configuration_plan_sha256")
        == _json_sha256(lock.get("configuration_plan"))
        and lock.get("classifier_configurations_sha256")
        == _json_sha256(lock.get("classifier_configurations"))
        and tuple(lock.get("seed_namespace", ())) == MODEL_ATTACK_SEEDS
        and int(lock.get("test_dataset_identity", {}).get("test_rows", -1))
        == EXPECTED_TEST_ROWS
        and int(lock.get("governed_evaluation_count", -1)) == EXPECTED_EVALUATION_COUNT
        and lock.get("test_used_for_selection") is False
        and lock.get("test_used_for_tuning") is False
        and lock.get("test_used_for_feature_selection") is False
        and lock.get("test_used_for_winner_selection") is False
        and lock.get("optimizer_invoked") is False
        and lock.get("feature_reselection_performed") is False
        and lock.get("test_driven_tuning_performed") is False
        and lock.get("threshold_tuning_performed") is False
        and lock.get("winner_replaced_after_test") is False
        and lock.get("hybrid_optimizer_rerun") is False
        and lock.get("bpso_rerun") is False
        and lock.get("bgwo_rerun") is False
        and lock.get("resource_benchmark_executed") is False
        and lock.get("direct_energy_measured") is False
        and result_lock_semantic_hash(lock) == lock.get("semantic_result_lock_sha256")
    )
    if not valid:
        raise V10EError("V1.0-E final-test result lock verification failed.")


def write_or_verify_result_lock(
    path: Path | str, lock: Mapping[str, Any]
) -> dict[str, Any]:
    verify_result_lock(lock)
    target = Path(path)
    if target.exists():
        existing = _read_json(target)
        verify_result_lock(existing)
        if existing["semantic_result_lock_sha256"] != lock["semantic_result_lock_sha256"]:
            raise V10EError("Existing V1.0-E result lock differs from the recomputed result.")
        return existing
    _atomic_write_json(target, lock)
    stored = _read_json(target)
    verify_result_lock(stored)
    return stored


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def _write_raw_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    frame_rows = []
    for row in rows:
        entry = {
            "configuration_id": row["configuration_id"],
            "source_configuration_id": row["source_configuration_id"],
            "classification": "TEST_USED_ONLY_AFTER_LOCK_FOR_FINAL_EVALUATION",
            "classifier": row["classifier"],
            "seed": row["seed"],
            "selected_feature_count": row["selected_feature_count"],
            "selected_features_sha256": row["selected_features_sha256"],
            "source_semantic_lock_sha256": row["source_semantic_lock_sha256"],
            "prediction_threshold": row["prediction_threshold"],
            "model_fit_on_test": row["model_fit_on_test"],
            "preprocessing_fit_on_test": row["preprocessing_fit_on_test"],
            "feature_selection_on_test": row["feature_selection_on_test"],
            "threshold_tuned_on_test": row["threshold_tuned_on_test"],
            "test_access_authorized": row["test_access_authorized"],
        }
        entry.update({metric: row["metrics"][metric] for metric in METRIC_COLUMNS})
        frame_rows.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frame_rows).to_csv(path, index=False)


def _write_summary_csv(summary: Mapping[str, Any], path: Path) -> None:
    rows = [
        {
            "configuration_id": row["configuration_id"],
            "classifier": row["classifier"],
            "metric": row["metric"],
            "n": row["n"],
            "mean": row["mean"],
            "std": row["std"],
            "min": row["min"],
            "max": row["max"],
        }
        for row in summary["summaries"]
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


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


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_v10e(
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
        raise V10EError("V1.0-E final-test campaign is already locked; rerun prohibited.")

    stage_start = time.perf_counter()
    started_at = _utc_now()
    preflight = verify_v10e_preflight()
    starting_head = preflight["starting_head"]
    _atomic_write_json(output / PREFLIGHT_PATH.name, preflight)

    hybrid_lock = load_verified_hybrid_winner_lock()
    bpso_lock = load_verified_bpso_winner_lock()
    bgwo_lock = load_verified_bgwo_winner_lock()
    basis = basis_loader()
    configurations = build_configuration_plan(basis, hybrid_lock, bpso_lock, bgwo_lock)
    classifiers = classifier_plan()

    output.mkdir(parents=True, exist_ok=True)
    immutable_before = snapshot_immutable_paths()
    immutability_before = build_immutability_snapshot("BEFORE", immutable_before)
    _atomic_write_json(output / IMMUTABILITY_BEFORE_PATH.name, immutability_before)
    hybrid_lock_sha_before = v10d._sha256_file(HYBRID_WINNER_LOCK_PATH)
    hybrid_lock_before = _deepcopy_json(hybrid_lock)

    authorization_path = output / AUTHORIZATION_PATH.name
    authorization = activate_test_authorization(authorization_path, preflight, hybrid_lock)

    benchmark_context = benchmark_context_loader()
    workloads = load_authorized_test_workloads(
        benchmark_context, authorization, loader=workload_loader
    )
    rows = evaluate_final_test_campaign(configurations, classifiers, workloads)
    if len(rows) != EXPECTED_EVALUATION_COUNT:
        raise V10EError("Governed evaluation count is not exactly sixty.")
    authorization = complete_test_authorization(
        authorization_path, authorization, evaluation_count=len(rows)
    )

    summary = aggregate_final_test(rows)
    paired = paired_comparisons(rows)
    preservation = preservation_analysis(summary)
    reduction = feature_reduction_analysis(configurations)
    overlap = feature_overlap_analysis(basis, hybrid_lock, bpso_lock, bgwo_lock)

    _write_raw_csv(rows, output / RAW_CSV_PATH.name)
    _write_summary_csv(summary, output / SUMMARY_CSV_PATH.name)
    _write_paired_csv(paired, output / PAIRED_CSV_PATH.name)
    _atomic_write_json(output / PRESERVATION_PATH.name, preservation)
    _atomic_write_json(output / REDUCTION_PATH.name, reduction)
    _atomic_write_json(output / OVERLAP_PATH.name, overlap)

    hybrid_lock_after = _read_json(HYBRID_WINNER_LOCK_PATH)
    v10d.verify_semantic_lock(hybrid_lock_after, V10D_OUTPUT_DIR)
    hybrid_lock_sha_after = v10d._sha256_file(HYBRID_WINNER_LOCK_PATH)
    immutable_after = snapshot_immutable_paths()
    immutability_after = build_immutability_snapshot("AFTER", immutable_after)
    _atomic_write_json(output / IMMUTABILITY_AFTER_PATH.name, immutability_after)
    if immutable_after != immutable_before:
        raise V10EError("V10E_NO_GO: frozen V0.6-V1.0-D artifacts changed during evaluation.")
    if hybrid_lock_sha_after != hybrid_lock_sha_before:
        raise V10EError("The V1.0-D winner lock changed during final-test evaluation.")

    ending_head = current_head_short(PROJECT_ROOT)
    audit = build_test_access_audit(
        authorization=authorization,
        hybrid_lock_before=hybrid_lock_before,
        hybrid_lock_after=hybrid_lock_after,
        hybrid_lock_sha256_before=hybrid_lock_sha_before,
        hybrid_lock_sha256_after=hybrid_lock_sha_after,
        immutability_before=immutability_before,
        immutability_after=immutability_after,
        starting_head=starting_head,
        ending_head=ending_head,
        evaluation_count=len(rows),
    )
    _atomic_write_json(output / AUDIT_PATH.name, audit)

    artifact_hashes = {
        name: v10d._sha256_file(output / name)
        for name in (
            PREFLIGHT_PATH.name,
            AUTHORIZATION_PATH.name,
            RAW_CSV_PATH.name,
            SUMMARY_CSV_PATH.name,
            PAIRED_CSV_PATH.name,
            PRESERVATION_PATH.name,
            REDUCTION_PATH.name,
            OVERLAP_PATH.name,
            IMMUTABILITY_BEFORE_PATH.name,
            IMMUTABILITY_AFTER_PATH.name,
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
        hybrid_lock=hybrid_lock,
        bpso_lock=bpso_lock,
        bgwo_lock=bgwo_lock,
        configurations=configurations,
        classifiers=classifiers,
        test_dataset_identity=test_dataset_identity,
        artifact_hashes=artifact_hashes,
        preservation=preservation,
        governed_evaluation_count=len(rows),
        started_at=started_at,
        completed_at=completed_at,
        wall_time_sec=time.perf_counter() - stage_start,
        rerun_count=int(authorization["rerun_count"]),
    )
    result_lock = write_or_verify_result_lock(result_lock_path, proposed_lock)

    execution = {
        "schema_version": V10E_SCHEMA_VERSION,
        "stage": V10E_STAGE,
        "status": "COMPLETED",
        "readiness_for_v10f": "GO",
        "starting_head": starting_head,
        "ending_head": ending_head,
        "starting_checkpoint": STARTING_CHECKPOINT,
        "provenance_policy": PROVENANCE_POLICY,
        "classification": TEST_ACCESS_CLASSIFICATION,
        "preflight": preflight,
        "dataset_identity": test_dataset_identity,
        "winner": {
            "optimizer": "GOVERNED_HYBRID_BPSO_BGWO",
            "source_optimizer_seed": hybrid_lock["optimizer_seed"],
            "selected_feature_count": hybrid_lock["selected_feature_count"],
            "mask_sha256": hybrid_lock["mask_sha256"],
            "feature_list_sha256": hybrid_lock["feature_list_sha256"],
            "feature_manifest_sha256": hybrid_lock["feature_manifest_sha256"],
            "semantic_lock_sha256": hybrid_lock["semantic_lock_sha256"],
        },
        "reference_optimizers": {
            "bpso_k10_semantic_lock_sha256": bpso_lock["semantic_lock_sha256"],
            "bgwo_k14_semantic_lock_sha256": bgwo_lock["semantic_lock_sha256"],
        },
        "configuration_ids": list(CONFIGURATION_IDS),
        "classifiers": list(CLASSIFIERS),
        "seeds": list(MODEL_ATTACK_SEEDS),
        "governed_evaluation_count": len(rows),
        "campaign_wall_time_sec": time.perf_counter() - stage_start,
        "campaign_rerun_count": authorization["rerun_count"],
        "preservation": {
            row["classifier"]: row["overall_preserved"] for row in preservation["results"]
        },
        "cross_classifier_interpretation": preservation["cross_classifier_interpretation"],
        "feature_reduction_percent": {
            row["configuration_id"]: row["reduction_percent"]
            for row in reduction["configurations"]
        },
        "feature_reduction": {
            "hybrid_k13_reduction_percent": reduction["configurations"][5][
                "reduction_percent"
            ]
        },
        "paired_comparison_count": len(paired["comparisons"]),
        "artifact_hashes": {
            **artifact_hashes,
            RESULT_LOCK_PATH.name: v10d._sha256_file(result_lock_path),
        },
        "semantic_result_lock_sha256": result_lock["semantic_result_lock_sha256"],
        "immutable_hashes_before_count": len(immutable_before),
        "immutable_hashes_after_count": len(immutable_after),
        "test_authorized": True,
        "test_accessed": True,
        "final_test_accessed": True,
        "test_used_for_evaluation_only": True,
        "test_used_for_selection": False,
        "test_used_for_tuning": False,
        "test_used_for_feature_selection": False,
        "test_used_for_winner_selection": False,
        "winner_modified_after_test_access": False,
        "optimizer_invoked": False,
        "feature_reselection_performed": False,
        "test_driven_tuning_performed": False,
        "threshold_tuning_performed": False,
        "hybrid_optimizer_rerun": False,
        "bpso_rerun": False,
        "bgwo_rerun": False,
        "resource_benchmark_executed": False,
        "direct_energy_measured": False,
    }
    _atomic_write_json(output / EXECUTION_PATH.name, execution)
    verify_v10e_artifacts(output)
    return execution


def verify_v10e_artifacts(output: Path | str = OUTPUT_DIR) -> None:
    directory = Path(output)
    execution = _read_json(directory / EXECUTION_PATH.name)
    lock = _read_json(directory / RESULT_LOCK_PATH.name)
    verify_result_lock(lock)
    audit = _read_json(directory / AUDIT_PATH.name)
    if (
        execution.get("status") != "COMPLETED"
        or execution.get("governed_evaluation_count") != EXPECTED_EVALUATION_COUNT
        or execution.get("winner_modified_after_test_access") is not False
        or execution.get("test_used_for_selection") is not False
        or audit.get("status") != "PASS"
        or audit.get("winner_lock_verified_before_test_access") is not True
        or audit.get("governed_evaluation_count") != EXPECTED_EVALUATION_COUNT
        or audit.get("test_used_for_evaluation_only") is not True
    ):
        raise V10EError("V1.0-E artifact governance verification failed.")
    for name, expected in execution["artifact_hashes"].items():
        if v10d._sha256_file(directory / name) != expected:
            raise V10EError(f"V1.0-E artifact hash mismatch: {name}")
    for name, expected in lock["result_artifact_hashes"].items():
        if v10d._sha256_file(directory / name) != expected:
            raise V10EError(f"V1.0-E result-lock hash mismatch: {name}")


def main() -> int:
    result = run_v10e()
    print(
        json.dumps(
            {
                "stage": V10E_STAGE,
                "status": result["status"],
                "governed_evaluation_count": result["governed_evaluation_count"],
                "preservation": result["preservation"],
                "cross_classifier_interpretation": result[
                    "cross_classifier_interpretation"
                ],
                "semantic_result_lock_sha256": result["semantic_result_lock_sha256"],
                "readiness_for_v10f": result["readiness_for_v10f"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
