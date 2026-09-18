"""V0.9-G artifact-only reporting and synthesis for the governed V0.9 BGWO study.

This module is the final reproducible reporting layer for the V0.9 BGWO
feature-selection evaluation. It is *strictly* artifact-only:

- it never imports optimizer execution paths (src.optimization.*),
- it never imports model-training / final-test evaluation paths,
- it never imports resource-campaign measurement paths,
- it only loads frozen V0.8/V0.9 lock, JSON and CSV evidence and derives
  reporting-only quantities from that evidence.

Every scientific number surfaced here is loaded from a frozen artifact under
``results/bpso/`` or ``results/bgwo/v09{d,e,f}/`` and is therefore traceable
back to that artifact. No BPSO/BGWO optimizer, final test or resource
campaign is ever (re)run by this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.config import PROJECT_ROOT

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

STAGE = "V0.9-G"
SCHEMA_VERSION = "v0.9-g-reporting-lock-1"
EXPECTED_HEAD_SHORT = "4e81f09"

CONFIGURATION_IDS = ("K43", "K42", "MI-K11", "BPSO-K10", "BGWO")
CLASSIFIERS = ("decision_tree", "logistic_regression")
METRICS = ("average_precision", "f1", "recall", "precision", "roc_auc")
FEATURE_SPACE_SIZE = 43
WORKING_ROWS = 40000
SPLIT = {"train": 28000, "validation": 6000, "test": 6000}

EXPECTED_DATASET = "DataCo SMART Supply Chain for Big Data Analysis"
EXPECTED_FEATURE_MANIFEST = "5146fd08fe766979adaf443bf9f4f7d32ee3c94cfdaf0fd46ec0efd10e92a10d"

EXPECTED_BGWO_SEMANTIC = "e228f4c619de7e2028e4089723c6ae066fe8c2ab9ed1d94d98e0439a63df2744"
EXPECTED_BGWO_MASK = "7ebb823374255f4f10c737c62a2111604aa50193a8f0d91b8483f3864cac7ad6"
EXPECTED_BGWO_FEATURES_SHA = "0d5219835f3dc96acb3c82f94d286813b834eceefc4156aa383234a842b9c359"
EXPECTED_BGWO_COUNT = 14
EXPECTED_BGWO_SEED = 2042
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

EXPECTED_BPSO_SEMANTIC = "0f356ccab422774b128ab82f5da8881f24816902d4ccc6a7cecc9760c6202ca8"
EXPECTED_BPSO_MASK = "5da981b5b87db97338ecdde9ca8a8b87db3a62771d03dc6a4ad901f6548a3299"
EXPECTED_BPSO_FEATURES_SHA = "5157d5bb6b17dc6c92b790671dac029e2b0d2b4454db9824870db1290a6c4121"
EXPECTED_BPSO_COUNT = 10
EXPECTED_BPSO_SEED = 1042
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

EXPECTED_V09E_RESULT_SEMANTIC = "8398f51ad4a17e559f741c1bc04690809903b7f688ab65fe4146fb1bc25d0556"
EXPECTED_V09F_RESULT_SEMANTIC = "d76795c7c329c52b2b97212ee15e5f4c32c55547d501308868a5ae6e12a3d2c1"

BGWO_RUN_SEEDS = ("2042", "2043", "2044", "2045", "2046")

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "bgwo" / "v09g_reporting"
DEFAULT_TABLE_DIR = DEFAULT_OUTPUT_DIR / "tables"
DEFAULT_FIGURE_DIR = DEFAULT_OUTPUT_DIR / "figures"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "v09_bgwo_feature_selection_evaluation.ipynb"
REPORT_PATH = PROJECT_ROOT / "docs" / "v09_bgwo_evaluation_report.md"
LOCK_PATH = DEFAULT_OUTPUT_DIR / "v09g_reporting_lock.json"

TABLE_FILENAMES = {
    "dataset_governance": "dataset_governance.csv",
    "feature_configurations": "feature_configurations.csv",
    "bgwo_validation_runs": "bgwo_validation_runs.csv",
    "final_test_dt": "final_test_dt.csv",
    "final_test_lr": "final_test_lr.csv",
    "paired_statistics": "paired_statistics.csv",
    "feature_overlap": "feature_overlap.csv",
    "resource_efficiency": "resource_efficiency.csv",
    "optimizer_overhead": "optimizer_overhead.csv",
    "bpso_vs_bgwo": "bpso_vs_bgwo.csv",
}

FIGURE_NAMES = (
    "01_bgwo_convergence",
    "02_bgwo_selected_k_by_seed",
    "03_bgwo_feature_selection_frequency",
    "04_feature_count_comparison",
    "05_dt_final_test",
    "06_lr_final_test",
    "07_bgwo_vs_bpso_paired_ci",
    "08_training_time_comparison",
    "09_inference_latency_comparison",
    "10_model_size_comparison",
    "11_optimizer_overhead_comparison",
    "12_predictive_resource_tradeoff",
)

SOURCE_EVIDENCE_PATHS = (
    "data/processed/dataset_metadata.json",
    "results/feature_selection/validation_lock.json",
    "results/feature_selection/validation_lock.sha256",
    "results/feature_selection/validation_summary.csv",
    # V0.8 BPSO evidence
    "results/bpso/v08a_protocol_validation.json",
    "results/bpso/v08c_execution_summary.json",
    "results/bpso/v08c_convergence.json",
    "results/bpso/v08c_stability.json",
    "results/bpso/v08c_winner_lock.json",
    "results/bpso/v08c_run_1042.json",
    "results/bpso/v08c_run_1043.json",
    "results/bpso/v08c_run_1044.json",
    "results/bpso/v08c_run_1045.json",
    "results/bpso/v08c_run_1046.json",
    "results/bpso/v08d_final_test_lock.json",
    "results/bpso/v08d_final_test_summary.json",
    "results/bpso/v08d_preservation.json",
    # V0.9-D BGWO validation evidence
    "results/bgwo/v09d/v09d_execution_summary.json",
    "results/bgwo/v09d/v09d_convergence.json",
    "results/bgwo/v09d/v09d_stability.json",
    "results/bgwo/v09d/v09d_validation_comparison.json",
    "results/bgwo/v09d/v09d_winner_lock.json",
    "results/bgwo/v09d/v09d_test_access_audit.json",
    "results/bgwo/v09d/v09d_run_2042.json",
    "results/bgwo/v09d/v09d_run_2043.json",
    "results/bgwo/v09d/v09d_run_2044.json",
    "results/bgwo/v09d/v09d_run_2045.json",
    "results/bgwo/v09d/v09d_run_2046.json",
    # V0.9-E final-test evidence
    "results/bgwo/v09e/v09e_execution_summary.json",
    "results/bgwo/v09e/v09e_test_summary.json",
    "results/bgwo/v09e/v09e_test_metrics.csv",
    "results/bgwo/v09e/v09e_paired_statistics.csv",
    "results/bgwo/v09e/v09e_feature_overlap.json",
    "results/bgwo/v09e/v09e_preservation_analysis.json",
    "results/bgwo/v09e/v09e_result_lock.json",
    "results/bgwo/v09e/v09e_test_access_audit.json",
    # V0.9-F resource evidence
    "results/bgwo/v09f/v09f_execution_summary.json",
    "results/bgwo/v09f/v09f_result_lock.json",
    "results/bgwo/v09f/v09f_resource_summary.json",
    "results/bgwo/v09f/v09f_resource_comparisons.json",
    "results/bgwo/v09f/v09f_model_size_comparison.json",
    "results/bgwo/v09f/v09f_optimizer_overhead.json",
    "results/bgwo/v09f/v09f_feature_reduction.json",
    "results/bgwo/v09f/v09f_timing_quality.json",
    "results/bgwo/v09f/v09f_tradeoff.json",
    "results/bgwo/v09f/v09f_break_even.json",
)

ENERGY_STATEMENT = (
    "DIRECT_ENERGY_UNAVAILABLE: no Joule, Watt, kWh, CO2 or CO2e measurements were "
    "collected. CPU time, wall-clock time, RSS and throughput are computational "
    "resource/efficiency proxies and are not direct physical-energy measurements. "
    "No energy value is derived from CPU TDP ratings; TDP multiplication is "
    "prohibited in this study."
)

PRESERVATION_STATEMENT = (
    "Preservation margins (AP loss <= 5%, F1 loss <= 5%, Recall loss <= 10%) are "
    "preregistered descriptive engineering criteria that governed validation and "
    "final-test interpretation. They are operational, not domain-validated "
    "clinical or security thresholds, and the final test was never used to "
    "reselect or retune anything."
)


class V09GError(RuntimeError):
    """Raised when governed V0.9-G reporting validations fail."""


def _read_json(path: Path | str) -> dict[str, Any]:
    payload = _read_json_any(path)
    if not isinstance(payload, dict):
        raise V09GError(f"JSON artifact must be an object: {path}")
    return payload


def _read_json_any(path: Path | str) -> Any:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V09GError(f"Cannot read JSON artifact: {target}") from exc
    return payload


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    target = Path(path)
    if not target.is_file():
        raise V09GError(f"Missing artifact for hashing: {target}")
    try:
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise V09GError(f"Cannot hash artifact: {target}") from exc
    return digest.hexdigest()


def locate_repository_root(start: Path | str = PROJECT_ROOT) -> Path:
    origin = Path(start).resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / "results" / "bgwo" / "v09d").is_dir() and (
            candidate / "docs"
        ).is_dir():
            return candidate
    raise V09GError("Could not locate repository root containing results/bgwo and docs/.")


def _git_head_short(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def current_head_short(root: Path | str = PROJECT_ROOT) -> str:
    return _git_head_short(locate_repository_root(Path(root)))


def _resolve_root(root: Path | str = PROJECT_ROOT) -> Path:
    return locate_repository_root(Path(root))


def _verify_file_manifest(manifest: Mapping[str, str], base_dir: Path) -> list[str]:
    mismatches: list[str] = []
    for filename, expected in manifest.items():
        artifact = base_dir / filename
        if not artifact.is_file() or sha256_file(artifact) != expected:
            mismatches.append(filename)
    return sorted(mismatches)


def verify_v09g_preflight(
    *,
    root: Path | str = PROJECT_ROOT,
    expected_head: str | None = EXPECTED_HEAD_SHORT,
) -> dict[str, Any]:
    """Verify governed V0.8/V0.9 evidence before any V0.9-G reporting output.

    This function is artifact-only. It never launches optimizer, training,
    final-test or resource-campaign execution.
    """

    repo_root = _resolve_root(root)
    observed_head = _git_head_short(repo_root)
    if expected_head and observed_head != expected_head:
        raise V09GError(f"V09G_NO_GO: expected HEAD {expected_head}, observed {observed_head}.")

    bgwo_lock = _read_json(repo_root / "results" / "bgwo" / "v09d" / "v09d_winner_lock.json")
    bpso_lock = _read_json(repo_root / "results" / "bpso" / "v08c_winner_lock.json")
    v09e_lock = _read_json(repo_root / "results" / "bgwo" / "v09e" / "v09e_result_lock.json")
    v09f_lock = _read_json(repo_root / "results" / "bgwo" / "v09f" / "v09f_result_lock.json")
    dataset_metadata = _read_json(repo_root / "data" / "processed" / "dataset_metadata.json")
    validation_lock = _read_json(
        repo_root / "results" / "feature_selection" / "validation_lock.json"
    )
    split = dataset_metadata.get("split", {})
    preprocessing = dataset_metadata.get("preprocessing", {})
    manifest = validation_lock.get("candidate_manifest", {})

    bgwo_ok = (
        bgwo_lock.get("stage") == "V0.9-D"
        and bgwo_lock.get("status") == "VALIDATION_LOCKED"
        and bgwo_lock.get("eligible_for_v09e") is True
        and bgwo_lock.get("semantic_lock_sha256") == EXPECTED_BGWO_SEMANTIC
        and bgwo_lock.get("mask_sha256") == EXPECTED_BGWO_MASK
        and bgwo_lock.get("selected_features_sha256") == EXPECTED_BGWO_FEATURES_SHA
        and int(bgwo_lock.get("selected_feature_count", -1)) == EXPECTED_BGWO_COUNT
        and int(bgwo_lock.get("source_optimizer_seed", -1)) == EXPECTED_BGWO_SEED
        and tuple(bgwo_lock.get("ordered_selected_features", ())) == EXPECTED_BGWO_FEATURES
        and bgwo_lock.get("selection_scope") == "TRAIN_AND_DEVELOPMENT_VALIDATION_ONLY"
        and bgwo_lock.get("final_test_accessed") is False
        and bgwo_lock.get("feature_manifest_sha256") == EXPECTED_FEATURE_MANIFEST
    )
    bpso_ok = (
        bpso_lock.get("stage") == "V0.8-C"
        and bpso_lock.get("status") == "VALIDATION_LOCKED"
        and bpso_lock.get("semantic_lock_sha256") == EXPECTED_BPSO_SEMANTIC
        and bpso_lock.get("mask_sha256") == EXPECTED_BPSO_MASK
        and bpso_lock.get("selected_features_sha256") == EXPECTED_BPSO_FEATURES_SHA
        and int(bpso_lock.get("selected_feature_count", -1)) == EXPECTED_BPSO_COUNT
        and int(bpso_lock.get("source_optimizer_seed", -1)) == EXPECTED_BPSO_SEED
        and tuple(bpso_lock.get("ordered_selected_features", ())) == EXPECTED_BPSO_FEATURES
        and bpso_lock.get("selection_scope") == "TRAIN_AND_DEVELOPMENT_VALIDATION_ONLY"
        and bpso_lock.get("final_test_accessed") is False
    )
    v09d_lock_sha = sha256_file(repo_root / "results" / "bgwo" / "v09d" / "v09d_winner_lock.json")
    v09e_ok = (
        v09e_lock.get("stage") == "V0.9-E"
        and v09e_lock.get("status") == "FINAL_TEST_EVALUATED"
        and v09e_lock.get("semantic_result_lock_sha256") == EXPECTED_V09E_RESULT_SEMANTIC
        and v09e_lock.get("source_winner_semantic_lock_sha256") == EXPECTED_BGWO_SEMANTIC
        and v09e_lock.get("bgwo_mask_sha256") == EXPECTED_BGWO_MASK
        and v09e_lock.get("bpso_source_semantic_lock_sha256") == EXPECTED_BPSO_SEMANTIC
        and v09e_lock.get("bpso_mask_sha256") == EXPECTED_BPSO_MASK
    )
    v09e_manifest_ok = not _verify_file_manifest(
        v09e_lock.get("result_artifact_hashes", {}),
        repo_root / "results" / "bgwo" / "v09e",
    )
    v09f_ok = (
        v09f_lock.get("stage") == "V0.9-F"
        and v09f_lock.get("status") == "RESOURCE_RESULT_LOCKED"
        and v09f_lock.get("semantic_result_lock_sha256") == EXPECTED_V09F_RESULT_SEMANTIC
        and v09f_lock.get("v09e_result_semantic_lock_sha256") == EXPECTED_V09E_RESULT_SEMANTIC
        and v09f_lock.get("source_winner_semantic_lock_sha256") == EXPECTED_BGWO_SEMANTIC
        and v09f_lock.get("bgwo_mask_sha256") == EXPECTED_BGWO_MASK
        and v09f_lock.get("bpso_mask_sha256") == EXPECTED_BPSO_MASK
        and v09f_lock.get("direct_energy_status") == "DIRECT_ENERGY_UNAVAILABLE"
        and v09f_lock.get("joule_estimation_from_cpu_or_wall_performed") is False
        and v09f_lock.get("optimizer_invoked") is False
    )
    v09f_manifest_ok = not _verify_file_manifest(
        v09f_lock.get("result_artifact_hashes", {}),
        repo_root / "results" / "bgwo" / "v09f",
    )

    frozen_snapshot = snapshot_evidence(repo_root)

    checks = {
        "starting_checkpoint": observed_head == EXPECTED_HEAD_SHORT,
        "bgwo_winner_lock_verified": bgwo_ok,
        "bpso_winner_lock_verified": bpso_ok,
        "v09e_result_lock_verified": v09e_ok,
        "v09e_artifact_hashes_verified": v09e_manifest_ok,
        "v09f_result_lock_verified": v09f_ok,
        "v09f_artifact_hashes_verified": v09f_manifest_ok,
        "dataset_identity": split.get("seed") == 42
        and split.get("strategy") == "order_grouped"
        and dict(split.get("sizes", {})) == SPLIT
        and preprocessing.get("fitted_on_rows") == 28000,
        "feature_manifest_identity": (
            bool(manifest.get("features")) and manifest.get("feature_count") == FEATURE_SPACE_SIZE
        )
        and manifest.get("sha256") == EXPECTED_FEATURE_MANIFEST,
        "test_locked_during_selection": (
            bgwo_lock.get("final_test_accessed") is False
            and bpso_lock.get("final_test_accessed") is False
        ),
        "evidence_snapshot_complete": len(frozen_snapshot) == len(SOURCE_EVIDENCE_PATHS),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise V09GError(f"V09G_NO_GO: preflight failed: {failed}")

    return {
        "stage": STAGE,
        "schema_version": SCHEMA_VERSION,
        "status": "GO",
        "head_short": observed_head,
        "v09d_winner_lock_sha256": v09d_lock_sha,
        "v09d_winner_semantic_lock_sha256": EXPECTED_BGWO_SEMANTIC,
        "v09e_result_semantic_lock_sha256": EXPECTED_V09E_RESULT_SEMANTIC,
        "v09f_result_semantic_lock_sha256": EXPECTED_V09F_RESULT_SEMANTIC,
        "bpso_winner_semantic_lock_sha256": EXPECTED_BPSO_SEMANTIC,
        "direct_energy_status": "DIRECT_ENERGY_UNAVAILABLE",
        "frozen_evidence_file_count": len(frozen_snapshot),
        "checks": checks,
    }


def snapshot_evidence(root: Path | str = PROJECT_ROOT) -> dict[str, str]:
    """Return {repo-relative_path: sha256} for every frozen source evidence file."""
    repo_root = _resolve_root(root)
    snapshot: dict[str, str] = {}
    for relative in SOURCE_EVIDENCE_PATHS:
        artifact = repo_root / relative
        if not artifact.is_file():
            raise V09GError(f"Missing frozen source evidence: {artifact}")
        snapshot[str(relative)] = sha256_file(artifact)
    return snapshot


def _resolve_relative(path: Path | str, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def load_v09g_artifacts(
    root: Path | str = PROJECT_ROOT,
    *,
    expected_head: str | None = EXPECTED_HEAD_SHORT,
) -> dict[str, Any]:
    """Load every frozen artifact consumed by the V0.9-G notebook/report."""
    repo_root = _resolve_root(root)
    preflight = verify_v09g_preflight(root=repo_root, expected_head=expected_head)

    bgwo_run_artifacts: dict[str, dict[str, Any]] = {}
    for seed in BGWO_RUN_SEEDS:
        bgwo_run_artifacts[seed] = _read_json(
            repo_root / "results" / "bgwo" / "v09d" / f"v09d_run_{seed}.json"
        )

    resource_summary = _read_json_any(
        repo_root / "results" / "bgwo" / "v09f" / "v09f_resource_summary.json"
    )

    return {
        "preflight": preflight,
        "evidence_snapshot": snapshot_evidence(repo_root),
        "dataset_metadata": _read_json(repo_root / "data" / "processed" / "dataset_metadata.json"),
        "feature_selection_validation_lock": _read_json(
            repo_root / "results" / "feature_selection" / "validation_lock.json"
        ),
        "feature_selection_validation_summary": pd.read_csv(
            repo_root / "results" / "feature_selection" / "validation_summary.csv"
        ),
        "v08a_protocol": _read_json(repo_root / "results" / "bpso" / "v08a_protocol_validation.json"),
        "v08c_execution": _read_json(repo_root / "results" / "bpso" / "v08c_execution_summary.json"),
        "v08c_convergence": _read_json(repo_root / "results" / "bpso" / "v08c_convergence.json"),
        "v08c_stability": _read_json(repo_root / "results" / "bpso" / "v08c_stability.json"),
        "v08c_winner_lock": _read_json(repo_root / "results" / "bpso" / "v08c_winner_lock.json"),
        "v08d_final_test_lock": _read_json(repo_root / "results" / "bpso" / "v08d_final_test_lock.json"),
        "v08d_final_test_summary": _read_json(
            repo_root / "results" / "bpso" / "v08d_final_test_summary.json"
        ),
        "v08d_preservation": _read_json(repo_root / "results" / "bpso" / "v08d_preservation.json"),
        "v09d_execution": _read_json(
            repo_root / "results" / "bgwo" / "v09d" / "v09d_execution_summary.json"
        ),
        "v09d_convergence": _read_json(
            repo_root / "results" / "bgwo" / "v09d" / "v09d_convergence.json"
        ),
        "v09d_stability": _read_json(repo_root / "results" / "bgwo" / "v09d" / "v09d_stability.json"),
        "v09d_validation_comparison": _read_json(
            repo_root / "results" / "bgwo" / "v09d" / "v09d_validation_comparison.json"
        ),
        "v09d_winner_lock": _read_json(
            repo_root / "results" / "bgwo" / "v09d" / "v09d_winner_lock.json"
        ),
        "bgwo_run_artifacts": bgwo_run_artifacts,
        "v09e_execution": _read_json(
            repo_root / "results" / "bgwo" / "v09e" / "v09e_execution_summary.json"
        ),
        "v09e_test_summary": _read_json(
            repo_root / "results" / "bgwo" / "v09e" / "v09e_test_summary.json"
        ),
        "v09e_test_metrics": pd.read_csv(
            repo_root / "results" / "bgwo" / "v09e" / "v09e_test_metrics.csv"
        ),
        "v09e_paired_statistics": pd.read_csv(
            repo_root / "results" / "bgwo" / "v09e" / "v09e_paired_statistics.csv"
        ),
        "v09e_feature_overlap": _read_json(
            repo_root / "results" / "bgwo" / "v09e" / "v09e_feature_overlap.json"
        ),
        "v09e_preservation": _read_json(
            repo_root / "results" / "bgwo" / "v09e" / "v09e_preservation_analysis.json"
        ),
        "v09e_result_lock": _read_json(
            repo_root / "results" / "bgwo" / "v09e" / "v09e_result_lock.json"
        ),
        "v09f_execution": _read_json(
            repo_root / "results" / "bgwo" / "v09f" / "v09f_execution_summary.json"
        ),
        "v09f_result_lock": _read_json(
            repo_root / "results" / "bgwo" / "v09f" / "v09f_result_lock.json"
        ),
        "v09f_resource_summary": resource_summary,
        "v09f_resource_comparisons": pd.read_csv(
            repo_root / "results" / "bgwo" / "v09f" / "v09f_resource_comparisons.csv"
        ),
        "v09f_model_size": _read_json(
            repo_root / "results" / "bgwo" / "v09f" / "v09f_model_size_comparison.json"
        ),
        "v09f_optimizer_overhead": _read_json(
            repo_root / "results" / "bgwo" / "v09f" / "v09f_optimizer_overhead.json"
        ),
        "v09f_feature_reduction": _read_json(
            repo_root / "results" / "bgwo" / "v09f" / "v09f_feature_reduction.json"
        ),
        "v09f_timing_quality": _read_json(
            repo_root / "results" / "bgwo" / "v09f" / "v09f_timing_quality.json"
        ),
        "v09f_tradeoff": _read_json(repo_root / "results" / "bgwo" / "v09f" / "v09f_tradeoff.json"),
        "v09f_break_even": _read_json(repo_root / "results" / "bgwo" / "v09f" / "v09f_break_even.json"),
    }


def _final_test_mean_frame(summary_payload: Mapping[str, Any], classifier: str) -> pd.DataFrame:
    rows_index: dict[str, dict[str, float]] = {}
    for entry in summary_payload.get("summaries", []):
        if entry.get("classifier") != classifier or entry.get("metric") not in METRICS:
            continue
        config = str(entry["configuration_id"])
        metric = str(entry["metric"])
        rows_index.setdefault(config, {})
        rows_index[config][f"mean_{metric}"] = float(entry["mean"])
        rows_index[config][f"std_{metric}"] = float(entry["std"])
    frame = pd.DataFrame.from_dict(rows_index, orient="index").reset_index()
    frame = frame.rename(columns={"index": "configuration_id"})
    frame["configuration_id"] = pd.Categorical(
        frame["configuration_id"], list(CONFIGURATION_IDS), ordered=True
    )
    frame = frame.sort_values("configuration_id").reset_index(drop=True)
    if len(frame) != len(CONFIGURATION_IDS):
        raise V09GError(f"Final-test summary incomplete for {classifier}.")
    return frame


def _resource_summary_value(
    rows: Sequence[Mapping[str, Any]], *, classifier: str, config: str, metric: str, phase: str
) -> float | None:
    for row in rows:
        if (
            row.get("classifier") == classifier
            and row.get("configuration_id") == config
            and row.get("metric") == metric
            and row.get("phase") == phase
        ):
            if row.get("mean_of_seed_means") is None:
                return None
            return float(row["mean_of_seed_means"])
    return None


def build_reporting_tables(artifacts: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Build the V0.9-G machine-readable reporting tables from frozen evidence."""

    # 1 ------------------------------------------------------------------ dataset governance
    dataset = artifacts["dataset_metadata"]
    governance = pd.DataFrame(
        [
            {"item": "dataset", "value": str(dataset.get("dataset", {}).get("name", EXPECTED_DATASET))},
            {"item": "working_rows", "value": WORKING_ROWS},
            {"item": "train_rows", "value": SPLIT["train"]},
            {"item": "validation_rows", "value": SPLIT["validation"]},
            {"item": "test_rows", "value": SPLIT["test"]},
            {"item": "split_strategy", "value": str(dataset.get("split", {}).get("strategy", "order_grouped"))},
            {"item": "split_seed", "value": dataset.get("split", {}).get("seed", 42)},
            {"item": "feature_space_dimensions", "value": FEATURE_SPACE_SIZE},
            {"item": "cyber_target", "value": "is_attack (controlled synthetic attack scenario)"},
            {"item": "test_availability", "value": "test unavailable during optimizer/feature selection; opened only after the BGWO winner lock"},
            {"item": "late_delivery_risk_ground_truth", "value": "Late_delivery_risk is NOT cyber ground truth"},
            {"item": "direct_energy_measured", "value": False},
        ]
    )

    # 2 ------------------------------------------------------------------ feature configurations
    feature_config_rows = [
        {
            "configuration_id": "K43",
            "feature_count": 43,
            "reduction_vs_k43_percent": 0.0,
            "selection_origin": "frozen V0.6 full baseline",
            "selection_category": "baseline",
            "test_used_during_selection": False,
        },
        {
            "configuration_id": "K42",
            "feature_count": 42,
            "reduction_vs_k43_percent": 100.0 * (43.0 - 42.0) / 43.0,
            "selection_origin": "frozen V0.6 unsupervised filter (pairwise correlation, natural)",
            "selection_category": "unsupervised",
            "test_used_during_selection": False,
        },
        {
            "configuration_id": "MI-K11",
            "feature_count": 11,
            "reduction_vs_k43_percent": 100.0 * (43.0 - 11.0) / 43.0,
            "selection_origin": "frozen V0.6 supervised MI select-k-best K11 (seed-specific feature set)",
            "selection_category": "supervised",
            "test_used_during_selection": False,
        },
        {
            "configuration_id": "BPSO-K10",
            "feature_count": 10,
            "reduction_vs_k43_percent": 100.0 * (43.0 - 10.0) / 43.0,
            "selection_origin": "governed V0.8 BPSO winner lock (v08c_winner_lock)",
            "selection_category": "metaheuristic",
            "test_used_during_selection": False,
        },
        {
            "configuration_id": "BGWO",
            "feature_count": 14,
            "reduction_vs_k43_percent": 100.0 * (43.0 - 14.0) / 43.0,
            "selection_origin": "governed V0.9-D BGWO winner lock (v09d_winner_lock)",
            "selection_category": "metaheuristic",
            "test_used_during_selection": False,
        },
    ]
    feature_configs = pd.DataFrame(feature_config_rows)
    feature_configs["test_used_during_selection"] = feature_configs[
        "test_used_during_selection"
    ].astype(bool)

    # 3 ------------------------------------------------------------------ BGWO validation runs
    v09d_exec = artifacts["v09d_execution"]
    stability = artifacts["v09d_stability"]
    run_rows: list[dict[str, Any]] = []
    for seed, run in artifacts["bgwo_run_artifacts"].items():
        best = run.get("best_evaluation", {})
        mean_metrics = dict(best.get("mean_metrics", {}))
        run_rows.append(
            {
                "optimizer_seed": int(run.get("optimizer_seed", seed)),
                "selected_feature_count": int(run.get("selected_feature_count", 0)),
                "validation_average_precision": float(mean_metrics.get("average_precision")),
                "validation_f1": float(mean_metrics.get("f1")),
                "validation_recall": float(mean_metrics.get("recall")),
                "candidate_requests": int(run.get("candidate_requests", 0)),
                "decision_tree_fits": int(run.get("actual_decision_tree_fits", 0)),
                "best_iteration": int(run.get("best_iteration", -1)),
                "evaluated_iteration_count": int(run.get("evaluated_iteration_count", 0)),
                "feasible": bool(best.get("feasible")) if best is not None else False,
                "stop_reason": str(run.get("stop_reason", "")),
            }
        )
    run_frame = pd.DataFrame(run_rows).sort_values("optimizer_seed").reset_index(drop=True)
    winner = v09d_exec.get("winner", {})
    summary_rows = [
        {
            "optimizer_seed": "ALL",
            "selected_feature_count": int(winner.get("selected_feature_count", 14)),
            "validation_average_precision": float(
                winner.get("validation_metrics", {}).get("average_precision")
            ),
            "validation_f1": float(winner.get("validation_metrics", {}).get("f1")),
            "validation_recall": float(winner.get("validation_metrics", {}).get("recall")),
            "candidate_requests": int(v09d_exec.get("candidate_requests", 0)),
            "decision_tree_fits": int(v09d_exec.get("actual_decision_tree_fits", 0)),
            "best_iteration": -1,
            "evaluated_iteration_count": -1,
            "feasible": True,
            "stop_reason": "AGGREGATE_WINNER",
        }
    ]
    validation_runs = pd.concat([run_frame, pd.DataFrame(summary_rows)], ignore_index=True)

    # 4-5 ---------------------------------------------------------------- final-test tables
    final_dt = _final_test_mean_frame(artifacts["v09e_test_summary"], "decision_tree")
    final_lr = _final_test_mean_frame(artifacts["v09e_test_summary"], "logistic_regression")

    # 6 ------------------------------------------------------------------ paired statistics
    paired = artifacts["v09e_paired_statistics"].copy()
    kept = paired[
        (paired["candidate_configuration_id"] == "BGWO")
        & (paired["reference_configuration_id"].isin(("K43", "BPSO-K10", "MI-K11")))
        & (paired["metric"].isin(METRICS))
    ].copy()
    kept["classifier"] = pd.Categorical(kept["classifier"], list(CLASSIFIERS), ordered=True)
    kept["reference_configuration_id"] = pd.Categorical(
        kept["reference_configuration_id"], ("K43", "BPSO-K10", "MI-K11"), ordered=True
    )
    paired_stats = kept.sort_values(
        ["classifier", "reference_configuration_id", "metric"]
    ).reset_index(drop=True)

    # 7 ------------------------------------------------------------------ feature overlap
    overlap = artifacts["v09e_feature_overlap"]
    bg_vs_bpso = overlap["bgwo_vs_bpso"]
    bg_vs_mi = overlap["bgwo_vs_mi_k11"]
    mi_seeds = bg_vs_mi.get("per_seed", [])
    mi_mean_intersection = (
        float(np.mean([float(s.get("intersection_size", 0)) for s in mi_seeds])) if mi_seeds else np.nan
    )
    mi_mean_union = float(np.mean([float(s.get("union_size", 0)) for s in mi_seeds])) if mi_seeds else np.nan

    def _join(names: Sequence[str] | list[str] | None) -> str:
        return "|".join(str(n) for n in (names or []))

    overlap_rows = [
        {
            "pair": "BGWO_vs_BPSO-K10",
            "bgwo_feature_count": 14,
            "other_feature_count": 10,
            "intersection_size": float(bg_vs_bpso["intersection_size"]),
            "union_size": float(bg_vs_bpso["union_size"]),
            "jaccard": float(bg_vs_bpso["jaccard"]),
            "mean_seed_jaccard": np.nan,
            "aggregation": "single_set",
            "shared_features": _join(bg_vs_bpso["shared_features"]),
            "bgwo_only_features": _join(bg_vs_bpso["bgwo_only_features"]),
            "other_only_features": _join(bg_vs_bpso["bpso_only_features"]),
        },
        {
            "pair": "BGWO_vs_MI-K11",
            "bgwo_feature_count": 14,
            "other_feature_count": 11,
            "intersection_size": mi_mean_intersection,
            "union_size": mi_mean_union,
            "jaccard": float(bg_vs_mi["union_jaccard"]),
            "mean_seed_jaccard": float(bg_vs_mi["mean_seed_jaccard"]),
            "aggregation": f"mean_across_{len(mi_seeds)}_mi_seeds; per-seed sets vary",
            "shared_features": "",
            "bgwo_only_features": "",
            "other_only_features": "",
        },
    ]
    feature_overlap = pd.DataFrame(overlap_rows)

    # 8 ------------------------------------------------------------------ resource efficiency
    resource = artifacts["v09f_resource_summary"]
    resource_rows: list[dict[str, Any]] = []
    for classifier in CLASSIFIERS:
        for config in CONFIGURATION_IDS:
            def _first_row(metric: str, phase: str) -> Mapping[str, Any] | None:
                for r in resource:
                    if (
                        r.get("classifier") == classifier
                        and r.get("configuration_id") == config
                        and r.get("metric") == metric
                        and r.get("phase") == phase
                    ):
                        return r
                return None

            training_row = _first_row("wall_clock_elapsed_sec", "training")
            inference_row = _first_row("per_record_latency_sec", "inference")
            model_size_row = next(
                (
                    row
                    for row in artifacts["v09f_model_size"]["entries"]
                    if row.get("classifier") == classifier
                    and row.get("configuration_id") == config
                ),
                None,
            )
            fc_value = training_row.get("feature_count") if training_row is not None else None
            if fc_value is None:
                raise V09GError(
                    f"Resource summary missing feature_count for {classifier}/{config}."
                )
            resource_rows.append(
                {
                    "classifier": classifier,
                    "configuration_id": config,
                    "feature_count": int(fc_value),
                    "training_wall_time_sec": _resource_summary_value(
                        resource, classifier=classifier, config=config,
                        metric="wall_clock_elapsed_sec", phase="training",
                    ),
                    "training_cpu_time_sec": _resource_summary_value(
                        resource, classifier=classifier, config=config,
                        metric="process_cpu_time_sec", phase="training",
                    ),
                    "training_absolute_rss_mib": _resource_summary_value(
                        resource, classifier=classifier, config=config,
                        metric="absolute_peak_rss_mib", phase="training",
                    ),
                    "training_input_memory_bytes": (
                        float(training_row["input_dataframe_memory_bytes"])
                        if training_row is not None
                        and training_row.get("input_dataframe_memory_bytes") is not None
                        else np.nan
                    ),
                    "inference_latency_per_record_sec": _resource_summary_value(
                        resource, classifier=classifier, config=config,
                        metric="per_record_latency_sec", phase="inference",
                    ),
                    "inference_throughput_records_sec": _resource_summary_value(
                        resource, classifier=classifier, config=config,
                        metric="throughput_records_sec", phase="inference",
                    ),
                    "inference_absolute_rss_mib": _resource_summary_value(
                        resource, classifier=classifier, config=config,
                        metric="absolute_peak_rss_mib", phase="inference",
                    ),
                    "inference_input_memory_bytes": (
                        float(inference_row["input_dataframe_memory_bytes"])
                        if inference_row is not None
                        and inference_row.get("input_dataframe_memory_bytes") is not None
                        else np.nan
                    ),
                    "serialized_model_bytes": (
                        float(model_size_row["serialized_model_bytes"])
                        if model_size_row is not None and model_size_row.get("serialized_model_bytes") is not None
                        else np.nan
                    ),
                }
            )
    resource_eff = pd.DataFrame(resource_rows)
    resource_eff["classifier"] = pd.Categorical(
        resource_eff["classifier"], list(CLASSIFIERS), ordered=True
    )
    resource_eff["configuration_id"] = pd.Categorical(
        resource_eff["configuration_id"], list(CONFIGURATION_IDS), ordered=True
    )
    resource_eff = resource_eff.sort_values(
        ["classifier", "configuration_id"]
    ).reset_index(drop=True)

    # 9 ------------------------------------------------------------------ optimizer overhead
    overhead = artifacts["v09f_optimizer_overhead"]
    overhead_rows = []
    for optimizer_name in ("bgwo", "bpso"):
        o = overhead[optimizer_name]
        overhead_rows.append(
            {
                "optimizer": str(o.get("optimizer", optimizer_name)).upper(),
                "optimizer_runs": int(o["optimizer_runs"]),
                "candidate_requests": int(o["candidate_requests"]),
                "unique_evaluations": int(o["unique_evaluations"]),
                "decision_tree_fits": int(o["decision_tree_fits"]),
                "optimizer_wall_time_sec": float(o["core_optimization_wall_time_sec"]),
                "optimizer_cpu_time_sec": float(o["core_optimization_cpu_time_sec"]),
                "pipeline_wall_time_sec": float(o["pipeline_wall_time_sec"]),
                "cache_hits": int(o["cache_hits"]),
                "provenance": o["measurement_provenance"],
                "source_stage": o["stage"],
            }
        )
    optimizer_overhead = pd.DataFrame(overhead_rows)

    # 10 ----------------------------------------------------------------- BPSO vs BGWO neutral comparison
    win_bgwo = artifacts["v09d_winner_lock"]
    win_bpso = artifacts["v08c_winner_lock"]
    bgwo_k = int(win_bgwo["selected_feature_count"])
    bpso_k = int(win_bpso["selected_feature_count"])

    def _mean_value(frame: pd.DataFrame, config: str, column: str) -> float:
        return float(frame.loc[frame["configuration_id"] == config, column].iloc[0])

    bg_vs_bpso_rows: list[dict[str, Any]] = [
        {
            "dimension": "feature_count",
            "value_type": "structural",
            "BPSO-K10": bpso_k,
            "BGWO-K14": bgwo_k,
            "interpretation": "BPSO-K10 locked fewer features than BGWO-K14; both are metaheuristic winners.",
        },
        {
            "dimension": "feature_reduction_vs_k43_percent",
            "value_type": "structural",
            "BPSO-K10": 100.0 * (43.0 - bpso_k) / 43.0,
            "BGWO-K14": 100.0 * (43.0 - bgwo_k) / 43.0,
            "interpretation": "Both reduce the 43-feature space; BPSO-K10 by more.",
        },
        {
            "dimension": "validation_average_precision",
            "value_type": "validation",
            "BPSO-K10": float(win_bpso["validation_metrics"]["average_precision"]),
            "BGWO-K14": float(win_bgwo["validation_metrics"]["average_precision"]),
            "interpretation": "Descriptive validation comparison; not a superiority claim.",
        },
        {
            "dimension": "validation_f1",
            "value_type": "validation",
            "BPSO-K10": float(win_bpso["validation_metrics"]["f1"]),
            "BGWO-K14": float(win_bgwo["validation_metrics"]["f1"]),
            "interpretation": "Descriptive validation comparison.",
        },
        {
            "dimension": "validation_recall",
            "value_type": "validation",
            "BPSO-K10": float(win_bpso["validation_metrics"]["recall"]),
            "BGWO-K14": float(win_bgwo["validation_metrics"]["recall"]),
            "interpretation": "Descriptive validation comparison.",
        },
        {
            "dimension": "final_test_dt_average_precision",
            "value_type": "final_test_mean",
            "BPSO-K10": _mean_value(final_dt, "BPSO-K10", "mean_average_precision"),
            "BGWO-K14": _mean_value(final_dt, "BGWO", "mean_average_precision"),
            "interpretation": "Metric-specific final-test means (n=5 pairs); no overall winner asserted.",
        },
        {
            "dimension": "final_test_dt_f1",
            "value_type": "final_test_mean",
            "BPSO-K10": _mean_value(final_dt, "BPSO-K10", "mean_f1"),
            "BGWO-K14": _mean_value(final_dt, "BGWO", "mean_f1"),
            "interpretation": "Difference is small; the paired 95% CI crosses zero.",
        },
        {
            "dimension": "final_test_lr_average_precision",
            "value_type": "final_test_mean",
            "BPSO-K10": _mean_value(final_lr, "BPSO-K10", "mean_average_precision"),
            "BGWO-K14": _mean_value(final_lr, "BGWO", "mean_average_precision"),
            "interpretation": "LR means are very close; metric-specific.",
        },
        {
            "dimension": "final_test_lr_f1",
            "value_type": "final_test_mean",
            "BPSO-K10": _mean_value(final_lr, "BPSO-K10", "mean_f1"),
            "BGWO-K14": _mean_value(final_lr, "BGWO", "mean_f1"),
            "interpretation": "LR means are very close; metric-specific.",
        },
    ]
    for classifier in CLASSIFIERS:
        for metric, column, label, phase in (
            ("wall_clock_elapsed_sec", "training_wall_time_sec",
             f"{classifier}_training_wall_time_sec", "training"),
            ("per_record_latency_sec", "inference_latency_per_record_sec",
             f"{classifier}_inference_latency_per_record_sec", "inference"),
        ):
            bpso_val = next(
                (
                    float(r["mean_of_seed_means"])
                    for r in resource
                    if r.get("classifier") == classifier
                    and r.get("configuration_id") == "BPSO-K10"
                    and r.get("metric") == metric
                    and r.get("phase") == phase
                ),
                None,
            )
            bgwo_val = next(
                (
                    float(r["mean_of_seed_means"])
                    for r in resource
                    if r.get("classifier") == classifier
                    and r.get("configuration_id") == "BGWO"
                    and r.get("metric") == metric
                    and r.get("phase") == phase
                ),
                None,
            )
            if bpso_val is None or bgwo_val is None:
                continue
            bg_vs_bpso_rows.append(
                {
                    "dimension": label,
                    "value_type": "resource_timing",
                    "BPSO-K10": bpso_val,
                    "BGWO-K14": bgwo_val,
                    "interpretation": "Indicative timing proxy; BPSO-K10 stays lighter for some resource metrics.",
                }
            )
    for classifier in CLASSIFIERS:
        bpso_size = float(
            next(
                (
                    row["serialized_model_bytes"]
                    for row in artifacts["v09f_model_size"]["entries"]
                    if row.get("classifier") == classifier
                    and row.get("configuration_id") == "BPSO-K10"
                ),
                np.nan,
            )
        )
        bgwo_size = float(
            next(
                (
                    row["serialized_model_bytes"]
                    for row in artifacts["v09f_model_size"]["entries"]
                    if row.get("classifier") == classifier
                    and row.get("configuration_id") == "BGWO"
                ),
                np.nan,
            )
        )
        bg_vs_bpso_rows.append(
            {
                "dimension": f"{classifier}_serialized_model_bytes",
                "value_type": "resource_structural",
                "BPSO-K10": bpso_size,
                "BGWO-K14": bgwo_size,
                "interpretation": "Fewer features do not guarantee a smaller serialized model.",
            }
        )
    oh = artifacts["v09f_optimizer_overhead"]
    bg_vs_bpso_rows.extend(
        [
            {
                "dimension": "optimizer_candidate_requests",
                "value_type": "one_time_overhead",
                "BPSO-K10": float(oh["bpso"]["candidate_requests"]),
                "BGWO-K14": float(oh["bgwo"]["candidate_requests"]),
                "interpretation": "One-time historical search cost; reported separately, not rerun.",
            },
            {
                "dimension": "optimizer_decision_tree_fits",
                "value_type": "one_time_overhead",
                "BPSO-K10": float(oh["bpso"]["decision_tree_fits"]),
                "BGWO-K14": float(oh["bgwo"]["decision_tree_fits"]),
                "interpretation": "One-time historical search cost; reported separately, not rerun.",
            },
            {
                "dimension": "optimizer_wall_time_sec",
                "value_type": "one_time_overhead",
                "BPSO-K10": float(oh["bpso"]["core_optimization_wall_time_sec"]),
                "BGWO-K14": float(oh["bgwo"]["core_optimization_wall_time_sec"]),
                "interpretation": "One-time historical search cost; reported separately, not rerun.",
            },
        ]
    )
    bpso_vs_bgwo = pd.DataFrame(bg_vs_bpso_rows)

    return {
        "dataset_governance": governance,
        "feature_configurations": feature_configs,
        "bgwo_validation_runs": validation_runs,
        "final_test_dt": final_dt,
        "final_test_lr": final_lr,
        "paired_statistics": paired_stats,
        "feature_overlap": feature_overlap,
        "resource_efficiency": resource_eff,
        "optimizer_overhead": optimizer_overhead,
        "bpso_vs_bgwo": bpso_vs_bgwo,
    }


def write_reporting_tables(
    tables: Mapping[str, pd.DataFrame],
    *,
    table_dir: Path | str = DEFAULT_TABLE_DIR,
) -> dict[str, Path]:
    output = Path(table_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for key, frame in tables.items():
        if key not in TABLE_FILENAMES:
            raise V09GError(f"Unknown table key: {key}")
        path = output / TABLE_FILENAMES[key]
        frame.to_csv(path, index=False)
        written[key] = path
    return written


def _set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def _save_figure(fig: plt.Figure, figure_dir: Path, basename: str) -> dict[str, Path]:
    png_path = figure_dir / f"{basename}.png"
    pdf_path = figure_dir / f"{basename}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def _figure_color_map() -> dict[str, str]:
    return {
        "K43": "#4c78a8",
        "K42": "#f58518",
        "MI-K11": "#54a24b",
        "BPSO-K10": "#f28e2b",
        "BGWO": "#e45756",
    }


_CLASSIFIER_LABELS = {
    "decision_tree": "Decision Tree",
    "logistic_regression": "Logistic Regression",
}
_METRIC_LABELS = {
    "average_precision": "AP",
    "f1": "F1",
    "recall": "Recall",
    "precision": "Precision",
    "roc_auc": "ROC-AUC",
}


def render_reporting_figures(
    artifacts: Mapping[str, Any],
    *,
    figure_dir: Path | str = DEFAULT_FIGURE_DIR,
) -> dict[str, dict[str, Path]]:
    _set_plot_style()
    output = Path(figure_dir)
    output.mkdir(parents=True, exist_ok=True)
    colors = _figure_color_map()
    rendered: dict[str, dict[str, Path]] = {}

    # 01 BGWO convergence -------------------------------------------------
    aggregate = pd.DataFrame(
        artifacts["v09d_convergence"]["aggregate_by_iteration"]
    ).sort_values("iteration_index")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    axes[0].plot(
        aggregate["iteration_index"], aggregate["mean_average_precision"],
        marker="o", color="#e45756",
    )
    axes[0].set_title("Mean Validation AP Across Iterations")
    axes[0].set_xlabel("Iteration Index")
    axes[0].set_ylabel("Mean AP")
    axes[1].plot(
        aggregate["iteration_index"], aggregate["mean_best_selected_feature_count"],
        marker="o", color="#4c78a8",
    )
    axes[1].set_title("Mean Best Selected Feature Count")
    axes[1].set_xlabel("Iteration Index")
    axes[1].set_ylabel("Selected Features")
    axes[2].plot(
        aggregate["iteration_index"], aggregate["mean_population_diversity"],
        marker="o", color="#54a24b",
    )
    axes[2].set_title("Mean Population Diversity")
    axes[2].set_xlabel("Iteration Index")
    axes[2].set_ylabel("Diversity")
    fig.suptitle("BGWO Convergence Across Five Optimizer Runs (V0.9-D)", fontsize=12)
    fig.tight_layout()
    rendered[FIGURE_NAMES[0]] = _save_figure(fig, output, FIGURE_NAMES[0])

    # 02 BGWO selected K by seed -----------------------------------------
    run_frame = pd.DataFrame.from_dict(artifacts["bgwo_run_artifacts"], orient="index")
    counts = run_frame["selected_feature_count"].astype(int).sort_index()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#e45756")
    ax.set_title("BGWO Selected Feature Count by Optimizer Run")
    ax.set_xlabel("Optimizer Seed")
    ax.set_ylabel("Selected Feature Count")
    ax.set_ylim(0, int(max(counts.values)) + 3)
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.25,
            str(count), ha="center", fontsize=9,
        )
    rendered[FIGURE_NAMES[1]] = _save_figure(fig, output, FIGURE_NAMES[1])

    # 03 BGWO feature-selection frequency ---------------------------------
    frequency = pd.DataFrame(artifacts["v09d_stability"]["feature_frequency"])
    frequency = frequency[frequency["selection_count"] > 0].sort_values(
        ["selection_frequency", "feature"], ascending=[True, True]
    )
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    ax.barh(frequency["feature"], frequency["selection_frequency"], color="#54a24b")
    ax.set_title("BGWO Feature Selection Frequency Across Five Runs")
    ax.set_xlabel("Selection Frequency")
    ax.set_ylabel("Feature")
    ax.set_xlim(0, 1.0)
    rendered[FIGURE_NAMES[2]] = _save_figure(fig, output, FIGURE_NAMES[2])

    # 04 feature-count comparison -----------------------------------------
    feature_counts = pd.DataFrame(
        {
            "configuration_id": list(CONFIGURATION_IDS),
            "feature_count": [43, 42, 11, 10, 14],
        }
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(
        feature_counts["configuration_id"], feature_counts["feature_count"],
        color=[colors[c] for c in feature_counts["configuration_id"]],
    )
    ax.set_title("Feature Count across Locked Configurations")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Feature Count")
    ax.set_ylim(0, 45)
    for bar, count in zip(bars, feature_counts["feature_count"]):
        reduction = 100.0 * (43.0 - count) / 43.0
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.8,
            f"{count} ({reduction:.2f}% red.)", ha="center", va="bottom", fontsize=8,
        )
    rendered[FIGURE_NAMES[3]] = _save_figure(fig, output, FIGURE_NAMES[3])

    # 05-06 final tests ----------------------------------------------------
    def _metric_heatmap(classifier: str, filename: str) -> None:
        summaries = artifacts["v09e_test_summary"]["summaries"]
        frame = pd.DataFrame(
            [
                {
                    "configuration_id": e["configuration_id"],
                    "metric": e["metric"],
                    "mean": float(e["mean"]),
                }
                for e in summaries
                if e.get("classifier") == classifier and e.get("metric") in METRICS
            ]
        )
        table = frame.pivot_table(
            index="configuration_id", columns="metric", values="mean", aggfunc="first"
        )
        table = table.reindex(index=list(CONFIGURATION_IDS), columns=list(METRICS)).astype(float)
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        image = ax.imshow(
            table.values, aspect="auto", cmap="YlGnBu",
            vmin=float(table.values.min()), vmax=float(table.values.max()),
        )
        ax.set_xticks(range(len(METRICS)), [_METRIC_LABELS[m] for m in METRICS])
        ax.set_yticks(range(len(CONFIGURATION_IDS)), list(CONFIGURATION_IDS))
        ax.set_title(f"{_CLASSIFIER_LABELS[classifier]} Final-Test Mean Metrics (V0.9-E)")
        for row in range(len(CONFIGURATION_IDS)):
            for col in range(len(METRICS)):
                value = table.values[row, col]
                ax.text(col, row, f"{value:.4f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, shrink=0.85, label="Mean metric value")
        rendered[filename] = _save_figure(fig, output, filename)

    _metric_heatmap("decision_tree", FIGURE_NAMES[4])
    _metric_heatmap("logistic_regression", FIGURE_NAMES[5])

    # 07 BGWO vs BPSO paired CI ------------------------------------------
    paired = artifacts["v09e_paired_statistics"].copy()
    paired = paired[
        (paired["candidate_configuration_id"] == "BGWO")
        & (paired["reference_configuration_id"] == "BPSO-K10")
        & (paired["metric"].isin(("average_precision", "f1", "recall")))
    ].sort_values(["classifier", "metric"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=False)
    for axis, classifier in zip(axes, CLASSIFIERS):
        subset = paired[paired["classifier"] == classifier]
        subset = (
            subset.set_index("metric").reindex(["average_precision", "f1", "recall"]).reset_index()
        )
        y = np.arange(len(subset))
        means = subset["mean_difference"].to_numpy(dtype=float)
        lows = subset["ci95_low"].to_numpy(dtype=float)
        highs = subset["ci95_high"].to_numpy(dtype=float)
        axis.errorbar(
            means, y, xerr=[means - lows, highs - means],
            fmt="o", color="#4c78a8", capsize=4, ms=6,
        )
        axis.axvline(0.0, color="#e45756", linestyle="--", linewidth=1.0)
        axis.set_yticks(y, [_METRIC_LABELS[m] for m in subset["metric"]])
        axis.set_title(f"{_CLASSIFIER_LABELS[classifier]} | BGWO over BPSO-K10")
        axis.set_xlabel("Paired mean difference (95% CI)")
    fig.suptitle("BGWO vs BPSO-K10 Paired Final-Test Differences (n=5 seed pairs)", fontsize=12)
    fig.tight_layout()
    rendered[FIGURE_NAMES[6]] = _save_figure(fig, output, FIGURE_NAMES[6])

    # 08 training time ----------------------------------------------------
    resource = artifacts["v09f_resource_summary"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for axis, classifier in zip(axes, CLASSIFIERS):
        values = {
            config: _resource_summary_value(
                resource, classifier=classifier, config=config,
                metric="wall_clock_elapsed_sec", phase="training",
            )
            for config in CONFIGURATION_IDS
        }
        configs = list(CONFIGURATION_IDS)
        axis.bar([str(c) for c in configs], [values[c] for c in configs], color=[colors[c] for c in configs])
        axis.set_title(f"{_CLASSIFIER_LABELS[classifier]} Training Wall Time")
        axis.set_ylabel("Seconds (mean of seed means)")
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle("Training Wall Time across Locked Configurations (V0.9-F)", fontsize=12)
    fig.tight_layout()
    rendered[FIGURE_NAMES[7]] = _save_figure(fig, output, FIGURE_NAMES[7])

    # 09 inference latency -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for axis, classifier in zip(axes, CLASSIFIERS):
        values = {
            config: _resource_summary_value(
                resource, classifier=classifier, config=config,
                metric="per_record_latency_sec", phase="inference",
            )
            for config in CONFIGURATION_IDS
        }
        configs = list(CONFIGURATION_IDS)
        axis.bar([str(c) for c in configs], [values[c] for c in configs], color=[colors[c] for c in configs])
        axis.set_title(f"{_CLASSIFIER_LABELS[classifier]} Inference Latency")
        axis.set_ylabel("Seconds per record (mean of seed means)")
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle("Inference Latency Per Record (V0.9-F)", fontsize=12)
    fig.tight_layout()
    rendered[FIGURE_NAMES[8]] = _save_figure(fig, output, FIGURE_NAMES[8])

    # 10 model size -------------------------------------------------------
    model_entries = artifacts["v09f_model_size"]["entries"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for axis, classifier in zip(axes, CLASSIFIERS):
        values = {}
        for row in model_entries:
            if row.get("classifier") == classifier:
                values[row["configuration_id"]] = float(row["serialized_model_bytes"])
        configs = list(CONFIGURATION_IDS)
        axis.bar(
            [str(c) for c in configs],
            [values.get(c, 0) for c in configs],
            color=[colors[c] for c in configs],
        )
        axis.set_title(f"{_CLASSIFIER_LABELS[classifier]} Serialized Model Size")
        axis.set_ylabel("Bytes")
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle("Serialized Model Size (V0.9-F)", fontsize=12)
    fig.tight_layout()
    rendered[FIGURE_NAMES[9]] = _save_figure(fig, output, FIGURE_NAMES[9])

    # 11 optimizer overhead -----------------------------------------------
    overhead = artifacts["v09f_optimizer_overhead"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    labels = ["BPSO", "BGWO"]
    pairs = [
        ("candidate_requests", "Candidate Requests"),
        ("decision_tree_fits", "Decision Tree Fits"),
        ("core_optimization_wall_time_sec", "Optimizer Wall Time (s)"),
    ]
    for axis, (key, title) in zip(axes, pairs):
        values = [float(overhead["bpso"][key]), float(overhead["bgwo"][key])]
        axis.bar(labels, values, color=["#f28e2b", "#e45756"])
        axis.set_title(title)
        axis.set_ylabel("One-time historical search cost")
    fig.suptitle("Optimizer Overhead Comparison (imported historical, not rerun)", fontsize=12)
    fig.tight_layout()
    rendered[FIGURE_NAMES[10]] = _save_figure(fig, output, FIGURE_NAMES[10])

    # 12 descriptive predictive/resource tradeoff --------------------------
    tradeoff_entries = artifacts["v09f_tradeoff"]["entries"]
    markers = {"decision_tree": "o", "logistic_regression": "s"}
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for entry in tradeoff_entries:
        config = str(entry["configuration_id"])
        fc = float(entry["feature_count"])
        for classifier in CLASSIFIERS:
            ap = float(entry["test_average_precision"][classifier])
            ax.scatter(fc, ap, color=colors[config], marker=markers[classifier], s=80, alpha=0.92)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#777777", markersize=7, label=_CLASSIFIER_LABELS["decision_tree"]),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#777777", markersize=7, label=_CLASSIFIER_LABELS["logistic_regression"]),
    ]
    for config in CONFIGURATION_IDS:
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[config], markersize=7, label=config)
        )
    ax.legend(handles=legend_handles, loc="lower right", ncol=2, fontsize=7)
    ax.set_title("Descriptive Predictive-Resource Tradeoff (Post-Lock, V0.9-F)")
    ax.set_xlabel("Feature count")
    ax.set_ylabel("Mean final-test AP")
    ax.set_xlim(8.5, 44.5)
    ax.set_ylim(bottom=0.0)
    rendered[FIGURE_NAMES[11]] = _save_figure(fig, output, FIGURE_NAMES[11])

    return rendered


def build_v09g_outputs(
    *,
    root: Path | str = PROJECT_ROOT,
    expected_head: str | None = EXPECTED_HEAD_SHORT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Generate all V0.9-G tables and figures from frozen evidence."""
    repo_root = _resolve_root(root)
    artifacts = load_v09g_artifacts(repo_root, expected_head=expected_head)
    preflight = artifacts["preflight"]

    output_root = Path(output_dir)
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"

    tables = build_reporting_tables(artifacts)
    written_tables = write_reporting_tables(tables, table_dir=table_dir)
    rendered_figures = render_reporting_figures(artifacts, figure_dir=figure_dir)

    return {
        "stage": STAGE,
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED",
        "preflight": preflight,
        "table_count": len(written_tables),
        "figure_count": len(rendered_figures),
        "table_dir": str(table_dir),
        "figure_dir": str(figure_dir),
        "tables": {key: str(path) for key, path in written_tables.items()},
        "figures": {
            name: {format_name: str(path) for format_name, path in outputs.items()}
            for name, outputs in rendered_figures.items()
        },
    }


def _table_file_hashes(root: Path, output_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key in TABLE_FILENAMES:
        path = output_dir / "tables" / TABLE_FILENAMES[key]
        if not path.is_file():
            raise V09GError(f"Missing reporting table: {path}")
        hashes[str(path.relative_to(root))] = sha256_file(path)
    return hashes


def _figure_file_hashes(root: Path, output_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in FIGURE_NAMES:
        for ext in ("png", "pdf"):
            path = output_dir / "figures" / f"{name}.{ext}"
            if not path.is_file():
                raise V09GError(f"Missing reporting figure: {path}")
            hashes[str(path.relative_to(root))] = sha256_file(path)
    return hashes


def create_reporting_lock(
    *,
    root: Path | str = PROJECT_ROOT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    notebook_path: Path | str = NOTEBOOK_PATH,
    report_path: Path | str = REPORT_PATH,
) -> dict[str, Any]:
    """Create the V0.9-G reporting lock from post-execution artifacts and verify it."""
    repo_root = _resolve_root(root)
    verify_v09g_preflight(root=repo_root, expected_head=EXPECTED_HEAD_SHORT)

    notebook = _resolve_relative(notebook_path, repo_root)
    report = _resolve_relative(report_path, repo_root)
    output_root = _resolve_relative(output_dir, repo_root)

    if not notebook.is_file():
        raise V09GError(f"Missing executed notebook for lock: {notebook}")
    if not report.is_file():
        raise V09GError(f"Missing report for lock: {report}")

    tables = _table_file_hashes(repo_root, output_root)
    figures = _figure_file_hashes(repo_root, output_root)
    evidence = snapshot_evidence(repo_root)

    payload = {
        "stage": STAGE,
        "schema_version": SCHEMA_VERSION,
        "starting_head": EXPECTED_HEAD_SHORT,
        "ending_head": EXPECTED_HEAD_SHORT,
        "preflight_status": "GO",
        "v09d_winner_semantic_hash": EXPECTED_BGWO_SEMANTIC,
        "v09e_result_semantic_hash": EXPECTED_V09E_RESULT_SEMANTIC,
        "v09f_result_semantic_hash": EXPECTED_V09F_RESULT_SEMANTIC,
        "bpso_winner_semantic_hash": EXPECTED_BPSO_SEMANTIC,
        "notebook_path": str(notebook.relative_to(repo_root)),
        "notebook_hash": sha256_file(notebook),
        "report_path": str(report.relative_to(repo_root)),
        "report_hash": sha256_file(report),
        "table_hashes": tables,
        "figure_hashes": figures,
        "source_evidence_hashes": evidence,
        "direct_energy_status": "DIRECT_ENERGY_UNAVAILABLE",
        "timer_interpretation": "OBSERVATION_LEVEL_TIMESTAMPED",
        "tradeoff_classification": "DESCRIPTIVE_POST_HOC_TRADEOFF_ANALYSIS",
    }
    semantic = _json_sha256(
        {key: value for key, value in payload.items() if key != "semantic_result_lock_sha256"}
    )
    payload["semantic_result_lock_sha256"] = semantic

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / "v09g_reporting_lock.json"
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return verify_reporting_lock(lock_path, root=repo_root, output_dir=output_root)


def verify_reporting_lock(
    lock: Path | str | Mapping[str, Any],
    *,
    root: Path | str = PROJECT_ROOT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Independently reload the reporting lock and verify every hash and identity."""
    repo_root = _resolve_root(root)
    payload: dict[str, Any]
    if isinstance(lock, Mapping):
        payload = dict(lock)
    else:
        payload = _read_json(lock)
        if payload.get("stage") != STAGE:
            raise V09GError("V09G_NO_GO: reporting lock stage mismatch.")

    problems: list[str] = []
    if payload.get("starting_head") != EXPECTED_HEAD_SHORT:
        problems.append("starting_head")
    if payload.get("ending_head") != EXPECTED_HEAD_SHORT:
        problems.append("ending_head")
    if payload.get("preflight_status") != "GO":
        problems.append("preflight_status")
    if payload.get("v09d_winner_semantic_hash") != EXPECTED_BGWO_SEMANTIC:
        problems.append("v09d")
    if payload.get("v09e_result_semantic_hash") != EXPECTED_V09E_RESULT_SEMANTIC:
        problems.append("v09e")
    if payload.get("v09f_result_semantic_hash") != EXPECTED_V09F_RESULT_SEMANTIC:
        problems.append("v09f")
    if payload.get("direct_energy_status") != "DIRECT_ENERGY_UNAVAILABLE":
        problems.append("direct_energy")

    for relative, expected in payload.get("table_hashes", {}).items():
        path = repo_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            problems.append(f"table:{relative}")
    for relative, expected in payload.get("figure_hashes", {}).items():
        path = repo_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            problems.append(f"figure:{relative}")
    for relative, expected in payload.get("source_evidence_hashes", {}).items():
        path = repo_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            problems.append(f"evidence:{relative}")

    notebook = repo_root / payload.get("notebook_path", "")
    report = repo_root / payload.get("report_path", "")
    if not notebook.is_file() or sha256_file(notebook) != payload.get("notebook_hash"):
        problems.append("notebook_hash")
    if not report.is_file() or sha256_file(report) != payload.get("report_hash"):
        problems.append("report_hash")

    recomputed_semantic = _json_sha256(
        {key: value for key, value in payload.items() if key != "semantic_result_lock_sha256"}
    )
    if recomputed_semantic != payload.get("semantic_result_lock_sha256"):
        problems.append("semantic_result_lock_sha256")

    if problems:
        raise V09GError(f"V09G_NO_GO: reporting lock verification failed: {sorted(problems)}")

    return {
        "stage": payload["stage"],
        "status": "VERIFIED",
        "schema_version": payload.get("schema_version"),
        "starting_head": payload["starting_head"],
        "semantic_result_lock_sha256": payload["semantic_result_lock_sha256"],
        "notebook_path": payload["notebook_path"],
        "report_path": payload["report_path"],
        "table_count": len(payload["table_hashes"]),
        "figure_count": len(payload["figure_hashes"]),
        "source_evidence_count": len(payload["source_evidence_hashes"]),
        "notebook_hash": payload.get("notebook_hash"),
        "report_hash": payload.get("report_hash"),
        "evidence_unchanged_since_lock": True,
    }