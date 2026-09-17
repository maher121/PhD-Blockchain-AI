"""V0.8-F reporting utilities for notebook, figures, tables, and synthesis.

This module is artifact-only. It loads frozen V0.8 artifacts, verifies governed
locks and hash identities, and builds publication-oriented tables and figures.
It does not rerun optimization, model training, predictive evaluation, or
resource benchmarking.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
import src.pipeline_v08d as v08d
import src.pipeline_v08e as v08e
import src.pipeline_v08e4 as v08e4


STAGE = "V0.8-F"
EXPECTED_HEAD_SHORT = "bcee6e9"
EXPECTED_E4_SEMANTIC_HASH = "7307fda1cdb5f100cb6268dc0f4eff05b65bfcefadeb06707c8a3e13fd6888d5"
EXPECTED_V08D_SEMANTIC_HASH = "ac19e5a0dba5d058f88485e6ab8ab9a96f97c30f697e95a744ccaa6563d2b657"
EXPECTED_T_CRITICAL = 2.776445
EXPECTED_CLASSIFICATION = "STRUCTURAL_GAIN_TIMING_UNCERTAIN"

CONFIG_ORDER = ("K43", "K42", "K11", "BPSO-K10")
CLASSIFIER_ORDER = ("decision_tree", "logistic_regression")
LOCKED_K10_FEATURES = (
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

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "bpso" / "v08f_reporting"
DEFAULT_TABLE_DIR = DEFAULT_OUTPUT_DIR / "tables"
DEFAULT_FIGURE_DIR = DEFAULT_OUTPUT_DIR / "figures"

TABLE_FILENAMES = {
    "table_a_bpso_run_summary": "table_a_bpso_run_summary.csv",
    "table_b_locked_k10_features": "table_b_locked_k10_features.csv",
    "table_c_predictive_results": "table_c_predictive_results.csv",
    "table_d_structural_comparison": "table_d_structural_comparison.csv",
    "table_e_paired_computational": "table_e_paired_computational.csv",
    "table_f_overhead_break_even": "table_f_overhead_break_even.csv",
    "table_g_limitations_claim_boundaries": "table_g_limitations_claim_boundaries.csv",
}

FIGURE_NAMES = (
    "01_bpso_selected_feature_count_by_run",
    "02_bpso_convergence",
    "03_feature_selection_frequency",
    "04_feature_count_reduction_comparison",
    "05_predictive_comparison",
    "06_input_memory_comparison",
    "07_model_size_comparison",
    "08_decision_tree_resource_comparison",
    "09_timing_variability",
    "10_predictive_resource_tradeoff",
)


class V08FError(RuntimeError):
    """Raised when governed V0.8-F reporting validations fail."""


def _read_json(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V08FError(f"Cannot read JSON artifact: {target}") from exc
    if not isinstance(payload, dict):
        raise V08FError(f"JSON artifact must be an object: {target}")
    return payload


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    target = Path(path)
    try:
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise V08FError(f"Cannot hash artifact: {target}") from exc
    return digest.hexdigest()


def locate_repository_root(start: Path | None = None) -> Path:
    origin = (start or PROJECT_ROOT).resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / "results" / "bpso").is_dir() and (candidate / "docs").is_dir():
            return candidate
    raise V08FError("Could not locate repository root containing results/bpso and docs/.")


def _git_head_short(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _verify_manifest_hashes(manifest: Mapping[str, str], base_dir: Path) -> None:
    mismatches: list[str] = []
    for filename, expected in manifest.items():
        artifact = base_dir / filename
        if not artifact.is_file() or _sha256_file(artifact) != expected:
            mismatches.append(filename)
    if mismatches:
        raise V08FError(f"Hash verification failed for: {', '.join(sorted(mismatches))}")


def verify_v08f_preflight(
    *,
    root: Path | str = PROJECT_ROOT,
    expected_head: str | None = EXPECTED_HEAD_SHORT,
) -> dict[str, Any]:
    """Verify governed V0.8 inputs for artifact-only V0.8-F reporting."""

    repo_root = locate_repository_root(Path(root))
    observed_head = _git_head_short(repo_root)
    if expected_head and observed_head != expected_head:
        raise V08FError(
            f"V08F_NO_GO: expected HEAD {expected_head}, observed {observed_head}."
        )

    v08c_lock = _read_json(repo_root / "results" / "bpso" / "v08c_winner_lock.json")
    v08d_lock = _read_json(repo_root / "results" / "bpso" / "v08d_final_test_lock.json")
    v08e_lock = _read_json(
        repo_root
        / "results"
        / "bpso"
        / "v08e_e4_analysis"
        / "v08e_scientific_result_lock.json"
    )
    v08e_manifest = _read_json(
        repo_root
        / "results"
        / "bpso"
        / "v08e_e4_analysis"
        / "v08e_e4_artifact_hashes.json"
    )

    if (
        v08c_lock.get("stage") != "V0.8-C"
        or v08c_lock.get("status") != "VALIDATION_LOCKED"
        or int(v08c_lock.get("selected_feature_count", -1)) != 10
        or tuple(v08c_lock.get("ordered_selected_features", ())) != LOCKED_K10_FEATURES
    ):
        raise V08FError("V08F_NO_GO: V0.8-C winner lock identity check failed.")

    v08d.verify_final_test_lock(v08d_lock)
    v08e4.verify_result_lock(v08e_lock)
    _verify_manifest_hashes(v08e_manifest, repo_root / "results" / "bpso" / "v08e_e4_analysis")

    if v08d_lock.get("semantic_result_lock_sha256") != EXPECTED_V08D_SEMANTIC_HASH:
        raise V08FError("V08F_NO_GO: V0.8-D semantic lock hash differs from governed value.")
    if v08e_lock.get("semantic_result_lock_sha256") != EXPECTED_E4_SEMANTIC_HASH:
        raise V08FError("V08F_NO_GO: V0.8-E semantic lock hash differs from governed value.")
    predictive_identity = v08e_lock.get("predictive_source_identity", {})
    if predictive_identity.get("semantic_result_lock_sha256") != v08d_lock.get(
        "semantic_result_lock_sha256"
    ):
        raise V08FError("V08F_NO_GO: V0.8-E predictive source identity mismatch.")

    preflight = _read_json(repo_root / "results" / "bpso" / "v08e_preflight.json")
    expected_hashes = preflight.get("immutable_hashes", {})
    mismatches: dict[str, dict[str, str]] = {}
    for path, expected in expected_hashes.items():
        observed = v08e.canonical_lf_sha256(path)
        if observed != expected:
            mismatches[str(path)] = {"expected": str(expected), "observed": observed}
    if mismatches:
        raise V08FError(
            f"V08F_NO_GO: frozen historical hash mismatch count = {len(mismatches)}"
        )

    return {
        "stage": STAGE,
        "status": "GO",
        "head_short": observed_head,
        "v08c_lock_verified": True,
        "v08d_lock_verified": True,
        "v08e_lock_verified": True,
        "v08d_semantic_result_lock_sha256": v08d_lock["semantic_result_lock_sha256"],
        "v08e_semantic_result_lock_sha256": v08e_lock["semantic_result_lock_sha256"],
        "resource_tradeoff_classification": v08e_lock["resource_tradeoff_classification"],
        "frozen_integrity_verified": True,
        "frozen_hash_count": len(expected_hashes),
    }


def _run_summary_from_v08c(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in (1042, 1043, 1044, 1045, 1046):
        payload = _read_json(root / "results" / "bpso" / f"v08c_run_{seed}.json")
        best = payload.get("best_evaluation", {})
        mean_metrics = dict(best.get("mean_metrics", {}))
        selected_count = int(best.get("selected_feature_count", sum(int(value) for value in best.get("mask", []))))
        rows.append(
            {
                "optimizer_seed": int(payload.get("optimizer_seed", seed)),
                "selected_feature_count": selected_count,
                "average_precision": float(mean_metrics.get("average_precision")),
                "f1": float(mean_metrics.get("f1")),
                "recall": float(mean_metrics.get("recall")),
                "artifact_sha256": _sha256_file(root / "results" / "bpso" / f"v08c_run_{seed}.json"),
            }
        )
    frame = pd.DataFrame(rows).sort_values("optimizer_seed").reset_index(drop=True)
    if len(frame) != 5:
        raise V08FError("Expected exactly five governed BPSO runs.")
    return frame


def _predictive_summary_frame(summary_payload: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(summary_payload.get("summaries", []))
    if frame.empty:
        raise V08FError("Predictive summary payload is empty.")
    return frame


def _preservation_frame(preservation_payload: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for classifier_entry in preservation_payload.get("results", []):
        classifier = str(classifier_entry["classifier"])
        for metric_entry in classifier_entry.get("metric_checks", []):
            rows.append(
                {
                    "classifier": classifier,
                    "metric": str(metric_entry["metric"]),
                    "k43_mean": float(metric_entry["k43_mean"]),
                    "bpso_k10_mean": float(metric_entry["bpso_k10_mean"]),
                    "relative_loss": float(metric_entry["relative_loss"]),
                    "margin": float(metric_entry["margin"]),
                    "preserved": bool(metric_entry["preserved"]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise V08FError("Preservation payload is empty.")
    return frame


def load_v08f_artifacts(root: Path | str = PROJECT_ROOT) -> dict[str, Any]:
    """Load all governed V0.8 artifacts required by the V0.8-F notebook/report."""

    repo_root = locate_repository_root(Path(root))
    preflight = verify_v08f_preflight(root=repo_root, expected_head=None)

    v08a_protocol = _read_json(repo_root / "results" / "bpso" / "v08a_protocol_validation.json")
    v08b_preflight = _read_json(repo_root / "results" / "bpso" / "v08b_preflight.json")
    v08b_pilot = _read_json(repo_root / "results" / "bpso" / "v08b_pilot.json")
    v08b_budget = _read_json(repo_root / "results" / "bpso" / "v08b_budget_projection.json")
    v08b_leakage = _read_json(repo_root / "results" / "bpso" / "v08b_leakage_audit.json")

    v08c_winner = _read_json(repo_root / "results" / "bpso" / "v08c_winner_lock.json")
    v08c_execution = _read_json(repo_root / "results" / "bpso" / "v08c_execution_summary.json")
    v08c_convergence = _read_json(repo_root / "results" / "bpso" / "v08c_convergence.json")
    v08c_stability = _read_json(repo_root / "results" / "bpso" / "v08c_stability.json")

    v08d_lock = _read_json(repo_root / "results" / "bpso" / "v08d_final_test_lock.json")
    v08d_execution = _read_json(repo_root / "results" / "bpso" / "v08d_execution_summary.json")
    v08d_summary = _read_json(repo_root / "results" / "bpso" / "v08d_final_test_summary.json")
    v08d_preservation = _read_json(repo_root / "results" / "bpso" / "v08d_preservation.json")

    e4_root = repo_root / "results" / "bpso" / "v08e_e4_analysis"
    v08e_lock = _read_json(e4_root / "v08e_scientific_result_lock.json")
    v08e_execution = _read_json(e4_root / "v08e_e4_execution_summary.json")
    v08e_overhead = _read_json(e4_root / "v08e_optimizer_overhead_summary.json")
    v08e_tradeoff = _read_json(e4_root / "v08e_tradeoff_analysis.json")
    v08e_sleep = _read_json(e4_root / "v08e_sleep_limitation.json")

    return {
        "preflight": preflight,
        "dataset_metadata": _read_json(repo_root / "data" / "processed" / "dataset_metadata.json"),
        "v08a_protocol": v08a_protocol,
        "v08b_preflight": v08b_preflight,
        "v08b_pilot": v08b_pilot,
        "v08b_budget": v08b_budget,
        "v08b_leakage": v08b_leakage,
        "v08c_winner": v08c_winner,
        "v08c_execution": v08c_execution,
        "v08c_convergence": v08c_convergence,
        "v08c_stability": v08c_stability,
        "v08c_run_summary": _run_summary_from_v08c(repo_root),
        "v08d_lock": v08d_lock,
        "v08d_execution": v08d_execution,
        "v08d_summary": _predictive_summary_frame(v08d_summary),
        "v08d_preservation": _preservation_frame(v08d_preservation),
        "v08e_lock": v08e_lock,
        "v08e_execution": v08e_execution,
        "v08e_overhead": v08e_overhead,
        "v08e_tradeoff": v08e_tradeoff,
        "v08e_sleep": v08e_sleep,
        "v08e_structural": pd.read_csv(e4_root / "v08e_structural_resource_summary.csv"),
        "v08e_paired": pd.read_csv(e4_root / "v08e_paired_resource_comparisons.csv"),
        "v08e_timing_variability": pd.read_csv(e4_root / "v08e_timing_variability.csv"),
        "v08e_break_even": pd.read_csv(e4_root / "v08e_break_even_analysis.csv"),
        "v08e_within_seed": pd.read_csv(e4_root / "v08e_within_seed_summary.csv"),
    }


def build_publication_tables(artifacts: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Build compact publication tables from governed artifacts."""

    run_summary = artifacts["v08c_run_summary"].copy()
    run_summary["selected_feature_count"] = run_summary["selected_feature_count"].astype(int)

    table_a = run_summary[[
        "optimizer_seed",
        "selected_feature_count",
        "average_precision",
        "f1",
        "recall",
    ]].copy()

    table_b = pd.DataFrame(
        {
            "rank": list(range(1, len(LOCKED_K10_FEATURES) + 1)),
            "feature": list(LOCKED_K10_FEATURES),
        }
    )

    predictive = artifacts["v08d_summary"].copy()
    predictive = predictive[
        predictive["metric"].isin(("average_precision", "f1", "recall", "precision", "roc_auc"))
    ]
    table_c = (
        predictive.pivot_table(
            index=["classifier", "configuration_id"],
            columns="metric",
            values="mean",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    preservation = artifacts["v08d_preservation"]
    preserved = (
        preservation.groupby("classifier", as_index=False)["preserved"]
        .all()
        .rename(columns={"preserved": "preservation_overall"})
    )
    table_c = table_c.merge(preserved, on="classifier", how="left")
    table_c["configuration_id"] = pd.Categorical(table_c["configuration_id"], CONFIG_ORDER, ordered=True)
    table_c["classifier"] = pd.Categorical(table_c["classifier"], CLASSIFIER_ORDER, ordered=True)
    table_c = table_c.sort_values(["classifier", "configuration_id"]).reset_index(drop=True)

    structural = artifacts["v08e_structural"].copy()
    feature_count = structural[
        (structural["metric"] == "feature_count") & (structural["phase"] == "inference")
    ][["classifier", "configuration_id", "mean_across_seeds"]].rename(
        columns={"mean_across_seeds": "feature_count"}
    )
    train_input = structural[
        (structural["metric"] == "input_dataframe_memory_bytes")
        & (structural["phase"] == "training")
    ][["classifier", "configuration_id", "mean_across_seeds"]].rename(
        columns={"mean_across_seeds": "train_input_memory_bytes"}
    )
    infer_input = structural[
        (structural["metric"] == "input_dataframe_memory_bytes")
        & (structural["phase"] == "inference")
    ][["classifier", "configuration_id", "mean_across_seeds"]].rename(
        columns={"mean_across_seeds": "inference_input_memory_bytes"}
    )
    model_size = structural[
        (structural["metric"] == "serialized_model_bytes")
        & (structural["phase"] == "inference")
    ][["classifier", "configuration_id", "mean_across_seeds"]].rename(
        columns={"mean_across_seeds": "mean_serialized_model_bytes"}
    )
    table_d = feature_count.merge(train_input, on=["classifier", "configuration_id"]).merge(
        infer_input, on=["classifier", "configuration_id"]
    ).merge(model_size, on=["classifier", "configuration_id"])
    baseline_feature_count = 43.0
    table_d["feature_reduction_vs_k43_percent"] = (
        100.0 * (baseline_feature_count - table_d["feature_count"]) / baseline_feature_count
    )
    table_d["configuration_id"] = pd.Categorical(table_d["configuration_id"], CONFIG_ORDER, ordered=True)
    table_d["classifier"] = pd.Categorical(table_d["classifier"], CLASSIFIER_ORDER, ordered=True)
    table_d = table_d.sort_values(["classifier", "configuration_id"]).reset_index(drop=True)

    paired = artifacts["v08e_paired"].copy()
    table_e = paired[
        (paired["metric"].isin(("wall_time_sec", "process_cpu_time_sec", "inference_latency_sec", "throughput_records_sec")))
        & (paired["reference_configuration_id"].isin(("K43", "K42", "K11")))
        & (
            ((paired["metric"] == "inference_latency_sec") & (paired["phase"] == "inference"))
            | ((paired["metric"] == "throughput_records_sec") & (paired["phase"] == "inference"))
            | ((paired["metric"] == "wall_time_sec") & (paired["phase"] == "training"))
            | ((paired["metric"] == "process_cpu_time_sec") & (paired["phase"] == "training"))
        )
    ][
        [
            "classifier",
            "reference_configuration_id",
            "phase",
            "metric",
            "reference_mean_across_seeds",
            "candidate_mean_across_seeds",
            "mean_percentage_change",
            "paired_difference_ci95_low",
            "paired_difference_ci95_high",
            "ci_includes_zero",
            "direction_consistency",
            "ci_interpretation",
            "timing_integrity_limited",
        ]
    ].copy()
    table_e["classifier"] = pd.Categorical(table_e["classifier"], CLASSIFIER_ORDER, ordered=True)
    table_e["reference_configuration_id"] = pd.Categorical(
        table_e["reference_configuration_id"], ("K43", "K42", "K11"), ordered=True
    )
    table_e = table_e.sort_values(["classifier", "reference_configuration_id", "phase", "metric"]).reset_index(drop=True)

    overhead = artifacts["v08e_overhead"]
    break_even = artifacts["v08e_break_even"].copy()
    break_even_counts = break_even["status"].value_counts().to_dict()
    table_f_rows = [
        {
            "section": "optimizer_overhead",
            "metric": "optimizer_runs",
            "value": float(overhead["optimizer_runs"]),
            "unit": "runs",
            "claim_boundary": "ONE_TIME_IMPORTED_HISTORICAL_SEARCH_COST",
        },
        {
            "section": "optimizer_overhead",
            "metric": "fitness_requests",
            "value": float(overhead["fitness_requests"]),
            "unit": "requests",
            "claim_boundary": "ONE_TIME_IMPORTED_HISTORICAL_SEARCH_COST",
        },
        {
            "section": "optimizer_overhead",
            "metric": "decision_tree_fits",
            "value": float(overhead["decision_tree_fits"]),
            "unit": "fits",
            "claim_boundary": "ONE_TIME_IMPORTED_HISTORICAL_SEARCH_COST",
        },
        {
            "section": "optimizer_overhead",
            "metric": "core_optimization_wall_time_sec",
            "value": float(overhead["core_optimization_wall_time_sec"]),
            "unit": "seconds",
            "claim_boundary": "ONE_TIME_IMPORTED_HISTORICAL_SEARCH_COST",
        },
        {
            "section": "optimizer_overhead",
            "metric": "core_optimization_cpu_time_sec",
            "value": float(overhead["core_optimization_cpu_time_sec"]),
            "unit": "cpu-seconds",
            "claim_boundary": "ONE_TIME_IMPORTED_HISTORICAL_SEARCH_COST",
        },
        {
            "section": "optimizer_overhead",
            "metric": "pipeline_wall_time_sec",
            "value": float(overhead["pipeline_wall_time_sec"]),
            "unit": "seconds",
            "claim_boundary": "ONE_TIME_IMPORTED_HISTORICAL_SEARCH_COST",
        },
        {
            "section": "break_even",
            "metric": "applicable_rows",
            "value": float(break_even_counts.get("APPLICABLE", 0)),
            "unit": "rows",
            "claim_boundary": "DERIVED_DESCRIPTIVE_TIMING_LIMITED",
        },
        {
            "section": "break_even",
            "metric": "not_applicable_rows",
            "value": float(break_even_counts.get("NOT_APPLICABLE", 0)),
            "unit": "rows",
            "claim_boundary": "DERIVED_DESCRIPTIVE_TIMING_LIMITED",
        },
    ]
    table_f = pd.DataFrame(table_f_rows)

    sleep = artifacts["v08e_sleep"]
    e4_lock = artifacts["v08e_lock"]
    table_g = pd.DataFrame(
        [
            {
                "boundary": "sleep_limitation",
                "status": sleep["sleep_audit_classification"],
                "statement": sleep["required_wording"],
            },
            {
                "boundary": "timing_interpretation",
                "status": sleep["timing_interpretation"],
                "statement": "Timing metrics are indicative rather than definitive.",
            },
            {
                "boundary": "direct_energy",
                "status": e4_lock["direct_energy_status"],
                "statement": "No Joules, Watts, kWh, CO2, or CO2e claims are reported.",
            },
            {
                "boundary": "statistical_protocol",
                "status": "CI_ONLY",
                "statement": "Seed-level paired CI protocol (n=5, df=4, t_critical=2.776445); no p-values.",
            },
            {
                "boundary": "classification",
                "status": e4_lock["resource_tradeoff_classification"],
                "statement": "Structural gains are supported; timing outcomes include uncertainty.",
            },
            {
                "boundary": "governance",
                "status": "NO_RERUN",
                "statement": "V0.8-F is presentation/synthesis over governed artifacts only.",
            },
        ]
    )

    return {
        "table_a_bpso_run_summary": table_a,
        "table_b_locked_k10_features": table_b,
        "table_c_predictive_results": table_c,
        "table_d_structural_comparison": table_d,
        "table_e_paired_computational": table_e,
        "table_f_overhead_break_even": table_f,
        "table_g_limitations_claim_boundaries": table_g,
    }


def write_publication_tables(
    tables: Mapping[str, pd.DataFrame],
    *,
    table_dir: Path | str = DEFAULT_TABLE_DIR,
) -> dict[str, Path]:
    """Write publication tables to CSV and return generated paths."""

    output = Path(table_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for key, frame in tables.items():
        if key not in TABLE_FILENAMES:
            raise V08FError(f"Unknown table key: {key}")
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
        "K11": "#54a24b",
        "BPSO-K10": "#e45756",
    }


def render_publication_figures(
    artifacts: Mapping[str, Any],
    *,
    figure_dir: Path | str = DEFAULT_FIGURE_DIR,
) -> dict[str, dict[str, Path]]:
    """Render publication figures from governed artifacts only."""

    _set_plot_style()
    output = Path(figure_dir)
    output.mkdir(parents=True, exist_ok=True)
    colors = _figure_color_map()
    rendered: dict[str, dict[str, Path]] = {}

    run_summary = artifacts["v08c_run_summary"].copy().sort_values("optimizer_seed")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(
        run_summary["optimizer_seed"].astype(str),
        run_summary["selected_feature_count"],
        color="#4c78a8",
    )
    ax.set_title("BPSO Selected Feature Count by Run")
    ax.set_xlabel("Optimizer Seed")
    ax.set_ylabel("Selected Feature Count")
    ax.set_ylim(0, max(16, int(run_summary["selected_feature_count"].max()) + 2))
    rendered[FIGURE_NAMES[0]] = _save_figure(fig, output, FIGURE_NAMES[0])

    aggregate = pd.DataFrame(artifacts["v08c_convergence"]["aggregate_by_generation"]).sort_values(
        "generation_index"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))
    axes[0].plot(
        aggregate["generation_index"],
        aggregate["mean_best_selected_feature_count"],
        marker="o",
        color="#e45756",
    )
    axes[0].set_title("Mean Best Feature Count Across Generations")
    axes[0].set_xlabel("Generation Index")
    axes[0].set_ylabel("Mean Selected Features")
    axes[0].set_ylim(0, max(26, float(aggregate["mean_best_selected_feature_count"].max()) + 1.0))

    axes[1].plot(
        aggregate["generation_index"],
        aggregate["mean_average_precision"],
        marker="o",
        color="#4c78a8",
    )
    axes[1].set_title("Mean Average Precision Across Generations")
    axes[1].set_xlabel("Generation Index")
    axes[1].set_ylabel("Mean AP")
    axes[1].set_ylim(0.20, max(0.245, float(aggregate["mean_average_precision"].max()) + 0.002))
    rendered[FIGURE_NAMES[1]] = _save_figure(fig, output, FIGURE_NAMES[1])

    frequency = pd.DataFrame(artifacts["v08c_stability"]["feature_frequency"])
    frequency = frequency[frequency["selection_count"] > 0].sort_values(
        ["selection_frequency", "selection_count", "feature"], ascending=[True, True, True]
    )
    fig, ax = plt.subplots(figsize=(9.0, 6.8))
    ax.barh(
        frequency["feature"],
        frequency["selection_frequency"],
        color="#54a24b",
    )
    ax.set_title("BPSO Feature Selection Frequency Across Five Runs")
    ax.set_xlabel("Selection Frequency")
    ax.set_ylabel("Feature")
    ax.set_xlim(0, 1.0)
    rendered[FIGURE_NAMES[2]] = _save_figure(fig, output, FIGURE_NAMES[2])

    feature_counts = pd.DataFrame(
        {
            "configuration_id": list(CONFIG_ORDER),
            "feature_count": [43, 42, 11, 10],
        }
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(
        feature_counts["configuration_id"],
        feature_counts["feature_count"],
        color=[colors[cfg] for cfg in feature_counts["configuration_id"]],
    )
    ax.set_title("Feature Count Reduction Across Locked Configurations")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Feature Count")
    ax.set_ylim(0, 45)
    for bar, count in zip(bars, feature_counts["feature_count"]):
        reduction = 100.0 * (43.0 - count) / 43.0
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.8,
            f"{count} ({reduction:.2f}% red.)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    rendered[FIGURE_NAMES[3]] = _save_figure(fig, output, FIGURE_NAMES[3])

    predictive = artifacts["v08d_summary"].copy()
    predictive = predictive[
        predictive["metric"].isin(("average_precision", "f1", "recall", "roc_auc"))
    ]
    metric_order = ["average_precision", "f1", "recall", "roc_auc"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=False)
    for axis, classifier in zip(axes, CLASSIFIER_ORDER):
        pivot = (
            predictive[predictive["classifier"] == classifier]
            .pivot_table(index="configuration_id", columns="metric", values="mean", aggfunc="first")
            .reindex(index=list(CONFIG_ORDER), columns=metric_order)
        )
        image = axis.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
        axis.set_xticks(range(len(metric_order)), ["AP", "F1", "Recall", "ROC-AUC"])
        axis.set_yticks(range(len(CONFIG_ORDER)), list(CONFIG_ORDER))
        axis.set_title(classifier.replace("_", " ").title())
        for row in range(len(CONFIG_ORDER)):
            for col in range(len(metric_order)):
                value = pivot.values[row, col]
                axis.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, label="Metric value")
    fig.suptitle("Frozen Predictive Comparison (V0.8-D)", fontsize=12)
    rendered[FIGURE_NAMES[4]] = _save_figure(fig, output, FIGURE_NAMES[4])

    structural = artifacts["v08e_structural"].copy()
    input_memory = structural[structural["metric"] == "input_dataframe_memory_bytes"].copy()
    input_memory["configuration_id"] = pd.Categorical(input_memory["configuration_id"], CONFIG_ORDER, ordered=True)
    input_memory = input_memory.sort_values(["phase", "configuration_id"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for axis, phase in zip(axes, ("training", "inference")):
        subset = input_memory[
            (input_memory["classifier"] == "decision_tree") & (input_memory["phase"] == phase)
        ]
        axis.bar(
            subset["configuration_id"].astype(str),
            subset["mean_across_seeds"],
            color=[colors[cfg] for cfg in subset["configuration_id"].astype(str)],
        )
        axis.set_title(f"Decision Tree {phase.title()} Input Memory")
        axis.set_xlabel("Configuration")
        axis.set_ylabel("Bytes")
        axis.set_ylim(bottom=0)
    rendered[FIGURE_NAMES[5]] = _save_figure(fig, output, FIGURE_NAMES[5])

    model_size = structural[
        (structural["metric"] == "serialized_model_bytes") & (structural["phase"] == "inference")
    ].copy()
    model_size["configuration_id"] = pd.Categorical(model_size["configuration_id"], CONFIG_ORDER, ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for axis, classifier in zip(axes, CLASSIFIER_ORDER):
        subset = model_size[model_size["classifier"] == classifier].sort_values("configuration_id")
        axis.bar(
            subset["configuration_id"].astype(str),
            subset["mean_across_seeds"],
            color=[colors[cfg] for cfg in subset["configuration_id"].astype(str)],
        )
        axis.set_title(f"{classifier.replace('_', ' ').title()} Model Size")
        axis.set_xlabel("Configuration")
        axis.set_ylabel("Serialized Bytes")
        axis.set_ylim(bottom=0)
    rendered[FIGURE_NAMES[6]] = _save_figure(fig, output, FIGURE_NAMES[6])

    paired = artifacts["v08e_paired"].copy()
    dt_key = paired[
        (paired["classifier"] == "decision_tree")
        & (paired["reference_configuration_id"] == "K43")
        & (
            (
                (paired["phase"] == "training")
                & (paired["metric"].isin(("wall_time_sec", "process_cpu_time_sec")))
            )
            | (
                (paired["phase"] == "inference")
                & (paired["metric"].isin(("inference_latency_sec", "throughput_records_sec")))
            )
        )
    ]
    metric_labels = {
        "wall_time_sec": "Training Wall (s)",
        "process_cpu_time_sec": "Training CPU (s)",
        "inference_latency_sec": "Inference Latency (s/op)",
        "throughput_records_sec": "Inference Throughput (rec/s)",
    }
    dt_key = dt_key.sort_values(["phase", "metric"])
    x = np.arange(len(dt_key))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.bar(
        x - width / 2.0,
        dt_key["reference_mean_across_seeds"],
        width=width,
        color=colors["K43"],
        label="K43",
    )
    ax.bar(
        x + width / 2.0,
        dt_key["candidate_mean_across_seeds"],
        width=width,
        color=colors["BPSO-K10"],
        label="BPSO-K10",
    )
    ax.set_xticks(x, [metric_labels[m] for m in dt_key["metric"]], rotation=20, ha="right")
    ax.set_title("Decision Tree K10 vs K43 Resource Means (Indicative Timing)")
    ax.set_ylabel("Metric value (native units)")
    ax.set_ylim(bottom=0)
    ax.legend()
    rendered[FIGURE_NAMES[7]] = _save_figure(fig, output, FIGURE_NAMES[7])

    variability = artifacts["v08e_timing_variability"].copy()
    variability = variability[variability["metric"].isin(("wall_time_sec", "process_cpu_time_sec", "inference_latency_sec"))]
    metric_order = ["wall_time_sec", "process_cpu_time_sec", "inference_latency_sec"]
    labels = ["Wall Time CV", "CPU Time CV", "Inference Latency CV"]
    samples = [
        variability[variability["metric"] == metric]["within_seed_cv_mean_percent"].to_numpy()
        for metric in metric_order
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.boxplot(samples, tick_labels=labels, showfliers=True)
    ax.set_title("Timing Variability Across Seed-Cells")
    ax.set_ylabel("Within-seed CV mean (%)")
    ax.set_ylim(bottom=0)
    rendered[FIGURE_NAMES[8]] = _save_figure(fig, output, FIGURE_NAMES[8])

    tradeoff = pd.DataFrame(artifacts["v08e_tradeoff"]["points"])
    fig, ax = plt.subplots(figsize=(8.7, 5.1))
    markers = {"decision_tree": "o", "logistic_regression": "s"}
    for classifier in CLASSIFIER_ORDER:
        subset = tradeoff[tradeoff["classifier"] == classifier]
        for _, row in subset.iterrows():
            ax.scatter(
                row["feature_count"],
                row["average_precision"],
                color=colors[str(row["configuration_id"])],
                marker=markers[classifier],
                s=70,
                alpha=0.9,
            )
            ax.text(
                row["feature_count"] + 0.3,
                row["average_precision"] + 0.00035,
                f"{row['configuration_id']} ({'DT' if classifier == 'decision_tree' else 'LR'})",
                fontsize=7,
            )
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#777777", markersize=7, label="Decision Tree"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#777777", markersize=7, label="Logistic Regression"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")
    ax.set_title("Predictive-Resource Tradeoff View (Descriptive Post-Lock)")
    ax.set_xlabel("Feature count")
    ax.set_ylabel("Average precision")
    ax.set_xlim(8.5, 44.5)
    ax.set_ylim(bottom=0.0)
    rendered[FIGURE_NAMES[9]] = _save_figure(fig, output, FIGURE_NAMES[9])

    return rendered


def build_v08f_outputs(
    *,
    root: Path | str = PROJECT_ROOT,
    expected_head: str | None = EXPECTED_HEAD_SHORT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Generate V0.8-F tables and figures from governed artifacts."""

    repo_root = locate_repository_root(Path(root))
    preflight = verify_v08f_preflight(root=repo_root, expected_head=expected_head)
    artifacts = load_v08f_artifacts(repo_root)

    output_root = Path(output_dir)
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"

    tables = build_publication_tables(artifacts)
    written_tables = write_publication_tables(tables, table_dir=table_dir)
    rendered_figures = render_publication_figures(artifacts, figure_dir=figure_dir)

    return {
        "stage": STAGE,
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
