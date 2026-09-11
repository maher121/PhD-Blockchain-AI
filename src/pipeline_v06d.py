"""Measured, validation-only V0.6-D feature-selection matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_SELECTION_CONFIG_FILE,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
)
from src.feature_selection import BaseFeatureSelector
from src.pipeline_v06 import (
    V06ProtocolConfig,
    create_v06_selector,
    load_v06_development_data,
    load_v06_protocol_config,
    run_validation_experiment,
)
from src.security.attack_generator import load_attack_config
from src.security.evaluation import (
    detector_visible_mask,
    evaluate_detection,
    evaluate_paired_detection,
    evaluate_visible_attacks,
)
from src.security.experiment_data import (
    DevelopmentExperimentData,
    fingerprint_feature_names,
    fingerprint_mapping,
)

V06D_SCHEMA_VERSION = "v0.6-d"
V06D_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
EXPECTED_V02_FEATURE_COUNT = 43
T_CRITICAL_DF4_95 = 2.7764451051977987
DEFAULT_RESULTS_DIR = RESULTS_DIR / "feature_selection"
DEFAULT_MODELS_DIR = MODELS_DIR / "feature_selection" / "core"

_NATURAL_SELECTORS = (
    "variance_threshold",
    "pairwise_correlation_filter",
)
_RANKING_SELECTORS = (
    "correlation_redundancy_ranking",
    "mutual_information_select_k_best",
    "anova_f_select_k_best",
)
_AGGREGATE_COLUMNS = {
    "average_precision": "average_precision",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "roc_auc": "roc_auc",
    "feature_count": "selected_feature_count",
    "training_sec": "combined_train_wall_time_sec",
    "inference_sec": "attacked_validation_inference_wall_time_sec",
    "peak_rss_mib": "peak_rss_mib",
    "model_size_bytes": "serialized_model_bytes",
}


class V06DMatrixFailure(RuntimeError):
    """Raised after all possible outputs are written when any run failed."""


def build_v06d_matrix(config: V06ProtocolConfig) -> list[dict[str, Any]]:
    """Return the frozen twelve validation configurations for one seed."""
    counts = tuple(config.candidate_feature_counts)
    if len(counts) != 3:
        raise ValueError("V0.6-D requires exactly three configured candidate K values.")
    matrix = [
        {
            "configuration_id": "none_natural",
            "selector_id": "none",
            "selector_type": "none",
            "count_strategy": "natural",
            "requested_feature_count": None,
        },
        *[
            {
                "configuration_id": f"{selector_id}_natural",
                "selector_id": selector_id,
                "selector_type": "unsupervised",
                "count_strategy": "natural",
                "requested_feature_count": None,
            }
            for selector_id in _NATURAL_SELECTORS
        ],
    ]
    for selector_id in _RANKING_SELECTORS:
        selector_type = (
            "unsupervised"
            if selector_id == "correlation_redundancy_ranking"
            else "supervised"
        )
        for count in counts:
            matrix.append(
                {
                    "configuration_id": f"{selector_id}_k{count}",
                    "selector_id": selector_id,
                    "selector_type": selector_type,
                    "count_strategy": "fixed_k",
                    "requested_feature_count": int(count),
                }
            )
    if len(matrix) != 12:
        raise RuntimeError("V0.6-D matrix construction did not produce 12 configurations.")
    return matrix


def run_v06d_validation_matrix(
    *,
    config_path: Path | str = FEATURE_SELECTION_CONFIG_FILE,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    processed_dir: Path | str = PROCESSED_DATA_DIR,
    development_data_loader: Callable[
        [V06ProtocolConfig], DevelopmentExperimentData
    ]
    | None = None,
) -> dict[str, Any]:
    """Execute all 60 measured development runs without exposing test data."""
    config_path = Path(config_path)
    results_dir = Path(results_dir)
    models_dir = Path(models_dir)
    processed_dir = Path(processed_dir)
    base_config = load_v06_protocol_config(config_path)
    if tuple(base_config.configured_seeds) != V06D_SEEDS:
        raise ValueError(f"V0.6-D seeds must be exactly {V06D_SEEDS}.")

    loader = development_data_loader
    if loader is None:
        loader = lambda config: load_v06_development_data(
            config, processed_dir=processed_dir
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    core_rows: list[dict[str, Any]] = []
    selected_sets: dict[str, Any] = {}
    ranking_rows: list[dict[str, Any]] = []
    manifest_checks: list[dict[str, Any]] = []
    load_counts: dict[int, int] = {seed: 0 for seed in V06D_SEEDS}

    for seed in V06D_SEEDS:
        config = load_v06_protocol_config(config_path, seed=seed)
        matrix = build_v06d_matrix(config)
        try:
            load_counts[seed] += 1
            data = loader(config)
            manifest_check = _validate_manifest(data, config, seed)
            manifest_checks.append(manifest_check)
        except Exception as exc:
            manifest_checks.append(
                {
                    "seed": seed,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            core_rows.extend(_failure_row(seed, item, exc) for item in matrix)
            continue

        for item in matrix:
            run_id = f"v06d_s{seed}_{item['configuration_id']}"
            artifact_path = models_dir / f"{run_id}.joblib"
            try:
                selector = create_v06_selector(
                    item["selector_id"],
                    data.candidate_features,
                    config,
                    n_features_to_select=item["requested_feature_count"],
                )
                outcome = run_validation_experiment(
                    data,
                    selector=selector,
                    config=config,
                    run_id=run_id,
                    measure_resources=True,
                    model_artifact_path=artifact_path,
                )
                row = _flatten_successful_run(
                    outcome.record.to_dict(), item, data, outcome.model, selector
                )
                core_rows.append(row)
                selected_sets[run_id] = {
                    "seed": seed,
                    "configuration_id": item["configuration_id"],
                    "selector_id": item["selector_id"],
                    "requested_feature_count": item["requested_feature_count"],
                    "selected_feature_count": outcome.record.selected_feature_count,
                    "selected_features": list(outcome.record.selected_features),
                    "selected_features_sha256": outcome.record.selected_features_sha256,
                }
                ranking_rows.extend(
                    _ranking_records(run_id, seed, item, selector)
                )
            except Exception as exc:
                core_rows.append(_failure_row(seed, item, exc, run_id=run_id))

    core = pd.DataFrame(core_rows)
    successful = core.loc[core["status"].eq("success")].copy()
    summary = aggregate_validation_runs(successful)
    paired_seed, paired_summary = paired_baseline_comparisons(successful)
    sensitivity = non_inferiority_sensitivity(summary)
    primary = sensitivity.loc[sensitivity["ap_f1_margin_percent"].eq(5.0)].copy()
    candidate_selection, candidate_identity = select_candidates(summary, primary)
    set_stability = selected_set_stability(successful)
    ranking_stability = ranking_correlation_stability(pd.DataFrame(ranking_rows))
    deterministic = deterministic_unsupervised_stability(
        successful, pd.DataFrame(ranking_rows)
    )
    resource_comparison = _resource_comparison(summary)
    leakage_audit = _matrix_leakage_audit(
        core,
        manifest_checks=manifest_checks,
        load_counts=load_counts,
        expected_runs=len(V06D_SEEDS) * 12,
        candidate_selection=candidate_identity,
    )

    _write_csv(core, results_dir / "core_runs.csv")
    _write_json(core_rows, results_dir / "core_runs.json")
    _write_json(selected_sets, results_dir / "selected_feature_sets.json")
    _write_csv(pd.DataFrame(ranking_rows), results_dir / "feature_selection_rankings.csv")
    _write_csv(resource_comparison, results_dir / "resource_comparison.csv")
    _write_csv(summary, results_dir / "validation_summary.csv")
    _write_csv(paired_seed, results_dir / "paired_baseline_differences.csv")
    _write_csv(paired_summary, results_dir / "paired_baseline_comparisons.csv")
    _write_csv(sensitivity, results_dir / "non_inferiority_sensitivity.csv")
    _write_csv(primary, results_dir / "preservation_primary.csv")
    _write_csv(candidate_selection, results_dir / "candidate_selection.csv")
    _write_json(candidate_identity, results_dir / "candidate_selection.json")
    _write_csv(set_stability, results_dir / "selected_set_stability.csv")
    _write_csv(ranking_stability, results_dir / "ranking_stability.csv")
    _write_csv(
        deterministic,
        results_dir / "deterministic_unsupervised_stability.csv",
    )
    _write_json(leakage_audit, results_dir / "leakage_audit.json")
    figure_paths = _write_figures(summary, results_dir / "figures")
    metadata = _run_metadata(
        config_path=config_path,
        processed_dir=processed_dir,
        results_dir=results_dir,
        models_dir=models_dir,
        base_config=base_config,
        manifest_checks=manifest_checks,
        core=core,
        figure_paths=figure_paths,
        candidate_identity=candidate_identity,
    )
    _write_json(metadata, results_dir / "run_metadata.json")

    failures = core.loc[core["status"].ne("success")]
    if not failures.empty:
        raise V06DMatrixFailure(
            f"V0.6-D recorded {len(failures)} failed runs; inspect core_runs.json."
        )
    if leakage_audit["status"] != "PASS":
        raise V06DMatrixFailure("V0.6-D leakage audit failed.")
    return {
        "status": "PASS",
        "run_count": len(core),
        "results_dir": results_dir,
        "models_dir": models_dir,
        "candidate_selection": candidate_identity,
        "leakage_audit": leakage_audit,
    }


def aggregate_validation_runs(core: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the requested performance and resource fields by configuration."""
    if core.empty:
        columns = [
            "configuration_id",
            "selector_id",
            "selector_type",
            "count_strategy",
            "requested_feature_count",
            "seed_count",
        ]
        return pd.DataFrame(columns=columns)
    group_columns = [
        "configuration_id",
        "selector_id",
        "selector_type",
        "count_strategy",
        "requested_feature_count",
    ]
    records: list[dict[str, Any]] = []
    for keys, group in core.groupby(group_columns, dropna=False, sort=False):
        record = dict(zip(group_columns, keys))
        record["seed_count"] = int(group["seed"].nunique())
        for output_name, source_name in _AGGREGATE_COLUMNS.items():
            values = pd.to_numeric(group[source_name], errors="coerce")
            record[f"{output_name}_mean"] = values.mean()
            record[f"{output_name}_std"] = values.std(ddof=1)
            record[f"{output_name}_min"] = values.min()
            record[f"{output_name}_max"] = values.max()
        records.append(record)
    return pd.DataFrame(records)


def paired_baseline_comparisons(
    core: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute candidate-minus-baseline paired differences and fixed-df t CIs."""
    difference_fields = {
        "average_precision": "average_precision",
        "f1": "f1",
        "recall": "recall",
        "features": "selected_feature_count",
        "training_sec": "combined_train_wall_time_sec",
        "inference_sec": "attacked_validation_inference_wall_time_sec",
    }
    if core.empty:
        return pd.DataFrame(), pd.DataFrame()
    baseline = core.loc[core["selector_id"].eq("none")].set_index("seed")
    seed_records: list[dict[str, Any]] = []
    candidates = core.loc[core["selector_id"].ne("none")]
    for row in candidates.to_dict(orient="records"):
        seed = int(row["seed"])
        if seed not in baseline.index:
            continue
        anchor = baseline.loc[seed]
        record = {
            "seed": seed,
            "configuration_id": row["configuration_id"],
            "selector_id": row["selector_id"],
            "requested_feature_count": row["requested_feature_count"],
        }
        for name, field in difference_fields.items():
            record[f"{name}_difference"] = float(row[field]) - float(anchor[field])
        baseline_features = float(anchor["selected_feature_count"])
        record["feature_reduction_percent"] = (
            100.0
            * (baseline_features - float(row["selected_feature_count"]))
            / baseline_features
            if baseline_features
            else None
        )
        seed_records.append(record)
    per_seed = pd.DataFrame(seed_records)
    summary_records: list[dict[str, Any]] = []
    if not per_seed.empty:
        identity = ["configuration_id", "selector_id", "requested_feature_count"]
        value_columns = [
            column
            for column in per_seed.columns
            if column.endswith("_difference") or column == "feature_reduction_percent"
        ]
        for keys, group in per_seed.groupby(identity, dropna=False, sort=False):
            record = dict(zip(identity, keys))
            record["paired_seed_count"] = int(len(group))
            for column in value_columns:
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                mean = values.mean()
                std = values.std(ddof=1)
                half_width = (
                    T_CRITICAL_DF4_95 * std / math.sqrt(5.0)
                    if len(values) == 5 and np.isfinite(std)
                    else None
                )
                record[f"{column}_mean"] = mean
                record[f"{column}_ci95_low"] = (
                    mean - half_width if half_width is not None else None
                )
                record[f"{column}_ci95_high"] = (
                    mean + half_width if half_width is not None else None
                )
            summary_records.append(record)
    return per_seed, pd.DataFrame(summary_records)


def non_inferiority_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    """Apply aggregate relative-degradation margins with explicit zero handling."""
    if summary.empty:
        return pd.DataFrame()
    baseline_rows = summary.loc[summary["selector_id"].eq("none")]
    if len(baseline_rows) != 1:
        raise ValueError("Validation summary must contain one baseline configuration.")
    baseline = baseline_rows.iloc[0]
    records: list[dict[str, Any]] = []
    for margin in (0.0, 5.0, 10.0):
        for candidate in summary.loc[summary["selector_id"].ne("none")].to_dict(
            orient="records"
        ):
            record = {
                "configuration_id": candidate["configuration_id"],
                "selector_id": candidate["selector_id"],
                "selector_type": candidate["selector_type"],
                "count_strategy": candidate["count_strategy"],
                "requested_feature_count": candidate["requested_feature_count"],
                "selected_feature_count_mean": candidate["feature_count_mean"],
                "ap_f1_margin_percent": margin,
                "recall_margin_percent": 10.0,
                "semantics": (
                    "aggregate five-seed mean relative degradation; AP and F1 use the "
                    "sensitivity margin, recall uses the frozen 10 percent margin"
                ),
            }
            metric_passes = []
            for metric in ("average_precision", "f1", "recall"):
                degradation, denominator_status = _relative_degradation_percent(
                    baseline[f"{metric}_mean"], candidate[f"{metric}_mean"]
                )
                relative_difference, _ = _relative_difference_percent(
                    baseline[f"{metric}_mean"], candidate[f"{metric}_mean"]
                )
                allowed = 10.0 if metric == "recall" else margin
                record[f"{metric}_relative_difference_percent"] = relative_difference
                record[f"{metric}_relative_degradation_percent"] = degradation
                record[f"{metric}_denominator_status"] = denominator_status
                passed = degradation is not None and degradation <= allowed + 1e-12
                record[f"{metric}_preserving"] = passed
                metric_passes.append(passed)
            baseline_features = float(baseline["feature_count_mean"])
            record["feature_reduction_percent"] = (
                100.0
                * (baseline_features - float(candidate["feature_count_mean"]))
                / baseline_features
                if baseline_features
                else None
            )
            record["preservation_status"] = (
                "preserving" if all(metric_passes) else "not_preserving"
            )
            records.append(record)
    return pd.DataFrame(records)


def select_candidates(
    summary: pd.DataFrame, primary: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select one validation candidate per method and identify named comparators."""
    if summary.empty:
        return pd.DataFrame(), {"status": "unavailable", "reason": "no successful runs"}
    records: list[dict[str, Any]] = []
    candidates = summary.loc[summary["selector_id"].ne("none")]
    for selector_id, group in candidates.groupby("selector_id", sort=False):
        preservation = primary.loc[primary["selector_id"].eq(selector_id)]
        merged = group.merge(
            preservation[
                ["configuration_id", "preservation_status", "feature_reduction_percent"]
            ],
            on="configuration_id",
            how="left",
            validate="one_to_one",
        )
        preserving = merged.loc[merged["preservation_status"].eq("preserving")]
        if selector_id in _NATURAL_SELECTORS:
            chosen = merged.iloc[0]
            rationale = "natural_single_configuration"
        elif not preserving.empty:
            chosen = preserving.sort_values(
                ["feature_count_mean", "average_precision_mean", "f1_mean"],
                ascending=[True, False, False],
            ).iloc[0]
            rationale = "smallest_k_satisfying_primary_preservation"
        else:
            chosen = merged.sort_values(
                ["average_precision_mean", "f1_mean", "feature_count_mean"],
                ascending=[False, False, True],
            ).iloc[0]
            rationale = "validation_best_k_no_preserving_candidate"
        record = chosen.to_dict()
        record["selection_rationale"] = rationale
        record["selection_status"] = (
            "preserving"
            if chosen.get("preservation_status") == "preserving"
            else "not_preserving"
        )
        records.append(record)
    selected = pd.DataFrame(records)
    baseline = summary.loc[summary["selector_id"].eq("none")].iloc[0]
    preserving = selected.loc[selected["selection_status"].eq("preserving")]
    identity = {
        "schema_version": V06D_SCHEMA_VERSION,
        "selection_scope": "development_validation_only",
        "primary_preservation_rule": (
            "five-seed mean AP and F1 relative degradation <=5%; recall <=10%"
        ),
        "ranking_tie_break": "higher AP, higher F1, then fewer features",
        "baseline": _candidate_descriptor(baseline),
        "best_unsupervised": _best_descriptor(selected, "unsupervised"),
        "best_supervised": _best_descriptor(selected, "supervised"),
        "smallest_preserving": (
            None
            if preserving.empty
            else _candidate_descriptor(
                preserving.sort_values(
                    ["feature_count_mean", "average_precision_mean", "f1_mean"],
                    ascending=[True, False, False],
                ).iloc[0]
            )
        ),
        "per_method": [_candidate_descriptor(row) for _, row in selected.iterrows()],
        "test_model_selected": False,
    }
    return selected, identity


def selected_set_stability(core: pd.DataFrame) -> pd.DataFrame:
    """Calculate all pairwise selected-set Jaccard values across seeds."""
    records: list[dict[str, Any]] = []
    for configuration_id, group in core.groupby("configuration_id", sort=False):
        rows = sorted(group.to_dict(orient="records"), key=lambda row: row["seed"])
        values: list[float] = []
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1 :]:
                left_set = set(json.loads(left["selected_features_json"]))
                right_set = set(json.loads(right["selected_features_json"]))
                union = left_set | right_set
                jaccard = len(left_set & right_set) / len(union) if union else 1.0
                values.append(jaccard)
                records.append(
                    {
                        "configuration_id": configuration_id,
                        "selector_id": left["selector_id"],
                        "seed_left": left["seed"],
                        "seed_right": right["seed"],
                        "jaccard": jaccard,
                        "comparison_type": "pairwise",
                    }
                )
        if values:
            records.append(
                {
                    "configuration_id": configuration_id,
                    "selector_id": rows[0]["selector_id"],
                    "seed_left": None,
                    "seed_right": None,
                    "jaccard": float(np.mean(values)),
                    "jaccard_min": float(np.min(values)),
                    "jaccard_max": float(np.max(values)),
                    "pair_count": len(values),
                    "comparison_type": "summary",
                }
            )
    return pd.DataFrame(records)


def ranking_correlation_stability(rankings: pd.DataFrame) -> pd.DataFrame:
    """Use pandas rank correlation for meaningful cross-seed ranking comparisons."""
    required = {"selector_id", "seed", "feature", "rank", "ranking_meaningful"}
    if rankings.empty or not required.issubset(rankings.columns):
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    meaningful = rankings.loc[rankings["ranking_meaningful"].eq(True)].drop_duplicates(
        ["selector_id", "seed", "feature"]
    )
    for selector_id, group in meaningful.groupby("selector_id", sort=False):
        seeds = sorted(group["seed"].unique())
        for index, left_seed in enumerate(seeds):
            for right_seed in seeds[index + 1 :]:
                left = group.loc[group["seed"].eq(left_seed)].set_index("feature")["rank"]
                right = group.loc[group["seed"].eq(right_seed)].set_index("feature")["rank"]
                aligned = pd.concat([left, right], axis=1, join="inner").dropna()
                try:
                    spearman = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
                    kendall = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="kendall")
                    status = "valid" if len(aligned) >= 2 else "insufficient_features"
                except ImportError:
                    spearman = None
                    kendall = None
                    status = "optional_correlation_dependency_unavailable"
                records.append(
                    {
                        "selector_id": selector_id,
                        "seed_left": int(left_seed),
                        "seed_right": int(right_seed),
                        "common_feature_count": len(aligned),
                        "spearman_rank_correlation": spearman,
                        "kendall_rank_correlation": kendall,
                        "status": status,
                    }
                )
    return pd.DataFrame(records)


def deterministic_unsupervised_stability(
    core: pd.DataFrame, rankings: pd.DataFrame
) -> pd.DataFrame:
    """Report whether deterministic unsupervised outputs are identical by seed."""
    records: list[dict[str, Any]] = []
    unsupervised = core.loc[core["selector_type"].eq("unsupervised")]
    for configuration_id, group in unsupervised.groupby("configuration_id", sort=False):
        selected_hashes = group["selected_features_sha256"].dropna().unique()
        selector_id = str(group["selector_id"].iloc[0])
        method_rankings = rankings.loc[rankings["selector_id"].eq(selector_id)]
        ranking_hashes = []
        if not method_rankings.empty and method_rankings["ranking_meaningful"].any():
            for _, seed_group in method_rankings.drop_duplicates(
                ["seed", "feature"]
            ).groupby("seed"):
                ordered = seed_group.sort_values("rank")["feature"].tolist()
                ranking_hashes.append(fingerprint_feature_names(ordered))
        records.append(
            {
                "configuration_id": configuration_id,
                "selector_id": selector_id,
                "seed_count": int(group["seed"].nunique()),
                "selected_sets_identical": len(selected_hashes) == 1,
                "rankings_identical": (
                    len(set(ranking_hashes)) == 1 if ranking_hashes else None
                ),
                "deterministic_identical_indicator": (
                    len(selected_hashes) == 1
                    and (not ranking_hashes or len(set(ranking_hashes)) == 1)
                ),
            }
        )
    return pd.DataFrame(records)


def _validate_manifest(
    data: DevelopmentExperimentData, config: V06ProtocolConfig, seed: int
) -> dict[str, Any]:
    observed = len(data.candidate_features)
    invalid = [count for count in config.candidate_feature_counts if count > observed]
    if invalid:
        raise ValueError(
            f"Configured K values exceed the {observed}-feature manifest: {invalid}"
        )
    return {
        "seed": seed,
        "status": "valid",
        "observed_feature_count": observed,
        "expected_v02_feature_count": EXPECTED_V02_FEATURE_COUNT,
        "expected_43_status": (
            "matches_expected_43" if observed == EXPECTED_V02_FEATURE_COUNT else "not_43"
        ),
        "configured_k": list(config.candidate_feature_counts),
        "all_k_within_manifest": True,
        "candidate_features_sha256": fingerprint_feature_names(data.candidate_features),
    }


def _flatten_successful_run(
    record: Mapping[str, Any],
    item: Mapping[str, Any],
    data: DevelopmentExperimentData,
    model: Any,
    selector: BaseFeatureSelector | None,
) -> dict[str, Any]:
    metrics = record["metrics"]
    resources = record["resource_measurement"]
    selected_features = list(record["selected_features"])
    clean_selected = (
        data.validation.clean_features.loc[:, selected_features].copy()
        if selector is None
        else selector.transform(data.validation.clean_features)
    )
    attacked_selected = (
        data.validation.features.loc[:, selected_features].copy()
        if selector is None
        else selector.transform(data.validation.features)
    )
    clean_raw = model.predict_frame(clean_selected)
    clean_predictions = pd.DataFrame(
        {
            "record_id": data.validation.features["row_id"].to_numpy(copy=True),
            "anomaly_score": clean_raw["anomaly_score"].to_numpy(copy=True),
            "anomaly_label": clean_raw["anomaly_label"].to_numpy(copy=True),
        }
    )
    attacked_raw = model.predict_frame(attacked_selected)
    attacked_predictions = pd.DataFrame(
        {
            "record_id": data.validation.features["row_id"].to_numpy(copy=True),
            "anomaly_score": attacked_raw["anomaly_score"].to_numpy(copy=True),
            "anomaly_label": attacked_raw["anomaly_label"].to_numpy(copy=True),
        }
    )
    visible_mask = detector_visible_mask(clean_selected, attacked_selected)
    visible = evaluate_visible_attacks(
        data.validation.ground_truth, attacked_predictions, visible_mask
    )
    paired = evaluate_paired_detection(
        data.validation.ground_truth, clean_predictions, attacked_predictions
    )
    clean_metrics = _normalize_external_metrics(
        evaluate_detection(data.validation.ground_truth, clean_predictions)
    )
    visible_metrics = _normalize_external_metrics(visible) if visible is not None else {}
    attack = record["attack_configuration"]
    training = record["training_provenance"]
    selector_provenance = record["selector_provenance"]
    selector_details = record["selector"]
    split = record["split_fingerprints"]
    row: dict[str, Any] = {
        "schema_version": V06D_SCHEMA_VERSION,
        "source_record_schema_version": record["schema_version"],
        "status": "success",
        "error_type": None,
        "error_message": None,
        "run_id": record["run_id"],
        "phase": record["phase"],
        "seed": record["seed"],
        "evaluation_split": record["evaluation_split"],
        "configuration_id": item["configuration_id"],
        "selector_id": item["selector_id"],
        "selector_type": item["selector_type"],
        "selector_mode": item["selector_type"],
        "selector_method": selector_details["method"],
        "selector_parameters_json": json.dumps(
            _json_safe(selector_details["parameters"]), sort_keys=True
        ),
        "selector_parameters": json.dumps(
            _json_safe(selector_details["parameters"]), sort_keys=True
        ),
        "count_strategy": item["count_strategy"],
        "requested_feature_count": item["requested_feature_count"],
        "requested_K": item["requested_feature_count"],
        "candidate_feature_count": record["candidate_feature_count"],
        "candidate_features_sha256": fingerprint_feature_names(data.candidate_features),
        "selected_feature_count": record["selected_feature_count"],
        "actual_feature_count": record["selected_feature_count"],
        "selected_features_json": json.dumps(selected_features),
        "selected_features": json.dumps(selected_features),
        "selected_features_sha256": record["selected_features_sha256"],
        "feature_fingerprint": record["selected_features_sha256"],
        "model_id": record["model_id"],
        "model_parameters_json": json.dumps(record["model_parameters"], sort_keys=True),
        "prediction_threshold": record["prediction_threshold"],
        "threshold_policy": record["threshold_policy"],
        "prediction_sha256": record["prediction_sha256"],
        "model_state_sha256": record["model_state_sha256"],
        "attack_type": attack["type"],
        "attack_rate": attack["rate"],
        "attack_severity": attack["severity"],
        "attack_config_path": attack["config_path"],
        "attack_generator_metadata_json": json.dumps(
            _json_safe(attack["generator_metadata"]), sort_keys=True
        ),
        "experiment_id": attack["generator_metadata"].get("experiment_id"),
        "training_split": training["split"],
        "training_row_ids_sha256": training["row_ids_sha256"],
        "training_clean_features_sha256": training["clean_features_sha256"],
        "training_attacked_features_sha256": training["attacked_features_sha256"],
        "training_data_fingerprint": training["attacked_features_sha256"],
        "training_labels_sha256": training["labels_sha256"],
        "training_ground_truth_sha256": data.train.ground_truth_sha256,
        "training_manifest_sha256": data.train.manifest_sha256,
        "training_attack_metadata_sha256": data.train.attack_metadata_sha256,
        "selector_labels_used": selector_provenance["labels_used"],
        "selector_training_rows_sha256": selector_provenance["training_rows_sha256"],
        "selector_training_features_sha256": selector_provenance[
            "training_features_sha256"
        ],
        "selector_training_labels_sha256": selector_provenance[
            "training_labels_sha256"
        ],
        "selector_training_label_source": selector_provenance[
            "training_label_source"
        ],
        "evaluation_rows_sha256": split["evaluation_rows_sha256"],
        "evaluation_features_sha256": split["evaluation_features_sha256"],
        "validation_data_fingerprint": split["evaluation_features_sha256"],
        "evaluation_labels_sha256": split["evaluation_labels_sha256"],
        "evaluation_ground_truth_sha256": data.validation.ground_truth_sha256,
        "evaluation_manifest_sha256": data.validation.manifest_sha256,
        "evaluation_attack_metadata_sha256": data.validation.attack_metadata_sha256,
        "leakage_audit_status": record["leakage_audit"]["status"],
        "leakage_checks_json": json.dumps(record["leakage_audit"]["checks"]),
        "warnings_json": json.dumps(record["warnings"]),
        "manifest_expected_feature_count": EXPECTED_V02_FEATURE_COUNT,
        "manifest_expected_43_status": (
            "matches_expected_43"
            if record["candidate_feature_count"] == EXPECTED_V02_FEATURE_COUNT
            else "not_43"
        ),
    }
    for name in (
        "precision",
        "recall",
        "f1",
        "average_precision",
        "pr_auc_trapezoidal",
        "pr_auc_trapezoidal_method",
        "pr_auc_status",
        "roc_auc",
        "roc_auc_status",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "evaluated_records",
        "attacked_records",
        "attack_prevalence",
    ):
        row[name] = metrics.get(name)
    row.update(
        {
            "tp": metrics["true_positives"],
            "tn": metrics["true_negatives"],
            "fp": metrics["false_positives"],
            "fn": metrics["false_negatives"],
            "TP": metrics["true_positives"],
            "TN": metrics["true_negatives"],
            "FP": metrics["false_positives"],
            "FN": metrics["false_negatives"],
            "confusion_matrix_json": json.dumps(metrics["confusion_matrix"]),
        }
    )
    for prefix, values in (("clean_validation", clean_metrics), ("visible_attack", visible_metrics)):
        for key, value in values.items():
            if key != "confusion_matrix":
                row[f"{prefix}_{key}"] = value
        row[f"{prefix}_confusion_matrix_json"] = (
            json.dumps(values["confusion_matrix"]) if values else None
        )
    for key, value in paired.items():
        row[f"paired_induced_{key}"] = value
    visible_count = int(visible_metrics.get("attacked_records", 0))
    row["visible_attacked_records"] = visible_count
    row["invisible_attacked_records"] = int(metrics["attacked_records"]) - visible_count
    row["visible_only_recall"] = visible_metrics.get("recall")
    for name in (
        "selector_fit_wall_time_sec",
        "model_fit_wall_time_sec",
        "combined_train_wall_time_sec",
        "attacked_validation_inference_wall_time_sec",
        "per_record_inference_sec",
        "peak_rss_bytes",
        "peak_rss_mib",
        "serialized_model_bytes",
        "serialized_model_kib",
        "model_artifact_path",
    ):
        row[name] = resources[name]
    row["resource_measurement_performed"] = resources["performed"]
    row["resource_energy_claim"] = resources["energy_claim"]
    row["selector_fit_time_seconds"] = resources["selector_fit_wall_time_sec"]
    row["model_train_time_seconds"] = resources["model_fit_wall_time_sec"]
    row["combined_train_time_seconds"] = resources["combined_train_wall_time_sec"]
    row["inference_time_seconds"] = resources[
        "attacked_validation_inference_wall_time_sec"
    ]
    row["inference_time_per_record"] = resources["per_record_inference_sec"]
    row["peak_RSS_MiB"] = resources["peak_rss_mib"]
    row["serialized_model_size_bytes"] = resources["serialized_model_bytes"]
    return row


def _normalize_external_metrics(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    if metrics is None:
        return {}
    evaluated = int(metrics["evaluated_records"])
    attacked = int(metrics["attacked_records"])
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "average_precision": metrics["average_precision"],
        "pr_auc_trapezoidal": metrics["pr_auc"],
        "pr_auc_trapezoidal_method": metrics["pr_auc_method"],
        "pr_auc_status": (
            "valid" if metrics["pr_auc"] is not None else "undefined_single_class"
        ),
        "roc_auc": metrics["roc_auc"],
        "roc_auc_status": (
            "valid" if metrics["roc_auc"] is not None else "undefined_single_class"
        ),
        "true_positives": metrics["true_positives"],
        "true_negatives": metrics["true_negatives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "evaluated_records": evaluated,
        "attacked_records": attacked,
        "attack_prevalence": float(attacked / evaluated) if evaluated else None,
        "confusion_matrix": metrics["confusion_matrix"],
    }


def _ranking_records(
    run_id: str,
    seed: int,
    item: Mapping[str, Any],
    selector: BaseFeatureSelector | None,
) -> list[dict[str, Any]]:
    if selector is None:
        return []
    ranking_frame = getattr(selector, "ranking_frame", None)
    if ranking_frame is None:
        return []
    records = []
    for row in ranking_frame().to_dict(orient="records"):
        records.append(
            {
                "run_id": run_id,
                "seed": seed,
                "configuration_id": item["configuration_id"],
                "selector_id": item["selector_id"],
                "selector_type": item["selector_type"],
                "requested_feature_count": item["requested_feature_count"],
                "feature": row["feature"],
                "rank": row["rank"],
                "score": row.get("score"),
                "score_status": row.get("score_status"),
                "selected": row.get("selected"),
                "ranking_meaningful": True,
            }
        )
    return records


def _failure_row(
    seed: int,
    item: Mapping[str, Any],
    error: Exception,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": V06D_SCHEMA_VERSION,
        "status": "failed",
        "run_id": run_id or f"v06d_s{seed}_{item['configuration_id']}",
        "seed": seed,
        "evaluation_split": "validation",
        "configuration_id": item["configuration_id"],
        "selector_id": item["selector_id"],
        "selector_type": item["selector_type"],
        "count_strategy": item["count_strategy"],
        "requested_feature_count": item["requested_feature_count"],
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def _matrix_leakage_audit(
    core: pd.DataFrame,
    *,
    manifest_checks: Sequence[Mapping[str, Any]],
    load_counts: Mapping[int, int],
    expected_runs: int,
    candidate_selection: Mapping[str, Any],
) -> dict[str, Any]:
    successful = core.loc[core["status"].eq("success")]
    checks = [
        _audit_check("test_accessed", True, value=False),
        _audit_check("test_loader_called", True, value=False),
        _audit_check(
            "validation_only_evaluation",
            bool(successful.empty or successful["evaluation_split"].eq("validation").all()),
            observed_splits=sorted(successful["evaluation_split"].dropna().unique()),
        ),
        _audit_check(
            "exact_frozen_seeds",
            tuple(sorted(core["seed"].unique())) == V06D_SEEDS,
            expected=list(V06D_SEEDS),
            observed=sorted(core["seed"].unique()),
        ),
        _audit_check(
            "twelve_configurations_per_seed",
            all(int(count) == 12 for count in core.groupby("seed").size()),
            counts={str(key): int(value) for key, value in core.groupby("seed").size().items()},
        ),
        _audit_check(
            "development_data_loaded_once_per_seed",
            all(load_counts[seed] == 1 for seed in V06D_SEEDS),
            counts={str(key): value for key, value in load_counts.items()},
        ),
        _audit_check(
            "configured_k_validated_against_manifest",
            all(check.get("all_k_within_manifest") is True for check in manifest_checks),
            manifest_checks=list(manifest_checks),
        ),
        _audit_check(
            "paired_attacks_reused_within_seed",
            _single_value_per_seed(successful, "evaluation_features_sha256")
            and _single_value_per_seed(successful, "evaluation_labels_sha256"),
        ),
        _audit_check(
            "train_validation_rows_disjoint",
            _embedded_check_passes(successful, "train_validation_rows_disjoint"),
        ),
        _audit_check(
            "preprocessing_fitted_on_training_rows_only",
            _embedded_check_passes(
                successful, "v02_preprocessing_fitted_on_training_rows"
            ),
        ),
        _audit_check(
            "candidate_manifest_preserved",
            _embedded_check_passes(successful, "v02_candidate_manifest_preserved"),
        ),
        _audit_check(
            "attack_metadata_absent_from_model_features",
            _embedded_check_passes(
                successful, "attack_metadata_absent_from_feature_matrices"
            ),
        ),
        _audit_check(
            "row_ids_are_not_model_features",
            _selected_features_exclude(successful, {"row_id", "record_id"}),
        ),
        _audit_check(
            "late_delivery_risk_is_not_cyber_ground_truth_or_model_feature",
            _selected_features_exclude(successful, {"Late_delivery_risk"}),
            cyber_ground_truth="controlled experiment-generated is_attack",
        ),
        _audit_check(
            "native_suspected_fraud_is_not_selector_input",
            _selected_features_exclude(successful, {"Order Status", "SUSPECTED_FRAUD"}),
        ),
        _audit_check(
            "model_outputs_are_not_selector_inputs",
            _selected_features_exclude(
                successful, {"anomaly_score", "anomaly_label", "prediction"}
            ),
        ),
        _audit_check(
            "selector_fitted_on_training_rows_only",
            _embedded_check_passes(
                successful, "selector_fitted_on_training_rows_only"
            ),
        ),
        _audit_check(
            "unsupervised_clean_train_and_supervised_controlled_label_governance",
            _embedded_check_passes(successful, "selector_training_representation")
            and _embedded_check_passes(successful, "selector_label_governance"),
        ),
        _audit_check(
            "validation_transform_did_not_refit_selector",
            _embedded_check_passes(
                successful, "validation_transform_did_not_refit_selector"
            ),
        ),
        _audit_check(
            "validation_rows_never_used_during_selector_fit",
            _embedded_check_passes(
                successful, "selector_fitted_on_training_rows_only"
            ),
        ),
        _audit_check(
            "selected_feature_order_is_canonical",
            _embedded_check_passes(
                successful, "selected_features_preserve_canonical_order"
            ),
        ),
        _audit_check(
            "validation_configuration_selection_recorded",
            candidate_selection.get("selection_scope")
            == "development_validation_only"
            and candidate_selection.get("test_model_selected") is False,
        ),
        _audit_check(
            "frozen_model_and_threshold",
            _embedded_check_passes(successful, "fixed_v05_decision_tree")
            and _embedded_check_passes(successful, "fixed_prediction_threshold"),
        ),
        _audit_check(
            "measured_resources_present",
            bool(
                successful.empty
                or successful["resource_measurement_performed"].eq(True).all()
            ),
        ),
        _audit_check(
            "all_expected_runs_recorded",
            len(core) == expected_runs,
            expected=expected_runs,
            observed=len(core),
        ),
        _audit_check(
            "no_run_failures",
            bool(core["status"].eq("success").all()),
            failed_runs=int(core["status"].ne("success").sum()),
        ),
    ]
    return {
        "schema_version": V06D_SCHEMA_VERSION,
        "status": "PASS" if all(check["passed"] for check in checks) else "FAIL",
        "scope": "development_train_and_validation_only",
        "test_accessed": False,
        "checks": checks,
    }


def _audit_check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _single_value_per_seed(frame: pd.DataFrame, column: str) -> bool:
    return bool(frame.empty or frame.groupby("seed")[column].nunique().le(1).all())


def _embedded_check_passes(frame: pd.DataFrame, check_name: str) -> bool:
    if frame.empty:
        return True
    for serialized in frame["leakage_checks_json"]:
        checks = json.loads(serialized)
        matching = [check for check in checks if check.get("name") == check_name]
        if len(matching) != 1 or matching[0].get("passed") is not True:
            return False
    return True


def _selected_features_exclude(frame: pd.DataFrame, forbidden: set[str]) -> bool:
    if frame.empty:
        return True
    return all(
        forbidden.isdisjoint(json.loads(serialized))
        for serialized in frame["selected_features_json"]
    )


def _relative_degradation_percent(
    baseline: Any, candidate: Any
) -> tuple[float | None, str]:
    if pd.isna(baseline) or pd.isna(candidate):
        return None, "undefined_metric"
    baseline_value = float(baseline)
    candidate_value = float(candidate)
    if baseline_value == 0.0:
        if candidate_value >= 0.0:
            return 0.0, "baseline_zero_candidate_nonnegative"
        return None, "baseline_zero_candidate_negative"
    return max(0.0, 100.0 * (baseline_value - candidate_value) / baseline_value), "nonzero"


def _relative_difference_percent(
    baseline: Any, candidate: Any
) -> tuple[float | None, str]:
    if pd.isna(baseline) or pd.isna(candidate):
        return None, "undefined_metric"
    baseline_value = float(baseline)
    candidate_value = float(candidate)
    if baseline_value == 0.0:
        if candidate_value == 0.0:
            return 0.0, "both_zero"
        return None, "baseline_zero"
    return 100.0 * (candidate_value - baseline_value) / baseline_value, "nonzero"


def _best_descriptor(frame: pd.DataFrame, selector_type: str) -> dict[str, Any] | None:
    candidates = frame.loc[frame["selector_type"].eq(selector_type)]
    if candidates.empty:
        return None
    chosen = candidates.sort_values(
        ["average_precision_mean", "f1_mean", "feature_count_mean"],
        ascending=[False, False, True],
    ).iloc[0]
    return _candidate_descriptor(chosen)


def _candidate_descriptor(row: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    return {
        "configuration_id": row["configuration_id"],
        "selector_id": row["selector_id"],
        "selector_type": row["selector_type"],
        "requested_feature_count": _none_if_nan(row.get("requested_feature_count")),
        "selected_feature_count_mean": _none_if_nan(row.get("feature_count_mean")),
        "average_precision_mean": _none_if_nan(row.get("average_precision_mean")),
        "f1_mean": _none_if_nan(row.get("f1_mean")),
        "recall_mean": _none_if_nan(row.get("recall_mean")),
        "selection_status": row.get("selection_status"),
        "selection_rationale": row.get("selection_rationale"),
    }


def _resource_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    identity = [
        "configuration_id",
        "selector_id",
        "selector_type",
        "requested_feature_count",
        "seed_count",
    ]
    resources = [
        column
        for column in summary.columns
        if column.startswith(
            ("feature_count_", "training_sec_", "inference_sec_", "peak_rss_mib_", "model_size_bytes_")
        )
    ]
    return summary.loc[:, identity + resources].copy()


def _run_metadata(
    *,
    config_path: Path,
    processed_dir: Path,
    results_dir: Path,
    models_dir: Path,
    base_config: V06ProtocolConfig,
    manifest_checks: Sequence[Mapping[str, Any]],
    core: pd.DataFrame,
    figure_paths: Sequence[Path],
    candidate_identity: Mapping[str, Any],
) -> dict[str, Any]:
    package_versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "matplotlib", "psutil", "PyYAML"):
        try:
            package_versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            package_versions[name] = None
    output_hashes = {}
    for path in sorted(results_dir.glob("*")):
        if path.is_file() and path.name != "run_metadata.json":
            output_hashes[path.name] = _sha256_file(path)
    metadata_path = processed_dir / "dataset_metadata.json"
    return {
        "schema_version": V06D_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_scope": "development_train_and_validation_only",
        "test_accessed": False,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "package_versions": package_versions,
        "protocol_config": base_config.to_dict(),
        "protocol_config_path": str(config_path),
        "protocol_config_sha256": _sha256_file(config_path),
        "attack_config_sha256": _sha256_file(base_config.attack_config_path),
        "attack_generator_config": load_attack_config(base_config.attack_config_path),
        "processed_dataset_metadata_sha256": (
            _sha256_file(metadata_path) if metadata_path.is_file() else None
        ),
        "manifest_validation": list(manifest_checks),
        "run_count": len(core),
        "successful_run_count": int(core["status"].eq("success").sum()),
        "failed_run_count": int(core["status"].ne("success").sum()),
        "input_fingerprints": {
            "training_rows": sorted(
                core.get("training_row_ids_sha256", pd.Series(dtype=str)).dropna().unique()
            ),
            "validation_rows": sorted(
                core.get("evaluation_rows_sha256", pd.Series(dtype=str)).dropna().unique()
            ),
            "validation_attacks": sorted(
                core.get("evaluation_features_sha256", pd.Series(dtype=str)).dropna().unique()
            ),
        },
        "results_dir": str(results_dir),
        "models_dir": str(models_dir),
        "figure_paths": [str(path) for path in figure_paths],
        "output_sha256": output_hashes,
        "candidate_selection": dict(candidate_identity),
    }


def _write_figures(summary: pd.DataFrame, figure_dir: Path) -> list[Path]:
    if summary.empty:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    ordered = summary.sort_values("feature_count_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(ordered["feature_count_mean"], ordered["average_precision_mean"], color="#24536b")
    for row in ordered.itertuples(index=False):
        ax.annotate(row.configuration_id, (row.feature_count_mean, row.average_precision_mean), fontsize=6)
    ax.set_xlabel("Mean selected feature count")
    ax.set_ylabel("Mean validation average precision")
    ax.set_title("V0.6-D validation performance and feature count")
    paths.append(_save_figure(fig, figure_dir / "validation_ap_vs_features.png", plt))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    positions = np.arange(len(ordered))
    ax.bar(positions, ordered["training_sec_mean"], color="#8a5a44")
    ax.set_xticks(positions, ordered["configuration_id"], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Mean selector + model fit seconds")
    ax.set_title("V0.6-D measured validation training time")
    paths.append(_save_figure(fig, figure_dir / "validation_training_resources.png", plt))
    return paths


def _save_figure(fig: Any, path: Path, pyplot: Any) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    pyplot.close(fig)
    return path


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [_json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _none_if_nan(value: Any) -> Any:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen V0.6-D five-seed validation-only matrix."
    )
    parser.add_argument("--config", type=Path, default=FEATURE_SELECTION_CONFIG_FILE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args(argv)
    result = run_v06d_validation_matrix(
        config_path=args.config,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
        processed_dir=args.processed_dir,
    )
    print(
        f"V0.6-D {result['status']}: {result['run_count']} validation runs; "
        f"results={result['results_dir']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
