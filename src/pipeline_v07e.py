"""Artifact-only statistical and performance-efficiency analysis for V0.7-E."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

from src.config import PROJECT_ROOT
import src.pipeline_v07d as v07d


STAGE = "V0.7-E"
SCHEMA_VERSION = "v0.7-e-statistical-analysis-1"
BASELINE = "none_natural"
CONFIGURATIONS = (
    BASELINE,
    "pairwise_correlation_filter_natural",
    "mutual_information_select_k_best_k11",
)
REDUCED_CONFIGURATIONS = CONFIGURATIONS[1:]
CLASSIFIERS = ("decision_tree", "logistic_regression")
SEEDS = (42, 43, 44, 45, 46)
PHASES = ("training", "inference")
REPETITIONS = tuple(range(1, 11))
T_CRITICAL_95_DF4 = 2.7764451051977987
HIGH_VARIABILITY_CV_PERCENT = 20.0
MODERATE_VARIABILITY_CV_PERCENT = 10.0
RESULTS_DIR = PROJECT_ROOT / "results" / "green_evaluation"
FINAL_TEST_DIR = PROJECT_ROOT / "results" / "feature_selection" / "final_test"
ROBUSTNESS_DIR = PROJECT_ROOT / "results" / "feature_selection" / "robustness"
STAGING_NAME = ".v07e_staging"

METRICS = (
    "wall_time_sec",
    "process_cpu_time_sec",
    "absolute_peak_rss_mib",
    "incremental_peak_rss_mib",
    "per_operation_latency_sec",
    "per_record_latency_sec",
    "throughput_records_sec",
)
STATIC_METRICS = ("feature_count", "selected_input_bytes", "serialized_model_bytes")
METRIC_UNITS = {
    "wall_time_sec": "seconds",
    "process_cpu_time_sec": "seconds",
    "absolute_peak_rss_mib": "MiB",
    "incremental_peak_rss_mib": "MiB",
    "per_operation_latency_sec": "seconds/operation",
    "per_record_latency_sec": "seconds/record",
    "throughput_records_sec": "records/second",
    "feature_count": "features",
    "selected_input_bytes": "bytes",
    "serialized_model_bytes": "bytes",
}
MAXIMIZE_METRICS = {"throughput_records_sec"}
TIMING_VARIABILITY_METRICS = {
    "wall_time_sec",
    "process_cpu_time_sec",
    "per_operation_latency_sec",
    "per_record_latency_sec",
    "throughput_records_sec",
}
PREDICTIVE_METRICS = ("average_precision", "f1", "recall", "precision", "roc_auc")
PRIMARY_SCENARIO = "mixed__r0p05__medium"
PARETO_OBJECTIVES = (
    ("average_precision", "maximize"),
    ("f1", "maximize"),
    ("recall", "maximize"),
    ("feature_count", "minimize"),
    ("training_wall_time_sec", "minimize"),
    ("inference_per_operation_latency_sec", "minimize"),
    ("inference_selected_input_bytes", "minimize"),
    ("serialized_model_bytes", "minimize"),
)


class V07EError(RuntimeError):
    """Raised when analysis inputs or outputs violate the frozen protocol."""


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Return sample CV as a percentage, using the absolute mean denominator."""

    if len(values) < 2:
        raise V07EError("CV requires at least two observations.")
    mean = statistics.fmean(values)
    if math.isclose(mean, 0.0, abs_tol=1e-15):
        return 0.0 if all(math.isclose(value, 0.0, abs_tol=1e-15) for value in values) else math.inf
    return 100.0 * statistics.stdev(values) / abs(mean)


def t95_interval(values: Sequence[float]) -> tuple[float, float]:
    """Return a two-sided 95% t interval for the fixed five-seed design."""

    if len(values) != 5:
        raise V07EError("Scientific confidence intervals require exactly five seed values.")
    mean = statistics.fmean(values)
    margin = T_CRITICAL_95_DF4 * statistics.stdev(values) / math.sqrt(5)
    return mean - margin, mean + margin


def relative_change_percent(reduced: float, baseline: float) -> float:
    """Use 100*(reduced-baseline)/baseline; negative denotes a reduction."""

    if math.isclose(baseline, 0.0, abs_tol=1e-15):
        raise V07EError("Relative change is undefined for a zero baseline.")
    return 100.0 * (reduced - baseline) / baseline


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V07EError(f"Cannot read required JSON artifact: {path}") from exc


def _verify_hash_manifest(directory: Path, manifest_path: Path) -> dict[str, str]:
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or not manifest:
        raise V07EError(f"Invalid artifact hash manifest: {manifest_path}")
    verified: dict[str, str] = {}
    for name, expected in manifest.items():
        path = directory / name
        actual = v07d.sha256_file(path)
        if actual != expected:
            raise V07EError(f"Artifact hash mismatch: {path}")
        verified[str(path)] = actual
    verified[str(manifest_path)] = v07d.sha256_file(manifest_path)
    return verified


def load_v07d_observations(
    results_dir: Path | str = RESULTS_DIR,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    """Load and validate the complete immutable V0.7-D computational matrix."""

    root = Path(results_dir)
    hashes = _verify_hash_manifest(root, root / "v07d_artifact_hashes.json")
    manifest = _read_json(root / "v07d_run_manifest.json")
    governance = _read_json(root / "v07d_governance_audit.json")
    rows = _read_json(root / "computational_observations.json")
    failures = _read_json(root / "measurement_failures.json")
    if (
        manifest.get("stage") != "V0.7-D"
        or manifest.get("status") != "COMPLETED"
        or manifest.get("run_completeness") != "COMPLETE"
        or manifest.get("direct_energy_decision") != "DIRECT_ENERGY_UNAVAILABLE"
        or manifest.get("electrical_energy_reported") is not False
        or manifest.get("carbon_reported") is not False
        or governance.get("status") != "PASS"
        or not all(check.get("passed") is True for check in governance.get("checks", ()))
    ):
        raise V07EError("V0.7-D completion and governance evidence is not valid.")
    if not isinstance(rows, list) or len(rows) != 600 or failures != []:
        raise V07EError("V0.7-E requires exactly 600 retained rows and zero failures.")
    if any(row.get("status") != "SUCCESS" for row in rows):
        raise V07EError("Failed V0.7-D observations are not permitted for this fixed analysis.")
    if Counter(row["phase"] for row in rows) != {"training": 300, "inference": 300}:
        raise V07EError("V0.7-D must contain 300 training and 300 inference rows.")
    if (
        set(row["configuration_id"] for row in rows) != set(CONFIGURATIONS)
        or set(row["classifier"] for row in rows) != set(CLASSIFIERS)
        or set(row["seed"] for row in rows) != set(SEEDS)
        or len({row["environment_id"] for row in rows}) != 1
    ):
        raise V07EError("V0.7-D matrix dimensions or environment are invalid.")
    expected = {
        (classifier, configuration, seed, phase, repetition)
        for classifier in CLASSIFIERS
        for configuration in CONFIGURATIONS
        for seed in SEEDS
        for phase in PHASES
        for repetition in REPETITIONS
    }
    actual = {
        (
            row["classifier"],
            row["configuration_id"],
            row["seed"],
            row["phase"],
            row["outer_repetition"],
        )
        for row in rows
    }
    if actual != expected or len(actual) != len(rows):
        raise V07EError("V0.7-D observations do not cover each planned repetition exactly once.")
    with (root / "computational_observations.csv").open(encoding="utf-8", newline="") as handle:
        csv_ids = [row["observation_id"] for row in csv.DictReader(handle)]
    if csv_ids != [row["observation_id"] for row in rows]:
        raise V07EError("V0.7-D JSON/CSV observation parity failed.")
    forbidden = ("joule", "energy_consum", "carbon", "co2")
    if any(any(marker in key.lower() for marker in forbidden) for row in rows for key in row):
        raise V07EError("Electrical-energy or carbon outcomes cannot enter V0.7-E.")
    return rows, manifest, hashes


def build_seed_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate ten measurement repetitions before any five-seed comparison."""

    groups: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["classifier"], row["configuration_id"], int(row["seed"]), row["phase"])].append(row)
    if len(groups) != 60 or sum(len(group) for group in groups.values()) != len(rows):
        raise V07EError("Expected exactly 60 ten-repetition aggregation groups.")
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        classifier, configuration, seed, phase = key
        group = groups[key]
        if len(group) != 10 or {int(row["outer_repetition"]) for row in group} != set(REPETITIONS):
            raise V07EError("Each seed summary must aggregate exactly repetitions 1-10.")
        feature_counts = {int(row["feature_count"]) for row in group}
        input_bytes = {int(row["selected_input_bytes"]) for row in group}
        model_bytes = {int(row["serialized_model_bytes"]) for row in group}
        workload_hashes = {row["workload_sha256"] for row in group}
        if any(len(values) != 1 for values in (feature_counts, input_bytes, model_bytes, workload_hashes)):
            raise V07EError("Static evidence changed within a seed/phase repetition block.")
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            output.append(
                {
                    "stage": STAGE,
                    "classifier": classifier,
                    "configuration_id": configuration,
                    "seed": seed,
                    "phase": phase,
                    "metric": metric,
                    "unit": METRIC_UNITS[metric],
                    "repetition_count": 10,
                    "mean": statistics.fmean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values),
                    "min": min(values),
                    "max": max(values),
                    "coefficient_of_variation_percent": coefficient_of_variation(values),
                    "feature_count": feature_counts.pop() if len(feature_counts) == 1 else None,
                    "selected_input_bytes": input_bytes.pop() if len(input_bytes) == 1 else None,
                    "serialized_model_bytes": model_bytes.pop() if len(model_bytes) == 1 else None,
                    "workload_sha256": next(iter(workload_hashes)),
                    "environment_id": group[0]["environment_id"],
                    "measurement_provenance": "DIRECT_COMPUTATIONAL",
                    "aggregation_scope": "ten_measurement_repetitions_within_seed",
                }
            )
            if feature_counts == set():
                feature_counts = {int(group[0]["feature_count"])}
            if input_bytes == set():
                input_bytes = {int(group[0]["selected_input_bytes"])}
            if model_bytes == set():
                model_bytes = {int(group[0]["serialized_model_bytes"])}
    if len(output) != 420:
        raise V07EError("Seed aggregation must produce 420 metric summaries.")
    return output


def _summary_groups(seed_summaries: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int, str], dict[str, float]]:
    groups: dict[tuple[str, str, int, str], dict[str, float]] = defaultdict(dict)
    for row in seed_summaries:
        key = (row["classifier"], row["configuration_id"], int(row["seed"]), row["phase"])
        groups[key][row["metric"]] = float(row["mean"])
        groups[key].update(
            {
                "feature_count": float(row["feature_count"]),
                "selected_input_bytes": float(row["selected_input_bytes"]),
                "serialized_model_bytes": float(row["serialized_model_bytes"]),
            }
        )
    return groups


def build_paired_comparisons(seed_summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pair each reduced configuration with K43 by classifier, phase, and seed."""

    groups = _summary_groups(seed_summaries)
    output: list[dict[str, Any]] = []
    for classifier in CLASSIFIERS:
        for reduced in REDUCED_CONFIGURATIONS:
            for phase in PHASES:
                for metric in (*METRICS, *STATIC_METRICS):
                    pairs = []
                    for seed in SEEDS:
                        baseline_value = groups[(classifier, BASELINE, seed, phase)][metric]
                        reduced_value = groups[(classifier, reduced, seed, phase)][metric]
                        difference = reduced_value - baseline_value
                        relative = relative_change_percent(reduced_value, baseline_value)
                        pairs.append((seed, baseline_value, reduced_value, difference, relative))
                    differences = [pair[3] for pair in pairs]
                    relatives = [pair[4] for pair in pairs]
                    ci_low, ci_high = t95_interval(differences)
                    relative_ci_low, relative_ci_high = t95_interval(relatives)
                    for seed, baseline_value, reduced_value, difference, relative in pairs:
                        output.append(
                            {
                                "stage": STAGE,
                                "comparison_id": f"{classifier}__{reduced}__vs__{BASELINE}__{phase}__{metric}",
                                "classifier": classifier,
                                "configuration_id": reduced,
                                "reference_configuration_id": BASELINE,
                                "seed": seed,
                                "phase": phase,
                                "metric": metric,
                                "unit": METRIC_UNITS[metric],
                                "baseline_seed_mean": baseline_value,
                                "reduced_seed_mean": reduced_value,
                                "paired_difference_reduced_minus_baseline": difference,
                                "relative_change_percent": relative,
                                "mean_paired_difference": statistics.fmean(differences),
                                "std_paired_difference": statistics.stdev(differences),
                                "ci95_low_paired_difference": ci_low,
                                "ci95_high_paired_difference": ci_high,
                                "mean_relative_change_percent": statistics.fmean(relatives),
                                "ci95_low_relative_change_percent": relative_ci_low,
                                "ci95_high_relative_change_percent": relative_ci_high,
                                "scientific_sample_size": 5,
                                "sign_convention": "negative_relative_change_is_reduction_positive_is_increase",
                                "comparison_scope": "paired_same_seed_means_after_within_seed_aggregation",
                            }
                        )
    if len(output) != 400:
        raise V07EError("Paired comparisons must contain 80 metrics by five seeds.")
    return output


def build_efficiency_summary(paired_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse seed-pair details into transparent computational effect estimates."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        groups[row["comparison_id"]].append(row)
    output: list[dict[str, Any]] = []
    for comparison_id in sorted(groups):
        group = groups[comparison_id]
        if len(group) != 5 or {int(row["seed"]) for row in group} != set(SEEDS):
            raise V07EError("Efficiency summaries require exactly five paired seeds.")
        first = group[0]
        metric = first["metric"]
        difference_low = float(first["ci95_low_paired_difference"])
        difference_high = float(first["ci95_high_paired_difference"])
        maximize = metric in MAXIMIZE_METRICS
        if maximize:
            interpretation = (
                "clear_gain" if difference_low > 0 else "clear_loss" if difference_high < 0 else "uncertain"
            )
            beneficial_change = float(first["mean_relative_change_percent"])
        else:
            interpretation = (
                "clear_reduction" if difference_high < 0 else "clear_increase" if difference_low > 0 else "uncertain"
            )
            beneficial_change = -float(first["mean_relative_change_percent"])
        output.append(
            {
                "stage": STAGE,
                "comparison_id": comparison_id,
                "classifier": first["classifier"],
                "configuration_id": first["configuration_id"],
                "reference_configuration_id": BASELINE,
                "phase": first["phase"],
                "metric": metric,
                "unit": first["unit"],
                "baseline_mean_across_seeds": statistics.fmean(float(row["baseline_seed_mean"]) for row in group),
                "reduced_mean_across_seeds": statistics.fmean(float(row["reduced_seed_mean"]) for row in group),
                "mean_paired_difference": first["mean_paired_difference"],
                "std_paired_difference": first["std_paired_difference"],
                "ci95_low_paired_difference": difference_low,
                "ci95_high_paired_difference": difference_high,
                "mean_relative_change_percent": first["mean_relative_change_percent"],
                "ci95_low_relative_change_percent": first["ci95_low_relative_change_percent"],
                "ci95_high_relative_change_percent": first["ci95_high_relative_change_percent"],
                "beneficial_change_percent": beneficial_change,
                "objective_direction": "maximize" if maximize else "minimize",
                "interpretation": interpretation,
                "scientific_sample_size": 5,
                "statistical_significance_claimed": False,
            }
        )
    if len(output) != 80:
        raise V07EError("Efficiency summary must contain 80 paired metric estimates.")
    return output


def build_variability_analysis(seed_summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Describe within-seed repetition noise and between-seed variation."""

    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_summaries:
        groups[(row["classifier"], row["configuration_id"], row["phase"], row["metric"])].append(row)
    output: list[dict[str, Any]] = []
    for (classifier, configuration, phase, metric), group in sorted(groups.items()):
        if len(group) != 5:
            raise V07EError("Between-seed variability requires exactly five seed summaries.")
        seed_means = [float(row["mean"]) for row in group]
        within_cvs = [float(row["coefficient_of_variation_percent"]) for row in group]
        between_cv = coefficient_of_variation(seed_means)
        max_cv = max(max(within_cvs), between_cv)
        level = "high" if max_cv >= HIGH_VARIABILITY_CV_PERCENT else "moderate" if max_cv >= MODERATE_VARIABILITY_CV_PERCENT else "low"
        output.append(
            {
                "stage": STAGE,
                "classifier": classifier,
                "configuration_id": configuration,
                "phase": phase,
                "metric": metric,
                "unit": METRIC_UNITS[metric],
                "seed_count": 5,
                "repetitions_per_seed": 10,
                "mean_within_seed_cv_percent": statistics.fmean(within_cvs),
                "median_within_seed_cv_percent": statistics.median(within_cvs),
                "max_within_seed_cv_percent": max(within_cvs),
                "between_seed_mean": statistics.fmean(seed_means),
                "between_seed_std": statistics.stdev(seed_means),
                "between_seed_cv_percent": between_cv,
                "variability_level": level,
                "high_timing_variability": metric in TIMING_VARIABILITY_METRICS and level == "high",
                "high_variability_threshold_percent": HIGH_VARIABILITY_CV_PERCENT,
                "outliers_removed": False,
            }
        )
    if len(output) != 84:
        raise V07EError("Variability analysis must contain 84 classifier/configuration/phase metrics.")
    return output


def load_frozen_performance(
    final_test_dir: Path | str = FINAL_TEST_DIR,
    robustness_dir: Path | str = ROBUSTNESS_DIR,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], bool], dict[str, str]]:
    """Import only frozen V0.6 predictive outcomes, never historical timing."""

    final_root = Path(final_test_dir)
    robust_root = Path(robustness_dir)
    hashes = {}
    hashes.update(_verify_hash_manifest(final_root, final_root / "final_test_artifact_hashes.json"))
    hashes.update(_verify_hash_manifest(robust_root, robust_root / "robustness_artifact_hashes.json"))
    final_rows = _read_json(final_root / "final_test_runs.json")
    robust_rows = _read_json(robust_root / "robustness_runs.json")
    dt_rows = [row for row in final_rows if row.get("configuration_id") in CONFIGURATIONS]
    lr_rows = [
        row
        for row in robust_rows
        if row.get("scenario_id") == PRIMARY_SCENARIO
        and row.get("classifier") == "logistic_regression"
        and row.get("configuration_id") in CONFIGURATIONS
    ]
    if len(dt_rows) != 15 or len(lr_rows) != 15:
        raise V07EError("Frozen performance import requires 15 V0.6-F DT and 15 V0.6-G LR rows.")
    output = []
    for classifier, source_stage, source_path, source_rows in (
        ("decision_tree", "V0.6-F", final_root / "final_test_runs.json", dt_rows),
        ("logistic_regression", "V0.6-G", robust_root / "robustness_runs.json", lr_rows),
    ):
        for row in source_rows:
            if row.get("status") != "COMPLETED" or row.get("seed") not in SEEDS:
                raise V07EError("Frozen predictive row is incomplete or outside seeds 42-46.")
            output.append(
                {
                    "classifier": classifier,
                    "configuration_id": row["configuration_id"],
                    "seed": int(row["seed"]),
                    **{metric: float(row[metric]) for metric in PREDICTIVE_METRICS},
                    "predictive_provenance": "IMPORTED_HISTORICAL",
                    "measurement_domain": "V0.6_predictive_performance",
                    "source_stage": source_stage,
                    "source_artifact": str(source_path.relative_to(PROJECT_ROOT)),
                    "source_artifact_sha256": v07d.sha256_file(source_path),
                    "historical_timing_imported": False,
                }
            )
    keys = {(row["classifier"], row["configuration_id"], row["seed"]) for row in output}
    expected = {(classifier, configuration, seed) for classifier in CLASSIFIERS for configuration in CONFIGURATIONS for seed in SEEDS}
    if keys != expected or len(output) != 30:
        raise V07EError("Frozen predictive evidence must cover exactly 30 classifier/configuration/seed rows.")

    final_preservation = _read_json(final_root / "final_test_preservation.json")
    robust_preservation = _read_json(robust_root / "robustness_preservation.json")
    preservation: dict[tuple[str, str], bool] = {}
    for configuration in CONFIGURATIONS:
        dt_checks = [
            row
            for row in final_preservation
            if row.get("configuration_id") == configuration and row.get("metric") in ("average_precision", "f1", "recall")
        ]
        preservation[("decision_tree", configuration)] = len(dt_checks) == 3 and all(
            row.get("descriptively_preserved") is True for row in dt_checks
        )
        if configuration == BASELINE:
            preservation[("logistic_regression", configuration)] = True
        else:
            lr_checks = [
                row
                for row in robust_preservation
                if row.get("scenario_id") == PRIMARY_SCENARIO
                and row.get("classifier") == "logistic_regression"
                and row.get("configuration_id") == configuration
                and row.get("metric") in ("average_precision", "f1", "recall")
            ]
            preservation[("logistic_regression", configuration)] = len(lr_checks) == 3 and all(
                row.get("descriptively_preserved") is True for row in lr_checks
            )
    return sorted(output, key=lambda row: (row["classifier"], row["configuration_id"], row["seed"])), preservation, hashes


def build_performance_efficiency(
    performance_rows: Sequence[Mapping[str, Any]],
    seed_summaries: Sequence[Mapping[str, Any]],
    preservation: Mapping[tuple[str, str], bool],
) -> list[dict[str, Any]]:
    """Place historical performance and V0.7-D computation side by side without pooling domains."""

    computational = _summary_groups(seed_summaries)
    output = []
    for classifier in CLASSIFIERS:
        for configuration in CONFIGURATIONS:
            performance = [
                row for row in performance_rows if row["classifier"] == classifier and row["configuration_id"] == configuration
            ]
            if len(performance) != 5:
                raise V07EError("Performance-efficiency rows require five imported predictive seeds.")
            row: dict[str, Any] = {
                "stage": STAGE,
                "classifier": classifier,
                "configuration_id": configuration,
                "feature_count": int(computational[(classifier, configuration, 42, "inference")]["feature_count"]),
                "feature_reduction_percent": 100.0 * (43 - computational[(classifier, configuration, 42, "inference")]["feature_count"]) / 43,
                "predictive_performance_preserved": bool(preservation[(classifier, configuration)]),
                "predictive_provenance": "IMPORTED_HISTORICAL",
                "predictive_measurement_domain": "V0.6_predictive_performance",
                "computational_provenance": "DIRECT_COMPUTATIONAL",
                "computational_measurement_domain": "V0.7-D_same_environment",
                "historical_timing_pooled": False,
                "scientific_sample_size": 5,
            }
            for metric in PREDICTIVE_METRICS:
                values = [float(item[metric]) for item in performance]
                low, high = t95_interval(values)
                row[f"{metric}_mean"] = statistics.fmean(values)
                row[f"{metric}_std"] = statistics.stdev(values)
                row[f"{metric}_ci95_low"] = low
                row[f"{metric}_ci95_high"] = high
            for phase, metric, output_name in (
                ("training", "wall_time_sec", "training_wall_time_sec"),
                ("training", "process_cpu_time_sec", "training_cpu_time_sec"),
                ("training", "absolute_peak_rss_mib", "training_peak_rss_mib"),
                ("inference", "per_operation_latency_sec", "inference_per_operation_latency_sec"),
                ("inference", "process_cpu_time_sec", "inference_cpu_time_sec"),
                ("inference", "absolute_peak_rss_mib", "inference_peak_rss_mib"),
                ("inference", "selected_input_bytes", "inference_selected_input_bytes"),
                ("inference", "serialized_model_bytes", "serialized_model_bytes"),
            ):
                values = [computational[(classifier, configuration, seed, phase)][metric] for seed in SEEDS]
                low, high = t95_interval(values)
                row[output_name] = statistics.fmean(values)
                row[f"{output_name}_ci95_low"] = low
                row[f"{output_name}_ci95_high"] = high
            output.append(row)
    return output


def dominates(candidate: Mapping[str, float], other: Mapping[str, float], objectives: Sequence[tuple[str, str]]) -> bool:
    """Return exact unweighted Pareto dominance for explicit objective directions."""

    no_worse = True
    strictly_better = False
    for metric, direction in objectives:
        left, right = float(candidate[metric]), float(other[metric])
        if direction == "maximize":
            no_worse &= left >= right
            strictly_better |= left > right
        elif direction == "minimize":
            no_worse &= left <= right
            strictly_better |= left < right
        else:
            raise V07EError(f"Unknown Pareto direction: {direction}")
    return no_worse and strictly_better


def build_pareto_analysis(performance_efficiency: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build separate descriptive, unweighted Pareto fronts for DT and LR."""

    output = []
    for classifier in CLASSIFIERS:
        points = [row for row in performance_efficiency if row["classifier"] == classifier]
        for point in points:
            objective_values = {
                "average_precision": point["average_precision_mean"],
                "f1": point["f1_mean"],
                "recall": point["recall_mean"],
                "feature_count": point["feature_count"],
                "training_wall_time_sec": point["training_wall_time_sec"],
                "inference_per_operation_latency_sec": point["inference_per_operation_latency_sec"],
                "inference_selected_input_bytes": point["inference_selected_input_bytes"],
                "serialized_model_bytes": point["serialized_model_bytes"],
            }
            dominators = []
            for other in points:
                if other is point:
                    continue
                other_values = {
                    "average_precision": other["average_precision_mean"],
                    "f1": other["f1_mean"],
                    "recall": other["recall_mean"],
                    "feature_count": other["feature_count"],
                    "training_wall_time_sec": other["training_wall_time_sec"],
                    "inference_per_operation_latency_sec": other["inference_per_operation_latency_sec"],
                    "inference_selected_input_bytes": other["inference_selected_input_bytes"],
                    "serialized_model_bytes": other["serialized_model_bytes"],
                }
                if dominates(other_values, objective_values, PARETO_OBJECTIVES):
                    dominators.append(other["configuration_id"])
            output.append(
                {
                    "stage": STAGE,
                    "classifier": classifier,
                    "configuration_id": point["configuration_id"],
                    **objective_values,
                    "is_pareto_optimal": not dominators,
                    "dominated_by": "|".join(sorted(dominators)),
                    "objective_directions": ";".join(f"{name}:{direction}" for name, direction in PARETO_OBJECTIVES),
                    "objective_weights_used": False,
                    "composite_score_used": False,
                    "analysis_scope": "descriptive_frozen_configuration_pareto_not_optimizer",
                }
            )
    return output


def _label(configuration: str) -> str:
    return {CONFIGURATIONS[0]: "K43", CONFIGURATIONS[1]: "K42", CONFIGURATIONS[2]: "K11"}[configuration]


def create_figures(
    output_dir: Path,
    performance_efficiency: Sequence[Mapping[str, Any]],
    efficiency_summary: Sequence[Mapping[str, Any]],
    variability: Sequence[Mapping[str, Any]],
    pareto: Sequence[Mapping[str, Any]],
) -> list[Path]:
    """Create eight artifact-only figures with explicit units and no weighted score."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"K43": "#264653", "K42": "#e9c46a", "K11": "#e76f51"}

    def save(fig: Any, name: str) -> Path:
        path = output_dir / name
        temporary = path.with_name(f".{path.name}.tmp")
        fig.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        temporary.replace(path)
        return path

    figures = []
    ordered = [next(row for row in performance_efficiency if row["classifier"] == classifier and row["configuration_id"] == configuration) for classifier in CLASSIFIERS for configuration in CONFIGURATIONS]
    for field, y_label, title, name in (
        ("inference_per_operation_latency_sec", "Inference latency (ms/operation)", "Feature Count vs Inference Latency", "01_feature_count_vs_inference_latency.png"),
        ("training_wall_time_sec", "Training wall time (seconds)", "Feature Count vs Training Time", "02_feature_count_vs_training_time.png"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
        for axis, classifier in zip(axes, CLASSIFIERS):
            subset = [row for row in ordered if row["classifier"] == classifier]
            scale = 1000.0 if "inference" in field else 1.0
            for row in subset:
                axis.errorbar(
                    row["feature_count"], row[field] * scale,
                    yerr=[[max(0.0, (row[field] - row[f"{field}_ci95_low"]) * scale)], [max(0.0, (row[f"{field}_ci95_high"] - row[field]) * scale)]],
                    fmt="o", color=colors[_label(row["configuration_id"])], capsize=4, label=_label(row["configuration_id"]),
                )
            axis.set_title(classifier.replace("_", " ").title())
            axis.set_xlabel("Feature count")
            axis.set_ylabel(y_label)
            axis.grid(alpha=0.25)
            axis.legend()
        fig.suptitle(title)
        figures.append(save(fig, name))

    for x_field, x_label, title, name in (
        ("inference_selected_input_bytes", "Inference input memory (MiB)", "Input Memory vs Imported Average Precision", "03_input_memory_vs_predictive_ap.png"),
        ("serialized_model_bytes", "Serialized model size (KiB)", "Model Size vs Imported Average Precision", "04_model_size_vs_ap.png"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, classifier in zip(axes, CLASSIFIERS):
            subset = [row for row in ordered if row["classifier"] == classifier]
            divisor = 1024.0**2 if "input" in x_field else 1024.0
            for row in subset:
                axis.scatter(row[x_field] / divisor, row["average_precision_mean"], s=80, color=colors[_label(row["configuration_id"])], label=_label(row["configuration_id"]), edgecolor="white")
            axis.set_title(classifier.replace("_", " ").title())
            axis.set_xlabel(x_label)
            axis.set_ylabel("Average Precision (imported V0.6)")
            axis.grid(alpha=0.25)
            axis.legend()
        fig.suptitle(title)
        figures.append(save(fig, name))

    selected_metrics = ("feature_count", "selected_input_bytes", "wall_time_sec", "per_operation_latency_sec", "serialized_model_bytes")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for axis, classifier in zip(axes, CLASSIFIERS):
        positions = np.arange(len(selected_metrics))
        width = 0.36
        for offset, configuration in enumerate(REDUCED_CONFIGURATIONS):
            values = []
            for metric in selected_metrics:
                phase = "training" if metric == "wall_time_sec" else "inference"
                match = next(row for row in efficiency_summary if row["classifier"] == classifier and row["configuration_id"] == configuration and row["phase"] == phase and row["metric"] == metric)
                values.append(match["beneficial_change_percent"])
            axis.bar(positions + (offset - 0.5) * width, values, width, label=_label(configuration), color=colors[_label(configuration)])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("Beneficial change (%)")
        axis.set_title(classifier.replace("_", " ").title())
        axis.legend()
    axes[-1].set_xticks(np.arange(len(selected_metrics)), ["Features", "Input memory", "Training time", "Inference latency", "Model size"])
    fig.suptitle("Relative Computational Savings (positive is beneficial)")
    figures.append(save(fig, "05_relative_computational_savings.png"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(3)
    for axis, field, label in zip(axes, ("training_wall_time_sec", "inference_per_operation_latency_sec"), ("Training wall time (seconds)", "Inference latency (ms/operation)")):
        for index, classifier in enumerate(CLASSIFIERS):
            values = [next(row for row in ordered if row["classifier"] == classifier and row["configuration_id"] == config)[field] for config in CONFIGURATIONS]
            if "inference" in field:
                values = [value * 1000 for value in values]
            axis.bar(x + (index - 0.5) * 0.35, values, 0.35, label=classifier.replace("_", " ").title())
        axis.set_xticks(x, [_label(config) for config in CONFIGURATIONS])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle("Decision Tree vs Logistic Regression Computational Comparison")
    figures.append(save(fig, "06_dt_vs_lr_computational_comparison.png"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, classifier in zip(axes, CLASSIFIERS):
        for row in [item for item in pareto if item["classifier"] == classifier]:
            axis.scatter(row["inference_per_operation_latency_sec"] * 1000, row["average_precision"], s=110 if row["is_pareto_optimal"] else 60, marker="o" if row["is_pareto_optimal"] else "x", color=colors[_label(row["configuration_id"])])
            axis.annotate(_label(row["configuration_id"]), (row["inference_per_operation_latency_sec"] * 1000, row["average_precision"]), xytext=(5, 5), textcoords="offset points")
        axis.set_title(classifier.replace("_", " ").title())
        axis.set_xlabel("Inference latency (ms/operation)")
        axis.set_ylabel("Average Precision (imported V0.6)")
        axis.grid(alpha=0.25)
    fig.suptitle("Descriptive Performance-Efficiency Pareto View")
    figures.append(save(fig, "07_pareto_performance_efficiency.png"))

    timing_rows = [row for row in variability if row["metric"] == "per_operation_latency_sec"]
    matrix = np.array([[next(row for row in timing_rows if row["classifier"] == classifier and row["configuration_id"] == config)["max_within_seed_cv_percent"] for config in CONFIGURATIONS] for classifier in CLASSIFIERS])
    fig, axis = plt.subplots(figsize=(7, 3.5))
    image = axis.imshow(matrix, cmap="magma", aspect="auto")
    axis.set_xticks(np.arange(3), [_label(config) for config in CONFIGURATIONS])
    axis.set_yticks(np.arange(2), [name.replace("_", " ").title() for name in CLASSIFIERS])
    axis.set_xlabel("Configuration")
    axis.set_title("Maximum Within-Seed Inference-Latency CV (%)")
    for i in range(2):
        for j in range(3):
            axis.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=axis, label="Coefficient of variation (%)")
    figures.append(save(fig, "08_measurement_variability_cv.png"))
    return figures


def _write_table(staging: Path, stem: str, rows: Sequence[Mapping[str, Any]]) -> tuple[Path, Path]:
    json_path = staging / f"{stem}.json"
    csv_path = staging / f"{stem}.csv"
    v07d.atomic_write_json(json_path, list(rows))
    v07d.atomic_write_csv(csv_path, rows, fieldnames=tuple(rows[0]))
    return json_path, csv_path


def _snapshot(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): v07d.sha256_file(path) for path in paths}


def run_v07e_analysis(
    *,
    results_dir: Path | str = RESULTS_DIR,
    final_test_dir: Path | str = FINAL_TEST_DIR,
    robustness_dir: Path | str = ROBUSTNESS_DIR,
) -> dict[str, Any]:
    """Run analysis only over immutable V0.6 and V0.7-D result artifacts."""

    root = Path(results_dir)
    if (root / "v07e_governance_audit.json").exists():
        raise V07EError("Completed V0.7-E artifacts already exist; refusing to overwrite them.")
    staging = root / STAGING_NAME
    staging.mkdir(parents=True, exist_ok=False)
    try:
        observations, manifest, v07d_hashes = load_v07d_observations(root)
        performance, preservation, v06_hashes = load_frozen_performance(final_test_dir, robustness_dir)
        immutable_before = {"v06": v06_hashes, "v07d": v07d_hashes}
        seed_summaries = build_seed_summaries(observations)
        paired = build_paired_comparisons(seed_summaries)
        efficiency = build_efficiency_summary(paired)
        variability = build_variability_analysis(seed_summaries)
        performance_efficiency = build_performance_efficiency(performance, seed_summaries, preservation)
        pareto = build_pareto_analysis(performance_efficiency)

        published: list[Path] = []
        for stem, rows in (
            ("v07e_seed_summaries", seed_summaries),
            ("v07e_paired_comparisons", paired),
            ("v07e_efficiency_summary", efficiency),
            ("v07e_performance_efficiency", performance_efficiency),
            ("v07e_pareto_analysis", pareto),
            ("v07e_variability_analysis", variability),
        ):
            published.extend(_write_table(staging, stem, rows))
        figures = create_figures(staging / "figures", performance_efficiency, efficiency, variability, pareto)
        published.extend(figures)

        immutable_after = {
            "v06": _snapshot(Path(path) for path in immutable_before["v06"]),
            "v07d": _snapshot(Path(path) for path in immutable_before["v07d"]),
        }
        if immutable_after != immutable_before:
            raise V07EError("Immutable V0.6 or V0.7-D evidence changed during analysis.")
        governance = {
            "schema_version": SCHEMA_VERSION,
            "stage": STAGE,
            "status": "PASS",
            "source_v07d_protocol_sha256": manifest["protocol_sha256"],
            "direct_energy_decision": "DIRECT_ENERGY_UNAVAILABLE",
            "scientific_hierarchy": "10 measurement repetitions aggregated within seed; inference across 5 seeds",
            "paired_difference_definition": "reduced_seed_mean_minus_baseline_seed_mean",
            "relative_change_definition": "100*(reduced-baseline)/baseline; negative is reduction",
            "confidence_interval": "two-sided 95% Student-t interval over five paired seeds; df=4",
            "variability_threshold": "high timing variability when max within-seed or between-seed CV >= 20%",
            "checks": [
                {"name": "exact_600_v07d_observations", "passed": len(observations) == 600},
                {"name": "zero_failures_and_no_observation_removal", "passed": True},
                {"name": "ten_repetitions_aggregated_within_seed", "passed": len(seed_summaries) == 420},
                {"name": "scientific_sample_size_is_five", "passed": True},
                {"name": "paired_by_same_classifier_phase_metric_seed", "passed": True},
                {"name": "v06_predictive_metrics_imported_only", "passed": len(performance) == 30},
                {"name": "v06_and_v07d_measurement_domains_separate", "passed": True},
                {"name": "no_model_training_or_workload_regeneration", "passed": True},
                {"name": "no_feature_reselection_or_tuning", "passed": True},
                {"name": "pareto_is_descriptive_unweighted", "passed": True},
                {"name": "no_direct_energy_or_carbon_outcomes", "passed": True},
                {"name": "no_statistical_significance_claim", "passed": True},
                {"name": "v06_and_v07d_hashes_unchanged", "passed": True},
            ],
            "allowed_claims": [
                "observed computational or resource-efficiency difference within the measured V0.7-D environment",
                "predictive behavior descriptively preserved according to immutable V0.6 evidence",
                "descriptive unweighted Pareto membership among the three frozen configurations",
            ],
            "prohibited_claims": [
                "electrical-energy savings",
                "carbon savings",
                "timing as measured energy",
                "statistical significance",
                "final multi-objective optimizer superiority",
                "generalization beyond the measured split, workload, classifiers, and environment",
            ],
            "immutable_hashes_after": immutable_after,
        }
        governance_path = staging / "v07e_governance_audit.json"
        v07d.atomic_write_json(governance_path, governance)
        published.append(governance_path)

        artifact_hashes = {
            str(path.relative_to(staging)): v07d.sha256_file(path)
            for path in sorted(published, key=lambda item: str(item.relative_to(staging)))
        }
        hash_path = staging / "v07e_artifact_hashes.json"
        v07d.atomic_write_json(hash_path, artifact_hashes)
        for path in published:
            destination = root / path.relative_to(staging)
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)
        hash_path.replace(root / hash_path.name)
        for directory in (staging / "figures", staging):
            directory.rmdir()
        return governance
    except BaseException:
        raise


def main() -> int:
    governance = run_v07e_analysis()
    print(json.dumps({"stage": STAGE, "status": governance["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
