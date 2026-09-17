"""Deterministic V0.8-E4 scientific analysis over frozen E3 evidence only."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

from src.config import PROJECT_ROOT
import src.pipeline_v08d as v08d
import src.pipeline_v08e as v08e


STAGE = "V0.8-E4"
SCHEMA_VERSION = "v0.8-e4-resource-analysis-1"
E3_DIR = PROJECT_ROOT / "results" / "bpso" / "v08e_e3_campaign"
OUTPUT_DIR = PROJECT_ROOT / "results" / "bpso" / "v08e_e4_analysis"

CONFIGURATIONS = v08e.CONFIGURATION_IDS
CLASSIFIERS = v08e.CLASSIFIERS
SEEDS = v08e.SEEDS
PHASES = v08e.PHASES
REPETITIONS = v08e.REPETITIONS
BASELINE = "K43"
CANDIDATE = "BPSO-K10"
REFERENCES = ("K43", "K42", "K11")
T_CRITICAL_95_DF4 = 2.776445
SLEEP_LIMITATION = (
    "Host Modern Standby occurred during portions of the campaign, and "
    "observation-level timestamps were unavailable; therefore timing-based "
    "resource results are interpreted as indicative rather than definitive."
)

METRICS: dict[str, dict[str, Any]] = {
    "wall_time_sec": {
        "unit": "seconds",
        "direction": "lower_is_better",
        "phases": PHASES,
        "timing_dependent": True,
    },
    "process_cpu_time_sec": {
        "unit": "cpu-seconds",
        "direction": "lower_is_better",
        "phases": PHASES,
        "timing_dependent": True,
    },
    "inference_latency_sec": {
        "unit": "seconds/prediction-operation",
        "direction": "lower_is_better",
        "phases": ("inference",),
        "timing_dependent": True,
    },
    "per_record_latency_sec": {
        "unit": "seconds/record",
        "direction": "lower_is_better",
        "phases": ("inference",),
        "timing_dependent": True,
    },
    "throughput_records_sec": {
        "unit": "records/second",
        "direction": "higher_is_better",
        "phases": ("inference",),
        "timing_dependent": True,
    },
    "absolute_peak_rss_mib": {
        "unit": "MiB",
        "direction": "lower_is_better",
        "phases": PHASES,
        "timing_dependent": False,
    },
    "incremental_peak_rss_mib": {
        "unit": "MiB",
        "direction": "lower_is_better",
        "phases": PHASES,
        "timing_dependent": False,
    },
    "feature_count": {
        "unit": "features",
        "direction": "lower_is_better",
        "phases": PHASES,
        "timing_dependent": False,
    },
    "input_dataframe_memory_bytes": {
        "unit": "bytes",
        "direction": "lower_is_better",
        "phases": PHASES,
        "timing_dependent": False,
    },
    "numpy_dense_nbytes": {
        "unit": "bytes",
        "direction": "lower_is_better",
        "phases": PHASES,
        "timing_dependent": False,
    },
    "serialized_model_bytes": {
        "unit": "bytes",
        "direction": "lower_is_better",
        "phases": PHASES,
        "timing_dependent": False,
    },
}
TIMING_METRICS = frozenset(
    metric for metric, metadata in METRICS.items() if metadata["timing_dependent"]
)
STRUCTURAL_METRICS = (
    "feature_count",
    "input_dataframe_memory_bytes",
    "numpy_dense_nbytes",
    "serialized_model_bytes",
)

SLEEP_LIMITED_CELLS = (
    "training/decision_tree/K11/s42",
    "training/logistic_regression/K11/s42",
    "training/logistic_regression/K43/s42",
    "training/logistic_regression/K11/s43",
    "training/logistic_regression/K42/s43",
    "training/decision_tree/BPSO-K10/s44",
    "training/decision_tree/K43/s45",
    "training/logistic_regression/K42/s45",
    "training/logistic_regression/K11/s45",
    "training/logistic_regression/BPSO-K10/s45",
    "training/logistic_regression/K43/s46",
    "inference/decision_tree/K42/s42",
    "inference/decision_tree/K11/s42",
    "inference/decision_tree/BPSO-K10/s42",
    "inference/decision_tree/K43/s42",
    "inference/logistic_regression/K11/s42",
    "inference/logistic_regression/BPSO-K10/s42",
    "inference/logistic_regression/K42/s42",
    "inference/decision_tree/K11/s43",
    "inference/decision_tree/BPSO-K10/s43",
    "inference/decision_tree/K43/s43",
    "inference/logistic_regression/K43/s43",
    "inference/logistic_regression/K42/s43",
    "inference/logistic_regression/K11/s43",
    "inference/decision_tree/BPSO-K10/s44",
    "inference/decision_tree/K43/s44",
    "inference/decision_tree/K11/s44",
    "inference/logistic_regression/K43/s44",
    "inference/logistic_regression/K42/s44",
    "inference/logistic_regression/K11/s44",
    "inference/decision_tree/K43/s45",
    "inference/decision_tree/K42/s45",
    "inference/logistic_regression/BPSO-K10/s45",
    "inference/logistic_regression/K43/s45",
    "inference/decision_tree/K42/s46",
    "inference/decision_tree/K11/s46",
    "inference/decision_tree/BPSO-K10/s46",
    "inference/decision_tree/K43/s46",
    "inference/logistic_regression/K11/s46",
    "inference/logistic_regression/BPSO-K10/s46",
    "inference/logistic_regression/K43/s46",
    "inference/logistic_regression/K42/s46",
)
TIMING_OUTLIER_IDS = (
    "v08e__training__decision_tree__K11__s42__r02",
    "v08e__training__logistic_regression__BPSO-K10__s42__r03",
    "v08e__training__decision_tree__K43__s43__r01",
    "v08e__training__decision_tree__K11__s44__r07",
    "v08e__training__decision_tree__BPSO-K10__s44__r09",
    "v08e__training__logistic_regression__K43__s44__r10",
    "v08e__training__logistic_regression__K42__s44__r06",
    "v08e__training__logistic_regression__K11__s44__r01",
    "v08e__training__decision_tree__K43__s45__r09",
    "v08e__training__logistic_regression__K42__s45__r08",
    "v08e__training__decision_tree__K11__s46__r02",
    "v08e__training__decision_tree__K11__s46__r03",
    "v08e__training__logistic_regression__BPSO-K10__s46__r02",
    "v08e__inference__logistic_regression__BPSO-K10__s42__r05",
    "v08e__inference__logistic_regression__BPSO-K10__s42__r10",
    "v08e__inference__decision_tree__K43__s44__r02",
)


class V08E4Error(RuntimeError):
    """Raised when governed E4 inputs or outputs fail validation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V08E4Error(f"Cannot read JSON artifact: {path}") from exc


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise V08E4Error(f"Cannot write empty table: {path.name}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_table(directory: Path, stem: str, rows: Sequence[Mapping[str, Any]]) -> list[Path]:
    json_path = directory / f"{stem}.json"
    csv_path = directory / f"{stem}.csv"
    _atomic_write_json(json_path, list(rows))
    _atomic_write_csv(csv_path, rows)
    return [json_path, csv_path]


def _snapshot(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): v08e.sha256_file(path) for path in paths}


def _canonical_frozen_snapshot(preflight: Mapping[str, Any]) -> dict[str, str]:
    expected = preflight.get("immutable_hashes", {})
    actual = {path: v08e.canonical_lf_sha256(path) for path in expected}
    if actual != expected:
        raise V08E4Error("Frozen V0.6/V0.7/V0.8-C/V0.8-D input hashes do not match.")
    return actual


def _configuration_identities(model_manifest: Mapping[str, Any]) -> dict[str, Any]:
    by_configuration: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in model_manifest.get("artifacts", []):
        by_configuration[str(row["configuration_id"])].append(row)
    expected_counts = {"K43": 43, "K42": 42, "K11": 11, "BPSO-K10": 10}
    identities: dict[str, Any] = {}
    for configuration in CONFIGURATIONS:
        rows = by_configuration.get(configuration, [])
        if len(rows) != 10 or {int(row["seed"]) for row in rows} != set(SEEDS):
            raise V08E4Error(f"Model identity coverage failed for {configuration}.")
        feature_counts = {int(row["feature_count"]) for row in rows}
        if feature_counts != {expected_counts[configuration]}:
            raise V08E4Error(f"Feature count identity failed for {configuration}.")
        seed_hashes = {
            str(seed): sorted(
                {str(row["feature_manifest_sha256"]) for row in rows if int(row["seed"]) == seed}
            )[0]
            for seed in SEEDS
        }
        identities[configuration] = {
            "feature_count": expected_counts[configuration],
            "source_configuration_ids": sorted({str(row["source_configuration_id"]) for row in rows}),
            "feature_hashes_by_seed": seed_hashes,
            "universal_identity": len(set(seed_hashes.values())) == 1,
        }
    if identities["BPSO-K10"]["feature_hashes_by_seed"] != {
        str(seed): v08e.EXPECTED_K10["selected_features_sha256"] for seed in SEEDS
    }:
        raise V08E4Error("BPSO-K10 universal identity changed.")
    if identities["K11"]["universal_identity"]:
        raise V08E4Error("MI-K11 must retain seed-specific identities.")
    return identities


def load_e4_inputs(e3_dir: Path | str = E3_DIR) -> dict[str, Any]:
    """Load and fail-closed validate frozen E3, V0.8-C, and V0.8-D inputs."""

    root = Path(e3_dir)
    input_paths = tuple(
        root / name
        for name in (
            "v08e_computational_observations.json",
            "v08e_computational_observations.csv",
            "v08e_measurement_failures.json",
            "v08e_run_manifest.json",
            "v08e_preflight.json",
            "v08e_environment_metadata.json",
            "v08e_model_artifact_manifest.json",
        )
    )
    input_hashes = _snapshot(input_paths)
    rows = _read_json(root / "v08e_computational_observations.json")
    failures = _read_json(root / "v08e_measurement_failures.json")
    manifest = _read_json(root / "v08e_run_manifest.json")
    preflight = _read_json(root / "v08e_preflight.json")
    environment = _read_json(root / "v08e_environment_metadata.json")
    model_manifest = _read_json(root / "v08e_model_artifact_manifest.json")
    if not isinstance(rows, list) or len(rows) != 800 or failures != []:
        raise V08E4Error("E4 requires exactly 800 E3 rows and zero failures.")
    v08e.validate_observation_rows(rows)
    if Counter(str(row["status"]) for row in rows) != {"SUCCESS": 800}:
        raise V08E4Error("All E3 observations must be successful.")
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
            str(row["classifier"]),
            str(row["configuration_id"]),
            int(row["seed"]),
            str(row["phase"]),
            int(row["outer_repetition"]),
        )
        for row in rows
    }
    if actual != expected or len(actual) != len(rows):
        raise V08E4Error("E3 matrix coverage or observation uniqueness failed.")
    if Counter(str(row["phase"]) for row in rows) != {"training": 400, "inference": 400}:
        raise V08E4Error("E3 phase counts are invalid.")
    environment_ids = {str(row["environment_id"]) for row in rows}
    if environment_ids != {str(manifest["environment_id"])}:
        raise V08E4Error("E3 environment identity is inconsistent.")
    if manifest.get("protocol_sha256") != preflight.get("protocol_sha256"):
        raise V08E4Error("E3 protocol SHA is inconsistent.")
    checkpoints = sorted((root / ".v08e_checkpoints").glob("*.json"))
    if len(checkpoints) != 80:
        raise V08E4Error("E3 must contain exactly 80 active checkpoints.")
    checkpoint_rows: list[Mapping[str, Any]] = []
    protocol_hashes: set[str] = set()
    for path in checkpoints:
        payload = _read_json(path)
        cell_rows = payload.get("rows", [])
        if (
            len(cell_rows) != 10
            or {int(row["outer_repetition"]) for row in cell_rows} != set(REPETITIONS)
            or any(row.get("status") != "SUCCESS" for row in cell_rows)
        ):
            raise V08E4Error(f"Checkpoint is not scientifically complete: {path.name}")
        protocol_hashes.add(str(payload.get("protocol_sha256")))
        checkpoint_rows.extend(cell_rows)
    if protocol_hashes != {str(manifest["protocol_sha256"])}:
        raise V08E4Error("Checkpoint protocol SHA coverage is invalid.")
    if {row["observation_id"] for row in checkpoint_rows} != {
        row["observation_id"] for row in rows
    }:
        raise V08E4Error("Raw E3 rows differ from active checkpoint evidence.")
    with (root / "v08e_computational_observations.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        csv_ids = [row["observation_id"] for row in csv.DictReader(handle)]
    if csv_ids != [row["observation_id"] for row in rows]:
        raise V08E4Error("E3 JSON/CSV observation parity failed.")
    identities = _configuration_identities(model_manifest)
    frozen_hashes = _canonical_frozen_snapshot(preflight)
    v08d.verify_v08d_artifacts()
    predictive = v08e.import_v08d_predictive_evidence()
    overhead = v08e.import_v08c_optimizer_overhead()
    return {
        "rows": rows,
        "manifest": manifest,
        "preflight": preflight,
        "environment": environment,
        "identities": identities,
        "input_hashes": input_hashes,
        "frozen_hashes": frozen_hashes,
        "predictive": predictive,
        "overhead": overhead,
    }


def coefficient_of_variation(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        raise V08E4Error("CV requires at least two values.")
    mean = statistics.fmean(values)
    if math.isclose(mean, 0.0, abs_tol=1e-15):
        return 0.0 if all(math.isclose(value, 0.0, abs_tol=1e-15) for value in values) else None
    return 100.0 * statistics.stdev(values) / abs(mean)


def build_within_seed_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["classifier"]), str(row["configuration_id"]), int(row["seed"]), str(row["phase"]))].append(row)
    if len(groups) != 80:
        raise V08E4Error("Expected 80 within-seed cells.")
    output: list[dict[str, Any]] = []
    for (classifier, configuration, seed, phase), group in sorted(groups.items()):
        if len(group) != 10:
            raise V08E4Error("Each within-seed cell must contain ten repetitions.")
        for metric, metadata in METRICS.items():
            if phase not in metadata["phases"]:
                continue
            values = [float(row[metric]) for row in group]
            output.append(
                {
                    "stage": STAGE,
                    "classifier": classifier,
                    "configuration_id": configuration,
                    "seed": seed,
                    "phase": phase,
                    "metric": metric,
                    "unit": metadata["unit"],
                    "objective_direction": metadata["direction"],
                    "repetition_count": 10,
                    "mean": statistics.fmean(values),
                    "median": statistics.median(values),
                    "sample_sd": statistics.stdev(values),
                    "min": min(values),
                    "max": max(values),
                    "coefficient_of_variation_percent": coefficient_of_variation(values),
                    "timing_integrity_limited": bool(metadata["timing_dependent"]),
                    "outliers_removed": False,
                    "aggregation_scope": "ten_nested_measurement_repetitions_within_seed",
                    "summary_provenance": "DERIVED",
                }
            )
    if len(output) != 760:
        raise V08E4Error("Within-seed summary must contain 760 rows.")
    return output


def _summary_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int, str, str], Mapping[str, Any]]:
    return {
        (
            str(row["classifier"]),
            str(row["configuration_id"]),
            int(row["seed"]),
            str(row["phase"]),
            str(row["metric"]),
        ): row
        for row in rows
    }


def _paired_t95(values: Sequence[float]) -> tuple[float, float]:
    if len(values) != 5:
        raise V08E4Error("Paired confidence intervals require exactly five seeds.")
    mean = statistics.fmean(values)
    margin = T_CRITICAL_95_DF4 * statistics.stdev(values) / math.sqrt(5.0)
    return mean - margin, mean + margin


def build_paired_resource_comparisons(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    index = _summary_index(summaries)
    output: list[dict[str, Any]] = []
    for classifier in CLASSIFIERS:
        for reference in REFERENCES:
            for phase in PHASES:
                for metric, metadata in METRICS.items():
                    if phase not in metadata["phases"]:
                        continue
                    baseline_values = [float(index[(classifier, reference, seed, phase, metric)]["mean"]) for seed in SEEDS]
                    candidate_values = [float(index[(classifier, CANDIDATE, seed, phase, metric)]["mean"]) for seed in SEEDS]
                    differences = [candidate - baseline for candidate, baseline in zip(candidate_values, baseline_values)]
                    relative_changes = [100.0 * difference / baseline for difference, baseline in zip(differences, baseline_values)]
                    low, high = _paired_t95(differences)
                    direction = str(metadata["direction"])
                    beneficial = [difference < 0 if direction == "lower_is_better" else difference > 0 for difference in differences]
                    adverse = [difference > 0 if direction == "lower_is_better" else difference < 0 for difference in differences]
                    if all(beneficial):
                        consistency = "beneficial_all_five_seeds"
                    elif all(adverse):
                        consistency = "adverse_all_five_seeds"
                    else:
                        consistency = "mixed_or_tied_across_seeds"
                    if direction == "lower_is_better":
                        ci_interpretation = "indicative_reduction" if high < 0 else "indicative_increase" if low > 0 else "direction_uncertain_ci_includes_zero"
                    else:
                        ci_interpretation = "indicative_gain" if low > 0 else "indicative_loss" if high < 0 else "direction_uncertain_ci_includes_zero"
                    output.append(
                        {
                            "stage": STAGE,
                            "comparison_id": f"{classifier}__{CANDIDATE}__vs__{reference}__{phase}__{metric}",
                            "classifier": classifier,
                            "configuration_id": CANDIDATE,
                            "reference_configuration_id": reference,
                            "phase": phase,
                            "metric": metric,
                            "unit": metadata["unit"],
                            "objective_direction": direction,
                            "reference_mean_across_seeds": statistics.fmean(baseline_values),
                            "candidate_mean_across_seeds": statistics.fmean(candidate_values),
                            "paired_mean_difference_candidate_minus_reference": statistics.fmean(differences),
                            "paired_difference_sample_sd": statistics.stdev(differences),
                            "paired_difference_ci95_low": low,
                            "paired_difference_ci95_high": high,
                            "mean_percentage_change": statistics.fmean(relative_changes),
                            "beneficial_seed_count": sum(beneficial),
                            "adverse_seed_count": sum(adverse),
                            "direction_consistency": consistency,
                            "ci_includes_zero": low <= 0.0 <= high,
                            "ci_interpretation": ci_interpretation,
                            "scientific_n": 5,
                            "df": 4,
                            "t_critical_95": T_CRITICAL_95_DF4,
                            "timing_integrity_limited": bool(metadata["timing_dependent"]),
                            "measurement_integrity_statement": SLEEP_LIMITATION if metadata["timing_dependent"] else "not_timing_dependent",
                            "p_value_computed": False,
                            "statistical_significance_claimed": False,
                        }
                    )
    if len(output) != 114:
        raise V08E4Error("Paired analysis must contain 114 comparison rows.")
    return output


def build_timing_variability(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        if row["metric"] in TIMING_METRICS:
            groups[(str(row["classifier"]), str(row["configuration_id"]), str(row["phase"]), str(row["metric"]))].append(row)
    output: list[dict[str, Any]] = []
    for (classifier, configuration, phase, metric), group in sorted(groups.items()):
        if len(group) != 5:
            raise V08E4Error("Timing variability requires five seed cells.")
        cvs = [float(row["coefficient_of_variation_percent"]) for row in group]
        maximum = max(group, key=lambda row: float(row["coefficient_of_variation_percent"]))
        seed_means = [float(row["mean"]) for row in group]
        output.append(
            {
                "stage": STAGE,
                "classifier": classifier,
                "configuration_id": configuration,
                "phase": phase,
                "metric": metric,
                "unit": METRICS[metric]["unit"],
                "within_seed_cv_min_percent": min(cvs),
                "within_seed_cv_median_percent": statistics.median(cvs),
                "within_seed_cv_mean_percent": statistics.fmean(cvs),
                "within_seed_cv_max_percent": max(cvs),
                "maximum_cv_seed": int(maximum["seed"]),
                "between_seed_cv_percent": coefficient_of_variation(seed_means),
                "high_variability_threshold_defined": False,
                "outliers_removed": False,
                "timing_integrity_limited": True,
                "measurement_integrity_statement": SLEEP_LIMITATION,
            }
        )
    if len(output) != 56:
        raise V08E4Error("Timing variability analysis must contain 56 rows.")
    return output


def build_structural_summary(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        if row["metric"] in STRUCTURAL_METRICS:
            groups[(str(row["classifier"]), str(row["configuration_id"]), str(row["phase"]), str(row["metric"]))].append(row)
    output: list[dict[str, Any]] = []
    for (classifier, configuration, phase, metric), group in sorted(groups.items()):
        values = [float(row["mean"]) for row in group]
        output.append(
            {
                "stage": STAGE,
                "classifier": classifier,
                "configuration_id": configuration,
                "phase": phase,
                "metric": metric,
                "unit": METRICS[metric]["unit"],
                "seed_count": 5,
                "mean_across_seeds": statistics.fmean(values),
                "median_across_seeds": statistics.median(values),
                "sample_sd_across_seeds": statistics.stdev(values),
                "min_across_seeds": min(values),
                "max_across_seeds": max(values),
                "timing_integrity_limited": False,
                "summary_provenance": "DERIVED",
            }
        )
    if len(output) != 64:
        raise V08E4Error("Structural summary must contain 64 rows.")
    return output


def _comparison_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    return {
        (
            str(row["classifier"]),
            str(row["reference_configuration_id"]),
            str(row["phase"]),
            str(row["metric"]),
        ): row
        for row in rows
    }


def build_break_even_analysis(
    paired: Sequence[Mapping[str, Any]], overhead: Mapping[str, Any]
) -> list[dict[str, Any]]:
    index = _comparison_index(paired)
    output: list[dict[str, Any]] = []
    specifications = (
        ("training", "wall_time_sec", "core_optimizer_wall", float(overhead["core_optimization_wall_time_sec"]), "training_fits"),
        ("training", "wall_time_sec", "pipeline_wall", float(overhead["pipeline_wall_time_sec"]), "training_fits"),
        ("training", "process_cpu_time_sec", "core_optimizer_cpu", float(overhead["core_optimization_cpu_time_sec"]), "training_fits"),
        ("inference", "inference_latency_sec", "core_optimizer_wall", float(overhead["core_optimization_wall_time_sec"]), "prediction_operations_of_6000_rows"),
        ("inference", "inference_latency_sec", "pipeline_wall", float(overhead["pipeline_wall_time_sec"]), "prediction_operations_of_6000_rows"),
        ("inference", "per_record_latency_sec", "core_optimizer_wall", float(overhead["core_optimization_wall_time_sec"]), "record_predictions"),
        ("inference", "per_record_latency_sec", "pipeline_wall", float(overhead["pipeline_wall_time_sec"]), "record_predictions"),
        ("inference", "process_cpu_time_sec", "core_optimizer_cpu", float(overhead["core_optimization_cpu_time_sec"]), "benchmark_blocks_of_512_prediction_operations"),
    )
    for classifier in CLASSIFIERS:
        for reference in REFERENCES:
            for phase, metric, overhead_basis, overhead_value, result_unit in specifications:
                row = index[(classifier, reference, phase, metric)]
                saving = float(row["reference_mean_across_seeds"]) - float(row["candidate_mean_across_seeds"])
                applicable = saving > 0.0
                output.append(
                    {
                        "stage": STAGE,
                        "classifier": classifier,
                        "configuration_id": CANDIDATE,
                        "reference_configuration_id": reference,
                        "phase": phase,
                        "metric": metric,
                        "overhead_basis": overhead_basis,
                        "overhead_value": overhead_value,
                        "overhead_unit": "cpu-seconds" if "cpu" in overhead_basis else "seconds",
                        "recurring_saving": saving,
                        "recurring_saving_unit": METRICS[metric]["unit"],
                        "status": "APPLICABLE" if applicable else "NOT_APPLICABLE",
                        "break_even_count": overhead_value / saving if applicable else None,
                        "break_even_count_unit": result_unit if applicable else None,
                        "prediction_operation_rows": 6000 if phase == "inference" else None,
                        "benchmark_inner_operation_count": 512 if phase == "inference" else 1,
                        "provenance": "DERIVED_DESCRIPTIVE_TIMING_LIMITED",
                        "measurement_integrity_statement": SLEEP_LIMITATION,
                    }
                )
    if len(output) != 48:
        raise V08E4Error("Break-even analysis must contain 48 rows.")
    return output


def build_sleep_limitation() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "source_stage": "V0.8-E3S",
        "sleep_audit_classification": "E3_PARTIAL_RERUN_REQUIRED",
        "project_governance_decision": "PRESERVE_AND_ANALYZE_WITH_LIMITATION_NO_RERUN_AT_E4",
        "recorded_modern_standby_intervals_intersecting_cell_windows": 21,
        "reconstructed_cell_execution_windows": 42,
        "conservatively_corresponding_observations": 420,
        "confirmed_observation_level_suspend_overlap": 0,
        "descriptive_timing_outlier_count": 16,
        "descriptive_timing_outlier_observation_ids": list(TIMING_OUTLIER_IDS),
        "classic_wall_only_suspend_signature_found": False,
        "cpu_increased_with_timing_outliers": True,
        "affected_cell_ids": list(SLEEP_LIMITED_CELLS),
        "outliers_removed": False,
        "rerun_performed": False,
        "timing_interpretation": "INDICATIVE_NOT_DEFINITIVE",
        "required_wording": SLEEP_LIMITATION,
    }


def _predictive_lookup(predictive: Mapping[str, Any]) -> dict[tuple[str, str, str], float]:
    return {
        (str(row["classifier"]), str(row["configuration_id"]), str(row["metric"])): float(row["mean"])
        for row in predictive["summaries"]
    }


def _mean_resource(
    index: Mapping[tuple[str, str, int, str, str], Mapping[str, Any]],
    classifier: str,
    configuration: str,
    phase: str,
    metric: str,
) -> float:
    return statistics.fmean(float(index[(classifier, configuration, seed, phase, metric)]["mean"]) for seed in SEEDS)


def _dominates(left: Mapping[str, float], right: Mapping[str, float], objectives: Sequence[tuple[str, str]]) -> bool:
    no_worse = True
    better = False
    for metric, direction in objectives:
        if direction == "maximize":
            no_worse &= left[metric] >= right[metric]
            better |= left[metric] > right[metric]
        else:
            no_worse &= left[metric] <= right[metric]
            better |= left[metric] < right[metric]
    return no_worse and better


def build_tradeoff_analysis(
    summaries: Sequence[Mapping[str, Any]], predictive: Mapping[str, Any]
) -> dict[str, Any]:
    resources = _summary_index(summaries)
    predictions = _predictive_lookup(predictive)
    points: list[dict[str, Any]] = []
    for classifier in CLASSIFIERS:
        for configuration in CONFIGURATIONS:
            point = {
                "classifier": classifier,
                "configuration_id": configuration,
                "average_precision": predictions[(classifier, configuration, "average_precision")],
                "f1": predictions[(classifier, configuration, "f1")],
                "recall": predictions[(classifier, configuration, "recall")],
                "precision": predictions[(classifier, configuration, "precision")],
                "roc_auc": predictions[(classifier, configuration, "roc_auc")],
                "feature_count": _mean_resource(resources, classifier, configuration, "inference", "feature_count"),
                "model_size_bytes": _mean_resource(resources, classifier, configuration, "inference", "serialized_model_bytes"),
                "input_memory_bytes": _mean_resource(resources, classifier, configuration, "inference", "input_dataframe_memory_bytes"),
                "training_wall_time_sec": _mean_resource(resources, classifier, configuration, "training", "wall_time_sec"),
                "training_cpu_time_sec": _mean_resource(resources, classifier, configuration, "training", "process_cpu_time_sec"),
                "inference_latency_sec": _mean_resource(resources, classifier, configuration, "inference", "inference_latency_sec"),
                "inference_cpu_time_sec": _mean_resource(resources, classifier, configuration, "inference", "process_cpu_time_sec"),
                "throughput_records_sec": _mean_resource(resources, classifier, configuration, "inference", "throughput_records_sec"),
                "training_absolute_peak_rss_mib": _mean_resource(resources, classifier, configuration, "training", "absolute_peak_rss_mib"),
                "inference_absolute_peak_rss_mib": _mean_resource(resources, classifier, configuration, "inference", "absolute_peak_rss_mib"),
                "predictive_provenance": "IMPORTED_HISTORICAL",
                "timing_interpretation": "INDICATIVE_NOT_DEFINITIVE",
            }
            points.append(point)
    objectives = (
        ("average_precision", "maximize"),
        ("f1", "maximize"),
        ("recall", "maximize"),
        ("feature_count", "minimize"),
        ("model_size_bytes", "minimize"),
        ("input_memory_bytes", "minimize"),
        ("training_wall_time_sec", "minimize"),
        ("inference_latency_sec", "minimize"),
        ("throughput_records_sec", "maximize"),
    )
    for point in points:
        values = {name: float(point[name]) for name, _ in objectives}
        peers = [candidate for candidate in points if candidate["classifier"] == point["classifier"] and candidate is not point]
        dominators = []
        for peer in peers:
            peer_values = {name: float(peer[name]) for name, _ in objectives}
            if _dominates(peer_values, values, objectives):
                dominators.append(str(peer["configuration_id"]))
        point["descriptive_post_lock_nondominated"] = not dominators
        point["descriptive_dominated_by"] = sorted(dominators)
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "analysis_scope": "DESCRIPTIVE_POST_LOCK_ANALYSIS_NOT_FORMAL_MULTIOBJECTIVE_OPTIMIZATION",
        "winner_modified": False,
        "objective_weights_used": False,
        "composite_score_used": False,
        "timing_integrity_statement": SLEEP_LIMITATION,
        "objective_directions": [{"metric": metric, "direction": direction} for metric, direction in objectives],
        "points": points,
        "resource_tradeoff_classification": "STRUCTURAL_GAIN_TIMING_UNCERTAIN",
        "k10_vs_k11_interpretation": "BPSO-K10 provides the smallest locked representation while MI-K11 offers a slightly stronger Decision Tree predictive tradeoff.",
        "k10_vs_k42_interpretation": "BPSO-K10 provides substantially greater dimensional reduction than K42; predictive superiority is not claimed.",
        "classifier_interpretation": "Decision Tree remains the primary predictive classifier; Logistic Regression is a resource robustness check with weak absolute predictive quality.",
    }


def result_lock_semantic_hash(lock: Mapping[str, Any]) -> str:
    payload = dict(lock)
    payload.pop("created_at_utc", None)
    payload.pop("semantic_result_lock_sha256", None)
    return _json_hash(payload)


def verify_result_lock(lock: Mapping[str, Any]) -> None:
    if (
        lock.get("stage") != STAGE
        or lock.get("status") != "RESOURCE_RESULT_LOCKED"
        or lock.get("observation_count") != 800
        or lock.get("resource_tradeoff_classification") != "STRUCTURAL_GAIN_TIMING_UNCERTAIN"
        or lock.get("direct_energy_status") != "DIRECT_ENERGY_UNAVAILABLE"
        or result_lock_semantic_hash(lock) != lock.get("semantic_result_lock_sha256")
    ):
        raise V08E4Error("V0.8-E4 result lock verification failed.")


def build_result_lock(
    inputs: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    paired_hashes: Mapping[str, str],
    break_even: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lock: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "status": "RESOURCE_RESULT_LOCKED",
        "created_at_utc": _utc_now(),
        "input_artifact_hashes": dict(inputs["input_hashes"]),
        "e3_observation_dataset_sha256": inputs["input_hashes"][str(E3_DIR / "v08e_computational_observations.json")],
        "environment_id": inputs["manifest"]["environment_id"],
        "protocol_sha256": inputs["manifest"]["protocol_sha256"],
        "configuration_identities": inputs["identities"],
        "seeds": list(SEEDS),
        "classifiers": list(CLASSIFIERS),
        "phases": list(PHASES),
        "observation_count": 800,
        "scientifically_complete_cells": 80,
        "statistical_protocol": {
            "scientific_unit": "seed_level_mean_after_ten_nested_measurement_repetitions",
            "n": 5,
            "df": 4,
            "paired_t_critical_95": T_CRITICAL_95_DF4,
            "p_values_computed": False,
            "outliers_removed": False,
        },
        "sleep_audit_limitation": {
            "classification": "E3_PARTIAL_RERUN_REQUIRED",
            "rerun_declined_by_project_governance": True,
            "cell_windows": 42,
            "conservative_observations": 420,
            "confirmed_observation_overlap": 0,
            "required_wording": SLEEP_LIMITATION,
        },
        "predictive_source_identity": {
            "stage": "V0.8-D",
            "provenance": "IMPORTED_HISTORICAL",
            "semantic_result_lock_sha256": inputs["predictive"]["source_semantic_result_lock_sha256"],
            "source_summary_sha256": inputs["predictive"]["source_summary_sha256"],
        },
        "optimizer_overhead_source_identity": {
            "stage": "V0.8-C",
            "provenance": "IMPORTED_HISTORICAL",
            "source_artifact_sha256": inputs["overhead"]["source_artifact_sha256"],
        },
        "direct_energy_status": "DIRECT_ENERGY_UNAVAILABLE",
        "paired_analysis_hashes": dict(paired_hashes),
        "analysis_artifact_hashes": dict(artifact_hashes),
        "resource_tradeoff_classification": "STRUCTURAL_GAIN_TIMING_UNCERTAIN",
        "break_even_status_counts": dict(Counter(str(row["status"]) for row in break_even)),
        "semantic_result_lock_sha256": None,
    }
    lock["semantic_result_lock_sha256"] = result_lock_semantic_hash(lock)
    verify_result_lock(lock)
    return lock


def run_e4_analysis(
    *, e3_dir: Path | str = E3_DIR, output_dir: Path | str = OUTPUT_DIR
) -> dict[str, Any]:
    """Generate governed E4 artifacts without executing models or optimizers."""

    output = Path(output_dir)
    if output.exists():
        raise V08E4Error(f"E4 output already exists; refusing overwrite: {output}")
    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        raise V08E4Error(f"E4 staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        inputs = load_e4_inputs(e3_dir)
        frozen_before = dict(inputs["frozen_hashes"])
        raw_before = dict(inputs["input_hashes"])
        summaries = build_within_seed_summaries(inputs["rows"])
        paired = build_paired_resource_comparisons(summaries)
        variability = build_timing_variability(summaries)
        structural = build_structural_summary(summaries)
        break_even = build_break_even_analysis(paired, inputs["overhead"])
        sleep = build_sleep_limitation()
        tradeoff = build_tradeoff_analysis(summaries, inputs["predictive"])
        overhead = {
            **inputs["overhead"],
            "analysis_stage": STAGE,
            "overhead_role": "ONE_TIME_IMPORTED_HISTORICAL_SEARCH_COST",
            "core_and_pipeline_overheads_kept_separate": True,
            "optimizer_rerun": False,
        }

        published: list[Path] = []
        for stem, table in (
            ("v08e_within_seed_summary", summaries),
            ("v08e_paired_resource_comparisons", paired),
            ("v08e_timing_variability", variability),
            ("v08e_structural_resource_summary", structural),
            ("v08e_break_even_analysis", break_even),
        ):
            published.extend(_write_table(staging, stem, table))
        for name, payload in (
            ("v08e_optimizer_overhead_summary.json", overhead),
            ("v08e_tradeoff_analysis.json", tradeoff),
            ("v08e_sleep_limitation.json", sleep),
        ):
            path = staging / name
            _atomic_write_json(path, payload)
            published.append(path)

        artifact_hashes = {path.name: v08e.sha256_file(path) for path in sorted(published)}
        paired_hashes = {
            name: digest
            for name, digest in artifact_hashes.items()
            if name.startswith("v08e_paired_resource_comparisons")
        }
        lock = build_result_lock(inputs, artifact_hashes, paired_hashes, break_even)
        lock_path = staging / "v08e_scientific_result_lock.json"
        _atomic_write_json(lock_path, lock)
        verify_result_lock(_read_json(lock_path))
        published.append(lock_path)

        frozen_after = _canonical_frozen_snapshot(inputs["preflight"])
        raw_after = _snapshot(Path(path) for path in raw_before)
        if frozen_after != frozen_before or raw_after != raw_before:
            raise V08E4Error("Frozen historical or raw E3 inputs changed during E4 analysis.")
        execution = {
            "schema_version": SCHEMA_VERSION,
            "stage": STAGE,
            "status": "COMPLETED",
            "readiness": "E4_COMPLETE",
            "observation_count": 800,
            "scientifically_complete_cells": 80,
            "within_seed_summary_rows": len(summaries),
            "paired_comparison_rows": len(paired),
            "variability_rows": len(variability),
            "structural_summary_rows": len(structural),
            "break_even_rows": len(break_even),
            "break_even_status_counts": dict(Counter(str(row["status"]) for row in break_even)),
            "sleep_audit_classification": sleep["sleep_audit_classification"],
            "resource_tradeoff_classification": tradeoff["resource_tradeoff_classification"],
            "semantic_result_lock_sha256": lock["semantic_result_lock_sha256"],
            "raw_e3_inputs_unchanged": True,
            "frozen_historical_inputs_unchanged": True,
            "optimizer_rerun": False,
            "predictive_evaluation_rerun": False,
            "feature_reselection_performed": False,
            "outliers_removed": False,
            "direct_energy_status": "DIRECT_ENERGY_UNAVAILABLE",
        }
        execution_path = staging / "v08e_e4_execution_summary.json"
        _atomic_write_json(execution_path, execution)
        published.append(execution_path)
        manifest = {path.name: v08e.sha256_file(path) for path in sorted(published)}
        _atomic_write_json(staging / "v08e_e4_artifact_hashes.json", manifest)
        staging.replace(output)
        return execution
    except BaseException:
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
        raise


def verify_e4_artifacts(output_dir: Path | str = OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    manifest = _read_json(output / "v08e_e4_artifact_hashes.json")
    for name, expected in manifest.items():
        if v08e.sha256_file(output / name) != expected:
            raise V08E4Error(f"E4 artifact hash mismatch: {name}")
    lock = _read_json(output / "v08e_scientific_result_lock.json")
    verify_result_lock(lock)
    execution = _read_json(output / "v08e_e4_execution_summary.json")
    if execution.get("status") != "COMPLETED" or execution.get("readiness") != "E4_COMPLETE":
        raise V08E4Error("E4 execution summary is incomplete.")
    return execution


def main() -> int:
    result = run_e4_analysis()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
