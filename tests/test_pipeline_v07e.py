"""Focused tests for artifact-only V0.7-E statistical analysis."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics

import pytest

import src.pipeline_v07e as v07e


@pytest.fixture(scope="module")
def actual_v07d():
    return v07e.load_v07d_observations()


@pytest.fixture(scope="module")
def actual_performance():
    return v07e.load_frozen_performance()


@pytest.fixture(scope="module")
def synthetic_observations() -> list[dict[str, object]]:
    rows = []
    feature_counts = dict(zip(v07e.CONFIGURATIONS, (43, 42, 11)))
    factors = dict(zip(v07e.CONFIGURATIONS, (1.0, 0.98, 0.65)))
    for classifier_index, classifier in enumerate(v07e.CLASSIFIERS):
        for configuration in v07e.CONFIGURATIONS:
            features = feature_counts[configuration]
            for seed in v07e.SEEDS:
                for phase in v07e.PHASES:
                    records = 28_000 if phase == "training" else 6_000
                    for repetition in v07e.REPETITIONS:
                        base = (0.2 if phase == "training" else 1.2) * (classifier_index + 1)
                        wall = base * factors[configuration] * (1 + (seed - 42) * 0.01 + repetition * 0.001)
                        operations = 1 if phase == "training" else 512
                        per_operation = wall / operations
                        rows.append(
                            {
                                "observation_id": f"{classifier}-{configuration}-{seed}-{phase}-{repetition}",
                                "classifier": classifier,
                                "configuration_id": configuration,
                                "seed": seed,
                                "phase": phase,
                                "outer_repetition": repetition,
                                "status": "SUCCESS",
                                "environment_id": "synthetic-environment",
                                "feature_count": features,
                                "selected_input_bytes": records * features * 8 + 132,
                                "serialized_model_bytes": (4000 if classifier == "decision_tree" else 2400) * features // 43,
                                "wall_time_sec": wall,
                                "process_cpu_time_sec": wall * 0.9,
                                "absolute_peak_rss_mib": 150 + features * 0.1 + repetition * 0.01,
                                "incremental_peak_rss_mib": 2 + features * 0.02 + repetition * 0.005,
                                "per_operation_latency_sec": per_operation,
                                "per_record_latency_sec": per_operation / records,
                                "throughput_records_sec": records / per_operation,
                                "workload_sha256": f"{seed}-{phase}",
                            }
                        )
    return rows


@pytest.fixture(scope="module")
def seed_summaries(synthetic_observations):
    return v07e.build_seed_summaries(synthetic_observations)


@pytest.fixture(scope="module")
def paired(seed_summaries):
    return v07e.build_paired_comparisons(seed_summaries)


@pytest.fixture(scope="module")
def synthetic_performance():
    rows = []
    for classifier in v07e.CLASSIFIERS:
        for config_index, configuration in enumerate(v07e.CONFIGURATIONS):
            for seed in v07e.SEEDS:
                rows.append(
                    {
                        "classifier": classifier,
                        "configuration_id": configuration,
                        "seed": seed,
                        "average_precision": 0.8 - config_index * 0.005 + (seed - 44) * 0.001,
                        "f1": 0.7 - config_index * 0.004 + (seed - 44) * 0.001,
                        "recall": 0.75 - config_index * 0.003 + (seed - 44) * 0.001,
                        "precision": 0.68 - config_index * 0.002 + (seed - 44) * 0.001,
                        "roc_auc": 0.9 - config_index * 0.001 + (seed - 44) * 0.001,
                        "predictive_provenance": "IMPORTED_HISTORICAL",
                        "measurement_domain": "V0.6_predictive_performance",
                        "historical_timing_imported": False,
                    }
                )
    return rows


def test_01_exactly_600_v07d_observations_loaded(actual_v07d) -> None:
    assert len(actual_v07d[0]) == 600


def test_02_zero_failed_observations_required(actual_v07d) -> None:
    assert {row["status"] for row in actual_v07d[0]} == {"SUCCESS"}


def test_03_training_and_inference_counts(actual_v07d) -> None:
    assert Counter(row["phase"] for row in actual_v07d[0]) == {"training": 300, "inference": 300}


def test_04_exactly_three_configurations(actual_v07d) -> None:
    assert {row["configuration_id"] for row in actual_v07d[0]} == set(v07e.CONFIGURATIONS)


def test_05_exactly_two_classifiers(actual_v07d) -> None:
    assert {row["classifier"] for row in actual_v07d[0]} == set(v07e.CLASSIFIERS)


def test_06_exactly_five_seeds(actual_v07d) -> None:
    assert {row["seed"] for row in actual_v07d[0]} == set(v07e.SEEDS)


def test_07_ten_repetitions_aggregated_within_seed(seed_summaries) -> None:
    assert len(seed_summaries) == 420
    assert {row["repetition_count"] for row in seed_summaries} == {10}


def test_08_scientific_sample_size_remains_five(paired) -> None:
    assert {row["scientific_sample_size"] for row in paired} == {5}
    assert all(len(group) == 5 for group in _paired_groups(paired).values())


def test_09_baseline_denominator_correct() -> None:
    assert v07e.relative_change_percent(80.0, 100.0) == pytest.approx(-20.0)


def test_10_relative_change_sign_convention_correct() -> None:
    assert v07e.relative_change_percent(120.0, 100.0) == pytest.approx(20.0)


def test_11_paired_comparison_uses_same_seed(paired) -> None:
    group = next(iter(_paired_groups(paired).values()))
    assert {row["seed"] for row in group} == set(v07e.SEEDS)
    assert all("paired_same_seed" in row["comparison_scope"] for row in group)


def test_12_t95_ci_calculation_correct() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    low, high = v07e.t95_interval(values)
    margin = v07e.T_CRITICAL_95_DF4 * statistics.stdev(values) / math.sqrt(5)
    assert (low, high) == pytest.approx((3.0 - margin, 3.0 + margin))


def test_13_cv_calculation_correct() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert v07e.coefficient_of_variation(values) == pytest.approx(100 * statistics.stdev(values) / 3)


def test_14_no_observation_silently_removed(synthetic_observations) -> None:
    with pytest.raises(v07e.V07EError, match="repetition"):
        v07e.build_seed_summaries(synthetic_observations[:-1])


def test_15_no_new_model_training_interface() -> None:
    source = inspect.getsource(v07e)
    assert ".fit(" not in source
    assert "sklearn" not in source
    assert "LightweightDetector" not in source


def test_16_no_workload_regeneration_interface() -> None:
    source = inspect.getsource(v07e)
    assert "prepare_verified_workloads" not in source
    assert "pipeline_v07b" not in source


def test_17_v06_performance_imported_only(actual_performance) -> None:
    rows = actual_performance[0]
    assert len(rows) == 30
    assert Counter(row["source_stage"] for row in rows) == {"V0.6-F": 15, "V0.6-G": 15}


def test_18_imported_performance_provenance_correct(actual_performance) -> None:
    assert {row["predictive_provenance"] for row in actual_performance[0]} == {"IMPORTED_HISTORICAL"}


def test_19_v06_timing_not_pooled(seed_summaries, synthetic_performance) -> None:
    rows = v07e.build_performance_efficiency(
        synthetic_performance,
        seed_summaries,
        {(classifier, configuration): True for classifier in v07e.CLASSIFIERS for configuration in v07e.CONFIGURATIONS},
    )
    assert all(row["historical_timing_pooled"] is False for row in rows)
    assert all(row["predictive_measurement_domain"] != row["computational_measurement_domain"] for row in rows)


def test_20_pareto_dominance_correct() -> None:
    objectives = (("performance", "maximize"), ("latency", "minimize"))
    assert v07e.dominates({"performance": 0.9, "latency": 1.0}, {"performance": 0.8, "latency": 1.2}, objectives)
    assert not v07e.dominates({"performance": 0.9, "latency": 1.3}, {"performance": 0.8, "latency": 1.2}, objectives)


def test_21_no_arbitrary_objective_weights(seed_summaries, synthetic_performance) -> None:
    performance_efficiency = v07e.build_performance_efficiency(
        synthetic_performance,
        seed_summaries,
        {(classifier, configuration): True for classifier in v07e.CLASSIFIERS for configuration in v07e.CONFIGURATIONS},
    )
    rows = v07e.build_pareto_analysis(performance_efficiency)
    assert rows and all(row["objective_weights_used"] is False and row["composite_score_used"] is False for row in rows)


def test_22_no_direct_energy_result(seed_summaries, paired) -> None:
    keys = {key.lower() for row in (*seed_summaries, *paired) for key in row}
    assert not any("joule" in key or "energy_consum" in key for key in keys)


def test_23_no_carbon_result(seed_summaries, paired) -> None:
    keys = {key.lower() for row in (*seed_summaries, *paired) for key in row}
    assert not any("carbon" in key or "co2" in key for key in keys)


def test_24_no_statistical_significance_overclaim_fields(paired) -> None:
    keys = {key.lower() for row in paired for key in row}
    assert not any(key.startswith("p_value") or "significant" in key for key in keys)


def test_25_json_csv_parity(tmp_path: Path, seed_summaries) -> None:
    json_path, csv_path = v07e._write_table(tmp_path, "table", seed_summaries[:5])
    json_rows = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    json_keys = [(row["classifier"], str(row["seed"]), row["metric"]) for row in json_rows]
    csv_keys = [(row["classifier"], row["seed"], row["metric"]) for row in csv_rows]
    assert json_keys == csv_keys


def test_26_artifact_hash_integrity(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "hashes.json"
    manifest.write_text(json.dumps({"artifact.json": digest}), encoding="utf-8")
    verified = v07e._verify_hash_manifest(tmp_path, manifest)
    assert verified[str(artifact)] == digest


def test_27_v06_immutable(actual_performance) -> None:
    for path, expected in actual_performance[2].items():
        assert v07e.v07d.sha256_file(path) == expected


def test_28_v07d_immutable(actual_v07d) -> None:
    for path, expected in actual_v07d[2].items():
        assert v07e.v07d.sha256_file(path) == expected


def _paired_groups(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["comparison_id"], []).append(row)
    return groups
