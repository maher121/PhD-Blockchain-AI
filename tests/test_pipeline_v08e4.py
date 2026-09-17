"""Focused tests for governed V0.8-E4 artifact-only resource analysis."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import src.pipeline_v08e4 as e4


@pytest.fixture(scope="module")
def inputs():
    return e4.load_e4_inputs()


@pytest.fixture(scope="module")
def summaries(inputs):
    return e4.build_within_seed_summaries(inputs["rows"])


@pytest.fixture(scope="module")
def paired(summaries):
    return e4.build_paired_resource_comparisons(summaries)


def test_e3_input_matrix_is_complete(inputs) -> None:
    assert len(inputs["rows"]) == 800
    assert all(row["status"] == "SUCCESS" for row in inputs["rows"])
    assert inputs["manifest"]["environment_id"] == "47053d2178876401fc9dcab4def25f61d70ec91304a60f878e245122c1830995"
    assert inputs["manifest"]["protocol_sha256"] == "fd18722b7d064ef1ae7447ff4d64028248c560182b54558f693da130e000c36f"


def test_configuration_identities_remain_frozen(inputs) -> None:
    identities = inputs["identities"]
    assert {name: row["feature_count"] for name, row in identities.items()} == {
        "K43": 43,
        "K42": 42,
        "K11": 11,
        "BPSO-K10": 10,
    }
    assert identities["K11"]["universal_identity"] is False
    assert identities["BPSO-K10"]["universal_identity"] is True


def test_within_seed_summary_uses_ten_nested_repetitions(summaries) -> None:
    assert len(summaries) == 760
    assert all(row["repetition_count"] == 10 for row in summaries)
    assert all(row["outliers_removed"] is False for row in summaries)


def test_paired_analysis_uses_five_seed_units(paired) -> None:
    assert len(paired) == 114
    assert all(row["scientific_n"] == 5 for row in paired)
    assert all(row["df"] == 4 for row in paired)
    assert all(row["t_critical_95"] == e4.T_CRITICAL_95_DF4 for row in paired)
    assert all(row["p_value_computed"] is False for row in paired)


def test_metric_directions_are_explicit(paired) -> None:
    by_metric = {row["metric"]: row["objective_direction"] for row in paired}
    assert by_metric["throughput_records_sec"] == "higher_is_better"
    for metric in set(by_metric) - {"throughput_records_sec"}:
        assert by_metric[metric] == "lower_is_better"


def test_timing_rows_carry_sleep_limitation(paired) -> None:
    timing = [row for row in paired if row["metric"] in e4.TIMING_METRICS]
    structural = [row for row in paired if row["metric"] in e4.STRUCTURAL_METRICS]
    assert timing and all(row["timing_integrity_limited"] is True for row in timing)
    assert all(row["measurement_integrity_statement"] == e4.SLEEP_LIMITATION for row in timing)
    assert structural and all(row["timing_integrity_limited"] is False for row in structural)


def test_variability_has_no_manufactured_threshold(summaries) -> None:
    rows = e4.build_timing_variability(summaries)
    assert len(rows) == 56
    assert all(row["high_variability_threshold_defined"] is False for row in rows)
    assert all(row["outliers_removed"] is False for row in rows)


def test_structural_summary_is_timing_independent(summaries) -> None:
    rows = e4.build_structural_summary(summaries)
    assert len(rows) == 64
    assert all(row["timing_integrity_limited"] is False for row in rows)


def test_break_even_never_reports_negative_count(inputs, paired) -> None:
    rows = e4.build_break_even_analysis(paired, inputs["overhead"])
    assert len(rows) == 48
    assert all(
        row["break_even_count"] is None or float(row["break_even_count"]) > 0.0
        for row in rows
    )
    assert all(row["provenance"] == "DERIVED_DESCRIPTIVE_TIMING_LIMITED" for row in rows)


def test_sleep_governance_is_exact() -> None:
    sleep = e4.build_sleep_limitation()
    assert sleep["sleep_audit_classification"] == "E3_PARTIAL_RERUN_REQUIRED"
    assert sleep["reconstructed_cell_execution_windows"] == 42
    assert sleep["conservatively_corresponding_observations"] == 420
    assert sleep["confirmed_observation_level_suspend_overlap"] == 0
    assert sleep["descriptive_timing_outlier_count"] == 16
    assert sleep["outliers_removed"] is False


def test_tradeoff_is_descriptive_post_lock(inputs, summaries) -> None:
    result = e4.build_tradeoff_analysis(summaries, inputs["predictive"])
    assert result["resource_tradeoff_classification"] == "STRUCTURAL_GAIN_TIMING_UNCERTAIN"
    assert result["winner_modified"] is False
    assert result["objective_weights_used"] is False
    assert result["composite_score_used"] is False
    assert len(result["points"]) == 8


def test_result_lock_hash_excludes_creation_timestamp(inputs, paired) -> None:
    lock = e4.build_result_lock(
        inputs,
        {"artifact.json": "a" * 64},
        {"paired.json": "b" * 64},
        e4.build_break_even_analysis(paired, inputs["overhead"]),
    )
    changed = {**lock, "created_at_utc": "2099-01-01T00:00:00+00:00"}
    assert e4.result_lock_semantic_hash(changed) == lock["semantic_result_lock_sha256"]
    e4.verify_result_lock(changed)


def test_pipeline_source_prohibits_experimental_recomputation() -> None:
    source = inspect.getsource(e4)
    forbidden = (
        "BinaryParticleSwarmOptimizer",
        "average_precision_score",
        "f1_score",
        "recall_score",
        "roc_auc_score",
        "run_fresh_worker_protocol",
    )
    assert all(marker not in source for marker in forbidden)


def test_full_analysis_is_semantically_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    raw_path = e4.E3_DIR / "v08e_computational_observations.json"
    before = e4.v08e.sha256_file(raw_path)
    e4.run_e4_analysis(output_dir=first)
    e4.run_e4_analysis(output_dir=second)
    first_lock = json.loads((first / "v08e_scientific_result_lock.json").read_text())
    second_lock = json.loads((second / "v08e_scientific_result_lock.json").read_text())
    assert first_lock["semantic_result_lock_sha256"] == second_lock["semantic_result_lock_sha256"]
    assert first_lock["analysis_artifact_hashes"] == second_lock["analysis_artifact_hashes"]
    assert e4.v08e.sha256_file(raw_path) == before
    assert e4.verify_e4_artifacts(first)["readiness"] == "E4_COMPLETE"
