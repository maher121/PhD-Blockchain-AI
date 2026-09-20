"""Governance, schedule, measurement and statistics tests for V1.0-F.

These tests exercise the V1.0-F computational-resource protocol against the real
frozen V0.8-E/V0.9-D/V0.9-E/V1.0-D/V1.0-E locks and synthetic observation rows.
They verify that starting provenance is HEAD-agnostic, prior locks are untouched,
only the six frozen configurations (including the V1.0-D locked HYBRID-K13
winner) are measured, the BPSO/BGWO/Hybrid subsets come from the locks, joined
v10d/v10e lock attributes are stamped on every observation, timestamps and units
are present, energy is never fabricated, paired comparisons align on seeds, and
the result lock is reproducible.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.pipeline_v08e as v08e
import src.pipeline_v09d as v09d
import src.pipeline_v09e as v09e
import src.pipeline_v09f as v09f
import src.pipeline_v10e as v10e
import src.pipeline_v10f as v10f
from src.green.measurement import (
    ComputationalObservation,
    WorkerResult,
    WorkerStatus,
)
from src.lightweight.resource_monitor import ResourceMeasurement


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_basis():
    return v10e.load_frozen_basis()


@pytest.fixture(scope="module")
def real_hybrid_lock():
    return v10e.load_verified_hybrid_winner_lock()


@pytest.fixture(scope="module")
def real_bpso_lock():
    return v10e.load_verified_bpso_winner_lock()


@pytest.fixture(scope="module")
def real_bgwo_lock():
    return v10e.load_verified_bgwo_winner_lock()


@pytest.fixture(scope="module")
def real_preflight(real_hybrid_lock, real_bpso_lock, real_bgwo_lock):
    """Run V1.0-F preflight HEAD-agnostically.

    Provenance is anchored to the recorded V1.0-E starting checkpoint e7465e9
    (HEAD_AGNOSTIC_ANCESTRY): the live repository HEAD may advance while the
    recorded checkpoint must remain an ancestor of it.
    """
    return v10f.verify_v10f_preflight(
        starting_checkpoint=None,
        hybrid_lock=real_hybrid_lock,
        bpso_lock=real_bpso_lock,
        bgwo_lock=real_bgwo_lock,
    )


@pytest.fixture(scope="module")
def configuration_plan(real_preflight):
    return real_preflight["configuration_plan"]


# --------------------------------------------------------------------------- #
# Synthetic observation helpers
# --------------------------------------------------------------------------- #
_CONFIG_BASE = {
    "K43": 1.00,
    "K42": 0.97,
    "MI-K11": 0.93,
    "BPSO-K10": 0.90,
    "BGWO-K14": 0.80,
    "HYBRID-K13": 0.78,
}
_CONFIG_FEATURES = {
    "K43": 43,
    "K42": 42,
    "MI-K11": 11,
    "BPSO-K10": 10,
    "BGWO-K14": 14,
    "HYBRID-K13": 13,
}
_CONFIG_MODEL_BYTES = {
    "K43": 9000,
    "K42": 8800,
    "MI-K11": 5000,
    "BPSO-K10": 4800,
    "BGWO-K14": 4600,
    "HYBRID-K13": 4500,
}


def _observation_row(
    phase: str,
    classifier: str,
    configuration_id: str,
    seed: int,
    repetition: int,
    *,
    standby_gap_sec: float = 0.0,
) -> dict:
    base = _CONFIG_BASE[configuration_id] * (1.0 + 0.01 * (repetition - 1))
    if phase == "inference":
        inner_count = v10f.INFERENCE_INNER_OPERATION_COUNT
        records = 100
    else:
        inner_count = v10f.TRAINING_INNER_OPERATION_COUNT
        records = 100
    measured_records = inner_count * records
    return {
        "observation_id": (
            f"v10f__{phase}__{classifier}__{configuration_id}__s{seed}__r{repetition:02d}"
        ),
        "schema_version": v10f.V10F_SCHEMA_VERSION,
        "stage": v10f.V10F_STAGE,
        "environment_id": "synthetic",
        "classifier": classifier,
        "configuration_id": configuration_id,
        "source_configuration_id": "synthetic",
        "seed": seed,
        "phase": phase,
        "outer_repetition": repetition,
        "inner_operation_count": inner_count,
        "measured_operation_count": inner_count,
        "record_count": records,
        "measured_record_count": measured_records,
        "feature_count": _CONFIG_FEATURES[configuration_id],
        "warmup_calls": v10f.WARMUP_CALLS,
        "worker_pid": 4321,
        "wall_time_sec": base,
        "process_cpu_time_sec": base * 0.9,
        "start_rss_mib": 100.0,
        "end_rss_mib": 101.0,
        "absolute_peak_rss_mib": 105.0,
        "incremental_peak_rss_mib": 5.0,
        "input_dataframe_memory_bytes": 1000 * _CONFIG_FEATURES[configuration_id],
        "numpy_dense_nbytes": 800 * _CONFIG_FEATURES[configuration_id],
        "serialized_model_bytes": _CONFIG_MODEL_BYTES[configuration_id],
        "inference_latency_sec": base / inner_count,
        "per_record_latency_sec": base / measured_records,
        "throughput_records_sec": measured_records / base,
        "time_unit": "seconds",
        "rss_unit": "MiB",
        "size_unit": "bytes",
        "throughput_unit": "records/second",
        "measurement_scope": "synthetic",
        "measurement_provenance": "DIRECT_COMPUTATIONAL",
        "derived_measurement_provenance": "DERIVED",
        "workload_sha256": "w" * 64,
        "model_sha256": "m" * 64,
        "configuration_features_sha256": "c" * 64,
        "v06_lock_sha256": "v" * 64,
        "v08c_semantic_lock_sha256": "b" * 64,
        "v09d_winner_semantic_lock_sha256": "d" * 64,
        "v09e_result_semantic_lock_sha256": "e" * 64,
        "v10d_winner_semantic_lock_sha256": "h" * 64,
        "v10e_result_semantic_lock_sha256": "x" * 64,
        "status": "SUCCESS",
        "failure_type": None,
        "failure_stage": None,
        "failure_reason": None,
        "observation_started_at_utc": "2026-01-01T00:00:00+00:00",
        "observation_finished_at_utc": "2026-01-01T00:00:01+00:00",
        "wall_clock_elapsed_sec": base + standby_gap_sec,
        "monotonic_elapsed_sec": base,
        "standby_gap_sec": standby_gap_sec,
    }


def _full_matrix() -> list[dict]:
    rows: list[dict] = []
    for phase in v10f.PHASES:
        for classifier in v10f.CLASSIFIERS:
            for configuration_id in v10f.CONFIGURATION_IDS:
                for seed in v10f.SEEDS:
                    for repetition in v10f.REPETITIONS:
                        rows.append(
                            _observation_row(
                                phase, classifier, configuration_id, seed, repetition
                            )
                        )
    return rows


def _resources(wall_time_sec: float = 0.5, inner_count: int = 1) -> ResourceMeasurement:
    return ResourceMeasurement(
        wall_time_sec=wall_time_sec,
        cpu_time_sec=wall_time_sec * 0.8,
        average_cpu_percent=80.0,
        peak_cpu_percent=None,
        peak_rss_bytes=300,
        start_rss_bytes=100,
        end_rss_bytes=200,
    )


def _fake_runner_factory(*, inner_count: int):
    def _runner(operation, *, prepare, inner_operation_count, phase, warmup_calls,
                outer_repetitions, records_per_operation, timeout_seconds,
                measurement_scope, start_method=None):
        assert outer_repetitions == 1, "V1.0-F must spawn one fresh worker per repetition."
        observation = ComputationalObservation(
            phase=phase,
            outer_repetition=1,
            inner_operation_count=inner_operation_count,
            measured_operation_count=inner_operation_count,
            warmup_calls=warmup_calls,
            resources=_resources(0.5, inner_count),
            records_per_operation=records_per_operation,
            dataframe_deep_memory_bytes=123,
            numpy_dense_nbytes=456,
            serialized_model_size_bytes=789,
            measurement_scope=measurement_scope,
        )
        result = WorkerResult(
            1,
            WorkerStatus.COMPLETED,
            999,
            warmup_calls,
            inner_operation_count,
            0.001,
            (),
            observation=observation,
        )
        return SimpleNamespace(results=(result,))

    return _runner


def _training_fake_context() -> tuple[SimpleNamespace, SimpleNamespace]:
    context = SimpleNamespace(plan=SimpleNamespace(validation_lock_sha256="v" * 64))
    features = pd.DataFrame({"f1": np.arange(4, dtype=float), "f2": np.arange(4, dtype=float)})
    workloads = SimpleNamespace(
        get=lambda seed, phase: SimpleNamespace(
            features=features, labels=pd.Series([0, 1, 0, 1])
        ),
        workload_hashes={(42, "training"): "w" * 64},
    )
    return context, workloads


# --------------------------------------------------------------------------- #
# Provenance and preflight
# --------------------------------------------------------------------------- #
def test_starting_head_and_stage_constants():
    assert v10f.V10F_STAGE == "V1.0-F"
    assert v10f.V10F_SCHEMA_VERSION == "v1.0-f-resource-efficiency-1"
    assert v10f.EXPECTED_STARTING_CHECKPOINT == v10e.STARTING_CHECKPOINT == "e7465e9"
    assert v10f.STARTING_COMMIT == v10e.STARTING_COMMIT
    assert v10f.PROVENANCE_POLICY == "HEAD_AGNOSTIC_ANCESTRY"
    current = v10f.current_head_short()
    assert re.fullmatch(r"[0-9a-f]{7,40}", current), current


def test_preflight_passes_with_frozen_identities(real_preflight):
    assert real_preflight["status"] == "PASS"
    assert all(real_preflight["checks"].values())
    assert real_preflight["direct_energy_status"] == v10f.DIRECT_ENERGY_STATUS
    assert real_preflight["starting_checkpoint"] == v10f.EXPECTED_STARTING_CHECKPOINT
    assert real_preflight["provenance_policy"] == v10f.PROVENANCE_POLICY
    assert real_preflight["hybrid_winner_selected_feature_count"] == 13
    assert real_preflight["bgwo_winner_selected_feature_count"] == 14
    assert real_preflight["bpso_winner_selected_feature_count"] == 10


def test_configuration_plan_is_six_frozen_configs(configuration_plan):
    assert tuple(plan["configuration_id"] for plan in configuration_plan) == v10f.CONFIGURATION_IDS
    counts = {plan["configuration_id"]: plan["feature_count"] for plan in configuration_plan}
    assert counts == {
        "K43": 43,
        "K42": 42,
        "MI-K11": 11,
        "BPSO-K10": 10,
        "BGWO-K14": 14,
        "HYBRID-K13": 13,
    }


def test_locked_subsets_come_from_locks(configuration_plan):
    by_id = {plan["configuration_id"]: plan for plan in configuration_plan}
    for seed in v10f.SEEDS:
        assert tuple(by_id["BGWO-K14"]["features_by_seed"][str(seed)]) == v10f.EXPECTED_BGWO_FEATURES
        assert tuple(by_id["BPSO-K10"]["features_by_seed"][str(seed)]) == v10f.EXPECTED_BPSO_FEATURES
        assert tuple(by_id["HYBRID-K13"]["features_by_seed"][str(seed)]) == v10f.EXPECTED_HYBRID_FEATURES
    assert by_id["BGWO-K14"]["source_semantic_lock_sha256"] == v10f.EXPECTED_BGWO_WINNER["semantic_lock_sha256"]
    assert by_id["BPSO-K10"]["source_semantic_lock_sha256"] == v10f.EXPECTED_BPSO_WINNER["semantic_lock_sha256"]
    assert by_id["HYBRID-K13"]["source_semantic_lock_sha256"] == v10f.EXPECTED_HYBRID_WINNER["semantic_lock_sha256"]


def test_classifier_settings_frozen(real_preflight):
    plan = real_preflight["classifier_plan"]
    assert plan["decision_tree"]["parameters"] == {
        "max_depth": 5,
        "min_samples_leaf": 20,
        "class_weight": "balanced",
    }
    assert plan["logistic_regression"]["parameters"] == {
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": 500,
        "C": 1.0,
    }
    assert plan["decision_tree"]["prediction_threshold"] == 0.5
    assert plan["logistic_regression"]["prediction_threshold"] == 0.5


def test_observation_fields_include_v10d_and_v10e_lock_attrs():
    assert "v10d_winner_semantic_lock_sha256" in v10f.OBSERVATION_FIELDS
    assert "v10e_result_semantic_lock_sha256" in v10f.OBSERVATION_FIELDS
    for field in v10f.OBSERVATION_FIELDS:
        assert isinstance(field, str) and field


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #
def test_schedule_accounting_and_determinism():
    schedule = v10f.build_observation_schedule()
    assert len(schedule) == v10f.EXPECTED_TOTAL_OBSERVATIONS == 1200
    assert v10f.EXPECTED_TRAINING_OBSERVATIONS == 600
    assert v10f.EXPECTED_INFERENCE_OBSERVATIONS == 600
    ids = [item.observation_id for item in schedule]
    assert len(set(ids)) == 1200
    assert schedule == v10f.build_observation_schedule()
    cells = v10f.schedule_cells(schedule)
    assert len(cells) == v10f.EXPECTED_CELL_COUNT == 120
    assert {cell.configuration_id for cell in cells} == set(v10f.CONFIGURATION_IDS)
    training = sum(item.phase == "training" for item in schedule)
    inference = sum(item.phase == "inference" for item in schedule)
    assert (training, inference) == (600, 600)


def test_schedule_has_timestamped_ten_repetition_blocks():
    schedule = v10f.build_observation_schedule()
    block = schedule[: v10f.OUTER_REPETITIONS]
    assert tuple(item.outer_repetition for item in block) == v10f.REPETITIONS
    assert len({(item.phase, item.classifier, item.configuration_id, item.seed) for item in block}) == 1


def test_observation_schema_includes_timestamps():
    assert set(v10f.TIMESTAMP_FIELDS) <= set(v10f.OBSERVATION_FIELDS)
    for field in v10f.TIMESTAMP_FIELDS:
        assert field in v10f.OBSERVATION_FIELDS


# --------------------------------------------------------------------------- #
# Cell measurement
# --------------------------------------------------------------------------- #
def test_run_cell_measurement_records_training_metadata(configuration_plan):
    context, workloads = _training_fake_context()
    plan = {p["configuration_id"]: p for p in configuration_plan}["HYBRID-K13"]
    features = tuple(plan["features_by_seed"]["42"])
    features_df = pd.DataFrame({name: np.arange(4, dtype=float) for name in features})
    workloads.get = lambda seed, phase: SimpleNamespace(
        features=features_df, labels=pd.Series([0, 1, 0, 1])
    )
    workloads.workload_hashes[(42, "training")] = "w" * 64
    cell = v08e.BenchmarkCell("decision_tree", "HYBRID-K13", "v10d_locked_hybrid_winner", 42, "training", 1)
    rows = v10f._run_cell_measurement(
        context=context,
        cell=cell,
        configuration_plan=plan,
        model_artifact=SimpleNamespace(
            file_sha256="m" * 64,
            serialized_model_bytes=2048,
            feature_manifest_sha256="c" * 64,
        ),
        workloads=workloads,
        inference_workload=None,
        environment_id="env",
        v08c_semantic_lock_sha256="b" * 64,
        v09d_semantic_lock_sha256="d" * 64,
        v09e_semantic_result_lock_sha256="e" * 64,
        v10d_winner_semantic_lock_sha256="h" * 64,
        v10e_result_semantic_lock_sha256="x" * 64,
        runner=_fake_runner_factory(inner_count=1),
    )
    assert len(rows) == v10f.OUTER_REPETITIONS
    for row in rows:
        assert row["phase"] == "training"
        assert row["measurement_scope"] == "model.fit_only_data_loading_selection_serialization_excluded"
        assert row["status"] == "SUCCESS"
        assert row["observation_started_at_utc"] and row["observation_finished_at_utc"]
        assert row["standby_gap_sec"] >= 0.0
        assert row["time_unit"] == "seconds" and row["rss_unit"] == "MiB"
        assert row["v10d_winner_semantic_lock_sha256"] == "h" * 64
        assert row["v10e_result_semantic_lock_sha256"] == "x" * 64


def test_run_cell_measurement_records_inference_metadata(configuration_plan):
    context, _ = _training_fake_context()
    plan = {p["configuration_id"]: p for p in configuration_plan}["BPSO-K10"]
    features = tuple(plan["features_by_seed"]["42"])
    features_df = pd.DataFrame({name: np.arange(4, dtype=float) for name in features})
    inference_workload = v08e.InferenceFeatureWorkload(
        seed=42,
        features=features_df,
        workload_sha256="w" * 64,
        row_ids_sha256="r" * 64,
        feature_matrix_sha256="f" * 64,
    )
    cell = v08e.BenchmarkCell("logistic_regression", "BPSO-K10", "v08c_locked_bpso_k10", 42, "inference", 512)
    rows = v10f._run_cell_measurement(
        context=context,
        cell=cell,
        configuration_plan=plan,
        model_artifact=SimpleNamespace(
            file_sha256="m" * 64,
            serialized_model_bytes=1024,
            feature_manifest_sha256="c" * 64,
        ),
        workloads=SimpleNamespace(),
        inference_workload=inference_workload,
        environment_id="env",
        v08c_semantic_lock_sha256="b" * 64,
        v09d_semantic_lock_sha256="d" * 64,
        v09e_semantic_result_lock_sha256="e" * 64,
        v10d_winner_semantic_lock_sha256="h" * 64,
        v10e_result_semantic_lock_sha256="x" * 64,
        runner=_fake_runner_factory(inner_count=512),
    )
    assert len(rows) == v10f.OUTER_REPETITIONS
    for row in rows:
        assert row["phase"] == "inference"
        assert row["measurement_scope"] == "prediction_only_model_loading_warmup_excluded"
        assert row["inner_operation_count"] == 512
        assert row["measured_record_count"] == 512 * 4


def test_validate_observation_rows_rejects_missing_timestamp():
    row = _observation_row("training", "decision_tree", "HYBRID-K13", 42, 1)
    good = [dict(row, observation_id=f"x{i}") for i in range(10)]
    v10f.validate_observation_rows(good)
    bad = [dict(item) for item in good]
    bad[0]["observation_started_at_utc"] = ""
    with pytest.raises(v10f.V10FError):
        v10f.validate_observation_rows(bad)


def test_validate_observation_rows_rejects_negative_metric():
    row = _observation_row("training", "decision_tree", "HYBRID-K13", 42, 1)
    rows = [dict(row, observation_id=f"x{i}") for i in range(10)]
    rows[2]["wall_time_sec"] = -1.0
    with pytest.raises(v10f.V10FError):
        v10f.validate_observation_rows(rows)


# --------------------------------------------------------------------------- #
# Statistics and comparisons
# --------------------------------------------------------------------------- #
_EXTRA_METRICS = ("wall_clock_elapsed_sec", "monotonic_elapsed_sec", "standby_gap_sec")


def test_within_and_across_seed_summaries():
    rows = _full_matrix()
    within = v10f.summarize_within_seed(rows)
    assert len(within) == 120 * len((*v10f.SCIENTIFIC_METRICS, *_EXTRA_METRICS))
    assert len(within) == 1200
    summary = v10f.summarize_across_seeds(within)
    keys = {(row["classifier"], row["configuration_id"], row["phase"], row["metric"]) for row in summary}
    assert len(keys) == 2 * 6 * 2 * 10
    assert len(summary) == 240
    assert all(row["seed_count"] == 5 for row in summary)


def test_paired_comparisons_align_on_seeds():
    within = v10f.summarize_within_seed(_full_matrix())
    comparisons = v10f.build_resource_comparisons(within)
    primary = [row for row in comparisons if row["comparison_tier"] == "primary"]
    secondary = [row for row in comparisons if row["comparison_tier"] == "secondary"]
    assert primary and secondary
    seen = {(row["candidate_configuration_id"], row["reference_configuration_id"]) for row in comparisons}
    assert {(row["candidate_configuration_id"], row["reference_configuration_id"]) for row in primary} == set(
        v10f.PRIMARY_PAIRS
    )
    assert {(row["candidate_configuration_id"], row["reference_configuration_id"]) for row in secondary} == set(
        v10f.SECONDARY_PAIRS
    )
    assert seen == set(v10f.PRIMARY_PAIRS) | set(v10f.SECONDARY_PAIRS)
    assert len(comparisons) == 5 * 2 * 2 * len(v10f.SCIENTIFIC_METRICS)
    for row in comparisons:
        assert row["n"] == 5 and row["df"] == 4 and row["t_critical"] == v10f.T_CRITICAL_95_DF4
        assert [pair["seed"] for pair in row["seed_pairs"]] == list(v10f.SEEDS)


def test_model_size_and_feature_reduction():
    rows = _full_matrix()
    size = v10f.build_model_size_comparison(rows)
    dt_hybrid = next(
        entry
        for entry in size["entries"]
        if entry["classifier"] == "decision_tree" and entry["configuration_id"] == "HYBRID-K13"
    )
    assert dt_hybrid["serialized_model_bytes"] == _CONFIG_MODEL_BYTES["HYBRID-K13"]
    assert dt_hybrid["difference_bytes_vs_k43"] == (
        _CONFIG_MODEL_BYTES["HYBRID-K13"] - _CONFIG_MODEL_BYTES["K43"]
    )
    reduction = v10f.build_feature_reduction()
    assert reduction["hybrid_k"] == 13
    assert reduction["hybrid_reduction_percent_vs_k43"] == pytest.approx(100.0 * 30 / 43)
    counts = {entry["configuration_id"]: entry["feature_count"] for entry in reduction["entries"]}
    assert counts == {
        "K43": 43,
        "K42": 42,
        "MI-K11": 11,
        "BPSO-K10": 10,
        "BGWO-K14": 14,
        "HYBRID-K13": 13,
    }


def test_timing_quality_flags_standby_without_deleting():
    rows = _full_matrix()
    rows[0]["standby_gap_sec"] = 5.0
    rows[0]["wall_clock_elapsed_sec"] += 5.0
    timing = v10f.build_timing_quality(rows)
    assert timing["observation_count"] == 1200
    assert timing["cell_count"] == 120
    assert timing["flagged_observation_count"] >= 1
    assert timing["outliers_removed"] is False
    assert timing["exclusion_policy"] == "PRESERVE_ALL_OBSERVATIONS_NO_SILENT_EXCLUSION"
    assert "STANDBY" in timing["timing_interpretation"]


# --------------------------------------------------------------------------- #
# Optimizer overhead, break-even, trade-off
# --------------------------------------------------------------------------- #
def test_optimizer_overhead_imports_frozen_evidence():
    overhead = v10f.build_optimizer_overhead()
    assert overhead["optimizer_rerun"] is False
    assert overhead["bgwo"]["optimizer_rerun"] is False
    assert overhead["bgwo"]["optimizer_runs"] == 5
    assert overhead["bgwo"]["candidate_requests"] == 768
    assert overhead["bgwo"]["decision_tree_fits"] == 3840
    assert overhead["bgwo"]["measurement_provenance"] == "IMPORTED_HISTORICAL"
    assert overhead["bpso"]["candidate_requests"] == 972
    assert overhead["bpso"]["decision_tree_fits"] == 4860
    assert overhead["bpso"]["measurement_provenance"] == "IMPORTED_HISTORICAL"
    assert overhead["hybrid"]["candidate_requests"] == 960
    assert overhead["hybrid"]["unique_evaluations"] == 940
    assert overhead["hybrid"]["cache_hits"] == 20
    assert overhead["hybrid"]["decision_tree_fits"] == 4700
    assert overhead["hybrid"]["core_optimization_wall_time_sec"] == pytest.approx(
        475.54950684303185, rel=1e-12
    )
    assert overhead["hybrid"]["optimizer_rerun"] is False
    assert overhead["hybrid"]["measurement_provenance"] == "IMPORTED_HISTORICAL"
    assert (
        overhead["hybrid"]["winner_semantic_lock_sha256"]
        == v10f.EXPECTED_HYBRID_WINNER["semantic_lock_sha256"]
    )


def test_break_even_guards_negative_savings():
    within = v10f.summarize_within_seed(_full_matrix())
    overhead = {"hybrid": {"core_optimization_wall_time_sec": 100.0}}
    analysis = v10f.build_break_even(within_seed=within, optimizer_overhead=overhead)
    assert analysis["timing_break_even_is_not_energy_break_even"] is True
    assert analysis["energy_break_even_status"] == "NOT_APPLICABLE_DIRECT_ENERGY_UNAVAILABLE"
    assert len(analysis["rows"]) == 3
    for row in analysis["rows"]:
        assert row["per_model_recurring_saving_sec"] > 0
        assert row["status"] == "APPLICABLE"
        assert row["break_even_repetitions"] is not None


def test_break_even_not_applicable_when_candidate_slower():
    within = [
        {
            "classifier": classifier,
            "configuration_id": config,
            "seed": seed,
            "phase": "training",
            "metric": "wall_time_sec",
            "mean": 2.0 if config == "HYBRID-K13" else 1.0,
        }
        for classifier in v10f.CLASSIFIERS
        for seed in v10f.SEEDS
        for config in v10f.CONFIGURATION_IDS
    ]
    overhead = {"hybrid": {"core_optimization_wall_time_sec": 100.0}}
    analysis = v10f.build_break_even(within_seed=within, optimizer_overhead=overhead)
    assert all(row["status"] == "NOT_APPLICABLE" for row in analysis["rows"])
    assert all(row["break_even_repetitions"] is None for row in analysis["rows"])


def test_tradeoff_is_descriptive_only():
    summary = v10f.summarize_across_seeds(v10f.summarize_within_seed(_full_matrix()))
    tradeoff = v10f.build_tradeoff(summary=summary)
    assert tradeoff["artifact_kind"] == "DESCRIPTIVE_POST_HOC_TRADEOFF_ANALYSIS"
    assert tradeoff["optimization_performed"] is False
    assert tradeoff["objective_function_introduced"] is False
    assert tradeoff["nsga_or_mopso_used"] is False
    assert len(tradeoff["entries"]) == len(v10f.CONFIGURATION_IDS)
    by_id = {entry["configuration_id"]: entry for entry in tradeoff["entries"]}
    assert set(by_id) == set(v10f.CONFIGURATION_IDS)
    assert by_id["HYBRID-K13"]["feature_count"] == 13
    assert by_id["K43"]["test_average_precision"] and by_id["K43"]["test_f1"]


# --------------------------------------------------------------------------- #
# Result lock and prior-lock preservation
# --------------------------------------------------------------------------- #
def _build_test_lock(real_preflight, artifact_hashes):
    return v10f.build_result_lock(
        starting_head=real_preflight["starting_head"],
        preflight=real_preflight,
        configurations=real_preflight["configuration_plan"],
        classifiers=real_preflight["classifier_plan"],
        artifact_hashes=artifact_hashes,
        environment_id="env",
        environment_sha256="a" * 64,
        observation_counts={
            "training": 600,
            "inference": 600,
            "total": 1200,
            "cells": 120,
            "failure_count": 0,
        },
        timing_quality={"observation_level_timestamps_present": True},
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T01:00:00+00:00",
        wall_time_sec=3600.0,
    )


def test_result_lock_roundtrip_and_semantic_hash(real_preflight):
    artifacts = {name: "0" * 64 for name in (
        v10f.RAW_CSV_PATH.name,
        v10f.SUMMARY_CSV_PATH.name,
        v10f.COMPARISONS_CSV_PATH.name,
        v10f.TIMING_QUALITY_PATH.name,
        v10f.OPTIMIZER_OVERHEAD_PATH.name,
        v10f.TRADEOFF_PATH.name,
    )}
    lock = _build_test_lock(real_preflight, artifacts)
    v10f.verify_result_lock(lock)
    assert lock["semantic_result_lock_sha256"] == v10f.result_lock_semantic_hash(lock)
    assert lock["optimizer_invoked"] is False
    assert lock["optimizer_rerun"] is False
    assert lock["final_test_accessed"] is False
    assert lock["feature_reselection_performed"] is False
    assert lock["winner_replaced"] is False
    assert lock["direct_energy_measured"] is False
    assert lock["joule_estimation_from_cpu_or_wall_performed"] is False
    assert lock["tdp_multiplication_performed"] is False


def test_result_lock_tamper_is_detected(real_preflight):
    artifacts = {name: "0" * 64 for name in (
        v10f.RAW_CSV_PATH.name,
        v10f.SUMMARY_CSV_PATH.name,
        v10f.COMPARISONS_CSV_PATH.name,
        v10f.TIMING_QUALITY_PATH.name,
        v10f.OPTIMIZER_OVERHEAD_PATH.name,
        v10f.TRADEOFF_PATH.name,
    )}
    lock = _build_test_lock(real_preflight, artifacts)
    tampered = dict(lock)
    tampered["hybrid_selected_feature_count"] = 12
    with pytest.raises(v10f.V10FError):
        v10f.verify_result_lock(tampered)

    semantic_tamper = dict(lock)
    semantic_tamper["semantic_result_lock_sha256"] = "f" * 64
    with pytest.raises(v10f.V10FError):
        v10f.verify_result_lock(semantic_tamper)


def test_previous_locks_unchanged():
    v09e_lock = v10f._read_json(v10f.V09E_RESULT_LOCK_PATH)
    v09e.verify_result_lock(v09e_lock)
    assert v09e_lock["semantic_result_lock_sha256"] == v10f.EXPECTED_V09E_RESULT_LOCK_SHA256

    v08e_lock = v10f._read_json(v10f.V08E_RESULT_LOCK_PATH)
    assert v08e_lock["semantic_result_lock_sha256"] == v10f.EXPECTED_V08E_RESULT_LOCK_SHA256

    bgwo_lock = v09e.load_verified_bgwo_winner_lock()
    v09d.verify_winner_lock(bgwo_lock)
    bpso_lock = v09e.load_verified_bpso_winner_lock()
    assert bpso_lock["mask_sha256"] == v10f.EXPECTED_BPSO_WINNER["mask_sha256"]

    hybrid_lock = v10e.load_verified_hybrid_winner_lock()
    assert hybrid_lock["mask_sha256"] == v10f.EXPECTED_HYBRID_WINNER["mask_sha256"]

    v10e_lock = v10f._read_json(v10f.V10E_RESULT_LOCK_PATH)
    v10e.verify_result_lock(v10e_lock)
    assert v10e_lock["semantic_result_lock_sha256"] == v10f.EXPECTED_V10E_RESULT_LOCK_SHA256


def test_no_optimizer_or_selection_invocation_in_source():
    source = inspect.getsource(v10f)
    for forbidden in (
        "mutual_info_classif",
        "SelectKBest",
        "feature_selection",
        "fit_transform",
        "GridSearchCV",
        "RandomizedSearchCV",
        "binary_grey_wolf",
        "binary_particle_swarm",
    ):
        assert forbidden not in source


# --------------------------------------------------------------------------- #
# Energy governance
# --------------------------------------------------------------------------- #
def test_sleep_governance_never_fabricates_energy():
    governance = v10f.build_sleep_governance(
        environment={"is_wsl": True},
        energy={"decision": "DIRECT_ENERGY_UNAVAILABLE"},
    )
    assert governance["direct_energy_status"] == "DIRECT_ENERGY_UNAVAILABLE"
    assert governance["energy_proxies_only"] is True
    assert governance["joule_estimation_from_cpu_or_wall_performed"] is False
    assert governance["tdp_multiplication_performed"] is False
    assert governance["sleep_governance"]["permanent_power_settings_modified"] is False
    assert governance["sleep_governance"]["host_sleep_availability_verified"] is False


def test_energy_helpers_are_proxy_only():
    energy = v08e.resolve_energy_capability()
    assert energy["decision"] == "DIRECT_ENERGY_UNAVAILABLE"
    assert v10f.DIRECT_ENERGY_STATUS == "DIRECT_ENERGY_UNAVAILABLE"


# --------------------------------------------------------------------------- #
# Checkpoint behaviour
# --------------------------------------------------------------------------- #
def test_checkpoint_roundtrip(tmp_path: Path):
    rows = [
        _observation_row("training", "decision_tree", "HYBRID-K13", 42, repetition)
        for repetition in v10f.REPETITIONS
    ]
    cell = v08e.BenchmarkCell("decision_tree", "HYBRID-K13", "v10d_locked_hybrid_winner", 42, "training", 1)
    path = tmp_path / "cell.json"
    v10f.write_cell_checkpoint(
        path, cell=cell, protocol_hash="p" * 64, environment_id="env", rows=rows
    )
    loaded = v10f.load_valid_checkpoint(
        path, cell=cell, protocol_hash="p" * 64, environment_id="env"
    )
    assert tuple(item["observation_id"] for item in loaded) == tuple(
        item["observation_id"] for item in rows
    )
    with pytest.raises(v10f.V10FCheckpointError):
        v10f.load_valid_checkpoint(
            path, cell=cell, protocol_hash="q" * 64, environment_id="env"
        )


def test_fingerprint_hashes_are_sha256(configuration_plan):
    for plan in configuration_plan:
        digest = hashlib.sha256("".join(plan["features_by_seed"]["42"]).encode()).hexdigest()
        assert len(digest) == 64
        assert isinstance(plan["source_semantic_lock_sha256"], (str, type(None)))


# --------------------------------------------------------------------------- #
# Readiness
# --------------------------------------------------------------------------- #
def test_run_v10f_readiness_without_executing(tmp_path: Path):
    result = v10f.run_v10f(
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoints",
        model_dir=tmp_path / "models",
        execute_matrix=False,
    )
    assert result["status"] == "INFRASTRUCTURE_READY"
    assert result["execute_matrix"] is False
    assert result["expected_total_observations"] == v10f.EXPECTED_TOTAL_OBSERVATIONS == 1200
    assert result["readiness_for_campaign"] == "GO"
    assert (tmp_path / "output" / v10f.MANIFEST_PATH.name).is_file()
    assert (tmp_path / "output" / v10f.PREFLIGHT_PATH.name).is_file()
    assert (tmp_path / "output" / v10f.ENVIRONMENT_PATH.name).is_file()
    assert (tmp_path / "output" / v10f.SLEEP_PATH.name).is_file()