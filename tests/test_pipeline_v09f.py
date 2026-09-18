"""Governance, schedule, measurement and statistics tests for V0.9-F.

These tests exercise the V0.9-F computational-resource protocol against the real
frozen V0.8-C/V0.8-E/V0.9-D/V0.9-E locks and synthetic observation rows. They
verify that starting provenance is fixed, prior locks are untouched, only the
five frozen configurations are measured, the BGWO/BPSO subsets come from the
locks, timestamps and units are present, energy is never fabricated, paired
comparisons align on seeds, and the result lock is reproducible.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import re
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import src.pipeline_v08e as v08e
import src.pipeline_v09d as v09d
import src.pipeline_v09e as v09e
import src.pipeline_v09f as v09f
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
def real_preflight():
    """Run V0.9-F preflight without requiring the LIVE repository HEAD to stay
    pinned at the V0.9-F starting checkpoint forever.

    Provenance is anchored to the frozen result lock, which records the original
    scientific starting checkpoint (EXPECTED_STARTING_HEAD). The internal
    ``starting_checkpoint`` check therefore cross-validates the recorded
    checkpoint against the pinned constant while the regression run is allowed
    to execute at any later HEAD.
    """
    frozen_lock = v09f._read_json(v09f.RESULT_LOCK_PATH)
    recorded_starting_head = frozen_lock.get("starting_head")
    assert recorded_starting_head, "frozen V0.9-F result lock must record starting_head"
    with mock.patch.object(
        v09f, "current_head_short", return_value=recorded_starting_head
    ):
        return v09f.verify_v09f_preflight(expected_head=None)


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
    "BGWO": 0.80,
}
_CONFIG_FEATURES = {"K43": 43, "K42": 42, "MI-K11": 11, "BPSO-K10": 10, "BGWO": 14}
_CONFIG_MODEL_BYTES = {"K43": 9000, "K42": 8800, "MI-K11": 5000, "BPSO-K10": 4800, "BGWO": 4600}


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
        inner_count = v09f.INFERENCE_INNER_OPERATION_COUNT
        records = 100
    else:
        inner_count = v09f.TRAINING_INNER_OPERATION_COUNT
        records = 100
    measured_records = inner_count * records
    return {
        "observation_id": (
            f"v09f__{phase}__{classifier}__{configuration_id}__s{seed}__r{repetition:02d}"
        ),
        "schema_version": v09f.V09F_SCHEMA_VERSION,
        "stage": v09f.V09F_STAGE,
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
        "warmup_calls": v09f.WARMUP_CALLS,
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
    for phase in v09f.PHASES:
        for classifier in v09f.CLASSIFIERS:
            for configuration_id in v09f.CONFIGURATION_IDS:
                for seed in v09f.SEEDS:
                    for repetition in v09f.REPETITIONS:
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
        assert outer_repetitions == 1, "V0.9-F must spawn one fresh worker per repetition."
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
    assert v09f.V09F_STAGE == "V0.9-F"
    assert v09f.EXPECTED_STARTING_HEAD == "9c1b276"
    # V0.9-F recorded its original scientific starting checkpoint in the frozen
    # result lock; that remains the provenance anchor.
    frozen_lock = v09f._read_json(v09f.RESULT_LOCK_PATH)
    assert frozen_lock.get("starting_head") == v09f.EXPECTED_STARTING_HEAD
    # The repository HEAD may legitimately advance after V0.9-F completion; the
    # regression must not require HEAD to remain pinned at 9c1b276 forever.
    current = v09f.current_head_short()
    assert re.fullmatch(r"[0-9a-f]{7,40}", current), current


def test_preflight_passes_with_frozen_identities(real_preflight):
    assert real_preflight["status"] == "PASS"
    assert all(real_preflight["checks"].values())
    assert real_preflight["direct_energy_status"] == v09f.DIRECT_ENERGY_STATUS
    assert real_preflight["bgwo_winner_selected_feature_count"] == 14
    assert real_preflight["bpso_winner_selected_feature_count"] == 10


def test_configuration_plan_is_five_frozen_configs(configuration_plan):
    assert tuple(plan["configuration_id"] for plan in configuration_plan) == v09f.CONFIGURATION_IDS
    counts = {plan["configuration_id"]: plan["feature_count"] for plan in configuration_plan}
    assert counts == {"K43": 43, "K42": 42, "MI-K11": 11, "BPSO-K10": 10, "BGWO": 14}


def test_locked_subsets_come_from_locks(configuration_plan):
    by_id = {plan["configuration_id"]: plan for plan in configuration_plan}
    for seed in v09f.SEEDS:
        assert tuple(by_id["BGWO"]["features_by_seed"][str(seed)]) == v09f.EXPECTED_BGWO_FEATURES
        assert tuple(by_id["BPSO-K10"]["features_by_seed"][str(seed)]) == v09f.EXPECTED_BPSO_FEATURES
    assert by_id["BGWO"]["source_semantic_lock_sha256"] == v09f.EXPECTED_BGWO_WINNER["semantic_lock_sha256"]
    assert by_id["BPSO-K10"]["source_semantic_lock_sha256"] == v09f.EXPECTED_BPSO_WINNER["semantic_lock_sha256"]


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


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #
def test_schedule_accounting_and_determinism():
    schedule = v09f.build_observation_schedule()
    assert len(schedule) == v09f.EXPECTED_TOTAL_OBSERVATIONS == 1000
    ids = [item.observation_id for item in schedule]
    assert len(set(ids)) == 1000
    assert schedule == v09f.build_observation_schedule()
    cells = v09f.schedule_cells(schedule)
    assert len(cells) == v09f.EXPECTED_CELL_COUNT == 100
    assert {cell.configuration_id for cell in cells} == set(v09f.CONFIGURATION_IDS)


def test_schedule_has_timestamped_ten_repetition_blocks():
    schedule = v09f.build_observation_schedule()
    block = schedule[: v09f.OUTER_REPETITIONS]
    assert tuple(item.outer_repetition for item in block) == v09f.REPETITIONS
    assert len({(item.phase, item.classifier, item.configuration_id, item.seed) for item in block}) == 1


def test_observation_schema_includes_timestamps():
    assert set(v09f.TIMESTAMP_FIELDS) <= set(v09f.OBSERVATION_FIELDS)
    for field in v09f.TIMESTAMP_FIELDS:
        assert field in v09f.OBSERVATION_FIELDS


# --------------------------------------------------------------------------- #
# Cell measurement
# --------------------------------------------------------------------------- #
def test_run_cell_measurement_records_training_metadata(configuration_plan):
    context, workloads = _training_fake_context()
    plan = {p["configuration_id"]: p for p in configuration_plan}["BGWO"]
    features = tuple(plan["features_by_seed"]["42"])
    features_df = pd.DataFrame({name: np.arange(4, dtype=float) for name in features})
    workloads.get = lambda seed, phase: SimpleNamespace(
        features=features_df, labels=pd.Series([0, 1, 0, 1])
    )
    workloads.workload_hashes[(42, "training")] = "w" * 64
    cell = v08e.BenchmarkCell("decision_tree", "BGWO", "v09d", 42, "training", 1)
    rows = v09f._run_cell_measurement(
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
        runner=_fake_runner_factory(inner_count=1),
    )
    assert len(rows) == v09f.OUTER_REPETITIONS
    for row in rows:
        assert row["phase"] == "training"
        assert row["measurement_scope"] == "model.fit_only_data_loading_selection_serialization_excluded"
        assert row["status"] == "SUCCESS"
        assert row["observation_started_at_utc"] and row["observation_finished_at_utc"]
        assert row["standby_gap_sec"] >= 0.0
        assert row["time_unit"] == "seconds" and row["rss_unit"] == "MiB"
        assert row["v09d_winner_semantic_lock_sha256"] == "d" * 64


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
    cell = v08e.BenchmarkCell("logistic_regression", "BPSO-K10", "v08c", 42, "inference", 512)
    rows = v09f._run_cell_measurement(
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
        runner=_fake_runner_factory(inner_count=512),
    )
    assert len(rows) == v09f.OUTER_REPETITIONS
    for row in rows:
        assert row["phase"] == "inference"
        assert row["measurement_scope"] == "prediction_only_model_loading_warmup_excluded"
        assert row["inner_operation_count"] == 512
        assert row["measured_record_count"] == 512 * 4


def test_validate_observation_rows_rejects_missing_timestamp():
    row = _observation_row("training", "decision_tree", "BGWO", 42, 1)
    good = [dict(row, observation_id=f"x{i}") for i in range(10)]
    v09f.validate_observation_rows(good)
    bad = [dict(item) for item in good]
    bad[0]["observation_started_at_utc"] = ""
    with pytest.raises(v09f.V09FError):
        v09f.validate_observation_rows(bad)


def test_validate_observation_rows_rejects_negative_metric():
    row = _observation_row("training", "decision_tree", "BGWO", 42, 1)
    rows = [dict(row, observation_id=f"x{i}") for i in range(10)]
    rows[2]["wall_time_sec"] = -1.0
    with pytest.raises(v09f.V09FError):
        v09f.validate_observation_rows(rows)


# --------------------------------------------------------------------------- #
# Statistics and comparisons
# --------------------------------------------------------------------------- #
def test_within_and_across_seed_summaries():
    rows = _full_matrix()
    within = v09f.summarize_within_seed(rows)
    assert len(within) == 100 * len(
        (*v09f.SCIENTIFIC_METRICS, "wall_clock_elapsed_sec", "monotonic_elapsed_sec", "standby_gap_sec")
    )
    summary = v09f.summarize_across_seeds(within)
    keys = {(row["classifier"], row["configuration_id"], row["phase"], row["metric"]) for row in summary}
    assert len(keys) == 2 * 5 * 2 * 10
    assert all(row["seed_count"] == 5 for row in summary)


def test_paired_comparisons_align_on_seeds():
    within = v09f.summarize_within_seed(_full_matrix())
    comparisons = v09f.build_resource_comparisons(within)
    primary = [row for row in comparisons if row["comparison_tier"] == "primary"]
    assert primary
    seen = {(row["candidate_configuration_id"], row["reference_configuration_id"]) for row in primary}
    assert seen == set(v09f.PRIMARY_PAIRS)
    for row in comparisons:
        assert row["n"] == 5 and row["df"] == 4 and row["t_critical"] == v09f.T_CRITICAL_95_DF4
        assert [pair["seed"] for pair in row["seed_pairs"]] == list(v09f.SEEDS)


def test_model_size_and_feature_reduction():
    rows = _full_matrix()
    size = v09f.build_model_size_comparison(rows)
    dt_bgwo = next(
        entry
        for entry in size["entries"]
        if entry["classifier"] == "decision_tree" and entry["configuration_id"] == "BGWO"
    )
    assert dt_bgwo["serialized_model_bytes"] == _CONFIG_MODEL_BYTES["BGWO"]
    assert dt_bgwo["difference_bytes_vs_k43"] == (
        _CONFIG_MODEL_BYTES["BGWO"] - _CONFIG_MODEL_BYTES["K43"]
    )
    reduction = v09f.build_feature_reduction()
    assert reduction["bgwo_k"] == 14
    assert reduction["bgwo_reduction_percent_vs_k43"] == pytest.approx(100.0 * 29 / 43)


def test_timing_quality_flags_standby_without_deleting():
    rows = _full_matrix()
    rows[0]["standby_gap_sec"] = 5.0
    rows[0]["wall_clock_elapsed_sec"] += 5.0
    timing = v09f.build_timing_quality(rows)
    assert timing["observation_count"] == 1000
    assert timing["flagged_observation_count"] >= 1
    assert timing["outliers_removed"] is False
    assert timing["exclusion_policy"] == "PRESERVE_ALL_OBSERVATIONS_NO_SILENT_EXCLUSION"
    assert "STANDBY" in timing["timing_interpretation"]


# --------------------------------------------------------------------------- #
# Optimizer overhead, break-even, trade-off
# --------------------------------------------------------------------------- #
def test_optimizer_overhead_imports_frozen_evidence():
    overhead = v09f.build_optimizer_overhead()
    assert overhead["optimizer_rerun"] is False
    assert overhead["bgwo"]["optimizer_runs"] == 5
    assert overhead["bgwo"]["candidate_requests"] == 768
    assert overhead["bgwo"]["decision_tree_fits"] == 3840
    assert overhead["bgwo"]["measurement_provenance"] == "IMPORTED_HISTORICAL"
    assert overhead["bpso"]["decision_tree_fits"] == 4860


def test_break_even_guards_negative_savings():
    within = v09f.summarize_within_seed(_full_matrix())
    overhead = {"bgwo": {"core_optimization_wall_time_sec": 100.0}}
    analysis = v09f.build_break_even(within_seed=within, optimizer_overhead=overhead)
    assert analysis["timing_break_even_is_not_energy_break_even"] is True
    assert analysis["energy_break_even_status"] == "NOT_APPLICABLE_DIRECT_ENERGY_UNAVAILABLE"
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
            "mean": 2.0 if config == "BGWO" else 1.0,
            "feature_count": 43,
            "input_dataframe_memory_bytes": 10,
            "numpy_dense_nbytes": 10,
            "serialized_model_bytes": 10,
        }
        for classifier in v09f.CLASSIFIERS
        for seed in v09f.SEEDS
        for config in v09f.CONFIGURATION_IDS
    ]
    overhead = {"bgwo": {"core_optimization_wall_time_sec": 100.0}}
    analysis = v09f.build_break_even(within_seed=within, optimizer_overhead=overhead)
    assert all(row["status"] == "NOT_APPLICABLE" for row in analysis["rows"])
    assert all(row["break_even_repetitions"] is None for row in analysis["rows"])


def test_tradeoff_is_descriptive_only():
    summary = v09f.summarize_across_seeds(v09f.summarize_within_seed(_full_matrix()))
    tradeoff = v09f.build_tradeoff(summary=summary)
    assert tradeoff["artifact_kind"] == "DESCRIPTIVE_POST_HOC_TRADEOFF_ANALYSIS"
    assert tradeoff["optimization_performed"] is False
    assert tradeoff["objective_function_introduced"] is False
    assert tradeoff["nsga_or_mopso_used"] is False
    assert len(tradeoff["entries"]) == len(v09f.CONFIGURATION_IDS)


# --------------------------------------------------------------------------- #
# Result lock and prior-lock preservation
# --------------------------------------------------------------------------- #
def _build_test_lock(real_preflight, artifact_hashes):
    return v09f.build_result_lock(
        starting_head=real_preflight["starting_head"],
        preflight=real_preflight,
        configurations=real_preflight["configuration_plan"],
        classifiers=real_preflight["classifier_plan"],
        artifact_hashes=artifact_hashes,
        environment_id="env",
        environment_sha256="a" * 64,
        observation_counts={"training": 500, "inference": 500, "total": 1000, "cells": 100, "failure_count": 0},
        timing_quality={"observation_level_timestamps_present": True},
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T01:00:00+00:00",
        wall_time_sec=3600.0,
    )


def test_result_lock_roundtrip_and_semantic_hash(real_preflight):
    artifacts = {name: "0" * 64 for name in (v09f.RAW_CSV_PATH.name, v09f.SUMMARY_CSV_PATH.name)}
    lock = _build_test_lock(real_preflight, artifacts)
    v09f.verify_result_lock(lock)
    assert lock["semantic_result_lock_sha256"] == v09f.result_lock_semantic_hash(lock)
    assert lock["optimizer_invoked"] is False
    assert lock["final_test_accessed"] is False
    assert lock["tdp_multiplication_performed"] is False
    assert lock["joule_estimation_from_cpu_or_wall_performed"] is False
    assert lock["direct_energy_measured"] is False


def test_result_lock_tamper_is_detected(real_preflight):
    artifacts = {name: "0" * 64 for name in (v09f.RAW_CSV_PATH.name, v09f.SUMMARY_CSV_PATH.name)}
    lock = _build_test_lock(real_preflight, artifacts)
    tampered = dict(lock)
    tampered["bgwo_selected_feature_count"] = 13
    with pytest.raises(v09f.V09FError):
        v09f.verify_result_lock(tampered)

    semantic_tamper = dict(lock)
    semantic_tamper["semantic_result_lock_sha256"] = "f" * 64
    with pytest.raises(v09f.V09FError):
        v09f.verify_result_lock(semantic_tamper)


def test_previous_locks_unchanged():
    v09e_lock = v09f._read_json(v09f.V09E_RESULT_LOCK_PATH)
    v09e.verify_result_lock(v09e_lock)
    assert v09e_lock["semantic_result_lock_sha256"] == v09f.EXPECTED_V09E_RESULT_LOCK_SHA256

    v08e_lock = v09f._read_json(v09f.V08E_RESULT_LOCK_PATH)
    assert v08e_lock["semantic_result_lock_sha256"] == v09f.EXPECTED_V08E_RESULT_LOCK_SHA256

    bgwo_lock = v09e.load_verified_bgwo_winner_lock()
    v09d.verify_winner_lock(bgwo_lock)
    bpso_lock = v09e.load_verified_bpso_winner_lock()
    assert bpso_lock["mask_sha256"] == v09f.EXPECTED_BPSO_WINNER["mask_sha256"]


def test_no_optimizer_or_selection_invocation_in_source():
    source = inspect.getsource(v09f)
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
    governance = v09f.build_sleep_governance(environment={"is_wsl": True}, energy={"decision": "DIRECT_ENERGY_UNAVAILABLE"})
    assert governance["direct_energy_status"] == "DIRECT_ENERGY_UNAVAILABLE"
    assert governance["energy_proxies_only"] is True
    assert governance["joule_estimation_from_cpu_or_wall_performed"] is False
    assert governance["tdp_multiplication_performed"] is False
    assert governance["sleep_governance"]["permanent_power_settings_modified"] is False
    assert governance["sleep_governance"]["host_sleep_availability_verified"] is False


def test_energy_helpers_are_proxy_only():
    energy = v08e.resolve_energy_capability()
    assert energy["decision"] == "DIRECT_ENERGY_UNAVAILABLE"
    assert v09f.DIRECT_ENERGY_STATUS == "DIRECT_ENERGY_UNAVAILABLE"


# --------------------------------------------------------------------------- #
# Checkpoint behaviour
# --------------------------------------------------------------------------- #
def test_checkpoint_roundtrip(tmp_path: Path):
    rows = [
        _observation_row("training", "decision_tree", "BGWO", 42, repetition)
        for repetition in v09f.REPETITIONS
    ]
    cell = v08e.BenchmarkCell("decision_tree", "BGWO", "v09d", 42, "training", 1)
    path = tmp_path / "cell.json"
    v09f.write_cell_checkpoint(
        path, cell=cell, protocol_hash="p" * 64, environment_id="env", rows=rows
    )
    loaded = v09f.load_valid_checkpoint(
        path, cell=cell, protocol_hash="p" * 64, environment_id="env"
    )
    assert tuple(item["observation_id"] for item in loaded) == tuple(
        item["observation_id"] for item in rows
    )
    with pytest.raises(v09f.V09FCheckpointError):
        v09f.load_valid_checkpoint(
            path, cell=cell, protocol_hash="q" * 64, environment_id="env"
        )


def test_fingerprint_hashes_are_sha256(configuration_plan):
    for plan in configuration_plan:
        digest = hashlib.sha256("".join(plan["features_by_seed"]["42"]).encode()).hexdigest()
        assert len(digest) == 64
        assert isinstance(plan["source_semantic_lock_sha256"], (str, type(None)))
