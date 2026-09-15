"""Synthetic governance and statistics tests for V0.8-D."""

from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.pipeline_v08d as v08d
from tests.test_bpso_fitness import SpyModel, make_context, spy_factory


@pytest.fixture(scope="module")
def real_basis():
    return v08d.load_frozen_basis()


@pytest.fixture(scope="module")
def real_winner_lock():
    return v08d.load_verified_winner_lock()


@pytest.fixture(scope="module")
def real_plans(real_basis, real_winner_lock):
    return v08d.build_configuration_plan(real_basis, real_winner_lock)


@pytest.fixture()
def synthetic_campaign():
    context = make_context()
    candidates = context.candidate_features
    bpso_features = candidates[:10]
    plans = (
        _plan("K43", candidates, candidates),
        _plan("K42", candidates[:-1], candidates),
        _seed_specific_k11_plan(candidates),
        _plan("BPSO-K10", bpso_features, candidates),
    )
    workload_map = {
        (seed, "training"): workload.train
        for seed, workload in zip(v08d.MODEL_ATTACK_SEEDS, context.workloads)
    }
    workload_map.update(
        {
            (seed, "inference"): workload.validation
            for seed, workload in zip(v08d.MODEL_ATTACK_SEEDS, context.workloads)
        }
    )
    workloads = SimpleNamespace(get=lambda seed, phase: workload_map[(seed, phase)])
    lock = {
        "semantic_lock_sha256": "a" * 64,
        "ordered_selected_features": list(bpso_features),
        "selected_features_sha256": v08d.fingerprint_feature_names(bpso_features),
    }
    SpyModel.fit_calls.clear()
    SpyModel.predict_calls.clear()
    rows = v08d.evaluate_final_test_campaign(
        plans,
        v08d.classifier_plan(),
        workloads,
        lock,
        model_factory=spy_factory,
    )
    return rows, plans, lock


def _plan(identifier, features, candidates):
    return {
        "configuration_id": identifier,
        "source_configuration_id": identifier.lower(),
        "feature_count": len(features),
        "selection_scope": "synthetic_test",
        "selection_semantics": "universal synthetic subset",
        "features_by_seed": {
            str(seed): list(features) for seed in v08d.MODEL_ATTACK_SEEDS
        },
    }


def _seed_specific_k11_plan(candidates):
    return {
        "configuration_id": "K11",
        "source_configuration_id": "mutual_information_select_k_best_k11",
        "feature_count": 11,
        "selection_scope": "synthetic historical rule",
        "selection_semantics": "seed-specific supervised training-derived subsets",
        "features_by_seed": {
            str(seed): list(candidates[offset : offset + 11])
            for offset, seed in enumerate(v08d.MODEL_ATTACK_SEEDS)
        },
    }


def _artifact_hashes():
    return {
        "v08d_final_test_raw.json": "1" * 64,
        "v08d_final_test_summary.json": "2" * 64,
        "v08d_paired_comparisons.json": "3" * 64,
        "v08d_preservation.json": "4" * 64,
        "v08d_test_access_audit.json": "5" * 64,
    }


def test_01_v08c_is_committed_pushed_and_exact() -> None:
    result = v08d.verify_v08c_prerequisites()
    assert result["status"] == "PASS"
    assert result["winner_semantic_lock_sha256"] == v08d.EXPECTED_WINNER[
        "semantic_lock_sha256"
    ]
    assert result["v08c_committed_and_pushed"] is True
    assert result["ahead"] == result["behind"] == 0


def test_02_winner_lock_exact_hashes_and_identity(real_winner_lock) -> None:
    assert real_winner_lock["source_optimizer_seed"] == 1042
    assert real_winner_lock["selected_feature_count"] == 10
    assert real_winner_lock["mask_sha256"] == v08d.EXPECTED_WINNER["mask_sha256"]
    assert real_winner_lock["selected_features_sha256"] == v08d.EXPECTED_WINNER[
        "selected_features_sha256"
    ]
    assert real_winner_lock["semantic_lock_sha256"] == v08d.EXPECTED_WINNER[
        "semantic_lock_sha256"
    ]
    assert tuple(real_winner_lock["ordered_selected_features"]) == v08d.EXPECTED_BPSO_FEATURES


def test_03_exact_four_configuration_semantics(real_plans) -> None:
    assert tuple(plan["configuration_id"] for plan in real_plans) == (
        "K43",
        "K42",
        "K11",
        "BPSO-K10",
    )
    assert [plan["feature_count"] for plan in real_plans] == [43, 42, 11, 10]
    assert real_plans[0]["selection_semantics"] == "universal full candidate manifest"
    assert "deterministic" in real_plans[1]["selection_semantics"]
    assert "seed-specific" in real_plans[2]["selection_semantics"]
    assert "universal locked" in real_plans[3]["selection_semantics"]


def test_04_historical_k11_features_remain_seed_specific(real_plans) -> None:
    k11 = real_plans[2]
    sets = {tuple(k11["features_by_seed"][str(seed)]) for seed in v08d.MODEL_ATTACK_SEEDS}
    assert len(sets) == 5
    assert all(len(features) == 11 for features in sets)


def test_05_frozen_classifier_plans_are_exact() -> None:
    plan = v08d.classifier_plan()
    assert plan["decision_tree"] == {
        "parameters": {
            "max_depth": 5,
            "min_samples_leaf": 20,
            "class_weight": "balanced",
        },
        "prediction_threshold": 0.5,
    }
    assert plan["logistic_regression"] == {
        "parameters": {
            "solver": "liblinear",
            "class_weight": "balanced",
            "max_iter": 500,
            "C": 1.0,
        },
        "prediction_threshold": 0.5,
    }


def test_06_test_access_audit_is_activated_after_lock_verification(tmp_path, real_winner_lock) -> None:
    audit = v08d.activate_test_access_audit(
        tmp_path / "audit.json",
        {"v08c_committed_and_pushed": True},
        real_winner_lock,
    )
    assert audit["status"] == "TEST_ACCESS_ACTIVATED"
    assert audit["winner_lock_verified_before_test_access"] is True
    assert audit["events"][0]["event"] == "WINNER_LOCK_VERIFIED"
    assert audit["events"][1]["event"] == "TEST_ACCESS_ACTIVATED"
    assert audit["final_test_opened"] is False
    assert audit["checks"]["optimizer_invoked"] is False
    assert audit["checks"]["feature_selection_invoked"] is False


def test_07_access_audit_completion_preserves_no_selection_or_tuning(tmp_path, real_winner_lock) -> None:
    path = tmp_path / "audit.json"
    audit = v08d.activate_test_access_audit(
        path, {"v08c_committed_and_pushed": True}, real_winner_lock
    )
    completed = v08d.complete_test_access_audit(path, audit, evaluation_count=40)
    assert completed["status"] == "PASS"
    assert completed["governed_evaluation_count"] == 40
    assert completed["winner_modified_after_access"] is False
    assert completed["partial_metrics_used_for_configuration"] is False


def test_08_synthetic_campaign_has_exact_40_paired_cells(synthetic_campaign) -> None:
    rows, _, _ = synthetic_campaign
    assert len(rows) == 40
    assert {row["configuration_id"] for row in rows} == set(v08d.CONFIGURATION_IDS)
    assert {row["classifier"] for row in rows} == set(v08d.CLASSIFIERS)
    assert {row["seed"] for row in rows} == set(v08d.MODEL_ATTACK_SEEDS)


def test_09_training_only_fit_and_final_test_only_score_are_enforced(synthetic_campaign) -> None:
    rows, _, _ = synthetic_campaign
    assert len(SpyModel.fit_calls) == len(SpyModel.predict_calls) == 40
    assert all(call["index"] == list(range(100, 106)) for call in SpyModel.fit_calls)
    assert all(call["index"] == list(range(200, 204)) for call in SpyModel.predict_calls)
    assert all(row["model_fit_on_test"] is False for row in rows)
    assert all(row["preprocessing_fit_on_test"] is False for row in rows)
    assert all(row["feature_selection_on_test"] is False for row in rows)
    assert all(row["threshold_tuned_on_test"] is False for row in rows)


def test_10_ground_truth_and_metric_outputs_are_complete(synthetic_campaign) -> None:
    rows, _, _ = synthetic_campaign
    metric = rows[0]["metrics"]
    assert metric["average_precision"] == 1.0
    assert metric["f1"] == 1.0
    assert metric["recall"] == 1.0
    assert metric["precision"] == 1.0
    assert metric["roc_auc"] == 1.0
    assert metric["true_positives"] == metric["true_negatives"] == 2
    assert metric["false_positives"] == metric["false_negatives"] == 0
    assert "pr_auc_trapezoidal" in metric
    assert metric["pr_auc_trapezoidal_method"] == "trapezoidal_precision_recall_curve"


def test_11_bpso_features_cannot_be_changed_by_test_data(synthetic_campaign) -> None:
    rows, _, lock = synthetic_campaign
    bpso = [row for row in rows if row["configuration_id"] == "BPSO-K10"]
    assert len(bpso) == 10
    assert all(tuple(row["selected_features"]) == tuple(lock["ordered_selected_features"]) for row in bpso)
    assert all(row["selected_features_sha256"] == lock["selected_features_sha256"] for row in bpso)


def test_12_aggregation_reports_mean_sd_min_max_for_five_seeds(synthetic_campaign) -> None:
    summary = v08d.aggregate_final_test(synthetic_campaign[0])
    expected_count = 4 * 2 * 15
    assert len(summary["summaries"]) == expected_count
    assert all(row["n"] == 5 for row in summary["summaries"])
    assert all({"mean", "std", "min", "max"} <= set(row) for row in summary["summaries"])


def test_13_paired_differences_align_seeds_and_compute_t_interval(synthetic_campaign) -> None:
    rows = deepcopy(synthetic_campaign[0])
    bpso_rows = sorted(
        [row for row in rows if row["classifier"] == "decision_tree" and row["configuration_id"] == "BPSO-K10"],
        key=lambda row: row["seed"],
    )
    differences = [0.01, 0.02, 0.03, 0.04, 0.05]
    for row, difference in zip(bpso_rows, differences):
        row["metrics"]["average_precision"] += difference
    paired = v08d.paired_comparisons(rows)
    result = next(
        row
        for row in paired["comparisons"]
        if row["classifier"] == "decision_tree"
        and row["reference_configuration_id"] == "K43"
        and row["metric"] == "average_precision"
    )
    expected_mean = statistics_mean = sum(differences) / 5
    expected_std = np.std(differences, ddof=1)
    half = v08d.T_CRITICAL_95_DF4 * expected_std / np.sqrt(5)
    assert result["mean_difference"] == pytest.approx(expected_mean)
    assert result["std_difference"] == pytest.approx(expected_std)
    assert result["ci95_low"] == pytest.approx(statistics_mean - half)
    assert result["ci95_high"] == pytest.approx(statistics_mean + half)
    assert result["n"] == 5 and result["df"] == 4
    assert [pair["seed"] for pair in result["pairs"]] == [42, 43, 44, 45, 46]


def test_14_zero_paired_difference_has_zero_interval_and_no_pvalue(synthetic_campaign) -> None:
    paired = v08d.paired_comparisons(synthetic_campaign[0])
    assert len(paired["comparisons"]) == 18
    assert all(row["mean_difference"] == 0.0 for row in paired["comparisons"])
    assert all(row["ci95_low"] == row["ci95_high"] == 0.0 for row in paired["comparisons"])
    assert all(row["p_value_reported"] is False for row in paired["comparisons"])


def test_15_final_test_preservation_uses_unchanged_five_five_ten(synthetic_campaign) -> None:
    preservation = v08d.preservation_analysis(v08d.aggregate_final_test(synthetic_campaign[0]))
    assert preservation["margins"] == {
        "average_precision": 0.05,
        "f1": 0.05,
        "recall": 0.10,
    }
    assert preservation["domain_validated_margins"] is False
    assert all(row["overall_preserved"] for row in preservation["results"])
    assert preservation["cross_classifier_interpretation"] == (
        "PRESERVATION_HOLDS_FOR_BOTH_CLASSIFIERS"
    )


def test_16_preservation_boundary_and_failure_are_exact(synthetic_campaign) -> None:
    summary = v08d.aggregate_final_test(synthetic_campaign[0])
    for row in summary["summaries"]:
        if row["configuration_id"] == "BPSO-K10" and row["metric"] == "average_precision":
            row["mean"] = 0.95
        if row["configuration_id"] == "BPSO-K10" and row["metric"] == "f1":
            row["mean"] = 0.95
        if row["configuration_id"] == "BPSO-K10" and row["metric"] == "recall":
            row["mean"] = 0.90
    preservation = v08d.preservation_analysis(summary)
    assert all(row["overall_preserved"] for row in preservation["results"])
    first = next(
        row
        for row in summary["summaries"]
        if row["configuration_id"] == "BPSO-K10"
        and row["classifier"] == "decision_tree"
        and row["metric"] == "average_precision"
    )
    first["mean"] = 0.949
    preservation = v08d.preservation_analysis(summary)
    dt = next(row for row in preservation["results"] if row["classifier"] == "decision_tree")
    assert dt["overall_preserved"] is False


def test_17_result_lock_semantic_hash_ignores_engineering_timestamps(real_winner_lock) -> None:
    classifiers = v08d.classifier_plan()
    preservation = {
        "results": [
            {"classifier": "decision_tree", "overall_preserved": True},
            {"classifier": "logistic_regression", "overall_preserved": False},
        ]
    }
    first = v08d.build_final_test_lock(
        real_winner_lock,
        classifiers,
        _artifact_hashes(),
        preservation,
        started_at="a",
        completed_at="b",
        wall_time_sec=1.0,
        rerun_count=0,
    )
    second = v08d.build_final_test_lock(
        real_winner_lock,
        classifiers,
        _artifact_hashes(),
        preservation,
        started_at="c",
        completed_at="d",
        wall_time_sec=999.0,
        rerun_count=0,
    )
    assert first["semantic_result_lock_sha256"] == second["semantic_result_lock_sha256"]
    v08d.verify_final_test_lock(first)


def test_18_result_lock_contains_winner_immutability_and_no_resource_claims(real_winner_lock) -> None:
    lock = v08d.build_final_test_lock(
        real_winner_lock,
        v08d.classifier_plan(),
        _artifact_hashes(),
        {
            "results": [
                {"classifier": "decision_tree", "overall_preserved": True},
                {"classifier": "logistic_regression", "overall_preserved": True},
            ]
        },
        started_at="a",
        completed_at="b",
        wall_time_sec=1.0,
        rerun_count=0,
    )
    assert lock["optimizer_invoked"] is False
    assert lock["feature_reselection_performed"] is False
    assert lock["test_driven_tuning_performed"] is False
    assert lock["resource_benchmark_executed"] is False
    assert lock["direct_energy_measured"] is False
    assert "not modified" in lock["winner_unchanged_statement"]


def test_19_existing_result_lock_prohibits_campaign_rerun(tmp_path) -> None:
    (tmp_path / v08d.RESULT_LOCK_PATH.name).write_text("{}", encoding="utf-8")
    with pytest.raises(v08d.V08DError, match="rerun prohibited"):
        v08d.run_v08d(output_dir=tmp_path)


def test_20_pipeline_source_cannot_invoke_optimizer_or_feature_selector() -> None:
    source = inspect.getsource(v08d)
    campaign_source = inspect.getsource(v08d.evaluate_final_test_campaign)
    assert "BinaryParticleSwarmOptimizer" not in source
    assert "feature_fitness_is_better" not in source
    assert "create_v06_selector" not in source
    assert "selector.fit" not in source
    assert "model.fit(train_x" in campaign_source
    assert "metric_evaluator(test.ground_truth" in campaign_source


def test_21_no_resource_benchmark_or_energy_measurement_path() -> None:
    source = inspect.getsource(v08d.run_v08d)
    assert "run_benchmark_cell" not in source
    assert "run_v07d_experiments" not in source
    assert "ResourceMonitor" not in source
    assert "energy_uj" not in source


def test_22_immutable_snapshot_covers_v06_through_v08c() -> None:
    snapshot = v08d.snapshot_immutable_paths()
    joined = "\n".join(snapshot)
    assert "validation_lock.json" in joined
    assert "v07d_artifact_hashes.json" in joined
    assert "v08a_protocol_validation.json" in joined
    assert "v08b_preflight.json" in joined
    assert "v08c_winner_lock.json" in joined


def test_23_completed_final_test_artifacts_verify_as_one_campaign() -> None:
    v08d.verify_v08d_artifacts()
    execution = v08d._read_json(v08d.EXECUTION_PATH)
    audit = v08d._read_json(v08d.AUDIT_PATH)
    assert execution["status"] == "COMPLETED"
    assert execution["governed_evaluation_count"] == 40
    assert execution["campaign_rerun_count"] == 0
    assert execution["winner_modified_after_test_access"] is False
    assert audit["winner_lock_verified_before_test_access"] is True
    assert audit["rerun_count"] == 0


def test_24_real_result_lock_preserves_exact_v08c_winner_and_both_classifiers() -> None:
    lock = v08d._read_json(v08d.RESULT_LOCK_PATH)
    v08d.verify_final_test_lock(lock)
    assert lock["source_winner_semantic_lock_sha256"] == v08d.EXPECTED_WINNER[
        "semantic_lock_sha256"
    ]
    assert lock["bpso_mask_sha256"] == v08d.EXPECTED_WINNER["mask_sha256"]
    assert lock["bpso_selected_features_sha256"] == v08d.EXPECTED_WINNER[
        "selected_features_sha256"
    ]
    assert lock["preservation_result"] == {
        "decision_tree": True,
        "logistic_regression": True,
    }
