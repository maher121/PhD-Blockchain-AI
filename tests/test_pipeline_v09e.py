"""Synthetic governance and statistics tests for V0.9-E.

These tests exercise the governed one-time final-test campaign against
synthetic train/test workloads and the real frozen V0.9-D / V0.8-C winner
locks. They verify that selection preceded test access, that the test split is
read only for evaluation, that the locked BGWO/BPSO subsets cannot be changed by
test data, and that the paired statistics and result lock behave exactly as
specified.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.pipeline_v09e as v09e
from src.security.experiment_data import fingerprint_feature_names
from tests.test_bpso_fitness import SpyModel, make_context, spy_factory


@pytest.fixture(scope="module")
def real_basis():
    return v09e.load_frozen_basis()


@pytest.fixture(scope="module")
def real_bgwo_lock():
    return v09e.load_verified_bgwo_winner_lock()


@pytest.fixture(scope="module")
def real_bpso_lock():
    return v09e.load_verified_bpso_winner_lock()


@pytest.fixture(scope="module")
def real_preflight(real_bgwo_lock, real_bpso_lock):
    # V0.9-E was committed at its own HEAD; the historical expected checkpoint
    # therefore predates the current commit. Pin the check to the live HEAD so
    # the regression suite stays commit-agnostic without editing frozen source.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(v09e, "EXPECTED_HEAD_SHORT", v09e.current_head_short())
    try:
        return v09e.verify_v09e_preflight(
            expected_head=None, bgwo_lock=real_bgwo_lock, bpso_lock=real_bpso_lock
        )
    finally:
        monkeypatch.undo()


def _plan(
    identifier: str,
    features: tuple[str, ...],
    candidates: tuple[str, ...],
    *,
    source: str,
    scope: str,
    semantics: str,
    source_semantic_lock_sha256: str | None = None,
    per_seed: dict[str, tuple[str, ...]] | None = None,
) -> dict:
    return {
        "configuration_id": identifier,
        "source_configuration_id": source,
        "feature_count": len(features),
        "selection_scope": scope,
        "selection_semantics": semantics,
        "source_semantic_lock_sha256": source_semantic_lock_sha256,
        "features_by_seed": per_seed
        or {str(seed): list(features) for seed in v09e.MODEL_ATTACK_SEEDS},
    }


def _synthetic_plans(context) -> tuple[dict, ...]:
    candidates = context.candidate_features
    bpso_features = candidates[:10]
    bgwo_features = candidates[:14]
    k11 = {
        str(seed): tuple(candidates[offset : offset + 11])
        for offset, seed in enumerate(v09e.MODEL_ATTACK_SEEDS)
    }
    return (
        _plan(
            "K43",
            candidates,
            candidates,
            source="none_natural",
            scope="synthetic full baseline",
            semantics="universal full candidate manifest",
        ),
        _plan(
            "K42",
            candidates[:-1],
            candidates,
            source="pairwise_correlation_filter_natural",
            scope="synthetic correlation baseline",
            semantics="universal deterministic training-derived subset",
        ),
        _plan(
            "MI-K11",
            k11["42"],
            candidates,
            source="mutual_information_select_k_best_k11",
            scope="synthetic historical MI rule",
            semantics="seed-specific supervised training-derived subsets",
            per_seed=k11,
        ),
        _plan(
            "BPSO-K10",
            bpso_features,
            candidates,
            source="v08c_locked_bpso_k10",
            scope="synthetic V0.8-C lock",
            semantics="one universal locked BPSO subset across all seeds",
            source_semantic_lock_sha256="b" * 64,
        ),
        _plan(
            "BGWO",
            bgwo_features,
            candidates,
            source="v09d_locked_bgwo_winner",
            scope="synthetic V0.9-D lock",
            semantics="one universal locked BGWO subset across all seeds",
            source_semantic_lock_sha256="c" * 64,
        ),
    )


def _synthetic_workloads(context):
    workload_map = {
        (seed, "training"): workload.train
        for seed, workload in zip(v09e.MODEL_ATTACK_SEEDS, context.workloads)
    }
    workload_map.update(
        {
            (seed, "inference"): workload.validation
            for seed, workload in zip(v09e.MODEL_ATTACK_SEEDS, context.workloads)
        }
    )
    return SimpleNamespace(get=lambda seed, phase: workload_map[(seed, phase)])


@pytest.fixture()
def synthetic_campaign():
    context = make_context()
    plans = _synthetic_plans(context)
    SpyModel.fit_calls.clear()
    SpyModel.predict_calls.clear()
    rows = v09e.evaluate_final_test_campaign(
        plans,
        v09e.classifier_plan(),
        _synthetic_workloads(context),
        model_factory=spy_factory,
    )
    return rows, plans, context


def _artifact_hashes() -> dict[str, str]:
    return {
        v09e.AUDIT_PATH.name: "5" * 64,
        v09e.RAW_PATH.name: "1" * 64,
        v09e.SUMMARY_PATH.name: "2" * 64,
        v09e.PAIRED_CSV_PATH.name: "3" * 64,
        v09e.PRESERVATION_PATH.name: "4" * 64,
    }


def _preservation_status() -> dict:
    return {
        "results": [
            {"classifier": "decision_tree", "overall_preserved": True},
            {"classifier": "logistic_regression", "overall_preserved": True},
        ]
    }


def _build_real_lock(bgwo_lock, bpso_lock, configurations, **overrides):
    arguments = {
        "starting_head": v09e.EXPECTED_HEAD_SHORT,
        "bgwo_lock": bgwo_lock,
        "bpso_lock": bpso_lock,
        "configurations": configurations,
        "classifiers": v09e.classifier_plan(),
        "test_dataset_identity": {
            "name": v09e.EXPECTED_DATASET,
            "test_rows": v09e.EXPECTED_TEST_ROWS,
            "dimensions": v09e.EXPECTED_DIMENSIONS,
        },
        "artifact_hashes": _artifact_hashes(),
        "preservation": _preservation_status(),
        "governed_evaluation_count": v09e.EXPECTED_EVALUATION_COUNT,
        "started_at": "a",
        "completed_at": "b",
        "wall_time_sec": 1.0,
        "rerun_count": 0,
    }
    arguments.update(overrides)
    return v09e.build_result_lock(**arguments)


def test_01_stage_constants_and_governed_identity() -> None:
    assert v09e.V09E_STAGE == "V0.9-E"
    assert v09e.EXPECTED_HEAD_SHORT == "1007d3f"
    assert v09e.MODEL_ATTACK_SEEDS == (42, 43, 44, 45, 46)
    assert v09e.CLASSIFIERS == ("decision_tree", "logistic_regression")
    assert v09e.CONFIGURATION_IDS == ("K43", "K42", "MI-K11", "BPSO-K10", "BGWO")
    assert v09e.CANDIDATE_COMPARISON_IDS == ("BGWO", "BPSO-K10")
    assert v09e.PRESERVATION_MARGINS == {
        "average_precision": 0.05,
        "f1": 0.05,
        "recall": 0.10,
    }
    assert v09e.EXPECTED_DIMENSIONS == 43
    assert v09e.EXPECTED_TEST_ROWS == 6000


def test_02_expected_evaluation_count_is_exactly_fifty() -> None:
    assert v09e.EXPECTED_EVALUATION_COUNT == 50
    assert (
        v09e.EXPECTED_EVALUATION_COUNT
        == len(v09e.CONFIGURATION_IDS) * len(v09e.CLASSIFIERS) * len(v09e.MODEL_ATTACK_SEEDS)
    )


def test_03_real_bgwo_winner_lock_is_exact(real_bgwo_lock) -> None:
    assert real_bgwo_lock["stage"] == "V0.9-D"
    assert real_bgwo_lock["status"] == "VALIDATION_LOCKED"
    assert real_bgwo_lock["eligible_for_v09e"] is True
    assert real_bgwo_lock["selection_scope"] == "TRAIN_AND_DEVELOPMENT_VALIDATION_ONLY"
    assert real_bgwo_lock["final_test_accessed"] is False
    assert real_bgwo_lock["test_authorized"] is False
    assert real_bgwo_lock["source_optimizer_seed"] == 2042
    assert real_bgwo_lock["selected_feature_count"] == 14
    assert real_bgwo_lock["mask_sha256"] == v09e.EXPECTED_BGWO_WINNER["mask_sha256"]
    assert (
        real_bgwo_lock["selected_features_sha256"]
        == v09e.EXPECTED_BGWO_WINNER["selected_features_sha256"]
    )
    assert (
        real_bgwo_lock["semantic_lock_sha256"]
        == v09e.EXPECTED_BGWO_WINNER["semantic_lock_sha256"]
    )
    assert tuple(real_bgwo_lock["ordered_selected_features"]) == v09e.EXPECTED_BGWO_FEATURES


def test_04_real_bpso_winner_lock_is_exact(real_bpso_lock) -> None:
    assert real_bpso_lock["stage"] == "V0.8-C"
    assert real_bpso_lock["eligible_for_v08d"] is True
    assert real_bpso_lock["source_optimizer_seed"] == 1042
    assert real_bpso_lock["selected_feature_count"] == 10
    assert real_bpso_lock["mask_sha256"] == v09e.EXPECTED_BPSO_WINNER["mask_sha256"]
    assert (
        real_bpso_lock["selected_features_sha256"]
        == v09e.EXPECTED_BPSO_WINNER["selected_features_sha256"]
    )
    assert (
        real_bpso_lock["semantic_lock_sha256"]
        == v09e.EXPECTED_BPSO_WINNER["semantic_lock_sha256"]
    )
    assert tuple(real_bpso_lock["ordered_selected_features"]) == v09e.EXPECTED_BPSO_FEATURES


def test_05_real_configuration_plan_has_exact_governed_semantics(
    real_basis, real_bgwo_lock, real_bpso_lock
) -> None:
    plans = v09e.build_configuration_plan(real_basis, real_bgwo_lock, real_bpso_lock)
    assert tuple(plan["configuration_id"] for plan in plans) == v09e.CONFIGURATION_IDS
    assert [plan["feature_count"] for plan in plans] == [43, 42, 11, 10, 14]
    assert plans[0]["selection_semantics"] == "universal full candidate manifest"
    assert "deterministic" in plans[1]["selection_semantics"]
    assert "seed-specific" in plans[2]["selection_semantics"]
    assert "universal locked" in plans[3]["selection_semantics"]
    assert "universal locked" in plans[4]["selection_semantics"]
    assert plans[3]["source_semantic_lock_sha256"] == real_bpso_lock["semantic_lock_sha256"]
    assert plans[4]["source_semantic_lock_sha256"] == real_bgwo_lock["semantic_lock_sha256"]
    assert tuple(plans[4]["features_by_seed"]["42"]) == v09e.EXPECTED_BGWO_FEATURES
    assert tuple(plans[3]["features_by_seed"]["42"]) == v09e.EXPECTED_BPSO_FEATURES


def test_06_mi_k11_historical_subset_stays_seed_specific(real_basis) -> None:
    plans = v09e.build_configuration_plan(
        real_basis,
        v09e.load_verified_bgwo_winner_lock(),
        v09e.load_verified_bpso_winner_lock(),
    )
    k11 = plans[2]
    sets = {tuple(k11["features_by_seed"][str(seed)]) for seed in v09e.MODEL_ATTACK_SEEDS}
    assert len(sets) == 5
    assert all(len(features) == 11 for features in sets)


def test_07_frozen_classifier_configurations_are_exact() -> None:
    plan = v09e.classifier_plan()
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


def test_08_preflight_passes_and_never_touches_test(real_preflight) -> None:
    assert real_preflight["status"] == "PASS"
    assert real_preflight["starting_head"] == v09e.current_head_short()
    assert real_preflight["checks"]["test_not_accessed_by_preflight"] is True
    assert all(real_preflight["checks"].values())
    assert real_preflight["frozen_immutable_hash_count"] > 0
    assert real_preflight["bgwo_winner_lock_sha256"]
    assert real_preflight["bgwo_winner_mask_sha256"] == v09e.EXPECTED_BGWO_WINNER["mask_sha256"]


def test_09_preflight_rejects_unexpected_starting_checkpoint() -> None:
    with pytest.raises(v09e.V09EError, match="V09E_NO_GO"):
        v09e.verify_v09e_preflight(expected_head="0000000")


def test_10_preflight_rejects_tampered_bgwo_lock(real_basis, real_bgwo_lock, real_bpso_lock) -> None:
    tampered = deepcopy(dict(real_bgwo_lock))
    tampered["mask"] = [1 if index < 5 else 0 for index in range(v09e.EXPECTED_DIMENSIONS)]
    with pytest.raises((v09e.V09EError, v09e.v09d.V09DError)):
        v09e.verify_v09e_preflight(
            expected_head=None,
            basis=real_basis,
            bgwo_lock=tampered,
            bpso_lock=real_bpso_lock,
        )


def test_11_test_access_is_activated_before_workloads(tmp_path, real_preflight, real_bgwo_lock, real_bpso_lock) -> None:
    audit = v09e.activate_test_access_audit(
        tmp_path / "audit.json", real_preflight, real_bgwo_lock, real_bpso_lock
    )
    assert audit["status"] == "TEST_ACCESS_ACTIVATED"
    assert audit["winner_lock_verified_before_test_access"] is True
    assert audit["final_test_accessed"] is True
    assert audit["final_test_opened"] is False
    assert audit["campaign_completed"] is False
    assert audit["test_used_for_evaluation_only"] is True
    assert audit["test_used_for_selection"] is False
    assert audit["test_used_for_tuning"] is False
    assert audit["test_used_for_feature_selection"] is False
    assert audit["checks"]["optimizer_invoked"] is False
    assert audit["checks"]["feature_selection_invoked"] is False
    assert [event["event"] for event in audit["events"]] == [
        "WINNER_LOCK_VERIFIED",
        "TEST_ACCESS_ACTIVATED",
    ]


def test_12_test_loader_refuses_unauthorized_access(real_preflight, real_bgwo_lock, real_bpso_lock, tmp_path) -> None:
    calls: list = []

    def spy_loader(context):
        calls.append(context)
        return "WORKLOADS"

    with pytest.raises(v09e.V09EError, match="inaccessible"):
        v09e.load_authorized_test_workloads(object(), {}, loader=spy_loader)
    assert calls == []

    audit = v09e.activate_test_access_audit(
        tmp_path / "audit.json", real_preflight, real_bgwo_lock, real_bpso_lock
    )
    audit_with_selection = dict(audit)
    audit_with_selection["test_used_for_selection"] = True
    with pytest.raises(v09e.V09EError, match="inaccessible"):
        v09e.load_authorized_test_workloads(object(), audit_with_selection, loader=spy_loader)
    assert calls == []


def test_13_test_loader_runs_only_after_authorization(tmp_path, real_preflight, real_bgwo_lock, real_bpso_lock) -> None:
    calls: list = []
    sentinel = object()

    def spy_loader(context):
        calls.append(context)
        return sentinel

    audit = v09e.activate_test_access_audit(
        tmp_path / "audit.json", real_preflight, real_bgwo_lock, real_bpso_lock
    )
    result = v09e.load_authorized_test_workloads("CONTEXT", audit, loader=spy_loader)
    assert result is sentinel
    assert calls == ["CONTEXT"]


def test_14_audit_completion_records_governed_count(tmp_path, real_preflight, real_bgwo_lock, real_bpso_lock) -> None:
    path = tmp_path / "audit.json"
    audit = v09e.activate_test_access_audit(path, real_preflight, real_bgwo_lock, real_bpso_lock)
    completed = v09e.complete_test_access_audit(
        path, audit, evaluation_count=v09e.EXPECTED_EVALUATION_COUNT
    )
    assert completed["status"] == "PASS"
    assert completed["final_test_opened"] is True
    assert completed["campaign_completed"] is True
    assert completed["governed_evaluation_count"] == 50
    assert completed["winner_modified_after_access"] is False
    assert [event["event"] for event in completed["events"]][-2:] == [
        "FINAL_TEST_OPENED_FOR_EVALUATION",
        "CAMPAIGN_COMPLETED",
    ]


def test_15_synthetic_campaign_has_exactly_fifty_paired_cells(synthetic_campaign) -> None:
    rows, _, _ = synthetic_campaign
    assert len(rows) == 50
    assert {row["configuration_id"] for row in rows} == set(v09e.CONFIGURATION_IDS)
    assert {row["classifier"] for row in rows} == set(v09e.CLASSIFIERS)
    assert {row["seed"] for row in rows} == set(v09e.MODEL_ATTACK_SEEDS)
    assert all(row["stage"] == v09e.V09E_STAGE for row in rows)
    assert all(row["training_split"] == "train" for row in rows)
    assert all(row["evaluation_split"] == "test" for row in rows)
    assert all(row["test_access_authorized"] is True for row in rows)


def test_16_training_only_fit_and_test_only_score(synthetic_campaign) -> None:
    rows, _, _ = synthetic_campaign
    assert len(SpyModel.fit_calls) == len(SpyModel.predict_calls) == 50
    assert all(call["index"] == list(range(100, 106)) for call in SpyModel.fit_calls)
    assert all(call["index"] == list(range(200, 204)) for call in SpyModel.predict_calls)
    assert all(call["columns"] == row["selected_features"] for call, row in zip(SpyModel.fit_calls, rows))
    assert all(call["columns"] == row["selected_features"] for call, row in zip(SpyModel.predict_calls, rows))
    assert all(row["model_fit_on_test"] is False for row in rows)
    assert all(row["preprocessing_fit_on_test"] is False for row in rows)
    assert all(row["feature_selection_on_test"] is False for row in rows)
    assert all(row["threshold_tuned_on_test"] is False for row in rows)
    assert all(row["prediction_threshold"] == 0.5 for row in rows)


def test_17_locked_subsets_cannot_be_changed_by_test_data(synthetic_campaign) -> None:
    rows, plans, _ = synthetic_campaign
    by_id = {plan["configuration_id"]: plan for plan in plans}
    for identifier in v09e.CANDIDATE_COMPARISON_IDS:
        locked = tuple(by_id[identifier]["features_by_seed"]["42"])
        subset = [row for row in rows if row["configuration_id"] == identifier]
        assert len(subset) == 10
        assert all(tuple(row["selected_features"]) == locked for row in subset)
        assert all(
            row["selected_features_sha256"] == fingerprint_feature_names(locked)
            for row in subset
        )


def test_18_aggregation_covers_every_configuration_classifier_metric(synthetic_campaign) -> None:
    summary = v09e.aggregate_final_test(synthetic_campaign[0])
    expected = len(v09e.CONFIGURATION_IDS) * len(v09e.CLASSIFIERS) * len(v09e.METRIC_COLUMNS)
    assert len(summary["summaries"]) == expected
    assert summary["test_driven_selection"] is False
    for row in summary["summaries"]:
        assert row["n"] == 5
        assert {"mean", "std", "min", "max"} <= set(row)
        assert row["min"] <= row["mean"] <= row["max"]


def test_19_paired_comparisons_align_seed_and_zero_interval(synthetic_campaign) -> None:
    paired = v09e.paired_comparisons(synthetic_campaign[0])
    assert len(paired["comparisons"]) == 7 * len(v09e.CLASSIFIERS) * len(v09e.PRIMARY_METRICS)
    assert all(row["mean_difference"] == 0.0 for row in paired["comparisons"])
    assert all(row["ci95_low"] == row["ci95_high"] == 0.0 for row in paired["comparisons"])
    assert all(row["direction_uncertain"] is True for row in paired["comparisons"])
    assert all(row["p_value_reported"] is False for row in paired["comparisons"])
    assert all([pair["seed"] for pair in row["pairs"]] == [42, 43, 44, 45, 46] for row in paired["comparisons"])
    assert paired["confidence_interval"].startswith("two-sided 95% Student-t")
    assert all(row["n"] == 5 and row["df"] == 4 for row in paired["comparisons"])


def test_20_paired_t_interval_math_is_exact(synthetic_campaign) -> None:
    rows = deepcopy(synthetic_campaign[0])
    bgwo_rows = sorted(
        [
            row
            for row in rows
            if row["classifier"] == "decision_tree" and row["configuration_id"] == "BGWO"
        ],
        key=lambda row: row["seed"],
    )
    differences = [0.01, 0.02, 0.03, 0.04, 0.05]
    for row, difference in zip(bgwo_rows, differences):
        row["metrics"]["average_precision"] += difference
    paired = v09e.paired_comparisons(rows)
    result = next(
        row
        for row in paired["comparisons"]
        if row["classifier"] == "decision_tree"
        and row["candidate_configuration_id"] == "BGWO"
        and row["reference_configuration_id"] == "K43"
        and row["metric"] == "average_precision"
    )
    mean = sum(differences) / 5
    std = np.std(differences, ddof=1)
    half = v09e.T_CRITICAL_95_DF4 * std / np.sqrt(5)
    assert result["mean_difference"] == pytest.approx(mean)
    assert result["std_difference"] == pytest.approx(std)
    assert result["ci95_low"] == pytest.approx(mean - half)
    assert result["ci95_high"] == pytest.approx(mean + half)
    assert result["direction_uncertain"] is False
    assert result["pairs"][0]["difference_candidate_minus_reference"] == pytest.approx(0.01)


def test_21_preservation_uses_unchanged_margins_for_both_classifiers(synthetic_campaign) -> None:
    preservation = v09e.preservation_analysis(v09e.aggregate_final_test(synthetic_campaign[0]))
    assert preservation["margins"] == {
        "average_precision": 0.05,
        "f1": 0.05,
        "recall": 0.10,
    }
    assert preservation["domain_validated_margins"] is False
    assert preservation["winner_reselection_performed"] is False
    assert all(row["overall_preserved"] for row in preservation["results"])
    assert preservation["cross_classifier_interpretation"] == (
        "PRESERVATION_HOLDS_FOR_BOTH_CLASSIFIERS"
    )


def test_22_preservation_boundary_and_failure_are_exact(synthetic_campaign) -> None:
    summary = v09e.aggregate_final_test(synthetic_campaign[0])
    for row in summary["summaries"]:
        if row["configuration_id"] == "BGWO" and row["metric"] == "average_precision":
            row["mean"] = 0.95
        if row["configuration_id"] == "BGWO" and row["metric"] == "f1":
            row["mean"] = 0.95
        if row["configuration_id"] == "BGWO" and row["metric"] == "recall":
            row["mean"] = 0.90
    preservation = v09e.preservation_analysis(summary)
    assert all(row["overall_preserved"] for row in preservation["results"])

    target = next(
        row
        for row in summary["summaries"]
        if row["configuration_id"] == "BGWO"
        and row["classifier"] == "decision_tree"
        and row["metric"] == "average_precision"
    )
    target["mean"] = 0.949
    preservation = v09e.preservation_analysis(summary)
    dt = next(row for row in preservation["results"] if row["classifier"] == "decision_tree")
    lr = next(row for row in preservation["results"] if row["classifier"] == "logistic_regression")
    assert dt["overall_preserved"] is False
    assert lr["overall_preserved"] is True
    assert preservation["cross_classifier_interpretation"] == (
        "PRESERVATION_DOES_NOT_HOLD_FOR_BOTH_CLASSIFIERS"
    )
    assert preservation["winner_reselection_performed"] is False

    lr_only = v09e.aggregate_final_test(synthetic_campaign[0])
    lr_target = next(
        row
        for row in lr_only["summaries"]
        if row["configuration_id"] == "BGWO"
        and row["classifier"] == "logistic_regression"
        and row["metric"] == "average_precision"
    )
    lr_target["mean"] = 0.949
    interpretation = v09e.preservation_analysis(lr_only)["cross_classifier_interpretation"]
    assert interpretation == (
        "PRESERVATION_HOLDS_FOR_DT_ONLY_SEARCH_CLASSIFIER_DEPENDENCE_POSSIBLE"
    )


def test_23_feature_overlap_is_descriptive_only(real_basis, real_bgwo_lock, real_bpso_lock) -> None:
    overlap = v09e.feature_overlap_analysis(real_bgwo_lock, real_bpso_lock, real_basis)
    assert overlap["descriptive_only"] is True
    assert overlap["mutated_any_subset"] is False
    assert overlap["bgwo_feature_count"] == 14
    assert overlap["bpso_feature_count"] == 10
    assert overlap["bgwo_vs_bpso"]["intersection_size"] == 3
    assert overlap["bgwo_vs_bpso"]["union_size"] == 21
    assert set(overlap["bgwo_vs_bpso"]["shared_features"]) == {
        "order_item_quantity",
        "order_item_total",
        "Market_LATAM",
    }
    assert len(overlap["bgwo_vs_mi_k11"]["per_seed"]) == 5
    assert 0.0 <= overlap["bgwo_vs_mi_k11"]["union_jaccard"] <= 1.0


def test_24_result_lock_semantic_hash_ignores_timestamps(
    real_bgwo_lock, real_bpso_lock, real_basis
) -> None:
    configurations = v09e.build_configuration_plan(real_basis, real_bgwo_lock, real_bpso_lock)
    first = _build_real_lock(real_bgwo_lock, real_bpso_lock, configurations)
    second = _build_real_lock(
        real_bgwo_lock,
        real_bpso_lock,
        configurations,
        started_at="later",
        completed_at="even-later",
        wall_time_sec=999.0,
    )
    assert first["semantic_result_lock_sha256"] == second["semantic_result_lock_sha256"]
    v09e.verify_result_lock(first)
    v09e.verify_result_lock(second)


def test_25_result_lock_contains_winner_immutability_and_no_resource_claims(
    real_bgwo_lock, real_bpso_lock, real_basis
) -> None:
    configurations = v09e.build_configuration_plan(real_basis, real_bgwo_lock, real_bpso_lock)
    lock = _build_real_lock(real_bgwo_lock, real_bpso_lock, configurations)
    assert lock["stage"] == "V0.9-E"
    assert lock["status"] == "FINAL_TEST_EVALUATED"
    assert lock["test_used_for_evaluation_only"] is True
    assert lock["test_used_for_selection"] is False
    assert lock["test_used_for_tuning"] is False
    assert lock["test_used_for_feature_selection"] is False
    assert lock["optimizer_invoked"] is False
    assert lock["feature_reselection_performed"] is False
    assert lock["test_driven_tuning_performed"] is False
    assert lock["threshold_tuning_performed"] is False
    assert lock["winner_replaced_after_test"] is False
    assert lock["bpso_rerun"] is False
    assert lock["bgwo_rerun"] is False
    assert lock["resource_benchmark_executed"] is False
    assert lock["direct_energy_measured"] is False
    assert "not modified" in lock["winner_unchanged_statement"]
    assert lock["test_access_audit_sha256"] == _artifact_hashes()[v09e.AUDIT_PATH.name]


def test_26_result_lock_tamper_is_rejected(real_bgwo_lock, real_bpso_lock, real_basis) -> None:
    configurations = v09e.build_configuration_plan(real_basis, real_bgwo_lock, real_bpso_lock)
    lock = _build_real_lock(real_bgwo_lock, real_bpso_lock, configurations)
    tampered = deepcopy(lock)
    tampered["bgwo_mask_sha256"] = "0" * 64
    with pytest.raises(v09e.V09EError, match="verification failed"):
        v09e.verify_result_lock(tampered)
    tampered_semantic = deepcopy(lock)
    tampered_semantic["winner_replaced_after_test"] = True
    with pytest.raises(v09e.V09EError, match="verification failed"):
        v09e.verify_result_lock(tampered_semantic)


def test_27_existing_result_lock_prohibits_campaign_rerun(tmp_path) -> None:
    (tmp_path / v09e.RESULT_LOCK_PATH.name).write_text("{}", encoding="utf-8")
    with pytest.raises(v09e.V09EError, match="already locked"):
        v09e.run_v09e(output_dir=tmp_path)


def test_28_leakage_audit_passes_and_fails_on_winner_drift(real_bgwo_lock) -> None:
    audit = {
        "winner_lock_verified_before_test_access": True,
        "winner_lock_verified_at_utc": "2026-01-01T00:00:00+00:00",
        "test_access_activated_at_utc": "2026-01-01T00:00:01+00:00",
        "campaign_completed_at_utc": "2026-01-01T00:01:00+00:00",
        "rerun_count": 0,
        "bpso_source_semantic_lock_sha256": v09e.EXPECTED_BPSO_WINNER["semantic_lock_sha256"],
    }
    before = deepcopy(dict(real_bgwo_lock))
    after = deepcopy(dict(real_bgwo_lock))
    result = v09e.build_test_leakage_audit(
        bgwo_lock_before=before,
        bgwo_lock_after=after,
        bgwo_lock_sha256_before="a" * 64,
        bgwo_lock_sha256_after="a" * 64,
        bpso_lock_sha256_before="b" * 64,
        bpso_lock_sha256_after="b" * 64,
        audit=audit,
        starting_head=v09e.EXPECTED_HEAD_SHORT,
        ending_head=v09e.EXPECTED_HEAD_SHORT,
        evaluation_count=50,
    )
    assert result["status"] == "PASS"
    assert result["final_test_accessed"] is True
    assert result["final_test_opened"] is True
    assert result["campaign_completed"] is True
    assert result["winner_lock_verified_before_test_access"] is True
    assert result["rerun_count"] == 0
    assert result["test_used_for_selection"] is False
    assert result["checks"]["governed_evaluation_count_is_fifty"] is True

    drifted = deepcopy(after)
    drifted["mask"][0] = 1 - int(drifted["mask"][0])
    with pytest.raises(v09e.V09EError, match="leakage audit failed"):
        v09e.build_test_leakage_audit(
            bgwo_lock_before=before,
            bgwo_lock_after=drifted,
            bgwo_lock_sha256_before="a" * 64,
            bgwo_lock_sha256_after="c" * 64,
            bpso_lock_sha256_before="b" * 64,
            bpso_lock_sha256_after="b" * 64,
            audit=audit,
            starting_head=v09e.EXPECTED_HEAD_SHORT,
            ending_head=v09e.EXPECTED_HEAD_SHORT,
            evaluation_count=50,
        )


def test_29_pipeline_source_cannot_invoke_optimizer_or_selector() -> None:
    source = inspect.getsource(v09e)
    campaign_source = inspect.getsource(v09e.evaluate_final_test_campaign)
    assert "BinaryParticleSwarmOptimizer" not in source
    assert "feature_fitness_is_better" not in source
    assert "create_v06_selector" not in source
    assert "selector.fit" not in source
    assert "select_search_winner" not in source
    assert "grid_search" not in source
    assert "run_v08c(" not in source
    assert "run_v08d(" not in source
    assert "run_v09d(" not in source
    assert "model.fit(train_x" in campaign_source
    assert "metric_evaluator(test.ground_truth" in campaign_source


def test_30_immutable_snapshot_covers_v06_through_v09d() -> None:
    snapshot = v09e.snapshot_immutable_paths()
    joined = "\n".join(snapshot)
    assert "validation_lock.json" in joined
    assert "v08c_winner_lock.json" in joined
    assert "v09d_winner_lock.json" in joined
    assert "v09d_run_2042.json" in joined
    assert "v09d_validation_comparison.json" in joined
    assert len(snapshot) >= len(v09e.v08d.IMMUTABLE_PATHS)


def test_31_completed_final_test_artifacts_verify_as_one_campaign() -> None:
    if not v09e.RESULT_LOCK_PATH.exists():
        pytest.skip("V0.9-E production campaign has not been executed in this checkout yet.")
    v09e.verify_v09e_artifacts()
    execution = v09e._read_json(v09e.EXECUTION_PATH)
    audit = v09e._read_json(v09e.AUDIT_PATH)
    lock = v09e._read_json(v09e.RESULT_LOCK_PATH)
    assert execution["status"] == "COMPLETED"
    assert execution["governed_evaluation_count"] == v09e.EXPECTED_EVALUATION_COUNT
    assert execution["winner_modified_after_test_access"] is False
    assert execution["readiness_for_v09f"] == "GO"
    assert audit["status"] == "PASS"
    assert audit["winner_lock_verified_before_test_access"] is True
    assert audit["rerun_count"] == 0
    assert lock["bgwo_mask_sha256"] == v09e.EXPECTED_BGWO_WINNER["mask_sha256"]
    assert lock["semantic_result_lock_sha256"] == execution["semantic_result_lock_sha256"]
