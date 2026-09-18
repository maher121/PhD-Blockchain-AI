"""Synthetic governance and statistics tests for V1.0-E.

These tests exercise the governed one-time final-test campaign against synthetic
train/test workloads and the real frozen V1.0-D hybrid / V0.8-C BPSO / V0.9-D
BGWO winner locks. They verify that selection preceded test access, that test is
read only for evaluation, that the locked Hybrid/BPSO/BGWO subsets cannot be
changed by test data, and that paired statistics, preservation, reduction,
feature overlap, immutability, and the result lock behave exactly as specified.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import src.pipeline_v10e as v10e
from src.security.experiment_data import fingerprint_feature_names
from tests.test_bpso_fitness import SpyModel, make_context, spy_factory


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
    return v10e.verify_v10e_preflight(
        expected_checkpoint=None,
        hybrid_lock=real_hybrid_lock,
        bpso_lock=real_bpso_lock,
        bgwo_lock=real_bgwo_lock,
    )


def _plan(
    identifier: str,
    features: tuple[str, ...],
    *,
    source: str,
    scope: str,
    semantics: str,
    source_semantic_lock_sha256: str | None = None,
    per_seed: dict[str, tuple[str, ...]] | None = None,
    feature_count: int | None = None,
) -> dict:
    return {
        "configuration_id": identifier,
        "source_configuration_id": source,
        "feature_count": feature_count if feature_count is not None else len(features),
        "selection_scope": scope,
        "selection_semantics": semantics,
        "source_semantic_lock_sha256": source_semantic_lock_sha256,
        "features_by_seed": per_seed
        or {str(seed): list(features) for seed in v10e.MODEL_ATTACK_SEEDS},
    }


def _synthetic_plans(context) -> tuple[dict, ...]:
    candidates = context.candidate_features
    k11 = {
        str(seed): tuple(candidates[offset : offset + 11])
        for offset, seed in enumerate(v10e.MODEL_ATTACK_SEEDS)
    }
    return (
        _plan(
            "K43",
            candidates,
            source="none_natural",
            scope="synthetic full baseline",
            semantics="universal full candidate manifest",
            feature_count=43,
        ),
        _plan(
            "K42",
            candidates[:-1],
            source="pairwise_correlation_filter_natural",
            scope="synthetic correlation baseline",
            semantics="universal deterministic training-derived subset",
            feature_count=42,
        ),
        _plan(
            "MI-K11",
            k11["42"],
            source="mutual_information_select_k_best_k11",
            scope="synthetic historical MI rule",
            semantics="seed-specific supervised training-derived subsets",
            per_seed=k11,
            feature_count=11,
        ),
        _plan(
            "BPSO-K10",
            candidates[:10],
            source="v08c_locked_bpso_k10",
            scope="synthetic V0.8-C lock",
            semantics="one universal locked BPSO subset across all seeds",
            source_semantic_lock_sha256="b" * 64,
            feature_count=10,
        ),
        _plan(
            "BGWO-K14",
            candidates[:14],
            source="v09d_locked_bgwo_winner",
            scope="synthetic V0.9-D lock",
            semantics="one universal locked BGWO subset across all seeds",
            source_semantic_lock_sha256="c" * 64,
            feature_count=14,
        ),
        _plan(
            "HYBRID-K13",
            candidates[:13],
            source="v10d_locked_hybrid_winner",
            scope="synthetic V1.0-D lock",
            semantics="one universal locked Hybrid subset across all seeds",
            source_semantic_lock_sha256="d" * 64,
            feature_count=13,
        ),
    )


def _synthetic_workloads(context):
    workload_map = {
        (seed, "training"): workload.train
        for seed, workload in zip(v10e.MODEL_ATTACK_SEEDS, context.workloads)
    }
    workload_map.update(
        {
            (seed, "inference"): workload.validation
            for seed, workload in zip(v10e.MODEL_ATTACK_SEEDS, context.workloads)
        }
    )
    return SimpleNamespace(get=lambda seed, phase: workload_map[(seed, phase)])


@pytest.fixture()
def synthetic_campaign():
    context = make_context()
    plans = _synthetic_plans(context)
    SpyModel.fit_calls.clear()
    SpyModel.predict_calls.clear()
    rows = v10e.evaluate_final_test_campaign(
        plans,
        v10e.classifier_plan(),
        _synthetic_workloads(context),
        model_factory=spy_factory,
    )
    return rows, plans, context


def _artifact_hashes() -> dict[str, str]:
    return {
        v10e.AUDIT_PATH.name: "5" * 64,
        v10e.RAW_CSV_PATH.name: "1" * 64,
        v10e.SUMMARY_CSV_PATH.name: "2" * 64,
        v10e.PAIRED_CSV_PATH.name: "3" * 64,
        v10e.PRESERVATION_PATH.name: "4" * 64,
    }


def _preservation_status() -> dict:
    return {
        "results": [
            {"classifier": "decision_tree", "overall_preserved": True},
            {"classifier": "logistic_regression", "overall_preserved": True},
        ]
    }


def _build_real_lock(hybrid_lock, bpso_lock, bgwo_lock, configurations, **overrides):
    arguments = {
        "starting_head": v10e.STARTING_CHECKPOINT,
        "hybrid_lock": hybrid_lock,
        "bpso_lock": bpso_lock,
        "bgwo_lock": bgwo_lock,
        "configurations": configurations,
        "classifiers": v10e.classifier_plan(),
        "test_dataset_identity": {
            "name": v10e.EXPECTED_DATASET,
            "test_rows": v10e.EXPECTED_TEST_ROWS,
            "dimensions": v10e.EXPECTED_DIMENSIONS,
        },
        "artifact_hashes": _artifact_hashes(),
        "preservation": _preservation_status(),
        "governed_evaluation_count": v10e.EXPECTED_EVALUATION_COUNT,
        "started_at": "a",
        "completed_at": "b",
        "wall_time_sec": 1.0,
        "rerun_count": 0,
    }
    arguments.update(overrides)
    return v10e.build_result_lock(**arguments)


# ---------------------------------------------------------------------------
# Constants and governed identity
# ---------------------------------------------------------------------------


def test_01_stage_constants_and_governed_identity() -> None:
    assert v10e.V10E_STAGE == "V1.0-E"
    assert v10e.STARTING_CHECKPOINT == "e7465e9"
    assert v10e.MODEL_ATTACK_SEEDS == (42, 43, 44, 45, 46)
    assert v10e.CLASSIFIERS == ("decision_tree", "logistic_regression")
    assert v10e.CONFIGURATION_IDS == (
        "K43",
        "K42",
        "MI-K11",
        "BPSO-K10",
        "BGWO-K14",
        "HYBRID-K13",
    )
    assert v10e.CANDIDATE_COMPARISON_IDS == ("HYBRID-K13",)
    assert v10e.CONFIGURATION_FEATURE_COUNTS == (43, 42, 11, 10, 14, 13)
    assert v10e.PRESERVATION_MARGINS == {
        "average_precision": 0.05,
        "f1": 0.05,
        "recall": 0.10,
    }
    assert v10e.EXPECTED_DIMENSIONS == 43
    assert v10e.EXPECTED_TEST_ROWS == 6000


def test_02_expected_evaluation_count_is_exactly_sixty() -> None:
    assert v10e.EXPECTED_EVALUATION_COUNT == 60
    assert (
        v10e.EXPECTED_EVALUATION_COUNT
        == len(v10e.CONFIGURATION_IDS) * len(v10e.CLASSIFIERS) * len(v10e.MODEL_ATTACK_SEEDS)
    )


def test_03_paired_comparison_count_is_exactly_fifty() -> None:
    assert v10e.EXPECTED_PAIRED_COMPARISON_COUNT == 50


def test_04_test_access_classification_and_reason() -> None:
    assert (
        v10e.TEST_ACCESS_CLASSIFICATION
        == "TEST_USED_ONLY_AFTER_LOCK_FOR_FINAL_EVALUATION"
    )
    assert v10e.TEST_AUTHORIZATION_REASON == "POST_WINNER_LOCK_FINAL_TEST_EVALUATION_ONLY"
    assert v10e.PROVENANCE_POLICY == "HEAD_AGNOSTIC_ANCESTRY"


# ---------------------------------------------------------------------------
# Frozen winner locks
# ---------------------------------------------------------------------------


def test_05_real_hybrid_winner_lock_is_exact(real_hybrid_lock) -> None:
    assert real_hybrid_lock["stage"] == "V1.0-D"
    assert real_hybrid_lock["selection_scope"] == "TRAIN_AND_DEVELOPMENT_VALIDATION_ONLY"
    assert real_hybrid_lock["test_accessed"] is False
    assert real_hybrid_lock["test_used_for_winner_selection"] is False
    assert real_hybrid_lock["final_test_evaluated"] is False
    assert real_hybrid_lock["optimizer_seed"] == 3045
    assert real_hybrid_lock["selected_feature_count"] == 13
    assert real_hybrid_lock["mask_sha256"] == v10e.EXPECTED_HYBRID_WINNER["mask_sha256"]
    assert (
        real_hybrid_lock["feature_list_sha256"]
        == v10e.EXPECTED_HYBRID_WINNER["feature_list_sha256"]
    )
    assert (
        real_hybrid_lock["feature_manifest_sha256"]
        == v10e.EXPECTED_HYBRID_WINNER["feature_manifest_sha256"]
    )
    assert (
        real_hybrid_lock["semantic_lock_sha256"]
        == v10e.EXPECTED_HYBRID_WINNER["semantic_lock_sha256"]
    )
    assert tuple(real_hybrid_lock["ordered_selected_features"]) == v10e.EXPECTED_HYBRID_FEATURES


def test_06_hybrid_mask_recomputes_from_ordered_features(real_hybrid_lock, real_basis) -> None:
    selected = set(real_hybrid_lock["ordered_selected_features"])
    mask = np.asarray(
        [feature in selected for feature in real_basis.candidate_features], dtype=np.uint8
    )
    assert v10e.v10d._mask_sha256(mask) == real_hybrid_lock["mask_sha256"]
    assert int(mask.sum()) == 13
    assert mask.shape == (43,)


def test_07_real_reference_locks_are_exact(real_bpso_lock, real_bgwo_lock) -> None:
    assert real_bpso_lock["stage"] == "V0.8-C"
    assert real_bpso_lock["selected_feature_count"] == 10
    assert real_bpso_lock["mask_sha256"] == v10e.v09e.EXPECTED_BPSO_WINNER["mask_sha256"]
    assert real_bgwo_lock["stage"] == "V0.9-D"
    assert real_bgwo_lock["selected_feature_count"] == 14
    assert real_bgwo_lock["mask_sha256"] == v10e.v09e.EXPECTED_BGWO_WINNER["mask_sha256"]


# ---------------------------------------------------------------------------
# Configuration plan
# ---------------------------------------------------------------------------


def test_08_real_configuration_plan_has_exact_governed_semantics(
    real_basis, real_hybrid_lock, real_bpso_lock, real_bgwo_lock
) -> None:
    plans = v10e.build_configuration_plan(
        real_basis, real_hybrid_lock, real_bpso_lock, real_bgwo_lock
    )
    assert tuple(plan["configuration_id"] for plan in plans) == v10e.CONFIGURATION_IDS
    assert [plan["feature_count"] for plan in plans] == [43, 42, 11, 10, 14, 13]
    assert plans[0]["selection_semantics"] == "universal full candidate manifest"
    assert "deterministic" in plans[1]["selection_semantics"]
    assert "seed-specific" in plans[2]["selection_semantics"]
    assert "universal locked" in plans[3]["selection_semantics"]
    assert "universal locked" in plans[4]["selection_semantics"]
    assert "universal locked" in plans[5]["selection_semantics"]
    assert plans[3]["source_semantic_lock_sha256"] == real_bpso_lock["semantic_lock_sha256"]
    assert plans[4]["source_semantic_lock_sha256"] == real_bgwo_lock["semantic_lock_sha256"]
    assert (
        plans[5]["source_semantic_lock_sha256"] == real_hybrid_lock["semantic_lock_sha256"]
    )
    assert tuple(plans[5]["features_by_seed"]["42"]) == v10e.EXPECTED_HYBRID_FEATURES


def test_09_mi_k11_historical_subset_stays_seed_specific(real_basis) -> None:
    plans = v10e.build_configuration_plan(
        real_basis,
        v10e.load_verified_hybrid_winner_lock(),
        v10e.load_verified_bpso_winner_lock(),
        v10e.load_verified_bgwo_winner_lock(),
    )
    k11 = plans[2]
    sets = {tuple(k11["features_by_seed"][str(seed)]) for seed in v10e.MODEL_ATTACK_SEEDS}
    assert len(sets) == 5
    assert all(len(features) == 11 for features in sets)


def test_10_frozen_classifier_configurations_are_exact() -> None:
    plan = v10e.classifier_plan()
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


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def test_11_preflight_passes_and_never_touches_test(real_preflight) -> None:
    assert real_preflight["status"] == "PASS"
    assert real_preflight["checks"]["test_not_accessed_by_preflight"] is True
    assert all(real_preflight["checks"].values())
    assert real_preflight["provenance_policy"] == "HEAD_AGNOSTIC_ANCESTRY"
    assert real_preflight["starting_checkpoint"] == "e7465e9"
    assert real_preflight["frozen_immutable_hash_count"] > 0
    assert (
        real_preflight["hybrid_winner_mask_sha256"]
        == v10e.EXPECTED_HYBRID_WINNER["mask_sha256"]
    )


def test_12_preflight_rejects_unexpected_starting_checkpoint() -> None:
    with pytest.raises(v10e.V10EError, match="V10E_NO_GO"):
        v10e.verify_v10e_preflight(expected_checkpoint="0000000")


def test_13_preflight_rejects_tampered_hybrid_lock(
    real_basis, real_hybrid_lock, real_bpso_lock, real_bgwo_lock
) -> None:
    tampered = deepcopy(dict(real_hybrid_lock))
    tampered["mask"] = [1 if index < 5 else 0 for index in range(v10e.EXPECTED_DIMENSIONS)]
    with pytest.raises(v10e.V10EError):
        v10e.verify_v10e_preflight(
            expected_checkpoint=None,
            basis=real_basis,
            hybrid_lock=tampered,
            bpso_lock=real_bpso_lock,
            bgwo_lock=real_bgwo_lock,
        )


def test_14_preflight_rejects_tampered_winner_features(
    real_basis, real_hybrid_lock, real_bpso_lock, real_bgwo_lock
) -> None:
    tampered = deepcopy(dict(real_hybrid_lock))
    tampered["ordered_selected_features"] = list(v10e.EXPECTED_HYBRID_FEATURES[::-1])
    with pytest.raises(v10e.V10EError):
        v10e.verify_v10e_preflight(
            expected_checkpoint=None,
            basis=real_basis,
            hybrid_lock=tampered,
            bpso_lock=real_bpso_lock,
            bgwo_lock=real_bgwo_lock,
        )


# ---------------------------------------------------------------------------
# Authorization gate
# ---------------------------------------------------------------------------


def test_15_test_authorization_activated_before_workloads(
    tmp_path, real_preflight, real_hybrid_lock
) -> None:
    authorization = v10e.activate_test_authorization(
        tmp_path / "authorization.json", real_preflight, real_hybrid_lock
    )
    assert authorization["status"] == "TEST_AUTHORIZED"
    assert authorization["test_authorized_before"] is False
    assert authorization["test_authorized"] is True
    assert authorization["test_access_activated_at_utc"]
    assert authorization["winner_lock_verified_before_test_access"] is True
    assert authorization["final_test_accessed"] is False
    assert authorization["test_used_for_evaluation_only"] is True
    assert authorization["test_used_for_selection"] is False
    assert authorization["test_used_for_tuning"] is False
    assert authorization["test_used_for_feature_selection"] is False
    assert authorization["test_used_for_winner_selection"] is False
    assert authorization["checks"]["optimizer_invoked"] is False
    assert authorization["checks"]["feature_selection_invoked"] is False
    assert authorization["checks"]["final_test_used_only_for_evaluation"] is True


def test_16_loader_refuses_unauthorized_access(
    real_preflight, real_hybrid_lock, tmp_path
) -> None:
    calls: list = []

    def spy_loader(context):
        calls.append(context)
        return "WORKLOADS"

    with pytest.raises(v10e.V10EError, match="inaccessible"):
        v10e.load_authorized_test_workloads(object(), {}, loader=spy_loader)
    assert calls == []

    authorization = v10e.activate_test_authorization(
        tmp_path / "authorization.json", real_preflight, real_hybrid_lock
    )
    unauthorized = dict(authorization)
    unauthorized["test_used_for_selection"] = True
    with pytest.raises(v10e.V10EError, match="inaccessible"):
        v10e.load_authorized_test_workloads(object(), unauthorized, loader=spy_loader)
    assert calls == []


def test_17_loader_runs_only_after_authorization(
    tmp_path, real_preflight, real_hybrid_lock
) -> None:
    calls: list = []
    sentinel = object()

    def spy_loader(context):
        calls.append(context)
        return sentinel

    authorization = v10e.activate_test_authorization(
        tmp_path / "authorization.json", real_preflight, real_hybrid_lock
    )
    result = v10e.load_authorized_test_workloads("CONTEXT", authorization, loader=spy_loader)
    assert result is sentinel
    assert calls == ["CONTEXT"]


def test_18_authorization_completion_records_governed_count(
    tmp_path, real_preflight, real_hybrid_lock
) -> None:
    path = tmp_path / "authorization.json"
    authorization = v10e.activate_test_authorization(path, real_preflight, real_hybrid_lock)
    completed = v10e.complete_test_authorization(
        path, authorization, evaluation_count=v10e.EXPECTED_EVALUATION_COUNT
    )
    assert completed["status"] == "PASS"
    assert completed["final_test_accessed"] is True
    assert completed["campaign_completed"] is True
    assert completed["governed_evaluation_count"] == 60
    assert [event["event"] for event in completed["events"]][-2:] == [
        "FINAL_TEST_OPENED_FOR_EVALUATION",
        "CAMPAIGN_COMPLETED",
    ]


def test_19_authorization_prohibits_rerun_when_locked(tmp_path, real_hybrid_lock) -> None:
    (tmp_path / v10e.RESULT_LOCK_PATH.name).write_text("{}", encoding="utf-8")
    with pytest.raises(v10e.V10EError, match="already exists"):
        v10e.activate_test_authorization(
            tmp_path / "authorization.json", {"starting_head": "x"}, real_hybrid_lock
        )


# ---------------------------------------------------------------------------
# Synthetic evaluation
# ---------------------------------------------------------------------------


def test_20_synthetic_campaign_has_exactly_sixty_paired_cells(synthetic_campaign) -> None:
    rows, _, _ = synthetic_campaign
    assert len(rows) == 60
    assert {row["configuration_id"] for row in rows} == set(v10e.CONFIGURATION_IDS)
    assert {row["classifier"] for row in rows} == set(v10e.CLASSIFIERS)
    assert {row["seed"] for row in rows} == set(v10e.MODEL_ATTACK_SEEDS)
    assert all(row["stage"] == v10e.V10E_STAGE for row in rows)
    assert all(row["training_split"] == "train" for row in rows)
    assert all(row["evaluation_split"] == "test" for row in rows)
    assert all(row["test_access_authorized"] is True for row in rows)


def test_21_training_only_fit_and_test_only_score(synthetic_campaign) -> None:
    rows, _, _ = synthetic_campaign
    assert len(SpyModel.fit_calls) == len(SpyModel.predict_calls) == 60
    assert all(call["index"] == list(range(100, 106)) for call in SpyModel.fit_calls)
    assert all(call["index"] == list(range(200, 204)) for call in SpyModel.predict_calls)
    assert all(
        call["columns"] == row["selected_features"]
        for call, row in zip(SpyModel.fit_calls, rows)
    )
    assert all(row["model_fit_on_test"] is False for row in rows)
    assert all(row["preprocessing_fit_on_test"] is False for row in rows)
    assert all(row["feature_selection_on_test"] is False for row in rows)
    assert all(row["threshold_tuned_on_test"] is False for row in rows)
    assert all(row["prediction_threshold"] == 0.5 for row in rows)


def test_22_locked_subsets_cannot_be_changed_by_test_data(synthetic_campaign) -> None:
    rows, plans, _ = synthetic_campaign
    by_id = {plan["configuration_id"]: plan for plan in plans}
    for identifier in ("BPSO-K10", "BGWO-K14", "HYBRID-K13"):
        locked = tuple(by_id[identifier]["features_by_seed"]["42"])
        subset = [row for row in rows if row["configuration_id"] == identifier]
        assert len(subset) == 10
        assert all(tuple(row["selected_features"]) == locked for row in subset)
        assert all(
            row["selected_features_sha256"] == fingerprint_feature_names(locked)
            for row in subset
        )


def test_23_aggregation_covers_every_configuration_classifier_metric(
    synthetic_campaign,
) -> None:
    summary = v10e.aggregate_final_test(synthetic_campaign[0])
    expected = len(v10e.CONFIGURATION_IDS) * len(v10e.CLASSIFIERS) * len(v10e.METRIC_COLUMNS)
    assert len(summary["summaries"]) == expected
    assert summary["test_driven_selection"] is False
    for row in summary["summaries"]:
        assert row["n"] == 5
        assert {"mean", "std", "min", "max"} <= set(row)
        assert row["min"] <= row["mean"] <= row["max"]


def test_24_paired_comparisons_align_seed_and_zero_interval(synthetic_campaign) -> None:
    paired = v10e.paired_comparisons(synthetic_campaign[0])
    assert len(paired["comparisons"]) == 50
    assert all(row["candidate_configuration_id"] == "HYBRID-K13" for row in paired["comparisons"])
    assert all(row["mean_difference"] == 0.0 for row in paired["comparisons"])
    assert all(row["ci95_low"] == row["ci95_high"] == 0.0 for row in paired["comparisons"])
    assert all(row["direction_uncertain"] is True for row in paired["comparisons"])
    assert all(row["p_value_reported"] is False for row in paired["comparisons"])
    assert all(
        [pair["seed"] for pair in row["pairs"]] == [42, 43, 44, 45, 46]
        for row in paired["comparisons"]
    )
    assert paired["confidence_interval"].startswith("two-sided 95% Student-t")
    assert all(row["n"] == 5 and row["df"] == 4 for row in paired["comparisons"])


def test_25_paired_t_interval_math_is_exact(synthetic_campaign) -> None:
    rows = deepcopy(synthetic_campaign[0])
    hybrid_rows = sorted(
        [
            row
            for row in rows
            if row["classifier"] == "decision_tree"
            and row["configuration_id"] == "HYBRID-K13"
        ],
        key=lambda row: row["seed"],
    )
    differences = [0.01, 0.02, 0.03, 0.04, 0.05]
    for row, difference in zip(hybrid_rows, differences):
        row["metrics"]["average_precision"] += difference
    paired = v10e.paired_comparisons(rows)
    result = next(
        row
        for row in paired["comparisons"]
        if row["classifier"] == "decision_tree"
        and row["candidate_configuration_id"] == "HYBRID-K13"
        and row["reference_configuration_id"] == "K43"
        and row["metric"] == "average_precision"
    )
    mean = sum(differences) / 5
    std = np.std(differences, ddof=1)
    half = v10e.T_CRITICAL_95_DF4 * std / np.sqrt(5)
    assert result["mean_difference"] == pytest.approx(mean)
    assert result["std_difference"] == pytest.approx(std)
    assert result["ci95_low"] == pytest.approx(mean - half)
    assert result["ci95_high"] == pytest.approx(mean + half)
    assert result["direction_uncertain"] is False
    assert result["pairs"][0]["difference_candidate_minus_reference"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Preservation and reduction
# ---------------------------------------------------------------------------


def test_26_preservation_uses_unchanged_margins_for_both_classifiers(
    synthetic_campaign,
) -> None:
    preservation = v10e.preservation_analysis(
        v10e.aggregate_final_test(synthetic_campaign[0])
    )
    assert preservation["margins"] == {
        "average_precision": 0.05,
        "f1": 0.05,
        "recall": 0.10,
    }
    assert preservation["reference_configuration_id"] == "K43"
    assert preservation["candidate_configuration_id"] == "HYBRID-K13"
    assert preservation["domain_validated_margins"] is False
    assert preservation["winner_reselection_performed"] is False
    assert all(row["overall_preserved"] for row in preservation["results"])
    assert preservation["cross_classifier_interpretation"] == (
        "PRESERVATION_HOLDS_FOR_BOTH_CLASSIFIERS"
    )


def test_27_preservation_boundary_and_failure_are_exact(synthetic_campaign) -> None:
    summary = v10e.aggregate_final_test(synthetic_campaign[0])
    for row in summary["summaries"]:
        if row["configuration_id"] == "HYBRID-K13" and row["metric"] == "average_precision":
            row["mean"] = 0.95
        if row["configuration_id"] == "HYBRID-K13" and row["metric"] == "f1":
            row["mean"] = 0.95
        if row["configuration_id"] == "HYBRID-K13" and row["metric"] == "recall":
            row["mean"] = 0.90
    preservation = v10e.preservation_analysis(summary)
    assert all(row["overall_preserved"] for row in preservation["results"])

    target = next(
        row
        for row in summary["summaries"]
        if row["configuration_id"] == "HYBRID-K13"
        and row["classifier"] == "decision_tree"
        and row["metric"] == "average_precision"
    )
    target["mean"] = 0.949
    preservation = v10e.preservation_analysis(summary)
    dt = next(row for row in preservation["results"] if row["classifier"] == "decision_tree")
    lr = next(
        row for row in preservation["results"] if row["classifier"] == "logistic_regression"
    )
    assert dt["overall_preserved"] is False
    assert lr["overall_preserved"] is True
    assert preservation["cross_classifier_interpretation"] == (
        "PRESERVATION_DOES_NOT_HOLD_FOR_BOTH_CLASSIFIERS"
    )


def test_28_feature_reduction_is_exact(real_basis, real_hybrid_lock, real_bpso_lock, real_bgwo_lock) -> None:
    plans = v10e.build_configuration_plan(
        real_basis, real_hybrid_lock, real_bpso_lock, real_bgwo_lock
    )
    reduction = v10e.feature_reduction_analysis(plans)
    by_id = {row["configuration_id"]: row for row in reduction["configurations"]}
    assert reduction["total_feature_count"] == 43
    assert by_id["K43"]["reduction_percent"] == 0.0
    assert by_id["HYBRID-K13"]["feature_count"] == 13
    assert by_id["HYBRID-K13"]["reduced_feature_count"] == 30
    assert by_id["HYBRID-K13"]["reduction_percent"] == pytest.approx(
        100.0 * 30 / 43
    )
    assert by_id["HYBRID-K13"]["reduction_percent"] == pytest.approx(69.76744186046511)


# ---------------------------------------------------------------------------
# Feature overlap
# ---------------------------------------------------------------------------


def test_29_feature_overlap_is_descriptive_only(
    real_basis, real_hybrid_lock, real_bpso_lock, real_bgwo_lock
) -> None:
    overlap = v10e.feature_overlap_analysis(
        real_basis, real_hybrid_lock, real_bpso_lock, real_bgwo_lock
    )
    assert overlap["descriptive_only"] is True
    assert overlap["mutated_any_subset"] is False
    assert overlap["hybrid_feature_count"] == 13
    assert overlap["bpso_feature_count"] == 10
    assert overlap["bgwo_feature_count"] == 14
    assert 0 <= overlap["hybrid_vs_bpso"]["intersection_size"] <= 10
    assert 0 <= overlap["hybrid_vs_bgwo"]["intersection_size"] <= 13
    assert len(overlap["hybrid_vs_mi_k11"]["per_seed"]) == 5
    assert 0.0 <= overlap["hybrid_vs_mi_k11"]["union_jaccard"] <= 1.0


# ---------------------------------------------------------------------------
# Result lock
# ---------------------------------------------------------------------------


def test_30_result_lock_semantic_hash_ignores_timestamps(
    real_hybrid_lock, real_bpso_lock, real_bgwo_lock, real_basis
) -> None:
    configurations = v10e.build_configuration_plan(
        real_basis, real_hybrid_lock, real_bpso_lock, real_bgwo_lock
    )
    first = _build_real_lock(
        real_hybrid_lock, real_bpso_lock, real_bgwo_lock, configurations
    )
    second = _build_real_lock(
        real_hybrid_lock,
        real_bpso_lock,
        real_bgwo_lock,
        configurations,
        started_at="later",
        completed_at="even-later",
        wall_time_sec=999.0,
    )
    assert first["semantic_result_lock_sha256"] == second["semantic_result_lock_sha256"]
    v10e.verify_result_lock(first)
    v10e.verify_result_lock(second)


def test_31_result_lock_contains_winner_immutability_and_no_resource_claims(
    real_hybrid_lock, real_bpso_lock, real_bgwo_lock, real_basis
) -> None:
    configurations = v10e.build_configuration_plan(
        real_basis, real_hybrid_lock, real_bpso_lock, real_bgwo_lock
    )
    lock = _build_real_lock(
        real_hybrid_lock, real_bpso_lock, real_bgwo_lock, configurations
    )
    assert lock["stage"] == "V1.0-E"
    assert lock["status"] == "FINAL_TEST_EVALUATED"
    assert lock["source_winner_stage"] == "V1.0-D"
    assert lock["test_used_for_evaluation_only"] is True
    assert lock["test_used_for_selection"] is False
    assert lock["test_used_for_tuning"] is False
    assert lock["test_used_for_feature_selection"] is False
    assert lock["test_used_for_winner_selection"] is False
    assert lock["optimizer_invoked"] is False
    assert lock["feature_reselection_performed"] is False
    assert lock["test_driven_tuning_performed"] is False
    assert lock["threshold_tuning_performed"] is False
    assert lock["winner_replaced_after_test"] is False
    assert lock["hybrid_optimizer_rerun"] is False
    assert lock["bpso_rerun"] is False
    assert lock["bgwo_rerun"] is False
    assert lock["resource_benchmark_executed"] is False
    assert lock["direct_energy_measured"] is False
    assert "not modified" in lock["winner_unchanged_statement"]
    assert lock["test_access_audit_sha256"] == _artifact_hashes()[v10e.AUDIT_PATH.name]
    assert lock["configuration_plan_sha256"] == v10e._json_sha256(lock["configuration_plan"])


def test_32_result_lock_tamper_is_rejected(
    real_hybrid_lock, real_bpso_lock, real_bgwo_lock, real_basis
) -> None:
    configurations = v10e.build_configuration_plan(
        real_basis, real_hybrid_lock, real_bpso_lock, real_bgwo_lock
    )
    lock = _build_real_lock(
        real_hybrid_lock, real_bpso_lock, real_bgwo_lock, configurations
    )
    tampered = deepcopy(lock)
    tampered["hybrid_mask_sha256"] = "0" * 64
    with pytest.raises(v10e.V10EError, match="verification failed"):
        v10e.verify_result_lock(tampered)
    tampered_flag = deepcopy(lock)
    tampered_flag["winner_replaced_after_test"] = True
    with pytest.raises(v10e.V10EError, match="verification failed"):
        v10e.verify_result_lock(tampered_flag)


def test_33_existing_result_lock_prohibits_campaign_rerun(tmp_path) -> None:
    (tmp_path / v10e.RESULT_LOCK_PATH.name).write_text("{}", encoding="utf-8")
    with pytest.raises(v10e.V10EError, match="already locked"):
        v10e.run_v10e(output_dir=tmp_path)


# ---------------------------------------------------------------------------
# Post-evaluation audit and immutability
# ---------------------------------------------------------------------------


def test_34_test_access_audit_passes_and_fails_on_winner_drift(
    real_hybrid_lock, real_bpso_lock, real_bgwo_lock
) -> None:
    authorization = {
        "winner_lock_verified_before_test_access": True,
        "test_authorized": True,
        "winner_lock_verified_at_utc": "2026-01-01T00:00:00+00:00",
        "test_access_activated_at_utc": "2026-01-01T00:00:01+00:00",
        "campaign_completed_at_utc": "2026-01-01T00:01:00+00:00",
        "rerun_count": 0,
    }
    immutability = {"hashes": {"a": "1", "b": "2"}, "hash_count": 2}
    before = deepcopy(dict(real_hybrid_lock))
    after = deepcopy(dict(real_hybrid_lock))
    result = v10e.build_test_access_audit(
        authorization=authorization,
        hybrid_lock_before=before,
        hybrid_lock_after=after,
        hybrid_lock_sha256_before="a" * 64,
        hybrid_lock_sha256_after="a" * 64,
        immutability_before=immutability,
        immutability_after=immutability,
        starting_head="e7465e9",
        ending_head="e7465e9",
        evaluation_count=60,
    )
    assert result["status"] == "PASS"
    assert result["classification"] == v10e.TEST_ACCESS_CLASSIFICATION
    assert result["test_authorized"] is True
    assert result["test_accessed"] is True
    assert result["test_used_for_evaluation"] is True
    assert result["test_used_for_evaluation_only"] is True
    assert result["test_used_for_selection"] is False
    assert result["test_used_for_tuning"] is False
    assert result["test_used_for_feature_selection"] is False
    assert result["test_used_for_winner_selection"] is False
    assert result["checks"]["governed_evaluation_count_is_sixty"] is True

    drifted = deepcopy(after)
    drifted["mask"][0] = 1 - int(drifted["mask"][0])
    with pytest.raises(v10e.V10EError, match="audit failed"):
        v10e.build_test_access_audit(
            authorization=authorization,
            hybrid_lock_before=before,
            hybrid_lock_after=drifted,
            hybrid_lock_sha256_before="a" * 64,
            hybrid_lock_sha256_after="c" * 64,
            immutability_before=immutability,
            immutability_after=immutability,
            starting_head="e7465e9",
            ending_head="e7465e9",
            evaluation_count=60,
        )


def test_35_immutability_snapshot_is_deterministic() -> None:
    first = v10e.snapshot_immutable_paths()
    second = v10e.snapshot_immutable_paths()
    assert first == second
    before = v10e.build_immutability_snapshot("BEFORE", first)
    after = v10e.build_immutability_snapshot("AFTER", second)
    assert before["hashes"] == after["hashes"]
    assert before["hash_count"] == after["hash_count"] == len(first)
    assert before["snapshot_phase"] == "BEFORE"
    assert after["snapshot_phase"] == "AFTER"


def test_36_immutable_snapshot_covers_v06_through_v10d() -> None:
    snapshot = v10e.snapshot_immutable_paths()
    joined = "\n".join(snapshot)
    assert "validation_lock.json" in joined
    assert "v08c_winner_lock.json" in joined
    assert "v09d_winner_lock.json" in joined
    assert "v09e_result_lock.json" in joined
    assert "v10d_winner_lock.json" in joined
    assert "v10d_run_3045.json" in joined
    assert "v10d_validation_comparison.json" in joined
    assert len(snapshot) >= len(v10e.v10c.FROZEN_ARTIFACT_PATHS)


# ---------------------------------------------------------------------------
# Source-level governance
# ---------------------------------------------------------------------------


def test_37_pipeline_source_cannot_invoke_optimizer_or_selector() -> None:
    source = inspect.getsource(v10e)
    campaign_source = inspect.getsource(v10e.evaluate_final_test_campaign)
    for forbidden in (
        "HybridBPSOBGWO",
        "BinaryParticleSwarmOptimizer",
        "feature_fitness_is_better",
        "create_v06_selector",
        "selector.fit",
        "select_search_winner",
        "grid_search",
        "run_v08c(",
        "run_v08d(",
        "run_v09d(",
        "run_v10d(",
    ):
        assert forbidden not in source, forbidden
    assert "model.fit(train_x" in campaign_source
    assert "metric_evaluator(test.ground_truth" in campaign_source


# ---------------------------------------------------------------------------
# Completed production artifacts
# ---------------------------------------------------------------------------


def test_38_completed_final_test_artifacts_verify_as_one_campaign() -> None:
    if not v10e.RESULT_LOCK_PATH.exists():
        pytest.skip("V1.0-E production campaign has not been executed in this checkout yet.")
    v10e.verify_v10e_artifacts()
    execution = v10e._read_json(v10e.EXECUTION_PATH)
    audit = v10e._read_json(v10e.AUDIT_PATH)
    lock = v10e._read_json(v10e.RESULT_LOCK_PATH)
    authorization = v10e._read_json(v10e.AUTHORIZATION_PATH)
    assert execution["status"] == "COMPLETED"
    assert execution["governed_evaluation_count"] == v10e.EXPECTED_EVALUATION_COUNT
    assert execution["winner_modified_after_test_access"] is False
    assert execution["readiness_for_v10f"] == "GO"
    assert audit["status"] == "PASS"
    assert audit["winner_lock_verified_before_test_access"] is True
    assert audit["test_authorized"] is True
    assert audit["test_accessed"] is True
    assert authorization["test_authorized"] is True
    assert authorization["test_authorized_before"] is False
    assert lock["hybrid_mask_sha256"] == v10e.EXPECTED_HYBRID_WINNER["mask_sha256"]
    assert (
        lock["semantic_result_lock_sha256"] == execution["semantic_result_lock_sha256"]
    )
