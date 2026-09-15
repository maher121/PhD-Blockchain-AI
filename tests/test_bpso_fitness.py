"""Synthetic tests for the V0.8-B real-fitness contract."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.optimization.bpso import BPSOConfig, BinaryParticleSwarmOptimizer
import src.optimization.feature_fitness as fitness
from src.security.experiment_data import DevelopmentExperimentData, fingerprint_feature_names


def _candidate_features() -> tuple[str, ...]:
    return tuple(f"feature_{index:02d}" for index in range(43))


def _prepared_split(
    name: str,
    seed: int,
    candidates: tuple[str, ...],
    row_start: int,
    rows: int,
) -> SimpleNamespace:
    index = pd.Index(np.arange(row_start, row_start + rows), name="source_index")
    row_ids = np.arange(row_start, row_start + rows)
    clean = pd.DataFrame(
        {
            "row_id": row_ids,
            **{
                feature: np.asarray(
                    [(position + column + seed) % 7 for position in range(rows)],
                    dtype=float,
                )
                for column, feature in enumerate(candidates)
            },
        },
        index=index,
    )
    attacked = clean.copy()
    if name == "validation":
        attacked.loc[index[1], candidates[0]] += 1.0
        attacked.loc[index[-1], candidates[1]] += 1.0
        labels_array = np.asarray([0, 1, 0, 1], dtype=np.int8)
    else:
        labels_array = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
    labels = pd.Series(labels_array, index=index, name="is_attack")
    truth = pd.DataFrame({"record_id": row_ids, "is_attack": labels_array})
    return SimpleNamespace(
        split_name=name,
        candidate_features=candidates,
        clean_features=clean,
        features=attacked,
        labels=labels,
        ground_truth=truth,
        attack_metadata={
            "attack_mode": "mixed",
            "configured_attack_rate": 0.05,
            "severity": "MEDIUM",
            "random_seed": seed,
        },
        row_ids_sha256=f"{name}-rows-{seed}",
        clean_features_sha256=f"{name}-clean-{seed}",
        attacked_features_sha256=f"{name}-attacked-{seed}",
        labels_sha256=f"{name}-labels-{seed}",
    )


def make_context(*, test_accessed: bool = False) -> fitness.FitnessContext:
    candidates = _candidate_features()
    metadata = {
        "split": {
            "seed": 42,
            "strategy": "order_grouped",
            "sizes": {"train": 6, "validation": 4, "test": 4},
        },
        "preprocessing": {
            "fitted_on_rows": 6,
            "ml_feature_columns": list(candidates),
        },
    }
    workloads = tuple(
        DevelopmentExperimentData(
            candidate_features=candidates,
            train=_prepared_split("train", seed, candidates, 100, 6),
            validation=_prepared_split("validation", seed, candidates, 200, 4),
            dataset_metadata=metadata,
        )
        for seed in fitness.EXPECTED_SEEDS
    )
    return fitness.FitnessContext(
        candidate_features=candidates,
        candidate_manifest_sha256=fingerprint_feature_names(candidates),
        workloads=workloads,
        seeds=fitness.EXPECTED_SEEDS,
        baseline=fitness.BaselineMetrics(1.0, 1.0, 1.0),
        dataset_split_seed=42,
        expected_train_rows=6,
        expected_validation_rows=4,
        test_accessed=test_accessed,
    )


class SpyModel:
    fit_calls: list[dict] = []
    predict_calls: list[dict] = []

    def __init__(self, model_name, features, parameters, random_state):
        self.model_name = model_name
        self.feature_names = list(features)
        self.parameters = dict(parameters)
        self.random_state = random_state

    def fit(self, frame, labels):
        self.fit_calls.append(
            {
                "seed": self.random_state,
                "columns": list(frame.columns),
                "index": frame.index.tolist(),
                "label_name": labels.name,
            }
        )
        return self

    def predict_frame(self, frame):
        self.predict_calls.append(
            {
                "seed": self.random_state,
                "columns": list(frame.columns),
                "index": frame.index.tolist(),
            }
        )
        scores = np.asarray([0.1, 0.9, 0.2, 0.8], dtype=float)
        return pd.DataFrame(
            {
                "anomaly_score": scores,
                "anomaly_label": (scores >= 0.5).astype(np.int8),
            },
            index=frame.index,
        )


def spy_factory(model_name, features, parameters, *, random_state):
    return SpyModel(model_name, features, parameters, random_state)


def evaluation(
    *,
    feasible: bool = True,
    violation: float = 0.0,
    count: int = 4,
    ap: float = 0.8,
    f1: float = 0.7,
    recall: float = 0.6,
    mask: tuple[int, ...] | None = None,
) -> fitness.FeatureFitnessEvaluation:
    resolved = mask or tuple([1] * count + [0] * (43 - count))
    return fitness.FeatureFitnessEvaluation(
        mask=resolved,
        mask_sha256=hashlib.sha256(np.asarray(resolved, dtype=np.uint8).tobytes()).hexdigest(),
        selected_features=tuple(f"f{index}" for index in range(count)),
        selected_features_sha256="a" * 64,
        selected_feature_count=count,
        feasible=feasible,
        normalized_violation=violation,
        relative_losses={"average_precision": 0.0, "f1": 0.0, "recall": 0.0},
        mean_metrics={
            "average_precision": ap,
            "f1": f1,
            "recall": recall,
            "precision": 0.5,
            "roc_auc": 0.5,
            "accuracy": 0.5,
            "false_positive_rate": 0.5,
            "false_negative_rate": 0.5,
            "attack_prevalence": 0.05,
            "selected_feature_visibility_rate": 0.5,
        },
        confusion_totals={
            "true_positives": 1,
            "true_negatives": 1,
            "false_positives": 1,
            "false_negatives": 1,
        },
        per_seed=(),
        decision_tree_fit_count=5,
    )


def test_01_candidate_mask_requires_exact_43_dimensions() -> None:
    with pytest.raises(ValueError, match="length 43"):
        fitness.validate_candidate_mask(np.ones(42, dtype=np.uint8))
    assert len(fitness.validate_candidate_mask(np.ones(43, dtype=np.uint8))) == 43


def test_02_candidate_mask_requires_binary_integer_values() -> None:
    bad = np.ones(43, dtype=float)
    with pytest.raises(ValueError, match="binary integer"):
        fitness.validate_candidate_mask(bad)
    bad_integer = np.ones(43, dtype=int)
    bad_integer[0] = 2
    with pytest.raises(ValueError, match="binary integer"):
        fitness.validate_candidate_mask(bad_integer)


def test_03_candidate_mask_must_be_nonempty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        fitness.validate_candidate_mask(np.zeros(43, dtype=np.uint8))


def test_04_leakage_audit_passes_development_only_context() -> None:
    audit = fitness.audit_fitness_context(make_context())
    assert audit.status == "PASS"
    assert audit.to_dict()["test_accessed"] is False
    assert all(check.passed for check in audit.checks)


def test_05_leakage_audit_blocks_test_access() -> None:
    audit = fitness.audit_fitness_context(make_context(test_accessed=True))
    assert audit.status == "FAIL"
    failed = {check.name for check in audit.checks if not check.passed}
    assert "test_access_prohibited" in failed


def test_06_forbidden_ground_truth_and_metadata_columns_are_registered() -> None:
    expected = {
        "Late_delivery_risk",
        "Order Status",
        "SUSPECTED_FRAUD",
        "is_attack",
        "row_id",
        "record_id",
        "original_value",
        "modified_value",
        "anomaly_score",
        "anomaly_label",
        "visibility_rate",
    }
    assert expected <= fitness.FORBIDDEN_MODEL_COLUMNS


def test_07_evaluator_preserves_feature_order_and_excludes_row_id() -> None:
    SpyModel.fit_calls.clear()
    SpyModel.predict_calls.clear()
    mask = np.zeros(43, dtype=np.uint8)
    mask[[7, 2, 11]] = 1
    result = fitness.FeatureFitnessEvaluator(
        make_context(), model_factory=spy_factory
    )(mask)
    expected = ("feature_02", "feature_07", "feature_11")
    assert result.selected_features == expected
    assert all(call["columns"] == list(expected) for call in SpyModel.fit_calls)
    assert all(call["columns"] == list(expected) for call in SpyModel.predict_calls)
    assert all("row_id" not in call["columns"] for call in SpyModel.fit_calls)


def test_08_models_fit_train_only_and_score_validation_only() -> None:
    SpyModel.fit_calls.clear()
    SpyModel.predict_calls.clear()
    fitness.FeatureFitnessEvaluator(make_context(), model_factory=spy_factory)(
        np.ones(43, dtype=np.uint8)
    )
    assert len(SpyModel.fit_calls) == len(SpyModel.predict_calls) == 5
    assert all(call["index"] == list(range(100, 106)) for call in SpyModel.fit_calls)
    assert all(call["index"] == list(range(200, 204)) for call in SpyModel.predict_calls)
    assert all(call["label_name"] == "is_attack" for call in SpyModel.fit_calls)


def test_09_five_seed_evaluation_is_deterministic_and_aggregated() -> None:
    evaluator = fitness.FeatureFitnessEvaluator(make_context(), model_factory=spy_factory)
    first = evaluator(np.ones(43, dtype=np.uint8))
    second = evaluator(np.ones(43, dtype=np.uint8))
    assert first.to_dict() == second.to_dict()
    assert tuple(record.seed for record in first.per_seed) == fitness.EXPECTED_SEEDS
    assert first.decision_tree_fit_count == 5
    assert first.mean_metrics["average_precision"] == 1.0
    assert first.mean_metrics["f1"] == 1.0
    assert first.mean_metrics["recall"] == 1.0
    assert first.mean_metrics["accuracy"] == 1.0
    assert first.confusion_totals == {
        "true_positives": 10,
        "true_negatives": 10,
        "false_positives": 0,
        "false_negatives": 0,
    }


def test_10_relative_loss_is_nonnegative() -> None:
    assert fitness.relative_loss(1.0, 1.1) == 0.0
    assert fitness.relative_loss(1.0, 0.95) == pytest.approx(0.05)


def test_11_feasibility_boundary_is_exactly_five_five_ten_percent() -> None:
    baseline = fitness.BaselineMetrics(1.0, 1.0, 1.0)
    losses, feasible, violation = fitness.constraint_outcome(
        baseline,
        {"average_precision": 0.95, "f1": 0.95, "recall": 0.90},
    )
    assert losses == pytest.approx(
        {"average_precision": 0.05, "f1": 0.05, "recall": 0.10}
    )
    assert feasible is True
    assert violation == 0.0


def test_12_each_margin_fails_immediately_outside_numerical_tolerance() -> None:
    baseline = fitness.BaselineMetrics(1.0, 1.0, 1.0)
    for metric, value in (
        ("average_precision", 0.949999999),
        ("f1", 0.949999999),
        ("recall", 0.899999999),
    ):
        means = {"average_precision": 1.0, "f1": 1.0, "recall": 1.0}
        means[metric] = value
        _, feasible, violation = fitness.constraint_outcome(baseline, means)
        assert feasible is False
        assert violation > 0.0


def test_13_normalized_violation_sums_all_margin_excesses() -> None:
    _, feasible, violation = fitness.constraint_outcome(
        fitness.BaselineMetrics(1.0, 1.0, 1.0),
        {"average_precision": 0.90, "f1": 0.90, "recall": 0.80},
    )
    assert feasible is False
    assert violation == pytest.approx(3.0)


def test_14_feasible_beats_smaller_infeasible() -> None:
    assert fitness.feature_fitness_is_better(
        evaluation(feasible=True, count=8),
        evaluation(feasible=False, violation=0.01, count=1),
    )


def test_15_smaller_feasible_subset_wins_before_predictive_ties() -> None:
    assert fitness.feature_fitness_is_better(
        evaluation(count=3, ap=0.6), evaluation(count=4, ap=1.0)
    )


def test_16_feasible_ap_then_f1_then_recall_tie_breaks() -> None:
    assert fitness.feature_fitness_is_better(
        evaluation(ap=0.9), evaluation(ap=0.8)
    )
    assert fitness.feature_fitness_is_better(
        evaluation(ap=0.8, f1=0.8), evaluation(ap=0.8, f1=0.7)
    )
    assert fitness.feature_fitness_is_better(
        evaluation(ap=0.8, f1=0.7, recall=0.7),
        evaluation(ap=0.8, f1=0.7, recall=0.6),
    )


def test_17_canonical_mask_is_final_feasible_tie_break() -> None:
    left_mask = tuple([0, 1, 1, 1] + [0] * 39)
    right_mask = tuple([1, 0, 1, 1] + [0] * 39)
    left = evaluation(count=3, mask=left_mask)
    right = evaluation(count=3, mask=right_mask)
    assert fitness.feature_fitness_is_better(left, right)
    assert not fitness.feature_fitness_is_better(right, left)


def test_18_infeasible_lower_violation_wins_before_other_fields() -> None:
    assert fitness.feature_fitness_is_better(
        evaluation(feasible=False, violation=0.1, count=20, ap=0.1),
        evaluation(feasible=False, violation=0.2, count=1, ap=1.0),
    )


def test_19_infeasible_ap_f1_recall_count_and_mask_order() -> None:
    base = dict(feasible=False, violation=0.1)
    assert fitness.feature_fitness_is_better(
        evaluation(**base, ap=0.9), evaluation(**base, ap=0.8)
    )
    assert fitness.feature_fitness_is_better(
        evaluation(**base, ap=0.8, f1=0.8),
        evaluation(**base, ap=0.8, f1=0.7),
    )
    assert fitness.feature_fitness_is_better(
        evaluation(**base, ap=0.8, f1=0.7, recall=0.7),
        evaluation(**base, ap=0.8, f1=0.7, recall=0.6),
    )
    assert fitness.feature_fitness_is_better(
        evaluation(**base, count=3), evaluation(**base, count=4)
    )


def test_20_engine_cache_and_actual_fit_accounting_are_compatible() -> None:
    context = make_context()
    evaluator = fitness.FeatureFitnessEvaluator(context, model_factory=spy_factory)
    config = BPSOConfig(43, 2, 1, ())
    duplicate = np.ones((2, 43), dtype=np.uint8)
    result = BinaryParticleSwarmOptimizer(
        config, evaluator, fitness.feature_fitness_is_better
    ).optimize(1042, initial_population=duplicate)
    assert result.total_fitness_requests == 2
    assert result.unique_evaluations == 1
    assert result.cache_hits == 1
    assert evaluator.decision_tree_fit_count == result.unique_evaluations * 5 == 5


def test_21_frozen_model_and_threshold_constants_are_exact() -> None:
    assert dict(fitness.EXPECTED_MODEL_PARAMETERS) == {
        "max_depth": 5,
        "min_samples_leaf": 20,
        "class_weight": "balanced",
    }
    assert fitness.EXPECTED_MODEL_NAME == "decision_tree"
    assert fitness.EXPECTED_THRESHOLD == 0.5


def test_22_result_records_nonobjective_metrics_and_visibility() -> None:
    result = fitness.FeatureFitnessEvaluator(make_context(), model_factory=spy_factory)(
        np.ones(43, dtype=np.uint8)
    )
    expected = {
        "precision",
        "roc_auc",
        "accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "attack_prevalence",
        "selected_feature_visibility_rate",
    }
    assert expected <= set(result.mean_metrics)
    assert "selected_feature_visibility_rate" not in result.relative_losses


def test_23_result_declares_no_weighted_pareto_or_resource_objective() -> None:
    result = fitness.FeatureFitnessEvaluator(make_context(), model_factory=spy_factory)(
        np.ones(43, dtype=np.uint8)
    ).to_dict()
    assert result["weighted_objective_used"] is False
    assert result["pareto_objective_used"] is False
    assert result["resource_or_energy_objective_used"] is False
