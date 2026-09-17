"""Synthetic unit tests for the dataset-independent V0.9-B BGWO engine."""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
import json

import numpy as np
import pytest

import src.optimization.bgwo as bgwo


@dataclass(frozen=True)
class Evaluation:
    feasible: bool
    violation: float
    cardinality: int
    primary_score: float
    secondary_score: float
    tie_break_key: str


def mask_key(mask: np.ndarray) -> str:
    return "".join(str(int(value)) for value in mask)


def constrained_better(left: Evaluation, right: Evaluation) -> bool:
    if left.feasible != right.feasible:
        return left.feasible
    if left.feasible:
        return (
            left.cardinality,
            -left.primary_score,
            -left.secondary_score,
            left.tie_break_key,
        ) < (
            right.cardinality,
            -right.primary_score,
            -right.secondary_score,
            right.tie_break_key,
        )
    return (
        left.violation,
        -left.primary_score,
        -left.secondary_score,
        left.cardinality,
        left.tie_break_key,
    ) < (
        right.violation,
        -right.primary_score,
        -right.secondary_score,
        right.cardinality,
        right.tie_break_key,
    )


def score_better(left: Evaluation, right: Evaluation) -> bool:
    return (
        -left.primary_score,
        -left.secondary_score,
        left.cardinality,
        left.tie_break_key,
    ) < (
        -right.primary_score,
        -right.secondary_score,
        right.cardinality,
        right.tie_break_key,
    )


def approved_config(**changes) -> bgwo.BGWOConfig:
    config = bgwo.BGWOConfig(
        dimensions=43,
        wolf_count=12,
        evaluated_iterations=20,
        random_cardinalities=(4, 8, 11, 14, 18, 22, 26, 30, 34, 38),
    )
    return replace(config, **changes) if changes else config


def cardinality_objective(mask: np.ndarray) -> Evaluation:
    return Evaluation(True, 0.0, int(mask.sum()), 0.0, 0.0, mask_key(mask))


def test_01_approved_dimensions_wolves_and_budget() -> None:
    config = approved_config()
    assert config.dimensions == 43
    assert config.wolf_count == 12
    assert config.evaluated_iterations == 20
    assert config.maximum_candidate_requests == 240


def test_02_configuration_rejects_iteration_and_cardinality_drift() -> None:
    with pytest.raises(ValueError, match="at least one iteration"):
        approved_config(evaluated_iterations=0)
    with pytest.raises(ValueError, match="wolf_count - 2"):
        approved_config(random_cardinalities=(4,))
    with pytest.raises(ValueError, match="exactly one minimum"):
        approved_config(minimum_selected_features=2)
    with pytest.raises(ValueError, match="standard logistic sigmoid"):
        approved_config(transfer_function="tanh")


def test_03_explicit_generator_uses_pcg64_and_not_global_seed() -> None:
    source = inspect.getsource(bgwo)
    assert "np.random.Generator(np.random.PCG64" in source
    assert "np.random.seed(" not in source
    generator = np.random.Generator(np.random.PCG64(2042))
    population = bgwo.build_initial_population(approved_config(), generator)
    assert population.shape == (12, 43)
    with pytest.raises(TypeError, match="PCG64"):
        bgwo.build_initial_population(
            approved_config(), np.random.Generator(np.random.MT19937(2042))
        )


def test_04_same_seed_reproduces_initial_population_and_latent_positions() -> None:
    optimizer = bgwo.BinaryGreyWolfOptimizer(
        approved_config(), cardinality_objective, constrained_better
    )
    first_population, first_latent = optimizer.initialize(2042)
    second_population, second_latent = optimizer.initialize(2042)
    assert np.array_equal(first_population, second_population)
    assert np.array_equal(first_latent, second_latent)


def test_05_different_optimizer_seeds_change_stochastic_state() -> None:
    optimizer = bgwo.BinaryGreyWolfOptimizer(
        approved_config(), cardinality_objective, constrained_better
    )
    first_population, first_latent = optimizer.initialize(2042)
    second_population, second_latent = optimizer.initialize(2043)
    assert not np.array_equal(first_population[2:], second_population[2:])
    assert not np.array_equal(first_latent[2:], second_latent[2:])


def test_06_initial_population_has_exact_anchor_cardinalities() -> None:
    optimizer = bgwo.BinaryGreyWolfOptimizer(
        approved_config(), cardinality_objective, constrained_better
    )
    population, _ = optimizer.initialize(2042, k42_inactive_index=42)
    assert population.sum(axis=1).tolist() == [
        43,
        42,
        4,
        8,
        11,
        14,
        18,
        22,
        26,
        30,
        34,
        38,
    ]
    assert population[0].tolist() == [1] * 43
    assert population[1, 42] == 0
    assert int(population[1].sum()) == 42


def test_07_external_k42_mask_is_supported_without_feature_semantics() -> None:
    mask = np.ones(43, dtype=np.uint8)
    mask[7] = 0
    generator = np.random.Generator(np.random.PCG64(2042))
    population = bgwo.build_initial_population(approved_config(), generator, k42_mask=mask)
    assert np.array_equal(population[1], mask)
    bad = mask.copy()
    bad[8] = 0
    with pytest.raises(ValueError, match="dimensions-1"):
        bgwo.build_initial_population(
            approved_config(),
            np.random.Generator(np.random.PCG64(2042)),
            k42_mask=bad,
        )


def test_08_all_initial_positions_are_binary_nonempty_and_length_43() -> None:
    optimizer = bgwo.BinaryGreyWolfOptimizer(
        approved_config(), cardinality_objective, constrained_better
    )
    population, _ = optimizer.initialize(2042)
    assert population.shape == (12, 43)
    assert set(np.unique(population)) <= {0, 1}
    assert np.all(population.sum(axis=1) >= 1)


def test_09_control_schedule_endpoints_length_and_monotonicity() -> None:
    schedule = bgwo.linear_control_schedule(20, 2.0, 0.0)
    assert len(schedule) == 19
    assert schedule[0] == pytest.approx(2.0)
    assert schedule[-1] == pytest.approx(0.0)
    assert all(left >= right for left, right in zip(schedule, schedule[1:]))


def test_10_iteration_zero_has_no_a_and_final_iteration_has_zero() -> None:
    result = bgwo.BinaryGreyWolfOptimizer(
        approved_config(early_stopping_min_iteration=20),
        cardinality_objective,
        constrained_better,
    ).optimize(2042)
    assert result.convergence_history[0].iteration_index == 0
    assert result.convergence_history[0].control_parameter_a is None
    assert result.convergence_history[1].control_parameter_a == pytest.approx(2.0)
    assert result.convergence_history[-1].control_parameter_a == pytest.approx(0.0)
    assert result.evaluated_iteration_count == 20
    assert result.total_candidate_requests == 240


def test_11_sigmoid_is_numerically_stable_and_standard() -> None:
    values = bgwo.sigmoid_probabilities(
        np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0]), clamp_min=-6.0, clamp_max=6.0
    )
    assert np.isfinite(values).all()
    assert values[2] == 0.5
    assert values[0] == pytest.approx(1.0 / (1.0 + np.exp(6.0)))
    assert values[-1] == pytest.approx(1.0 / (1.0 + np.exp(-6.0)))
    assert values[1] == pytest.approx(1.0 / (1.0 + np.exp(1.0)))
    assert values[3] == pytest.approx(1.0 / (1.0 + np.exp(-1.0)))


def test_12_empty_mask_repair_uses_highest_absolute_latent_and_lowest_index() -> None:
    repaired, occurred, index = bgwo.repair_empty_mask([0, 0, 0], [0.2, -3.0, 3.0])
    assert occurred is True
    assert index == 1
    assert repaired.tolist() == [0, 1, 0]


def test_13_nonempty_mask_is_not_repaired() -> None:
    mask, occurred, index = bgwo.repair_empty_mask([0, 1, 0], [0.9, 0.1, 0.8])
    assert mask.tolist() == [0, 1, 0]
    assert occurred is False
    assert index is None


def test_14_sampled_positions_remain_binary_and_nonempty() -> None:
    generator = np.random.Generator(np.random.PCG64(2042))
    for _ in range(50):
        sampled, _, _, _, _ = bgwo.sample_binary_mask(
            np.full(43, -6.0),
            clamp_min=-6.0,
            clamp_max=6.0,
            generator=generator,
        )
        assert set(sampled.tolist()) <= {0, 1}
        assert int(sampled.sum()) >= 1


def test_15_comparator_is_required_and_evaluations_need_no_python_ordering() -> None:
    with pytest.raises(TypeError):
        bgwo.BinaryGreyWolfOptimizer(approved_config(), cardinality_objective)  # type: ignore[call-arg]
    result = bgwo.BinaryGreyWolfOptimizer(
        approved_config(evaluated_iterations=1),
        cardinality_objective,
        constrained_better,
    ).optimize(2042)
    assert isinstance(result.best_evaluation, Evaluation)


def test_16_comparator_must_be_strict_and_asymmetric() -> None:
    optimizer = bgwo.BinaryGreyWolfOptimizer(
        approved_config(evaluated_iterations=1),
        cardinality_objective,
        lambda left, right: True,
    )
    with pytest.raises(bgwo.ComparatorContractError, match="strict and asymmetric"):
        optimizer.optimize(2042)


def test_17_feasible_candidate_beats_smaller_infeasible_candidate() -> None:
    config = bgwo.BGWOConfig(3, 3, 1, (1,))
    population = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.uint8)

    def objective(mask: np.ndarray) -> Evaluation:
        feasible = bool(mask[0])
        return Evaluation(
            feasible,
            0.0 if feasible else 1.0,
            int(mask.sum()),
            0.0,
            0.0,
            mask_key(mask),
        )

    result = bgwo.BinaryGreyWolfOptimizer(config, objective, constrained_better).optimize(
        2042, initial_population=population
    )
    assert result.best_evaluation.feasible is True
    assert result.best_selected_feature_count == 2


def test_18_constrained_comparator_minimizes_cardinality_among_feasible() -> None:
    config = bgwo.BGWOConfig(3, 3, 1, (1,))
    population = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.uint8)

    def objective(mask: np.ndarray) -> Evaluation:
        return Evaluation(True, 0.0, int(mask.sum()), 0.0, 0.0, mask_key(mask))

    result = bgwo.BinaryGreyWolfOptimizer(config, objective, constrained_better).optimize(
        2042, initial_population=population
    )
    assert result.best_mask.tolist() == [1, 0, 0]
    assert result.best_selected_feature_count == 1


def test_19_run_local_cache_evaluates_duplicate_masks_once() -> None:
    config = approved_config(evaluated_iterations=1)
    population = np.zeros((12, 43), dtype=np.uint8)
    population[:, 0] = 1
    calls = 0

    def objective(mask: np.ndarray) -> Evaluation:
        nonlocal calls
        calls += 1
        return Evaluation(True, 0.0, 1, 0.0, 0.0, mask_key(mask))

    result = bgwo.BinaryGreyWolfOptimizer(config, objective, constrained_better).optimize(
        2042, initial_population=population
    )
    assert calls == result.unique_evaluations == 1
    assert result.total_candidate_requests == 12
    assert result.cache_hits == 11
    assert result.total_candidate_requests == result.unique_evaluations + result.cache_hits


def test_20_cache_is_isolated_between_optimizer_runs() -> None:
    config = approved_config(evaluated_iterations=1)
    population = np.zeros((12, 43), dtype=np.uint8)
    population[:, 0] = 1
    calls = 0

    def objective(mask: np.ndarray) -> Evaluation:
        nonlocal calls
        calls += 1
        return Evaluation(True, 0.0, 1, 0.0, 0.0, mask_key(mask))

    optimizer = bgwo.BinaryGreyWolfOptimizer(config, objective, constrained_better)
    first = optimizer.optimize(2042, initial_population=population)
    second = optimizer.optimize(2042, initial_population=population)
    assert calls == 2
    assert first.unique_evaluations == second.unique_evaluations == 1


def test_21_cache_does_not_change_optimization_decisions() -> None:
    cached_config = approved_config(
        evaluated_iterations=8,
        early_stopping_min_iteration=20,
    )
    uncached_config = replace(cached_config, cache_enabled=False)
    base = bgwo.BinaryGreyWolfOptimizer(
        cached_config,
        cardinality_objective,
        constrained_better,
    )
    population, latent = base.initialize(2042)
    cached = base.optimize(2042, initial_population=population, initial_latent_positions=latent)
    uncached = bgwo.BinaryGreyWolfOptimizer(
        uncached_config,
        cardinality_objective,
        constrained_better,
    ).optimize(2042, initial_population=population, initial_latent_positions=latent)
    assert np.array_equal(cached.best_mask, uncached.best_mask)
    assert cached.best_evaluation == uncached.best_evaluation
    assert [row.best_mask.tolist() for row in cached.convergence_history] == [
        row.best_mask.tolist() for row in uncached.convergence_history
    ]


def test_22_convergence_history_has_complete_iteration_schema() -> None:
    result = bgwo.BinaryGreyWolfOptimizer(
        approved_config(evaluated_iterations=3, early_stopping_min_iteration=20),
        cardinality_objective,
        constrained_better,
    ).optimize(2042)
    assert [row.iteration_index for row in result.convergence_history] == [0, 1, 2]
    assert all(row.request_count == 12 for row in result.convergence_history)
    expected = {
        "iteration_index",
        "best_mask",
        "best_selected_feature_count",
        "best_evaluation",
        "request_count",
        "cumulative_unique_evaluations",
        "cumulative_cache_hits",
        "population_diversity",
        "best_rank_improved",
        "best_changed",
        "repair_count",
        "control_parameter_a",
    }
    assert set(result.convergence_history[0].to_dict()) == expected


def test_23_hamming_diversity_and_degenerate_behavior() -> None:
    population = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.uint8)
    assert bgwo.mean_pairwise_normalized_hamming(population) == 1.0
    assert bgwo.mean_pairwise_normalized_hamming(population[:1]) == 0.0
    three = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.uint8)
    assert bgwo.mean_pairwise_normalized_hamming(three) == pytest.approx(2.0 / 3.0)


def test_24_early_stopping_cannot_occur_before_iteration_ten() -> None:
    def objective(mask: np.ndarray) -> Evaluation:
        return Evaluation(
            True,
            0.0,
            int(mask.sum()),
            float(np.all(mask == 1)),
            0.0,
            mask_key(mask),
        )

    result = bgwo.BinaryGreyWolfOptimizer(
        approved_config(),
        objective,
        score_better,
    ).optimize(2042)
    assert result.stop_reason == bgwo.STOP_EARLY_NO_IMPROVEMENT
    assert result.convergence_history[-1].iteration_index == 10
    assert result.evaluated_iteration_count == 11
    assert result.best_iteration == 0


def test_25_max_iterations_has_exact_request_ceiling() -> None:
    config = bgwo.BGWOConfig(
        dimensions=3,
        wolf_count=3,
        evaluated_iterations=5,
        random_cardinalities=(1,),
        cache_enabled=False,
        early_stopping_min_iteration=5,
    )
    result = bgwo.BinaryGreyWolfOptimizer(
        config,
        cardinality_objective,
        constrained_better,
    ).optimize(2042)
    assert result.stop_reason == bgwo.STOP_MAX_ITERATIONS
    assert result.evaluated_iteration_count == 5
    assert result.total_candidate_requests == 15
    assert result.maximum_iterations == 5


def test_26_best_iteration_and_stop_reason_are_serialized() -> None:
    result = bgwo.BinaryGreyWolfOptimizer(
        approved_config(),
        cardinality_objective,
        constrained_better,
    ).optimize(2042)
    payload = result.to_dict()
    assert payload["best_iteration"] == result.best_iteration
    assert payload["stop_reason"] in {
        bgwo.STOP_MAX_ITERATIONS,
        bgwo.STOP_EARLY_NO_IMPROVEMENT,
    }


def test_27_complete_result_reproduces_for_same_seed() -> None:
    optimizer = bgwo.BinaryGreyWolfOptimizer(
        approved_config(),
        cardinality_objective,
        constrained_better,
    )
    first = optimizer.optimize(2042)
    second = optimizer.optimize(2042)
    assert first.to_json() == second.to_json()


def test_28_result_json_serialization_is_safe_and_deterministic() -> None:
    result = bgwo.BinaryGreyWolfOptimizer(
        approved_config(evaluated_iterations=2, early_stopping_min_iteration=20),
        cardinality_objective,
        constrained_better,
    ).optimize(2042)
    first = result.to_json()
    second = result.to_json()
    payload = json.loads(first)
    assert first == second
    assert payload["best_mask"] == result.best_mask.tolist()
    assert isinstance(payload["best_evaluation"], dict)
    assert payload["configuration_snapshot"]["optimizer_seed"] == 2042
    with pytest.raises(ValueError, match="Non-finite"):
        bgwo.to_json_compatible(float("nan"))


def test_29_engine_has_no_dataset_or_classifier_dependency() -> None:
    source = inspect.getsource(bgwo)
    forbidden = (
        "DataCo",
        "sklearn",
        "DecisionTree",
        "LogisticRegression",
        "pandas",
        "final_test",
        "feature_fitness",
    )
    assert all(marker not in source for marker in forbidden)


def test_30_optimizer_records_repairs_in_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bgwo.BGWOConfig(
        dimensions=3,
        wolf_count=3,
        evaluated_iterations=2,
        random_cardinalities=(1,),
        early_stopping_min_iteration=20,
    )

    def forced_repair(latent, *, clamp_min, clamp_max, generator):
        del latent, clamp_min, clamp_max, generator
        return (
            np.array([1, 0, 0], dtype=np.uint8),
            True,
            0,
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([1.0, 0.0, 0.0], dtype=float),
        )

    monkeypatch.setattr(bgwo, "sample_binary_mask", forced_repair)
    result = bgwo.BinaryGreyWolfOptimizer(
        config,
        cardinality_objective,
        constrained_better,
    ).optimize(2042)
    assert result.repair_count == 3
    assert result.convergence_history[0].repair_count == 0
    assert result.convergence_history[1].repair_count == 3
