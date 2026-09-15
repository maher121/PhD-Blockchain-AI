"""Synthetic unit tests for the dataset-independent V0.8-A BPSO engine."""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
import json

import numpy as np
import pytest

import src.optimization.bpso as bpso


@dataclass(frozen=True)
class Evaluation:
    feasible: bool
    violation: float
    cardinality: int
    score: float


def constrained_better(left: Evaluation, right: Evaluation) -> bool:
    if left.feasible != right.feasible:
        return left.feasible
    if left.feasible:
        return (left.cardinality, -left.score) < (right.cardinality, -right.score)
    return (left.violation, -left.score, left.cardinality) < (
        right.violation,
        -right.score,
        right.cardinality,
    )


def score_better(left: Evaluation, right: Evaluation) -> bool:
    return (-left.score, left.cardinality) < (-right.score, right.cardinality)


def approved_config(**changes) -> bpso.BPSOConfig:
    config = bpso.BPSOConfig(
        dimensions=43,
        particle_count=12,
        evaluated_generations=20,
        random_cardinalities=(4, 8, 11, 14, 18, 22, 26, 30, 34, 38),
    )
    return replace(config, **changes) if changes else config


def cardinality_objective(mask: np.ndarray) -> Evaluation:
    return Evaluation(True, 0.0, int(mask.sum()), 0.0)


def test_01_approved_dimensions_particles_and_budget() -> None:
    config = approved_config()
    assert config.dimensions == 43
    assert config.particle_count == 12
    assert config.evaluated_generations == 20
    assert config.maximum_fitness_requests == 240


def test_02_configuration_rejects_generation_and_cardinality_drift() -> None:
    with pytest.raises(ValueError, match="at least one generation"):
        approved_config(evaluated_generations=0)
    with pytest.raises(ValueError, match="particle_count - 2"):
        approved_config(random_cardinalities=(4,))
    with pytest.raises(ValueError, match="minimum of exactly one"):
        approved_config(minimum_selected_features=2)


def test_03_explicit_generator_uses_pcg64_and_not_global_seed() -> None:
    source = inspect.getsource(bpso)
    assert "np.random.Generator(np.random.PCG64" in source
    assert "np.random.seed(" not in source
    generator = np.random.Generator(np.random.PCG64(1042))
    population = bpso.build_initial_population(approved_config(), generator)
    assert population.shape == (12, 43)
    with pytest.raises(TypeError, match="PCG64"):
        bpso.build_initial_population(
            approved_config(), np.random.Generator(np.random.MT19937(1042))
        )


def test_04_same_seed_reproduces_initial_population_and_velocities() -> None:
    optimizer = bpso.BinaryParticleSwarmOptimizer(
        approved_config(), cardinality_objective, constrained_better
    )
    first_population, first_velocities = optimizer.initialize(1042)
    second_population, second_velocities = optimizer.initialize(1042)
    assert np.array_equal(first_population, second_population)
    assert np.array_equal(first_velocities, second_velocities)


def test_05_different_optimizer_seeds_change_stochastic_state() -> None:
    optimizer = bpso.BinaryParticleSwarmOptimizer(
        approved_config(), cardinality_objective, constrained_better
    )
    first_population, first_velocities = optimizer.initialize(1042)
    second_population, second_velocities = optimizer.initialize(1043)
    assert not np.array_equal(first_population[2:], second_population[2:])
    assert not np.array_equal(first_velocities, second_velocities)


def test_06_initial_population_has_exact_anchor_cardinalities() -> None:
    optimizer = bpso.BinaryParticleSwarmOptimizer(
        approved_config(), cardinality_objective, constrained_better
    )
    population, _ = optimizer.initialize(1042, k42_inactive_index=42)
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
    generator = np.random.Generator(np.random.PCG64(1042))
    population = bpso.build_initial_population(
        approved_config(), generator, k42_mask=mask
    )
    assert np.array_equal(population[1], mask)
    bad = mask.copy()
    bad[8] = 0
    with pytest.raises(ValueError, match="D-1"):
        bpso.build_initial_population(
            approved_config(),
            np.random.Generator(np.random.PCG64(1042)),
            k42_mask=bad,
        )


def test_08_all_initial_positions_are_binary_nonempty_and_length_43() -> None:
    optimizer = bpso.BinaryParticleSwarmOptimizer(
        approved_config(), cardinality_objective, constrained_better
    )
    population, _ = optimizer.initialize(1042)
    assert population.shape == (12, 43)
    assert set(np.unique(population)) <= {0, 1}
    assert np.all(population.sum(axis=1) >= 1)


def test_09_initial_velocities_follow_approved_uniform_bounds() -> None:
    optimizer = bpso.BinaryParticleSwarmOptimizer(
        approved_config(), cardinality_objective, constrained_better
    )
    _, velocities = optimizer.initialize(1042)
    assert velocities.shape == (12, 43)
    assert np.all(velocities >= -1.0)
    assert np.all(velocities <= 1.0)
    assert np.unique(velocities).size > 100


def test_10_velocity_updates_are_clamped_to_six() -> None:
    generator = np.random.Generator(np.random.PCG64(1042))
    updated = bpso.update_velocity(
        np.full(43, 100.0),
        np.zeros(43, dtype=np.uint8),
        np.ones(43, dtype=np.uint8),
        np.ones(43, dtype=np.uint8),
        inertia=1.0,
        cognitive_coefficient=100.0,
        social_coefficient=100.0,
        clamp_min=-6.0,
        clamp_max=6.0,
        generator=generator,
    )
    assert np.all(updated <= 6.0)
    assert np.all(updated >= -6.0)
    assert np.any(updated == 6.0)


def test_11_sigmoid_is_numerically_stable_and_standard() -> None:
    values = bpso.stable_sigmoid(np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0]))
    assert np.isfinite(values).all()
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert values[2] == 0.5
    assert values[1] == pytest.approx(1.0 / (1.0 + np.exp(1.0)))
    assert values[3] == pytest.approx(1.0 / (1.0 + np.exp(-1.0)))


def test_12_inertia_schedule_endpoints_length_and_monotonicity() -> None:
    schedule = bpso.linear_inertia_schedule(20, 0.9, 0.4)
    assert len(schedule) == 19
    assert schedule[0] == pytest.approx(0.9)
    assert schedule[-1] == pytest.approx(0.4)
    assert all(left >= right for left, right in zip(schedule, schedule[1:]))


def test_13_generation_zero_has_no_inertia_and_final_update_has_point_four() -> None:
    config = approved_config(early_stopping_min_generation=20)
    result = bpso.BinaryParticleSwarmOptimizer(
        config, cardinality_objective, constrained_better
    ).optimize(1042)
    assert result.convergence_history[0].generation_index == 0
    assert result.convergence_history[0].inertia is None
    assert result.convergence_history[1].inertia == pytest.approx(0.9)
    assert result.convergence_history[-1].inertia == pytest.approx(0.4)
    assert result.evaluated_generation_count == 20
    assert result.total_fitness_requests == 240


def test_14_empty_mask_repair_uses_highest_probability() -> None:
    probabilities = np.array([0.1, 0.7, 0.2])
    repaired, occurred, index = bpso.repair_empty_mask([0, 0, 0], probabilities)
    assert occurred is True
    assert index == 1
    assert repaired.tolist() == [0, 1, 0]


def test_15_empty_mask_tie_repair_uses_lowest_index_deterministically() -> None:
    first = bpso.repair_empty_mask([0, 0, 0], [0.5, 0.5, 0.5])
    second = bpso.repair_empty_mask([0, 0, 0], [0.5, 0.5, 0.5])
    assert first[0].tolist() == [1, 0, 0]
    assert first[1:] == (True, 0)
    assert np.array_equal(first[0], second[0])


def test_16_nonempty_mask_is_not_repaired() -> None:
    mask, occurred, index = bpso.repair_empty_mask([0, 1, 0], [0.9, 0.1, 0.8])
    assert mask.tolist() == [0, 1, 0]
    assert occurred is False
    assert index is None


def test_17_sampled_positions_remain_binary_and_nonempty() -> None:
    generator = np.random.Generator(np.random.PCG64(1042))
    for _ in range(50):
        position, _, _ = bpso.sample_binary_position(np.full(43, -6.0), generator)
        assert set(position.tolist()) <= {0, 1}
        assert int(position.sum()) >= 1


def test_18_comparator_is_required_and_evaluations_need_no_python_ordering() -> None:
    with pytest.raises(TypeError):
        bpso.BinaryParticleSwarmOptimizer(approved_config(), cardinality_objective)  # type: ignore[call-arg]
    result = bpso.BinaryParticleSwarmOptimizer(
        approved_config(evaluated_generations=1),
        cardinality_objective,
        constrained_better,
    ).optimize(1042)
    assert isinstance(result.best_evaluation, Evaluation)


def test_19_comparator_must_be_strict_and_asymmetric() -> None:
    optimizer = bpso.BinaryParticleSwarmOptimizer(
        approved_config(evaluated_generations=1),
        cardinality_objective,
        lambda left, right: True,
    )
    with pytest.raises(bpso.ComparatorContractError, match="strict and asymmetric"):
        optimizer.optimize(1042)


def test_20_feasible_candidate_beats_smaller_infeasible_candidate() -> None:
    config = bpso.BPSOConfig(3, 2, 1, ())
    population = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.uint8)

    def objective(mask: np.ndarray) -> Evaluation:
        feasible = bool(mask[0])
        return Evaluation(feasible, 0.0 if feasible else 1.0, int(mask.sum()), 0.0)

    result = bpso.BinaryParticleSwarmOptimizer(
        config, objective, constrained_better
    ).optimize(1042, initial_population=population)
    assert result.best_evaluation.feasible is True
    assert result.best_selected_feature_count == 2


def test_21_constrained_comparator_minimizes_cardinality_among_feasible() -> None:
    config = bpso.BPSOConfig(3, 2, 1, ())
    population = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.uint8)

    def objective(mask: np.ndarray) -> Evaluation:
        return Evaluation(True, 0.0, int(mask.sum()), 0.0)

    result = bpso.BinaryParticleSwarmOptimizer(
        config, objective, constrained_better
    ).optimize(1042, initial_population=population)
    assert result.best_mask.tolist() == [1, 0, 0]
    assert result.best_selected_feature_count == 1


def test_22_personal_and_global_bests_initialize_from_injected_comparator() -> None:
    config = bpso.BPSOConfig(4, 4, 1, (1, 2))
    result = bpso.BinaryParticleSwarmOptimizer(
        config, cardinality_objective, constrained_better
    ).optimize(1042)
    initial_counts = result.initial_population.sum(axis=1).astype(int).tolist()
    assert [item.cardinality for item in result.personal_best_evaluations] == initial_counts
    assert result.best_evaluation.cardinality == min(initial_counts) == 1
    assert np.array_equal(result.best_mask, result.personal_best_masks[2])


def test_23_personal_and_global_bests_never_regress() -> None:
    target = np.zeros(43, dtype=np.uint8)
    target[[0, 1, 2]] = 1

    def objective(mask: np.ndarray) -> Evaluation:
        return Evaluation(True, 0.0, int(mask.sum()), -float(np.count_nonzero(mask != target)))

    result = bpso.BinaryParticleSwarmOptimizer(
        approved_config(), objective, score_better
    ).optimize(1042)
    initial = [objective(mask) for mask in result.initial_population]
    assert not any(score_better(item, result.best_evaluation) for item in initial)
    for pbest, initial_evaluation in zip(result.personal_best_evaluations, initial):
        assert not score_better(initial_evaluation, pbest)


def test_24_run_local_cache_evaluates_duplicate_masks_once() -> None:
    config = approved_config(evaluated_generations=1)
    population = np.zeros((12, 43), dtype=np.uint8)
    population[:, 0] = 1
    calls = 0

    def objective(mask: np.ndarray) -> Evaluation:
        nonlocal calls
        calls += 1
        return Evaluation(True, 0.0, 1, 0.0)

    result = bpso.BinaryParticleSwarmOptimizer(
        config, objective, constrained_better
    ).optimize(1042, initial_population=population)
    assert calls == result.unique_evaluations == 1
    assert result.total_fitness_requests == 12
    assert result.cache_hits == 11
    assert result.total_fitness_requests == result.unique_evaluations + result.cache_hits


def test_25_cache_is_isolated_between_optimizer_runs() -> None:
    config = approved_config(evaluated_generations=1)
    population = np.zeros((12, 43), dtype=np.uint8)
    population[:, 0] = 1
    calls = 0

    def objective(mask: np.ndarray) -> Evaluation:
        nonlocal calls
        calls += 1
        return Evaluation(True, 0.0, 1, 0.0)

    optimizer = bpso.BinaryParticleSwarmOptimizer(config, objective, constrained_better)
    first = optimizer.optimize(1042, initial_population=population)
    second = optimizer.optimize(1042, initial_population=population)
    assert calls == 2
    assert first.unique_evaluations == second.unique_evaluations == 1


def test_26_cache_does_not_change_optimization_decisions() -> None:
    cached_config = approved_config(evaluated_generations=8, early_stopping_min_generation=20)
    uncached_config = replace(cached_config, cache_enabled=False)
    base = bpso.BinaryParticleSwarmOptimizer(
        cached_config, cardinality_objective, constrained_better
    )
    population, velocities = base.initialize(1042)
    cached = base.optimize(
        1042, initial_population=population, initial_velocities=velocities
    )
    uncached = bpso.BinaryParticleSwarmOptimizer(
        uncached_config, cardinality_objective, constrained_better
    ).optimize(1042, initial_population=population, initial_velocities=velocities)
    assert np.array_equal(cached.best_mask, uncached.best_mask)
    assert cached.best_evaluation == uncached.best_evaluation
    assert [row.best_mask.tolist() for row in cached.convergence_history] == [
        row.best_mask.tolist() for row in uncached.convergence_history
    ]


def test_27_convergence_history_has_complete_generation_schema() -> None:
    result = bpso.BinaryParticleSwarmOptimizer(
        approved_config(evaluated_generations=3, early_stopping_min_generation=20),
        cardinality_objective,
        constrained_better,
    ).optimize(1042)
    assert [row.generation_index for row in result.convergence_history] == [0, 1, 2]
    assert all(row.request_count == 12 for row in result.convergence_history)
    expected = {
        "generation_index",
        "best_mask",
        "best_selected_feature_count",
        "best_evaluation",
        "request_count",
        "cumulative_unique_evaluations",
        "cumulative_cache_hits",
        "population_diversity",
        "global_best_improved",
        "global_best_changed",
        "repair_count",
        "inertia",
    }
    assert set(result.convergence_history[0].to_dict()) == expected


def test_28_hamming_diversity_and_degenerate_behavior() -> None:
    population = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.uint8)
    assert bpso.mean_pairwise_normalized_hamming(population) == 1.0
    assert bpso.mean_pairwise_normalized_hamming(population[:1]) == 0.0
    three = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.uint8)
    assert bpso.mean_pairwise_normalized_hamming(three) == pytest.approx(2.0 / 3.0)


def test_29_early_stopping_cannot_occur_before_generation_ten() -> None:
    def objective(mask: np.ndarray) -> Evaluation:
        return Evaluation(True, 0.0, int(mask.sum()), float(np.all(mask == 1)))

    result = bpso.BinaryParticleSwarmOptimizer(
        approved_config(), objective, score_better
    ).optimize(1042)
    assert result.stop_reason == bpso.STOP_EARLY_NO_IMPROVEMENT
    assert result.convergence_history[-1].generation_index == 10
    assert result.evaluated_generation_count == 11
    assert result.best_generation == 0


def test_30_seven_generation_patience_is_exact_after_improvement() -> None:
    config = bpso.BPSOConfig(
        dimensions=3,
        particle_count=2,
        evaluated_generations=20,
        random_cardinalities=(),
        cache_enabled=False,
        early_stopping_patience=7,
        early_stopping_min_generation=10,
    )
    calls = 0

    def objective(mask: np.ndarray) -> Evaluation:
        nonlocal calls
        generation = calls // config.particle_count
        calls += 1
        score = 1.0 if generation >= 3 else 0.0
        return Evaluation(True, 0.0, 0, score)

    population = np.ones((2, 3), dtype=np.uint8)
    velocities = np.zeros((2, 3), dtype=float)
    result = bpso.BinaryParticleSwarmOptimizer(
        config, objective, score_better
    ).optimize(1042, initial_population=population, initial_velocities=velocities)
    improved_generations = [
        row.generation_index
        for row in result.convergence_history
        if row.global_best_improved
    ]
    assert improved_generations == [0, 3]
    assert result.convergence_history[-1].generation_index == 10
    assert result.stop_reason == bpso.STOP_EARLY_NO_IMPROVEMENT


def test_31_max_generations_has_exact_request_ceiling() -> None:
    config = bpso.BPSOConfig(
        dimensions=3,
        particle_count=2,
        evaluated_generations=5,
        random_cardinalities=(),
        cache_enabled=False,
        early_stopping_min_generation=5,
    )
    result = bpso.BinaryParticleSwarmOptimizer(
        config, cardinality_objective, constrained_better
    ).optimize(1042)
    assert result.stop_reason == bpso.STOP_MAX_GENERATIONS
    assert result.evaluated_generation_count == 5
    assert result.total_fitness_requests == 10
    assert result.maximum_generations == 5


def test_32_best_generation_and_stop_reason_are_serialized() -> None:
    result = bpso.BinaryParticleSwarmOptimizer(
        approved_config(), cardinality_objective, constrained_better
    ).optimize(1042)
    payload = result.to_dict()
    assert payload["best_generation"] == result.best_generation
    assert payload["stop_reason"] in {
        bpso.STOP_MAX_GENERATIONS,
        bpso.STOP_EARLY_NO_IMPROVEMENT,
    }


def test_33_complete_result_reproduces_for_same_seed() -> None:
    optimizer = bpso.BinaryParticleSwarmOptimizer(
        approved_config(), cardinality_objective, constrained_better
    )
    first = optimizer.optimize(1042)
    second = optimizer.optimize(1042)
    assert first.to_json() == second.to_json()


def test_34_result_json_serialization_is_safe_and_deterministic() -> None:
    result = bpso.BinaryParticleSwarmOptimizer(
        approved_config(evaluated_generations=2, early_stopping_min_generation=20),
        cardinality_objective,
        constrained_better,
    ).optimize(1042)
    first = result.to_json()
    second = result.to_json()
    payload = json.loads(first)
    assert first == second
    assert payload["best_mask"] == result.best_mask.tolist()
    assert isinstance(payload["best_evaluation"], dict)
    assert payload["configuration_snapshot"]["optimizer_seed"] == 1042
    with pytest.raises(ValueError, match="Non-finite"):
        bpso.to_json_compatible(float("nan"))


def test_35_result_contains_governed_initial_final_and_pbest_state() -> None:
    result = bpso.BinaryParticleSwarmOptimizer(
        approved_config(evaluated_generations=2, early_stopping_min_generation=20),
        cardinality_objective,
        constrained_better,
    ).optimize(1042)
    assert result.initial_population.shape == (12, 43)
    assert result.initial_velocities.shape == (12, 43)
    assert result.final_population.shape == (12, 43)
    assert result.final_velocities.shape == (12, 43)
    assert result.personal_best_masks.shape == (12, 43)
    assert len(result.personal_best_evaluations) == 12


def test_36_engine_has_no_dataset_or_classifier_dependency() -> None:
    source = inspect.getsource(bpso)
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


def test_37_optimizer_records_repairs_in_history(monkeypatch: pytest.MonkeyPatch) -> None:
    config = bpso.BPSOConfig(
        dimensions=3,
        particle_count=2,
        evaluated_generations=2,
        random_cardinalities=(),
        early_stopping_min_generation=20,
    )

    def forced_repair(velocity, generator):
        return np.array([1, 0, 0], dtype=np.uint8), True, 0

    monkeypatch.setattr(bpso, "sample_binary_position", forced_repair)
    result = bpso.BinaryParticleSwarmOptimizer(
        config, cardinality_objective, constrained_better
    ).optimize(1042)
    assert result.repair_count == 2
    assert result.convergence_history[0].repair_count == 0
    assert result.convergence_history[1].repair_count == 2
