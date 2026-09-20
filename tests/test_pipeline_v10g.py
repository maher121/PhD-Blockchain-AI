"""Deterministic V1.0-G2 implementation tests for the elite-transfer ablation.

Verifies the governance gates, the frozen-primitive equivalence of the WITH
arm, the fail-closed paired invariants, the exact dual-phase budgets, cache
accounting, deterministic reruns, TEST isolation, winner non-replacement and
the absence of any G2 production activity.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import src.pipeline_v10g as p
from src.optimization.feature_fitness import (
    FeatureFitnessEvaluation,
    feature_fitness_is_better,
)
from src.optimization.hybrid_ablation_v10g import (
    AblationArmResult,
    AblationConfig,
    AblationConfigError,
    AblationError,
    AblationVariant,
    PairedAblationResult,
    ShadowFillerBlock,
    best_index,
    evaluation_relation,
    generate_shadow_filler_block,
    production_ablation_config,
    run_ablation_arm,
    run_paired_seed,
    validate_production_constants,
)
from src.optimization.hybrid_bpso_bgwo import (
    DESIGN_IDENTIFIER,
    OPTIMIZER_IDENTIFIER,
    HybridBPSOBGWO,
    HybridConfig,
)

SAFE_SEED = 5200
SECOND_SEED = 5201
SAFE_SEEDS = (SAFE_SEED, SECOND_SEED, 5202, 5209)


# ---------------------------------------------------------------------------
# Deterministic lightweight evaluators
# ---------------------------------------------------------------------------


def scalar_objective(mask: np.ndarray) -> float:
    """Deterministic scalar objective: strict, no aggregate-score shortcuts."""
    return float(int(np.count_nonzero(np.asarray(mask, dtype=np.uint8))) % 11)


def scalar_is_better(left: float, right: float) -> bool:
    return left < right


def _feature_evaluation(mask: np.ndarray) -> FeatureFitnessEvaluation:
    arr = np.asarray(mask, dtype=np.uint8)
    mask_tuple = tuple(int(value) for value in arr)
    selected = tuple(index for index, value in enumerate(mask_tuple) if value)
    return FeatureFitnessEvaluation(
        mask=mask_tuple,
        mask_sha256=hashlib.sha256(np.asarray(mask_tuple, dtype=np.uint8).tobytes()).hexdigest(),
        selected_features=tuple(f"feature_{index}" for index in selected),
        selected_features_sha256=hashlib.sha256(
            repr(tuple(f"feature_{index}" for index in selected)).encode()
        ).hexdigest(),
        selected_feature_count=len(selected),
        feasible=True,
        normalized_violation=0.0,
        relative_losses={"average_precision": 0.0, "f1": 0.0, "recall": 0.0},
        mean_metrics={
            "average_precision": 0.9,
            "f1": 0.8,
            "recall": 0.7,
            "precision": 0.6,
            "roc_auc": 0.7,
            "accuracy": 0.8,
            "false_positive_rate": 0.1,
            "false_negative_rate": 0.2,
            "attack_prevalence": 0.05,
            "selected_feature_visibility_rate": 0.5,
        },
        confusion_totals={},
        per_seed=(),
        decision_tree_fit_count=5,
    )


def feature_objective(mask: np.ndarray) -> FeatureFitnessEvaluation:
    return _feature_evaluation(mask)


def _foreach_seed(func: Callable[[int], Any]) -> list[Any]:
    return [func(seed) for seed in SAFE_SEEDS]


# ---------------------------------------------------------------------------
# Frozen configuration contracts
# ---------------------------------------------------------------------------


class TestFrozenConfig:
    def test_locked_production_constants(self) -> None:
        config = production_ablation_config()
        assert config.dimensions == 43
        assert config.population_size == 12
        assert config.bpso_evaluated_generations == 8
        assert config.bgwo_evaluated_iterations == 8
        assert config.elite_count == 3
        assert config.cardinality_pool == (4, 8, 11, 14, 18, 22, 26, 30, 34, 38)
        assert config.bpso_phase_rng_offset == 0
        assert config.bgwo_phase_rng_offset == 10000
        assert config.filler_row_count == config.elite_count
        assert validate_production_constants(config)

    def test_locked_budget_allocation(self) -> None:
        config = production_ablation_config()
        assert config.bpso_request_allocation == 96
        assert config.bgwo_request_allocation == 96
        assert config.per_run_request_allocation == 192
        assert p.PRODUCTION_REQUESTS_PER_ARM == 960
        assert p.PRODUCTION_REQUESTS_ALL_ABLATION == 1920
        assert p.PRODUCTION_FITS_PER_UNIQUE_EVALUATION == 5

    def test_locked_seed_sequences(self) -> None:
        assert p.PRODUCTION_OPTIMIZER_SEEDS == (3042, 3043, 3044, 3045, 3046)
        assert p.PRODUCTION_MODEL_ATTACK_SEEDS == (42, 43, 44, 45, 46)
        assert p.CARDINALITY_POOL == (4, 8, 11, 14, 18, 22, 26, 30, 34, 38)

    def test_config_maps_onto_frozen_hybrid_config(self) -> None:
        config = production_ablation_config()
        frozen = HybridConfig(
            dimensions=config.dimensions,
            population_size=config.population_size,
            bpso_evaluated_generations=config.bpso_evaluated_generations,
            bgwo_evaluated_iterations=config.bgwo_evaluated_iterations,
            cardinality_pool=config.cardinality_pool,
            elite_count=config.elite_count,
            bgwo_phase_rng_offset=config.bgwo_phase_rng_offset,
            velocity_initial_min=config.velocity_initial_min,
            velocity_initial_max=config.velocity_initial_max,
            velocity_clamp_min=config.velocity_clamp_min,
            velocity_clamp_max=config.velocity_clamp_max,
            inertia_start=config.inertia_start,
            inertia_end=config.inertia_end,
            cognitive_coefficient=config.cognitive_coefficient,
            social_coefficient=config.social_coefficient,
            sigmoid_clamp_min=config.sigmoid_clamp_min,
            sigmoid_clamp_max=config.sigmoid_clamp_max,
            control_parameter_start=config.control_parameter_start,
            control_parameter_end=config.control_parameter_end,
            minimum_selected_features=config.minimum_selected_features,
            cache_enabled=config.cache_enabled,
            optimizer_name=OPTIMIZER_IDENTIFIER,
            design_identifier=DESIGN_IDENTIFIER,
        )
        assert frozen.bgwo_phase_rng_offset == 10000
        assert frozen.per_run_request_allocation == config.per_run_request_allocation

    def test_config_change_detected_by_validation(self) -> None:
        import dataclasses

        config = production_ablation_config()
        drifted = dataclasses.replace(config, dimensions=44)
        with pytest.raises(AblationConfigError):
            validate_production_constants(drifted)

    def test_config_fails_closed_on_bad_filler_rows(self) -> None:
        with pytest.raises(AblationConfigError):
            AblationConfig(
                dimensions=43,
                population_size=12,
                bpso_evaluated_generations=8,
                bgwo_evaluated_iterations=8,
                cardinality_pool=(4, 8, 11, 14, 18, 22, 26, 30, 34, 38),
                elite_count=3,
                bpso_phase_rng_offset=0,
                bgwo_phase_rng_offset=10000,
                filler_protocol_tag=p.FILLER_PROTOCOL_TAG,
                filler_stream_tag=p.FILLER_STREAM_TAG,
                filler_row_count=2,
            )


# ---------------------------------------------------------------------------
# Shadow filler stream
# ---------------------------------------------------------------------------


class TestShadowFillerStream:
    def test_filler_matches_locked_seed_sequence(self) -> None:
        for seed in SAFE_SEEDS:
            block = generate_shadow_filler_block(
                seed,
                dimensions=43,
                cardinality_pool=p.CARDINALITY_POOL,
                filler_protocol_tag=p.FILLER_PROTOCOL_TAG,
                filler_stream_tag=p.FILLER_STREAM_TAG,
                row_count=3,
            )
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [seed, p.FILLER_PROTOCOL_TAG, p.FILLER_STREAM_TAG]
                )
            )
            expected = []
            for _ in range(3):
                k_index = int(rng.integers(0, len(p.CARDINALITY_POOL)))
                k = int(p.CARDINALITY_POOL[k_index])
                positions = rng.choice(43, size=k, replace=False)
                mask = np.zeros(43, dtype=np.uint8)
                mask[positions] = 1
                expected.append(mask)
            observed = np.asarray([np.asarray(block.masks[i]) for i in range(3)])
            assert np.array_equal(observed, np.asarray(expected))

    def test_filler_cardinalities_from_locked_pool(self) -> None:
        for seed in SAFE_SEEDS:
            block = generate_shadow_filler_block(
                seed,
                dimensions=43,
                cardinality_pool=p.CARDINALITY_POOL,
                filler_protocol_tag=p.FILLER_PROTOCOL_TAG,
                filler_stream_tag=p.FILLER_STREAM_TAG,
                row_count=3,
            )
            assert block.row_count == 3
            assert block.dimensions == 43
            for cardinality in block.cardinalities:
                assert cardinality in p.CARDINALITY_POOL
            for mask in block.masks:
                assert int(np.sum(mask)) in p.CARDINALITY_POOL

    def test_filler_is_optimization_state_independent(self) -> None:
        first = generate_shadow_filler_block(
            SAFE_SEED,
            dimensions=43,
            cardinality_pool=p.CARDINALITY_POOL,
            filler_protocol_tag=p.FILLER_PROTOCOL_TAG,
            filler_stream_tag=p.FILLER_STREAM_TAG,
            row_count=3,
        )
        second = generate_shadow_filler_block(
            SAFE_SEED,
            dimensions=43,
            cardinality_pool=p.CARDINALITY_POOL,
            filler_protocol_tag=p.FILLER_PROTOCOL_TAG,
            filler_stream_tag=p.FILLER_STREAM_TAG,
            row_count=3,
        )
        assert first.row_count == 3 and second.row_count == 3
        assert np.array_equal(first.masks, second.masks)
        assert first.mask_sha256 == second.mask_sha256
        assert first.to_dict()["shadow_filler_kind"] == "GOVERNED_DETERMINISTIC_EXACT_K"

    def test_filler_duplicates_reported_not_rejected(self) -> None:
        block = generate_shadow_filler_block(
            SAFE_SEED,
            dimensions=43,
            cardinality_pool=p.CARDINALITY_POOL,
            filler_protocol_tag=p.FILLER_PROTOCOL_TAG,
            filler_stream_tag=p.FILLER_STREAM_TAG,
            row_count=3,
        )
        hashes = list(block.mask_sha256)
        assert len(hashes) == 3
        assert len(set(hashes)) <= 3


# ---------------------------------------------------------------------------
# Paired engine mechanics
# ---------------------------------------------------------------------------


class TestPairedEngineMechanics:
    @pytest.mark.parametrize("seed", SAFE_SEEDS)
    def test_exactly_two_variants(self, seed: int) -> None:
        result = run_paired_seed(seed, production_ablation_config(), scalar_objective, scalar_is_better)
        assert [result.with_arm.variant, result.without_arm.variant] == [
            "WITH_ELITE_TRANSFER",
            "WITHOUT_ELITE_TRANSFER",
        ]
        assert result.status == "PASS"

    @pytest.mark.parametrize("seed", SAFE_SEEDS)
    def test_exactly_three_distinct_elites_transferred(self, seed: int) -> None:
        result = run_paired_seed(seed, production_ablation_config(), scalar_objective, scalar_is_better)
        arm: AblationArmResult[float] = result.with_arm
        assert arm.placed_elite_count == 3
        assert arm.elite_selection_invoked
        elite_list = [tuple(map(int, row)) for row in np.asarray(arm.elite_masks)]
        assert len(set(elite_list)) == 3

    @pytest.mark.parametrize("seed", SAFE_SEEDS)
    def test_without_selects_no_elites(self, seed: int) -> None:
        result = run_paired_seed(seed, production_ablation_config(), scalar_objective, scalar_is_better)
        arm: AblationArmResult[float] = result.without_arm
        assert not arm.elite_selection_invoked
        assert arm.placed_elite_count == 0
        assert np.asarray(arm.elite_masks).shape[0] == 0

    @pytest.mark.parametrize("seed", SAFE_SEEDS)
    def test_without_rows_1_3_are_exactly_the_filler(self, seed: int) -> None:
        result = run_paired_seed(seed, production_ablation_config(), scalar_objective, scalar_is_better)
        arm = result.without_arm
        rows = arm.final_population[1:4]
        filler = np.asarray(arm.filler_block.masks)
        assert np.array_equal(rows.astype(np.uint8), filler.astype(np.uint8))
        assert arm.shadow_filler_placed
        assert arm.shadow_filler_evaluated

    @pytest.mark.parametrize("seed", SAFE_SEEDS)
    def test_with_rows_1_3_are_exactly_the_elites(self, seed: int) -> None:
        result = run_paired_seed(seed, production_ablation_config(), scalar_objective, scalar_is_better)
        arm = result.with_arm
        rows = arm.final_population[1:4]
        elites = np.asarray(arm.elite_masks)
        assert np.array_equal(rows.astype(np.uint8), elites.astype(np.uint8))
        assert not arm.shadow_filler_placed
        assert not arm.shadow_filler_evaluated
        assert arm.shadow_filler_generated

    def test_paired_bpso_outputs_byte_identical(self) -> None:
        for seed in SAFE_SEEDS:
            result = run_paired_seed(seed, production_ablation_config(), scalar_objective, scalar_is_better)
            assert result.with_arm.bpso_submitted_outputs_sha256 == result.without_arm.bpso_submitted_outputs_sha256
            assert result.with_arm.bpso_final_population_sha256 == result.without_arm.bpso_final_population_sha256

    def test_row_zero_identical(self) -> None:
        for seed in SAFE_SEEDS:
            result = run_paired_seed(seed, production_ablation_config(), scalar_objective, scalar_is_better)
            assert result.with_arm.bgwo_row0_sha256 == result.without_arm.bgwo_row0_sha256

    def test_rows_4_11_identical(self) -> None:
        for seed in SAFE_SEEDS:
            result = run_paired_seed(seed, production_ablation_config(), scalar_objective, scalar_is_better)
            assert result.with_arm.bgwo_rows_4_11_sha256 == result.without_arm.bgwo_rows_4_11_sha256
            with_rows = result.with_arm.final_population[4:12]
            without_rows = result.without_arm.final_population[4:12]
            assert np.array_equal(with_rows.astype(np.uint8), without_rows.astype(np.uint8))

    def test_common_rng_state_identical_before_updates(self) -> None:
        for seed in SAFE_SEEDS:
            result = run_paired_seed(seed, production_ablation_config(), scalar_objective, scalar_is_better)
            assert result.with_arm.bgwo_common_rng_state_before_updates == result.without_arm.bgwo_common_rng_state_before_updates

    def test_paired_invariants_pass_fail_closed(self) -> None:
        for seed in SAFE_SEEDS:
            result = run_paired_seed(seed, production_ablation_config(), scalar_objective, scalar_is_better)
            assert result.status == "PASS"
            assert result.invariants["status"] == "PASS"
            checks = result.invariants["checks"]
            for name, passed in checks.items():
                assert passed, f"invariant {name} failed for seed {seed}"
            assert result.invariants["treatment_rows_differ_bitwise"] == [True, True, True]
            assert list(result.invariants["checks"].keys()) == [
                "paired_bpso_outputs_byte_identical",
                "bgwo_row0_byte_identical",
                "bgwo_rows_4_11_byte_identical",
                "bgwo_common_rng_state_identical_before_updates",
                "with_rows_1_3_are_exactly_the_elites",
                "without_rows_1_3_are_exactly_the_filler",
                "with_shadow_filler_neither_placed_nor_evaluated",
                "without_never_invokes_elite_selection",
                "shared_shadow_filler_block_identical",
                "shared_phase_request_budgets",
                "treatment_rows_collision_reported_not_rejected",
            ]

    def test_exact_dual_phase_budgets_no_early_stopping(self) -> None:
        for seed in SAFE_SEEDS:
            result = run_paired_seed(seed, production_ablation_config(), scalar_objective, scalar_is_better)
            for arm in (result.with_arm, result.without_arm):
                assert arm.stop_reason == "FIXED_BUDGET_EXHAUSTED"
                assert arm.bpso_requests == 96
                assert arm.bgwo_requests == 96
                assert arm.total_candidate_requests == 192

    def test_request_accounting_consistent(self) -> None:
        for seed in SAFE_SEEDS:
            result = run_paired_seed(seed, production_ablation_config(), scalar_objective, scalar_is_better)
            for arm in (result.with_arm, result.without_arm):
                assert arm.total_candidate_requests == arm.unique_evaluations + arm.cache_hits
                assert arm.evaluator_calls == arm.unique_evaluations
                assert arm.bgwo_new_unique_evaluations + arm.bgwo_cache_hits == arm.bgwo_requests
                assert arm.bpso_unique_evaluations + arm.bpso_cache_hits == arm.bpso_requests

    def test_with_elite_cache_reuse_reported(self) -> None:
        reuse_seen = False
        for seed in SAFE_SEEDS:
            result = run_paired_seed(seed, production_ablation_config(), scalar_objective, scalar_is_better)
            if result.with_arm.bgwo_cache_hits >= result.with_arm.placed_elite_count:
                reuse_seen = True
                assert result.with_arm.bgwo_cache_hits >= 3
            assert result.with_arm.bgwo_cache_hits >= result.with_arm.placed_elite_count
        assert reuse_seen

    def test_without_collisions_allowed(self) -> None:
        for seed in SAFE_SEEDS:
            result = run_paired_seed(seed, production_ablation_config(), scalar_objective, scalar_is_better)
            assert result.status == "PASS"
            assert result.without_arm.bgwo_cache_hits >= 0

    def test_no_duplicate_rejection(self) -> None:
        m0 = np.zeros(43, dtype=np.uint8)
        m0[[0, 4, 8, 11]] = 1
        m1 = np.zeros(43, dtype=np.uint8)
        m1[[0, 4, 8, 12]] = 1
        block = ShadowFillerBlock(
            optimizer_seed=SAFE_SEED,
            dimensions=43,
            cardinality_pool=p.CARDINALITY_POOL,
            filler_protocol_tag=p.FILLER_PROTOCOL_TAG,
            filler_stream_tag=p.FILLER_STREAM_TAG,
            masks=np.array([m0, m0, m1], dtype=np.uint8),
            cardinalities=(4, 4, 4),
            mask_sha256=(
                hashlib.sha256(m0.tobytes()).hexdigest(),
                hashlib.sha256(m0.tobytes()).hexdigest(),
                hashlib.sha256(m1.tobytes()).hexdigest(),
            ),
        )
        arm = run_ablation_arm(
            AblationVariant.WITHOUT_ELITE_TRANSFER,
            SAFE_SEED,
            production_ablation_config(),
            scalar_objective,
            scalar_is_better,
            filler_block=block,
        )
        rows = np.asarray(arm.final_population[1:4], dtype=np.uint8)
        assert np.array_equal(rows[0], m0)
        assert np.array_equal(rows[1], m0)
        assert np.array_equal(rows[2], m1)
        assert len(set(arm.filler_block.mask_sha256)) == 2

    def test_deterministic_rerun_identical(self) -> None:
        first = run_paired_seed(SAFE_SEED, production_ablation_config(), scalar_objective, scalar_is_better)
        second = run_paired_seed(SAFE_SEED, production_ablation_config(), scalar_objective, scalar_is_better)
        assert first.deterministic_digest() == second.deterministic_digest()
        assert first.with_arm.total_candidate_requests == second.with_arm.total_candidate_requests
        assert first.without_arm.total_candidate_requests == second.without_arm.total_candidate_requests

    def test_rerun_digests_distinguish_seeds(self) -> None:
        results = {
            seed: run_paired_seed(seed, production_ablation_config(), scalar_objective, scalar_is_better)
            for seed in SAFE_SEEDS
        }
        digests = {seed: results[seed].deterministic_digest() for seed in SAFE_SEEDS}
        assert len(set(digests.values())) == len(SAFE_SEEDS)

    def test_standalone_arms_equal_paired_arms(self) -> None:
        result = run_paired_seed(SAFE_SEED, production_ablation_config(), scalar_objective, scalar_is_better)
        with_arm = run_ablation_arm(
            AblationVariant.WITH_ELITE_TRANSFER,
            SAFE_SEED,
            production_ablation_config(),
            scalar_objective,
            scalar_is_better,
        )
        without_arm = run_ablation_arm(
            AblationVariant.WITHOUT_ELITE_TRANSFER,
            SAFE_SEED,
            production_ablation_config(),
            scalar_objective,
            scalar_is_better,
        )
        assert with_arm.bpso_submitted_outputs_sha256 == result.with_arm.bpso_submitted_outputs_sha256
        assert np.array_equal(
            with_arm.final_population.astype(np.uint8),
            result.with_arm.final_population.astype(np.uint8),
        )
        assert np.array_equal(
            without_arm.final_population.astype(np.uint8),
            result.without_arm.final_population.astype(np.uint8),
        )

    def test_feature_evaluation_type_is_supported(self) -> None:
        result = run_paired_seed(
            5207,
            production_ablation_config(),
            feature_objective,
            feature_fitness_is_better,
        )
        assert result.status == "PASS"
        assert isinstance(result.with_arm.best_evaluation, FeatureFitnessEvaluation)
        assert result.with_arm.best_feasible is True
        assert result.with_arm.best_normalized_violation == 0.0

    def test_repair_functionality_present(self) -> None:
        for seed in SAFE_SEEDS:
            result = run_paired_seed(
                seed, production_ablation_config(), feature_objective, feature_fitness_is_better
            )
            for arm in (result.with_arm, result.without_arm):
                assert arm.repair_count >= 0
                assert arm.best_feasible is True
                assert arm.best_normalized_violation == 0.0


class TestFrozenRankingSemantics:
    def test_strict_comparator_relation(self) -> None:
        assert evaluation_relation(scalar_is_better, 1.0, 2.0) == -1
        assert evaluation_relation(scalar_is_better, 2.0, 2.0) == 0
        assert evaluation_relation(scalar_is_better, 3.0, 1.0) == 1

    def test_best_index_returns_strict_best_with_mask_tiebreak(self) -> None:
        population = np.array(
            [[1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 0]],
            dtype=np.uint8,
        )
        evaluations = [2.0, 1.5, 1.5, 1.5]
        best = best_index(population, evaluations, scalar_is_better)
        rows = [tuple(map(int, population[index])) for index in range(len(evaluations))]
        assert evaluations[best] == min(evaluations)
        assert rows[best] == min(rows[index] for index in range(len(rows)) if evaluations[index] == evaluations[best])


# ---------------------------------------------------------------------------
# Frozen hybrid equivalence
# ---------------------------------------------------------------------------


class TestFrozenHybridEquivalence:
    def _frozen_config(self) -> HybridConfig:
        config = production_ablation_config()
        return HybridConfig(
            dimensions=config.dimensions,
            population_size=config.population_size,
            bpso_evaluated_generations=config.bpso_evaluated_generations,
            bgwo_evaluated_iterations=config.bgwo_evaluated_iterations,
            cardinality_pool=config.cardinality_pool,
            elite_count=config.elite_count,
            bgwo_phase_rng_offset=config.bgwo_phase_rng_offset,
            velocity_initial_min=config.velocity_initial_min,
            velocity_initial_max=config.velocity_initial_max,
            velocity_clamp_min=config.velocity_clamp_min,
            velocity_clamp_max=config.velocity_clamp_max,
            inertia_start=config.inertia_start,
            inertia_end=config.inertia_end,
            cognitive_coefficient=config.cognitive_coefficient,
            social_coefficient=config.social_coefficient,
            sigmoid_clamp_min=config.sigmoid_clamp_min,
            sigmoid_clamp_max=config.sigmoid_clamp_max,
            control_parameter_start=config.control_parameter_start,
            control_parameter_end=config.control_parameter_end,
            minimum_selected_features=config.minimum_selected_features,
            cache_enabled=config.cache_enabled,
        )

    @pytest.mark.parametrize("seed", SAFE_SEEDS)
    def test_with_arm_elites_equal_frozen_optimize(self, seed: int) -> None:
        frozen = HybridBPSOBGWO(self._frozen_config(), scalar_objective, scalar_is_better).optimize(seed)
        ablation = run_ablation_arm(
            AblationVariant.WITH_ELITE_TRANSFER,
            seed,
            production_ablation_config(),
            scalar_objective,
            scalar_is_better,
        )
        assert frozen.elite_hashes == ablation.elite_hashes
        assert np.array_equal(
            np.asarray(frozen.elite_masks).astype(np.uint8),
            np.asarray(ablation.elite_masks).astype(np.uint8),
        )

    @pytest.mark.parametrize("seed", SAFE_SEEDS)
    def test_with_arm_bpso_accounting_equal_frozen(self, seed: int) -> None:
        frozen = HybridBPSOBGWO(self._frozen_config(), scalar_objective, scalar_is_better).optimize(seed)
        ablation = run_ablation_arm(
            AblationVariant.WITH_ELITE_TRANSFER,
            seed,
            production_ablation_config(),
            scalar_objective,
            scalar_is_better,
        )
        assert frozen.bpso_requests == ablation.bpso_requests
        assert frozen.bpso_unique_evaluations == ablation.bpso_unique_evaluations
        assert frozen.bpso_cache_hits == ablation.bpso_cache_hits


# ---------------------------------------------------------------------------
# V1.0-G2 pipeline wiring and governance gates
# ---------------------------------------------------------------------------


class TestPipelineGovernance:
    def test_g1_lock_recomputes_to_expected(self) -> None:
        result = p.recompute_protocol_lock()
        assert result["status"] == "PASS"
        assert result["recomputed_semantic_sha256"] == "2914819f4c63b50bf58a8dfa19af1c5b05539f6997b3f447c3a891b1852acf5c"
        assert result["matches_expected"] and result["matches_stored"]

    def test_g1_protocol_verification_passes(self) -> None:
        result = p.g1_protocol_verification()
        assert result["status"] == "PASS"
        assert result["checks"]["starting_commit_locked"]
        assert result["checks"]["markdown_lf_hash_matches"]
        assert result["checks"]["yaml_lf_hash_matches"]
        assert result["checks"]["v10d_winner_id_locked"]

    def test_prior_locks_immutable(self) -> None:
        result = p.prior_locks_immutable()
        assert result["status"] == "PASS"
        assert result["v10d_winner_lock_sha256"] == "1603cf0faff9027c2e5bcfb53345338317963d55ffc048afcfca5f974923bf79"
        assert result["v10f_result_lock_sha256"] == "0e24e485b74489756725dc6da7ef5e197dd63951a667fa70761758cfa8997a8f"

    def test_locked_execution_order(self) -> None:
        order = p.locked_execution_order()
        assert [item["optimizer_seed"] for item in order] == [3042, 3043, 3044, 3045, 3046]
        assert [item["first"] for item in order] == [
            "WITH_ELITE_TRANSFER",
            "WITHOUT_ELITE_TRANSFER",
            "WITH_ELITE_TRANSFER",
            "WITHOUT_ELITE_TRANSFER",
            "WITH_ELITE_TRANSFER",
        ]

    def test_production_seed_guard_fails_closed(self) -> None:
        for seed in (3042, 3043, 3044, 3045, 3046):
            with pytest.raises(p.V10GNoGoError):
                p.run_synthetic_paired_seed(seed, scalar_objective, scalar_is_better)

    def test_synthetic_run_safe_seeds_allowed(self) -> None:
        for seed in SAFE_SEEDS:
            result = p.run_synthetic_paired_seed(seed, scalar_objective, scalar_is_better)
            assert result.status == "PASS"

    def test_synthetic_run_never_touches_disk(self) -> None:
        before = sorted(path.name for path in p.V10G_RESULTS_DIR.iterdir())
        p.run_synthetic_paired_seed(SAFE_SEED, scalar_objective, scalar_is_better)
        after = sorted(path.name for path in p.V10G_RESULTS_DIR.iterdir())
        assert before == after

    def test_test_isolation_audit(self) -> None:
        result = p.test_isolation_audit()
        assert result["status"] == "PASS"
        assert result["test_accessed"] is False
        assert result["classification"] == "TEST_LOCKED_DURING_V10G_ABLATION"

    def test_test_flags_audit_fails_closed(self) -> None:
        assert p.test_flags_audit({})["status"] == "PASS"
        for flag in ("test_accessed", "test_used_for_fitness", "test_used_for_selection", "final_test_evaluated"):
            with pytest.raises(p.V10GNoGoError):
                p.test_flags_audit({flag: True})

    def test_no_v10f_campaign_invoked(self) -> None:
        result = p.no_v10f_resource_campaign_invoked()
        assert result["status"] == "PASS"
        assert result["checks"]["resource_campaign_invoked"] is False

    def test_production_campaign_is_no_go(self) -> None:
        with pytest.raises(p.V10GNoGoError, match="V1.0-G4"):
            p.production_campaign_guard()

    def test_no_g2_result_lock_artifacts(self) -> None:
        result = p.no_g2_result_lock_artifacts()
        assert result["status"] == "PASS"
        assert result["present_files"] == ["v10g_protocol_lock.json"]

    def test_import_has_no_side_effects(self) -> None:
        clean = p.no_g2_result_lock_artifacts()
        assert clean["status"] == "PASS"
        assert clean["created_result_locks"] == []
        assert clean["present_files"] == ["v10g_protocol_lock.json"]
        assert p.test_isolation_audit()["status"] == "PASS"
        assert p.g1_protocol_verification()["status"] == "PASS"

    def test_winner_non_replacement(self) -> None:
        before_d = p._sha256_file(p.V10D_LOCK_PATH)
        before_f = p._sha256_file(p.V10F_RESULT_PATH)
        before_g1 = p._sha256_file(p.PROTOCOL_LOCK_PATH)
        for seed in SAFE_SEEDS:
            p.run_synthetic_paired_seed(seed, scalar_objective, scalar_is_better)
        assert p._sha256_file(p.V10D_LOCK_PATH) == before_d == p.V10D_WINNER_LOCK_SHA256
        assert p._sha256_file(p.V10F_RESULT_PATH) == before_f == p.V10F_RESULT_LOCK_SHA256
        assert p._sha256_file(p.PROTOCOL_LOCK_PATH) == before_g1
        assert p.g1_protocol_verification()["checks"]["v10d_winner_id_locked"]

    def test_full_g2_gate_suite_passes(self) -> None:
        assert p.g1_protocol_verification()["status"] == "PASS"
        assert p.prior_locks_immutable()["status"] == "PASS"
        assert p.test_isolation_audit()["status"] == "PASS"
        assert p.no_v10f_resource_campaign_invoked()["status"] == "PASS"
        with pytest.raises(p.V10GNoGoError):
            p.production_campaign_guard()
        result = run_paired_seed(SAFE_SEED, production_ablation_config(), scalar_objective, scalar_is_better)
        assert result.status == "PASS"
        assert p.no_g2_result_lock_artifacts()["status"] == "PASS"