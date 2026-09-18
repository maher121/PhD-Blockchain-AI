"""Synthetic protocol tests for the GOVERNED hybrid BPSO+BGWO engine (V1.0-B).

Pure, dataset-free tests. Every optimization run below executes a synthetic
objective to verify the locked V1.0-A design contract; the production budget
(5 runs x 192 requests = 960) is verified only as configuration arithmetic and
never executed here.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from src.optimization import hybrid_bpso_bgwo as hybrid
from src.optimization.hybrid_bpso_bgwo import (
    HybridBPSOBGWO,
    HybridConfig,
    RunLocalCache,
    build_bgwo_initial_population,
    build_hybrid_initial_population,
    hybrid_config_from_yaml,
    select_elites,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
HYBRID_YAML = REPO / "config" / "hybrid_v10.yaml"
MODULE_PATH = REPO / "src" / "optimization" / "hybrid_bpso_bgwo.py"

HYBRID_YAML_SHA256 = "0ac491b50abeb60e65efc2295eb88c03b4c988a5182ff0e2e3ed729deb9367d8"
PROTOCOL_CLASSIFICATION = "HYBRID_PROTOCOL_LOCKED_FAIR_BUDGET"
DESIGN_IDENTIFIER = "BPSO_BGWO_SEQUENTIAL_50_50_ELITE3"
OPTIMIZER_IDENTIFIER = "GOVERNED_HYBRID_BPSO_BGWO"

BPSO_K10_MASK_SHA256 = (
    "5da981b5b87db97338ecdde9ca8a8b87db3a62771d03dc6a4ad901f6548a3299"
)
BGWO_K14_MASK_SHA256 = (
    "7ebb823374255f4f10c737c62a2111604aa50193a8f0d91b8483f3864cac7ad6"
)

PRIMARY_SEED_FAMILY = (3042, 3043, 3044, 3045, 3046)
MODEL_SEED_FAMILY = (42, 43, 44, 45, 46)
FORBIDDEN_SEED_FAMILIES = ((1042, 1043, 1044, 1045, 1046), (2042, 2043, 2044, 2045, 2046))
BGWO_RNG_OFFSET = 10000

REFERENCE_AP = 0.23068406113411433
REFERENCE_F1 = 0.1977010726398371
REFERENCE_RECALL = 0.322
MARGINS = {"average_precision": 0.05, "f1": 0.05, "recall": 0.10}

MANDATORY_FEATURES = (0, 1)
FEATURE_GAINS = {
    2: (0.010, 0.008, 0.012),
    5: (0.015, 0.010, 0.020),
    7: (0.020, 0.015, 0.010),
    11: (0.025, 0.020, 0.015),
    13: (0.010, 0.025, 0.020),
    19: (0.005, 0.010, 0.030),
    23: (0.030, 0.005, 0.010),
    29: (0.020, 0.030, 0.005),
    31: (0.010, 0.010, 0.025),
    37: (0.005, 0.020, 0.015),
}


@dataclass(frozen=True)
class SyntheticFeatureEvaluation:
    average_precision: float
    f1: float
    recall: float
    selected_feature_count: int
    normalized_violation: float
    feasible: bool
    mask: tuple[int, ...]


def _normalized_violation(ap: float, f1: float, recall: float) -> float:
    losses = {
        "average_precision": (REFERENCE_AP - ap) / REFERENCE_AP,
        "f1": (REFERENCE_F1 - f1) / REFERENCE_F1,
        "recall": (REFERENCE_RECALL - recall) / REFERENCE_RECALL,
    }
    return float(
        sum(max(0.0, (losses[key] - MARGINS[key]) / MARGINS[key]) for key in MARGINS)
    )


def synthetic_feature_fitness(mask: np.ndarray) -> SyntheticFeatureEvaluation:
    """Deterministic synthetic fitness mirroring the frozen constrained semantics."""
    binary = np.asarray(mask, dtype=np.uint8).flatten()
    if not np.isin(binary, (0, 1)).all():
        raise ValueError("synthetic objective requires a binary mask")
    k = int(binary.sum())
    gain_ap = gain_f1 = gain_rec = 0.0
    for feature in np.flatnonzero(binary):
        if int(feature) in FEATURE_GAINS:
            g = FEATURE_GAINS[int(feature)]
            gain_ap += g[0]
            gain_f1 += g[1]
            gain_rec += g[2]
    feasible = all(binary[i] == 1 for i in MANDATORY_FEATURES)
    if feasible:
        ap = REFERENCE_AP + gain_ap
        f1 = REFERENCE_F1 + gain_f1
        recall = REFERENCE_RECALL + gain_rec
        violation = 0.0
    else:
        ap = 0.60 * REFERENCE_AP + gain_ap
        f1 = 0.60 * REFERENCE_F1 + gain_f1
        recall = 0.60 * REFERENCE_RECALL + gain_rec
        violation = _normalized_violation(ap, f1, recall)
    return SyntheticFeatureEvaluation(
        average_precision=ap,
        f1=f1,
        recall=recall,
        selected_feature_count=k,
        normalized_violation=violation,
        feasible=bool(feasible),
        mask=tuple(int(value) for value in binary),
    )


def synthetic_is_better(
    left: SyntheticFeatureEvaluation, right: SyntheticFeatureEvaluation
) -> bool:
    """Mirror feature_fitness_is_better: feasible-first deterministic ranking."""
    if left.feasible != right.feasible:
        return left.feasible
    if left.feasible:
        left_key = (
            left.selected_feature_count,
            -left.average_precision,
            -left.f1,
            -left.recall,
            left.mask,
        )
        right_key = (
            right.selected_feature_count,
            -right.average_precision,
            -right.f1,
            -right.recall,
            right.mask,
        )
    else:
        left_key = (
            left.normalized_violation,
            -left.average_precision,
            -left.f1,
            -left.recall,
            left.selected_feature_count,
            left.mask,
        )
        right_key = (
            right.normalized_violation,
            -right.average_precision,
            -right.f1,
            -right.recall,
            right.selected_feature_count,
            right.mask,
        )
    return left_key < right_key


def flat_objective(mask: np.ndarray) -> SyntheticFeatureEvaluation:
    binary = np.asarray(mask, dtype=np.uint8).flatten()
    return SyntheticFeatureEvaluation(
        average_precision=REFERENCE_AP,
        f1=REFERENCE_F1,
        recall=REFERENCE_RECALL,
        selected_feature_count=int(binary.sum()),
        normalized_violation=0.0,
        feasible=True,
        mask=tuple(int(value) for value in binary),
    )


def reduced_config(bpso_generations: int = 4, bgwo_iterations: int = 4) -> HybridConfig:
    return replace(
        hybrid_config_from_yaml(),
        bpso_evaluated_generations=bpso_generations,
        bgwo_evaluated_iterations=bgwo_iterations,
    )


def make_counting_objective():
    state = {"calls": 0, "masks": []}

    def objective(mask):
        state["calls"] += 1
        result = synthetic_feature_fitness(mask)
        state["masks"].append(np.asarray(mask, dtype=np.uint8).copy())
        return result

    return objective, state


def mask_digest(mask: np.ndarray | list | tuple) -> str:
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Imports / purity
# ---------------------------------------------------------------------------


def test_module_imports_without_dataset_access():
    from src.optimization import hybrid_bpso_bgwo  # noqa: F401


def test_module_exports_lock_identifiers():
    assert hybrid.OPTIMIZER_IDENTIFIER == OPTIMIZER_IDENTIFIER
    assert hybrid.DESIGN_IDENTIFIER == DESIGN_IDENTIFIER
    assert hybrid.STOP_FIXED_BUDGET_EXHAUSTED == "FIXED_BUDGET_EXHAUSTED"


def test_module_ast_import_allowlist():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    allowed_prefixes = (
        "__future__",
        "numpy",
        "hashlib",
        "json",
        "re",
        "math",
        "dataclasses",
        "pathlib",
        "typing",
        "yaml",
        "src.optimization.bpso",
        "src.optimization.bgwo",
    )
    for name in imported:
        assert name.startswith(allowed_prefixes), f"forbidden hybrid import: {name}"


def test_module_source_free_of_forbidden_semantics():
    source = MODULE_PATH.read_text(encoding="utf-8")
    for token in (
        "energy",
        "rss",
        "wall_time",
        "pandas",
        "sklearn",
        "is_attack",
        "Late_delivery",
        "final_test",
        "test_accessed",
        "predict",
        "decision_tree",
    ):
        assert token not in source.lower(), f"forbidden token leaked into hybrid: {token}"


def test_module_source_does_not_hardcode_frozen_winners():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert BPSO_K10_MASK_SHA256 not in source
    assert BGWO_K14_MASK_SHA256 not in source


# ---------------------------------------------------------------------------
# Protocol / configuration
# ---------------------------------------------------------------------------


def test_yaml_hash_unchanged():
    digest = hashlib.sha256(HYBRID_YAML.read_bytes()).hexdigest()
    assert digest == HYBRID_YAML_SHA256


def test_protocol_classification_frozen():
    import yaml

    payload = yaml.safe_load(HYBRID_YAML.read_text(encoding="utf-8"))
    assert payload["protocol_classification"] == PROTOCOL_CLASSIFICATION
    assert payload["scientific_experiment"] is False


def test_config_loads_locked_constants():
    config = hybrid_config_from_yaml()
    assert config.dimensions == 43
    assert config.population_size == 12
    assert config.bpso_evaluated_generations == 8
    assert config.bgwo_evaluated_iterations == 8
    assert config.elite_count == 3
    assert config.bgwo_phase_rng_offset == BGWO_RNG_OFFSET
    assert config.cache_enabled is True
    assert config.cardinality_pool == (4, 8, 11, 14, 18, 22, 26, 30, 34, 38)


def test_production_budget_arithmetic():
    config = hybrid_config_from_yaml()
    assert config.bpso_request_allocation == 12 * 8 == 96
    assert config.bgwo_request_allocation == 12 * 8 == 96
    assert config.per_run_request_allocation == 192
    assert config.five_run_request_allocation == 960
    assert config.phase_budget_ratio == (96, 96)


def test_production_fits_cap_4800():
    config = hybrid_config_from_yaml()
    assert config.five_run_request_allocation * 5 == 4800


def test_seed_families_respect_lock():
    import yaml

    payload = yaml.safe_load(HYBRID_YAML.read_text(encoding="utf-8"))
    seeds = payload["seeds"]
    assert seeds["optimizer_seed_family_primary"] == list(PRIMARY_SEED_FAMILY)
    assert seeds["optimizer_seed_family_ablation"] == list(PRIMARY_SEED_FAMILY)
    assert seeds["model_attack_seeds"] == list(MODEL_SEED_FAMILY)
    assert seeds["dataset_split_seed"] == 42
    assert seeds["optimizer_seed_phase_rng_offset_bgwo"] == BGWO_RNG_OFFSET


def test_seed_families_do_not_collide():
    primary = set(PRIMARY_SEED_FAMILY)
    for family in FORBIDDEN_SEED_FAMILIES:
        assert not (primary & set(family))
    assert not (primary & set(MODEL_SEED_FAMILY))


def test_config_validates_bad_inputs():
    base = hybrid_config_from_yaml()
    with pytest.raises(ValueError):
        replace(base, dimensions=1)
    with pytest.raises(ValueError):
        replace(base, population_size=2)
    with pytest.raises(ValueError):
        replace(base, bpso_evaluated_generations=0)
    with pytest.raises(ValueError):
        replace(base, elite_count=0)
    with pytest.raises(ValueError):
        replace(base, bgwo_phase_rng_offset=0)
    with pytest.raises(ValueError):
        replace(base, cache_enabled=False)
    with pytest.raises(ValueError):
        replace(base, minimum_selected_features=2)
    with pytest.raises(ValueError):
        replace(base, velocity_initial_min=-8.0, velocity_initial_max=-7.0)
    with pytest.raises(ValueError):
        replace(base, inertia_start=0.2, inertia_end=0.9)


def test_config_snapshot_reports_fixed_budget():
    config = reduced_config(4, 4)
    snapshot = config.to_dict()
    assert snapshot["optimizer_name"] == OPTIMIZER_IDENTIFIER
    assert snapshot["design_identifier"] == DESIGN_IDENTIFIER
    assert snapshot["early_stopping"] == "disabled (fixed-budget execution locked by V1.0-A)"
    assert snapshot["bpso_request_allocation"] == 48
    assert snapshot["bgwo_request_allocation"] == 48


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_hybrid_initial_population_anchor_and_pool():
    config = hybrid_config_from_yaml()
    rng = np.random.Generator(np.random.PCG64(PRIMARY_SEED_FAMILY[0]))
    population = build_hybrid_initial_population(config, rng)
    assert population.shape == (12, 43)
    assert np.all(population[0] == 1)
    for row in population[1:]:
        assert int(row.sum()) in config.cardinality_pool
    assert np.isin(population, (0, 1)).all()


def test_hybrid_initial_population_requires_pcg64():
    config = reduced_config()
    with pytest.raises(TypeError):
        build_hybrid_initial_population(config, np.random.Generator(np.random.MT19937(1)))
    with pytest.raises(TypeError):
        build_hybrid_initial_population(config, np.random.RandomState(1))
    with pytest.raises(TypeError):
        build_hybrid_initial_population(config, "not-an-rng")


def test_hybrid_initial_population_never_seeds_frozen_winners():
    config = hybrid_config_from_yaml()
    rng = np.random.Generator(np.random.PCG64(PRIMARY_SEED_FAMILY[0]))
    population = build_hybrid_initial_population(config, rng)
    for row in population:
        assert mask_digest(row) != BPSO_K10_MASK_SHA256
        assert mask_digest(row) != BGWO_K14_MASK_SHA256


def test_bgwo_initial_population_places_elites_up_to_three():
    config = hybrid_config_from_yaml()
    rng = np.random.Generator(np.random.PCG64(PRIMARY_SEED_FAMILY[0]))
    elites = build_hybrid_initial_population(config, rng)[:3]
    population = build_bgwo_initial_population(config, elites, rng)
    assert population.shape == (12, 43)
    assert np.all(population[0] == 1)
    for slot in range(3):
        assert np.array_equal(population[slot + 1], elites[slot])


def test_bgwo_initial_population_pads_missing_elites():
    config = hybrid_config_from_yaml()
    rng = np.random.Generator(np.random.PCG64(PRIMARY_SEED_FAMILY[0] + BGWO_RNG_OFFSET))
    one_elite = np.ones(43, dtype=np.uint8)
    one_elite[40] = 0
    population = build_bgwo_initial_population(config, one_elite, rng)
    assert np.array_equal(population[1], one_elite)
    for row in range(2, 12):
        assert int(population[row].sum()) in config.cardinality_pool


def test_bgwo_initial_population_maintains_binary_and_min_one():
    config = reduced_config()
    rng = np.random.Generator(np.random.PCG64(PRIMARY_SEED_FAMILY[0]))
    population = build_bgwo_initial_population(config, [], rng)
    assert np.isin(population, (0, 1)).all()
    assert population.sum(axis=1).min() >= 1


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_counts_requests_unique_hits():
    cache = RunLocalCache(synthetic_feature_fitness)
    mask_a = np.array([1, 1] + [0] * 41, dtype=np.uint8)
    mask_b = np.array([1, 0] + [1] + [0] * 40, dtype=np.uint8)
    cache.request(mask_a)
    cache.request(mask_a)
    cache.request(mask_b)
    cache.request(mask_a)
    assert cache.total_requests == 4
    assert cache.unique_evaluations == 2
    assert cache.cache_hits == 2
    assert cache.evaluator_calls == 2


def test_cache_reuses_recorded_evaluation():
    cache = RunLocalCache(synthetic_feature_fitness)
    mask = np.array([1, 1] + [0] * 41, dtype=np.uint8)
    first = cache.request(mask)
    second = cache.request(mask)
    assert first is second
    assert first.mask == tuple(int(v) for v in mask)


def test_cache_is_run_local():
    cache_a = RunLocalCache(synthetic_feature_fitness)
    cache_b = RunLocalCache(synthetic_feature_fitness)
    mask = np.array([1, 1] + [0] * 41, dtype=np.uint8)
    cache_a.request(mask)
    assert cache_b.total_requests == 0
    assert cache_b.unique_evaluations == 0


def test_cache_key_matches_sha256_convention():
    from src.optimization.bgwo import canonical_mask_key

    mask = np.array([1, 1, 0, 1, 1], dtype=np.uint8)
    assert canonical_mask_key(mask) == mask.astype(np.uint8).tobytes()
    assert hashlib.sha256(mask.tobytes()).hexdigest() == mask_digest(mask)


def test_cache_rejects_non_binary_mask():
    cache = RunLocalCache(synthetic_feature_fitness)
    with pytest.raises(ValueError):
        cache.request(np.array([2, 0, 0, 0, 0], dtype=np.uint8))


# ---------------------------------------------------------------------------
# Elite selection
# ---------------------------------------------------------------------------


def test_elite_selection_top3_distinct():
    cache = RunLocalCache(synthetic_feature_fitness)
    seed_rng = np.random.Generator(np.random.PCG64(PRIMARY_SEED_FAMILY[0]))
    candidates = build_hybrid_initial_population(reduced_config(), seed_rng)
    for mask in candidates:
        cache.request(mask)
    elites = select_elites(cache, 3, synthetic_is_better)
    assert elites.shape == (3, 43)
    keys = {mask_digest(elites[i]) for i in range(3)}
    assert len(keys) == 3


def test_elite_selection_feasible_first():
    cache = RunLocalCache(synthetic_feature_fitness)
    feasible = np.array([1, 1, 1] + [0] * 40, dtype=np.uint8)
    infeasible = np.array([1, 0, 1] + [0] * 40, dtype=np.uint8)
    cache.request(feasible)
    cache.request(infeasible)
    elites = select_elites(cache, 2, synthetic_is_better)
    first = cache.evaluation_for_mask(elites[0])
    second = cache.evaluation_for_mask(elites[1])
    assert first.feasible is True
    assert second.feasible is False
    assert synthetic_is_better(first, second)


def test_elite_selection_zero_evaluator_calls():
    objective, state = make_counting_objective()
    cache = RunLocalCache(objective)
    seed_rng = np.random.Generator(np.random.PCG64(PRIMARY_SEED_FAMILY[1]))
    candidates = build_hybrid_initial_population(reduced_config(2, 2), seed_rng)
    for mask in candidates:
        cache.request(mask)
    before = state["calls"]
    elites = select_elites(cache, 3, synthetic_is_better)
    assert state["calls"] == before
    assert elites.shape == (3, 43)


def test_elite_selection_results_were_evaluated_in_run_cache():
    objective, state = make_counting_objective()
    cache = RunLocalCache(objective)
    seed_rng = np.random.Generator(np.random.PCG64(PRIMARY_SEED_FAMILY[2]))
    candidates = build_hybrid_initial_population(reduced_config(2, 2), seed_rng)
    for mask in candidates:
        cache.request(mask)
    elites = select_elites(cache, 3, synthetic_is_better)
    evaluated = {tuple(int(v) for v in m) for m in state["masks"]}
    for row in elites:
        assert tuple(int(v) for v in row) in evaluated


# ---------------------------------------------------------------------------
# Full optimize accounting (reduced budget)
# ---------------------------------------------------------------------------


def run_reduced(seed: int = PRIMARY_SEED_FAMILY[0], config: HybridConfig | None = None):
    engine = HybridBPSOBGWO(config or reduced_config(), synthetic_feature_fitness, synthetic_is_better)
    return engine.optimize(seed), engine


def test_run_stop_reason_and_budget():
    result, _ = run_reduced()
    assert result.stop_reason == "FIXED_BUDGET_EXHAUSTED"
    assert result.total_candidate_requests == 96
    assert result.bpso_requests == 48
    assert result.bgwo_requests == 48


def test_run_accounting_invariants():
    result, _ = run_reduced()
    assert result.total_candidate_requests == result.unique_evaluations + result.cache_hits
    assert result.evaluator_calls == result.unique_evaluations
    assert result.total_candidate_requests == result.bpso_requests + result.bgwo_requests


def test_run_unique_accounting_matches_cache_statistics():
    result, engine = run_reduced()
    assert result.bpso_requests == 48
    assert result.bgwo_requests == 48
    assert result.bpso_cache_hits + result.bgwo_cache_hits == result.cache_hits
    assert result.bpso_unique_evaluations + result.bgwo_new_unique_evaluations == result.unique_evaluations


def test_run_elites_transferred_without_hidden_evaluation():
    result, _ = run_reduced()
    placed = result.configuration_snapshot["placed_elite_count"]
    assert placed == 3
    assert result.bgwo_cache_hits >= placed


def test_run_shared_cache_reduced_bgwo_unique():
    result, _ = run_reduced()
    assert result.bgwo_new_unique_evaluations + result.bgwo_cache_hits == result.bgwo_requests
    assert result.bgwo_cache_hits >= 3


def test_run_flat_landscape_consumes_full_fixed_budget():
    engine = HybridBPSOBGWO(reduced_config(6, 6), flat_objective, synthetic_is_better)
    result = engine.optimize(PRIMARY_SEED_FAMILY[0])
    assert result.total_candidate_requests == 144
    assert result.stop_reason == "FIXED_BUDGET_EXHAUSTED"
    assert result.unique_evaluations + result.cache_hits == 144


def test_run_best_evaluation_reflects_run_local_optimum():
    result, _ = run_reduced()
    assert np.isin(result.best_mask, (0, 1)).all()
    assert result.best_selected_feature_count == int(result.best_mask.sum())
    assert result.best_evaluation.selected_feature_count == result.best_selected_feature_count
    assert result.best_evaluation.feasible is True


def test_run_history_phase_order_and_length():
    result, _ = run_reduced()
    assert len(result.convergence_history) == 8
    phases = [record.phase for record in result.convergence_history]
    assert phases == ["BPSO"] * 4 + ["BGWO"] * 4


def test_run_history_cumulative_counts_monotone():
    result, _ = run_reduced()
    requests = [record.cumulative_requests for record in result.convergence_history]
    uniques = [record.cumulative_unique_evaluations for record in result.convergence_history]
    hits = [record.cumulative_cache_hits for record in result.convergence_history]
    assert requests == sorted(requests)
    assert uniques == sorted(uniques)
    assert hits == sorted(hits)


def test_run_history_reports_phase_request_count():
    result, _ = run_reduced()
    for record in result.convergence_history:
        assert record.request_count == 12
    assert result.convergence_history[-1].cumulative_requests == 96


def test_run_elites_are_in_bgwo_initial_population():
    result, _ = run_reduced()
    for slot in range(min(3, result.elite_masks.shape[0])):
        assert np.array_equal(result.bgwo_initial_population[slot + 1], result.elite_masks[slot])
    assert np.all(result.bgwo_initial_population[0] == 1)


def test_run_bgwo_caches_shared_evaluations_across_phases():
    objective, state = make_counting_objective()
    config = reduced_config(2, 2)
    engine = HybridBPSOBGWO(config, objective, synthetic_is_better)
    result = engine.optimize(PRIMARY_SEED_FAMILY[3])
    assert result.bgwo_cache_hits >= result.configuration_snapshot["placed_elite_count"]
    assert result.evaluator_calls == state["calls"] == result.unique_evaluations


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_is_reproducible():
    left, _ = run_reduced(PRIMARY_SEED_FAMILY[0])
    right, _ = run_reduced(PRIMARY_SEED_FAMILY[0])
    assert left.to_json() == right.to_json()


def test_different_seeds_diverge():
    left, _ = run_reduced(PRIMARY_SEED_FAMILY[0])
    right, _ = run_reduced(PRIMARY_SEED_FAMILY[4])
    assert left.to_json() != right.to_json()


def test_primary_seed_family_runnable():
    for seed in PRIMARY_SEED_FAMILY:
        result, _ = run_reduced(seed, reduced_config(2, 2))
        assert result.total_candidate_requests == 48


def test_bgwo_phase_uses_offset_generator():
    config = reduced_config(2, 2)
    elite = np.array([1, 1, 1] + [0] * 40, dtype=np.uint8)
    offset_rng = np.random.Generator(np.random.PCG64(PRIMARY_SEED_FAMILY[0] + BGWO_RNG_OFFSET))
    expected = build_bgwo_initial_population(config, elite.reshape(1, -1), offset_rng)
    wrong_rng = np.random.Generator(np.random.PCG64(PRIMARY_SEED_FAMILY[0]))
    wrong = build_bgwo_initial_population(config, elite.reshape(1, -1), wrong_rng)
    assert not np.array_equal(expected, wrong)


def test_result_serializes_to_json():
    result, _ = run_reduced(PRIMARY_SEED_FAMILY[0], reduced_config(2, 2))
    payload = json.loads(result.to_json())
    assert payload["optimizer_name"] == OPTIMIZER_IDENTIFIER
    assert payload["stop_reason"] == "FIXED_BUDGET_EXHAUSTED"
    assert payload["total_candidate_requests"] == 48


def test_result_schema_fields_present():
    result, _ = run_reduced(PRIMARY_SEED_FAMILY[0], reduced_config(2, 2))
    payload = result.to_dict()
    for field in (
        "optimizer_seed",
        "best_mask",
        "best_selected_feature_count",
        "best_evaluation",
        "best_phase",
        "elite_masks",
        "elite_hashes",
        "bgwo_initial_population",
        "convergence_history",
        "configuration_snapshot",
    ):
        assert field in payload


def test_result_reproducible_re_code_stable():
    left, _ = run_reduced(PRIMARY_SEED_FAMILY[0])
    right, _ = run_reduced(PRIMARY_SEED_FAMILY[0])
    assert left.reproducible_re_code == right.reproducible_re_code
    assert len(left.reproducible_re_code) == 64


# ---------------------------------------------------------------------------
# Non-injection across an actual run
# ---------------------------------------------------------------------------


def test_evaluated_landscape_avoids_frozen_winner_masks():
    objective, state = make_counting_objective()
    config = reduced_config(6, 6)
    engine = HybridBPSOBGWO(config, objective, synthetic_is_better)
    result = engine.optimize(PRIMARY_SEED_FAMILY[0])
    digests = {mask_digest(mask) for mask in state["masks"]}
    digests.add(mask_digest(result.best_mask))
    for row in result.bgwo_initial_population:
        digests.add(mask_digest(row))
    assert BPSO_K10_MASK_SHA256 not in digests
    assert BGWO_K14_MASK_SHA256 not in digests


def test_no_transition_evaluations_between_phases():
    objective, state = make_counting_objective()
    config = reduced_config(2, 2)
    engine = HybridBPSOBGWO(config, objective, synthetic_is_better)
    result = engine.optimize(PRIMARY_SEED_FAMILY[2])
    assert result.bpso_unique_evaluations + result.bgwo_new_unique_evaluations == result.unique_evaluations
    assert result.evaluator_calls == state["calls"]


def test_elite_transfer_keeps_recovery_when_identical_masks():
    config = reduced_config(6, 6)
    engine = HybridBPSOBGWO(config, flat_objective, synthetic_is_better)
    result = engine.optimize(PRIMARY_SEED_FAMILY[0])
    assert result.bgwo_requests == 72
    assert result.bgwo_cache_hits >= result.configuration_snapshot["placed_elite_count"]