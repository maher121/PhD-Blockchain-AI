"""Focused dry, checkpoint, analysis, and lock tests for V0.8-C."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from src.optimization.bpso import BinaryParticleSwarmOptimizer
from src.optimization.feature_fitness import (
    FeatureFitnessEvaluation,
    FeatureFitnessEvaluator,
    feature_fitness_is_better,
)
from src.pipeline_v08a import APPROVED_CARDINALITIES, load_v08a_protocol
import src.pipeline_v08c as v08c
from tests.test_bpso_fitness import make_context, spy_factory
from tests.test_pipeline_v08b import synthetic_basis as synthetic_basis_fixture


@pytest.fixture()
def synthetic_basis(request):
    return synthetic_basis_fixture.__wrapped__()


@pytest.fixture()
def production_config():
    return load_v08a_protocol().config


@pytest.fixture()
def synthetic_context():
    return make_context()


@pytest.fixture()
def protocol_identity(production_config, synthetic_basis):
    return v08c.production_protocol_identity(production_config, synthetic_basis)


def _evaluate(mask: np.ndarray, context) -> FeatureFitnessEvaluation:
    return FeatureFitnessEvaluator(context, model_factory=spy_factory)(mask)


def _run_record(seed, config, context, basis, identity, mask=None):
    mask = np.ones(43, dtype=np.uint8) if mask is None else np.asarray(mask, dtype=np.uint8)
    evaluation = _evaluate(mask, context)
    dummy = BinaryParticleSwarmOptimizer(config, lambda value: None, lambda left, right: False)
    population, velocities = dummy.initialize(seed, k42_mask=basis.k42_mask)
    history = [
        {
            "generation_index": 0,
            "best_mask": mask.tolist(),
            "best_selected_feature_count": int(mask.sum()),
            "best_evaluation": evaluation.to_dict(),
            "request_count": 12,
            "cumulative_unique_evaluations": 12,
            "cumulative_cache_hits": 0,
            "population_diversity": 0.5,
            "global_best_improved": True,
            "global_best_changed": True,
            "repair_count": 0,
            "inertia": None,
        }
    ]
    return {
        "schema_version": v08c.V08C_SCHEMA_VERSION,
        "stage": v08c.V08C_STAGE,
        "artifact_kind": v08c.RUN_ARTIFACT_KIND,
        "status": "COMPLETED_AND_VALIDATED",
        "execution_mode": "FRESH",
        "optimizer_seed": seed,
        "model_attack_seeds": list(v08c.MODEL_ATTACK_SEEDS),
        "selection_scope": v08c.VALIDATION_SCOPE,
        "test_accessed": False,
        "logistic_regression_evaluated": False,
        "resource_benchmark_executed": False,
        "direct_energy_measured": False,
        "protocol_identity": identity.to_dict(),
        "configuration": config.to_dict(),
        "feature_manifest_sha256": basis.candidate_manifest_sha256,
        "real_k42_mask_sha256": basis.k42_mask_sha256,
        "initial_population": population.tolist(),
        "initial_population_sha256": v08c._json_sha256(population.tolist()),
        "initial_cardinalities": population.sum(axis=1).astype(int).tolist(),
        "initial_velocities_sha256": v08c._json_sha256(velocities.tolist()),
        "evaluated_generation_count": 1,
        "stop_reason": "MAX_GENERATIONS",
        "best_generation": 0,
        "best_mask": mask.tolist(),
        "best_mask_sha256": evaluation.mask_sha256,
        "selected_features": list(evaluation.selected_features),
        "selected_features_sha256": evaluation.selected_features_sha256,
        "selected_feature_count": evaluation.selected_feature_count,
        "best_evaluation": evaluation.to_dict(),
        "fitness_requests": 12,
        "unique_evaluations": 12,
        "cache_hits": 0,
        "actual_decision_tree_fits": 60,
        "repairs": 0,
        "convergence_history": history,
        "wall_time_sec": 1.0,
        "process_cpu_time_sec": 0.9,
        "evaluation_timing": {},
    }


def _analysis_evaluation(candidates, mask, ap=0.9, f1=0.8, recall=0.7, feasible=True):
    selected = tuple(feature for feature, active in zip(candidates, mask) if active)
    mask_array = np.asarray(mask, dtype=np.uint8)
    return FeatureFitnessEvaluation(
        mask=tuple(int(value) for value in mask),
        mask_sha256=hashlib.sha256(mask_array.tobytes()).hexdigest(),
        selected_features=selected,
        selected_features_sha256=v08c.fingerprint_feature_names(selected),
        selected_feature_count=len(selected),
        feasible=feasible,
        normalized_violation=0.0 if feasible else 0.1,
        relative_losses={"average_precision": 0.0, "f1": 0.0, "recall": 0.0},
        mean_metrics={
            "average_precision": ap,
            "f1": f1,
            "recall": recall,
            "precision": 0.6,
            "roc_auc": 0.7,
            "accuracy": 0.8,
            "false_positive_rate": 0.1,
            "false_negative_rate": 0.2,
            "attack_prevalence": 0.05,
            "selected_feature_visibility_rate": 0.5,
        },
        confusion_totals={
            "true_positives": 10,
            "true_negatives": 10,
            "false_positives": 1,
            "false_negatives": 1,
        },
        per_seed=(),
        decision_tree_fit_count=5,
    )


def _analysis_record(seed, evaluation, wall_time=1.0):
    history_evaluation = evaluation.to_dict()
    return {
        "optimizer_seed": seed,
        "best_mask": list(evaluation.mask),
        "best_evaluation": evaluation.to_dict(),
        "selected_feature_count": evaluation.selected_feature_count,
        "best_generation": 1,
        "evaluated_generation_count": 2,
        "stop_reason": "MAX_GENERATIONS",
        "fitness_requests": 24,
        "unique_evaluations": 20,
        "cache_hits": 4,
        "actual_decision_tree_fits": 100,
        "repairs": 0,
        "wall_time_sec": wall_time,
        "process_cpu_time_sec": wall_time,
        "convergence_history": [
            {
                "generation_index": 0,
                "best_mask": list(evaluation.mask),
                "best_selected_feature_count": evaluation.selected_feature_count + 1,
                "best_evaluation": history_evaluation,
                "global_best_improved": True,
                "population_diversity": 0.5,
                "cumulative_unique_evaluations": 10,
                "cumulative_cache_hits": 2,
                "repair_count": 0,
            },
            {
                "generation_index": 1,
                "best_mask": list(evaluation.mask),
                "best_selected_feature_count": evaluation.selected_feature_count,
                "best_evaluation": history_evaluation,
                "global_best_improved": False,
                "population_diversity": 0.4,
                "cumulative_unique_evaluations": 20,
                "cumulative_cache_hits": 4,
                "repair_count": 0,
            },
        ],
    }


def _five_analysis_records(basis):
    masks = []
    for offset in range(5):
        mask = np.zeros(43, dtype=np.uint8)
        mask[0] = 1
        if offset < 4:
            mask[1] = 1
        mask[2 + offset] = 1
        masks.append(mask)
    return [
        _analysis_record(
            seed,
            _analysis_evaluation(basis.candidate_features, mask),
            wall_time=float(10 - index),
        )
        for index, (seed, mask) in enumerate(zip(v08c.OPTIMIZER_SEEDS, masks))
    ]


def test_01_v08b_prerequisites_are_committed_and_pass() -> None:
    result = v08c.verify_v08b_prerequisites()
    assert result["status"] == "PASS"
    assert result["v08b_source_commit"].startswith("59c2235")
    assert result["final_test_prohibition_pass"] is True
    assert result["pilot_label"] == "QUARANTINED_PILOT"


def test_02_production_configuration_and_seed_sets_are_exact(production_config) -> None:
    v08c.verify_production_config(production_config)
    assert v08c.OPTIMIZER_SEEDS == (1042, 1043, 1044, 1045, 1046)
    assert v08c.MODEL_ATTACK_SEEDS == (42, 43, 44, 45, 46)
    assert production_config.maximum_fitness_requests == 240


def test_03_generation_zero_and_maximum_budget_are_frozen(production_config) -> None:
    snapshot = production_config.to_dict()
    assert snapshot["generation_index_origin"] == 0
    assert snapshot["evaluated_generations"] == 20
    assert snapshot["maximum_fitness_requests"] == 240
    assert "generation 0" in snapshot["generation_semantics"]


def test_04_dry_validation_uses_real_k42_and_exact_cardinalities(
    production_config, synthetic_basis
) -> None:
    result = v08c.production_dry_validation(production_config, synthetic_basis)
    assert result["status"] == "PASS"
    assert result["fitness_evaluations"] == 0
    assert len(result["initializations"]) == 5
    assert all(
        row["cardinalities"] == [43, 42, *APPROVED_CARDINALITIES]
        for row in result["initializations"]
    )


def test_05_dry_initialization_is_same_seed_deterministic(
    production_config, synthetic_basis
) -> None:
    first = v08c.production_dry_validation(production_config, synthetic_basis)
    second = v08c.production_dry_validation(production_config, synthetic_basis)
    assert first == second
    assert len(
        {row["initial_population_sha256"] for row in first["initializations"]}
    ) == 5


def test_06_run_artifact_validation_recomputes_all_invariants(
    production_config, synthetic_context, synthetic_basis, protocol_identity
) -> None:
    record = _run_record(
        1042, production_config, synthetic_context, synthetic_basis, protocol_identity
    )
    checks = v08c.validate_run_artifact(
        record,
        1042,
        production_config,
        synthetic_context,
        synthetic_basis,
        protocol_identity,
    )
    assert all(checks.values())


def test_07_run_artifact_rejects_request_or_generation_overrun(
    production_config, synthetic_context, synthetic_basis, protocol_identity
) -> None:
    record = _run_record(
        1042, production_config, synthetic_context, synthetic_basis, protocol_identity
    )
    record["evaluated_generation_count"] = 21
    with pytest.raises(v08c.V08CCheckpointError, match="validation failed"):
        v08c.validate_run_artifact(
            record,
            1042,
            production_config,
            synthetic_context,
            synthetic_basis,
            protocol_identity,
        )


def test_08_completed_checkpoint_resumes_without_executor(
    tmp_path: Path,
    production_config,
    synthetic_context,
    synthetic_basis,
    protocol_identity,
) -> None:
    record = _run_record(
        1042, production_config, synthetic_context, synthetic_basis, protocol_identity
    )
    path = tmp_path / "v08c_run_1042.json"
    v08c._atomic_write_json(path, record)

    def forbidden_executor(*args, **kwargs):
        raise AssertionError("Accepted checkpoint must not be rerun")

    resumed, mode, digest = v08c.execute_or_resume_run(
        1042,
        tmp_path,
        production_config,
        synthetic_context,
        synthetic_basis,
        protocol_identity,
        executor=forbidden_executor,
    )
    assert mode == "RESUMED"
    assert resumed == record
    assert digest == v08c.sha256_file(path)


def test_09_incompatible_checkpoint_is_rejected_without_overwrite(
    tmp_path: Path,
    production_config,
    synthetic_context,
    synthetic_basis,
    protocol_identity,
) -> None:
    record = _run_record(
        1042, production_config, synthetic_context, synthetic_basis, protocol_identity
    )
    record["protocol_identity"]["production_protocol_sha256"] = "0" * 64
    path = tmp_path / "v08c_run_1042.json"
    v08c._atomic_write_json(path, record)
    before = path.read_bytes()
    with pytest.raises(v08c.V08CCheckpointError):
        v08c.execute_or_resume_run(
            1042,
            tmp_path,
            production_config,
            synthetic_context,
            synthetic_basis,
            protocol_identity,
        )
    assert path.read_bytes() == before


def test_10_jaccard_and_hamming_are_correct_for_all_ten_pairs(synthetic_basis) -> None:
    records = _five_analysis_records(synthetic_basis)
    stability = v08c.build_stability_analysis(records, synthetic_basis.candidate_features)
    assert len(stability["pairwise"]) == 10
    by_seed = {record["optimizer_seed"]: set(np.flatnonzero(record["best_mask"])) for record in records}
    for row in stability["pairwise"]:
        left = by_seed[row["seed_left"]]
        right = by_seed[row["seed_right"]]
        assert row["jaccard"] == pytest.approx(len(left & right) / len(left | right))
        assert row["normalized_hamming"] == pytest.approx(
            len(left ^ right) / 43
        )


def test_11_feature_frequency_consensus_and_majority_are_descriptive(synthetic_basis) -> None:
    stability = v08c.build_stability_analysis(
        _five_analysis_records(synthetic_basis), synthetic_basis.candidate_features
    )
    frequency = {row["feature_index"]: row for row in stability["feature_frequency"]}
    assert len(frequency) == 43
    assert frequency[0]["selection_count"] == 5
    assert frequency[1]["selection_count"] == 4
    assert synthetic_basis.candidate_features[0] in stability["strict_consensus_features"]
    assert synthetic_basis.candidate_features[1] in stability["majority_features"]
    assert stability["consensus_used_for_selection"] is False


def test_12_cardinality_and_fitness_variation_summaries(synthetic_basis) -> None:
    stability = v08c.build_stability_analysis(
        _five_analysis_records(synthetic_basis), synthetic_basis.candidate_features
    )
    assert stability["cardinality"]["count"] == 5
    assert set(stability["cardinality"]) == {
        "count",
        "mean",
        "std",
        "min",
        "max",
        "median",
    }
    assert set(stability["winner_variation"]) == {
        "average_precision",
        "f1",
        "recall",
        "normalized_violation",
        "cardinality",
    }


def test_13_convergence_aggregation_contains_run_and_plot_data(synthetic_basis) -> None:
    analysis = v08c.build_convergence_analysis(_five_analysis_records(synthetic_basis))
    assert len(analysis["run_summaries"]) == 5
    assert len(analysis["plot_data"]) == 10
    assert [row["generation_index"] for row in analysis["aggregate_by_generation"]] == [0, 1]
    assert all(row["run_count"] == 5 for row in analysis["aggregate_by_generation"])


def test_14_validation_comparison_preserves_k11_semantic_distinction(
    synthetic_basis,
) -> None:
    basis = deepcopy(synthetic_basis)
    basis.lock.update(
        {
            "roles": {
                "full_baseline": {
                    "metrics": {"average_precision": 1.0, "f1": 1.0, "recall": 1.0}
                },
                "best_unsupervised": {
                    "metrics": {"average_precision": 1.0, "f1": 1.0, "recall": 1.0}
                },
                "best_supervised": {
                    "metrics": {"average_precision": 1.0, "f1": 1.0, "recall": 1.0}
                },
            }
        }
    )
    comparison = v08c.build_validation_comparison(
        _five_analysis_records(basis), basis
    )
    assert comparison["historical_validation_anchors"]["K11"]["selection_semantics"] == (
        "seed-specific historical MI rule"
    )
    assert comparison["bpso_selection_semantics"] == (
        "one universal subset candidate per optimizer run"
    )
    assert comparison["final_test_comparison_performed"] is False


def test_15_winner_is_selected_only_by_committed_comparator(synthetic_basis) -> None:
    records = _five_analysis_records(synthetic_basis)
    smallest_mask = np.zeros(43, dtype=np.uint8)
    smallest_mask[0] = 1
    smallest = _analysis_evaluation(
        synthetic_basis.candidate_features, smallest_mask, ap=0.1, f1=0.1, recall=0.1
    )
    records[-1] = _analysis_record(1046, smallest, wall_time=1000.0)
    winner, result = v08c.select_validation_winner(records)
    assert winner["optimizer_seed"] == 1046
    assert result.selected_feature_count == 1
    assert not any(
        feature_fitness_is_better(
            v08c.evaluation_from_dict(record["best_evaluation"]), result
        )
        for record in records
    )


def test_16_winner_selection_is_independent_of_runtime_and_test_fields(synthetic_basis) -> None:
    records = _five_analysis_records(synthetic_basis)
    first_winner = v08c.select_validation_winner(records)[0]["optimizer_seed"]
    for index, record in enumerate(records):
        record["wall_time_sec"] = float(index * 1000)
        record["unused_test_metric"] = float(100 - index)
    second_winner = v08c.select_validation_winner(records)[0]["optimizer_seed"]
    assert second_winner == first_winner


def test_17_winner_lock_hash_is_creation_time_independent(
    synthetic_basis, protocol_identity
) -> None:
    records = _five_analysis_records(synthetic_basis)
    winner, evaluation = v08c.select_validation_winner(records)
    hashes = {seed: f"{seed:064x}" for seed in v08c.OPTIMIZER_SEEDS}
    first = v08c.build_winner_lock(
        winner, evaluation, protocol_identity, hashes, created_at="2026-01-01T00:00:00Z"
    )
    second = v08c.build_winner_lock(
        winner, evaluation, protocol_identity, hashes, created_at="2027-01-01T00:00:00Z"
    )
    assert first["semantic_lock_sha256"] == second["semantic_lock_sha256"]
    v08c.verify_winner_lock(first)


def test_18_winner_lock_contains_required_provenance(synthetic_basis, protocol_identity) -> None:
    records = _five_analysis_records(synthetic_basis)
    winner, evaluation = v08c.select_validation_winner(records)
    lock = v08c.build_winner_lock(
        winner,
        evaluation,
        protocol_identity,
        {seed: f"{seed:064x}" for seed in v08c.OPTIMIZER_SEEDS},
    )
    assert lock["status"] == "VALIDATION_LOCKED"
    assert lock["eligible_for_v08d"] is True
    assert lock["selection_scope"] == "TRAIN_AND_DEVELOPMENT_VALIDATION_ONLY"
    assert lock["final_test_accessed"] is False
    assert lock["logistic_regression_evaluated"] is False
    assert lock["comparator"]["resource_metrics_used"] is False
    assert len(lock["five_run_evidence_sha256"]) == 5


def test_19_immutable_lock_cannot_be_silently_overwritten(
    tmp_path: Path, synthetic_basis, protocol_identity
) -> None:
    records = _five_analysis_records(synthetic_basis)
    winner, evaluation = v08c.select_validation_winner(records)
    hashes = {seed: f"{seed:064x}" for seed in v08c.OPTIMIZER_SEEDS}
    lock = v08c.build_winner_lock(winner, evaluation, protocol_identity, hashes)
    path = tmp_path / "winner.json"
    accepted = v08c.write_or_verify_winner_lock(path, lock)
    incompatible = dict(lock)
    incompatible["mask_sha256"] = "f" * 64
    with pytest.raises(v08c.V08CError):
        v08c.write_or_verify_winner_lock(path, incompatible)
    assert v08c._read_json(path) == accepted


def test_20_source_has_no_final_test_loader_lr_fit_or_resource_comparator() -> None:
    source = inspect.getsource(v08c)
    execute_source = inspect.getsource(v08c.execute_production_run)
    select_source = inspect.getsource(v08c.select_validation_winner)
    assert "load_v06_locked_test_data" not in source
    assert "run_locked_test_experiment" not in source
    assert "create_model" not in execute_source
    assert "FeatureFitnessEvaluator" in execute_source
    assert "wall_time" not in select_source
    assert "test_metric" not in select_source
    assert "feature_fitness_is_better" in select_source


def test_21_each_run_constructs_a_new_evaluator_for_run_local_cache() -> None:
    source = inspect.getsource(v08c.execute_production_run)
    assert "evaluator = FeatureFitnessEvaluator(context)" in source
    assert "BinaryParticleSwarmOptimizer(config, evaluator" in source


def test_22_pipeline_has_exact_five_seed_loop_and_no_full_search_duplication() -> None:
    source = inspect.getsource(v08c.run_v08c)
    assert "for seed in OPTIMIZER_SEEDS" in source
    assert "execute_or_resume_run" in source
    assert "tuple(record[\"optimizer_seed\"] for record in records)" in source


def test_23_frozen_integrity_snapshot_covers_all_prior_stages() -> None:
    snapshot = v08c.snapshot_immutable_paths()
    joined = "\n".join(snapshot)
    assert "validation_lock.json" in joined
    assert "v07d_artifact_hashes.json" in joined
    assert "v08a_protocol_validation.json" in joined
    assert "v08b_preflight.json" in joined


def test_24_completed_campaign_artifacts_are_complete_and_hash_consistent() -> None:
    summary = v08c._read_json(v08c.OUTPUT_DIR / "v08c_execution_summary.json")
    assert summary["status"] == "COMPLETED"
    assert summary["completed_run_count"] == 5
    assert summary["final_test_accessed"] is False
    assert summary["logistic_regression_evaluated"] is False
    assert summary["fitness_requests"] == summary["unique_evaluations"] + summary["cache_hits"]
    assert summary["actual_decision_tree_fits"] == summary["unique_evaluations"] * 5
    for seed in v08c.OPTIMIZER_SEEDS:
        path = v08c.OUTPUT_DIR / f"v08c_run_{seed}.json"
        record = v08c._read_json(path)
        assert record["status"] == "COMPLETED_AND_VALIDATED"
        assert record["optimizer_seed"] == seed
        assert record["test_accessed"] is False
        assert summary["run_artifact_sha256"][str(seed)] == v08c.sha256_file(path)


def test_25_real_winner_lock_matches_comparator_selected_run() -> None:
    records = [
        v08c._read_json(v08c.OUTPUT_DIR / f"v08c_run_{seed}.json")
        for seed in v08c.OPTIMIZER_SEEDS
    ]
    winner_record, evaluation = v08c.select_validation_winner(records)
    lock = v08c._read_json(v08c.OUTPUT_DIR / "v08c_winner_lock.json")
    v08c.verify_winner_lock(lock)
    assert lock["source_optimizer_seed"] == winner_record["optimizer_seed"]
    assert lock["mask"] == list(evaluation.mask)
    assert lock["ordered_selected_features"] == list(evaluation.selected_features)
    assert lock["semantic_lock_sha256"] == v08c.winner_lock_semantic_hash(lock)
