"""Focused tests for V0.9-D five-run BGWO production search and winner lock."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import re
import statistics
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

import src.pipeline_v08b as v08b
import src.pipeline_v09b as v09b
import src.pipeline_v09c as v09c
import src.pipeline_v09d as v09d
from src.optimization import bgwo
import src.optimization.feature_fitness as fitness
from src.security.experiment_data import DevelopmentExperimentData, fingerprint_feature_names


def _candidate_features() -> tuple[str, ...]:
    return tuple(f"feature_{index:02d}" for index in range(43))


def _synthetic_split(
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
    labels_array = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)[:rows]
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
        test_authorization_id=None,
        test_authorization_capability=None,
    )


def synthetic_context(*, test_accessed: bool = False) -> fitness.FitnessContext:
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
            train=_synthetic_split("train", seed, candidates, 100, 6),
            validation=_synthetic_split("validation", seed, candidates, 200, 4),
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


def _synthetic_k11_features() -> dict[str, tuple[str, ...]]:
    candidates = _candidate_features()
    return {
        str(seed): tuple(candidates[index] for index in range(11))
        for seed in fitness.EXPECTED_SEEDS
    }


BP_SO_K10_MASK = np.asarray(
    [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    dtype=np.uint8,
)


def synthetic_basis() -> v08b.FrozenBasis:
    context = synthetic_context()
    k42_mask = np.ones(43, dtype=np.uint8)
    k42_mask[2] = 0
    k42_features = tuple(
        feature
        for feature, active in zip(context.candidate_features, k42_mask)
        if active
    )
    return v08b.FrozenBasis(
        candidate_features=context.candidate_features,
        candidate_manifest_sha256=context.candidate_manifest_sha256,
        k42_features=k42_features,
        k42_mask=k42_mask,
        k42_features_sha256=fingerprint_feature_names(k42_features),
        k42_mask_sha256=hashlib.sha256(k42_mask.tobytes()).hexdigest(),
        k11_seed_specific=True,
        k11_feature_hashes={str(seed): str(seed) * 64 for seed in fitness.EXPECTED_SEEDS},
        baseline=context.baseline,
        baseline_seed_metrics={},
        dataset_metadata=context.workloads[0].dataset_metadata,
        lock={
            "roles": {
                "full_baseline": {
                    "metrics": {
                        "average_precision": 1.0,
                        "f1": 1.0,
                        "recall": 1.0,
                    }
                },
                "best_unsupervised": {
                    "metrics": {
                        "average_precision": 0.9,
                        "f1": 0.9,
                        "recall": 0.9,
                    }
                },
                "best_supervised": {
                    "selected_features": _synthetic_k11_features(),
                    "metrics": {
                        "average_precision": 0.8,
                        "f1": 0.8,
                        "recall": 0.8,
                    },
                },
            }
        },
        source_hashes={},
    )


def _mask_with_indices(indices: Sequence[int], dimensions: int = 43) -> np.ndarray:
    mask = np.zeros(dimensions, dtype=np.uint8)
    for index in indices:
        mask[index] = 1
    return mask


def small_config() -> bgwo.BGWOConfig:
    protocol = v09b.load_v09b_protocol()
    return replace(protocol.config, evaluated_iterations=2)


def test_01_stage_constants_match_frozen_protocol() -> None:
    protocol = v09b.load_v09b_protocol()
    assert protocol.config.optimizer_name == "BGWO"
    assert protocol.config.dimensions == 43
    assert protocol.config.wolf_count == 12
    assert protocol.config.evaluated_iterations == 20
    assert protocol.config.maximum_candidate_requests == 240
    assert v09d.V09D_STAGE == "V0.9-D"
    assert v09d.EXPECTED_WOLF_COUNT == 12
    assert v09d.EXPECTED_BUDGET_PER_RUN == 240
    assert v09d.EXPECTED_BUDGET_ACROSS_RUNS == 1200
    assert v09d.EXPECTED_INITIAL_CARDINALITIES == (43, 42, 4, 8, 11, 14, 18, 22, 26, 30, 34, 38)
    assert v09d.OPTIMIZER_SEEDS == (2042, 2043, 2044, 2045, 2046)
    assert v09d.CARDINALITIES == (4, 8, 11, 14, 18, 22, 26, 30, 34, 38)


def test_02_frozen_protocol_yaml_hash_is_stable() -> None:
    observed = v09d.sha256_file(v09d.PROJECT_ROOT / "config" / "bgwo_v09.yaml")
    assert observed == "6b2ccf82a9d5bc382916567dd1eac076ad5b5a4927048b5c97f7ef457a6d27b2"


def test_03_production_config_verification_passes_and_rejects_drift() -> None:
    protocol = v09b.load_v09b_protocol()
    v09d.verify_production_config(protocol.config)
    drifted_budget = replace(protocol.config, evaluated_iterations=19)
    with pytest.raises(v09d.V09DError, match="drift"):
        v09d.verify_production_config(drifted_budget)
    drifted_update = replace(protocol.config, early_stopping_patience=6)
    with pytest.raises(v09d.V09DError, match="drift"):
        v09d.verify_production_config(drifted_update)


def test_04_v09c_artifacts_on_disk_verify_clean() -> None:
    result = v09d.verify_v09c_artifacts()
    assert result["status"] == "PASS"
    assert all(result["checks"].values())
    assert result["checks"]["v09c_pilot_quarantined"] is True


def test_05_no_v09e_artifacts_gate(tmp_path: Path, monkeypatch) -> None:
    v09d.verify_no_v09e_artifacts()
    monkeypatch.setattr(v09d, "PROJECT_ROOT", tmp_path)
    v09d.verify_no_v09e_artifacts()
    (tmp_path / "results" / "bgwo").mkdir(parents=True)
    (tmp_path / "results" / "bgwo" / "v09e_marker.json").write_text("{}", encoding="utf-8")
    with pytest.raises(v09d.V09DError, match="V0.9-E"):
        v09d.verify_no_v09e_artifacts()


def test_06_dry_validation_has_zero_fitness_and_locked_initialization() -> None:
    protocol = v09b.load_v09b_protocol()
    basis = synthetic_basis()
    winner_lock = {"mask": BP_SO_K10_MASK.tolist()}
    dry = v09d.production_dry_validation(protocol.config, basis, winner_lock)
    assert dry["status"] == "PASS"
    assert dry["fitness_evaluations"] == 0
    assert dry["final_test_accessed"] is False
    assert len(dry["initializations"]) == 5
    for row in dry["initializations"]:
        assert row["cardinalities"] == list(v09d.EXPECTED_INITIAL_CARDINALITIES)
        assert row["initialization_independence"]["status"] == "PASS"


def test_07_execute_and_validate_production_run_record() -> None:
    protocol = v09b.load_v09b_protocol()
    context = synthetic_context()
    basis = synthetic_basis()
    config = small_config()
    identity = v09d.production_protocol_identity(config, basis, protocol)
    record = v09d.execute_production_run(2042, config, context, basis, identity)
    assert record["optimizer_name"] == "BGWO"
    assert record["optimizer_seed"] == 2042
    assert record["stop_reason"] == bgwo.STOP_MAX_ITERATIONS
    assert record["evaluated_iteration_count"] == 2
    assert record["candidate_requests"] == 24
    assert record["selected_feature_count"] == int(np.asarray(record["best_mask"]).sum())
    assert record["actual_decision_tree_fits"] == record["unique_evaluations"] * 5
    assert record["candidate_requests"] == record["unique_evaluations"] + record["cache_hits"]
    checks = v09d.validate_run_artifact(record, 2042, config, context, basis, identity)
    assert all(checks.values())
    assert v09d.validate_run_artifact(record, 2042, config, context, basis, identity)["dt_fit_accounting"] is True


def test_08_run_artifact_rejects_wrong_seed() -> None:
    protocol = v09b.load_v09b_protocol()
    context = synthetic_context()
    basis = synthetic_basis()
    config = small_config()
    identity = v09d.production_protocol_identity(config, basis, protocol)
    record = v09d.execute_production_run(2042, config, context, basis, identity)
    with pytest.raises(v09d.V09DCheckpointError, match="approved_optimizer_seed"):
        v09d.validate_run_artifact(record, 2043, config, context, basis, identity)


def test_09_run_artifact_rejects_tampered_mask_hash() -> None:
    protocol = v09b.load_v09b_protocol()
    context = synthetic_context()
    basis = synthetic_basis()
    config = small_config()
    identity = v09d.production_protocol_identity(config, basis, protocol)
    record = v09d.execute_production_run(2042, config, context, basis, identity)
    tampered = dict(record)
    tampered["best_mask"] = [1 if index in (0, 1) else 0 for index in range(43)]
    with pytest.raises(v09d.V09DCheckpointError, match="mask_hash"):
        v09d.validate_run_artifact(tampered, 2042, config, context, basis, identity)


def test_10_run_checkpoint_resume_and_no_duplicate(tmp_path: Path) -> None:
    protocol = v09b.load_v09b_protocol()
    context = synthetic_context()
    basis = synthetic_basis()
    config = small_config()
    identity = v09d.production_protocol_identity(config, basis, protocol)
    calls = []

    def executor(seed, config_, context_, basis_, identity_):
        calls.append(seed)
        return v09d.execute_production_run(seed, config_, context_, basis_, identity_)

    first, mode_first, hash_first = v09d.execute_or_resume_run(
        2042, tmp_path, config, context, basis, identity, executor=executor
    )
    second, mode_second, hash_second = v09d.execute_or_resume_run(
        2042, tmp_path, config, context, basis, identity, executor=executor
    )
    assert mode_first == "FRESH"
    assert mode_second == "RESUMED"
    assert calls == [2042]
    assert hash_first == hash_second
    assert first["status"] == "COMPLETED_AND_VALIDATED"
    assert first == second


def test_11_resume_rejects_corrupt_or_incompatible_checkpoint(tmp_path: Path) -> None:
    protocol = v09b.load_v09b_protocol()
    context = synthetic_context()
    basis = synthetic_basis()
    config = small_config()
    identity = v09d.production_protocol_identity(config, basis, protocol)
    path = tmp_path / v09d.RUN_FILE_TEMPLATE.format(seed=2043)
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(v09d.V09DCheckpointError, match="Cannot read JSON"):
        v09d.execute_or_resume_run(2043, tmp_path, config, context, basis, identity)

    record = v09d.execute_production_run(2043, config, context, basis, identity)
    incompatible = dict(record)
    incompatible["protocol_identity"] = {
        **incompatible["protocol_identity"],
        "production_protocol_sha256": "0" * 64,
    }
    (tmp_path / v09d.RUN_FILE_TEMPLATE.format(seed=2043)).write_text(
        json.dumps(incompatible, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(v09d.V09DCheckpointError, match="protocol_identity"):
        v09d.execute_or_resume_run(2043, tmp_path, config, context, basis, identity)


def test_12_stability_math_is_exact() -> None:
    protocol = v09b.load_v09b_protocol()
    basis = synthetic_basis()
    context = synthetic_context()
    config = small_config()
    identity = v09d.production_protocol_identity(config, basis, protocol)
    template = v09d.execute_production_run(2042, config, context, basis, identity)
    template_evaluation = template["best_evaluation"]
    masks = {
        2042: np.asarray([1 if index < 11 else 0 for index in range(43)], dtype=np.uint8),
        2043: np.asarray([1 if index < 11 else 0 for index in range(43)], dtype=np.uint8),
        2044: np.asarray([1 if index < 10 else 0 for index in range(43)], dtype=np.uint8),
        2045: np.asarray([1 if index < 9 else 0 for index in range(43)], dtype=np.uint8),
        2046: np.asarray([1 if index < 11 else 0 for index in range(43)], dtype=np.uint8),
    }
    records = [
        {
            "optimizer_seed": seed,
            "best_mask": mask.tolist(),
            "best_evaluation": template_evaluation,
        }
        for seed, mask in masks.items()
    ]
    stability = v09d.build_stability_analysis(records, basis.candidate_features)
    assert stability["cardinality"] == {
        "count": 5,
        "mean": 10.4,
        "std": pytest.approx(statistics.stdev([11, 11, 10, 9, 11]), rel=1e-12),
        "min": 9,
        "max": 11,
        "median": 11.0,
    }
    pairs = {frozenset((row["seed_left"], row["seed_right"])): row for row in stability["pairwise"]}
    identical = pairs[frozenset((2042, 2043))]
    assert identical["jaccard"] == 1.0
    assert identical["normalized_hamming"] == 0.0
    assert identical["absolute_hamming_distance"] == 0
    assert stability["strict_consensus_features"] == list(basis.candidate_features[:9])
    assert stability["majority_features"] == list(basis.candidate_features[:10])
    assert stability["consensus_used_for_selection"] is False
    assert stability["test_accessed"] is False


def test_13_winner_selection_uses_governed_comparator_ordering() -> None:
    feasible_4 = v09c._synthetic_evaluation(
        mask=_mask_with_indices((0, 1, 2)), feasible=True, violation=0.0,
        average_precision=0.7, f1=0.7, recall=0.7,
    )
    feasible_4_high = v09c._synthetic_evaluation(
        mask=_mask_with_indices((0, 1, 2, 3)), feasible=True, violation=0.0,
        average_precision=0.71, f1=0.71, recall=0.71,
    )
    infeasible_low = v09c._synthetic_evaluation(
        mask=_mask_with_indices((0, 1, 2, 3, 4)), feasible=False, violation=0.4,
        average_precision=0.9, f1=0.9, recall=0.9,
    )
    infeasible_high = v09c._synthetic_evaluation(
        mask=_mask_with_indices((0, 1, 2, 3, 4, 5)), feasible=False, violation=1.2,
        average_precision=0.91, f1=0.91, recall=0.91,
    )
    records = [
        {"optimizer_seed": seed, "best_evaluation": evaluation.to_dict()}
        for seed, evaluation in zip(
            v09d.OPTIMIZER_SEEDS, (infeasible_high, infeasible_low, feasible_4_high, feasible_4, feasible_4_high)
        )
    ]
    winner_record, winner_evaluation = v09d.select_search_winner(records)
    assert winner_record["optimizer_seed"] == 2045
    assert winner_evaluation.selected_feature_count == 3
    with pytest.raises(v09d.V09DError, match="exactly five"):
        v09d.select_search_winner(records[:4])


def test_14_winner_lock_roundtrip_semantic_hash_and_verification(tmp_path: Path) -> None:
    protocol = v09b.load_v09b_protocol()
    context = synthetic_context()
    basis = synthetic_basis()
    config = small_config()
    identity = v09d.production_protocol_identity(config, basis, protocol)
    records = [
        v09d.execute_production_run(seed, config, context, basis, identity)
        for seed in v09d.OPTIMIZER_SEEDS
    ]
    winner_record, winner_evaluation = v09d.select_search_winner(records)
    budget = {"candidate_requests": sum(int(record["candidate_requests"]) for record in records)}
    run_hashes = {seed: hashlib.sha256(str(seed).encode()).hexdigest() for seed in v09d.OPTIMIZER_SEEDS}
    lock = v09d.build_winner_lock(
        winner_record, winner_evaluation, identity, run_hashes,
        protocol=protocol, budget=budget, created_at="2026-01-01T00:00:00+00:00",
    )
    v09d.verify_winner_lock(lock)
    expected_semantic = v09d.winner_lock_semantic_hash(lock)
    assert lock["semantic_lock_sha256"] == expected_semantic
    stored = v09d.write_or_verify_winner_lock(tmp_path / "v09d_winner_lock.json", lock)
    assert stored["semantic_lock_sha256"] == expected_semantic
    again = v09d.write_or_verify_winner_lock(tmp_path / "v09d_winner_lock.json", lock)
    assert again["semantic_lock_sha256"] == expected_semantic

    tampered = dict(lock)
    tampered["mask"] = [1 if index < 5 else 0 for index in range(43)]
    with pytest.raises(v09d.V09DError, match="verification failed"):
        v09d.verify_winner_lock(tampered)


def test_15_winner_lock_rejects_test_access_flags() -> None:
    protocol = v09b.load_v09b_protocol()
    context = synthetic_context()
    basis = synthetic_basis()
    config = small_config()
    identity = v09d.production_protocol_identity(config, basis, protocol)
    record = v09d.execute_production_run(2042, config, context, basis, identity)
    winner_evaluation = v09d.evaluation_from_dict(record["best_evaluation"])
    budget = {"candidate_requests": 24}
    run_hashes = {seed: hashlib.sha256(str(seed).encode()).hexdigest() for seed in v09d.OPTIMIZER_SEEDS}
    lock = v09d.build_winner_lock(
        record, winner_evaluation, identity, run_hashes,
        protocol=protocol, budget=budget,
    )
    for flag_name in ("final_test_accessed", "test_authorized", "test_used_for_winner_selection"):
        tampered = dict(lock)
        tampered[flag_name] = True
        with pytest.raises(v09d.V09DError, match="verification failed"):
            v09d.verify_winner_lock(tampered)


def test_16_winner_lock_enforces_across_run_budget() -> None:
    protocol = v09b.load_v09b_protocol()
    context = synthetic_context()
    basis = synthetic_basis()
    config = small_config()
    identity = v09d.production_protocol_identity(config, basis, protocol)
    record = v09d.execute_production_run(2042, config, context, basis, identity)
    winner_evaluation = v09d.evaluation_from_dict(record["best_evaluation"])
    run_hashes = {seed: hashlib.sha256(str(seed).encode()).hexdigest() for seed in v09d.OPTIMIZER_SEEDS}
    over_budget = v09d.build_winner_lock(
        record, winner_evaluation, identity, run_hashes,
        protocol=protocol, budget={"candidate_requests": 1201},
    )
    with pytest.raises(v09d.V09DError, match="verification failed"):
        v09d.verify_winner_lock(over_budget)
    within_budget = v09d.build_winner_lock(
        record, winner_evaluation, identity, run_hashes,
        protocol=protocol, budget={"candidate_requests": 1100},
    )
    v09d.verify_winner_lock(within_budget)
    assert within_budget["optimization_budget"]["five_run_total_candidate_requests"] == 1100


def test_17_test_access_audit_flags_are_false() -> None:
    protocol = v09b.load_v09b_protocol()
    context = synthetic_context()
    basis = synthetic_basis()
    config = small_config()
    identity = v09d.production_protocol_identity(config, basis, protocol)
    records = [
        v09d.execute_production_run(seed, config, context, basis, identity)
        for seed in v09d.OPTIMIZER_SEEDS
    ]
    winner_record, winner_evaluation = v09d.select_search_winner(records)
    budget = {"candidate_requests": sum(int(record["candidate_requests"]) for record in records)}
    run_hashes = {seed: hashlib.sha256(str(seed).encode()).hexdigest() for seed in v09d.OPTIMIZER_SEEDS}
    lock = v09d.build_winner_lock(
        winner_record, winner_evaluation, identity, run_hashes,
        protocol=protocol, budget=budget,
    )
    audit = v09d.build_test_access_audit(
        winner_lock=lock,
        records=records,
        leakage_audit={"status": "PASS", "test_accessed": False},
    )
    assert audit["status"] == "PASS"
    assert audit["final_test_accessed"] is False
    assert audit["test_authorized"] is False
    assert audit["test_used_for_winner_selection"] is False
    assert audit["no_v09e_artifacts"] is True
    assert audit["bpso_rerun"] is False
    assert all(
        flags["test_accessed"] is False
        for flags in audit["run_test_flags"].values()
    )


def test_18_stage_has_no_test_loader_or_bpso_rerun_entrypoints() -> None:
    source = inspect.getsource(v09d)
    assert "load_v06_locked_test_data" not in source
    assert "prepare_v06_locked_test_data" not in source
    assert "run_locked_test_experiment" not in source
    assert "BinaryParticleSwarmOptimizer" not in source
    assert "run_v08c(" not in source
    assert "run_v08d(" not in source
    assert '"logistic_regression_evaluated": False' in source
    assert "logistic_regression_predict_frame" not in source


def test_19_five_run_campaign_resumes_and_completes(tmp_path: Path) -> None:
    protocol = v09b.load_v09b_protocol()
    context = synthetic_context()
    basis = synthetic_basis()
    config = small_config()
    identity = v09d.production_protocol_identity(config, basis, protocol)
    fresh_records: list[dict] = []
    fresh_modes: list[str] = []
    hashes: dict[int, str] = {}
    for seed in v09d.OPTIMIZER_SEEDS:
        record, mode, run_hash = v09d.execute_or_resume_run(
            seed, tmp_path, config, context, basis, identity
        )
        fresh_records.append(record)
        fresh_modes.append(mode)
        hashes[seed] = run_hash
    assert fresh_modes == ["FRESH"] * 5
    assert len(list(tmp_path.glob("v09d_run_*.json"))) == 5

    resumed_records: list[dict] = []
    resumed_modes: list[str] = []
    for seed in v09d.OPTIMIZER_SEEDS:
        record, mode, run_hash = v09d.execute_or_resume_run(
            seed, tmp_path, config, context, basis, identity
        )
        resumed_records.append(record)
        resumed_modes.append(mode)
        assert run_hash == hashes[seed]
    assert resumed_modes == ["RESUMED"] * 5
    assert resumed_records == fresh_records
    assert len(list(tmp_path.glob("v09d_run_*.json"))) == 5

    records = resumed_records
    assert tuple(record["optimizer_seed"] for record in records) == v09d.OPTIMIZER_SEEDS

    stability = v09d.build_stability_analysis(records, basis.candidate_features)
    convergence = v09d.build_convergence_analysis(records)
    comparison = v09d.build_validation_comparison(
        records, basis, {"validation_metrics": {"average_precision": 0.31, "f1": 0.25, "recall": 0.34}, "selected_feature_count": 10}
    )
    winner_record, winner_evaluation = v09d.select_search_winner(records)
    budget = {"candidate_requests": sum(int(record["candidate_requests"]) for record in records)}
    lock = v09d.build_winner_lock(
        winner_record, winner_evaluation, identity, hashes,
        protocol=protocol, budget=budget,
    )
    assert v09d.write_or_verify_winner_lock(tmp_path / "v09d_winner_lock.json", lock) == lock
    assert stability["optimizer_run_count"] == 5
    assert stability["cardinality"]["count"] == 5
    assert len(convergence["run_summaries"]) == 5
    assert len(comparison["bgwo_run_winners"]) == 5
    assert comparison["bpso_descriptive_context"]["selected_feature_count"] == 10
    assert comparison["bpso_descriptive_context"]["no_superiority_declared"] is True


def test_20_winner_lock_uses_canonical_feature_list_hash() -> None:
    protocol = v09b.load_v09b_protocol()
    context = synthetic_context()
    basis = synthetic_basis()
    config = small_config()
    identity = v09d.production_protocol_identity(config, basis, protocol)
    record = v09d.execute_production_run(2042, config, context, basis, identity)
    winner_evaluation = v09d.evaluation_from_dict(record["best_evaluation"])
    run_hashes = {seed: hashlib.sha256(str(seed).encode()).hexdigest() for seed in v09d.OPTIMIZER_SEEDS}
    lock = v09d.build_winner_lock(
        record, winner_evaluation, identity, run_hashes,
        protocol=protocol, budget={"candidate_requests": 24},
    )
    selected = tuple(lock["ordered_selected_features"])
    assert lock["mask_sha256"] == hashlib.sha256(
        np.asarray(lock["mask"], dtype=np.uint8).tobytes()
    ).hexdigest()
    assert lock["selected_features_sha256"] == fingerprint_feature_names(selected)
    assert lock["selected_feature_count"] == len(selected) == int(np.asarray(lock["mask"]).sum())
    assert lock["feature_manifest_sha256"] == identity.feature_manifest_sha256


def test_head_commit_regex() -> None:
    head = v09d.sha256_file(v09d.PROJECT_ROOT / "src" / "pipeline_v09d.py")
    assert re.fullmatch(r"[0-9a-f]{64}", head)