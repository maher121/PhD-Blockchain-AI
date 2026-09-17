"""Focused tests for V0.9-C leakage-safe real fitness integration and pilot."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.pipeline_v08b as v08b
import src.pipeline_v09b as v09b
import src.pipeline_v09c as v09c
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
                "best_supervised": {
                    "selected_features": _synthetic_k11_features(),
                }
            }
        },
        source_hashes={},
    )


def test_01_stage_constants_match_frozen_protocol() -> None:
    protocol = v09b.load_v09b_protocol()
    assert protocol.config.optimizer_name == "BGWO"
    assert protocol.config.dimensions == 43
    assert protocol.config.wolf_count == 12
    assert protocol.config.evaluated_iterations == 20
    assert protocol.config.maximum_candidate_requests == 240
    assert protocol.optimizer_seeds == (2042, 2043, 2044, 2045, 2046)
    assert v09c.V09C_STAGE == "V0.9-C"
    assert v09c.V09C_PILOT_LABEL == "QUARANTINED_PILOT"
    assert v09c.V09C_PILOT_ELIGIBILITY == "NOT_ELIGIBLE_FOR_SCIENTIFIC_WINNER_SELECTION"
    assert v09c.EXPECTED_PILOT_OPTIMIZER_SEED == 2042
    assert v09c.EXPECTED_PILOT_MAX_REQUESTS == 24


def test_02_pilot_config_forces_24_request_budget() -> None:
    protocol = v09b.load_v09b_protocol()
    pilot = v09c.build_pilot_config(protocol.config)
    assert pilot.wolf_count == 12
    assert pilot.evaluated_iterations == 2
    assert pilot.maximum_candidate_requests == 24
    assert pilot.to_dict()["maximum_candidate_requests"] == 24
    assert pilot.cache_enabled is True


def test_03_pilot_config_rejects_drift() -> None:
    protocol = v09b.load_v09b_protocol()
    drifted = v08b.replace(
        protocol.config,
        wolf_count=13,
        random_cardinalities=(4, 8, 11, 14, 18, 22, 26, 30, 34, 38, 40),
    )
    with pytest.raises(v09c.V09CError, match="pilot configuration drift"):
        v09c.build_pilot_config(drifted)


def test_04_v08_frozen_integrity_passes() -> None:
    result = v09b.verify_v08_frozen_integrity()
    assert result["status"] == "PASS"
    assert (
        result["v08d_semantic_result_lock_sha256"]
        == "ac19e5a0dba5d058f88485e6ab8ab9a96f97c30f697e95a744ccaa6563d2b657"
    )
    assert (
        result["v08e_semantic_result_lock_sha256"]
        == "7307fda1cdb5f100cb6268dc0f4eff05b65bfcefadeb06707c8a3e13fd6888d5"
    )


def test_05_k43_baseline_and_preservation_thresholds_match_fitness() -> None:
    protocol = v09b.load_v09b_protocol()
    baseline = protocol.raw_snapshot["predictive_baseline"]
    assert baseline["average_precision"] == pytest.approx(0.2306840611, abs=1e-10)
    assert baseline["f1"] == pytest.approx(0.1977010726, abs=1e-10)
    assert baseline["recall"] == pytest.approx(0.322, abs=1e-10)
    constraints = baseline["preservation_constraints"]
    assert constraints["average_precision_relative_loss_max"] == 0.05
    assert constraints["f1_relative_loss_max"] == 0.05
    assert constraints["recall_relative_loss_max"] == 0.10
    assert dict(fitness.MARGINS) == {
        "average_precision": 0.05,
        "f1": 0.05,
        "recall": 0.10,
    }


def test_06_shared_governed_fitness_reused() -> None:
    source = inspect.getsource(v09c)
    assert "FeatureFitnessEvaluator(context)" in source
    assert "feature_fitness_is_better" in source
    assert "audit_fitness_context" in source
    audit = v09c.build_v09c_leakage_audit(synthetic_context())
    assert audit["status"] == "PASS"
    assert audit["checks"]["shared_feature_fitness_reused"] is True
    assert audit["checks"]["shared_constrained_comparator_reused"] is True
    assert audit["checks"]["winner_selection_disabled_in_stage"] is True
    assert audit["ground_truth"] == "is_attack"
    assert audit["forbidden_ground_truth"] == ["Late_delivery_risk", "SUSPECTED_FRAUD"]
    assert audit["pass_count"] == audit["total_check_count"]


def test_07_synthetic_development_context_passes_test_inaccessibility() -> None:
    context = synthetic_context()
    v09c.ensure_test_split_inaccessible(context)
    audit = v09c.build_v09c_leakage_audit(context)
    assert audit["status"] == "PASS"
    assert audit["test_accessed"] is False


def test_08_guard_rejects_test_access_flag() -> None:
    context = synthetic_context(test_accessed=True)
    with pytest.raises(v09c.V09CError, match="test access"):
        v09c.ensure_test_split_inaccessible(context)


def test_09_guard_rejects_workload_exposing_test_split() -> None:
    context = synthetic_context()
    setattr(context.workloads[0], "test", "forbidden")
    with pytest.raises(v09c.V09CError, match="test split"):
        v09c.ensure_test_split_inaccessible(context)


def test_10_guard_rejects_test_authorization_on_development_split() -> None:
    context = synthetic_context()
    authorization = SimpleNamespace(
        **{name: getattr(context.workloads[0].train, name) for name in ("split_name", "candidate_features", "clean_features", "features", "labels", "ground_truth", "attack_metadata", "row_ids_sha256", "clean_features_sha256", "attacked_features_sha256", "labels_sha256", "test_authorization_id", "test_authorization_capability")}
    )
    authorization.test_authorization_id = "forbidden-validation-lock"
    context.workloads[0].train = authorization
    with pytest.raises(v09c.V09CError, match="authorization"):
        v09c.ensure_test_split_inaccessible(context)


def test_11_comparator_consistency_checks_pass() -> None:
    result = v09c.run_comparator_consistency_checks()
    assert result["status"] == "PASS"
    assert result["feasible_comparator_verified"] is True
    assert result["infeasible_comparator_verified"] is True
    assert all(result["checks"].values())


def test_12_initialization_identity_audit_detects_prohibited_identities() -> None:
    basis = synthetic_basis()
    winner_lock = {"mask": BP_SO_K10_MASK.tolist()}
    population = np.zeros((12, 43), dtype=np.uint8)
    population[0, :] = 1
    population[1] = basis.k42_mask
    k11_mask = v09c._mask_from_selected_features(
        basis.candidate_features, _synthetic_k11_features()["42"]
    )
    population[3] = k11_mask
    audit = v09c.audit_initialization_independence(
        population, basis.candidate_features, basis, winner_lock
    )
    assert audit["status"] == "FAIL"
    assert audit["mi_k11_present"] is True
    assert audit["checks"]["real_k42_anchor_used"] is True

    population_10 = np.zeros((12, 43), dtype=np.uint8)
    population_10[0, :] = 1
    population_10[1] = basis.k42_mask
    population_10[5] = BP_SO_K10_MASK
    audit_10 = v09c.audit_initialization_independence(
        population_10, basis.candidate_features, basis, winner_lock
    )
    assert audit_10["status"] == "FAIL"
    assert audit_10["bpso_k10_present"] is True


def test_13_clean_initialization_identity_audit_passes() -> None:
    protocol = v09b.load_v09b_protocol()
    basis = synthetic_basis()
    winner_lock = {"mask": BP_SO_K10_MASK.tolist()}
    generator = np.random.Generator(np.random.PCG64(2042))
    population = bgwo.build_initial_population(
        v09c.build_pilot_config(protocol.config),
        generator,
        k42_mask=basis.k42_mask,
    )
    assert population.sum(axis=1).astype(int).tolist() == [43, 42, 4, 8, 11, 14, 18, 22, 26, 30, 34, 38]
    audit = v09c.audit_initialization_independence(
        population, basis.candidate_features, basis, winner_lock
    )
    assert audit["status"] == "PASS"
    assert audit["checks"]["real_k42_anchor_used"] is True
    assert audit["checks"]["bpso_k10_identity_not_present"] is True
    assert audit["checks"]["mi_k11_seed_identities_not_present"] is True


def test_14_quarantined_pilot_runs_within_budget_and_accounting() -> None:
    protocol = v09b.load_v09b_protocol()
    context = synthetic_context()
    basis = synthetic_basis()
    winner_lock = {"mask": BP_SO_K10_MASK.tolist()}
    pilot = v09c.run_quarantined_pilot(protocol, context, basis, winner_lock)
    assert pilot["status"] == "PASS"
    assert pilot["label"] == "QUARANTINED_PILOT"
    assert pilot["eligibility"] == "NOT_ELIGIBLE_FOR_SCIENTIFIC_WINNER_SELECTION"
    assert pilot["scientific_experiment"] is False
    assert pilot["optimizer_seed"] == 2042
    assert pilot["population"] == 12
    assert pilot["evaluated_iterations"] == 2
    assert pilot["candidate_requests"] == 24
    assert pilot["evaluated_iteration_count"] == 2
    assert pilot["stop_reason"] == bgwo.STOP_MAX_ITERATIONS
    assert pilot["decision_tree_fits"] == pilot["unique_candidate_evaluations"] * 5
    assert pilot["accounting"]["request_budget"] is True
    assert pilot["accounting"]["dt_fit_accounting"] is True
    assert pilot["accounting"]["pilot_boundary_no_early_stop"] is True
    assert pilot["identity_audit"]["status"] == "PASS"
    assert pilot["governance"]["winner_selected"] is False
    assert pilot["governance"]["winner_lock_created"] is False
    assert pilot["governance"]["production_search_executed"] is False
    assert pilot["governance"]["final_test_accessed"] is False
    assert pilot["governance"]["bpso_rerun"] is False
    assert 0 < pilot["runtime"]["wall_time_sec"]


def test_15_no_winner_lock_artifacts_allowed(tmp_path: Path) -> None:
    v09c.verify_no_winner_lock_artifacts(tmp_path)
    (tmp_path / "v09c_fake_winner_lock.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(v09c.V09CError, match="winner locks"):
        v09c.verify_no_winner_lock_artifacts(tmp_path)


def test_16_stage_has_no_test_loader_or_production_search_entrypoint() -> None:
    source = inspect.getsource(v09c)
    assert "load_v06_locked_test_data" not in source
    assert "prepare_v06_locked_test_data" not in source
    assert "run_locked_test_experiment" not in source
    assert "logistic_regression" not in source
    assert "BinaryParticleSwarmOptimizer" not in source
    assert "run_v08d" not in source
    assert "run_v08c" not in source


def test_17_pipeline_execution_is_fail_closed_and_ordered() -> None:
    source = inspect.getsource(v09c.run_v09c)
    ordered = (
        "verify_v09c_preflight",
        "build_v09c_fitness_context",
        "build_v09c_leakage_audit",
        "reproduce_k43_baseline",
        "run_comparator_consistency_checks",
        "run_quarantined_pilot",
        "verify_no_winner_lock_artifacts",
    )
    positions = [source.index(name) for name in ordered]
    assert positions == sorted(positions)
    assert "if leakage[\"status\"] != \"PASS\"" in source
    assert "if baseline[\"status\"] != \"PASS\"" in source
    assert "if comparator[\"status\"] != \"PASS\"" in source


def test_18_stage_registration_is_machine_readable() -> None:
    result = v09c.run_comparator_consistency_checks()
    json.dumps(result, sort_keys=True)
    assert v09c.sha256_file(v09c.PROJECT_ROOT / "config" / "bgwo_v09.yaml")


def test_head_commit_regex() -> None:
    head = v09c.current_head_short()
    assert re.fullmatch(r"[0-9a-f]{7}", head)