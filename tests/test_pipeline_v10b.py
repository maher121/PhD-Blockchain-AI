"""Focused tests for V1.0-B synthetic hybrid protocol validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

import src.pipeline_v10b as v10b


ROOT = Path(__file__).resolve().parents[1]


def test_load_protocol_returns_locked_configuration() -> None:
    loaded = v10b.load_v10b_protocol()
    config = loaded["config"]
    assert config.optimizer_name == "GOVERNED_HYBRID_BPSO_BGWO"
    assert config.dimensions == 43
    assert config.population_size == 12
    assert config.bpso_evaluated_generations == 8
    assert config.bgwo_evaluated_iterations == 8
    assert config.elite_count == 3
    assert config.bgwo_phase_rng_offset == 10000
    assert config.bpso_request_allocation == 96
    assert config.bgwo_request_allocation == 96
    assert config.per_run_request_allocation == 192
    assert config.five_run_request_allocation == 960
    assert loaded["config_sha256"] == v10b.HYBRID_YAML_SHA256
    assert len(loaded["config_sha256"]) == 64


def test_load_protocol_rejects_yaml_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(v10b.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["hybrid_design"]["population_size_per_phase"] = 13
    config_path = tmp_path / "hybrid_v10_drift.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(v10b.V10BProtocolError, match="frozen SHA-256"):
        v10b.load_v10b_protocol(config_path=config_path)


def test_provenance_lock_verification_matches_frozen_yaml() -> None:
    provenance = v10b.provenance_lock_verification()
    assert provenance["commit_exists"] is True
    assert provenance["locked_on_commit"] == v10b.PROVENANCE_LOCK_COMMIT
    assert provenance["yaml_sha256_matches_lock"] is True
    assert len(provenance["yaml_sha256_at_lock_commit"]) == 64


def test_current_head_short_is_observable_but_not_pinned() -> None:
    head = v10b.current_head_short()
    assert len(head) == 7
    int(head, 16)


def test_module_immutability_of_frozen_engines() -> None:
    assert v10b.module_unchanged_by_git(v10b.DEFAULT_BPSO_MODULE)
    assert v10b.module_unchanged_by_git(v10b.DEFAULT_BGWO_MODULE)


def test_synthetic_comparator_prioritizes_feasibility_then_violation() -> None:
    base_kwargs = dict(
        normalized_violation=0.0,
        cardinality=1,
        average_precision=0.24,
        f1=0.20,
        recall=0.33,
        mask=(0,),
    )
    feasible = v10b._SyntheticEvaluation(feasible=True, **base_kwargs)
    infeasible = v10b._SyntheticEvaluation(
        feasible=False, normalized_violation=0.9, **{
            key: value for key, value in base_kwargs.items() if key != "normalized_violation"
        }
    )
    lower_violation = v10b._SyntheticEvaluation(
        feasible=False, normalized_violation=0.1, **{
            key: value for key, value in base_kwargs.items() if key != "normalized_violation"
        }
    )
    assert v10b._synthetic_is_better(feasible, infeasible)
    assert not v10b._synthetic_is_better(infeasible, feasible)
    assert v10b._synthetic_is_better(lower_violation, infeasible)


def test_run_synthetic_validation_produces_pass_artifact(tmp_path: Path) -> None:
    output_path = tmp_path / "v10b_protocol_validation.json"
    summary = v10b.run_v10b_synthetic_validation(output_path=output_path)
    assert summary["stage"] == v10b.V10B_STAGE
    assert summary["status"] == "PASS"
    assert summary["validation_kind"] == v10b.V10B_VALIDATION_KIND
    assert summary["scientific_experiment"] is False
    assert summary["scientific_claims_supported"] is False
    assert all(summary["checks"].values())
    assert summary["governance"]["dataco_accessed"] is False
    assert summary["governance"]["production_hybrid_search_executed"] is False
    assert summary["governance"]["final_test_accessed"] is False
    assert summary["provenance"]["yaml_sha256_matches_lock"] is True
    assert summary["synthetic_smoke"]["replay_identical"] is True
    assert summary["synthetic_smoke"]["requests"] == 96
    assert (
        summary["synthetic_smoke"]["requests"]
        == summary["synthetic_smoke"]["unique_evaluations"]
        + summary["synthetic_smoke"]["cache_hits"]
    )
    assert summary["synthetic_smoke"]["bgwo_cache_hits"] >= summary["synthetic_smoke"]["placed_elites"]

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted == summary


def test_governed_yaml_hash_is_frozen() -> None:
    digest = hashlib.sha256(v10b.DEFAULT_CONFIG_PATH.read_bytes()).hexdigest()
    assert digest == v10b.HYBRID_YAML_SHA256


def test_production_budget_never_executed_in_pipeline() -> None:
    loaded = v10b.load_v10b_protocol()
    config = loaded["config"]
    assert config.bpso_evaluated_generations == 8
    assert config.bgwo_evaluated_iterations == 8
    summary = v10b.run_v10b_synthetic_validation(output_path=None)
    assert summary["synthetic_smoke"]["bpso_generations"] == 4
    assert summary["synthetic_smoke"]["bgwo_iterations"] == 4
    assert summary["governance"]["hybrid_production_budget_executed"] is False