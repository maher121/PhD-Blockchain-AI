"""Protocol and synthetic-only governance tests for V0.8-A."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

import src.pipeline_v08a as v08a


FROZEN_ARTIFACTS = (
    Path("results/feature_selection/validation_lock.json"),
    Path("results/feature_selection/validation_lock.sha256"),
    Path("results/feature_selection/leakage_audit.json"),
    Path("results/feature_selection/final_test/final_test_artifact_hashes.json"),
    Path("results/feature_selection/robustness/robustness_artifact_hashes.json"),
    Path("results/green_evaluation/energy_capability.json"),
    Path("results/green_evaluation/v07d_artifact_hashes.json"),
    Path("results/green_evaluation/v07e_artifact_hashes.json"),
    Path("results/green_evaluation/v07d_governance_audit.json"),
    Path("results/green_evaluation/v07e_governance_audit.json"),
)


def _hashes() -> dict[str, str]:
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in FROZEN_ARTIFACTS
    }


def test_01_config_loads_with_approved_dimensions_budget_and_seeds() -> None:
    protocol = v08a.load_v08a_protocol()
    assert protocol.config.dimensions == 43
    assert protocol.config.particle_count == 12
    assert protocol.config.evaluated_generations == 20
    assert protocol.config.maximum_fitness_requests == 240
    assert protocol.optimizer_seeds == (1042, 1043, 1044, 1045, 1046)


def test_02_config_freezes_initialization_velocity_inertia_and_early_stop() -> None:
    protocol = v08a.load_v08a_protocol()
    config = protocol.config
    assert config.random_cardinalities == (4, 8, 11, 14, 18, 22, 26, 30, 34, 38)
    assert (config.velocity_initial_min, config.velocity_initial_max) == (-1.0, 1.0)
    assert (config.velocity_clamp_min, config.velocity_clamp_max) == (-6.0, 6.0)
    assert (config.inertia_start, config.inertia_end) == (0.9, 0.4)
    assert config.cognitive_coefficient == config.social_coefficient == 2.0
    assert config.minimum_selected_features == 1
    assert config.early_stopping_patience == 7
    assert config.early_stopping_min_generation == 10
    assert config.cache_enabled is True
    assert protocol.synthetic_k42_inactive_index == 42


def test_03_config_rejects_protocol_drift(tmp_path: Path) -> None:
    text = v08a.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    changed = tmp_path / "bpso.yaml"
    changed.write_text(text.replace("particles: 12", "particles: 13"), encoding="utf-8")
    with pytest.raises(v08a.V08AProtocolError, match="protocol drift"):
        v08a.load_v08a_protocol(changed)


def test_04_constrained_comparator_prefers_feasible_over_smaller_infeasible() -> None:
    infeasible = v08a.SyntheticEvaluation(False, 0.1, 1, 100.0, 100.0, "a")
    feasible = v08a.SyntheticEvaluation(True, 0.0, 5, 0.0, 0.0, "b")
    assert v08a.constrained_evaluation_is_better(feasible, infeasible)
    assert not v08a.constrained_evaluation_is_better(infeasible, feasible)


def test_05_constrained_comparator_minimizes_feasible_cardinality() -> None:
    small = v08a.SyntheticEvaluation(True, 0.0, 2, 0.0, 0.0, "b")
    large = v08a.SyntheticEvaluation(True, 0.0, 3, 100.0, 100.0, "a")
    assert v08a.constrained_evaluation_is_better(small, large)
    assert not v08a.constrained_evaluation_is_better(large, small)


def test_06_synthetic_protocol_validation_passes_and_writes_labeled_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "v08a_protocol_validation.json"
    summary = v08a.run_synthetic_protocol_validation(output_path=output)
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert summary == stored
    assert summary["status"] == "PASS"
    assert summary["validation_kind"] == "SYNTHETIC_PROTOCOL_VALIDATION"
    assert summary["scientific_experiment"] is False
    assert summary["scientific_claims_supported"] is False
    assert all(summary["checks"].values())


def test_07_protocol_validation_is_byte_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "validation.json"
    first = v08a.run_synthetic_protocol_validation(output_path=output)
    first_bytes = output.read_bytes()
    second = v08a.run_synthetic_protocol_validation(output_path=output)
    assert first == second
    assert output.read_bytes() == first_bytes


def test_08_generation_budget_cache_and_early_stop_are_validated(tmp_path: Path) -> None:
    summary = v08a.run_synthetic_protocol_validation(output_path=tmp_path / "result.json")
    assert summary["generation_semantics"] == {
        "evaluated_indices": [0, 19],
        "initialization_generation": 0,
        "initialization_counts_toward_budget": True,
        "maximum_requests_per_run": 240,
    }
    cache = summary["synthetic_cases"]["cache"]
    assert cache["requests"] == cache["unique_evaluations"] + cache["cache_hits"]
    assert cache["cache_hits"] > 0
    early = summary["synthetic_cases"]["early_stopping"]
    assert early["stop_reason"] == "EARLY_STOP_NO_IMPROVEMENT"
    assert early["evaluated_generation_count"] == 11
    assert early["last_generation_index"] == 10


def test_09_pipeline_is_synthetic_only_and_has_no_data_or_classifier_imports() -> None:
    source = inspect.getsource(v08a)
    forbidden = (
        "load_processed_dataco",
        "prepare_attack_split",
        "DecisionTreeClassifier",
        "LogisticRegression",
        "sklearn",
        "pandas",
        "pipeline_v06",
        "pipeline_v07",
        "feature_fitness",
    )
    assert all(marker not in source for marker in forbidden)
    assert "SYNTHETIC_PROTOCOL_VALIDATION" in source
    assert "scientific_experiment" in source


def test_10_governance_confirms_no_real_access_fit_or_experiment(tmp_path: Path) -> None:
    summary = v08a.run_synthetic_protocol_validation(output_path=tmp_path / "result.json")
    assert summary["governance"] == {
        "dataset_accessed": False,
        "classifier_fitted": False,
        "real_feature_subset_selected": False,
        "resource_benchmark_executed": False,
        "direct_energy_measured": False,
        "next_stage_started": False,
    }


def test_11_validation_does_not_modify_frozen_v06_v07_artifacts(tmp_path: Path) -> None:
    before = _hashes()
    v08a.run_synthetic_protocol_validation(output_path=tmp_path / "result.json")
    assert _hashes() == before


def test_12_default_artifact_is_restricted_to_results_bpso() -> None:
    protocol = v08a.load_v08a_protocol()
    assert protocol.artifact_path == v08a.DEFAULT_OUTPUT_PATH
    assert protocol.artifact_path.relative_to(v08a.PROJECT_ROOT) == Path(
        "results/bpso/v08a_protocol_validation.json"
    )


def test_13_pipeline_does_not_start_v08b_or_make_scientific_claims(tmp_path: Path) -> None:
    summary = v08a.run_synthetic_protocol_validation(output_path=tmp_path / "result.json")
    serialized = json.dumps(summary, sort_keys=True)
    assert "V0.8-B" not in serialized
    assert summary["scientific_claims_supported"] is False
    assert summary["initialization"]["real_feature_identities_bound"] is False
