from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "resource_efficiency_v11f.yaml"
PROTOCOL_PATH = PROJECT_ROOT / "docs" / "v11f_green_evaluation_protocol.md"
LOCK_PATH = (
    PROJECT_ROOT / "results" / "blockchain" / "v11f" / "v11f_protocol_lock.json"
)
UPSTREAM_LOCK_PATH = (
    PROJECT_ROOT / "results" / "blockchain" / "v11e" / "v11e_experiment_result_lock.json"
)

EXPECTED_HEAD = "d425cf1c89e391edd1a68fd2b14f3565ab487b88"
EXPECTED_V11E_SEMANTIC = (
    "058aeca8ac97101356bcf1c4dc5b74fb3affbda6833a85b5d423a53556cd1749"
)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def _semantic_sha256(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage_identity_and_f1_boundary(cfg, lock):
    assert cfg["stage"] == "V1.1-F1"
    assert cfg["artifact_kind"] == "V11F1_RESOURCE_EFFICIENCY_PROTOCOL_LOCK"
    assert cfg["protocol_classification"] == (
        "RESOURCE_EFFICIENCY_GREEN_EVALUATION_PROTOCOL_LOCKED"
    )
    assert cfg["stage_identity"]["name"] == "Resource-Efficiency / Green Evaluation"
    assert cfg["stage_identity"]["read_only_analytical_stage"] is True
    assert cfg["stage_identity"]["implementation_excluded"] is True
    assert cfg["stage_identity"]["execution_excluded"] is True
    assert lock["stage"] == cfg["stage"]


def test_starting_checkpoint_and_upstream_lock_are_pinned(cfg):
    assert cfg["starting_checkpoint"] == {
        "head": EXPECTED_HEAD,
        "head_origin_main": EXPECTED_HEAD,
    }
    upstream_cfg = cfg["upstream_v11e"]
    upstream_disk = json.loads(UPSTREAM_LOCK_PATH.read_text(encoding="utf-8"))
    assert upstream_cfg["result_lock_semantic_sha256"] == EXPECTED_V11E_SEMANTIC
    assert upstream_disk["semantic_result_lock_sha256"] == EXPECTED_V11E_SEMANTIC
    assert upstream_cfg["result_lock_file_sha256"] == _file_sha256(UPSTREAM_LOCK_PATH)
    assert upstream_cfg["expected_cells"] == 390
    assert upstream_cfg["measured_cells"] == 345
    assert upstream_cfg["e10_derived_cells"] == 45
    assert upstream_cfg["missing_cells"] == 0
    assert upstream_cfg["duplicate_cells"] == 0
    assert upstream_cfg["immutable"] is True
    assert upstream_cfg["regeneration_allowed"] is False


def test_only_frozen_policies_are_allowed(cfg):
    assert cfg["policies"]["allowed"] == ["B0", "B1", "P"]
    assert cfg["policies"]["additions_allowed"] is False
    assert cfg["policies"]["tuning_allowed"] is False
    assert cfg["policies"]["definition_changes_allowed"] is False
    assert cfg["policies"]["execution_allowed"] is False


def test_data_and_test_access_are_forbidden(cfg):
    access = cfg["access_policy"]
    assert access["data_access"] == "READ_ONLY_PERSISTED_ARTIFACTS_ONLY"
    assert access["raw_dataco_reopen_allowed"] is False
    assert access["test_access"] == "FORBIDDEN"
    assert access["test_access_count_required"] == 0
    assert access["upstream_mutation_allowed"] is False


def test_ai_fit_and_inference_are_forbidden(cfg):
    policy = cfg["ai_policy"]
    assert policy["ai_fit_allowed"] is False
    assert policy["ai_inference_allowed"] is False
    assert policy["model_loading_for_new_scoring_allowed"] is False
    assert policy["retraining_allowed"] is False
    assert policy["new_risk_prediction_allowed"] is False


def test_all_new_measurement_and_e06_to_e10_reruns_are_forbidden(cfg):
    policy = cfg["new_measurement_policy"]
    assert policy
    assert all(value is False for value in policy.values())


def test_allowed_metric_families_are_exact(cfg):
    evidence = cfg["allowed_metric_families"]
    assert set(evidence) == {"runtime", "computational_work", "memory", "storage", "throughput"}
    assert evidence["computational_work"]["metrics"] == [
        "validation_check_count",
        "validator_invocation_count",
        "measurable_hash_operation_count",
    ]
    assert evidence["throughput"]["metrics"] == ["persisted_validated_orders_per_second"]


def test_e07_is_tracemalloc_only_memory_proxy(cfg):
    e07 = cfg["e07"]
    assert e07["measurement"] == "tracemalloc_fresh_worker"
    assert e07["label"] == "COMPUTATIONAL_MEMORY_PROXY"
    assert e07["rss_measurement"] is False
    assert e07["electrical_energy_measurement"] is False
    assert "fresh-process tracemalloc peak" in e07["required_terms"]
    assert "Python traced-allocation memory proxy" in e07["required_terms"]


def test_e08_is_policy_independent(cfg):
    e08 = cfg["e08"]
    assert e08["status"] == "POLICY_INDEPENDENT"
    assert e08["persisted_policy_token"] == "B0"
    assert e08["policy_storage_differences_allowed"] is False
    assert e08["manufactured_b0_b1_p_storage_differences_forbidden"] is True


def test_e10_is_derived_only_without_fourth_campaign(cfg):
    e10 = cfg["e10"]
    assert e10["status"] == "DERIVED_ONLY"
    assert e10["sources"] == ["E06", "E07", "E08", "E09"]
    assert e10["fourth_campaign_allowed"] is False
    assert e10["independent_measurement_allowed"] is False


def test_direct_energy_policy_is_explicit(cfg):
    policy = cfg["direct_energy_policy"]
    assert policy["status"] == "DIRECT_ENERGY_UNAVAILABLE"
    for key in [
        "cpu_time_is_energy",
        "wall_time_is_energy",
        "tracemalloc_is_energy",
        "throughput_is_energy",
        "storage_is_energy",
    ]:
        assert policy[key] is False
    assert policy["prohibited_estimates"] == [
        "Joules",
        "Wh",
        "kWh",
        "Watts",
        "TDP-derived_energy",
        "carbon_emissions",
        "CO2e",
    ]


def test_population_limitations_are_frozen(cfg):
    population = cfg["population"]
    assert population["risk_band_counts"] == {"LOW": 0, "MEDIUM": 4582, "HIGH": 6}
    assert population["low_marker"] == "LOW_ABSENT_IN_GOVERNED_POPULATION"
    assert population["workload_100_high_marker"] == "HIGH_NOT_OBSERVED_IN_CONDITION"
    assert population["strong_high_subgroup_inference_allowed"] is False


def test_nested_workloads_are_not_independent_replicates(cfg):
    workloads = cfg["workloads"]
    assert workloads["order_counts"] == [100, 250, 500, 1000, 2500]
    assert workloads["seeds"] == [522, 523, 524]
    assert workloads["nesting"] == (
        "100 subset 250 subset 500 subset 1000 subset 2500 within each seed"
    )
    assert workloads["independent_replicates"] is False
    assert workloads["seed_workload_cells_may_be_treated_as_15_independent_replicates"] is False


def test_analysis_is_descriptive_only_and_has_no_winner(cfg):
    analysis = cfg["descriptive_analysis"]
    assert analysis["status"] == "DESCRIPTIVE_ONLY"
    assert analysis["allowed_summaries"] == [
        "mean",
        "median",
        "sample_standard_deviation_ddof_1",
        "min",
        "max",
        "coefficient_of_variation",
    ]
    for forbidden in [
        "p_values",
        "statistical_significance_claims",
        "bootstrap_inference",
        "order_level_inference",
        "inferential_confidence_intervals",
        "composite_overall_policy_score",
        "overall_policy_winner",
        "universal_superiority",
        "global_policy_ranking",
    ]:
        assert forbidden in analysis["forbidden"]
    assert cfg["stage_identity"]["overall_policy_winner_allowed"] is False
    assert cfg["trade_off_interpretation"]["overall_winner_allowed"] is False


def test_matched_differences_and_ratio_formula_are_frozen(cfg):
    analysis = cfg["descriptive_analysis"]
    assert set(analysis["allowed_matched_differences"]) == {
        "B1_minus_B0",
        "P_minus_B0",
        "P_minus_B1",
    }
    assert analysis["normalized_ratio_formula"] == (
        "metric(numerator_policy, seed, workload) / metric(denominator_policy, seed, workload)"
    )
    assert "denominator is strictly positive" in analysis["normalized_ratio_preconditions"]
    assert "E08 storage is excluded from policy ratios" in analysis["normalized_ratio_preconditions"]
    assert cfg["pairing"]["minimum_keys"] == ["seed", "workload"]
    assert cfg["pairing"]["unmatched_comparison_policy"] == "FAIL_CLOSED"


def test_security_is_context_only(cfg):
    context = cfg["security_context"]
    assert context["role"] == "FROZEN_UPSTREAM_TRADE_OFF_CONTEXT_ONLY"
    assert context["new_security_evaluation_allowed"] is False
    assert context["e02_detection_counts"] == {
        "denominator_per_policy": 135,
        "B0": 30,
        "B1": 105,
        "P": 75,
    }
    assert context["e04_unaffected_order_preservation"] == {
        "numerator_per_policy": 117315,
        "denominator_per_policy": 117315,
    }
    assert context["e03_independent_confirmatory_evidence"] is False


def test_future_f2_fail_closed_contract_is_complete(cfg):
    expected = {
        "expected_v11e_result_lock_absent",
        "semantic_lock_mismatch",
        "required_artifact_fingerprint_mismatch",
        "unexpected_test_access",
        "ai_fit_detected",
        "ai_inference_detected",
        "new_measurement_campaign_attempted",
        "e10_not_derived_only",
        "nested_workloads_treated_as_independent_replicates",
        "direct_energy_terminology_or_claim_introduced",
        "upstream_artifact_mutation_detected",
        "unexpected_policy",
        "b0_b1_p_policy_contract_changed",
        "unmatched_comparison",
    }
    assert set(cfg["future_f2_fail_closed"]) == expected
    assert all(cfg["future_f2_fail_closed"].values())


def test_protocol_config_consistency(cfg, lock):
    payload = lock["semantic_payload"]
    assert payload["stage"] == cfg["stage"]
    assert payload["protocol_version"] == cfg["protocol_version"]
    assert payload["starting_checkpoint"] == cfg["starting_checkpoint"]["head"]
    assert payload["upstream_v11e"] == cfg["upstream_v11e"]
    assert payload["access_policy"] == cfg["access_policy"]
    assert payload["ai_policy"] == cfg["ai_policy"]
    assert payload["new_measurement_policy"] == cfg["new_measurement_policy"]
    assert payload["policies"] == cfg["policies"]
    assert payload["direct_energy_policy"] == cfg["direct_energy_policy"]
    assert payload["workloads"] == cfg["workloads"]
    assert payload["descriptive_analysis"] == cfg["descriptive_analysis"]
    assert payload["future_f2_fail_closed"] == cfg["future_f2_fail_closed"]


def test_semantic_lock_and_artifact_fingerprints_are_deterministic(lock):
    payload = lock["semantic_payload"]
    assert lock["semantic_result_lock_sha256"] == _semantic_sha256(payload)
    fingerprints = payload["artifact_fingerprints"]
    assert fingerprints["protocol_config_sha256"] == _file_sha256(CONFIG_PATH)
    assert fingerprints["protocol_document_sha256"] == _file_sha256(PROTOCOL_PATH)


def test_protocol_lock_records_no_execution(lock):
    metadata = lock["execution_metadata"]
    assert metadata["head_origin_main"] == EXPECTED_HEAD
    assert metadata["protocol_only"] is True
    assert metadata["f2_started"] is False
    assert metadata["experiment_execution_count"] == 0
    assert metadata["test_access_count"] == 0
    assert metadata["ai_fit_count"] == 0
    assert metadata["ai_inference_count"] == 0
    assert metadata["new_measurement_campaign_count"] == 0


def test_f1_outputs_exclude_implementation_and_results(cfg):
    outputs = cfg["f1_outputs"]
    assert outputs["allowed"] == [
        "docs/v11f_green_evaluation_protocol.md",
        "config/resource_efficiency_v11f.yaml",
        "tests/test_protocol_v11f.py",
        "results/blockchain/v11f/v11f_protocol_lock.json",
    ]
    assert outputs["implementation_files_allowed"] is False
    assert outputs["execution_files_allowed"] is False
    assert outputs["result_artifacts_allowed"] is False
    assert not (PROJECT_ROOT / "src" / "pipeline_v11f.py").exists()
    assert not (PROJECT_ROOT / "scripts" / "run_v11f.py").exists()


def test_protocol_document_states_critical_boundaries():
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    for marker in [
        "Resource-Efficiency / Green Evaluation",
        "READ_ONLY_PERSISTED_ARTIFACTS_ONLY",
        "TEST_ACCESS = FORBIDDEN",
        "AI_FIT_ALLOWED = FALSE",
        "AI_INFERENCE_ALLOWED = FALSE",
        "DIRECT_ENERGY_UNAVAILABLE",
        "fresh-process tracemalloc peak",
        "POLICY_INDEPENDENT",
        "DERIVED_ONLY",
        "LOW_ABSENT_IN_GOVERNED_POPULATION",
        "HIGH_NOT_OBSERVED_IN_CONDITION",
        "DESCRIPTIVE_ONLY",
        "overall policy winner",
    ]:
        assert marker in text
