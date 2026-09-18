"""Focused V1.0-C tests: real-fitness gate, baseline reproduction, quarantined pilot.

These tests are artifact-based: they verify the persisted V1.0-C evidence under
``results/hybrid/v10c_pilot/`` without re-running the 48-request real pilot.
A single in-test K43 real re-evaluation proves baseline determinism.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import yaml

import src.pipeline_v10b as v10b
import src.pipeline_v10c as v10c
import src.pipeline_v08b as v08b


ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = v10c.DEFAULT_OUTPUT_DIR

FROZEN_BASELINE = {
    "average_precision": 0.23068406113411433,
    "f1": 0.1977010726398371,
    "recall": 0.32199999999999995,
}
FROZEN_WINNERS = {
    "bpso_k10": "5da981b5b87db97338ecdde9ca8a8b87db3a62771d03dc6a4ad901f6548a3299",
    "bgwo_k14": "7ebb823374255f4f10c737c62a2111604aa50193a8f0d91b8483f3864cac7ad6",
}


@pytest.fixture(scope="module")
def v10c_artifacts() -> dict[str, dict]:
    payload = {}
    for name in (
        "v10c_preflight.json",
        "v10c_baseline_reproduction.json",
        "v10c_leakage_audit.json",
        "v10c_comparator_audit.json",
        "v10c_pilot_result.json",
        "v10c_accounting_audit.json",
        "v10c_test_access_audit.json",
        "v10c_execution_summary.json",
    ):
        payload[name] = json.loads((PILOT_DIR / name).read_text(encoding="utf-8"))
    return payload


# ---------------------------------------------------------------------------
# Governance, provenance, and immutable locks
# ---------------------------------------------------------------------------


def test_governed_yaml_hash_is_frozen() -> None:
    digest = hashlib.sha256(v10c.HYBRID_YAML_PATH.read_bytes()).hexdigest()
    assert digest == v10b.HYBRID_YAML_SHA256 == v10c.HYBRID_YAML_SHA256


def test_protocol_classification_frozen() -> None:
    payload = yaml.safe_load(v10c.HYBRID_YAML_PATH.read_text(encoding="utf-8"))
    assert payload["protocol_classification"] == v10b.PROTOCOL_CLASSIFICATION


def test_starting_checkpoint_exists_in_history_The_ADG_policy() -> None:
    assert v10c._commit_exists(v10c.PROJECT_ROOT, v10c.STARTING_CHECKPOINT)
    assert v10c._is_ancestor_of_head(v10c.PROJECT_ROOT, v10c.STARTING_CHECKPOINT)


def test_live_head_never_pinned() -> None:
    preflight = json.loads((PILOT_DIR / "v10c_preflight.json").read_text(encoding="utf-8"))
    assert preflight["checks"]["live_head_never_pinned"] is True
    assert preflight["provenance_policy"] == "HEAD_AGNOSTIC_ANCESTRY"


def test_preflight_records_observable_head_only() -> None:
    preflight = json.loads((PILOT_DIR / "v10c_preflight.json").read_text(encoding="utf-8"))
    observed = preflight["observed_head_short"]
    assert len(observed) == 7
    int(observed, 16)


def test_v10b_implementation_tracked_and_unchanged(v10c_artifacts: dict) -> None:
    preflight = v10c_artifacts["v10c_preflight.json"]
    assert preflight["checks"]["v10b_implementation_tracked"] is True
    assert preflight["checks"]["v10b_implementation_unchanged_by_git"] is True
    assert v10c._git_tracked(v10c.PROJECT_ROOT, v10c.V10B_IMPL_PATHS)


def test_protocol_document_present() -> None:
    preflight = json.loads((PILOT_DIR / "v10c_preflight.json").read_text(encoding="utf-8"))
    assert preflight["checks"]["protocol_document_present"] is True


def test_engine_modules_immutable_by_git() -> None:
    assert v10b.module_unchanged_by_git(v10c.PROJECT_ROOT / "src" / "optimization" / "hybrid_bpso_bgwo.py")
    assert v10b.module_unchanged_by_git(v10c.PROJECT_ROOT / "src" / "pipeline_v10b.py")


# ---------------------------------------------------------------------------
# Dataset identity
# ---------------------------------------------------------------------------


def test_dataset_identity_from_context() -> None:
    integration = v10c.load_integration_context()
    ds = integration["dataset"]
    assert ds["raw_rows"] == 180519
    assert ds["raw_columns_before_row_id"] == 53
    assert ds["working_cap"] == 40000
    assert ds["split_seed"] == 42
    assert ds["split_strategy"] == "order_grouped"
    assert ds["expected_split_sizes"] == {"train": 28000, "validation": 6000, "test": 6000}
    assert ds["feature_dimensions"] == 43
    assert len(ds["candidate_feature_order"]) == 43


def test_dataset_identity_split_sizes_match_metadata() -> None:
    metadata = json.loads(
        (v10c.PROJECT_ROOT / "data" / "processed" / "dataset_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["split"]["sizes"] == v10c.SPLIT_SIZES
    assert metadata["split"]["seed"] == v10c.SPLIT_SEED


def test_forbidden_labels_absent_from_candidates() -> None:
    integration = v10c.load_integration_context()
    candidates = set(integration["context"].candidate_features)
    assert "Late_delivery_risk" not in candidates
    assert "SUSPECTED_FRAUD" not in candidates
    assert "Order Status" not in candidates
    assert "is_attack" not in candidates


def test_target_is_attack_only() -> None:
    integration = v10c.load_integration_context()
    for workload in integration["context"].workloads:
        assert workload.train.labels.name == "is_attack"
        assert workload.validation.labels.name == "is_attack"


def test_development_data_has_no_test_attribute() -> None:
    integration = v10c.load_integration_context()
    for workload in integration["context"].workloads:
        assert not hasattr(workload, "test")
        assert not hasattr(workload.train, "test")
        assert not hasattr(workload.validation, "test")


def test_candidate_fingerprint_matches_manifest() -> None:
    integration = v10c.load_integration_context()
    assert integration["context"].candidate_manifest_sha256 == (
        integration["basis"].candidate_manifest_sha256
    )


# ---------------------------------------------------------------------------
# Frozen user-accepted baseline reproduction
# ---------------------------------------------------------------------------


def test_baseline_gate_passed(v10c_artifacts: dict) -> None:
    assert v10c_artifacts["v10c_baseline_reproduction.json"]["status"] == "PASS"


def test_baseline_actual_values_match_frozen_stored(v10c_artifacts: dict) -> None:
    baseline = v10c_artifacts["v10c_baseline_reproduction.json"]
    for metric, stored in FROZEN_BASELINE.items():
        assert float(baseline["actual_values"][metric]) == pytest.approx(stored, abs=1e-12)
        assert float(baseline["expected_values"][metric]) == pytest.approx(stored, abs=0.0)


def test_baseline_tolerance_is_1e_12_absolute(v10c_artifacts: dict) -> None:
    tolerance = v10c_artifacts["v10c_baseline_reproduction.json"]["tolerance"]
    assert tolerance["kind"] == "absolute"
    assert tolerance["value"] == 1e-12


def test_baseline_differences_within_tolerance(v10c_artifacts: dict) -> None:
    baseline = v10c_artifacts["v10c_baseline_reproduction.json"]
    assert baseline["checks"]["baseline_reproduced_within_tolerance"] is True
    assert baseline["checks"]["stored_baseline_metrics_match_within_tolerance"] is True
    for metric in FROZEN_BASELINE:
        diff = abs(float(baseline["differences"][metric]))
        assert diff <= 1e-12


def test_baseline_exact_two_evaluations_ten_fits(v10c_artifacts: dict) -> None:
    baseline = v10c_artifacts["v10c_baseline_reproduction.json"]
    assert baseline["evaluator_requests"] == 2
    assert baseline["decision_tree_fits"] == 10
    assert baseline["checks"]["evaluator_requests_two"] is True
    assert baseline["checks"]["decision_tree_fits_ten"] is True


def test_baseline_deterministic_repeat(v10c_artifacts: dict) -> None:
    baseline = v10c_artifacts["v10c_baseline_reproduction.json"]
    assert baseline["deterministic_repeat"] is True
    assert baseline["checks"]["deterministic_repeat_identical"] is True


def test_baseline_reproduced_evaluation_is_k43(v10c_artifacts: dict) -> None:
    baseline = v10c_artifacts["v10c_baseline_reproduction.json"]
    assert baseline["reproduced_evaluation_feature_count"] == 43
    assert len(baseline["reproduced_evaluation_mask_sha256"]) == 64


def test_baseline_test_not_accessed(v10c_artifacts: dict) -> None:
    baseline = v10c_artifacts["v10c_baseline_reproduction.json"]
    assert baseline["checks"]["test_not_accessed"] is True


def test_in_test_real_k43_determinism() -> None:
    integration = v10c.load_integration_context()
    record, _ = v08b.reproduce_k43(integration["context"], integration["basis"])
    assert record["status"] == "PASS"
    assert record["deterministic_repeat"] is True
    assert record["checks"]["deterministic_repeat_identical"] is True


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------


def test_leakage_audit_passed(v10c_artifacts: dict) -> None:
    assert v10c_artifacts["v10c_leakage_audit.json"]["status"] == "PASS"


def test_leakage_scope_is_development_validation_only(v10c_artifacts: dict) -> None:
    leak = v10c_artifacts["v10c_leakage_audit.json"]
    assert leak["scope"] == "training_and_development_validation_only"


def test_leakage_test_flags_all_false(v10c_artifacts: dict) -> None:
    leak = v10c_artifacts["v10c_leakage_audit.json"]
    assert leak["test_accessed"] is False
    assert leak["test_used_for_fitness"] is False
    assert leak["test_used_for_winner_selection"] is False
    assert leak["checks"]["test_accessed_false"] is True
    assert leak["checks"]["test_used_for_fitness_false"] is True
    assert leak["checks"]["test_used_for_winner_selection_false"] is True


def test_leakage_all_checks_true(v10c_artifacts: dict) -> None:
    leak = v10c_artifacts["v10c_leakage_audit.json"]
    assert all(value is True for value in leak["checks"].values())
    assert len(leak["checks"]) >= 19


def test_leakage_forbidden_ground_truths_declared(v10c_artifacts: dict) -> None:
    leak = v10c_artifacts["v10c_leakage_audit.json"]
    assert set(leak["forbidden_as_cyber_ground_truth"]) == {"Late_delivery_risk", "SUSPECTED_FRAUD"}


# ---------------------------------------------------------------------------
# Comparator audit (frozen constrained ranking)
# ---------------------------------------------------------------------------


def test_comparator_audit_passed(v10c_artifacts: dict) -> None:
    comparator = v10c_artifacts["v10c_comparator_audit.json"]
    assert comparator["status"] == "PASS"
    assert comparator["ranking_rules_unchanged"] is True


def test_comparator_audit_cases_all_consistent(v10c_artifacts: dict) -> None:
    comparator = v10c_artifacts["v10c_comparator_audit.json"]
    assert len(comparator["cases"]) >= 10
    assert all(case["consistent"] for case in comparator["cases"])


def test_comparator_feasible_beats_infeasible(v10c_artifacts: dict) -> None:
    cases = {case["case"]: case for case in v10c_artifacts["v10c_comparator_audit.json"]["cases"]}
    assert cases["feasible_vs_infeasible"]["left_better"] is True
    assert cases["infeasible_vs_feasible"]["right_better"] is True


def test_comparator_smaller_k_wins(v10c_artifacts: dict) -> None:
    cases = {case["case"]: case for case in v10c_artifacts["v10c_comparator_audit.json"]["cases"]}
    assert cases["feasible_smaller_k_wins"]["left_better"] is True
    assert cases["feasible_larger_k_loses"]["right_better"] is True


def test_comparator_never_prefers_both(v10c_artifacts: dict) -> None:
    cases = v10c_artifacts["v10c_comparator_audit.json"]["cases"]
    for case in cases:
        assert not (case["left_better"] and case["right_better"])


# ---------------------------------------------------------------------------
# Quarantined pilot
# ---------------------------------------------------------------------------


def test_pilot_label_and_eligibility(v10c_artifacts: dict) -> None:
    pilot = v10c_artifacts["v10c_pilot_result.json"]
    assert pilot["label"] == v10c.PILOT_LABEL == "QUARANTINED_PILOT"
    assert pilot["eligibility"] == v10c.PILOT_ELIGIBILITY
    assert pilot["eligible_for_scientific_winner_selection"] is False


def test_pilot_not_scientific_no_winner(v10c_artifacts: dict) -> None:
    pilot = v10c_artifacts["v10c_pilot_result.json"]
    assert pilot["scientific_experiment"] is False
    assert pilot["winner_lock_created"] is False
    assert pilot["candidate"]["not_winner"] is True


def test_pilot_seed_population_generations(v10c_artifacts: dict) -> None:
    pilot = v10c_artifacts["v10c_pilot_result.json"]
    assert pilot["optimizer_seed"] == v10c.PILOT_SEED == 3042
    assert pilot["population_size"] == 12
    assert pilot["bpso_evaluated_generations"] == 2
    assert pilot["bgwo_evaluated_iterations"] == 2
    assert pilot["allocated_bpso_requests"] == 24
    assert pilot["allocated_bgwo_requests"] == 24
    assert pilot["allocated_total_requests"] == 48


def test_pilot_uses_locked_cardinality_pool_and_offsets(v10c_artifacts: dict) -> None:
    pilot = v10c_artifacts["v10c_pilot_result.json"]
    assert pilot["configuration"]["cardinality_pool"] == list((4, 8, 11, 14, 18, 22, 26, 30, 34, 38))
    assert pilot["configuration"]["elite_count"] == 3
    assert pilot["configuration"]["design_identifier"] == "BPSO_BGWO_SEQUENTIAL_50_50_ELITE3"


def test_pilot_is_derived_from_locked_protocol_not_production(v10c_artifacts: dict) -> None:
    pilot = v10c_artifacts["v10c_pilot_result.json"]
    assert pilot["production_configuration"] is False
    assert pilot["production_campaign_executed"] is False
    config = v10c.pilot_configuration()
    assert config.per_run_request_allocation == 48
    assert config.five_run_request_allocation == 240


def test_production_8_8_configuration_never_run() -> None:
    locked = v10c.hybrid_config_from_yaml()
    assert locked.bpso_evaluated_generations == 8
    assert locked.bgwo_evaluated_iterations == 8
    assert locked.per_run_request_allocation == 192
    assert locked.five_run_request_allocation == 960
    summary = json.loads((PILOT_DIR / "v10c_execution_summary.json").read_text(encoding="utf-8"))
    assert summary["production_campaign_executed"] is False
    assert summary["pilot_allocated_requests"] == 48


def test_pilot_has_no_production_double_run_budget() -> None:
    summary = json.loads((PILOT_DIR / "v10c_execution_summary.json").read_text(encoding="utf-8"))
    pilot = json.loads((PILOT_DIR / "v10c_pilot_result.json").read_text(encoding="utf-8"))
    assert summary["pilot_allocated_requests"] == 48
    assert pilot["accounting"]["total_candidate_requests"] == 48


def test_pilot_no_frozen_winner_injection(v10c_artifacts: dict) -> None:
    pilot = v10c_artifacts["v10c_pilot_result.json"]
    identity_checks = pilot["forbidden_identity_checks"]
    assert identity_checks, "identity checks must be recorded"
    assert all(value is True for value in identity_checks.values())
    forbidden = pilot["elite_transfer"]["elite_masks_sha256"]
    candidate_hash = pilot["candidate"]["mask_sha256"]
    assert candidate_hash not in FROZEN_WINNERS.values()
    assert all(digest not in FROZEN_WINNERS.values() for digest in forbidden)


def test_pilot_test_never_accessed(v10c_artifacts: dict) -> None:
    pilot = v10c_artifacts["v10c_pilot_result.json"]
    assert pilot["test_accessed"] is False
    assert pilot["test_used_for_fitness"] is False
    assert pilot["test_used_for_winner_selection"] is False


def test_pilot_status_pass_and_stop_reason(v10c_artifacts: dict) -> None:
    pilot = v10c_artifacts["v10c_pilot_result.json"]
    assert pilot["status"] == "PASS"
    assert pilot["stop_reason"] == "FIXED_BUDGET_EXHAUSTED"


def test_pilot_convergence_history_covers_both_phases(v10c_artifacts: dict) -> None:
    pilot = v10c_artifacts["v10c_pilot_result.json"]
    phases = [row["phase"] for row in pilot["convergence_history"]]
    assert phases.count("BPSO") == 2
    assert phases.count("BGWO") == 2
    assert pilot["evaluated_generation_count"] == 4


def test_pilot_elites_distinct_selection_zero_calls(v10c_artifacts: dict) -> None:
    pilot = v10c_artifacts["v10c_pilot_result.json"]
    elite = pilot["elite_transfer"]
    assert elite["configured_elite_count"] == 3
    assert elite["placed_elite_count"] == 3
    assert elite["elite_selection_evaluator_calls"] == 0
    assert len(set(elite["elite_masks_sha256"])) == len(elite["elite_masks_sha256"])


def test_pilot_uses_shared_run_local_cache(v10c_artifacts: dict) -> None:
    pilot = v10c_artifacts["v10c_pilot_result.json"]
    assert pilot["accounting"]["cache_hits"] >= pilot["elite_transfer"]["placed_elite_count"]
    assert pilot["accounting"]["bgwo_cache_hits"] >= pilot["elite_transfer"]["placed_elite_count"]


def test_pilot_essence_metric_fields_finite(v10c_artifacts: dict) -> None:
    pilot = v10c_artifacts["v10c_pilot_result.json"]
    candidate = pilot["candidate"]
    for field in ("average_precision", "f1", "recall", "precision", "roc_auc"):
        assert math.isfinite(float(candidate[field]))


# ---------------------------------------------------------------------------
# Accounting audit
# ---------------------------------------------------------------------------


def test_accounting_gate_passed(v10c_artifacts: dict) -> None:
    assert v10c_artifacts["v10c_accounting_audit.json"]["status"] == "PASS"
    assert all(
        value is True
        for value in v10c_artifacts["v10c_accounting_audit.json"]["checks"].values()
    )


def test_accounting_requests_equals_unique_plus_cache(v10c_artifacts: dict) -> None:
    pilot = v10c_artifacts["v10c_pilot_result.json"]["accounting"]
    accounting = v10c_artifacts["v10c_accounting_audit.json"]["reported"]
    assert accounting["candidate_requests"] == accounting["unique_evaluations"] + accounting["cache_hits"]
    assert accounting["candidate_requests"] == pilot["total_candidate_requests"]


def test_accounting_exactly_48_candidate_requests(v10c_artifacts: dict) -> None:
    accounting = v10c_artifacts["v10c_accounting_audit.json"]["reported"]
    assert accounting["candidate_requests"] == 48


def test_accounting_bpso_bgwo_split_exact_24_24(v10c_artifacts: dict) -> None:
    accounting = v10c_artifacts["v10c_accounting_audit.json"]["reported"]
    assert accounting["bpso_requests"] == 24
    assert accounting["bgwo_requests"] == 24
    assert accounting["bpso_requests"] + accounting["bgwo_requests"] == accounting["candidate_requests"]


def test_accounting_no_hidden_evaluations(v10c_artifacts: dict) -> None:
    accounting = v10c_artifacts["v10c_accounting_audit.json"]["reported"]
    pilot = v10c_artifacts["v10c_pilot_result.json"]
    assert accounting["unique_evaluations"] == pilot["evaluator_instrumentation"]["objective_evaluations"]
    assert accounting["unique_evaluations"] == accounting["evaluator_calls"]
    assert accounting["bpso_unique_evaluations"] + accounting["bgwo_new_unique_evaluations"] == accounting["unique_evaluations"]


def test_accounting_cache_span_total(v10c_artifacts: dict) -> None:
    accounting = v10c_artifacts["v10c_accounting_audit.json"]["reported"]
    assert accounting["bpso_cache_hits"] + accounting["bgwo_cache_hits"] == accounting["cache_hits"]


def test_accounting_decision_tree_fits_exactly_5x_unique(v10c_artifacts: dict) -> None:
    accounting = v10c_artifacts["v10c_accounting_audit.json"]
    reported = accounting["reported"]
    assert reported["decision_tree_fits"] == reported["unique_evaluations"] * 5
    assert accounting["pilot_decision_tree_fits"] == reported["decision_tree_fits"]


def test_accounting_baseline_fits_reported_separately(v10c_artifacts: dict) -> None:
    accounting = v10c_artifacts["v10c_accounting_audit.json"]
    assert accounting["baseline_reproduction_decision_tree_fits"] == 10
    assert accounting["decision_tree_fits_reported_separately"] is True


# ---------------------------------------------------------------------------
# Test access audit
# ---------------------------------------------------------------------------


def test_test_access_audit_passed(v10c_artifacts: dict) -> None:
    access = v10c_artifacts["v10c_test_access_audit.json"]
    assert access["status"] == "PASS"
    assert all(value is True for value in access["checks"].values())


def test_test_access_flags(v10c_artifacts: dict) -> None:
    access = v10c_artifacts["v10c_test_access_audit.json"]
    assert access["recorded"]["test_accessed"] is False
    assert access["recorded"]["test_authorized"] is False
    assert access["recorded"]["test_used_for_fitness"] is False
    assert access["recorded"]["test_used_for_winner_selection"] is False


def test_test_access_splits(v10c_artifacts: dict) -> None:
    access = v10c_artifacts["v10c_test_access_audit.json"]
    assert access["recorded"]["allowed_splits"] == ["train", "validation"]
    assert access["recorded"]["optimizer_forbidden_splits"] == ["test"]


def test_frozen_test_split_identity(v10c_artifacts: dict) -> None:
    access = v10c_artifacts["v10c_test_access_audit.json"]
    split = access["recorded"]["frozen_test_split"]
    assert split["strategy"] == "order_grouped"
    assert split["seed"] == 42
    assert split["sizes"]["test"] == 6000
    assert len(split["raw_file_sha256"]) == 64


def test_historical_final_test_locks_not_selection(v10c_artifacts: dict) -> None:
    access = v10c_artifacts["v10c_test_access_audit.json"]
    historical = access["recorded"]["historical_final_test_locks"]
    assert historical["v09e_result_lock"]["test_used_for_selection"] is False
    assert historical["v09f_result_lock"]["final_test_accessed"] is False


# ---------------------------------------------------------------------------
# Execution summary and product-of-execution requirements
# ---------------------------------------------------------------------------


def test_execution_summary_all_steps_pass(v10c_artifacts: dict) -> None:
    summary = v10c_artifacts["v10c_execution_summary.json"]
    assert summary["status"] == "PASS"
    assert all(step == "PASS" for step in summary["step_status"].values())


def test_execution_summary_governance_headers(v10c_artifacts: dict) -> None:
    summary = v10c_artifacts["v10c_execution_summary.json"]
    assert summary["label"] == v10c.PILOT_LABEL
    assert summary["eligibility"] == v10c.PILOT_ELIGIBILITY
    assert summary["protocol_classification"] == v10b.PROTOCOL_CLASSIFICATION
    assert summary["protocol_config_sha256"] == v10b.HYBRID_YAML_SHA256
    assert summary["starting_checkpoint"] == v10c.STARTING_CHECKPOINT
    assert summary["winner_lock_created"] is False
    assert summary["production_campaign_executed"] is False


def test_all_eight_artifacts_exist_and_are_json() -> None:
    expected = (
        "v10c_preflight.json",
        "v10c_baseline_reproduction.json",
        "v10c_leakage_audit.json",
        "v10c_comparator_audit.json",
        "v10c_pilot_result.json",
        "v10c_accounting_audit.json",
        "v10c_test_access_audit.json",
        "v10c_execution_summary.json",
    )
    for name in expected:
        path = PILOT_DIR / name
        assert path.is_file()
        json.loads(path.read_text(encoding="utf-8"))


def test_artifacts_under_governed_results_dir() -> None:
    assert str(PILOT_DIR.resolve()).startswith(str(v10c.PROJECT_ROOT / "results"))


def test_frozen_artifact_hashes_snapshot_has_16_entries(v10c_artifacts: dict) -> None:
    summary = v10c_artifacts["v10c_execution_summary.json"]
    assert len(summary["frozen_artifact_hashes"]) >= 15
    assert all(len(digest) == 64 for digest in summary["frozen_artifact_hashes"].values())


def test_governed_yaml_and_locks_still_frozen(v10c_artifacts: dict) -> None:
    snapshot = v10c_artifacts["v10c_execution_summary.json"]["frozen_artifact_hashes"]
    key = "config/hybrid_v10.yaml"
    assert snapshot[key] == v10b.HYBRID_YAML_SHA256


def test_no_winner_lock_created_by_pipeline() -> None:
    for name in ("v10_winner_lock.json", "v10c_winner_lock.json", "hybrid_winner_lock.json"):
        assert not (v10c.PROJECT_ROOT / "results" / name).exists()


def test_pilot_deterministic_single_run_not_repeated() -> None:
    summary = json.loads((PILOT_DIR / "v10c_execution_summary.json").read_text(encoding="utf-8"))
    assert summary["pilot_seed"] == 3042
    assert summary["pilot_allocated_requests"] == 48
    # only ONE pilot artifact set was produced for seed 3042
    assert len(list(PILOT_DIR.glob("v10c_pilot_result.json"))) == 1