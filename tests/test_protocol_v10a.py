from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "hybrid_v10.yaml"
PROTOCOL_PATH = PROJECT_ROOT / "docs" / "v10_hybrid_bpso_bgwo_protocol.md"
VALIDATION_LOCK = PROJECT_ROOT / "results" / "feature_selection" / "validation_lock.json"
BPSO_EXEC = PROJECT_ROOT / "results" / "bpso" / "v08c_execution_summary.json"
BPSO_LOCK = PROJECT_ROOT / "results" / "bpso" / "v08c_winner_lock.json"
BGWO_EXEC = PROJECT_ROOT / "results" / "bgwo" / "v09d" / "v09d_execution_summary.json"
BGWO_LOCK = PROJECT_ROOT / "results" / "bgwo" / "v09d" / "v09d_winner_lock.json"


@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# --------------------------------------------------------------------------- #
# 1. YAML schema / required top-level keys
# --------------------------------------------------------------------------- #
def test_yaml_loads_and_declares_required_sections(cfg):
    required = [
        "stage",
        "kind",
        "protocol_classification",
        "scope",
        "dataset",
        "predictive_baseline",
        "optimization_objective",
        "deterministic_ranking",
        "search_classifier",
        "hybrid_design",
        "seeds",
        "initialization",
        "forbidden_injected_subsets",
        "knowledge_transfer",
        "cache_and_accounting",
        "early_stopping_policy",
        "test_access_gate",
        "production_runs",
        "winner_selection",
        "final_test_plan",
        "resource_policy",
        "energy_policy",
        "optimizer_efficiency_comparison",
        "ablation_plan",
        "statistics",
        "claim_governance",
        "roadmap",
        "governed_sources",
    ]
    assert cfg["stage"] == "V1.0-A"
    assert cfg["kind"] == "SCIENTIFIC_PROTOCOL_LOCK"
    for key in required:
        assert key in cfg, f"missing YAML section: {key}"


def test_scope_forbids_execution(cfg):
    scope = cfg["scope"]
    for flag, expected in [
        ("hybrid_optimizer_execution_allowed", False),
        ("model_training_execution_allowed", False),
        ("final_test_access_allowed", False),
        ("bpso_rerun_allowed", False),
        ("bgwo_rerun_allowed", False),
        ("optimizer_parameter_tuning_allowed", False),
        ("resource_benchmark_execution_allowed", False),
    ]:
        assert scope[flag] is expected, f"{flag} must be false in V1.0-A"


def test_protocol_classification_present(cfg):
    assert cfg["protocol_classification"] == "HYBRID_PROTOCOL_LOCKED_FAIR_BUDGET"


# --------------------------------------------------------------------------- #
# 2. Dataset / feature space / baseline / margins / objective
# --------------------------------------------------------------------------- #
def test_dataset_governance_frozen(cfg):
    dataset = cfg["dataset"]
    assert dataset["feature_space_count"] == 43
    assert dataset["split"]["seed"] == 42
    assert dataset["ground_truth"] == "is_attack"
    assert dataset["optimizer_visible_splits"] == ["train", "validation"]
    assert dataset["optimizer_forbidden_splits"] == ["test"]


def test_baseline_matches_frozen_validation_lock(cfg):
    baseline = cfg["predictive_baseline"]["metrics"]
    lock = json.loads(VALIDATION_LOCK.read_text(encoding="utf-8"))
    frozen = lock["roles"]["full_baseline"]["metrics"]
    for metric in ["average_precision", "f1", "recall"]:
        assert baseline[metric] == pytest.approx(frozen[metric], rel=1e-12)


def test_preservation_margins_match_frozen_lock(cfg):
    margins = cfg["predictive_baseline"]["preservation_constraints"]
    assert margins["average_precision_relative_loss_max"] == pytest.approx(0.05)
    assert margins["f1_relative_loss_max"] == pytest.approx(0.05)
    assert margins["recall_relative_loss_max"] == pytest.approx(0.10)


def test_objective_excludes_resource_energy(cfg):
    objective = cfg["optimization_objective"]
    assert objective["formulation"] == "constrained_minimum_cardinality"
    assert objective["resource_or_energy_in_fitness"] is False
    for metric in ["wall_time_sec", "direct_energy", "serialized_model_bytes", "absolute_peak_rss_mib"]:
        assert metric in objective["excluded_from_fitness"]


def test_ranking_match_frozen_convention(cfg):
    ranking = cfg["deterministic_ranking"]
    assert ranking["feasible"] == [
        "smaller_k",
        "higher_average_precision",
        "higher_f1",
        "higher_recall",
        "canonical_tie_break",
    ]
    assert ranking["infeasible"] == [
        "lower_total_normalized_constraint_violation",
        "higher_average_precision",
        "higher_f1",
        "higher_recall",
        "smaller_k",
        "canonical_tie_break",
    ]


# --------------------------------------------------------------------------- #
# 3. Budget arithmetic
# --------------------------------------------------------------------------- #
def test_budget_arithmetic(cfg):
    budgets = cfg["hybrid_design"]["phase_budgets"]
    population = cfg["hybrid_design"]["population_size_per_phase"]
    bpso_phase = budgets["bpso_exploration_per_run_requests"]
    bgwo_phase = budgets["bgwo_refinement_per_run_requests"]
    per_run = budgets["per_run_requests_total"]
    total = budgets["five_run_requests_total"]
    feats = cfg["dataset"]["feature_space_count"]
    fits_total = budgets["allocated_decision_tree_fits_total"]
    fits_per_request = budgets["fits_per_request"]

    # 12 agents * 8 evaluated iterations per phase
    assert bpso_phase == population * 8 == 96
    assert bgwo_phase == population * 8 == 96
    # 50/50 split
    assert bpso_phase == bgwo_phase
    assert per_run == bpso_phase + bgwo_phase == 192
    assert total == per_run * 5 == 960
    assert per_run % population == 0
    assert per_run // population == 16
    assert fits_total == total * fits_per_request == 4800
    assert fits_per_request == len(cfg["seeds"]["model_attack_seeds"]) == 5
    assert feats == 43


def test_budget_inside_frozen_envelope(cfg):
    budgets = cfg["hybrid_design"]["phase_budgets"]
    per_run = budgets["per_run_requests_total"]
    # Frozen per-run request ranges: BPSO [132, 228], BGWO [132, 204]
    bpso = json.loads(BPSO_EXEC.read_text(encoding="utf-8"))
    bgwo = json.loads(BGWO_EXEC.read_text(encoding="utf-8"))
    bpso_frozen_total, bgwo_frozen_total = bpso["fitness_requests"], bgwo["candidate_requests"]
    assert bpso_frozen_total == 972 and bgwo_frozen_total == 768
    assert 132 <= per_run <= 204  # common frozen envelope
    assert budgets["five_run_requests_total"] <= bpso_frozen_total  # 960 <= 972
    assert budgets["five_run_requests_total"] > bgwo_frozen_total  # 960 > 768


# --------------------------------------------------------------------------- #
# 4. Phase totals and iteration semantics
# --------------------------------------------------------------------------- #
def test_phase_totals(cfg):
    design = cfg["hybrid_design"]
    assert design["name"] == "BPSO_BGWO_SEQUENTIAL_50_50_ELITE3"
    assert design["phase_budget_ratio_bpso_bgwo"] == [50, 50]
    assert design["phase_budgets"]["bpso_evaluated_generations_per_run"] == 8
    assert design["phase_budgets"]["bgwo_evaluated_iterations_per_run"] == 8
    # phase requests = population * evaluated iterations
    assert design["phase_budgets"]["bpso_exploration_per_run_requests"] == design["population_size_per_phase"] * 8
    assert design["phase_budgets"]["bgwo_refinement_per_run_requests"] == design["population_size_per_phase"] * 8


def test_early_stopping_decision_fixed_budget(cfg):
    policy = cfg["early_stopping_policy"]
    assert policy["method"] == "A_fixed_budget_no_early_stopping"
    assert policy["report_both_allocated_and_consumed"] is True


# --------------------------------------------------------------------------- #
# 5. Seed families
# --------------------------------------------------------------------------- #
def test_seed_families(cfg):
    seeds = cfg["seeds"]
    assert seeds["optimizer_seed_family_primary"] == [3042, 3043, 3044, 3045, 3046]
    assert seeds["model_attack_seeds"] == [42, 43, 44, 45, 46]
    assert seeds["dataset_split_seed"] == 42
    assert len(seeds["optimizer_seed_family_primary"]) == 5
    for family in seeds["forbidden_optimizer_seed_families"]:
        assert not set(family) & set(seeds["optimizer_seed_family_primary"])


def test_seed_families_match_frozen_evidence(cfg):
    seeds = cfg["seeds"]
    assert seeds["forbidden_optimizer_seed_families"] == [[1042, 1043, 1044, 1045, 1046], [2042, 2043, 2044, 2045, 2046]]
    bpso = json.loads(BPSO_EXEC.read_text(encoding="utf-8"))
    bgwo = json.loads(BGWO_EXEC.read_text(encoding="utf-8"))
    assert bpso["optimizer_seeds"] == [1042, 1043, 1044, 1045, 1046]
    assert bgwo["optimizer_seeds"] == [2042, 2043, 2044, 2045, 2046]
    assert bpso["model_attack_seeds"] == [42, 43, 44, 45, 46]


# --------------------------------------------------------------------------- #
# 6. Forbidden winner injection / initialization
# --------------------------------------------------------------------------- #
def test_forbidden_injection_denylist(cfg):
    denylist = cfg["forbidden_injected_subsets"]
    joined = "\n".join(str(item) for item in denylist).lower()
    assert "bpso-k10" in joined
    assert "bgwo-k14" in joined
    assert "mi-k11" in joined
    assert "test-derived" in joined
    assert "5da981b5" in joined
    assert "7ebb8233" in joined


def test_initialization_has_governed_anchors(cfg):
    init = cfg["initialization"]
    assert init["governed_anchors"]
    assert "wolf_0" in init["governed_anchors"][0]
    assert init["cardinality_repair"]["minimum_selected_features"] == 1
    assert init["cardinality_repair"]["dry_run_repair_budget"] == 0


def test_forbidden_optimizer_security_hashes_in_sources(cfg):
    bpso = json.loads(BPSO_EXEC.read_text(encoding="utf-8"))
    bgwo = json.loads(BGWO_EXEC.read_text(encoding="utf-8"))
    bpso_winner = json.loads(BPSO_LOCK.read_text(encoding="utf-8"))
    bgwo_winner = json.loads(BGWO_LOCK.read_text(encoding="utf-8"))
    assert bpso_winner["mask_sha256"].startswith("5da981b5")
    assert bgwo_winner["mask_sha256"].startswith("7ebb8233")
    assert bpso["cache_hits"] == 0
    assert bgwo["cache_hits"] == 0


# --------------------------------------------------------------------------- #
# 7. Knowledge transfer
# --------------------------------------------------------------------------- #
def test_knowledge_transfer_spec(cfg):
    kt = cfg["knowledge_transfer"]
    assert kt["elite_count"] == 3
    assert kt["transferred_candidates_count_against_budget"] is True
    assert kt["transition_evaluations_between_phases"] == 0
    assert kt["remaining_wolves_initialization"]
    assert "fewer_than_N_feasible_elites" in kt


def test_elite_transfer_within_budget(cfg):
    budgets = cfg["hybrid_design"]["phase_budgets"]
    kt = cfg["knowledge_transfer"]
    # 3 elites each counted within the 96 BGWO-phase requests
    assert kt["elite_count"] * 5 <= budgets["bgwo_refinement_per_run_requests"]


# --------------------------------------------------------------------------- #
# 8. Cache and accounting
# --------------------------------------------------------------------------- #
def test_cache_accounting(cfg):
    acc = cfg["cache_and_accounting"]
    assert acc["cache_scope"] == "one optimizer run (both phases), cleared between runs"
    assert acc["candidate_request_definition"]
    assert acc["unique_evaluation_definition"]
    assert acc["decision_tree_fit_definition"]
    assert acc["no_hidden_evaluations_between_phases"] is True
    assert acc["expected_allocated_fits_total"] == 4800


# --------------------------------------------------------------------------- #
# 9. TEST-access policy
# --------------------------------------------------------------------------- #
def test_test_access_gate_fail_closed(cfg):
    gate = cfg["test_access_gate"]
    assert gate["policy"] == "fail_closed"
    assert gate["required_states_during_search"] == {
        "test_accessed": False,
        "test_authorized": False,
        "test_used_for_winner_selection": False,
    }
    assert len(gate["test_accessible_only_after"]) == 3


def test_final_test_not_used_for_selection(cfg):
    plan = cfg["final_test_plan"]
    assert plan["winner_selection_forbidden_on_test"] is True
    assert "Hybrid-V1.0-winner" in plan["participants"]
    assert "BPSO-K10" in plan["participants"]
    assert "BGWO-K14" in plan["participants"]
    assert plan["statistics"]


# --------------------------------------------------------------------------- #
# 10. Energy policy
# --------------------------------------------------------------------------- #
def test_energy_policy_honest(cfg):
    energy = cfg["energy_policy"]
    assert energy["direct_energy_policy"]
    assert energy["fallback_marker"] == "DIRECT_ENERGY_UNAVAILABLE"
    assert "TDP" in energy["forbidden_inference"]


# --------------------------------------------------------------------------- #
# 11. Roadmap
# --------------------------------------------------------------------------- #
def test_roadmap_stages(cfg):
    roadmap = cfg["roadmap"]["stages"]
    expected = ["V1.0-A", "V1.0-B", "V1.0-C", "V1.0-D", "V1.0-E", "V1.0-F", "V1.0-G", "V1.0-H"]
    assert [stage.split(":")[0] for stage in roadmap] == expected


def test_ablation_plan_locked(cfg):
    ablation = cfg["ablation_plan"]
    assert "without elite transfer" in ablation["primary_contrast"]
    assert "with elite transfer" in ablation["primary_contrast"]
    assert ablation["scale"] == "single controlled ablation; no hyperparameter sweep"
    assert "validation only" in ablation["evaluation_scope"]


# --------------------------------------------------------------------------- #
# 12. Protocol SHA-256 recorded in the document
# --------------------------------------------------------------------------- #
def test_protocol_sha256_matches_document(cfg):
    actual = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    match = re.search(r"Protocol-config SHA-256:\s*`([0-9a-f]{64})`", text)
    assert match, "protocol SHA-256 not recorded in protocol document"
    assert match.group(1) == actual


def test_classification_constant_in_document(cfg):
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "HYBRID_PROTOCOL_LOCKED_FAIR_BUDGET" in text


# --------------------------------------------------------------------------- #
# 13. Frozen evidence totals recorded in YAML match artifacts
# --------------------------------------------------------------------------- #
def test_frozen_budget_totals_recorded(cfg):
    sources = cfg["governed_sources"]
    bpso = json.loads(BPSO_EXEC.read_text(encoding="utf-8"))
    bgwo = json.loads(BGWO_EXEC.read_text(encoding="utf-8"))
    assert sources["bpso_frozen_budget_requests"] == bpso["fitness_requests"] == 972
    assert sources["bpso_frozen_decision_tree_fits"] == bpso["actual_decision_tree_fits"] == 4860
    assert sources["bgwo_frozen_budget_requests"] == bgwo["candidate_requests"] == 768
    assert sources["bgwo_frozen_decision_tree_fits"] == bgwo["actual_decision_tree_fits"] == 3840
    assert abs(bpso["optimization_wall_time_sec"] - 645.5028752319995) < 1e-6
    assert abs(bgwo["optimizer_wall_time_sec"] - 325.42544312802784) < 1e-6


def test_claim_governance(cfg):
    claims = cfg["claim_governance"]
    asserted = [c.lower() for c in claims["prohibited_statements_without_direct_evidence"]]
    assert any("universally superior" in item for item in asserted)
    assert any("greener" in item for item in asserted)
    assert any("consumes less energy" in item for item in asserted)