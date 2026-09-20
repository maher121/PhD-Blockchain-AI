from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "ablation_v10g.yaml"
PROTOCOL_PATH = PROJECT_ROOT / "docs" / "v10g_elite_transfer_ablation_protocol.md"
LOCK_PATH = PROJECT_ROOT / "results" / "hybrid" / "v10g" / "v10g_protocol_lock.json"
V10D_LOCK_PATH = PROJECT_ROOT / "results" / "hybrid" / "v10d" / "v10d_winner_lock.json"
V10F_LOCK_PATH = PROJECT_ROOT / "results" / "hybrid" / "v10f" / "v10f_result_lock.json"

STARTING_COMMIT = "b086be7de998d18b6154301c0f29c5f43cb48d41"
V10D_SEMANTIC_SHA256 = "1c258dc1a90ad78197d25e62bbc9fca24904cbcb77edafd24b12726059e49d84"
V10F_SEMANTIC_SHA256 = "f714b4a95070ee26718b3b2f8dc2ed820c5c2dadb7c5fb58fa1c09f95505b412"
V10D_FILE_SHA256 = "1603cf0faff9027c2e5bcfb53345338317963d55ffc048afcfca5f974923bf79"
V10F_FILE_SHA256 = "0e24e485b74489756725dc6da7ef5e197dd63951a667fa70761758cfa8997a8f"
EXPECTED_PAYLOAD_FIELDS = {
    "additional_secondary_endpoints",
    "ancestry",
    "architecture",
    "budgets",
    "cache_policy",
    "computational_accounting",
    "convergence_analysis",
    "data_governance",
    "evaluator",
    "fail_closed_conditions",
    "feasibility_constraints",
    "fitness_metrics",
    "frozen_dependencies",
    "interpretation_rules",
    "lock_semantics",
    "paired_execution",
    "primary_inferential_endpoints",
    "primary_supporting_endpoint",
    "protocol_artifacts",
    "protocol_identity",
    "ranking_rules",
    "repair_and_clamping",
    "required_audits",
    "rng_policy",
    "roadmap",
    "safeguard_secondary_endpoint",
    "scientific_question",
    "seed_policy",
    "stability_analysis",
    "statistical_analysis",
    "study_role",
    "v10f_relationship",
    "variant_definitions",
    "winner_governance",
}


def _lf_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _canonical_semantic_sha256(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def payload(lock: dict) -> dict:
    return lock["semantic_payload"]


def test_01_protocol_status_and_schema(cfg: dict, lock: dict, payload: dict):
    assert cfg["stage"] == lock["stage"] == "V1.0-G1"
    assert cfg["status"] == lock["status"] == "ABLATION_PROTOCOL_LOCKED"
    assert lock["artifact_kind"] == "V10G_ABLATION_PROTOCOL_LOCK"
    assert lock["schema_version"] == "v1.0-g-ablation-protocol-lock-1"
    assert set(payload) == EXPECTED_PAYLOAD_FIELDS


def test_02_starting_ancestry(cfg: dict, payload: dict):
    assert cfg["ancestry"]["starting_git_commit"] == STARTING_COMMIT
    assert payload["ancestry"]["starting_git_commit"] == STARTING_COMMIT


def test_03_v10d_winner_lock_ancestry(cfg: dict, payload: dict):
    frozen = json.loads(V10D_LOCK_PATH.read_text(encoding="utf-8"))
    assert frozen["semantic_lock_sha256"] == V10D_SEMANTIC_SHA256
    assert cfg["ancestry"]["v10d_winner_semantic_lock_sha256"] == V10D_SEMANTIC_SHA256
    assert payload["ancestry"]["v10d_winner_semantic_lock_sha256"] == V10D_SEMANTIC_SHA256
    assert payload["ancestry"]["v10d_winner_id"] == "HYBRID-K13"


def test_04_v10f_result_lock_ancestry(cfg: dict, payload: dict):
    frozen = json.loads(V10F_LOCK_PATH.read_text(encoding="utf-8"))
    assert frozen["semantic_result_lock_sha256"] == V10F_SEMANTIC_SHA256
    assert cfg["ancestry"]["v10f_semantic_result_lock_sha256"] == V10F_SEMANTIC_SHA256
    assert payload["ancestry"]["v10f_semantic_result_lock_sha256"] == V10F_SEMANTIC_SHA256


def test_05_exact_two_arm_definitions(cfg: dict, payload: dict):
    variants = cfg["variant_definitions"]
    ids = {
        variants["with_elite_transfer"]["variant_id"],
        variants["without_elite_transfer"]["variant_id"],
    }
    assert ids == {"WITH_ELITE_TRANSFER", "WITHOUT_ELITE_TRANSFER"}
    assert variants["with_elite_transfer"]["elite_count_placed"] == 3
    assert variants["without_elite_transfer"]["elite_count_placed"] == 0
    assert payload["variant_definitions"]["manipulated_factor"] == "elite transfer present vs absent"


def test_06_exact_dimensions(cfg: dict, payload: dict):
    assert cfg["architecture"]["dimensions"] == payload["architecture"]["dimensions"] == 43


def test_07_phase_and_run_request_accounting(cfg: dict, payload: dict):
    budget = cfg["budgets"]
    assert budget["bpso_requests_per_run"] == 12 * 8 == 96
    assert budget["bgwo_requests_per_run"] == 12 * 8 == 96
    assert budget["total_requests_per_run"] == 96 + 96 == 192
    assert payload["budgets"]["total_requests_per_run"] == 192


def test_08_requests_per_arm(cfg: dict, payload: dict):
    assert cfg["budgets"]["requests_per_arm"] == 192 * 5 == 960
    assert payload["budgets"]["requests_per_arm"] == 960


def test_09_full_ablation_requests(cfg: dict, payload: dict):
    assert cfg["budgets"]["requests_total_ablation"] == 960 * 2 == 1920
    assert payload["budgets"]["requests_total_ablation"] == 1920


def test_10_optimizer_seeds(cfg: dict, payload: dict):
    expected = [3042, 3043, 3044, 3045, 3046]
    assert cfg["seed_policy"]["optimizer_seeds"] == expected
    assert payload["seed_policy"]["optimizer_seeds"] == expected
    assert payload["seed_policy"]["paired_by_optimizer_seed"] is True


def test_11_model_attack_seeds(cfg: dict, payload: dict):
    expected = [42, 43, 44, 45, 46]
    assert cfg["seed_policy"]["model_attack_seeds"] == expected
    assert payload["seed_policy"]["model_attack_seeds"] == expected
    assert payload["evaluator"]["model_attack_seed_count"] == len(expected)


def test_12_no_early_stopping(cfg: dict, payload: dict):
    assert cfg["architecture"]["early_stopping"] is False
    assert payload["architecture"]["early_stopping"] is False


def test_13_test_is_inaccessible(cfg: dict, payload: dict):
    for governance in (cfg["data_governance"], payload["data_governance"]):
        assert governance["allowed_splits"] == ["train", "validation"]
        assert governance["test_features_accessible"] is False
        assert governance["test_labels_accessible"] is False
        assert governance["final_test_evaluation"] is False
        assert governance["fail_closed"] is True


def test_14_winner_replacement_prohibited(cfg: dict, payload: dict):
    for governance in (cfg["winner_governance"], payload["winner_governance"]):
        assert governance["v10d_winner_id"] == "HYBRID-K13"
        assert governance["winner_reselection_performed"] is False
        assert governance["winner_replaced"] is False


def test_15_ap_and_f1_are_only_primary_inferential_endpoints(cfg: dict, payload: dict):
    expected = ["validation_average_precision", "validation_f1"]
    config_ids = [item["endpoint_id"] for item in cfg["endpoint_hierarchy"]["primary_inferential_endpoints"]]
    lock_ids = [item["endpoint_id"] for item in payload["primary_inferential_endpoints"]]
    assert config_ids == lock_ids == expected


def test_16_k_is_primary_supporting_only(cfg: dict, payload: dict):
    config_k = cfg["endpoint_hierarchy"]["primary_supporting_endpoint"]
    lock_k = payload["primary_supporting_endpoint"]
    assert config_k["endpoint_id"] == lock_k["endpoint_id"] == "selected_feature_count_k"
    assert config_k["discrete_outcome"] is lock_k["discrete_outcome"] is True
    assert config_k["required_integer_differences"] == lock_k["required_integer_differences"] == 5
    assert lock_k["required_direction_counts"] == ["negative", "zero", "positive"]


def test_17_k_standalone_inference_is_false(cfg: dict, payload: dict):
    assert cfg["endpoint_hierarchy"]["primary_supporting_endpoint"]["standalone_inferential_evidence_permitted"] is False
    assert payload["primary_supporting_endpoint"]["standalone_inferential_evidence_permitted"] is False
    assert payload["primary_supporting_endpoint"]["approximate_t_ci_descriptive_only"] is True


def test_18_recall_is_safeguard(cfg: dict, payload: dict):
    config_recall = cfg["endpoint_hierarchy"]["safeguard_secondary_endpoint"]
    lock_recall = payload["safeguard_secondary_endpoint"]
    assert config_recall["endpoint_id"] == lock_recall["endpoint_id"] == "validation_recall"
    assert "safeguard" in lock_recall["role"]


def test_19_exact_t_critical(cfg: dict, payload: dict):
    assert cfg["statistical_analysis"]["t_critical"] == 2.7764451051977987
    assert payload["statistical_analysis"]["t_critical"] == 2.7764451051977987
    assert payload["statistical_analysis"]["paired_sample_size"] == 5
    assert payload["statistical_analysis"]["degrees_of_freedom"] == 4


def test_20_no_p_values_or_significance_language(cfg: dict, payload: dict):
    for statistics in (cfg["statistical_analysis"], payload["statistical_analysis"]):
        assert statistics["p_values_permitted"] is False
        assert statistics["significance_language_permitted"] is False
        assert statistics["allowed_interval_language"] == ["CI excludes zero.", "Direction uncertain."]


def test_21_no_candidate_level_replication(cfg: dict, payload: dict):
    assert cfg["statistical_analysis"]["candidate_level_replication_permitted"] is False
    assert payload["statistical_analysis"]["scientific_unit"] == "paired_optimizer_seed"
    assert payload["statistical_analysis"]["candidate_level_replication_permitted"] is False


def test_22_exact_rng_derivations(cfg: dict, payload: dict):
    for rng in (cfg["rng_policy"], payload["rng_policy"]):
        assert rng["bit_generator"] == "PCG64"
        assert rng["bpso_stream"] == "PCG64(optimizer_seed)"
        assert rng["bgwo_common_stream"] == "PCG64(optimizer_seed + 10000)"
        assert rng["bgwo_filler_stream"] == "PCG64(SeedSequence([optimizer_seed, 0x56313047, 0x46494C4C]))"
        assert rng["cardinality_pool"] == [4, 8, 11, 14, 18, 22, 26, 30, 34, 38]
        assert rng["duplicate_rejection"] is False
    assert payload["rng_policy"]["filler_protocol_tag"] == 0x56313047
    assert payload["rng_policy"]["filler_stream_tag"] == 0x46494C4C


def test_23_filler_stream_excludes_bpso_information(cfg: dict, payload: dict):
    expected = ["elites", "BPSO candidates", "fitness values", "cache contents", "previous winners", "TEST information"]
    assert cfg["rng_policy"]["filler_forbidden_inputs"] == expected
    assert payload["rng_policy"]["filler_forbidden_inputs"] == expected


def test_24_row_zero_invariant(cfg: dict, payload: dict):
    config_rng = cfg["rng_policy"]["common_random_number_invariants"]
    lock_rng = payload["rng_policy"]["common_random_number_invariants"]
    assert config_rng["row_0_byte_identical"] is True
    assert lock_rng["row_0_byte_identical"] is True
    assert "K43" in payload["variant_definitions"]["with_elite_transfer"]["row_0_rule"]


def test_25_common_rows_and_rng_invariants(cfg: dict, payload: dict):
    for rng in (cfg["rng_policy"], payload["rng_policy"]):
        invariants = rng["common_random_number_invariants"]
        assert invariants["paired_bpso_outputs_byte_identical"] is True
        assert invariants["rows_4_11_byte_identical"] is True
        assert invariants["bgwo_common_rng_state_before_updates_identical"] is True


def test_26_cache_separation_between_arms(cfg: dict, payload: dict):
    assert cfg["cache_policy"]["independent_between_arms"] is True
    assert payload["cache_policy"]["independent_between_arms"] is True
    assert payload["paired_execution"]["cache_independence_between_arms"] is True


def test_27_shared_cache_within_run(cfg: dict, payload: dict):
    assert cfg["cache_policy"]["shared_within_run"] is True
    assert payload["cache_policy"]["shared_within_run"] is True
    assert "shared across BPSO and BGWO" in payload["cache_policy"]["scope"]


def test_28_fail_closed_rules(cfg: dict, payload: dict):
    assert cfg["fail_closed_conditions"] == payload["fail_closed_conditions"]
    joined = "\n".join(payload["fail_closed_conditions"])
    for required in ["BPSO outputs differ", "rows 4-11 differ", "WITHOUT calls elite selection", "TEST features or labels"]:
        assert required in joined
    assert all(payload["required_audits"].values())


def test_29_semantic_lock_recomputes_exactly(lock: dict, payload: dict):
    assert _canonical_semantic_sha256(payload) == lock["semantic_protocol_lock_sha256"]
    assert lock["semantic_protocol_lock_sha256"] == "2914819f4c63b50bf58a8dfa19af1c5b05539f6997b3f447c3a891b1852acf5c"


def test_30_markdown_lf_sha_matches_lock(payload: dict):
    assert _lf_sha256(PROTOCOL_PATH) == payload["protocol_artifacts"]["markdown_lf_sha256"]


def test_31_yaml_lf_sha_matches_lock(payload: dict):
    assert _lf_sha256(CONFIG_PATH) == payload["protocol_artifacts"]["yaml_lf_sha256"]


def test_32_no_circular_hash_dependency(lock: dict):
    digest = lock["semantic_protocol_lock_sha256"]
    assert digest not in PROTOCOL_PATH.read_text(encoding="utf-8")
    assert digest not in CONFIG_PATH.read_text(encoding="utf-8")
    assert yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["semantic_lock_rules"]["protocol_artifacts_embed_semantic_lock_hash"] is False


def test_33_no_test_derived_content(cfg: dict, payload: dict):
    assert cfg["data_governance"]["test_derived_ranking"] is False
    assert payload["data_governance"]["test_derived_ranking"] is False
    assert payload["winner_governance"]["threshold_tuning_performed"] is False
    assert payload["winner_governance"]["model_tuning_performed"] is False
    assert payload["winner_governance"]["feature_reselection_performed"] is False


def test_34_protected_prior_locks_unchanged():
    assert hashlib.sha256(V10D_LOCK_PATH.read_bytes()).hexdigest() == V10D_FILE_SHA256
    assert hashlib.sha256(V10F_LOCK_PATH.read_bytes()).hexdigest() == V10F_FILE_SHA256


def test_35_frozen_evaluator_and_constraints(cfg: dict, payload: dict):
    expected_parameters = {"class_weight": "balanced", "max_depth": 5, "min_samples_leaf": 20}
    assert cfg["evaluator"]["parameters"] == expected_parameters
    assert payload["evaluator"]["parameters"] == expected_parameters
    assert payload["evaluator"]["prediction_threshold"] == 0.5
    assert payload["feasibility_constraints"]["ap_relative_loss_max"] == 0.05
    assert payload["feasibility_constraints"]["f1_relative_loss_max"] == 0.05
    assert payload["feasibility_constraints"]["recall_relative_loss_max"] == 0.1
    assert payload["feasibility_constraints"]["boundary_tolerance"] == 1e-12


def test_36_frozen_deterministic_ranking(payload: dict):
    ranking = payload["ranking_rules"]
    assert ranking["aggregate_score_permitted"] is False
    assert ranking["feasible_order"] == [
        "smaller_k", "higher_average_precision", "higher_f1", "higher_recall", "canonical_lexicographic_mask"
    ]
    assert ranking["infeasible_order"] == [
        "lower_total_normalized_constraint_violation", "higher_average_precision", "higher_f1",
        "higher_recall", "smaller_k", "canonical_lexicographic_mask",
    ]


def test_37_fresh_execution_and_v10f_non_rerun(payload: dict):
    assert payload["study_role"]["fresh_execution_both_arms"] is True
    assert payload["study_role"]["historical_v10d_with_as_primary_arm"] is False
    assert payload["v10f_relationship"]["v10f_frozen"] is True
    assert payload["v10f_relationship"]["resource_campaign_rerun"] is False
    assert payload["v10f_relationship"]["physical_energy_claims"] is False


def test_38_semantic_payload_has_no_runtime_identity_or_results(payload: dict):
    forbidden_keys = {
        "timestamp", "created_at", "created_at_utc", "hostname", "environment_id",
        "experimental_results", "observations", "winner_result",
    }

    def walk(value):
        if isinstance(value, dict):
            assert not forbidden_keys.intersection(value)
            for key, nested in value.items():
                if key.endswith("_path"):
                    assert not str(nested).startswith(("/", "\\"))
                    assert ":" not in str(nested)
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)
        elif isinstance(value, float):
            assert math.isfinite(value)

    walk(payload)


def test_39_pilot_omitted_by_default(payload: dict):
    g3 = next(item for item in payload["roadmap"] if item["stage"] == "V1.0-G3")
    assert g3["purpose"] == "pilot omitted by default"


def test_40_protocol_document_contains_approved_endpoint_hierarchy():
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "### Primary Inferential Endpoints" in text
    assert "### Primary-Supporting Mechanistic/Parsimony Endpoint" in text
    assert "### Safeguard / Secondary Endpoint" in text
    assert "K cannot independently establish inferential benefit" in normalized
