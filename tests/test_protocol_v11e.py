from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "experiments_v11e.yaml"
PROTOCOL_PATH = PROJECT_ROOT / "docs" / "v11e_experiment_protocol.md"
PROTOCOL_LOCK_PATH = (
    PROJECT_ROOT / "results" / "blockchain" / "v11e" / "v11e_experiment_protocol_lock.json"
)
V11C_LOCK_PATH = (
    PROJECT_ROOT / "results" / "blockchain" / "v11c" / "v11c_mapping_lock.json"
)
V11D_LOCK_PATH = (
    PROJECT_ROOT / "results" / "blockchain" / "v11d" / "v11d_ai_risk_result_lock.json"
)

EXPECTED_UPSTREAM = {
    "v11a_protocol_lock_semantic_sha256": (
        "aaf3ac3139e8296fb7f976a8a7ed9f18eb317af2f73b10cd0208b6bb870ebd8d"
    ),
    "v11c_mapping_semantic_sha256": (
        "c992429b2165f18649d36b0339260d9a95e1ae8524d7dfd586a168504d3cc765"
    ),
    "v11d_result_lock_semantic_sha256": (
        "440d351f4254cc74c2779b35fd161acecf452b071448a3592131d19ed87f5641"
    ),
    "v11d_artifact_semantic_sha256": (
        "ab7b1c8ef82035dc01005d8ac3ebaad379a77dd4bed743dbee5cae37b573186e"
    ),
    "v11d_artifact_disk_sha256": (
        "70b1d74c800baaa5d060a5d22739b4d142a578523a847a094899cb9746239b7b"
    ),
    "v11d_summary_semantic_sha256": (
        "d91c0102273a55379bc498d82323b670be523044f49de74c9f8234e8559afc6c"
    ),
    "v11d_summary_disk_sha256": (
        "99adcc25588e4bf431e61c52c68b6f06907bd55d7df32daf67852b39ece5007d"
    ),
}

EXPECTED_WORKLOADS = [100, 250, 500, 1000, 2500]
EXPECTED_SEEDS = [522, 523, 524]
EXPECTED_ATTACKS = [
    "PA-01",
    "PA-02",
    "PA-03",
    "PA-04",
    "PA-05",
    "PA-06",
    "PA-07",
    "PA-08",
    "PA-09",
]
EXPECTED_EXPERIMENT_IDS = [
    "E01",
    "E02",
    "E03",
    "E04",
    "E05",
    "E06",
    "E07",
    "E08",
    "E09",
    "E10",
]


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lock() -> dict:
    return json.loads(PROTOCOL_LOCK_PATH.read_text(encoding="utf-8"))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_sha256(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_stage_is_final_experiment_protocol_lock(cfg, lock):
    assert cfg["stage"] == "V1.1-E2B"
    assert cfg["artifact_kind"] == "V11E2_EXPERIMENT_PROTOCOL_LOCK"
    assert lock["stage"] == "V1.1-E2B"
    assert lock["artifact_kind"] == "V11E2_EXPERIMENT_PROTOCOL_LOCK"


def test_scientific_matrix_resolution_is_go(cfg, lock):
    assert cfg["scientific_matrix_resolution"] == "V1.1-E2A (FROZEN, GO)"
    assert lock["semantic_payload"]["scientific_matrix_resolution"] == (
        "V11E2A_SCIENTIFIC_MATRIX_RESOLVED_GO"
    )


def test_starting_checkpoint_head_matches_current_main(cfg, lock):
    expected = "415e58ce260251dbbf1efbff9bbaab0a23c86a9c"
    assert cfg["starting_checkpoint"]["head"] == expected
    assert cfg["starting_checkpoint"]["head_origin_main"] == expected
    assert lock["semantic_payload"]["starting_checkpoint"]["head_origin_main"] == expected
    assert lock["execution_metadata"]["head_origin_main"] == expected


def test_governed_population_counts_are_frozen(cfg, lock):
    population_cfg = cfg["governed_population"]
    assert population_cfg["governed_order_count"] == 4588
    assert population_cfg["low"] == 0
    assert population_cfg["medium"] == 4582
    assert population_cfg["high"] == 6
    assert cfg["validation_rows"] == 6000
    payload = lock["semantic_payload"]["governed_population"]
    assert payload["governed_order_count"] == 4588
    assert payload["risk_band_counts"] == {"LOW": 0, "MEDIUM": 4582, "HIGH": 6}
    assert payload["validation_rows"] == 6000


def test_no_low_synthesis_and_no_high_oversampling(cfg, lock):
    assert "no LOW synthesis" in cfg["governed_population"]["risk_rule"]
    assert "no HIGH oversampling" in cfg["governed_population"]["risk_rule"]
    assert lock["semantic_payload"]["governed_population"]["risk_rule"] == (
        "not rebalanced; NOT stratified; no LOW synthesis; no HIGH oversampling"
    )


def test_workloads_are_nested_prefixes_frozen(cfg, lock):
    assert cfg["workloads"] == EXPECTED_WORKLOADS
    assert lock["semantic_payload"]["workloads"] == EXPECTED_WORKLOADS
    assert 4588 not in EXPECTED_WORKLOADS


def test_seeds_and_repetitions_are_frozen(cfg, lock):
    assert cfg["blockchain_seeds"] == EXPECTED_SEEDS
    assert cfg["repetitions"] == 3
    assert lock["semantic_payload"]["blockchain_seeds"] == EXPECTED_SEEDS
    assert lock["semantic_payload"]["repetitions"] == 3


def test_policy_b0_is_light_baseline(cfg, lock):
    assert cfg["policies"]["B0"]["checks"] == ["c1", "c2", "c3", "c4"]
    assert cfg["policies"]["B0"]["validators"] == 1
    assert lock["semantic_payload"]["validation_policies"]["B0"]["checks"] == (
        ["c1", "c2", "c3", "c4"]
    )
    assert lock["semantic_payload"]["validation_policies"]["B0"]["validators"] == 1


def test_policy_b1_is_heavy_baseline(cfg, lock):
    heavy = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
    assert cfg["policies"]["B1"]["checks"] == heavy
    assert cfg["policies"]["B1"]["validators"] == 3
    assert cfg["policies"]["B1"]["quorum"] == 3
    assert lock["semantic_payload"]["validation_policies"]["B1"]["checks"] == heavy
    assert lock["semantic_payload"]["validation_policies"]["B1"]["validators"] == 3
    assert lock["semantic_payload"]["validation_policies"]["B1"]["quorum"] == 3


def test_policy_p_is_ai_linked_adaptive_with_frozen_bands(cfg, lock):
    cfg_p = cfg["policies"]["P"]
    lock_p = lock["semantic_payload"]["validation_policies"]["P"]
    assert cfg_p["mode"] == "AI_DRIVEN_ADAPTIVE_ALV"
    assert cfg_p["risk_input"] == "GOVERNED_AI_RISK"
    assert lock_p["mode"] == "AI_DRIVEN_ADAPTIVE_ALV"
    assert lock_p["risk_input"] == "GOVERNED_AI_RISK"
    assert lock_p["bands"]["LOW"] == {
        "checks": ["c1", "c2", "c3", "c4"],
        "validators": 1,
    }
    assert lock_p["bands"]["MEDIUM"] == {
        "checks": ["c1", "c2", "c3", "c4", "c5", "c6"],
        "validators": 1,
    }
    assert lock_p["bands"]["HIGH"] == {
        "checks": ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"],
        "validators": 3,
    }


def test_attacks_are_frozen_pa01_to_pa09(cfg, lock):
    assert cfg["attacks"] == EXPECTED_ATTACKS
    assert lock["semantic_payload"]["attacks"] == EXPECTED_ATTACKS


def test_attack_instance_contract_is_frozen(cfg, lock):
    contract = cfg["attack_instance_contract"]
    assert contract["same_attacked_copies_reused_across_b0_b1_p"] is True
    assert contract["same_attacked_copies_reused_across_e02_e03_e04"] is True
    assert contract["attack_generation_injection_outside_validation_timing"] is True
    assert contract["pa08_pa09_donor_read_only"] is True
    assert contract["pa08_pa09_target_and_donor_distinct"] is True
    assert lock["semantic_payload"]["attack_instance_contract"] == contract


def test_experiment_matrix_is_complete_and_frozen(cfg, lock):
    ids_cfg = [e["experiment_id"] for e in cfg["experiment_matrix"]]
    assert ids_cfg == EXPECTED_EXPERIMENT_IDS
    ids_lock = [e["experiment_id"] for e in lock["semantic_payload"]["experiment_matrix"]]
    assert ids_lock == EXPECTED_EXPERIMENT_IDS
    cfg_by_id = {e["experiment_id"]: e for e in cfg["experiment_matrix"]}
    lock_by_id = {e["experiment_id"]: e for e in lock["semantic_payload"]["experiment_matrix"]}
    assert cfg_by_id.keys() == lock_by_id.keys()


def test_e01_functional_correctness_is_frozen(lock):
    e01 = lock["semantic_payload"]["experiment_matrix"][0]
    assert e01["experiment_id"] == "E01"
    assert e01["name"] == "FUNCTIONAL_CORRECTNESS"
    assert e01["policies"] == ["B0", "B1", "P"]
    assert e01["attacks"] == []
    assert e01["integrity_detection"] == "NO"


def test_e02_integrity_detection_binds_all_attacks(lock):
    e02 = lock["semantic_payload"]["experiment_matrix"][1]
    assert e02["experiment_id"] == "E02"
    assert e02["name"] == "INTEGRITY_TAMPER_DETECTION"
    assert e02["policies"] == ["B0", "B1", "P"]
    assert e02["attacks"] == EXPECTED_ATTACKS
    assert e02["detection_semantics"] == "THAT_POLICY_REJECTS_ATTACKED_CHAIN"
    assert e02["integrity_detection"] == "YES"


def test_e03_localization_undetected_counts_as_not_localized(lock):
    e03 = lock["semantic_payload"]["experiment_matrix"][2]
    assert e03["experiment_id"] == "E03"
    assert e03["name"] == "TAMPER_LOCALIZATION"
    assert e03["localization_denominator"] == (
        "ALL_INJECTED_ATTACKS_UNDETECTED_COUNTS_AS_NOT_LOCALIZED"
    )
    assert e03["integrity_detection"] == "YES"


def test_e04_fault_isolation_binds_all_attacks(lock):
    e04 = lock["semantic_payload"]["experiment_matrix"][3]
    assert e04["experiment_id"] == "E04"
    assert e04["name"] == "ORDER_LEVEL_FAULT_ISOLATION"
    assert e04["attacks"] == EXPECTED_ATTACKS
    assert e04["integrity_detection"] == "YES"


def test_e05_ai_linked_uses_only_governed_risk(lock):
    e05 = lock["semantic_payload"]["experiment_matrix"][4]
    assert e05["experiment_id"] == "E05"
    assert e05["name"] == "AI_LINKED_ADAPTIVE_BEHAVIOR"
    assert e05["policies"] == ["P"]
    assert e05["matched_reference"] == ["B0", "B1"]
    assert e05["risk_rule"] == "MUST_USE_GOVERNED_AI_RISK"


def test_e06_execution_time_is_principal_timing(lock):
    e06 = lock["semantic_payload"]["experiment_matrix"][5]
    assert e06["experiment_id"] == "E06"
    assert e06["name"] == "EXECUTION_TIME_OVERHEAD"
    assert e06["timing"] == "PRINCIPAL"
    assert e06["attacks"] == []


def test_e07_memory_is_computational_proxy_not_energy(lock):
    e07 = lock["semantic_payload"]["experiment_matrix"][6]
    assert e07["experiment_id"] == "E07"
    assert e07["name"] == "MEMORY_OVERHEAD"
    assert e07["measurement"] == "tracemalloc_fresh_worker"
    assert e07["unit"] == "MiB"
    assert e07["label"] == "COMPUTATIONAL_MEMORY_PROXY"


def test_e08_storage_is_policy_independent(lock):
    e08 = lock["semantic_payload"]["experiment_matrix"][7]
    assert e08["experiment_id"] == "E08"
    assert e08["name"] == "STORAGE_OVERHEAD"
    assert e08["policy_independent"] is True
    assert e08["policies"] == []


def test_e09_throughput_counts_orders_per_second(lock):
    e09 = lock["semantic_payload"]["experiment_matrix"][8]
    assert e09["experiment_id"] == "E09"
    assert e09["name"] == "THROUGHPUT"
    assert e09["primary"] == ["validated_orders_per_second"]


def test_e10_green_proxy_is_derived_only_with_no_fourth_campaign(lock):
    e10 = lock["semantic_payload"]["experiment_matrix"][9]
    assert e10["experiment_id"] == "E10"
    assert e10["name"] == "RESOURCE_GREEN_PROXY"
    assert e10["derived_only"] is True
    assert e10["no_fourth_campaign"] is True
    assert e10["sources"] == ["E06", "E07", "E08", "E09"]


def test_timing_excludes_ai_and_attack_generation(cfg, lock):
    timing_cfg = cfg["timing_contract"]
    timing_lock = lock["semantic_payload"]["timing_contract"]
    assert timing_cfg["boundary"] == "validation policy call only"
    assert "ai_generation" in timing_cfg["exclude"]
    assert "ai_inference" in timing_cfg["exclude"]
    assert "attack_generation" in timing_cfg["exclude"]
    assert "attack_injection" in timing_cfg["exclude"]
    assert timing_cfg["counterbalance_across_seeds"] is True
    assert timing_lock == timing_cfg


def test_aggregation_is_descriptive_only_without_inference(cfg, lock):
    agg_cfg = cfg["aggregation"]
    agg_lock = lock["semantic_payload"]["aggregation"]
    assert agg_cfg["continuous"] == ["mean", "median", "std_ddof_1", "min", "max"]
    assert agg_cfg["paired_differences"] == [
        "B1_minus_B0",
        "P_minus_B0",
        "P_minus_B1",
    ]
    banned = agg_cfg["do_not_preregister"]
    assert "confidence_intervals" in banned
    assert "significance_tests" in banned
    assert "bootstrap_intervals" in banned
    assert "order_level_inferential_tests" in banned
    assert agg_lock == agg_cfg


def test_risk_band_treatment_low_absent_medium_dominant_high_natural(cfg, lock):
    low_cfg = cfg["risk_band_treatment"]["LOW"]
    assert low_cfg["count"] == 0
    assert low_cfg["marker"] == "LOW_ABSENT_IN_GOVERNED_POPULATION"
    low_lock = lock["semantic_payload"]["risk_band_treatment"]["LOW"]
    assert low_lock["count"] == 0
    medium_lock = lock["semantic_payload"]["risk_band_treatment"]["MEDIUM"]
    assert medium_lock["count"] == 4582
    high_lock = lock["semantic_payload"]["risk_band_treatment"]["HIGH"]
    assert high_lock["count"] == 6
    assert "natural sampling only" in high_lock["rule"]


def test_energy_governance_forbids_joules_and_tdp(cfg, lock):
    energy_cfg = cfg["energy"]
    assert energy_cfg["direct_energy"] == "DIRECT_ENERGY_UNAVAILABLE"
    for banned in ["Joules", "Wh", "TDP x runtime", "physical_energy_savings"]:
        assert banned in energy_cfg["never_report"]
    assert lock["semantic_payload"]["energy"] == energy_cfg


def test_fairness_contract_is_frozen(cfg, lock):
    fairness_cfg = cfg["fairness_contract"]
    assert fairness_cfg["exact_order_ids"] is True
    assert fairness_cfg["exact_attack_instance"] is True
    assert fairness_cfg["exact_seed"] is True
    assert fairness_cfg["exact_workload_size"] is True
    assert fairness_cfg["exact_timing_boundary"] is True
    assert fairness_cfg["same_machine"] is True
    assert fairness_cfg["only_validation_policy_differs"] is True
    assert lock["semantic_payload"]["fairness_contract"] == fairness_cfg


def test_protocol_lock_recomputes_semantic_sha256(lock):
    payload = lock["semantic_payload"]
    assert lock["semantic_result_lock_sha256"] == _semantic_sha256(payload)
    assert lock["semantic_payload"]["artifact_fingerprints"][
        "protocol_config_sha256"
    ] == _sha256_file(CONFIG_PATH)
    assert lock["semantic_payload"]["artifact_fingerprints"][
        "protocol_document_sha256"
    ] == _sha256_file(PROTOCOL_PATH)


def test_protocol_lock_tamper_is_detected(lock):
    tampered = dict(lock)
    tampered["semantic_result_lock_sha256"] = "f" * 64
    assert tampered["semantic_result_lock_sha256"] != _semantic_sha256(
        tampered["semantic_payload"]
    )


def test_upstream_semantic_bindings_match_on_disk(lock):
    upstream = lock["semantic_payload"]["upstream"]
    assert upstream["v11c_mapping_semantic_sha256"] == _read_json(V11C_LOCK_PATH)[
        "semantic_result_lock_sha256"
    ]
    assert upstream["v11d_result_lock_semantic_sha256"] == _read_json(V11D_LOCK_PATH)[
        "semantic_result_lock_sha256"
    ]


def test_upstream_bindings_match_frozen_protocol_doc(lock):
    upstream = lock["semantic_payload"]["upstream"]
    expected = {
        "v11a_protocol_lock_sha256": (
            "aaf3ac3139e8296fb7f976a8a7ed9f18eb317af2f73b10cd0208b6bb870ebd8d"
        ),
        "v11c_mapping_semantic_sha256": (
            "c992429b2165f18649d36b0339260d9a95e1ae8524d7dfd586a168504d3cc765"
        ),
        "v11d_result_lock_semantic_sha256": (
            "440d351f4254cc74c2779b35fd161acecf452b071448a3592131d19ed87f5641"
        ),
        "v11d_artifact_semantic_sha256": (
            "ab7b1c8ef82035dc01005d8ac3ebaad379a77dd4bed743dbee5cae37b573186e"
        ),
        "v11d_artifact_disk_sha256": (
            "70b1d74c800baaa5d060a5d22739b4d142a578523a847a094899cb9746239b7b"
        ),
        "v11d_summary_semantic_sha256": (
            "d91c0102273a55379bc498d82323b670be523044f49de74c9f8234e8559afc6c"
        ),
        "v11d_summary_disk_sha256": (
            "99adcc25588e4bf431e61c52c68b6f06907bd55d7df32daf67852b39ece5007d"
        ),
    }
    assert set(upstream.keys()) == set(expected.keys())
    for key, value in expected.items():
        assert upstream[key] == value


def test_execution_metadata_records_zero_access(lock):
    metadata = lock["execution_metadata"]
    assert metadata["experiment_status"] == "NOT_EXECUTED"
    assert metadata["test_access_count"] == 0
    assert metadata["ai_fit_count"] == 0
    assert metadata["ai_inference_count"] == 0
    assert metadata["experiment_execution_count"] == 0


def test_test_isolation_contract_is_frozen(cfg, lock):
    isolation_cfg = cfg["test_isolation"]
    assert isolation_cfg["test_access_count"] == 0
    assert isolation_cfg["ai_fit_count"] == 0
    assert isolation_cfg["ai_inference_count"] == 0
    assert isolation_cfg["experiment_execution_count"] == 0
    assert lock["semantic_payload"]["test_isolation"] == isolation_cfg


def test_protocol_document_states_required_boundaries():
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    for marker in [
        "DIRECT_ENERGY_UNAVAILABLE",
        "E01",
        "E02",
        "E03",
        "E04",
        "E05",
        "E06",
        "E07",
        "E08",
        "E09",
        "E10",
        "4588",
        "4582",
        "LOW = 0",
        "HIGH = 6",
        "LOW_ABSENT_IN_GOVERNED_POPULATION",
        "HIGH_NOT_OBSERVED_IN_CONDITION",
        "GOVERNED_AI_RISK",
    ]:
        assert marker in text


def test_protocol_document_forbids_joules_wh_and_tdp():
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    for banned in ["Joules", "Wh", "TDP", "energy savings"]:
        assert banned in text


def test_protocol_lock_file_is_complete_and_stage_correct(lock):
    assert lock["stage"] == "V1.1-E2B"
    assert lock["semantic_payload"]["stage"] == "V1.1-E2B"
    assert lock["semantic_payload"]["protocol_version"] == (
        "v1.1-e2b-final-experiment-protocol-lock-1"
    )
    assert lock["semantic_payload"]["upstream_lock_stage"] == "V1.1-E1"
    assert lock["semantic_payload"]["energy"]["direct_energy"] == (
        "DIRECT_ENERGY_UNAVAILABLE"
    )