from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "blockchain_v11.yaml"
PROTOCOL_PATH = PROJECT_ROOT / "docs" / "v11_blockchain_protocol.md"
PROTOCOL_LOCK = PROJECT_ROOT / "results" / "blockchain" / "v11a" / "v11a_protocol_lock.json"
V10D_LOCK = PROJECT_ROOT / "results" / "hybrid" / "v10d" / "v10d_winner_lock.json"
V10E_LOCK = PROJECT_ROOT / "results" / "hybrid" / "v10e" / "v10e_result_lock.json"
V10F_LOCK = PROJECT_ROOT / "results" / "hybrid" / "v10f" / "v10f_result_lock.json"
V10G_LOCK = PROJECT_ROOT / "results" / "hybrid" / "v10g" / "v10g_result_lock.json"


@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def protocol_text() -> str:
    return PROTOCOL_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def lock() -> dict:
    return json.loads(PROTOCOL_LOCK.read_text(encoding="utf-8"))


def _semantic_sha256(payload: dict) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


# --------------------------------------------------------------------------- #
# 1. Stage identity / YAML schema
# --------------------------------------------------------------------------- #
def test_yaml_loads_and_declares_required_sections(cfg):
    required = [
        "stage",
        "kind",
        "protocol_classification",
        "stage_identity",
        "upstream",
        "scope",
        "architecture",
        "dataco_semantics",
        "order_identity",
        "block_schema",
        "genesis",
        "events",
        "canonical_serialization",
        "hash",
        "payload",
        "ai_integration",
        "threat_model",
        "security_properties",
        "terminology",
        "validation",
        "adaptive_lightweight_validation",
        "baselines",
        "experiments_preregistered",
        "metrics",
        "workloads",
        "seeds",
        "energy_policy",
        "implementation_constraints",
        "future_boundaries",
        "governed_sources",
    ]
    assert cfg["stage"] == "V1.1-A"
    assert cfg["kind"] == "SCIENTIFIC_PROTOCOL_LOCK"
    assert cfg["protocol_classification"] == "BLOCKCHAIN_LIGHTWEIGHT_PROTOCOL_LOCKED"
    for key in required:
        assert key in cfg, f"missing YAML section: {key}"


def test_stage_identity_locks_design_only(cfg):
    identity = cfg["stage_identity"]
    assert identity["implementation_excluded"] is True
    assert identity["experiment_execution_excluded"] is True


# --------------------------------------------------------------------------- #
# 2. Scope: everything forbidden in V1.1-A
# --------------------------------------------------------------------------- #
def test_scope_forbids_everything(cfg):
    scope = cfg["scope"]
    for flag, expected in [
        ("blockchain_engine_implementation_allowed", False),
        ("blockchain_experiment_execution_allowed", False),
        ("adaptive_lightweight_validation_implementation_allowed", False),
        ("tampering_scenario_execution_allowed", False),
        ("ai_retraining_allowed", False),
        ("ai_model_reconstruction_allowed", False),
        ("v10_rerun_allowed", False),
        ("v10_artifacts_mutation_allowed", False),
        ("dataco_raw_read_allowed", False),
        ("split_or_test_access_allowed", False),
        ("pow_required_allowed", False),
        ("mining_allowed", False),
        ("external_platform_required_allowed", False),
    ]:
        assert scope[flag] is expected, f"{flag} must be false in V1.1-A"


# --------------------------------------------------------------------------- #
# 3. Upstream frozen V1.0 evidence
# --------------------------------------------------------------------------- #
def test_upstream_v10_frozen(cfg):
    up = cfg["upstream"]
    assert up["v10_superstage"] == "V1.0"
    assert up["v10_close_stage"] == "V1.0-I"
    assert up["v10_start_checkpoint_sha256"] == "813a49b93db3271a275097e44c947ba96a09f763"
    assert up["v10_closed_immutable"] is True
    assert up["ai_winner_configuration"] == "HYBRID-K13"
    assert up["ai_winner_selected_feature_count"] == 13
    assert up["ai_feature_space_dimension"] == 43


def test_winner_feature_manifest_matches_v10d(cfg):
    assert cfg["upstream"]["ai_winner_feature_manifest_sha256"] == json.loads(
        V10D_LOCK.read_text(encoding="utf-8")
    )["feature_manifest_sha256"]


def test_upstream_semantic_locks_match_frozen_artifacts(cfg):
    referenced = cfg["upstream"]["referenced_v10_semantic_locks"]
    artifacts = {
        "v10d_hybrid_winner_lock": V10D_LOCK,
        "v10e_result_lock": V10E_LOCK,
        "v10f_resource_lock": V10F_LOCK,
        "v10g_ablation_lock": V10G_LOCK,
    }
    for name, path in artifacts.items():
        record = json.loads(path.read_text(encoding="utf-8"))
        value = record.get("semantic_result_lock_sha256") or record.get("semantic_lock_sha256")
        assert referenced[name] == value, f"{name} does not match frozen artifact"
    assert referenced["v10d_hybrid_winner_lock"] == "1c258dc1a90ad78197d25e62bbc9fca24904cbcb77edafd24b12726059e49d84"


# --------------------------------------------------------------------------- #
# 4. Order-centric architecture
# --------------------------------------------------------------------------- #
def test_architecture_is_order_centric(cfg):
    arch = cfg["architecture"]
    assert arch["primary_classification"] == "ORDER_CENTRIC_PER_ORDER_LOGICAL_BLOCKCHAIN"
    assert arch["unit_of_chain"] == "order"
    assert arch["ordering_rule"] == "blocks are order-local; unrelated orders never appear in a shared block chain"
    required_core = {"chain_id (canonical Order Id)", "presented block schema"}
    assert required_core.issubset(set(arch["core_mandatory_components"]))
    assert "order" in arch["core_mandatory_components"][0].lower()


def test_summary_layer_is_optional_not_a_second_chain(cfg):
    summary = cfg["architecture"]["summary_optional_components"]
    assert isinstance(summary, list)
    assert "none proposed in V1.0" in summary[0]


# --------------------------------------------------------------------------- #
# 5. DataCo honesty
# --------------------------------------------------------------------------- #
def test_dataco_honest_semantics(cfg):
    ds = cfg["dataco_semantics"]
    assert ds["dataset_name"] == "DataCo SMART Supply Chain"
    assert ds["native_event_log"] is False
    assert ds["native_provenance_dag"] is False
    assert ds["never_claimed_as_observed"] is True


def test_event_provenance_classes_are_valid(cfg):
    for name, event in cfg["events"]["types"].items():
        assert event["provenance_class"] in {
            "OBSERVED", "DERIVED", "RESEARCH_GENERATED", "MODE_DEPENDENT",
        }, event
    assert cfg["events"]["types"]["AI_RISK_ASSESSED"]["provenance_class"] == "MODE_DEPENDENT"


# --------------------------------------------------------------------------- #
# 6. Order identity
# --------------------------------------------------------------------------- #
def test_order_identity_rules(cfg):
    oi = cfg["order_identity"]
    assert oi["canonical_field"] == "Order Id"
    assert oi["item_line_field"] == "Order Item Id"
    assert oi["kaggle_to_canonical_rule"] == "Order Id -> order_id (34-column canonical preprocessing)"
    assert oi["multiple_rows_per_order"] is True
    assert "exactly one logical chain per distinct canonical order_id" in oi["unique_chain_per_order_rule"]
    assert "deterministic" in oi["order_level_aggregation_policy"].lower()


# --------------------------------------------------------------------------- #
# 7. Block schema
# --------------------------------------------------------------------------- #
def test_block_schema_required_fields(cfg):
    schema = cfg["block_schema"]
    assert schema["schema_version"] == "1"
    fields = schema["fields"]
    for name in ["schema_version", "order_id", "block_index", "event_type", "event_timestamp",
                 "payload_digest", "ai_risk_reference", "validation_policy", "previous_hash", "block_hash"]:
        assert name in fields, f"missing block field: {name}"
    constraints = schema["constraints"]
    assert constraints["block_index_nonnegative"] is True
    assert constraints["block_index_0_is_genesis"] is True
    assert constraints["previous_hash_null_only_index_0"] is True


# --------------------------------------------------------------------------- #
# 8. Genesis / events / canonical serialization / hashing
# --------------------------------------------------------------------------- #
def test_genesis_deterministic(cfg):
    genesis = cfg["genesis"]
    assert "no random nonce" in genesis["rule"]
    assert genesis["index"] == 0
    assert genesis["event_type"] == "ORDER_GENESIS"
    assert genesis["previous_hash"] == "null"


def test_event_catalog_static(cfg):
    assert cfg["events"]["static_catalog"] is True
    types = cfg["events"]["types"]
    for name in ["ORDER_GENESIS", "ORDER_CREATED", "SHIPMENT_RECORDED", "DELIVERY_STATUS_RECORDED", "AI_RISK_ASSESSED"]:
        assert name in types
    assert "fixed_event_sequence_rule" in cfg["events"]


def test_canonical_serialization_frozen(cfg):
    rule = cfg["canonical_serialization"]
    assert rule["encoding"] == "utf-8"
    assert rule["sort_keys"] is True
    assert rule["separators"] == [",", ":"]
    assert rule["ensure_ascii"] is True
    assert rule["allow_nan"] is False
    assert rule["trailing_newline"] is False


def test_hash_policy(cfg):
    h = cfg["hash"]
    assert h["algorithm"] == "sha256"
    assert h["integrity_only"] is True
    assert h["not_encryption"] is True
    assert h["confidentiality_claim_from_hash"] is False
    assert h["preimage"] == "canonical_serialization(block_without_block_hash)"
    assert h["preimage_excludes"] == ["block_hash"]


def test_payload_strategy_documented_tradeoff(cfg):
    payload = cfg["payload"]
    assert payload["primary_strategy"] == "DIGEST_MINIMAL_METADATA"
    assert payload["off_chain_repository"]
    assert "reproducibility choice" in payload["tradeoff_statement"]


# --------------------------------------------------------------------------- #
# 9. AI integration contract
# --------------------------------------------------------------------------- #
def test_ai_integration_honest(cfg):
    ai = cfg["ai_integration"]
    assert ai["reads_frozen_only"] is True
    assert ai["no_retraining"] is True
    assert ai["per_order_risk_existence_in_v10_frozen_artifacts"] is False
    assert "no persisted per-order AI risk predictions" in ai["frozen_ai_reference"]["evidence_kind"]
    ref = ai["frozen_ai_reference"]
    assert ref["configuration_id"] == "HYBRID-K13"
    assert ref["selected_feature_count"] == 13
    assert ref["feature_space_dimension"] == 43


def test_risk_bands_preregistered(cfg):
    rule = cfg["ai_integration"]["risk_band_from_score"]
    assert "LOW" in rule and "MEDIUM" in rule and "HIGH" in rule
    assert "preregistered" in cfg["ai_integration"]["threshold_preregistration"].lower()
    assert "final TEST" in cfg["ai_integration"]["threshold_preregistration"]


def test_two_distinct_risk_input_modes(cfg):
    modes = cfg["ai_integration"]["risk_input_modes"]
    assert set(modes.keys()) == {"GOVERNED_AI_RISK", "SYNTHETIC_RISK"}
    governed = modes["GOVERNED_AI_RISK"]
    assert governed["classification"] == "A"
    assert governed["role"] == "PRIMARY"
    assert governed["research_generated"] is False
    assert "HYBRID-K13 remains immutable" in governed["production_rule"]
    assert "provenance-locked" in governed["production_rule"]
    assert "before any ALV experiment consumes them" in governed["production_rule"]
    synthetic = modes["SYNTHETIC_RISK"]
    assert synthetic["classification"] == "B"
    assert synthetic["role"] == "SIMULATION_ABLATION_TESTING_ONLY"
    assert synthetic["research_generated"] is True


def test_synthetic_risk_never_labeled_ai_prediction(cfg):
    synthetic = cfg["ai_integration"]["risk_input_modes"]["SYNTHETIC_RISK"]
    assert "never presented as an AI prediction" in synthetic["rules"]
    assert any("never the sole evidence" in rule for rule in synthetic["rules"])
    assert cfg["ai_integration"]["per_order_risk_existence_in_v10_frozen_artifacts"] is False
    assert "no persisted per-order AI risk predictions" in cfg["ai_integration"]["frozen_ai_reference"]["evidence_kind"]
    doc = PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "never presented as an AI prediction" in doc.lower() or "never presented as an AI prediction" in doc


def test_hybrid_k13_cannot_be_changed_by_v11(cfg):
    assert cfg["upstream"]["hybrid_k13_immutable_in_v11"] is True
    assert cfg["upstream"]["ai_winner_configuration"] == "HYBRID-K13"
    ai = cfg["ai_integration"]
    assert ai["no_feature_reselection"] is True
    assert ai["no_optimizer_execution"] is True
    assert ai["no_test_based_threshold_tuning"] is True
    assert ai["frozen_ai_reference"]["configuration_id"] == "HYBRID-K13"
    assert cfg["scope"]["ai_retraining_allowed"] is False
    assert cfg["scope"]["ai_model_reconstruction_allowed"] is False


def test_main_ai_alv_experiment_requires_governed_risk(cfg):
    exp = cfg["experiments_preregistered"]
    e05 = exp["E05_ai_linked_adaptive_behavior"]
    assert e05["principal_evidence_mode"] == "GOVERNED_AI_RISK"
    assert e05["synthetic_as_sole_evidence"] is False
    assert "GOVERNED_AI_RISK" in e05["principal_evidence_rule"]
    assert "never be the sole evidence" in e05["principal_evidence_rule"]
    assert "never be the sole evidence" in exp["synthetic_risk_forbidden_uses"] or "sole evidence" in exp["synthetic_risk_forbidden_uses"]
    assert "GOVERNED_AI_RISK" in cfg["adaptive_lightweight_validation"]["risk_mode_policy"]
    assert "SYNTHETIC_RISK" in cfg["adaptive_lightweight_validation"]["risk_mode_policy"]


def test_alv_record_contract_requires_locked_artifact(cfg):
    contract = cfg["ai_integration"]["per_order_risk_record_contract"]
    fields = contract["fields"]
    for field in ["order_id", "risk_score", "risk_level", "model_configuration_provenance",
                  "hybrid_k13_provenance", "generation_stage_provenance", "record_digest"]:
        assert field in fields, f"missing contract field: {field}"
    assert contract["unlocked_consumption_forbidden"] is True
    assert "before ALV consumption" in contract["lock_before_use"]
    assert "SHA-256 over canonical serialization" in fields["record_digest"]["description"] or fields["record_digest"]["type"] == "string"


def test_generation_mapping_validation_kept_distinct(cfg):
    steps = cfg["ai_integration"]["risk_generation_vs_mapping_vs_validation"]
    for key in ["risk_score_generation", "risk_level_mapping", "blockchain_validation_policy"]:
        assert key in steps
    assert "preregistered" in steps["risk_level_mapping"]
    assert "independent of generation mode" in steps["risk_level_mapping"]


# --------------------------------------------------------------------------- #
# 9b. Corrected V1.1 future stage boundaries
# --------------------------------------------------------------------------- #
def test_stage_boundaries_corrected(cfg):
    stage_map = cfg["future_boundaries"]["stage_map"]
    assert stage_map["V1.1-A"] == "Blockchain Protocol & Architecture Lock"
    assert stage_map["V1.1-B"] == "Core Lightweight Blockchain Engine Implementation"
    assert stage_map["V1.1-C"] == "DataCo Order/Event Mapping and Blockchain Construction"
    assert stage_map["V1.1-D"] == "Governed Per-Order AI Risk Artifact and Frozen-AI Integration"
    assert stage_map["V1.1-E"] == "AI-Driven Adaptive Lightweight Validation and Integrity/Tampering Experiments"
    assert stage_map["V1.1-F"] == "Resource-Efficiency / Green Evaluation"
    assert stage_map["V1.1-G"] == "Final Blockchain Scientific Analysis and Notebook"


def test_v11b_is_engine_not_risk_generation(cfg):
    v11b = cfg["future_boundaries"]["v11b"]
    assert "engine" in v11b.lower()
    assert "GOVERNED_AI_RISK" not in v11b
    assert "provenance-lock" not in v11b
    assert "produce" not in v11b.lower()
    assert "freeze" not in v11b.lower()


def test_v11c_is_dataco_mapping_and_construction(cfg):
    v11c = cfg["future_boundaries"]["v11c"]
    assert "dataco" in v11c.lower()
    assert "order_id" in v11c
    assert "construct" in v11c.lower()


def test_v11d_generates_locks_and_freezes_governed_risk(cfg):
    v11d = cfg["future_boundaries"]["v11d"]
    assert "GOVERNED_AI_RISK" in v11d
    assert "provenance-lock" in v11d
    assert "freeze" in v11d.lower()
    assert "before" in v11d.lower()


def test_v11e_is_principal_alv_experiment_and_requires_v11d_lock(cfg):
    v11e = cfg["future_boundaries"]["v11e"]
    low = v11e.lower()
    assert "alv" in low or "adaptive" in low
    assert "v1.1-d" in low
    assert "forbidden" in low
    assert "provenance/lock" in low
    assert "synthetic_risk" in v11e or "SYNTHETIC_RISK" in v11e
    e05 = cfg["experiments_preregistered"]["E05_ai_linked_adaptive_behavior"]
    assert "V1.1-D" in e05["principal_evidence_rule"]
    assert "V1.1-E" in e05["principal_evidence_rule"]


# --------------------------------------------------------------------------- #
# 10. Threat model
# --------------------------------------------------------------------------- #
def test_threat_model_covers_nine_scenarios(cfg):
    scenarios = cfg["threat_model"]["scenarios"]
    assert list(scenarios.keys()) == [
        "PA_01", "PA_02", "PA_03", "PA_04", "PA_05", "PA_06", "PA_07", "PA_08", "PA_09",
    ]
    for key, item in scenarios.items():
        assert item["detection"], f"{key} missing detection"
        assert item["localization"], f"{key} missing localization"
    assert "generated experiment copies" in cfg["threat_model"]["tampering_execution_rule"]


# --------------------------------------------------------------------------- #
# 11. Security properties / claim boundaries
# --------------------------------------------------------------------------- #
def test_security_properties_bounded(cfg):
    props = cfg["security_properties"]
    for claimed in ["integrity", "deterministic verification", "traceability",
                    "order-level fault isolation", "AI-result provenance linkage"]:
        found = any(claimed in item for item in props["demonstrated_scope"])
        assert found, f"missing claimed property: {claimed}"
    for not_claimed in ["confidentiality", "anonymity", "decentralization", "BFT consensus",
                        "non-repudiation", "legal immutability"]:
        found = any(not_claimed in item for item in props["explicitly_not_claimed"])
        assert found, f"missing not_claimed entry: {not_claimed}"


def test_terminology_governance(cfg):
    terms = cfg["terminology"]
    assert "lightweight research prototype" in terms["preferred"]
    assert "full blockchain network" in terms["avoided"]
    assert "mining layer" in terms["avoided"]


# --------------------------------------------------------------------------- #
# 12. Validation and ALV
# --------------------------------------------------------------------------- #
def test_validation_authorized_validator_no_pow(cfg):
    validation = cfg["validation"]
    assert validation["mode"] == "AUTHORIZED_VALIDATOR_LIGHTWEIGHT"
    assert "all applied checks pass" in validation["baseline_acceptance"]
    for check in ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]:
        assert check in validation["core_checks"], f"missing core check {check}"


def test_alv_bands_monotone_and_preregistered(cfg):
    alv = cfg["adaptive_lightweight_validation"]
    assert alv["name"] == "ADAPTIVE_LIGHTWEIGHT_VALIDATION"
    assert alv["short_name"] == "ALV"
    low = alv["band_to_checks"]["LOW"]["checks"]
    medium = alv["band_to_checks"]["MEDIUM"]["checks"]
    high = alv["band_to_checks"]["HIGH"]["checks"]
    assert set(low) < set(medium) < set(high)
    assert alv["band_to_checks"]["LOW"]["validator_count"] == 1
    assert alv["band_to_checks"]["MEDIUM"]["validator_count"] == 1
    assert alv["band_to_checks"]["HIGH"]["validator_count"] == 3
    assert "escalates the block to MEDIUM validation" in alv["fail_safe"]
    assert "deterministic" in alv["tie_handling"]


def test_baselines_registered_and_fair(cfg):
    baselines = cfg["baselines"]
    assert set(baselines["registered"].keys()) == {"B0", "B1", "P"}
    assert "identical workload" in baselines["fairness_rule"]
    assert "only the validation policy differs" in baselines["fairness_rule"]


def test_experiments_registered_not_executed(cfg):
    exp = cfg["experiments_preregistered"]
    names = {k.split("_", 1)[1] for k in exp if k.startswith("E0") or k.startswith("E1")}
    assert {"functional_correctness", "integrity_tamper_detection", "tamper_localization",
            "order_level_fault_isolation", "ai_linked_adaptive_behavior",
            "execution_time_overhead", "memory_overhead", "storage_overhead",
            "throughput", "resource_and_green_proxy"}.issubset(names)
    assert exp["none_executed_in_v11a"] is True


def test_metrics_proxy_rule(cfg):
    metrics = cfg["metrics"]
    assert metrics["functional"] and metrics["performance"]
    assert metrics["adaptive"] and metrics["green"]
    assert "never labeled or inferred as measured Joules" in metrics["proxy_rule"]


# --------------------------------------------------------------------------- #
# 13. Workloads / seeds
# --------------------------------------------------------------------------- #
def test_workloads_frozen(cfg):
    workloads = cfg["workloads"]
    assert workloads["order_counts"] == [100, 250, 500, 1000, 2500]
    assert workloads["repetitions"] == 3
    assert "40000" in workloads["basis"]
    assert "deterministic" in workloads["sampling_rule"]


def test_seed_family_dedicated(cfg):
    seeds = cfg["seeds"]
    family = seeds["blockchain_seed_family"]
    assert family == [522, 523, 524]
    prior = seeds["recorded_prior_families"]
    joined = [s for fl in prior for s in fl]
    assert not set(family) & set(joined)
    assert set(joined) >= set([42, 43, 44, 45, 46, 1042, 3042, 3043, 3044, 3045, 3046])


# --------------------------------------------------------------------------- #
# 14. Energy / implementation / future boundaries
# --------------------------------------------------------------------------- #
def test_energy_policy_honest(cfg):
    energy = cfg["energy_policy"]
    assert energy["fallback_marker"] == "DIRECT_ENERGY_UNAVAILABLE"
    assert "TDP-times-time" in energy["forbidden_inference"]
    assert "measured Joules claim" in energy["forbidden_inference"]


def test_implementation_constraints(cfg):
    impl = cfg["implementation_constraints"]
    assert "local Python 3.12" in impl["language"]
    assert "none" in impl["external_platforms"]
    assert "read-only frozen V1.0 artifacts" in impl["upstream_consumption"]


def test_future_boundaries_no_empirical_claims(cfg):
    assert cfg["future_boundaries"]["claim_boundary"]
    assert "no empirical claims from V1.1-A" in cfg["future_boundaries"]["claim_boundary"]


# --------------------------------------------------------------------------- #
# 15. Protocol hash recorded in the document
# --------------------------------------------------------------------------- #
def test_protocol_config_sha256_matches_document(cfg, protocol_text):
    actual = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    match = re.search(r"Protocol-config SHA-256:\s*`([0-9a-f]{64})`", protocol_text)
    assert match, "protocol-config SHA-256 not recorded in protocol document"
    assert match.group(1) == actual


def test_classification_constant_in_document(protocol_text):
    assert "BLOCKCHAIN_LIGHTWEIGHT_PROTOCOL_LOCKED" in protocol_text


# --------------------------------------------------------------------------- #
# 16. Semantic protocol lock verification
# --------------------------------------------------------------------------- #
def test_lock_semantic_hash_recomputes(cfg, lock):
    semantic_payload = lock["semantic_payload"]
    recomputed = _semantic_sha256(semantic_payload)
    assert lock["semantic_result_lock_sha256"] == recomputed
    assert len(recomputed) == 64


def test_lock_artifact_fingerprints_match_files(lock):
    fp = lock["semantic_payload"]["artifact_fingerprints"]
    assert fp["protocol_document_sha256"] == hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    assert fp["protocol_config_sha256"] == hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def test_lock_reflects_stage_and_protocol_identity(cfg, lock):
    payload = lock["semantic_payload"]
    assert payload["artifact_kind"] == "V11A_PROTOCOL_LOCK"
    assert payload["stage"] == cfg["stage"]
    assert payload["starting_checkpoint"] == "813a49b93db3271a275097e44c947ba96a09f763"
    assert payload["protocol_identity"]["architecture"] == "ORDER_CENTRIC_PER_ORDER_LOGICAL_BLOCKCHAIN"
    assert payload["protocol_identity"]["proposed_mechanism"] == "ADAPTIVE_LIGHTWEIGHT_VALIDATION_ALV"
    assert payload["protocol_identity"]["governed_risk_artifact_stage"] == "V1.1-D"
    assert payload["protocol_identity"]["primary_ai_risk_mode"] == "GOVERNED_AI_RISK"
    assert payload["protocol_identity"]["ai_risk_input_modes"] == ["GOVERNED_AI_RISK", "SYNTHETIC_RISK"]