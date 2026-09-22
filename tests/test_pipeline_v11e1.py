"""V1.1-E1 governed AI-to-chain integration readiness tests.

These tests exercise the E1 orchestrator and its deterministic helpers with
hand-constructed synthetic fixtures. They never run experiments (E01-E10),
never fit models, never perform AI inference, never run optimizers, and never
append to a real corpus. The only real artifacts touched are the frozen
read-only provenance files re-verified by the preflight (D1/D3/V1.1-C locks,
the governed risk artifact, and the V1.1-C mapping summary).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import src.pipeline_v11e1 as p

PROJECT_ROOT = Path(__file__).resolve().parent.parent

D3_ARTIFACT_PATH = PROJECT_ROOT / "results" / "blockchain" / "v11d" / "v11d_governed_risk_artifact.json"


def _synthetic_order_rows(
    order_id: str,
    *,
    created: str = "01/05/2014 09:30",
    shipped: str = "01/08/2014 11:00",
    delivery: str = "COMPLETE",
) -> list[dict[str, object]]:
    return [
        {
            "Order Id": order_id,
            "Order Item Id": 1,
            "order date (DateOrders)": created,
            "shipping date (DateOrders)": shipped,
            "Delivery Status": delivery,
            "Order Status": "COMPLETE",
            "Order Item Quantity": 2,
            "Order Item Total": 99.90,
        },
        {
            "Order Id": order_id,
            "Order Item Id": 2,
            "order date (DateOrders)": created,
            "shipping date (DateOrders)": shipped,
            "Delivery Status": delivery,
            "Order Status": "COMPLETE",
            "Order Item Quantity": 1,
            "Order Item Total": 55.50,
        },
    ]


# --------------------------------------------------------------------------- #
# module identity and governance
# --------------------------------------------------------------------------- #


def test_module_identity():
    assert p.STAGE == "V1.1-E1"
    assert p.KIND == "V11E1_INTEGRATION_READINESS"
    assert p.PA_SCENARIOS == (
        "PA-01", "PA-02", "PA-03", "PA-04", "PA-05",
        "PA-06", "PA-07", "PA-08", "PA-09",
    )
    assert p.ALV_BASELINES == ("B0", "B1", "P")
    assert p.E1_MARKER == "V11E1_INTEGRATION_READINESS_COMPLETE_REVIEW_REQUIRED"


def test_energy_marker_frozen():
    assert p.ENERGY_MARKER_DIRECT_UNAVAILABLE == "DIRECT_ENERGY_UNAVAILABLE"


# --------------------------------------------------------------------------- #
# on-chain reference (c7 reconciliation) + payload digest
# --------------------------------------------------------------------------- #


def test_onchain_reference_is_c7_valid():
    record = p.synthetic_governed_record("1007", 0.5)
    reference = p.onchain_ai_reference(record)
    p.verify_onchain_reference(reference, record)
    assert tuple(reference) == tuple(p.ONCHAIN_REFERENCE_FIELD_ORDER)
    assert reference["mode"] == "GOVERNED_AI_RISK"
    assert reference["stage"] == "V1.1-D"
    assert reference["record_digest"] != record.record_digest  # c7 digest vs off-chain
    six = {key: reference[key] for key in (
        "order_id", "risk_score", "risk_level",
        "model_configuration_provenance",
        "hybrid_k13_provenance", "generation_stage_provenance",
    )}
    from src.blockchain_engine.validation import risk_record_digest
    assert reference["record_digest"] == risk_record_digest(six)
    assert not (set(reference) & p.FORBIDDEN_REFERENCE_KEYS)


def test_onchain_reference_rejects_tampered_score():
    record = p.synthetic_governed_record("1007", 0.5)
    reference = p.onchain_ai_reference(record)
    tampered = dict(reference)
    tampered["risk_score"] = 0.9
    with pytest.raises(p.V11E1Error):
        p.verify_onchain_reference(tampered, record)


def test_payload_digest_deterministic_and_content_free():
    a = p.synthetic_governed_record("1007", 0.5)
    b = p.synthetic_governed_record("1008", 0.5)
    ra = p.onchain_ai_reference(a)
    rb = p.onchain_ai_reference(b)
    assert p.ai_risk_assessed_payload_digest(ra) == p.ai_risk_assessed_payload_digest(dict(ra))
    assert p.ai_risk_assessed_payload_digest(ra) != p.ai_risk_assessed_payload_digest(rb)
    with pytest.raises(p.V11E1Error):
        p.ai_risk_assessed_payload_digest({"order_id": "1", "raw_features": []})


# --------------------------------------------------------------------------- #
# synthetic chain + c7 acceptance end-to-end
# --------------------------------------------------------------------------- #


def test_synthetic_chain_length_and_sequence():
    with_ai = p.build_synthetic_chain("2001", risk_level="HIGH", risk_score=0.9)
    assert with_ai.length == 5
    assert [b.event_type for b in with_ai.blocks] == [
        "ORDER_GENESIS", "ORDER_CREATED", "SHIPMENT_RECORDED",
        "DELIVERY_STATUS_RECORDED", "AI_RISK_ASSESSED",
    ]
    without_ai = p.build_synthetic_chain("2001", with_ai=False)
    assert without_ai.length == 4


def test_governed_last_block_passes_full_validation():
    record = p.synthetic_governed_record("3001", 0.5)
    reference = p.onchain_ai_reference(record)
    chain = p.build_synthetic_chain("3001", with_ai=False)
    p.extend_chain_with_ai_risk(chain, onchain_reference=reference)
    assert chain.length == 5
    from src.blockchain_engine.validation import validate_block
    result = validate_block(
        chain.last_block, p.TOTAL_CHECK_IDS, chain=chain.blocks, order_id="3001"
    )
    assert result.accepted, result.reasons


def test_extend_requires_delivery_head():
    chain = p.OrderChain("3002")
    chain.seed_genesis("2014-01-01T00:00:00Z")
    chain.append_event(
        "ORDER_CREATED",
        "2014-01-01T00:00:00Z",
        p.sha256_hex({"e": "oc"}),
    )
    record = p.synthetic_governed_record("3002", 0.5)
    reference = p.onchain_ai_reference(record)
    with pytest.raises(p.V11E1Error):
        p.extend_chain_with_ai_risk(chain, onchain_reference=reference)


def test_extend_rejects_cross_order_reference():
    chain = p.build_synthetic_chain("3003", with_ai=False)
    record = p.synthetic_governed_record("999999", 0.5)
    reference = p.onchain_ai_reference(record)
    with pytest.raises(p.V11E1Error):
        p.extend_chain_with_ai_risk(chain, onchain_reference=reference)


# --------------------------------------------------------------------------- #
# order-chain reconstruction + mapping binding
# --------------------------------------------------------------------------- #


def test_reconstruct_and_extend_governed_order():
    order_id = "1007"
    rows = _synthetic_order_rows(order_id)
    descriptor = p.reconstruct_pre_ai_chain(order_id, rows)
    assert descriptor["chain_length"] == 4
    entry = dict(descriptor)
    entry["passed_checks"] = list(p.VALIDATION_CHECKS_V11C)
    binding = p.verify_chain_matches_mapping_entry(descriptor, entry)
    assert binding["passed"]

    record = p.synthetic_governed_record(order_id, 0.5)
    outcome = p.extend_governed_order_chain(
        order_id, rows, record=record, mapping_entry=entry
    )
    assert outcome["chain_length"] == 5
    assert outcome["full_validation_accepted"] is True
    assert outcome["failed_checks"] == []
    assert outcome["historical_blocks_unaltered"] is True
    assert outcome["binding"]["passed"] is True


def test_coverage_index_rejects_rebind():
    order_id = "1008"
    rows = _synthetic_order_rows(order_id, created="01/05/2014 09:30", shipped="01/06/2014 09:00")
    descriptor = p.reconstruct_pre_ai_chain(order_id, rows)
    other = p.reconstruct_pre_ai_chain(order_id, _synthetic_order_rows(order_id, shipped="01/09/2014 09:00"))
    entry = dict(other)
    entry["passed_checks"] = list(p.VALIDATION_CHECKS_V11C)
    with pytest.raises(p.V11E1Error):
        p.verify_chain_matches_mapping_entry(descriptor, entry)


# --------------------------------------------------------------------------- #
# ALV routing + baselines (frozen policy)
# --------------------------------------------------------------------------- #


def test_alv_policy_frozen():
    low = p.alv_policy_for_level("LOW")
    medium = p.alv_policy_for_level("MEDIUM")
    high = p.alv_policy_for_level("HIGH")
    assert low["checks"] == ["c1", "c2", "c3", "c4"] and low["validator_count"] == 1
    assert medium["checks"] == ["c1", "c2", "c3", "c4", "c5", "c6"] and medium["validator_count"] == 1
    assert high["checks"] == ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"] and high["validator_count"] == 3
    assert p.alv_policy_for_level("MEDIUM")["fail_safe_level"] == "MEDIUM"
    with pytest.raises(p.V11E1Error):
        p.alv_policy_for_level("CRITICAL")


@pytest.mark.parametrize("level,score", [("LOW", 0.2), ("MEDIUM", 0.5), ("HIGH", 0.9)])
def test_alv_routes_each_band(level, score):
    chain = p.build_synthetic_chain("4002", risk_level=level, risk_score=score)
    result = p.validate_under_policy("P", chain.last_block, chain.blocks, chain.order_id)
    assert result.accepted
    assert result.validation_level == level
    assert result.validator_count == p.ALV_VALIDATOR_COUNTS[level]


def test_baselines_registered_and_over_identical_workload():
    fixture = p.baseline_readiness_fixture()
    results = fixture["results"]
    assert set(results) == {"B0", "B1", "P"}
    for policy in p.ALV_BASELINES:
        assert results[policy]["accepted"] is True
    assert results["B0"]["applied_checks"] == ["c1", "c2", "c3", "c4"]
    assert results["B0"]["validator_count"] == 1
    assert results["B1"]["validator_count"] == 3
    assert results["P"]["validation_level"] == "MEDIUM"
    with pytest.raises(p.V11E1Error):
        p.validate_under_policy("Q", None, (), "1")


def test_empty_low_band_accepted_and_supported():
    handling = p.empty_low_band_handling()
    assert handling["governed_low_count"] == 0
    assert handling["framework_supports_low_band"] is True
    assert handling["rebalancing"] == "NONE" and handling["retuning"] == "NONE"
    chain = p.build_synthetic_chain("4101", risk_level="LOW", risk_score=0.2)
    result = p.validate_under_policy("P", chain.last_block, chain.blocks, chain.order_id)
    assert result.accepted and result.validation_level == "LOW"


def test_fail_safe_escalates_malformed_reference():
    chain = p.build_synthetic_chain("4102", with_ai=False)
    head = chain.blocks[-1]
    result = p.validate_under_policy("P", head, chain.blocks, chain.order_id)
    assert result.validation_level == "MEDIUM"  # fail-safe for non-AI head block


# --------------------------------------------------------------------------- #
# tamper harness PA-01..PA-09
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scenario", p.PA_SCENARIOS)
def test_tamper_scenarios_detected(scenario):
    chain = p.build_synthetic_chain("5101", risk_level="MEDIUM", risk_score=0.5)
    outcome = p.run_tamper_scenario(
        chain.blocks, scenario, expected_order_id="5101", declared_length=5
    )
    assert outcome.applied is True
    assert outcome.detected is True, (
        f"{scenario} not detected: {outcome.detection_mechanism} | {outcome.failed_checks}"
    )
    assert outcome.tamper_kind == p.PA_TAMPER_KINDS[scenario]


def test_tamper_harness_never_mutates_original():
    chain = p.build_synthetic_chain("5102", risk_level="HIGH", risk_score=0.9)
    original = [block.block_hash for block in chain.blocks]
    for scenario in p.PA_SCENARIOS:
        p.run_tamper_scenario(chain.blocks, scenario, expected_order_id="5102", declared_length=5)
        assert [block.block_hash for block in chain.blocks] == original


def test_pa06_requires_ai_block():
    chain = p.build_synthetic_chain("5103", with_ai=False)
    with pytest.raises(p.V11E1Error):
        p.apply_tamper(chain.blocks, "PA-06", expected_order_id="5103")


def test_all_harness_ready():
    harness = p.tamper_harness_readiness()
    assert harness["all_detected"] is True
    assert set(harness["scenarios"]) == set(p.PA_SCENARIOS)


# --------------------------------------------------------------------------- #
# measurement instrumentation
# --------------------------------------------------------------------------- #


def test_measure_validation_records_deterministic():
    chain = p.build_synthetic_chain("5201", risk_level="MEDIUM", risk_score=0.5)
    blocks = chain.blocks
    head = blocks[-1]

    b0 = p.measure_validation("B0", head, blocks, chain.order_id)
    b1 = p.measure_validation("B1", head, blocks, chain.order_id)
    alv = p.measure_validation("P", head, blocks, chain.order_id)

    assert b0.check_operations == 4 and b0.validator_count == 1
    assert b1.check_operations == 24 and b1.validator_count == 3
    assert alv.check_operations == 6 and alv.validator_count == 1
    assert b1.hash_operations == 18  # (c2:1 + c8:5) * 3 validators
    assert alv.hash_operations == 1  # c2 only (MEDIUM band)
    for record in (b0, b1, alv):
        assert record.accepted is True
        assert record.energy_marker == "DIRECT_ENERGY_UNAVAILABLE"
        assert record.wall_time_ms >= 0.0 and record.cpu_time_ms >= 0.0
        assert record.workload_blocks == 5


def test_instrumentation_readiness_structure():
    instruments = p.instrumentation_readiness()
    assert set(instruments["records"]) == {"B0", "B1", "P"}
    for value in instruments["records"].values():
        assert value["energy_marker"] == "DIRECT_ENERGY_UNAVAILABLE"


# --------------------------------------------------------------------------- #
# record loading + artifact provenance (read-only frozen artifacts)
# --------------------------------------------------------------------------- #


def test_load_governed_records_from_frozen_artifact():
    if not D3_ARTIFACT_PATH.exists():
        pytest.skip("frozen D3 artifact not present")
    records = p.load_governed_risk_records()
    assert len(records) == p.EXPECTED_ORDERS
    assert len({record.order_id for record in records}) == p.EXPECTED_ORDERS
    counts = p.route_risk_levels(records)
    assert counts == {"LOW": 0, "MEDIUM": 4582, "HIGH": 6}


def test_preflight_guard_constant_matches_current_head():
    assert p.EXPECTED_HEAD == "511490248cce0bd683c4beed21f6d9cb35fd70a8"


def test_preflight_passes_against_frozen_artifacts():
    if not D3_ARTIFACT_PATH.exists():
        pytest.skip("frozen D3 artifact not present")
    result = p.run_v11e1_preflight(checkpoint_guard=False)
    assert result.passed is True
    assert result.artifact_record_count == 4588
    assert (result.low, result.medium, result.high) == (0, 4582, 6)
    assert result.verified_record_digests == 4588
    assert result.test_access_count == 0
    assert result.artifact_semantic_sha256 == p.ARTIFACT_SEMANTIC_SHA256
    assert result.summary_semantic_sha256 == p.SUMMARY_SEMANTIC_SHA256
    assert result.d3_result_lock_semantic_sha256 == p.D3_RESULT_LOCK_SEMANTIC
    assert result.d1_lock_semantic_sha256 == p.D1_SEMANTIC_LOCK_SHA256
    assert result.v11c_lock_semantic_sha256 == p.V11C_MAPPING_SEMANTIC_SHA256


def test_preflight_fails_closed_on_bad_distribution():
    base = p.E1PreflightResult(
        stage=p.STAGE,
        observed_head=p.EXPECTED_HEAD,
        origin_main=p.EXPECTED_HEAD,
        d1_lock_semantic_sha256=p.D1_SEMANTIC_LOCK_SHA256,
        v11c_lock_semantic_sha256=p.V11C_MAPPING_SEMANTIC_SHA256,
        d3_result_lock_semantic_sha256=p.D3_RESULT_LOCK_SEMANTIC,
        artifact_semantic_sha256=p.ARTIFACT_SEMANTIC_SHA256,
        summary_semantic_sha256=p.SUMMARY_SEMANTIC_SHA256,
        artifact_record_count=p.EXPECTED_ORDERS,
        low=0,
        medium=4582,
        high=6,
        verified_record_digests=p.EXPECTED_ORDERS,
        test_access_count=0,
        artifact_file_sha256=p.ARTIFACT_DISK_SHA256,
        summary_file_sha256=p.SUMMARY_DISK_SHA256,
    )
    assert base.passed is True
    assert base.semantic_sha256
    changed = p.E1PreflightResult(
        stage=p.STAGE,
        observed_head=p.EXPECTED_HEAD,
        origin_main=p.EXPECTED_HEAD,
        d1_lock_semantic_sha256=p.D1_SEMANTIC_LOCK_SHA256,
        v11c_lock_semantic_sha256=p.V11C_MAPPING_SEMANTIC_SHA256,
        d3_result_lock_semantic_sha256=p.D3_RESULT_LOCK_SEMANTIC,
        artifact_semantic_sha256=p.ARTIFACT_SEMANTIC_SHA256,
        summary_semantic_sha256=p.SUMMARY_SEMANTIC_SHA256,
        artifact_record_count=p.EXPECTED_ORDERS,
        low=10,
        medium=4572,
        high=6,
        verified_record_digests=p.EXPECTED_ORDERS,
        test_access_count=0,
        artifact_file_sha256=p.ARTIFACT_DISK_SHA256,
        summary_file_sha256=p.SUMMARY_DISK_SHA256,
    )
    assert changed.passed is False  # drifted LOW band fails closed


def test_governed_order_coverage_from_frozen_mapping():
    if not D3_ARTIFACT_PATH.exists():
        pytest.skip("frozen D3 artifact not present")
    records = p.load_governed_risk_records()
    summary = p.load_v11c_mapping_summary()
    entries = p.mapping_entries_index(summary)
    coverage = p.verify_governed_order_coverage(records, entries)
    assert coverage["status"] == "PREPARED"
    assert coverage["missing_count"] == 0


# --------------------------------------------------------------------------- #
# record mapping round-trip + malformed provenance
# --------------------------------------------------------------------------- #


def test_record_from_artifact_mapping_round_trip():
    record = p.synthetic_governed_record("6001", 0.5)
    mapping = record.to_mapping(include_digest=True)
    rebuilt = p.record_from_artifact_mapping(mapping)
    assert rebuilt == record
    with pytest.raises(p.V11E1Error):
        p.record_from_artifact_mapping({**mapping, "extra": 1})


def test_fail_closed_on_inconsistent_record_provenance():
    record = p.synthetic_governed_record("6002", 0.5)
    altered = p.RiskRecord(
        **{
            **record.to_mapping(include_digest=True),
            "model_configuration_provenance": "DRIFTED",
            "record_digest": "",
        }
    )
    with pytest.raises(Exception):
        altered.verify()