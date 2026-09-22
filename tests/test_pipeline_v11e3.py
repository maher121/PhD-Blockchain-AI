"""V1.1-E3 governed experiment-runner machinery tests.

These tests exercise the E3 deterministic machinery with hand-constructed
synthetic fixtures only. They never execute governed experiments (E01-E10 over
the real 4588-order population), never fit models, never perform AI inference,
never run optimizers, never touch TEST, and never write result artifacts. The
only real artifacts touched are the frozen read-only files re-verified by the
read-only preflight (E2B protocol lock + mirrored YAML/document fingerprints).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import src.pipeline_v11e3 as p
import src.pipeline_v11e1 as e1
from src.blockchain_engine.canonical import sha256_hex

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SMALL_IDS = ["1001", "1002", "1003", "1004"]
SMALL_LEVELS = ["LOW", "MEDIUM", "MEDIUM", "HIGH"]


# --------------------------------------------------------------------------- #
# module identity and governance
# --------------------------------------------------------------------------- #


def test_module_identity():
    assert p.STAGE == "V1.1-E3"
    assert p.KIND == "V11E3_EXPERIMENT_RUNNER_MACHINERY"
    assert p.EXPECTED_HEAD == "b1ff42e9b2e71cce9e829261b2cd0f8a8a79a3a0"
    assert p.E3_MARKER == "V11E3_EXPERIMENT_RUNNER_MACHINERY_READY_REVIEW_REQUIRED"
    assert p.PA_SCENARIOS == e1.PA_SCENARIOS
    assert p.WORKLOADS == (100, 250, 500, 1000, 2500)
    assert p.BLOCKCHAIN_SEEDS == (522, 523, 524)
    assert p.REPETITIONS == 3
    assert p.POLICIES == ("B0", "B1", "P")


def test_governance_constants():
    assert p.ENERGY_MARKER == "DIRECT_ENERGY_UNAVAILABLE"
    assert "Joules" in p.ENERGY_FORBIDDEN_TOKENS
    assert "TDP" in p.ENERGY_FORBIDDEN_TOKENS
    assert p.MEMORY_PROXY_LABEL == "COMPUTATIONAL_MEMORY_PROXY"
    assert p.LOW_ABSENT_MARKER == "LOW_ABSENT_IN_GOVERNED_POPULATION"
    assert p.HIGH_ABSENT_MARKER == "HIGH_NOT_OBSERVED_IN_CONDITION"
    assert p.NA is None
    assert p.E2_PROTOCOL_LOCK_SEMANTIC_SHA256 == (
        "8047fbe356c964e26f7acdcde930a1e2a6734026ea303b56334498ffa295f239"
    )


def test_frozen_policy_descriptors():
    b0 = p.policy_check_spec("B0")
    assert b0["checks"] == ["c1", "c2", "c3", "c4"]
    assert b0["validators"] == 1
    b1 = p.policy_check_spec("B1")
    assert b1["checks"] == ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
    assert b1["validators"] == 3
    assert b1["quorum"] == 3
    pr = p.policy_check_spec("P")
    assert pr["risk_input"] == "GOVERNED_AI_RISK"
    assert set(pr["bands"]) == {"LOW", "MEDIUM", "HIGH"}
    with pytest.raises(p.V11E3Error):
        p.policy_check_spec("Q9")


def test_policy_execution_order_counterbalanced():
    assert p.policy_execution_order(522) == ("B0", "B1", "P")
    assert p.policy_execution_order(523) == ("B1", "P", "B0")
    assert p.policy_execution_order(524) == ("P", "B0", "B1")
    firsts = {p.policy_execution_order(seed)[0] for seed in p.BLOCKCHAIN_SEEDS}
    assert firsts == set(p.POLICIES)


def test_policy_execution_order_rejects_outside_seeds():
    with pytest.raises(ValueError):
        p.policy_execution_order(1)


# --------------------------------------------------------------------------- #
# deterministic sampler
# --------------------------------------------------------------------------- #


def test_rank_order_ids_deterministic_and_seed_dependent():
    r1 = p.rank_order_ids(522, SMALL_IDS)
    r2 = p.rank_order_ids(522, SMALL_IDS)
    assert r1 == r2
    assert set(r1) == set(SMALL_IDS)
    r_523 = p.rank_order_ids(523, SMALL_IDS)
    assert r_523 != r1
    assert set(r_523) == set(SMALL_IDS)


def test_rank_order_ids_duplicate_detection():
    with pytest.raises(p.V11E3Error):
        p.rank_order_ids(522, ["1001", "1001", "1002"])


def test_rank_order_ids_canonical_time():
    r = p.rank_order_ids("522", [1001, 1002, 1003, 1004])
    assert r == p.rank_order_ids(522, ["1001", "1002", "1003", "1004"])


def test_workload_sample_nested_prefixes():
    nested = p.nested_workloads(522, SMALL_IDS, sizes=[2, 3, 4])
    for larger in sorted(nested):
        for smaller in sorted(nested):
            if smaller < larger:
                assert set(nested[smaller]).issubset(set(nested[larger]))
    assert len(nested[2]) == 2 and len(nested[4]) == 4


def test_workload_sample_rejects_zero():
    with pytest.raises(p.V11E3Error):
        p.workload_sample(522, SMALL_IDS, 0)


def test_modern_prereq_sampler_smoke():
    smoke = p.modern_prereq_sampler_smoke(SMALL_IDS, size=4)
    assert smoke["deterministic"] is True
    assert len(smoke["workload"]) == 4
    assert p.nested_workloads(522, SMALL_IDS, [4])[4] == tuple(smoke["workload"])


# --------------------------------------------------------------------------- #
# experiment context
# --------------------------------------------------------------------------- #


def test_context_for_repetition_mapping():
    c522 = p.context_for("E01", 522, 4, SMALL_IDS, "B0")
    c523 = p.context_for("E01", 523, 4, SMALL_IDS, "B0")
    c524 = p.context_for("E01", 524, 4, SMALL_IDS, "B0")
    assert (c522.repetition, c523.repetition, c524.repetition) == (1, 2, 3)
    assert c522.as_mapping()["order_ids"] == [p.canonical_order_id(o) for o in SMALL_IDS]
    assert c522.policy == "B0"


def test_context_for_validation():
    with pytest.raises(p.V11E3Error):
        p.context_for("E01", 7, 4, SMALL_IDS, "B0")
    with pytest.raises(p.V11E3Error):
        p.context_for("E01", 522, 4, SMALL_IDS, "Q9")
    with pytest.raises(p.V11E3Error):
        p.context_for("E01", 522, 5, SMALL_IDS, "B0")
    with pytest.raises(p.V11E3Error):
        p.context_for("E07", 522, 4, SMALL_IDS, "B0", attack_id="PA-90")


def test_context_semantic_sha256_deterministic():
    c1 = p.context_for("E01", 522, 4, SMALL_IDS, "B0")
    c2 = p.context_for("E01", 522, 4, SMALL_IDS, "B0")
    assert c1.semantic_sha256 == c2.semantic_sha256
    assert len(c1.semantic_sha256) == 64


def test_context_provenance():
    c = p.context_for("E01", 522, 4, SMALL_IDS, "B0")
    assert c.protocol_sha256 == p.E2_PROTOCOL_LOCK_SEMANTIC_SHA256
    assert c.upstream_provenance["v11c_mapping_semantic_sha256"] == e1.V11C_MAPPING_SEMANTIC_SHA256
    assert c.environment["threads"] == "1"
    assert c.environment["energy_marker"] == p.ENERGY_MARKER


# --------------------------------------------------------------------------- #
# raw observation schema
# --------------------------------------------------------------------------- #


def test_raw_observation_field_order_and_na():
    ctx = p.context_for("E07", 522, 4, SMALL_IDS, "B0", attack_id="PA-01")
    rec = p.raw_observation(ctx, blocks=5, risk_counts={"LOW": 0, "MEDIUM": 4, "HIGH": 0})
    assert tuple(rec) == p.RAW_OBSERVATION_FIELDS
    assert rec["energy_marker"] == p.ENERGY_MARKER
    assert rec["attack"] == "PA-01"
    assert rec["risk_counts"] == {"LOW": 0, "MEDIUM": 4, "HIGH": 0}
    assert rec["risk_markers"] == [p.LOW_ABSENT_MARKER, p.HIGH_ABSENT_MARKER]
    assert rec["wall_time_ns"] is None
    assert rec["storage_bytes"] is None
    assert len(rec["semantic_sha256"]) == 64


def test_raw_observation_attack_override():
    ctx = p.context_for("E02", 522, 4, SMALL_IDS, "B0")
    rec = p.raw_observation(ctx, blocks=5, attack="PA-02")
    assert rec["attack"] == "PA-02"
    assert "PA-02" not in (ctx.attack_id or "")


# --------------------------------------------------------------------------- #
# descriptive aggregation (no inferential tools)
# --------------------------------------------------------------------------- #


def test_descriptive_aggregation():
    d = p.describe_continuous([1.0, 2.0, 3.0, 4.0])
    assert d["mean"] == 2.5
    assert d["median"] == 2.5
    assert abs(d["std_ddof_1"] - 1.2909944487358056) < 1e-9
    assert (d["min"], d["max"]) == (1.0, 4.0)
    assert p.describe_continuous([]) == {
        "mean": None, "median": None, "std_ddof_1": None, "min": None, "max": None,
    }


def test_pooled_rate_and_paired_differences():
    rate = p.pooled_rate([1, 1, 0, 1], [1, 1, 1, 1])
    assert rate == {"numerator": 3, "denominator": 4, "rate": 0.75}
    assert p.pooled_rate([], [])["rate"] is None
    diffs = p.paired_differences([10.0, 20.0], [5.0, 7.0])
    assert diffs == [5.0, 13.0]


# --------------------------------------------------------------------------- #
# attack-instance factory
# --------------------------------------------------------------------------- #


def _small_fixture(levels: list[str] | None = None) -> dict:
    fixture = p.synthetic_workload_fixture(SMALL_IDS, risk_levels=levels or SMALL_LEVELS)
    instances = p.attack_instances_for_workload(SMALL_IDS, fixture["chains"], 522)
    return {**fixture, "instances": instances}


def test_attack_instances_deterministic_and_copy_only():
    a = _small_fixture()
    b = _small_fixture()
    for sid in p.PA_SCENARIOS:
        assert a["instances"][sid].instance_id == b["instances"][sid].instance_id
        assert a["instances"][sid].tampered_blocks == b["instances"][sid].tampered_blocks
        assert a["instances"][sid].copies_only is True
        assert a["instances"][sid].tamper_kind == e1.PA_TAMPER_KINDS[sid]
    # originals untouched after every build
    for oid in SMALL_IDS:
        original = a["chains"][oid]
        rebuilt_instances = p.attack_instances_for_workload(SMALL_IDS, a["chains"], 522)
        assert a["chains"][oid] == original
        assert rebuilt_instances["PA-01"].tampered_blocks != original


def test_attack_instances_pa08_pa09_donor_contract():
    a = _small_fixture()
    for sid in ("PA-08", "PA-09"):
        inst = a["instances"][sid]
        assert inst.donor_order_id is not None
        assert inst.donor_order_id != inst.target_order_id
        assert inst.donor_read_only is True
    # non-cross-order scenarios have no donor
    assert a["instances"]["PA-01"].donor_order_id is None


def test_attack_instance_semantic_sha():
    a = _small_fixture()["instances"]["PA-06"]
    sig = a.as_mapping()["semantic_sha256"]
    assert sig == sha256_hex(
        {
            "scenario_id": a.scenario_id,
            "seed": int(a.seed),
            "workload_size": int(a.workload_size),
            "target_order_id": a.target_order_id,
            "donor_order_id": a.donor_order_id,
        }
    )


# --------------------------------------------------------------------------- #
# full synthetic dry-run (branch coverage, §13)
# --------------------------------------------------------------------------- #

FIXTURE = None


@pytest.fixture(scope="module")
def dry_run():
    global FIXTURE
    if FIXTURE is None:
        FIXTURE = p.dry_run_all_experiments(SMALL_IDS, seeds=[522], risk_levels=SMALL_LEVELS)
    return FIXTURE["results"]["522"]


def test_dry_run_all_flags(dry_run):
    assert set(dry_run) == set(("E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10"))
    assert all(len(cells) >= 1 for cells in dry_run.values())


def test_dry_run_risk_bands(dry_run):
    for ex in ("E01", "E02", "E03", "E06", "E07", "E09"):
        for cell in dry_run[ex]:
            assert cell["raw_records"]
            for rec in cell["raw_records"]:
                assert rec["risk_counts"] == {"LOW": 1, "MEDIUM": 2, "HIGH": 1}
                assert rec["risk_markers"] == []


def test_dry_run_e01_clean_pass(dry_run):
    full_family = {"c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"}
    allocations = {c["context"]["policy"]: set(c["primary"]["policy_consistent_check_allocation"]) for c in dry_run["E01"]}
    assert allocations["B0"] == {"c1", "c2", "c3", "c4"}
    assert allocations["B1"] == full_family
    assert allocations["P"] == full_family
    for c in dry_run["E01"]:
        assert c["primary"]["clean_verification_pass_count"] == 4
        assert c["primary"]["clean_verification_pass_rate"] == 1.0
        assert c["primary"]["clean_false_rejection_count"] == 0
        assert c["primary"]["valid_chain_construction_count"] == 4


def test_dry_run_e02_detection(dry_run):
    for cell in dry_run["E02"]:
        assert cell["primary"]["denominator"] == 9
        detected = cell["primary"]["policy_specific_detected_count"]
        assert 0 <= detected <= 9
        assert cell["primary"]["policy_specific_detection_rate"] == detected / 9
        assert all(r["detection_outcome"] in (True, False) for r in cell["raw_records"])
    combined = {c["context"]["policy"]: c["primary"]["policy_specific_detected_count"] for c in dry_run["E02"]}
    assert combined["B0"] < combined["B1"]
    assert combined["P"] < combined["B1"]


def test_dry_run_e03_localization(dry_run):
    for cell in dry_run["E03"]:
        assert cell["primary"]["denominator"] == 9
        for rec in cell["raw_records"]:
            assert "localized" in rec["localization_outcome"]
            assert isinstance(rec["localization_outcome"]["localized"], bool)
        rate = cell["primary"]["unconditional_correct_localization_rate"]
        assert rate is not None and 0.0 <= rate <= 1.0


def test_dry_run_e04_isolated(dry_run):
    for cell in dry_run["E04"]:
        for rec in cell["raw_records"]:
            loc = rec["localization_outcome"]
            assert loc["donor_read_only"] is True
            assert loc["cross_order_propagation_count"] == 0
            assert loc["unaffected_order_count"] == 3
        assert cell["primary"]["unaffected_order_preservation_rate"] == 1.0


def test_dry_run_e05_p_principal_and_bands(dry_run):
    e05 = dry_run["E05"]
    assert len(e05) == 1
    assert e05[0]["context"]["policy"] == "P"
    allocation = e05[0]["primary"]["applied_validation_check_allocation_by_governed_risk"]
    assert set(allocation) == {"LOW", "MEDIUM", "HIGH"}
    assert e05[0]["primary"]["validation_level_distribution"] == {"LOW": 1, "MEDIUM": 2, "HIGH": 1}


def test_dry_run_e06_timing(dry_run):
    for cell in dry_run["E06"]:
        assert cell["timing_principal"] is True
        assert cell["energy_marker"] == p.ENERGY_MARKER
        for rec in cell["raw_records"]:
            assert int(rec["wall_time_ns"]) >= 0
            assert int(rec["cpu_time_ns"]) >= 0
        prim = cell["primary"]["validation_only_wall_clock_runtime_ns"]
        assert prim["mean"] >= prim["min"]
        assert prim["mean"] <= prim["max"]


def test_dry_run_e07_memory_proxy(dry_run):
    for cell in dry_run["E07"]:
        assert cell["label"] == p.MEMORY_PROXY_LABEL
        assert cell["unit"] == "MiB"
        assert cell["measurement"] == "tracemalloc_fresh_worker"
        for rec in cell["raw_records"]:
            assert float(rec["memory_proxy_mib"]) > 0.0


def test_dry_run_e08_storage_policy_independent(dry_run):
    e08 = dry_run["E08"]
    assert len(e08) == 1
    cell = e08[0]
    assert cell["policy_independent"] is True
    assert cell["primary"]["canonical_serialized_blockchain_bytes_per_workload"] > 0
    assert cell["primary"]["bytes_per_order"] > 0


def test_dry_run_e09_throughput(dry_run):
    for cell in dry_run["E09"]:
        assert cell["primary"]["total_validated_orders"] == 4
        assert cell["primary"]["validated_orders_per_second"] is not None


def test_dry_run_e10_derived_only(dry_run):
    for cell in dry_run["E10"]:
        assert cell["derived_only"] is True
        assert cell["no_fourth_campaign"] is True
        assert cell["direct_energy"] == p.ENERGY_MARKER
        assert cell["semantic_sha256"] and len(cell["semantic_sha256"]) == 64


def test_dry_run_energy_marker_everywhere(dry_run):
    for ex, cells in dry_run.items():
        for cell in cells:
            assert cell["energy_marker"] == p.ENERGY_MARKER
            if isinstance(cell.get("primary"), dict):
                keys = " ".join(cell["primary"]).lower()
                assert "joule" not in keys
                assert "watt" not in keys


# --------------------------------------------------------------------------- #
# preflight (read-only, fail-closed)
# --------------------------------------------------------------------------- #


def test_preflight_read_only_and_fail_closed():
    evidence = p.run_v11e3_preflight(checkpoint_guard=True)
    assert evidence["head"] == p.EXPECTED_HEAD
    assert evidence["origin_main"] == p.EXPECTED_HEAD
    assert evidence["e2_protocol_lock_semantic_sha256"] == p.E2_PROTOCOL_LOCK_SEMANTIC_SHA256
    assert evidence["e2_config_fingerprint"] == p.E2_CONFIG_FINGERPRINT
    assert evidence["e2_document_fingerprint"] == p.E2_DOC_FINGERPRINT
    assert evidence["experiment_status"] == "NOT_EXECUTED"
    assert evidence["test_access_count"] == 0
    assert evidence["ai_fit_count"] == 0
    assert evidence["ai_inference_count"] == 0
    assert evidence["experiment_execution_count"] == 0
    assert evidence["upstream_provenance_verified"] is True
    assert evidence["marker"] == p.E3_MARKER