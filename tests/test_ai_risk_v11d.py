"""V1.1-D2 governed AI risk implementation tests (synthetic fixtures only).

These tests exercise the deterministic components of the D2 implementation
with hand-constructed fixtures. They never fit models, never read real
partition data, never run optimizers, and never touch a real blockchain
chain. The protocol loading tests additionally re-verify the frozen D1
config/lock artifact linkage on disk (read-only).
"""

from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path

import pytest

from src.ai_risk import (
    AIRiskReference,
    ReconstructionNotAuthorizedError,
    ReconstructionSpec,
    RiskRecord,
    RowScore,
    aggregate_rows,
    build_ai_risk_reference,
    build_risk_record,
    combine_member_scores,
    load_verified_protocol,
)
import src.pipeline_v11d as pipeline
from src.ai_risk.blockchain_linkage import (
    EVENT_TYPE_AI_RISK_ASSESSED,
    FORBIDDEN_REFERENCE_KEYS,
    BlockchainLinkageError,
    verify_reference_matches_record,
)
from src.ai_risk.model_reconstruction import (
    PartitionGuardError,
    assert_fit_split,
    assert_generation_split,
    assert_test_blocked,
    validate_reconstruction_spec,
    verify_fingerprint_names,
)
from src.ai_risk.order_aggregation import (
    OrderAggregationError,
    OrderAggregate,
    source_rows_digest,
)
from src.ai_risk.protocol import (
    D1_SEMANTIC_LOCK_SHA256,
    AIRiskProtocolError,
    verify_d1_lock_semantic,
)
from src.blockchain_engine.errors import BlockchainEngineError
from src.ai_risk.risk_levels import RiskLevelError, risk_level_for_score
from src.ai_risk.risk_record import (
    SCHEMA_VERSION,
    RiskRecordError,
)
from src.ai_risk.row_scoring import FROZEN_MEMBER_SEEDS, RowScoreError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "ai_risk_v11d.yaml"
LOCK_PATH = (
    PROJECT_ROOT / "results" / "blockchain" / "v11d" / "v11d_ai_risk_protocol_lock.json"
)

EXPECTED_FEATURES = [
    "order_item_quantity",
    "product_price",
    "order_item_total",
    "is_weekend",
    "Type_DEBIT",
    "Type_TRANSFER",
    "Type_CASH",
    "Market_LATAM",
    "Market_Pacific Asia",
    "Shipping Mode_First Class",
    "Department Name_Golf",
    "Department Name_Fitness",
    "Department Name_Health and Beauty",
]


@pytest.fixture(scope="module")
def protocol():
    return load_verified_protocol()


@pytest.fixture()
def seeded_scores():
    return [0.10, 0.20, 0.30, 0.40, 0.50]


# --------------------------------------------------------------------------- #
# protocol loading / lock verification
# --------------------------------------------------------------------------- #


def test_protocol_loader_verifies_linked_artifacts(protocol):
    assert protocol.lock_semantic_sha256 == D1_SEMANTIC_LOCK_SHA256
    assert protocol.config["stage"] == "V1.1-D1"
    assert protocol.config["kind"] == "SCIENTIFIC_DECISION_PROTOCOL_LOCK"


def test_protocol_feature_identity_is_frozen(protocol):
    assert list(protocol.features()) == EXPECTED_FEATURES
    assert len(protocol.features()) == 13
    identity = protocol.feature_identity()
    assert identity["configuration_id"] == "HYBRID-K13"
    assert identity["selected_feature_count"] == 13
    assert identity["feature_space_dimension"] == 43


def test_protocol_seeds_are_frozen(protocol):
    assert protocol.seeds() == FROZEN_MEMBER_SEEDS
    assert protocol.classifier_parameters() == {
        "class_weight": "balanced",
        "max_depth": 5,
        "min_samples_leaf": 20,
    }


def test_protocol_lock_semantic_constant_matches_disk():
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    verify_d1_lock_semantic(lock)
    assert lock["semantic_result_lock_sha256"] == D1_SEMANTIC_LOCK_SHA256


def test_protocol_loader_fails_closed_on_tampered_lock(tmp_path):
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    lock["semantic_result_lock_sha256"] = "0" * 64
    tampered = tmp_path / "lock.json"
    tampered.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    with pytest.raises(AIRiskProtocolError):
        load_verified_protocol(lock_path=tampered)


def test_protocol_loader_rejects_wrong_config_version(tmp_path):
    config = CONFIG_PATH.read_text(encoding="utf-8").replace("stage: \"V1.1-D1\"", 'stage: "V9.9-X"')
    tampered = tmp_path / "ai_risk_v11d.yaml"
    tampered.write_text(config, encoding="utf-8")
    with pytest.raises(AIRiskProtocolError):
        load_verified_protocol(config_path=tampered)


def test_protocol_loader_fails_closed_on_missing_artifact(tmp_path):
    with pytest.raises(AIRiskProtocolError):
        load_verified_protocol(lock_path=tmp_path / "does-not-exist.json")


# --------------------------------------------------------------------------- #
# feature identity / D1 lock linkage
# --------------------------------------------------------------------------- #


def test_frozen_feature_list_matches_lock_semantic_payload():
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    payload_features = lock["semantic_payload"]["feature_identity"]["ordered_features"]
    assert payload_features == EXPECTED_FEATURES
    assert lock["semantic_payload"]["feature_identity"]["configuration_id"] == "HYBRID-K13"


def test_frozen_seed_list_matches_lock_semantic_payload():
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    seeds = lock["semantic_payload"]["decisions"]["model_strategy"]["seeds"]
    assert seeds == list(FROZEN_MEMBER_SEEDS)


# --------------------------------------------------------------------------- #
# model reconstruction contract
# --------------------------------------------------------------------------- #


def test_reconstruction_spec_from_protocol(protocol):
    spec = ReconstructionSpec.from_protocol(protocol)
    validate_reconstruction_spec(spec)
    assert spec.fit_split == "TRAIN"
    assert spec.fit_rows == 28000
    assert spec.source_contract == "src.pipeline_v08b.load_frozen_development_workloads"
    assert spec.attack_type == "mixed"
    assert spec.attack_rate == 0.05
    assert spec.attack_severity == "MEDIUM"
    assert spec.seeds == (42, 43, 44, 45, 46)


def test_reconstruction_requires_authorization_by_default(protocol):
    from src.ai_risk.model_reconstruction import require_reconstruction_authorization

    spec = ReconstructionSpec.from_protocol(protocol)
    validate_reconstruction_spec(spec)
    with pytest.raises(ReconstructionNotAuthorizedError):
        require_reconstruction_authorization(authorized=False)


def test_fit_split_guard_is_train_only():
    assert_fit_split("TRAIN")
    with pytest.raises(PartitionGuardError):
        assert_fit_split("VALIDATION")
    with pytest.raises(PartitionGuardError):
        assert_fit_split("TEST")


def test_generation_split_guard_is_validation_only():
    assert_generation_split("VALIDATION")
    with pytest.raises(PartitionGuardError):
        assert_generation_split("TRAIN")
    with pytest.raises(PartitionGuardError):
        assert_generation_split("TEST")


def test_test_partition_is_blocked():
    assert_test_blocked("TRAIN")
    assert_test_blocked("VALIDATION")
    with pytest.raises(PartitionGuardError):
        assert_test_blocked("TEST")


def test_fingerprint_names_required_frozen_set():
    verify_fingerprint_names(
        [
            "training_row_ids_sha256",
            "training_clean_features_sha256",
            "training_attacked_features_sha256",
            "training_labels_sha256",
        ]
    )
    with pytest.raises(PartitionGuardError):
        verify_fingerprint_names(["training_labels_sha256"])


def test_full_reconstruction_stub_fails_closed():
    from src.ai_risk.model_reconstruction import reconstruct_member_models

    with pytest.raises(ReconstructionNotAuthorizedError):
        reconstruct_member_models(None)
    with pytest.raises(ReconstructionNotAuthorizedError):
        reconstruct_member_models(None, authorization_capability=object())


# --------------------------------------------------------------------------- #
# row scoring / ensemble arithmetic
# --------------------------------------------------------------------------- #


def test_ensemble_arithmetic_mean(seeded_scores):
    assert combine_member_scores(seeded_scores) == 0.3
    assert combine_member_scores([1.0, 1.0, 1.0, 1.0, 1.0]) == 1.0
    assert combine_member_scores([0.0, 0.0, 0.0, 0.0, 0.0]) == 0.0


def test_ensemble_rejects_wrong_member_count():
    with pytest.raises(RowScoreError):
        combine_member_scores([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(RowScoreError):
        combine_member_scores([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def test_ensemble_rejects_nan_and_inf():
    with pytest.raises(RowScoreError):
        combine_member_scores([0.1, 0.2, math.nan, 0.4, 0.5])
    with pytest.raises(RowScoreError):
        combine_member_scores([0.1, 0.2, 0.3, math.inf, 0.5])


def test_ensemble_rejects_out_of_range():
    with pytest.raises(RowScoreError):
        combine_member_scores([-0.1, 0.2, 0.3, 0.4, 0.5])
    with pytest.raises(RowScoreError):
        combine_member_scores([0.1, 0.2, 1.1, 0.4, 0.5])


def test_ensemble_score_range_is_unit_interval():
    scores = combine_member_scores([0.0, 0.25, 0.5, 0.75, 1.0])
    assert 0.0 <= scores <= 1.0


def test_ensemble_rejects_reordered_input():
    with pytest.raises(RowScoreError):
        combine_member_scores([0.5, 0.4, 0.3, 0.2, 0.1], in_frozen_seed_order=False)


# --------------------------------------------------------------------------- #
# row-to-order MAX aggregation
# --------------------------------------------------------------------------- #


def test_max_aggregation_picks_highest_row_score():
    rows = [
        RowScore(row_id="1", order_id="1001", score=0.2),
        RowScore(row_id="2", order_id="1001", score=0.8),
        RowScore(row_id="3", order_id="1001", score=0.4),
    ]
    result = aggregate_rows(rows)
    assert result["1001"].risk_score == 0.8
    assert result["1001"].source_row_count == 3


def test_max_aggregation_is_row_order_invariant():
    rows_forward = [
        RowScore("1", "2001", 0.3),
        RowScore("2", "2001", 0.9),
        RowScore("3", "2001", 0.6),
    ]
    rows_shuffled = [
        RowScore("3", "2001", 0.6),
        RowScore("1", "2001", 0.3),
        RowScore("2", "2001", 0.9),
    ]
    a = aggregate_rows(rows_forward)
    b = aggregate_rows(rows_shuffled)
    assert a["2001"].risk_score == b["2001"].risk_score
    assert a["2001"].source_rows_digest == b["2001"].source_rows_digest


def test_aggregation_normalizes_order_and_row_ids():
    rows = [
        RowScore(row_id="07", order_id="3001", score=0.5),
        RowScore(row_id=8, order_id=3001, score=0.6),
    ]
    result = aggregate_rows(rows)
    assert "3001" in result
    assert result["3001"].source_row_count == 2
    assert result["3001"].source_rows_digest == source_rows_digest(["7", "8"])


def test_aggregation_rejects_non_finite_score():
    with pytest.raises(OrderAggregationError):
        aggregate_rows([RowScore("1", "3001", math.nan)])
    with pytest.raises(OrderAggregationError):
        aggregate_rows([RowScore("1", "3001", math.inf)])


def test_aggregation_rejects_out_of_range_score():
    with pytest.raises(OrderAggregationError):
        aggregate_rows([RowScore("1", "3001", -0.1)])
    with pytest.raises(OrderAggregationError):
        aggregate_rows([RowScore("1", "3001", 1.5)])


def test_aggregation_rejects_malformed_inputs():
    assert aggregate_rows([]) == {}
    with pytest.raises(OrderAggregationError):
        aggregate_rows([("1", "3001", 0.5)])
    with pytest.raises(BlockchainEngineError):
        aggregate_rows([RowScore("1", "abc", 0.5)])


def test_aggregation_output_is_ordered_aggregate_type():
    rows = [RowScore("1", "4001", 0.3)]
    result = aggregate_rows(rows)
    assert isinstance(result["4001"], OrderAggregate)


# --------------------------------------------------------------------------- #
# risk level bands
# --------------------------------------------------------------------------- #


def test_low_medium_high_boundaries():
    assert risk_level_for_score(0.0) == "LOW"
    assert risk_level_for_score(0.3332) == "LOW"
    assert risk_level_for_score(0.3333) == "MEDIUM"
    assert risk_level_for_score(0.5) == "MEDIUM"
    assert risk_level_for_score(0.6666) == "MEDIUM"
    assert risk_level_for_score(0.6667) == "HIGH"
    assert risk_level_for_score(1.0) == "HIGH"


def test_risk_level_rejects_invalid_scores():
    with pytest.raises(RiskLevelError):
        risk_level_for_score(-0.1)
    with pytest.raises(RiskLevelError):
        risk_level_for_score(1.1)
    with pytest.raises(RiskLevelError):
        risk_level_for_score(math.nan)


# --------------------------------------------------------------------------- #
# risk record schema + deterministic digest
# --------------------------------------------------------------------------- #


def _sample_aggregate() -> OrderAggregate:
    return aggregate_rows(
        [
            RowScore("10", "5001", 0.3),
            RowScore("11", "5001", 0.8),
        ]
    )["5001"]


def test_risk_record_schema_and_field_order(protocol):
    record = build_risk_record(_sample_aggregate(), protocol)
    fields = list(record.to_mapping())
    assert fields[0] == "schema_version"
    assert fields[-1] == "record_digest"
    assert set(fields) == {
        "schema_version", "order_id", "risk_score", "risk_level",
        "source_row_count", "source_rows_digest", "classifier",
        "model_strategy", "model_seeds", "aggregation_rule",
        "source_partition", "score_semantics_version",
        "model_configuration_provenance", "hybrid_k13_provenance",
        "generation_stage_provenance", "record_digest",
    }


def test_risk_record_digest_is_deterministic(protocol):
    first = build_risk_record(_sample_aggregate(), protocol)
    second = build_risk_record(_sample_aggregate(), protocol)
    assert first.record_digest == second.record_digest
    assert first.record_digest == first.recompute_digest()


def test_risk_record_digest_excludes_itself(protocol):
    record = build_risk_record(_sample_aggregate(), protocol)
    payload = record.to_mapping(include_digest=False)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == record.record_digest


def test_risk_record_verifies_consistent(protocol):
    record = build_risk_record(_sample_aggregate(), protocol)
    record.verify()


def test_risk_record_rejects_inconsistent_digest(protocol):
    record = build_risk_record(_sample_aggregate(), protocol)
    altered = RiskRecord(**{**record.to_mapping(), "risk_score": 0.7})
    with pytest.raises(RiskRecordError):
        altered.verify()


def test_risk_record_carries_provenance(protocol):
    record = build_risk_record(_sample_aggregate(), protocol)
    assert record.source_partition == "VALIDATION"
    assert record.aggregation_rule == "MAX"
    assert record.classifier == "decision_tree"
    assert record.model_strategy == "FIXED_FIVE_SEED_MEAN_ENSEMBLE"
    assert record.model_seeds == (42, 43, 44, 45, 46)
    assert record.hybrid_k13_provenance.startswith("HYBRID-K13;")
    assert "TRAIN attack workloads" in record.model_configuration_provenance
    assert "VALIDATION" in record.generation_stage_provenance


def test_risk_record_rejects_non_validation_partition(protocol):
    with pytest.raises(RiskRecordError):
        build_risk_record(_sample_aggregate(), protocol, source_partition="TEST")


def test_risk_record_rejects_non_frozen_seeds(protocol):
    with pytest.raises(RiskRecordError):
        build_risk_record(
            _sample_aggregate(), protocol, model_seeds=(1, 2, 3, 4, 5)
        )


def test_risk_record_rejects_empty_aggregate(protocol):
    empty_order = OrderAggregate(
        order_id="5002", risk_score=0.0, source_row_count=0, source_rows_digest=""
    )
    with pytest.raises(RiskRecordError):
        build_risk_record(empty_order, protocol)


def test_source_rows_digest_is_sorted_and_stable():
    assert source_rows_digest(["10", "9", "2"]) == source_rows_digest(["2", "10", "9"])
    assert source_rows_digest(["7"]) == source_rows_digest([7])
    assert len(source_rows_digest(["1", "2", "3"])) == 64


def test_risk_record_schema_version_frozen():
    assert SCHEMA_VERSION == "v1.1-d-governed-order-risk-1"


# --------------------------------------------------------------------------- #
# blockchain linkage payload policy
# --------------------------------------------------------------------------- #


def _sample_record(protocol) -> RiskRecord:
    return build_risk_record(_sample_aggregate(), protocol)


def test_ai_risk_reference_payload_is_minimal_and_ordered(protocol):
    record = _sample_record(protocol)
    reference = build_ai_risk_reference(
        record,
        protocol,
        artifact_ref="results/v11d/risk_lock.json",
        artifact_lock_sha256="1" * 64,
    )
    payload = reference.to_mapping()
    assert list(payload) == [
        "mode", "stage", "order_id", "risk_score", "risk_level",
        "configuration_id", "classifier", "aggregation_rule",
        "record_digest", "artifact_ref", "artifact_lock_sha256",
    ]
    assert payload["mode"] == "GOVERNED_AI_RISK"
    assert payload["stage"] == "V1.1-D"
    assert payload["configuration_id"] == "HYBRID-K13"


def test_ai_risk_reference_contains_no_raw_data(protocol):
    record = _sample_record(protocol)
    reference = build_ai_risk_reference(
        record, protocol, artifact_ref="ref", artifact_lock_sha256="1" * 64
    )
    payload = reference.to_mapping()
    for forbidden in FORBIDDEN_REFERENCE_KEYS:
        assert forbidden not in payload
    assert "model_seeds" not in payload
    assert "source_row_count" not in payload
    assert "row_ids" not in payload


def test_ai_risk_reference_digest_binds_to_record(protocol):
    record = _sample_record(protocol)
    reference = build_ai_risk_reference(
        record, protocol, artifact_ref="ref", artifact_lock_sha256="1" * 64
    )
    verify_reference_matches_record(reference, record)
    mismatched = AIRiskReference(
        **{**reference.to_mapping(), "record_digest": "0" * 64}
    )
    with pytest.raises(BlockchainLinkageError):
        verify_reference_matches_record(mismatched, record)


def test_ai_risk_reference_requires_record_digest(protocol):
    record = RiskRecord(order_id="5001")
    with pytest.raises(BlockchainLinkageError):
        build_ai_risk_reference(
            record, protocol, artifact_ref="ref", artifact_lock_sha256="1" * 64
        )


def test_event_type_constant_is_ai_risk_assessed():
    assert EVENT_TYPE_AI_RISK_ASSESSED == "AI_RISK_ASSESSED"


# --------------------------------------------------------------------------- #
# pipeline execution guard
# --------------------------------------------------------------------------- #


def test_pipeline_generation_is_not_authorized_in_d2():
    assert pipeline.GOVERNED_RISK_GENERATION_AUTHORIZED is False


def test_pipeline_entry_points_blocked_without_authorization(protocol):
    with pytest.raises(pipeline.GovernedRiskExecutionNotAuthorizedError):
        pipeline.run_governed_risk_generation()
    with pytest.raises(pipeline.GovernedRiskExecutionNotAuthorizedError):
        pipeline.combine_row_score([0.1, 0.2, 0.3, 0.4, 0.5])
    with pytest.raises(pipeline.GovernedRiskExecutionNotAuthorizedError):
        pipeline.aggregate_order_rows([RowScore("1", "6001", 0.5)])
    with pytest.raises(pipeline.GovernedRiskExecutionNotAuthorizedError):
        pipeline.score_to_level(0.5)
    with pytest.raises(pipeline.GovernedRiskExecutionNotAuthorizedError):
        pipeline.make_risk_record(_sample_aggregate(), protocol)
    with pytest.raises(pipeline.GovernedRiskExecutionNotAuthorizedError):
        pipeline.make_ai_risk_reference(
            RiskRecord(order_id="6001", record_digest="1" * 64),
            protocol,
            artifact_ref="ref",
            artifact_lock_sha256="1" * 64,
        )


def test_pipeline_protocol_loading_and_contracts_are_read_only(protocol):
    spec = pipeline.reconstruction_contract(protocol)
    assert spec.fit_split == "TRAIN"
    assert pipeline.load_protocol().config_sha256 == protocol.config_sha256


# --------------------------------------------------------------------------- #
# fail-closed malformed inputs (cross-component)
# --------------------------------------------------------------------------- #


def test_fail_closed_malformed_scores_never_reach_aggregation():
    with pytest.raises(RowScoreError):
        combine_member_scores([0.1, 0.2, 0.3, 0.4, float("nan")])


def test_fail_closed_out_of_range_score_never_creates_record(protocol):
    rows = [RowScore("1", "7001", 2.0)]
    with pytest.raises(OrderAggregationError):
        aggregate_rows(rows)


def test_fail_closed_non_canonical_order_id():
    with pytest.raises(BlockchainEngineError):
        aggregate_rows([RowScore("1", "Order 7001", 0.5)])


def test_frozen_aggregation_rule_is_max(protocol):
    record = _sample_record(protocol)
    assert record.aggregation_rule == "MAX"
    assert protocol.config["row_to_order"]["rule"] == "MAX"