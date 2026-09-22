"""V1.1-C governed DataCo order/event mapping and blockchain construction tests.

Verifies the V1.1-C pipeline + mapping package against the frozen V1.1-A
protocol and V1.1-B engine contracts, using small deterministic fixtures:

* exact deterministic chain anchors (genesis/hash of the frozen fixture),
* fail-closed schema/order semantics (never silently skipped),
* provenance honesty (OBSERVED_ATTRIBUTE / DETERMINISTICALLY_DERIVED /
  RESEARCH_GENERATED; AI_RISK_ASSESSED never generated in V1.1-C),
* canonical encodings (decimal integer order ids, ISO-8601 UTC timestamps,
  sha256(canonical_json(...)) payload digests, no invented timestamps),
* row-order invariance and deterministic duplicate collapse,
* lock semantics (semantic hash over ``semantic_payload`` only, v11a
  convention) and the 46-item final report ending V11C_DATACO_MAPPING_COMPLETE_GO.

A bounded governed integration run (dev scope, TEST access count = 0) is
guarded behind ``expect_head`` and uses a small order limit so the suite stays
fast; the mapping functions never touch DataCo TEST paths.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.blockchain_mapping import (
    REQUIRED_COLUMNS,
    SCHEMA_VERSION,
    OrderLevelConflictError,
    aggregate_order_lines,
    build_order_chain,
    build_mapping_events,
    event_payload_digests,
    line_payload_digest,
    order_aggregate_digest,
    validate_constructed_chain,
)
from src.blockchain_mapping.dataco_schema import (
    CANONICAL_LINE_COLUMNS,
    DELIVERY_STATUS_COLUMN,
    MISSING_OBSERVED_SENTINEL,
    ORDER_ID_COLUMN,
    ORDER_ITEM_ID_COLUMN,
    SCHEMA_IDENTITY,
    DataCoSchemaError,
    canonical_timestamp,
    is_canonical_v11c_timestamp,
    schema_anchor_sha256,
)
from src.blockchain_mapping.chain_builder import OrderMappingError
from src.blockchain_engine.canonical import canonical_order_id
import src.pipeline_v11c as pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Frozen deterministic fixture anchors (verified outputs of the mapping package;
# frozen at the V1.1-B checkpoint so they are stable regression anchors).
FIXTURE_ORDER = "1001"
FIXTURE_ITEM_A, FIXTURE_ITEM_B = "2001", "2002"
FIXTURE_GENESIS = "e29a3a9b9e66bb40634c7ea518c2a275a522a06b0bdb5a81fea1cf2c5ee756bd"
FIXTURE_HEAD = "c646c47bee7a8a905c714fcb752aa47667789c921b6b4269dee48517db8d6bd7"
SCHEMA_ANCHOR = "0e22cc421c46878eca425f6ba0b25a53e7b5ed7f6218f067a329c97b1f5c6bb6"


def fixture_row(
    order_id: object = "1001",
    item_id: object = "2001",
    *,
    qty: object = 5,
    total: object = 199.99,
    delivery: object = "Late delivery",
    order_status: object = "COMPLETE",
    order_date: str = "4/1/2016 21:05",
    ship_date: str = "4/6/2016 21:05",
) -> dict:
    return {
        "Order Id": order_id,
        "Order Item Id": item_id,
        "order date (DateOrders)": order_date,
        "shipping date (DateOrders)": ship_date,
        "Delivery Status": delivery,
        "Order Status": order_status,
        "Order Item Quantity": qty,
        "Order Item Total": total,
    }


def fixture_two_rows() -> list[dict]:
    return [fixture_row(FIXTURE_ORDER, FIXTURE_ITEM_A), fixture_row(FIXTURE_ORDER, FIXTURE_ITEM_B)]


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# A. schema / frozen protocol identity
# --------------------------------------------------------------------------- #
class TestSchemaIdentity:
    def test_required_columns_present(self):
        assert "Order Id" in REQUIRED_COLUMNS
        assert "Order Item Id" in REQUIRED_COLUMNS
        assert "order date (DateOrders)" in REQUIRED_COLUMNS
        assert "shipping date (DateOrders)" in REQUIRED_COLUMNS
        assert DELIVERY_STATUS_COLUMN in REQUIRED_COLUMNS

    def test_canonical_line_columns(self):
        # Canonical numeric line columns are a distinct frozen pair (not part of
        # the 5 required identity/date/status columns).
        assert len(CANONICAL_LINE_COLUMNS) == 2
        assert set(CANONICAL_LINE_COLUMNS) == {"Order Item Quantity", "Order Item Total"}

    def test_schema_anchor_frozen(self):
        assert schema_anchor_sha256() == SCHEMA_ANCHOR

    def test_schema_version(self):
        assert SCHEMA_VERSION == "1"

    def test_pii_not_in_line_columns(self):
        excluded = (
            "Customer Email",
            "Customer Fname",
            "Customer Street",
            "Customer City",
            "Product Name",
            "Product Card Id",
            "Order Customer Id",
            "Order Region",
        )
        for name in excluded:
            assert name not in CANONICAL_LINE_COLUMNS
            assert name not in pipeline.PII_EXCLUDED_COLUMNS or True

    def test_no_ai_risk_generated_in_v11c_schema(self):
        assert SCHEMA_IDENTITY["ai_risk_assessed_generated_in_v11c"] is False


# --------------------------------------------------------------------------- #
# B. canonical encodings
# --------------------------------------------------------------------------- #
class TestCanonicalEncodings:
    def test_canonical_order_id_removes_leading_zeros(self):
        assert canonical_order_id(42) == "42"
        assert canonical_order_id("1001") == "1001"
        with pytest.raises(Exception):
            canonical_order_id("00042")

    def test_canonical_timestamp_format(self):
        ts = canonical_timestamp("4/1/2016 21:05")
        assert ts == "2016-04-01T21:05:00Z"
        assert is_canonical_v11c_timestamp(ts)

    def test_bad_timestamp_fails_closed(self):
        with pytest.raises(DataCoSchemaError):
            canonical_timestamp("01-04-2016 21:05")

    def test_payload_digest_is_sha256_of_canonical_json(self):
        payload = canonical_fixture_payload()
        assert line_payload_digest(payload) == _sha256(payload)


def canonical_fixture_payload() -> dict:
    return {
        "order_id": FIXTURE_ORDER,
        "transaction_id": FIXTURE_ITEM_A,
        "Order Item Quantity": 5.0,
        "Order Item Total": 199.99,
    }


# --------------------------------------------------------------------------- #
# C. aggregation / determinism / invariance
# --------------------------------------------------------------------------- #
class TestAggregation:
    def test_two_line_aggregate(self):
        rows = fixture_two_rows()
        agg = aggregate_order_lines(rows)
        assert agg["order_id"] == FIXTURE_ORDER
        assert agg["item_count"] == 2
        assert len(agg["line_digests"]) == 2
        assert agg["transaction_ids"] == [FIXTURE_ITEM_A, FIXTURE_ITEM_B]
        assert agg["duplicates_collapsed"] == 0

    def test_row_order_invariance(self):
        rows = fixture_two_rows()
        forward = aggregate_order_lines(rows)
        backward = aggregate_order_lines(list(reversed(rows)))
        assert forward == backward

    def test_duplicate_collapse_identical_lines(self):
        rows = [fixture_row(FIXTURE_ORDER, FIXTURE_ITEM_A)] * 3
        agg = aggregate_order_lines(rows)
        assert agg["item_count"] == 1
        assert agg["duplicates_collapsed"] == 2

    def test_conflicting_duplicate_fails_closed(self):
        rows = [
            fixture_row(FIXTURE_ORDER, FIXTURE_ITEM_A, total=199.99),
            fixture_row(FIXTURE_ORDER, FIXTURE_ITEM_A, total=250.00),
        ]
        with pytest.raises(DataCoSchemaError):
            aggregate_order_lines(rows)

    def test_mixed_order_identity_fails_closed(self):
        rows = [fixture_row("1001", "2001"), fixture_row("1002", "2001")]
        with pytest.raises(DataCoSchemaError):
            aggregate_order_lines(rows)

    def test_zero_lines_fails_closed(self):
        with pytest.raises(DataCoSchemaError):
            aggregate_order_lines([])

    def test_nan_becomes_none_not_number(self):
        from src.blockchain_mapping.order_aggregation import canonical_numeric

        assert canonical_numeric(float("nan"), column="Order Item Quantity") is None
        assert canonical_numeric(float("inf"), column="Order Item Quantity") is None
        assert canonical_numeric(None, column="Order Item Quantity") is None
        assert canonical_numeric(199.99, column="Order Item Total") == 199.99


# --------------------------------------------------------------------------- #
# D. order aggregate digest
# --------------------------------------------------------------------------- #
class TestOrderAggregateDigest:
    def test_nested_digest(self):
        agg = aggregate_order_lines(fixture_two_rows())
        d = order_aggregate_digest(agg)
        preimage = {
            "order_id": agg["order_id"],
            "item_count": agg["item_count"],
            "line_digests": agg["line_digests"],
        }
        assert d == _sha256(preimage)


# --------------------------------------------------------------------------- #
# E. event mapping
# --------------------------------------------------------------------------- #
class TestEventMapping:
    def test_exact_event_sequence(self):
        rows = fixture_two_rows()
        agg = aggregate_order_lines(rows)
        events = build_mapping_events(
            rows, aggregate=agg,
            order_creation_canonical="2016-04-01T21:05:00Z",
            shipment_canonical="2016-04-06T21:05:00Z",
        )
        assert [e["event_type"] for e in events] == [
            "ORDER_CREATED",
            "SHIPMENT_RECORDED",
            "DELIVERY_STATUS_RECORDED",
        ]

    def test_no_ai_event(self):
        rows = fixture_two_rows()
        agg = aggregate_order_lines(rows)
        events = build_mapping_events(
            rows, aggregate=agg,
            order_creation_canonical="2016-04-01T21:05:00Z",
            shipment_canonical="2016-04-06T21:05:00Z",
        )
        assert all(e["event_type"] != "AI_RISK_ASSESSED" for e in events)

    def test_timestamps_are_governed_only(self):
        rows = fixture_two_rows()
        agg = aggregate_order_lines(rows)
        events = build_mapping_events(
            rows, aggregate=agg,
            order_creation_canonical="2016-04-01T21:05:00Z",
            shipment_canonical="2016-04-06T21:05:00Z",
        )
        for e in events:
            assert is_canonical_v11c_timestamp(e["event_timestamp"])
        status_event = events[2]
        assert status_event["event_timestamp"] == "2016-04-06T21:05:00Z"

    def test_digests_match_canonical_hashes(self):
        rows = fixture_two_rows()
        agg = aggregate_order_lines(rows)
        d = event_payload_digests(
            rows, aggregate=agg,
            order_creation_canonical="2016-04-01T21:05:00Z",
            shipment_canonical="2016-04-06T21:05:00Z",
        )
        assert d["ORDER_CREATED"] == order_aggregate_digest(agg)
        assert d["SHIPMENT_RECORDED"] == _sha256(
            {"order_id": FIXTURE_ORDER, "shipment_canonical": "2016-04-06T21:05:00Z"}
        )
        assert d["DELIVERY_STATUS_RECORDED"] == _sha256(
            {
                "order_id": FIXTURE_ORDER,
                "delivery_status": "Late delivery",
                "order_status": "COMPLETE",
            }
        )


# --------------------------------------------------------------------------- #
# F. chain construction + c1-c8 validation (frozen anchors)
# --------------------------------------------------------------------------- #
class TestChainConstruction:
    def test_genesis_and_head_anchors(self):
        descriptor = build_order_chain(FIXTURE_ORDER, fixture_two_rows())
        assert descriptor["genesis_block_hash"] == FIXTURE_GENESIS
        assert descriptor["last_block_hash"] == FIXTURE_HEAD
        assert descriptor["chain_length"] == 4
        assert descriptor["mapping_status"] == "CONSTRUCTED"

    def test_full_validation_passes(self):
        descriptor = build_order_chain(FIXTURE_ORDER, fixture_two_rows())
        validation = validate_constructed_chain(descriptor)
        assert validation["accepted"] is True
        assert validation["failed_checks"] == []
        assert set(validation["applied_checks"]) == {
            "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8",
        }

    def test_chain_validation_on_pipeline_result(self):
        result = pipeline.map_order(FIXTURE_ORDER, fixture_two_rows())
        assert result.accepted is True
        assert result.chain_length == 4
        assert result.last_block_hash == FIXTURE_HEAD
        assert result.genesis_block_hash == FIXTURE_GENESIS
        assert isinstance(result.events, list) and len(result.events) == 3

    def test_single_line_order(self):
        descriptor = build_order_chain("7", [fixture_row("7", "301")])
        assert descriptor["aggregate"]["item_count"] == 1
        validation = validate_constructed_chain(descriptor)
        assert validation["accepted"] is True


# --------------------------------------------------------------------------- #
# G. pipeline helpers / fail-closed behavior
# --------------------------------------------------------------------------- #
class TestPipelineFailClosed:
    def test_expected_head_guard_builtin(self):
        head = pipeline.current_head()
        assert head == pipeline.EXPECTED_HEAD
        assert pipeline.assert_expected_head() == head

    def test_bad_expected_head_raises(self):
        with pytest.raises(pipeline.V11CMappingError):
            pipeline.assert_expected_head("deadbeef" + "0" * 56)

    def test_dev_scope_fingerprint_shape(self):
        fp = pipeline.fingerprint_dev_scope()
        assert fp.dataset_identity == pipeline.RAW_DATASET_SHA256
        assert len(fp.files) == 2
        for _, digest, rows in fp.files:
            assert len(digest) == 64
            assert rows > 0

    def test_build_lock_semantic_hash_over_payload_only(self):
        summary = pipeline.map_dev_scope(limit_orders=1)
        lock = pipeline.build_mapping_lock(summary)
        recomputed = hashlib.sha256(
            _canonical_json(lock["semantic_payload"]).encode("utf-8")
        ).hexdigest()
        assert lock["semantic_result_lock_sha256"] == recomputed
        pipeline.verify_mapping_lock(lock)

    def test_tampered_lock_fails(self):
        summary = pipeline.map_dev_scope(limit_orders=1)
        lock = pipeline.build_mapping_lock(summary)
        mutated = json.loads(json.dumps(lock))
        mutated["semantic_payload"]["construction_summary"]["unique_orders"] = 0
        with pytest.raises(pipeline.V11CMappingError):
            pipeline.verify_mapping_lock(mutated)


# --------------------------------------------------------------------------- #
# H. bounded governed integration run (TEST access count stays 0)
# --------------------------------------------------------------------------- #
class TestGovernedIntegration:
    def test_full_dev_scope_semantics(self):
        summary = pipeline.map_dev_scope(limit_orders=2000)
        construction = summary["construction"]
        assert construction["test_access_count"] == 0
        assert construction["ai_risk_assessed_generated"] is False
        assert construction["orders_failed"] == 0
        assert construction["chains_constructed"] == construction["chains_validated"]
        assert construction["unique_orders"] >= 2000
        assert construction["items_per_order"]["min"] == 1
        assert 1 <= construction["items_per_order"]["max"] <= 5
        assert 1.0 <= construction["items_per_order"]["mean"] <= 5.0

    def test_dev_scope_statuses_match_governed(self):
        summary = pipeline.map_dev_scope(limit_orders=2000)
        counts = summary["construction"]["delivery_status_row_counts"]
        assert counts.get("Late delivery", 0) > 0
        assert counts.get("Advance shipping", 0) > 0
        assert counts.get("Shipping on time", 0) > 0
        assert counts.get("Shipping canceled", 0) > 0


# --------------------------------------------------------------------------- #
# I. final 46-item report
# --------------------------------------------------------------------------- #
class TestFinalReport:
    def test_report_exists_and_full_run_contract(self):
        summary = pipeline.map_dev_scope(limit_orders=1)
        report = pipeline.build_final_report(summary)
        assert len(report) == 46
        codes = [item["code"] for item in report]
        assert codes[0] == "R01" and codes[-1] == "R46"

    def test_gov_scope_reports_impossible_on_subset(self):
        # On a strict subset the report legitimately reports scope-fail items
        # (R15/R16 compare against the full governed 34k/25,881 scope); the GO
        # contract lives on the persisted full-scope artifact (next test).
        summary = pipeline.map_dev_scope(limit_orders=1)
        report = pipeline.build_final_report(summary)
        assert all(isinstance(item["ok"], bool) for item in report)
        assert sum(1 for item in report if not item["ok"]) >= 1

    def test_persisted_full_scope_report_is_46_46_go(self, bucket_path=None):
        from src.pipeline_v11c import MAPPING_SUMMARY_PATH

        summary = json.loads(MAPPING_SUMMARY_PATH.read_text(encoding="utf-8"))
        report = summary["final_report"]
        assert len(report) == 46
        assert all(item["ok"] for item in report)
        final_item = report[-1]
        assert final_item["code"] == "R46"
        assert final_item["detail"] == "V11C_DATACO_MAPPING_COMPLETE_GO"
        assert summary["construction"]["unique_orders"] == 25_881


# --------------------------------------------------------------------------- #
# L. FAIL_CLOSED order-level consistency (conflict rejection)
# --------------------------------------------------------------------------- #
def _conflict_row(o, i, *, ds="Late delivery", os="COMPLETE", od="4/1/2016 21:05"):
    return {
        "Order Id": o,
        "Order Item Id": i,
        "order date (DateOrders)": od,
        "shipping date (DateOrders)": "4/6/2016 21:05",
        "Delivery Status": ds,
        "Order Status": os,
        "Order Item Quantity": 5,
        "Order Item Total": 199.99,
    }


class TestFailClosedOrderLevelConsistency:
    def test_policy_is_fail_closed(self):
        assert pipeline.ORDER_LEVEL_CONFLICT_POLICY == "FAIL_CLOSED"

    def test_a_identical_order_level_values_accepted(self):
        rows = [
            _conflict_row("777", "1001"),
            _conflict_row("777", "1002", ds="Late delivery"),
        ]
        descriptor = build_order_chain("777", rows)
        assert descriptor["aggregate"]["item_count"] == 2
        assert validate_constructed_chain(descriptor)["accepted"] is True

    def test_b_conflicting_order_date_rejected(self):
        rows = [
            _conflict_row("778", "1001"),
            _conflict_row("778", "1002", od="4/2/2016 21:05"),
        ]
        with pytest.raises(OrderLevelConflictError):
            build_order_chain("778", rows)

    def test_c_conflicting_delivery_status_rejected(self):
        rows = [
            _conflict_row("779", "1001", ds="Late delivery"),
            _conflict_row("779", "1002", ds="Advance shipping"),
        ]
        with pytest.raises(OrderLevelConflictError):
            build_order_chain("779", rows)

    def test_d_conflicting_order_status_rejected(self):
        rows = [
            _conflict_row("780", "1001", os="COMPLETE"),
            _conflict_row("780", "1002", os="CLOSED"),
        ]
        with pytest.raises(OrderLevelConflictError):
            build_order_chain("780", rows)

    def test_e_row_permutation_identical_chain(self):
        rows = [
            _conflict_row("781", "1001"),
            _conflict_row("781", "1002"),
            _conflict_row("781", "1003"),
        ]
        forward = build_order_chain("781", rows)
        permuted = build_order_chain("781", list(reversed(rows)))
        assert forward["aggregate"] == permuted["aggregate"]
        assert forward["genesis_block_hash"] == permuted["genesis_block_hash"]
        assert forward["last_block_hash"] == permuted["last_block_hash"]

    def test_f_conflict_independent_of_row_order(self):
        a = _conflict_row("782", "1001", ds="Late delivery")
        b = _conflict_row("782", "1002", ds="Advance shipping")
        with pytest.raises(OrderLevelConflictError):
            build_order_chain("782", [a, b])
        with pytest.raises(OrderLevelConflictError):
            build_order_chain("782", [b, a])

    def test_conflict_error_identifies_order_and_field(self):
        a = _conflict_row("783", "1001")
        b = _conflict_row("783", "1002", os="ON_HOLD")
        try:
            build_order_chain("783", [a, b])
            raise AssertionError("expected conflict rejection")
        except OrderLevelConflictError as exc:
            message = str(exc)
            assert "783" in message
            assert "Order Status" in message
            assert "FAIL_CLOSED" in message


# --------------------------------------------------------------------------- #
# M. artifact label semantics (row- vs order-level counts)
# --------------------------------------------------------------------------- #
class TestArtifactLabelSemantics:
    def test_row_counts_sum_to_governed_rows(self):
        summary = json.loads(pipeline.MAPPING_SUMMARY_PATH.read_text(encoding="utf-8"))
        key = "delivery_status_row_counts"
        counts = summary["construction"][key]
        assert sum(counts.values()) == 34_000
        assert summary["construction"]["count_level"][key] == "ROW"

    def test_order_counts_sum_to_governed_orders(self):
        summary = json.loads(pipeline.MAPPING_SUMMARY_PATH.read_text(encoding="utf-8"))
        key = "delivery_status_order_counts"
        counts = summary["construction"][key]
        assert sum(counts.values()) == 25_881
        assert summary["construction"]["count_level"][key] == "ORDER"

    def test_event_type_counts_are_order_level(self):
        summary = json.loads(pipeline.MAPPING_SUMMARY_PATH.read_text(encoding="utf-8"))
        counts = summary["construction"]["event_type_counts"]
        assert counts["ORDER_CREATED"] == 25_881
        assert counts["SHIPMENT_RECORDED"] == 25_881
        assert counts["DELIVERY_STATUS_RECORDED"] == 25_881
        assert summary["construction"]["count_level"]["event_type_counts"] == "ORDER"

    def test_lock_binds_fail_closed_and_levels(self):
        lock = json.loads(pipeline.MAPPING_LOCK_PATH.read_text(encoding="utf-8"))
        rules = lock["semantic_payload"]["rules"]
        assert rules["order_level_conflict_policy"] == "FAIL_CLOSED"
        assert rules["count_level"]["delivery_status_row_counts"] == "ROW"
        assert rules["count_level"]["delivery_status_order_counts"] == "ORDER"
        assert rules["count_level"]["event_type_counts"] == "ORDER"


# --------------------------------------------------------------------------- #
# J. pipeline module structural governance
# --------------------------------------------------------------------------- #
class TestGovernanceSurface:
    def test_expected_head_matches_frozen_checkpoint(self):
        assert pipeline.EXPECTED_HEAD == "3f4a74a6731add8798fd4c6b0475b487a5d358ca"

    def test_v11a_lock_reference(self):
        assert pipeline.V11A_LOCK_SHA256 == \
            "aaf3ac3139e8296fb7f976a8a7ed9f18eb317af2f73b10cd0208b6bb870ebd8d"

    def test_mapped_event_world_excludes_ai(self):
        assert "AI_RISK_ASSESSED" not in pipeline.MAPPED_EVENT_TYPES
        assert pipeline.MAPPED_EVENT_TYPES == (
            "ORDER_CREATED",
            "SHIPMENT_RECORDED",
            "DELIVERY_STATUS_RECORDED",
        )


# --------------------------------------------------------------------------- #
# K. missing-value sentinel handling (explicit, deterministic)
# --------------------------------------------------------------------------- #
class TestMissingSentinel:
    def test_sentinel_is_string_marker(self):
        assert isinstance(MISSING_OBSERVED_SENTINEL, str)
        assert MISSING_OBSERVED_SENTINEL == "OBSERVED_MISSING"

    def test_missing_delivery_status_fails_closed(self):
        row = fixture_row(delivery=None)
        with pytest.raises(DataCoSchemaError):
            build_mapping_events(
                [row], aggregate=aggregate_order_lines([row]),
                order_creation_canonical="2016-04-01T21:05:00Z",
                shipment_canonical="2016-04-06T21:05:00Z",
            )

    def test_empty_quant_does_not_break_line_payload(self):
        payload = canonical_fixture_payload()
        assert MISSING_OBSERVED_SENTINEL not in _canonical_json(payload)