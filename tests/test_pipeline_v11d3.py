"""V1.1-D3 governed risk generation tests (synthetic fixtures + read-only).

These tests exercise the D3 orchestrator and its deterministic helpers with
hand-constructed fixtures. They never fit models on real partition data, never
read real partition features, never run the real reconstruction campaign, and
never append anywhere. The only real artifacts touched are the frozen
read-only provenance files re-verified by the preflight (config, D1/D2 locks,
HYBRID-K13 winner lock, core_runs.csv, and row/order metadata counts).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import types

import numpy as np
import pandas as pd
import pytest

from src.ai_risk.protocol import D1_SEMANTIC_LOCK_SHA256, AIRiskProtocol
from src.ai_risk.order_aggregation import RowScore, aggregate_rows
from src.ai_risk.risk_record import RiskRecord
from src.blockchain_engine.canonical import sha256_hex
import src.pipeline_v11d3 as pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FROZEN_FEATURES = [
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

SYNTH_CONFIG = {
    "feature_identity": {
        "configuration_id": "HYBRID-K13",
        "winner_mask_sha256": pipeline.HYBRID_K13_MASK_SHA256,
        "canonical_feature_manifest_sha256": pipeline.HYBRID_K13_MANIFEST_SHA256,
        "ordered_features": FROZEN_FEATURES,
    },
    "classifier": {
        "primary": "decision_tree",
        "parameters": {
            "class_weight": "balanced",
            "max_depth": 5,
            "min_samples_leaf": 20,
        },
        "hard_prediction_threshold": 0.5,
    },
    "model_strategy": {
        "type": "FIXED_FIVE_SEED_MEAN_ENSEMBLE",
        "seeds": [42, 43, 44, 45, 46],
    },
    "risk_score": {"semantics_version": "MODEL_DERIVED_ATTACK_RISK_V1"},
    "row_to_order": {"rule": "MAX"},
}


def _synth_protocol() -> AIRiskProtocol:
    return AIRiskProtocol(
        config=SYNTH_CONFIG,
        config_sha256="a" * 64,
        protocol_doc_sha256="b" * 64,
        lock={},
        lock_semantic_sha256="c" * 64,
    )


def _passing_preflight() -> pipeline.D3PreflightResult:
    return pipeline.D3PreflightResult(
        stage=pipeline.STAGE,
        observed_head=pipeline.D3_STARTING_CHECKPOINT,
        origin_main=pipeline.D3_STARTING_CHECKPOINT,
        d1_lock_semantic_sha256=D1_SEMANTIC_LOCK_SHA256,
        v11c_lock_semantic_sha256=pipeline.V11C_MAPPING_SEMANTIC_SHA256,
        hybrid_k13_configuration_id="HYBRID-K13",
        hybrid_k13_mask_sha256=pipeline.HYBRID_K13_MASK_SHA256,
        hybrid_k13_feature_list_sha256=pipeline.HYBRID_K13_FEATURE_LIST_SHA256,
        hybrid_k13_manifest_sha256=pipeline.HYBRID_K13_MANIFEST_SHA256,
        train_rows=28000,
        validation_rows=6000,
        validation_orders=4588,
        test_access_count=0,
    )


def _records_for(protocol, order_scores: dict[str, float]) -> list[RiskRecord]:
    rows = [
        RowScore(row_id=str(index), order_id=order, score=score)
        for index, (order, score) in enumerate(order_scores.items())
    ]
    aggregates = aggregate_rows(rows)
    return pipeline.build_ordered_records(protocol, aggregates)


class FakeScorer:
    def __init__(self, scores) -> None:
        self._scores = scores

    def anomaly_scores(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series(self._scores, index=features.index, dtype=float)


def _fake_workloads(features: pd.DataFrame, metadata: pd.DataFrame, n=5):
    workloads = []
    for _ in range(n):
        validation = types.SimpleNamespace(
            clean_features=features.copy(),
            clean_metadata=metadata.copy(),
            candidate_features=tuple(FROZEN_FEATURES),
        )
        workloads.append(types.SimpleNamespace(validation=validation))
    return workloads


def _clean_view():
    features = pd.DataFrame(
        {name: [1.0, 2.0, 3.0] for name in FROZEN_FEATURES},
        index=[10, 11, 12],
    )
    metadata = pd.DataFrame(
        {"row_id": [1, 2, 3], "Order Id": [111, 111, 222]}
    )
    return features, metadata


# --------------------------------------------------------------------------- #
# preflight and authorization gates
# --------------------------------------------------------------------------- #


class TestPreflight:
    def test_real_preflight_passes_with_checkpoint_guard(self):
        result = pipeline.run_v11d3_preflight()
        assert result.passed
        assert result.semantic_sha256 == sha256_hex(result.semantic_payload())

    def test_real_preflight_passes_without_git_guard(self):
        result = pipeline.run_v11d3_preflight(checkpoint_guard=False)
        assert result.observed_head == pipeline.D3_STARTING_CHECKPOINT
        assert result.passed

    def test_preflight_fail_closed_on_drifted_head(self):
        drifted = dataclasses.replace(
            _passing_preflight(), observed_head="0" * 40
        )
        assert drifted.passed is False

    def test_preflight_fail_closed_on_drifted_train_rows(self):
        drifted = dataclasses.replace(_passing_preflight(), train_rows=0)
        assert drifted.passed is False

    def test_preflight_fail_closed_on_nonzero_test_access(self):
        drifted = dataclasses.replace(_passing_preflight(), test_access_count=1)
        assert drifted.passed is False


class TestAuthorization:
    def test_authorize_refuses_failed_preflight(self):
        drifted = dataclasses.replace(_passing_preflight(), validation_orders=0)
        with pytest.raises(pipeline.V11D3NotAuthorizedError):
            pipeline.authorize_d3_execution(drifted)

    def test_authorize_grants_capability_bound_to_preflight(self):
        preflight = _passing_preflight()
        capability = pipeline.authorize_d3_execution(preflight)
        assert capability.stage == pipeline.STAGE
        assert capability.preflight_semantic_sha256 == preflight.semantic_sha256

    def test_require_capability_rejects_none(self):
        with pytest.raises(pipeline.V11D3NotAuthorizedError):
            pipeline.require_d3_capability(None)

    def test_require_capability_rejects_wrong_stage(self):
        with pytest.raises(pipeline.V11D3NotAuthorizedError):
            pipeline.require_d3_capability(
                pipeline.D3ExecutionCapability(
                    stage="V1.1-D", preflight_semantic_sha256="x" * 64
                )
            )

    def test_require_capability_rejects_empty_semantic(self):
        with pytest.raises(pipeline.V11D3NotAuthorizedError):
            pipeline.require_d3_capability(
                pipeline.D3ExecutionCapability(
                    stage=pipeline.STAGE, preflight_semantic_sha256=""
                )
            )

    def test_generation_fails_closed_without_capability(self):
        with pytest.raises(pipeline.V11D3NotAuthorizedError):
            pipeline.run_governed_risk_generation(None)

    def test_generation_fails_closed_when_module_gate_closed(self):
        capability = pipeline.D3ExecutionCapability(
            stage=pipeline.STAGE, preflight_semantic_sha256="x" * 64
        )
        with pytest.raises(pipeline.V11D3NotAuthorizedError):
            pipeline.run_governed_risk_generation(capability)


# --------------------------------------------------------------------------- #
# deterministic row scoring and model fitting
# --------------------------------------------------------------------------- #


class TestScoreValidationRows:
    def test_row_ensemble_is_arithmetic_mean_in_seed_order(self):
        protocol = _synth_protocol()
        features, metadata = _clean_view()
        workloads = _fake_workloads(features, metadata)
        scorers = [
            FakeScorer([0.10, 0.15, 0.20]),
            FakeScorer([0.20, 0.25, 0.30]),
            FakeScorer([0.30, 0.35, 0.40]),
            FakeScorer([0.40, 0.45, 0.50]),
            FakeScorer([0.50, 0.55, 0.60]),
        ]
        rows = pipeline.score_validation_rows(protocol, scorers, workloads)
        expected = {
            ("1", "111"): 0.30,
            ("2", "111"): 0.35,
            ("3", "222"): 0.40,
        }
        assert len(rows) == 3
        for row in rows:
            assert row.score == pytest.approx(expected[(row.row_id, row.order_id)])

    def test_scored_rows_aggregate_to_max_per_order(self):
        protocol = _synth_protocol()
        features, metadata = _clean_view()
        workloads = _fake_workloads(features, metadata)
        scorers = [
            FakeScorer([0.10, 0.99, 0.20]),
            FakeScorer([0.20, 0.25, 0.30]),
            FakeScorer([0.30, 0.35, 0.40]),
            FakeScorer([0.40, 0.45, 0.50]),
            FakeScorer([0.50, 0.55, 0.60]),
        ]
        rows = pipeline.score_validation_rows(protocol, scorers, workloads)
        aggregates = aggregate_rows(rows)
        assert list(aggregates) == ["111", "222"]
        row1_mean = (0.99 + 0.25 + 0.35 + 0.45 + 0.55) / 5
        assert aggregates["111"].risk_score == pytest.approx(row1_mean)
        row2_mean = (0.20 + 0.30 + 0.40 + 0.50 + 0.60) / 5
        assert aggregates["222"].risk_score == pytest.approx(row2_mean)

    def test_member_count_must_match_frozen_seed_count(self):
        protocol = _synth_protocol()
        features, metadata = _clean_view()
        workloads = _fake_workloads(features, metadata, n=4)
        scorers = [FakeScorer([0.1, 0.1, 0.1]) for _ in range(4)]
        with pytest.raises(pipeline.V11D3Error):
            pipeline.score_validation_rows(protocol, scorers, workloads)

    def test_clean_view_fails_closed_on_missing_frozen_feature(self):
        protocol = _synth_protocol()
        features, metadata = _clean_view()
        features = features.drop(columns=[FROZEN_FEATURES[0]])
        workloads = _fake_workloads(features, metadata)
        scorers = [FakeScorer([0.1, 0.1, 0.1]) for _ in range(5)]
        with pytest.raises(pipeline.V11D3Error):
            pipeline.score_validation_rows(protocol, scorers, workloads)


class TestFitMemberModels:
    def test_fits_five_decision_trees_with_frozen_features(self):
        protocol = _synth_protocol()
        workloads = []
        for seed in protocol.seeds():
            rng = np.random.default_rng(seed)
            features = pd.DataFrame(
                rng.normal(size=(40, len(FROZEN_FEATURES))),
                columns=FROZEN_FEATURES,
            )
            labels = pd.Series([0] * 36 + [1] * 4)
            train = types.SimpleNamespace(features=features, labels=labels)
            workloads.append(types.SimpleNamespace(train=train))
        models = pipeline.fit_member_models(protocol, workloads)
        assert len(models) == 5
        for model in models:
            assert model.model_name == "decision_tree"
            assert tuple(model.feature_names) == tuple(FROZEN_FEATURES)
            assert model.is_fitted


# --------------------------------------------------------------------------- #
# records, artifact, summary, lock, references
# --------------------------------------------------------------------------- #


class TestRecordsAndArtifact:
    def test_ordered_records_sorted_and_digest_consistent(self):
        protocol = _synth_protocol()
        records = _records_for(
            protocol, {"111": 0.2, "222": 0.5, "333": 0.9}
        )
        assert [record.order_id for record in records] == ["111", "222", "333"]
        for record in records:
            record.verify()

    def test_risk_level_matches_pre_registered_bands(self):
        protocol = _synth_protocol()
        records = _records_for(
            protocol, {"111": 0.2, "222": 0.5, "333": 0.9}
        )
        levels = {record.order_id: record.risk_level for record in records}
        assert levels == {"111": "LOW", "222": "MEDIUM", "333": "HIGH"}

    def test_artifact_document_digest_is_recomputable(self):
        protocol = _synth_protocol()
        records = _records_for(protocol, {"111": 0.2, "222": 0.5})
        document = pipeline.build_artifact_document(
            records, generated_at_utc="2026-09-22T00:00:00Z"
        )
        assert document["record_count"] == 2
        assert document["aggregation_rule"] == "MAX"
        assert document["input_view"] == "CLEAN_FROZEN_VALIDATION_FEATURES"
        assert document["artifact_sha256"] == sha256_hex(
            [record.to_mapping() for record in records]
        )


def _full_scope_scores() -> dict[str, float]:
    """Carefully balanced 4588-order scope across the three frozen bands."""
    scores: dict[str, float] = {}
    for index in range(pipeline.EXPECTED_VALIDATION_ORDERS):
        if index < 2000:
            value = 0.2
        elif index < 3588:
            value = 0.5
        else:
            value = 0.9
        scores[str(index + 1)] = value
    return scores


def _generated_artifacts(tmp_path, order_scores: dict[str, float] | None = None):
    """Build synthetic records + artifact/summary/lock files in tmp_path."""
    if order_scores is None:
        order_scores = {"111": 0.2, "222": 0.5, "333": 0.9}
    protocol = _synth_protocol()
    records = _records_for(protocol, order_scores)
    artifact = pipeline.build_artifact_document(
        records, generated_at_utc="2026-09-22T00:00:00Z"
    )
    artifact_path = tmp_path / "artifact.json"
    pipeline._atomic_write_json(artifact_path, artifact)
    artifact["artifact_file_sha256"] = pipeline._sha256_file(artifact_path)

    preflight = _passing_preflight()
    summary = pipeline.build_generation_summary(
        preflight=preflight,
        protocol=protocol,
        records=records,
        artifact_path=artifact_path,
        artifact_document=artifact,
        generated_at_utc="2026-09-22T00:00:00Z",
        test_access_count=0,
    )
    summary["summary_sha256"] = sha256_hex(
        {k: v for k, v in summary.items() if k != "summary_sha256"}
    )
    summary_path = tmp_path / "summary.json"
    pipeline._atomic_write_json(summary_path, summary)

    lock = pipeline.build_result_lock(
        preflight=preflight,
        protocol=protocol,
        artifact_document=artifact,
        artifact_file_sha256=artifact["artifact_file_sha256"],
        summary=summary,
        summary_file_sha256=pipeline._sha256_file(summary_path),
        generated_at_utc="2026-09-22T00:00:00Z",
    )
    lock_path = tmp_path / "lock.json"
    pipeline._atomic_write_json(lock_path, lock)
    return protocol, records, preflight, artifact, summary, lock, (
        artifact_path,
        summary_path,
        lock_path,
    )


class TestSummaryAndLock:
    def test_summary_carries_expected_distribution_and_governance(self, tmp_path):
        _, records, _, _, summary, _, _ = _generated_artifacts(tmp_path)
        dist = summary["risk_distribution"]
        assert (dist["LOW"], dist["MEDIUM"], dist["HIGH"]) == (1, 1, 1)
        assert dist["min"] >= 0.0 and dist["max"] <= 1.0
        assert summary["governance"]["no_optimizer"] is True
        assert summary["governance"]["no_threshold_tuning"] is True
        assert summary["governance"]["no_corpus_append"] is True
        assert summary["test_isolation"]["test_access_count"] == 0
        assert summary["partitions"]["validation_orders"] == len(records)

    def test_result_lock_semantic_is_recomputable_and_verifiable(self, tmp_path):
        _, records, _, _, _, lock, _ = _generated_artifacts(tmp_path)
        assert lock["semantic_result_lock_sha256"] == sha256_hex(
            lock["semantic_payload"]
        )
        pipeline.verify_result_lock(lock, expected_records=len(records))

    def test_result_lock_fail_closed_on_wrong_order_count(self, tmp_path):
        _, records, _, _, _, lock, _ = _generated_artifacts(tmp_path)
        with pytest.raises(pipeline.V11D3Error):
            pipeline.verify_result_lock(lock, expected_records=len(records) + 1)

    def test_reference_preparation_verifies_every_record(self):
        protocol = _synth_protocol()
        records = _records_for(protocol, {"111": 0.2, "222": 0.5, "333": 0.9})
        count = pipeline.verify_ai_risk_reference_preparation(
            protocol,
            records,
            artifact_sha256="d" * 64,
            result_lock_sha256="e" * 64,
        )
        assert count == len(records)


# --------------------------------------------------------------------------- #
# frozen 33-item final report
# --------------------------------------------------------------------------- #


class TestFinalReport:
    def test_report_is_exactly_33_items_ending_in_review_required_marker(self, tmp_path):
        protocol, records, preflight, _, summary, _, paths = _generated_artifacts(
            tmp_path
        )
        artifact_path, summary_path, lock_path = paths
        report = pipeline.build_final_report(
            preflight=preflight,
            protocol=protocol,
            summary=summary,
            record_count=len(records),
            artifact_path=artifact_path,
            artifact_file_sha256=summary["artifact"]["file_sha256"],
            summary_path=summary_path,
            summary_file_sha256=pipeline._sha256_file(summary_path),
            result_lock_path=lock_path,
            result_lock_file_sha256=pipeline._sha256_file(lock_path),
            references_verified=len(records),
        )
        assert len(report) == pipeline.FINAL_REPORT_COUNT
        codes = [item["code"] for item in report]
        assert codes[0] == "R01" and codes[-1] == "R33"
        assert report[-1]["detail"] == pipeline.FINAL_REPORT_MARKER

    def test_report_all_pass_on_consistent_full_scope_artifacts(self, tmp_path):
        protocol, records, preflight, _, summary, _, paths = _generated_artifacts(
            tmp_path, _full_scope_scores()
        )
        assert len(records) == pipeline.EXPECTED_VALIDATION_ORDERS
        artifact_path, summary_path, lock_path = paths
        report = pipeline.build_final_report(
            preflight=preflight,
            protocol=protocol,
            summary=summary,
            record_count=len(records),
            artifact_path=artifact_path,
            artifact_file_sha256=summary["artifact"]["file_sha256"],
            summary_path=summary_path,
            summary_file_sha256=pipeline._sha256_file(summary_path),
            result_lock_path=lock_path,
            result_lock_file_sha256=pipeline._sha256_file(lock_path),
            references_verified=len(records),
        )
        assert all(item["ok"] for item in report)

    def test_report_subset_honestly_fails_scope_items(self, tmp_path):
        # A 3-order synthetic subset legitimately reports R26 (governed record
        # count == 4588) as FAIL, mirroring the V1.1-C subset-report contract.
        protocol, records, preflight, _, summary, _, paths = _generated_artifacts(
            tmp_path
        )
        assert len(records) == 3
        artifact_path, summary_path, lock_path = paths
        report = pipeline.build_final_report(
            preflight=preflight,
            protocol=protocol,
            summary=summary,
            record_count=len(records),
            artifact_path=artifact_path,
            artifact_file_sha256=summary["artifact"]["file_sha256"],
            summary_path=summary_path,
            summary_file_sha256=pipeline._sha256_file(summary_path),
            result_lock_path=lock_path,
            result_lock_file_sha256=pipeline._sha256_file(lock_path),
            references_verified=len(records),
        )
        r26 = [item for item in report if item["code"] == "R26"][0]
        assert r26["ok"] is False
        assert r26["status"] == "FAIL"

    def test_report_fails_closed_on_tampered_artifact(self, tmp_path):
        protocol, records, preflight, _, summary, _, paths = _generated_artifacts(
            tmp_path
        )
        artifact_path, summary_path, lock_path = paths
        artifact_path.write_text("{}", encoding="utf-8")
        report = pipeline.build_final_report(
            preflight=preflight,
            protocol=protocol,
            summary=summary,
            record_count=len(records),
            artifact_path=artifact_path,
            artifact_file_sha256=summary["artifact"]["file_sha256"],
            summary_path=summary_path,
            summary_file_sha256=pipeline._sha256_file(summary_path),
            result_lock_path=lock_path,
            result_lock_file_sha256=pipeline._sha256_file(lock_path),
            references_verified=len(records),
        )
        r30 = [item for item in report if item["code"] == "R30"][0]
        assert r30["ok"] is False
        assert r30["status"] == "FAIL"


# --------------------------------------------------------------------------- #
# read-only frozen provenance metadata (cheap, never test partition data)
# --------------------------------------------------------------------------- #


class TestFrozenCounts:
    def test_validation_metadata_counts(self):
        rows, orders = pipeline._count_metadata(
            PROJECT_ROOT / "data" / "processed" / "validation" / "metadata.csv"
        )
        assert (rows, orders) == (6000, 4588)

    def test_train_metadata_rows(self):
        rows, _ = pipeline._count_metadata(
            PROJECT_ROOT / "data" / "processed" / "train" / "metadata.csv"
        )
        assert rows == 28000