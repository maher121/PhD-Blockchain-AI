"""Focused tests for V0.9-B synthetic BGWO protocol validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import src.pipeline_v09b as v09b


ROOT = Path(__file__).resolve().parents[1]


def test_load_protocol_returns_locked_configuration() -> None:
    protocol = v09b.load_v09b_protocol()
    assert protocol.config.optimizer_name == "BGWO"
    assert protocol.config.dimensions == 43
    assert protocol.config.wolf_count == 12
    assert protocol.config.evaluated_iterations == 20
    assert protocol.config.maximum_candidate_requests == 240
    assert protocol.optimizer_seeds == (2042, 2043, 2044, 2045, 2046)
    assert protocol.model_attack_seeds == (42, 43, 44, 45, 46)
    assert protocol.split_seed == 42
    assert len(protocol.config_sha256) == 64


def test_load_protocol_rejects_yaml_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(v09b.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["optimizer"]["wolves"] = 11
    config_path = tmp_path / "bgwo_v09_drift.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(v09b.V09BProtocolError, match="Frozen V0.9-A protocol drift"):
        v09b.load_v09b_protocol(config_path=config_path)


def test_load_protocol_rejects_doc_yaml_consistency_drift(tmp_path: Path) -> None:
    text = v09b.DEFAULT_PROTOCOL_DOC_PATH.read_text(encoding="utf-8")
    drifted = text.replace("wolves (population) = `12`", "wolves (population) = `13`")
    doc_path = tmp_path / "v09_bgwo_protocol_drift.md"
    doc_path.write_text(drifted, encoding="utf-8")
    with pytest.raises(v09b.V09BProtocolError, match="consistency"):
        v09b.load_v09b_protocol(protocol_doc_path=doc_path)


def test_verify_v08_frozen_integrity_passes() -> None:
    result = v09b.verify_v08_frozen_integrity()
    assert result["status"] == "PASS"
    assert result["immutable_hash_count"] >= 1
    assert (
        result["v08d_semantic_result_lock_sha256"]
        == "ac19e5a0dba5d058f88485e6ab8ab9a96f97c30f697e95a744ccaa6563d2b657"
    )
    assert (
        result["v08e_semantic_result_lock_sha256"]
        == "7307fda1cdb5f100cb6268dc0f4eff05b65bfcefadeb06707c8a3e13fd6888d5"
    )


def test_constrained_comparator_prioritizes_feasibility_then_violation() -> None:
    infeasible = v09b.SyntheticEvaluation(
        feasible=False,
        violation=0.4,
        cardinality=1,
        primary_score=10.0,
        secondary_score=10.0,
        tie_break_key="a",
    )
    feasible = v09b.SyntheticEvaluation(
        feasible=True,
        violation=0.0,
        cardinality=3,
        primary_score=0.0,
        secondary_score=0.0,
        tie_break_key="b",
    )
    lower_violation = v09b.SyntheticEvaluation(
        feasible=False,
        violation=0.1,
        cardinality=3,
        primary_score=0.0,
        secondary_score=0.0,
        tie_break_key="c",
    )

    assert v09b.constrained_evaluation_is_better(feasible, infeasible)
    assert not v09b.constrained_evaluation_is_better(infeasible, feasible)
    assert v09b.constrained_evaluation_is_better(lower_violation, infeasible)


def test_run_synthetic_validation_produces_pass_artifact(tmp_path: Path) -> None:
    output_path = tmp_path / "v09b_protocol_validation.json"
    summary = v09b.run_v09b_synthetic_validation(output_path=output_path)
    assert summary["stage"] == v09b.V09B_STAGE
    assert summary["status"] == "PASS"
    assert summary["validation_kind"] == v09b.V09B_VALIDATION_KIND
    assert summary["scientific_experiment"] is False
    assert summary["scientific_claims_supported"] is False
    assert all(summary["checks"].values())
    assert summary["governance"]["dataco_accessed"] is False
    assert summary["governance"]["production_bgwo_search_executed"] is False
    assert summary["governance"]["bpso_rerun"] is False

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted == summary
