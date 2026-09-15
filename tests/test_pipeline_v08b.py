"""Focused governance and real-preflight tests for V0.8-B."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import src.pipeline_v08b as v08b
from src.optimization.feature_fitness import audit_fitness_context
from tests.test_bpso_fitness import make_context


@pytest.fixture(scope="module")
def real_basis() -> v08b.FrozenBasis:
    return v08b.load_frozen_basis()


@pytest.fixture(scope="module")
def real_context(real_basis):
    workloads = v08b.load_frozen_development_workloads(real_basis)
    return v08b.build_fitness_context(real_basis, workloads)


@pytest.fixture()
def synthetic_basis() -> v08b.FrozenBasis:
    context = make_context()
    k42_mask = np.ones(43, dtype=np.uint8)
    k42_mask[2] = 0
    k42_features = tuple(
        feature
        for feature, active in zip(context.candidate_features, k42_mask)
        if active
    )
    return v08b.FrozenBasis(
        candidate_features=context.candidate_features,
        candidate_manifest_sha256=context.candidate_manifest_sha256,
        k42_features=k42_features,
        k42_mask=k42_mask,
        k42_features_sha256=v08b.fingerprint_feature_names(k42_features),
        k42_mask_sha256=v08b.hashlib.sha256(k42_mask.tobytes()).hexdigest(),
        k11_seed_specific=True,
        k11_feature_hashes={seed: str(seed) * 64 for seed in v08b.EXPECTED_SEEDS},
        baseline=context.baseline,
        baseline_seed_metrics={},
        dataset_metadata=context.workloads[0].dataset_metadata,
        lock={},
        source_hashes={},
    )


def test_01_v08a_is_committed_and_unmodified() -> None:
    result = v08b.verify_v08a_committed()
    assert result["status"] == "PASS"
    assert len(result["commit"]) == 40
    assert result["working_tree_changes_for_v08a"] is False


def test_02_frozen_feature_manifest_and_protocol_are_verified(real_basis) -> None:
    assert len(real_basis.candidate_features) == 43
    assert (
        real_basis.candidate_manifest_sha256
        == "5146fd08fe766979adaf443bf9f4f7d32ee3c94cfdaf0fd46ec0efd10e92a10d"
    )
    assert real_basis.dataset_metadata["split"]["seed"] == 42
    assert real_basis.dataset_metadata["split"]["strategy"] == "order_grouped"
    assert real_basis.dataset_metadata["split"]["sizes"] == {
        "train": 28000,
        "validation": 6000,
        "test": 6000,
    }


def test_03_real_k42_binding_comes_from_lock_and_is_deterministic(real_basis) -> None:
    assert len(real_basis.k42_features) == 42
    assert int(real_basis.k42_mask.sum()) == 42
    assert real_basis.k42_features_sha256 == (
        "c355fe6d1f8ed5557b3f94f48cd0c4400d9f4a6894f52916c79a7721a912eef6"
    )
    assert real_basis.k42_mask.tolist() == [1, 1, 0, *([1] * 40)]


def test_04_k11_remains_seed_specific(real_basis) -> None:
    assert real_basis.k11_seed_specific is True
    assert set(real_basis.k11_feature_hashes) == set(v08b.EXPECTED_SEEDS)
    assert len(set(real_basis.k11_feature_hashes.values())) > 1


def test_05_real_fitness_context_passes_leakage_audit(real_context) -> None:
    audit = audit_fitness_context(real_context)
    assert audit.status == "PASS"
    assert all(check.passed for check in audit.checks)
    assert real_context.test_accessed is False


def test_06_real_k43_reproduces_frozen_metrics_and_repeat(real_basis, real_context) -> None:
    record, evaluation = v08b.reproduce_k43(real_context, real_basis)
    assert record["status"] == "PASS"
    assert all(record["checks"].values())
    assert record["tolerance"]["value"] == 1e-12
    assert record["deterministic_repeat"] is True
    assert evaluation.selected_feature_count == 43
    assert evaluation.decision_tree_fit_count == 5
    assert record["preflight_decision_tree_fits"] == 10
    assert record["reproduced_metrics"] == pytest.approx(
        {
            "average_precision": 0.23068406113411433,
            "f1": 0.19770107263983716,
            "recall": 0.32199999999999995,
        },
        abs=1e-12,
    )


def test_07_pilot_is_quarantined_and_uses_only_twelve_requests(synthetic_basis) -> None:
    context = make_context()
    pilot = v08b.run_quarantined_pilot(context, synthetic_basis)
    budget = v08b.build_budget_projection(pilot)
    assert pilot["label"] == "QUARANTINED_PILOT"
    assert pilot["eligibility"] == "NOT_ELIGIBLE_FOR_FINAL_SELECTION"
    assert pilot["final_selection_performed"] is False
    assert pilot["test_accessed"] is False
    assert pilot["fitness_requests"] == 12
    assert pilot["actual_decision_tree_fits"] == pilot["unique_masks"] * 5
    assert pilot["initial_cardinalities"] == [43, 42, 4, 11]
    assert pilot["quarantined_candidate"]["feature_names_published"] is False
    assert budget["full_search_executed"] is False
    assert budget["approved_full_search"]["maximum_decision_tree_fits"] == 6000


def test_08_precheck_failure_prohibits_pilot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_basis,
) -> None:
    context = make_context()
    monkeypatch.setattr(v08b, "snapshot_immutable_paths", lambda: {"frozen": "same"})
    monkeypatch.setattr(v08b, "verify_v08a_committed", lambda: {"status": "PASS"})
    monkeypatch.setattr(v08b, "build_fitness_context", lambda basis, workloads: context)
    monkeypatch.setattr(
        v08b,
        "reproduce_k43",
        lambda context, basis: (
            {
                "stage": "V0.8-B",
                "label": "REAL_FITNESS_PREFLIGHT",
                "status": "PRECHECK_FAIL",
            },
            None,
        ),
    )

    def forbidden_pilot(*args, **kwargs):
        raise AssertionError("Pilot must not run after PRECHECK_FAIL")

    monkeypatch.setattr(v08b, "run_quarantined_pilot", forbidden_pilot)
    with pytest.raises(v08b.V08BPrecheckError, match="PRECHECK_FAIL"):
        v08b.run_v08b(
            output_dir=tmp_path,
            basis_loader=lambda: synthetic_basis,
            workload_loader=lambda basis: context.workloads,
        )
    assert not (tmp_path / "v08b_pilot.json").exists()
    assert (tmp_path / "v08b_preflight.json").is_file()


def test_09_pipeline_order_is_fail_closed() -> None:
    source = inspect.getsource(v08b.run_v08b)
    ordered = (
        "immutable_before = snapshot_immutable_paths",
        "v08a_integrity = verify_v08a_committed",
        "basis = basis_loader",
        "workloads = workload_loader",
        "context = build_fitness_context",
        "leakage = audit_fitness_context",
        "preflight, _ = reproduce_k43",
        "pilot = run_quarantined_pilot",
        "budget = build_budget_projection",
        "_verify_generated_artifacts(output)",
    )
    positions = [source.index(name) for name in ordered]
    assert positions == sorted(positions)


def test_10_pipeline_has_no_final_test_loader_or_full_search_entrypoint() -> None:
    source = inspect.getsource(v08b)
    assert "load_v06_locked_test_data" not in source
    assert "prepare_v06_locked_test_data" not in source
    assert "run_locked_test_experiment" not in source
    assert "logistic_regression" not in source
    assert "DIRECT_ENERGY" not in source
    assert "12, 20" not in source


def test_11_immutable_snapshot_includes_v06_v07_and_v08a() -> None:
    snapshot = v08b.snapshot_immutable_paths()
    paths = "\n".join(snapshot)
    assert "validation_lock.json" in paths
    assert "v07d_artifact_hashes.json" in paths
    assert "src/optimization/bpso.py" in paths
    assert "v08a_protocol_validation.json" in paths
