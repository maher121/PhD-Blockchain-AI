"""V1.1-F4-A governed read-only runner + persistence harness tests.

The runner is a thin orchestration layer over the frozen read-only V1.1-F2
analysis. These tests verify the runner's behavior without ever invoking the
governed ``execute`` command against the real results directory, without
executing any experiment, and without writing any production artifact. All
persistence tests use temporary directories.

Real frozen V1.1-E evidence is read-only when present; synthetic families
(temp dirs) are used wherever an intentional failure or a persistence run is
needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.pipeline_v11f as f2
from scripts import run_v11f as runner
from tests.test_pipeline_v11f import SyntheticFamily

from src.pipeline_v11f import (
    V11FProtocolLockMismatchError,
    V11FResultLockMismatchError,
    V11FUnmatchedPairError,
)
from scripts.run_v11f import (
    ANALYSIS_RESULTS_NAME,
    EXECUTION_MANIFEST_NAME,
    EXPECTED_F2_SEMANTIC_ANALYSIS_SHA256,
    F2_IMPLEMENTATION_COMMIT,
    RESULT_LOCK_NAME,
    V11F4AuthorizationError,
    V11F4IntegrityError,
    V11F4OutputCollisionError,
)

REAL_V11E_DIR = Path(f2.V11E_DIR)


def _has_frozen_checkpoint() -> bool:
    return (REAL_V11E_DIR / f2.V11E_LOCK_NAME).exists()


# --------------------------------------------------------------------------- #
# shared fixtures (read-only real analysis computed once per module)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real_readonly_payload():
    return runner.run_f2_analysis_readonly()


@pytest.fixture(scope="module")
def real_verify_report():
    return runner.run_readonly_verification(
        authorized_commit=F2_IMPLEMENTATION_COMMIT, require_git=False
    )


@pytest.fixture()
def synthetic_family(tmp_path: Path):
    return SyntheticFamily(tmp_path / "f4v11e")


def _tampered_f1_lock(tmp_path: Path) -> Path:
    lock = json.loads(f2.F1_LOCK_PATH.read_text(encoding="utf-8"))
    lock["semantic_result_lock_sha256"] = "0" * 64
    path = tmp_path / "tampered_v11f_protocol_lock.json"
    path.write_text(f2.canonical_json(lock), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 1. verify mode writes nothing
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _has_frozen_checkpoint(), reason="frozen V1.1-E family absent")
def test_verify_mode_writes_nothing(tmp_path: Path, real_verify_report):
    out_dir = tmp_path / "out"
    assert real_verify_report["marker"] == runner.VERIFY_REPORT_MARKER
    assert real_verify_report["outputs_written"] == []
    assert real_verify_report["mode"] == "verify"
    assert not out_dir.exists() or list(out_dir.iterdir()) == []


# --------------------------------------------------------------------------- #
# 2. execute requires explicit authorization
# --------------------------------------------------------------------------- #


def test_execute_requires_authorization(tmp_path: Path):
    with pytest.raises(V11F4AuthorizationError):
        runner.run_execute(
            authorized_commit=F2_IMPLEMENTATION_COMMIT,
            allow_governed_execution=False,
            require_git=False,
            out_dir=tmp_path / "out",
        )


# --------------------------------------------------------------------------- #
# 3. thin runner: calls the frozen pipeline, never reimplements science
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _has_frozen_checkpoint(), reason="frozen V1.1-E family absent")
def test_runner_calls_frozen_pipeline_not_reimplement(
    monkeypatch, tmp_path: Path
):
    calls: list[dict] = []
    original = f2.run_v11f_analysis

    def spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(f2, "run_v11f_analysis", spy)
    runner.run_readonly_verification(
        authorized_commit=F2_IMPLEMENTATION_COMMIT,
        require_git=False,
        base_dir=REAL_V11E_DIR,
    )
    assert len(calls) == 1
    assert calls[0]["v11e_base_dir"] == Path(f2.V11E_DIR)
    # the runner module contains no scientific primitives of its own
    scientific = {
        "describe_values", "build_paired_descriptive_comparisons",
        "build_normalized_ratios", "build_descriptive_summaries",
        "mean", "median",
    }
    assert not scientific.intersection(dir(runner))
    assert "def describe_values" not in (Path(runner.__file__).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 4. F2 semantic hash pins
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _has_frozen_checkpoint(), reason="frozen V1.1-E family absent")
def test_f2_semantic_hash_equals_frozen_pin(real_verify_report, real_readonly_payload):
    assert (
        real_verify_report["f2_semantic_analysis_sha256"]
        == EXPECTED_F2_SEMANTIC_ANALYSIS_SHA256
    )
    assert (
        real_readonly_payload["semantic_analysis_sha256"]
        == EXPECTED_F2_SEMANTIC_ANALYSIS_SHA256
    )


# --------------------------------------------------------------------------- #
# 5. wrong F1 lock fails closed
# --------------------------------------------------------------------------- #


def test_wrong_f1_lock_fails_closed(tmp_path: Path):
    tampered = _tampered_f1_lock(tmp_path)
    with pytest.raises(V11FProtocolLockMismatchError):
        runner.run_readonly_verification(
            authorized_commit=F2_IMPLEMENTATION_COMMIT,
            require_git=False,
            base_dir=REAL_V11E_DIR,
            f1_lock_path=tampered,
        )


# --------------------------------------------------------------------------- #
# 6. wrong V1.1-E lock fails closed
# --------------------------------------------------------------------------- #


def test_wrong_v11e_lock_fails_closed(synthetic_family: SyntheticFamily):
    # synthetic family's lock can never satisfy the frozen upstream binding
    with pytest.raises(V11FResultLockMismatchError):
        runner.run_readonly_verification(
            authorized_commit=F2_IMPLEMENTATION_COMMIT,
            require_git=False,
            base_dir=synthetic_family.base,
            v11e_binding=None,
        )


# --------------------------------------------------------------------------- #
# 7. output persistence is deterministic
# --------------------------------------------------------------------------- #


def test_persistence_deterministic(tmp_path: Path, synthetic_family: SyntheticFamily):
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    first = runner.run_execute(
        authorized_commit=F2_IMPLEMENTATION_COMMIT,
        allow_governed_execution=True,
        base_dir=synthetic_family.base,
        out_dir=out_a,
        v11e_binding=dict(synthetic_family.binding),
        expected_f2_semantic_analysis_sha256=None,
        require_git=False,
    )
    second = runner.run_execute(
        authorized_commit=F2_IMPLEMENTATION_COMMIT,
        allow_governed_execution=True,
        base_dir=synthetic_family.base,
        out_dir=out_b,
        v11e_binding=dict(synthetic_family.binding),
        expected_f2_semantic_analysis_sha256=None,
        require_git=False,
    )
    for name in (ANALYSIS_RESULTS_NAME, EXECUTION_MANIFEST_NAME, RESULT_LOCK_NAME):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()
    assert (
        first["result_lock_semantic_sha256"] == second["result_lock_semantic_sha256"]
    )
    assert (
        first["f2_semantic_analysis_sha256"] == second["f2_semantic_analysis_sha256"]
    )


# --------------------------------------------------------------------------- #
# 8. result-lock construction is deterministic
# --------------------------------------------------------------------------- #


def test_result_lock_construction_deterministic(real_readonly_payload):
    manifest = runner.build_execution_manifest(
        payload=real_readonly_payload,
        authorized_execution_commit=F2_IMPLEMENTATION_COMMIT,
        f2_implementation={"file_sha256": "x", "git_check": "test"},
    )
    lock_a = runner.build_result_lock(
        payload=real_readonly_payload,
        manifest=manifest,
        analysis_results_sha256="a" * 64,
        manifest_sha256="b" * 64,
    )
    lock_b = runner.build_result_lock(
        payload=real_readonly_payload,
        manifest=manifest,
        analysis_results_sha256="a" * 64,
        manifest_sha256="b" * 64,
    )
    assert lock_a == lock_b
    semantics = lock_a["semantic_payload"]
    assert lock_a["semantic_result_lock_sha256"] == f2.sha256_of_canonical(semantics)
    assert f2.canonical_json(lock_a) == f2.canonical_json(lock_b)


# --------------------------------------------------------------------------- #
# 9. TEST/AI/new-measurement counters remain zero
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _has_frozen_checkpoint(), reason="frozen V1.1-E family absent")
def test_governance_counters_zero(real_readonly_payload):
    governance = real_readonly_payload["governance_checks"]
    for key in ("test_access_requests", "ai_fit_requests",
                "ai_inference_requests", "new_measurement_attempts"):
        assert governance[key] == 0
    assert governance["results_written"] == 0
    manifest = runner.build_execution_manifest(
        payload=real_readonly_payload,
        authorized_execution_commit=F2_IMPLEMENTATION_COMMIT,
        f2_implementation={"file_sha256": "x", "git_check": "test"},
    )
    for key in ("test_access_requests", "ai_fit_requests",
                "ai_inference_requests", "new_measurement_attempts"):
        assert manifest["governance_counters"][key] == 0
    assert manifest["results_written_by_analysis"] == 0


# --------------------------------------------------------------------------- #
# 10. DIRECT_ENERGY_UNAVAILABLE preserved
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _has_frozen_checkpoint(), reason="frozen V1.1-E family absent")
def test_direct_energy_unavailable_preserved(real_readonly_payload, real_verify_report):
    assert real_readonly_payload["direct_energy_status"] == f2.ENERGY_MARKER
    assert real_verify_report["direct_energy_status"] == f2.ENERGY_MARKER
    manifest = runner.build_execution_manifest(
        payload=real_readonly_payload,
        authorized_execution_commit=F2_IMPLEMENTATION_COMMIT,
        f2_implementation={"file_sha256": "x", "git_check": "test"},
    )
    assert manifest["direct_energy_status"] == f2.ENERGY_MARKER
    # no Joules/energy estimates may leak into persisted output bytes
    text = f2.canonical_json(real_readonly_payload)
    for token in ("joule", "kwh", "watts", "co2e", "carbon"):
        assert token.lower() not in text.lower()


# --------------------------------------------------------------------------- #
# 11. ranking/winner/composite forbidden
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _has_frozen_checkpoint(), reason="frozen V1.1-E family absent")
def test_ranking_winner_composite_forbidden(real_readonly_payload):
    f2.assert_no_ranking_fields(real_readonly_payload)
    manifest = runner.build_execution_manifest(
        payload=real_readonly_payload,
        authorized_execution_commit=F2_IMPLEMENTATION_COMMIT,
        f2_implementation={"file_sha256": "x", "git_check": "test"},
    )
    lock = runner.build_result_lock(
        payload=real_readonly_payload,
        manifest=manifest,
        analysis_results_sha256="a" * 64,
        manifest_sha256="b" * 64,
    )
    semantics = f2.canonical_json(lock["semantic_payload"])
    assert '"overall_winner_present":false' in semantics
    assert '"composite_score_present":false' in semantics
    assert '"ranking_fields_present":false' in semantics
    for forbidden in ("overall_score", "composite_overall_policy_score",
                      "global_policy_ranking", "best_policy"):
        assert forbidden not in semantics


# --------------------------------------------------------------------------- #
# 12. E08 remains policy-independent
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _has_frozen_checkpoint(), reason="frozen V1.1-E family absent")
def test_e08_policy_independent(real_readonly_payload):
    storage = real_readonly_payload["policy_independent_storage"]
    assert storage["status"] == "POLICY_INDEPENDENT"
    assert storage["storage_policy_differences_allowed"] is False
    assert list(storage["per_workload"]) == ["100", "250", "500", "1000", "2500"]
    # storage never appears as a policy pair/ratio metric
    assert "persisted_canonical_serialized_bytes" not in real_readonly_payload[
        "paired_descriptive_comparisons"
    ]["per_contrast"]
    assert "persisted_canonical_serialized_bytes" not in real_readonly_payload[
        "normalized_ratios"
    ]["ratios"]


# --------------------------------------------------------------------------- #
# 13. E10 remains derived-only
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _has_frozen_checkpoint(), reason="frozen V1.1-E family absent")
def test_e10_derived_only(real_readonly_payload):
    upstream = real_readonly_payload["upstream_result_lock"]
    assert upstream["e10_derived_cells"] == 45
    e10 = real_readonly_payload["governance_checks"]["v11e_e10_derived_only_verified"]
    assert e10["verified"] is True
    assert e10["sources"] == ["E06", "E07", "E08", "E09"]
    computational_work = real_readonly_payload["metric_definitions"]["families"][
        "computational_work"
    ]
    assert computational_work["source_experiment"] == "E10"


# --------------------------------------------------------------------------- #
# 14. E07 remains COMPUTATIONAL_MEMORY_PROXY
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _has_frozen_checkpoint(), reason="frozen V1.1-E family absent")
def test_e07_memory_proxy_pinned(real_readonly_payload, real_verify_report):
    memory = real_readonly_payload["metric_definitions"]["families"]["memory"]
    assert memory["memory_label"] == "COMPUTATIONAL_MEMORY_PROXY"
    assert memory["canonical_metric"] == "persisted_tracemalloc_peak_proxy"
    assert memory["energy_basis"] == f2.ENERGY_MARKER
    assert real_verify_report["checks"]["memory_label"] == "COMPUTATIONAL_MEMORY_PROXY"
    assert (
        real_verify_report["checks"]["e07_canonical_metric"]
        == "persisted_tracemalloc_peak_proxy"
    )


# --------------------------------------------------------------------------- #
# 15. protected unrelated worktree entries do not block execution
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _has_frozen_checkpoint(), reason="requires git + frozen data")
def test_protected_worktree_entries_do_not_block():
    # this repository has a legitimately dirty worktree (protected entries);
    # the govlier runner must authorize the frozen commit without requiring a
    # clean tree (read-only git rev-parse / git show only)
    head = runner._git("HEAD")
    report = runner.verify_authorized_execution_commit(
        head, require_git=True, git_resolver=runner._git
    )
    assert report["head"] == report["origin_main"] == report["authorized_execution_commit"]
    implementation = runner.verify_f2_implementation_commit(require_git=True)
    assert implementation["verified"] is True
    assert implementation["commit"] == F2_IMPLEMENTATION_COMMIT


# --------------------------------------------------------------------------- #
# 16. no authorized output during readiness against the real results dir
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _has_frozen_checkpoint(), reason="frozen V1.1-E family absent")
def test_readiness_does_not_mutate_frozen_real_outputs(real_verify_report):
    """F4-B evolved the lifecycle guarantee.

    Before F4-B the governed F4 result artifacts did not exist, so VERIFY proved
    "no real F4 outputs are produced". F4-B has legitimately frozen them, so VERIFY
    now proves the stronger read-only property: it must not create, rewrite, mutate,
    or delete the already-frozen real outputs, and every authoritative semantic /
    result-lock binding must still recompute.
    """
    real_out = Path(f2.V11F_DIR)
    names = (ANALYSIS_RESULTS_NAME, EXECUTION_MANIFEST_NAME, RESULT_LOCK_NAME)
    missing = [name for name in names if not (real_out / name).exists()]
    assert not missing, f"frozen F4 artifacts missing: {missing}"

    before = {
        name: {
            "sha256": f2.sha256_file(real_out / name),
            "bytes": (real_out / name).stat().st_size,
            "mtime_ns": (real_out / name).stat().st_mtime_ns,
        }
        for name in names
    }

    assert real_verify_report["mode"] == "verify"
    assert real_verify_report["outputs_written"] == []

    for name in names:
        path = real_out / name
        stat = path.stat()
        assert f2.sha256_file(path) == before[name]["sha256"]
        assert stat.st_size == before[name]["bytes"]
        assert stat.st_mtime_ns == before[name]["mtime_ns"]

    # authoritative bindings remain valid after VERIFY
    assert real_verify_report["f2_semantic_analysis_sha256"] == (
        "0765ea4d79cdad956d09156983395c9c85fe748002cd91a084e2dc5ef6bf3899"
    )
    assert before[ANALYSIS_RESULTS_NAME]["sha256"] == (
        "eb4ce1c69d56385a0bddc606bed40ec4d9dcc049e77a172c612a0414f2357d3a"
    )
    assert before[EXECUTION_MANIFEST_NAME]["sha256"] == (
        "d1fb42d0bfb8b11267fc6572c2c333d7d77ca1b559533eff6309a128bfd599f8"
    )

    lock = json.loads((real_out / RESULT_LOCK_NAME).read_text(encoding="utf-8"))
    assert lock["semantic_result_lock_sha256"] == (
        "d639be8970bd6662ed0efe688ec2c0375414942ebf636d2b958abd88d722bd1a"
    )
    assert lock["semantic_result_lock_sha256"] == f2.sha256_of_canonical(
        lock["semantic_payload"]
    )
    assert lock["artifacts_fingerprints_sha256"][ANALYSIS_RESULTS_NAME] == (
        before[ANALYSIS_RESULTS_NAME]["sha256"]
    )
    assert lock["artifacts_fingerprints_sha256"][EXECUTION_MANIFEST_NAME] == (
        before[EXECUTION_MANIFEST_NAME]["sha256"]
    )

    v11e_lock = json.loads((Path(f2.V11E_DIR) / f2.V11E_LOCK_NAME).read_text(encoding="utf-8"))
    assert f2.sha256_file(Path(f2.V11E_DIR) / f2.V11E_LOCK_NAME) == (
        "dd3b926eb08206f505e739ed61c5f29a3e7f03a370e9afc1a077e0b260fd6e64"
    )
    assert v11e_lock["semantic_result_lock_sha256"] == (
        "058aeca8ac97101356bcf1c4dc5b74fb3affbda6833a85b5d423a53556cd1749"
    )
    assert v11e_lock["semantic_result_lock_sha256"] == f2.sha256_of_canonical(
        v11e_lock["semantic_payload"]
    )


# --------------------------------------------------------------------------- #
# fail-closed propagation: unexpected governed evidence + output collision
# --------------------------------------------------------------------------- #


def test_unexpected_governed_evidence_fails_closed(synthetic_family: SyntheticFamily):
    synthetic_family.mutate_e08_duplicate_placeholder()
    with pytest.raises(V11FUnmatchedPairError):
        runner.run_f2_analysis_readonly(
            base_dir=synthetic_family.base,
            v11e_binding=dict(synthetic_family.binding),
        )


def test_execute_output_collision_fails_closed(tmp_path: Path, synthetic_family: SyntheticFamily):
    out_dir = tmp_path / "out"
    runner.run_execute(
        authorized_commit=F2_IMPLEMENTATION_COMMIT,
        allow_governed_execution=True,
        base_dir=synthetic_family.base,
        out_dir=out_dir,
        v11e_binding=dict(synthetic_family.binding),
        expected_f2_semantic_analysis_sha256=None,
        require_git=False,
    )
    (out_dir / ANALYSIS_RESULTS_NAME).write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(V11F4OutputCollisionError):
        runner.run_execute(
            authorized_commit=F2_IMPLEMENTATION_COMMIT,
            allow_governed_execution=True,
            base_dir=synthetic_family.base,
            out_dir=out_dir,
            v11e_binding=dict(synthetic_family.binding),
            expected_f2_semantic_analysis_sha256=None,
            require_git=False,
        )


def test_execute_idempotent_resume_identical_bytes(tmp_path: Path, synthetic_family: SyntheticFamily):
    out_dir = tmp_path / "out"
    first = runner.run_execute(
        authorized_commit=F2_IMPLEMENTATION_COMMIT,
        allow_governed_execution=True,
        base_dir=synthetic_family.base,
        out_dir=out_dir,
        v11e_binding=dict(synthetic_family.binding),
        expected_f2_semantic_analysis_sha256=None,
        require_git=False,
    )
    second = runner.run_execute(
        authorized_commit=F2_IMPLEMENTATION_COMMIT,
        allow_governed_execution=True,
        base_dir=synthetic_family.base,
        out_dir=out_dir,
        v11e_binding=dict(synthetic_family.binding),
        expected_f2_semantic_analysis_sha256=None,
        require_git=False,
    )
    assert first["result_lock_semantic_sha256"] == second["result_lock_semantic_sha256"]
    for name in (ANALYSIS_RESULTS_NAME, EXECUTION_MANIFEST_NAME, RESULT_LOCK_NAME):
        assert (out_dir / name).exists()