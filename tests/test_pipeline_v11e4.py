"""V1.1-E4L governed execution launcher + persistence layer tests.

These tests exercise the launcher with hand-constructed synthetic fixtures in
temporary directories only. They never run the governed E01-E10 campaigns over
the real population (``run_execution_session`` is always given
``require_git=False`` and a :class:`SyntheticDataProvider`), never fit models,
never perform AI inference, never touch TEST, and never write into
``results/blockchain/v11e/``.

Coverage map (E4L requirements):
* authorized-commit validation + fail-closed HEAD/origin checks;
* deterministic cell identity/ordering and the frozen per-experiment counts;
* atomic canonical JSON writing (temp-file + fsync + os.replace, no remnants);
* workload/attack manifest determinism (nested prefixes, LOW/HIGH markers);
* plan mode (no execution) and resume semantics (skip valid / fail closed);
* fail-closed conditions: conflicting cell, corrupted semantic hash, protocol
  mismatch, provenance mismatch, authorized-commit mismatch, missing cell;
* E10 derived ONLY from persisted verified E06-E09 (zero measurement), and
  fail-closed when a source is missing;
* aggregated descriptive summaries restricted to the frozen statistics;
* read-only verify mode; result-lock fail-closed conditions + leak counters.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

import src.pipeline_v11e4 as m
import src.pipeline_v11e3 as p

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UNIVERSE = [str(index) for index in range(1, 11)]
LEVELS = ["LOW"] + ["MEDIUM"] * 7 + ["HIGH"] * 2
SIZES = (2, 4, 6)
SEEDS = (522,)
ALL_MEDIUM = ["MEDIUM"] * 10

AUTH = "ab" * 20  # 40-hex, format-valid only (no git in tests)


def make_provider(order_ids=UNIVERSE, levels=LEVELS):
    return m.SyntheticDataProvider(order_ids, risk_levels=levels)


def session_kwargs(base_dir: Path, *, provider=None, auth=AUTH, sizes=SIZES, seeds=SEEDS):
    return {
        "provider": provider or make_provider(),
        "authorized_commit": auth,
        "base_dir": base_dir,
        "sizes": sizes,
        "seeds": seeds,
        "require_git": False,
    }


def _family_hashes(base_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(base_dir)): m.sha256_file(path)
        for path in sorted(base_dir.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def full_session(tmp_path_factory) -> tuple[Path, m.SyntheticDataProvider]:
    """One complete synthetic session over a temp dir (module-wide reuse)."""
    base_dir = tmp_path_factory.mktemp("v11e4_sess")
    provider = make_provider()
    result = m.run_execution_session(**session_kwargs(base_dir, provider=provider))
    assert len(result["executed_cells"]) == 78
    assert result["skipped_cells"] == []
    return base_dir, provider


# --------------------------------------------------------------------------- #
# module identity + governance constants
# --------------------------------------------------------------------------- #


def test_module_identity():
    assert m.STAGE == "V1.1-E4L"
    assert m.KIND == "V11E4_GOVERNED_EXECUTION_LAYER"
    assert m.RESULTS_STAGE == "V1.1-E4"
    assert m.EXECUTION_ORDER[:2] == ("E01", "E02")
    assert m.EXECUTION_ORDER[-1] == "E09"
    assert m.POLICY_INDEPENDENT_EXPERIMENTS == ("E08",)
    assert m.P_ONLY_EXPERIMENTS == ("E05",)


def test_frozen_output_artifact_names():
    assert m.RAW_OBS_DIR_NAME == "v11e_raw_run_observations"
    assert m.ATTACK_MANIFEST_NAME == "v11e_attack_instance_manifest.json"
    assert m.WORKLOAD_MANIFEST_NAME == "v11e_workload_manifest.json"
    assert m.RESULTS_DIR_NAME == "v11e_experiment_results"
    assert m.SUMMARY_DIR_NAME == "v11e_aggregated_descriptive_summaries"
    assert m.RESULT_LOCK_NAME == "v11e_experiment_result_lock.json"


# --------------------------------------------------------------------------- #
# authorization
# --------------------------------------------------------------------------- #


def test_governed_execution_refused_without_allowance(tmp_path):
    # Guard must fire BEFORE any materialization/provider use.
    provider = m.GovernedDataProvider.__new__(m.GovernedDataProvider)
    with pytest.raises(m.V11E4NotAuthorizedError):
        m.run_execution_session(
            provider=provider,
            authorized_commit=AUTH,
            base_dir=tmp_path,
            sizes=SIZES,
            seeds=SEEDS,
            require_git=False,
        )
    assert list(tmp_path.rglob("*")) == []


def test_authorized_commit_accepts_40_hex():
    report = m.verify_authorized_execution_commit(AUTH, require_git=False)
    assert report["authorized_execution_commit"] == AUTH
    assert report["head"] == AUTH and report["origin_main"] == AUTH


def test_authorized_commit_rejects_invalid_format():
    with pytest.raises(m.V11E4NotAuthorizedError):
        m.verify_authorized_execution_commit("not-a-sha", require_git=False)
    with pytest.raises(m.V11E4NotAuthorizedError):
        m.verify_authorized_execution_commit("ZZ" * 20, require_git=False)
    with pytest.raises(m.V11E4NotAuthorizedError):
        m.verify_authorized_execution_commit("", require_git=False)


def test_authorized_commit_fails_closed_on_head_drift():
    resolver = {"HEAD": "1" * 40, "origin/main": AUTH}
    with pytest.raises(m.V11E4NotAuthorizedError):
        m.verify_authorized_execution_commit(AUTH, require_git=True,
                                             git_resolver=resolver.get)


def test_authorized_commit_fails_closed_on_origin_drift():
    resolver = {"HEAD": AUTH, "origin/main": "2" * 40}
    with pytest.raises(m.V11E4NotAuthorizedError):
        m.verify_authorized_execution_commit(AUTH, require_git=True,
                                             git_resolver=resolver.get)


def test_authorized_commit_passes_when_all_match():
    resolver = {"HEAD": AUTH, "origin/main": AUTH}
    report = m.verify_authorized_execution_commit(AUTH, require_git=True,
                                                  git_resolver=resolver.get)
    assert report["authorized_execution_commit"] == AUTH


def test_no_hardcoded_future_execution_sha_anywhere():
    # The E4L launcher and the E3 machinery must never bake a future execution
    # commit (or the obsolete E4 authorization constant). Authorization is a
    # runtime input (--authorized-commit), verified against HEAD/origin/main.
    for path in (PROJECT_ROOT / "src" / "pipeline_v11e4.py",
                 PROJECT_ROOT / "src" / "pipeline_v11e3.py"):
        source = path.read_text()
        assert "E4_AUTHORIZED_HEAD" not in source
        assert "E4_RUNTIME_READINESS" not in source
        assert "run_v11e4_runtime_preflight" not in source
        assert "verify_e4_runtime_head_guard" not in source
        assert "90c9d7e935d95b6226029fe2be1aeace1358e9ad" not in source
        assert "332555dc1a3b7760801b501d4177044091481364" not in source
    # historical E3 stage guard remains untouched by the launcher
    assert hasattr(p, "EXPECTED_HEAD")


def test_launcher_preflight_composes_e3_data_gates_and_authorization(tmp_path):
    report = m.run_launcher_preflight(
        authorized_commit=AUTH,
        base_dir=tmp_path,
        sizes=SIZES,
        seeds=SEEDS,
        require_git=False,
    )
    assert report["marker"] == m.PREFLIGHT_MARKER
    assert report["authorization"] == {
        "authorized_execution_commit": AUTH, "head": AUTH, "origin_main": AUTH}
    assert report["e3_data_gates_passed"] is True
    assert report["leaks_zero"] is True
    assert report["leak_counts"]["test_access_count"] == 0
    assert report["leak_counts"]["ai_fit_count"] == 0
    assert report["leak_counts"]["ai_inference_count"] == 0
    assert report["leak_counts"]["experiment_execution_count"] == 0
    assert report["plan"]["total_cells"] == 78
    assert report["plan"]["expected_total"] == 78
    assert report["family_state"]["persisted_cells"] == 0
    assert report["family_state"]["stale_temp_files"] == []


def test_launcher_preflight_fails_closed_on_head_or_origin_drift(tmp_path):
    resolve = {"HEAD": "1" * 40, "origin/main": AUTH}
    with pytest.raises(m.V11E4NotAuthorizedError):
        m.run_launcher_preflight(
            authorized_commit=AUTH, base_dir=tmp_path, sizes=SIZES, seeds=SEEDS,
            require_git=True, git_resolver=resolve.get)
    resolve = {"HEAD": AUTH, "origin/main": "2" * 40}
    with pytest.raises(m.V11E4NotAuthorizedError):
        m.run_launcher_preflight(
            authorized_commit=AUTH, base_dir=tmp_path, sizes=SIZES, seeds=SEEDS,
            require_git=True, git_resolver=resolve.get)
    assert list(tmp_path.rglob("*")) == []


# --------------------------------------------------------------------------- #
# canonical serialization + atomic writer
# --------------------------------------------------------------------------- #


def test_canonical_json_deterministic():
    doc = {"b": [2, 1], "a": {"z": None, "y": "x"}}
    assert m.canonical_json(doc) == m.canonical_json(doc)
    manual = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    assert m.canonical_json(doc) == manual
    assert "\n" not in m.canonical_json(doc)


def test_canonical_json_rejects_non_finite():
    with pytest.raises(ValueError):
        m.canonical_json({"x": float("nan")})
    with pytest.raises(ValueError):
        m.canonical_json({"x": float("inf")})


def test_atomic_write_json(tmp_path):
    target = tmp_path / "nested" / "out.json"
    m.atomic_write_json(target, {"a": 1, "b": [True, None]})
    assert target.read_text() == json.dumps({"a": 1, "b": [True, None]},
                                            sort_keys=True, separators=(",", ":"))
    assert list(tmp_path.rglob("*.tmp.*")) == []


def test_atomic_write_json_failure_leaves_no_temp(monkeypatch, tmp_path):
    original_replace = os.replace
    def broken_replace(src: os.PathLike, dst: os.PathLike):
        raise OSError("boom")
    monkeypatch.setattr(m.os, "replace", broken_replace)
    target = tmp_path / "out.json"
    with pytest.raises(OSError):
        m.atomic_write_json(target, {"x": 1})
    assert not target.exists()
    assert list(tmp_path.rglob("*.tmp.*")) == []
    monkeypatch.setattr(m.os, "replace", original_replace)


# --------------------------------------------------------------------------- #
# plan + deterministic cell identity
# --------------------------------------------------------------------------- #


def test_expected_cell_counts_frozen():
    assert m.expected_cell_count("E01") == 45
    assert m.expected_cell_count("E04") == 45
    assert m.expected_cell_count("E05") == 15
    assert m.expected_cell_count("E06") == 45
    assert m.expected_cell_count("E07") == 45
    assert m.expected_cell_count("E08") == 15
    assert m.expected_cell_count("E09") == 45
    assert m.expected_cell_count("E10") == 45
    assert m.expected_total_cell_count() == 390


def test_build_execution_plan_structure():
    plan = m.build_execution_plan()
    ids = [spec.experiment_id for spec in plan]
    assert ids.count("E10") == 45
    assert all(spec.experiment_id != "E10" for spec in plan[:-45])
    assert ids[-1] == "E10"
    assert plan[:-45][-1].experiment_id == "E09"
    # counterbalanced per seed (E01 cells for a fixed workload size)
    for seed in p.BLOCKCHAIN_SEEDS:
        policies = [spec.policy for spec in plan
                    if spec.experiment_id == "E01" and spec.workload_size == 100
                    and spec.seed == seed]
        assert tuple(policies) == p.policy_execution_order(seed)


def test_cell_identity_deterministic_and_commit_free():
    spec_a = m.CellSpec("E06", 522, 100, "B0")
    spec_b = m.CellSpec("E06", 522, 100, "B0")
    assert spec_a.cell_identity_sha256() == spec_b.cell_identity_sha256()
    assert spec_a.repetition == 1 and m.CellSpec("E06", 523, 100, "B0").repetition == 2
    assert spec_a.cell_key() == "E06|s522|w100|B0"
    assert spec_a.file_stem() == "E06_s522_w00100_pB0"
    assert "authorized_execution_commit" not in spec_a.identity_payload()


def test_cell_keys_unique_in_plan():
    plan = m.build_execution_plan()
    keys = [spec.cell_key() for spec in plan]
    assert len(keys) == len(set(keys))
    stems = [spec.file_stem() for spec in plan]
    assert len(stems) == len(set(stems))


def test_e08_and_e05_policy_shape_in_plan():
    plan = m.build_execution_plan()
    e08 = [spec for spec in plan if spec.experiment_id == "E08"]
    e05 = [spec for spec in plan if spec.experiment_id == "E05"]
    assert all(spec.policy == "B0" and spec.policy_independent for spec in e08)
    assert all(spec.policy == "P" for spec in e05)


# --------------------------------------------------------------------------- #
# workload + attack manifests
# --------------------------------------------------------------------------- #


def test_workload_manifest_deterministic_and_nested():
    provider = make_provider()
    a = m.build_workload_manifest(provider, sizes=SIZES, seeds=SEEDS)
    b = m.build_workload_manifest(provider, sizes=SIZES, seeds=SEEDS)
    assert a == b
    assert a["nested_prefixes"] is True and a["no_replacement"] is True
    per_size = a["workloads_by_seed"]["522"]
    assert list(per_size) == ["2", "4", "6"]
    # nested prefixes
    assert set(per_size["2"]["order_ids"]) <= set(per_size["4"]["order_ids"])
    assert set(per_size["4"]["order_ids"]) <= set(per_size["6"]["order_ids"])
    counts = per_size["6"]["risk_counts"]
    assert counts["LOW"] + counts["MEDIUM"] + counts["HIGH"] == 6
    assert all(value >= 0 for value in counts.values())
    markers = per_size["6"]["markers"]
    assert ("LOW_ABSENT_IN_GOVERNED_POPULATION" in markers) == (counts["LOW"] == 0)
    assert ("HIGH_NOT_OBSERVED_IN_CONDITION" in markers) == (counts["HIGH"] == 0)
    assert "LOW_ABSENT_IN_GOVERNED_POPULATION" in per_size["2"]["markers"]


def test_workload_manifest_low_absent_high_absent_all_medium():
    provider = make_provider(order_ids=UNIVERSE, levels=ALL_MEDIUM)
    manifest = m.build_workload_manifest(provider, sizes=SIZES, seeds=SEEDS)
    per_size = manifest["workloads_by_seed"]["522"]["6"]["markers"]
    assert "LOW_ABSENT_IN_GOVERNED_POPULATION" in per_size
    assert "HIGH_NOT_OBSERVED_IN_CONDITION" in per_size


def test_attack_manifest_deterministic_and_copy_only():
    provider = make_provider()
    a = m.build_attack_instance_manifest(provider, sizes=SIZES, seeds=SEEDS)
    b = m.build_attack_instance_manifest(provider, sizes=SIZES, seeds=SEEDS)
    assert a == b
    instances = a["by_seed"]["522"]["6"]
    assert set(instances) == set(p.PA_SCENARIOS)
    for scenario_id in p.PA_SCENARIOS:
        instance = instances[scenario_id]
        assert instance["copies_only"] is True
        assert instance["donor_read_only"] is True
        assert len(instance["tampered_block_hashes"]) == instance["tampered_block_count"]
        assert instance["semantic_sha256"]
    # PA-08/PA-09 donors distinct from targets
    for scenario_id in ("PA-08", "PA-09"):
        instance = instances[scenario_id]
        assert instance["donor_order_id"] not in (None, instance["target_order_id"])


# --------------------------------------------------------------------------- #
# plan mode (no execution) + manifests persistence
# --------------------------------------------------------------------------- #


def test_plan_mode_writes_manifests_without_cells(tmp_path):
    provider = make_provider()
    workload, attack, paths = m._fill_family_artifacts(
        tmp_path, provider=provider, sizes=SIZES, seeds=SEEDS,
        write_outputs=True, require_clean_write=False,
    )
    assert paths["workload_manifest"].exists()
    assert paths["attack_manifest"].exists()
    assert not paths["stage_dir"].exists()
    on_disk = m.read_json(paths["workload_manifest"], "workload manifest")
    assert on_disk == workload


def test_plan_mode_fails_closed_on_manifest_drift(tmp_path):
    provider = make_provider()
    m._fill_family_artifacts(tmp_path, provider=provider, sizes=SIZES, seeds=SEEDS,
                             write_outputs=True, require_clean_write=False)
    other = make_provider(order_ids=[str(i) for i in range(20, 30)], levels=ALL_MEDIUM)
    with pytest.raises(m.V11E4IntegrityError):
        m._fill_family_artifacts(tmp_path, provider=other, sizes=SIZES, seeds=SEEDS,
                                 write_outputs=True, require_clean_write=False)


# --------------------------------------------------------------------------- #
# session + resume + fail-closed conditions
# --------------------------------------------------------------------------- #


def test_full_session_family_and_lock(full_session):
    base_dir, _ = full_session
    paths = m.lineate_output_paths(base_dir)
    assert paths["workload_manifest"].exists()
    assert paths["attack_manifest"].exists()
    assert paths["result_lock"].exists()
    stage = paths["stage_dir"]
    assert len(list(stage.glob("*.json"))) == 78
    # raw observation files exist only for measurement cells (E10 is derived-only)
    raw_count = sum(1 for _ in (paths["raw_obs_dir"].rglob("*.json")))
    assert raw_count == 69
    assert len(list(paths["summary_dir"].glob("E*.json"))) == 10


def test_resume_skips_valid_cells_and_is_idempotent(full_session, tmp_path):
    base_dir, provider = full_session
    out = m.run_execution_session(**session_kwargs(base_dir, provider=provider))
    assert out["executed_cells"] == []
    assert len(out["skipped_cells"]) == 78


def test_missing_cell_resumes_and_completes(full_session, tmp_path):
    base_dir, provider = full_session
    work = tmp_path / "copy"
    shutil.copytree(base_dir, work)
    victim = (work / m.RESULTS_DIR_NAME / m.RESULTS_STAGE_DIR_NAME
              / "E06_s522_w00002_pB0.json")
    victim.unlink()
    out = m.run_execution_session(**session_kwargs(work, provider=provider))
    assert out["executed_cells"] == ["E06|s522|w2|B0"]
    assert len(out["skipped_cells"]) == 77
    victim = _result_lock(work)
    assert victim["execution_metadata"]["experiment_status"] == "EXECUTED"


def test_conflicting_cell_fails_closed(full_session, tmp_path):
    base_dir, provider = full_session
    work = tmp_path / "conflict"
    shutil.copytree(base_dir, work)
    target = work / m.RESULTS_DIR_NAME / m.RESULTS_STAGE_DIR_NAME / "E06_s522_w00002_pB0.json"
    document = json.loads(target.read_text())
    document["result"]["primary"] = {"hacked": 1}
    m.atomic_write_json(target, document)
    with pytest.raises(m.V11E4IntegrityError):
        m.run_execution_session(**session_kwargs(work, provider=provider))


def test_corrupted_semantic_hash_fails_closed(full_session, tmp_path):
    base_dir, provider = full_session
    work = tmp_path / "corrupt"
    shutil.copytree(base_dir, work)
    target = work / m.RESULTS_DIR_NAME / m.RESULTS_STAGE_DIR_NAME / "E06_s522_w00002_pB1.json"
    document = json.loads(target.read_text())
    document["semantic_sha256"] = "0" * 64
    m.atomic_write_json(target, document)
    with pytest.raises(m.V11E4IntegrityError):
        m.run_execution_session(**session_kwargs(work, provider=provider))


def test_protocol_mismatch_fails_closed(full_session, tmp_path):
    base_dir, provider = full_session
    work = tmp_path / "protocol"
    shutil.copytree(base_dir, work)
    target = work / m.RESULTS_DIR_NAME / m.RESULTS_STAGE_DIR_NAME / "E06_s522_w00002_pP.json"
    document = json.loads(target.read_text())
    document["protocol_sha256"] = "0" * 64
    m.atomic_write_json(target, document)
    with pytest.raises(m.V11E4IntegrityError, match="protocol"):
        m.run_execution_session(**session_kwargs(work, provider=provider))


def test_provenance_mismatch_fails_closed(full_session, tmp_path):
    base_dir, provider = full_session
    work = tmp_path / "provenance"
    shutil.copytree(base_dir, work)
    target = work / m.RESULTS_DIR_NAME / m.RESULTS_STAGE_DIR_NAME / "E09_s522_w00004_pB0.json"
    document = json.loads(target.read_text())
    document["upstream_provenance"] = {"drift": "yes"}
    m.atomic_write_json(target, document)
    with pytest.raises(m.V11E4IntegrityError, match="semantic"):
        m.run_execution_session(**session_kwargs(work, provider=provider))


def test_authorized_commit_mismatch_fails_closed(full_session, tmp_path):
    base_dir, provider = full_session
    work = tmp_path / "commit"
    other_auth = "cd" * 20
    m.run_execution_session(**session_kwargs(work, provider=provider, auth=other_auth))
    with pytest.raises(m.V11E4IntegrityError, match="authorized-execution-commit"):
        m.run_execution_session(**session_kwargs(work, provider=provider, auth=AUTH))


# --------------------------------------------------------------------------- #
# E10: derived only, from persisted verified E06-E09, zero measurement
# --------------------------------------------------------------------------- #


def _result_lock(base_dir: Path) -> dict:
    return m.read_json(base_dir / m.RESULT_LOCK_NAME, "result lock")


def test_e10_cells_are_derived_only_without_measurement(full_session):
    base_dir, _ = full_session
    stage = base_dir / m.RESULTS_DIR_NAME / m.RESULTS_STAGE_DIR_NAME
    e10_files = sorted(stage.glob("E10_*.json"))
    assert len(e10_files) == 9
    for path in e10_files:
        document = m.read_json(path, "E10 cell")
        result = document["result"]
        assert result["derived_only"] is True
        assert result["no_fourth_campaign"] is True
        assert "raw_records" not in result
        assert set(document["source_cells"]) == {"E06", "E07", "E08", "E09"}
    lock = _result_lock(base_dir)
    assert lock["semantic_payload"]["fail_closed"]["e10_measurement_campaign_count"] == 0


def test_e10_derivation_from_persisted_cells_direct(full_session):
    base_dir, _ = full_session
    stage = base_dir / m.RESULTS_DIR_NAME / m.RESULTS_STAGE_DIR_NAME
    workload = m.read_json(base_dir / m.WORKLOAD_MANIFEST_NAME, "workload manifest")
    spec = m.CellSpec("E10", 522, 4, "P")
    authorization = m.verify_authorized_execution_commit(AUTH, require_git=False)
    workload_model = m._WorkloadModel(make_provider(), sizes=SIZES, seeds=SEEDS)
    document = m.derive_e10_from_persisted_cells(
        spec,
        provider=make_provider(),
        authorization=authorization,
        workload_semantic_sha256=workload["semantic_sha256"],
        workload_order_ids=workload_model.order_ids(522, 4),
        stage_dir=stage,
        workloads=workload_model,
    )
    assert document["result"]["derived_only"] is True
    assert document["result"]["direct_energy"] == p.ENERGY_MARKER


def test_e10_derivation_fails_closed_when_source_missing(tmp_path):
    base_dir = tmp_path / "empty"
    base_dir.mkdir()
    stage = base_dir / "V1.1-E4"
    stage.mkdir()
    spec = m.CellSpec("E10", 522, 4, "P")
    authorization = m.verify_authorized_execution_commit(AUTH, require_git=False)
    workload_model = m._WorkloadModel(make_provider(), sizes=SIZES, seeds=SEEDS)
    with pytest.raises(m.V11E4IntegrityError, match="E06"):
        m.derive_e10_from_persisted_cells(
            spec,
            provider=make_provider(),
            authorization=authorization,
            workload_semantic_sha256="0" * 64,
            workload_order_ids=workload_model.order_ids(522, 4),
            stage_dir=stage,
            workloads=workload_model,
        )


# --------------------------------------------------------------------------- #
# aggregated descriptive summaries (frozen statistics only)
# --------------------------------------------------------------------------- #


def test_summary_uses_frozen_statistics_only(full_session):
    base_dir, _ = full_session
    summary_dir = base_dir / m.SUMMARY_DIR_NAME
    for experiment_id in m.EXECUTION_ORDER + ("E10",):
        summary = m.read_json(summary_dir / f"{experiment_id}.json", "summary")
        assert summary["experiment_id"] == experiment_id
        blob = json.dumps({
            "per_policy": summary["per_policy"],
            "paired_differences": summary["paired_differences"],
        })
        for forbidden in ("pvalue", "p_value", "bootstrap", "confidence",
                          "ci_", "inferential"):
            assert forbidden.lower() not in blob.lower()
        policies = list(summary["per_policy"])
        assert policies
        for policy in policies:
            for metric, stats in summary["per_policy"][policy]["metrics"].items():
                assert set(stats) == {"mean", "median", "std_ddof_1", "min", "max"}
        if experiment_id in ("E02", "E03", "E04", "E01", "E06", "E07", "E09"):
            for policy in policies:
                rates = summary["per_policy"][policy]["rates"]
                for rate in rates.values():
                    assert set(rate) == {"numerator", "denominator", "rate"}


def test_summary_paired_differences_match_policy_pairs(full_session):
    base_dir, _ = full_session
    summary = m.read_json(base_dir / m.SUMMARY_DIR_NAME / "E06.json", "E06 summary")
    assert set(summary["paired_differences"]) == {
        "B1_minus_B0", "P_minus_B0", "P_minus_B1"}
    pd = summary["paired_differences"]["B1_minus_B0"]["wall_time_ns"]
    assert set(pd) == {"mean", "median", "std_ddof_1", "min", "max"}


def test_summary_e08_grouped_policy_independent(full_session):
    base_dir, _ = full_session
    summary = m.read_json(base_dir / m.SUMMARY_DIR_NAME / "E08.json", "E08 summary")
    assert list(summary["per_policy"]) == ["POLICY_INDEPENDENT"]
    metrics = summary["per_policy"]["POLICY_INDEPENDENT"]["metrics"]
    assert "canonical_serialized_blockchain_bytes_per_workload" in metrics


def test_summary_descriptive_stats_are_deterministic(full_session):
    base_dir, provider = full_session
    a = m.read_json(base_dir / m.SUMMARY_DIR_NAME / "E06.json", "E06 summary")
    rebuilt = m.build_descriptive_summaries(
        base_dir=base_dir, authorized_commit=AUTH, sizes=SIZES, seeds=SEEDS,
        write_outputs=False,
    )["E06"]
    assert a == rebuilt


# --------------------------------------------------------------------------- #
# verify mode (read-only) + result lock fail-closed conditions
# --------------------------------------------------------------------------- #


def test_verify_mode_is_read_only(full_session):
    base_dir, _ = full_session
    before = _family_hashes(base_dir)
    size_before = len(before)
    report = m.verify_result_family(
        base_dir, authorized_commit=AUTH, sizes=SIZES, seeds=SEEDS, require_git=False)
    assert report["well_formed"] is True and report["errors"] == []
    after = _family_hashes(base_dir)
    assert len(after) == size_before and after == before


def test_verify_detects_missing_cell(full_session, tmp_path):
    base_dir, _ = full_session
    work = tmp_path / "missing"
    shutil.copytree(base_dir, work)
    (work / m.RESULTS_DIR_NAME / m.RESULTS_STAGE_DIR_NAME / "E01_s522_w00006_pB0.json").unlink()
    report = m.verify_result_family(
        work, authorized_commit=AUTH, sizes=SIZES, seeds=SEEDS, require_git=False)
    assert report["well_formed"] is False
    assert any("missing cells" in error for error in report["errors"])


def test_verify_detects_duplicate_cell(full_session, tmp_path):
    base_dir, _ = full_session
    work = tmp_path / "dup"
    shutil.copytree(base_dir, work)
    stage = work / m.RESULTS_DIR_NAME / m.RESULTS_STAGE_DIR_NAME
    shutil.copy(stage / "E01_s522_w00002_pB0.json", stage / "E01_s522_w00002_pB0_copy.json")
    report = m.verify_result_family(
        work, authorized_commit=AUTH, sizes=SIZES, seeds=SEEDS, require_git=False)
    assert report["well_formed"] is False
    assert any("duplicate" in error or "unexpected" in error for error in report["errors"])


def test_result_lock_fails_closed_on_leak_counts(full_session, tmp_path, monkeypatch):
    base_dir, provider = full_session
    work = tmp_path / "leak"
    shutil.copytree(base_dir, work)
    monkeypatch.setattr(
        m, "_session_leak_counts",
        lambda: {"test_access_count": 1, "ai_fit_count": 1,
                 "ai_inference_count": 1, "experiment_execution_count": 1},
    )
    lock = m.build_result_lock(
        work, authorized_commit=AUTH,
        executed_cells=[], skipped_cells=[],
        sizes=SIZES, seeds=SEEDS,
    )
    assert lock["execution_metadata"]["experiment_status"] == "INCOMPLETE"
    fail_closed = lock["semantic_payload"]["fail_closed"]
    assert fail_closed["test_access_zero"] is False
    assert fail_closed["ai_fit_zero"] is False
    assert fail_closed["ai_inference_zero"] is False
    assert fail_closed["experiment_execution_zero"] is False
    assert fail_closed["e10_measurement_campaign_count"] == 0
    with pytest.raises(m.V11E4IntegrityError):
        m.verify_result_lock_document(
            lock, authorized_commit=AUTH, sizes=SIZES, seeds=SEEDS)


def test_result_lock_semantic_verification(full_session):
    base_dir, _ = full_session
    lock = _result_lock(base_dir)
    semantic = lock["semantic_payload"]
    recomputed = m.sha256_of_canonical(semantic)
    assert recomputed == lock["semantic_result_lock_sha256"]
    m.verify_result_lock_document(
        lock, authorized_commit=AUTH, sizes=SIZES, seeds=SEEDS)

    broken = json.loads(json.dumps(lock))
    broken["semantic_payload"] = dict(broken["semantic_payload"], authorized_execution_commit="0" * 40)
    with pytest.raises(m.V11E4IntegrityError):
        m.verify_result_lock_document(
            broken, authorized_commit=AUTH, sizes=SIZES, seeds=SEEDS)


def test_output_family_uses_exact_frozen_relative_paths(full_session):
    base_dir, _ = full_session
    relative = {str(path.relative_to(base_dir)) for path in base_dir.rglob("*") if path.is_file()}
    assert (base_dir / m.WORKLOAD_MANIFEST_NAME).exists()
    assert (base_dir / m.ATTACK_MANIFEST_NAME).exists()
    assert (base_dir / m.RESULT_LOCK_NAME).exists()
    assert (base_dir / m.RAW_OBS_DIR_NAME).is_dir()
    assert (base_dir / m.RESULTS_DIR_NAME / m.RESULTS_STAGE_DIR_NAME).is_dir()
    assert (base_dir / m.SUMMARY_DIR_NAME).is_dir()
    # frozen relative layout: no artifact may live outside the five families
    for rel in relative:
        assert rel.split("/")[0] in {
            m.RAW_OBS_DIR_NAME, m.RESULTS_DIR_NAME, m.SUMMARY_DIR_NAME,
            m.ATTACK_MANIFEST_NAME, m.WORKLOAD_MANIFEST_NAME, m.RESULT_LOCK_NAME,
        }, rel
    # every measurement cell is mirrored in the raw-observations family
    stage_stems = {path.stem for path in
                   (base_dir / m.RESULTS_DIR_NAME / m.RESULTS_STAGE_DIR_NAME).glob("*.json")
                   if not path.name.startswith("E10_")}
    raw_stems = {path.stem for path in
                 (base_dir / m.RAW_OBS_DIR_NAME).rglob("*.json")}
    assert stage_stems == raw_stems
    assert len(stage_stems) == 69