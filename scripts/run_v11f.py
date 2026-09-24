#!/usr/bin/env python3
"""Governed read-only runner + persistence harness for V1.1-F (V1.1-F4).

A thin orchestration/persistence layer over the frozen read-only V1.1-F2
analysis (``src.pipeline_v11f.run_v11f_analysis``). It performs no scientific
calculation, no experiment, and no measurement of any kind: every invariant is
validated by the frozen F2 pipeline and re-validated here before any output is
persisted.

Modes:
  verify    read-only integrity/readiness verification (writes NOTHING)
  execute   requires ``--allow-governed-execution``; persists the authorized
            V1.1-F analysis outputs under results/blockchain/v11f/
            deterministically (idempotent resume; conflicting existing
            artifacts fail closed)

Authorization: ``--authorized-commit`` must equal HEAD == origin/main, and
``src/pipeline_v11f.py`` must byte-match the reviewed F2 implementation commit.

Planned F4 outputs (produced only by execute, never by verify):
  results/blockchain/v11f/v11f_analysis_results.json
  results/blockchain/v11f/v11f_execution_manifest.json
  results/blockchain/v11f/v11f_result_lock.json

Run with the repo virtualenv:
    .venv/bin/python scripts/run_v11f.py verify   --authorized-commit <sha>
    .venv/bin/python scripts/run_v11f.py execute  --authorized-commit <sha> \
        --allow-governed-execution
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.pipeline_v11f as f2

# --------------------------------------------------------------------------- #
# frozen governance identifiers (V1.1-F4-A pin set)
# --------------------------------------------------------------------------- #

STAGE = "V1.1-F4"
ANALYSIS_STAGE = "V1.1-F2"

F2_IMPLEMENTATION_COMMIT = "a33cb2fd7ceed03dde4be6094baa41089d707530"
F2_IMPLEMENTATION_RELATIVE_PATH = "src/pipeline_v11f.py"
EXPECTED_F2_SEMANTIC_ANALYSIS_SHA256 = (
    "0765ea4d79cdad956d09156983395c9c85fe748002cd91a084e2dc5ef6bf3899"
)

ANALYSIS_RESULTS_NAME = "v11f_analysis_results.json"
EXECUTION_MANIFEST_NAME = "v11f_execution_manifest.json"
RESULT_LOCK_NAME = "v11f_result_lock.json"
AUTHORIZED_OUTPUTS = (ANALYSIS_RESULTS_NAME, EXECUTION_MANIFEST_NAME, RESULT_LOCK_NAME)

ALLOWED_AUTHORIZED_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

MANIFEST_KIND = "V11F_EXECUTION_MANIFEST"
RESULT_LOCK_KIND = "V11F_RESULT_LOCK"
VERIFY_REPORT_MARKER = "V11F4A_READINESS_REPORT"
EXECUTE_REPORT_MARKER = "V11F4_EXECUTE_REPORT"

ZERO_COUNTER_KEYS = (
    "test_access_requests",
    "ai_fit_requests",
    "ai_inference_requests",
    "new_measurement_attempts",
)

# --------------------------------------------------------------------------- #
# runner errors
# --------------------------------------------------------------------------- #


class V11F4Error(Exception):
    """Base error for the V1.1-F4 runner layer."""


class V11F4AuthorizationError(V11F4Error):
    """Unauthorized execution commit, reference commit, or missing sanction."""


class V11F4IntegrityError(V11F4Error):
    """The frozen F2 semantic/reference pin or governance record drifted."""


class V11F4OutputCollisionError(V11F4Error):
    """An existing governed output artifact conflicts with the planned bytes."""


# --------------------------------------------------------------------------- #
# git / authorization gates
# --------------------------------------------------------------------------- #


def _git(rev: str, *, cwd: Path = REPO_ROOT) -> str:
    try:
        raw = subprocess.run(
            ["git", "rev-parse", rev],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as exc:
        raise V11F4AuthorizationError(
            f"git required to guard the V1.1-F4 launch checkpoint: {exc}"
        ) from exc
    if not raw:
        raise V11F4AuthorizationError(f"could not resolve git revision {rev!r}")
    return raw


def validate_authorized_commit_sha(authorized_commit: str) -> str:
    candidate = str(authorized_commit or "").strip().lower()
    if not ALLOWED_AUTHORIZED_COMMIT_RE.match(candidate):
        raise V11F4AuthorizationError(
            "authorized execution commit must be a 40-hex git object id."
        )
    return candidate


def verify_authorized_execution_commit(
    authorized_commit: str,
    *,
    require_git: bool = True,
    git_resolver: Callable[[str], str] | None = None,
) -> dict[str, str]:
    """Fail closed unless ``authorized_commit == HEAD == origin/main``.

    The dirty-worktree check is deliberately absent: unrelated protected
    worktree entries must never block a governed read-only execution.
    """
    authorized = validate_authorized_commit_sha(authorized_commit)
    resolve = git_resolver or _git
    if require_git:
        head = resolve("HEAD")
        origin_main = resolve("origin/main")
    else:
        head = authorized
        origin_main = authorized
    if head != authorized or origin_main != authorized:
        raise V11F4AuthorizationError(
            "V1.1-F4 authorization failed: HEAD/origin drifted from the "
            "supplied authorized execution commit."
        )
    return {
        "authorized_execution_commit": authorized,
        "head": head,
        "origin_main": origin_main,
    }


def verify_f2_implementation_commit(
    *,
    require_git: bool = True,
    repo_root: Path = REPO_ROOT,
) -> dict[str, str]:
    """Fail closed unless the local F2 module byte-matches the frozen F2 commit."""
    local_path = repo_root / F2_IMPLEMENTATION_RELATIVE_PATH
    local_sha = f2.sha256_file(local_path)
    if not require_git:
        return {
            "commit": F2_IMPLEMENTATION_COMMIT,
            "file_sha256": local_sha,
            "verified": str(local_sha) == str(f2.sha256_file(local_path)),
            "git_check": "skipped_no_git",
        }
    try:
        frozen_raw = subprocess.run(
            ["git", "show", f"{F2_IMPLEMENTATION_COMMIT}:{F2_IMPLEMENTATION_RELATIVE_PATH}"],
            cwd=str(repo_root),
            capture_output=True,
            check=True,
        ).stdout
    except Exception as exc:
        raise V11F4AuthorizationError(
            f"cannot resolve the frozen F2 implementation blob: {exc}"
        ) from exc
    frozen_sha = hashlib.sha256(frozen_raw).hexdigest()
    if frozen_sha != local_sha:
        raise V11F4AuthorizationError(
            "F2 implementation drift: src/pipeline_v11f.py no longer matches "
            f"the reviewed F2 implementation commit {F2_IMPLEMENTATION_COMMIT}."
        )
    return {
        "commit": F2_IMPLEMENTATION_COMMIT,
        "file_sha256": local_sha,
        "verified": True,
        "git_check": "verified",
    }


# --------------------------------------------------------------------------- #
# read-only analysis + re-validation (the F4 fail-closed ceiling)
# --------------------------------------------------------------------------- #


def run_f2_analysis_readonly(
    *,
    base_dir: Path | str = f2.V11E_DIR,
    f1_lock_path: Path | str = f2.F1_LOCK_PATH,
    f1_config_path: Path | str = f2.F1_CONFIG_PATH,
    f1_protocol_path: Path | str = f2.F1_PROTOCOL_PATH,
    v11e_binding: Mapping[str, Any] | None = None,
    expected_f2_semantic_analysis_sha256: str | None = (
        EXPECTED_F2_SEMANTIC_ANALYSIS_SHA256
    ),
    cross_check_frozen_summaries_enabled: bool = True,
) -> dict[str, Any]:
    """Invoke the frozen F2 analysis and re-validate its governed pins.

    This is the confidence ceiling of the runner: the analysis itself already
    fail-closes on F1/V1.1-E integrity, upstream fingerprints, policy/pairing,
    TEST/AI/measurement requests, direct-energy wording, ranking fields,
    memory-proxy labeling and nesting rules. The runner re-checks the material
    pins here so a future pipeline drift cannot silently reach persistence.

    When ``expected_f2_semantic_analysis_sha256`` is None (unit/CI temp-dir
    runs over synthetic evidence families) the pinned-equality gate is skipped;
    every other fail-closed gate stays active. Real-data runs always pin.
    """
    payload = f2.run_v11f_analysis(
        v11e_base_dir=base_dir,
        f1_lock_path=f1_lock_path,
        f1_config_path=f1_config_path,
        f1_protocol_path=f1_protocol_path,
        v11e_binding=v11e_binding,
        cross_check_frozen_summaries_enabled=cross_check_frozen_summaries_enabled,
    )
    semantic = payload.get("semantic_analysis_sha256")
    if (
        expected_f2_semantic_analysis_sha256 is not None
        and semantic != expected_f2_semantic_analysis_sha256
    ):
        raise V11F4IntegrityError(
            "F2 semantic analysis drift: "
            f"{semantic} != {expected_f2_semantic_analysis_sha256}."
        )
    governance = payload.get("governance_checks") or {}
    for key in ZERO_COUNTER_KEYS:
        if governance.get(key) != 0:
            raise V11F4IntegrityError(
                f"governance counter drifted from zero: {key}={governance.get(key)}."
            )
    if payload.get("direct_energy_status") != f2.ENERGY_MARKER:
        raise V11F4IntegrityError(
            "direct-energy status drifted from DIRECT_ENERGY_UNAVAILABLE."
        )
    f2.assert_no_ranking_fields(payload)
    f2.assert_memory_proxy_labeling(payload)
    f2.assert_nested_workload_safe(payload)
    return payload


def _readiness_checks(payload: Mapping[str, Any]) -> dict[str, Any]:
    governance = payload.get("governance_checks") or {}
    memory = (payload.get("metric_definitions") or {}).get("families") or {}
    return {
        "f1_protocol_lock_verified": governance.get("f1_protocol_lock_verified"),
        "f1_config_verified": governance.get("f1_config_verified"),
        "f1_protocol_document_verified": governance.get("f1_protocol_document_verified"),
        "v11e_result_lock_verified": governance.get("v11e_result_lock_verified"),
        "v11e_family_fingerprints_verified": governance.get(
            "v11e_family_fingerprints_verified"
        ),
        "v11e_e10_derived_only_verified": governance.get("v11e_e10_derived_only_verified"),
        "workload_manifest_verified": governance.get("workload_manifest_verified"),
        "policy_set_verified": governance.get("policy_set_verified"),
        "test_access_requests": governance.get("test_access_requests"),
        "ai_fit_requests": governance.get("ai_fit_requests"),
        "ai_inference_requests": governance.get("ai_inference_requests"),
        "new_measurement_attempts": governance.get("new_measurement_attempts"),
        "energy_terminology_scan": governance.get("energy_terminology_scan"),
        "memory_proxy_labeling": governance.get("memory_proxy_labeling"),
        "ranking_fields_scan": governance.get("ranking_fields_scan"),
        "nested_workload_guard": governance.get("nested_workload_guard"),
        "results_written": governance.get("results_written"),
        "e08_policy_independent": (
            (payload.get("policy_independent_storage") or {}).get("status")
            == "POLICY_INDEPENDENT"
        ),
        "e10_derived_only": (memory.get("computational_work") or {}).get(
            "source_experiment"
        )
        == "E10",
        "memory_label": (memory.get("memory") or {}).get("memory_label"),
        "e07_canonical_metric": (memory.get("memory") or {}).get("canonical_metric"),
    }


def run_readonly_verification(
    *,
    authorized_commit: str | None = None,
    base_dir: Path | str = f2.V11E_DIR,
    f1_lock_path: Path | str = f2.F1_LOCK_PATH,
    f1_config_path: Path | str = f2.F1_CONFIG_PATH,
    f1_protocol_path: Path | str = f2.F1_PROTOCOL_PATH,
    v11e_binding: Mapping[str, Any] | None = None,
    expected_f2_semantic_analysis_sha256: str | None = (
        EXPECTED_F2_SEMANTIC_ANALYSIS_SHA256
    ),
    require_git: bool = True,
    git_resolver: Callable[[str], str] | None = None,
    cross_check_frozen_summaries_enabled: bool = True,
) -> dict[str, Any]:
    """Read-only integrity/readiness report. Never writes any artifact."""
    authorization = None
    if authorized_commit is not None:
        authorization = verify_authorized_execution_commit(
            authorized_commit, require_git=require_git, git_resolver=git_resolver
        )
    payload = run_f2_analysis_readonly(
        base_dir=base_dir,
        f1_lock_path=f1_lock_path,
        f1_config_path=f1_config_path,
        f1_protocol_path=f1_protocol_path,
        v11e_binding=v11e_binding,
        expected_f2_semantic_analysis_sha256=expected_f2_semantic_analysis_sha256,
        cross_check_frozen_summaries_enabled=cross_check_frozen_summaries_enabled,
    )
    return {
        "marker": VERIFY_REPORT_MARKER,
        "mode": "verify",
        "authorization": authorization,
        "f2_semantic_analysis_sha256": payload.get("semantic_analysis_sha256"),
        "checks": _readiness_checks(payload),
        "direct_energy_status": payload.get("direct_energy_status"),
        "outputs_written": [],
    }


# --------------------------------------------------------------------------- #
# deterministic F4 persisted-output design (created only by execute)
# --------------------------------------------------------------------------- #


def build_execution_manifest(
    *,
    payload: Mapping[str, Any],
    authorized_execution_commit: str,
    f2_implementation: Mapping[str, Any],
    expected_f2_semantic_analysis_sha256: str | None = (
        EXPECTED_F2_SEMANTIC_ANALYSIS_SHA256
    ),
) -> dict[str, Any]:
    """Deterministic per-run execution/binding record (no timestamps, no paths)."""
    governance = payload.get("governance_checks") or {}
    upstream = payload.get("upstream_result_lock") or {}
    observed_semantic = payload.get("semantic_analysis_sha256")
    return {
        "artifact_kind": MANIFEST_KIND,
        "stage": STAGE,
        "analysis_stage": payload.get("stage"),
        "analysis_artifact_kind": payload.get("artifact_kind"),
        "analysis_mode": payload.get("analysis_mode"),
        "policy_set": payload.get("policy_set"),
        "authorized_execution_commit": str(authorized_execution_commit),
        "f1_protocol_lock_semantic_sha256": (
            (payload.get("protocol_lock") or {}).get("semantic_result_lock_sha256")
        ),
        "f2_implementation_commit": F2_IMPLEMENTATION_COMMIT,
        "f2_implementation_file_sha256": f2_implementation.get("file_sha256"),
        "f2_implementation_git_check": f2_implementation.get("git_check"),
        "f2_semantic_analysis_sha256": observed_semantic,
        "f2_semantic_governed_pin": expected_f2_semantic_analysis_sha256,
        "f2_semantic_gate_verified": bool(
            expected_f2_semantic_analysis_sha256 is not None
            and observed_semantic == expected_f2_semantic_analysis_sha256
        ),
        "upstream_v11e_result_lock_semantic_sha256": upstream.get(
            "semantic_result_lock_sha256"
        ),
        "upstream_v11e_result_lock_file_sha256": f2.EXPECTED_V11E_RESULT_LOCK_FILE_SHA256,
        "direct_energy_status": payload.get("direct_energy_status"),
        "governance_counters": {
            key: governance.get(key) for key in ZERO_COUNTER_KEYS
        },
        "results_written_by_analysis": governance.get("results_written"),
        "authorized_outputs": list(AUTHORIZED_OUTPUTS),
    }


def build_result_lock(
    *,
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    analysis_results_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Deterministic V1.1-F result lock (semantic lock convention)."""
    semantics = {
        "stage": "V1.1-F",
        "artifact_kind": RESULT_LOCK_KIND,
        "analysis_mode": manifest.get("analysis_mode"),
        "direct_energy_status": manifest.get("direct_energy_status"),
        "policy_set": manifest.get("policy_set"),
        "f1_protocol_lock_semantic_sha256": manifest.get(
            "f1_protocol_lock_semantic_sha256"
        ),
        "f2_implementation_commit": manifest.get("f2_implementation_commit"),
        "f2_semantic_analysis_sha256": manifest.get("f2_semantic_analysis_sha256"),
        "upstream_v11e_result_lock_semantic_sha256": manifest.get(
            "upstream_v11e_result_lock_semantic_sha256"
        ),
        "upstream_v11e_result_lock_file_sha256": manifest.get(
            "upstream_v11e_result_lock_file_sha256"
        ),
        "authorized_execution_commit": manifest.get("authorized_execution_commit"),
        "governance_counters": manifest.get("governance_counters"),
        "results_written_by_analysis": manifest.get("results_written_by_analysis"),
        "ranking_fields_present": False,
        "overall_winner_present": False,
        "composite_score_present": False,
        "authorized_outputs": {
            ANALYSIS_RESULTS_NAME: analysis_results_sha256,
            EXECUTION_MANIFEST_NAME: manifest_sha256,
        },
    }
    return {
        "artifact_kind": RESULT_LOCK_KIND,
        "stage": "V1.1-F",
        "semantic_payload": semantics,
        "semantic_result_lock_sha256": f2.sha256_of_canonical(semantics),
        "artifacts_fingerprints_sha256": {
            ANALYSIS_RESULTS_NAME: analysis_results_sha256,
            EXECUTION_MANIFEST_NAME: manifest_sha256,
        },
        "execution_metadata": {
            "protocol_only": False,
            "read_only_analysis": True,
            "governed_execute_authorization": True,
        },
    }


def _check_output_collision(
    out_dir: Path, planned: Mapping[str, bytes]
) -> dict[str, str]:
    """Fail closed when an existing output differs from the planned bytes."""
    existing: dict[str, str] = {}
    for name, planned_bytes in planned.items():
        path = out_dir / name
        if path.exists():
            current = path.read_bytes()
            if current != planned_bytes:
                raise V11F4OutputCollisionError(
                    "governed output collision: "
                    f"{path.name} exists with different content; refusing to "
                    "overwrite a possibly governed artifact."
                )
            existing[name] = f2.sha256_file(path)
    return existing


def _write_deterministic(out_dir: Path, name: str, planned_bytes: bytes) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    if path.exists() and path.read_bytes() == planned_bytes:
        return f2.sha256_file(path)
    path.write_bytes(planned_bytes)
    return hashlib.sha256(planned_bytes).hexdigest()


def persist_f4_outputs(
    *,
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    out_dir: Path | str,
) -> dict[str, Any]:
    """Persist the three authorized F4 artifacts deterministically.

    All three outputs are byte-planned first and the whole set is checked for
    conflicts before any bytes are written, so the run is all-or-nothing with
    respect to a governing collision. Identical existing bytes are left in
    place (explicit deterministic resume policy).
    """
    out_dir = Path(out_dir)
    analysis_bytes = f2.canonical_json(payload).encode("utf-8")
    manifest_bytes = f2.canonical_json(manifest).encode("utf-8")
    lock = build_result_lock(
        payload=payload,
        manifest=manifest,
        analysis_results_sha256=hashlib.sha256(analysis_bytes).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    lock_bytes = f2.canonical_json(lock).encode("utf-8")
    planned = {
        ANALYSIS_RESULTS_NAME: analysis_bytes,
        EXECUTION_MANIFEST_NAME: manifest_bytes,
        RESULT_LOCK_NAME: lock_bytes,
    }
    _check_output_collision(out_dir, planned)
    outputs = {
        name: _write_deterministic(out_dir, name, planned_bytes)
        for name, planned_bytes in planned.items()
    }
    return {
        "outputs_written": list(outputs),
        "output_fingerprints": outputs,
        "result_lock_semantic_sha256": lock["semantic_result_lock_sha256"],
    }


def run_execute(
    *,
    authorized_commit: str,
    allow_governed_execution: bool,
    base_dir: Path | str = f2.V11E_DIR,
    out_dir: Path | str = f2.V11F_DIR,
    f1_lock_path: Path | str = f2.F1_LOCK_PATH,
    f1_config_path: Path | str = f2.F1_CONFIG_PATH,
    f1_protocol_path: Path | str = f2.F1_PROTOCOL_PATH,
    v11e_binding: Mapping[str, Any] | None = None,
    expected_f2_semantic_analysis_sha256: str | None = (
        EXPECTED_F2_SEMANTIC_ANALYSIS_SHA256
    ),
    require_git: bool = True,
    git_resolver: Callable[[str], str] | None = None,
    cross_check_frozen_summaries_enabled: bool = True,
) -> dict[str, Any]:
    """Authorized persistence run. Performs no measurement and no experiment.

    Authorization order (all fail closed before any analysis or output):
    authorized-commit gate -> explicit ``allow_governed_execution`` sanction ->
    F2 implementation-commit byte-match -> frozen F2 analysis + semantic pin.
    """
    authorization = verify_authorized_execution_commit(
        authorized_commit, require_git=require_git, git_resolver=git_resolver
    )
    if not allow_governed_execution:
        raise V11F4AuthorizationError(
            "execute requires --allow-governed-execution; no governed output "
            "was produced."
        )
    f2_implementation = verify_f2_implementation_commit(require_git=require_git)
    payload = run_f2_analysis_readonly(
        base_dir=base_dir,
        f1_lock_path=f1_lock_path,
        f1_config_path=f1_config_path,
        f1_protocol_path=f1_protocol_path,
        v11e_binding=v11e_binding,
        expected_f2_semantic_analysis_sha256=expected_f2_semantic_analysis_sha256,
        cross_check_frozen_summaries_enabled=cross_check_frozen_summaries_enabled,
    )
    manifest = build_execution_manifest(
        payload=payload,
        authorized_execution_commit=authorization["authorized_execution_commit"],
        f2_implementation=f2_implementation,
        expected_f2_semantic_analysis_sha256=expected_f2_semantic_analysis_sha256,
    )
    persisted = persist_f4_outputs(payload=payload, manifest=manifest, out_dir=out_dir)
    return {
        "marker": EXECUTE_REPORT_MARKER,
        "mode": "execute",
        "authorization": authorization,
        "f2_implementation": f2_implementation,
        "f2_semantic_analysis_sha256": payload.get("semantic_analysis_sha256"),
        "checks": _readiness_checks(payload),
        "direct_energy_status": payload.get("direct_energy_status"),
        "governance_counters": manifest["governance_counters"],
        "result_lock_semantic_sha256": persisted["result_lock_semantic_sha256"],
        "outputs": {
            name: str(Path(out_dir) / name) for name in AUTHORIZED_OUTPUTS
        },
        "output_fingerprints": persisted["output_fingerprints"],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run_v11f.py",
        description="V1.1-F4 governed read-only runner + persistence harness.",
    )
    parser.add_argument(
        "mode", choices=("verify", "execute"), help="verify (read-only) or execute (persist)"
    )
    parser.add_argument(
        "--authorized-commit",
        required=True,
        help="40-hex git commit that must equal HEAD == origin/main",
    )
    parser.add_argument("--base-dir", default=str(f2.V11E_DIR))
    parser.add_argument("--out-dir", default=str(f2.V11F_DIR))
    parser.add_argument("--f1-lock-path", default=str(f2.F1_LOCK_PATH))
    parser.add_argument("--f1-config-path", default=str(f2.F1_CONFIG_PATH))
    parser.add_argument("--f1-protocol-path", default=str(f2.F1_PROTOCOL_PATH))
    parser.add_argument(
        "--allow-governed-execution",
        action="store_true",
        help="sanctioned operator path: permits persistence of the authorized "
        "F4 outputs (execute mode only)",
    )
    parser.add_argument(
        "--no-git-guard",
        action="store_true",
        help="skip git HEAD/origin + implementation-blob checks (unit/CI "
        "temp-dir runs only)",
    )
    parser.add_argument(
        "--expected-f2-semantic",
        default=EXPECTED_F2_SEMANTIC_ANALYSIS_SHA256,
        help="expected V1.1-F2 semantic_analysis_sha256",
    )
    args = parser.parse_args()
    require_git = not args.no_git_guard

    if args.mode == "verify":
        report = run_readonly_verification(
            authorized_commit=args.authorized_commit,
            base_dir=args.base_dir,
            f1_lock_path=args.f1_lock_path,
            f1_config_path=args.f1_config_path,
            f1_protocol_path=args.f1_protocol_path,
            expected_f2_semantic_analysis_sha256=args.expected_f2_semantic,
            require_git=require_git,
        )
        print(f"[v11fr] verify {report['marker']}")
        print(f"  f2 semantic: {report['f2_semantic_analysis_sha256']}")
        print(f"  checks: {report['checks']}")
        print(f"  outputs_written: {report['outputs_written']}")
        return 0

    if args.mode == "execute":
        report = run_execute(
            authorized_commit=args.authorized_commit,
            allow_governed_execution=args.allow_governed_execution,
            base_dir=args.base_dir,
            out_dir=args.out_dir,
            f1_lock_path=args.f1_lock_path,
            f1_config_path=args.f1_config_path,
            f1_protocol_path=args.f1_protocol_path,
            expected_f2_semantic_analysis_sha256=args.expected_f2_semantic,
            require_git=require_git,
        )
        print(f"[v11fr] execute {report['marker']}")
        print(f"  f2 semantic: {report['f2_semantic_analysis_sha256']}")
        print(f"  result lock: {report['result_lock_semantic_sha256']}")
        for name, path in report["outputs"].items():
            print(f"  {name}: {path}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())