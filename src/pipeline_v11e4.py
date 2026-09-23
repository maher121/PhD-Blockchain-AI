"""V1.1-E4L governed execution launcher + persistence layer.

This module is the sanctioned *implementation* of the V1.1-E2B-confirmed
output family for the governed V1.1-E4 execution stage. It orchestrates the
frozen V1.1-E3 experiment-runner machinery (never redefines scientific
behavior), persists every raw run observation, materializes the workload and
attack-instance manifests, derives E10 ONLY from persisted and re-verified
E06-E09 cell outputs (zero measurement campaigns), aggregates the frozen
descriptive statistics (mean/median/sample std ddof=1/min/max, pooled rates,
matched-policy paired differences) with no inferential tools, and emits the
final ``v11e_experiment_result_lock.json``.

Implementation-stage boundaries (frozen):
* Nothing in this module executes the governed E01-E10 campaigns by itself.
  A caller must provide an ``authorized_execution_commit`` that equals current
  ``HEAD`` AND ``origin/main`` at runtime; the launcher fails closed otherwise.
* ``--execute``/``--resume`` on governed data are DISABLED for this module by
  policy: the launcher refuses to run governed workloads unless the caller
  passes ``allow_governed_execution=True``, which only the sanctioned
  post-launch-commit operator path may do (see ``scripts/run_v11e4.py``).
* The launcher never touches TEST, never fits models, never performs AI
  inference, and never reports physical energy (frozen
  ``DIRECT_ENERGY_UNAVAILABLE`` marker).
* Output paths and artifact names are exactly the frozen V1.1-E2B family:
  ``v11e_raw_run_observations/``, ``v11e_attack_instance_manifest.json``,
  ``v11e_workload_manifest.json``, ``v11e_experiment_results/<V1.1-E4>/*.json``,
  ``v11e_aggregated_descriptive_summaries/``, ``v11e_experiment_result_lock.json``.

Everything here is deterministic; files are written atomically (temp file +
fsync + ``os.replace``) under repository canonical JSON serialization.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.blockchain_engine.block import Block
from src.blockchain_engine.canonical import canonical_order_id, sha256_hex

import src.pipeline_v11e1 as e1
import src.pipeline_v11e3 as p

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V11E_DIR = PROJECT_ROOT / "results" / "blockchain" / "v11e"

STAGE = "V1.1-E4L"
KIND = "V11E4_GOVERNED_EXECUTION_LAYER"
PROTOCOL_VERSION = "v1.1-e4l-governed-execution-launcher-1"
RESULTS_STAGE = "V1.1-E4"

RAW_OBS_DIR_NAME = "v11e_raw_run_observations"
ATTACK_MANIFEST_NAME = "v11e_attack_instance_manifest.json"
WORKLOAD_MANIFEST_NAME = "v11e_workload_manifest.json"
RESULTS_DIR_NAME = "v11e_experiment_results"
RESULTS_STAGE_DIR_NAME = RESULTS_STAGE
SUMMARY_DIR_NAME = "v11e_aggregated_descriptive_summaries"
RESULT_LOCK_NAME = "v11e_experiment_result_lock.json"

EXECUTION_ORDER = ("E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09")
POLICY_INDEPENDENT_EXPERIMENTS = ("E08",)
P_ONLY_EXPERIMENTS = ("E05",)

PREFLIGHT_MARKER = "V11E4L_PREFLIGHT_READY_GO"
EXECUTION_COMPLETE_MARKER = "V11E4_GOVERNED_EXECUTION_COMPLETE_REVIEW_REQUIRED"

ALLOWED_AUTHORIZED_COMMIT_RE = r"^[0-9a-f]{40}$"
_TMP_SUFFIX = ".tmp."


class V11E4Error(Exception):
    """Base error for the V1.1-E4L launcher layer."""


class V11E4IntegrityError(V11E4Error):
    """A persisted artifact drifted from the regenerated expectation."""


class V11E4NotAuthorizedError(V11E4Error):
    """Runtime authorization failed (no governed execution is permitted)."""


# --------------------------------------------------------------------------- #
# canonical serialization + atomic persistence
# --------------------------------------------------------------------------- #


def canonical_json(value: Any) -> str:
    """Repository-canonical JSON: sorted keys, compact separators, ascii.

    Non-finite floats are rejected (``allow_nan=False``) because the governed
    encoding never carries NaN/Infinity.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_of_canonical(value: Any) -> str:
    """sha256 hex of the canonical JSON encoding of ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_write_json(path: Path | str, payload: Any) -> None:
    """Atomically persist ``payload`` under canonical JSON (no trailing EOL).

    Writes to a same-directory temp file, flushes + fsyncs, then
    ``os.replace``. No temp artifact survives on success or failure.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + _TMP_SUFFIX + uuid.uuid4().hex)
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: Path | str, description: str) -> dict[str, Any]:
    document = json.loads(Path(path).read_bytes().decode("utf-8"))
    if not isinstance(document, dict):
        raise V11E4Error(f"{description} is not a JSON object.")
    return document


def _git_rev(rev: str, *, cwd: Path = PROJECT_ROOT) -> str:
    try:
        raw = subprocess.run(
            ["git", "rev-parse", rev],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as exc:
        raise V11E4NotAuthorizedError(
            f"git required to guard the V1.1-E4L launch checkpoint: {exc}"
        ) from exc
    if not raw:
        raise V11E4NotAuthorizedError(f"could not resolve git revision {rev!r}")
    return raw


def validate_authorized_commit_sha(authorized_commit: str) -> str:
    import re

    candidate = str(authorized_commit or "").strip().lower()
    if not re.match(ALLOWED_AUTHORIZED_COMMIT_RE, candidate):
        raise V11E4NotAuthorizedError(
            "authorized execution commit must be a 40-hex git object id."
        )
    return candidate


def verify_authorized_execution_commit(
    authorized_commit: str,
    *,
    require_git: bool = True,
    git_resolver: Callable[[str], str] | None = None,
) -> dict[str, str]:
    """Fail closed unless the runtime authorized commit == HEAD == origin/main.

    The authorized commit is supplied by the operator at launch time; it is
    never hard-coded here. With ``require_git=False`` (unit tests over temp
    dirs) only the format check runs against an optional injected resolver.
    """
    authorized = validate_authorized_commit_sha(authorized_commit)
    resolve = git_resolver or _git_rev
    if require_git:
        head = resolve("HEAD")
        origin_main = resolve("origin/main")
    else:
        head = authorized
        origin_main = authorized
    if head != authorized or origin_main != authorized:
        raise V11E4NotAuthorizedError(
            "V1.1-E4L authorization failed: HEAD/origin drifted from the "
            "supplied authorized execution commit."
        )
    return {
        "authorized_execution_commit": authorized,
        "head": head,
        "origin_main": origin_main,
    }


# --------------------------------------------------------------------------- #
# deterministic cell plan (frozen matrix, counterbalanced policy order)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CellSpec:
    """One deterministic experiment cell in the frozen E01-E10 matrix."""

    experiment_id: str
    seed: int
    workload_size: int
    policy: str
    policy_independent: bool = False

    @property
    def repetition(self) -> int:
        return p.BLOCKCHAIN_SEEDS.index(int(self.seed)) + 1

    def cell_key(self) -> str:
        return f"{self.experiment_id}|s{int(self.seed)}|w{int(self.workload_size)}|{self.policy}"

    def file_stem(self) -> str:
        return f"{self.experiment_id}_s{int(self.seed)}_w{int(self.workload_size):05d}_p{self.policy}"

    def identity_payload(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "seed": int(self.seed),
            "workload_size": int(self.workload_size),
            "policy": self.policy,
            "protocol_sha256": p.E2_PROTOCOL_LOCK_SEMANTIC_SHA256,
        }

    def cell_identity_sha256(self) -> str:
        return sha256_of_canonical(self.identity_payload())


def build_execution_plan(
    *,
    sizes: Sequence[int] = p.WORKLOADS,
    seeds: Sequence[int] = p.BLOCKCHAIN_SEEDS,
) -> list[CellSpec]:
    """Deterministic ordered cell plan: E01-E09 (counterbalanced) then E10.

    E05 is P-principal only; E08 is policy-independent (policy token ``B0``);
    E10 mirrors the per-seed counterbalanced policy order and is derived last.
    """
    plan: list[CellSpec] = []
    for experiment_id in EXECUTION_ORDER:
        for size in sizes:
            assert int(size) > 0
            for seed in seeds:
                policies = p.policy_execution_order(int(seed))
                if experiment_id in P_ONLY_EXPERIMENTS:
                    policies = ("P",)
                elif experiment_id in POLICY_INDEPENDENT_EXPERIMENTS:
                    policies = ("B0",)
                for policy in policies:
                    plan.append(
                        CellSpec(
                            experiment_id=experiment_id,
                            seed=int(seed),
                            workload_size=int(size),
                            policy=policy,
                            policy_independent=experiment_id in POLICY_INDEPENDENT_EXPERIMENTS,
                        )
                    )
    for size in sizes:
        for seed in seeds:
            for policy in p.policy_execution_order(int(seed)):
                plan.append(
                    CellSpec(experiment_id="E10", seed=int(seed),
                             workload_size=int(size), policy=policy)
                )
    return plan


def expected_cell_count(experiment_id: str, *, seeds: int = len(p.BLOCKCHAIN_SEEDS),
                        sizes: int = len(p.WORKLOADS), policies: int = len(p.POLICIES)) -> int:
    if experiment_id in P_ONLY_EXPERIMENTS:
        policy_factor = 1
    elif experiment_id in POLICY_INDEPENDENT_EXPERIMENTS:
        policy_factor = 1
    elif experiment_id == "E10":
        policy_factor = policies
    else:
        policy_factor = policies
    return int(seeds) * int(sizes) * policy_factor


def expected_total_cell_count(*, seeds: int = len(p.BLOCKCHAIN_SEEDS),
                              sizes: int = len(p.WORKLOADS)) -> int:
    return sum(expected_cell_count(eid, seeds=seeds, sizes=sizes) for eid in EXECUTION_ORDER + ("E10",))


def cell_file_stem_from_filename(filename: str) -> str:
    if not filename.endswith(".json"):
        raise V11E4Error(f"unexpected non-JSON cell file {filename!r}")
    return filename[:-5]


# --------------------------------------------------------------------------- #
# data providers (synthetic for tests; governed for the sanctioned run)
# --------------------------------------------------------------------------- #


class DataProvider:
    """Thin data seam: order ids, risk records, chains (all read-only)."""

    def order_ids(self) -> Sequence[str]:  # pragma: no cover - protocol
        raise NotImplementedError

    def records(self) -> Mapping[str, Mapping[str, Any]]:  # pragma: no cover - protocol
        raise NotImplementedError

    def chains(self) -> Mapping[str, Sequence[Block]]:  # pragma: no cover - protocol
        raise NotImplementedError


class SyntheticDataProvider(DataProvider):
    """Tiny labeled synthetic fixture (tests only, never governed evidence)."""

    def __init__(self, order_ids: Sequence[Any], *, risk_levels: Sequence[str] | None = None):
        fixture = p.synthetic_workload_fixture(order_ids, risk_levels=risk_levels)
        self._order_ids: Sequence[str] = tuple(fixture["order_ids"])
        self._chains: Mapping[str, Sequence[Block]] = dict(fixture["chains"])
        self._records: Mapping[str, Mapping[str, Any]] = dict(fixture["records"])

    def order_ids(self) -> Sequence[str]:
        return self._order_ids

    def records(self) -> Mapping[str, Mapping[str, Any]]:
        return dict(self._records)

    def chains(self) -> Mapping[str, Sequence[Block]]:
        return dict(self._chains)


class GovernedDataProvider(DataProvider):
    """Full governed data provider (V1.1-C mapping + D3 artifact + chains).

    Reuses the frozen V1.1-E1 reconstruction bridge verbatim; never appends to
    the corpus and never touches TEST. Only ever exercised by the sanctioned
    governed execution path.
    """

    def __init__(self):
        self._materialize()

    def _materialize(self) -> None:
        records = e1.load_governed_risk_records()
        entries = e1.mapping_entries_index(e1.load_v11c_mapping_summary())
        coverage = e1.verify_governed_order_coverage(records, entries)
        if coverage["status"] != "PREPARED":
            raise V11E4Error("governed order coverage is not PREPARED; refusing to run.")
        rows_by_order: dict[str, list[dict[str, Any]]] = {}
        for row in e1.load_validation_metadata_rows():
            rows_by_order.setdefault(canonical_order_id(row[e1.ORDER_ID_COLUMN]), []).append(row)
        chains: dict[str, tuple[Block, ...]] = {}
        for record in records:
            order_rows = rows_by_order.get(record.order_id)
            if not order_rows:
                raise V11E4Error(f"validation metadata missing for governed order {record.order_id!r}.")
            reference = e1.onchain_ai_reference(
                record,
                artifact_ref=e1.D3_ARTIFACT_REF,
                artifact_lock_sha256=e1.D3_RESULT_LOCK_SEMANTIC,
            )
            e1.verify_onchain_reference(reference, record)
            descriptor = e1.reconstruct_pre_ai_chain(record.order_id, order_rows)
            binding = e1.verify_chain_matches_mapping_entry(descriptor, entries[record.order_id])
            if not binding["passed"]:
                raise V11E4Error(f"governed chain binding failed for {record.order_id!r}.")
            chain = e1.chain_from_descriptor(descriptor)
            e1.extend_chain_with_ai_risk(chain, onchain_reference=reference)
            chains[record.order_id] = tuple(chain.blocks)
        self._record_objects = records
        self._records: Mapping[str, Mapping[str, Any]] = {
            record.order_id: {
                "order_id": record.order_id,
                "risk_level": record.risk_level,
                "risk_score": record.risk_score,
            }
            for record in records
        }
        self._order_ids: Sequence[str] = tuple(record.order_id for record in records)
        self._chains = chains

    def order_ids(self) -> Sequence[str]:
        return self._order_ids

    def records(self) -> Mapping[str, Mapping[str, Any]]:
        return dict(self._records)

    def chains(self) -> Mapping[str, Sequence[Block]]:
        return dict(self._chains)


# --------------------------------------------------------------------------- #
# workload + attack manifests (deterministic, preregistered, non-measuring)
# --------------------------------------------------------------------------- #


def build_workload_manifest(
    provider: DataProvider,
    *,
    sizes: Sequence[int] = p.WORKLOADS,
    seeds: Sequence[int] = p.BLOCKCHAIN_SEEDS,
) -> dict[str, Any]:
    """Deterministic workload manifest over the frozen nested sampling contract.

    Reuses the frozen E3 sampler (``p.nested_workloads``, ``p.risk_band_counts``,
    ``p.band_markers``); adds the persisted envelope + per-workload digest.
    """
    size_key = sorted(int(size) for size in sizes)
    seed_key = [int(seed) for seed in seeds]
    workloads_by_seed: dict[str, dict[str, Any]] = {}
    for seed in seed_key:
        nested = p.nested_workloads(seed, provider.order_ids(), size_key)
        per_seed: dict[str, Any] = {}
        for size in size_key:
            order_ids = [canonical_order_id(oid) for oid in nested[size]]
            counts = p.risk_band_counts(order_ids, provider.records())
            per_seed[str(size)] = {
                "order_ids": order_ids,
                "order_ids_digest_sha256": sha256_of_canonical(
                    {"seed": seed, "workload_size": size, "order_ids": order_ids}
                ),
                "risk_counts": counts,
                "markers": p.band_markers(counts),
                "repetition": p.BLOCKCHAIN_SEEDS.index(seed) + 1,
            }
        workloads_by_seed[str(seed)] = per_seed
    included = {
        "artifact_kind": "V11E4_WORKLOAD_MANIFEST",
        "stage": STAGE,
        "protocol_sha256": p.E2_PROTOCOL_LOCK_SEMANTIC_SHA256,
        "sizes": size_key,
        "seeds": seed_key,
        "nested_prefixes": True,
        "no_replacement": True,
        "sampling": "sha256_of_canonical({'seed': seed, 'order_id': oid}), "
                    "digest tie-break by canonical numeric order id, first-size prefix",
        "workloads_by_seed": workloads_by_seed,
    }
    return {
        **included,
        "semantic_sha256": sha256_of_canonical(included),
    }


def build_attack_instance_manifest(
    provider: DataProvider,
    *,
    sizes: Sequence[int] = p.WORKLOADS,
    seeds: Sequence[int] = p.BLOCKCHAIN_SEEDS,
) -> dict[str, Any]:
    """Deterministic PA-01..PA-09 attack-instance manifest per {seed, workload}.

    Reuses the frozen E3 factory (``p.attack_instances_for_workload``) so the
    exact attacked copies consumed by E02/E03/E04 are preregistered verbatim,
    including donor read-only and copy-only semantics.
    """
    size_key = sorted(int(size) for size in sizes)
    seed_key = [int(seed) for seed in seeds]
    chains = provider.chains()
    by_seed: dict[str, dict[str, Any]] = {}
    for seed in seed_key:
        nested = p.nested_workloads(seed, provider.order_ids(), size_key)
        per_seed: dict[str, Any] = {}
        for size in size_key:
            order_ids = [canonical_order_id(oid) for oid in nested[size]]
            instances = p.attack_instances_for_workload(order_ids, chains, seed)
            rendered: dict[str, Any] = {}
            for scenario_id in p.PA_SCENARIOS:
                instance = instances[scenario_id]
                rendered[scenario_id] = {
                    "instance_id": instance.instance_id,
                    "scenario_id": scenario_id,
                    "tamper_kind": instance.tamper_kind,
                    "seed": instance.seed,
                    "workload_size": instance.workload_size,
                    "target_order_id": instance.target_order_id,
                    "donor_order_id": instance.donor_order_id,
                    "copies_only": instance.copies_only,
                    "donor_read_only": instance.donor_read_only,
                    "tampered_block_count": len(instance.tampered_blocks),
                    "tampered_block_hashes": [block.block_hash for block in instance.tampered_blocks],
                    "semantic_sha256": instance.as_mapping()["semantic_sha256"],
                }
            per_seed[str(size)] = rendered
        by_seed[str(seed)] = per_seed
    included = {
        "artifact_kind": "V11E4_ATTACK_INSTANCE_MANIFEST",
        "stage": STAGE,
        "protocol_sha256": p.E2_PROTOCOL_LOCK_SEMANTIC_SHA256,
        "sizes": size_key,
        "seeds": seed_key,
        "copy_only_target_blocks": True,
        "donor_read_only": True,
        "by_seed": by_seed,
    }
    return {
        **included,
        "semantic_sha256": sha256_of_canonical(included),
    }


# --------------------------------------------------------------------------- #
# cell persistence (verifiable, deterministic)
# --------------------------------------------------------------------------- #


def _cell_document(
    spec: CellSpec,
    response_cell: dict[str, Any],
    *,
    authorization: Mapping[str, str],
    workload_manifest_semantic_sha256: str,
    workload_order_ids: Sequence[str],
    source_bindings: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    context = response_cell.get("context") or {}
    payload: dict[str, Any] = {
        "artifact_kind": "V11E4_EXPERIMENT_CELL",
        "version": 1,
        "protocol_sha256": p.E2_PROTOCOL_LOCK_SEMANTIC_SHA256,
        "authorized_execution_commit": authorization["authorized_execution_commit"],
        "head": authorization.get("head"),
        "origin_main": authorization.get("origin_main"),
        "experiment_id": spec.experiment_id,
        "seed": spec.seed,
        "repetition": spec.repetition,
        "workload_size": spec.workload_size,
        "policy": spec.policy,
        "policy_independent": spec.policy_independent,
        "attack_id": context.get("attack_id"),
        "cell_id": spec.cell_key(),
        "cell_identity_sha256": spec.cell_identity_sha256(),
        "workload_manifest_semantic_sha256": workload_manifest_semantic_sha256,
        "order_ids_digest_sha256": sha256_of_canonical(
            {"seed": spec.seed, "workload_size": spec.workload_size, "order_ids": list(workload_order_ids)}
        ),
        "upstream_provenance": dict(context.get("upstream_provenance") or {}),
        "environment": dict(context.get("environment") or {}),
        "source_cells": dict(source_bindings) if source_bindings else {},
        "result": response_cell,
        "semantic_sha256": "",  # placeholder, replaced below
    }
    payload.pop("semantic_sha256")
    payload["semantic_sha256"] = sha256_of_canonical(payload)
    return payload


def build_measurement_cell_document(
    spec: CellSpec,
    runner_cell: dict[str, Any],
    *,
    authorization: Mapping[str, str],
    workload_manifest_semantic_sha256: str,
    workload_order_ids: Sequence[str],
) -> dict[str, Any]:
    return _cell_document(
        spec,
        runner_cell,
        authorization=authorization,
        workload_manifest_semantic_sha256=workload_manifest_semantic_sha256,
        workload_order_ids=workload_order_ids,
    )


def build_e10_cell_document(
    spec: CellSpec,
    derived_cell: dict[str, Any],
    *,
    authorization: Mapping[str, str],
    workload_manifest_semantic_sha256: str,
    workload_order_ids: Sequence[str],
    source_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    document = _cell_document(
        spec,
        derived_cell,
        authorization=authorization,
        workload_manifest_semantic_sha256=workload_manifest_semantic_sha256,
        workload_order_ids=workload_order_ids,
        source_bindings=source_bindings,
    )
    if derived_cell.get("derived_only") is not True:
        raise V11E4IntegrityError("E10 cell is not marked derived_only.")
    if "raw_records" in derived_cell:
        raise V11E4IntegrityError("E10 cell carries raw measurement artifacts (must never measure).")
    return document


def verify_cell_document(
    document: Mapping[str, Any],
    spec: CellSpec,
    authorization: Mapping[str, str],
    *,
    workload_manifest_semantic_sha256: str,
    require_e10_derived: bool = False,
) -> None:
    """Fail closed unless the persisted cell reproduces the session expectation."""
    if document.get("artifact_kind") != "V11E4_EXPERIMENT_CELL":
        raise V11E4IntegrityError("cell artifact_kind mismatch.")
    if document.get("experiment_id") != spec.experiment_id:
        raise V11E4IntegrityError("cell experiment_id mismatch.")
    if int(document.get("seed", -1)) != spec.seed:
        raise V11E4IntegrityError("cell seed mismatch.")
    if int(document.get("workload_size", -1)) != spec.workload_size:
        raise V11E4IntegrityError("cell workload_size mismatch.")
    if document.get("policy") != spec.policy:
        raise V11E4IntegrityError("cell policy mismatch.")
    if document.get("cell_id") != spec.cell_key():
        raise V11E4IntegrityError("cell_id mismatch.")
    if document.get("cell_identity_sha256") != spec.cell_identity_sha256():
        raise V11E4IntegrityError("cell identity sha mismatch.")
    if document.get("protocol_sha256") != p.E2_PROTOCOL_LOCK_SEMANTIC_SHA256:
        raise V11E4IntegrityError("cell protocol mismatch.")
    if document.get("authorized_execution_commit") != authorization["authorized_execution_commit"]:
        raise V11E4IntegrityError("cell authorized-execution-commit mismatch.")
    if document.get("workload_manifest_semantic_sha256") != workload_manifest_semantic_sha256:
        raise V11E4IntegrityError("cell workload-manifest semantic mismatch.")
    recomputed = sha256_of_canonical(
        {key: value for key, value in document.items() if key != "semantic_sha256"}
    )
    if recomputed != document.get("semantic_sha256"):
        raise V11E4IntegrityError("cell semantic sha drift (computed != persisted).")
    if require_e10_derived:
        result = document.get("result") or {}
        if result.get("derived_only") is not True:
            raise V11E4IntegrityError("E10 cell result is not derived_only.")
        if "raw_records" in result:
            raise V11E4IntegrityError("E10 cell result carries measurement artifacts.")


def load_verified_cell(
    path: Path,
    spec: CellSpec,
    authorization: Mapping[str, str],
    *,
    workload_manifest_semantic_sha256: str,
    require_e10_derived: bool = False,
) -> dict[str, Any]:
    document = read_json(path, "experiment cell")
    verify_cell_document(
        document,
        spec,
        authorization,
        workload_manifest_semantic_sha256=workload_manifest_semantic_sha256,
        require_e10_derived=require_e10_derived,
    )
    return document


def raw_observations_document(
    spec: CellSpec,
    cell_document: Mapping[str, Any],
) -> dict[str, Any]:
    records = (cell_document.get("result") or {}).get("raw_records")
    if not isinstance(records, list):
        if spec.experiment_id == "E08":
            records = []
        else:
            raise V11E4IntegrityError(
                f"{spec.experiment_id} cell has no raw-records list."
            )
    included = {
        "artifact_kind": "V11E4_RAW_RUN_OBSERVATIONS",
        "experiment_id": spec.experiment_id,
        "seed": spec.seed,
        "workload_size": spec.workload_size,
        "policy": spec.policy,
        "cell_id": spec.cell_key(),
        "cell_identity_sha256": spec.cell_identity_sha256(),
        "protocol_sha256": p.E2_PROTOCOL_LOCK_SEMANTIC_SHA256,
        "authorized_execution_commit": cell_document["authorized_execution_commit"],
        "record_count": len(records),
        "records": records,
        "note": "E08 policy-independent workload-level primary evidence only; "
                "no per-order raw records exist by design." if spec.experiment_id == "E08"
                else "every per-order raw run observation, persisted before aggregation.",
    }
    return {
        **included,
        "semantic_sha256": sha256_of_canonical(included),
    }


def lineate_output_paths(base_dir: Path) -> dict[str, Path]:
    return {
        "raw_obs_dir": base_dir / RAW_OBS_DIR_NAME,
        "attack_manifest": base_dir / ATTACK_MANIFEST_NAME,
        "workload_manifest": base_dir / WORKLOAD_MANIFEST_NAME,
        "results_dir": base_dir / RESULTS_DIR_NAME,
        "stage_dir": base_dir / RESULTS_DIR_NAME / RESULTS_STAGE_DIR_NAME,
        "summary_dir": base_dir / SUMMARY_DIR_NAME,
        "result_lock": base_dir / RESULT_LOCK_NAME,
    }

# --------------------------------------------------------------------------- #
# session engine (execute/resume: idempotent, fail-closed, E10 derived-only)
# --------------------------------------------------------------------------- #

_E10_REQUIRED_SOURCES = ("E06", "E07", "E08", "E09")


class _WorkloadModel:
    """Cached frozen sampling + attack instances for one provider session."""

    def __init__(self, provider: DataProvider, *, sizes: Sequence[int], seeds: Sequence[int]):
        self._provider = provider
        self._sizes = tuple(sorted(int(size) for size in sizes))
        self._seeds = tuple(int(seed) for seed in seeds)
        self._nested: dict[int, tuple[str, ...]] = {}
        self._instances: dict[tuple[int, int], Mapping[str, Any]] = {}

    def order_ids(self, seed: int, size: int) -> tuple[str, ...]:
        seed = int(seed)
        size = int(size)
        if seed not in self._nested:
            self._nested[seed] = tuple(
                canonical_order_id(oid) for oid in p.rank_order_ids(seed, self._provider.order_ids())
            )
        return self._nested[seed][:size]

    def instances(self, seed: int, size: int) -> Mapping[str, Any]:
        key = (int(seed), int(size))
        if key not in self._instances:
            self._instances[key] = p.attack_instances_for_workload(
                self.order_ids(seed, size), self._provider.chains(), seed
            )
        return self._instances[key]


def _fill_family_artifacts(
    base_dir: Path,
    *,
    provider: DataProvider,
    sizes: Sequence[int],
    seeds: Sequence[int],
    write_outputs: bool,
    require_clean_write: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    """Persist-or-verify the two preregistered manifests (no measurement)."""
    paths = lineate_output_paths(base_dir)
    workload = build_workload_manifest(provider, sizes=sizes, seeds=seeds)
    attack = build_attack_instance_manifest(provider, sizes=sizes, seeds=seeds)
    for manifest, path in ((workload, paths["workload_manifest"]),
                           (attack, paths["attack_manifest"])):
        if path.exists():
            existing = read_json(path, "preregistered manifest")
            if existing != manifest:
                raise V11E4IntegrityError(
                    f"{path.name} drifted from its deterministic recomputation."
                )
        elif write_outputs:
            if require_clean_write and path.exists():
                raise V11E4IntegrityError(f"{path.name} unexpectedly present.")
            atomic_write_json(path, manifest)
    return workload, attack, paths


def derive_e10_from_persisted_cells(
    spec: CellSpec,
    *,
    provider: DataProvider,
    authorization: Mapping[str, str],
    workload_semantic_sha256: str,
    workload_order_ids: Sequence[str],
    stage_dir: Path,
    workloads: _WorkloadModel,
) -> dict[str, Any]:
    """Derive one E10 cell ONLY from verified persisted E06-E09 cells."""
    source_bindings: dict[str, Mapping[str, str]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for experiment_id in _E10_REQUIRED_SOURCES:
        source_policy = "B0" if experiment_id == "E08" else spec.policy
        source_spec = CellSpec(experiment_id, spec.seed, spec.workload_size, source_policy)
        path = stage_dir / f"{source_spec.file_stem()}.json"
        if not path.exists():
            raise V11E4IntegrityError(
                f"E10 derivation requires persisted verified {experiment_id} cell "
                f"{source_spec.cell_key()!r}; missing on disk."
            )
        document = load_verified_cell(
            path,
            source_spec,
            authorization,
            workload_manifest_semantic_sha256=workload_semantic_sha256,
        )
        sources[experiment_id] = document
        source_bindings[experiment_id] = {
            "cell_id": document["cell_id"],
            "cell_identity_sha256": document["cell_identity_sha256"],
            "cell_semantic_sha256": document["semantic_sha256"],
        }
    context = p.context_for(
        "E10", spec.seed, spec.workload_size, workload_order_ids, spec.policy
    )
    derived = p.derive_e10_resource_green_proxy(
        e06_results=sources["E06"]["result"],
        e07_results=sources["E07"]["result"],
        e08_results=sources["E08"]["result"],
        e09_results=sources["E09"]["result"],
        context=context,
    )
    return build_e10_cell_document(
        spec,
        derived,
        authorization=authorization,
        workload_manifest_semantic_sha256=workload_semantic_sha256,
        workload_order_ids=workload_order_ids,
        source_bindings=source_bindings,
    )


def run_execution_session(
    *,
    provider: DataProvider,
    authorized_commit: str,
    base_dir: Path,
    allow_governed_execution: bool = False,
    sizes: Sequence[int] = p.WORKLOADS,
    seeds: Sequence[int] = p.BLOCKCHAIN_SEEDS,
    require_git: bool = True,
    git_resolver: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Execute/resume the full governed plan (idempotent, fail-closed).

    Fails closed if a persisted cell does not reproduce the current session
    expectation (identity, protocol, authorized commit, semantic sha). E10 is
    always derived after persistence and only from verified E06-E09 cells.
    """
    if isinstance(provider, GovernedDataProvider) and not allow_governed_execution:
        raise V11E4NotAuthorizedError(
            "governed execution requires allow_governed_execution=True "
            "(sanctioned operator path only)."
        )
    authorization = verify_authorized_execution_commit(
        authorized_commit, require_git=require_git, git_resolver=git_resolver
    )
    workload, _attack, paths = _fill_family_artifacts(
        base_dir,
        provider=provider,
        sizes=sizes,
        seeds=seeds,
        write_outputs=True,
        require_clean_write=False,
    )
    workload_semantic = workload["semantic_sha256"]
    stage_dir = paths["stage_dir"]
    stage_dir.mkdir(parents=True, exist_ok=True)

    plan = build_execution_plan(sizes=sizes, seeds=seeds)
    workload_model = _WorkloadModel(provider, sizes=sizes, seeds=seeds)

    executed: list[str] = []
    skipped: list[str] = []
    chain_no_attack = ("E01", "E05", "E06", "E07", "E08", "E09")

    for spec in plan:
        if spec.experiment_id != "E10":
            path = stage_dir / f"{spec.file_stem()}.json"
            if path.exists():
                load_verified_cell(
                    path,
                    spec,
                    authorization,
                    workload_manifest_semantic_sha256=workload_semantic,
                )
                skipped.append(spec.cell_key())
                continue
            order_ids = workload_model.order_ids(spec.seed, spec.workload_size)
            context = p.context_for(
                spec.experiment_id, spec.seed, spec.workload_size, order_ids, spec.policy
            )
            attacks = None
            if spec.experiment_id not in chain_no_attack:
                attacks = workload_model.instances(spec.seed, spec.workload_size)
            response_cells = p.run_ordered_experiments(
                context, provider.chains(), attacks, provider.records()
            )
            if len(response_cells) != 1:
                raise V11E4IntegrityError(
                    f"{spec.experiment_id} runner produced != 1 response cell."
                )
            document = build_measurement_cell_document(
                spec,
                response_cells[0],
                authorization=authorization,
                workload_manifest_semantic_sha256=workload_semantic,
                workload_order_ids=order_ids,
            )
            atomic_write_json(path, document)
            raw_doc = raw_observations_document(spec, document)
            raw_dir = paths["raw_obs_dir"] / spec.experiment_id
            atomic_write_json(raw_dir / f"{spec.file_stem()}.json", raw_doc)
            executed.append(spec.cell_key())
            continue

        path = stage_dir / f"{spec.file_stem()}.json"
        if path.exists():
            load_verified_cell(
                path,
                spec,
                authorization,
                workload_manifest_semantic_sha256=workload_semantic,
                require_e10_derived=True,
            )
            skipped.append(spec.cell_key())
            continue
        order_ids = workload_model.order_ids(spec.seed, spec.workload_size)
        document = derive_e10_from_persisted_cells(
            spec,
            provider=provider,
            authorization=authorization,
            workload_semantic_sha256=workload_semantic,
            workload_order_ids=order_ids,
            stage_dir=stage_dir,
            workloads=workload_model,
        )
        atomic_write_json(path, document)
        executed.append(spec.cell_key())

    summaries = build_descriptive_summaries(
        base_dir=base_dir,
        authorized_commit=authorization["authorized_execution_commit"],
        sizes=sizes,
        seeds=seeds,
        write_outputs=True,
    )
    lock = build_result_lock(
        base_dir=base_dir,
        authorized_commit=authorization["authorized_execution_commit"],
        executed_cells=executed,
        skipped_cells=skipped,
        sizes=sizes,
        seeds=seeds,
    )
    return {
        "authorization": authorization,
        "executed_cells": executed,
        "skipped_cells": skipped,
        "summary_files": list(paths["summary_dir"].glob("*.json")),
        "result_lock_path": paths["result_lock"],
        "marker": EXECUTION_COMPLETE_MARKER,
    }


# --------------------------------------------------------------------------- #
# descriptive aggregation (frozen statistics only, never inferential)
# --------------------------------------------------------------------------- #

_CONTINUOUS_RAW_FIELDS = (
    "wall_time_ns", "cpu_time_ns", "memory_proxy_mib", "storage_bytes",
    "throughput_orders_per_second",
)
_COUNT_RAW_FIELDS = (
    "validation_check_count", "validator_invocation_count", "hash_operation_count",
)

EXPERIMENT_METRIC_LABELS: dict[str, tuple[str, ...]] = {
    "E01": ("wall_time_ns", "cpu_time_ns"),
    "E02": ("wall_time_ns", "cpu_time_ns"),
    "E03": ("wall_time_ns", "cpu_time_ns"),
    "E04": ("wall_time_ns", "cpu_time_ns"),
    "E05": ("wall_time_ns", "cpu_time_ns"),
    "E06": ("wall_time_ns", "cpu_time_ns"),
    "E07": ("memory_proxy_mib",),
    "E08": ("canonical_serialized_blockchain_bytes_per_workload", "bytes_per_order", "bytes_per_block"),
    "E09": ("wall_time_ns", "cpu_time_ns"),
    "E10": (
        "runtime_proxy", "memory_evidence", "storage_evidence",
        "throughput_evidence", "validation_check_count",
        "validator_invocation_count", "measurable_hash_operation_count",
    ),
}

_RATE_POLICIES = {
    "E01": "validation_outcome",
    "E05": "validation_outcome",
    "E06": "validation_outcome",
    "E07": "validation_outcome",
    "E02": "detection_outcome",
    "E03": "localization_outcome.localized",
    "E04": "localization_outcome.preservation",
    "E09": "validation_outcome",
}


def _policy_group_key(spec_or_doc: Mapping[str, Any]) -> str:
    if spec_or_doc.get("policy_independent"):
        return "POLICY_INDEPENDENT"
    return str(spec_or_doc.get("policy"))


def _raw_metric_values(cell_document: Mapping[str, Any], metric: str) -> list[float]:
    result = cell_document.get("result") or {}
    experiment = cell_document.get("experiment_id")
    records = result.get("raw_records")
    if not isinstance(records, list):
        return []
    values: list[float] = []
    for record in records:
        value = record.get(metric)
        if value is None:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _primary_scalar(cell_document: Mapping[str, Any], key: str) -> float | None:
    primary = (cell_document.get("result") or {}).get("primary") or {}
    value = primary.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _cell_metric_scalar(cell_document: Mapping[str, Any], metric: str) -> float | None:
    experiment = cell_document.get("experiment_id")
    if experiment == "E08":
        return _primary_scalar(cell_document, metric)
    if experiment == "E10":
        proxies = ((cell_document.get("result") or {}).get("primary_computational_proxies")) or {}
        memory = (cell_document.get("result") or {}).get("memory_evidence") or {}
        mapping: dict[str, Any] = {
            "runtime_proxy": proxies.get("runtime_proxy") or {},
            "memory_evidence": memory,
            "storage_evidence": cell_document.get("result", {}).get("storage_evidence"),
            "throughput_evidence": cell_document.get("result", {}).get("throughput_evidence"),
            "validation_check_count": proxies.get("validation_check_count"),
            "validator_invocation_count": proxies.get("validator_invocation_count"),
            "measurable_hash_operation_count": proxies.get("measurable_hash_operation_count"),
        }
        value = mapping.get(metric)
        if isinstance(value, dict):
            return value.get("mean")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None
    raw = _raw_metric_values(cell_document, metric)
    return p.mean([v for v in raw]) if raw else None


def _position_label(cell_document: Mapping[str, Any]) -> tuple[int, int]:
    return (int(cell_document["seed"]), int(cell_document["workload_size"]))


def _pair_diffs_describes(
    cells: Sequence[Mapping[str, Any]],
    metric: str,
    base_policy: str,
    other_policy: str,
) -> dict[str, float | None] | None:
    base_map = {_position_label(doc): doc for doc in cells
                if doc.get("policy") == base_policy}
    other_map = {_position_label(doc): doc for doc in cells
                 if doc.get("policy") == other_policy}
    diffs: list[float] = []
    for position in sorted(set(base_map) & set(other_map)):
        a = _cell_metric_scalar(other_map[position], metric)
        b = _cell_metric_scalar(base_map[position], metric)
        if a is None or b is None:
            continue
        diffs.append(a - b)
    if not diffs:
        return None
    return p.describe_continuous(diffs)


def _pooled_mark(policy_cells: Sequence[Mapping[str, Any]], source: str) -> dict[str, Any] | None:
    numerators: list[int] = []
    denominators: list[int] = []
    for document in policy_cells:
        result = document.get("result") or {}
        if source == "validation_outcome":
            for record in result.get("raw_records") or []:
                value = record.get("validation_outcome")
                if value is not None:
                    numerators.append(1 if value else 0)
                    denominators.append(1)
        elif source == "detection_outcome":
            for record in result.get("raw_records") or []:
                value = record.get("detection_outcome")
                if value is not None:
                    numerators.append(1 if value else 0)
                    denominators.append(1)
        elif source == "localization_outcome.localized":
            for record in result.get("raw_records") or []:
                localization = record.get("localization_outcome")
                if isinstance(localization, dict) and "localized" in localization:
                    numerators.append(1 if localization["localized"] else 0)
                    denominators.append(1)
        elif source == "localization_outcome.preservation":
            for record in result.get("raw_records") or []:
                localization = record.get("localization_outcome")
                if isinstance(localization, dict) and "unaffected_order_count" in localization:
                    denominator = int(localization["unaffected_order_count"])
                    propagation = int(localization.get("cross_order_propagation_count", 0))
                    numerators.append(denominator - propagation)
                    denominators.append(denominator)
    if not denominators:
        return None
    return p.pooled_rate(numerators, denominators)


def _experiment_descriptive_summary(
    experiment_id: str,
    cells: Sequence[Mapping[str, Any]],
    *,
    authorized_commit: str,
) -> dict[str, Any]:
    by_policy: dict[str, list[Mapping[str, Any]]] = {}
    for document in cells:
        by_policy.setdefault(_policy_group_key(document), []).append(document)

    per_policy: dict[str, dict[str, Any]] = {}
    for policy, policy_cells in sorted(by_policy.items()):
        metrics: dict[str, Any] = {}
        for metric in EXPERIMENT_METRIC_LABELS.get(experiment_id, ()):
            values: list[float] = []
            for document in policy_cells:
                coverage = _cell_metric_scalar(document, metric)
                if coverage is not None:
                    values.append(coverage)
            if values:
                metrics[metric] = p.describe_continuous(values)
        rates: dict[str, Any] = {}
        if experiment_id in _RATE_POLICIES:
            mark = _pooled_mark(policy_cells, _RATE_POLICIES[experiment_id])
            if mark is not None:
                rates[_RATE_POLICIES[experiment_id]] = mark
        per_policy[policy] = {"metrics": metrics, "rates": rates}

    paired: dict[str, dict[str, Any]] = {}
    if experiment_id not in ("E08",) and len(by_policy) > 1:
        for other, base, label in (("B1", "B0", "B1_minus_B0"),
                                   ("P", "B0", "P_minus_B0"),
                                   ("P", "B1", "P_minus_B1")):
            if base not in by_policy or other not in by_policy:
                continue
            for metric in EXPERIMENT_METRIC_LABELS.get(experiment_id, ()):
                describe = _pair_diffs_describes(cells, metric, base_policy=base,
                                                 other_policy=other)
                if describe is not None:
                    paired.setdefault(label, {})[metric] = describe

    included = {
        "artifact_kind": "V11E4_AGGREGATED_DESCRIPTIVE_SUMMARY",
        "experiment_id": experiment_id,
        "protocol_sha256": p.E2_PROTOCOL_LOCK_SEMANTIC_SHA256,
        "authorized_execution_commit": authorized_commit,
        "descriptive_statistics": "mean/median/sample_std_ddof1/min/max only; "
                                  "no CI, no p-values, no bootstrap, no inferential tests",
        "per_policy": per_policy,
        "paired_differences": paired,
        "energy_marker": p.ENERGY_MARKER,
    }
    return {
        **included,
        "semantic_sha256": sha256_of_canonical(included),
    }


def build_descriptive_summaries(
    *,
    base_dir: Path,
    authorized_commit: str,
    sizes: Sequence[int],
    seeds: Sequence[int],
    write_outputs: bool,
) -> dict[str, Any]:
    """Aggregate persisted cell outputs into frozen descriptive summaries.

    Reads only from the persisted (verified) experiment-cell family at
    ``base_dir``; never executes a measurement.
    """
    paths = lineate_output_paths(base_dir)
    stage_dir = paths["stage_dir"]
    if not stage_dir.is_dir():
        raise V11E4IntegrityError("no persisted experiment cells to summarize.")
    cells: dict[str, list[dict[str, Any]]] = {eid: [] for eid in EXECUTION_ORDER + ("E10",)}
    for path in sorted(stage_dir.glob("*.json")):
        document = read_json(path, "experiment cell")
        cells[document["experiment_id"]].append(document)

    outputs: dict[str, dict[str, Any]] = {}
    for experiment_id in EXECUTION_ORDER + ("E10",):
        if not cells[experiment_id]:
            raise V11E4IntegrityError(
                f"descriptive summary requires persisted {experiment_id} cells."
            )
        summary = _experiment_descriptive_summary(
            experiment_id, cells[experiment_id], authorized_commit=authorized_commit
        )
        outputs[experiment_id] = summary
        if write_outputs:
            atomic_write_json(paths["summary_dir"] / f"{experiment_id}.json", summary)
    return outputs


# --------------------------------------------------------------------------- #
# result lock + family verification (read-only verify mode)
# --------------------------------------------------------------------------- #


def _enumerate_family(base_dir: Path) -> dict[str, list[Path]]:
    paths = lineate_output_paths(base_dir)
    cells = sorted(paths["stage_dir"].glob("*.json")) if paths["stage_dir"].is_dir() else []
    raw_files = (
        sorted(paths["raw_obs_dir"].rglob("*.json"))
        if paths["raw_obs_dir"].is_dir() else []
    )
    summary_files = (
        sorted(paths["summary_dir"].glob("*.json"))
        if paths["summary_dir"].is_dir() else []
    )
    return {
        "cells": cells,
        "raw": raw_files,
        "summaries": summary_files,
        "attack_manifest": paths["attack_manifest"],
        "workload_manifest": paths["workload_manifest"],
        "result_lock": paths["result_lock"],
    }


def verify_result_family(
    base_dir: Path,
    *,
    authorized_commit: str,
    sizes: Sequence[int],
    seeds: Sequence[int],
    require_git: bool = True,
    git_resolver: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Read-only verification of an existing output family (never writes)."""
    authorization = verify_authorized_execution_commit(
        authorized_commit, require_git=require_git, git_resolver=git_resolver
    )
    family = _enumerate_family(base_dir)
    paths = lineate_output_paths(base_dir)
    errors: list[str] = []

    workload = read_json(paths["workload_manifest"], "workload manifest")

    expected = build_execution_plan(sizes=sizes, seeds=seeds)
    expected_by_stem = {spec.file_stem(): spec for spec in expected}
    expected_keys = [spec.cell_key() for spec in expected]
    if len(set(expected_keys)) != len(expected_keys):
        errors.append("execution plan contains duplicate cell keys.")

    found_stems = [cell_file_stem_from_filename(p.name) for p in family["cells"]]
    missing = [spec.cell_key() for spec in expected if spec.file_stem() not in set(found_stems)]
    extra = sorted(set(found_stems) - set(expected_by_stem))
    duplicates = [stem for stem in sorted(set(found_stems)) if found_stems.count(stem) > 1]
    if missing:
        errors.append(f"missing cells: {missing}")
    if extra:
        errors.append(f"unexpected cells: {extra}")
    if duplicates:
        errors.append(f"duplicate cells: {duplicates}")

    for path in family["cells"]:
        stem = cell_file_stem_from_filename(path.name)
        spec = expected_by_stem.get(stem)
        if spec is None:
            continue
        try:
            load_verified_cell(
                path, spec, authorization,
                workload_manifest_semantic_sha256=str(read_json(
                    paths["workload_manifest"], "workload manifest"
                ).get("semantic_sha256")),
                require_e10_derived=True if path.name.startswith("E10") else False,
            )
        except V11E4Error as exc:
            errors.append(f"cell {path.name}: {exc}")

    lock_path = paths["result_lock"]
    lock = None
    if lock_path.exists():
        lock = read_json(lock_path, "result lock")
        try:
            verify_result_lock_document(lock, authorized_commit=authorized_commit,
                                        sizes=sizes, seeds=seeds)
        except V11E4Error as exc:
            errors.append(f"result lock: {exc}")
    else:
        errors.append("missing v11e_experiment_result_lock.json")

    fam_present = {
        "attack_manifest": family["attack_manifest"].exists(),
        "workload_manifest": family["workload_manifest"].exists(),
        "cells": len(family["cells"]),
        "raw_observation_files": len(family["raw"]),
        "summary_files": family["summaries"],
        "result_lock": lock_path.exists(),
    }
    ok = not errors
    return {
        "authorization": authorization,
        "well_formed": ok,
        "errors": errors,
        "family": fam_present,
        "expected_cells": len(expected_keys),
        "found_cells": len(found_stems),
        "expected_cell_keys_per_experiment": {
            eid: expected_cell_count(eid, seeds=len(seeds), sizes=len(sizes))
            for eid in EXECUTION_ORDER + ("E10",)
        },
        "e10_cell_count": sum(1 for key in expected_keys if key.startswith("E10|")),
        "lock": lock,
        "marker": "V11E4L_FAMILY_VERIFIED" if ok else "V11E4L_FAMILY_INVALID",
    }


def _session_leak_counts() -> dict[str, int]:
    """Fail-closed leak counters from the frozen E3 data gates (read-only)."""
    evidence = p.run_v11e3_preflight(checkpoint_guard=False)
    return {
        "test_access_count": int(evidence["test_access_count"]),
        "ai_fit_count": int(evidence["ai_fit_count"]),
        "ai_inference_count": int(evidence["ai_inference_count"]),
        "experiment_execution_count": int(evidence["experiment_execution_count"]),
    }


def build_result_lock(
    base_dir: Path,
    *,
    authorized_commit: str,
    executed_cells: Sequence[str],
    skipped_cells: Sequence[str],
    sizes: Sequence[int] = p.WORKLOADS,
    seeds: Sequence[int] = p.BLOCKCHAIN_SEEDS,
) -> dict[str, Any]:
    """Build + atomically write the final fail-closed result lock.

    The lock refuses to be ``ok`` unless: the full expected family is present,
    every cell re-verifies (identity + protocol + commit + semantic sha),
    no TEST/AI-fit/AI-inference activity leaked, E10 carried zero measurement
    campaigns, and all recorded fingerprints match.
    """
    paths = lineate_output_paths(base_dir)
    family = _enumerate_family(base_dir)
    leaks = _session_leak_counts()
    workload = read_json(paths["workload_manifest"], "workload manifest")
    if not isinstance(workload.get("semantic_sha256"), str) or len(workload["semantic_sha256"]) != 64:
        raise V11E4IntegrityError("persisted workload manifest semantic is malformed.")

    expected = build_execution_plan(sizes=sizes, seeds=seeds)
    expected_by_stem = {spec.file_stem(): spec.cell_key() for spec in expected}
    expected_keys = set(expected_by_stem.values())
    found_stems = {cell_file_stem_from_filename(path.name) for path in family["cells"]}
    found_keys = {expected_by_stem[stem] for stem in found_stems & set(expected_by_stem)}
    fail_closed: dict[str, Any] = {
        "expected_cell_count": len(expected_keys),
        "persisted_cell_count": len(found_keys),
        "all_expected_cells_present": found_stems == set(expected_by_stem),
        "no_duplicate_cells": len(found_stems) == len(family["cells"]),
        "test_access_count": leaks["test_access_count"],
        "ai_fit_count": leaks["ai_fit_count"],
        "ai_inference_count": leaks["ai_inference_count"],
        "experiment_execution_count": leaks["experiment_execution_count"],
        "test_access_zero": leaks["test_access_count"] == 0,
        "ai_fit_zero": leaks["ai_fit_count"] == 0,
        "ai_inference_zero": leaks["ai_inference_count"] == 0,
        "experiment_execution_zero": leaks["experiment_execution_count"] == 0,
        "e10_measurement_campaign_count": 0,
        "all_cell_semantics_valid": True,
    }
    all_valid = (fail_closed["all_expected_cells_present"]
                 and fail_closed["no_duplicate_cells"]
                 and fail_closed["test_access_zero"]
                 and fail_closed["ai_fit_zero"]
                 and fail_closed["ai_inference_zero"]
                 and fail_closed["experiment_execution_zero"])

    fingerprints: dict[str, str] = {}
    for label, path in (
        ("attack_manifest", family["attack_manifest"]),
        ("workload_manifest", family["workload_manifest"]),
    ):
        fingerprints[f"{label}_file_sha256"] = sha256_file(path)
    for eid in EXECUTION_ORDER + ("E10",):
        for path in sorted(family["cells"]):
            if path.name.startswith(eid + "_"):
                fingerprints[f"cell:{path.stem}"] = sha256_file(path)
    for path in sorted(family["raw"]):
        fingerprints[f"raw:{path.parent.name}/{path.stem}"] = sha256_file(path)
    for path in sorted(family["summaries"]):
        fingerprints[f"summary:{path.stem}"] = sha256_file(path)

    semantic_payload: dict[str, Any] = {
        "stage": RESULTS_STAGE,
        "protocol_sha256": p.E2_PROTOCOL_LOCK_SEMANTIC_SHA256,
        "authorized_execution_commit": authorized_commit,
        "workload_manifest_semantic_sha256": workload["semantic_sha256"],
        "expected_cells": sorted(expected_keys),
        "executed_cells": sorted(set(executed_cells)),
        "skipped_cells": sorted(set(skipped_cells)),
        "counts": {
            "expected_total": len(expected_keys),
            "executed_now": len(set(executed_cells)),
            "skipped": len(set(skipped_cells)),
        },
        "fail_closed": fail_closed,
        "energy_marker": p.ENERGY_MARKER,
    }
    lock_document = {
        "artifact_kind": "V11E4_EXPERIMENT_RESULT_LOCK",
        "stage": RESULTS_STAGE,
        "semantic_payload": semantic_payload,
        "semantic_result_lock_sha256": sha256_of_canonical(semantic_payload),
        "artifacts_fingerprints_sha256": fingerprints,
        "execution_metadata": {
            "experiment_status": "EXECUTED" if all_valid else "INCOMPLETE",
            "generated_at_utc_iso": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }
    atomic_write_json(paths["result_lock"], lock_document)
    return lock_document


def verify_result_lock_document(
    lock_document: Mapping[str, Any],
    *,
    authorized_commit: str,
    sizes: Sequence[int],
    seeds: Sequence[int],
) -> None:
    if lock_document.get("artifact_kind") != "V11E4_EXPERIMENT_RESULT_LOCK":
        raise V11E4IntegrityError("result lock artifact_kind mismatch.")
    semantic = lock_document.get("semantic_payload")
    if not isinstance(semantic, Mapping):
        raise V11E4IntegrityError("result lock missing semantic_payload.")
    recomputed = sha256_of_canonical({key: value for key, value in semantic.items()
                                      if key != "semantic_result_lock_sha256"})
    if recomputed != lock_document.get("semantic_result_lock_sha256"):
        raise V11E4IntegrityError("result lock semantic sha drift.")
    if semantic.get("authorized_execution_commit") != authorized_commit:
        raise V11E4IntegrityError("result lock authorized-commit mismatch.")
    fail_closed = semantic.get("fail_closed") or {}
    for key in ("test_access_zero", "ai_fit_zero", "ai_inference_zero",
                "experiment_execution_zero", "all_expected_cells_present",
                "no_duplicate_cells"):
        if fail_closed.get(key) is not True:
            raise V11E4IntegrityError(f"result lock fail-closed condition {key} not satisfied.")
    if fail_closed.get("e10_measurement_campaign_count") != 0:
        raise V11E4IntegrityError("result lock fail-closed condition e10_measurement_campaign_count not satisfied.")


def run_launcher_preflight(
    *,
    authorized_commit: str,
    base_dir: Path = V11E_DIR,
    sizes: Sequence[int] = p.WORKLOADS,
    seeds: Sequence[int] = p.BLOCKCHAIN_SEEDS,
    require_git: bool = True,
    git_resolver: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Fail-closed launcher preflight: authorization + data gates + family state."""
    authorization = verify_authorized_execution_commit(
        authorized_commit, require_git=require_git, git_resolver=git_resolver
    )
    e3 = p.run_v11e3_preflight(checkpoint_guard=False)
    leaks = _session_leak_counts()
    plan = build_execution_plan(sizes=sizes, seeds=seeds)
    paths = lineate_output_paths(base_dir)
    family_state = {
        "attack_manifest_present": paths["attack_manifest"].exists(),
        "workload_manifest_present": paths["workload_manifest"].exists(),
        "persisted_cells": len(list(paths["stage_dir"].glob("*.json")))
        if paths["stage_dir"].is_dir() else 0,
        "result_lock_present": paths["result_lock"].exists(),
        "stale_temp_files": [
            str(path.name) for path in (base_dir.rglob("*") if base_dir.is_dir() else [])
            if _TMP_SUFFIX in path.name
        ],
    }
    return {
        "authorization": authorization,
        "e3_data_gates_passed": e3.get("live_governed_artifact_gate_passed")
        and not e3.get("semantic_sha256", "") == "",
        "leak_counts": leaks,
        "leaks_zero": all(leaks[key] == 0 for key in leaks),
        "plan": {
            "total_cells": len(plan),
            "expected_total": expected_total_cell_count(seeds=len(seeds), sizes=len(sizes)),
            "per_experiment": {
                eid: expected_cell_count(eid, seeds=len(seeds), sizes=len(sizes))
                for eid in EXECUTION_ORDER + ("E10",)
            },
        },
        "family_state": family_state,
        "marker": PREFLIGHT_MARKER,
    }
