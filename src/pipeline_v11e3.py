"""V1.1-E3 governed experiment-runner machinery for the frozen E01-E10 matrix.

E3 is an implementation stage, not an execution stage: it provides the
deterministic machinery that a later authorized run will use to execute the
preregistered E01-E10 experiments from the frozen V1.1-E2B protocol lock.
Everything here is deterministic and read-only over the frozen upstream
artifacts (V1.1-A engine, V1.1-B chains, V1.1-C mapping, V1.1-D risk artifact,
V1.1-E1 integration readiness, V1.1-E2B protocol lock). Nothing here runs the
final governed experiments, fits models, performs AI inference, runs
optimizers, touches TEST, or writes results.

Frozen inputs consumed (read-only):
* V1.1-E2B experiment protocol lock (semantic sha pinned) and the mirroring
  YAML config + protocol document;
* V1.1-E1 integration-readiness harness (chain reconstruction, tamper
  PA-01..PA-09, policy adapter, measurement proxies) reused verbatim;
* V1.1-A engine validation (c1-c8 + ALV + canonical serialization).

E3 machinery surface (deterministic, unit-tested on synthetic fixtures):
* workload sampler: SHA-256 ranking of governed order ids per blockchain seed
  with repository canonical serialization, digest tie-breaks by canonical
  numeric order id, nested prefixes [100, 250, 500, 1000, 2500];
* experiment context: frozen, serializable execution context;
* policy adapters B0/B1/P (reused engine validation, never reimplemented);
* attack-instance factory for PA-01..PA-09 (E1 tamper reuse, copy-only,
  PA-08/PA-09 donor read-only);
* timing harness: wall perf_counter_ns + cpu process_time_ns around the
  validation-policy call only; per-policy unmeasured warm-up; single thread;
  seed-counterbalanced policy order;
* E01-E10 runners emitting raw-observation records with explicit null/NA
  semantics for non-applicable fields;
* descriptive aggregation (mean/median/sample std ddof=1/min/max, pooled
  rates, paired differences) without CI/p-value/bootstrap/inferential tools;
* LOW/HIGH band governance markers and the frozen DIRECT_ENERGY_UNAVAILABLE
  marker (no Joules/Wh/TDP inference anywhere).

The only executable activity in this module is a synthetic dry-run harness
over tiny labeled fixtures (branch coverage) plus a read-only readiness
preflight; neither accesses TEST nor runs governed workloads.
"""

from __future__ import annotations

import subprocess
import sys
import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import src.pipeline_v11e1 as e1
from src.blockchain_engine.block import Block
from src.blockchain_engine.canonical import (
    canonical_order_id,
    sha256_hex,
)
from src.blockchain_engine.validation import ValidationResult
from src.blockchain_engine.adaptive_validation import (
    ALV_BAND_TO_CHECKS,
    ALV_VALIDATOR_COUNTS,
)
from src.blockchain_engine.errors import BlockchainEngineError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STAGE = "V1.1-E3"
KIND = "V11E3_EXPERIMENT_RUNNER_MACHINERY"
PROTOCOL_VERSION = "v1.1-e3-governed-experiment-runner-machinery-1"
EXPECTED_HEAD = "b1ff42e9b2e71cce9e829261b2cd0f8a8a79a3a0"
E3_MARKER = "V11E3_EXPERIMENT_RUNNER_MACHINERY_READY_REVIEW_REQUIRED"
E10_DERIVED_ONLY_MSG = (
    "E10 is DERIVED ONLY: derive_e10_resource_green_proxy requires cached E06-E09 "
    "cell outputs and must never re-run measurement campaigns."
)

E2_PROTOCOL_LOCK_SEMANTIC_SHA256 = (
    "8047fbe356c964e26f7acdcde930a1e2a6734026ea303b56334498ffa295f239"
)
E2_CONFIG_FINGERPRINT = (
    "f895115c201c55685615e5dba2feb88ce7b1ddd790812bf4696c31722fc21334"
)
E2_DOC_FINGERPRINT = (
    "e2e1cf976098f9cfbea5c693775447bf0936b057f3fd1da4c2672fa03ea969c1"
)

V11E_DIR = e1.BLOCKCHAIN_RESULTS_DIR / "v11e"
E2_PROTOCOL_LOCK_PATH = V11E_DIR / "v11e_experiment_protocol_lock.json"
E2_PROTOCOL_CONFIG_PATH = PROJECT_ROOT / "config" / "experiments_v11e.yaml"
E2_PROTOCOL_DOC_PATH = PROJECT_ROOT / "docs" / "v11e_experiment_protocol.md"

WORKLOADS = (100, 250, 500, 1000, 2500)
BLOCKCHAIN_SEEDS = (522, 523, 524)
REPETITIONS = 3
POLICIES = ("B0", "B1", "P")
PA_SCENARIOS = e1.PA_SCENARIOS

ENERGY_MARKER = "DIRECT_ENERGY_UNAVAILABLE"
ENERGY_FORBIDDEN_TOKENS = (
    "Joules",
    "joule",
    "Wh",
    "TDP",
    "kWh",
    "physical_energy_savings",
)

LOW_ABSENT_MARKER = "LOW_ABSENT_IN_GOVERNED_POPULATION"
HIGH_ABSENT_MARKER = "HIGH_NOT_OBSERVED_IN_CONDITION"
NA = None  # explicit null/NA semantics: non-applicable fields serialize to JSON null

MEMORY_PROXY_LABEL = "COMPUTATIONAL_MEMORY_PROXY"

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")


class V11E3Error(Exception):
    """Base error for the V1.1-E3 experiment-runner machinery stage."""


class V11E3PreflightError(V11E3Error):
    """Raised when an E3 readiness gate fails closed."""


class V11E3NotAuthorizedError(V11E3Error):
    """Raised when E3 machinery is asked to do an unsanctioned action."""


def _read_json(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise V11E3Error(f"frozen artifact is not a JSON object: {path}")
    return payload


def _canonical_json(value: Any) -> str:
    from src.blockchain_engine.canonical import canonical_json

    return canonical_json(value)


def _semantic_sha256(payload: Mapping[str, Any]) -> str:
    return sha256_hex(dict(payload))


# --------------------------------------------------------------------------- #
# frozen policy descriptors (mirror the E2B config exactly; reuse validation)
# --------------------------------------------------------------------------- #


def policy_check_spec(policy: str) -> dict[str, Any]:
    """Frozen check/validator descriptor for one of B0/B1/P (never edited)."""
    if policy == "B0":
        return {"policy": "B0", "checks": ["c1", "c2", "c3", "c4"], "validators": 1, "mode": "LIGHTWEIGHT_FIXED"}
    if policy == "B1":
        return {
            "policy": "B1",
            "checks": ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"],
            "validators": 3,
            "quorum": 3,
            "mode": "FULL_FROZEN_QUORUM",
        }
    if policy == "P":
        return {
            "policy": "P",
            "mode": "AI_DRIVEN_ADAPTIVE_ALV",
            "risk_input": "GOVERNED_AI_RISK",
            "bands": {
                level: {
                    "checks": list(ALV_BAND_TO_CHECKS[level]),
                    "validators": ALV_VALIDATOR_COUNTS[level],
                }
                for level in RISK_LEVELS
            },
        }
    raise V11E3Error(f"unknown policy {policy!r}")


def validate_under_policy(policy: str, block: Block, chain: Sequence[Block], order_id: str) -> ValidationResult:
    """Run one B0/B1/P verdict reusing the frozen E1 policy adapter."""
    return e1.validate_under_policy(policy, block, chain, order_id)


def policy_execution_order(seed: int) -> tuple[str, ...]:
    """Deterministic seed-counterbalanced B0/B1/P execution order."""
    offset = BLOCKCHAIN_SEEDS.index(int(seed))
    return tuple(POLICIES[offset:] + POLICIES[:offset])


# --------------------------------------------------------------------------- #
# deterministic workload sampler (E2B sampling contract, no replacement)
# --------------------------------------------------------------------------- #


def rank_order_ids(seed: Any, order_ids: Sequence[Any]) -> tuple[str, ...]:
    """Rank canonical governed order ids for one blockchain seed.

    Each order id is ranked by SHA-256 over the repository-canonical
    serialization of ``{"seed": <int seed>, "order_id": <canonical oid>}``
    (sorted keys, compact separators, ensure_ascii, no trailing newline).
    Digest ties are broken by the canonical numeric order id. Deterministic,
    platform-independent, no replacement.
    """
    canonical_ids = [canonical_order_id(oid) for oid in order_ids]
    if len(set(canonical_ids)) != len(canonical_ids):
        raise V11E3Error("order id set contains duplicates")
    decorated = [
        (
            sha256_hex({"seed": int(seed), "order_id": oid}),
            int(oid),
            oid,
        )
        for oid in canonical_ids
    ]
    decorated.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in decorated)


def workload_sample(seed: Any, order_ids: Sequence[Any], size: int) -> tuple[str, ...]:
    """First ``size`` orders of the seed ranking (nested-prefix contract)."""
    if int(size) <= 0:
        raise V11E3Error("workload size must be positive")
    return rank_order_ids(seed, order_ids)[: int(size)]


def nested_workloads(
    seed: Any, order_ids: Sequence[Any], sizes: Sequence[int] = WORKLOADS
) -> dict[int, tuple[str, ...]]:
    """Nested prefix workloads for the frozen sizes, verified nested."""
    ranking = rank_order_ids(seed, order_ids)
    workloads: dict[int, tuple[str, ...]] = {}
    for size in sizes:
        workload = tuple(ranking[: int(size)])
        workloads[int(size)] = workload
    for larger in sorted(workloads):
        for smaller in sorted(workloads):
            if smaller < larger and not set(workloads[smaller]).issubset(set(workloads[larger])):
                raise V11E3Error("nested prefixes violated")
    return workloads


def risk_band_counts(order_ids: Sequence[Any], records: Mapping[str, Any]) -> dict[str, int]:
    """LOW/MEDIUM/HIGH counts over an order-id set from governed records."""
    counts = Counter(records[canonical_order_id(oid)]["risk_level"] for oid in order_ids)
    return {level: int(counts.get(level, 0)) for level in RISK_LEVELS}


def band_markers(counts: Mapping[str, int]) -> list[str]:
    """LOW/HIGH governance markers for a workload's risk counts."""
    markers: list[str] = []
    if counts.get("LOW", 0) == 0:
        markers.append(LOW_ABSENT_MARKER)
    if counts.get("HIGH", 0) == 0:
        markers.append(HIGH_ABSENT_MARKER)
    return markers


def workload_manifest(
    order_ids: Sequence[Any],
    records: Mapping[str, Any],
    *,
    sizes: Sequence[int] = WORKLOADS,
    seeds: Sequence[int] = BLOCKCHAIN_SEEDS,
) -> dict[str, Any]:
    """Deterministic workload manifest mirroring the frozen sampling contract."""
    workloads_by_seed: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        workloads = nested_workloads(seed, order_ids, sizes)
        workloads_by_seed[str(seed)] = {
            str(size): {
                "order_ids": list(workloads[int(size)]),
                "risk_counts": risk_band_counts(workloads[int(size)], records),
                "markers": band_markers(risk_band_counts(workloads[int(size)], records)),
            }
            for size in sizes
        }
    return {
        "stage": STAGE,
        "kind": KIND,
        "protocol_sha256": E2_PROTOCOL_LOCK_SEMANTIC_SHA256,
        "sizes": list(sizes),
        "seeds": list(seeds),
        "nested_prefixes": True,
        "no_replacement": True,
        "workloads_by_seed": workloads_by_seed,
        "semantic_sha256": _semantic_sha256(
            {
                key: value
                for key, value in {
                    "stage": STAGE,
                    "kind": KIND,
                    "protocol_sha256": E2_PROTOCOL_LOCK_SEMANTIC_SHA256,
                    "sizes": list(sizes),
                    "seeds": list(seeds),
                    "nested_prefixes": True,
                    "workloads_by_seed": workloads_by_seed,
                }.items()
            }
        ),
    }


# --------------------------------------------------------------------------- #
# experiment context (frozen, deterministic, serializable)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExperimentContext:
    """Frozen execution context for one experiment cell (deterministic)."""

    experiment_id: str
    seed: int
    repetition: int
    workload_size: int
    order_ids: tuple[str, ...]
    policy: str
    attack_id: str | None
    protocol_sha256: str
    upstream_provenance: Mapping[str, str]
    environment: Mapping[str, str]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "repetition": self.repetition,
            "workload_size": self.workload_size,
            "order_ids": list(self.order_ids),
            "policy": self.policy,
            "attack_id": self.attack_id,
            "protocol_sha256": self.protocol_sha256,
            "upstream_provenance": dict(self.upstream_provenance),
            "environment": dict(self.environment),
        }

    @property
    def semantic_sha256(self) -> str:
        return sha256_hex(self.as_mapping())


def context_for(
    experiment_id: str,
    seed: int,
    workload_size: int,
    order_ids: Sequence[Any],
    policy: str,
    *,
    attack_id: str | None = None,
    repetitions: int = REPETITIONS,
    upstream_provenance: Mapping[str, str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> ExperimentContext:
    """Build an E3 experiment context over an explicitly frozen workload.

    ``repetition`` is the deterministic 1-indexed position of ``seed`` within
    the frozen blockchain-seed family [522, 523, 524].
    """
    canonical_ids = tuple(canonical_order_id(oid) for oid in order_ids)
    if len(canonical_ids) != int(workload_size):
        raise V11E3Error("context order_ids length must equal workload_size")
    if int(seed) not in BLOCKCHAIN_SEEDS:
        raise V11E3Error(f"seed {seed!r} is outside the frozen blockchain-seed family")
    if policy not in POLICIES:
        raise V11E3Error(f"unknown policy {policy!r}")
    if attack_id is not None and attack_id not in PA_SCENARIOS:
        raise V11E3Error(f"unknown attack {attack_id!r}")
    repetition = BLOCKCHAIN_SEEDS.index(int(seed)) + 1
    if repetition > repetitions:
        raise V11E3Error("repetition exceeds the frozen repetition count")
    return ExperimentContext(
        experiment_id=experiment_id,
        seed=int(seed),
        repetition=repetition,
        workload_size=int(workload_size),
        order_ids=canonical_ids,
        policy=policy,
        attack_id=attack_id,
        protocol_sha256=E2_PROTOCOL_LOCK_SEMANTIC_SHA256,
        upstream_provenance=dict(upstream_provenance or _upstream_provenance()),
        environment=dict(environment or _environment_metadata()),
    )


def _upstream_provenance() -> dict[str, str]:
    return {
        "v11a_protocol_lock_sha256": "aaf3ac3139e8296fb7f976a8a7ed9f18eb317af2f73b10cd0208b6bb870ebd8d",
        "v11c_mapping_semantic_sha256": e1.V11C_MAPPING_SEMANTIC_SHA256,
        "v11d_result_lock_semantic_sha256": e1.D3_RESULT_LOCK_SEMANTIC,
        "v11d_artifact_semantic_sha256": e1.ARTIFACT_SEMANTIC_SHA256,
        "v11d_artifact_disk_sha256": e1.ARTIFACT_DISK_SHA256,
        "v11d_summary_semantic_sha256": e1.SUMMARY_SEMANTIC_SHA256,
        "v11d_summary_disk_sha256": e1.SUMMARY_DISK_SHA256,
        "v11e_protocol_lock_semantic_sha256": E2_PROTOCOL_LOCK_SEMANTIC_SHA256,
    }


def _environment_metadata() -> dict[str, str]:
    """Fixed, deterministic environment facts (no machine-specific drift)."""
    return {
        "threads": "1",
        "timing_boundary": "validation_policy_call_only",
        "energy_marker": ENERGY_MARKER,
        "warmup": "one_unmeasured_synthetic_warmup_per_policy",
    }


# --------------------------------------------------------------------------- #
# timing harness (wall perf_counter_ns + cpu process_time_ns)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TimingSample:
    policy: str
    wall_ns: int
    cpu_ns: int

    def as_mapping(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "wall_ns": self.wall_ns,
            "cpu_ns": self.cpu_ns,
            "timing_boundary": "validation_policy_call_only",
        }


def time_policy_validation(
    policy: str, block: Block, chain: Sequence[Block], order_id: str
) -> tuple[ValidationResult, TimingSample]:
    """Time one validation-policy verdict with the frozen ns boundary.

    The timer starts immediately before the validation-policy call and stops
    immediately after the complete verdict is returned. Chain construction,
    AI generation/inference, attack generation/injection, serialization prep,
    and result writing are excluded by construction (callers materialize those
    before invoking this function).
    """
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    result = validate_under_policy(policy, block, chain, order_id)
    wall_ns = time.perf_counter_ns() - wall_start
    cpu_ns = time.process_time_ns() - cpu_start
    return result, TimingSample(policy=policy, wall_ns=wall_ns, cpu_ns=cpu_ns)


def warm_up_validation(policy: str, block: Block, chain: Sequence[Block], order_id: str) -> None:
    """One unmeasured per-policy warm-up (frozen timing contract)."""
    validate_under_policy(policy, block, chain, order_id)


# --------------------------------------------------------------------------- #
# deterministic operation counts (proxy evidence, never energy)
# --------------------------------------------------------------------------- #


def deterministic_operation_counts(
    result: ValidationResult, blocks: Sequence[Block]
) -> dict[str, int]:
    """Check / validator-invocation / hash-operation counts for one verdict.

    Hash operations use the frozen E1 deterministic c2/c8 counting
    (``hash_operation_count``). These are computational proxies, never Joules.
    """
    checks = tuple(result.applied_checks)
    return {
        "validation_check_count": len(checks) * result.validator_count,
        "validator_invocation_count": result.validator_count,
        "hash_operation_count": e1.hash_operation_count(tuple(blocks), checks)
        * result.validator_count,
    }


# --------------------------------------------------------------------------- #
# attack-instance factory (E1 tamper reuse, copy-only, donor read-only)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AttackInstance:
    """One deterministic PA attack instance over copied target blocks."""

    instance_id: str
    scenario_id: str
    tamper_kind: str
    seed: int
    workload_size: int
    target_order_id: str
    donor_order_id: str | None
    tampered_blocks: tuple[Block, ...]
    applied_metadata: Mapping[str, Any]
    copies_only: bool = True
    donor_read_only: bool = True

    def as_mapping(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "scenario_id": self.scenario_id,
            "tamper_kind": self.tamper_kind,
            "seed": self.seed,
            "workload_size": self.workload_size,
            "target_order_id": self.target_order_id,
            "donor_order_id": self.donor_order_id,
            "copies_only": self.copies_only,
            "donor_read_only": self.donor_read_only,
            "tampered_block_count": len(self.tampered_blocks),
            "applied_metadata": dict(self.applied_metadata),
            "semantic_sha256": sha256_hex(
                {
                    "scenario_id": self.scenario_id,
                    "seed": self.seed,
                    "workload_size": self.workload_size,
                    "target_order_id": self.target_order_id,
                    "donor_order_id": self.donor_order_id,
                }
            ),
        }


def attack_target_index(seed: Any, workload_size: int, scenario_id: str, *, offset: int = 0) -> int:
    """Deterministic affected-target index within a workload for one attack."""
    digest = sha256_hex(
        {
            "role": "target",
            "seed": int(seed),
            "workload_size": int(workload_size),
            "scenario_id": scenario_id,
            "offset": int(offset),
        }
    )
    return int(digest[-8:], 16) % int(workload_size)


def attack_donor_index(seed: Any, workload_size: int, scenario_id: str, *, target_index: int) -> int:
    """Deterministic donor index dist in ct from the target (PA-08/PA-09)."""
    if int(workload_size) < 2:
        raise V11E3Error("PA-08/PA-09 require at least two workload orders")
    digest = sha256_hex(
        {
            "role": "donor",
            "seed": int(seed),
            "workload_size": int(workload_size),
            "scenario_id": scenario_id,
            "target_index": int(target_index),
        }
    )
    index = int(digest[-8:], 16) % int(workload_size)
    if index == int(target_index):
        index = (index + 1) % int(workload_size)
    if index == int(target_index):
        raise V11E3Error("donor cannot equal target under the frozen contract")
    return index


def build_attack_instance(
    scenario_id: str,
    seed: Any,
    workload_size: int,
    target_order_id: Any,
    donor_order_id: Any | None,
    target_blocks: Sequence[Block],
    *,
    declared_length: int | None = None,
) -> AttackInstance:
    """Deterministically build one PA attack instance on a copy of the target.

    Reuses the frozen E1 ``apply_tamper`` for PA-01..PA-09 semantics (never
    redefines them). Tampering only ever runs on a deep copy of the target
    chain blocks; the authoritative chain and any donor chain are never
    mutated. For PA-08/PA-09 the donor order is explicit, distinct from the
    target, and read-only. The same attacked copy is consumable by every
    policy (B0/B1/P) and by E02/E03/E04 because the instance is fully
    deterministic from its input context.
    """
    if scenario_id not in e1.PA_TAMPER_KINDS:
        raise V11E3Error(f"unknown tamper scenario {scenario_id!r}")
    target = canonical_order_id(target_order_id)
    donor = canonical_order_id(donor_order_id) if donor_order_id is not None else target
    cross_order = scenario_id in ("PA-08", "PA-09")
    if cross_order and donor == target:
        raise V11E3Error("PA-08/PA-09 require target and donor to be distinct")
    instance_id = sha256_hex(
        {
            "scenario_id": scenario_id,
            "seed": int(seed),
            "workload_size": int(workload_size),
            "target_order_id": target,
            "donor_order_id": donor if cross_order else None,
        }
    )
    extra: dict[str, Any] = {}
    if scenario_id == "PA-08":
        extra["substitute_order_id"] = donor
    elif scenario_id == "PA-09":
        extra["replay_order_id"] = donor
    tampered, metadata = e1.apply_tamper(
        list(target_blocks),
        scenario_id,
        expected_order_id=target,
        declared_length=declared_length or len(list(target_blocks)),
        extra=extra,
    )
    return AttackInstance(
        instance_id=instance_id,
        scenario_id=scenario_id,
        tamper_kind=e1.PA_TAMPER_KINDS[scenario_id],
        seed=int(seed),
        workload_size=int(workload_size),
        target_order_id=target,
        donor_order_id=donor if cross_order else None,
        tampered_blocks=tuple(tampered),
        applied_metadata=dict(metadata),
    )


def attack_instances_for_workload(
    order_ids: Sequence[Any],
    chains: Mapping[str, Sequence[Block]],
    seed: Any,
    *,
    scenarios: Sequence[str] = PA_SCENARIOS,
) -> dict[str, AttackInstance]:
    """One deterministic attack instance per {seed, workload, attack}."""
    canonical_ids = [canonical_order_id(oid) for oid in order_ids]
    size = len(canonical_ids)
    instances: dict[str, AttackInstance] = {}
    for scenario_id in scenarios:
        target_index = attack_target_index(seed, size, scenario_id)
        target = canonical_ids[target_index]
        donor: str | None = None
        if scenario_id in ("PA-08", "PA-09"):
            donor_index = attack_donor_index(seed, size, scenario_id, target_index=target_index)
            donor = canonical_ids[donor_index]
        instance = build_attack_instance(
            scenario_id,
            seed,
            size,
            target,
            donor,
            chains[target],
            declared_length=len(list(chains[target])),
        )
        instances[scenario_id] = instance
    return instances


# --------------------------------------------------------------------------- #
# raw observation schema (explicit null/NA semantics per E2)
# --------------------------------------------------------------------------- #

RAW_OBSERVATION_FIELDS = (
    "experiment_id",
    "seed",
    "repetition",
    "workload",
    "policy",
    "attack",
    "orders",
    "blocks",
    "risk_counts",
    "risk_markers",
    "validation_outcome",
    "detection_outcome",
    "localization_outcome",
    "wall_time_ns",
    "cpu_time_ns",
    "memory_proxy_mib",
    "storage_bytes",
    "throughput_orders_per_second",
    "validation_check_count",
    "validator_invocation_count",
    "hash_operation_count",
    "energy_marker",
    "semantic_sha256",
)


def raw_observation(
    context: ExperimentContext,
    *,
    blocks: int | None = None,
    attack: str | None = None,
    risk_counts: Mapping[str, int] | None = None,
    validation_outcome: bool | None = NA,
    detection_outcome: bool | None = NA,
    localization_outcome: Any | None = NA,
    wall_time_ns: int | None = NA,
    cpu_time_ns: int | None = NA,
    memory_proxy_mib: float | None = NA,
    storage_bytes: int | None = NA,
    throughput_orders_per_second: float | None = NA,
    validation_check_count: int | None = NA,
    validator_invocation_count: int | None = NA,
    hash_operation_count: int | None = NA,
) -> dict[str, Any]:
    """Build one raw-run observation record with explicit null/NA semantics.

    Fields not applicable to the experiment cell are written as JSON ``null``
    (the E2-frozen NA semantics). ``semantic_sha256`` covers the non-signature
    payload under repository canonical serialization.
    """
    if risk_counts is None:
        risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    payload = {
        "experiment_id": context.experiment_id,
        "seed": context.seed,
        "repetition": context.repetition,
        "workload": context.workload_size,
        "policy": context.policy,
        "attack": attack if attack is not None else context.attack_id,
        "orders": context.workload_size,
        "blocks": blocks,
        "risk_counts": {
            level: int(risk_counts.get(level, 0)) for level in RISK_LEVELS
        },
        "risk_markers": band_markers(risk_counts),
        "validation_outcome": validation_outcome,
        "detection_outcome": detection_outcome,
        "localization_outcome": localization_outcome,
        "wall_time_ns": wall_time_ns,
        "cpu_time_ns": cpu_time_ns,
        "memory_proxy_mib": memory_proxy_mib,
        "storage_bytes": storage_bytes,
        "throughput_orders_per_second": throughput_orders_per_second,
        "validation_check_count": validation_check_count,
        "validator_invocation_count": validator_invocation_count,
        "hash_operation_count": hash_operation_count,
        "energy_marker": ENERGY_MARKER,
    }
    included = {key: payload[key] for key in RAW_OBSERVATION_FIELDS if key != "semantic_sha256"}
    record = dict(payload)
    record["semantic_sha256"] = sha256_hex(included)
    if tuple(record) != RAW_OBSERVATION_FIELDS:
        raise V11E3Error("raw observation field order violated")
    return record


# --------------------------------------------------------------------------- #
# descriptive aggregation (frozen: no CI/p-value/bootstrap/inferential)
# --------------------------------------------------------------------------- #


def mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def sample_std_ddof1(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def min_max(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return min(values), max(values)


def describe_continuous(values: Sequence[float]) -> dict[str, float | None]:
    """Frozen descriptive summary: mean/median/std(ddof=1)/min/max."""
    lo, hi = min_max(values)
    return {
        "mean": mean(values),
        "median": median(values),
        "std_ddof_1": sample_std_ddof1(values),
        "min": lo,
        "max": hi,
    }


def pooled_rate(numerators: Sequence[int], denominators: Sequence[int]) -> dict[str, float | None]:
    """Pooled descriptive rate = sum(numerators)/sum(denominators)."""
    numerator = int(sum(numerators))
    denominator = int(sum(denominators))
    rate = numerator / denominator if denominator else None
    return {"numerator": numerator, "denominator": denominator, "rate": rate}


def paired_differences(a_values: Sequence[float], b_values: Sequence[float]) -> list[float]:
    """Raw paired differences a - b over identically ordered matched runs."""
    if len(a_values) != len(b_values):
        raise V11E3Error("paired differences require identically ordered matched runs")
    return [a - b for a, b in zip(a_values, b_values)]


# --------------------------------------------------------------------------- #
# memory proxy (tracemalloc, MiB, COMPUTATIONAL_MEMORY_PROXY label)
# --------------------------------------------------------------------------- #


def peak_traced_validation_bytes(
    policy: str, block: Block, chain: Sequence[Block], order_id: str
) -> tuple[float, Mapping[str, Any]]:
    """Peak traced Python allocation during one validation (proxy, MiB).

    Uses tracemalloc over the validation-policy call in the calling process.
    The E3 governed path quantifies this inside a fresh worker process
    (``memory_proxy_fresh_worker``); the label is COMPUTATIONAL_MEMORY_PROXY
    and the value is NEVER presented as whole-system RSS or energy.
    """
    tracemalloc.start()
    try:
        result = validate_under_policy(policy, block, chain, order_id)
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return peak / (1024 * 1024), {
        "accepted": result.accepted,
        "current_bytes": int(current),
        "peak_bytes": int(peak),
        "label": MEMORY_PROXY_LABEL,
        "unit": "MiB",
        "energy_marker": ENERGY_MARKER,
    }


def memory_proxy_fresh_worker(
    policy: str, block: Block, chain: Sequence[Block], order_id: str
) -> tuple[float, Mapping[str, Any]]:
    """Peak traced allocation inside a fresh worker process (E07 contract).

    Serializes the (already materialized, un-attacked) chain to the worker via
    the repository canonical serialization; the worker measures with
    tracemalloc and returns MiB + verdict. Not a whole-system RSS measure and
    never an energy measure.
    """
    import base64
    import json
    import os
    import tempfile

    payload = {
        "policy": policy,
        "order_id": canonical_order_id(order_id),
        "blocks": [block.as_dict() for block in chain],
    }
    encoded = base64.b64encode(
        _canonical_json(payload).encode("utf-8")
    ).decode("ascii")
    code = (
        "import base64,json,sys,tracemalloc\n"
        "raw=base64.b64decode(sys.argv[1]).decode('utf-8')\n"
        "req=json.loads(raw)\n"
        "from src.blockchain_engine.block import Block\n"
        "import src.pipeline_v11e3 as m\n"
        "blocks=tuple(Block(**b) for b in req['blocks'])\n"
        "tracemalloc.start()\n"
        "try:\n"
        "  r=m.validate_under_policy(req['policy'], blocks[-1], blocks, req['order_id'])\n"
        "finally:\n"
        "  cur,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()\n"
        "print(json.dumps({'peak_bytes':peak,'current_bytes':cur,'accepted':r.accepted}))\n"
    )
    env = dict(os.environ)
    proc = subprocess.run(
        [sys.executable, "-c", code, encoded],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
        timeout=300,
    )
    if proc.returncode != 0:
        raise V11E3NotAuthorizedError(
            f"memory-proxy worker failed: {proc.stderr.strip()}"
        )
    parsed = json.loads(proc.stdout.strip().splitlines()[-1])
    return parsed["peak_bytes"] / (1024 * 1024), {
        "accepted": parsed["accepted"],
        "current_bytes": int(parsed["current_bytes"]),
        "peak_bytes": int(parsed["peak_bytes"]),
        "label": MEMORY_PROXY_LABEL,
        "unit": "MiB",
        "fresh_worker": True,
        "energy_marker": ENERGY_MARKER,
    }


# --------------------------------------------------------------------------- #
# storage proxy (canonical serialized blockchain bytes, policy-independent)
# --------------------------------------------------------------------------- #


def canonical_storage_bytes(blocks: Sequence[Block]) -> int:
    """Canonical serialized bytes of a chain (blocks) under the frozen contract."""
    return sum(len(_canonical_json(block.as_dict()).encode("utf-8")) for block in blocks)


def storage_observation(
    order_ids: Sequence[Any], chains: Mapping[str, Sequence[Block]]
) -> dict[str, Any]:
    """Policy-independent canonical storage bytes per workload (E08)."""
    canonical_ids = [canonical_order_id(oid) for oid in order_ids]
    total_blocks = 0
    total_bytes = 0
    per_order: dict[str, int] = {}
    for oid in canonical_ids:
        blocks = tuple(chains[oid])
        total_blocks += len(blocks)
        total_bytes += canonical_storage_bytes(blocks)
        per_order[oid] = canonical_storage_bytes(blocks)
    return {
        "orders": len(canonical_ids),
        "blocks": total_blocks,
        "canonical_blockchain_bytes": total_bytes,
        "bytes_per_order": total_bytes / len(canonical_ids) if canonical_ids else None,
        "bytes_per_block": total_bytes / total_blocks if total_blocks else None,
        "policy_independent": True,
        "note": "validation policy does not mutate canonical chain representation",
    }


# --------------------------------------------------------------------------- #
# E01-E10 runners (raw observations per frozen matrix; no file writes)
# --------------------------------------------------------------------------- #


def run_e01_functional_correctness(
    context: ExperimentContext,
    chains: Mapping[str, Sequence[Block]],
    governed_records: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """E01: clean verification pass count/rate + valid-chain construction."""
    records: list[dict[str, Any]] = []
    canonical_ids = list(context.order_ids)
    workload_rc = risk_band_counts(context.order_ids, governed_records) if governed_records else None
    valid_constructed = 0
    full_family = ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8")
    for oid in canonical_ids:
        blocks = tuple(chains[oid])
        head = blocks[-1]
        kind = validate_under_policy("B1", head, blocks, oid)
        if kind.accepted:
            valid_constructed += 1
            if tuple(kind.applied_checks) != tuple(full_family):
                raise V11E3Error("B1 did not apply the full frozen check family")
    valid = 0
    clean_rejections = 0
    check_acc: dict[str, int] = {}
    for oid in canonical_ids:
        blocks = tuple(chains[oid])
        head = blocks[-1]
        result, timing = time_policy_validation(context.policy, head, blocks, oid)
        if result.accepted:
            valid += 1
        else:
            clean_rejections += 1
        ops = deterministic_operation_counts(result, blocks)
        for check in result.applied_checks:
            check_acc[check] = check_acc.get(check, 0) + 1
        records.append(
            raw_observation(
                context,
                blocks=len(blocks),
                risk_counts=workload_rc,
                validation_outcome=result.accepted,
                wall_time_ns=int(timing.wall_ns),
                cpu_time_ns=int(timing.cpu_ns),
                validation_check_count=ops["validation_check_count"],
                validator_invocation_count=ops["validator_invocation_count"],
                hash_operation_count=ops["hash_operation_count"],
            )
        )
    primary = {
        "clean_verification_pass_count": valid,
        "clean_verification_pass_rate": valid / len(canonical_ids) if canonical_ids else None,
        "clean_false_rejection_count": clean_rejections,
        "valid_chain_construction_count": valid_constructed,
        "policy_consistent_check_allocation": check_acc,
    }
    return [
        {
            "experiment": "E01",
            "context": context.as_mapping(),
            "primary": primary,
            "raw_records": records,
            "timing_role": "SECONDARY_DESCRIPTIVE",
            "energy_marker": ENERGY_MARKER,
            "semantic_sha256": sha256_hex(
                {
                    "experiment": "E01",
                    "context": context.as_mapping(),
                    "primary": primary,
                }
            ),
        }
    ]


def run_e02_tamper_detection(
    context: ExperimentContext,
    chains: Mapping[str, Sequence[Block]],
    attack_instances: Mapping[str, AttackInstance],
    governed_records: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """E02: policy-specific tamper detection (THAT_POLICY_REJECTS_ATTACKED)."""
    raw_records: list[dict[str, Any]] = []
    workload_rc = risk_band_counts(context.order_ids, governed_records) if governed_records else None
    for scenario_id in PA_SCENARIOS:
        instance = attack_instances[scenario_id]
        tampered = instance.tampered_blocks
        head = tampered[-1]
        result, timing = time_policy_validation(context.policy, head, tampered, instance.target_order_id)
        detected = not result.accepted
        ops = deterministic_operation_counts(result, tampered)
        raw_records.append(
            raw_observation(
                context,
                blocks=len(tampered),
                risk_counts=workload_rc,
                attack=scenario_id,
                validation_outcome=result.accepted,
                detection_outcome=bool(detected),
                storage_bytes=canonical_storage_bytes(tampered),
                wall_time_ns=int(timing.wall_ns),
                cpu_time_ns=int(timing.cpu_ns),
                validation_check_count=ops["validation_check_count"],
                validator_invocation_count=ops["validator_invocation_count"],
                hash_operation_count=ops["hash_operation_count"],
            )
        )
    numerators = [int(r["detection_outcome"]) for r in raw_records]
    rate = pooled_rate(numerators, [1] * len(raw_records))
    return [
        {
            "experiment": "E02",
            "context": context.as_mapping(),
            "primary": {
                "policy_specific_detected_count": sum(numerators),
                "policy_specific_detection_rate": rate["rate"],
                "false_acceptance_count": len(raw_records) - sum(numerators),
                "false_acceptance_rate": (
                    1.0 - rate["rate"] if rate["rate"] is not None else None
                ),
                "denominator": len(raw_records),
            },
            "raw_records": raw_records,
            "timing_role": "SECONDARY_DESCRIPTIVE",
            "energy_marker": ENERGY_MARKER,
            "semantic_sha256": sha256_hex(
                {
                    "experiment": "E02",
                    "context": context.as_mapping(),
                    "detected_count": sum(numerators),
                    "denominator": len(raw_records),
                }
            ),
        }
    ]


def run_e03_localization(
    context: ExperimentContext,
    chains: Mapping[str, Sequence[Block]],
    attack_instances: Mapping[str, AttackInstance],
    governed_records: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """E03: unconditional correct-localization (undetected == NOT localized)."""
    raw_records: list[dict[str, Any]] = []
    workload_rc = risk_band_counts(context.order_ids, governed_records) if governed_records else None
    for scenario_id in PA_SCENARIOS:
        instance = attack_instances[scenario_id]
        tampered = instance.tampered_blocks
        head = tampered[-1]
        result, timing = time_policy_validation(context.policy, head, tampered, instance.target_order_id)
        detected = not result.accepted
        ops = deterministic_operation_counts(result, tampered)
        outcome = e1.detect_tamper(
            tampered,
            scenario_id,
            expected_order_id=instance.target_order_id,
            declared_length=len(tampered),
            applied_metadata=instance.applied_metadata,
        )
        localized = bool(detected and outcome.detected)
        localization = {
            "localized": localized,
            "detection_mechanism": outcome.detection_mechanism if detected else "undetected by evaluated policy",
            "localization": (
                outcome.localization
                if detected
                else "NOT_LOCALIZED_UNDETECTED_ATTACK"
            ),
            "failed_checks": list(outcome.failed_checks),
        }
        raw_records.append(
            raw_observation(
                context,
                blocks=len(tampered),
                risk_counts=workload_rc,
                attack=scenario_id,
                validation_outcome=result.accepted,
                detection_outcome=bool(detected),
                localization_outcome=localization,
                wall_time_ns=int(timing.wall_ns),
                cpu_time_ns=int(timing.cpu_ns),
                validation_check_count=ops["validation_check_count"],
                validator_invocation_count=ops["validator_invocation_count"],
                hash_operation_count=ops["hash_operation_count"],
            )
        )
    rate = pooled_rate([int(d["localization_outcome"]["localized"]) for d in raw_records], [1] * len(raw_records))
    return [
        {
            "experiment": "E03",
            "context": context.as_mapping(),
            "primary": {
                "unconditional_correct_localization_count": rate["numerator"],
                "unconditional_correct_localization_rate": rate["rate"],
                "denominator": len(raw_records),
            },
            "raw_records": raw_records,
            "timing_role": "SECONDARY_DESCRIPTIVE",
            "energy_marker": ENERGY_MARKER,
            "semantic_sha256": sha256_hex(
                {
                    "experiment": "E03",
                    "context": context.as_mapping(),
                    "localized_count": rate["numerator"],
                    "denominator": len(raw_records),
                }
            ),
        }
    ]


def run_e04_order_level_fault_isolation(
    context: ExperimentContext,
    chains: Mapping[str, Sequence[Block]],
    attack_instances: Mapping[str, AttackInstance],
    governed_records: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """E04: unaffected-order preservation (target affected, donors read-only)."""
    raw_records: list[dict[str, Any]] = []
    canonical_ids = set(context.order_ids)
    workload_rc = risk_band_counts(context.order_ids, governed_records) if governed_records else None
    for scenario_id in PA_SCENARIOS:
        instance = attack_instances[scenario_id]
        affected = {instance.target_order_id}
        unaffected = sorted(canonical_ids - affected)
        preserved = 0
        propagated = 0
        wall_ns_sum = 0
        cpu_ns_sum = 0
        for oid in unaffected:
            blocks = tuple(chains[oid])
            head = blocks[-1]
            result, timing = time_policy_validation(context.policy, head, blocks, oid)
            wall_ns_sum += timing.wall_ns
            cpu_ns_sum += timing.cpu_ns
            if result.accepted:
                preserved += 1
            else:
                propagated += 1
        denominator = len(unaffected)
        raw_records.append(
            raw_observation(
                context,
                blocks=len(tuple(chains[instance.target_order_id])),
                risk_counts=workload_rc,
                attack=scenario_id,
                validation_outcome=NA,
                detection_outcome=NA,
                localization_outcome={
                    "affected_order_ids": [instance.target_order_id],
                    "donor_order_id": instance.donor_order_id,
                    "donor_read_only": instance.donor_read_only,
                    "cross_order_propagation_count": propagated,
                    "unaffected_order_count": denominator,
                },
                wall_time_ns=int(wall_ns_sum),
                cpu_time_ns=int(cpu_ns_sum),
                validation_check_count=NA,
                validator_invocation_count=NA,
                hash_operation_count=NA,
            )
        )
    rate = pooled_rate(
        [int(r["localization_outcome"]["unaffected_order_count"] - r["localization_outcome"]["cross_order_propagation_count"]) for r in raw_records],
        [int(r["localization_outcome"]["unaffected_order_count"]) for r in raw_records],
    )
    return [
        {
            "experiment": "E04",
            "context": context.as_mapping(),
            "primary": {
                "unaffected_order_preservation_count": rate["numerator"],
                "unaffected_order_preservation_rate": rate["rate"],
                "denominator": rate["denominator"],
            },
            "raw_records": raw_records,
            "timing_role": "SECONDARY_DESCRIPTIVE",
            "energy_marker": ENERGY_MARKER,
            "semantic_sha256": sha256_hex(
                {
                    "experiment": "E04",
                    "context": context.as_mapping(),
                    "preservation_count": rate["numerator"],
                    "denominator": rate["denominator"],
                }
            ),
        }
    ]


def run_e05_ai_linked_adaptive_behavior(
    context: ExperimentContext,
    chains: Mapping[str, Sequence[Block]],
    governed_records: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """E05: P principal, GOVERNED_AI_RISK allocation, matched B0/B1 reference.

    No integrity-detection outcome is produced (E02 owns detection).
    """
    if context.policy != "P":
        raise V11E3Error("E05 principal evidence is policy P only")
    raw_records: list[dict[str, Any]] = []
    workload_rc = risk_band_counts(context.order_ids, governed_records) if governed_records else None
    allocation_by_level: dict[str, dict[str, Any]] = {
        level: {"validation_checks": 0, "validator_count": 0, "orders": 0}
        for level in RISK_LEVELS
    }
    level_dist: Counter[str] = Counter()
    for oid in context.order_ids:
        blocks = tuple(chains[oid])
        head = blocks[-1]
        record = governed_records.get(oid)
        governed_level = record["risk_level"] if record else e1.route_risk_levels([e1.synthetic_governed_record(oid, 0.5)])["MEDIUM"]
        result, timing = time_policy_validation("P", head, blocks, oid)
        ops = deterministic_operation_counts(result, blocks)
        level_dist[governed_level] += 1
        allocation_by_level[governed_level]["validation_checks"] += len(result.applied_checks)
        allocation_by_level[governed_level]["validator_count"] += result.validator_count
        allocation_by_level[governed_level]["orders"] += 1
        raw_records.append(
            raw_observation(
                context,
                blocks=len(blocks),
                risk_counts=workload_rc,
                validation_outcome=result.accepted,
                wall_time_ns=int(timing.wall_ns),
                cpu_time_ns=int(timing.cpu_ns),
                validation_check_count=ops["validation_check_count"],
                validator_invocation_count=ops["validator_invocation_count"],
                hash_operation_count=ops["hash_operation_count"],
            )
        )
    allocation = {
        level: {
            "validation_checks": allocation_by_level[level]["validation_checks"],
            "validator_count": allocation_by_level[level]["validator_count"],
            "orders": allocation_by_level[level]["orders"],
        }
        for level in RISK_LEVELS
        if allocation_by_level[level]["orders"]
    }
    matched: dict[str, dict[str, Any]] = {}
    for policy in ("B0", "B1"):
        checks: Counter[str] = Counter()
        validators = 0
        for oid in context.order_ids:
            blocks = tuple(chains[oid])
            result = validate_under_policy(policy, blocks[-1], blocks, oid)
            for check in result.applied_checks:
                checks[check] += 1
            validators += result.validator_count
        matched[policy] = {"check_allocation": dict(checks), "validator_count": validators}
    return [
        {
            "experiment": "E05",
            "context": context.as_mapping(),
            "primary": {
                "applied_validation_check_allocation_by_governed_risk": allocation,
                "validator_allocation_by_governed_risk": {
                    level: allocation[level]["validator_count"] for level in allocation
                },
                "validation_level_distribution": dict(level_dist),
            },
            "matched_reference": matched,
            "raw_records": raw_records,
            "timing_role": "SECONDARY_DESCRIPTIVE",
            "integrity_detection": "NO",
            "energy_marker": ENERGY_MARKER,
            "semantic_sha256": sha256_hex(
                {
                    "experiment": "E05",
                    "context": context.as_mapping(),
                    "allocation": allocation,
                    "matched_reference": matched,
                }
            ),
        }
    ]


def run_e06_execution_time_overhead(
    context: ExperimentContext,
    chains: Mapping[str, Sequence[Block]],
    governed_records: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """E06: validation-only wall + cpu runtime over clean chains (principal)."""
    raw_records: list[dict[str, Any]] = []
    workload_rc = risk_band_counts(context.order_ids, governed_records) if governed_records else None
    for oid in context.order_ids:
        blocks = tuple(chains[oid])
        head = blocks[-1]
        result, timing = time_policy_validation(context.policy, head, blocks, oid)
        ops = deterministic_operation_counts(result, blocks)
        raw_records.append(
            raw_observation(
                context,
                blocks=len(blocks),
                risk_counts=workload_rc,
                validation_outcome=result.accepted,
                wall_time_ns=int(timing.wall_ns),
                cpu_time_ns=int(timing.cpu_ns),
                validation_check_count=ops["validation_check_count"],
                validator_invocation_count=ops["validator_invocation_count"],
                hash_operation_count=ops["hash_operation_count"],
            )
        )
    wall = [float(r["wall_time_ns"]) for r in raw_records]
    cpu = [float(r["cpu_time_ns"]) for r in raw_records]
    return [
        {
            "experiment": "E06",
            "context": context.as_mapping(),
            "primary": {
                "validation_only_wall_clock_runtime_ns": describe_continuous(wall),
                "validation_only_cpu_runtime_ns": describe_continuous(cpu),
                "per_order_validation_time_ns": describe_continuous(
                    [w / max(1, len(context.order_ids)) for w in wall]
                ),
            },
            "raw_records": raw_records,
            "timing_principal": True,
            "timing_role": "PRINCIPAL",
            "energy_marker": ENERGY_MARKER,
            "semantic_sha256": sha256_hex(
                {
                    "experiment": "E06",
                    "context": context.as_mapping(),
                    "wall_mean_ns": describe_continuous(wall)["mean"],
                }
            ),
        }
    ]


def run_e07_memory_overhead(
    context: ExperimentContext,
    chains: Mapping[str, Sequence[Block]],
    governed_records: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """E07: peak traced allocation proxy per policy over clean chains.

    Uses a fresh worker process (the E07 frozen contract). Deterministic over
    the already-materialized chains; label COMPUTATIONAL_MEMORY_PROXY, MiB.
    """
    raw_records: list[dict[str, Any]] = []
    workload_rc = risk_band_counts(context.order_ids, governed_records) if governed_records else None
    for oid in context.order_ids:
        blocks = tuple(chains[oid])
        head = blocks[-1]
        peak_mib, meta = memory_proxy_fresh_worker(context.policy, head, blocks, oid)
        result = validate_under_policy(context.policy, head, blocks, oid)
        ops = deterministic_operation_counts(result, blocks)
        raw_records.append(
            raw_observation(
                context,
                blocks=len(blocks),
                risk_counts=workload_rc,
                validation_outcome=result.accepted,
                memory_proxy_mib=round(float(peak_mib), 6),
                validation_check_count=ops["validation_check_count"],
                validator_invocation_count=ops["validator_invocation_count"],
                hash_operation_count=ops["hash_operation_count"],
            )
        )
    values = [float(r["memory_proxy_mib"]) for r in raw_records]
    return [
        {
            "experiment": "E07",
            "context": context.as_mapping(),
            "primary": {
                "peak_traced_python_allocation_during_validation_mib": describe_continuous(values),
            },
            "measurement": "tracemalloc_fresh_worker",
            "unit": "MiB",
            "label": MEMORY_PROXY_LABEL,
            "raw_records": raw_records,
            "energy_marker": ENERGY_MARKER,
            "semantic_sha256": sha256_hex(
                {
                    "experiment": "E07",
                    "context": context.as_mapping(),
                    "peak_mean_mib": describe_continuous(values)["mean"],
                }
            ),
        }
    ]


def run_e08_storage_overhead(
    context: ExperimentContext,
    chains: Mapping[str, Sequence[Block]],
) -> dict[str, Any]:
    """E08: policy-independent canonical storage bytes per workload."""
    observation = storage_observation(context.order_ids, chains)
    return {
        "experiment": "E08",
        "context": context.as_mapping(),
        "primary": {
            "canonical_serialized_blockchain_bytes_per_workload": observation[
                "canonical_blockchain_bytes"
            ],
            "bytes_per_order": observation["bytes_per_order"],
            "bytes_per_block": observation["bytes_per_block"],
        },
        "policy_independent": True,
        "note": "no B0/B1/P storage variants; validation does not mutate representation",
        "energy_marker": ENERGY_MARKER,
        "semantic_sha256": sha256_hex(
            {
                "experiment": "E08",
                "context": context.as_mapping(),
                "bytes": observation["canonical_blockchain_bytes"],
            }
        ),
    }


def run_e09_throughput(
    context: ExperimentContext,
    chains: Mapping[str, Sequence[Block]],
    governed_records: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """E09: validated orders/second (and blocks/second) over clean chains."""
    raw_records: list[dict[str, Any]] = []
    total_wall = 0
    total_blocks = 0
    workload_rc = risk_band_counts(context.order_ids, governed_records) if governed_records else None
    for oid in context.order_ids:
        blocks = tuple(chains[oid])
        head = blocks[-1]
        result, timing = time_policy_validation(context.policy, head, blocks, oid)
        total_wall += timing.wall_ns
        total_blocks += len(blocks)
        ops = deterministic_operation_counts(result, blocks)
        raw_records.append(
            raw_observation(
                context,
                blocks=len(blocks),
                risk_counts=workload_rc,
                validation_outcome=result.accepted,
                wall_time_ns=int(timing.wall_ns),
                cpu_time_ns=int(timing.cpu_ns),
                validation_check_count=ops["validation_check_count"],
                validator_invocation_count=ops["validator_invocation_count"],
                hash_operation_count=ops["hash_operation_count"],
            )
        )
    orders = len(context.order_ids)
    wall_seconds = total_wall / 1e9 if total_wall else None
    throughput_orders = orders / wall_seconds if wall_seconds else None
    throughput_blocks = total_blocks / wall_seconds if wall_seconds else None
    return [
        {
            "experiment": "E09",
            "context": context.as_mapping(),
            "primary": {
                "validated_orders_per_second": throughput_orders,
                "validated_blocks_per_second": throughput_blocks,
                "total_validated_orders": orders,
                "total_validated_blocks": total_blocks,
            },
            "raw_records": raw_records,
            "energy_marker": ENERGY_MARKER,
            "semantic_sha256": sha256_hex(
                {
                    "experiment": "E09",
                    "context": context.as_mapping(),
                    "orders": orders,
                    "blocks": total_blocks,
                }
            ),
        }
    ]


def _require_cache_cell(cell: Mapping[str, Any], experiment: str, keys: Sequence[str]) -> None:
    """Fail closed if a cached E06-E09 output is absent, mislabeled, or partial."""
    if not isinstance(cell, Mapping) or not cell:
        raise V11E3Error(
            f"E10 derivation requires cached {experiment} results; got {type(cell).__name__}"
        )
    if cell.get("experiment") != experiment:
        raise V11E3Error(
            f"E10 derivation received {cell.get('experiment')!r}; requires {experiment}"
        )
    missing = [key for key in keys if key not in cell]
    if missing:
        raise V11E3Error(f"E10 derivation: {experiment} cache missing fields {sorted(missing)}")


def _require_cache_primary(cell: Mapping[str, Any], key: str, kind: str = "scalar") -> None:
    primary = cell.get("primary")
    if not isinstance(primary, Mapping) or key not in primary:
        raise V11E3Error(
            f"E10 derivation: cached {cell.get('experiment')} missing primary field {key!r}"
        )
    value = primary[key]
    if kind == "mapping":
        if not isinstance(value, Mapping) or not value:
            raise V11E3Error(
                f"E10 derivation: cached {cell.get('experiment')} {key!r} is not a non-empty summary"
            )
        return
    if value is not None and not isinstance(value, (int, float)):
        raise V11E3Error(
            f"E10 derivation: cached {cell.get('experiment')} {key!r} is not numeric"
        )


def derive_e10_resource_green_proxy(
    e06_results: Mapping[str, Any],
    e07_results: Mapping[str, Any],
    e08_results: Mapping[str, Any],
    e09_results: Mapping[str, Any],
    context: ExperimentContext,
) -> dict[str, Any]:
    """E10: derived ONLY from cached E06-E09 outputs + deterministic op counts.

    Never invokes any measurement runner, timing harness, tracemalloc worker,
    or storage campaign. If the required cached E06-E09 results are absent or
    incomplete it FAILS CLOSED rather than silently regenerating them.
    """
    _require_cache_cell(e06_results, "E06", ("experiment", "primary", "raw_records"))
    _require_cache_cell(e07_results, "E07", ("experiment", "primary"))
    _require_cache_cell(e08_results, "E08", ("experiment", "primary"))
    _require_cache_cell(e09_results, "E09", ("experiment", "primary"))
    _require_cache_primary(e06_results, "validation_only_wall_clock_runtime_ns", kind="mapping")
    _require_cache_primary(e07_results, "peak_traced_python_allocation_during_validation_mib", kind="mapping")
    _require_cache_primary(e08_results, "canonical_serialized_blockchain_bytes_per_workload", kind="scalar")
    _require_cache_primary(e09_results, "validated_orders_per_second", kind="scalar")
    all_records = e06_results.get("raw_records")
    if not isinstance(all_records, list) or not all_records:
        raise V11E3Error("E10 derivation: E06 cache raw_records empty or invalid")
    for record in all_records:
        for key in ("validation_check_count", "validator_invocation_count", "hash_operation_count"):
            if not isinstance(record.get(key), int):
                raise V11E3Error(f"E10 derivation: E06 cache raw record missing int field {key!r}")
    check_count = sum(int(r["validation_check_count"]) for r in all_records)
    validator_count = sum(int(r["validator_invocation_count"]) for r in all_records)
    hash_count = sum(int(r["hash_operation_count"]) for r in all_records)
    return {
        "experiment": "E10",
        "context": context.as_mapping(),
        "derived_only": True,
        "no_fourth_campaign": True,
        "primary_computational_proxies": {
            "runtime_proxy": e06_results["primary"]["validation_only_wall_clock_runtime_ns"],
            "validation_check_count": check_count,
            "validator_invocation_count": validator_count,
            "measurable_hash_operation_count": hash_count,
        },
        "memory_evidence": e07_results["primary"]["peak_traced_python_allocation_during_validation_mib"],
        "storage_evidence": e08_results["primary"]["canonical_serialized_blockchain_bytes_per_workload"],
        "throughput_evidence": e09_results["primary"]["validated_orders_per_second"],
        "direct_energy": ENERGY_MARKER,
        "never_report": list(ENERGY_FORBIDDEN_TOKENS),
        "energy_marker": ENERGY_MARKER,
        "semantic_sha256": sha256_hex(
            {
                "experiment": "E10",
                "context": context.as_mapping(),
                "check_count": check_count,
                "hash_count": hash_count,
            }
        ),
    }


ALL_EXPERIMENTS = {
    "E01": run_e01_functional_correctness,
    "E02": run_e02_tamper_detection,
    "E03": run_e03_localization,
    "E04": run_e04_order_level_fault_isolation,
    "E05": run_e05_ai_linked_adaptive_behavior,
    "E06": run_e06_execution_time_overhead,
    "E07": run_e07_memory_overhead,
    "E08": run_e08_storage_overhead,
    "E09": run_e09_throughput,
    "E10": derive_e10_resource_green_proxy,
}


def run_ordered_experiments(
    context: ExperimentContext,
    chains: Mapping[str, Sequence[Block]],
    attack_instances: Mapping[str, AttackInstance] | None,
    records: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run one experiment cell dispatch (callers choose the cell).

    Dispatch is explicit and exact: E01-E09 call the matching runner, E10 is
    derived from the E06-E09 outputs. The caller is responsible for ordering
    cells by the frozen counterbalanced seed policy order and for materializing
    chains/attacks before any timing call (see the timing boundary).
    """
    eid = context.experiment_id
    governed_records = records or {}
    if eid == "E01":
        return run_e01_functional_correctness(context, chains, governed_records or None)
    if eid == "E02":
        if attack_instances is None:
            raise V11E3Error("E02 requires attack instances")
        return run_e02_tamper_detection(context, chains, attack_instances, governed_records or None)
    if eid in ("E03", "E04"):
        if attack_instances is None:
            raise V11E3Error(f"{eid} requires attack instances")
        if eid == "E03":
            return run_e03_localization(context, chains, attack_instances, governed_records or None)
        return run_e04_order_level_fault_isolation(context, chains, attack_instances, governed_records or None)
    if eid == "E05":
        return run_e05_ai_linked_adaptive_behavior(context, chains, governed_records)
    if eid == "E06":
        return run_e06_execution_time_overhead(context, chains, governed_records or None)
    if eid == "E07":
        return run_e07_memory_overhead(context, chains, governed_records or None)
    if eid == "E08":
        return [run_e08_storage_overhead(context, chains)]
    if eid == "E09":
        return run_e09_throughput(context, chains, governed_records or None)
    if eid == "E10":
        raise V11E3Error(E10_DERIVED_ONLY_MSG)
    raise V11E3Error(f"unknown experiment id {eid!r}")


def e2_semantic_lock_sha256() -> str:
    """Recompute the E2 protocol-lock semantic sha from its locked payload."""
    lock = _read_json(E2_PROTOCOL_LOCK_PATH)
    if lock.get("stage") != "V1.1-E2B":
        raise V11E3PreflightError("E2 protocol lock stage mismatch.")
    semantic = lock.get("semantic_payload")
    if not isinstance(semantic, dict):
        raise V11E3PreflightError("E2 protocol lock is missing its semantic_payload.")
    recomputed = sha256_hex(semantic)
    recorded = lock.get("semantic_result_lock_sha256")
    if recorded != recomputed:
        raise V11E3PreflightError("E2 protocol lock semantic sha drift.")
    if recomputed != E2_PROTOCOL_LOCK_SEMANTIC_SHA256:
        raise V11E3PreflightError("E2 protocol lock semantic sha drifted from the frozen constant.")
    return recomputed


def modern_prereq_sampler_smoke(order_ids: Sequence[Any], *, size: int = 4) -> dict[str, Any]:
    """Tiny deterministic sampler smoke over explicit order ids (test-only)."""
    if len(order_ids) < size:
        raise V11E3Error("sampler smoke requires at least the requested size")
    workloads = nested_workloads(BLOCKCHAIN_SEEDS[0], order_ids, sizes=[size])
    return {
        "seed": BLOCKCHAIN_SEEDS[0],
        "ranking": list(rank_order_ids(BLOCKCHAIN_SEEDS[0], order_ids)),
        "workload": list(workloads[size]),
        "deterministic": True,
    }


# --------------------------------------------------------------------------- #
# E3 preflight (fail-closed gates, read-only)
# --------------------------------------------------------------------------- #


def _git_rev(rev: str) -> str:
    try:
        raw = subprocess.run(
            ["git", "rev-parse", rev],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as exc:
        raise V11E3PreflightError(f"git required to guard the E3 checkpoint: {exc}") from exc
    if not raw:
        raise V11E3PreflightError(f"could not resolve git revision {rev!r}")
    return raw


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def live_governed_artifact_gate() -> dict[str, Any]:
    """Read-only live re-verification of the governed V1.1-D3 artifact/population.

    Reuses the frozen V1.1-E1 integration preflight in its documented post-stage
    mode (``checkpoint_guard=False``): the historical E1 stage guard pinned its
    own starting checkpoint, while current-HEAD runtime authorization is owned
    by the V1.1-E4L launcher (operator-supplied ``--authorized-commit`` with
    ``HEAD == origin/main == authorized_commit`` enforced at launch time, never
    hard-coded here). Fails closed on any drift of the population size, risk
    bands, or record-digest reverification.
    """
    upstream = e1.run_v11e1_preflight(checkpoint_guard=False)
    return {
        "governed_records": upstream.artifact_record_count,
        "low": upstream.low,
        "medium": upstream.medium,
        "high": upstream.high,
        "verified_record_digests": upstream.verified_record_digests,
        "d1_lock_semantic_sha256": upstream.d1_lock_semantic_sha256,
        "v11c_lock_semantic_sha256": upstream.v11c_lock_semantic_sha256,
        "d3_result_lock_semantic_sha256": upstream.d3_result_lock_semantic_sha256,
        "artifact_semantic_sha256": upstream.artifact_semantic_sha256,
        "artifact_disk_sha256": upstream.artifact_file_sha256,
        "summary_semantic_sha256": upstream.summary_semantic_sha256,
        "summary_disk_sha256": upstream.summary_file_sha256,
        "test_access_count": upstream.test_access_count,
    }


def run_v11e3_preflight(*, checkpoint_guard: bool = True) -> dict[str, Any]:
    """Fail-closed E3 readiness gates over the frozen upstream + E2 artifacts.

    ``checkpoint_guard=True`` asserts the historical E3 stage guard
    (``HEAD == origin/main == EXPECTED_HEAD``), pinning the checkpoint for which
    E3 was originally implemented. ``checkpoint_guard=False`` is the documented
    post-stage mode: it skips git entirely and reruns the frozen semantic/data
    gates (live governed artifact re-verification, D3 provenance checks,
    population + risk-band distribution checks, TEST/AI governance counters),
    making no claim about the current HEAD. Current-HEAD runtime authorization
    is enforced by the V1.1-E4L launcher against the operator-supplied
    authorized execution commit, never hard-coded in this module.
    """
    evidence: dict[str, Any] = {}
    evidence["stage"] = STAGE
    evidence["protocol_version"] = PROTOCOL_VERSION
    if checkpoint_guard:
        head = _git_rev("HEAD")
        origin_main = _git_rev("origin/main")
    else:
        head = origin_main = EXPECTED_HEAD
    evidence["head"] = head
    evidence["origin_main"] = origin_main
    evidence["expected_head"] = EXPECTED_HEAD
    if head != EXPECTED_HEAD or origin_main != EXPECTED_HEAD:
        raise V11E3PreflightError("E3 checkpoint guard failed (HEAD/origin drift).")

    semantic = e2_semantic_lock_sha256()
    evidence["e2_protocol_lock_semantic_sha256"] = semantic
    evidence["e2_config_fingerprint"] = _sha256_file(E2_PROTOCOL_CONFIG_PATH)
    evidence["e2_document_fingerprint"] = _sha256_file(E2_PROTOCOL_DOC_PATH)
    if evidence["e2_config_fingerprint"] != E2_CONFIG_FINGERPRINT:
        raise V11E3PreflightError("E2 config fingerprint drifted from the frozen constant.")
    if evidence["e2_document_fingerprint"] != E2_DOC_FINGERPRINT:
        raise V11E3PreflightError("E2 document fingerprint drifted from the frozen constant.")

    lock = _read_json(E2_PROTOCOL_LOCK_PATH)
    exec_meta = lock.get("execution_metadata", {})
    evidence["experiment_status"] = exec_meta.get("experiment_status")
    evidence["test_access_count"] = int(exec_meta.get("test_access_count", -1))
    evidence["ai_fit_count"] = int(exec_meta.get("ai_fit_count", -1))
    evidence["ai_inference_count"] = int(exec_meta.get("ai_inference_count", -1))
    evidence["experiment_execution_count"] = int(exec_meta.get("experiment_execution_count", -1))
    if evidence["experiment_status"] != "NOT_EXECUTED":
        raise V11E3PreflightError("E2 lock reports executed experiments.")
    if any(
        evidence[key] != 0
        for key in ("test_access_count", "ai_fit_count", "ai_inference_count", "experiment_execution_count")
    ):
        raise V11E3PreflightError("E2 lock reports non-zero TEST/AI/experiment activity.")

    upstream = lock.get("semantic_payload", {}).get("upstream", {})
    provenance = _upstream_provenance()
    drift = [
        key
        for key, value in upstream.items()
        if key in provenance and upstream[key] != provenance[key]
    ]
    if drift:
        raise V11E3PreflightError(f"E2 lock upstream provenance drifted: {sorted(drift)}")
    evidence["upstream_provenance_verified"] = True

    live = live_governed_artifact_gate()
    evidence.update({f"live_{key}": val for key, val in live.items()})
    evidence["live_governed_artifact_gate_passed"] = True
    if live["governed_records"] != e1.EXPECTED_ORDERS:
        raise V11E3PreflightError("live governed population count drifted.")
    if (live["low"], live["medium"], live["high"]) != (e1.EXPECTED_LOW, e1.EXPECTED_MEDIUM, e1.EXPECTED_HIGH):
        raise V11E3PreflightError("live governed risk-band distribution drifted.")
    if live["verified_record_digests"] != e1.EXPECTED_ORDERS:
        raise V11E3PreflightError("live governed record-digest reverification incomplete.")
    if live["test_access_count"] != 0:
        raise V11E3PreflightError("live upstream verification reports TEST access.")
    if live["d3_result_lock_semantic_sha256"] != e1.D3_RESULT_LOCK_SEMANTIC:
        raise V11E3PreflightError("live D3 result-lock semantic hash drifted.")
    if live["artifact_semantic_sha256"] != e1.ARTIFACT_SEMANTIC_SHA256:
        raise V11E3PreflightError("live governed artifact semantic hash drifted.")
    if live["artifact_disk_sha256"] != e1.ARTIFACT_DISK_SHA256:
        raise V11E3PreflightError("live governed artifact disk hash drifted.")
    if live["summary_semantic_sha256"] != e1.SUMMARY_SEMANTIC_SHA256:
        raise V11E3PreflightError("live summary semantic hash drifted.")
    if live["summary_disk_sha256"] != e1.SUMMARY_DISK_SHA256:
        raise V11E3PreflightError("live summary disk hash drifted.")

    evidence["marker"] = E3_MARKER
    evidence["semantic_sha256"] = sha256_hex({key: val for key, val in evidence.items() if key != "semantic_sha256"})
    return evidence


# --------------------------------------------------------------------------- #
# synthetic dry-run harness (§13): tiny labeled fixtures, branch coverage only
# --------------------------------------------------------------------------- #


def synthetic_workload_fixture(
    order_ids: Sequence[Any],
    *,
    risk_levels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build tiny labeled synthetic chains + risk records (tests only)."""
    canonical_ids = [canonical_order_id(oid) for oid in order_ids]
    levels = list(risk_levels or ["MEDIUM"] * len(canonical_ids))
    if len(levels) != len(canonical_ids):
        raise V11E3Error("synthetic risk_levels length mismatch")
    chains: dict[str, tuple[Block, ...]] = {}
    records: dict[str, dict[str, Any]] = {}
    for index, oid in enumerate(canonical_ids):
        level = levels[index]
        score = {"LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.9}[level]
        chain = e1.build_synthetic_chain(oid, risk_level=level, risk_score=score)
        chains[oid] = chain.blocks
        record = e1.synthetic_governed_record(oid, score)
        records[oid] = {"order_id": oid, "risk_level": level, "risk_score": score}
    return {"order_ids": canonical_ids, "chains": chains, "records": records}


def _derive_e10_from_seed_cache(
    per_seed: Mapping[str, Any], seed: int, fixture: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Derive each E10 cell from the already-run E06-E09 seed cells (no re-run)."""
    e08_cell = per_seed["E08"][0]
    e10_cells: list[dict[str, Any]] = []
    for idx, policy in enumerate(policy_execution_order(seed)):
        context = context_for(
            "E10",
            seed,
            len(fixture["order_ids"]),
            fixture["order_ids"],
            policy,
        )
        e10_cells.append(
            derive_e10_resource_green_proxy(
                per_seed["E06"][idx],
                per_seed["E07"][idx],
                e08_cell,
                per_seed["E09"][idx],
                context,
            )
        )
    return e10_cells


def dry_run_all_experiments(
    order_ids: Sequence[Any],
    *,
    seeds: Sequence[int] = (BLOCKCHAIN_SEEDS[0],),
    risk_levels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run every E01-E10 cell over tiny synthetic fixtures (branch coverage).

    In-memory only: no file writes, no TEST access, no governed workloads, no
    AI fit/infer. Every experiment cell for every policy in the requested seeds
    is exercised so the machinery is fully branch-covered before any governed
    run is authorized. E10 is always derived from the cached E06-E09 cells run
    once in the same loop; it never re-runs a measurement campaign.
    """
    fixture = synthetic_workload_fixture(order_ids, risk_levels=risk_levels)
    results: dict[str, Any] = {}
    for seed in seeds:
        per_seed: dict[str, Any] = {}
        for experiment_id in ("E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09"):
            policies = policy_execution_order(seed)
            if experiment_id == "E05":
                policies = ("P",)
            if experiment_id == "E08":
                policies = ("B0",)
            per_experiment: list[dict[str, Any]] = []
            for policy in policies:
                context = context_for(
                    experiment_id,
                    seed,
                    len(fixture["order_ids"]),
                    fixture["order_ids"],
                    policy,
                )
                if experiment_id == "E02":
                    instances = attack_instances_for_workload(
                        fixture["order_ids"], fixture["chains"], seed
                    )
                    per_experiment.extend(run_e02_tamper_detection(context, fixture["chains"], instances, fixture["records"]))
                elif experiment_id == "E03":
                    instances = attack_instances_for_workload(
                        fixture["order_ids"], fixture["chains"], seed
                    )
                    per_experiment.extend(run_e03_localization(context, fixture["chains"], instances, fixture["records"]))
                elif experiment_id == "E04":
                    instances = attack_instances_for_workload(
                        fixture["order_ids"], fixture["chains"], seed
                    )
                    per_experiment.extend(run_e04_order_level_fault_isolation(context, fixture["chains"], instances, fixture["records"]))
                elif experiment_id == "E01":
                    per_experiment.extend(run_e01_functional_correctness(context, fixture["chains"], fixture["records"]))
                elif experiment_id == "E05":
                    per_experiment.extend(run_e05_ai_linked_adaptive_behavior(context, fixture["chains"], fixture["records"]))
                elif experiment_id == "E08":
                    per_experiment.append(run_e08_storage_overhead(context, fixture["chains"]))
                elif experiment_id == "E06":
                    per_experiment.extend(run_e06_execution_time_overhead(context, fixture["chains"], fixture["records"]))
                elif experiment_id == "E09":
                    per_experiment.extend(run_e09_throughput(context, fixture["chains"], fixture["records"]))
                elif experiment_id == "E07":
                    per_experiment.extend(run_e07_memory_overhead(context, fixture["chains"], fixture["records"]))
                else:  # pragma: no cover - defensive
                    raise V11E3Error(f"unhandled experiment {experiment_id!r}")
            per_seed[experiment_id] = per_experiment
        per_seed["E10"] = _derive_e10_from_seed_cache(per_seed, seed, fixture)
        results[str(seed)] = per_seed
    return {
        "stage": STAGE,
        "kind": KIND,
        "mode": "SYNTHETIC_DRY_RUN",
        "no_file_writes": True,
        "no_test_access": True,
        "no_governed_workloads": True,
        "no_ai_fit_infer": True,
        "results": results,
        "marker": E3_MARKER,
    }


# --------------------------------------------------------------------------- #
# readiness report + CLI
# --------------------------------------------------------------------------- #


def build_e3_readiness_report(*, checkpoint_guard: bool = True) -> dict[str, Any]:
    """Assemble the E3 machinery-ready report (read-only; no experiments)."""
    preflight = run_v11e3_preflight(checkpoint_guard=checkpoint_guard)
    sampler = modern_prereq_sampler_smoke(["1001", "1002", "1003", "1004"], size=4)
    return {
        "stage": STAGE,
        "kind": KIND,
        "protocol_version": PROTOCOL_VERSION,
        "preflight": preflight,
        "sampler_smoke": sampler,
        "machinery_ready": True,
        "no_experiments_executed": True,
        "no_resources_written": True,
        "marker": E3_MARKER,
    }


def main() -> None:
    """Print the E3 machinery-ready report (read-only; never runs experiments)."""
    report = build_e3_readiness_report()
    import json as _json

    print(f"{KIND} machinery readiness (read-only; no experiments executed)")
    print("PROTOCOL:", report["protocol_version"])
    print("PREFLIGHT:", _json.dumps(report["preflight"], sort_keys=True))
    print(f"MARKER: {report['marker']}")


if __name__ == "__main__":
    main()