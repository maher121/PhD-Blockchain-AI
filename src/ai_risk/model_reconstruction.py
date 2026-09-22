"""Deterministic TRAIN-only model reconstruction contract for V1.1-D.

V1.1-D2 authorizes NO full reconstruction, fitting, prediction, or test-path
execution. This module therefore exposes only the frozen contract, partition
gates, and re-verify-able fingerprints; every entry point that would consume
the real frozen TRAIN workloads is guarded so the D2 stage cannot launch a
campaign. Reconstruction specification is validated against the frozen config
cycles, not against any fitted model state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .protocol import AIRiskProtocol

PARTITION_TRAIN = "TRAIN"
PARTITION_VALIDATION = "VALIDATION"
PARTITION_TEST = "TEST"

FROZEN_FINGERPRINT_NAMES = frozenset(
    {
        "training_row_ids_sha256",
        "training_clean_features_sha256",
        "training_attacked_features_sha256",
        "training_labels_sha256",
    }
)


class ReconstructionNotAuthorizedError(PermissionError):
    """Raised when reconstruction operations are attempted before authorization."""


class PartitionGuardError(ValueError):
    """Raised when a frozen partition gate is violated."""


@dataclass(frozen=True)
class ReconstructionSpec:
    """The frozen, D1-locked reconstruction contract (no fitted state)."""

    source_contract: str
    source_fingerprint_artifact: str
    fit_split: str
    fit_rows: int
    attack_type: str
    attack_rate: float
    attack_severity: str
    attack_config: str
    seeds: tuple[int, ...]
    fingerprint_requirements: frozenset[str]
    verification_rule: str

    @classmethod
    def from_protocol(cls, protocol: AIRiskProtocol) -> "ReconstructionSpec":
        section = protocol.config["training_reconstruction"]
        return cls(
            source_contract=section["source_contract"],
            source_fingerprint_artifact=section["source_fingerprint_artifact"],
            fit_split=section["fit_split"],
            fit_rows=int(section["fit_rows"]),
            attack_type=section["attack_type"],
            attack_rate=float(section["attack_rate"]),
            attack_severity=section["attack_severity"],
            attack_config=section["attack_config"],
            seeds=tuple(int(seed) for seed in section["attack_and_model_seeds"]),
            fingerprint_requirements=frozenset(section["fingerprint_requirements"]),
            verification_rule=section["verification_rule"],
        )


def validate_reconstruction_spec(spec: ReconstructionSpec) -> None:
    """Fail closed unless every frozen reconstruction invariant holds."""
    if spec.fit_split != PARTITION_TRAIN:
        raise PartitionGuardError(
            f"reconstruction fit split must be {PARTITION_TRAIN}, got {spec.fit_split!r}"
        )
    if spec.fit_rows != 28000:
        raise PartitionGuardError(
            f"reconstruction fit rows must be 28000, got {spec.fit_rows}"
        )
    expected_seeds = tuple(range(42, 47))
    if spec.seeds != expected_seeds:
        raise PartitionGuardError(
            f"reconstruction seeds must be {expected_seeds}, got {spec.seeds}"
        )
    if not FROZEN_FINGERPRINT_NAMES.issubset(spec.fingerprint_requirements):
        missing = FROZEN_FINGERPRINT_NAMES - spec.fingerprint_requirements
        raise PartitionGuardError(
            f"reconstruction is missing frozen fingerprint requirements: {sorted(missing)}"
        )


def require_reconstruction_authorization(*, authorized: bool = False) -> None:
    """Gate the reconstruction contract; D2 never auto-authorizes."""
    if not authorized:
        raise ReconstructionNotAuthorizedError(
            "V1.1-D2 authorizes no governed reconstruction; a separately "
            "authorized implementation stage must grant the capability."
        )


def assert_fit_split(split: str) -> None:
    """Reject any non-TRAIN reconstruction or fit input, fail closed."""
    if split != PARTITION_TRAIN:
        raise PartitionGuardError(
            f"reconstruction/fit input must be {PARTITION_TRAIN}, got {split!r}"
        )


def assert_generation_split(split: str) -> None:
    """Reject any non-VALIDATION generation input, fail closed."""
    if split != PARTITION_VALIDATION:
        raise PartitionGuardError(
            f"generation input must be {PARTITION_VALIDATION}, got {split!r}"
        )


def assert_test_blocked(split: str) -> None:
    """Reject TEST partition inputs absolutely (must remain unopen)."""
    if split == PARTITION_TEST:
        raise PartitionGuardError(
            "TEST partition access is forbidden by the frozen V1.1-D gate"
        )


def verify_fingerprint_names(present: Iterable[str]) -> None:
    """Confirm the frozen TRAIN fingerprint set is fully present."""
    present_set = frozenset(present)
    missing = FROZEN_FINGERPRINT_NAMES - present_set
    if missing:
        raise PartitionGuardError(
            f"frozen reconstruction fingerprint names are missing: {sorted(missing)}"
        )


def reconstruct_member_models(
    protocol: AIRiskProtocol,
    *,
    authorization_capability: object | None = None,
    workloads: Sequence[Any] | None = None,
) -> list[Any]:
    """D2 stub: full reconstruction is out of scope and always fails closed."""
    if authorization_capability is None:
        raise ReconstructionNotAuthorizedError(
            "full V1.1-D model reconstruction is out of the D2 implementation "
            "scope and requires a future authorized capability"
        )
    if workloads is None:
        raise ReconstructionNotAuthorizedError(
            "reconstruction requires an authorized frozen TRAIN workload source"
        )
    raise NotImplementedError(
        "V1.1-D2 does not implement the full governed reconstruction campaign"
    )