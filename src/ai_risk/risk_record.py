"""Frozen governing risk-record schema and deterministic digest.

Implements the D1-locked risk record exactly: every required field in the
frozen order, SHA-256 digest over the canonical JSON of all required fields
except ``record_digest``, and a stable digest over the sorted contributing
stable row identifiers. Records are entirely off-chain; the on-chain linkage
is a separate minimal reference built in ``blockchain_linkage``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from ..blockchain_engine.canonical import sha256_hex
from .order_aggregation import OrderAggregate
from .protocol import AIRiskProtocol
from .risk_levels import RISK_LEVEL_LOW, risk_level_for_score

SCHEMA_VERSION = "v1.1-d-governed-order-risk-1"

REQUIRED_FIELDS_IN_ORDER = (
    "schema_version",
    "order_id",
    "risk_score",
    "risk_level",
    "source_row_count",
    "source_rows_digest",
    "classifier",
    "model_strategy",
    "model_seeds",
    "aggregation_rule",
    "source_partition",
    "score_semantics_version",
    "model_configuration_provenance",
    "hybrid_k13_provenance",
    "generation_stage_provenance",
    "record_digest",
)


class RiskRecordError(ValueError):
    """Raised when a risk record violates the frozen schema."""


def _validated_score(score: float) -> float:
    value = float(score)
    if math.isnan(value) or math.isinf(value):
        raise RiskRecordError(f"risk score must be finite, got {score!r}")
    if value < 0.0 or value > 1.0:
        raise RiskRecordError(f"risk score out of the frozen [0,1] range: {value!r}")
    return value


@dataclass(frozen=True)
class RiskRecord:
    """One governed per-order risk record (off-chain authority)."""

    schema_version: str = SCHEMA_VERSION
    order_id: str = ""
    risk_score: float = 0.0
    risk_level: str = RISK_LEVEL_LOW
    source_row_count: int = 0
    source_rows_digest: str = ""
    classifier: str = "decision_tree"
    model_strategy: str = "FIXED_FIVE_SEED_MEAN_ENSEMBLE"
    model_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    aggregation_rule: str = "MAX"
    source_partition: str = "VALIDATION"
    score_semantics_version: str = "MODEL_DERIVED_ATTACK_RISK_V1"
    model_configuration_provenance: str = ""
    hybrid_k13_provenance: str = ""
    generation_stage_provenance: str = ""
    record_digest: str = ""

    def to_mapping(self, *, include_digest: bool = True) -> dict[str, Any]:
        """Serialize the record in the frozen required-field order."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "order_id": self.order_id,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "source_row_count": self.source_row_count,
            "source_rows_digest": self.source_rows_digest,
            "classifier": self.classifier,
            "model_strategy": self.model_strategy,
            "model_seeds": list(self.model_seeds),
            "aggregation_rule": self.aggregation_rule,
            "source_partition": self.source_partition,
            "score_semantics_version": self.score_semantics_version,
            "model_configuration_provenance": self.model_configuration_provenance,
            "hybrid_k13_provenance": self.hybrid_k13_provenance,
            "generation_stage_provenance": self.generation_stage_provenance,
        }
        if include_digest:
            payload["record_digest"] = self.record_digest
        return payload

    def recompute_digest(self) -> str:
        """Recompute the digest over all required fields except record_digest."""
        base = self.to_mapping(include_digest=False)
        return sha256_hex(base)

    def verify(self) -> None:
        """Fail closed unless the record is schema-valid and digest-consistent."""
        expected_order = tuple(REQUIRED_FIELDS_IN_ORDER)
        actual = tuple(self.to_mapping(include_digest=True))
        if actual != expected_order:
            raise RiskRecordError(
                f"risk record fields must follow the frozen order {expected_order}"
            )
        if self.risk_level != risk_level_for_score(self.risk_score):
            raise RiskRecordError(
                f"risk_level {self.risk_level!r} does not match "
                f"risk_score {self.risk_score}"
            )
        if self.record_digest != self.recompute_digest():
            raise RiskRecordError("risk record_digest is inconsistent with its payload")


def build_risk_record(
    aggregate: OrderAggregate,
    protocol: AIRiskProtocol,
    *,
    source_partition: str = "VALIDATION",
    classifier: str | None = None,
    model_seeds: tuple[int, ...] | None = None,
) -> RiskRecord:
    """Build a fully-provenanced, digest-consistent off-chain risk record.

    Provenance fields are derived deterministically from the frozen D1
    protocol. ``model_seeds`` may be overridden only by an exact match of the
    frozen seed family; anything else fails closed.
    """
    if aggregate.source_row_count < 1:
        raise RiskRecordError("a risk record requires at least one contributing row")
    score = _validated_score(aggregate.risk_score)
    received_seeds = tuple(
        int(seed) for seed in (model_seeds or protocol.seeds())
    )
    if received_seeds != (42, 43, 44, 45, 46):
        raise RiskRecordError(
            f"model_seeds must be exactly (42, 43, 44, 45, 46), got {received_seeds}"
        )
    if source_partition != "VALIDATION":
        raise RiskRecordError(
            f"risk records are generated on VALIDATION only, got {source_partition!r}"
        )
    selected_classifier = classifier or protocol.config["classifier"]["primary"]
    if selected_classifier != "decision_tree":
        raise RiskRecordError(
            f"primary classifier must be decision_tree, got {selected_classifier!r}"
        )

    record = RiskRecord(
        schema_version=SCHEMA_VERSION,
        order_id=aggregate.order_id,
        risk_score=score,
        risk_level=risk_level_for_score(score),
        source_row_count=aggregate.source_row_count,
        source_rows_digest=aggregate.source_rows_digest,
        classifier=selected_classifier,
        model_strategy=protocol.config["model_strategy"]["type"],
        model_seeds=received_seeds,
        aggregation_rule=protocol.config["row_to_order"]["rule"],
        source_partition=source_partition,
        score_semantics_version=protocol.config["risk_score"]["semantics_version"],
        model_configuration_provenance=(
            "deterministic reconstruction from frozen fingerprint-verified "
            "TRAIN attack workloads"
        ),
        hybrid_k13_provenance=(
            f'{protocol.feature_identity()["configuration_id"]}; '
            f'mask {protocol.feature_identity()["winner_mask_sha256"]}; '
            f'manifest {protocol.feature_identity()["canonical_feature_manifest_sha256"]}'
        ),
        generation_stage_provenance=(
            "CLEAN_FROZEN_VALIDATION_FEATURES; off-chain governed V1.1-D "
            "risk artifact"
        ),
        record_digest="",
    )
    digest = record.recompute_digest()
    return replace(record, record_digest=digest)