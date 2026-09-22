"""Minimal digest-bound AI_RISK_ASSESSED on-chain reference construction.

V1.1-D2 builds only the metadata payloads and NEVER appends to the real
V1.1-C order chains. The frozen payload policy is DIGEST_MINIMAL_METADATA:
raw features, raw labels, row identifiers, and model objects are strictly
excluded; the authoritative record remains the off-chain risk-record digest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .risk_record import RiskRecord
from .protocol import AIRiskProtocol


class BlockchainLinkageError(ValueError):
    """Raised when an on-chain reference violates the frozen payload policy."""


EVENT_TYPE_AI_RISK_ASSESSED = "AI_RISK_ASSESSED"
RISK_MODE_GOVERNED_AI_RISK = "GOVERNED_AI_RISK"
PAYLOAD_POLICY_DIGEST_MINIMAL_METADATA = "DIGEST_MINIMAL_METADATA"

FORBIDDEN_REFERENCE_KEYS = frozenset(
    {
        "raw_features",
        "raw_labels",
        "row_ids",
        "row_identifiers",
        "model_object",
        "model",
        "prediction",
    }
)

REFERENCE_FIELD_ORDER = (
    "mode",
    "stage",
    "order_id",
    "risk_score",
    "risk_level",
    "configuration_id",
    "classifier",
    "aggregation_rule",
    "record_digest",
    "artifact_ref",
    "artifact_lock_sha256",
)


@dataclass(frozen=True)
class AIRiskReference:
    """The frozen on-chain AI_RISK_ASSESSED reference payload (dataset-free)."""

    mode: str = RISK_MODE_GOVERNED_AI_RISK
    stage: str = "V1.1-D"
    order_id: str = ""
    risk_score: float = 0.0
    risk_level: str = "LOW"
    configuration_id: str = "HYBRID-K13"
    classifier: str = "decision_tree"
    aggregation_rule: str = "MAX"
    record_digest: str = ""
    artifact_ref: str = ""
    artifact_lock_sha256: str = ""

    def to_mapping(self) -> dict[str, Any]:
        order = REFERENCE_FIELD_ORDER
        payload = {
            "mode": self.mode,
            "stage": self.stage,
            "order_id": self.order_id,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "configuration_id": self.configuration_id,
            "classifier": self.classifier,
            "aggregation_rule": self.aggregation_rule,
            "record_digest": self.record_digest,
            "artifact_ref": self.artifact_ref,
            "artifact_lock_sha256": self.artifact_lock_sha256,
        }
        if tuple(payload) != tuple(order):
            raise BlockchainLinkageError("ai_risk_reference field order violated")
        forbidden = set(payload) & FORBIDDEN_REFERENCE_KEYS
        if forbidden:
            raise BlockchainLinkageError(
                f"ai_risk_reference must not carry dataset content: {sorted(forbidden)}"
            )
        return payload


def build_ai_risk_reference(
    record: RiskRecord,
    protocol: AIRiskProtocol,
    *,
    artifact_ref: str,
    artifact_lock_sha256: str,
) -> AIRiskReference:
    """Build the minimal digest-bound reference validating all payload rules."""
    if record.record_digest == "":
        raise BlockchainLinkageError("a risk record must carry a record_digest first")
    if record.source_partition != "VALIDATION":
        raise BlockchainLinkageError(
            f"on-chain references require VALIDATION records, got {record.source_partition!r}"
        )
    feature_identity = protocol.feature_identity()
    if feature_identity["configuration_id"] != "HYBRID-K13":
        raise BlockchainLinkageError(
            "on-chain references require the frozen HYBRID-K13 configuration"
        )
    return AIRiskReference(
        mode=RISK_MODE_GOVERNED_AI_RISK,
        stage="V1.1-D",
        order_id=record.order_id,
        risk_score=record.risk_score,
        risk_level=record.risk_level,
        configuration_id="HYBRID-K13",
        classifier=record.classifier,
        aggregation_rule=record.aggregation_rule,
        record_digest=record.record_digest,
        artifact_ref=artifact_ref,
        artifact_lock_sha256=artifact_lock_sha256,
    )


def verify_reference_matches_record(
    reference: AIRiskReference,
    record: RiskRecord,
) -> None:
    """Fail closed unless the on-chain reference is digest-bound to the record."""
    if reference.order_id != record.order_id:
        raise BlockchainLinkageError("reference order_id does not match the record")
    if reference.risk_score != record.risk_score:
        raise BlockchainLinkageError("reference risk_score does not match the record")
    if reference.risk_level != record.risk_level:
        raise BlockchainLinkageError("reference risk_level does not match the record")
    if reference.record_digest != record.record_digest:
        raise BlockchainLinkageError("reference record_digest does not match the record")