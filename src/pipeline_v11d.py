"""V1.1-D governed AI risk implementation stage (orchestrator).

D2 ships the frozen protocol loader and the deterministic components that a
later authorized stage will use to produce the off-chain governed risk
artifact. This orchestrator must NEVER launch the full reconstruction
campaign, validation inference, optimization, or blockchain experiments on
its own: an explicit execution guard gates every scientific entry point, and
the default state is D2_IMPLEMENTATION_ONLY.
"""

from __future__ import annotations

from typing import Any, Sequence

from .ai_risk.blockchain_linkage import (
    AIRiskReference,
    build_ai_risk_reference,
    verify_reference_matches_record,
)
from .ai_risk.model_reconstruction import (
    ReconstructionSpec,
    require_reconstruction_authorization,
    validate_reconstruction_spec,
    assert_fit_split,
    assert_generation_split,
    assert_test_blocked,
)
from .ai_risk.order_aggregation import OrderAggregate, RowScore, aggregate_rows
from .ai_risk.protocol import AIRiskProtocol, load_verified_protocol
from .ai_risk.risk_levels import risk_level_for_score
from .ai_risk.risk_record import RiskRecord, build_risk_record
from .ai_risk.row_scoring import combine_member_scores

#: Frozen D2 execution posture: no governed scientific execution is allowed.
GOVERNED_RISK_GENERATION_AUTHORIZED = False
STAGE = "V1.1-D"


class GovernedRiskExecutionNotAuthorizedError(PermissionError):
    """Raised when a scientific entry point runs before authorization."""


def require_generation_authorization(*, authorized: bool | None = None) -> None:
    """Gate any generation/inference/optimization entry point, fail closed."""
    allowed = GOVERNED_RISK_GENERATION_AUTHORIZED if authorized is None else authorized
    if not allowed:
        raise GovernedRiskExecutionNotAuthorizedError(
            "V1.1-D2 authorizes no governed risk generation; raise the "
            "execution gate only in a separately authorized stage."
        )


def run_governed_risk_generation(*args: Any, **kwargs: Any) -> None:
    """D2 stub: full governed generation/inference is out of scope."""
    require_generation_authorization()
    raise NotImplementedError(
        "V1.1-D2 does not implement the full governed risk generation campaign"
    )


def load_protocol() -> AIRiskProtocol:
    """Load and strictly verify the frozen D1 config and lock artifacts."""
    return load_verified_protocol()


def reconstruction_contract(protocol: AIRiskProtocol) -> ReconstructionSpec:
    """Return and validate the frozen reconstruction contract (no fitting)."""
    spec = ReconstructionSpec.from_protocol(protocol)
    validate_reconstruction_spec(spec)
    return spec


def combine_row_score(member_scores: Sequence[float]) -> float:
    """Combine five frozen-seed member scores into the row ensemble score."""
    require_generation_authorization()
    return combine_member_scores(member_scores)


def aggregate_order_rows(rows: Sequence[RowScore]) -> dict[str, OrderAggregate]:
    """Aggregate validated rows to canonical orders under the frozen MAX rule."""
    require_generation_authorization()
    result = aggregate_rows(rows)
    return dict(result)


def score_to_level(score: float) -> str:
    """Map a validated score to the frozen risk band."""
    require_generation_authorization()
    return risk_level_for_score(score)


def make_risk_record(
    aggregate: OrderAggregate,
    protocol: AIRiskProtocol,
) -> RiskRecord:
    """Build an off-chain risk record with full provenance and digest."""
    require_generation_authorization()
    return build_risk_record(aggregate, protocol)


def make_ai_risk_reference(
    record: RiskRecord,
    protocol: AIRiskProtocol,
    *,
    artifact_ref: str,
    artifact_lock_sha256: str,
) -> AIRiskReference:
    """Build the minimal digest-bound on-chain reference for a record."""
    require_generation_authorization()
    reference = build_ai_risk_reference(
        record,
        protocol,
        artifact_ref=artifact_ref,
        artifact_lock_sha256=artifact_lock_sha256,
    )
    verify_reference_matches_record(reference, record)
    return reference