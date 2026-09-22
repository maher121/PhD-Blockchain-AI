"""Governed V1.1-D AI risk implementation components.

D2 introduces the deterministic software components that later convert frozen
V1.1-A HYBRID-K13 row-level attack evidence into per-order governed risk
records and minimal, digest-bound on-chain references. Every component is
purely deterministic, fail-closed, and guarded so that this stage cannot
launch scientific inference, reconstruction, optimization, or blockchain
experiments. The execution guard lives in ``src.pipeline_v11d``.
"""

from __future__ import annotations

from .blockchain_linkage import AIRiskReference, build_ai_risk_reference
from .model_reconstruction import (
    ReconstructionNotAuthorizedError,
    ReconstructionSpec,
    require_reconstruction_authorization,
)
from .order_aggregation import OrderAggregate, RowScore, aggregate_rows
from .protocol import AIRiskProtocol, load_verified_protocol
from .risk_levels import risk_level_for_score
from .risk_record import RiskRecord, build_risk_record
from .row_scoring import combine_member_scores

__all__ = [
    "AIRiskProtocol",
    "AIRiskReference",
    "OrderAggregate",
    "ReconstructionNotAuthorizedError",
    "ReconstructionSpec",
    "RiskRecord",
    "RowScore",
    "aggregate_rows",
    "build_ai_risk_reference",
    "build_risk_record",
    "combine_member_scores",
    "load_verified_protocol",
    "require_reconstruction_authorization",
    "risk_level_for_score",
]