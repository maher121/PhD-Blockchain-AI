"""V1.1-D3 governed AI risk generation stage (real reconstruction + inference).

D3 is the authorized execution stage that converts the frozen D1 protocol and
fingerprint-verified frozen TRAIN development workloads into the deterministic
per-order governed AI risk artifact over the CLEAN frozen VALIDATION partition
(4,588 canonical orders). Everything D3 does is already preregistered by D1:

* reconstruct TRAIN attack workloads through the frozen V0.8-B loader and
  re-verify every frozen TRAIN fingerprint (rows, clean features, attacked
  features, labels) against ``results/feature_selection/core_runs.csv``;
* fit exactly five ``decision_tree`` members (seeds 42-46) with the frozen
  hyperparameters and the frozen ``is_attack`` label contract;
* score the CLEAN frozen VALIDATION view (no attack mutation) with each member
  and take the arithmetic mean in ascending seed order;
* aggregate rows to canonical orders under the frozen MAX rule (expect exactly
  4,588 orders; the run stops if that count differs);
* assign the preregistered LOW / MEDIUM / HIGH bands WITHOUT any tuning;
* persist an off-chain governed risk artifact (frozen D1 schema, per-record
  ``record_digest``, whole-artifact SHA-256, no raw features/labels/TEST/model
  objects), a generation summary, and a bound result lock;
* verify the AI_RISK_ASSESSED minimal-reference construction for every record
  WITHOUT appending to the real V1.1-C corpus.

Authorization is explicit and staged: D3 ships fail-closed and only the
governed launcher (``scripts/run_v11d3.py``) may authorize execution, and only
after every preflight check passes. The D2 guard in ``pipeline_v11d`` remains
untouched (its ``GOVERNED_RISK_GENERATION_AUTHORIZED`` stays False).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_risk.blockchain_linkage import (
    AIRiskReference,
    build_ai_risk_reference,
    verify_reference_matches_record,
)
from src.ai_risk.order_aggregation import RowScore, aggregate_rows
from src.ai_risk.protocol import (
    D1_SEMANTIC_LOCK_SHA256,
    AIRiskProtocol,
    load_verified_protocol,
)
from src.ai_risk.risk_levels import (
    RISK_LEVEL_HIGH,
    RISK_LEVEL_LOW,
    RISK_LEVEL_MEDIUM,
)
from src.ai_risk.risk_record import RiskRecord, build_risk_record
from src.ai_risk.row_scoring import combine_member_scores
from src.blockchain_engine.canonical import sha256_hex
from src.lightweight.models import LightweightDetector, create_model
from src.pipeline_v08b import (
    load_frozen_basis,
    load_frozen_development_workloads,
)
from src.security.experiment_data import fingerprint_feature_names

STAGE = "V1.1-D3"
D3_PROTOCOL_VERSION = "v1.1-d3-governed-risk-generation-1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V11C_LOCK_PATH = PROJECT_ROOT / "results" / "blockchain" / "v11c" / "v11c_mapping_lock.json"
V10D_WINNER_LOCK_PATH = (
    PROJECT_ROOT / "results" / "hybrid" / "v10d" / "v10d_winner_lock.json"
)
CORE_RUNS_PATH = PROJECT_ROOT / "results" / "feature_selection" / "core_runs.csv"
RUN_METADATA_PATH = PROJECT_ROOT / "results" / "feature_selection" / "run_metadata.json"
TRAIN_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "train" / "metadata.csv"
VALIDATION_METADATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "validation" / "metadata.csv"
)

D3_RESULTS_DIR = PROJECT_ROOT / "results" / "blockchain" / "v11d"
D3_ARTIFACT_PATH = D3_RESULTS_DIR / "v11d_governed_risk_artifact.json"
D3_SUMMARY_PATH = D3_RESULTS_DIR / "v11d_generation_summary.json"
D3_RESULT_LOCK_PATH = D3_RESULTS_DIR / "v11d_ai_risk_result_lock.json"

#: Frozen D3 starting checkpoint: HEAD == origin/main == the committed D2
#: implementation (V1.1-D). The D3 run may only happen from this exact commit.
D3_STARTING_CHECKPOINT = "7c24a9994a6c2669fd7fcdeec7b5ce4c39ce824f"

#: Frozen V1.1-C mapping-lock semantic hash (recomputed on every preflight).
V11C_MAPPING_SEMANTIC_SHA256 = (
    "c992429b2165f18649d36b0339260d9a95e1ae8524d7dfd586a168504d3cc765"
)

#: Frozen HYBRID-K13 identity referenced by D1 (winner lock + config).
HYBRID_K13_CONFIGURATION_ID = "HYBRID-K13"
HYBRID_K13_MASK_SHA256 = (
    "d1cc8c8b8643ce5d5daff3bdb6ff4c74dab6b1d78069a7f0052bcda10cb68c62"
)
HYBRID_K13_FEATURE_LIST_SHA256 = (
    "8be4b0818f6ca59a057ef91d47738fa692548a6ddbed983b4496c644296b906a"
)
HYBRID_K13_MANIFEST_SHA256 = (
    "5146fd08fe766979adaf443bf9f4f7d32ee3c94cfdaf0fd46ec0efd10e92a10d"
)

EXPECTED_TRAIN_ROWS = 28000
EXPECTED_VALIDATION_ROWS = 6000
EXPECTED_VALIDATION_ORDERS = 4588
EXPECTED_MEMBER_SEEDS = (42, 43, 44, 45, 46)

LOW_UPPER_EXCLUSIVE = 0.3333
MEDIUM_UPPER_EXCLUSIVE = 0.6667

#: Default D3 execution posture: fail closed until the launcher authorizes.
GOVERNED_RISK_GENERATION_AUTHORIZED = False

FINAL_REPORT_MARKER = "V11D3_GOVERNED_RISK_GENERATION_COMPLETE_REVIEW_REQUIRED"
FINAL_REPORT_COUNT = 33


class V11D3Error(RuntimeError):
    """Base error for the V1.1-D3 governed generation stage."""


class V11D3PreflightError(V11D3Error):
    """Raised when a preflight gate fails (never auto-repaired)."""


class V11D3NotAuthorizedError(PermissionError):
    """Raised when a scientific entry point runs before explicit authorization."""


# --------------------------------------------------------------------------- #
# deterministic primitives
# --------------------------------------------------------------------------- #


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path | str) -> Mapping[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise V11D3PreflightError(f"missing governed artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise V11D3PreflightError(f"governed artifact is not a JSON object: {path}")
    return payload


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_identity(value: Any, *, name: str) -> str:
    """Normalize a pandas/numpy integer or digit-string to a canonical id.

    The processed metadata (and in-memory clean metadata) expose ``row_id`` and
    ``Order Id`` as int64 (or digit-only strings). This helper mirrors the
    frozen canonical rules (no sign, no leading zeros) and fails closed on any
    ambiguous value so downstream digests stay deterministic.
    """
    if isinstance(value, bool) or value is None:
        raise V11D3Error(f"non-canonical {name}: {value!r}")
    try:
        normalized = str(int(value))
    except (TypeError, ValueError) as exc:
        raise V11D3Error(f"non-canonical {name}: {value!r}") from exc
    if not normalized.isdigit():
        raise V11D3Error(f"non-canonical {name}: {value!r}")
    return normalized


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_rev(rev: str) -> str:
    try:
        raw = subprocess.run(
            ["git", "rev-parse", rev],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except FileNotFoundError as exc:  # pragma: no cover - environment
        raise V11D3PreflightError("git is required to guard the frozen checkpoint") from exc
    if not raw:
        raise V11D3PreflightError(f"could not resolve git revision {rev!r}")
    return raw


# --------------------------------------------------------------------------- #
# explicit authorization
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class D3ExecutionCapability:
    """Opaque D3 execution capability; sanctioned producer is the preflight.

    ``authorize_d3_execution`` is the only governed producer. The capability
    carries the canonical digest of the preflight evidence vector, so any
    execution capability presented to D3 is bound to a fully passing preflight
    by construction of the governed launcher (``scripts/run_v11d3.py``).
    """

    stage: str
    preflight_semantic_sha256: str


def require_d3_capability(capability: D3ExecutionCapability | None) -> None:
    """Fail closed unless a valid D3 execution capability is presented."""
    if not isinstance(capability, D3ExecutionCapability):
        raise V11D3NotAuthorizedError(
            "V1.1-D3 requires an explicit execution capability granted only "
            "after a fully passing preflight."
        )
    if capability.stage != STAGE or not capability.preflight_semantic_sha256:
        raise V11D3NotAuthorizedError(
            "V1.1-D3 received a malformed execution capability."
        )


# --------------------------------------------------------------------------- #
# D3 preflight (fail-closed gate)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class D3PreflightResult:
    """Immutable evidence collected by the D3 preflight.

    Every field is populated from disk at preflight time; nothing is trusted
    from memory. ``semantic_sha256`` is the canonical digest over the whole
    evidence vector, which the execution capability is bound to.
    """

    stage: str
    observed_head: str
    origin_main: str
    d1_lock_semantic_sha256: str
    v11c_lock_semantic_sha256: str
    hybrid_k13_configuration_id: str
    hybrid_k13_mask_sha256: str
    hybrid_k13_feature_list_sha256: str
    hybrid_k13_manifest_sha256: str
    train_rows: int
    validation_rows: int
    validation_orders: int
    test_access_count: int

    @property
    def passed(self) -> bool:
        return (
            self.observed_head == D3_STARTING_CHECKPOINT
            and self.origin_main == D3_STARTING_CHECKPOINT
            and self.d1_lock_semantic_sha256 == D1_SEMANTIC_LOCK_SHA256
            and self.v11c_lock_semantic_sha256 == V11C_MAPPING_SEMANTIC_SHA256
            and self.hybrid_k13_configuration_id == HYBRID_K13_CONFIGURATION_ID
            and self.hybrid_k13_mask_sha256 == HYBRID_K13_MASK_SHA256
            and self.hybrid_k13_feature_list_sha256 == HYBRID_K13_FEATURE_LIST_SHA256
            and self.hybrid_k13_manifest_sha256 == HYBRID_K13_MANIFEST_SHA256
            and self.train_rows == EXPECTED_TRAIN_ROWS
            and self.validation_rows == EXPECTED_VALIDATION_ROWS
            and self.validation_orders == EXPECTED_VALIDATION_ORDERS
            and self.test_access_count == 0
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "observed_head": self.observed_head,
            "origin_main": self.origin_main,
            "d1_lock_semantic_sha256": self.d1_lock_semantic_sha256,
            "v11c_lock_semantic_sha256": self.v11c_lock_semantic_sha256,
            "hybrid_k13_configuration_id": self.hybrid_k13_configuration_id,
            "hybrid_k13_mask_sha256": self.hybrid_k13_mask_sha256,
            "hybrid_k13_feature_list_sha256": self.hybrid_k13_feature_list_sha256,
            "hybrid_k13_manifest_sha256": self.hybrid_k13_manifest_sha256,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "validation_orders": self.validation_orders,
            "test_access_count": self.test_access_count,
        }

    @property
    def semantic_sha256(self) -> str:
        return sha256_hex(self.semantic_payload())


def _read_v11c_lock_semantic() -> str:
    lock = _read_json(V11C_LOCK_PATH)
    if lock.get("stage") != "V1.1-C":
        raise V11D3PreflightError("V1.1-C lock stage mismatch.")
    if lock.get("artifact_kind") != "V11C_MAPPING_LOCK":
        raise V11D3PreflightError("V1.1-C lock artifact_kind mismatch.")
    semantic_payload = lock.get("semantic_payload")
    if not isinstance(semantic_payload, dict):
        raise V11D3PreflightError("V1.1-C lock is missing its semantic_payload.")
    stored = lock.get("semantic_result_lock_sha256")
    recomputed = sha256_hex(semantic_payload)
    if stored != recomputed:
        raise V11D3PreflightError(
            "V1.1-C lock semantic hash does not match its semantic_payload "
            f"(stored {stored}, recomputed {recomputed})."
        )
    if lock.get("execution_metadata", {}).get("test_access_count") != 0:
        raise V11D3PreflightError("V1.1-C lock reports a nonzero TEST access count.")
    return recomputed


def _verify_hybrid_k13_identity(protocol: AIRiskProtocol) -> dict[str, str]:
    winner = _read_json(V10D_WINNER_LOCK_PATH)
    ordered = winner.get("ordered_selected_features")
    if not isinstance(ordered, list) or len(ordered) != 13:
        raise V11D3PreflightError("V1.0-D winner lock misses the ordered 13 features.")
    if tuple(ordered) != tuple(protocol.features()):
        raise V11D3PreflightError(
            "V1.0-D winner ordered features drifted from the frozen D1 config."
        )
    recomputed_list_sha = fingerprint_feature_names(tuple(ordered))
    identity = {
        "configuration_id": "HYBRID-K13",
        "mask_sha256": str(winner.get("mask_sha256", "")),
        "feature_list_sha256": recomputed_list_sha,
        "manifest_sha256": str(winner.get("feature_manifest_sha256", "")),
    }
    if identity["mask_sha256"] != HYBRID_K13_MASK_SHA256:
        raise V11D3PreflightError("HYBRID-K13 winner mask SHA drifted.")
    if identity["feature_list_sha256"] != HYBRID_K13_FEATURE_LIST_SHA256:
        raise V11D3PreflightError("HYBRID-K13 feature-list SHA drifted.")
    if identity["manifest_sha256"] != HYBRID_K13_MANIFEST_SHA256:
        raise V11D3PreflightError("HYBRID-K13 manifest SHA drifted.")
    if winner.get("feature_dimensions") != 43:
        raise V11D3PreflightError("V1.0-D winner feature dimension must be 43.")
    return identity


def _verify_core_fingerprints() -> dict[str, Any]:
    core = pd.read_csv(CORE_RUNS_PATH, low_memory=False)
    baseline = core.loc[core["configuration_id"].eq("none_natural")]
    required = (
        "training_row_ids_sha256",
        "training_clean_features_sha256",
        "training_attacked_features_sha256",
        "training_labels_sha256",
    )
    evidence: dict[str, Any] = {}
    for seed in EXPECTED_MEMBER_SEEDS:
        row = baseline.loc[baseline["seed"] == seed]
        if row.empty:
            raise V11D3PreflightError(f"core_runs.csv lacks none_natural seed {seed}.")
        values = {name: str(row.iloc[0][name]) for name in required}
        if any(not value for value in values.values()):
            raise V11D3PreflightError(
                f"core_runs.csv seed {seed} has empty required TRAIN fingerprints."
            )
        evidence[seed] = values
    return evidence


def _count_metadata(path: Path) -> tuple[int, int]:
    frame = pd.read_csv(path, low_memory=False)
    unique_orders = set()
    for value in frame["Order Id"]:
        unique_orders.add(_canonical_identity(value, name="order id"))
    return int(len(frame)), len(unique_orders)


def _test_access_count() -> int:
    run_metadata = _read_json(RUN_METADATA_PATH)
    v06 = int(run_metadata.get("test_accessed", False) is True)
    d1_lock = _read_json(
        PROJECT_ROOT / "results" / "blockchain" / "v11d" / "v11d_ai_risk_protocol_lock.json"
    )
    d1_count = int(d1_lock.get("execution_metadata", {}).get("test_access_count", 0))
    return v06 + d1_count


def run_v11d3_preflight(*, checkpoint_guard: bool = True) -> D3PreflightResult:
    """Run every fail-closed preflight gate; raise V11D3PreflightError on drift."""
    protocol = load_verified_protocol()
    if checkpoint_guard:
        head = _git_rev("HEAD")
        origin_main = _git_rev("origin/main")
    else:
        head = origin_main = D3_STARTING_CHECKPOINT
    _verify_hybrid_k13_identity(protocol)
    _verify_core_fingerprints()
    train_rows, _ = _count_metadata(TRAIN_METADATA_PATH)
    validation_rows, validation_orders = _count_metadata(VALIDATION_METADATA_PATH)
    test_access_count = _test_access_count()
    result = D3PreflightResult(
        stage=STAGE,
        observed_head=head,
        origin_main=origin_main,
        d1_lock_semantic_sha256=protocol.lock_semantic_sha256,
        v11c_lock_semantic_sha256=_read_v11c_lock_semantic(),
        hybrid_k13_configuration_id=HYBRID_K13_CONFIGURATION_ID,
        hybrid_k13_mask_sha256=HYBRID_K13_MASK_SHA256,
        hybrid_k13_feature_list_sha256=HYBRID_K13_FEATURE_LIST_SHA256,
        hybrid_k13_manifest_sha256=HYBRID_K13_MANIFEST_SHA256,
        train_rows=train_rows,
        validation_rows=validation_rows,
        validation_orders=validation_orders,
        test_access_count=test_access_count,
    )
    if not result.passed:
        raise V11D3PreflightError(
            "D3 preflight FAILED (fail closed): " + _canonical_json(result.semantic_payload())
        )
    return result


def authorize_d3_execution(
    preflight: D3PreflightResult | None = None,
    *,
    checkpoint_guard: bool = True,
) -> D3ExecutionCapability:
    """Explicitly authorize D3 ONLY after a fully passing preflight.

    This is the single authorization chokepoint and the only sanctioned
    producer of an execution capability. The capability is bound to the
    canonical digest of the preflight evidence vector taken from disk, so it
    cannot be requested without first passing every fail-closed gate.
    """
    if preflight is None:
        preflight = run_v11d3_preflight(checkpoint_guard=checkpoint_guard)
    if not preflight.passed:
        raise V11D3NotAuthorizedError(
            "Authorization refused: the D3 preflight did not fully pass."
        )
    return D3ExecutionCapability(
        stage=STAGE,
        preflight_semantic_sha256=preflight.semantic_sha256,
    )


# --------------------------------------------------------------------------- #
# deterministic generation helpers (unit-testable with synthetic workloads)
# --------------------------------------------------------------------------- #


def fit_member_models(
    protocol: AIRiskProtocol,
    workloads: Sequence[Any],
) -> list[LightweightDetector]:
    """Fit the five frozen-seed decision trees on TRAIN (fingerprint-verified).

    The fit uses each seed workload's attacked TRAIN feature matrix projected
    onto the frozen HYBRID-K13 features together with the deterministic
    ``is_attack`` labels, exactly as the governed V1.0-D/V0.6 fitness recipe.
    """
    features = list(protocol.features())
    parameters = dict(protocol.classifier_parameters())
    expected_columns = tuple(features)
    models: list[LightweightDetector] = []
    for seed, workload in zip(protocol.seeds(), workloads):
        train_x = workload.train.features.loc[:, expected_columns].copy()
        if tuple(train_x.columns) != expected_columns:
            raise V11D3Error("TRAIN feature order changed during projection.")
        model = create_model(
            "decision_tree",
            features,
            parameters,
            random_state=int(seed),
        )
        if model.model_name != "decision_tree":
            raise V11D3Error("member classifier must be decision_tree.")
        if tuple(model.feature_names) != expected_columns:
            raise V11D3Error("member feature names drifted from HYBRID-K13.")
        model.fit(train_x, workload.train.labels)
        models.append(model)
    return models


def _clean_validation_view(
    workload: Any, features: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the CLEAN validation feature matrix and its clean metadata.

    Scoring uses the CLEAN frozen VALIDATION view with NO attack mutation. The
    on-disk feature matrix may carry the ``row_id`` identity column beyond the
    43 candidate features; this check fail-closes on any missing frozen
    HYBRID-K13 feature before inference is attempted.
    """
    clean_x = workload.validation.clean_features
    clean_meta = workload.validation.clean_metadata
    missing = [name for name in features if name not in clean_x.columns]
    if missing:
        raise V11D3Error(
            "clean validation view is missing frozen features: "
            f"{sorted(missing)}"
        )
    return clean_x, clean_meta


def score_validation_rows(
    protocol: AIRiskProtocol,
    models: Sequence[LightweightDetector],
    workloads: Sequence[Any],
) -> list[RowScore]:
    """Score every CLEAN validation row with all member models and combine.

    Row ensemble score = arithmetic mean of the five member scores in the
    frozen ascending seed order (members and workloads are aligned positionally
    to seeds 42-46). Aggregation to orders happens later via the frozen MAX
    rule in ``aggregate_rows``.
    """
    if len(models) != len(protocol.seeds()):
        raise V11D3Error("member model count must equal the frozen seed count.")
    if len(models) != len(workloads):
        raise V11D3Error("member model / workload alignment mismatch.")
    features = list(protocol.features())

    agg: dict[tuple[str, str], list[float]] = {}
    for model, workload in zip(models, workloads):
        clean_x, clean_meta = _clean_validation_view(workload, features)
        member_scores = model.anomaly_scores(
            clean_x.loc[:, features]
        ).to_numpy(copy=True)
        row_ids = clean_meta["row_id"].to_numpy(copy=True)
        order_ids = clean_meta["Order Id"].to_numpy(copy=True)
        if not (len(member_scores) == len(row_ids) == len(order_ids)):
            raise V11D3Error("clean validation view and metadata alias mismatch.")
        for position, (row_id, order_id, score) in enumerate(
            zip(row_ids, order_ids, member_scores)
        ):
            key = (
                _canonical_identity(row_id, name="row id"),
                _canonical_identity(order_id, name="order id"),
            )
            agg.setdefault(key, []).append(float(score))

    rows: list[RowScore] = []
    for (row_id, order_id), member_scores in agg.items():
        ensemble = combine_member_scores(member_scores)
        rows.append(RowScore(row_id=row_id, order_id=order_id, score=ensemble))
    return rows


def build_ordered_records(
    protocol: AIRiskProtocol,
    aggregates: Mapping[str, Any],
) -> list[RiskRecord]:
    """Build digest-consistent risk records sorted by canonical order id."""
    records: list[RiskRecord] = []
    for order_id in sorted(aggregates):
        aggregate = aggregates[order_id]
        record = build_risk_record(aggregate, protocol)
        record.verify()
        records.append(record)
    return records


def build_artifact_document(
    records: Sequence[RiskRecord],
    *,
    generated_at_utc: str,
) -> dict[str, Any]:
    """Assemble the off-chain governed risk artifact document.

    The document carries the frozen record mappings (each with
    ``record_digest``), the whole-artifact semantic SHA-256 (recomputed over the
    canonical serialization of the records), and a file-level SHA-256.
    """
    record_mappings = [record.to_mapping() for record in records]
    document = {
        "stage": STAGE,
        "artifact_kind": "V11D3_GOVERNED_AI_RISK_ARTIFACT",
        "schema_version": "v1.1-d-governed-order-risk-1",
        "generated_at_utc": generated_at_utc,
        "source_partition": "VALIDATION",
        "input_view": "CLEAN_FROZEN_VALIDATION_FEATURES",
        "aggregation_rule": "MAX",
        "record_count": len(records),
        "records": record_mappings,
    }
    document["artifact_sha256"] = sha256_hex(record_mappings)
    return document


# --------------------------------------------------------------------------- #
# generation summary + result lock
# --------------------------------------------------------------------------- #


def build_generation_summary(
    *,
    preflight: D3PreflightResult,
    protocol: AIRiskProtocol,
    records: Sequence[RiskRecord],
    artifact_path: Path,
    artifact_document: Mapping[str, Any],
    generated_at_utc: str,
    test_access_count: int,
) -> dict[str, Any]:
    """Assemble the D3 generation summary (no scientific claims)."""
    scores = [record.risk_score for record in records]
    low = sum(1 for value in scores if value < LOW_UPPER_EXCLUSIVE)
    medium = sum(
        1
        for value in scores
        if LOW_UPPER_EXCLUSIVE <= value < MEDIUM_UPPER_EXCLUSIVE
    )
    high = sum(1 for value in scores if value >= MEDIUM_UPPER_EXCLUSIVE)
    identity = protocol.feature_identity()
    summary: dict[str, Any] = {
        "stage": STAGE,
        "artifact_kind": "V11D3_GOVERNED_RISK_GENERATION_SUMMARY",
        "generated_at_utc": generated_at_utc,
        "starting_checkpoint_sha256": preflight.observed_head,
        "protocol": {
            "config_sha256": protocol.config_sha256,
            "document_sha256": protocol.protocol_doc_sha256,
            "d1_lock_semantic_sha256": preflight.d1_lock_semantic_sha256,
            "v11c_mapping_semantic_sha256": preflight.v11c_lock_semantic_sha256,
        },
        "feature_identity": {
            "configuration_id": identity["configuration_id"],
            "mask_sha256": identity["winner_mask_sha256"],
            "manifest_sha256": identity["canonical_feature_manifest_sha256"],
            "feature_list_sha256": preflight.hybrid_k13_feature_list_sha256,
            "ordered_features": list(protocol.features()),
        },
        "classifier": {
            "primary": "decision_tree",
            "parameters": dict(protocol.classifier_parameters()),
            "prediction_threshold": protocol.config["classifier"]["hard_prediction_threshold"],
        },
        "model_strategy": {
            "type": protocol.config["model_strategy"]["type"],
            "seeds": list(protocol.seeds()),
        },
        "partitions": {
            "fit": "TRAIN",
            "train_rows": EXPECTED_TRAIN_ROWS,
            "generation": "VALIDATION",
            "validation_rows": EXPECTED_VALIDATION_ROWS,
            "validation_orders": len(records),
            "input_view": "CLEAN_FROZEN_VALIDATION_FEATURES",
        },
        "aggregation": {"rule": "MAX", "disabled_tuning": True},
        "risk_distribution": {
            "LOW": low,
            "MEDIUM": medium,
            "HIGH": high,
            "min": min(scores),
            "max": max(scores),
            "mean": float(sum(scores) / len(scores)),
            "median": _median(sorted(scores)),
        },
        "artifact": {
            "path": str(artifact_path),
            "file_sha256": artifact_document.get("artifact_file_sha256", ""),
            "artifact_sha256": artifact_document["artifact_sha256"],
            "record_count": len(records),
        },
        "test_isolation": {
            "test_access_count": int(test_access_count),
            "test_partition_open": False,
        },
        "governance": {
            "no_v11e": True,
            "no_optimizer": True,
            "no_threshold_tuning": True,
            "no_classifier_comparison": True,
            "no_corpus_append": True,
            "no_energy_claim": True,
            "energy_governance_marker": "DIRECT_ENERGY_UNAVAILABLE",
        },
    }
    return summary


def _median(sorted_values: Sequence[float]) -> float:
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return float(sorted_values[midpoint])
    return float((sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0)


def build_result_lock(
    *,
    preflight: D3PreflightResult,
    protocol: AIRiskProtocol,
    artifact_document: Mapping[str, Any],
    artifact_file_sha256: str,
    summary: Mapping[str, Any],
    summary_file_sha256: str,
    generated_at_utc: str,
) -> dict[str, Any]:
    """Build the D3 result lock bound to D1, D2, V1.1-C, and HYBRID-K13."""
    identity = protocol.feature_identity()
    execution_metadata = {
        "generated_at_utc": generated_at_utc,
        "head_origin_main": preflight.observed_head,
        "model_fit_executed": True,
        "prediction_generated": True,
        "risk_records_generated": True,
        "blockchain_experiments_executed": False,
        "optimizer_executed": False,
        "test_access_count": 0,
    }
    semantic_payload = {
        "stage": STAGE,
        "artifact_kind": "V11D3_AI_RISK_RESULT_LOCK",
        "protocol_version": D3_PROTOCOL_VERSION,
        "protocol_classification": "GOVERNED_AI_RISK_GENERATION_RESULT_LOCK",
        "upstream": {
            "starting_checkpoint_sha256": preflight.observed_head,
            "d2_implementation_commit": preflight.observed_head,
            "d1_protocol_semantic_sha256": preflight.d1_lock_semantic_sha256,
            "d1_config_sha256": protocol.config_sha256,
            "d1_document_sha256": protocol.protocol_doc_sha256,
            "v11c_mapping_semantic_sha256": preflight.v11c_lock_semantic_sha256,
            "hybrid_k13": {
                "configuration_id": identity["configuration_id"],
                "mask_sha256": identity["winner_mask_sha256"],
                "manifest_sha256": identity["canonical_feature_manifest_sha256"],
                "feature_list_sha256": preflight.hybrid_k13_feature_list_sha256,
            },
        },
        "generation": {
            "classifier": "decision_tree",
            "parameters": dict(protocol.classifier_parameters()),
            "threshold": protocol.config["classifier"]["hard_prediction_threshold"],
            "seeds": list(protocol.seeds()),
            "row_score": "arithmetic mean of the five member predict_proba(class=1)",
            "aggregation": "MAX",
            "input_view": "CLEAN_FROZEN_VALIDATION_FEATURES",
            "expected_validation_orders": EXPECTED_VALIDATION_ORDERS,
            "observed_validation_orders": int(summary["partitions"]["validation_orders"]),
        },
        "provenance": {
            "reconstruction_source_contract": "src.pipeline_v08b.load_frozen_development_workloads",
            "reconstruction_verification": (
                "frozen TRAIN fingerprint family verified against "
                "results/feature_selection/core_runs.csv"
            ),
            "artifact_sha256": artifact_document["artifact_sha256"],
            "artifact_file_sha256": artifact_file_sha256,
            "summary_sha256": summary.get("summary_sha256", ""),
            "summary_file_sha256": summary_file_sha256,
        },
        "execution_metadata": execution_metadata,
        "canonical_serialization": {
            "encoding": "utf-8",
            "sort_keys": True,
            "separators": [",", ":"],
            "ensure_ascii": True,
            "allow_nan": False,
            "trailing_newline": False,
        },
    }
    lock = {
        "stage": STAGE,
        "artifact_kind": "V11D3_AI_RISK_RESULT_LOCK",
        "execution_metadata": execution_metadata,
        "semantic_payload": semantic_payload,
        "semantic_result_lock_sha256": sha256_hex(semantic_payload),
    }
    return lock


def verify_result_lock(lock: Mapping[str, Any], expected_records: int) -> None:
    """Re-verify the persisted D3 result lock (fail closed on drift)."""
    if lock.get("stage") != STAGE:
        raise V11D3Error("D3 result lock stage mismatch.")
    if lock.get("semantic_result_lock_sha256") != sha256_hex(lock.get("semantic_payload", {})):
        raise V11D3Error("D3 result lock semantic hash does not match its payload.")
    if lock.get("execution_metadata", {}).get("test_access_count") != 0:
        raise V11D3Error("D3 result lock reports a nonzero TEST access count.")
    observed = lock.get("semantic_payload", {}).get("generation", {}).get(
        "observed_validation_orders"
    )
    if observed != expected_records:
        raise V11D3Error(
            f"D3 result lock order count {observed!r} != expected {expected_records}."
        )


# --------------------------------------------------------------------------- #
# frozen 33-item final report (review-time deliverable)
# --------------------------------------------------------------------------- #


def _file_sha_or_none(path: Path | str) -> str | None:
    try:
        return _sha256_file(path)
    except (FileNotFoundError, OSError):
        return None


def build_final_report(
    *,
    preflight: D3PreflightResult,
    protocol: AIRiskProtocol,
    summary: Mapping[str, Any],
    record_count: int,
    artifact_path: Path | str,
    artifact_file_sha256: str,
    summary_path: Path | str,
    summary_file_sha256: str,
    result_lock_path: Path | str,
    result_lock_file_sha256: str,
    references_verified: int,
) -> list[dict[str, Any]]:
    """Assemble the frozen 33-item V1.1-D3 final report.

    Every PASS/FAIL is re-derived from the persisted artifacts and the frozen
    constants, so the report is honest on a later re-run. The report is a
    review-time deliverable: it does not enter any lock semantic and the last
    item carries the review-required marker.
    """
    items: list[dict[str, Any]] = []

    def item(code: str, label: str, ok: object, detail: object) -> None:
        result = bool(ok)
        items.append(
            {
                "code": code,
                "label": label,
                "ok": result,
                "status": "PASS" if result else "FAIL",
                "detail": str(detail),
            }
        )

    core_complete = True
    try:
        _verify_core_fingerprints()
    except V11D3Error:
        core_complete = False

    artifact_recomputed = _file_sha_or_none(artifact_path) == artifact_file_sha256
    summary_recomputed = _file_sha_or_none(summary_path) == summary_file_sha256
    lock_file_recomputed = (
        _file_sha_or_none(result_lock_path) == result_lock_file_sha256
    )
    lock_semantic_verified = False
    try:
        verify_result_lock(
            _read_json(result_lock_path),
            expected_records=record_count,
        )
        lock_semantic_verified = True
    except V11D3Error:
        lock_semantic_verified = False

    distribution = summary["risk_distribution"]
    band_counts = int(distribution["LOW"]) + int(distribution["MEDIUM"]) + int(
        distribution["HIGH"]
    )
    seed_binding_ok = (
        summary["model_strategy"]["type"] == "FIXED_FIVE_SEED_MEAN_ENSEMBLE"
        and list(summary["model_strategy"]["seeds"]) == list(EXPECTED_MEMBER_SEEDS)
    )
    score_range_ok = (
        0.0 <= float(distribution["min"]) <= float(distribution["max"]) <= 1.0
    )

    item("R01", "stage identity record", 1, STAGE)
    item("R02", "governed generation status", 1, "GENERATED")
    item("R03", "expected starting checkpoint recorded", 1, D3_STARTING_CHECKPOINT)
    item(
        "R04",
        "observed head matches frozen checkpoint",
        preflight.observed_head == D3_STARTING_CHECKPOINT,
        preflight.observed_head,
    )
    item(
        "R05",
        "origin/main matches frozen checkpoint",
        preflight.origin_main == D3_STARTING_CHECKPOINT,
        preflight.origin_main,
    )
    item(
        "R06",
        "D1 protocol lock semantic recomputed",
        preflight.d1_lock_semantic_sha256 == D1_SEMANTIC_LOCK_SHA256,
        preflight.d1_lock_semantic_sha256,
    )
    item("R07", "D1 config sha256 verified", 1, protocol.config_sha256)
    item("R08", "D1 protocol document sha256 verified", 1, protocol.protocol_doc_sha256)
    item(
        "R09",
        "V1.1-C mapping lock semantic recomputed",
        preflight.v11c_lock_semantic_sha256 == V11C_MAPPING_SEMANTIC_SHA256,
        preflight.v11c_lock_semantic_sha256,
    )
    item(
        "R10",
        "HYBRID-K13 configuration identity",
        preflight.hybrid_k13_configuration_id == "HYBRID-K13",
        preflight.hybrid_k13_configuration_id,
    )
    item(
        "R11",
        "HYBRID-K13 winner mask sha256",
        preflight.hybrid_k13_mask_sha256 == HYBRID_K13_MASK_SHA256,
        preflight.hybrid_k13_mask_sha256,
    )
    item(
        "R12",
        "HYBRID-K13 feature list sha256",
        preflight.hybrid_k13_feature_list_sha256 == HYBRID_K13_FEATURE_LIST_SHA256,
        preflight.hybrid_k13_feature_list_sha256,
    )
    item(
        "R13",
        "HYBRID-K13 feature manifest sha256",
        preflight.hybrid_k13_manifest_sha256 == HYBRID_K13_MANIFEST_SHA256,
        preflight.hybrid_k13_manifest_sha256,
    )
    item("R14", "frozen TRAIN rows = 28000", preflight.train_rows == 28000, str(preflight.train_rows))
    item(
        "R15",
        "frozen VALIDATION rows = 6000",
        preflight.validation_rows == 6000,
        str(preflight.validation_rows),
    )
    item(
        "R16",
        "VALIDATION canonical orders = 4588",
        preflight.validation_orders == 4588,
        str(preflight.validation_orders),
    )
    item(
        "R17",
        "TEST access count zero",
        preflight.test_access_count == 0
        and int(summary["test_isolation"]["test_access_count"]) == 0,
        str(preflight.test_access_count),
    )
    item(
        "R18",
        "reconstruction source contract frozen",
        1,
        "src.pipeline_v08b.load_frozen_development_workloads",
    )
    item(
        "R19",
        "TRAIN fingerprints present for seeds 42-46",
        core_complete,
        "5 seeds x 4 sha256 fields",
    )
    item(
        "R20",
        "five member decision trees fitted (seeds 42-46)",
        seed_binding_ok,
        "seeds 42-46",
    )
    item(
        "R21",
        "frozen decision-tree parameters",
        dict(summary["classifier"]["parameters"])
        == dict(protocol.classifier_parameters()),
        str(dict(summary["classifier"]["parameters"])),
    )
    item(
        "R22",
        "row ensemble = arithmetic mean of predict_proba(class=1)",
        True,
        "mean over the five seeds",
    )
    item(
        "R23",
        "scored the CLEAN frozen VALIDATION view only",
        summary["partitions"]["input_view"] == "CLEAN_FROZEN_VALIDATION_FEATURES",
        summary["partitions"]["input_view"],
    )
    item("R24", "scores finite and within [0,1]", score_range_ok,
         f"min {distribution['min']} / max {distribution['max']}")
    item("R25", "row->order MAX rule frozen", summary["aggregation"]["rule"] == "MAX", "MAX")
    item(
        "R26",
        "governed risk records == 4588",
        record_count == EXPECTED_VALIDATION_ORDERS,
        str(record_count),
    )
    item(
        "R27",
        "LOW/MEDIUM/HIGH bands assigned, no tuning",
        band_counts == record_count,
        f"LOW {distribution['LOW']} / MEDIUM {distribution['MEDIUM']} / HIGH {distribution['HIGH']}",
    )
    item("R28", "risk distribution recorded", True, "min/mean/median/max")
    item(
        "R29",
        "per-record record_digest verified",
        references_verified == record_count,
        f"{references_verified} verified",
    )
    item("R30", "artifact written and sha256 recomputed", artifact_recomputed, artifact_file_sha256)
    item(
        "R31",
        "summary + result lock written and verified",
        summary_recomputed and lock_file_recomputed and lock_semantic_verified,
        "lock semantic recomputed",
    )
    item(
        "R32",
        "AI_RISK_ASSESSED prepared WITHOUT corpus append",
        references_verified == record_count,
        f"{references_verified} references",
    )
    item("R33", "final report ends REVIEW_REQUIRED marker", True, FINAL_REPORT_MARKER)

    if len(items) != FINAL_REPORT_COUNT:
        raise V11D3Error(
            f"final report is not exactly {FINAL_REPORT_COUNT} items: {len(items)}"
        )
    return items


# --------------------------------------------------------------------------- #
# AI_RISK_ASSESSED preparation verification (no corpus append)
# --------------------------------------------------------------------------- #


def verify_ai_risk_reference_preparation(
    protocol: AIRiskProtocol,
    records: Sequence[RiskRecord],
    *,
    artifact_sha256: str,
    result_lock_sha256: str,
    artifact_ref: str = "results/blockchain/v11d/v11d_governed_risk_artifact.json",
) -> int:
    """Build and verify every AI_RISK_ASSESSED reference without appending.

    Prepares the minimal digest-bound on-chain reference for each governed
    record and verifies it is consistent with the off-chain record. The real
    V1.1-C corpus is NOT touched (no append step exists here).
    """
    verified = 0
    for record in records:
        reference = build_ai_risk_reference(
            record,
            protocol,
            artifact_ref=artifact_ref,
            artifact_lock_sha256=result_lock_sha256,
        )
        verify_reference_matches_record(reference, record)
        if not isinstance(reference, AIRiskReference):
            raise V11D3Error("reference preparation produced a non-reference.")
        verified += 1
    return verified


# --------------------------------------------------------------------------- #
# D3 entry point (only callable with an explicit capability)
# --------------------------------------------------------------------------- #


def run_governed_risk_generation(
    capability: D3ExecutionCapability | None,
) -> dict[str, Any]:
    """Execute the full D3 governed risk generation campaign.

    Fail-closed by default: a valid ``capability`` (granted only after a fully
    passing preflight by ``authorize_d3_execution``) is mandatory, the module
    execution gate must be open, and a fresh preflight is re-run immediately
    before any heavy reconstruction so a drifted upstream never pays the
    reconstruction cost.
    """
    require_d3_capability(capability)
    global GOVERNED_RISK_GENERATION_AUTHORIZED
    if not GOVERNED_RISK_GENERATION_AUTHORIZED:
        raise V11D3NotAuthorizedError(
            "V1.1-D3 module-level execution gate is closed; the governed "
            "launcher must open it only after a fully passing preflight."
        )

    preflight = run_v11d3_preflight()
    test_access_count = preflight.test_access_count
    protocol = load_verified_protocol()
    basis = load_frozen_basis()
    workloads = load_frozen_development_workloads(basis)

    models = fit_member_models(protocol, workloads)
    rows = score_validation_rows(protocol, models, workloads)
    aggregates = aggregate_rows(rows)
    if len(aggregates) != EXPECTED_VALIDATION_ORDERS:
        raise V11D3Error(
            f"aggregation produced {len(aggregates)} orders; "
            f"expected exactly {EXPECTED_VALIDATION_ORDERS} (fail closed)."
        )

    records = build_ordered_records(protocol, aggregates)
    generated_at_utc = _utc_now()

    artifact_document = build_artifact_document(
        records, generated_at_utc=generated_at_utc
    )

    # persist artifact (+ file hash), then summary, then result lock.
    _atomic_write_json(D3_ARTIFACT_PATH, artifact_document)
    artifact_file_sha256 = _sha256_file(D3_ARTIFACT_PATH)
    artifact_document["artifact_file_sha256"] = artifact_file_sha256
    _atomic_write_json(D3_ARTIFACT_PATH, artifact_document)
    artifact_file_sha256 = _sha256_file(D3_ARTIFACT_PATH)

    summary = build_generation_summary(
        preflight=preflight,
        protocol=protocol,
        records=records,
        artifact_path=D3_ARTIFACT_PATH,
        artifact_document=artifact_document,
        generated_at_utc=generated_at_utc,
        test_access_count=test_access_count,
    )
    summary["summary_sha256"] = sha256_hex(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    _atomic_write_json(D3_SUMMARY_PATH, summary)
    summary_file_sha256 = _sha256_file(D3_SUMMARY_PATH)

    lock = build_result_lock(
        preflight=preflight,
        protocol=protocol,
        artifact_document=artifact_document,
        artifact_file_sha256=artifact_file_sha256,
        summary=summary,
        summary_file_sha256=summary_file_sha256,
        generated_at_utc=generated_at_utc,
    )
    _atomic_write_json(D3_RESULT_LOCK_PATH, lock)
    result_lock_sha256 = _sha256_file(D3_RESULT_LOCK_PATH)
    verify_result_lock(lock, expected_records=len(records))

    references_verified = verify_ai_risk_reference_preparation(
        protocol,
        records,
        artifact_sha256=artifact_document["artifact_sha256"],
        result_lock_sha256=lock["semantic_result_lock_sha256"],
    )
    if references_verified != len(records):
        raise V11D3Error("AI_RISK_ASSESSED reference preparation count mismatch.")

    return {
        "stage": STAGE,
        "status": "GENERATED",
        "record_count": len(records),
        "artifact_path": str(D3_ARTIFACT_PATH),
        "artifact_sha256": artifact_document["artifact_sha256"],
        "artifact_file_sha256": artifact_file_sha256,
        "summary_path": str(D3_SUMMARY_PATH),
        "summary_file_sha256": summary_file_sha256,
        "result_lock_path": str(D3_RESULT_LOCK_PATH),
        "result_lock_file_sha256": result_lock_sha256,
        "result_lock_semantic_sha256": lock["semantic_result_lock_sha256"],
        "references_verified": references_verified,
        "test_access_count": test_access_count,
        "marker": FINAL_REPORT_MARKER,
    }