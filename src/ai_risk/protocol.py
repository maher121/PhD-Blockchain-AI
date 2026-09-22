"""Strict loader and validator for the frozen V1.1-D AI risk decisions.

Locks the D1 protocol/config and D1 lock artifact together: any drift in the
config file, the frozen protocol document, or the stored D1 semantic lock
fails closed before a single risk operation can be attempted. This module is
a dependency leaf of the D2 package: it reads ``config/ai_risk_v11d.yaml``
and ``results/blockchain/v11d/v11d_ai_risk_protocol_lock.json`` only, and
imports nothing else from the D2 package.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..blockchain_engine.canonical import canonical_json, sha256_hex

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

AI_RISK_CONFIG_PATH = PROJECT_ROOT / "config" / "ai_risk_v11d.yaml"
AI_RISK_PROTOCOL_DOC_PATH = (
    PROJECT_ROOT / "docs" / "v11d_governed_ai_risk_protocol.md"
)
AI_RISK_PROTOCOL_LOCK_PATH = (
    PROJECT_ROOT / "results" / "blockchain" / "v11d" / "v11d_ai_risk_protocol_lock.json"
)

STAGE = "V1.1-D1"
PROTOCOL_VERSION = "v1.1-d1-governed-ai-risk-1"
KIND = "SCIENTIFIC_DECISION_PROTOCOL_LOCK"

D1_SEMANTIC_LOCK_SHA256 = "917ad246e85d5086d71f8d8850af6d145345589af956461165099434f812e3d3"


class AIRiskProtocolError(ValueError):
    """Raised when a frozen D1 protocol identity or linkage fails validation."""


@dataclass(frozen=True)
class AIRiskProtocol:
    """Verified view over the frozen D1 config plus its lock linkage."""

    config: Mapping[str, Any]
    config_sha256: str
    protocol_doc_sha256: str
    lock: Mapping[str, Any]
    lock_semantic_sha256: str
    config_path: Path = AI_RISK_CONFIG_PATH
    protocol_doc_path: Path = AI_RISK_PROTOCOL_DOC_PATH
    lock_path: Path = AI_RISK_PROTOCOL_LOCK_PATH

    def decision(self, key: str) -> Mapping[str, Any]:
        return self.config[key]

    def feature_identity(self) -> Mapping[str, Any]:
        return self.config["feature_identity"]

    def features(self) -> tuple[str, ...]:
        return tuple(self.config["feature_identity"]["ordered_features"])

    def seeds(self) -> tuple[int, ...]:
        return tuple(self.config["model_strategy"]["seeds"])

    def classifier_parameters(self) -> Mapping[str, Any]:
        return self.config["classifier"]["parameters"]


def _require_file(path: Path) -> None:
    if not path.exists():
        raise AIRiskProtocolError(f"Missing frozen D1 artifact: {path}")
    if not path.is_file():
        raise AIRiskProtocolError(f"Not a regular file: {path}")


def _sha256_file(path: Path) -> str:
    _require_file(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    _require_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AIRiskProtocolError(f"Frozen lock is not a JSON object: {path}")
    return payload


def _verify_config_shape(config: Mapping[str, Any]) -> None:
    if config.get("stage") != STAGE:
        raise AIRiskProtocolError(
            f"config stage must be {STAGE!r}, got {config.get('stage')!r}"
        )
    if config.get("kind") != KIND:
        raise AIRiskProtocolError(
            f"config kind must be {KIND!r}, got {config.get('kind')!r}"
        )
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise AIRiskProtocolError(
            f"protocol_version must be {PROTOCOL_VERSION!r}, "
            f"got {config.get('protocol_version')!r}"
        )
    for section in ("feature_identity", "classifier", "model_strategy", "risk_score",
                    "row_to_order", "thresholds", "risk_record", "blockchain_linkage",
                    "generation_partition", "training_reconstruction"):
        if section not in config:
            raise AIRiskProtocolError(f"config is missing frozen section {section!r}")


def _verify_lock_shape(lock: Mapping[str, Any]) -> None:
    if lock.get("stage") != STAGE:
        raise AIRiskProtocolError(
            f"lock stage must be {STAGE!r}, got {lock.get('stage')!r}"
        )
    if lock.get("artifact_kind") != "V11D1_AI_RISK_PROTOCOL_LOCK":
        raise AIRiskProtocolError(
            f"lock artifact_kind must be V11D1_AI_RISK_PROTOCOL_LOCK, "
            f"got {lock.get('artifact_kind')!r}"
        )
    if "semantic_payload" not in lock:
        raise AIRiskProtocolError("lock is missing its semantic_payload")


def _recompute_lock_semantic(lock: Mapping[str, Any]) -> str:
    return sha256_hex(lock["semantic_payload"])


def verify_d1_lock_semantic(
    lock: Mapping[str, Any],
    *,
    expected: str = D1_SEMANTIC_LOCK_SHA256,
) -> None:
    """Recompute the D1 lock semantic hash and require constant equality."""
    recomputed = _recompute_lock_semantic(lock)
    stored = lock.get("semantic_result_lock_sha256")
    if stored is None:
        raise AIRiskProtocolError("lock is missing semantic_result_lock_sha256")
    if stored != recomputed:
        raise AIRiskProtocolError(
            "D1 lock semantic hash does not match its semantic_payload "
            f"(stored {stored}, recomputed {recomputed})"
        )
    if recomputed != expected:
        raise AIRiskProtocolError(
            "D1 lock semantic hash drifted from the frozen constant "
            f"(expected {expected}, recomputed {recomputed})"
        )


def verify_config_protocol_linkage(
    config_sha256: str,
    protocol_doc_sha256: str,
    lock: Mapping[str, Any],
) -> None:
    """Bind the D1 config and protocol document hashes into the lock."""
    fingerprints = lock["semantic_payload"]["artifact_fingerprints"]
    expected_config = fingerprints["protocol_config_sha256"]
    expected_doc = fingerprints["protocol_document_sha256"]
    if config_sha256 != expected_config:
        raise AIRiskProtocolError(
            "config/ai_risk_v11d.yaml drifted from the D1 lock "
            f"(expected {expected_config}, actual {config_sha256})"
        )
    if protocol_doc_sha256 != expected_doc:
        raise AIRiskProtocolError(
            "docs/v11d_governed_ai_risk_protocol.md drifted from the D1 lock "
            f"(expected {expected_doc}, actual {protocol_doc_sha256})"
        )


def load_verified_protocol(
    *,
    config_path: Path = AI_RISK_CONFIG_PATH,
    protocol_doc_path: Path = AI_RISK_PROTOCOL_DOC_PATH,
    lock_path: Path = AI_RISK_PROTOCOL_LOCK_PATH,
) -> AIRiskProtocol:
    """Load and strictly verify the frozen D1 config and lock artifacts.

    Every verifiable linkage is re-derived from the artifacts on disk, so a
    stale or tampered file fails closed instead of being silently reused.
    """
    config_path = Path(config_path)
    protocol_doc_path = Path(protocol_doc_path)
    lock_path = Path(lock_path)

    _require_file(config_path)
    _require_file(protocol_doc_path)
    _require_file(lock_path)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise AIRiskProtocolError("frozen config is not a YAML mapping")
    _verify_config_shape(config)

    lock = _load_json(lock_path)
    _verify_lock_shape(lock)
    verify_d1_lock_semantic(lock)

    config_sha256 = _sha256_file(config_path)
    protocol_doc_sha256 = _sha256_file(protocol_doc_path)
    verify_config_protocol_linkage(config_sha256, protocol_doc_sha256, lock)

    return AIRiskProtocol(
        config=config,
        config_sha256=config_sha256,
        protocol_doc_sha256=protocol_doc_sha256,
        lock=lock,
        lock_semantic_sha256=_recompute_lock_semantic(lock),
        config_path=config_path,
        protocol_doc_path=protocol_doc_path,
        lock_path=lock_path,
    )


def canonical_json_for_test(value: Any) -> str:
    """Expose the frozen canonical serialization for cross-module tests."""
    return canonical_json(value)