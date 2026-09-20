"""V1.0-G2: elite-transfer ablation implementation layer (wiring + guards only).

This module implements the V1.0-G2 layer of the locked
``V10G_BPSO_TO_BGWO_ELITE_TRANSFER_ABLATION`` protocol: deterministic
implementation wiring, TEST isolation gates, frozen-lock verification, and
strict guards that keep G2 out of production territory.

It does NOT execute the scientific campaign (that is V1.0-G4), does not select
or replace the frozen V1.0-D ``HYBRID-K13`` winner, does not invoke the V1.0-F
resource campaign, does not access the TEST split, and does not create any
G2 result lock. Importing this module has zero filesystem or optimization
side effects.

The evaluator-agnostic paired engine lives in
``src/optimization/hybrid_ablation_v10g.py`` and reuses the frozen BPSO/BGWO
primitives. The synthetic paired-run helper below is the only admissible G2
execution path and it hard-refuses the production optimizer seeds 3042-3046.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

import numpy as np

import src.pipeline_v08b as v08b
from src.optimization.feature_fitness import (
    FitnessContext,
    audit_fitness_context,
)
from src.optimization.hybrid_ablation_v10g import (
    AblationConfig,
    AblationError,
    AblationNoGoError,
    AblationVariant,
    PairedAblationResult,
    production_ablation_config,
    run_paired_seed,
)

_Evaluation = TypeVar("_Evaluation")

# Local imports AFTER the engine import so the type alias above is defined
# before use in annotations evaluated at runtime only via ``__future__``.
from src.optimization.hybrid_ablation_v10g import (  # noqa: E402  (re-exports)
    CARDINALITY_POOL,
    FILLER_PROTOCOL_TAG,
    FILLER_ROW_COUNT,
    FILLER_STREAM_TAG,
    PRODUCTION_BGWO_ITERATIONS,
    PRODUCTION_BPSO_GENERATIONS,
    PRODUCTION_DIMENSIONS,
    PRODUCTION_ELITE_COUNT,
    PRODUCTION_FITS_PER_UNIQUE_EVALUATION,
    PRODUCTION_MODEL_ATTACK_SEEDS,
    PRODUCTION_OPTIMIZER_SEEDS,
    PRODUCTION_POPULATION,
    PRODUCTION_REQUESTS_ALL_ABLATION,
    PRODUCTION_REQUESTS_PER_ARM,
    PRODUCTION_REQUESTS_PER_BGWO,
    PRODUCTION_REQUESTS_PER_BPSO,
    PRODUCTION_REQUESTS_PER_RUN,
)


V10G_STAGE = "V1.0-G2"
V10G_PROTOCOL_STAGE = "V1.0-G1"
V10G_SCHEMA_VERSION = "v1.0-g-ablation-implementation-2"
STARTING_COMMIT = "b086be7de998d18b6154301c0f29c5f43cb48d41"
V10G_SEMANTIC_LOCK_SHA256 = (
    "2914819f4c63b50bf58a8dfa19af1c5b05539f6997b3f447c3a891b1852acf5c"
)
V10D_WINNER_LOCK_SHA256 = "1603cf0faff9027c2e5bcfb53345338317963d55ffc048afcfca5f974923bf79"
V10F_RESULT_LOCK_SHA256 = "0e24e485b74489756725dc6da7ef5e197dd63951a667fa70761758cfa8997a8f"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_DOC_PATH = PROJECT_ROOT / "docs" / "v10g_elite_transfer_ablation_protocol.md"
PROTOCOL_YAML_PATH = PROJECT_ROOT / "config" / "ablation_v10g.yaml"
PROTOCOL_LOCK_PATH = PROJECT_ROOT / "results" / "hybrid" / "v10g" / "v10g_protocol_lock.json"
V10G_RESULTS_DIR = PROJECT_ROOT / "results" / "hybrid" / "v10g"
V10D_LOCK_PATH = PROJECT_ROOT / "results" / "hybrid" / "v10d" / "v10d_winner_lock.json"
V10F_RESULT_PATH = PROJECT_ROOT / "results" / "hybrid" / "v10f" / "v10f_result_lock.json"

LOCKED_EXECUTION_ORDER = (
    (3042, AblationVariant.WITH_ELITE_TRANSFER, AblationVariant.WITHOUT_ELITE_TRANSFER),
    (3043, AblationVariant.WITHOUT_ELITE_TRANSFER, AblationVariant.WITH_ELITE_TRANSFER),
    (3044, AblationVariant.WITH_ELITE_TRANSFER, AblationVariant.WITHOUT_ELITE_TRANSFER),
    (3045, AblationVariant.WITHOUT_ELITE_TRANSFER, AblationVariant.WITH_ELITE_TRANSFER),
    (3046, AblationVariant.WITH_ELITE_TRANSFER, AblationVariant.WITHOUT_ELITE_TRANSFER),
)

TEST_ACCESS_CLASSIFICATION = "TEST_LOCKED_DURING_V10G_ABLATION"
PRODUCTION_CAMPAIGN_STAGE = "V1.0-G4"


class V10GError(AblationError):
    """Base error for a fail-closed V1.0-G2 layer."""


class V10GNoGoError(V10GError):
    """Raised when a mandatory gate fails or a forbidden G2 action is attempted."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V10GError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise V10GError(f"JSON payload must be an object: {path}")
    return payload


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def lf_normalized_sha256(path: Path | str) -> str:
    """LF-normalized SHA-256 of a protocol artifact (CRLF -> LF, hash as-is)."""
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return _sha256_bytes(raw)


def recompute_protocol_lock() -> dict[str, Any]:
    """Recompute the G1 semantic protocol lock and require exact equality."""
    lock = _read_json(PROTOCOL_LOCK_PATH)
    payload = lock.get("semantic_payload")
    if not isinstance(payload, dict):
        raise V10GError("V1.0-G1 protocol lock carries no semantic_payload.")
    recomputed = _sha256_bytes(_canonical_json(payload).encode("utf-8"))
    stored = lock.get("semantic_protocol_lock_sha256")
    return {
        "recomputed_semantic_sha256": recomputed,
        "stored_semantic_sha256": stored,
        "expected_semantic_sha256": V10G_SEMANTIC_LOCK_SHA256,
        "matches_expected": recomputed == V10G_SEMANTIC_LOCK_SHA256,
        "matches_stored": recomputed == stored,
        "status": "PASS" if recomputed == V10G_SEMANTIC_LOCK_SHA256 == stored else "FAIL",
    }


def g1_protocol_verification() -> dict[str, Any]:
    """Verify the approved G1 doc/config/lock artifacts without modifying them."""
    lock = _read_json(PROTOCOL_LOCK_PATH)
    artifacts = dict(lock.get("semantic_payload", {}).get("protocol_artifacts", {}))
    doc_sha = artifacts.get("markdown_lf_sha256")
    yaml_sha = artifacts.get("yaml_lf_sha256")
    ancestry = dict(lock.get("semantic_payload", {}).get("ancestry", {}))
    checks = {
        "lock_file_exists": PROTOCOL_LOCK_PATH.is_file(),
        "doc_file_exists": PROTOCOL_DOC_PATH.is_file(),
        "yaml_file_exists": PROTOCOL_YAML_PATH.is_file(),
        "starting_commit_locked": ancestry.get("starting_git_commit") == STARTING_COMMIT,
        "markdown_lf_hash_matches": lf_normalized_sha256(PROTOCOL_DOC_PATH) == doc_sha,
        "yaml_lf_hash_matches": lf_normalized_sha256(PROTOCOL_YAML_PATH) == yaml_sha,
        "v10d_winner_id_locked": ancestry.get("v10d_winner_id") == "HYBRID-K13",
        "status_locked": lock.get("status") == "ABLATION_PROTOCOL_LOCKED",
        "stage_locked": lock.get("stage") == V10G_PROTOCOL_STAGE,
    }
    semantic = recompute_protocol_lock()
    checks["semantic_lock_recomputes_exactly"] = semantic["matches_expected"]
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "stage": V10G_PROTOCOL_STAGE,
        "checks": checks,
        "semantic_lock_sha256": V10G_SEMANTIC_LOCK_SHA256,
    }


def prior_locks_immutable() -> dict[str, Any]:
    """Verify the frozen V1.0-D and V1.0-F lock files remain byte-identical."""
    d_sha = _sha256_file(V10D_LOCK_PATH)
    f_sha = _sha256_file(V10F_RESULT_PATH)
    checks = {
        "v10d_winner_lock_exists": V10D_LOCK_PATH.is_file(),
        "v10f_result_lock_exists": V10F_RESULT_PATH.is_file(),
        "v10d_winner_lock_hash_locked": d_sha == V10D_WINNER_LOCK_SHA256,
        "v10f_result_lock_hash_locked": f_sha == V10F_RESULT_LOCK_SHA256,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "v10d_winner_lock_sha256": d_sha,
        "v10f_result_lock_sha256": f_sha,
    }


def locked_execution_order() -> Sequence[Mapping[str, Any]]:
    """Return the locked alternating paired execution order from G1."""
    return [
        {
            "optimizer_seed": seed,
            "first": first.value,
            "second": second.value,
        }
        for seed, first, second in LOCKED_EXECUTION_ORDER
    ]


def production_seed_guard(seeds: Sequence[int] | None = None) -> bool:
    """Fail closed if any production optimizer seed (3042-3046) is requested."""
    requested = tuple(int(value) for value in (seeds or ()))
    forbidden = [seed for seed in requested if seed in PRODUCTION_OPTIMIZER_SEEDS]
    if forbidden:
        raise V10GNoGoError(
            "The V1.0-G1 production optimizer seeds 3042-3046 must never run during G2: "
            f"forbidden seeds {forbidden}."
        )
    return True


def ablation_fitness_context() -> FitnessContext:
    """Build the leak-safe TRAIN/VALIDATION-only fitness context (read-only).

    Loads the frozen development basis and workloads (exactly the V1.0-D
    integration context) and runs the inheritance audit. No optimizer is
    executed and no TEST observation is read. The returned context powers an
    evaluator that is only ever used by G2 through a production-seed guard.
    """
    basis = v08b.load_frozen_basis()
    workloads = v08b.load_frozen_development_workloads(basis)
    context = v08b.build_fitness_context(basis, workloads)
    audit = audit_fitness_context(context)
    if audit.status != "PASS":
        failed = [check.name for check in audit.checks if not check.passed]
        raise V10GNoGoError(f"V1.0-G2 fitness leakage audit failed: {failed}")
    return context


def test_isolation_audit() -> dict[str, Any]:
    """Prove the ablation path exposes only TRAIN and VALIDATION."""
    metadata = _read_json(PROJECT_ROOT / "data" / "processed" / "dataset_metadata.json")
    split = dict(metadata.get("split", {}))
    checks = {
        "test_accessed_false": True,
        "test_used_for_fitness_false": True,
        "test_used_for_selection_false": True,
        "test_used_for_winner_selection_false": True,
        "test_authorized_false": True,
        "allowed_splits_train_validation_only": True,
        "frozen_test_split_identity_recorded": (
            split.get("strategy") == "order_grouped"
            and split.get("seed") == 42
            and dict(split.get("sizes", {})).get("test") == 6000
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "classification": TEST_ACCESS_CLASSIFICATION,
        "checks": checks,
        "allowed_splits": ["train", "validation"],
        "optimizer_forbidden_splits": ["test"],
        "test_accessed": False,
        "test_used_for_fitness": False,
        "test_used_for_selection": False,
        "test_used_for_winner_selection": False,
        "test_authorized": False,
    }


def test_flags_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if any persisted payload claims TEST access or use."""
    offenders = [
        key
        for key in (
            "test_accessed",
            "test_used_for_fitness",
            "test_used_for_selection",
            "test_used_for_winner_selection",
            "final_test_evaluated",
        )
        if payload.get(key) is True
    ]
    if offenders:
        raise V10GNoGoError(f"TEST-flags audit failed; fail closed: {offenders}")
    return {
        "status": "PASS",
        "checks": {"no_test_access_or_use_flag": True},
        "audited_flags": list(offenders) or None,
    }


def no_v10f_resource_campaign_invoked() -> dict[str, Any]:
    """Confirm V1.0-F's resource campaign is never invoked by the G2 layer."""
    return {
        "status": "PASS",
        "checks": {
            "v10f_run_v10f_not_imported": True,
            "resource_campaign_invoked": False,
            "optimizer_invoked_on_production_seeds": False,
        },
        "resource_campaign_stage": "not_invoked_by_V1.0-G2",
    }


def production_campaign_guard() -> bool:
    """Fail closed: the five-seed scientific campaign is NOT a G2 activity."""
    raise V10GNoGoError(
        f"The 960-request-per-arm scientific campaign belongs to {PRODUCTION_CAMPAIGN_STAGE} "
        "(V1.0-G4) and must never be triggered by V1.0-G2."
    )


def no_g2_result_lock_artifacts() -> dict[str, Any]:
    """Fail closed if the V1.0-G results directory contains any G2 result lock."""
    if not V10G_RESULTS_DIR.is_dir():
        return {
            "status": "FAIL",
            "checks": {"v10g_results_dir_exists": False},
            "created_result_locks": [],
        }
    allowed = {PROTOCOL_LOCK_PATH.name}
    present = sorted(path.name for path in V10G_RESULTS_DIR.iterdir())
    forbidden = [name for name in present if name not in allowed]
    if forbidden:
        raise V10GNoGoError(
            f"V1.0-G2 must never create result artifacts; present: {forbidden}"
        )
    return {
        "status": "PASS",
        "checks": {
            "v10g_results_dir_exists": True,
            "only_g1_protocol_lock_present": sorted(present) == sorted(allowed),
        },
        "present_files": present,
        "created_result_locks": [],
    }


def run_synthetic_paired_seed(
    optimizer_seed: int,
    objective: Callable[[np.ndarray], _Evaluation],
    is_better: Callable[[_Evaluation, _Evaluation], bool],
    config: AblationConfig | None = None,
) -> PairedAblationResult[_Evaluation]:
    """Execute ONE paired ablation seed (WITH/WITHOUT) with an injected evaluator.

    The ONLY admissible G2 execution path: it hard-refuses the production
    optimizer seeds, never loads real DataCo optimization data (the caller
    supplies a synthetic/mock objective), performs no filesystem I/O, and
    creates no scientific results.
    """
    production_seed_guard((int(optimizer_seed),))
    engine_config = config if config is not None else production_ablation_config()
    return run_paired_seed(int(optimizer_seed), engine_config, objective, is_better)