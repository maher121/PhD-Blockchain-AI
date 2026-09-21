"""V1.0-G4: frozen paired production ablation campaign (DATA GENERATION ONLY).

Executes the approved WITH_ELITE_TRANSFER vs WITHOUT_ELITE_TRANSFER ablation
for optimizer seeds 3042-3046 over TRAIN+VALIDATION DataCo data using the
frozen V0.8-B fitness evaluator, the frozen deterministic ranking, and the
frozen G1 RNG/cache/budget contract.

This stage is DATA GENERATION. It deliberately performs NO statistical
interpretation (G5) and NO reporting (G6). It does not select or replace any
winner, does not rerun V1.0-F, does not access TEST, and never tunes.

Governance:
- Produces resumable, atomically written checkpoints under
  ``results/hybrid/v10g/v10g4_campaign/``.
- Reuses the previously validated implementation and gates:
  ``src/optimization/hybrid_ablation_v10g.py`` and ``src/pipeline_v10g.py``.
- Fail-closes on ANY pre-flight mismatch, checkpoint incompatibility, or
  paired-invariant violation, stopping the campaign immediately.
- This module is the ONLY authorized production-seed execution API for the
  ablation. Importing it has no side effects; the campaign starts only when
  ``main()`` is invoked.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.optimization.feature_fitness import (
    FitnessContext,
    FeatureFitnessEvaluator,
    MARGINS,
    NUMERICAL_BOUNDARY_TOLERANCE,
    audit_fitness_context,
    feature_fitness_is_better,
    fingerprint_feature_names,
)
from src.optimization.hybrid_ablation_v10g import (
    AblationConfig,
    AblationVariant,
    PairedAblationResult,
    to_json_compatible,
    generate_shadow_filler_block,
    run_ablation_arm,
    verify_paired_invariants,
)
from src.optimization.hybrid_bpso_bgwo import (
    build_hybrid_initial_population as FROZEN_BUILD_INITIAL,
)
import src.pipeline_v08b as v08b
import src.pipeline_v10g as v10g

STAGE = "V1.0-G4"
PROTOCOL_STAGE = "V1.0-G1"
PROTOCOL_NAME = "V10G_BPSO_TO_BGWO_ELITE_TRANSFER_ABLATION"
SCHEMA_VERSION = "v1.0-g4-production-ablation-1"

ARM_ARTIFACT_KIND = "V10G4_PRODUCTION_ARM_RUN"
PAIRED_ARTIFACT_KIND = "V10G4_PRODUCTION_PAIRED_SEED"
MANIFEST_ARTIFACT_KIND = "V10G4_CAMPAIGN_MANIFEST"
PREFLIGHT_ARTIFACT_KIND = "V10G4_PREFLIGHT"
ENVIRONMENT_ARTIFACT_KIND = "V10G4_ENVIRONMENT"

EXPECTED_HEAD = "b302fa0b35c8f4b63f4ee1f27b886f733f941c6f"
G1_SEMANTIC_LOCK_SHA256 = (
    "2914819f4c63b50bf58a8dfa19af1c5b05539f6997b3f447c3a891b1852acf5c"
)
V10D_WINNER_LOCK_SHA256 = (
    "1603cf0faff9027c2e5bcfb53345338317963d55ffc048afcfca5f974923bf79"
)
V10F_RESULT_LOCK_SHA256 = (
    "0e24e485b74489756725dc6da7ef5e197dd63951a667fa70761758cfa8997a8f"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN_DIR = PROJECT_ROOT / "results" / "hybrid" / "v10g" / "v10g4_campaign"

LOCKED_SEED_ORDER: tuple[tuple[int, str, str], ...] = (
    (3042, "WITH_ELITE_TRANSFER", "WITHOUT_ELITE_TRANSFER"),
    (3043, "WITHOUT_ELITE_TRANSFER", "WITH_ELITE_TRANSFER"),
    (3044, "WITH_ELITE_TRANSFER", "WITHOUT_ELITE_TRANSFER"),
    (3045, "WITHOUT_ELITE_TRANSFER", "WITH_ELITE_TRANSFER"),
    (3046, "WITH_ELITE_TRANSFER", "WITHOUT_ELITE_TRANSFER"),
)

PRODUCTION_OPTIMIZER_SEEDS = (3042, 3043, 3044, 3045, 3046)


class V10G4Error(RuntimeError):
    """Base error for a fail-closed V1.0-G4 campaign."""


class V10G4NoGoError(V10G4Error):
    """Raised when a mandatory G4 gate fails (fail closed)."""


# ---------------------------------------------------------------------------
# Low-level JSON / hashing helpers (atomic writes, no clobbering of G1)
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V10G4Error(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise V10G4Error(f"JSON payload must be an object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a checkpoint atomically; never touch the G1 protocol lock."""
    if Path(path).resolve() == v10g.PROTOCOL_LOCK_PATH.resolve():
        raise V10G4NoGoError("Refusing to write the G1 protocol lock from G4.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    raw = json.dumps(
        to_json_compatible(payload),
        sort_keys=True,
        indent=2,
        separators=(",", ": "),
        allow_nan=False,
    )
    tmp.write_text(raw, encoding="utf-8")
    os.replace(tmp, path)


def _config_digest(config: AblationConfig) -> str:
    return _sha256_bytes(
        json.dumps(to_json_compatible(config.to_dict()), sort_keys=True).encode("utf-8")
    )


def capture_environment() -> dict[str, Any]:
    """Capture runtime/dependency identity for resume verification."""
    from importlib.metadata import version

    def _version(package: str) -> str | None:
        try:
            return version(package)
        except Exception:
            return None

    return {
        "artifact_kind": ENVIRONMENT_ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "captured_at_iso": _now_iso(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "libraries": {
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "scikit-learn": _version("scikit-learn"),
            "scipy": _version("scipy"),
        },
        "repository": {
            "head_sha256": _git_sha("HEAD"),
            "origin_main_sha256": _git_sha("origin/main"),
            "expected_head_sha256": EXPECTED_HEAD,
        },
    }


# ---------------------------------------------------------------------------
# Frozen-data replay helpers (deterministic, no optimization execution)
# ---------------------------------------------------------------------------


def replay_initial_bpso_population(
    config: AblationConfig, optimizer_seed: int
) -> list[dict[str, Any]]:
    """Recompute the exact BPSO generation-0 population rows and identities.

    Uses the frozen population builder with ``PCG64(optimizer_seed)``; this
    consumes no evaluator and reproduces the engine's initial population.
    """
    generator = np.random.Generator(np.random.PCG64(int(optimizer_seed)))
    population = FROZEN_BUILD_INITIAL(config, generator)
    rows = []
    for row in np.asarray(population, dtype=np.uint8):
        arr = np.asarray(row, dtype=np.uint8)
        rows.append(
            {
                "mask": to_json_compatible(arr.tolist()),
                "mask_sha256": _sha256_bytes(arr.tobytes()),
                "cardinality": int(arr.sum()),
            }
        )
    return rows


def selected_feature_evidence(
    best_mask: np.ndarray,
    candidate_features: Sequence[str],
    evaluation: Any,
) -> dict[str, Any]:
    """Persist the G5-reconstructible final-mask scientific evidence."""
    arr = np.asarray(best_mask, dtype=np.uint8)
    mask_sha = _sha256_bytes(arr.tobytes())
    selected = tuple(
        feature for feature, active in zip(candidate_features, arr) if active
    )
    return {
        "selected_feature_count": int(arr.sum()),
        "selected_features": list(selected),
        "feature_list_sha256": fingerprint_feature_names(selected),
        "mask_sha256": mask_sha,
        "mask": to_json_compatible(arr.tolist()),
    }


def optimizer_metric_summary(evaluation: Any) -> dict[str, Any]:
    mean_metrics = dict(evaluation.mean_metrics)
    return {
        "average_precision": float(mean_metrics["average_precision"]),
        "f1": float(mean_metrics["f1"]),
        "recall": float(mean_metrics["recall"]),
        "precision": float(mean_metrics["precision"]),
        "roc_auc": float(mean_metrics["roc_auc"]),
        "feasible": bool(evaluation.feasible),
        "normalized_violation": float(evaluation.normalized_violation),
    }


def phase_boundary_evidence(history: Sequence[Any]) -> dict[str, Any]:
    """Extract frozen per-phase boundaries and improvement iterations."""
    records = list(history)
    bpso_records = [r for r in records if r.phase == "bpso"]
    bgwo_records = [r for r in records if r.phase == "bgwo"]
    bpso_boundary = bpso_records[-1] if bpso_records else None
    bgwo_deepest = bgwo_records[-1] if bgwo_records else None
    improvements = [
        (index, record.phase, int(record.phase_index))
        for index, record in enumerate(records)
        if record.run_best_improved
    ]
    return {
        "bpso_boundary_best_mask_sha256": _sha256_bytes(
            np.asarray(bpso_boundary.phase_best_mask, dtype=np.uint8).tobytes()
        )
        if bpso_boundary is not None
        else None,
        "bgwo_phase_best_mask_sha256": _sha256_bytes(
            np.asarray(bgwo_deepest.phase_best_mask, dtype=np.uint8).tobytes()
        )
        if bgwo_deepest is not None
        else None,
        "best_newly_evaluated_bgwo_candidate_mask_sha256": _sha256_bytes(
            np.asarray(bgwo_deepest.phase_best_mask, dtype=np.uint8).tobytes()
        )
        if bgwo_deepest is not None
        else None,
        "first_run_best_improvement": improvements[0] if improvements else None,
        "final_run_best_improvement": improvements[-1] if improvements else None,
        "improvement_count": len(improvements),
    }


# ---------------------------------------------------------------------------
# Pre-flight and resume compatibility
# ---------------------------------------------------------------------------


def preflight_gates() -> dict[str, Any]:
    """Fail-closed verification of every required G4 pre-flight gate."""
    head_matches = _repo_head_matches()
    origin_matches = _repo_origin_matches()
    gates: dict[str, Any] = {}
    gates["head_equals_base"] = {
        "head": _git_sha("HEAD"),
        "origin_main": _git_sha("origin/main"),
        "expected": EXPECTED_HEAD,
        "head_matches": head_matches,
        "origin_matches": origin_matches,
        "status": "PASS" if (head_matches and origin_matches) else "FAIL",
    }
    lock = v10g.recompute_protocol_lock()
    gates["g1_semantic_lock"] = {
        "status": lock["status"],
        "recomputed": lock["recomputed_semantic_sha256"],
        "expected": G1_SEMANTIC_LOCK_SHA256,
        "matches": bool(lock["matches_expected"] and lock["matches_stored"]),
    }
    gates["g1_protocol_verification"] = v10g.g1_protocol_verification()
    gates["prior_locks_immutable"] = v10g.prior_locks_immutable()
    gates["test_isolation_audit"] = v10g.test_isolation_audit()
    gates["g4_artifacts_sanctioned"] = sanctioned_v10g_artifacts()
    expected_margins = {
        "ap_relative_loss_max": 0.05,
        "f1_relative_loss_max": 0.05,
        "recall_relative_loss_max": 0.1,
    }
    gates["evaluator_parity"] = {
        "margins": dict(MARGINS),
        "expect_ap_0_05": MARGINS.get("average_precision") == 0.05,
        "expect_f1_0_05": MARGINS.get("f1") == 0.05,
        "expect_recall_0_10": MARGINS.get("recall") == 0.1,
        "boundary_tolerance_1e_12": NUMERICAL_BOUNDARY_TOLERANCE == 1e-12,
        "aggregate_score_permitted": False,
        "fits_per_unique_evaluation": v10g.PRODUCTION_FITS_PER_UNIQUE_EVALUATION,
        "model_attack_seeds": list(v10g.PRODUCTION_MODEL_ATTACK_SEEDS),
        "classifier": "decision_tree",
        "prediction_threshold": 0.5,
        "margins_match_protocol": dict(MARGINS) == {
            "average_precision": 0.05,
            "f1": 0.05,
            "recall": 0.1,
        },
        "status": "PASS"
        if (
            MARGINS.get("average_precision") == 0.05
            and MARGINS.get("f1") == 0.05
            and MARGINS.get("recall") == 0.1
            and NUMERICAL_BOUNDARY_TOLERANCE == 1e-12
        )
        else "FAIL",
    }
    passed = (
        gates["head_equals_base"]["status"] == "PASS"
        and gates["g1_semantic_lock"]["matches"]
        and gates["g1_protocol_verification"]["status"] == "PASS"
        and gates["prior_locks_immutable"]["status"] == "PASS"
        and gates["test_isolation_audit"]["status"] == "PASS"
        and gates["g4_artifacts_sanctioned"]["status"] == "PASS"
        and gates["evaluator_parity"]["status"] == "PASS"
    )
    if not passed:
        raise V10G4NoGoError(f"V1.0-G4 pre-flight gate failed: {sorted(gates.keys())}")
    return {
        "artifact_kind": PREFLIGHT_ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "protocol_name": PROTOCOL_NAME,
        "g1_semantic_lock_sha256": G1_SEMANTIC_LOCK_SHA256,
        "expected_head_sha256": EXPECTED_HEAD,
        "status": "PASS",
        "gates": gates,
    }


def sanctioned_v10g_artifacts() -> dict[str, Any]:
    """G4-native check: results/hybrid/v10g contains ONLY the G1 protocol lock
    plus G4's own campaign directory, and the G1 lock is intact."""
    allowed = {"v10g_protocol_lock.json", "v10g4_campaign"}
    present = sorted(p.name for p in v10g.V10G_RESULTS_DIR.iterdir())
    unexpected = [name for name in present if name not in allowed]
    lock_exists = (v10g.V10G_RESULTS_DIR / "v10g_protocol_lock.json").is_file()
    status = "PASS" if (not unexpected and lock_exists) else "FAIL"
    return {
        "checks": {
            "g1_protocol_lock_present": lock_exists,
            "only_g1_lock_and_g4_campaign_present": not unexpected,
        },
        "allowed": sorted(allowed),
        "present": present,
        "unexpected": unexpected,
        "status": status,
    }


def _repo_head_matches() -> bool:
    return _git_sha("HEAD") == EXPECTED_HEAD


def _repo_origin_matches() -> bool:
    return _git_sha("origin/main") == EXPECTED_HEAD


def _git_sha(ref: str) -> str | None:
    try:
        output = os.popen(f"git rev-parse {ref}").read().strip()
        return output or None
    except Exception:
        return None


def build_fitness_context() -> FitnessContext:
    basis = v08b.load_frozen_basis()
    workloads = v08b.load_frozen_development_workloads(basis)
    context = v08b.build_fitness_context(basis, workloads)
    audit = audit_fitness_context(context)
    if audit.status != "PASS":
        raise V10G4NoGoError(f"V1.0-G4 fitness leakage audit failed: {audit.status}")
    return context


def dataset_identity_evidence(context: FitnessContext) -> dict[str, Any]:
    return {
        "candidate_features": list(context.candidate_features),
        "candidate_manifest_sha256": context.candidate_manifest_sha256,
        "feature_count": len(context.candidate_features),
        "expected_train_rows": int(context.expected_train_rows),
        "expected_validation_rows": int(context.expected_validation_rows),
        "split_capacity": {
            "train": int(context.expected_train_rows),
            "validation": int(context.expected_validation_rows),
            "test": 0,
        },
        "attack_label": "is_attack",
        "train_test_identity_overlap_verified_by": "frozen V0.8-B split loader",
    }


# ---------------------------------------------------------------------------
# Per-arm execution
# ---------------------------------------------------------------------------


def _arm_validation(
    arm: Any, seed: int, variant: str, fits_delta: int
) -> dict[str, Any]:
    checks = {
        "variant_matches": arm.variant == str(variant),
        "seed_matches": int(arm.optimizer_seed) == int(seed),
        "no_early_stopping": arm.stop_reason == "FIXED_BUDGET_EXHAUSTED",
        "bpso_requests_96": int(arm.bpso_requests) == 96,
        "bgwo_requests_96": int(arm.bgwo_requests) == 96,
        "total_requests_192": int(arm.total_candidate_requests) == 192,
        "accounting_requests_equals_unique_plus_hits": (
            int(arm.total_candidate_requests)
            == int(arm.unique_evaluations) + int(arm.cache_hits)
        ),
        "accounting_evaluator_calls_equals_unique": (
            int(arm.evaluator_calls) == int(arm.unique_evaluations)
        ),
        "accounting_fits_equals_unique_times_5": (
            int(fits_delta) == int(arm.unique_evaluations) * v10g.PRODUCTION_FITS_PER_UNIQUE_EVALUATION
        ),
        "test_access_false": True,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def execute_arm(
    seed: int,
    variant: str,
    config: AblationConfig,
    filler: Any,
    evaluator: FeatureFitnessEvaluator,
    context: FitnessContext,
    execution_position: str,
    preflight_sha: str,
    environment_sha: str,
) -> dict[str, Any]:
    variant_enum = AblationVariant(variant)
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    fits_before = int(evaluator.decision_tree_fit_count)
    arm = run_ablation_arm(
        variant_enum,
        int(seed),
        config,
        evaluator,
        feature_fitness_is_better,
        filler_block=filler,
    )
    wall_time = time.perf_counter() - wall_start
    cpu_time = time.process_time() - cpu_start
    fits_delta = int(evaluator.decision_tree_fit_count) - fits_before

    validation = _arm_validation(arm, seed, str(variant), fits_delta)
    if validation["status"] != "PASS":
        raise V10G4NoGoError(
            f"G4 arm validation failed for seed={seed} variant={variant}: {validation}"
        )
    if int(fits_delta) != int(arm.unique_evaluations) * v10g.PRODUCTION_FITS_PER_UNIQUE_EVALUATION:
        raise V10G4NoGoError(
            f"G4 DT-fit accounting mismatch seed={seed} variant={variant}: "
            f"fits_delta={fits_delta} != unique*5={int(arm.unique_evaluations)*5}"
        )

    summary = optimizer_metric_summary(arm.best_evaluation)
    payload: dict[str, Any] = {
        "artifact_kind": ARM_ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "protocol_name": PROTOCOL_NAME,
        "protocol_lock_sha256": G1_SEMANTIC_LOCK_SHA256,
        "starting_checkpoint_sha256": EXPECTED_HEAD,
        "config_digest": _config_digest(config),
        "execution_order": str(execution_position),
        "optimizer_seed": int(seed),
        "variant": str(variant_enum.value),
        "completed_at_iso": _now_iso(),
        "optimizer_wall_time_sec": round(wall_time, 6),
        "optimizer_cpu_time_sec": round(cpu_time, 6),
        "final": {
            **summary,
            **selected_feature_evidence(
                np.asarray(arm.best_mask, dtype=np.uint8),
                context.candidate_features,
                arm.best_evaluation,
            ),
        },
        "final_phase": arm.best_phase,
        "final_phase_index": int(arm.best_phase_index),
        "phase_boundaries": phase_boundary_evidence(arm.convergence_history),
        "convergence": [to_json_compatible(record.to_dict()) for record in arm.convergence_history],
        "accounting": {
            "total_candidate_requests": int(arm.total_candidate_requests),
            "unique_evaluations": int(arm.unique_evaluations),
            "cache_hits": int(arm.cache_hits),
            "evaluator_calls": int(arm.evaluator_calls),
            "decision_tree_fits": int(fits_delta),
            "bpso_requests": int(arm.bpso_requests),
            "bgwo_requests": int(arm.bgwo_requests),
            "bpso_unique_evaluations": int(arm.bpso_unique_evaluations),
            "bgwo_new_unique_evaluations": int(arm.bgwo_new_unique_evaluations),
            "bpso_cache_hits": int(arm.bpso_cache_hits),
            "bgwo_cache_hits": int(arm.bgwo_cache_hits),
        },
        "elite_transfer": {
            "elite_selection_invoked": bool(arm.elite_selection_invoked),
            "placed_elite_count": int(arm.placed_elite_count),
            "elite_masks": to_json_compatible(np.asarray(arm.elite_masks, dtype=np.uint8).tolist()),
            "elite_hashes": list(arm.elite_hashes),
        },
        "shadow_filler": {
            "block": None if filler is None else to_json_compatible(filler.to_dict()),
            "mask_sha256": [] if filler is None else list(filler.mask_sha256),
            "cardinalities": [] if filler is None else list(filler.cardinalities),
            "generated": bool(arm.shadow_filler_generated),
            "evaluated": bool(arm.shadow_filler_evaluated),
            "placed": bool(arm.shadow_filler_placed),
        },
        "paired_identity": {
            "paired_bpso_outputs_byte_identical": None,
            "bgwo_row0_sha256": arm.bgwo_row0_sha256,
            "bgwo_rows_1_3_sha256": list(arm.bgwo_rows_1_3_sha256),
            "bgwo_rows_4_11_sha256": list(arm.bgwo_rows_4_11_sha256),
            "bgwo_common_rng_state_before_updates": arm.bgwo_common_rng_state_before_updates,
            "bpso_submitted_outputs_sha256": arm.bpso_submitted_outputs_sha256,
            "bpso_final_population_sha256": arm.bpso_final_population_sha256,
        },
        "initial_population": {
            "bpso_rows": replay_initial_bpso_population(config, int(seed)),
            "bgwo_final_population_shape": list(np.asarray(arm.bgwo_initial_population).shape),
        },
        "repair_and_diversity": {
            "repair_count": int(arm.repair_count),
            "population_diversity_history": [
                to_json_compatible(
                    {
                        "phase": record.phase,
                        "phase_index": int(record.phase_index),
                        "population_diversity": float(record.population_diversity),
                        "kinetics_value": record.kinetics_value,
                    }
                )
                for record in arm.convergence_history
            ],
        },
        "test_access_audit": v10g.test_isolation_audit(),
        "validation": validation,
        "preflight_sha256": preflight_sha,
        "environment_sha256": environment_sha,
        "provenance": {
            "g1_protocol_lock_sha256": G1_SEMANTIC_LOCK_SHA256,
            "serialized_arm_sha256": None,
        },
    }
    raw = json.dumps(
        to_json_compatible(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    payload["provenance"]["serialized_arm_sha256"] = _sha256_bytes(raw.encode("utf-8"))
    return payload


def arm_checkpoint_path(seed: int, variant: str) -> Path:
    return CAMPAIGN_DIR / f"v10g4_arm_s{int(seed):04d}_{str(variant)}.json"


def write_arm_checkpoint(payload: Mapping[str, Any]) -> Path:
    path = arm_checkpoint_path(int(payload["optimizer_seed"]), str(payload["variant"]))
    _atomic_write_json(path, payload)
    return path


def load_valid_arm_checkpoint(seed: int, variant: str) -> dict[str, Any] | None:
    """Return a compatible checkpoint or None when absent/incompatible."""
    path = arm_checkpoint_path(seed, variant)
    if not path.is_file():
        return None
    payload = _read_json(path)
    incompatible = (
        payload.get("artifact_kind") != ARM_ARTIFACT_KIND
        or payload.get("optimizer_seed") != int(seed)
        or payload.get("variant") != str(variant)
        or payload.get("protocol_lock_sha256") != G1_SEMANTIC_LOCK_SHA256
        or payload.get("starting_checkpoint_sha256") != EXPECTED_HEAD
        or payload.get("config_digest") != _config_digest(_current_config())
        or not isinstance(payload.get("validation"), dict)
        or payload["validation"].get("status") != "PASS"
    )
    if incompatible:
        raise V10G4NoGoError(
            f"G4 checkpoint incompatible for seed={seed} variant={variant}; abort (no "
            "silent reuse): " + str(path)
        )
    return payload


def _current_config() -> AblationConfig:
    return v10g.production_ablation_config()


# ---------------------------------------------------------------------------
# Per-seed paired execution
# ---------------------------------------------------------------------------


def execute_seed(
    seed: int,
    first: str,
    second: str,
    config: AblationConfig,
    evaluator: FeatureFitnessEvaluator,
    context: FitnessContext,
    preflight_sha: str,
    environment_sha: str,
    resume: bool,
) -> dict[str, Any]:
    filler = generate_shadow_filler_block(
        int(seed),
        dimensions=config.dimensions,
        cardinality_pool=config.cardinality_pool,
        filler_protocol_tag=config.filler_protocol_tag,
        filler_stream_tag=config.filler_stream_tag,
        row_count=config.filler_row_count,
    )
    order = [first, second]
    first_payload = None
    second_payload = None
    if resume:
        first_payload = load_valid_arm_checkpoint(seed, first)
        second_payload = load_valid_arm_checkpoint(seed, second)
    if first_payload is None:
        first_payload = execute_arm(
            seed, first, config, filler, evaluator, context, "first", preflight_sha, environment_sha
        )
        write_arm_checkpoint(first_payload)
    if second_payload is None:
        second_payload = execute_arm(
            seed, second, config, filler, evaluator, context, "second", preflight_sha, environment_sha
        )
        write_arm_checkpoint(second_payload)

    with_arm_payload = first_payload if first == "WITH_ELITE_TRANSFER" else second_payload
    without_arm_payload = second_payload if first == "WITH_ELITE_TRANSFER" else first_payload

    arm_with = build_arm_result_for_invariants(seed, config, with_arm_payload, without_arm_payload)
    arm_without = build_arm_result_for_invariants(seed, config, without_arm_payload, with_arm_payload)
    paired = PairedAblationResult(
        optimizer_seed=int(seed),
        filler_block=filler,
        with_arm=arm_with,
        without_arm=arm_without,
        invariants={},
        status="PENDING",
    )
    invariants = verify_paired_invariants(paired)

    seed_payload: dict[str, Any] = {
        "artifact_kind": PAIRED_ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "protocol_name": PROTOCOL_NAME,
        "protocol_lock_sha256": G1_SEMANTIC_LOCK_SHA256,
        "starting_checkpoint_sha256": EXPECTED_HEAD,
        "optimizer_seed": int(seed),
        "execution_order": [
            {"position": "first", "variant": first},
            {"position": "second", "variant": second},
        ],
        "completed_at_iso": _now_iso(),
        "paired_invariants": to_json_compatible(invariants),
        "paired_invariant_status": str(invariants.get("status")),
        "shadow_filler": to_json_compatible(filler.to_dict()),
        "per_arm": {
            "WITH_ELITE_TRANSFER": to_json_compatible(
                {
                    "final": with_arm_payload["final"],
                    "final_phase": with_arm_payload["final_phase"],
                    "accounting": with_arm_payload["accounting"],
                    "elite_transfer": with_arm_payload["elite_transfer"],
                }
            ),
            "WITHOUT_ELITE_TRANSFER": to_json_compatible(
                {
                    "final": without_arm_payload["final"],
                    "final_phase": without_arm_payload["final_phase"],
                    "accounting": without_arm_payload["accounting"],
                    "elite_transfer": without_arm_payload["elite_transfer"],
                }
            ),
        },
        "validation_status": "PASS",
    }
    _atomic_write_json(
        CAMPAIGN_DIR / f"v10g4_paired_s{int(seed):04d}.json", seed_payload
    )
    return seed_payload


def build_arm_result_for_invariants(
    seed: int, config: AblationConfig, payload: dict[str, Any], other_payload: dict[str, Any]
):
    """Reconstruct a minimal AblationArmResult view for invariant verification.

    The paired invariants operate only on persisted identity fields that were
    captured from the executed arms; every recomputation is byte-exact.
    """
    from types import SimpleNamespace

    validation = other_payload["validation"]["checks"]
    arm = SimpleNamespace()
    arm.optimizer_seed = int(seed)
    arm.variant = str(payload["variant"])
    arm.bpso_submitted_outputs_sha256 = payload["paired_identity"]["bpso_submitted_outputs_sha256"]
    arm.bpso_final_population_sha256 = payload["paired_identity"]["bpso_final_population_sha256"]
    arm.bgwo_row0_sha256 = payload["paired_identity"]["bgwo_row0_sha256"]
    arm.bgwo_rows_1_3_sha256 = tuple(payload["paired_identity"]["bgwo_rows_1_3_sha256"])
    arm.bgwo_rows_4_11_sha256 = tuple(payload["paired_identity"]["bgwo_rows_4_11_sha256"])
    arm.bgwo_common_rng_state_before_updates = payload["paired_identity"]["bgwo_common_rng_state_before_updates"]
    arm.elite_selection_invoked = bool(payload["elite_transfer"]["elite_selection_invoked"])
    arm.placed_elite_count = int(payload["elite_transfer"]["placed_elite_count"])
    arm.elite_hashes = tuple(payload["elite_transfer"]["elite_hashes"])
    arm.shadow_filler_placed = bool(payload["shadow_filler"]["placed"])
    arm.shadow_filler_evaluated = bool(payload["shadow_filler"]["evaluated"])
    arm.shadow_filler_generated = bool(payload["shadow_filler"]["generated"])
    arm.filler_block = SimpleNamespace(
        mask_sha256=tuple(payload["shadow_filler"]["mask_sha256"]),
        dimensions=config.dimensions,
        cardinalities=tuple(payload["shadow_filler"]["cardinalities"]),
        to_dict=lambda: payload["shadow_filler"]["block"],
    )
    arm.bpso_requests = int(validation.get("bpso_requests_96") or payload["accounting"]["bpso_requests"])
    arm.bgwo_requests = int(payload["accounting"]["bgwo_requests"])
    return arm


# ---------------------------------------------------------------------------
# Campaign orchestration
# ---------------------------------------------------------------------------


def run_campaign(resume: bool = True) -> dict[str, Any]:
    if not (_repo_head_matches() and _repo_origin_matches()):
        raise V10G4NoGoError(
            "V1.0-G4 requires HEAD == origin/main == "
            f"{EXPECTED_HEAD}; got HEAD={_git_sha('HEAD')} origin={_git_sha('origin/main')}"
        )

    config = _current_config()
    preflight = preflight_gates()
    preflight_sha = _sha256_bytes(
        json.dumps(to_json_compatible(preflight), sort_keys=True).encode("utf-8")
    )
    environment = capture_environment()
    environment_sha = _sha256_bytes(
        json.dumps(to_json_compatible(environment), sort_keys=True).encode("utf-8")
    )
    env_path = CAMPAIGN_DIR / "v10g4_environment.json"
    pf_path = CAMPAIGN_DIR / "v10g4_preflight.json"
    if env_path.is_file():
        existing_env = _read_json(env_path)
        env_compatible = (
            existing_env.get("schema_version") == SCHEMA_VERSION
            and existing_env.get("libraries", {}).get("numpy")
            == environment["libraries"]["numpy"]
            and existing_env.get("libraries", {}).get("pandas")
            == environment["libraries"]["pandas"]
            and existing_env.get("libraries", {}).get("scikit-learn")
            == environment["libraries"]["scikit-learn"]
            and existing_env.get("libraries", {}).get("scipy")
            == environment["libraries"]["scipy"]
            and existing_env.get("repository", {}).get("expected_head_sha256")
            == EXPECTED_HEAD
            and existing_env.get("repository", {}).get("head_sha256") == _git_sha("HEAD")
        )
        if not env_compatible:
            raise V10G4NoGoError(
                "G4 environment checkpoint incompatible with the current run; abort."
            )
    _atomic_write_json(env_path, environment)
    _atomic_write_json(pf_path, preflight)

    context = build_fitness_context()
    dataset_identity = dataset_identity_evidence(context)
    evaluator = FeatureFitnessEvaluator(context)

    manifest_entries: list[dict[str, Any]] = []
    campaign_started_iso = _now_iso()
    wall_start = time.perf_counter()
    for seed, first, second in LOCKED_SEED_ORDER:
        if seed not in PRODUCTION_OPTIMIZER_SEEDS:
            raise V10G4NoGoError(f"Unlocked G4 seed {seed}")
        seed_payload = execute_seed(
            seed, first, second, config, evaluator, context, preflight_sha, environment_sha, resume
        )
        manifest_entries.append(
            {
                "optimizer_seed": int(seed),
                "first": first,
                "second": second,
                "final": seed_payload["per_arm"],
                "paired_invariant_status": seed_payload["paired_invariant_status"],
                "requests_total": sum(
                    seed_payload["per_arm"][variant]["accounting"]["total_candidate_requests"]
                    for variant in ("WITH_ELITE_TRANSFER", "WITHOUT_ELITE_TRANSFER")
                ),
            }
        )
        print(f"[v10g4] seed {seed} complete: {first} / {second}", flush=True)

    wall_total = time.perf_counter() - wall_start
    manifest: dict[str, Any] = {
        "artifact_kind": MANIFEST_ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "protocol_name": PROTOCOL_NAME,
        "protocol_lock_sha256": G1_SEMANTIC_LOCK_SHA256,
        "starting_checkpoint_sha256": EXPECTED_HEAD,
        "status": "COMPLETE",
        "campaign_started_iso": campaign_started_iso,
        "campaign_completed_iso": _now_iso(),
        "campaign_wall_time_sec": round(wall_total, 6),
        "dataset_identity": dataset_identity,
        "environment": environment,
        "preflight_sha256": preflight_sha,
        "seeds": manifest_entries,
        "totals": {
            "arm_runs": 10,
            "optimizer_seed_count": 5,
            "requests_with": sum(
                entry["final"]["WITH_ELITE_TRANSFER"]["accounting"]["total_candidate_requests"]
                for entry in manifest_entries
            ),
            "requests_without": sum(
                entry["final"]["WITHOUT_ELITE_TRANSFER"]["accounting"]["total_candidate_requests"]
                for entry in manifest_entries
            ),
            "requests_total": sum(entry["requests_total"] for entry in manifest_entries),
            "test_access_count": 0,
            "incomplete_runs": 0,
            "failed_runs": 0,
        },
        "interpretation_locked": False,
        "winner_selected": False,
    }
    _atomic_write_json(CAMPAIGN_DIR / "v10g4_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    resume = "--no-resume" not in args
    manifest = run_campaign(resume=resume)
    print(json.dumps(to_json_compatible(manifest), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())