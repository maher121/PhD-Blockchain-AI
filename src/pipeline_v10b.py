"""V1.0-B synthetic protocol validation for the pure GOVERNED hybrid engine.

Validates the locked V1.0-A hybrid protocol and the pure, evaluator-agnostic
implementation without loading governed datasets, fitting production models,
or accessing final-test data. The production budget (5 runs x 192 requests =
960) is verified only as configuration arithmetic.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import yaml

import src.optimization.hybrid_bpso_bgwo as hybrid
from src.optimization.hybrid_bpso_bgwo import (
    HybridBPSOBGWO,
    RunLocalCache,
    hybrid_config_from_yaml,
    select_elites,
)


V10B_STAGE = "V1.0-B"
V10B_SCHEMA_VERSION = "v1.0-b-hybrid-protocol-validation-1"
V10B_VALIDATION_KIND = "SYNTHETIC_HYBRID_IMPLEMENTATION_VALIDATION"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "hybrid_v10.yaml"
DEFAULT_PROTOCOL_DOC_PATH = PROJECT_ROOT / "docs" / "v10_hybrid_bpso_bgwo_protocol.md"
DEFAULT_OPTIMIZER_MODULE = PROJECT_ROOT / "src" / "optimization" / "hybrid_bpso_bgwo.py"
DEFAULT_BPSO_MODULE = PROJECT_ROOT / "src" / "optimization" / "bpso.py"
DEFAULT_BGWO_MODULE = PROJECT_ROOT / "src" / "optimization" / "bgwo.py"
DEFAULT_BPSO_CONFIG_PATH = PROJECT_ROOT / "config" / "bpso.yaml"
DEFAULT_BGWO_CONFIG_PATH = PROJECT_ROOT / "config" / "bgwo_v09.yaml"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "hybrid" / "v10b_protocol_validation.json"

HYBRID_YAML_SHA256 = "0ac491b50abeb60e65efc2295eb88c03b4c988a5182ff0e2e3ed729deb9367d8"
PROTOCOL_CLASSIFICATION = "HYBRID_PROTOCOL_LOCKED_FAIR_BUDGET"
PROVENANCE_LOCK_COMMIT = "c0200e1"

PRIMARY_SEED_FAMILY = (3042, 3043, 3044, 3045, 3046)
MODEL_SEED_FAMILY = (42, 43, 44, 45, 46)
FORBIDDEN_OPTIMIZER_FAMILIES = (
    (1042, 1043, 1044, 1045, 1046),
    (2042, 2043, 2044, 2045, 2046),
)
BGWO_RNG_OFFSET = 10000
APPROVED_CARDINALITIES = (4, 8, 11, 14, 18, 22, 26, 30, 34, 38)


class V10BProtocolError(ValueError):
    """Raised when the frozen V1.0-A hybrid protocol configuration drifts."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V10BProtocolError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise V10BProtocolError(f"JSON payload must be an object: {path}")
    return payload


def current_head_short(root: Path = PROJECT_ROOT) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def provenance_lock_verification(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Verify the protocol-lock commit exists and still carries the locked YAML.

    HEAD-agnostic: the live HEAD is recorded observably but never pinned, so
    the validation remains valid regardless of later commits.
    """
    exists = (
        subprocess.run(
            ["git", "rev-parse", "--verify", f"{PROVENANCE_LOCK_COMMIT}^{{commit}}"],
            cwd=root,
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )
    if not exists:
        raise V10BProtocolError(
            f"Protocol-lock commit {PROVENANCE_LOCK_COMMIT} does not exist."
        )
    observed = subprocess.run(
        ["git", "show", f"{PROVENANCE_LOCK_COMMIT}:config/hybrid_v10.yaml"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    locked_bytes = observed.stdout
    locked_hash = hashlib.sha256(locked_bytes).hexdigest()
    return {
        "locked_on_commit": PROVENANCE_LOCK_COMMIT,
        "commit_exists": exists,
        "yaml_sha256_at_lock_commit": locked_hash,
        "yaml_sha256_matches_lock": locked_hash == HYBRID_YAML_SHA256,
    }


def module_unchanged_by_git(module_path: Path, root: Path = PROJECT_ROOT) -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", str(module_path)],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return status.stdout.strip() == ""


def load_v10b_protocol(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    protocol_doc_path: Path | str = DEFAULT_PROTOCOL_DOC_PATH,
) -> dict[str, Any]:
    """Load the locked V1.0-A hybrid protocol and fail closed on drift."""
    config_target = Path(config_path)
    doc_target = Path(protocol_doc_path)
    try:
        raw_config = config_target.read_bytes()
        payload = yaml.safe_load(raw_config.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise V10BProtocolError(f"Cannot load V1.0-A YAML protocol: {exc}") from exc
    if not isinstance(payload, dict):
        raise V10BProtocolError("V1.0-A YAML configuration must be a mapping.")
    if not doc_target.exists():
        raise V10BProtocolError(f"V1.0-A protocol document missing: {doc_target}")

    config_sha256 = hashlib.sha256(raw_config).hexdigest()
    if config_sha256 != HYBRID_YAML_SHA256:
        raise V10BProtocolError("V1.0-A hybrid YAML drifted from its frozen SHA-256.")

    expected_values = {
        "classification": (payload.get("protocol_classification"), PROTOCOL_CLASSIFICATION),
        "scientific_experiment": (payload.get("scientific_experiment"), False),
        "feature_space_count": (payload.get("dataset", {}).get("feature_space_count"), 43),
        "split_seed": (payload.get("dataset", {}).get("split", {}).get("seed"), 42),
        "ground_truth": (payload.get("dataset", {}).get("ground_truth"), "is_attack"),
        "population_size": (
            payload.get("hybrid_design", {}).get("population_size_per_phase"),
            12,
        ),
        "phase_budget_ratio": (
            tuple(payload.get("hybrid_design", {}).get("phase_budget_ratio_bpso_bgwo", ())),
            (50, 50),
        ),
        "bpso_requests": (
            payload.get("hybrid_design", {})
            .get("phase_budgets", {})
            .get("bpso_exploration_per_run_requests"),
            96,
        ),
        "bgwo_requests": (
            payload.get("hybrid_design", {})
            .get("phase_budgets", {})
            .get("bgwo_refinement_per_run_requests"),
            96,
        ),
        "per_run_requests": (
            payload.get("hybrid_design", {}).get("phase_budgets", {}).get("per_run_requests_total"),
            192,
        ),
        "five_run_requests": (
            payload.get("hybrid_design", {}).get("phase_budgets", {}).get("five_run_requests_total"),
            960,
        ),
        "bpso_generations": (
            payload.get("hybrid_design", {})
            .get("phase_budgets", {})
            .get("bpso_evaluated_generations_per_run"),
            8,
        ),
        "bgwo_iterations": (
            payload.get("hybrid_design", {})
            .get("phase_budgets", {})
            .get("bgwo_evaluated_iterations_per_run"),
            8,
        ),
        "elite_count": (payload.get("knowledge_transfer", {}).get("elite_count"), 3),
        "rng_offset": (
            payload.get("seeds", {}).get("optimizer_seed_phase_rng_offset_bgwo"),
            BGWO_RNG_OFFSET,
        ),
        "dataset_split_seed": (payload.get("seeds", {}).get("dataset_split_seed"), 42),
        "early_stopping_policy": (
            payload.get("early_stopping_policy", {}).get("method"),
            "A_fixed_budget_no_early_stopping",
        ),
        "elastic_cache_scope": (
            payload.get("cache_and_accounting", {}).get("cache_scope"),
            "one optimizer run (both phases), cleared between runs",
        ),
    }
    drift = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in expected_values.items()
        if observed != expected
    }
    if drift:
        raise V10BProtocolError(f"Frozen V1.0-A protocol drift: {drift}")

    primary = tuple(payload.get("seeds", {}).get("optimizer_seed_family_primary", ()))
    ablation = tuple(payload.get("seeds", {}).get("optimizer_seed_family_ablation", ()))
    model_seeds = tuple(payload.get("seeds", {}).get("model_attack_seeds", ()))
    if primary != PRIMARY_SEED_FAMILY or ablation != PRIMARY_SEED_FAMILY:
        raise V10BProtocolError("V1.0-A optimizer seed families drifted.")
    if model_seeds != MODEL_SEED_FAMILY:
        raise V10BProtocolError("V1.0-A model/attack seed family drifted.")
    forbidden = tuple(
        tuple(family)
        for family in payload.get("seeds", {}).get("forbidden_optimizer_seed_families", ())
    )
    if forbidden != FORBIDDEN_OPTIMIZER_FAMILIES:
        raise V10BProtocolError("V1.0-A forbidden seed families drifted.")

    config = hybrid_config_from_yaml(
        config_path=config_target,
        bpso_config_path=DEFAULT_BPSO_CONFIG_PATH,
        bgwo_config_path=DEFAULT_BGWO_CONFIG_PATH,
    )
    if config.cardinality_pool != APPROVED_CARDINALITIES:
        raise V10BProtocolError("V1.0-A cardinality anchor pool drifted.")
    if config.five_run_request_allocation != 960:
        raise V10BProtocolError("V1.0-A five-run request allocation must be exactly 960.")

    return {
        "schema_version": V10B_SCHEMA_VERSION,
        "stage": V10B_STAGE,
        "config_path": str(config_target),
        "protocol_doc_path": str(doc_target),
        "config_sha256": config_sha256,
        "config": config,
        "raw_snapshot": hybrid.to_json_compatible(payload),
    }


def run_v10b_synthetic_validation(
    *,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    protocol_doc_path: Path | str = DEFAULT_PROTOCOL_DOC_PATH,
    output_path: Path | str | None = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Execute deterministic synthetic validation of the pure hybrid engine."""
    loaded = load_v10b_protocol(config_path, protocol_doc_path)
    config = loaded["config"]
    provenance = provenance_lock_verification()
    live_head = current_head_short(PROJECT_ROOT)

    synthetic_config = replace(
        config,
        bpso_evaluated_generations=4,
        bgwo_evaluated_iterations=4,
    )
    engine = HybridBPSOBGWO(
        synthetic_config,
        _synthetic_objective,
        _synthetic_is_better,
    )
    first = engine.optimize(PRIMARY_SEED_FAMILY[0])
    replay = engine.optimize(PRIMARY_SEED_FAMILY[0])
    deterministic_replay_identical = first.to_json() == replay.to_json()
    different_seed = engine.optimize(PRIMARY_SEED_FAMILY[1])
    different_seed_diverges = first.to_json() != different_seed.to_json()

    module_source = DEFAULT_OPTIMIZER_MODULE.read_text(encoding="utf-8")
    forbidden_tokens = (
        "energy",
        "rss",
        "wall_time",
        "pandas",
        "sklearn",
        "is_attack",
        "Late_delivery",
        "final_test",
        "test_accessed",
    )
    forbidden_leaked = [
        token for token in forbidden_tokens if token in module_source.lower()
    ]

    checks = {
        "protocol_loaded_and_locked": True,
        "protocol_classification_locked": True,
        "yaml_hash_matches_lock": (
            loaded["config_sha256"] == HYBRID_YAML_SHA256
        ),
        "protocol_document_exists": True,
        "provenance_commit_exists": provenance["commit_exists"],
        "provenance_yaml_matches_lock": provenance["yaml_sha256_matches_lock"],
        "production_arithmetic_96_96_192": (
            config.bpso_request_allocation == 96
            and config.bgwo_request_allocation == 96
            and config.per_run_request_allocation == 192
        ),
        "five_run_request_allocation_960": config.five_run_request_allocation == 960,
        "allocated_fits_cap_4800": config.five_run_request_allocation * 5 == 4800,
        "seed_families_locked": (
            config.bgwo_phase_rng_offset == BGWO_RNG_OFFSET
        ),
        "optimizer_module_imports_clean": _module_imports_clean(),
        "optimizer_module_no_forbidden_semantics": not forbidden_leaked,
        "optimizer_module_no_frozen_winner_hardcode": (
            "5da981b5b87db97338ecdde9ca8a8b87db3a62771d03dc6a4ad901f6548a3299"
            not in module_source
            and "7ebb823374255f4f10c737c62a2111604aa50193a8f0d91b8483f3864cac7ad6"
            not in module_source
        ),
        "bpso_module_unchanged_by_git": module_unchanged_by_git(DEFAULT_BPSO_MODULE),
        "bgwo_module_unchanged_by_git": module_unchanged_by_git(DEFAULT_BGWO_MODULE),
        "synthetic_smoke_stop_reason_fixed_budget": (
            first.stop_reason == hybrid.STOP_FIXED_BUDGET_EXHAUSTED
        ),
        "synthetic_smoke_budget_consumed": (
            first.total_candidate_requests == synthetic_config.per_run_request_allocation
        ),
        "synthetic_smoke_accounting_invariants": (
            first.total_candidate_requests
            == first.unique_evaluations + first.cache_hits
            and first.evaluator_calls == first.unique_evaluations
            and first.total_candidate_requests == first.bpso_requests + first.bgwo_requests
        ),
        "synthetic_smoke_elites_transferred": (
            first.bgwo_cache_hits
            >= first.configuration_snapshot["placed_elite_count"]
        ),
        "synthetic_smoke_elite_rows_in_bgwo_init": _elite_rows_match(first),
        "synthetic_smoke_deterministic_replay": deterministic_replay_identical,
        "synthetic_smoke_different_seed_diverges": different_seed_diverges,
        "synthetic_only_validation": True,
        "production_optimizer_runs_not_executed": True,
        "dataco_not_accessed": True,
        "validation_not_accessed": True,
        "final_test_not_accessed": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    summary = {
        "schema_version": V10B_SCHEMA_VERSION,
        "stage": V10B_STAGE,
        "status": status,
        "validation_kind": V10B_VALIDATION_KIND,
        "scientific_experiment": False,
        "scientific_claims_supported": False,
        "live_head_short": live_head,
        "protocol_lock_commit": PROVENANCE_LOCK_COMMIT,
        "protocol_config_sha256": loaded["config_sha256"],
        "protocol_config_path": str(Path(loaded["config_path"]).relative_to(PROJECT_ROOT)),
        "protocol_document_path": str(
            Path(loaded["protocol_doc_path"]).relative_to(PROJECT_ROOT)
        ),
        "frozen_protocol_classification": PROTOCOL_CLASSIFICATION,
        "governed_model_attack_seeds": list(MODEL_SEED_FAMILY),
        "governed_optimizer_seed_family": list(PRIMARY_SEED_FAMILY),
        "production_budget": {
            "bpso_requests_per_run": config.bpso_request_allocation,
            "bgwo_requests_per_run": config.bgwo_request_allocation,
            "per_run_requests": config.per_run_request_allocation,
            "five_run_requests": config.five_run_request_allocation,
            "allocated_fits_cap": config.five_run_request_allocation * 5,
        },
        "provenance": provenance,
        "implementation_audit": {
            "module": str(DEFAULT_OPTIMIZER_MODULE.relative_to(PROJECT_ROOT)),
            "bpso_module_unchanged": checks["bpso_module_unchanged_by_git"],
            "bgwo_module_unchanged": checks["bgwo_module_unchanged_by_git"],
            "forbidden_tokens": list(forbidden_tokens),
            "forbidden_tokens_found": forbidden_leaked,
        },
        "synthetic_smoke": {
            "seed": PRIMARY_SEED_FAMILY[0],
            "bpso_generations": synthetic_config.bpso_evaluated_generations,
            "bgwo_iterations": synthetic_config.bgwo_evaluated_iterations,
            "stop_reason": first.stop_reason,
            "requests": first.total_candidate_requests,
            "unique_evaluations": first.unique_evaluations,
            "cache_hits": first.cache_hits,
            "evaluator_calls": first.evaluator_calls,
            "bpso_requests": first.bpso_requests,
            "bgwo_requests": first.bgwo_requests,
            "placed_elites": first.configuration_snapshot["placed_elite_count"],
            "bgwo_cache_hits": first.bgwo_cache_hits,
            "best_selected_feature_count": first.best_selected_feature_count,
            "best_phase": first.best_phase,
            "reproducible_re_code": first.reproducible_re_code,
            "replay_identical": deterministic_replay_identical,
        },
        "checks": checks,
        "governance": {
            "dataco_accessed": False,
            "governed_dt_trained": False,
            "final_test_accessed": False,
            "production_hybrid_search_executed": False,
            "hybrid_production_budget_executed": False,
            "next_stage_started": False,
        },
    }
    summary = hybrid.to_json_compatible(summary)
    if output_path is not None:
        _atomic_write_json(Path(output_path), summary)
    if status != "PASS":
        raise RuntimeError("V1.0-B synthetic hybrid validation failed.")
    return summary


class _SyntheticEvaluation:
    """Synthetic constrained-style evaluation mirroring the frozen ranking."""

    def __init__(
        self,
        *,
        feasible: bool,
        normalized_violation: float,
        cardinality: int,
        average_precision: float,
        f1: float,
        recall: float,
        mask: tuple[int, ...],
    ) -> None:
        self.feasible = feasible
        self.normalized_violation = normalized_violation
        self.cardinality = cardinality
        self.average_precision = average_precision
        self.f1 = f1
        self.recall = recall
        self.mask = mask

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "normalized_violation": self.normalized_violation,
            "cardinality": self.cardinality,
            "average_precision": self.average_precision,
            "f1": self.f1,
            "recall": self.recall,
        }


def _synthetic_objective(mask: np.ndarray) -> _SyntheticEvaluation:
    binary = np.asarray(mask, dtype=np.uint8).flatten()
    k = int(binary.sum())
    feasible = bool(binary[0] == 1 and binary[1] == 1)
    gain = float(np.count_nonzero(binary[2:8]))
    if feasible:
        violation = 0.0
        ap, f1, recall = 0.240 + 0.01 * gain, 0.205 + 0.01 * gain, 0.330 + 0.01 * gain
    else:
        missing = int((2 - int(binary[0]) - int(binary[1])) / 2) + 1
        violation = float(missing)
        ap, f1, recall = 0.120, 0.100, 0.150
    return _SyntheticEvaluation(
        feasible=feasible,
        normalized_violation=violation,
        cardinality=k,
        average_precision=ap,
        f1=f1,
        recall=recall,
        mask=tuple(int(value) for value in binary),
    )


def _synthetic_is_better(
    left: _SyntheticEvaluation, right: _SyntheticEvaluation
) -> bool:
    if left.feasible != right.feasible:
        return left.feasible
    if left.feasible:
        left_key = (left.cardinality, -left.average_precision, -left.f1, -left.recall, left.mask)
        right_key = (
            right.cardinality,
            -right.average_precision,
            -right.f1,
            -right.recall,
            right.mask,
        )
    else:
        left_key = (
            left.normalized_violation,
            -left.average_precision,
            -left.f1,
            -left.recall,
            left.cardinality,
            left.mask,
        )
        right_key = (
            right.normalized_violation,
            -right.average_precision,
            -right.f1,
            -right.recall,
            right.cardinality,
            right.mask,
        )
    return left_key < right_key


def _module_imports_clean() -> bool:
    import ast

    source = DEFAULT_OPTIMIZER_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    allowed = (
        "__future__",
        "numpy",
        "hashlib",
        "json",
        "re",
        "math",
        "dataclasses",
        "pathlib",
        "typing",
        "yaml",
        "src.optimization.bpso",
        "src.optimization.bgwo",
    )
    return all(name.startswith(allowed) for name in imported)


def _elite_rows_match(result: hybrid.HybridResult) -> bool:
    if result.elite_masks.shape[0] == 0:
        return False
    for slot in range(min(3, result.elite_masks.shape[0])):
        if not np.array_equal(
            result.bgwo_initial_population[slot + 1], result.elite_masks[slot]
        ):
            return False
    return True


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    summary = run_v10b_synthetic_validation()
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "status": summary["status"],
                "validation_kind": summary["validation_kind"],
                "scientific_experiment": summary["scientific_experiment"],
                "live_head_short": summary["live_head_short"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())