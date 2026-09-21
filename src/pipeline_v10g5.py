"""V1.0-G5: statistical and mechanistic analysis of the frozen V1.0-G4
elite-transfer production campaign.

READ-ONLY stage. G5 consumes ONLY the completed governed G4 artifacts under
``results/hybrid/v10g/v10g4_campaign/``. It does not run any optimizer, does
not access the TEST split, does not reselect, and does not create the G6 result
lock. Output is persisted deterministically under
``results/hybrid/v10g/v10g5_analysis/``.

Study design:
    - five paired optimizer seeds 3042..3046
    - comparison direction WITH_ELITE_TRANSFER - WITHOUT_ELITE_TRANSFER
    - primary inferential endpoints exactly: Average Precision and F1
    - primary-supporting endpoint: selected feature count K (descriptive only)
    - safeguard secondary: Recall
    - further secondary endpoints descriptive only

No p-values are computed anywhere. The paired 95% Student-t interval uses
n = 5, df = 4, t_critical = 2.7764451051977987.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import importlib
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.optimization.feature_fitness as feature_fitness
import src.pipeline_v10g4 as g4
import src.pipeline_v10g as v10g

PROTOCOL_NAME = "ELITE_TRANSFER_ABLATION_V10G"
STAGE = "V1.0-G5"
SCHEMA_VERSION = 1
EXPECTED_HEAD = g4.EXPECTED_HEAD
G1_SEMANTIC_LOCK_SHA256 = g4.G1_SEMANTIC_LOCK_SHA256
LOCKED_SEED_ORDER = g4.LOCKED_SEED_ORDER
PRODUCTION_OPTIMIZER_SEEDS = g4.PRODUCTION_OPTIMIZER_SEEDS
CAMPAIGN_DIR = g4.CAMPAIGN_DIR
G5_RESULTS_DIR = v10g.V10G_RESULTS_DIR / "v10g5_analysis"

WITH_VARIANT = "WITH_ELITE_TRANSFER"
WITHOUT_VARIANT = "WITHOUT_ELITE_TRANSFER"
VARIANTS = (WITH_VARIANT, WITHOUT_VARIANT)

N = 5
DF = 4
T_CRITICAL = 2.7764451051977987

ARM_ARTIFACT_KIND = g4.ARM_ARTIFACT_KIND
PAIRED_ARTIFACT_KIND = g4.PAIRED_ARTIFACT_KIND
MANIFEST_ARTIFACT_KIND = g4.MANIFEST_ARTIFACT_KIND

OUTPUT_FILES = (
    "v10g5_prevalidation.json",
    "v10g5_environment.json",
    "v10g5_paired_endpoint_table.json",
    "v10g5_paired_statistics.json",
    "v10g5_k_analysis.json",
    "v10g5_recall_safeguard.json",
    "v10g5_stability.json",
    "v10g5_feature_frequencies.json",
    "v10g5_mechanistic_analysis.json",
    "v10g5_computational_accounting.json",
    "v10g5_analysis_summary.json",
)


class V10G5NoGoError(RuntimeError):
    """Raised when a G5 analysis gate fails (fail-closed)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    raw = json.dumps(payload, sort_keys=True, indent=2, separators=(",", ": "), allow_nan=False)
    tmp.write_text(raw, encoding="utf-8")
    os.replace(tmp, path)
    return path


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _version(module_name: str) -> str:
    mod = importlib.import_module(module_name)
    if hasattr(mod, "__version__"):
        return str(mod.__version__)
    return "unknown"


def _feature_names() -> list[str]:
    manifest = _read_json(CAMPAIGN_DIR / "v10g4_manifest.json")
    names = manifest["dataset_identity"]["candidate_features"]
    if len(names) != 43:
        raise V10G5NoGoError(f"G5 found {len(names)} candidate features; expected 43.")
    return names


def _mask_selected(mask: list[int]) -> list[int]:
    return [i for i, bit in enumerate(mask) if int(bit) == 1]


def jaccard_indices(a: list[int], b: list[int]) -> float:
    set_a = set(a)
    set_b = set(b)
    union = set_a | set_b
    if not union:
        return 1.0
    return float(len(set_a & set_b) / len(union))


def paired_statistics(values: list[float]) -> dict[str, Any]:
    """Two-sided 95% Student-t interval on n = 5 paired differences (no p-values)."""
    if len(values) != N:
        raise V10G5NoGoError("Paired statistic must be computed on exactly 5 differences.")
    mean = float(sum(values) / N)
    sd = (
        math.sqrt(sum((v - mean) ** 2 for v in values) / (N - 1))
        if N > 1
        else 0.0
    )
    se = sd / math.sqrt(N)
    halfwidth = T_CRITICAL * se
    lower = mean - halfwidth
    upper = mean + halfwidth
    return {
        "n": N,
        "df": DF,
        "t_critical": T_CRITICAL,
        "paired_differences": [float(v) for v in values],
        "mean_difference": mean,
        "sample_sd": sd,
        "standard_error": se,
        "ci_lower_95": lower,
        "ci_upper_95": upper,
        "ci_excludes_zero": (lower > 0.0) or (upper < 0.0),
        "all_zero": all(float(v) == 0.0 for v in values),
    }


# ---------------------------------------------------------------------------
# Arm / pair loading
# ---------------------------------------------------------------------------


def load_arms() -> dict[tuple[int, str], dict[str, Any]]:
    arms: dict[tuple[int, str], dict[str, Any]] = {}
    for name in sorted(CAMPAIGN_DIR.glob("v10g4_arm_s*.json")):
        payload = _read_json(name)
        if payload.get("artifact_kind") != ARM_ARTIFACT_KIND:
            raise V10G5NoGoError(f"G5 unexpected arm artifact kind in {name.name}")
        arms[(int(payload["optimizer_seed"]), str(payload["variant"]))] = payload
    return arms


def load_pairs() -> dict[int, dict[str, Any]]:
    pairs: dict[int, dict[str, Any]] = {}
    for name in sorted(CAMPAIGN_DIR.glob("v10g4_paired_s*.json")):
        payload = _read_json(name)
        if payload.get("artifact_kind") != PAIRED_ARTIFACT_KIND:
            raise V10G5NoGoError(f"G5 unexpected paired artifact kind in {name.name}")
        pairs[int(payload["optimizer_seed"])] = payload
    return pairs


def endpoint_value(arm: dict[str, Any], endpoint: str) -> Any:
    final = arm["final"]
    if endpoint == "selected_feature_count":
        return int(final["selected_feature_count"])
    return final[endpoint]


# ---------------------------------------------------------------------------
# Pre-validation
# ---------------------------------------------------------------------------


def prevalidation_gates(arms: dict[tuple[int, str], dict[str, Any]],
                        pairs: dict[int, dict[str, Any]]) -> dict[str, Any]:
    manifest = _read_json(CAMPAIGN_DIR / "v10g4_manifest.json")
    checks: dict[str, Any] = {}

    checks["manifest_status_complete"] = manifest.get("status") == "COMPLETE"
    checks["exactly_10_arm_runs"] = len(arms) == 10
    expected_keys = {
        (s, v) for s, first, second in LOCKED_SEED_ORDER for v in (first, second)
    }
    checks["arm_identity_set_exact"] = set(arms.keys()) == expected_keys

    checks["exactly_5_paired_seeds"] = len(pairs) == 5
    checks["paired_seed_set_exact"] = set(pairs.keys()) == set(PRODUCTION_OPTIMIZER_SEEDS)

    invariant_statuses: dict[str, str] = {}
    for seed in sorted(pairs):
        pair = pairs[seed]
        invariant_statuses[str(seed)] = str(pair["paired_invariant_status"])
    checks["all_g4_paired_invariants_pass"] = all(s == "PASS" for s in invariant_statuses.values())

    totals = manifest.get("totals", {})
    checks["total_candidate_requests_1920"] = int(totals.get("requests_total", -1)) == 1920
    checks["requests_with_960"] = int(totals.get("requests_with", -1)) == 960
    checks["requests_without_960"] = int(totals.get("requests_without", -1)) == 960
    checks["test_access_count_zero"] = int(totals.get("test_access_count", -1)) == 0
    checks["no_incomplete_or_failed"] = (
        int(totals.get("incomplete_runs", -1)) == 0 and int(totals.get("failed_runs", -1)) == 0
    )

    for arm in arms.values():
        acc = arm["accounting"]
        if not (
            int(acc["total_candidate_requests"]) == 192
            and int(acc["bpso_requests"]) == 96
            and int(acc["bgwo_requests"]) == 96
            and int(acc["total_candidate_requests"])
            == int(acc["unique_evaluations"]) + int(acc["cache_hits"])
            and int(acc["evaluator_calls"]) == int(acc["unique_evaluations"])
            and int(acc["decision_tree_fits"]) == int(acc["unique_evaluations"]) * 5
        ):
            checks[f"accounting_{arm['optimizer_seed']}_{arm['variant']}"] = False
            continue
        checks[f"accounting_{arm['optimizer_seed']}_{arm['variant']}"] = True

    lock = v10g.recompute_protocol_lock()
    checks["g1_semantic_lock_unchanged"] = bool(
        lock["matches_stored"] and lock["matches_expected"]
        and lock["recomputed_semantic_sha256"] == G1_SEMANTIC_LOCK_SHA256
    )

    for arm in arms.values():
        if arm["test_access_audit"].get("test_accessed") is not False:
            checks[f"test_unaccessed_{arm['optimizer_seed']}_{arm['variant']}"] = False
            continue
        checks[f"test_unaccessed_{arm['optimizer_seed']}_{arm['variant']}"] = True

    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "artifact_kind": "V10G5_PREVALIDATION",
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "protocol_name": PROTOCOL_NAME,
        "g1_semantic_lock_sha256": G1_SEMANTIC_LOCK_SHA256,
        "input_campaign": "result",
        "campaign_manifest_sha256": _sha256_file(CAMPAIGN_DIR / "v10g4_manifest.json"),
        "status": status,
        "checks": checks,
        "paired_invariant_statuses": invariant_statuses,
    }


# ---------------------------------------------------------------------------
# Paired endpoint table + primary stats
# ---------------------------------------------------------------------------


def build_paired_endpoint_table(
    arms: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    feature_names = _feature_names()
    rows: list[dict[str, Any]] = []
    for seed, first, second in LOCKED_SEED_ORDER:
        with_arm = arms[(seed, WITH_VARIANT)]
        without_arm = arms[(seed, WITHOUT_VARIANT)]
        if {first, second} != {WITH_VARIANT, WITHOUT_VARIANT}:
            raise V10G5NoGoError(f"G5 locked order mismatch for seed {seed}.")
        row: dict[str, Any] = {
            "optimizer_seed": int(seed),
            "execution_order": [first, second],
        }
        for endpoint in (
            "average_precision",
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "selected_feature_count",
            "normalized_violation",
        ):
            row[f"{endpoint}_with"] = endpoint_value(with_arm, endpoint)
            row[f"{endpoint}_without"] = endpoint_value(without_arm, endpoint)
            if endpoint == "selected_feature_count":
                row[f"{endpoint}_difference"] = int(endpoint_value(with_arm, endpoint)) - int(
                    endpoint_value(without_arm, endpoint)
                )
            else:
                row[f"{endpoint}_difference"] = float(endpoint_value(with_arm, endpoint)) - float(
                    endpoint_value(without_arm, endpoint)
                )
        row["feasible_with"] = bool(with_arm["final"]["feasible"])
        row["feasible_without"] = bool(without_arm["final"]["feasible"])
        row["feasibility_transition"] = (
            "STAY_FEASIBLE" if row["feasible_with"] and row["feasible_without"] else "OTHER"
        )
        row["final_mask_with"] = with_arm["final"]["mask_sha256"]
        row["final_mask_without"] = without_arm["final"]["mask_sha256"]
        row["final_mask_identical"] = (
            row["final_mask_with"] == row["final_mask_without"]
        )
        row["selected_features_with"] = with_arm["final"]["selected_features"]
        row["selected_features_without"] = without_arm["final"]["selected_features"]
        row["selected_indices_with"] = _mask_selected(with_arm["final"]["mask"])
        row["selected_indices_without"] = _mask_selected(without_arm["final"]["mask"])
        row["cross_arm_jaccard"] = jaccard_indices(
            with_arm["final"]["mask"], without_arm["final"]["mask"]
        )
        row["feature_list_sha256_with"] = with_arm["final"]["feature_list_sha256"]
        row["feature_list_sha256_without"] = without_arm["final"]["feature_list_sha256"]
        rows.append(row)
    return {
        "artifact_kind": "V10G5_PAIRED_ENDPOINT_TABLE",
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "protocol_name": PROTOCOL_NAME,
        "comparison_direction": f"{WITH_VARIANT} - {WITHOUT_VARIANT}",
        "feature_names": feature_names,
        "rows": rows,
    }


def build_paired_statistics(table: dict[str, Any]) -> dict[str, Any]:
    primary = {
        "average_precision": {
            "label": "Average Precision (AP)",
            "values": [row["average_precision_difference"] for row in table["rows"]],
        },
        "f1": {
            "label": "F1",
            "values": [row["f1_difference"] for row in table["rows"]],
        },
    }
    statistics: dict[str, Any] = {
        "artifact_kind": "V10G5_PAIRED_STATISTICS",
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "protocol_name": PROTOCOL_NAME,
        "n": N,
        "df": DF,
        "t_critical": T_CRITICAL,
        "p_value_reported": False,
        "comparison_direction": f"{WITH_VARIANT} - {WITHOUT_VARIANT}",
        "endpoints": {},
    }
    for key, spec in primary.items():
        statistics["endpoints"][key] = {
            "label": spec["label"],
            "statistics": paired_statistics(spec["values"]),
        }
    return statistics


# ---------------------------------------------------------------------------
# K analysis (primary-supporting, descriptive only)
# ---------------------------------------------------------------------------


def build_k_analysis(table: dict[str, Any]) -> dict[str, Any]:
    differences = [int(row["selected_feature_count_difference"]) for row in table["rows"]]
    with_values = [int(row["selected_feature_count_with"]) for row in table["rows"]]
    without_values = [int(row["selected_feature_count_without"]) for row in table["rows"]]
    sorted_diffs = sorted(differences)
    median = (sorted_diffs[2]) if len(sorted_diffs) == 5 else None
    stats = {
        "n": N,
        "mean_difference": float(sum(differences) / N),
        "median_difference": float(median),
        "min_difference": float(min(differences)),
        "max_difference": float(max(differences)),
        "negative_count": sum(1 for d in differences if d < 0),
        "zero_count": sum(1 for d in differences if d == 0),
        "positive_count": sum(1 for d in differences if d > 0),
    }
    ci = paired_statistics([float(d) for d in differences])
    ci["descriptive_only"] = True
    ci["descriptive_rationale"] = (
        "K cannot independently establish inferential benefit."
    )
    k_with = {
        "values": with_values,
        "mean": float(sum(with_values) / N),
        "median": float(sorted(with_values)[2]),
        "min": int(min(with_values)),
        "max": int(max(with_values)),
    }
    k_without = {
        "values": without_values,
        "mean": float(sum(without_values) / N),
        "median": float(sorted(without_values)[2]),
        "min": int(min(without_values)),
        "max": int(max(without_values)),
    }
    return {
        "artifact_kind": "V10G5_K_ANALYSIS",
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "protocol_name": PROTOCOL_NAME,
        "comparison_direction": f"{WITH_VARIANT} - {WITHOUT_VARIANT}",
        "paired_differences": differences,
        "statistics": stats,
        "descriptive_95_ci": ci,
        "k_distribution_with": k_with,
        "k_distribution_without": k_without,
        "interpretation_role": (
            "K is a primary-SUPPORTING endpoint only; it cannot independently "
            "establish inferential elite-transfer benefit."
        ),
    }


# ---------------------------------------------------------------------------
# Recall safeguard
# ---------------------------------------------------------------------------


def build_recall_safeguard(table: dict[str, Any]) -> dict[str, Any]:
    differences = [float(row["recall_difference"]) for row in table["rows"]]
    stats = paired_statistics(differences)
    resolved_harm = any(d < 0.0 for d in differences)
    return {
        "artifact_kind": "V10G5_RECALL_SAFEGUARD",
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "protocol_name": PROTOCOL_NAME,
        "comparison_direction": f"{WITH_VARIANT} - {WITHOUT_VARIANT}",
        "paired_differences": differences,
        "statistics": stats,
        "resolved_recall_harm_detected": bool(resolved_harm),
        "recall_safeguard_status": "PASS" if not resolved_harm else "REVIEW",
    }


# ---------------------------------------------------------------------------
# Stability analysis
# ---------------------------------------------------------------------------


def build_stability(table: dict[str, Any]) -> dict[str, Any]:
    with_masks = [row["selected_indices_with"] for row in table["rows"]]
    without_masks = [row["selected_indices_without"] for row in table["rows"]]
    cross_arm = {
        str(row["optimizer_seed"]): row["cross_arm_jaccard"] for row in table["rows"]
    }

    def within_pairwise(jaccards: dict[tuple[int, int], float], masks: list[list[int]]) -> None:
        for i in range(5):
            for j in range(i + 1, 5):
                jaccards[(i, j)] = jaccard_indices(masks[i], masks[j])

    with_pairs: dict[tuple[int, int], float] = {}
    without_pairs: dict[tuple[int, int], float] = {}
    within_pairwise(with_pairs, with_masks)
    within_pairwise(without_pairs, without_masks)

    return {
        "artifact_kind": "V10G5_STABILITY_ANALYSIS",
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "protocol_name": PROTOCOL_NAME,
        "cross_arm_paired_jaccards": cross_arm,
        "within_with_pairwise_jaccards": {
            f"({i},{j})": with_pairs[(i, j)] for i in range(5) for j in range(i + 1, 5)
        },
        "within_without_pairwise_jaccards": {
            f"({i},{j})": without_pairs[(i, j)] for i in range(5) for j in range(i + 1, 5)
        },
        "k_distribution_with": sorted(len(m) for m in with_masks),
        "k_distribution_without": sorted(len(m) for m in without_masks),
        "note": "Cross-arm paired Jaccard = 1.0 means the paired arms reached identical masks.",
    }


# ---------------------------------------------------------------------------
# Feature-frequency / consensus
# ---------------------------------------------------------------------------


def _frequencies(masks: list[list[int]]) -> list[int]:
    counts = [0] * 43
    for mask in masks:
        for idx in mask:
            counts[idx] += 1
    return counts


def build_feature_frequencies(table: dict[str, Any], feature_names: list[str]) -> dict[str, Any]:
    with_masks = [row["selected_indices_with"] for row in table["rows"]]
    without_masks = [row["selected_indices_without"] for row in table["rows"]]
    freq_with = _frequencies(with_masks)
    freq_without = _frequencies(without_masks)
    freq_diff = [w - o for w, o in zip(freq_with, freq_without)]
    consensus = [i for i, c in enumerate(list(freq_with)) if c >= 3]
    strict_consensus = [i for i, c in enumerate(list(freq_with)) if c == 5]
    return {
        "artifact_kind": "V10G5_FEATURE_FREQUENCIES",
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "protocol_name": PROTOCOL_NAME,
        "feature_names": feature_names,
        "frequency_with_0_to_5": freq_with,
        "frequency_without_0_to_5": freq_without,
        "frequency_with_minus_without": freq_diff,
        "consensus_features_ge3_of_5": {
            "indices": consensus,
            "names": [feature_names[i] for i in consensus],
        },
        "strict_consensus_features_5_of_5": {
            "indices": strict_consensus,
            "names": [feature_names[i] for i in strict_consensus],
        },
    }


# ---------------------------------------------------------------------------
# Mechanistic analysis
# ---------------------------------------------------------------------------


def build_mechanistic(
    arms: dict[tuple[int, str], dict[str, Any]],
    pairs: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for seed, _, _ in LOCKED_SEED_ORDER:
        with_arm = arms[(seed, WITH_VARIANT)]
        without_arm = arms[(seed, WITHOUT_VARIANT)]
        acc_w = with_arm["accounting"]
        acc_o = without_arm["accounting"]
        pi_w = with_arm["paired_identity"]
        pi_o = without_arm["paired_identity"]
        pair = pairs[seed]
        invariants = pair["paired_invariants"]["checks"]
        per_seed.append(
            {
                "optimizer_seed": int(seed),
                "rows_1_3_with_are_elites": bool(with_arm["elite_transfer"]["elite_selection_invoked"])
                and bool(with_arm["elite_transfer"]["placed_elite_count"] == 3),
                "rows_1_3_without_are_filler": list(pi_o["bgwo_rows_1_3_sha256"])
                == list(with_arm["shadow_filler"]["mask_sha256"]),
                "elite_hashes_equal_with_rows_1_3": list(pi_w["bgwo_rows_1_3_sha256"])
                == list(with_arm["elite_transfer"]["elite_hashes"]),
                "paired_bpso_outputs_identical": pi_w["bpso_submitted_outputs_sha256"]
                == pi_o["bpso_submitted_outputs_sha256"],
                "row0_paired_identical": pi_w["bgwo_row0_sha256"] == pi_o["bgwo_row0_sha256"],
                "rows_4_11_paired_identical": list(pi_w["bgwo_rows_4_11_sha256"])
                == list(pi_o["bgwo_rows_4_11_sha256"]),
                "final_mask_paired_identical": with_arm["final"]["mask_sha256"]
                == without_arm["final"]["mask_sha256"],
                "final_metrics_paired_identical": all(
                    endpoint_value(with_arm, e) == endpoint_value(without_arm, e)
                    for e in ("average_precision", "f1", "precision", "recall", "roc_auc")
                ),
                "cache_hits_with_minus_without": int(acc_w["cache_hits"]) - int(acc_o["cache_hits"]),
                "unique_with_minus_without": int(acc_w["unique_evaluations"]) - int(acc_o["unique_evaluations"]),
                "bgwo_new_unique_with_minus_without": int(acc_w["bgwo_new_unique_evaluations"])
                - int(acc_o["bgwo_new_unique_evaluations"]),
                "bgwo_cache_hits_with_minus_without": int(acc_w["bgwo_cache_hits"])
                - int(acc_o["bgwo_cache_hits"]),
                "convergence_history_differ": with_arm["convergence"] != without_arm["convergence"],
                "convergence_length": len(with_arm["convergence"]),
                "g4_invariant_row_collisions": invariants.get("treatment_rows_collision_reported_not_rejected"),
            }
        )
    totals_w = {k: 0 for k in ("cache_hits", "unique_evaluations", "bgwo_new_unique_evaluations", "decision_tree_fits")}
    totals_o = {k: 0 for k in ("cache_hits", "unique_evaluations", "bgwo_new_unique_evaluations", "decision_tree_fits")}
    for seed, _, _ in LOCKED_SEED_ORDER:
        for key in totals_w:
            totals_w[key] += int(arms[(seed, WITH_VARIANT)]["accounting"][key])
            totals_o[key] += int(arms[(seed, WITHOUT_VARIANT)]["accounting"][key])
    return {
        "artifact_kind": "V10G5_MECHANISTIC_ANALYSIS",
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "protocol_name": PROTOCOL_NAME,
        "per_seed": per_seed,
        "totals_with": totals_w,
        "totals_without": totals_o,
        "totals_with_minus_without": {k: totals_w[k] - totals_o[k] for k in totals_w},
    }


# ---------------------------------------------------------------------------
# Computational accounting
# ---------------------------------------------------------------------------


def build_computational_accounting(
    arms: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed, _, _ in LOCKED_SEED_ORDER:
        with_arm = arms[(seed, WITH_VARIANT)]
        without_arm = arms[(seed, WITHOUT_VARIANT)]
        for variant, arm in ((WITH_VARIANT, with_arm), (WITHOUT_VARIANT, without_arm)):
            acc = arm["accounting"]
            rows.append(
                {
                    "optimizer_seed": int(seed),
                    "variant": variant,
                    "candidate_requests": int(acc["total_candidate_requests"]),
                    "unique_evaluations": int(acc["unique_evaluations"]),
                    "evaluator_calls": int(acc["evaluator_calls"]),
                    "cache_hits": int(acc["cache_hits"]),
                    "bpso_unique_evaluations": int(acc["bpso_unique_evaluations"]),
                    "bgwo_new_unique_evaluations": int(acc["bgwo_new_unique_evaluations"]),
                    "bpso_cache_hits": int(acc["bpso_cache_hits"]),
                    "bgwo_cache_hits": int(acc["bgwo_cache_hits"]),
                    "decision_tree_fits": int(acc["decision_tree_fits"]),
                    "optimizer_wall_time_sec": float(arm["optimizer_wall_time_sec"]),
                    "optimizer_cpu_time_sec": float(arm["optimizer_cpu_time_sec"]),
                    "final_phase": str(arm["final_phase"]),
                    "final_phase_index": int(arm["final_phase_index"]),
                }
            )
    return {
        "artifact_kind": "V10G5_COMPUTATIONAL_ACCOUNTING",
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "protocol_name": PROTOCOL_NAME,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Environment + summary
# ---------------------------------------------------------------------------


def capture_environment() -> dict[str, Any]:
    return {
        "artifact_kind": "V10G5_ENVIRONMENT",
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "protocol_name": PROTOCOL_NAME,
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
            "scikit-learn": _version("sklearn"),
            "scipy": _version("scipy"),
        },
        "repository": {
            "head_sha256": _git_sha("HEAD"),
            "expected_head_sha256": EXPECTED_HEAD,
        },
    }


def _git_sha(ref: str) -> str | None:
    try:
        output = os.popen(f"git rev-parse {ref}").read().strip()
        return output or None
    except Exception:
        return None


def build_summary(
    table: dict[str, Any],
    statistics: dict[str, Any],
    k_analysis: dict[str, Any],
    recall_safeguard: dict[str, Any],
    stability: dict[str, Any],
    frequencies: dict[str, Any],
    mechanistic: dict[str, Any],
    accounting: dict[str, Any],
    prevalidation: dict[str, Any],
) -> dict[str, Any]:
    ap = statistics["endpoints"]["average_precision"]["statistics"]
    f1 = statistics["endpoints"]["f1"]["statistics"]

    def phase_direction(stat: dict[str, Any]) -> str:
        if stat["all_zero"]:
            return "identical_zero_difference"
        return "uncertain" if not stat["ci_excludes_zero"] else ("favors_with" if stat["mean_difference"] > 0 else "favors_without")

    ap_direction = phase_direction(ap)
    f1_direction = phase_direction(f1)
    feasibility_preserved = all(
        row["feasible_with"] and row["feasible_without"] for row in table["rows"]
    )
    recall_no_harm = not recall_safeguard["resolved_recall_harm_detected"]
    endpoint_support = {
        "average_precision": bool(
            ap_direction == "favors_with" and ap["ci_excludes_zero"]
        ),
        "f1": bool(f1_direction == "favors_with" and f1["ci_excludes_zero"]),
    }
    effective_equivalence = (
        ap["mean_difference"] == 0.0
        and f1["mean_difference"] == 0.0
        and all(row["cross_arm_jaccard"] == 1.0 for row in table["rows"])
    )
    interpretation = {
        "comparison_direction": f"{WITH_VARIANT} - {WITHOUT_VARIANT}",
        "average_precision": {
            "direction": ap_direction,
            "mean_difference": ap["mean_difference"],
            "ci_95": [ap["ci_lower_95"], ap["ci_upper_95"]],
            "endpoint_specific_support": endpoint_support["average_precision"],
        },
        "f1": {
            "direction": f1_direction,
            "mean_difference": f1["mean_difference"],
            "ci_95": [f1["ci_lower_95"], f1["ci_upper_95"]],
            "endpoint_specific_support": endpoint_support["f1"],
        },
        "feasibility_preserved": bool(feasibility_preserved),
        "recall_no_resolved_harm": bool(recall_no_harm),
    }
    if effective_equivalence:
        interpretation["conclusion"] = (
            "This experiment does not demonstrate a resolved elite-transfer contribution."
        )
    elif not (ap["ci_excludes_zero"] or f1["ci_excludes_zero"]):
        interpretation["conclusion"] = "Direction uncertain."
    elif endpoint_support["average_precision"] or endpoint_support["f1"]:
        interpretation["conclusion"] = (
            "Endpoint-specific support for elite transfer is present (AP or F1 favors WITH "
            "with 95% CI excluding zero, feasibility preserved, no resolved recall harm)."
        )
    else:
        interpretation["conclusion"] = (
            "Direction uncertain."
        )
    return {
        "artifact_kind": "V10G5_ANALYSIS_SUMMARY",
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "protocol_name": PROTOCOL_NAME,
        "input_campaign_status": "COMPLETE",
        "prevalidation": {
            "status": prevalidation["status"],
            "checks": prevalidation["checks"],
        },
        "endpoints_paired_differences_ap": ap["paired_differences"],
        "endpoints_paired_differences_f1": f1["paired_differences"],
        "endpoint_statistics": {
            "ap": ap,
            "f1": f1,
        },
        "k_analysis": {
            "statistics": k_analysis["statistics"],
            "descriptive_95_ci": k_analysis["descriptive_95_ci"],
        },
        "recall_safeguard": {
            "status": recall_safeguard["recall_safeguard_status"],
            "statistics": recall_safeguard["statistics"],
        },
        "stability": {
            "cross_arm_paired_jaccards": stability["cross_arm_paired_jaccards"],
            "consensus_features_ge3_of_5": frequencies["consensus_features_ge3_of_5"]["names"],
            "strict_consensus_features_5_of_5": frequencies["strict_consensus_features_5_of_5"]["names"],
            "frequency_with_minus_without": frequencies["frequency_with_minus_without"],
        },
        "mechanistic": {
            "totals_with_minus_without": mechanistic["totals_with_minus_without"],
            "all_final_masks_paired_identical": all(
                row["final_mask_paired_identical"] for row in mechanistic["per_seed"]
            ),
            "all_convergence_histories_differ": all(
                row["convergence_history_differ"] for row in mechanistic["per_seed"]
            ),
        },
        "interpretation": interpretation,
        "p_value_reported": False,
        "overall_winner": "none",
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_analysis(verify_only: bool = False) -> dict[str, str]:
    if not (_git_sha("HEAD") == EXPECTED_HEAD):
        raise V10G5NoGoError(
            f"V1.0-G5 requires HEAD == {EXPECTED_HEAD}; got {_git_sha('HEAD')}"
        )
    arms = load_arms()
    pairs = load_pairs()
    prevalidation = prevalidation_gates(arms, pairs)
    if prevalidation["status"] != "PASS":
        raise V10G5NoGoError("V1.0-G5 pre-validation gate failed; abort.")
    table = build_paired_endpoint_table(arms)
    statistics = build_paired_statistics(table)
    k_analysis = build_k_analysis(table)
    recall_safeguard = build_recall_safeguard(table)
    stability = build_stability(table)
    frequencies = build_feature_frequencies(table, table["feature_names"])
    mechanistic = build_mechanistic(arms, pairs)
    accounting = build_computational_accounting(arms)
    environment = capture_environment()
    summary = build_summary(
        table, statistics, k_analysis, recall_safeguard, stability,
        frequencies, mechanistic, accounting, prevalidation,
    )
    outputs = {
        "v10g5_prevalidation.json": prevalidation,
        "v10g5_environment.json": environment,
        "v10g5_paired_endpoint_table.json": table,
        "v10g5_paired_statistics.json": statistics,
        "v10g5_k_analysis.json": k_analysis,
        "v10g5_recall_safeguard.json": recall_safeguard,
        "v10g5_stability.json": stability,
        "v10g5_feature_frequencies.json": frequencies,
        "v10g5_mechanistic_analysis.json": mechanistic,
        "v10g5_computational_accounting.json": accounting,
        "v10g5_analysis_summary.json": summary,
    }
    written: dict[str, str] = {}
    for name, payload in outputs.items():
        if name not in OUTPUT_FILES:
            raise V10G5NoGoError(f"Unexpected output file {name}")
        path = _atomic_write_json(G5_RESULTS_DIR / name, payload)
        written[name] = _sha256_file(path)
    return written


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    verify_only = "--verify-only" in args
    written = run_analysis(verify_only=verify_only)
    print(f"[v10g5] analysis persisted {len(written)} artifacts under {G5_RESULTS_DIR}")
    for name, sha in sorted(written.items()):
        print(f"  {name}  {sha}")
    if not verify_only:
        second = run_analysis(verify_only=True)
        if second == written:
            print("[v10g5] deterministic outputs verified (byte-identical rerun)")
        else:
            raise V10G5NoGoError("V1.0-G5 outputs are NOT deterministic across runs.")
    print("[v10g5] V10G5 analysis complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())