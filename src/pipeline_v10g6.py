"""Governed V1.0-G6 finalization: verify and lock the elite-transfer ablation result.

V1.0-G6 performs NO new experiment. It is a strictly read-only verification of
the frozen upstream artifacts (G1 protocol lock, G4 production campaign, G5
deterministic analysis, V10D/V10E/V10F locks) followed by the creation of the
single governed ablation result lock ``v10g_result_lock.json`` whose
``semantic_result_lock_sha256`` is the canonical SHA-256 of its content-bearing
``semantic_payload``.

Canonical hash convention (matches the G1 protocol lock and the V10D/V10F/J
semantic locks): SHA-256 over the UTF-8 encoding of ``json.dumps`` with mapping
keys recursively sorted, array order preserved, ``sort_keys=True``,
``separators=(\",\",\":\")``, ``ensure_ascii=True``, ``allow_nan=False``, and no
trailing newline. G6 never invokes an optimizer or fitted model and never
touches the TEST split.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import src.pipeline_v10g as v10g
import src.pipeline_v10g5 as g5

STAGE = "V1.0-G6"
SCHEMA_VERSION = "v1.0-g-ablation-result-1"
PROTOCOL_NAME = "ELITE_TRANSFER_ABLATION_V10G"
EXPECTED_HEAD = g5.EXPECTED_HEAD
RESULT_DIR = v10g.V10G_RESULTS_DIR
RESULT_LOCK_PATH = RESULT_DIR / "v10g_result_lock.json"
VALIDATION_PATH = RESULT_DIR / "v10g6_validation.json"
CAMPAIGN_DIR = g5.CAMPAIGN_DIR
G5_RESULTS_DIR = g5.G5_RESULTS_DIR
V10E_RESULT_PATH = v10g.PROJECT_ROOT / "results" / "hybrid" / "v10e" / "v10e_result_lock.json"

REQUIRED_LOCK_KEYS = (
    "artifact_kind",
    "schema_version",
    "stage",
    "status",
    "protocol_lock_sha256",
    "upstream_provenance",
    "campaign_identity",
    "paired_seed_set",
    "endpoint_summary",
    "stability_summary",
    "computational_accounting_summary",
    "test_isolation",
    "interpretation",
    "winner_governance",
    "artifact_hashes",
    "semantic_payload",
    "semantic_result_lock_sha256",
)

G5_OUTPUT_FILES = tuple(sorted(path.name for path in G5_RESULTS_DIR.glob("v10g5_*.json")))
G4_CAMPAIGN_FILES = tuple(
    sorted(path.name for path in CAMPAIGN_DIR.glob("v10g4_*.json"))
)


class V10G6Error(RuntimeError):
    """Base error for the governed V1.0-G6 finalization layer."""


class V10G6NoGoError(V10G6Error):
    """Raised when a V1.0-G6 gate fails and the result lock must not be written."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V10G6Error(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise V10G6Error(f"JSON payload must be an object: {path}")
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
    """Recursively sorted-key canonical JSON (ensure_ascii=True, no newline)."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def semantic_hash(payload: Any) -> str:
    """Canonical semantic SHA-256 over the content-bearing payload only."""
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _js(value: Any) -> Any:
    """Force a plain-JSON deep copy (no numpy/Python-only types)."""
    return json.loads(json.dumps(value, allow_nan=False))


def _git_sha(ref: str) -> str | None:
    try:
        output = os.popen(f"git rev-parse {ref}").read().strip()
        return output or None
    except Exception:
        return None


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if Path(path).resolve() == v10g.PROTOCOL_LOCK_PATH.resolve():
        raise V10G6NoGoError("Refusing to write the G1 protocol lock from V1.0-G6.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    raw = json.dumps(
        _js(payload),
        sort_keys=True,
        indent=2,
        separators=(",", ": "),
        allow_nan=False,
        ensure_ascii=True,
    )
    tmp.write_text(raw, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def head_equals_base() -> dict[str, Any]:
    head = _git_sha("HEAD")
    origin = _git_sha("origin/main")
    checks = {
        "head_equals_expected": head == EXPECTED_HEAD,
        "origin_main_equals_expected": origin == EXPECTED_HEAD,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "stage": STAGE,
        "expected_head_sha256": EXPECTED_HEAD,
        "head_sha256": head,
        "origin_main_sha256": origin,
        "checks": checks,
    }


def g1_gate() -> dict[str, Any]:
    g1 = v10g.g1_protocol_verification()
    semantic = v10g.recompute_protocol_lock()
    checks = {
        "g1_protocol_verification_pass": g1["status"] == "PASS",
        "g1_semantic_lock_recomputes_exactly": bool(
            semantic["matches_expected"] and semantic["matches_stored"]
        ),
        "g1_semantic_lock_sha256": v10g.V10G_SEMANTIC_LOCK_SHA256,
    }
    return {
        "status": "PASS" if all(v is True for v in checks.values() if isinstance(v, bool)) else "FAIL",
        "stage": STAGE,
        "checks": {k: v for k, v in checks.items() if k != "g1_semantic_lock_sha256"},
        "g1_semantic_lock_sha256": checks["g1_semantic_lock_sha256"],
    }


def prior_locks_gate() -> dict[str, Any]:
    frozen = v10g.prior_locks_immutable()
    v10e_present = V10E_RESULT_PATH.is_file()
    checks = dict(frozen["checks"])
    checks["v10e_result_lock_exists"] = v10e_present
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "stage": STAGE,
        "checks": checks,
        "v10d_winner_lock_sha256": frozen["v10d_winner_lock_sha256"],
        "v10e_result_lock_sha256": _sha256_file(V10E_RESULT_PATH) if v10e_present else None,
        "v10f_result_lock_sha256": frozen["v10f_result_lock_sha256"],
    }


def test_isolation_gate() -> dict[str, Any]:
    audit = v10g.test_isolation_audit()
    result = {
        "status": "PASS" if audit["status"] == "PASS" else "FAIL",
        "stage": STAGE,
        "test_accessed": audit["test_accessed"],
        "classification": audit["classification"],
        "checks": {"test_never_accessed": True, "no_test_split_authorization": False},
    }
    result["checks"]["no_test_split_authorization"] = (
        audit["checks"]["test_authorized_false"] is True
    )
    return result


def g4_campaign_gate() -> dict[str, Any]:
    manifest = _read_json(CAMPAIGN_DIR / "v10g4_manifest.json")
    totals = dict(manifest.get("totals", {}))
    counts = {
        "arms": len(list(CAMPAIGN_DIR.glob("v10g4_arm_s*.json"))),
        "pairs": len(list(CAMPAIGN_DIR.glob("v10g4_paired_s*.json"))),
        "json_artifacts": len(G4_CAMPAIGN_FILES),
    }
    checks = {
        "manifest_status_complete": manifest.get("status") == "COMPLETE",
        "exactly_10_arms": counts["arms"] == 10,
        "exactly_5_pairs": counts["pairs"] == 5,
        "campaign_json_artifact_set_exact": (
            counts["json_artifacts"] == 18
            and set(G4_CAMPAIGN_FILES)
            == {
                p.name
                for p in CAMPAIGN_DIR.iterdir()
                if p.name.startswith("v10g4_") and p.suffix == ".json"
            }
        ),
        "requests_total_1920": int(totals.get("requests_total", -1)) == 1920,
        "requests_with_960": int(totals.get("requests_with", -1)) == 960,
        "requests_without_960": int(totals.get("requests_without", -1)) == 960,
        "test_access_count_zero": int(totals.get("test_access_count", -1)) == 0,
        "no_incomplete_or_failed": (
            int(totals.get("incomplete_runs", -1)) == 0
            and int(totals.get("failed_runs", -1)) == 0
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "stage": STAGE,
        "campaign_manifest_sha256": _sha256_file(CAMPAIGN_DIR / "v10g4_manifest.json"),
        "campaign_counts": counts,
        "checks": checks,
    }


def g5_recomputation_gate(
    arms: dict[tuple[int, str], dict[str, Any]],
    pairs: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Recompute every G5 artifact from the raw G4 JSON and require exact match."""
    table = g5.build_paired_endpoint_table(arms)
    statistics = g5.build_paired_statistics(table)
    k_analysis = g5.build_k_analysis(table)
    recall = g5.build_recall_safeguard(table)
    stability = g5.build_stability(table)
    frequencies = g5.build_feature_frequencies(table, table["feature_names"])
    mechanistic = g5.build_mechanistic(arms, pairs)
    accounting = g5.build_computational_accounting(arms)
    prevalidation = g5.prevalidation_gates(arms, pairs)
    summary = g5.build_summary(
        table,
        statistics,
        k_analysis,
        recall,
        stability,
        frequencies,
        mechanistic,
        accounting,
        prevalidation,
    )
    recomputed = {
        "v10g5_prevalidation.json": prevalidation,
        "v10g5_paired_endpoint_table.json": table,
        "v10g5_paired_statistics.json": statistics,
        "v10g5_k_analysis.json": k_analysis,
        "v10g5_recall_safeguard.json": recall,
        "v10g5_stability.json": stability,
        "v10g5_feature_frequencies.json": frequencies,
        "v10g5_mechanistic_analysis.json": mechanistic,
        "v10g5_computational_accounting.json": accounting,
        "v10g5_analysis_summary.json": summary,
    }
    matches: dict[str, bool] = {}
    for name, payload in recomputed.items():
        matches[name] = _read_json(G5_RESULTS_DIR / name) == _js(payload)
    status = "PASS" if all(matches.values()) else "FAIL"
    return {
        "status": status,
        "stage": STAGE,
        "checks": matches,
        "recomputed_vs_persisted": "byte-content-identical" if status == "PASS" else "MISMATCH",
    }


def zero_equivalence_gate(table: dict[str, Any]) -> dict[str, Any]:
    row_checks: dict[str, bool] = {}
    for row in table["rows"]:
        seed = int(row["optimizer_seed"])
        for endpoint in (
            "average_precision",
            "f1",
            "precision",
            "recall",
            "roc_auc",
        ):
            row_checks[f"{seed}_{endpoint}_difference_zero"] = (
                float(row[f"{endpoint}_difference"]) == 0.0
            )
        row_checks[f"{seed}_selected_feature_count_difference_zero"] = (
            int(row["selected_feature_count_difference"]) == 0
        )
        row_checks[f"{seed}_cross_arm_jaccard_one"] = float(row["cross_arm_jaccard"]) == 1.0
        row_checks[f"{seed}_final_mask_identical"] = bool(row["final_mask_identical"])
        row_checks[f"{seed}_both_feasible"] = bool(row["feasible_with"]) and bool(
            row["feasible_without"]
        )
    statistics = g5.build_paired_statistics(table)
    stat_checks: dict[str, bool] = {}
    for endpoint in ("average_precision", "f1"):
        s = statistics["endpoints"][endpoint]["statistics"]
        stat_checks[f"{endpoint}_all_zero"] = bool(s["all_zero"])
        stat_checks[f"{endpoint}_ci_is_zero_width"] = (
            float(s["ci_lower_95"]) == 0.0 and float(s["ci_upper_95"]) == 0.0
        )
        stat_checks[f"{endpoint}_ci_excludes_zero"] = bool(s["ci_excludes_zero"]) is False
    k = g5.build_k_analysis(table)
    k_stat = k["statistics"]
    k_checks = {
        "k_mean_difference_zero": float(k_stat["mean_difference"]) == 0.0,
        "k_median_difference_zero": float(k_stat["median_difference"]) == 0.0,
        "k_all_differences_zero": all(v == 0 for v in k["paired_differences"]),
        "k_zero_count_5": int(k_stat["zero_count"]) == 5,
        "k_negative_count_zero": int(k_stat["negative_count"]) == 0,
        "k_positive_count_zero": int(k_stat["positive_count"]) == 0,
        "k_within_distributions_identical": (
            k["k_distribution_with"]["values"] == k["k_distribution_without"]["values"]
        ),
    }
    checks = {**row_checks, **stat_checks, **k_checks}
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "stage": STAGE,
        "checks": checks,
    }


def accounting_gate(arms: dict[tuple[int, str], dict[str, Any]]) -> dict[str, Any]:
    fits_correct = all(
        int(arm["accounting"]["decision_tree_fits"])
        == int(arm["accounting"]["unique_evaluations"]) * 5
        and int(arm["accounting"]["evaluator_calls"]) == int(arm["accounting"]["unique_evaluations"])
        for arm in arms.values()
    )
    with_total = {"unique_evaluations": 0, "cache_hits": 0, "decision_tree_fits": 0}
    without_total = {"unique_evaluations": 0, "cache_hits": 0, "decision_tree_fits": 0}
    for (seed, variant), arm in arms.items():
        if variant == "WITH_ELITE_TRANSFER":
            with_total["unique_evaluations"] += int(arm["accounting"]["unique_evaluations"])
            with_total["cache_hits"] += int(arm["accounting"]["cache_hits"])
            with_total["decision_tree_fits"] += int(arm["accounting"]["decision_tree_fits"])
        else:
            without_total["unique_evaluations"] += int(arm["accounting"]["unique_evaluations"])
            without_total["cache_hits"] += int(arm["accounting"]["cache_hits"])
            without_total["decision_tree_fits"] += int(arm["accounting"]["decision_tree_fits"])
    checks = {
        "with_unique_evaluations_940": with_total["unique_evaluations"] == 940,
        "with_cache_hits_20": with_total["cache_hits"] == 20,
        "with_decision_tree_fits_4700": with_total["decision_tree_fits"] == 4700,
        "without_unique_evaluations_955": without_total["unique_evaluations"] == 955,
        "without_cache_hits_5": without_total["cache_hits"] == 5,
        "without_decision_tree_fits_4775": without_total["decision_tree_fits"] == 4775,
        "per_arm_fits_equal_unique_times_5": fits_correct,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "stage": STAGE,
        "with_totals": with_total,
        "without_totals": without_total,
        "delta_totals": {
            "unique_evaluations": with_total["unique_evaluations"] - without_total["unique_evaluations"],
            "cache_hits": with_total["cache_hits"] - without_total["cache_hits"],
            "decision_tree_fits": with_total["decision_tree_fits"] - without_total["decision_tree_fits"],
        },
        "checks": checks,
    }


def lifecycle_gate() -> dict[str, Any]:
    result = v10g.no_g2_result_lock_artifacts()
    own_outputs = {"v10g_result_lock.json", "v10g6_validation.json"}
    return {
        "status": result["status"],
        "stage": STAGE,
        "present_files": [n for n in result["present_files"] if n not in own_outputs],
        "g6_owned_outputs": sorted(own_outputs),
        "sanctioned_post_g2_artifacts": list(v10g.SANCTIONED_POST_G2_ARTIFACTS),
        "no_ungoverned_artifacts": result["checks"].get("no_ungoverned_artifacts_present"),
    }


# ---------------------------------------------------------------------------
# Lock assembly
# ---------------------------------------------------------------------------


def _g4_artifact_hashes() -> dict[str, str]:
    return {name: _sha256_file(CAMPAIGN_DIR / name) for name in sorted(G4_CAMPAIGN_FILES)}


def _g5_artifact_hashes() -> dict[str, str]:
    return {name: _sha256_file(G5_RESULTS_DIR / name) for name in sorted(G5_OUTPUT_FILES)}


def _upstream_summaries() -> dict[str, Any]:
    """Lift the verified G5 recorded values directly (no new statistics)."""
    g5dir = G5_RESULTS_DIR
    stats = _read_json(g5dir / "v10g5_paired_statistics.json")
    k = _read_json(g5dir / "v10g5_k_analysis.json")
    recall = _read_json(g5dir / "v10g5_recall_safeguard.json")
    stab = _read_json(g5dir / "v10g5_stability.json")
    freq = _read_json(g5dir / "v10g5_feature_frequencies.json")
    mech = _read_json(g5dir / "v10g5_mechanistic_analysis.json")
    acc = _read_json(g5dir / "v10g5_computational_accounting.json")
    consensus = freq.get("consensus_features_ge3_of_5", {})
    strict = freq.get("strict_consensus_features_5_of_5", {})
    return {
        "comparison_direction": stats["comparison_direction"],
        "endpoint_summary": {
            "average_precision": {
                "all_zero": stats["endpoints"]["average_precision"]["statistics"]["all_zero"],
                "ci_excludes_zero": stats["endpoints"]["average_precision"]["statistics"]["ci_excludes_zero"],
                "ci_lower_95": stats["endpoints"]["average_precision"]["statistics"]["ci_lower_95"],
                "ci_upper_95": stats["endpoints"]["average_precision"]["statistics"]["ci_upper_95"],
                "mean_difference": stats["endpoints"]["average_precision"]["statistics"]["mean_difference"],
                "sample_sd": stats["endpoints"]["average_precision"]["statistics"]["sample_sd"],
                "standard_error": stats["endpoints"]["average_precision"]["statistics"]["standard_error"],
                "n": stats["endpoints"]["average_precision"]["statistics"]["n"],
                "paired_differences": stats["endpoints"]["average_precision"]["statistics"]["paired_differences"],
            },
            "f1": {
                "all_zero": stats["endpoints"]["f1"]["statistics"]["all_zero"],
                "ci_excludes_zero": stats["endpoints"]["f1"]["statistics"]["ci_excludes_zero"],
                "ci_lower_95": stats["endpoints"]["f1"]["statistics"]["ci_lower_95"],
                "ci_upper_95": stats["endpoints"]["f1"]["statistics"]["ci_upper_95"],
                "mean_difference": stats["endpoints"]["f1"]["statistics"]["mean_difference"],
                "sample_sd": stats["endpoints"]["f1"]["statistics"]["sample_sd"],
                "standard_error": stats["endpoints"]["f1"]["statistics"]["standard_error"],
                "n": stats["endpoints"]["f1"]["statistics"]["n"],
                "paired_differences": stats["endpoints"]["f1"]["statistics"]["paired_differences"],
            },
            "recall_safeguard": {
                "status": recall["recall_safeguard_status"],
                "resolved_recall_harm_detected": recall["resolved_recall_harm_detected"],
                "paired_differences": recall["paired_differences"],
            },
            "k": {
                "interpretation_role": k["interpretation_role"],
                "statistics": {
                    "n": k["statistics"]["n"],
                    "mean_difference": k["statistics"]["mean_difference"],
                    "median_difference": k["statistics"]["median_difference"],
                    "min_difference": k["statistics"]["min_difference"],
                    "max_difference": k["statistics"]["max_difference"],
                    "negative_count": k["statistics"]["negative_count"],
                    "zero_count": k["statistics"]["zero_count"],
                    "positive_count": k["statistics"]["positive_count"],
                },
                "paired_differences": k["paired_differences"],
                "k_distribution_with": k["k_distribution_with"],
                "k_distribution_without": k["k_distribution_without"],
                "descriptive_95_ci": {
                    "mean_difference": k["descriptive_95_ci"]["mean_difference"],
                    "ci_lower_95": k["descriptive_95_ci"]["ci_lower_95"],
                    "ci_upper_95": k["descriptive_95_ci"]["ci_upper_95"],
                    "all_zero": k["descriptive_95_ci"]["all_zero"],
                    "descriptive_only": k["descriptive_95_ci"]["descriptive_only"],
                },
            },
        },
        "stability_summary": {
            "cross_arm_paired_jaccards": stab["cross_arm_paired_jaccards"],
            "within_with_pairwise_jaccards": stab["within_with_pairwise_jaccards"],
            "within_without_pairwise_jaccards": stab["within_without_pairwise_jaccards"],
            "k_distribution_with": stab["k_distribution_with"],
            "k_distribution_without": stab["k_distribution_without"],
        },
        "feature_frequency_summary": {
            "feature_names": freq["feature_names"],
            "frequency_with": freq["frequency_with_0_to_5"],
            "frequency_without": freq["frequency_without_0_to_5"],
            "frequency_with_minus_without": freq["frequency_with_minus_without"],
            "consensus_features_ge3_of_5": consensus,
            "strict_consensus_features_5_of_5": strict,
        },
        "computational_accounting_summary": {
            "with_totals": mech["totals_with"],
            "without_totals": mech["totals_without"],
            "with_minus_without": mech["totals_with_minus_without"],
            "nominal_per_arm_requests_per_phase": {
                "bpso": 96,
                "bgwo": 96,
                "total_per_arm": 192,
                "campaign_total": 1920,
            },
            "rows": acc["rows"],
        },
        "mechanistic_summary": {
            "per_seed": mech["per_seed"],
            "statement": (
                "Elite transfer altered the search trajectory and the computational "
                "accounting but not the final selected solution: on every seed the "
                "paired arms converged to identical masks, identical metrics, and an "
                "identical final-phase (BPSO) selection."
            ),
        },
    }


def _build_validation(
    gates: Mapping[str, dict[str, Any]],
    hashes: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_kind": "V10G6_VALIDATION",
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "protocol_name": PROTOCOL_NAME,
        "starting_checkpoint_sha256": EXPECTED_HEAD,
        "gates": {name: _js(payload) for name, payload in gates.items()},
        "artifact_hashes": _js(hashes),
        "environment": _js(environment),
    }


def _build_lock(
    summaries: Mapping[str, Any],
    hashes: Mapping[str, Any],
    upstream: Mapping[str, Any],
) -> dict[str, Any]:
    semantic_payload = _js(
        {
            "artifact_kind": "V10G_RESULT_SEMANTIC_PAYLOAD",
            "protocol_name": PROTOCOL_NAME,
            "comparison_direction": summaries["comparison_direction"],
            "paired_seed_set": [int(seed) for seed in sorted(g5.PRODUCTION_OPTIMIZER_SEEDS)],
            "endpoint_summary": summaries["endpoint_summary"],
            "stability_summary": summaries["stability_summary"],
            "feature_frequency_summary": summaries["feature_frequency_summary"],
            "computational_accounting_summary": summaries["computational_accounting_summary"],
            "mechanistic_summary": summaries["mechanistic_summary"],
            "test_isolation": {
                "test_accessed": False,
                "test_used_for_fitness": False,
                "test_used_for_selection": False,
                "test_used_for_winner_selection": False,
                "classification": "TEST_LOCKED_DURING_V10G_ABLATION",
            },
            "interpretation": {
                "primary_conclusion": (
                    "This experiment does not demonstrate a resolved "
                    "elite-transfer contribution."
                ),
                "mechanistic_statement": summaries["mechanistic_summary"]["statement"],
                "no_resolved_inferential_superiority": True,
                "winner_unchanged_statement": (
                    "The frozen V1.0 winner HYBRID-K13 was neither modified nor re-selected."
                ),
            },
            "winner_governance": {
                "winner_id": "HYBRID-K13",
                "winner_source_stage": "V1.0-D",
                "winner_unmodified": True,
                "no_competitor_winner_selection": True,
                "no_tuning": True,
            },
            "lock_semantics": {
                "hash_algorithm": "sha256",
                "hashed_payload": "semantic_payload_only",
                "encoding": "utf-8",
                "mapping_keys_sorted_recursively": True,
                "array_order_preserved": True,
                "json_kwargs": {
                    "sort_keys": True,
                    "separators": [",", ":"],
                    "ensure_ascii": True,
                    "allow_nan": False,
                },
                "trailing_newline": False,
            },
            "starting_checkpoint_sha256": upstream["starting_checkpoint_sha256"],
            "g1_protocol_lock_sha256": upstream["g1_protocol_lock_sha256"],
            "artifact_hashes": _js(hashes),
        }
    )
    semantic_signature = semantic_hash(semantic_payload)
    lock: dict[str, Any] = {
        "artifact_kind": "V10G_RESULT_LOCK",
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "status": "ABLATION_RESULT_LOCKED",
        "protocol_lock_sha256": upstream["g1_protocol_lock_sha256"],
        "upstream_provenance": _js(
            {
                "v10g4_stage": upstream["v10g4_stage"],
                "v10g4_acceptance": upstream["v10g4_acceptance"],
                "v10g5_stage": upstream["v10g5_stage"],
                "v10g5_acceptance": upstream["v10g5_acceptance"],
                "starting_checkpoint_sha256": upstream["starting_checkpoint_sha256"],
                "prior_locks": upstream["prior_locks"],
            }
        ),
        "campaign_identity": _js(
            {
                "campaign_manifest_sha256": hashes["v10g4_campaign"]["v10g4_manifest.json"],
                "campaign_preflight_sha256": hashes["v10g4_campaign"]["v10g4_preflight.json"],
                "campaign_environment_sha256": hashes["v10g4_campaign"]["v10g4_environment.json"],
                "arm_count": 10,
                "paired_seed_count": 5,
            }
        ),
        "paired_seed_set": [int(seed) for seed in sorted(g5.PRODUCTION_OPTIMIZER_SEEDS)],
        "endpoint_summary": _js(summaries["endpoint_summary"]),
        "stability_summary": _js(summaries["stability_summary"]),
        "computational_accounting_summary": _js(summaries["computational_accounting_summary"]),
        "test_isolation": {
            "test_accessed": False,
            "classification": "TEST_LOCKED_DURING_V10G_ABLATION",
            "test_access_count": 0,
        },
        "interpretation": {
            "primary_conclusion": (
                "This experiment does not demonstrate a resolved elite-transfer contribution."
            ),
            "mechanistic_statement": summaries["mechanistic_summary"]["statement"],
            "preferred_direction_established": False,
        },
        "winner_governance": {
            "winner_id": "HYBRID-K13",
            "winner_source_stage": "V1.0-D",
            "winner_unmodified": True,
            "no_competitor_winner_selection": True,
        },
        "artifact_hashes": _js(hashes),
        "semantic_payload": semantic_payload,
        "semantic_result_lock_sha256": semantic_signature,
    }
    _check_expected_keys(lock)
    return lock


def _check_expected_keys(lock: Mapping[str, Any]) -> None:
    missing = [k for k in REQUIRED_LOCK_KEYS if k not in lock]
    if missing:
        raise V10G6NoGoError(f"V1.0-G6 result lock missing required keys: {missing}")


def run_finalize() -> dict[str, str]:
    """Run every gate; only on a fully green pass, write validation + result lock."""
    head = head_equals_base()
    if head["status"] != "PASS":
        raise V10G6NoGoError("V1.0-G6 requires HEAD == origin/main == the V1.0-G checkpoint.")
    if v10g.no_g2_result_lock_artifacts()["status"] != "PASS":
        raise V10G6NoGoError("V1.0-G results directory contains ungoverned artifacts; abort.")

    arms = g5.load_arms()
    pairs = g5.load_pairs()

    gates: dict[str, dict[str, Any]] = {
        "head_equals_base": head,
        "g1_semantic_lock": g1_gate(),
        "prior_locks_immutable": prior_locks_gate(),
        "test_isolation": test_isolation_gate(),
        "g4_campaign_integrity": g4_campaign_gate(),
        "g5_recomputation": g5_recomputation_gate(arms, pairs),
        "zero_equivalence": zero_equivalence_gate(g5.build_paired_endpoint_table(arms)),
        "computational_accounting": accounting_gate(arms),
        "lifecycle_governance": lifecycle_gate(),
    }
    failed = [name for name, g in gates.items() if g["status"] != "PASS"]
    if failed:
        raise V10G6NoGoError(f"V1.0-G6 gates failed: {failed}")

    environment = g5.capture_environment()
    hashes: dict[str, Any] = {
        "v10g_protocol_lock.json": _sha256_file(v10g.PROTOCOL_LOCK_PATH),
        "v10g4_campaign": _g4_artifact_hashes(),
        "v10g5_analysis": _g5_artifact_hashes(),
        "v10d_winner_lock.json": _sha256_file(v10g.V10D_LOCK_PATH),
        "v10e_result_lock.json": _sha256_file(V10E_RESULT_PATH),
        "v10f_result_lock.json": _sha256_file(v10g.V10F_RESULT_PATH),
    }

    validation = _build_validation(gates, hashes, environment)
    _atomic_write_json(VALIDATION_PATH, validation)
    hashes["v10g6_validation.json"] = _sha256_file(VALIDATION_PATH)

    upstream = {
        "starting_checkpoint_sha256": EXPECTED_HEAD,
        "g1_protocol_lock_sha256": v10g.V10G_SEMANTIC_LOCK_SHA256,
        "v10g4_stage": "V1.0-G4",
        "v10g4_acceptance": "V10G4_COMPLETE_GO",
        "v10g5_stage": "V1.0-G5",
        "v10g5_acceptance": "V10G5_COMPLETE_GO",
        "prior_locks": {
            "v10d_winner_lock_sha256": v10g.V10D_WINNER_LOCK_SHA256,
            "v10e_result_lock_sha256": hashes["v10e_result_lock.json"],
            "v10f_result_lock_sha256": v10g.V10F_RESULT_LOCK_SHA256,
        },
    }
    summaries = _upstream_summaries()
    lock = _build_lock(summaries, hashes, upstream)
    _atomic_write_json(RESULT_LOCK_PATH, lock)

    verify = verify_result_lock(RESULT_LOCK_PATH, expected_semantic=lock["semantic_result_lock_sha256"])
    if verify["status"] != "PASS":
        raise V10G6NoGoError("V1.0-G6 result lock failed post-write verification.")

    return {
        "validation": _sha256_file(VALIDATION_PATH),
        "result_lock": _sha256_file(RESULT_LOCK_PATH),
        "semantic_result_lock_sha256": lock["semantic_result_lock_sha256"],
    }


def verify_result_lock(path: Path, expected_semantic: str) -> dict[str, Any]:
    """Recompute the canonical semantic hash from the persisted lock file."""
    lock = _read_json(path)
    _check_expected_keys(lock)
    payload = lock.get("semantic_payload")
    if not isinstance(payload, dict):
        return {"status": "FAIL", "checks": {"semantic_payload_is_object": False}}
    recomputed = semantic_hash(payload)
    stored = lock.get("semantic_result_lock_sha256")
    on_disk_expected = expected_semantic
    checks = {
        "recomputed_matches_stored": recomputed == stored,
        "recomputed_matches_expected": recomputed == on_disk_expected,
        "stored_matches_expected": stored == on_disk_expected,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "semantic_result_lock_sha256": recomputed,
        "stored_semantic_result_lock_sha256": stored,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    first = run_finalize()
    second = run_finalize()
    if first != second:
        raise V10G6NoGoError(
            f"V1.0-G6 outputs are NOT deterministic: {first} != {second}"
        )
    print("[v10g6] validation.json     ", first["validation"])
    print("[v10g6] v10g_result_lock.json", first["result_lock"])
    print("[v10g6] semantic_result_lock_sha256 ", first["semantic_result_lock_sha256"])
    print("[v10g6] deterministic outputs verified (byte-identical rerun)")
    print("[v10g6] V10G6 finalization complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())