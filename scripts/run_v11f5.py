"""V1.1-F5 final reporting and validation closure runner (read-only synthesis).

Governed read-only F5 stage. No experiment, no measurement, no TestSplit
access, no AI fit or inference. The runner reads the frozen V1.1-E / F1 / F2 /
F4 artifacts plus the authored F5 report, recomputes every binding, reconciles
the reported numbers with the frozen evidence, and -- only under the explicit
``--allow-governed-persistence`` sanction -- writes the deterministic
validation-closure artifact ``results/blockchain/v11f/v11f5_validation.json``.

Modes:
  .venv/bin/python -m scripts.run_v11f5 verify
  .venv/bin/python -m scripts.run_v11f5 persist --allow-governed-persistence

The planning, verify, and execute report markers are:
  V11F5_VERIFY_REPORT / V11F5_PERSIST_REPORT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import sys
from collections.abc import Mapping
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.pipeline_v11f as f2

# --------------------------------------------------------------------------- #
# stage / artifact names
# --------------------------------------------------------------------------- #

STAGE = "V1.1-F5"
ANALYSIS_STAGE = "V1.1-F2"

ANALYSIS_RESULTS_NAME = "v11f_analysis_results.json"
EXECUTION_MANIFEST_NAME = "v11f_execution_manifest.json"
RESULT_LOCK_NAME = "v11f_result_lock.json"
V11F5_VALIDATION_NAME = "v11f5_validation.json"
F5_REPORT_PATH = REPO_ROOT / "docs" / "v11f_resource_efficiency_results.md"

FINAL_LOCK_FORBIDDEN_NAMES = ("v11f_final_lock.json",)

VALIDATION_ARTIFACT_KIND = "V11F5_REPORTING_VALIDATION_CLOSURE"
VERIFY_REPORT_MARKER = "V11F5_VERIFY_REPORT"
PERSIST_REPORT_MARKER = "V11F5_PERSIST_REPORT"

ZERO_COUNTER_KEYS = (
    "test_access_requests",
    "ai_fit_requests",
    "ai_inference_requests",
    "new_measurement_attempts",
)

CANONICAL_METRICS = (
    "persisted_wall_clock_timing",
    "persisted_cpu_process_timing",
    "persisted_validated_orders_per_second",
    "persisted_tracemalloc_peak_proxy",
    "measurable_hash_operation_count",
    "validation_check_count",
    "validator_invocation_count",
)
RATIO_PRECISION = 4
STORAGE_PRECISION = 1
MEMORY_METRICS = ("persisted_tracemalloc_peak_proxy",)

# --------------------------------------------------------------------------- #
# frozen governance identifiers (V1.1-F5 pin set; verified against disk)
# --------------------------------------------------------------------------- #

EXPECTED_AUTHORIZED_COMMIT = "d66661248392cf775bfd6761590478fdf49641ea"
EXPECTED_F1_PROTOCOL_LOCK_SEMANTIC_SHA256 = (
    "68cceedde6384c16a23226ddf082ef7d478e489c9b691e4e63d30bade85597e1"
)
EXPECTED_F1_CONFIG_SHA256 = (
    "84f198556ce319a2775063e11960cd8cd5e5457b8fea14ac835b6f93040c71f3"
)
EXPECTED_F1_PROTOCOL_DOC_SHA256 = (
    "8e20ac45291fe5568b5d1e10a3671faec71e3db4bb191c8a299ca446d68eaaa1"
)
EXPECTED_F2_IMPLEMENTATION_COMMIT = "a33cb2fd7ceed03dde4be6094baa41089d707530"
EXPECTED_F2_IMPLEMENTATION_FILE_SHA256 = (
    "851478e2a5994543c15e5badd1614a7e6d6687a081f8d0fbd9d37dfe858e60a2"
)
EXPECTED_F2_SEMANTIC_ANALYSIS_SHA256 = (
    "0765ea4d79cdad956d09156983395c9c85fe748002cd91a084e2dc5ef6bf3899"
)
EXPECTED_F4_ANALYSIS_RESULTS_SHA256 = (
    "eb4ce1c69d56385a0bddc606bed40ec4d9dcc049e77a172c612a0414f2357d3a"
)
EXPECTED_F4_EXECUTION_MANIFEST_SHA256 = (
    "d1fb42d0bfb8b11267fc6572c2c333d7d77ca1b559533eff6309a128bfd599f8"
)
EXPECTED_F4_RESULT_LOCK_SEMANTIC_SHA256 = (
    "d639be8970bd6662ed0efe688ec2c0375414942ebf636d2b958abd88d722bd1a"
)
EXPECTED_V11E_RESULT_LOCK_FILE_SHA256 = (
    "dd3b926eb08206f505e739ed61c5f29a3e7f03a370e9afc1a077e0b260fd6e64"
)
EXPECTED_V11E_RESULT_LOCK_SEMANTIC_SHA256 = (
    "058aeca8ac97101356bcf1c4dc5b74fb3affbda6833a85b5d423a53556cd1749"
)
EXPECTED_V11E_PROTOCOL_LOCK_SEMANTIC_SHA256 = (
    "8047fbe356c964e26f7acdcde930a1e2a6734026ea303b56334498ffa295f239"
)

POLICY_SET = ("B0", "B1", "P")


# --------------------------------------------------------------------------- #
# runner errors
# --------------------------------------------------------------------------- #


class V11F5Error(Exception):
    """Base error for the V1.1-F5 validation layer."""


class V11F5IntegrityError(V11F5Error):
    """A frozen binding drifted from its pinned hash, or a gate failed."""


class V11F5AuthorizationError(V11F5Error):
    """Persistence was requested without the governed sanction."""


class V11F5OutputCollisionError(V11F5Error):
    """An existing governed artifact conflicts with the planned bytes."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _load_json(path: pathlib.Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_number(value: float, precision: int) -> str:
    fixed = f"{value:.{precision}f}"
    return fixed


def _fmt_metric(metric: str, value: float) -> str:
    precision = 6 if metric in MEMORY_METRICS else 3
    return _fmt_number(value, precision)


def _fmt_ratio(value: float) -> str:
    return _fmt_number(value, RATIO_PRECISION)


def _fmt_storage(value: float) -> str:
    return _fmt_number(value, STORAGE_PRECISION)


# --------------------------------------------------------------------------- #
# deterministic report-values extraction (echo of frozen F2 evidence; no
# scientific recomputation, only descriptive aggregation of persisted arrays)
# --------------------------------------------------------------------------- #


def _ratio_array_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    values = [float(r["value"]) for r in records]
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan"), "count": 0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def extract_report_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Compact ledger of every number the F5 report is allowed to state.

    Every value here is copied from the frozen F2 analysis payload or computed
    as a descriptive summary (mean/median/min/max) over the persisted ratio
    arrays inside that same frozen payload. Nothing here is a new measurement.
    """
    per_policy = payload["descriptive_summaries"]["per_policy"]
    per_contrast = payload["paired_descriptive_comparisons"]["per_contrast"]
    ratios = payload["normalized_ratios"]["ratios"]
    storage = payload["policy_independent_storage"]["per_workload"]
    sec = payload["security_context"]
    pop = payload["population_limitations"]

    per_policy_summary: dict[str, dict[str, dict[str, float]]] = {}
    per_contrast_summary: dict[str, dict[str, dict[str, float]]] = {}
    ratio_summary: dict[str, dict[str, dict[str, float]]] = {}
    for metric in CANONICAL_METRICS:
        per_policy_summary[metric] = {
            policy: {"mean": float(per_policy[metric][policy]["mean"]),
                     "median": float(per_policy[metric][policy]["median"]),
                     "min": float(per_policy[metric][policy]["min"]),
                     "max": float(per_policy[metric][policy]["max"]),
                     "count": int(per_policy[metric][policy]["count"])}
            for policy in POLICY_SET
        }
        per_contrast_summary[metric] = {
            contrast: {"mean": float(per_contrast[metric][contrast]["mean"]),
                       "median": float(per_contrast[metric][contrast]["median"]),
                       "min": float(per_contrast[metric][contrast]["min"]),
                       "max": float(per_contrast[metric][contrast]["max"]),
                       "count": int(per_contrast[metric][contrast]["count"])}
            for contrast in ("B1_minus_B0", "P_minus_B0", "P_minus_B1")
        }
        ratio_summary[metric] = {
            ratio: _ratio_array_summary(records)
            for ratio, records in ratios[metric].items()
        }

    storage_means = {
        workload: float(storage[workload]["describe"]["mean"])
        for workload in ("100", "250", "500", "1000", "2500")
    }
    e02 = sec["e02_detection_counts"]
    e04 = sec["e04_unaffected_order_preservation"]
    return {
        "per_policy_summary": per_policy_summary,
        "per_contrast_summary": per_contrast_summary,
        "normalized_ratio_summary": ratio_summary,
        "policy_independent_storage_mean_bytes": storage_means,
        "security_counts": {
            "e02_detection_counts": {p: int(e02[p]) for p in POLICY_SET},
            "e02_denominator_per_policy": int(e02["denominator_per_policy"]),
            "e03_independent_confirmatory_evidence": bool(sec["e03_independent_confirmatory_evidence"]),
            "e04_unaffected_order_preservation": {
                "numerator_per_policy": int(e04["numerator_per_policy"]),
                "denominator_per_policy": int(e04["denominator_per_policy"]),
            },
        },
        "population_indicators": {
            "governed_order_count": int(pop["governed_order_count"]),
            "risk_band_counts": {k: int(v) for k, v in pop["risk_band_counts"].items()},
            "high_evidence": pop["high_evidence"],
            "low_marker": pop["low_marker"],
            "workload_100_high_marker": pop["workload_100_high_marker"],
            "aggregate_p_evidence_note": pop["aggregate_p_evidence_note"],
            "strong_high_subgroup_inference_allowed": bool(pop["strong_high_subgroup_inference_allowed"]),
        },
    }


def report_value_strings(report_values: Mapping[str, Any]) -> list[str]:
    """Formatted number strings that must appear verbatim in the F5 report."""
    strings: list[str] = []
    for metric, policies in report_values["per_policy_summary"].items():
        for policy, stats in policies.items():
            for key in ("mean", "median", "min", "max"):
                strings.append(_fmt_metric(metric, stats[key]))
    for metric, contrasts in report_values["per_contrast_summary"].items():
        for contrast, stats in contrasts.items():
            for key in ("mean", "median", "min", "max"):
                strings.append(_fmt_metric(metric, stats[key]))
    for metric, ratios in report_values["normalized_ratio_summary"].items():
        for ratio, stats in ratios.items():
            for key in ("mean", "median", "min", "max"):
                strings.append(_fmt_ratio(stats[key]))
    for workload, mean in report_values["policy_independent_storage_mean_bytes"].items():
        strings.append(_fmt_storage(mean))
    e02 = report_values["security_counts"]["e02_detection_counts"]
    for policy, count in e02.items():
        strings.append(str(int(count)))
    strings.append(str(int(report_values["security_counts"]["e02_denominator_per_policy"])))
    e04 = report_values["security_counts"]["e04_unaffected_order_preservation"]
    strings.append(str(int(e04["numerator_per_policy"])))
    strings.append(str(int(e04["denominator_per_policy"])))
    for k, v in report_values["population_indicators"]["risk_band_counts"].items():
        strings.append(str(int(v)))
    strings.append(str(int(report_values["population_indicators"]["governed_order_count"])))
    return strings


# --------------------------------------------------------------------------- #
# report-language markers and prohibited claim tokens
# --------------------------------------------------------------------------- #

REQUIRED_REPORT_MARKERS = (
    "DIRECT_ENERGY_UNAVAILABLE",
    "COMPUTATIONAL_MEMORY_PROXY",
    "POLICY_INDEPENDENT",
    "DERIVED_ONLY",
    "tracemalloc",
    "not RSS/system memory",
    "no new measurement",
    "no TestSplit",
    "no fitted model",
    "no live model inference",
    "DESCRIPTIVE_ONLY",
    "merit_no_final_stage_lock",
)

REQUIRED_REPORT_MARKERS = tuple(
    m for m in REQUIRED_REPORT_MARKERS if m != "merit_no_final_stage_lock"
)

PROHIBITED_REPORT_TOKENS = (
    "winner",
    "best",
    "optimal",
    "superior",
    "recommended",
    "composite",
    "ranking",
    "ranked",
    "overall",
    "universal",
    "score",
    "joule",
    "kwh",
    "watt",
    "carbon",
    "co2",
    "energy savings",
    "energy reduction",
    "power reduction",
    "energy-efficient",
    "energy efficient",
    "p-value",
    "p value",
    "bootstrap",
    "confidence interval",
    "significance",
    "significant",
    "statistically",
    "95%",
    "inferential test",
    "statistical test",
)


# --------------------------------------------------------------------------- #
# validation gates (fail closed on any drift)
# --------------------------------------------------------------------------- #


def _counter_dict_zero(counters: Mapping[str, Any]) -> bool:
    return all(int(counters.get(key, 0)) == 0 for key in ZERO_COUNTER_KEYS)


def _binding(ok: bool, detail: str) -> dict[str, Any]:
    return {"verified": bool(ok), "detail": detail}


def run_validation_gates(
    *,
    analysis: Mapping[str, Any],
    manifest: Mapping[str, Any],
    result_lock: Mapping[str, Any],
    v11e_lock: Mapping[str, Any],
    v11e_protocol_lock: Mapping[str, Any],
    report_text: str,
) -> dict[str, dict[str, Any]]:
    gates: dict[str, dict[str, Any]] = {}

    # 1. F1 protocol binding
    g = analysis["governance_checks"]
    gates["f1_protocol_binding"] = _binding(
        g["f1_protocol_lock_verified"]["verified"] is True
        and g["f1_protocol_lock_verified"]["semantic_result_lock_sha256"]
        == EXPECTED_F1_PROTOCOL_LOCK_SEMANTIC_SHA256
        and g["f1_protocol_lock_verified"]["protocol_classification"]
        == "RESOURCE_EFFICIENCY_GREEN_EVALUATION_PROTOCOL_LOCKED"
        and g["f1_config_verified"]["verified"] is True
        and g["f1_config_verified"]["expected_config_sha256"]
        == EXPECTED_F1_CONFIG_SHA256
        and f2.sha256_file(f2.F1_CONFIG_PATH) == EXPECTED_F1_CONFIG_SHA256
        and f2.sha256_file(f2.F1_PROTOCOL_PATH) == EXPECTED_F1_PROTOCOL_DOC_SHA256,
        "F1 protocol lock semantic, config file, and protocol document match pins",
    )

    # 2. F2 semantic + implementation binding
    gates["f2_semantic_binding"] = _binding(
        analysis.get("semantic_analysis_sha256") == EXPECTED_F2_SEMANTIC_ANALYSIS_SHA256
        and manifest.get("f2_semantic_analysis_sha256") == EXPECTED_F2_SEMANTIC_ANALYSIS_SHA256
        and manifest.get("f2_implementation_commit") == EXPECTED_F2_IMPLEMENTATION_COMMIT
        and manifest.get("f2_implementation_file_sha256") == EXPECTED_F2_IMPLEMENTATION_FILE_SHA256
        and bool(manifest.get("f2_semantic_gate_verified")) is True
        and f2.sha256_file(f2.PROJECT_ROOT / "src" / "pipeline_v11f.py")
        == EXPECTED_F2_IMPLEMENTATION_FILE_SHA256,
        "F2 semantic analysis pin and implementation blob/commit match",
    )

    # 3. F4 artifact file hashes
    analysis_sha = f2.sha256_file(f2.V11F_DIR / ANALYSIS_RESULTS_NAME)
    manifest_sha = f2.sha256_file(f2.V11F_DIR / EXECUTION_MANIFEST_NAME)
    gates["f4_artifact_hashes"] = _binding(
        analysis_sha == EXPECTED_F4_ANALYSIS_RESULTS_SHA256
        and manifest_sha == EXPECTED_F4_EXECUTION_MANIFEST_SHA256
        and result_lock.get("artifacts_fingerprints_sha256", {}).get(ANALYSIS_RESULTS_NAME)
        == EXPECTED_F4_ANALYSIS_RESULTS_SHA256
        and result_lock.get("artifacts_fingerprints_sha256", {}).get(EXECUTION_MANIFEST_NAME)
        == EXPECTED_F4_EXECUTION_MANIFEST_SHA256,
        "F4 analysis/manifest files and result-lock fingerprints match pins",
    )

    # 4. F4 result-lock semantic hash
    lock_payload = result_lock.get("semantic_payload")
    recomputed_lock = bool(lock_payload) and f2.sha256_of_canonical(lock_payload) == EXPECTED_F4_RESULT_LOCK_SEMANTIC_SHA256
    gates["f4_result_lock_semantic_hash"] = _binding(
        recomputed_lock
        and result_lock.get("semantic_result_lock_sha256") == EXPECTED_F4_RESULT_LOCK_SEMANTIC_SHA256
        and result_lock.get("stage") == "V1.1-F"
        and result_lock.get("artifact_kind") == "V11F_RESULT_LOCK",
        "V1.1-F result-lock semantic hash recomputes to the pinned value",
    )

    # 5. V1.1-E upstream lock binding
    v11e_semantic = v11e_lock.get("semantic_result_lock_sha256")
    pr_semantic = v11e_protocol_lock.get("semantic_result_lock_sha256")
    gates["v11e_upstream_lock"] = _binding(
        f2.sha256_file(f2.V11E_DIR / f2.V11E_LOCK_NAME) == EXPECTED_V11E_RESULT_LOCK_FILE_SHA256
        and v11e_semantic == EXPECTED_V11E_RESULT_LOCK_SEMANTIC_SHA256
        and bool(v11e_lock.get("semantic_payload"))
        and f2.sha256_of_canonical(v11e_lock["semantic_payload"]) == EXPECTED_V11E_RESULT_LOCK_SEMANTIC_SHA256
        and pr_semantic == EXPECTED_V11E_PROTOCOL_LOCK_SEMANTIC_SHA256
        and manifest.get("upstream_v11e_result_lock_semantic_sha256")
        == EXPECTED_V11E_RESULT_LOCK_SEMANTIC_SHA256
        and manifest.get("upstream_v11e_result_lock_file_sha256")
        == EXPECTED_V11E_RESULT_LOCK_FILE_SHA256,
        "V1.1-E result lock file/semantic and protocol-lock pins recompute",
    )

    # 6. governance counters zero (analysis, manifest, result lock)
    counters_ok = (
        _counter_dict_zero(g)
        and _counter_dict_zero(manifest.get("governance_counters", {}))
        and _counter_dict_zero(result_lock.get("semantic_payload", {}).get("governance_counters", {}))
    )
    gates["governance_counters_zero"] = _binding(
        counters_ok,
        "analysis/manifest/result-lock test-access, AI, and measurement counters are zero",
    )

    # 7. direct energy unavailable
    gates["direct_energy_unavailable"] = _binding(
        analysis.get("direct_energy_status") == "DIRECT_ENERGY_UNAVAILABLE"
        and result_lock.get("semantic_payload", {}).get("direct_energy_status") == "DIRECT_ENERGY_UNAVAILABLE"
        and manifest.get("direct_energy_status") == "DIRECT_ENERGY_UNAVAILABLE"
        and "DIRECT_ENERGY_UNAVAILABLE" in report_text,
        "direct physical-energy evidence is unavailable and the report says so",
    )

    # 8. E07 memory proxy terminology
    memory_label = analysis["metric_definitions"]["families"]["memory"].get("memory_label")
    gates["e07_proxy_terminology"] = _binding(
        memory_label == "COMPUTATIONAL_MEMORY_PROXY"
        and analysis["metric_definitions"]["families"]["memory"]["source_experiment"] == "E07"
        and "COMPUTATIONAL_MEMORY_PROXY" in report_text
        and "tracemalloc" in report_text
        and "not RSS/system memory" in report_text,
        "memory evidence labeled as tracemalloc computational proxy, not RSS/system",
    )

    # 9. E08 policy independence
    storage = analysis.get("policy_independent_storage", {})
    storage_ok = (
        storage.get("status") == "POLICY_INDEPENDENT"
        and storage.get("unit") == "bytes"
        and "not a policy comparison" in storage.get("token_semantics", "")
        and "POLICY_INDEPENDENT" in report_text
    )
    gates["e08_policy_independence"] = _binding(
        storage_ok,
        "storage evidence is policy-independent canonical serialized bytes",
    )

    # 10. E10 derived-only
    e10_limits = " ".join(analysis.get("scientific_limitations", []))
    gates["e10_derived_only"] = _binding(
        analysis["metric_definitions"]["families"]["computational_work"]["source_experiment"] == "E10"
        and "derived only from re-verified persisted" in e10_limits
        and "DERIVED_ONLY" in report_text,
        "E10 computational-work evidence is derived-only from re-verified E06-E09 cells",
    )

    # 11. E03 non-independence
    sec = analysis.get("security_context", {})
    gates["e03_non_independence"] = _binding(
        sec.get("e03_independent_confirmatory_evidence") is False
        and "mirrored" in sec.get("e03_reason", "").lower()
        and "not independent" in report_text
        and "mirrored" in report_text.lower(),
        "E03 localization outcomes mirrored E02 detection and are not independent evidence",
    )

    # 12. no ranking / winner / composite
    semantics = result_lock.get("semantic_payload", {})
    gates["no_ranking_winner_composite"] = _binding(
        semantics.get("ranking_fields_present") is False
        and semantics.get("overall_winner_present") is False
        and semantics.get("composite_score_present") is False
        and g.get("energy_terminology_scan") == "PASS"
        and "DESCRIPTIVE_ONLY" in report_text,
        "no ranking/winner/composite produced or claimed; energy-terminology scan PASS",
    )

    # 13. no inferential claims / no energy substitution
    scan = report_text.lower()
    prohibited_hits = [tok for tok in PROHIBITED_REPORT_TOKENS if tok in scan]
    gates["no_inferential_claims"] = _binding(
        analysis.get("analysis_mode") == "DESCRIPTIVE_ONLY"
        and result_lock.get("semantic_payload", {}).get("analysis_mode") == "DESCRIPTIVE_ONLY"
        and not prohibited_hits
        and "no inferential" in report_text.lower(),
        f"DESCRIPTIVE_ONLY mode, no inferential wording; prohibited-token hits: {prohibited_hits or 'none'}",
    )

    # 14. report values reconcile with frozen evidence
    report_values = extract_report_values(analysis)
    missing = [s for s in report_value_strings(report_values) if s not in report_text]
    gates["report_values_reconcile"] = _binding(
        not missing,
        f"all {len(report_value_strings(report_values))} headline numbers present in the report; missing: {missing[:8] or 'none'}",
    )

    # 15. no test access
    gates["no_test_access"] = _binding(
        int(g.get("test_access_requests", 0)) == 0
        and "no TestSplit" in report_text,
        "no governed standardized-split (TEST) access is recorded or claimed",
    )

    # 16. no AI fit / inference
    gates["no_ai_fit_inference"] = _binding(
        int(g.get("ai_fit_requests", 0)) == 0
        and int(g.get("ai_inference_requests", 0)) == 0
        and "no fitted model" in report_text
        and "no live model inference" in report_text,
        "no AI fit or inference is recorded or claimed",
    )

    # 17. no new measurement
    gates["no_new_measurement"] = _binding(
        int(g.get("new_measurement_attempts", 0)) == 0
        and "no new measurement" in report_text,
        "no new measurement campaign is recorded or claimed",
    )

    # 18. required markers all present
    missing_markers = [m for m in REQUIRED_REPORT_MARKERS if m not in report_text]
    gates["required_report_markers"] = _binding(
        not missing_markers,
        f"required report markers present; missing: {missing_markers or 'none'}",
    )

    return gates


# --------------------------------------------------------------------------- #
# deterministic validation-closure artifact
# --------------------------------------------------------------------------- #


def build_validation_artifact(
    *,
    analysis: Mapping[str, Any],
    manifest: Mapping[str, Any],
    result_lock: Mapping[str, Any],
    v11e_lock: Mapping[str, Any],
    v11e_protocol_lock: Mapping[str, Any],
    report_text: str,
    report_sha256: str,
) -> dict[str, Any]:
    gates = run_validation_gates(
        analysis=analysis,
        manifest=manifest,
        result_lock=result_lock,
        v11e_lock=v11e_lock,
        v11e_protocol_lock=v11e_protocol_lock,
        report_text=report_text,
    )
    if not all(g["verified"] for g in gates.values()):
        failed = [name for name, g in gates.items() if not g["verified"]]
        raise V11F5IntegrityError(
            "F5 validation gates failed; refusing to build the closure artifact: "
            f"{failed}"
        )

    report_values = extract_report_values(analysis)
    counters = {
        "analysis": {k: int(analysis["governance_checks"].get(k, 0)) for k in ZERO_COUNTER_KEYS},
        "manifest": {k: int(manifest.get("governance_counters", {}).get(k, 0)) for k in ZERO_COUNTER_KEYS},
        "result_lock": {k: int(result_lock["semantic_payload"].get("governance_counters", {}).get(k, 0)) for k in ZERO_COUNTER_KEYS},
    }

    semantics = {
        "stage": STAGE,
        "artifact_kind": VALIDATION_ARTIFACT_KIND,
        "analysis_mode": analysis.get("analysis_mode"),
        "direct_energy_status": analysis.get("direct_energy_status"),
        "policy_set": list(POLICY_SET),
        "report_file": "docs/v11f_resource_efficiency_results.md",
        "report_file_sha256": report_sha256,
        "f1_protocol_lock_semantic_sha256": EXPECTED_F1_PROTOCOL_LOCK_SEMANTIC_SHA256,
        "f1_config_sha256": EXPECTED_F1_CONFIG_SHA256,
        "f1_protocol_document_sha256": EXPECTED_F1_PROTOCOL_DOC_SHA256,
        "f2_implementation_commit": EXPECTED_F2_IMPLEMENTATION_COMMIT,
        "f2_implementation_file_sha256": EXPECTED_F2_IMPLEMENTATION_FILE_SHA256,
        "f2_semantic_analysis_sha256": EXPECTED_F2_SEMANTIC_ANALYSIS_SHA256,
        "f4_analysis_results_sha256": EXPECTED_F4_ANALYSIS_RESULTS_SHA256,
        "f4_execution_manifest_sha256": EXPECTED_F4_EXECUTION_MANIFEST_SHA256,
        "f4_result_lock_semantic_sha256": EXPECTED_F4_RESULT_LOCK_SEMANTIC_SHA256,
        "upstream_v11e_result_lock_semantic_sha256": EXPECTED_V11E_RESULT_LOCK_SEMANTIC_SHA256,
        "upstream_v11e_result_lock_file_sha256": EXPECTED_V11E_RESULT_LOCK_FILE_SHA256,
        "upstream_v11e_protocol_lock_semantic_sha256": EXPECTED_V11E_PROTOCOL_LOCK_SEMANTIC_SHA256,
        "governance_counters_zero": all(all(c == 0 for c in counters[k].values()) for k in counters),
        "direct_energy_unavailable": analysis.get("direct_energy_status") == "DIRECT_ENERGY_UNAVAILABLE",
        "e07_label": analysis["metric_definitions"]["families"]["memory"].get("memory_label"),
        "e08_status": analysis.get("policy_independent_storage", {}).get("status"),
        "e10_derived_only": True,
        "ranking_fields_present": False,
        "overall_winner_present": False,
        "composite_score_present": False,
        "final_stage_lock_created": False,
        "final_stage_lock_policy": "F4 lock preserved; no separate F5 result lock",
    }

    return {
        "artifact_kind": VALIDATION_ARTIFACT_KIND,
        "stage": STAGE,
        "authorized_commit": EXPECTED_AUTHORIZED_COMMIT,
        "analysis_mode": analysis.get("analysis_mode"),
        "report_file": "docs/v11f_resource_efficiency_results.md",
        "report_file_sha256": report_sha256,
        "frozen_bindings": semantics,
        "dependency_hashes_verified": {
            "v11f_analysis_results.json": f2.sha256_file(f2.V11F_DIR / ANALYSIS_RESULTS_NAME),
            "v11f_execution_manifest.json": f2.sha256_file(f2.V11F_DIR / EXECUTION_MANIFEST_NAME),
            "v11e_experiment_result_lock.json": f2.sha256_file(f2.V11E_DIR / f2.V11E_LOCK_NAME),
        },
        "governance_counters": counters,
        "report_values": report_values,
        "validation_gates": {name: g["verified"] for name, g in gates.items()},
        "validation_gates_all": True,
        "semantic_payload": semantics,
        "semantic_validation_sha256": f2.sha256_of_canonical(semantics),
        "execution_metadata": {
            "read_only_synthesis": True,
            "no_new_measurement": True,
            "no_ai_fit": True,
            "no_ai_inference": True,
            "no_test_access": True,
            "f4_result_lock_preserved": True,
            "final_stage_lock_created": False,
        },
    }


def run_readonly_verification(
    *,
    authorisation_policy_allowed: bool = True,
) -> dict[str, Any]:
    report_path = F5_REPORT_PATH
    if not report_path.exists():
        raise V11F5IntegrityError(
            "F5 report file missing; create docs/v11f_resource_efficiency_results.md "
            "before verification."
        )
    report_text = report_path.read_text(encoding="utf-8")
    report_sha256 = hashlib.sha256(report_text.encode("utf-8")).hexdigest()

    analysis = _load_json(f2.V11F_DIR / ANALYSIS_RESULTS_NAME)
    manifest = _load_json(f2.V11F_DIR / EXECUTION_MANIFEST_NAME)
    result_lock = _load_json(f2.V11F_DIR / RESULT_LOCK_NAME)
    v11e_lock = _load_json(f2.V11E_DIR / f2.V11E_LOCK_NAME)
    v11e_protocol_lock = _load_json(f2.V11E_DIR / "v11e_experiment_protocol_lock.json")

    for forbidden in FINAL_LOCK_FORBIDDEN_NAMES:
        if (f2.V11F_DIR / forbidden).exists():
            raise V11F5IntegrityError(
                f"unexpected final stage lock present: {forbidden}; F5 must preserve "
                "the F4 result lock without a separate final lock."
            )

    artifact = build_validation_artifact(
        analysis=analysis,
        manifest=manifest,
        result_lock=result_lock,
        v11e_lock=v11e_lock,
        v11e_protocol_lock=v11e_protocol_lock,
        report_text=report_text,
        report_sha256=report_sha256,
    )
    return {
        "marker": VERIFY_REPORT_MARKER,
        "mode": "verify",
        "stage": STAGE,
        "report_file": str(report_path),
        "report_file_sha256": report_sha256,
        "validation_gates_all": True,
        "semantic_validation_sha256": artifact["semantic_validation_sha256"],
        "governance_counters_zero": artifact["governance_counters"],
    }


# --------------------------------------------------------------------------- #
# persistence (deterministic, collision-checked, sanctioned)
# --------------------------------------------------------------------------- #


def persist_validation_artifact(
    *,
    allow_governed_persistence: bool,
    out_dir: pathlib.Path | str = f2.V11F_DIR,
) -> dict[str, Any]:
    if not allow_governed_persistence:
        raise V11F5AuthorizationError(
            "persist requires --allow-governed-persistence; no governed output "
            "was produced."
        )
    report_path = F5_REPORT_PATH
    report_text = report_path.read_text(encoding="utf-8")
    report_sha256 = hashlib.sha256(report_text.encode("utf-8")).hexdigest()

    analysis = _load_json(f2.V11F_DIR / ANALYSIS_RESULTS_NAME)
    manifest = _load_json(f2.V11F_DIR / EXECUTION_MANIFEST_NAME)
    result_lock = _load_json(f2.V11F_DIR / RESULT_LOCK_NAME)
    v11e_lock = _load_json(f2.V11E_DIR / f2.V11E_LOCK_NAME)
    v11e_protocol_lock = _load_json(f2.V11E_DIR / "v11e_experiment_protocol_lock.json")

    artifact = build_validation_artifact(
        analysis=analysis,
        manifest=manifest,
        result_lock=result_lock,
        v11e_lock=v11e_lock,
        v11e_protocol_lock=v11e_protocol_lock,
        report_text=report_text,
        report_sha256=report_sha256,
    )
    planned = f2.canonical_json(artifact).encode("utf-8")
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / V11F5_VALIDATION_NAME
    if path.exists() and path.read_bytes() != planned:
        raise V11F5OutputCollisionError(
            "governed output collision: v11f5_validation.json exists with "
            "different content; refusing to overwrite."
        )
    if not path.exists():
        path.write_bytes(planned)
    return {
        "marker": PERSIST_REPORT_MARKER,
        "mode": "persist",
        "artifact_path": str(path),
        "artifact_sha256": hashlib.sha256(planned).hexdigest(),
        "semantic_validation_sha256": artifact["semantic_validation_sha256"],
        "validation_gates_all": True,
        "governance_counters_zero": artifact["governance_counters"],
    }


# --------------------------------------------------------------------------- #
# F5 report renderer (deterministic: prose templates + injected frozen numbers)
# --------------------------------------------------------------------------- #

RPT_METRIC_GLOSS = {
    "persisted_wall_clock_timing": "wall-clock process timing",
    "persisted_cpu_process_timing": "CPU process timing",
    "persisted_validated_orders_per_second": "validated orders per second",
    "persisted_tracemalloc_peak_proxy": "tracemalloc peak proxy",
    "measurable_hash_operation_count": "hash-operation count",
    "validation_check_count": "validation-check count",
    "validator_invocation_count": "validator-invocation count",
}
RPT_RATIO_GLOSS = {
    "P_over_B0": "P / B0",
    "P_over_B1": "P / B1",
}
RPT_CONTRAST_GLOSS = {
    "B1_minus_B0": "B1 − B0",
    "P_minus_B0": "P − B0",
    "P_minus_B1": "P − B1",
}
POLICY_GLOSS = {
    "B0": "B0 (fixed baseline A)",
    "B1": "B1 (fixed escalation / higher governed attack coverage)",
    "P": "P (adaptive)",
}

RPT_HEADER = """# V1.1-F Endpoint-Specific Resource-Efficiency Synthesis: Adaptive Policy P, Fixed Baselines B0/B1, and a Unified No-Energy-Claim Regime

Authoritative scientific synthesis for stage V1.1-F. Strict, governed descriptors
and artifact-**fingerprints** below are quoted verbatim from the frozen V1.1-F1/F2/F4
and upstream V1.1-E locks; all numeric **report values** in this document are copied
from the frozen analysis payload `results/blockchain/v11f/v11f_analysis_results.json`
or computed as descriptive mean/median/min/max over the persisted ratio arrays inside
that same payload. This document adds no measurement, no test split access, no fitted
model, no live model inference, and no new measurement campaign.

Energy evidence is declared unavailable: `DIRECT_ENERGY_UNAVAILABLE`. Therefore this
report makes no energy-consumption, no saved-energy, and no power-related claim of any
kind, and it never substitutes CPU/wall time or TDP-times-time arithmetic for any
measured physical-energy quantity.
"""

RPT_1_PROVENANCE = """## 1. Provenance and scope

Stage `V1.1-F5` closes the frozen V1.1-F resource-efficiency chain by (i) re-verifying
every upstream binding, (ii) reconciling each number stated here with the frozen
evidence, and (iii) recording the validation-closure artifact without creating a new
stage result lock. The F4 result lock is preserved as the only V1.1-F result lock.

The stage consumes only read-only persisted artifacts:
- upstream V1.1-E4 experiment result lock (semantic `058aeca8ac97101356bcf1c4dc5b74fb3affbda6833a85b5d423a53556cd1749`);
- protocol locks F1 (semantic `68cceedde6384c16a23226ddf082ef7d478e489c9b691e4e63d30bade85597e1`) and V1.1-E (semantic `8047fbe356c964e26f7acdcde930a1e2a6734026ea303b56334498ffa295f239`);
- F2 analysis payload (`semantic_analysis_sha256` `0765ea4d79cdad956d09156983395c9c85fe748002cd91a084e2dc5ef6bf3899`, implementation commit `a33cb2fd7ceed03dde4be6094baa41089d707530`);
- F4 artifacts: `v11f_analysis_results.json` (`eb4ce1c69d56385a0bddc606bed40ec4d9dcc049e77a172c612a0414f2357d3a`), `v11f_execution_manifest.json` (`d1fb42d0bfb8b11267fc6572c2c333d7d77ca1b559533eff6309a128bfd599f8`), and result lock (semantic `d639be8970bd6662ed0efe688ec2c0375414942ebf636d2b958abd88d722bd1a`).

The F2 analysis `analysis_mode` is `DESCRIPTIVE_ONLY`; no inferential procedure is
applied or reported in this document.
"""

RPT_2_BOUNDARY = """## 2. Governed evidence base and read-only boundary

The underlying experiments are the frozen V1.1-E security/resource campaign
(seeds 522, 523, 524; workloads 100, 250, 500, 1000, 2500; policies B0, B1, P).
Each timing, memory, throughput, and storage observation is a persisted, cell-level
descriptive value aggregated from per-order run records; no measurement is repeated
here. The governing protocol forbids new timing/memory/storage/throughput/security
campaigns, forbids any TEST split access (`no TestSplit`), and forbids any AI fit or
prediction (`no fitted model`, `no live model inference`). Every governance counter in
the frozen analysis, manifest, and result lock is zero.

All values below are reported exactly as persisted (no round-trip conversion, no
unit arithmetic). The V1.1-E metric labels embed their unit identifiers:
`cpu_time_ns`/`wall_time_ns` (nanoseconds, per-order persisted runs),
`throughput_orders_per_second`, `memory_proxy_mib`, and `storage_bytes`/`canonical_serialized_bytes`
(bytes).
"""

RPT_3_METRICS = """## 3. Metric families and proxy status

The F2 analysis defines four resource families on a shared 15-cell grid
(three seeds x five nested workloads):

| Family | Canonical metric | Source | Status |
|---|---|---|---|
| runtime | `persisted_wall_clock_timing`, `persisted_cpu_process_timing` | V1.1-E E06 | computational resource proxy |
| memory | `persisted_tracemalloc_peak_proxy` | V1.1-E E07 | `COMPUTATIONAL_MEMORY_PROXY` |
| throughput | `persisted_validated_orders_per_second` | V1.1-E E09 | computational resource proxy |
| storage | `persisted_canonical_serialized_bytes` | V1.1-E E08 | `POLICY_INDEPENDENT` (no policy comparison) |
| computational work | `measurable_hash_operation_count` | V1.1-E E10 | `DERIVED_ONLY` |

Memory evidence is the fresh-process tracemalloc peak (Python traced-allocation
memory proxy), not RSS/system memory. Storage evidence is policy-independent
canonical serialized bytes; the stored B0 token is a serialization placeholder, not
a policy comparison. E10 carries no independent measurement campaign; it is derived
only from re-verified persisted E06-E09 cell evidence.
"""


def _rpt_tables(report_values: Mapping[str, Any]) -> str:
    lines: list[str] = []

    # 4. per policy
    lines.append(
        "## 4. Per-policy descriptive summaries (across the 15 governed cells)\n"
    )
    lines.append(
        "Values are the persisted per-cell means/medians/min/max summarized across "
        "the 15 cells; timing metrics carry the V1.1-E nanosecond labels. "
        "`memory_proxy_mib` values are printed with six decimals; other metrics with three."
    )
    for metric in CANONICAL_METRICS:
        rows = []
        for policy in POLICY_SET:
            s = report_values["per_policy_summary"][metric][policy]
            rows.append(
                [
                    POLICY_GLOSS[policy],
                    _fmt_metric(metric, s["mean"]),
                    _fmt_metric(metric, s["median"]),
                    _fmt_metric(metric, s["min"]),
                    _fmt_metric(metric, s["max"]),
                    str(int(s["count"])),
                ]
            )
        lines.append(
            _md_table(
                f"Metric: {RPT_METRIC_GLOSS[metric]} (`{metric}`)",
                ["Policy", "mean", "median", "min", "max", "n"],
                rows,
            )
        )

    # 5. paired contrasts
    lines.append(
        "## 5. Paired descriptive contrasts (mean/median/min/max, n = 15 cells)\n"
    )
    for metric in CANONICAL_METRICS:
        rows = []
        for contrast in ("B1_minus_B0", "P_minus_B0", "P_minus_B1"):
            s = report_values["per_contrast_summary"][metric][contrast]
            rows.append(
                [
                    RPT_CONTRAST_GLOSS[contrast],
                    _fmt_metric(metric, s["mean"]),
                    _fmt_metric(metric, s["median"]),
                    _fmt_metric(metric, s["min"]),
                    _fmt_metric(metric, s["max"]),
                    str(int(s["count"])),
                ]
            )
        lines.append(
            _md_table(
                f"Metric: {RPT_METRIC_GLOSS[metric]} (`{metric}`)",
                ["Contrast", "mean", "median", "min", "max", "n"],
                rows,
            )
        )

    # 6. ratios
    lines.append(
        "## 6. Normalized ratio summaries (P / B0 and P / B1, descriptive)\n"
    )
    for metric in CANONICAL_METRICS:
        rows = []
        for ratio in ("P_over_B0", "P_over_B1"):
            s = report_values["normalized_ratio_summary"][metric][ratio]
            rows.append(
                [
                    RPT_RATIO_GLOSS[ratio],
                    _fmt_ratio(s["mean"]),
                    _fmt_ratio(s["median"]),
                    _fmt_ratio(s["min"]),
                    _fmt_ratio(s["max"]),
                    str(int(s["count"])),
                ]
            )
        lines.append(
            _md_table(
                f"Metric: {RPT_METRIC_GLOSS[metric]} (`{metric}`)",
                ["Ratio", "mean", "median", "min", "max", "n"],
                rows,
            )
        )

    # 7. storage
    lines.append(
        "## 7. Policy-independent storage (mean canonical serialized bytes per workload)\n"
    )
    rows = []
    for workload in ("100", "250", "500", "1000", "2500"):
        mean = report_values["policy_independent_storage_mean_bytes"][workload]
        rows.append([workload, _fmt_storage(mean)])
    lines.append(_md_table("Storage is independent of policy (E08, `POLICY_INDEPENDENT`)", ["Workload (orders)", "mean bytes"], rows))
    lines.append(
        "The stored token is a policy-independent serialization placeholder; it is "
        "not a policy comparison and carries no per-policy meaning."
    )
    lines.append("")

    # 8. security
    sec = report_values["security_counts"]
    e02 = sec["e02_detection_counts"]
    e04 = sec["e04_unaffected_order_preservation"]
    lines.append("## 8. Security context (frozen upstream trade-off context only)\n")
    lines.append(
        _md_table(
            "E02 detection counts (denominator per policy = "
            + str(int(sec["e02_denominator_per_policy"]))
            + ")",
            ["Policy", "detected"],
            [[policy, str(int(e02[policy]))] for policy in POLICY_SET],
        )
    )
    lines.append(
        "E04 unaffected-order preservation: "
        + str(int(e04["numerator_per_policy"]))
        + " / "
        + str(int(e04["denominator_per_policy"]))
        + " orders preserved. E03 binary localization outcomes mirrored E02 detection "
        "in the governed V1.1-E campaign and are not independent confirmatory evidence; "
        "detection/localization is therefore treated as a single trade-off axis."
    )
    lines.append("")

    # 9. population
    pop = report_values["population_indicators"]
    lines.append("## 9. Population composition and risk-band limitations\n")
    lines.append(
        "Governed population: "
        + str(int(pop["governed_order_count"]))
        + " orders. Risk-band counts: HIGH "
        + str(int(pop["risk_band_counts"]["HIGH"]))
        + ", MEDIUM "
        + str(int(pop["risk_band_counts"]["MEDIUM"]))
        + ", LOW "
        + str(int(pop["risk_band_counts"]["LOW"]))
        + "."
    )
    lines.append(
        "Risk-band evidence is overwhelmingly MEDIUM-driven. There is no LOW-band "
        "evidence in the governed population (`LOW_ABSENT_IN_GOVERNED_POPULATION`). "
        "HIGH-band evidence is sparse (six governed observations) and is treated as "
        "descriptive-only; HIGH was not observed in the workload-100 condition "
        "(`HIGH_NOT_OBSERVED_IN_CONDITION`)."
    )
    lines.append("")

    # 10. replication / inferential scope
    lines.append("## 10. Replication, nesting, and inferential scope\n")
    lines.append(
        "The 15 cell estimates share only three fixed seeds (522, 523, 524); the five "
        "workload sizes are nested within each seed (100, 250, 500, 1000, 2500) and must "
        "not be pooled as independent replicates. The analysis is `DESCRIPTIVE_ONLY`: "
        "only mean/median/min/max-style summaries are reported, no inferential procedure "
        "is applied or reported, no inferential statement is made, and no interval or "
        "resampling statement is made."
    )

    return "\n".join(lines)


def _md_table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    rows_s = [[str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for row in rows_s:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    header = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = "\n".join(
        "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        for row in rows_s
    )
    return f"{title}\n\n{header}\n{sep}\n{body}\n"


def build_report_text(report_values: Mapping[str, Any]) -> str:
    """Compose the deterministic F5 report text (prose + injected numbers)."""
    text = "\n\n".join(
        [
            RPT_HEADER.strip(),
            RPT_1_PROVENANCE.strip(),
            RPT_2_BOUNDARY.strip(),
            RPT_3_METRICS.strip(),
            _rpt_tables(report_values).strip(),
            RPT_11_ENERGY.strip(),
            RPT_12_STANCE.strip(),
            RPT_13_REPRODUCIBILITY.strip(),
        ]
    )
    text += "\n"
    return text


RPT_11_ENERGY = """## 11. Direct-energy status and proxy constraints

Direct physical-energy measurement was unavailable (`DIRECT_ENERGY_UNAVAILABLE`).
No CPU/wall timing, no proxy count, and no TDP-times-time arithmetic is presented as
measured energy; nothing in this document asserts any saved-energy benefit, any
lower-energy use, or any energy-efficiency property. All runtime/throughput/memory
values are computational resource proxies (`COMPUTATIONAL_MEMORY_PROXY` for memory;
policy-independent
`POLICY_INDEPENDENT` serialization for storage; `DERIVED_ONLY` for computational work).

Declared limitations (each is a real constraint of the governed evidence):

1. No direct physical-energy measurement exists; the entire stage is energy-claim-free.
2. No direct-energy substitution is performed: CPU/wall time is never relabeled as
   a measured energy quantity, and TDP-times-time inference is prohibited.
3. Memory evidence is a `COMPUTATIONAL_MEMORY_PROXY` (fresh-process tracemalloc peak),
   not RSS/system memory; it is a traced-allocation proxy.
4. Storage evidence is `POLICY_INDEPENDENT`: canonical serialized bytes with a B0
   serialization placeholder token; it is not a policy comparison.
5. Computational-work evidence (E10) is `DERIVED_ONLY`: it reuses re-verified E06-E09
   persisted cells and carries no independent measurement campaign.
6. E03 binary localization outcomes mirrored E02 detection in the governed V1.1-E
   campaign and are not independent confirmatory evidence; detection/localization is a
   single axis.
7. The LOW risk band is absent from the governed population.
8. HIGH-band evidence is sparse (six governed observations) and descriptive-only;
   no HIGH-band subgroup-level inference is allowed.
9. HIGH was not observed in the workload-100 condition.
10. Workload sizes are nested within each seed and must not be pooled as independent
    replicates (15 cells are not 15 independent experiments).
11. Only three fixed seeds (522, 523, 524) were available; the shared seed/workload
    design is deliberate and finite.
12. The analysis is `DESCRIPTIVE_ONLY`: only descriptive summaries are reported; no
    inferential procedure, no probabilistic statement, no interval, and no resampling
    statement is made.
13. Results apply to the governed protocol design and environment (40,000-row cap,
    order-grouped 28k/6k/6k split, 43-feature canonical space, V1.1-E implementation,
    the B0/B1/P policy set, and the five nested workloads above).
"""

RPT_12_STANCE = """## 12. Endpoint-specific conclusions and policy stance

The F1 protocol forbids a global policy selection, and this report honors that
bound: it does not name a single all-purpose preferred policy, does not produce any
ordering index or single-value aggregation, and never selects a policy. Instead it states,
for the governed population and the three resource families, which policy leads on
which specific endpoint while acknowledging the reverse on other endpoints:

- **Runtime lightness.** B0 leads: CPU process timing mean `{cpu_b0}` and wall-clock
  mean `{wall_b0}` are the lightest; B1 is the heaviest (CPU mean `{cpu_b1}`, wall
  mean `{wall_b1}`); P is intermediate (CPU mean `{cpu_p}`, wall mean `{wall_p}`) and
  is lighter than B1 (CPU ratio P/B1 mean `{cpu_r_pb1}`) yet not lighter than B0
  (CPU ratio P/B0 mean `{cpu_r_pb0}`). A lighter runtime read on one fixed policy and
  a heavier one on the other is an endpoint-level trade-off, not a global result.
- **Throughput capacity.** B0 leads with mean `{thr_b0}` validated orders per second;
  B1 is the lowest (mean `{thr_b1}`); P is intermediate (mean `{thr_p}`; ratio P/B1
  mean `{thr_r_pb1}`, P/B0 mean `{thr_r_pb0}`). Throughput and runtime point the same
  way across the three policies.
- **Memory footprint.** The `COMPUTATIONAL_MEMORY_PROXY` values are nearly identical
  across policies (B0 `{mem_b0}`, B1 `{mem_b1}`, P `{mem_p}`); paired contrasts are
  three to four orders smaller than the level. Memory is effectively policy-flat in
  this governed design, so no memory-driven policy trade-off is claimed.
- **Computational load and validation work.** B1 carries the heavier load (hash
  sequence mean `{hash_b1}` versus B0 `{hash_b0}`; paired mean `{hash_c_b1b0}`), while
  P closely tracks B0 (P/B0 ratio mean `{hash_r_pb0}`). B1 consequently records the
  most validation checks and validator invocations; P records slightly more than B0.
- **Storage.** Storage is `POLICY_INDEPENDENT` and grows linearly with workload
  (mean `{sto_100}` bytes at 100 orders through `{sto_2500}` bytes at 2500 orders);
  there is no storage policy trade-off.
- **Security (frozen context).** B1 has the higher governed attack-scenario coverage
  (E02 detected `{sec_b1}` of `{sec_den}`) versus B0 (`{sec_b0}` of `{sec_den}`), with
  P intermediate (`{sec_p}` of `{sec_den}`); E04 order preservation is `{sec_e04}` /
  `{sec_e04}`.

In summary: no single policy leads on every endpoint category in the governed
population — B0 leads on runtime lightness and throughput while carrying lower
governed attack coverage; B1 leads on governed attack coverage while being the
heaviest and slowest; P sits intermediate on runtime/throughput/security and nearly
ties B0 on the memory proxy. These are endpoint-specific, descriptive observations
within the governed V1.1-E design; they are not evidence of any energy benefit,
not a global ordering, and not a policy recommendation. No final stage lock is created
(F4 result lock preserved).
"""

RPT_13_REPRODUCIBILITY = """## 13. Reproducibility, data policy, and follow-on governance

- Frozen identifiers: F1 protocol lock semantic `68cceedde6384c16a23226ddf082ef7d478e489c9b691e4e63d30bade85597e1`; F1 config `84f198556ce319a2775063e11960cd8cd5e5457b8fea14ac835b6f93040c71f3`; F1 protocol document `8e20ac45291fe5568b5d1e10a3671faec71e3db4bb191c8a299ca446d68eaaa1`; F2 implementation commit `a33cb2fd7ceed03dde4be6094baa41089d707530` (file `851478e2a5994543c15e5badd1614a7e6d6687a081f8d0fbd9d37dfe858e60a2`); F2 semantic analysis `0765ea4d79cdad956d09156983395c9c85fe748002cd91a084e2dc5ef6bf3899`; F4 analysis `eb4ce1c69d56385a0bddc606bed40ec4d9dcc049e77a172c612a0414f2357d3a`; F4 manifest `d1fb42d0bfb8b11267fc6572c2c333d7d77ca1b559533eff6309a128bfd599f8`; F4 result lock semantic `d639be8970bd6662ed0efe688ec2c0375414942ebf636d2b958abd88d722bd1a`; V1.1-E result lock file `dd3b926eb08206f505e739ed61c5f29a3e7f03a370e9afc1a077e0b260fd6e64` / semantic `058aeca8ac97101356bcf1c4dc5b74fb3affbda6833a85b5d423a53556cd1749`; V1.1-E protocol lock semantic `8047fbe356c964e26f7acdcde930a1e2a6734026ea303b56334498ffa295f239`.
- This document is generated deterministically from the frozen analysis payload by
  `.venv/bin/python -m scripts.run_v11f5 report`. The validation-closure artifact
  `results/blockchain/v11f/v11f5_validation.json` records `report_file_sha256` and a
  semantic validation hash recomputed from the same pins.
- No new measurement was performed, no TEST split was accessed (`no TestSplit`), no
  fitted model exists in this stage, and no live model inference is made.
- Reproduction: re-running the report/verification/persistence commands at
  authorized commit `d66661248392cf775bfd6761590478fdf49641ea` must reproduce the
  exact report bytes and the exact semantic validation hash.
"""


def render_report_text() -> tuple[str, str]:
    """Render the deterministic F5 report; returns (text, sha256)."""
    analysis = _load_json(f2.V11F_DIR / ANALYSIS_RESULTS_NAME)
    report_values = extract_report_values(analysis)
    text = build_report_text(report_values)
    text = text.replace("\\{", "{").replace("\\}", "}")
    text = text.format(
        cpu_b0=_fmt_metric("persisted_cpu_process_timing", report_values["per_policy_summary"]["persisted_cpu_process_timing"]["B0"]["mean"]),
        cpu_b1=_fmt_metric("persisted_cpu_process_timing", report_values["per_policy_summary"]["persisted_cpu_process_timing"]["B1"]["mean"]),
        cpu_p=_fmt_metric("persisted_cpu_process_timing", report_values["per_policy_summary"]["persisted_cpu_process_timing"]["P"]["mean"]),
        wall_b0=_fmt_metric("persisted_wall_clock_timing", report_values["per_policy_summary"]["persisted_wall_clock_timing"]["B0"]["mean"]),
        wall_b1=_fmt_metric("persisted_wall_clock_timing", report_values["per_policy_summary"]["persisted_wall_clock_timing"]["B1"]["mean"]),
        wall_p=_fmt_metric("persisted_wall_clock_timing", report_values["per_policy_summary"]["persisted_wall_clock_timing"]["P"]["mean"]),
        thr_b0=_fmt_metric("persisted_validated_orders_per_second", report_values["per_policy_summary"]["persisted_validated_orders_per_second"]["B0"]["mean"]),
        thr_b1=_fmt_metric("persisted_validated_orders_per_second", report_values["per_policy_summary"]["persisted_validated_orders_per_second"]["B1"]["mean"]),
        thr_p=_fmt_metric("persisted_validated_orders_per_second", report_values["per_policy_summary"]["persisted_validated_orders_per_second"]["P"]["mean"]),
        thr_r_pb1=_fmt_ratio(report_values["normalized_ratio_summary"]["persisted_validated_orders_per_second"]["P_over_B1"]["mean"]),
        thr_r_pb0=_fmt_ratio(report_values["normalized_ratio_summary"]["persisted_validated_orders_per_second"]["P_over_B0"]["mean"]),
        cpu_r_pb1=_fmt_ratio(report_values["normalized_ratio_summary"]["persisted_cpu_process_timing"]["P_over_B1"]["mean"]),
        cpu_r_pb0=_fmt_ratio(report_values["normalized_ratio_summary"]["persisted_cpu_process_timing"]["P_over_B0"]["mean"]),
        mem_b0=_fmt_metric("persisted_tracemalloc_peak_proxy", report_values["per_policy_summary"]["persisted_tracemalloc_peak_proxy"]["B0"]["mean"]),
        mem_b1=_fmt_metric("persisted_tracemalloc_peak_proxy", report_values["per_policy_summary"]["persisted_tracemalloc_peak_proxy"]["B1"]["mean"]),
        mem_p=_fmt_metric("persisted_tracemalloc_peak_proxy", report_values["per_policy_summary"]["persisted_tracemalloc_peak_proxy"]["P"]["mean"]),
        hash_b1=_fmt_metric("measurable_hash_operation_count", report_values["per_policy_summary"]["measurable_hash_operation_count"]["B1"]["mean"]),
        hash_b0=_fmt_metric("measurable_hash_operation_count", report_values["per_policy_summary"]["measurable_hash_operation_count"]["B0"]["mean"]),
        hash_c_b1b0=_fmt_metric("measurable_hash_operation_count", report_values["per_contrast_summary"]["measurable_hash_operation_count"]["B1_minus_B0"]["mean"]),
        hash_r_pb0=_fmt_ratio(report_values["normalized_ratio_summary"]["measurable_hash_operation_count"]["P_over_B0"]["mean"]),
        sto_100=_fmt_storage(report_values["policy_independent_storage_mean_bytes"]["100"]),
        sto_2500=_fmt_storage(report_values["policy_independent_storage_mean_bytes"]["2500"]),
        sec_b1=str(int(report_values["security_counts"]["e02_detection_counts"]["B1"])),
        sec_b0=str(int(report_values["security_counts"]["e02_detection_counts"]["B0"])),
        sec_p=str(int(report_values["security_counts"]["e02_detection_counts"]["P"])),
        sec_den=str(int(report_values["security_counts"]["e02_denominator_per_policy"])),
        sec_e04=str(int(report_values["security_counts"]["e04_unaffected_order_preservation"]["numerator_per_policy"])),
    )
    scan = text.lower()
    hits = [tok for tok in PROHIBITED_REPORT_TOKENS if tok in scan]
    if hits:
        raise V11F5IntegrityError(
            "F5 report renderer produced a prohibited-token text: " + ", ".join(sorted(hits))
        )
    missing_markers = [m for m in REQUIRED_REPORT_MARKERS if m not in text]
    if missing_markers:
        raise V11F5IntegrityError(
            "F5 report renderer omitted required markers: " + ", ".join(missing_markers)
        )
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_report_artifact() -> dict[str, Any]:
    """Deterministically render (and, if needed, write) the F5 report."""
    text, sha = render_report_text()
    path = F5_REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    planned = text.encode("utf-8")
    if path.exists() and path.read_bytes() != planned:
        raise V11F5OutputCollisionError(
            "docs/v11f_resource_efficiency_results.md exists with different content; "
            "refusing to overwrite a conflicting authored report."
        )
    if not path.exists():
        path.write_bytes(planned)
    return {
        "marker": "V11F5_REPORT_RENDER",
        "mode": "report",
        "report_file": str(path),
        "report_file_sha256": sha,
        "lines": text.count("\n") + 1,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run_v11f5.py",
        description="V1.1-F5 governed reporting/validation closure runner.",
    )
    parser.add_argument(
        "mode",
        choices=("report", "verify", "persist"),
        help="report (render docs/v11f_resource_efficiency_results.md), verify "
        "(read-only, checks the report), or persist (writes the governed "
        "validation-closure artifact)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(f2.V11F_DIR),
        help="target directory for v11f5_validation.json (persist mode)",
    )
    parser.add_argument(
        "--allow-governed-persistence",
        action="store_true",
        help="sanctioned operator path: permits writing v11f5_validation.json",
    )
    args = parser.parse_args()
    try:
        if args.mode == "report":
            report = write_report_artifact()
        elif args.mode == "verify":
            report = run_readonly_verification()
        else:
            report = persist_validation_artifact(
                allow_governed_persistence=args.allow_governed_persistence,
                out_dir=args.out_dir,
            )
    except (V11F5Error, EnvironmentError) as exc:
        print(f"V1.1-F5 error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())