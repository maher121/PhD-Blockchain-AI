"""V1.1-F5 governed reporting/validation-closure tests.

Read-only. These tests verify that the F5 report, the persisted validation-closure
artifact, and every frozen upstream binding are mutually consistent, that the report
respects the governed language (no energy claims, no winner/ranking/composite claims,
no inferential wording), and that the F4/V1.1-E locks are byte-immutable. No
experiment, no measurement, and no artifact write is performed here.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

import src.pipeline_v11f as f2
from scripts import run_v11f5 as f5
from scripts.run_v11f5 import (
    ANALYSIS_RESULTS_NAME,
    EXECUTION_MANIFEST_NAME,
    F5_REPORT_PATH,
    FINAL_LOCK_FORBIDDEN_NAMES,
    PROHIBITED_REPORT_TOKENS,
    REQUIRED_REPORT_MARKERS,
    RESULT_LOCK_NAME,
    V11F5_VALIDATION_NAME,
    VERIFY_REPORT_MARKER,
    extract_report_values,
    report_value_strings,
)

REPO_ROOT = Path(f2.PROJECT_ROOT)
V11E_DIR = Path(f2.V11E_DIR)
V11F_DIR = Path(f2.V11F_DIR)

# --------------------------------------------------------------------------- #
# shared fixtures (all read-only)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def analysis():
    return json.loads((V11F_DIR / ANALYSIS_RESULTS_NAME).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest():
    return json.loads((V11F_DIR / EXECUTION_MANIFEST_NAME).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def result_lock():
    return json.loads((V11F_DIR / RESULT_LOCK_NAME).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def validation_artifact():
    return json.loads((V11F_DIR / V11F5_VALIDATION_NAME).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report_text() -> str:
    return F5_REPORT_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def verify_report():
    return f5.run_readonly_verification()


# --------------------------------------------------------------------------- #
# frozen pins stay pinned on disk
# --------------------------------------------------------------------------- #


def test_f1_config_and_protocol_document_pins_match_disk():
    assert f2.sha256_file(f2.F1_CONFIG_PATH) == f5.EXPECTED_F1_CONFIG_SHA256
    assert f2.sha256_file(f2.F1_PROTOCOL_PATH) == f5.EXPECTED_F1_PROTOCOL_DOC_SHA256


def test_f2_implementation_blob_pin_matches_disk():
    assert f2.sha256_file(REPO_ROOT / "src" / "pipeline_v11f.py") == (
        f5.EXPECTED_F2_IMPLEMENTATION_FILE_SHA256
    )


def test_f4_artifacts_recompute_to_pins():
    assert f2.sha256_file(V11F_DIR / ANALYSIS_RESULTS_NAME) == (
        f5.EXPECTED_F4_ANALYSIS_RESULTS_SHA256
    )
    assert f2.sha256_file(V11F_DIR / EXECUTION_MANIFEST_NAME) == (
        f5.EXPECTED_F4_EXECUTION_MANIFEST_SHA256
    )


def test_f4_result_lock_semantic_recomputes(result_lock):
    assert f2.sha256_of_canonical(result_lock["semantic_payload"]) == (
        f5.EXPECTED_F4_RESULT_LOCK_SEMANTIC_SHA256
    )
    assert result_lock["semantic_result_lock_sha256"] == (
        f5.EXPECTED_F4_RESULT_LOCK_SEMANTIC_SHA256
    )


def test_v11e_result_lock_pins_recompute():
    lock = json.loads((V11E_DIR / f2.V11E_LOCK_NAME).read_text(encoding="utf-8"))
    assert f2.sha256_file(V11E_DIR / f2.V11E_LOCK_NAME) == (
        f5.EXPECTED_V11E_RESULT_LOCK_FILE_SHA256
    )
    assert f2.sha256_of_canonical(lock["semantic_payload"]) == (
        f5.EXPECTED_V11E_RESULT_LOCK_SEMANTIC_SHA256
    )


def test_v11e_protocol_lock_semantic_recomputes():
    lock = json.loads(
        (V11E_DIR / "v11e_experiment_protocol_lock.json").read_text(encoding="utf-8")
    )
    assert lock["semantic_result_lock_sha256"] == (
        f5.EXPECTED_V11E_PROTOCOL_LOCK_SEMANTIC_SHA256
    )


# --------------------------------------------------------------------------- #
# frozen V1.1-E / F4 records remained byte-immutable through F5 authoring
# --------------------------------------------------------------------------- #


def test_upstream_artifacts_unchanged():
    assert f2.sha256_file(V11E_DIR / f2.V11E_LOCK_NAME) == (
        f5.EXPECTED_V11E_RESULT_LOCK_FILE_SHA256
    )
    assert f2.sha256_file(V11F_DIR / ANALYSIS_RESULTS_NAME) == (
        f5.EXPECTED_F4_ANALYSIS_RESULTS_SHA256
    )
    assert f2.sha256_file(V11F_DIR / EXECUTION_MANIFEST_NAME) == (
        f5.EXPECTED_F4_EXECUTION_MANIFEST_SHA256
    )


def test_f4_result_lock_preserved_and_no_final_lock():
    assert not any((V11F_DIR / n).exists() for n in FINAL_LOCK_FORBIDDEN_NAMES)
    lock = json.loads((V11F_DIR / RESULT_LOCK_NAME).read_text(encoding="utf-8"))
    assert lock["semantic_result_lock_sha256"] == (
        f5.EXPECTED_F4_RESULT_LOCK_SEMANTIC_SHA256
    )


# --------------------------------------------------------------------------- #
# governance counters and analysis-mode facts in the frozen payloads
# --------------------------------------------------------------------------- #


def test_governance_counters_zero(analysis, manifest, result_lock):
    for source in (
        analysis["governance_checks"],
        manifest["governance_counters"],
        result_lock["semantic_payload"]["governance_counters"],
    ):
        for key in f5.ZERO_COUNTER_KEYS:
            assert int(source.get(key, 0)) == 0


def test_direct_energy_unavailable_everywhere(analysis, manifest, result_lock):
    assert analysis["direct_energy_status"] == "DIRECT_ENERGY_UNAVAILABLE"
    assert manifest["direct_energy_status"] == "DIRECT_ENERGY_UNAVAILABLE"
    assert result_lock["semantic_payload"]["direct_energy_status"] == (
        "DIRECT_ENERGY_UNAVAILABLE"
    )


def test_no_winner_ranking_composite_in_frozen_lock(result_lock):
    sp = result_lock["semantic_payload"]
    assert sp["overall_winner_present"] is False
    assert sp["ranking_fields_present"] is False
    assert sp["composite_score_present"] is False


def test_analysis_mode_descriptive(analysis, result_lock):
    assert analysis["analysis_mode"] == "DESCRIPTIVE_ONLY"
    assert result_lock["semantic_payload"]["analysis_mode"] == "DESCRIPTIVE_ONLY"


# --------------------------------------------------------------------------- #
# the persisted validation-closure artifact is self-consistent
# --------------------------------------------------------------------------- #


def test_validation_artifact_gates_all_true(validation_artifact):
    assert validation_artifact["validation_gates_all"] is True
    gates = validation_artifact["validation_gates"]
    assert all(gates[name] is True for name in gates)


def test_required_gate_keys_present(validation_artifact):
    gates = set(validation_artifact["validation_gates"].keys())
    required = {
        "f1_protocol_binding",
        "f2_semantic_binding",
        "f4_artifact_hashes",
        "f4_result_lock_semantic_hash",
        "v11e_upstream_lock",
        "governance_counters_zero",
        "direct_energy_unavailable",
        "e07_proxy_terminology",
        "e08_policy_independence",
        "e10_derived_only",
        "e03_non_independence",
        "no_ranking_winner_composite",
        "no_inferential_claims",
        "report_values_reconcile",
        "no_test_access",
        "no_ai_fit_inference",
        "no_new_measurement",
        "required_report_markers",
    }
    assert required.issubset(gates)


def test_validation_artifact_semantic_hash_deterministic(validation_artifact):
    recomputed = f2.sha256_of_canonical(validation_artifact["semantic_payload"])
    assert recomputed == validation_artifact["semantic_validation_sha256"]


def test_validation_artifact_payload_matches_frozen_pins(validation_artifact):
    sp = validation_artifact["semantic_payload"]
    pins = {
        "f1_protocol_lock_semantic_sha256": f5.EXPECTED_F1_PROTOCOL_LOCK_SEMANTIC_SHA256,
        "f1_config_sha256": f5.EXPECTED_F1_CONFIG_SHA256,
        "f1_protocol_document_sha256": f5.EXPECTED_F1_PROTOCOL_DOC_SHA256,
        "f2_implementation_commit": f5.EXPECTED_F2_IMPLEMENTATION_COMMIT,
        "f2_implementation_file_sha256": f5.EXPECTED_F2_IMPLEMENTATION_FILE_SHA256,
        "f2_semantic_analysis_sha256": f5.EXPECTED_F2_SEMANTIC_ANALYSIS_SHA256,
        "f4_analysis_results_sha256": f5.EXPECTED_F4_ANALYSIS_RESULTS_SHA256,
        "f4_execution_manifest_sha256": f5.EXPECTED_F4_EXECUTION_MANIFEST_SHA256,
        "f4_result_lock_semantic_sha256": f5.EXPECTED_F4_RESULT_LOCK_SEMANTIC_SHA256,
        "upstream_v11e_result_lock_semantic_sha256": (
            f5.EXPECTED_V11E_RESULT_LOCK_SEMANTIC_SHA256
        ),
        "upstream_v11e_result_lock_file_sha256": (
            f5.EXPECTED_V11E_RESULT_LOCK_FILE_SHA256
        ),
        "upstream_v11e_protocol_lock_semantic_sha256": (
            f5.EXPECTED_V11E_PROTOCOL_LOCK_SEMANTIC_SHA256
        ),
    }
    for key, pin in pins.items():
        assert sp[key] == pin
    assert sp["final_stage_lock_created"] is False
    assert sp["e07_label"] == "COMPUTATIONAL_MEMORY_PROXY"
    assert sp["e08_status"] == "POLICY_INDEPENDENT"
    assert sp["e10_derived_only"] is True


def test_validation_artifact_report_values_reconcile(
    analysis, validation_artifact
):
    expected = extract_report_values(analysis)
    assert validation_artifact["report_values"] == expected


def test_validation_artifact_report_sha_matches_disk(validation_artifact):
    actual = hashlib.sha256(F5_REPORT_PATH.read_bytes()).hexdigest()
    assert actual == validation_artifact["report_file_sha256"]


def test_verify_report_runner_passes_all_gates(verify_report):
    assert verify_report["marker"] == f5.VERIFY_REPORT_MARKER
    assert verify_report["validation_gates_all"] is True


# --------------------------------------------------------------------------- #
# the F5 report itself: numbers, markers, and governed language
# --------------------------------------------------------------------------- #


def test_report_contains_all_headline_numbers(analysis, report_text):
    report_values = extract_report_values(analysis)
    missing = [s for s in report_value_strings(report_values) if s not in report_text]
    assert not missing


def test_report_required_markers_present(report_text):
    missing = [m for m in REQUIRED_REPORT_MARKERS if m not in report_text]
    assert not missing


def test_report_forbidden_tokens_absent(report_text):
    lowered = report_text.lower()
    hits = [tok for tok in PROHIBITED_REPORT_TOKENS if tok in lowered]
    assert not hits


def test_report_energy_claim_free(report_text):
    for phrase in (
        "DIRECT_ENERGY_UNAVAILABLE",
        "COMPUTATIONAL_MEMORY_PROXY",
        "POLICY_INDEPENDENT",
        "DERIVED_ONLY",
    ):
        assert phrase in report_text
    assert "no single policy" in report_text.lower()


def test_report_states_required_limitations(report_text):
    lower = report_text.lower()
    required_phrases = (
        "no direct physical-energy measurement exists",
        "not rss/system memory",
        "not a policy comparison",
        "no independent measurement campaign",
        "mirrored",
        "not independent",
        "low-band evidence",
        "high-band evidence is sparse",
        "not observed in the workload-100 condition",
        "nested within each seed",
        "only three fixed seeds",
        "descriptive_only",
        "no inferential",
    )
    missing = [p for p in required_phrases if p not in lower]
    assert not missing


def test_report_declares_only_descriptive_scope(report_text):
    assert "DESCRIPTIVE_ONLY" in report_text
    assert "no interval or resampling statement" in report_text.lower()
    assert "15 cells are not 15 independent experiments" in report_text.lower()


def test_report_has_thirteen_sections_in_order(report_text):
    headings = [
        "## 1. Provenance and scope",
        "## 2. Governed evidence base and read-only boundary",
        "## 3. Metric families and proxy status",
        "## 4. Per-policy descriptive summaries",
        "## 5. Paired descriptive contrasts",
        "## 6. Normalized ratio summaries",
        "## 7. Policy-independent storage",
        "## 8. Security context",
        "## 9. Population composition and risk-band limitations",
        "## 10. Replication, nesting, and inferential scope",
        "## 11. Direct-energy status and proxy constraints",
        "## 12. Endpoint-specific conclusions and policy stance",
        "## 13. Reproducibility, data policy, and follow-on governance",
    ]
    idx = -1
    for heading in headings:
        found = report_text.find(heading)
        assert found > idx, f"section out of order or missing: {heading}"
        idx = found


def test_report_reproduction_is_deterministic(analysis):
    rendered, _ = f5.render_report_text()
    assert rendered == F5_REPORT_PATH.read_text(encoding="utf-8")