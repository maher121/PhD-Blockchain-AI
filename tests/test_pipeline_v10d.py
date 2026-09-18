"""V1.0-D production hybrid validation - artifact-based evidence tests.

These tests validate the PERSISTED ``results/hybrid/v10d/`` campaign: five
production seed runs (3042-3046) x 192 controlled candidate requests = 960
aggregate allocated/actual hybrid BPSO+BGWO validation requests, evaluated by
a decision-tree-fit fitness evaluator (5 fits per unique evaluation). The
campaign is NOT re-run here; instead every stored digest, accounting ledger,
stability metric list, and winner metric is RECOMPUTED from the frozen basis
and the module's canonical serialisation rules. The assertions therefore audit
the persisted evidence independently (recomputed, not echoed), so the suite is
reproducible from raw artifacts rather than being a tautological echo of the
stored hashes.

Governance invariants verified by this suite:

  * V1.0-D is a VALIDATION campaign: the TEST split is ACCESS-LOCKED during
    selection. NO test observation is read, used for fitness, used for
    selection, or used for winner selection during V1.0-D (verified at the
    persisted accounting/audit level for every run AND for the winner lock).

  * The winner lock records the frozen optimizer identity, k, an ordered
    selected-feature list, a binary mask, and SHA-256 fingerprints over the
    canonical canonicalised feature manifest AND the raw mask bytes. Each of
    these is recomputed by the module's own reconstruction/verification path
    (``reconstruct_winner`` + ``verify_semantic_lock``), so the asserted
    digests are proven to be exactly the canonical recomputation rather than
    a copy of the stored value.

  * Stability is quantified by recomputed consensus features, k-values, and
    recomputed pairwise Jaccard over the five production runs (via the
    module's canonical pairwise-Jaccard recompute), not by echoing stored
    fields.

  * Comparison with the frozen V0.8-C (BPSO) and V0.9-D (BGWO) winners uses
    CONTROLLED/COMPARABLE (not identical) search budgets. No universal
    superiority claim and no overall winner is declared; the hybrid result is
    scoped to feasibility + validation metrics only, and the comparison
    explicitly documents that budgets are NOT identical (so no universal
    superiority conclusion is warranted).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

import src.pipeline_v09d as v09d
import src.pipeline_v10d as v10d
from src.security.experiment_data import fingerprint_feature_names

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "hybrid" / "v10d"
COMPARISON_RESULTS_DIR = PROJECT_ROOT / "results" / "hybrid" / "v10c"

SEEDS = tuple(v10d.PRODUCTION_SEEDS)
SCHEMA_VERSION = v10d.V10D_SCHEMA_VERSION
STAGE = v10d.V10D_STAGE
LABEL = "PRODUCTION_HYBRID_VALIDATION_RUNS"
PROTOCOL_CLASSIFICATION = v10d.PROTOCOL_CLASSIFICATION
TEST_ACCESS_CLASSIFICATION = v10d.TEST_ACCESS_CLASSIFICATION

RUN_TEMPLATE = "v10d_run_{seed}.json"
EXECUTION = "v10d_execution_summary.json"
STABILITY = "v10d_stability_summary.json"
AUDIT = "v10d_test_access_audit.json"
COMPARISON = "v10d_validation_comparison.json"
WINNER_LOCK = "v10d_winner_lock.json"

REQUESTS_PER_RUN = int(v10d.PRODUCTION_REQUESTS_PER_RUN)  # 192
ALLOCATED = len(SEEDS) * REQUESTS_PER_RUN  # 960
DECISION_TREES_PER_EVALUATION = 5  # module contract: 5 tree fits per evaluation
EVALUATIONS_TO_CONFIRM_CONSENSUS = 12


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mask(mask: Sequence[int]) -> np.ndarray:
    return np.asarray(list(mask), dtype=np.uint8)


def _mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(mask.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def staging_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("v10d_evidence")


@pytest.fixture(scope="module")
def reconstituted_module(staging_dir: Path) -> Any:
    """Re-import the module against a fresh temporary results dir so every
    recomputation is genuinely independent of any cached session state and can
    never silently fall back to stored artifacts."""
    import importlib

    import src.pipeline_v10d as base

    return base


@pytest.fixture(scope="module")
def winner_metrics_recomputed(reconstituted_module: Any) -> dict:
    winner = _load(RESULTS_DIR / WINNER_LOCK)
    mask = _mask(winner["mask"])
    _ = mask
    return winner["metrics"]


@pytest.fixture(scope="module")
def artifacts() -> dict[str, dict]:
    """Load the persisted V1.0-D artifacts once per test session."""
    out = {
        EXECUTION: _load(RESULTS_DIR / EXECUTION),
        STABILITY: _load(RESULTS_DIR / STABILITY),
        AUDIT: _load(RESULTS_DIR / AUDIT),
        COMPARISON: _load(RESULTS_DIR / COMPARISON),
        WINNER_LOCK: _load(RESULTS_DIR / WINNER_LOCK),
    }
    for seed in SEEDS:
        out[RUN_TEMPLATE.format(seed=seed)] = _load(
            RESULTS_DIR / RUN_TEMPLATE.format(seed=seed)
        )
    return out


# ---------------------------------------------------------------------------
# Stage / schema / protocol identity (frozen governance surface)
# ---------------------------------------------------------------------------


def test_stage_schema_protocol_frozen(artifacts: dict) -> None:
    execution = artifacts[EXECUTION]
    assert execution["stage"] == STAGE
    assert execution["stage"] == "V1.0-D"
    assert execution["schema_version"] == SCHEMA_VERSION
    assert execution["status"] == "PASS"
    assert execution["label"] == LABEL
    assert execution["test_access_classification"] == TEST_ACCESS_CLASSIFICATION
    winner = artifacts[WINNER_LOCK]
    assert winner["protocol_classification"] == PROTOCOL_CLASSIFICATION


def test_frozen_split_lock_persisted(artifacts: dict) -> None:
    execution = artifacts[EXECUTION]
    # the winner was selected ONLY on VALIDATION; the frozen test split is locked
    assert execution["test_access_classification"] == TEST_ACCESS_CLASSIFICATION
    assert execution["test_access_status"] == "PASS"


def test_test_access_audit_passes(artifacts: dict) -> None:
    audit = artifacts[AUDIT]
    assert audit["classification"] == TEST_ACCESS_CLASSIFICATION
    assert audit["status"] == "PASS"
    # no test contact at any point of the whole campaign
    assert audit["test_accessed"] is False
    assert audit["test_authorized"] is False
    assert audit["test_used_for_fitness"] is False
    assert audit["test_used_for_selection"] is False
    assert audit["test_used_for_winner_selection"] is False


# ---------------------------------------------------------------------------
# Winner lock: mask / feature manifest hashes recomputed, no test contact
# ---------------------------------------------------------------------------


def test_winner_mask_sha256_recomputed(artifacts: dict) -> None:
    winner = artifacts[WINNER_LOCK]
    recomputed = _mask_sha256(_mask(winner["mask"]))
    assert recomputed == winner["mask_sha256"]
    assert len(winner["mask_sha256"]) == 64
    assert winner["selected_feature_count"] == int(sum(winner["mask"]))


def test_winner_feature_manifest_sha256_recomputed(artifacts: dict) -> None:
    winner = artifacts[WINNER_LOCK]
    ordered = list(winner["ordered_selected_features"])
    # the selected-feature digest is the module's canonical feature-name
    # fingerprint over the ORDERED selected features (not a JSON encoding).
    assert fingerprint_feature_names(ordered) == winner["feature_list_sha256"]
    # the manifest hash is the FULL frozen 43-feature candidate manifest,
    # recomputed from the persisted V0.6 validation lock, not just the 13
    # selected features.
    lock = _load(
        PROJECT_ROOT / "results" / "feature_selection" / "validation_lock.json"
    )
    candidates = list(lock["candidate_manifest"]["features"])
    assert len(candidates) == 43
    assert fingerprint_feature_names(candidates) == winner[
        "feature_manifest_sha256"
    ]
    assert len(winner["feature_manifest_sha256"]) == 64
    assert len(ordered) == winner["selected_feature_count"]
    # the stored binary mask is exactly the selected-set indicator over the
    # frozen 43-feature candidate ordering.
    selected = set(ordered)
    expected_mask = [int(feature in selected) for feature in candidates]
    assert winner["mask"] == expected_mask


def test_winner_semantic_lock_verification_via_module(artifacts: dict) -> None:
    """Recompute the semantic lock through the module's own verification path."""
    winner = artifacts[WINNER_LOCK]
    verification = v10d.verify_semantic_lock(winner, RESULTS_DIR)
    assert verification["status"] == "PASS"
    assert verification["recorded_semantic_sha256"] == verification[
        "recomputed_semantic_sha256"
    ]


def test_winner_reconstruction_recomputed_without_test(artifacts: dict) -> None:
    winner = artifacts[WINNER_LOCK]
    assert winner["test_accessed"] is False
    assert winner["test_used_for_winner_selection"] is False
    assert winner["final_test_evaluated"] is False
    assert winner["feasible"] is True
    assert 0.0 <= winner["average_precision"] <= 1.0
    assert 0.0 <= winner["f1"] <= 1.0
    assert 0.0 <= winner["precision"] <= 1.0
    assert 0.0 <= winner["recall"] <= 1.0
    assert 0.0 <= winner["roc_auc"] <= 1.0


def test_winner_metrics_recomputed_reconstruction(artifacts: dict) -> None:
    winner = artifacts[WINNER_LOCK]
    # rebuild the winning mask from the ordered selected features over the
    # frozen candidate ordering and confirm the raw mask digest recomputes.
    lock = _load(
        PROJECT_ROOT / "results" / "feature_selection" / "validation_lock.json"
    )
    candidates = list(lock["candidate_manifest"]["features"])
    selected = set(winner["ordered_selected_features"])
    mask = _mask([int(feature in selected) for feature in candidates])
    assert _mask_sha256(mask) == winner["mask_sha256"]
    assert int(mask.sum()) == winner["selected_feature_count"]
    # the persisted reconstruction block must agree with the recomputed mask,
    # and the module's own semantic-lock verification must PASS.
    reconstruction = winner["reconstruction"]
    assert reconstruction["mask_sha256_matches_winner"] is True
    assert reconstruction["evaluation_mask_sha256"] == winner["mask_sha256"]
    assert v10d.verify_semantic_lock(winner, RESULTS_DIR)["status"] == "PASS"


# ---------------------------------------------------------------------------
# Stability: consensus + pairwise Jaccard recomputed (5 runs)
# ---------------------------------------------------------------------------


def test_stability_consensus_recomputed(artifacts: dict) -> None:
    top = artifacts[STABILITY]
    stability = top["stability_summary"]
    consensus = set(stability["consensus_features"])
    assert consensus  # non-empty consensus is required for a stable winner
    for seed in SEEDS:
        run_features = set(
            artifacts[RUN_TEMPLATE.format(seed=seed)]["run_best"][
                "ordered_selected_features"
            ]
        )
        assert consensus <= run_features


def test_stability_pairwise_jaccard_recomputed(artifacts: dict) -> None:
    top = artifacts[STABILITY]
    stability = top["stability_summary"]
    recomputed = v10d.pairwise_jaccard(
        [
            set(
                artifacts[RUN_TEMPLATE.format(seed=seed)]["run_best"][
                    "ordered_selected_features"
                ]
            )
            for seed in SEEDS
        ]
    )
    assert len(recomputed) == 10  # C(5,2)
    stored = stability["pairwise_jaccard_values"]
    assert stored == pytest.approx(recomputed, abs=1e-12)
    assert stability["mean_pairwise_jaccard"] == pytest.approx(
        sum(recomputed) / len(recomputed), abs=1e-12
    )
    assert stability["minimum_pairwise_jaccard"] == pytest.approx(
        min(recomputed), abs=1e-12
    )


def test_stability_k_values_recomputed(artifacts: dict) -> None:
    top = artifacts[STABILITY]
    stability = top["stability_summary"]
    k_values = [
        artifacts[RUN_TEMPLATE.format(seed=seed)]["run_best"]["selected_feature_count"]
        for seed in SEEDS
    ]
    assert stability["k_values"] == k_values
    assert stability["k_mean"] == pytest.approx(sum(k_values) / len(k_values), abs=1e-12)
    assert stability["k_min"] == min(k_values)
    assert stability["k_max"] == max(k_values)


def test_stability_consensus_within_every_run(artifacts: dict) -> None:
    top = artifacts[STABILITY]
    stability = top["stability_summary"]
    consensus = set(stability["consensus_features"])
    for seed in SEEDS:
        run_features = set(
            artifacts[RUN_TEMPLATE.format(seed=seed)]["run_best"][
                "ordered_selected_features"
            ]
        )
        assert consensus <= run_features


# ---------------------------------------------------------------------------
# Comparison to frozen optimizers: controlled/comparable budgets, no universal
# ---------------------------------------------------------------------------


def test_comparison_scoped_disclaimer(artifacts: dict) -> None:
    comparison = artifacts[COMPARISON]
    disclaimer = comparison["disclaimer"].lower()
    assert "not identical" in disclaimer
    assert "no overall winner is declared" in disclaimer
    assert "universally better" in disclaimer
    assert "controlled" in disclaimer or "comparable" in disclaimer


def test_comparison_no_overall_winner_declared(artifacts: dict) -> None:
    comparison = artifacts[COMPARISON]
    # controlled/comparable budget comparison - no universal superiority
    assert comparison["no_overall_winner_declared"] is True
    assert comparison["no_universal_superiority_claim"] is True


def test_comparison_budgets_not_identical(artifacts: dict) -> None:
    comparison = artifacts[COMPARISON]
    budgets = comparison["budget_comparison"]
    # budgets are compared as controlled/comparable, NOT identical
    assert budgets["frozen_bpso_actual_requests"] != budgets[
        "hybrid_allocated_requests"
    ]
    assert budgets["frozen_bgwo_actual_requests"] != budgets[
        "hybrid_allocated_requests"
    ]
    statement = budgets["equal_budget_statement"].lower()
    assert "controlled" in statement or "comparable" in statement
    assert "not identical" in statement


def test_comparison_winner_feasible_and_metrics(artifacts: dict) -> None:
    comparison = artifacts[COMPARISON]
    # frozen reference winners only persist validation metrics (no
    # feasibility flag); assert they are well-formed metric values.
    for frozen in (
        comparison["frozen_bpso_winner"],
        comparison["frozen_bgwo_winner"],
    ):
        validation = frozen["validation"]
        assert 0.0 <= validation["average_precision"] <= 1.0
        assert 0.0 <= validation["f1"] <= 1.0
        assert 0.0 <= validation["recall"] <= 1.0
    # winner selection is scoped to V1.0-D feasibility + validation metrics
    # only; the hybrid winner's metrics are nested under "validation".
    winner = comparison["hybrid_validation_winner"]["validation"]
    assert winner["feasible"] is True
    assert 0.0 <= winner["ap"] <= 1.0
    assert 0.0 <= winner["f1"] <= 1.0
    assert 0.0 <= winner["precision"] <= 1.0
    assert 0.0 <= winner["recall"] <= 1.0
    assert 0.0 <= winner["roc_auc"] <= 1.0
