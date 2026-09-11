"""Focused synthetic tests for artifact-only V0.6-E configuration locking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline_v06e import (
    AUDIT_FILE_NAME,
    LOCK_FILE_NAME,
    REPORT_FILE_NAME,
    SIDECAR_FILE_NAME,
    ValidationLockError,
    build_validation_lock,
    create_validation_lock,
    fingerprint_feature_names,
    semantic_payload_sha256,
    verify_validation_lock,
    _validation_best_key,
)


SEEDS = (42, 43, 44, 45, 46)
FEATURES = [f"feature_{index:02d}" for index in range(43)]
CONFIGURATIONS = (
    ("none_natural", "none", "none", "natural", None, 43, 0.800, 0.700, 0.600),
    ("variance_threshold_natural", "variance_threshold", "unsupervised", "natural", None, 43, 0.790, 0.690, 0.590),
    ("pairwise_correlation_filter_natural", "pairwise_correlation_filter", "unsupervised", "natural", None, 42, 0.740, 0.650, 0.540),
    ("correlation_redundancy_ranking_k32", "correlation_redundancy_ranking", "unsupervised", "fixed_k", 32, 32, 0.790, 0.690, 0.590),
    ("correlation_redundancy_ranking_k22", "correlation_redundancy_ranking", "unsupervised", "fixed_k", 22, 22, 0.780, 0.680, 0.570),
    ("correlation_redundancy_ranking_k11", "correlation_redundancy_ranking", "unsupervised", "fixed_k", 11, 11, 0.700, 0.620, 0.510),
    ("mutual_information_select_k_best_k32", "mutual_information_select_k_best", "supervised", "fixed_k", 32, 32, 0.790, 0.690, 0.590),
    ("mutual_information_select_k_best_k22", "mutual_information_select_k_best", "supervised", "fixed_k", 22, 22, 0.770, 0.680, 0.570),
    ("mutual_information_select_k_best_k11", "mutual_information_select_k_best", "supervised", "fixed_k", 11, 11, 0.730, 0.610, 0.500),
    ("anova_f_select_k_best_k32", "anova_f_select_k_best", "supervised", "fixed_k", 32, 32, 0.760, 0.665, 0.550),
    ("anova_f_select_k_best_k22", "anova_f_select_k_best", "supervised", "fixed_k", 22, 22, 0.730, 0.610, 0.500),
    ("anova_f_select_k_best_k11", "anova_f_select_k_best", "supervised", "fixed_k", 11, 11, 0.690, 0.590, 0.480),
)


def test_create_lock_writes_and_verifies_all_outputs(tmp_path: Path) -> None:
    results = _artifact_dir(tmp_path)
    outcome = create_validation_lock(results, created_at="2026-01-01T00:00:00+00:00")

    assert outcome["status"] == "PASS"
    assert {LOCK_FILE_NAME, SIDECAR_FILE_NAME, REPORT_FILE_NAME, AUDIT_FILE_NAME} <= {
        path.name for path in results.iterdir()
    }
    assert verify_validation_lock(results / LOCK_FILE_NAME)["status"] == "PASS"
    audit = _read_json(results / AUDIT_FILE_NAME)
    assert audit["status"] == "PASS"
    assert audit["test_accessed"] is False


def test_semantic_hash_excludes_created_at_and_is_deterministic(tmp_path: Path) -> None:
    results = _artifact_dir(tmp_path)
    first = build_validation_lock(results, created_at="2026-01-01T00:00:00+00:00")
    second = build_validation_lock(results, created_at="2027-02-02T00:00:00+00:00")

    assert first["created_at"] != second["created_at"]
    assert first["semantic_payload_sha256"] == second["semantic_payload_sha256"]


def test_recomputes_dynamic_roles_and_smallest_preserving(tmp_path: Path) -> None:
    lock = build_validation_lock(_artifact_dir(tmp_path))

    assert lock["per_method_choices"]["correlation_redundancy_ranking"] == (
        "correlation_redundancy_ranking_k22"
    )
    assert lock["per_method_choices"]["mutual_information_select_k_best"] == (
        "mutual_information_select_k_best_k22"
    )
    assert lock["roles"]["best_unsupervised"]["configuration_id"] == (
        "variance_threshold_natural"
    )
    assert lock["roles"]["best_supervised"]["configuration_id"] == (
        "mutual_information_select_k_best_k22"
    )
    assert lock["roles"]["smallest_preserving"]["configuration_id"] == (
        "correlation_redundancy_ranking_k22"
    )


def test_validation_best_tie_break_is_ap_then_f1_then_fewer_features() -> None:
    records = [
        {"configuration_id": "fewer", "means": {"average_precision": 0.8, "f1": 0.7}, "actual_feature_count": 1},
        {"configuration_id": "higher_f1", "means": {"average_precision": 0.8, "f1": 0.71}, "actual_feature_count": 3},
        {"configuration_id": "higher_ap", "means": {"average_precision": 0.81, "f1": 0.1}, "actual_feature_count": 4},
    ]
    assert sorted(records, key=_validation_best_key)[0]["configuration_id"] == "higher_ap"
    assert sorted(records[:2], key=_validation_best_key)[0]["configuration_id"] == "higher_f1"
    records[1]["means"]["f1"] = 0.7
    assert sorted(records[:2], key=_validation_best_key)[0]["configuration_id"] == "fewer"


def test_duplicate_roles_may_reference_same_configuration(tmp_path: Path) -> None:
    results = _artifact_dir(tmp_path)
    core = pd.read_csv(results / "core_runs.csv")
    candidate = core["configuration_id"].eq("mutual_information_select_k_best_k11")
    core.loc[candidate, ["average_precision", "f1", "recall"]] = [0.79, 0.69, 0.59]
    _rewrite_core_and_selected(results, core)

    lock = build_validation_lock(results)

    assert lock["roles"]["best_supervised"]["configuration_id"] == (
        "mutual_information_select_k_best_k11"
    )
    assert lock["roles"]["smallest_preserving"]["configuration_id"] == (
        "mutual_information_select_k_best_k11"
    )
    assert lock["locked_configurations"]["mutual_information_select_k_best_k11"]["locked_roles"] == [
        "best_supervised",
        "smallest_preserving",
    ]


def test_lock_exposes_required_role_schema_and_five_validation_metrics(tmp_path: Path) -> None:
    lock = build_validation_lock(_artifact_dir(tmp_path))
    role = lock["roles"]["best_supervised"]

    assert {
        "role", "selector_id", "selector_method", "selector_mode",
        "selector_parameters", "requested_K", "actual_feature_count",
        "selected_features", "selected_feature_fingerprint", "validation_metrics",
        "paired_changes_vs_baseline", "feature_reduction_percentage",
        "preservation_status", "resource_summary", "stability_summary",
        "reason_for_lock",
    } <= set(role)
    assert set(role["validation_metrics"]) == {
        "mean_average_precision", "mean_f1", "mean_recall", "mean_precision",
        "mean_roc_auc",
    }


def test_locks_seed_specific_supervised_sets_and_model_state_hashes(tmp_path: Path) -> None:
    lock = build_validation_lock(_artifact_dir(tmp_path))
    role = lock["roles"]["best_supervised"]

    assert role["selected_features"]["42"] != role["selected_features"]["43"]
    assert set(role["selected_features"]) == {str(seed) for seed in SEEDS}
    assert all(len(value["model_state_sha256"]) == 64 for value in role["model_artifacts"].values())


def test_no_preserving_candidate_uses_validation_best_and_empty_smallest_role(
    tmp_path: Path,
) -> None:
    results = _artifact_dir(tmp_path)
    core = pd.read_csv(results / "core_runs.csv")
    candidates = core["selector_id"].ne("none")
    core.loc[candidates, ["average_precision", "f1", "recall"]] = [0.1, 0.1, 0.1]
    _rewrite_core_and_selected(results, core)

    lock = build_validation_lock(results)

    assert lock["evaluated_configurations"]["correlation_redundancy_ranking_k11"]["selection_status"] == "not_preserving"
    assert lock["per_method_choices"]["correlation_redundancy_ranking"] == (
        "correlation_redundancy_ranking_k11"
    )
    assert lock["roles"]["smallest_preserving"]["configuration_id"] is None
    assert lock["roles"]["smallest_preserving"]["preservation_status"] == "not_available"


def test_hashes_available_referenced_model_file(tmp_path: Path) -> None:
    results = _artifact_dir(tmp_path)
    model = tmp_path / "shared.joblib"
    model.write_bytes(b"stored-model")
    core = pd.read_csv(results / "core_runs.csv")
    core["model_artifact_path"] = str(model)
    core["serialized_model_bytes"] = model.stat().st_size
    core.to_csv(results / "core_runs.csv", index=False)
    _rehash_metadata(results)

    lock = build_validation_lock(results)

    expected = hashlib.sha256(b"stored-model").hexdigest()
    assert any(
        source["kind"] == "referenced_model_artifact" and source["sha256"] == expected
        for source in lock["source_artifacts"]
    )
    assert lock["roles"]["full_baseline"]["model_artifacts"]["42"]["file_sha256"] == expected


def test_rejects_non_sixty_success_rows_before_writes(tmp_path: Path) -> None:
    results = _artifact_dir(tmp_path)
    core = pd.read_csv(results / "core_runs.csv").iloc[:-1]
    core.to_csv(results / "core_runs.csv", index=False)
    _rehash_metadata(results)

    with pytest.raises(ValidationLockError, match="60 successful"):
        create_validation_lock(results)

    assert not (results / LOCK_FILE_NAME).exists()
    assert not (results / REPORT_FILE_NAME).exists()


@pytest.mark.parametrize("column,value", [("evaluation_split", "test"), ("training_split", "test")])
def test_rejects_non_validation_protocol_splits(
    tmp_path: Path, column: str, value: str
) -> None:
    results = _artifact_dir(tmp_path)
    core = pd.read_csv(results / "core_runs.csv")
    core.loc[0, column] = value
    _rewrite_core_and_selected(results, core)

    with pytest.raises(ValidationLockError, match="split"):
        build_validation_lock(results)


def test_rejects_wrong_exact_seed_set(tmp_path: Path) -> None:
    results = _artifact_dir(tmp_path)
    core = pd.read_csv(results / "core_runs.csv")
    core.loc[core["seed"].eq(46), "seed"] = 47
    _rewrite_core_and_selected(results, core)

    with pytest.raises(ValidationLockError, match="Seeds must be exactly"):
        build_validation_lock(results)


def test_rejects_configuration_matrix_mismatch(tmp_path: Path) -> None:
    results = _artifact_dir(tmp_path)
    core = pd.read_csv(results / "core_runs.csv")
    core.loc[
        (core["seed"].eq(42))
        & (core["configuration_id"].eq("correlation_redundancy_ranking_k11")),
        "configuration_id",
    ] = "other"
    _rewrite_core_and_selected(results, core)

    with pytest.raises(ValidationLockError, match="same 12 unique configurations"):
        build_validation_lock(results)


def test_rejects_missing_per_seed_baseline(tmp_path: Path) -> None:
    results = _artifact_dir(tmp_path)
    core = pd.read_csv(results / "core_runs.csv")
    index = core.index[(core["seed"].eq(42)) & (core["selector_id"].eq("none"))][0]
    core.loc[index, "selector_id"] = "not_baseline"
    _rewrite_core_and_selected(results, core)

    with pytest.raises(ValidationLockError, match="one full baseline"):
        build_validation_lock(results)


@pytest.mark.parametrize("column,value", [("attack_rate", 0.2), ("prediction_threshold", 0.7)])
def test_rejects_attack_model_or_threshold_drift(
    tmp_path: Path, column: str, value: float
) -> None:
    results = _artifact_dir(tmp_path)
    core = pd.read_csv(results / "core_runs.csv")
    core.loc[0, column] = value
    _rewrite_core_and_selected(results, core)

    with pytest.raises(ValidationLockError, match="must be invariant"):
        build_validation_lock(results)


def test_rejects_exact_pr_auc_column(tmp_path: Path) -> None:
    results = _artifact_dir(tmp_path)
    core = pd.read_csv(results / "core_runs.csv")
    core["pr_auc"] = 0.5
    _rewrite_core_and_selected(results, core)

    with pytest.raises(ValidationLockError, match="exact column pr_auc"):
        build_validation_lock(results)


def test_rejects_recursive_exact_pr_auc_key(tmp_path: Path) -> None:
    results = _artifact_dir(tmp_path)
    metadata = _read_json(results / "run_metadata.json")
    metadata["nested"] = {"deeper": [{"pr_auc": 0.5}]}
    _write_json(results / "run_metadata.json", metadata)

    with pytest.raises(ValidationLockError, match="exact key pr_auc"):
        build_validation_lock(results)


@pytest.mark.parametrize("artifact", ["leakage_audit.json", "run_metadata.json"])
def test_rejects_source_d_test_attestation(tmp_path: Path, artifact: str) -> None:
    results = _artifact_dir(tmp_path)
    payload = _read_json(results / artifact)
    payload["test_accessed"] = True
    _write_json(results / artifact, payload)
    if artifact == "leakage_audit.json":
        _rehash_metadata(results)

    with pytest.raises(ValidationLockError, match="test_accessed=false"):
        build_validation_lock(results)


def test_rejects_source_d_audit_failure(tmp_path: Path) -> None:
    results = _artifact_dir(tmp_path)
    audit = _read_json(results / "leakage_audit.json")
    audit["status"] = "FAIL"
    _write_json(results / "leakage_audit.json", audit)
    _rehash_metadata(results)

    with pytest.raises(ValidationLockError, match="PASS status"):
        build_validation_lock(results)


def test_rejects_metadata_output_hash_claim_mismatch(tmp_path: Path) -> None:
    results = _artifact_dir(tmp_path)
    metadata = _read_json(results / "run_metadata.json")
    metadata["output_sha256"]["core_runs.csv"] = "0" * 64
    _write_json(results / "run_metadata.json", metadata)

    with pytest.raises(ValidationLockError, match="hash claim mismatch"):
        build_validation_lock(results)


def test_rejects_selected_feature_fingerprint_mismatch(tmp_path: Path) -> None:
    results = _artifact_dir(tmp_path)
    core = pd.read_csv(results / "core_runs.csv")
    core.loc[1, "selected_features_sha256"] = "f" * 64
    _rewrite_core_and_selected(results, core)

    with pytest.raises(ValidationLockError, match="fingerprint mismatch"):
        build_validation_lock(results)


def test_rejects_selected_set_artifact_disagreement(tmp_path: Path) -> None:
    results = _artifact_dir(tmp_path)
    selected = _read_json(results / "selected_feature_sets.json")
    selected["run_42_correlation_redundancy_ranking_k22"]["selected_features"] = [
        FEATURES[0]
    ]
    _write_json(results / "selected_feature_sets.json", selected)
    _rehash_metadata(results)

    with pytest.raises(ValidationLockError, match="disagrees"):
        build_validation_lock(results)


def test_verifier_detects_source_mutation(tmp_path: Path) -> None:
    results = _artifact_dir(tmp_path)
    create_validation_lock(results)
    with (results / "core_runs.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ValidationLockError, match="Source artifact hash mismatch"):
        verify_validation_lock(results / LOCK_FILE_NAME)


def test_verifier_detects_lock_and_sidecar_mutation(tmp_path: Path) -> None:
    results = _artifact_dir(tmp_path)
    create_validation_lock(results)
    lock_path = results / LOCK_FILE_NAME
    lock = _read_json(lock_path)
    lock["created_at"] = "changed-but-semantically-volatile"
    _write_json(lock_path, lock)

    with pytest.raises(ValidationLockError, match="sidecar mismatch"):
        verify_validation_lock(lock_path)


def test_verifier_rejects_role_change_even_with_recomputed_semantic_hash(tmp_path: Path) -> None:
    lock = build_validation_lock(_artifact_dir(tmp_path))
    lock["roles"]["best_supervised"]["configuration_id"] = "anova_f_select_k_best_k32"
    lock["roles"]["best_supervised"]["locked_configuration_ref"] = "anova_f_select_k_best_k32"
    lock["semantic_payload_sha256"] = semantic_payload_sha256(lock)

    with pytest.raises(ValidationLockError, match="immutably mirror|selection invariants"):
        verify_validation_lock(lock, verify_sources=False)


def test_artifact_only_module_never_calls_loader_or_final_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.ai.model_utils as model_utils
    import src.pipeline_v06 as pipeline_v06

    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("forbidden")
        raise AssertionError("A data loader or final-test runner was called")

    monkeypatch.setattr(model_utils, "load_processed_dataco_splits", forbidden)
    monkeypatch.setattr(pipeline_v06, "run_locked_test_experiment", forbidden)

    assert build_validation_lock(_artifact_dir(tmp_path))["test_accessed"] is False
    assert calls == []


def test_report_contains_separate_costs_and_ground_truth_caveat(tmp_path: Path) -> None:
    results = _artifact_dir(tmp_path)
    create_validation_lock(results)
    report = (results / REPORT_FILE_NAME).read_text(encoding="utf-8")

    assert "Mean selector fit" in report
    assert "Mean model fit" in report
    assert "Mean combined fit" in report
    assert "Mean validation inference" in report
    assert "Mean peak RSS" in report
    assert "not native DataCo cybersecurity labeling" in report


def _artifact_dir(tmp_path: Path) -> Path:
    results = tmp_path / "feature_selection"
    results.mkdir()
    rows: list[dict[str, object]] = []
    selected_sets: dict[str, dict[str, object]] = {}
    for seed in SEEDS:
        for (
            configuration_id,
            selector_id,
            selector_type,
            strategy,
            requested,
            count,
            average_precision,
            f1,
            recall,
        ) in CONFIGURATIONS:
            run_id = f"run_{seed}_{configuration_id}"
            selected = _selected_features(selector_type, configuration_id, count, seed)
            fingerprint = fingerprint_feature_names(selected)
            row = {
                "status": "success",
                "run_id": run_id,
                "seed": seed,
                "evaluation_split": "validation",
                "training_split": "train",
                "configuration_id": configuration_id,
                "selector_id": selector_id,
                "selector_type": selector_type,
                "selector_method": f"synthetic_{selector_id}",
                "selector_mode": selector_type,
                "selector_parameters_json": json.dumps(
                    {} if requested is None else {"k": requested}, sort_keys=True
                ),
                "count_strategy": strategy,
                "requested_feature_count": requested,
                "candidate_feature_count": len(FEATURES),
                "candidate_features_sha256": fingerprint_feature_names(FEATURES),
                "selected_feature_count": count,
                "selected_features_json": json.dumps(selected),
                "selected_features_sha256": fingerprint,
                "model_id": "decision_tree",
                "model_parameters_json": json.dumps(
                    {"class_weight": "balanced", "max_depth": 5, "min_samples_leaf": 20},
                    sort_keys=True,
                ),
                "model_state_sha256": hashlib.sha256(run_id.encode()).hexdigest(),
                "prediction_threshold": 0.5,
                "threshold_policy": "fixed 0.5",
                "attack_type": "mixed",
                "attack_rate": 0.05,
                "attack_severity": "MEDIUM",
                "attack_config_path": "configs/attack.yaml",
                "average_precision": average_precision + (seed - 44) * 0.0001,
                "f1": f1 + (seed - 44) * 0.0001,
                "recall": recall + (seed - 44) * 0.0001,
                "precision": 0.65 + (seed - 44) * 0.0001,
                "roc_auc": 0.75 + (seed - 44) * 0.0001,
                "selector_fit_wall_time_sec": 0.01 * count,
                "model_fit_wall_time_sec": 0.02 * count,
                "combined_train_wall_time_sec": 0.03 * count,
                "attacked_validation_inference_wall_time_sec": 0.004 * count,
                "peak_rss_mib": 100.0 + count,
                "serialized_model_bytes": 1000 + count,
                "model_artifact_path": str(tmp_path / "absent_models" / f"{run_id}.joblib"),
                "resource_measurement_performed": True,
            }
            rows.append(row)
            selected_sets[run_id] = {
                "seed": seed,
                "configuration_id": configuration_id,
                "selector_id": selector_id,
                "requested_feature_count": requested,
                "selected_feature_count": count,
                "selected_features": selected,
                "selected_features_sha256": fingerprint,
            }
    pd.DataFrame(rows).to_csv(results / "core_runs.csv", index=False)
    _write_json(results / "selected_feature_sets.json", selected_sets)
    _write_json(
        results / "leakage_audit.json",
        {
            "schema_version": "v0.6-d",
            "scope": "development_train_and_validation_only",
            "status": "PASS",
            "test_accessed": False,
            "checks": [
                {"name": name, "passed": True}
                for name in (
                    "validation_only_evaluation",
                    "preprocessing_fitted_on_training_rows_only",
                    "attack_metadata_absent_from_model_features",
                    "row_ids_are_not_model_features",
                    "late_delivery_risk_is_not_cyber_ground_truth_or_model_feature",
                    "selector_fitted_on_training_rows_only",
                    "unsupervised_clean_train_and_supervised_controlled_label_governance",
                    "validation_rows_never_used_during_selector_fit",
                    "selected_feature_order_is_canonical",
                    "validation_configuration_selection_recorded",
                    "frozen_model_and_threshold",
                    "all_expected_runs_recorded",
                    "no_run_failures",
                )
            ],
        },
    )
    hashes = {
        name: _file_hash(results / name)
        for name in ("core_runs.csv", "selected_feature_sets.json", "leakage_audit.json")
    }
    _write_json(
        results / "run_metadata.json",
        {
            "schema_version": "v0.6-d",
            "execution_scope": "development_train_and_validation_only",
            "test_accessed": False,
            "run_count": 60,
            "successful_run_count": 60,
            "failed_run_count": 0,
            "output_sha256": hashes,
        },
    )
    return results


def _selected_features(
    selector_type: str, configuration_id: str, count: int, seed: int
) -> list[str]:
    if selector_type == "supervised" and seed % 2:
        return [*FEATURES[: count - 1], FEATURES[count]]
    return FEATURES[:count]


def _rewrite_core_and_selected(results: Path, core: pd.DataFrame) -> None:
    core.to_csv(results / "core_runs.csv", index=False)
    selected: dict[str, dict[str, object]] = {}
    for row in core.itertuples(index=False):
        selected[row.run_id] = {
            "seed": int(row.seed),
            "configuration_id": row.configuration_id,
            "selector_id": row.selector_id,
            "requested_feature_count": None
            if pd.isna(row.requested_feature_count)
            else int(row.requested_feature_count),
            "selected_feature_count": int(row.selected_feature_count),
            "selected_features": json.loads(row.selected_features_json),
            "selected_features_sha256": row.selected_features_sha256,
        }
    _write_json(results / "selected_feature_sets.json", selected)
    _rehash_metadata(results)


def _rehash_metadata(results: Path) -> None:
    metadata = _read_json(results / "run_metadata.json")
    metadata["output_sha256"] = {
        name: _file_hash(results / name)
        for name in ("core_runs.csv", "selected_feature_sets.json", "leakage_audit.json")
    }
    _write_json(results / "run_metadata.json", metadata)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
