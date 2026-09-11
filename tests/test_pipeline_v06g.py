"""Synthetic-only invariant tests for the V0.6-G robustness stage."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.ai.model_utils import ProcessedSplit
from src.config import ATTACK_CONFIG_FILE
from src.lightweight.models import LightweightDetector
from src.lightweight.resource_monitor import ResourceMeasurement
import src.pipeline_v06f as v06f
import src.pipeline_v06g as v06g
from src.pipeline_v06e import fingerprint_feature_names
from src.security.attack_generator import load_attack_config
from src.security.experiment_data import (
    PreparedAttackSplit,
    fingerprint_attack_labels,
    fingerprint_feature_matrix,
    fingerprint_frame,
    fingerprint_mapping,
    fingerprint_row_ids,
)


FEATURES = ("feature_a", "feature_b")
DT_PARAMETERS = {"max_depth": 5, "min_samples_leaf": 20, "class_weight": "balanced"}
CONFIGURATIONS = (
    ("baseline", ("full_baseline",), FEATURES),
    ("unsupervised", ("best_unsupervised",), ("feature_a",)),
    ("supervised", ("best_supervised",), ("feature_b",)),
)
RESOURCE = ResourceMeasurement(0.01, 0.005, 50.0, None, 1024 * 1024, 0, 1024)


def _base_frames(offset: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = np.arange(offset, offset + 20)
    features = pd.DataFrame(
        {
            "row_id": rows,
            "feature_a": np.linspace(-2.0, 2.0, len(rows)),
            "feature_b": np.tile([0.0, 1.0], len(rows) // 2),
        }
    )
    metadata = pd.DataFrame({"row_id": rows, "source": ["synthetic"] * len(rows)})
    return features, metadata


def _make_context(tmp_path: Path) -> v06g.RobustnessContext:
    results = tmp_path / "feature_selection"
    final_dir = results / "final_test"
    final_dir.mkdir(parents=True)
    final_manifest = final_dir / "final_test_artifact_hashes.json"
    final_manifest.write_text("{}\n", encoding="utf-8")
    training, _ = _base_frames()
    labels = pd.Series(([0] * 10) + ([1] * 10), dtype="int8")
    models: dict[tuple[str, int], LightweightDetector] = {}
    paths: dict[tuple[str, int], Path] = {}
    plans: list[v06g.LockedConfiguration] = []
    for configuration_id, roles, selected in CONFIGURATIONS:
        seed_plans = []
        for seed in v06g.EXPECTED_SEEDS:
            model = LightweightDetector(
                "decision_tree", selected, DT_PARAMETERS, random_state=seed
            ).fit(training.loc[:, list(selected)], labels)
            path = results / "locked_models" / f"{configuration_id}_{seed}.joblib"
            model.save(path)
            seed_plan = v06f.SeedPlan(
                seed=seed,
                selected_features=selected,
                selected_features_sha256=fingerprint_feature_names(selected),
                model_reference=str(path),
                model_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                model_state_sha256=v06f._model_state_sha256(model),
                model_size_bytes=path.stat().st_size,
            )
            seed_plans.append(seed_plan)
            models[(configuration_id, seed)] = LightweightDetector.load(path)
            paths[(configuration_id, seed)] = path
        plans.append(v06g.LockedConfiguration(configuration_id, roles, tuple(seed_plans)))
    scenarios = v06g.build_scenario_matrix(load_attack_config(ATTACK_CONFIG_FILE))
    plan = v06g.RobustnessPlan(
        semantic_lock_sha256="a" * 64,
        validation_lock_file_sha256="b" * 64,
        candidate_features=FEATURES,
        configurations=tuple(plans),
        scenarios=scenarios,
    )
    sources = {final_manifest, *paths.values()}
    return v06g.RobustnessContext(
        lock=MappingProxyType(
            {"attack_configuration": MappingProxyType({"attack_config_path": str(ATTACK_CONFIG_FILE)})}
        ),
        final_test_metadata=MappingProxyType(
            {"processed_dataset_metadata_sha256": "c" * 64}
        ),
        plan=plan,
        plan_sha256=v06g.fingerprint_robustness_plan(plan),
        gate_timestamp="2026-01-01T00:00:00+00:00",
        decision_tree_models=MappingProxyType(models),
        decision_tree_paths=MappingProxyType(paths),
        decision_tree_state_hashes=MappingProxyType(
            {key: v06g.model_state_sha256(model) for key, model in models.items()}
        ),
        source_hashes=MappingProxyType(
            {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        ),
        results_dir=results.resolve(),
        authorization_capability=v06g._CONTEXT_CAPABILITY,
    )


def _bundle(extra_split: bool = False) -> Any:
    train_features, train_metadata = _base_frames()
    test_features, test_metadata = _base_frames(100)
    splits = {
        "train": ProcessedSplit("train", train_features, train_metadata, None),
        "test": ProcessedSplit("test", test_features, test_metadata, None),
    }
    if extra_split:
        splits["validation"] = ProcessedSplit(
            "validation", test_features.copy(), test_metadata.copy(), None
        )
    return SimpleNamespace(selected_features=list(FEATURES), splits=splits)


def _preparer(calls: list[dict[str, Any]]):
    def prepare(**kwargs: Any) -> PreparedAttackSplit:
        calls.append(kwargs)
        clean = kwargs["split"].features.copy(deep=True)
        attacked = clean.copy(deep=True)
        count = min(len(clean), max(1, round(len(clean) * float(kwargs["attack_rate"]))))
        selected_rows = np.arange(count)
        magnitude = float(kwargs["attack_rate"]) * {
            "LOW": 10.0, "MEDIUM": 20.0, "HIGH": 30.0
        }[kwargs["severity"]]
        attacked.loc[selected_rows, "feature_a"] += magnitude
        labels = pd.Series(
            np.isin(np.arange(len(clean)), selected_rows).astype("int8"),
            index=clean.index,
            name="is_attack",
        )
        attack_type = kwargs["attack_type"]
        configured_types = list(load_attack_config(ATTACK_CONFIG_FILE)["scenarios"])
        types = configured_types if attack_type == "mixed" else [attack_type]
        mode = "mixed" if attack_type == "mixed" else "single"
        seed = kwargs["random_seed"]
        record_ids = clean.loc[selected_rows, "row_id"].to_numpy()
        ground_truth = pd.DataFrame(
            {"record_id": clean["row_id"], "is_attack": labels, "random_seed": seed}
        )
        manifest = pd.DataFrame(
            {
                "record_id": record_ids,
                "random_seed": seed,
                "attack_type": [types[index % len(types)] for index in range(count)],
                "change": magnitude,
            }
        )
        metadata = {
            "attack_mode": mode,
            "attack_types": types,
            "configured_attack_rate": float(kwargs["attack_rate"]),
            "severity": kwargs["severity"],
            "random_seed": seed,
            "eligible_records": len(clean),
            "selected_records": count,
            "actual_modified_records": count,
            "attack_rate_achieved": count / len(clean),
        }
        return PreparedAttackSplit(
            split_name=kwargs["split"].name,
            candidate_features=FEATURES,
            clean_features=clean,
            features=attacked,
            clean_metadata=kwargs["split"].metadata.copy(),
            metadata=kwargs["split"].metadata.copy(),
            labels=labels,
            ground_truth=ground_truth,
            manifest=manifest,
            attack_metadata=metadata,
            row_ids_sha256=fingerprint_row_ids(attacked),
            clean_features_sha256=fingerprint_feature_matrix(clean, FEATURES),
            attacked_features_sha256=fingerprint_feature_matrix(attacked, FEATURES),
            labels_sha256=fingerprint_attack_labels(attacked, labels),
            ground_truth_sha256=fingerprint_frame(ground_truth),
            manifest_sha256=fingerprint_frame(manifest),
            attack_metadata_sha256=fingerprint_mapping(metadata),
            test_authorization_id=kwargs.get("test_authorization_id"),
            test_authorization_capability=kwargs.get("test_authorization_capability"),
        )

    return prepare


def _trainer(records: list[dict[str, Any]]):
    def train(
        model: LightweightDetector,
        features: pd.DataFrame,
        labels: pd.Series,
        artifact_path: Path,
        **kwargs: Any,
    ) -> Any:
        records.append(
            {
                "model": model,
                "columns": tuple(features.columns),
                "labels": labels.copy(),
                "isolated": kwargs.get("isolated"),
            }
        )
        model.fit(features, labels)
        model.save(artifact_path)
        return SimpleNamespace(model=model, artifact_path=artifact_path, resources=RESOURCE)

    return train


def _inference(model: LightweightDetector, features: pd.DataFrame, **_: Any) -> Any:
    return SimpleNamespace(predictions=model.predict_frame(features), resources=RESOURCE)


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("v06g")
    context = _make_context(root)
    attack_calls: list[dict[str, Any]] = []
    training_calls: list[dict[str, Any]] = []
    result = v06g.run_v06g_robustness(
        context.results_dir,
        context=context,
        loader=lambda *_args, **_kwargs: _bundle(),
        attack_preparer=_preparer(attack_calls),
        training_runner=_trainer(training_calls),
        inference_runner=_inference,
        synthetic_test=True,
    )
    runs = pd.read_csv(context.results_dir / "robustness" / "robustness_runs.csv")
    return {
        "context": context,
        "result": result,
        "runs": runs,
        "attack_calls": attack_calls,
        "training_calls": training_calls,
    }


def _install_fake_v06f_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, corrupt: bool = False, bad_state: bool = False
) -> Path:
    context = _make_context(tmp_path)
    final_plan_configs = []
    for item in context.plan.configurations:
        seeds = list(item.seed_plans)
        final_plan_configs.append(v06f.ConfigurationPlan(item.configuration_id, item.roles, tuple(seeds)))
    final_plan = v06f.ExecutionPlan(
        semantic_lock_sha256=context.plan.semantic_lock_sha256,
        lock_file_sha256=context.plan.validation_lock_file_sha256,
        attack_config_sha256=hashlib.sha256(ATTACK_CONFIG_FILE.read_bytes()).hexdigest(),
        processed_dataset_metadata_sha256="c" * 64,
        candidate_features=FEATURES,
        configurations=tuple(final_plan_configs),
    )
    roles = {
        "full_baseline": {"configuration_id": "baseline", "locked_configuration_ref": "baseline"},
        "best_unsupervised": {"configuration_id": "unsupervised", "locked_configuration_ref": "unsupervised"},
        "best_supervised": {"configuration_id": "supervised", "locked_configuration_ref": "supervised"},
        "smallest_preserving": {"configuration_id": "supervised", "locked_configuration_ref": "supervised"},
    }
    lock = {
        "roles": roles,
        "attack_configuration": {"attack_config_path": str(ATTACK_CONFIG_FILE)},
    }
    fake = SimpleNamespace(
        plan=final_plan,
        lock=lock,
        models=context.decision_tree_models,
        model_paths=context.decision_tree_paths,
    )
    if bad_state:
        fake.models[("baseline", 42)].estimator.tree_.threshold[0] += 0.25
    monkeypatch.setattr(v06g.v06f, "pre_test_integrity_gate", lambda *_a, **_k: fake)
    results = context.results_dir
    for name in v06g.REQUIRED_FINAL_TEST_FILES:
        path = results / "final_test" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "final_test_metadata.json":
            payload = {
                "stage": "V0.6-F", "status": "COMPLETED", "test_accessed": True,
                "failed_run_count": 0,
                "semantic_lock_sha256": final_plan.semantic_lock_sha256,
                "validation_lock_file_sha256": final_plan.lock_file_sha256,
            }
        elif name == "final_test_governance_audit.json":
            payload = {"status": "PASS", "test_accessed": True}
        elif name == "test_access_state.json":
            payload = {"test_accessed": True}
        elif name.endswith(".json"):
            payload = {}
        else:
            path.write_text("synthetic\n", encoding="utf-8")
            continue
        path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = {
        name: hashlib.sha256((results / "final_test" / name).read_bytes()).hexdigest()
        for name in v06g.REQUIRED_FINAL_TEST_FILES
    }
    (results / "final_test" / "final_test_artifact_hashes.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    if corrupt:
        (results / "final_test" / "final_test_runs.csv").write_text("changed\n", encoding="utf-8")
    for name in ("validation_lock.json", "validation_lock.sha256", "validation_lock_audit.json"):
        (results / name).write_text("synthetic\n", encoding="utf-8")
    return results


def test_01_scenario_matrix_has_seven_families_and_thirteen_unique_scenarios() -> None:
    matrix = v06g.build_scenario_matrix(load_attack_config(ATTACK_CONFIG_FILE))
    assert len(matrix) == 13
    assert sum(item.in_family_panel for item in matrix) == 7
    assert len({item.scenario_id for item in matrix}) == 13


def test_02_scenario_matrix_rejects_nonofficial_rates() -> None:
    config = load_attack_config(ATTACK_CONFIG_FILE)
    config["attack_rates"] = [0.01, 0.05]
    with pytest.raises(v06g.V06GIntegrityError, match="official rates"):
        v06g.build_scenario_matrix(config)
    with pytest.raises(ValueError, match="Unknown attack types"):
        from src.security.attack_generator import generate_attack

        _, metadata = _base_frames()
        generate_attack(metadata, "invented_attack", 0.05, "MEDIUM", 42)


def test_03_scenario_matrix_rejects_nonofficial_severities() -> None:
    config = load_attack_config(ATTACK_CONFIG_FILE)
    config["severity_levels"] = ["LOW", "HIGH"]
    with pytest.raises(v06g.V06GIntegrityError, match="severities"):
        v06g.build_scenario_matrix(config)


def test_04_primary_mixed_scenario_is_deduplicated_into_both_panels() -> None:
    matrix = v06g.build_scenario_matrix(load_attack_config(ATTACK_CONFIG_FILE))
    primary = [x for x in matrix if x.attack_type == "mixed" and x.attack_rate == 0.05 and x.attack_severity == "MEDIUM"]
    assert len(primary) == 1
    assert primary[0].in_rate_panel and primary[0].in_severity_panel


def test_05_exact_roles_are_derived_and_duplicate_references_collapse(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    lock = {
        "roles": {
            "full_baseline": {"configuration_id": "baseline", "locked_configuration_ref": "baseline"},
            "best_unsupervised": {"configuration_id": "unsupervised", "locked_configuration_ref": "unsupervised"},
            "best_supervised": {"configuration_id": "unsupervised", "locked_configuration_ref": "unsupervised"},
            "smallest_preserving": {"configuration_id": "supervised", "locked_configuration_ref": "supervised"},
        }
    }
    final = v06f.ExecutionPlan("a", "b", "c", "d", FEATURES, tuple(
        v06f.ConfigurationPlan(x.configuration_id, x.roles, x.seed_plans) for x in context.plan.configurations
    ))
    derived = v06g.derive_locked_configurations(lock, final)
    assert len(derived) == 2
    assert derived[1].roles == ("best_unsupervised", "best_supervised")


def test_06_plan_has_exact_seeds_and_is_fingerprinted(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    assert context.plan.seeds == (42, 43, 44, 45, 46)
    mutated = replace(context.plan, seeds=(42,))
    assert v06g.fingerprint_robustness_plan(mutated) != context.plan_sha256
    changed = replace(context.plan.configurations[0].seed_plans[0], selected_features=("changed",))
    assert changed.selected_features != context.plan.configurations[0].seed_plans[0].selected_features
    first = context.plan.configurations[0]
    changed_configuration = replace(first, seed_plans=(changed, *first.seed_plans[1:]))
    object.__setattr__(
        context.plan,
        "configurations",
        (changed_configuration, *context.plan.configurations[1:]),
    )
    with pytest.raises(v06g.V06GIntegrityError, match="execution plan changed"):
        v06g.verify_source_hashes(context)


def test_07_gate_accepts_completed_v06f_and_verifies_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results = _install_fake_v06f_gate(tmp_path, monkeypatch)
    context = v06g.robustness_integrity_gate(results)
    assert context.final_test_metadata["test_accessed"] is True
    assert len(context.source_hashes) > len(v06g.REQUIRED_FINAL_TEST_FILES)


def test_08_gate_rejects_corrupt_v06f_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results = _install_fake_v06f_gate(tmp_path, monkeypatch, corrupt=True)
    with pytest.raises(v06g.V06GIntegrityError, match="artifact hash mismatch"):
        v06g.robustness_integrity_gate(results)


def test_09_gate_verifies_loaded_decision_tree_state_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results = _install_fake_v06f_gate(tmp_path, monkeypatch, bad_state=True)
    with pytest.raises(v06g.V06GIntegrityError, match="semantic state"):
        v06g.robustness_integrity_gate(results)


def test_10_source_mutation_is_detected_before_or_after_execution(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    source = Path(next(iter(context.source_hashes)))
    source.write_bytes(source.read_bytes() + b"mutation")
    with pytest.raises(v06g.V06GIntegrityError, match="source changed"):
        v06g.verify_source_hashes(context)


def test_11_synthetic_mode_requires_all_data_bearing_injections(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    with pytest.raises(v06g.V06GExecutionError, match="injected loader"):
        v06g.run_v06g_robustness(context.results_dir, context=context, synthetic_test=True)


def test_12_loader_may_expose_only_original_train_and_test(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    with pytest.raises(v06g.V06GExecutionError, match="exactly original train and test"):
        v06g.run_v06g_robustness(
            context.results_dir, context=context, loader=lambda *_a, **_k: _bundle(True),
            training_runner=_trainer([]), inference_runner=_inference, synthetic_test=True,
        )
    bundle = _bundle()
    bundle.splits["test"].features["row_id"] = bundle.splits["train"].features["row_id"]
    bundle.splits["test"].metadata["row_id"] = bundle.splits["train"].metadata["row_id"]
    with pytest.raises(v06g.V06GExecutionError, match="disjoint"):
        v06g.verify_loaded_splits(bundle, FEATURES)
    attack_metadata = _bundle()
    attack_metadata.splits["train"].features["is_attack"] = 0
    with pytest.raises(ValueError, match="Attack metadata"):
        v06g.verify_loaded_splits(attack_metadata, FEATURES)
    delivery_target = _bundle()
    delivery_target.splits["train"].features["Late_delivery_risk"] = 0
    with pytest.raises(v06g.V06GExecutionError, match="Late_delivery_risk"):
        v06g.verify_loaded_splits(delivery_target, FEATURES)


def test_13_attack_preparation_never_receives_validation(synthetic_run: dict[str, Any]) -> None:
    assert {call["split"].name for call in synthetic_run["attack_calls"]} == {"train", "test"}


def test_14_one_training_and_one_test_manifestation_per_seed_scenario(synthetic_run: dict[str, Any]) -> None:
    calls = synthetic_run["attack_calls"]
    assert sum(call["split"].name == "train" for call in calls) == 5
    assert sum(call["split"].name == "test" for call in calls) == 65
    kwargs = next(call for call in calls if call["split"].name == "test")
    prepare = _preparer([])
    assert v06f._manifestation_fingerprint(prepare(**kwargs)) == v06f._manifestation_fingerprint(
        prepare(**kwargs)
    )


def test_15_manifestation_hash_is_identical_across_all_six_cells(synthetic_run: dict[str, Any]) -> None:
    runs = synthetic_run["runs"]
    grouped = runs.groupby(["seed", "scenario_id"])
    assert grouped.size().eq(6).all()
    assert grouped["manifestation_sha256"].nunique().eq(1).all()


def test_16_lr_parameters_seed_features_and_threshold_are_exact(synthetic_run: dict[str, Any]) -> None:
    for call in synthetic_run["training_calls"]:
        model = call["model"]
        assert model.parameters == dict(v06g.LR_PARAMETERS)
        assert model.estimator.random_state in v06g.EXPECTED_SEEDS
        assert tuple(model.feature_names) == call["columns"]
    assert synthetic_run["runs"]["prediction_threshold"].eq(0.5).all()
    dt = synthetic_run["runs"].loc[synthetic_run["runs"]["classifier"].eq("decision_tree")]
    assert all(
        synthetic_run["context"].decision_tree_models[(row.configuration_id, row.seed)].parameters
        == DT_PARAMETERS
        for row in dt.itertuples()
    )


def test_17_lr_trains_once_per_seed_configuration_on_controlled_train_labels(synthetic_run: dict[str, Any]) -> None:
    calls = synthetic_run["training_calls"]
    assert len(calls) == 15
    assert all(call["isolated"] is True for call in calls)
    assert all(call["labels"].name == "is_attack" and set(call["labels"].unique()) == {0, 1} for call in calls)
    assert all(call["columns"] in {FEATURES, ("feature_a",), ("feature_b",)} for call in calls)


def test_18_frozen_dt_has_null_selector_and_model_fit_resources(synthetic_run: dict[str, Any]) -> None:
    dt = synthetic_run["runs"].loc[synthetic_run["runs"]["classifier"].eq("decision_tree")]
    assert dt["selector_fit_wall_time_sec"].isna().all()
    assert dt["model_fit_wall_time_sec"].isna().all()
    assert not dt["model_fit_performed"].any()
    assert not synthetic_run["runs"]["selector_fit_performed"].any()


def test_19_dt_and_lr_have_the_same_complete_scenario_matrix(synthetic_run: dict[str, Any]) -> None:
    runs = synthetic_run["runs"]
    dt = set(map(tuple, runs.loc[runs["classifier"].eq("decision_tree"), ["configuration_id", "seed", "scenario_id"]].to_numpy()))
    lr = set(map(tuple, runs.loc[runs["classifier"].eq("logistic_regression"), ["configuration_id", "seed", "scenario_id"]].to_numpy()))
    assert dt == lr
    assert len(runs) == 390


def test_20_runs_contain_explicit_metrics_confusion_paired_and_resources(synthetic_run: dict[str, Any]) -> None:
    runs = synthetic_run["runs"]
    required = {
        "average_precision", "pr_auc_trapezoidal", "TP", "TN", "FP", "FN",
        "paired_induced_attack_induced_detection_rate", "inference_wall_time_sec",
        "serialized_model_bytes", "model_state_sha256",
    }
    assert required.issubset(runs.columns)
    assert "pr_auc" not in runs.columns
    assert runs["inference_wall_time_sec"].gt(0).all()
    assert runs["per_record_inference_sec"].gt(0).all()
    assert runs["peak_rss_mib"].gt(0).all()
    assert runs["serialized_model_bytes"].gt(0).all()


def test_21_visibility_counts_rates_and_direct_recalls_use_selected_features(synthetic_run: dict[str, Any]) -> None:
    runs = synthetic_run["runs"]
    visible = runs.loc[runs["configuration_id"].eq("unsupervised")]
    invisible = runs.loc[runs["configuration_id"].eq("supervised")]
    assert visible["visible_feature_attack_count"].eq(visible["total_attacks"]).all()
    assert visible["invisible_feature_attack_count"].eq(0).all()
    assert invisible["visible_feature_attack_count"].eq(0).all()
    assert invisible["selected_feature_visibility_rate"].eq(0.0).all()
    assert invisible["visible_attack_recall"].isna().all()
    assert (visible["visibility_rate"] == 1.0).all()


def test_22_aggregation_reports_mean_std_min_max_and_five_seed_t_ci(synthetic_run: dict[str, Any]) -> None:
    summary = v06g.aggregate_metrics(
        synthetic_run["runs"], ["classifier", "configuration_id", "scenario_id"]
    )
    row = summary.loc[summary["metric"].eq("f1")].iloc[0]
    assert row["count"] == 5
    assert all(name in summary for name in ("mean", "std", "min", "max", "ci95_low", "ci95_high"))
    assert pd.notna(row["ci95_low"]) and pd.notna(row["ci95_high"])


def test_23_preservation_and_classifier_reports_include_paired_differences_and_cis(synthetic_run: dict[str, Any]) -> None:
    runs = synthetic_run["runs"]
    preservation = v06g.preservation_analysis(runs)
    classifier = v06g.paired_seed_differences(runs, comparison="classifier")
    assert set(preservation["margin_percent"]) == {5.0, 10.0}
    assert preservation["rate_scenario"].any()
    assert preservation["paired_seed_count"].eq(5).all()
    assert preservation["ci95_low"].notna().all()
    assert classifier["paired_seed_count"].eq(5).all()
    assert classifier["ci95_high"].notna().all()
    row = preservation.iloc[0]
    expected = max(0.0, 100.0 * (row["baseline_mean"] - row["configuration_mean"]) / row["baseline_mean"])
    assert row["relative_degradation_percent"] == pytest.approx(expected)
    assert not preservation["selection_performed"].any()
    assert not runs["reselection_performed"].any()


def test_24_atomic_output_suite_metadata_audit_and_eight_figures_exist(synthetic_run: dict[str, Any]) -> None:
    output = synthetic_run["context"].results_dir / "robustness"
    for stem in (
        "attack_family_summary", "attack_rate_summary", "attack_severity_summary",
        "visibility_analysis", "robustness_preservation", "classifier_comparison", "resource_comparison",
    ):
        assert (output / f"{stem}.csv").is_file()
        assert (output / f"{stem}.json").is_file()
    metadata = json.loads((output / "robustness_metadata.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "robustness_governance_audit.json").read_text(encoding="utf-8"))
    hashes = json.loads((output / "robustness_artifact_hashes.json").read_text(encoding="utf-8"))
    assert metadata["completed_run_count"] == 390 and metadata["logistic_fit_count"] == 15
    assert audit["status"] == "PASS"
    assert len(list((output / "figures").glob("*.png"))) == 8
    assert all(name in hashes for name in (f"figures/{figure}" for figure in v06g.FIGURE_NAMES))
    assert all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
        for path, expected in synthetic_run["context"].source_hashes.items()
    )
