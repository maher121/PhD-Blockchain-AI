"""End-to-end V0.5 lightweight AI and computational-efficiency workflow."""

from __future__ import annotations

import datetime as dt
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import sklearn
import yaml

from src.ai.anomaly_detector import IsolationForestBaseline
from src.ai.model_utils import load_processed_dataco, write_json
from src.config import (
    AI_BASELINE_DIR,
    GLOBAL_SEED,
    LIGHTWEIGHT_CONFIG_FILE,
    LIGHTWEIGHT_FIGURES_DIR,
    LIGHTWEIGHT_MODELS_DIR,
    LIGHTWEIGHT_PREDICTIONS_DIR,
    LIGHTWEIGHT_RESULTS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    SECURITY_MANIFESTS_DIR,
    SECURITY_RESULTS_DIR,
    V03_MODEL_FILE,
    V05_EXPERIMENT_DIR,
)
from src.energy.measurement import report_system_info
from src.lightweight.evaluation import (
    calculate_metrics,
    identify_pareto_candidates,
    save_lightweight_figures,
)
from src.lightweight.feature_reduction import (
    V05_FORBIDDEN_FEATURES,
    CorrelationFeatureReducer,
    validate_model_features,
)
from src.lightweight.models import LightweightDetector, create_model
from src.lightweight.resource_monitor import measure_model_size
from src.lightweight.training import (
    measure_artifact_inference,
    reproduce_v03_training,
    run_inference,
    train_model,
)
from src.security.attack_generator import generate_attack, load_attack_config
from src.security.attack_scenarios import eligible_mask
from src.security.evaluation import (
    detector_visible_modified_count,
    detector_visible_mask,
    evaluate_paired_detection,
    evaluate_visible_attacks,
    project_attacks_to_feature_space,
)
from src.security.ground_truth import MANIFEST_COLUMNS, assert_no_attack_metadata

PUBLICATION_COLUMNS: tuple[str, ...] = (
    "model",
    "feature_count",
    "precision",
    "recall",
    "f1",
    "pr_auc",
    "roc_auc",
    "training_time_sec",
    "inference_time_sec",
    "memory_mb",
    "model_size_kb",
)


def load_lightweight_config(
    path: Path | str = LIGHTWEIGHT_CONFIG_FILE,
) -> dict[str, Any]:
    """Load and validate the versioned V0.5 experiment configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"V0.5 configuration not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = {"experiment_id", "random_seed", "models", "feature_reduction", "evaluation"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"V0.5 configuration must define {sorted(required)}")
    if not payload["models"]:
        raise ValueError("V0.5 requires at least one lightweight model.")
    return payload


def run_lightweight_evaluation_v05(
    processed_dir: Path | str = PROCESSED_DATA_DIR,
    results_dir: Path | str = LIGHTWEIGHT_RESULTS_DIR,
    models_dir: Path | str = LIGHTWEIGHT_MODELS_DIR,
    experiment_dir: Path | str = V05_EXPERIMENT_DIR,
    config_path: Path | str = LIGHTWEIGHT_CONFIG_FILE,
    v03_model_path: Path | str = V03_MODEL_FILE,
    v03_results_dir: Path | str = AI_BASELINE_DIR,
    security_results_dir: Path | str = SECURITY_RESULTS_DIR,
) -> dict[str, Any]:
    """Train, evaluate, and persist the complete V0.5 experiment."""
    started = time.perf_counter()
    config = load_lightweight_config(config_path)
    random_seed = int(config.get("random_seed", GLOBAL_SEED))
    if random_seed != GLOBAL_SEED:
        raise ValueError("V0.5 is governed by random_state=42 for direct comparison.")

    results_dir = Path(results_dir)
    models_dir = Path(models_dir)
    experiment_dir = Path(experiment_dir)
    figures_dir = results_dir / LIGHTWEIGHT_FIGURES_DIR.name
    predictions_dir = results_dir / LIGHTWEIGHT_PREDICTIONS_DIR.name
    for directory in (results_dir, models_dir, experiment_dir, figures_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)

    bundle = load_processed_dataco(processed_dir)
    selected_features = list(bundle.selected_features)
    validate_model_features(selected_features)
    for split in bundle.splits.values():
        assert_no_attack_metadata(split.features[selected_features])

    reduction = config["feature_reduction"]
    if reduction.get("method") != CorrelationFeatureReducer.method:
        raise ValueError(
            "V0.5 supports only transparent train_correlation_redundancy_ranking."
        )
    reducer = CorrelationFeatureReducer(selected_features).fit(
        bundle.splits["train"].features
    )
    fractions = tuple(float(value) for value in reduction["fractions"])
    feature_subsets = reducer.subsets(fractions)
    ranking = reducer.ranking_frame()
    ranking.to_csv(results_dir / "feature_ranking.csv", index=False)
    write_json(
        {
            "method": reducer.method,
            "fitted_on": "clean V0.2 training split only",
            "labels_used": False,
            "test_data_used": False,
            "fractions": list(fractions),
            "subsets": {str(key): value for key, value in feature_subsets.items()},
        },
        results_dir / "feature_subsets.json",
    )

    attack_config = load_attack_config()
    evaluation_config = config["evaluation"]
    primary_rate = float(evaluation_config["primary_attack_rate"])
    primary_severity = str(evaluation_config["primary_severity"]).upper()
    primary_attack_type = evaluation_config["primary_attack_type"]
    inference_repeats = int(evaluation_config.get("inference_repeats", 10))
    if inference_repeats < 2:
        raise ValueError("Use at least two inference repeats to amortize timing noise.")
    primary_splits = {
        name: _prepare_attack_split(
            clean_features=bundle.splits[name].features,
            clean_metadata=bundle.splits[name].metadata,
            reference_features=bundle.splits["train"].features,
            reference_metadata=bundle.splits["train"].metadata,
            attack_type=primary_attack_type,
            attack_rate=primary_rate,
            severity=primary_severity,
            random_seed=random_seed,
            experiment_id=(
                "v0_4_mixed_r0p05_medium_s42"
                if name == "test"
                else f"{config['experiment_id']}_{name}_primary"
            ),
        )
        for name in ("train", "validation", "test")
    }
    primary_test = primary_splits["test"]
    primary_test["ground_truth"].to_csv(
        predictions_dir / "primary_test_ground_truth.csv", index=False
    )
    primary_test["manifest"].to_csv(
        results_dir / "primary_test_attack_manifest.csv", index=False
    )

    existing_primary_manifest = (
        Path(security_results_dir)
        / SECURITY_MANIFESTS_DIR.name
        / "v0_4_mixed_r0p05_medium_s42.csv"
    )
    primary_manifest_verification = _verify_manifest_if_available(
        primary_test["manifest"], existing_primary_manifest
    )
    if primary_manifest_verification["available"] and not primary_manifest_verification["matches"]:
        raise RuntimeError("Regenerated primary test scenario does not match V0.4.")

    feature_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    full_model_paths: dict[str, Path] = {}
    reload_checks: dict[str, bool] = {}
    for model_name, parameters in config["models"].items():
        for fraction, features in feature_subsets.items():
            configuration_id = f"{model_name}_f{len(features)}"
            model = create_model(
                model_name,
                features,
                parameters,
                random_state=random_seed,
            )
            artifact_path = models_dir / f"{configuration_id}.joblib"
            training = train_model(
                model,
                primary_splits["train"]["features"],
                primary_splits["train"]["ground_truth"]["is_attack"],
                artifact_path,
            )
            model = training.model
            validation_predictions = _add_record_ids(
                model.predict_frame(primary_splits["validation"]["features"]),
                primary_splits["validation"]["features"],
            )
            validation_metrics = calculate_metrics(
                primary_splits["validation"]["ground_truth"], validation_predictions
            )
            validation_inference = run_inference(
                model,
                primary_splits["validation"]["features"],
                artifact_path=training.artifact_path,
                repeats=inference_repeats,
            )
            validation_rows.append(
                {
                    "configuration_id": configuration_id,
                    "model": model_name,
                    "feature_count": len(features),
                    **_metric_columns(validation_metrics),
                }
            )

            inference = run_inference(
                model,
                primary_test["features"],
                artifact_path=training.artifact_path,
                repeats=inference_repeats,
            )
            predictions = _add_record_ids(inference.predictions, primary_test["features"])
            metrics = calculate_metrics(primary_test["ground_truth"], predictions)
            subset_visible_mask = detector_visible_mask(
                bundle.splits["test"].features[["row_id", *features]],
                primary_test["features"][["row_id", *features]],
            )
            visible_metrics = evaluate_visible_attacks(
                primary_test["ground_truth"], predictions, subset_visible_mask
            )
            predictions.to_csv(
                predictions_dir / f"{configuration_id}_test_predictions.csv", index=False
            )
            reloaded = LightweightDetector.load(training.artifact_path)
            reloaded_predictions = reloaded.predict_frame(primary_test["features"])
            reload_verified = bool(
                np.array_equal(
                    inference.predictions["anomaly_label"].to_numpy(),
                    reloaded_predictions["anomaly_label"].to_numpy(),
                )
                and np.allclose(
                    inference.predictions["anomaly_score"].to_numpy(),
                    reloaded_predictions["anomaly_score"].to_numpy(),
                    rtol=0.0,
                    atol=0.0,
                )
            )
            reload_checks[configuration_id] = reload_verified
            if not reload_verified:
                raise RuntimeError(f"Reload verification failed for {configuration_id}.")

            feature_row = {
                "configuration_id": configuration_id,
                "model": model_name,
                "feature_fraction": fraction,
                "feature_count": len(features),
                **_metric_columns(metrics),
                "attacked_records": metrics["attacked_records"],
                "validation_f1": validation_metrics["f1"],
                "validation_pr_auc": validation_metrics["pr_auc"],
                "validation_roc_auc": validation_metrics["roc_auc"],
                "validation_inference_time_sec": validation_inference.resources.wall_time_sec,
                "validation_memory_mb": validation_inference.resources.peak_rss_mb,
                **_visible_metric_columns(visible_metrics),
                "training_time_sec": training.resources.wall_time_sec,
                "inference_time_sec": inference.resources.wall_time_sec,
                "total_execution_time_sec": (
                    training.resources.wall_time_sec + inference.resources.wall_time_sec
                ),
                "memory_mb": inference.resources.peak_rss_mb,
                "model_size_bytes": training.model_size_bytes,
                "model_size_kb": training.model_size_kb,
                "number_of_training_samples": len(primary_splits["train"]["features"]),
                "number_of_test_samples": len(primary_test["features"]),
                "artifact_path": _portable_path(training.artifact_path),
                "training_regime": "V0.4 5% MEDIUM mixed attacked training split",
                "threshold_policy": _threshold_policy(model_name),
            }
            feature_rows.append(feature_row)
            resource_rows.append(
                {
                    "configuration_id": configuration_id,
                    "model": model_name,
                    "feature_count": len(features),
                    "training_time_sec": training.resources.wall_time_sec,
                    "training_cpu_time_sec": training.resources.cpu_time_sec,
                    "training_average_cpu_percent": training.resources.average_cpu_percent,
                    "training_peak_cpu_percent": training.resources.peak_cpu_percent,
                    "training_peak_rss_mb": training.resources.peak_rss_mb,
                    "training_rss_delta_mb": training.resources.rss_delta_mb,
                    "inference_time_sec": inference.resources.wall_time_sec,
                    "inference_cpu_time_sec": inference.resources.cpu_time_sec,
                    "inference_average_cpu_percent": inference.resources.average_cpu_percent,
                    "inference_peak_cpu_percent": inference.resources.peak_cpu_percent,
                    "inference_peak_rss_mb": inference.resources.peak_rss_mb,
                    "inference_rss_delta_mb": inference.resources.rss_delta_mb,
                    "validation_inference_time_sec": validation_inference.resources.wall_time_sec,
                    "validation_inference_peak_rss_mb": validation_inference.resources.peak_rss_mb,
                    "model_size_bytes": training.model_size_bytes,
                    "model_size_kb": training.model_size_kb,
                    "energy_directly_measured": False,
                }
            )
            complexity_rows.append(
                {
                    "configuration_id": configuration_id,
                    **model.complexity(),
                    "indicators_are_energy_measurements": False,
                }
            )
            if np.isclose(fraction, 1.0):
                full_model_paths[model_name] = training.artifact_path

    feature_results = pd.DataFrame(feature_rows)
    resources = pd.DataFrame(resource_rows)
    complexities = pd.DataFrame(complexity_rows)
    validation_results = pd.DataFrame(validation_rows)
    feature_results.to_csv(results_dir / "feature_reduction.csv", index=False)
    complexities.to_csv(results_dir / "model_complexity.csv", index=False)
    validation_results.to_csv(results_dir / "validation_metrics.csv", index=False)

    baseline = _evaluate_v03_reference(
        v03_model_path=Path(v03_model_path),
        v03_results_dir=Path(v03_results_dir),
        clean_train_features=bundle.splits["train"].features,
        validation_features=primary_splits["validation"]["features"],
        validation_ground_truth=primary_splits["validation"]["ground_truth"],
        clean_test_features=bundle.splits["test"].features,
        test_features=primary_test["features"],
        ground_truth=primary_test["ground_truth"],
        reproduced_model_path=models_dir / "v0_3_isolation_forest_reproduced_f43.joblib",
        inference_repeats=inference_repeats,
        predictions_path=predictions_dir / "v0_3_isolation_forest_test_predictions.csv",
    )
    resources = pd.concat(
        [pd.DataFrame([baseline["resource_row"]]), resources], ignore_index=True
    )
    resources.to_csv(results_dir / "resource_comparison.csv", index=False)
    security_summary_path = Path(security_results_dir) / "security_summary.json"
    v04_metric_verification = _verify_v04_metrics_if_available(
        baseline["metrics"], security_summary_path
    )
    if v04_metric_verification["available"] and not v04_metric_verification["matches"]:
        raise RuntimeError("V0.3 reference metrics do not reproduce the V0.4 primary result.")

    full_rows = feature_results.loc[np.isclose(feature_results["feature_fraction"], 1.0)]
    publication_internal = pd.concat(
        [pd.DataFrame([baseline["comparison_row"]]), full_rows], ignore_index=True
    )
    model_comparison = publication_internal.loc[:, PUBLICATION_COLUMNS]
    model_comparison.to_csv(results_dir / "model_comparison.csv", index=False)

    pareto_input = pd.concat(
        [pd.DataFrame([baseline["pareto_row"]]), feature_results], ignore_index=True
    )
    pareto = identify_pareto_candidates(
        pareto_input,
        maximize=("validation_f1",),
        minimize=(
            "validation_inference_time_sec",
            "validation_memory_mb",
            "feature_count",
        ),
    )
    pareto.to_csv(results_dir / "pareto_candidates.csv", index=False)

    visible_comparison = publication_internal[
        [
            "model",
            "feature_count",
            "visible_attacked_records",
            "visible_precision",
            "visible_recall",
            "visible_f1",
            "visible_pr_auc",
            "visible_pr_auc_method",
            "visible_average_precision",
            "visible_roc_auc",
        ]
    ]
    visible_comparison.to_csv(
        results_dir / "visible_attack_comparison.csv", index=False
    )
    publication_internal[
        ["model", "training_regime", "threshold_policy"]
    ].to_csv(results_dir / "model_methodology.csv", index=False)

    v03_comparison = _build_v03_comparison(full_rows, baseline["pareto_row"])
    v03_comparison.to_csv(results_dir / "v03_comparison.csv", index=False)

    security_scenarios = pd.DataFrame()
    scenario_verification = {"available": 0, "matched": 0, "mismatched": 0}
    if bool(evaluation_config.get("security_scenario_sweep", True)):
        security_scenarios, scenario_verification = _run_security_scenario_sweep(
            bundle=bundle,
            attack_config=attack_config,
            random_seed=random_seed,
            full_model_paths=full_model_paths,
            v03_model_path=Path(v03_model_path),
            security_results_dir=Path(security_results_dir),
        )
        security_scenarios.to_csv(
            results_dir / "security_scenario_evaluation.csv", index=False
        )
        if scenario_verification["mismatched"]:
            raise RuntimeError("One or more regenerated security scenarios differ from V0.4.")

    figure_paths = save_lightweight_figures(
        feature_results, model_comparison, pareto, figures_dir
    )
    total_time = time.perf_counter() - started
    leakage_audit = {
        "status": "PASS",
        "forbidden_columns": sorted(V05_FORBIDDEN_FEATURES),
        "feature_ranking_fitted_on": "clean_training_split_only",
        "feature_ranking_used_labels": False,
        "feature_ranking_used_test_data": False,
        "configuration_selection_metric_source": "validation_split_only",
        "test_metrics_used_for_configuration_selection": False,
        "model_inputs": "selected V0.2 processed features only",
    }

    summary = _build_summary(
        config=config,
        bundle=bundle,
        feature_results=feature_results,
        model_comparison=model_comparison,
        pareto=pareto,
        v03_comparison=v03_comparison,
        security_scenarios=security_scenarios,
        scenario_verification=scenario_verification,
        primary_manifest_verification=primary_manifest_verification,
        v04_metric_verification=v04_metric_verification,
        v03_reproduction_verified=baseline["reproduction_matches_saved_v03"],
        leakage_audit=leakage_audit,
        reload_checks=reload_checks,
        figure_paths=figure_paths,
        total_time=total_time,
    )
    write_json(summary, results_dir / "lightweight_summary.json")
    write_json(leakage_audit, results_dir / "leakage_audit.json")

    reproducibility = {
        "stage": "V0.5",
        "experiment_id": config["experiment_id"],
        "random_seed": random_seed,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "psutil_version": psutil.__version__,
        "pyyaml_version": yaml.__version__,
        "hardware": report_system_info(),
        "dataset": bundle.dataset_metadata.get("dataset", {}),
        "split": bundle.dataset_metadata.get("split", {}),
        "model_configuration": config["models"],
        "feature_configuration": config["feature_reduction"],
        "configuration_selection": {
            "metric_source": "controlled validation split",
            "test_metrics_used": False,
        },
        "attack_configuration": {
            "primary_attack_type": primary_attack_type,
            "primary_attack_rate": primary_rate,
            "primary_severity": primary_severity,
            "source": "unchanged V0.4 attack_scenarios.yaml",
        },
        "timing_note": "Wall-clock timings may vary with system load.",
    }
    write_json(reproducibility, experiment_dir / "reproducibility.json")
    write_json(config, experiment_dir / "config.json")
    write_json(summary, experiment_dir / "experiment_results.json")
    write_json(
        {
            "method": reducer.method,
            "ranking": ranking.to_dict(orient="records"),
            "subsets": {str(key): value for key, value in feature_subsets.items()},
        },
        experiment_dir / "feature_configuration.json",
    )

    return {
        "summary": summary,
        "model_comparison": model_comparison,
        "feature_reduction": feature_results,
        "resource_comparison": resources,
        "pareto_candidates": pareto,
        "security_scenarios": security_scenarios,
        "figure_paths": figure_paths,
        "paths": {
            "results": results_dir,
            "models": models_dir,
            "experiment": experiment_dir,
            "notebook": PROJECT_ROOT / "notebooks" / "05_lightweight_ai_comparison.ipynb",
        },
    }


def _prepare_attack_split(
    *,
    clean_features: pd.DataFrame,
    clean_metadata: pd.DataFrame,
    reference_features: pd.DataFrame,
    reference_metadata: pd.DataFrame,
    attack_type: str | list[str],
    attack_rate: float,
    severity: str,
    random_seed: int,
    experiment_id: str,
) -> dict[str, Any]:
    attacked_metadata, metadata, ground_truth = generate_attack(
        clean_metadata,
        attack_type,
        attack_rate,
        severity,
        random_seed,
        experiment_id=experiment_id,
    )
    manifest = metadata.pop("manifest")
    attacked_features = project_attacks_to_feature_space(
        clean_features,
        clean_metadata,
        attacked_metadata,
        manifest,
        reference_features=reference_features,
        reference_metadata=reference_metadata,
    )
    assert_no_attack_metadata(attacked_features)
    return {
        "features": attacked_features,
        "metadata": attacked_metadata,
        "ground_truth": ground_truth,
        "manifest": manifest,
        "attack_metadata": metadata,
    }


def _evaluate_v03_reference(
    *,
    v03_model_path: Path,
    v03_results_dir: Path,
    clean_train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    validation_ground_truth: pd.DataFrame,
    clean_test_features: pd.DataFrame,
    test_features: pd.DataFrame,
    ground_truth: pd.DataFrame,
    reproduced_model_path: Path,
    inference_repeats: int,
    predictions_path: Path,
) -> dict[str, Any]:
    original = IsolationForestBaseline.load(v03_model_path)
    training_resources = reproduce_v03_training(
        original.feature_names,
        original.model_parameters,
        clean_train_features,
        reproduced_model_path,
    )
    detector = IsolationForestBaseline.load(reproduced_model_path)
    original_predictions = original.predict_frame(test_features[original.feature_names])
    reproduced_predictions = detector.predict_frame(test_features[detector.feature_names])
    reproduction_matches = bool(
        np.array_equal(
            original_predictions["anomaly_label"].to_numpy(),
            reproduced_predictions["anomaly_label"].to_numpy(),
        )
        and np.allclose(
            original_predictions["anomaly_score"].to_numpy(),
            reproduced_predictions["anomaly_score"].to_numpy(),
            rtol=0.0,
            atol=0.0,
        )
    )
    if not reproduction_matches:
        raise RuntimeError("Current-environment V0.3 reproduction differs from saved V0.3 model.")
    resources = measure_artifact_inference(
        reproduced_model_path,
        test_features[detector.feature_names],
        artifact_kind="v03",
        repeats=inference_repeats,
    )
    predictions = _add_record_ids(reproduced_predictions, test_features)
    predictions.to_csv(predictions_path, index=False)
    metrics = calculate_metrics(ground_truth, predictions)
    validation_predictions = _add_record_ids(
        detector.predict_frame(validation_features[detector.feature_names]),
        validation_features,
    )
    validation_metrics = calculate_metrics(
        validation_ground_truth, validation_predictions
    )
    validation_resources = measure_artifact_inference(
        reproduced_model_path,
        validation_features[detector.feature_names],
        artifact_kind="v03",
        repeats=inference_repeats,
    )
    visible_metrics = evaluate_visible_attacks(
        ground_truth,
        predictions,
        detector_visible_mask(clean_test_features, test_features),
    )
    size = measure_model_size(reproduced_model_path)
    original_size = measure_model_size(v03_model_path)
    baseline_report_path = v03_results_dir / "baseline_report.json"
    if not baseline_report_path.is_file():
        raise FileNotFoundError(f"V0.3 baseline report not found: {baseline_report_path}")
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    row = {
        "configuration_id": "v0_3_isolation_forest_reference_f43",
        "model": "v0_3_isolation_forest",
        "feature_fraction": 1.0,
        "feature_count": len(detector.feature_names),
        **_metric_columns(metrics),
        "validation_f1": validation_metrics["f1"],
        "validation_pr_auc": validation_metrics["pr_auc"],
        "validation_roc_auc": validation_metrics["roc_auc"],
        "validation_inference_time_sec": validation_resources.wall_time_sec,
        "validation_memory_mb": validation_resources.peak_rss_mb,
        **_visible_metric_columns(visible_metrics),
        "training_time_sec": training_resources.wall_time_sec,
        "inference_time_sec": resources.wall_time_sec,
        "total_execution_time_sec": training_resources.wall_time_sec + resources.wall_time_sec,
        "memory_mb": resources.peak_rss_mb,
        "model_size_bytes": int(size["model_size_bytes"]),
        "model_size_kb": float(size["model_size_kb"]),
        "number_of_training_samples": baseline_report["dataset"]["split_sizes"]["train"],
        "number_of_test_samples": len(test_features),
        "artifact_path": _portable_path(reproduced_model_path),
        "training_regime": "clean V0.2 training split (exact V0.3 methodology)",
        "threshold_policy": "sklearn IsolationForest decision_function threshold at zero",
    }
    resource_row = {
        "configuration_id": row["configuration_id"],
        "model": row["model"],
        "feature_count": row["feature_count"],
        "training_time_sec": training_resources.wall_time_sec,
        "training_cpu_time_sec": training_resources.cpu_time_sec,
        "training_average_cpu_percent": training_resources.average_cpu_percent,
        "training_peak_cpu_percent": training_resources.peak_cpu_percent,
        "training_peak_rss_mb": training_resources.peak_rss_mb,
        "training_rss_delta_mb": training_resources.rss_delta_mb,
        "inference_time_sec": resources.wall_time_sec,
        "inference_cpu_time_sec": resources.cpu_time_sec,
        "inference_average_cpu_percent": resources.average_cpu_percent,
        "inference_peak_cpu_percent": resources.peak_cpu_percent,
        "inference_peak_rss_mb": resources.peak_rss_mb,
        "inference_rss_delta_mb": resources.rss_delta_mb,
        "validation_inference_time_sec": validation_resources.wall_time_sec,
        "validation_inference_peak_rss_mb": validation_resources.peak_rss_mb,
        "model_size_bytes": row["model_size_bytes"],
        "model_size_kb": row["model_size_kb"],
        "energy_directly_measured": False,
    }
    return {
        "comparison_row": row,
        "pareto_row": row,
        "resource_row": resource_row,
        "metrics": metrics,
        "reproduction_matches_saved_v03": reproduction_matches,
        "original_model_size_bytes": int(original_size["model_size_bytes"]),
    }


def _run_security_scenario_sweep(
    *,
    bundle: Any,
    attack_config: dict[str, Any],
    random_seed: int,
    full_model_paths: dict[str, Path],
    v03_model_path: Path,
    security_results_dir: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    test = bundle.splits["test"]
    train = bundle.splits["train"]
    attack_types = [
        name
        for name, definition in attack_config["scenarios"].items()
        if bool(eligible_mask(test.metadata, definition).any())
    ]
    primary_rate = 0.05
    primary_severity = "MEDIUM"
    scenarios = {
        (name, float(rate), primary_severity)
        for name in attack_types
        for rate in attack_config["attack_rates"]
    }
    scenarios.update(
        {
            (name, primary_rate, str(severity).upper())
            for name in attack_types
            for severity in attack_config["severity_levels"]
        }
    )

    models: dict[str, Any] = {
        "v0_3_isolation_forest": IsolationForestBaseline.load(v03_model_path)
    }
    models.update(
        {name: LightweightDetector.load(path) for name, path in full_model_paths.items()}
    )
    clean_predictions = {
        name: _add_record_ids(model.predict_frame(test.features), test.features)
        for name, model in models.items()
    }
    verification = {"available": 0, "matched": 0, "mismatched": 0}
    rows: list[dict[str, Any]] = []
    for attack_type, rate, severity in sorted(scenarios):
        experiment_id = _v04_experiment_id(attack_type, rate, severity, random_seed)
        scenario = _prepare_attack_split(
            clean_features=test.features,
            clean_metadata=test.metadata,
            reference_features=train.features,
            reference_metadata=train.metadata,
            attack_type=attack_type,
            attack_rate=rate,
            severity=severity,
            random_seed=random_seed,
            experiment_id=experiment_id,
        )
        existing = (
            security_results_dir
            / SECURITY_MANIFESTS_DIR.name
            / f"{experiment_id}.csv"
        )
        checked = _verify_manifest_if_available(scenario["manifest"], existing)
        if checked["available"]:
            verification["available"] += 1
            key = "matched" if checked["matches"] else "mismatched"
            verification[key] += 1
        visible_count = detector_visible_modified_count(
            test.features, scenario["features"], scenario["ground_truth"]
        )
        visible_mask = detector_visible_mask(test.features, scenario["features"])
        for model_name, model in models.items():
            predictions = _add_record_ids(
                model.predict_frame(scenario["features"]), scenario["features"]
            )
            metrics = calculate_metrics(scenario["ground_truth"], predictions)
            visible_metrics = evaluate_visible_attacks(
                scenario["ground_truth"], predictions, visible_mask
            )
            paired = evaluate_paired_detection(
                scenario["ground_truth"], clean_predictions[model_name], predictions
            )
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "model": model_name,
                    "attack_type": attack_type,
                    "attack_rate": rate,
                    "severity": severity,
                    "selected_records": scenario["attack_metadata"]["actual_modified_records"],
                    "detector_visible_modified_records": visible_count,
                    "metrics_scope": "all controlled labels; see visible_* for input-changing attacks",
                    **_metric_columns(metrics),
                    **_visible_metric_columns(visible_metrics),
                    "newly_detected_after_attack": paired["newly_detected_after_attack"],
                    "attack_induced_detection_rate": paired[
                        "attack_induced_detection_rate"
                    ],
                    "mean_anomaly_score_delta": paired["mean_anomaly_score_delta"],
                }
            )
    return pd.DataFrame(rows), verification


def _build_v03_comparison(
    full_rows: pd.DataFrame, baseline: dict[str, Any]
) -> pd.DataFrame:
    rows = []
    for row in full_rows.to_dict(orient="records"):
        rows.append(
            {
                "model": row["model"],
                "reference_model": baseline["model"],
                "f1_change": row["f1"] - baseline["f1"],
                "pr_auc_change": row["pr_auc"] - baseline["pr_auc"],
                "roc_auc_change": row["roc_auc"] - baseline["roc_auc"],
                "inference_time_change_sec": (
                    row["inference_time_sec"] - baseline["inference_time_sec"]
                ),
                "inference_time_ratio": (
                    row["inference_time_sec"] / baseline["inference_time_sec"]
                ),
                "memory_change_mb": row["memory_mb"] - baseline["memory_mb"],
                "model_size_change_kb": row["model_size_kb"] - baseline["model_size_kb"],
                "feature_count_change": row["feature_count"] - baseline["feature_count"],
                "interpretation": "trade-off; no single metric establishes superiority",
            }
        )
    return pd.DataFrame(rows)


def _build_summary(
    *,
    config: dict[str, Any],
    bundle: Any,
    feature_results: pd.DataFrame,
    model_comparison: pd.DataFrame,
    pareto: pd.DataFrame,
    v03_comparison: pd.DataFrame,
    security_scenarios: pd.DataFrame,
    scenario_verification: dict[str, int],
    primary_manifest_verification: dict[str, Any],
    v04_metric_verification: dict[str, Any],
    v03_reproduction_verified: bool,
    leakage_audit: dict[str, Any],
    reload_checks: dict[str, bool],
    figure_paths: list[Path],
    total_time: float,
) -> dict[str, Any]:
    best_f1 = feature_results.sort_values(
        ["validation_f1", "validation_pr_auc", "configuration_id"],
        ascending=[False, False, True],
    ).iloc[0]
    fastest = feature_results.sort_values(
        ["validation_inference_time_sec", "feature_count", "configuration_id"]
    ).iloc[0]
    lowest_memory = feature_results.sort_values(
        ["validation_memory_mb", "feature_count", "configuration_id"]
    ).iloc[0]
    smallest = feature_results.sort_values(
        ["model_size_bytes", "validation_f1"], ascending=[True, False]
    ).iloc[0]
    minimum_features = int(feature_results["feature_count"].min())
    lowest_feature = feature_results.loc[
        feature_results["feature_count"].eq(minimum_features)
    ].sort_values(["validation_f1", "configuration_id"], ascending=[False, True]).iloc[0]
    full_visibility = feature_results.loc[
        feature_results["feature_count"].eq(feature_results["feature_count"].max())
    ].iloc[0]
    scenario_count = int(security_scenarios["experiment_id"].nunique()) if not security_scenarios.empty else 0
    return {
        "stage": "V0.5",
        "status": "PASS",
        "scientific_objective": (
            "Establish lightweight detection and computational-resource baselines; "
            "not maximize predictive performance alone."
        ),
        "experiment_id": config["experiment_id"],
        "dataset": {
            "source": "V0.2 processed DataCo splits",
            "split_sizes": {
                name: len(split.features) for name, split in bundle.splits.items()
            },
            "full_feature_count": len(bundle.selected_features),
            "ground_truth": "V0.4 controlled experimental labels",
        },
        "models": list(config["models"]),
        "model_configurations": config["models"],
        "excluded_model": {
            "one_class_svm": (
                "Excluded because kernel scaling on 28,000 training records is not "
                "reliably lightweight for this baseline stage."
            )
        },
        "best_f1_model": {
            **_configuration_summary(best_f1),
            "selection_basis": "highest validation F1; test F1 reported without test-based reselection",
        },
        "fastest_inference_model": {
            **_configuration_summary(fastest),
            "selection_basis": "lowest validation inference time",
        },
        "lowest_memory_model": {
            **_configuration_summary(lowest_memory),
            "selection_basis": "lowest validation fresh-process peak RSS",
        },
        "smallest_model": {
            **_configuration_summary(smallest),
            "selection_basis": "smallest actual serialized artifact",
        },
        "lowest_feature_configuration": {
            **_configuration_summary(lowest_feature),
            "selection_basis": "highest validation F1 among minimum-feature configurations",
        },
        "pareto_method": (
            "Deterministic non-dominance: maximize validation F1; minimize validation "
            "inference time, validation peak process RSS, and feature count."
        ),
        "pareto_candidates": [
            _configuration_summary(row) for _, row in pareto.iterrows()
        ],
        "v0_3_comparison": {
            "reference": "current-environment reproduction of the unchanged V0.3 methodology",
            "reproduction_matches_saved_v0_3_predictions": v03_reproduction_verified,
            "comparison_basis": "same regenerated V0.4 primary test scenario",
            "rows": v03_comparison.to_dict(orient="records"),
            "interpretation": "Results expose trade-offs and do not declare one model universally superior.",
        },
        "security_evaluation": {
            "primary_attack_type": config["evaluation"]["primary_attack_type"],
            "primary_attack_rate": config["evaluation"]["primary_attack_rate"],
            "primary_severity": config["evaluation"]["primary_severity"],
            "scenario_count": scenario_count,
            "evaluation_rows": len(security_scenarios),
            "attack_types": sorted(security_scenarios["attack_type"].unique().tolist()) if not security_scenarios.empty else [],
            "attack_rates": sorted(security_scenarios["attack_rate"].unique().tolist()) if not security_scenarios.empty else [],
            "severity_levels": sorted(security_scenarios["severity"].unique().tolist()) if not security_scenarios.empty else [],
            "v0_4_manifest_verification": scenario_verification,
            "primary_manifest_verification": primary_manifest_verification,
            "v0_4_reference_metrics_reproduced": v04_metric_verification,
            "primary_detector_visible_records": int(
                full_visibility["visible_attacked_records"]
            ),
            "primary_detector_invisible_records": int(
                full_visibility["attacked_records"]
                - full_visibility["visible_attacked_records"]
            ),
            "metrics_scope": (
                "Headline metrics include all controlled labels; visible_attack_comparison.csv "
                "separately evaluates attacked records whose legitimate model inputs changed."
            ),
        },
        "computational_efficiency": {
            "classification": "computational efficiency proxies",
            "proxies": config["energy"]["computational_proxies"],
            "normalized_efficiency_score_created": False,
            "complexity_note": "Complexity indicators are not measurements of energy consumption.",
            "memory_definition": (
                "Peak resident set size in a fresh isolated inference process, including "
                "Python/sklearn, selected input data, and the loaded model."
            ),
            "inference_timing_definition": (
                "Mean wall time per prediction call over configured repeated calls in a "
                "fresh isolated process; model loading is excluded."
            ),
            "configuration_selection_resource_split": "validation",
        },
        "energy_measurement": {
            "directly_measured": False,
            "status": "NOT DIRECTLY MEASURED",
            "electrical_energy_values_reported": False,
        },
        "leakage_audit": leakage_audit,
        "reproducibility": {
            "random_seed": int(config["random_seed"]),
            "model_reload_checks_passed": all(reload_checks.values()),
            "model_reload_checks": reload_checks,
            "timing_variation_expected": True,
        },
        "artifacts": {
            "model_count": len(reload_checks) + 1,
            "figure_count": len(figure_paths),
            "figures": [_portable_path(path) for path in figure_paths],
        },
        "total_execution_time_sec": total_time,
        "limitations": [
            "Controlled synthetic modifications are not equivalent to real attacks.",
            "DataCo has no native cybersecurity labels.",
            "Supervised models learn the configured controlled attack distribution.",
            "Fields excluded by V0.2 remain invisible to every AI model.",
            "Isolated peak process RSS includes interpreter, selected input data, and loaded-library memory.",
            "Wall-clock timings depend on system load and are computational proxies only.",
            "Default decision thresholds differ by model family and are documented rather than tuned on test labels.",
            "No electrical energy consumption was directly measured.",
            "No hyperparameter optimization or future-stage optimizer was used.",
        ],
        "conclusion": (
            "V0.5 establishes candidate configurations and detection-resource trade-offs "
            "for later research-grade feature selection."
        ),
    }


def _metric_columns(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "pr_auc": metrics["pr_auc"],
        "pr_auc_method": metrics["pr_auc_method"],
        "average_precision": metrics["average_precision"],
        "roc_auc": metrics["roc_auc"],
        "true_positives": metrics["true_positives"],
        "true_negatives": metrics["true_negatives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
    }


def _visible_metric_columns(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if metrics is None:
        return {
            "visible_attacked_records": 0,
            "visible_precision": None,
            "visible_recall": None,
            "visible_f1": None,
            "visible_pr_auc": None,
            "visible_pr_auc_method": None,
            "visible_average_precision": None,
            "visible_roc_auc": None,
        }
    return {
        "visible_attacked_records": metrics["attacked_records"],
        "visible_precision": metrics["precision"],
        "visible_recall": metrics["recall"],
        "visible_f1": metrics["f1"],
        "visible_pr_auc": metrics["pr_auc"],
        "visible_pr_auc_method": metrics["pr_auc_method"],
        "visible_average_precision": metrics["average_precision"],
        "visible_roc_auc": metrics["roc_auc"],
    }


def _add_record_ids(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    result = predictions.reset_index(drop=True).copy()
    result.insert(0, "record_id", features["row_id"].to_numpy())
    return result


def _configuration_summary(row: pd.Series) -> dict[str, Any]:
    return {
        "configuration_id": row.get("configuration_id"),
        "model": row["model"],
        "feature_count": int(row["feature_count"]),
        "f1": float(row["f1"]),
        "validation_f1": (
            float(row["validation_f1"]) if "validation_f1" in row else None
        ),
        "pr_auc": float(row["pr_auc"]),
        "average_precision": float(row["average_precision"]),
        "pr_auc_method": row["pr_auc_method"],
        "validation_inference_time_sec": (
            float(row["validation_inference_time_sec"])
            if "validation_inference_time_sec" in row
            else None
        ),
        "validation_memory_mb": (
            float(row["validation_memory_mb"])
            if "validation_memory_mb" in row
            else None
        ),
        "inference_time_sec": float(row["inference_time_sec"]),
        "memory_mb": float(row["memory_mb"]),
        "model_size_kb": float(row["model_size_kb"]),
    }


def _verify_v04_metrics_if_available(
    metrics: dict[str, Any], summary_path: Path
) -> dict[str, Any]:
    if not summary_path.is_file():
        return {"available": False, "matches": None, "path": str(summary_path)}
    previous = json.loads(summary_path.read_text(encoding="utf-8"))["detection"]
    keys = ("precision", "recall", "f1", "pr_auc", "roc_auc")
    matches = all(np.isclose(metrics[key], previous[key], rtol=0.0, atol=1e-12) for key in keys)
    return {
        "available": True,
        "matches": bool(matches),
        "path": _portable_path(summary_path),
    }


def _verify_manifest_if_available(
    generated: pd.DataFrame, existing_path: Path
) -> dict[str, Any]:
    if not existing_path.is_file():
        return {"available": False, "matches": None, "path": str(existing_path)}
    existing = pd.read_csv(existing_path)
    columns = list(MANIFEST_COLUMNS)
    matches = (
        len(generated) == len(existing)
        and all(column in generated for column in columns)
        and all(column in existing for column in columns)
    )
    if matches:
        for column in columns:
            for left, right in zip(generated[column], existing[column]):
                if not _manifest_values_equal(left, right):
                    matches = False
                    break
            if not matches:
                break
    return {
        "available": True,
        "matches": bool(matches),
        "path": _portable_path(existing_path),
        "record_count": len(existing),
        "columns_checked": columns,
    }


def _manifest_values_equal(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    try:
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12))
    except (TypeError, ValueError):
        return str(left) == str(right)


def _threshold_policy(model_name: str) -> str:
    if model_name == "isolation_forest":
        return "sklearn IsolationForest decision_function threshold at zero"
    return "fixed predict_proba threshold 0.5 via sklearn predict; no test tuning"


def _v04_experiment_id(attack_type: str, rate: float, severity: str, seed: int) -> str:
    rate_text = f"{rate:.2f}".replace(".", "p")
    return f"v0_4_{attack_type}_r{rate_text}_{severity.lower()}_s{seed}"


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    result = run_lightweight_evaluation_v05()
    summary = result["summary"]
    print("=== V0.5 Lightweight AI and Computational Efficiency ===")
    print("Status        :", summary["status"])
    print("Best F1       :", summary["best_f1_model"])
    print("Pareto count  :", len(summary["pareto_candidates"]))
    print("Energy        :", summary["energy_measurement"]["status"])
    print("Leakage audit :", summary["leakage_audit"]["status"])
