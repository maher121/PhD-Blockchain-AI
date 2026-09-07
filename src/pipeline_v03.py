"""End-to-end V0.3 unsupervised Isolation Forest baseline workflow."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from src.ai.anomaly_detector import IsolationForestBaseline
from src.ai.evaluation import (
    build_leakage_audit,
    contamination_sensitivity,
    save_baseline_figures,
    seed_stability,
    summarize_predictions,
    top_anomalies,
)
from src.ai.model_utils import (
    ProcessedDataBundle,
    build_feature_manifest,
    determine_label_situation,
    load_processed_dataco,
    write_json,
)
from src.config import (
    AI_BASELINE_DIR,
    AI_BASELINE_FIGURES_DIR,
    AI_BASELINE_PREDICTIONS_DIR,
    DATASET_METADATA_FILE,
    GLOBAL_SEED,
    PROCESSED_DATA_DIR,
    V03_CONTAMINATION_VALUES,
    V03_EXPERIMENT_DIR,
    V03_ISOLATION_FOREST_PARAMS,
    V03_MODEL_FILE,
    V03_STABILITY_SEEDS,
)
from src.energy.measurement import ExecutionTimer


@dataclass
class BaselineConfig:
    stage: str = "V0.3"
    model_name: str = "IsolationForest"
    random_seed: int = GLOBAL_SEED
    model_parameters: dict[str, Any] = field(
        default_factory=lambda: dict(V03_ISOLATION_FOREST_PARAMS)
    )
    contamination_sensitivity_values: tuple[float, ...] = V03_CONTAMINATION_VALUES
    stability_seeds: tuple[int, ...] = V03_STABILITY_SEEDS
    top_anomaly_count: int = 50
    evaluation_split: str = "test"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contamination_sensitivity_values"] = list(
            self.contamination_sensitivity_values
        )
        payload["stability_seeds"] = list(self.stability_seeds)
        payload.update(
            {
                "label_mode": "unsupervised",
                "contamination_note": (
                    "Sensitivity values are bounded scenario assumptions only; they do not "
                    "represent measured anomaly prevalence or ground truth."
                ),
                "preprocessing": (
                    "Reuse V0.2 train-fitted processed features; saved sklearn pipeline "
                    "contains an exact-column selector and Isolation Forest."
                ),
                "score_semantics": (
                    "anomaly_score=-score_samples (higher is more anomalous); normalized "
                    "score is not a calibrated risk probability"
                ),
                "python_version": platform.python_version(),
                "package_versions": {
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scikit_learn": sklearn.__version__,
                },
            }
        )
        return payload


def run_baseline_anomaly_detection_v03(
    processed_dir: Path | str = PROCESSED_DATA_DIR,
    results_dir: Path | str = AI_BASELINE_DIR,
    model_path: Path | str = V03_MODEL_FILE,
    experiment_dir: Path | str = V03_EXPERIMENT_DIR,
    config: BaselineConfig | None = None,
) -> dict[str, Any]:
    """Run V0.3 from V0.2 processed splits and save every required artifact."""
    config = config or BaselineConfig()
    results_dir = Path(results_dir)
    model_path = Path(model_path)
    experiment_dir = Path(experiment_dir)
    figures_dir = results_dir / AI_BASELINE_FIGURES_DIR.name
    predictions_dir = results_dir / AI_BASELINE_PREDICTIONS_DIR.name
    predictions_path = predictions_dir / "isolation_forest_v0_3_predictions.csv"

    with ExecutionTimer(label="v0.3-total-compute") as total_timer:
        bundle = load_processed_dataco(
            processed_dir,
            metadata_path=Path(processed_dir) / DATASET_METADATA_FILE.name,
        )
        label_situation = determine_label_situation(bundle)
        if label_situation["supervised_metrics_allowed"]:
            raise RuntimeError(
                "This V0.3 implementation is configured for the audited unlabeled DataCo baseline."
            )

        feature_manifest = build_feature_manifest(bundle)
        leakage_audit = build_leakage_audit(
            bundle.selected_features, bundle.dataset_metadata
        )
        if leakage_audit["overall_status"] != "PASS":
            raise RuntimeError("V0.3 leakage audit failed; refusing to train.")

        train_features = bundle.splits["train"].features
        with ExecutionTimer(label="v0.3-isolation-forest-training") as training_timer:
            detector = IsolationForestBaseline(
                bundle.selected_features, **config.model_parameters
            ).fit(train_features)

        model_path = detector.save(model_path)

        prediction_blocks: list[pd.DataFrame] = []
        inference_measurements: dict[str, dict[str, Any]] = {}
        split_predictions: dict[str, pd.DataFrame] = {}
        for split_name, split in bundle.splits.items():
            with ExecutionTimer(label=f"v0.3-{split_name}-inference") as inference_timer:
                predictions = detector.predict_frame(split.features)
            inference_measurements[split_name] = inference_timer.measurement.to_dict()
            predictions.insert(0, "row_id", split.features["row_id"].to_numpy())
            predictions.insert(0, "split", split_name)
            split_predictions[split_name] = predictions
            prediction_blocks.append(_add_safe_traceability(predictions, split.metadata))

        all_predictions = pd.concat(prediction_blocks, ignore_index=True)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        all_predictions.to_csv(predictions_path, index=False)

        loaded_detector = IsolationForestBaseline.load(model_path)
        reloaded_test = loaded_detector.predict_frame(bundle.splits["test"].features)
        original_test = split_predictions["test"].drop(columns=["split", "row_id"])
        reload_verified = bool(
            np.array_equal(
                reloaded_test["anomaly_label"].to_numpy(),
                original_test["anomaly_label"].to_numpy(),
            )
            and np.allclose(
                reloaded_test["anomaly_score"].to_numpy(),
                original_test["anomaly_score"].to_numpy(),
                rtol=0.0,
                atol=0.0,
            )
        )
        if not reload_verified:
            raise RuntimeError("Saved-model reload inference did not reproduce predictions.")

        evaluation_features = {
            name: bundle.splits[name].features for name in ("validation", "test")
        }
        sensitivity = contamination_sensitivity(
            train_features,
            evaluation_features,
            bundle.selected_features,
            config.contamination_sensitivity_values,
            config.model_parameters,
        )
        sensitivity_path = results_dir / "contamination_sensitivity.csv"
        sensitivity_path.parent.mkdir(parents=True, exist_ok=True)
        sensitivity.to_csv(sensitivity_path, index=False)

        stability = seed_stability(
            train_features,
            bundle.splits["test"].features,
            bundle.selected_features,
            config.stability_seeds,
            config.model_parameters,
        )
        stability_path = results_dir / "seed_stability.csv"
        stability.to_csv(stability_path, index=False)

        top = top_anomalies(
            split_predictions["test"],
            bundle.splits["test"].metadata,
            config.top_anomaly_count,
        )
        top_path = results_dir / "top_anomalies.csv"
        top.to_csv(top_path, index=False)

        figure_paths = save_baseline_figures(
            split_predictions["test"],
            sensitivity,
            bundle.splits["test"].features,
            bundle.selected_features,
            figures_dir,
        )

    config_payload = config.to_dict()
    config_payload["dataset"] = {
        "processed_data_directory": str(Path(processed_dir)),
        "metadata_path": str(Path(processed_dir) / DATASET_METADATA_FILE.name),
        "raw_file_sha256": bundle.dataset_metadata.get("dataset", {}).get("file_sha256"),
        "split_strategy": bundle.dataset_metadata.get("split", {}).get("strategy"),
    }

    training_measurement = training_timer.measurement.to_dict()
    inference_seconds = float(
        sum(item["wall_time_seconds"] for item in inference_measurements.values())
    )
    test_summary = summarize_predictions(split_predictions["test"])
    per_split_summary = {
        name: summarize_predictions(predictions)
        for name, predictions in split_predictions.items()
    }
    stability_summary = {
        "seeds": [int(value) for value in stability["seed"]],
        "minimum_jaccard_with_reference": float(stability["jaccard_with_reference"].min()),
        "minimum_score_rank_correlation_with_reference": float(
            stability["score_rank_correlation_with_reference"].min()
        ),
    }
    performance = {
        "training_time_seconds": training_measurement["wall_time_seconds"],
        "inference_time_seconds_all_splits": inference_seconds,
        "total_execution_time_seconds": total_timer.measurement.wall_time_seconds,
        "training_measurement": training_measurement,
        "inference_measurements": inference_measurements,
        "total_compute_measurement": total_timer.measurement.to_dict(),
        "energy_consumption": {
            "reported": False,
            "reason": "No direct energy instrumentation was used; no energy value is inferred.",
        },
    }
    metrics = {
        "evaluation_mode": "unsupervised_anomaly_detection",
        "evaluation_split": config.evaluation_split,
        "label_situation": label_situation,
        "supervised_metrics": {
            "calculated": False,
            "reason": label_situation.get("reason", "No valid anomaly ground truth."),
        },
        "anomaly_detection": per_split_summary,
        "stability": stability_summary,
        "performance": performance,
    }
    model_info = {
        "name": config.model_name,
        "artifact_version": IsolationForestBaseline.artifact_version,
        "artifact_path": str(model_path),
        "parameters": detector.model_parameters,
        "random_seed": config.random_seed,
        "selected_feature_count": len(bundle.selected_features),
        "embedded_preprocessing": "exact V0.2 processed-feature selector",
        "reload_and_inference_verification": "PASS" if reload_verified else "FAIL",
        "additional_models": [],
        "additional_models_note": (
            "No additional model was added: Isolation Forest is the intentionally simple, "
            "computationally bounded primary reference for V0.3."
        ),
    }
    experiment_metadata = {
        "stage": "V0.3",
        "objective": "Reproducible unsupervised anomaly-detection baseline",
        "dataset": bundle.dataset_metadata.get("dataset", {}),
        "split": bundle.dataset_metadata.get("split", {}),
        "label_situation": label_situation,
        "evaluation_scope": "Test-split anomaly behavior; no cyber/classification claims",
        "conclusion": (
            "The V0.3 baseline establishes a reproducible anomaly-detection reference point "
            "for subsequent optimization and lightweight-model experiments."
        ),
        "limitations": [
            "DataCo has no native cybersecurity or anomaly ground-truth labels.",
            "Isolation Forest is unsupervised; its flags are not confirmed attacks.",
            "Contamination scenarios are assumptions, not measured prevalence.",
            "Anomaly interpretation is operational/business-oriented and requires domain review.",
            "The V0.2 processed experiment uses a deterministic 40,000-row cap.",
        ],
    }
    baseline_report = {
        "stage": "V0.3",
        "status": "PASS",
        "dataset": {
            "number_of_samples": int(sum(len(item.features) for item in bundle.splits.values())),
            "number_of_features": len(bundle.selected_features),
            "split_sizes": {
                name: int(len(item.features)) for name, item in bundle.splits.items()
            },
            "label_situation": label_situation,
        },
        "model": model_info,
        "anomaly_detection": test_summary,
        "performance": performance,
        "evaluation": metrics["supervised_metrics"],
        "leakage_audit": leakage_audit["overall_status"],
        "reproducibility": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "random_seed": config.random_seed,
            "configuration_path": str(results_dir / "config.json"),
            "model_reload_verified": reload_verified,
        },
        "research_interpretation": experiment_metadata["conclusion"],
        "limitations": experiment_metadata["limitations"],
    }

    output_payloads = {
        "config.json": config_payload,
        "metrics.json": metrics,
        "feature_list.json": feature_manifest,
        "model_info.json": model_info,
        "experiment_metadata.json": experiment_metadata,
    }
    for filename, payload in output_payloads.items():
        write_json(payload, results_dir / filename)
        write_json(payload, experiment_dir / filename)
    write_json(leakage_audit, results_dir / "leakage_audit.json")
    write_json(baseline_report, results_dir / "baseline_report.json")

    return {
        "bundle": bundle,
        "detector": detector,
        "label_situation": label_situation,
        "feature_manifest": feature_manifest,
        "leakage_audit": leakage_audit,
        "predictions": all_predictions,
        "sensitivity": sensitivity,
        "stability": stability,
        "top_anomalies": top,
        "metrics": metrics,
        "baseline_report": baseline_report,
        "model_info": model_info,
        "paths": {
            "model": model_path,
            "predictions": predictions_path,
            "sensitivity": sensitivity_path,
            "stability": stability_path,
            "top_anomalies": top_path,
            "figures": figure_paths,
            "results": results_dir,
            "experiment": experiment_dir,
        },
    }


def _add_safe_traceability(
    predictions: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    safe_columns = [
        "row_id",
        "Order Id",
        "Order Item Id",
        "order date (DateOrders)",
        "Type",
        "Market",
        "Shipping Mode",
    ]
    available = [column for column in safe_columns if column in metadata.columns]
    return predictions.merge(
        metadata[available], on="row_id", how="left", validate="one_to_one"
    )


if __name__ == "__main__":
    from src.logging_config import setup_logging

    setup_logging()
    result = run_baseline_anomaly_detection_v03()
    report = result["baseline_report"]
    print("=== V0.3 Baseline AI Anomaly Detection ===")
    print("Status       :", report["status"])
    print("Samples      :", report["dataset"]["number_of_samples"])
    print("Features     :", report["dataset"]["number_of_features"])
    print("Test flags   :", report["anomaly_detection"]["number_of_detected_anomalies"])
    print("Flagged %    :", round(report["anomaly_detection"]["anomaly_percentage"], 3))
    print("Leakage audit:", report["leakage_audit"])
    print("Model        :", result["paths"]["model"])
