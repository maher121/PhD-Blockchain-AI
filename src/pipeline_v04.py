"""End-to-end V0.4 controlled cybersecurity evaluation workflow."""

from __future__ import annotations

import datetime as dt
import hashlib
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml

from src.ai.anomaly_detector import IsolationForestBaseline
from src.ai.model_utils import load_processed_dataco, write_json
from src.config import (
    ATTACK_CONFIG_FILE,
    PROJECT_ROOT,
    PROCESSED_DATA_DIR,
    SECURITY_FIGURES_DIR,
    SECURITY_MANIFESTS_DIR,
    SECURITY_PREDICTIONS_DIR,
    SECURITY_RESULTS_DIR,
    V03_MODEL_FILE,
    V04_EXPERIMENT_DIR,
)
from src.security.attack_generator import generate_attack, load_attack_config
from src.security.attack_scenarios import eligible_mask
from src.security.evaluation import (
    detector_visible_modified_count,
    detector_visible_mask,
    evaluate_detection,
    evaluate_paired_detection,
    evaluate_visible_attacks,
    project_attacks_to_feature_space,
    save_security_figures,
)
from src.security.experiments import (
    combined_security_result,
    run_blockchain_integrity_experiment,
)
from src.security.ground_truth import ATTACK_METADATA_COLUMNS, assert_no_attack_metadata


@dataclass(frozen=True)
class SecurityExperimentConfig:
    stage: str = "V0.4"
    random_seed: int | None = None
    attack_rates: tuple[float, ...] = ()
    severity_levels: tuple[str, ...] = ()
    primary_attack_rate: float = 0.05
    primary_severity: str = "MEDIUM"
    attack_config_path: str = str(ATTACK_CONFIG_FILE)


def run_security_evaluation_v04(
    processed_dir: Path | str = PROCESSED_DATA_DIR,
    results_dir: Path | str = SECURITY_RESULTS_DIR,
    model_path: Path | str = V03_MODEL_FILE,
    experiment_dir: Path | str = V04_EXPERIMENT_DIR,
    config: SecurityExperimentConfig | None = None,
) -> dict[str, Any]:
    """Run controlled attacks on clean test records and save V0.4 artifacts."""
    config = config or SecurityExperimentConfig()
    results_dir = Path(results_dir)
    experiment_dir = Path(experiment_dir)
    manifests_dir = results_dir / SECURITY_MANIFESTS_DIR.name
    predictions_dir = results_dir / SECURITY_PREDICTIONS_DIR.name
    figures_dir = results_dir / SECURITY_FIGURES_DIR.name
    for directory in (results_dir, experiment_dir, manifests_dir, predictions_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    total_started = time.perf_counter()
    bundle = load_processed_dataco(processed_dir)
    clean_features = bundle.splits["test"].features
    clean_metadata = bundle.splits["test"].metadata
    reference_features = bundle.splits["train"].features
    reference_metadata = bundle.splits["train"].metadata
    detector = IsolationForestBaseline.load(model_path)
    if detector.feature_names != bundle.selected_features:
        raise RuntimeError("V0.3 model feature contract differs from V0.2 processed features.")
    assert_no_attack_metadata(clean_features)
    attack_config = load_attack_config(config.attack_config_path)
    random_seed = int(
        attack_config["default_random_state"]
        if config.random_seed is None
        else config.random_seed
    )
    attack_rates = tuple(config.attack_rates or attack_config["attack_rates"])
    severity_levels = tuple(config.severity_levels or attack_config["severity_levels"])
    if config.primary_attack_rate <= 0.0:
        raise ValueError("primary_attack_rate must be positive for the combined experiment.")
    if config.primary_severity not in severity_levels:
        raise ValueError("primary_severity must be included in configured severity levels.")
    attack_types = _available_attack_types(clean_metadata, attack_config)
    if not attack_types:
        raise RuntimeError("No configured V0.4 scenario is supported by the test metadata.")

    clean_inference_started = time.perf_counter()
    clean_predictions = detector.predict_frame(clean_features[detector.feature_names])
    clean_predictions.insert(0, "record_id", clean_features["row_id"].to_numpy())
    clean_inference_seconds = time.perf_counter() - clean_inference_started
    clean_predictions.to_csv(predictions_dir / "clean_test_predictions.csv", index=False)

    cache: dict[tuple[str, float, str], dict[str, Any]] = {}

    def run_single(attack_type: str, rate: float, severity: str) -> dict[str, Any]:
        key = (attack_type, float(rate), severity)
        if key not in cache:
            experiment_id = _experiment_id(attack_type, rate, severity, random_seed)
            cache[key] = _run_detection_experiment(
                clean_features=clean_features,
                clean_metadata=clean_metadata,
                reference_features=reference_features,
                reference_metadata=reference_metadata,
                detector=detector,
                clean_predictions=clean_predictions,
                attack_type=attack_type,
                attack_rate=rate,
                severity=severity,
                random_seed=random_seed,
                experiment_id=experiment_id,
                config_path=config.attack_config_path,
                manifest_path=manifests_dir / f"{experiment_id}.csv",
                prediction_path=predictions_dir / f"{experiment_id}_evaluation.csv",
            )
        return cache[key]

    attack_type_rows = [
        _analysis_row(
            run_single(name, config.primary_attack_rate, config.primary_severity)
        )
        for name in attack_types
    ]
    attack_type_analysis = pd.DataFrame(attack_type_rows)

    attack_rate_analysis = pd.DataFrame(
        [
            _analysis_row(run_single(name, rate, config.primary_severity))
            for name in attack_types
            for rate in attack_rates
        ]
    )
    severity_analysis = pd.DataFrame(
        [
            _analysis_row(run_single(name, config.primary_attack_rate, severity))
            for name in attack_types
            for severity in severity_levels
        ]
    )

    primary_id = _experiment_id(
        "mixed", config.primary_attack_rate, config.primary_severity, random_seed
    )
    primary = _run_detection_experiment(
        clean_features=clean_features,
        clean_metadata=clean_metadata,
        reference_features=reference_features,
        reference_metadata=reference_metadata,
        detector=detector,
        clean_predictions=clean_predictions,
        attack_type=attack_types,
        attack_rate=config.primary_attack_rate,
        severity=config.primary_severity,
        random_seed=random_seed,
        experiment_id=primary_id,
        config_path=config.attack_config_path,
        manifest_path=manifests_dir / f"{primary_id}.csv",
        prediction_path=predictions_dir / f"{primary_id}_evaluation.csv",
    )

    repeated = _run_detection_experiment(
        clean_features=clean_features,
        clean_metadata=clean_metadata,
        reference_features=reference_features,
        reference_metadata=reference_metadata,
        detector=detector,
        clean_predictions=clean_predictions,
        attack_type=attack_types,
        attack_rate=config.primary_attack_rate,
        severity=config.primary_severity,
        random_seed=random_seed,
        experiment_id=primary_id,
        config_path=config.attack_config_path,
        manifest_path=manifests_dir / f"{primary_id}.csv",
        prediction_path=predictions_dir / f"{primary_id}_evaluation.csv",
    )
    reproducible = bool(
        primary["attacked_metadata"].equals(repeated["attacked_metadata"])
        and primary["ground_truth"].equals(repeated["ground_truth"])
        and primary["manifest"].equals(repeated["manifest"])
        and primary["attacked_features"].equals(repeated["attacked_features"])
        and primary["predictions"].equals(repeated["predictions"])
        and primary["metrics"] == repeated["metrics"]
    )

    first_attack = primary["manifest"].iloc[0]
    record_id = first_attack["record_id"]
    metadata_row = clean_metadata.loc[clean_metadata["row_id"].eq(record_id)].iloc[0]
    blockchain = run_blockchain_integrity_experiment(
        metadata_row.to_dict(),
        tamper_source_field=first_attack["original_field"],
        modified_value=first_attack["modified_value"],
    )
    ai_row = primary["predictions"].loc[
        primary["predictions"]["record_id"].eq(record_id)
    ].iloc[0]
    combined = combined_security_result(
        ai_detected_anomaly=bool(ai_row["anomaly_label"]),
        blockchain_integrity_valid=bool(blockchain["tampered_chain_verification"]),
    )
    combined["record_id"] = record_id
    combined["transaction_id"] = blockchain["tampered_transaction_id"]
    combined["attack_type"] = first_attack["attack_type"]

    attack_rate_analysis.to_csv(results_dir / "attack_rate_analysis.csv", index=False)
    severity_analysis.to_csv(results_dir / "severity_analysis.csv", index=False)
    attack_type_analysis.to_csv(results_dir / "attack_type_analysis.csv", index=False)
    confusion = pd.DataFrame(
        primary["metrics"]["confusion_matrix"],
        index=["actual_clean", "actual_attack"],
        columns=["predicted_clean", "predicted_attack"],
    )
    confusion.to_csv(results_dir / "confusion_matrix.csv")

    primary["predictions"].to_csv(
        predictions_dir / f"{primary_id}_predictions.csv", index=False
    )
    primary["ground_truth"].to_csv(
        predictions_dir / f"{primary_id}_ground_truth.csv", index=False
    )
    figure_paths = save_security_figures(
        attack_rate_analysis,
        severity_analysis,
        primary["ground_truth"],
        primary["predictions"],
        figures_dir,
    )

    leakage_audit = {
        "status": "PASS",
        "forbidden_columns": list(ATTACK_METADATA_COLUMNS),
        "model_feature_count": len(bundle.selected_features),
        "evidence": "Model inference received only the exact V0.3 selected feature list.",
    }
    contract_checks = {
        "leakage_prevention": leakage_audit["status"] == "PASS",
        "attack_count_achieved": (
            primary["metadata"]["actual_modified_records"]
            == primary["metadata"]["selected_records"]
            and primary["metadata"]["actual_modified_records"] > 0
        ),
        "blockchain_clean_valid": blockchain["clean_chain_verification"] is True,
        "blockchain_tampered_invalid": blockchain["tampered_chain_verification"] is False,
        "reproducibility": reproducible,
        "required_figures_created": len(figure_paths) == 6,
        "required_artifacts_created": all(
            path.exists()
            for path in (
                results_dir / "attack_rate_analysis.csv",
                results_dir / "severity_analysis.csv",
                results_dir / "attack_type_analysis.csv",
                results_dir / "confusion_matrix.csv",
                manifests_dir / f"{primary_id}.csv",
                predictions_dir / "clean_test_predictions.csv",
                predictions_dir / f"{primary_id}_evaluation.csv",
            )
        ),
    }
    contract_status = "PASS" if all(contract_checks.values()) else "FAIL"
    total_seconds = time.perf_counter() - total_started
    summary = {
        "stage": "V0.4",
        "status": contract_status,
        "pipeline_execution_status": "PASS",
        "contract_validation_status": contract_status,
        "contract_checks": contract_checks,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": {
            "source": "Real operational DataCo records from the V0.2 test split",
            "native_cybersecurity_labels": False,
            "ground_truth_kind": "controlled experimental ground truth",
            "test_sample_count": int(len(clean_features)),
            "clean_sample_count": int(primary["ground_truth"]["is_attack"].eq(0).sum()),
            "attacked_sample_count": int(primary["ground_truth"]["is_attack"].sum()),
            "feature_count": len(bundle.selected_features),
            "processed_dataset_row_cap": bundle.dataset_metadata.get("extra", {}).get(
                "dataset_used_rows"
            ),
            "raw_file_sha256": bundle.dataset_metadata.get("dataset", {}).get("file_sha256"),
        },
        "attack": {
            "attack_types": attack_types,
            "attack_rates": list(attack_rates),
            "severity_levels": list(severity_levels),
            "primary_experiment_id": primary_id,
            "primary_attack_rate_requested": config.primary_attack_rate,
            "primary_attack_rate_achieved": primary["metadata"]["attack_rate_achieved"],
            "primary_severity": config.primary_severity,
            "primary_eligible_records": primary["metadata"]["eligible_records"],
            "primary_selected_records": primary["metadata"]["selected_records"],
            "primary_actual_modified_records": primary["metadata"]["actual_modified_records"],
            "primary_per_attack_type": primary["metadata"]["per_attack_type"],
            "primary_manifest": _portable_path(manifests_dir / f"{primary_id}.csv"),
            "detector_visible_modified_records": primary["detector_visible_modified_records"],
            "detector_invisible_modified_records": (
                primary["metadata"]["actual_modified_records"]
                - primary["detector_visible_modified_records"]
            ),
        },
        "detection": primary["metrics"],
        "visible_attack_detection": primary["visible_metrics"],
        "paired_clean_vs_attacked_detection": primary["paired_detection"],
        "blockchain": {key: value for key, value in blockchain.items() if key not in {"chain", "tampered_chain"}},
        "combined_security": combined,
        "performance": {
            "clean_ai_inference_time_seconds": clean_inference_seconds,
            "ai_inference_time_seconds": primary["inference_time_seconds"],
            "attack_generation_time_seconds": primary["attack_generation_time_seconds"],
            "feature_projection_time_seconds": primary["feature_projection_time_seconds"],
            "blockchain_verification_time_seconds": blockchain["verification_time_seconds"],
            "total_pipeline_time_seconds": total_seconds,
        },
        "leakage_audit": leakage_audit,
        "reproducibility": {
            "status": "PASS" if reproducible else "FAIL",
            "random_seed": random_seed,
            "same_configuration_reproduced_full_evaluation": reproducible,
            "model_sha256": _sha256(Path(model_path)),
            "v0_4_source_sha256": _source_fingerprint(),
        },
        "research_interpretation": (
            "Final-label metrics compare V0.3 flags with controlled ground truth, while the paired "
            "clean-versus-attacked analysis isolates newly induced flags and score changes. The "
            "observed attack-induced detection rate is descriptive, not evidence of universal "
            "effectiveness. AI tests statistical behavior; the simulated blockchain tests "
            "post-sealing integrity, so their roles are complementary."
        ),
        "limitations": [
            "Synthetic controlled attacks are not equivalent to real attacks.",
            "Attack distributions are determined by experiment parameters.",
            "Isolation Forest remains an unsupervised baseline with an existing decision threshold.",
            "DataCo is an operational supply-chain dataset, not a dedicated cybersecurity dataset.",
            "Results measure controlled detection behavior and do not establish universal cybersecurity effectiveness.",
            "Fields excluded from V0.3 for leakage safety may be modified but remain intentionally invisible to AI.",
            "The blockchain is a single-authority hash-chain simulation without consensus or signatures.",
            "The experiment uses a deterministic 40,000-row DataCo cap and its 6,000-row test split.",
        ],
    }
    write_json(summary, results_dir / "security_summary.json")
    write_json(leakage_audit, results_dir / "leakage_audit.json")

    experiment_payloads = {
        "attack_configuration.json": attack_config,
        "dataset_metadata.json": bundle.dataset_metadata,
        "feature_list.json": {
            "stage": "V0.4",
            "selected_feature_count": len(bundle.selected_features),
            "selected_features": bundle.selected_features,
            "attack_metadata_included": False,
        },
        "model_configuration.json": {
            "artifact_path": _portable_path(Path(model_path)),
            "artifact_sha256": _sha256(Path(model_path)),
            "artifact_version": detector.artifact_version,
            "parameters": detector.model_parameters,
            "trained_on": "V0.2 clean training split",
        },
        "experiment_results.json": summary,
        "config.json": {
            **asdict(config),
            "attack_config_path": _portable_path(Path(config.attack_config_path)),
            "resolved_random_seed": random_seed,
            "resolved_attack_rates": list(attack_rates),
            "resolved_severity_levels": list(severity_levels),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "pyyaml_version": yaml.__version__,
        },
    }
    for filename, payload in experiment_payloads.items():
        write_json(payload, experiment_dir / filename)

    return {
        "summary": summary,
        "primary": primary,
        "attack_type_analysis": attack_type_analysis,
        "attack_rate_analysis": attack_rate_analysis,
        "severity_analysis": severity_analysis,
        "blockchain": blockchain,
        "combined_security": combined,
        "figure_paths": figure_paths,
        "paths": {
            "results": results_dir,
            "experiment": experiment_dir,
            "manifests": manifests_dir,
            "predictions": predictions_dir,
        },
    }


def _run_detection_experiment(
    *,
    clean_features: pd.DataFrame,
    clean_metadata: pd.DataFrame,
    reference_features: pd.DataFrame,
    reference_metadata: pd.DataFrame,
    detector: IsolationForestBaseline,
    clean_predictions: pd.DataFrame,
    attack_type: str | list[str],
    attack_rate: float,
    severity: str,
    random_seed: int,
    experiment_id: str,
    config_path: Path | str,
    manifest_path: Path,
    prediction_path: Path,
) -> dict[str, Any]:
    generation_started = time.perf_counter()
    attacked_metadata, metadata, ground_truth = generate_attack(
        clean_metadata,
        attack_type,
        attack_rate,
        severity,
        random_seed,
        experiment_id=experiment_id,
        config_path=config_path,
    )
    manifest = metadata.pop("manifest")
    generation_seconds = time.perf_counter() - generation_started
    projection_started = time.perf_counter()
    attacked_features = project_attacks_to_feature_space(
        clean_features,
        clean_metadata,
        attacked_metadata,
        manifest,
        reference_features=reference_features,
        reference_metadata=reference_metadata,
    )
    projection_seconds = time.perf_counter() - projection_started
    assert_no_attack_metadata(attacked_features)
    inference_started = time.perf_counter()
    predictions = detector.predict_frame(attacked_features[detector.feature_names])
    predictions.insert(0, "record_id", attacked_features["row_id"].to_numpy())
    inference_seconds = time.perf_counter() - inference_started
    metrics = evaluate_detection(ground_truth, predictions)
    paired_detection = evaluate_paired_detection(
        ground_truth, clean_predictions, predictions
    )
    visible = detector_visible_mask(clean_features, attacked_features)
    visible_metrics = evaluate_visible_attacks(ground_truth, predictions, visible)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    evaluation = ground_truth[["record_id", "is_attack", "attack_type"]].merge(
        clean_predictions[
            ["record_id", "anomaly_label", "anomaly_score"]
        ].rename(
            columns={
                "anomaly_label": "clean_anomaly_label",
                "anomaly_score": "clean_anomaly_score",
            }
        ),
        on="record_id",
        validate="one_to_one",
    )
    evaluation = evaluation.merge(
        predictions[["record_id", "anomaly_label", "anomaly_score"]].rename(
            columns={
                "anomaly_label": "attacked_anomaly_label",
                "anomaly_score": "attacked_anomaly_score",
            }
        ),
        on="record_id",
        validate="one_to_one",
    )
    evaluation["model_input_changed"] = visible
    evaluation["anomaly_score_delta"] = (
        evaluation["attacked_anomaly_score"] - evaluation["clean_anomaly_score"]
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(prediction_path, index=False)
    return {
        "experiment_id": experiment_id,
        "attacked_metadata": attacked_metadata,
        "attacked_features": attacked_features,
        "metadata": metadata,
        "manifest": manifest,
        "ground_truth": ground_truth,
        "predictions": predictions,
        "metrics": metrics,
        "visible_metrics": visible_metrics,
        "paired_detection": paired_detection,
        "detector_visible_modified_records": detector_visible_modified_count(
            clean_features, attacked_features, ground_truth
        ),
        "attack_generation_time_seconds": generation_seconds,
        "feature_projection_time_seconds": projection_seconds,
        "inference_time_seconds": inference_seconds,
    }


def _available_attack_types(
    metadata: pd.DataFrame, attack_config: dict[str, Any]
) -> list[str]:
    return [
        name
        for name, definition in attack_config["scenarios"].items()
        if bool(eligible_mask(metadata, definition).any())
    ]


def _analysis_row(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    metadata = result["metadata"]
    return {
        "experiment_id": result["experiment_id"],
        "attack_type": metadata["attack_types"][0],
        "attack_rate": metadata["configured_attack_rate"],
        "attack_rate_achieved": metadata["attack_rate_achieved"],
        "severity": metadata["severity"],
        "eligible_records": metadata["eligible_records"],
        "selected_records": metadata["selected_records"],
        "actual_modified_records": metadata["actual_modified_records"],
        "detector_visible_modified_records": result["detector_visible_modified_records"],
        "metrics_scope": (
            "attacked_inputs_changed"
            if result["detector_visible_modified_records"]
            else "baseline_overlap_only_model_inputs_unchanged"
        ),
        "severity_interpretation": (
            "nominal_transition_variant_not_ordered_magnitude"
            if metadata["attack_types"][0]
            in {"transaction_status_manipulation", "geographic_route_manipulation"}
            else "ordered_configured_magnitude"
        ),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "pr_auc": metrics["pr_auc"],
        "average_precision": metrics["average_precision"],
        "roc_auc": metrics["roc_auc"],
        "visible_recall": (
            result["visible_metrics"]["recall"] if result["visible_metrics"] else None
        ),
        "newly_detected_after_attack": result["paired_detection"][
            "newly_detected_after_attack"
        ],
        "attack_induced_detection_rate": result["paired_detection"][
            "attack_induced_detection_rate"
        ],
        "mean_anomaly_score_delta": result["paired_detection"][
            "mean_anomaly_score_delta"
        ],
        "true_positives": metrics["true_positives"],
        "true_negatives": metrics["true_negatives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
    }


def _experiment_id(attack_type: str, rate: float, severity: str, seed: int) -> str:
    rate_text = f"{rate:.2f}".replace(".", "p")
    return f"v0_4_{attack_type}_r{rate_text}_{severity.lower()}_s{seed}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _source_fingerprint() -> str:
    paths = [
        Path(__file__),
        PROJECT_ROOT / "config" / "attack_scenarios.yaml",
        PROJECT_ROOT / "src" / "security" / "attack_generator.py",
        PROJECT_ROOT / "src" / "security" / "attack_scenarios.py",
        PROJECT_ROOT / "src" / "security" / "evaluation.py",
        PROJECT_ROOT / "src" / "security" / "experiments.py",
        PROJECT_ROOT / "src" / "security" / "ground_truth.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(_portable_path(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    result = run_security_evaluation_v04()
    summary = result["summary"]
    print("=== V0.4 Controlled Cybersecurity Evaluation ===")
    print("Status      :", summary["status"])
    print("Test samples:", summary["dataset"]["test_sample_count"])
    print("Attacked    :", summary["dataset"]["attacked_sample_count"])
    print("F1          :", round(summary["detection"]["f1"], 4))
    print("Blockchain :", summary["blockchain"]["integrity_failure_detected"])
    print("Leakage     :", summary["leakage_audit"]["status"])
