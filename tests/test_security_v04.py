"""Miniature-data tests for V0.4 controlled security experiments."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.ai.model_utils import validate_selected_features
from src.ai.anomaly_detector import IsolationForestBaseline
from src.pipeline_v04 import SecurityExperimentConfig, run_security_evaluation_v04
from src.security.attack_generator import generate_attack, restore_original
from src.security.evaluation import (
    detector_visible_modified_count,
    evaluate_detection,
    evaluate_paired_detection,
    project_attacks_to_feature_space,
)
from src.security.experiments import (
    combined_security_result,
    run_blockchain_integrity_experiment,
)
from src.security.ground_truth import ATTACK_METADATA_COLUMNS, assert_no_attack_metadata
from src.security.integrity import detect_tampering, verify_chain_integrity


@pytest.fixture()
def security_frame() -> pd.DataFrame:
    count = 200
    return pd.DataFrame(
        {
            "row_id": np.arange(count),
            "Order Id": 10_000 + np.arange(count),
            "Order Item Id": 20_000 + np.arange(count),
            "Order Customer Id": 30_000 + np.arange(count),
            "Product Card Id": 40_000 + (np.arange(count) % 7),
            "Order Item Quantity": 2 + (np.arange(count) % 5),
            "Order Item Total": 25.0 + np.arange(count),
            "Days for shipping (real)": 2 + (np.arange(count) % 4),
            "Order Status": np.where(np.arange(count) % 2, "PENDING", "COMPLETE"),
            "order date (DateOrders)": pd.date_range(
                "2016-01-01", periods=count, freq="12h"
            ).astype(str),
            "Order Region": np.where(np.arange(count) % 2, "West", "East"),
        }
    )


@pytest.fixture()
def processed_features(security_frame: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.to_datetime(security_frame["order date (DateOrders)"])
    quantity = security_frame["Order Item Quantity"].astype(float)
    total = security_frame["Order Item Total"].astype(float)
    return pd.DataFrame(
        {
            "row_id": security_frame["row_id"],
            "order_item_quantity": (quantity - quantity.mean()) / quantity.std(ddof=0),
            "order_item_total": (total - total.mean()) / total.std(ddof=0),
            "year": timestamp.dt.year.astype(float),
            "month": timestamp.dt.month.astype(float),
            "day": timestamp.dt.day.astype(float),
            "day_of_week": timestamp.dt.dayofweek.astype(float),
            "hour": timestamp.dt.hour.astype(float),
            "is_weekend": timestamp.dt.dayofweek.ge(5).astype(float),
        }
    )


def test_attack_generator_rate_and_ground_truth(security_frame: pd.DataFrame) -> None:
    attacked, metadata, truth = generate_attack(
        security_frame, "quantity_manipulation", 0.10, "MEDIUM", 42
    )
    assert attacked is not security_frame
    assert metadata["eligible_records"] == 200
    assert metadata["selected_records"] == 20
    assert metadata["actual_modified_records"] == 20
    assert metadata["attack_rate_achieved"] == pytest.approx(0.10)
    assert truth["is_attack"].sum() == 20
    assert truth.loc[truth["is_attack"].eq(0), "attack_type"].isna().all()


@pytest.mark.parametrize("attack_rate, expected", [(0.01, 2), (0.03, 6), (0.05, 10), (0.10, 20)])
def test_configured_attack_rates(
    security_frame: pd.DataFrame, attack_rate: float, expected: int
) -> None:
    _, metadata, _ = generate_attack(
        security_frame, "value_manipulation", attack_rate, "LOW", 42
    )
    assert metadata["actual_modified_records"] == expected


def test_severity_changes_magnitude_but_not_selection(security_frame: pd.DataFrame) -> None:
    _, low, _ = generate_attack(
        security_frame, "value_manipulation", 0.10, "LOW", 42
    )
    _, high, _ = generate_attack(
        security_frame, "value_manipulation", 0.10, "HIGH", 42
    )
    low_manifest = low["manifest"]
    high_manifest = high["manifest"]
    assert low_manifest["record_id"].tolist() == high_manifest["record_id"].tolist()
    low_change = (
        low_manifest["modified_value"].astype(float)
        - low_manifest["original_value"].astype(float)
    ).abs().mean()
    high_change = (
        high_manifest["modified_value"].astype(float)
        - high_manifest["original_value"].astype(float)
    ).abs().mean()
    assert high_change > low_change


def test_attack_selection_is_deterministic(security_frame: pd.DataFrame) -> None:
    first = generate_attack(security_frame, "mixed", 0.10, "MEDIUM", 42)
    second = generate_attack(security_frame, "mixed", 0.10, "MEDIUM", 42)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1]["manifest"], second[1]["manifest"])
    pd.testing.assert_frame_equal(first[2], second[2])


def test_mixed_attacks_are_disjoint_and_traceable(security_frame: pd.DataFrame) -> None:
    _, metadata, truth = generate_attack(security_frame, "mixed", 0.10, "HIGH", 42)
    manifest = metadata["manifest"]
    assert len(manifest) == 20
    assert manifest["record_id"].is_unique
    assert manifest["attack_type"].nunique() == 7
    assert manifest["original_field"].notna().all()
    assert manifest["original_value"].notna().all()
    assert manifest["modified_value"].notna().all()
    assert set(truth.loc[truth["is_attack"].eq(1), "record_id"]) == set(
        manifest["record_id"]
    )


def test_restore_original_recovers_input(
    security_frame: pd.DataFrame, tmp_path
) -> None:
    attacked, metadata, _ = generate_attack(
        security_frame, "mixed", 0.20, "HIGH", 42
    )
    restored = restore_original(attacked, metadata["manifest"])
    pd.testing.assert_frame_equal(restored, security_frame)
    manifest_path = tmp_path / "manifest.csv"
    metadata["manifest"].to_csv(manifest_path, index=False)
    reloaded = pd.read_csv(manifest_path)
    pd.testing.assert_frame_equal(restore_original(attacked, reloaded), security_frame)


def test_unsupported_scenario_has_zero_eligible_records() -> None:
    frame = pd.DataFrame({"row_id": range(10), "unrelated": range(10)})
    attacked, metadata, truth = generate_attack(
        frame, "quantity_manipulation", 0.10, "LOW", 42
    )
    pd.testing.assert_frame_equal(attacked, frame)
    assert metadata["eligible_records"] == 0
    assert metadata["actual_modified_records"] == 0
    assert truth["is_attack"].sum() == 0


@pytest.mark.parametrize("forbidden", ATTACK_METADATA_COLUMNS)
def test_attack_metadata_never_enters_ai_features(forbidden: str) -> None:
    with pytest.raises(ValueError, match="Attack metadata"):
        assert_no_attack_metadata(pd.DataFrame({"legitimate": [1.0], forbidden: [0]}))
    with pytest.raises(ValueError, match="Leakage-prone"):
        validate_selected_features(["legitimate", forbidden])


def test_projection_changes_only_legitimate_feature_space(
    security_frame: pd.DataFrame, processed_features: pd.DataFrame
) -> None:
    attacked, metadata, truth = generate_attack(
        security_frame, "quantity_manipulation", 0.10, "HIGH", 42
    )
    projected = project_attacks_to_feature_space(
        processed_features, security_frame, attacked, metadata["manifest"]
    )
    assert_no_attack_metadata(projected)
    assert detector_visible_modified_count(processed_features, projected, truth) == 20
    clean_rows = truth["is_attack"].eq(0)
    pd.testing.assert_frame_equal(
        projected.loc[clean_rows].reset_index(drop=True),
        processed_features.loc[clean_rows].reset_index(drop=True),
    )


def test_excluded_status_attack_is_not_projected(
    security_frame: pd.DataFrame, processed_features: pd.DataFrame
) -> None:
    attacked, metadata, truth = generate_attack(
        security_frame, "transaction_status_manipulation", 0.10, "LOW", 42
    )
    projected = project_attacks_to_feature_space(
        processed_features, security_frame, attacked, metadata["manifest"]
    )
    pd.testing.assert_frame_equal(projected, processed_features)
    assert detector_visible_modified_count(processed_features, projected, truth) == 0


def test_ai_evaluation_uses_experimental_ground_truth() -> None:
    truth = pd.DataFrame({"record_id": range(4), "is_attack": [0, 0, 1, 1]})
    predictions = pd.DataFrame(
        {
            "record_id": range(4),
            "anomaly_label": [0, 1, 1, 0],
            "anomaly_score": [0.1, 0.8, 0.9, 0.2],
        }
    )
    metrics = evaluate_detection(truth, predictions)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["pr_auc"] is not None
    assert metrics["roc_auc"] is not None


def test_ai_evaluation_aligns_by_record_id() -> None:
    truth = pd.DataFrame({"record_id": [10, 11], "is_attack": [0, 1]})
    predictions = pd.DataFrame(
        {
            "record_id": [11, 10],
            "anomaly_label": [1, 0],
            "anomaly_score": [0.9, 0.1],
        }
    )
    metrics = evaluate_detection(truth, predictions)
    assert metrics["f1"] == pytest.approx(1.0)


def test_paired_evaluation_separates_preexisting_flags() -> None:
    truth = pd.DataFrame(
        {"record_id": [0, 1, 2, 3], "is_attack": [0, 0, 1, 1]}
    )
    clean = pd.DataFrame(
        {
            "record_id": [0, 1, 2, 3],
            "anomaly_label": [0, 0, 1, 0],
            "anomaly_score": [0.1, 0.2, 0.8, 0.3],
        }
    )
    attacked = clean.copy()
    attacked.loc[attacked["record_id"].eq(3), ["anomaly_label", "anomaly_score"]] = [1, 0.9]
    paired = evaluate_paired_detection(truth, clean, attacked)
    assert paired["already_flagged_before_attack"] == 1
    assert paired["newly_detected_after_attack"] == 1
    assert paired["attack_induced_detection_rate"] == pytest.approx(0.5)


def test_blockchain_tampering_detection(security_frame: pd.DataFrame) -> None:
    result = run_blockchain_integrity_experiment(security_frame.iloc[0].to_dict())
    assert result["clean_chain_verification"] is True
    assert result["tampered_chain_verification"] is False
    assert result["integrity_failure_detected"] is True
    assert verify_chain_integrity(result["chain"]) is True
    assert detect_tampering(result["tampered_chain"])["tampered"] is True


def test_combined_security_experiment_reports_both_signals() -> None:
    result = combined_security_result(
        ai_detected_anomaly=True, blockchain_integrity_valid=False
    )
    assert result == {
        "ai_detected_anomaly": True,
        "blockchain_integrity_valid": False,
    }


def test_all_required_attack_metadata_names_are_governed() -> None:
    required = {
        "is_attack",
        "attack_type",
        "attack_severity",
        "original_value",
        "modified_value",
        "experiment_id",
        "attack_rate",
        "original_field",
        "random_seed",
    }
    assert required.issubset(ATTACK_METADATA_COLUMNS)


def test_value_attack_accepts_integer_source_column(security_frame: pd.DataFrame) -> None:
    frame = security_frame.copy()
    frame["Order Item Total"] = frame["Order Item Total"].astype(int)
    attacked, metadata, _ = generate_attack(
        frame, "value_manipulation", 0.10, "LOW", 42
    )
    assert metadata["actual_modified_records"] == 20
    assert attacked["Order Item Total"].dtype == float


def test_v04_pipeline_writes_required_outputs(
    security_frame: pd.DataFrame, tmp_path
) -> None:
    processed_dir = tmp_path / "processed"
    train_source = security_frame.iloc[:100].reset_index(drop=True)
    mean_quantity = train_source["Order Item Quantity"].mean()
    scale_quantity = train_source["Order Item Quantity"].std(ddof=0)
    mean_total = train_source["Order Item Total"].mean()
    scale_total = train_source["Order Item Total"].std(ddof=0)

    split_sources = {
        "train": train_source,
        "validation": security_frame.iloc[100:150].reset_index(drop=True),
        "test": security_frame.iloc[150:].reset_index(drop=True),
    }
    split_features: dict[str, pd.DataFrame] = {}
    for name, source in split_sources.items():
        features = pd.DataFrame(
            {
                "row_id": source["row_id"],
                "order_item_quantity": (
                    source["Order Item Quantity"] - mean_quantity
                )
                / scale_quantity,
                "order_item_total": (source["Order Item Total"] - mean_total)
                / scale_total,
            }
        )
        split_features[name] = features
        split_dir = processed_dir / name
        split_dir.mkdir(parents=True)
        features.to_csv(split_dir / "features.csv", index=False)
        source.to_csv(split_dir / "metadata.csv", index=False)

    metadata = {
        "dataset": {"file_sha256": "synthetic-miniature"},
        "split": {"sizes": {name: len(frame) for name, frame in split_features.items()}},
        "preprocessing": {"fitted_on_rows": len(split_features["train"])},
        "features": {
            "columns": ["order_item_quantity", "order_item_total"],
            "cybersecurity_labels": False,
        },
        "extra": {"dataset_used_rows": 200},
    }
    (processed_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    model_path = tmp_path / "model.joblib"
    IsolationForestBaseline(
        ["order_item_quantity", "order_item_total"], n_estimators=10
    ).fit(split_features["train"]).save(model_path)

    result = run_security_evaluation_v04(
        processed_dir=processed_dir,
        results_dir=tmp_path / "results",
        model_path=model_path,
        experiment_dir=tmp_path / "experiment",
        config=SecurityExperimentConfig(
            random_seed=42,
            attack_rates=(0.10,),
            severity_levels=("LOW",),
            primary_attack_rate=0.10,
            primary_severity="LOW",
        ),
    )
    assert result["summary"]["status"] == "PASS"
    assert len(result["figure_paths"]) == 6
    assert (tmp_path / "results" / "security_summary.json").exists()
    assert (tmp_path / "results" / "attack_manifests").is_dir()
    assert (tmp_path / "results" / "predictions" / "clean_test_predictions.csv").exists()
