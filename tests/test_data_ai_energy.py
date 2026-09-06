"""Unit tests for the data + AI + energy modules of Prototype V0.1."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ai.anomaly_detection import IsolationForestAnomalyDetector, report_anomaly_metrics
from src.data.generator import generate_supply_chain_dataset
from src.data.loading import load_supply_chain_data
from src.energy.estimation import EnergyEstimate, estimate_energy_consumption
from src.energy.measurement import ExecutionTimer, Measurement
from src.preprocessing.feature_engineering import build_feature_matrix
from src.preprocessing.preprocessing import preprocess_supply_chain_data


@pytest.fixture(scope="module")
def raw_dataset() -> pd.DataFrame:
    return generate_supply_chain_dataset(n_rows=400, seed=7)


@pytest.fixture(scope="module")
def processed_dataset(raw_dataset: pd.DataFrame) -> pd.DataFrame:
    return preprocess_supply_chain_data(raw_dataset)


# --- data ----------------------------------------------------------------


def test_generator_output_schema(raw_dataset: pd.DataFrame) -> None:
    assert list(raw_dataset.columns) == [
        "transaction_id",
        "participant_id",
        "product_id",
        "order_id",
        "timestamp",
        "quantity",
        "unit_price",
        "total_amount",
        "shipping_days",
        "location_region",
        "transaction_status",
        "known_anomaly",
        "anomaly_type",
    ]
    assert len(raw_dataset) == 400


def test_generator_reproducibility() -> None:
    first = generate_supply_chain_dataset(n_rows=100, seed=3)
    second = generate_supply_chain_dataset(n_rows=100, seed=3)
    pd.testing.assert_frame_equal(first, second)


def test_generator_invalid_anomaly_fraction() -> None:
    with pytest.raises(ValueError):
        generate_supply_chain_dataset(n_rows=10, anomaly_fraction=0.9)


def test_preprocessing_cleans_and_keeps_rows(processed_dataset: pd.DataFrame) -> None:
    assert not processed_dataset.empty
    assert (processed_dataset["quantity"] > 0).all()
    assert (processed_dataset["unit_price"] > 0).all()


def test_feature_matrix_shape(processed_dataset: pd.DataFrame) -> None:
    features = build_feature_matrix(processed_dataset)
    assert features.shape[0] == len(processed_dataset)
    assert not features.isna().any().any()
    expected_columns = {
        "quantity",
        "unit_price",
        "total_amount",
        "shipping_days",
        "amount_per_unit",
        "log_total_amount",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "quantity_ratio_to_mean",
    }
    assert expected_columns.issubset(set(features.columns))


# --- AI ------------------------------------------------------------------


def test_isolation_forest_train_predict(processed_dataset: pd.DataFrame) -> None:
    features = build_feature_matrix(processed_dataset)
    detector = IsolationForestAnomalyDetector()
    detector.fit(features)
    predictions = detector.predict_anomalies(features)
    scores = detector.anomaly_scores(features)
    assert len(predictions) == len(features)
    assert predictions.dtype == bool
    assert scores.isna().sum() == 0


def test_isolation_forest_save_load(tmp_path, processed_dataset: pd.DataFrame) -> None:
    features = build_feature_matrix(processed_dataset)
    detector = IsolationForestAnomalyDetector()
    detector.fit(features)
    model_path = tmp_path / "model.joblib"
    detector.save_model(model_path)

    loaded = IsolationForestAnomalyDetector()
    loaded.load_model(model_path)
    assert loaded.is_trained
    pd.testing.assert_series_equal(
        loaded.predict_anomalies(features), detector.predict_anomalies(features)
    )


def test_anomaly_metric_reporting(processed_dataset: pd.DataFrame) -> None:
    features = build_feature_matrix(processed_dataset)
    detector = IsolationForestAnomalyDetector()
    detector.fit(features)
    predictions = detector.predict_anomalies(features)
    scores = detector.anomaly_scores(features)
    labels = processed_dataset.loc[features.index, "known_anomaly"]
    metrics = report_anomaly_metrics(labels, predictions, scores)
    assert {
        "precision",
        "recall",
        "f1_score",
        "accuracy",
        "roc_auc",
    }.issubset(set(metrics.keys()))
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_predict_before_fit_raises(processed_dataset: pd.DataFrame) -> None:
    detector = IsolationForestAnomalyDetector()
    with pytest.raises(RuntimeError):
        detector.predict_anomalies(pd.DataFrame({"a": [1.0, 2.0]}))


# --- energy --------------------------------------------------------------


def test_execution_timer_measures_time() -> None:
    with ExecutionTimer(label="test") as timer:
        for _ in range(100_000):
            pass
    measurement: Measurement = timer.measurement
    assert measurement.wall_time_seconds >= 0.0
    assert measurement.cpu_time_seconds >= 0.0
    assert measurement.peak_memory_rss_bytes > 0


def test_energy_estimate_uses_measured_time() -> None:
    with ExecutionTimer(label="test") as timer:
        nonce = 0
        for i in range(200_000):
            nonce += i
    estimate: EnergyEstimate = estimate_energy_consumption(timer.measurement)
    assert estimate.is_estimated is True
    assert estimate.energy_joules >= 0.0
    assert estimate.wall_time_seconds == pytest.approx(timer.measurement.wall_time_seconds)
    assert estimate.cpu_energy_joules + estimate.ram_energy_joules == pytest.approx(
        estimate.energy_joules
    )