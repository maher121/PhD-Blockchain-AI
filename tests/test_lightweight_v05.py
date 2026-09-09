"""Synthetic-data tests for V0.5 lightweight model evaluation."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.lightweight.evaluation import calculate_metrics, identify_pareto_candidates
from src.lightweight.feature_reduction import (
    V05_FORBIDDEN_FEATURES,
    CorrelationFeatureReducer,
    validate_model_features,
)
from src.lightweight.models import SUPPORTED_MODELS, LightweightDetector, create_model
from src.lightweight.resource_monitor import measure_call, measure_model_size
from src.lightweight.training import run_inference, train_model


@pytest.fixture()
def binary_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    clean = rng.normal(0.0, 1.0, size=(90, 4))
    attacks = rng.normal(4.0, 0.5, size=(30, 4))
    values = np.vstack([clean, attacks])
    frame = pd.DataFrame(values, columns=["f1", "f2", "f3", "f4"])
    labels = pd.Series([0] * len(clean) + [1] * len(attacks), dtype="int8")
    return frame, labels


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_model_factory(model_name: str) -> None:
    parameters = {
        "isolation_forest": {"n_estimators": 10, "n_jobs": 1},
        "logistic_regression": {"solver": "liblinear", "max_iter": 100},
        "decision_tree": {"max_depth": 3},
        "random_forest": {"n_estimators": 5, "max_depth": 3, "n_jobs": 1},
    }[model_name]
    model = create_model(model_name, ["f1", "f2"], parameters)
    assert model.model_name == model_name
    assert model.random_state == 42
    assert model.is_fitted is False


def test_model_training_and_prediction(binary_data) -> None:
    features, labels = binary_data
    model = create_model("decision_tree", list(features), {"max_depth": 3})
    assert model.fit(features, labels) is model
    predictions = model.predict_frame(features)
    assert model.is_fitted
    assert list(predictions) == ["anomaly_score", "anomaly_label"]
    assert set(predictions["anomaly_label"].unique()).issubset({0, 1})
    assert predictions["anomaly_score"].between(0.0, 1.0).all()


def test_measured_training_and_model_size(binary_data, tmp_path) -> None:
    features, labels = binary_data
    result = train_model(
        create_model("decision_tree", list(features), {"max_depth": 3}),
        features,
        labels,
        tmp_path / "tree.joblib",
    )
    inference = run_inference(result.model, features)
    measured_size = measure_model_size(result.artifact_path)
    assert result.resources.wall_time_sec > 0.0
    assert inference.resources.wall_time_sec > 0.0
    assert inference.resources.peak_rss_bytes > 0
    assert measured_size["model_size_bytes"] == result.artifact_path.stat().st_size
    assert measured_size["model_size_kb"] > 0.0


def test_resource_monitor_measures_wall_time() -> None:
    value, measurement = measure_call(lambda: (time.sleep(0.01), 7)[1], label="test")
    assert value == 7
    assert measurement.wall_time_sec >= 0.01
    assert measurement.peak_rss_mb > 0.0


def test_feature_reduction_is_nested_and_train_only(binary_data) -> None:
    features, _ = binary_data
    reducer = CorrelationFeatureReducer(list(features)).fit(features.iloc[:80])
    subsets = reducer.subsets([1.0, 0.75, 0.5, 0.25])
    assert [len(subsets[value]) for value in (1.0, 0.75, 0.5, 0.25)] == [4, 3, 2, 1]
    assert set(subsets[0.25]).issubset(subsets[0.5])
    assert set(subsets[0.5]).issubset(subsets[0.75])
    ranking = reducer.ranking_frame()
    assert ranking["labels_used"].eq(False).all()
    assert ranking["selection_data"].eq("clean_training_split_only").all()


def test_metric_calculation() -> None:
    truth = pd.DataFrame({"record_id": [1, 2, 3, 4], "is_attack": [0, 0, 1, 1]})
    predictions = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "anomaly_label": [0, 1, 1, 0],
            "anomaly_score": [0.1, 0.8, 0.9, 0.2],
        }
    )
    metrics = calculate_metrics(truth, predictions)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["pr_auc"] is not None
    assert metrics["roc_auc"] is not None


def test_reproducibility(binary_data) -> None:
    features, labels = binary_data
    parameters = {"n_estimators": 8, "max_depth": 3, "n_jobs": 1}
    first = create_model("random_forest", list(features), parameters).fit(features, labels)
    second = create_model("random_forest", list(features), parameters).fit(features, labels)
    pd.testing.assert_frame_equal(first.predict_frame(features), second.predict_frame(features))


@pytest.mark.parametrize(
    "forbidden",
    [
        "is_attack",
        "attack_type",
        "attack_severity",
        "original_value",
        "modified_value",
        "experiment_id",
        "attack_rate",
        "target",
        "prediction",
        "anomaly_score",
    ],
)
def test_forbidden_columns_never_enter_model_input(forbidden: str) -> None:
    assert forbidden in V05_FORBIDDEN_FEATURES
    with pytest.raises(ValueError, match="Forbidden V0.5"):
        validate_model_features(["legitimate", forbidden])
    with pytest.raises(ValueError, match="Forbidden V0.5"):
        LightweightDetector("decision_tree", ["legitimate", forbidden])


def test_pareto_candidate_identification() -> None:
    frame = pd.DataFrame(
        [
            {"configuration_id": "balanced", "f1": 0.8, "inference_time_sec": 0.2, "memory_mb": 20.0, "feature_count": 4},
            {"configuration_id": "dominated", "f1": 0.7, "inference_time_sec": 0.3, "memory_mb": 25.0, "feature_count": 4},
            {"configuration_id": "fast", "f1": 0.6, "inference_time_sec": 0.1, "memory_mb": 10.0, "feature_count": 2},
        ]
    )
    result = identify_pareto_candidates(frame)
    assert set(result["configuration_id"]) == {"balanced", "fast"}


def test_model_complexity_indicators(binary_data) -> None:
    features, labels = binary_data
    tree = create_model("decision_tree", list(features), {"max_depth": 3}).fit(
        features, labels
    )
    linear = create_model(
        "logistic_regression", list(features), {"solver": "liblinear"}
    ).fit(features, labels)
    assert tree.complexity()["number_of_trees"] == 1
    assert tree.complexity()["maximum_observed_depth"] <= 3
    assert linear.complexity()["number_of_coefficients"] == len(features.columns)
