"""Synthetic-fixture tests for the V0.3 unsupervised AI baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ai.anomaly_detector import IsolationForestBaseline
from src.ai.evaluation import build_leakage_audit
from src.ai.model_utils import validate_selected_features


@pytest.fixture()
def baseline_features() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    normal = rng.normal(size=(80, 3))
    extremes = np.array([[8.0, 8.0, 8.0], [-8.0, -8.0, -8.0]])
    values = np.vstack([normal, extremes])
    return pd.DataFrame(
        {
            "row_id": np.arange(len(values)),
            "amount": values[:, 0],
            "quantity": values[:, 1],
            "schedule": values[:, 2],
        }
    )


@pytest.fixture()
def selected_features() -> list[str]:
    return ["amount", "quantity", "schedule"]


@pytest.fixture()
def leakage_metadata() -> dict:
    return {
        "split": {
            "sizes": {"train": 57, "validation": 12, "test": 13},
            "identity_overlap_checks": {
                "Order Id": {
                    "train_vs_validation": 0,
                    "train_vs_test": 0,
                    "validation_vs_test": 0,
                }
            },
        },
        "preprocessing": {"fitted_on_rows": 57},
    }


def test_model_initialization(selected_features: list[str]) -> None:
    detector = IsolationForestBaseline(selected_features, n_estimators=20)
    model = detector.pipeline.named_steps["isolation_forest"]
    assert model.n_estimators == 20
    assert model.random_state == 42
    assert detector.is_fitted is False


def test_model_training(
    baseline_features: pd.DataFrame, selected_features: list[str]
) -> None:
    detector = IsolationForestBaseline(selected_features, n_estimators=20)
    returned = detector.fit(baseline_features.iloc[:57])
    assert returned is detector
    assert detector.is_fitted


def test_prediction_generation(
    baseline_features: pd.DataFrame, selected_features: list[str]
) -> None:
    detector = IsolationForestBaseline(selected_features, n_estimators=20).fit(
        baseline_features.iloc[:57]
    )
    result = detector.predict_frame(baseline_features.iloc[57:])
    assert len(result) == 25
    assert set(result["anomaly_label"].unique()).issubset({0, 1})
    assert set(result["isolation_forest_prediction"].unique()).issubset({-1, 1})


def test_anomaly_score_generation(
    baseline_features: pd.DataFrame, selected_features: list[str]
) -> None:
    detector = IsolationForestBaseline(selected_features, n_estimators=20).fit(
        baseline_features.iloc[:57]
    )
    test = baseline_features.iloc[57:]
    raw = detector.raw_scores(test)
    anomaly = detector.anomaly_scores(test)
    normalized = detector.normalized_anomaly_scores(test)
    assert np.allclose(anomaly.to_numpy(), -raw.to_numpy())
    assert np.isfinite(normalized).all()
    assert normalized.between(0.0, 1.0).all()


def test_reproducibility(
    baseline_features: pd.DataFrame, selected_features: list[str]
) -> None:
    first = IsolationForestBaseline(selected_features, n_estimators=20).fit(
        baseline_features.iloc[:57]
    )
    second = IsolationForestBaseline(selected_features, n_estimators=20).fit(
        baseline_features.iloc[:57]
    )
    pd.testing.assert_frame_equal(
        first.predict_frame(baseline_features.iloc[57:]),
        second.predict_frame(baseline_features.iloc[57:]),
    )


def test_feature_exclusion_and_identifier_invariance(
    baseline_features: pd.DataFrame, selected_features: list[str]
) -> None:
    with pytest.raises(ValueError, match="Leakage-prone"):
        validate_selected_features([*selected_features, "row_id"])

    detector = IsolationForestBaseline(selected_features, n_estimators=20).fit(
        baseline_features.iloc[:57]
    )
    test = baseline_features.iloc[57:].copy()
    changed_identifier = test.copy()
    changed_identifier["row_id"] += 1_000_000
    pd.testing.assert_series_equal(
        detector.predict(test), detector.predict(changed_identifier)
    )


def test_no_target_leakage(
    selected_features: list[str], leakage_metadata: dict
) -> None:
    passing = build_leakage_audit(selected_features, leakage_metadata)
    assert passing["overall_status"] == "PASS"
    failing = build_leakage_audit(
        [*selected_features, "Late_delivery_risk", "anomaly_label"],
        leakage_metadata,
    )
    assert failing["overall_status"] == "FAIL"
    assert failing["checks"]["target_not_in_features"]["status"] == "FAIL"
    assert failing["checks"]["model_outputs_not_in_features"]["status"] == "FAIL"


def test_saved_model_loading(
    tmp_path, baseline_features: pd.DataFrame, selected_features: list[str]
) -> None:
    detector = IsolationForestBaseline(selected_features, n_estimators=20).fit(
        baseline_features.iloc[:57]
    )
    artifact = detector.save(tmp_path / "isolation_forest_v0_3.joblib")
    loaded = IsolationForestBaseline.load(artifact)
    assert loaded.is_fitted
    assert loaded.feature_names == selected_features


def test_inference_using_saved_model(
    tmp_path, baseline_features: pd.DataFrame, selected_features: list[str]
) -> None:
    detector = IsolationForestBaseline(selected_features, n_estimators=20).fit(
        baseline_features.iloc[:57]
    )
    loaded = IsolationForestBaseline.load(detector.save(tmp_path / "model.joblib"))
    expected = detector.predict_frame(baseline_features.iloc[57:])
    actual = loaded.predict_frame(baseline_features.iloc[57:])
    pd.testing.assert_frame_equal(actual, expected)
