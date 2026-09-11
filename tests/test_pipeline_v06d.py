"""Focused synthetic tests for the validation-only V0.6-D matrix."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.ai.model_utils import ProcessedSplit
from src.pipeline_v06 import V06ProtocolConfig, run_validation_experiment
from src.pipeline_v06d import (
    V06D_SEEDS,
    build_v06d_matrix,
    non_inferiority_sensitivity,
    run_v06d_validation_matrix,
)
from src.security.experiment_data import prepare_development_experiment_data


def test_measured_validation_records_actual_model_artifact(tmp_path) -> None:
    config = V06ProtocolConfig(
        experiment_id="v06d_measured_s42",
        candidate_feature_counts=(3, 2, 1),
    )
    data = _development_data(config)
    artifact = tmp_path / "models" / "measured.joblib"
    outcome = run_validation_experiment(
        data,
        selector=None,
        config=config,
        run_id="measured_validation",
        measure_resources=True,
        model_artifact_path=artifact,
    )

    resources = outcome.record.resource_measurement
    assert resources["performed"] is True
    assert resources["selector_fit_wall_time_sec"] == 0.0
    assert resources["combined_train_wall_time_sec"] >= resources["model_fit_wall_time_sec"]
    assert resources["attacked_validation_inference_wall_time_sec"] > 0.0
    assert resources["per_record_inference_sec"] > 0.0
    assert resources["peak_rss_mib"] > 0.0
    assert resources["serialized_model_bytes"] == artifact.stat().st_size
    assert resources["selected_feature_count"] == 3


def test_v06d_matrix_has_exact_twelve_configurations() -> None:
    config = V06ProtocolConfig(
        experiment_id="v06d_matrix_s42",
        configured_seeds=V06D_SEEDS,
        candidate_feature_counts=(3, 2, 1),
    )
    matrix = build_v06d_matrix(config)

    assert len(matrix) == 12
    assert [item["selector_id"] for item in matrix[:3]] == [
        "none",
        "variance_threshold",
        "pairwise_correlation_filter",
    ]
    for selector_id in (
        "correlation_redundancy_ranking",
        "mutual_information_select_k_best",
        "anova_f_select_k_best",
    ):
        assert [
            item["requested_feature_count"]
            for item in matrix
            if item["selector_id"] == selector_id
        ] == [3, 2, 1]


def test_non_inferiority_handles_zero_baseline_denominators() -> None:
    summary = pd.DataFrame(
        [
            _summary_row("none_natural", "none", "none", 3, 0.0, 0.0, 0.0),
            _summary_row("variance", "variance_threshold", "unsupervised", 2, 0.0, 0.1, 0.0),
        ]
    )

    sensitivity = non_inferiority_sensitivity(summary)
    primary = sensitivity.loc[sensitivity["ap_f1_margin_percent"].eq(5.0)].iloc[0]
    assert primary["average_precision_denominator_status"] == (
        "baseline_zero_candidate_nonnegative"
    )
    assert primary["average_precision_relative_degradation_percent"] == 0.0
    assert primary["preservation_status"] == "preserving"


def test_synthetic_matrix_writes_complete_validation_only_outputs(tmp_path) -> None:
    config_path = tmp_path / "feature_selection.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.6-c",
                "experiment_id": "v0_6_d_synthetic",
                "seeds": list(V06D_SEEDS),
                "feature_selection": {
                    "candidate_k": [3, 2, 1],
                    "variance_threshold": 0.0,
                    "correlation_threshold": 0.95,
                },
                "attack": {"type": "mixed", "rate": 0.05, "severity": "MEDIUM"},
                "primary_model": {
                    "name": "decision_tree",
                    "parameters": {
                        "max_depth": 5,
                        "min_samples_leaf": 20,
                        "class_weight": "balanced",
                    },
                    "prediction_threshold": 0.5,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    load_counts = {seed: 0 for seed in V06D_SEEDS}

    def load(config: V06ProtocolConfig):
        load_counts[config.seed] += 1
        return _development_data(config)

    results_dir = tmp_path / "results"
    result = run_v06d_validation_matrix(
        config_path=config_path,
        results_dir=results_dir,
        models_dir=tmp_path / "models",
        development_data_loader=load,
    )

    assert result["status"] == "PASS"
    assert result["run_count"] == 60
    assert set(load_counts.values()) == {1}
    required = {
        "core_runs.csv",
        "core_runs.json",
        "selected_feature_sets.json",
        "feature_selection_rankings.csv",
        "resource_comparison.csv",
        "validation_summary.csv",
        "run_metadata.json",
        "leakage_audit.json",
        "paired_baseline_differences.csv",
        "paired_baseline_comparisons.csv",
        "non_inferiority_sensitivity.csv",
        "preservation_primary.csv",
        "candidate_selection.csv",
        "candidate_selection.json",
        "selected_set_stability.csv",
        "ranking_stability.csv",
        "deterministic_unsupervised_stability.csv",
    }
    assert required.issubset({path.name for path in results_dir.iterdir()})
    core = pd.read_csv(results_dir / "core_runs.csv")
    assert len(core) == 60
    assert set(core["seed"]) == set(V06D_SEEDS)
    assert core.groupby("seed").size().eq(12).all()
    assert core["evaluation_split"].eq("validation").all()
    assert core["resource_measurement_performed"].all()
    assert "pr_auc" not in core.columns
    assert {"average_precision", "pr_auc_trapezoidal", "TP", "TN", "FP", "FN"} <= set(core)
    assert {
        "selector_mode",
        "selector_parameters",
        "requested_K",
        "actual_feature_count",
        "selected_features",
        "feature_fingerprint",
        "training_data_fingerprint",
        "validation_data_fingerprint",
        "selector_fit_time_seconds",
        "model_train_time_seconds",
        "combined_train_time_seconds",
        "inference_time_seconds",
        "inference_time_per_record",
        "peak_RSS_MiB",
        "serialized_model_size_bytes",
        "visible_only_recall",
    } <= set(core)
    audit = json.loads((results_dir / "leakage_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["test_accessed"] is False
    assert len(list((results_dir / "figures").glob("*.png"))) >= 2


def _summary_row(
    configuration_id: str,
    selector_id: str,
    selector_type: str,
    feature_count: int,
    average_precision: float,
    f1: float,
    recall: float,
) -> dict[str, object]:
    return {
        "configuration_id": configuration_id,
        "selector_id": selector_id,
        "selector_type": selector_type,
        "count_strategy": "natural",
        "requested_feature_count": np.nan,
        "feature_count_mean": feature_count,
        "average_precision_mean": average_precision,
        "f1_mean": f1,
        "recall_mean": recall,
    }


def _development_data(config: V06ProtocolConfig):
    candidates = ("order_item_quantity", "order_item_total", "days_schedule")
    train = _split("train", 1000, candidates)
    validation = _split("validation", 2000, candidates)
    metadata = {
        "features": {"columns": list(candidates)},
        "preprocessing": {
            "fitted_on_rows": len(train.features),
            "ml_feature_count": len(candidates),
            "ml_feature_columns": list(candidates),
        },
        "split": {
            "seed": 42,
            "strategy": "synthetic_grouped",
            "sizes": {"train": len(train.features), "validation": len(validation.features)},
        },
    }
    return prepare_development_experiment_data(
        train=train,
        validation=validation,
        candidate_features=candidates,
        dataset_metadata=metadata,
        attack_type=config.attack_type,
        attack_rate=config.attack_rate,
        severity=config.attack_severity,
        random_seed=config.seed,
        experiment_id=config.experiment_id,
    )


def _split(name: str, row_start: int, candidates: tuple[str, ...]) -> ProcessedSplit:
    rows = 100
    position = np.arange(rows)
    row_ids = np.arange(row_start, row_start + rows)
    index = pd.Index(row_ids, name="source_index")
    quantity = (position % 7 + 1).astype(float)
    total = (30.0 + position * 1.7 + quantity * 3.0).astype(float)
    scheduled = (position % 5 + 1).astype(float)
    dates = pd.Timestamp("2017-01-02") + pd.to_timedelta(position % 20, unit="D")
    trace = pd.DataFrame(
        {
            "row_id": row_ids,
            "Order Item Quantity": quantity,
            "Order Item Total": total,
            "Days for shipment (scheduled)": scheduled,
            "Order Status": np.asarray(["A", "B", "C", "D"])[position % 4],
            "order date (DateOrders)": dates.astype(str),
            "Order Region": np.asarray(["North", "South", "East", "West"])[position % 4],
        },
        index=index,
    )
    features = pd.DataFrame(
        {
            "row_id": row_ids,
            "order_item_quantity": quantity,
            "order_item_total": total,
            "days_schedule": scheduled,
        },
        index=index,
    )
    assert tuple(features.columns[1:]) == candidates
    return ProcessedSplit(name, features, trace, None)
