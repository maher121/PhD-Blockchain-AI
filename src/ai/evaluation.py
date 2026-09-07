"""Unsupervised evaluation, leakage audit, sensitivity, and figures for V0.3."""

from __future__ import annotations

import time
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.ai.anomaly_detector import IsolationForestBaseline
from src.ai.model_utils import MODEL_GENERATED_COLUMNS
from src.config import DATACO_OUTCOME_COLUMNS, DATACO_TARGET_COLUMN


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """Describe model flags and scores without ground-truth classification claims."""
    count = int(len(predictions))
    anomaly_count = int(predictions["anomaly_label"].sum())
    anomaly_scores = predictions["anomaly_score"]
    raw_scores = predictions["raw_isolation_score"]
    return {
        "number_of_samples": count,
        "number_of_detected_anomalies": anomaly_count,
        "anomaly_percentage": float(100.0 * anomaly_count / count) if count else 0.0,
        "score_direction": {
            "raw_isolation_score": "lower values are more anomalous",
            "anomaly_score": "higher values are more anomalous (-raw_isolation_score)",
            "normalized_anomaly_score": (
                "higher values are more anomalous; training-bound min-max score, not a probability"
            ),
        },
        "raw_isolation_score_statistics": _series_statistics(raw_scores),
        "anomaly_score_statistics": _series_statistics(anomaly_scores),
    }


def contamination_sensitivity(
    train_features: pd.DataFrame,
    evaluation_features: dict[str, pd.DataFrame],
    selected_features: list[str],
    contamination_values: Iterable[float],
    base_parameters: dict[str, Any],
) -> pd.DataFrame:
    """Measure flag sensitivity to hypothetical contamination assumptions."""
    rows: list[dict[str, Any]] = []
    for contamination in contamination_values:
        parameters = dict(base_parameters)
        parameters["contamination"] = float(contamination)
        started = time.perf_counter()
        detector = IsolationForestBaseline(selected_features, **parameters).fit(train_features)
        fit_seconds = time.perf_counter() - started
        for split_name, frame in evaluation_features.items():
            infer_started = time.perf_counter()
            labels = detector.predict(frame)
            infer_seconds = time.perf_counter() - infer_started
            anomaly_count = int(labels.sum())
            rows.append(
                {
                    "contamination": float(contamination),
                    "split": split_name,
                    "number_of_samples": int(len(labels)),
                    "number_of_detected_anomalies": anomaly_count,
                    "anomaly_percentage": float(100.0 * anomaly_count / len(labels)),
                    "training_time_seconds": fit_seconds,
                    "inference_time_seconds": infer_seconds,
                    "interpretation": "sensitivity assumption, not ground truth prevalence",
                }
            )
    return pd.DataFrame(rows)


def seed_stability(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    selected_features: list[str],
    seeds: Iterable[int],
    base_parameters: dict[str, Any],
) -> pd.DataFrame:
    """Compare test flags and score ranks across a small fixed seed set."""
    seed_values = list(seeds)
    if not seed_values:
        raise ValueError("At least one stability seed is required.")
    predictions: dict[int, pd.Series] = {}
    scores: dict[int, pd.Series] = {}
    times: dict[int, float] = {}
    for seed in seed_values:
        parameters = dict(base_parameters)
        parameters["random_state"] = int(seed)
        started = time.perf_counter()
        detector = IsolationForestBaseline(selected_features, **parameters).fit(train_features)
        predictions[seed] = detector.predict(test_features)
        scores[seed] = detector.anomaly_scores(test_features)
        times[seed] = time.perf_counter() - started

    reference = seed_values[0]
    reference_set = set(predictions[reference].index[predictions[reference].eq(1)])
    reference_rank = scores[reference].rank(method="average")
    rows: list[dict[str, Any]] = []
    for seed in seed_values:
        current_set = set(predictions[seed].index[predictions[seed].eq(1)])
        union = reference_set | current_set
        jaccard = float(len(reference_set & current_set) / len(union)) if union else 1.0
        rank_correlation = float(reference_rank.corr(scores[seed].rank(method="average")))
        rows.append(
            {
                "seed": int(seed),
                "reference_seed": int(reference),
                "number_of_detected_anomalies": int(predictions[seed].sum()),
                "anomaly_percentage": float(100.0 * predictions[seed].mean()),
                "jaccard_with_reference": jaccard,
                "score_rank_correlation_with_reference": rank_correlation,
                "fit_and_inference_time_seconds": times[seed],
            }
        )
    return pd.DataFrame(rows)


def build_leakage_audit(
    selected_features: list[str],
    dataset_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Produce explicit PASS/FAIL evidence for the V0.3 leakage contract."""
    selected_set = set(selected_features)
    target_names = {"target", DATACO_TARGET_COLUMN}
    identifier_names = {
        "row_id",
        "Order Id",
        "Order Item Id",
        "Order Customer Id",
        "Customer Id",
        "Product Card Id",
    }
    future_names = {
        "shipping date (DateOrders)",
        "Days for shipping (real)",
        "Delivery Status",
        "Order Status",
    }

    preprocessing = dataset_metadata.get("preprocessing", {})
    split = dataset_metadata.get("split", {})
    train_size = split.get("sizes", {}).get("train")
    fitted_rows = preprocessing.get("fitted_on_rows")
    order_overlap = split.get("identity_overlap_checks", {}).get("Order Id", {})

    checks = {
        "target_not_in_features": _check(
            selected_set.isdisjoint(target_names), sorted(selected_set & target_names)
        ),
        "identifiers_not_in_features": _check(
            selected_set.isdisjoint(identifier_names), sorted(selected_set & identifier_names)
        ),
        "post_outcome_variables_not_in_features": _check(
            selected_set.isdisjoint(set(DATACO_OUTCOME_COLUMNS)),
            sorted(selected_set & set(DATACO_OUTCOME_COLUMNS)),
        ),
        "preprocessing_fitted_on_training_only": _check(
            train_size is not None
            and fitted_rows == train_size
            and not any(int(value) for value in order_overlap.values()),
            {
                "preprocessor_fitted_rows": fitted_rows,
                "training_rows": train_size,
                "order_id_overlap": order_overlap,
            },
        ),
        "model_outputs_not_in_features": _check(
            selected_set.isdisjoint(MODEL_GENERATED_COLUMNS),
            sorted(selected_set & set(MODEL_GENERATED_COLUMNS)),
        ),
        "future_information_not_in_features": _check(
            selected_set.isdisjoint(future_names), sorted(selected_set & future_names)
        ),
    }
    overall = "PASS" if all(item["status"] == "PASS" for item in checks.values()) else "FAIL"
    return {
        "stage": "V0.3",
        "overall_status": overall,
        "checks": checks,
        "note": (
            "The operational Late_delivery_risk files are retained for provenance only and "
            "are not loaded into the model feature pipeline or used for V0.3 metrics."
        ),
    }


def top_anomalies(
    test_predictions: pd.DataFrame,
    test_metadata: pd.DataFrame,
    limit: int = 50,
) -> pd.DataFrame:
    """Return highest-scoring flagged test records with non-PII traceability fields."""
    safe_columns = [
        "row_id",
        "Order Id",
        "Order Item Id",
        "order date (DateOrders)",
        "Type",
        "Market",
        "Shipping Mode",
        "Order Item Quantity",
        "Sales",
        "Order Item Total",
    ]
    safe_columns = [column for column in safe_columns if column in test_metadata.columns]
    merged = test_predictions.merge(
        test_metadata[safe_columns], on="row_id", how="left", validate="one_to_one"
    )
    flagged = merged[merged["anomaly_label"].eq(1)]
    candidates = flagged if not flagged.empty else merged
    return candidates.sort_values("anomaly_score", ascending=False).head(limit).reset_index(drop=True)


def save_baseline_figures(
    test_predictions: pd.DataFrame,
    sensitivity: pd.DataFrame,
    test_features: pd.DataFrame,
    selected_features: list[str],
    output_dir: Path | str,
) -> list[Path]:
    """Create the four curated V0.3 figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    normal = test_predictions.loc[test_predictions["anomaly_label"].eq(0), "anomaly_score"]
    anomaly = test_predictions.loc[test_predictions["anomaly_label"].eq(1), "anomaly_score"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(normal, bins=45, alpha=0.75, label="Normal flag", color="#356859")
    if not anomaly.empty:
        ax.hist(anomaly, bins=30, alpha=0.75, label="Anomaly flag", color="#c14953")
    ax.set_title("Isolation Forest anomaly-score distribution (test split)")
    ax.set_xlabel("Anomaly score (-score_samples); higher is more anomalous")
    ax.set_ylabel("Observations")
    ax.legend()
    written.append(_save_figure(fig, output_dir / "anomaly_score_distribution.png", plt))

    counts = test_predictions["anomaly_label"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(["Normal", "Anomaly"], counts.values, color=["#356859", "#c14953"])
    ax.bar_label(bars, labels=[str(int(value)) for value in counts.values])
    ax.set_title("Isolation Forest flags (test split)")
    ax.set_ylabel("Observations")
    written.append(_save_figure(fig, output_dir / "anomaly_vs_normal_count.png", plt))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = cycle(["#264653", "#e76f51", "#2a9d8f"])
    for split_name, group in sensitivity.groupby("split", sort=False):
        ax.plot(
            group["contamination"] * 100,
            group["anomaly_percentage"],
            marker="o",
            label=split_name,
            color=next(colors),
        )
    ax.set_title("Contamination sensitivity (assumptions, not ground truth)")
    ax.set_xlabel("Configured contamination (%)")
    ax.set_ylabel("Observations flagged (%)")
    ax.legend()
    written.append(_save_figure(fig, output_dir / "contamination_sensitivity.png", plt))

    profile_candidates = [
        "order_item_quantity",
        "order_item_product_price",
        "order_item_discount_rate",
        "days_schedule",
    ]
    profile_features = [name for name in profile_candidates if name in selected_features]
    fig, axes = plt.subplots(1, max(1, len(profile_features)), figsize=(4 * max(1, len(profile_features)), 4.5))
    axes_array = np.atleast_1d(axes)
    labels = test_predictions["anomaly_label"].to_numpy(dtype=int)
    for ax, name in zip(axes_array, profile_features):
        values = test_features[name].to_numpy(dtype=float)
        groups = [values[labels == 0], values[labels == 1]]
        if len(groups[1]) == 0:
            groups[1] = np.array([np.nan])
        ax.boxplot(groups, tick_labels=["Normal", "Anomaly"], showfliers=False)
        ax.set_title(name.replace("_", " "))
        ax.set_ylabel("V0.2 standardized value")
    fig.suptitle("Selected feature profiles by model flag (test split)")
    written.append(_save_figure(fig, output_dir / "feature_anomaly_profiles.png", plt))
    return written


def _series_statistics(series: pd.Series) -> dict[str, float]:
    return {
        "minimum": float(series.min()),
        "q1": float(series.quantile(0.25)),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "q3": float(series.quantile(0.75)),
        "maximum": float(series.max()),
        "standard_deviation": float(series.std(ddof=0)),
    }


def _check(passed: bool, evidence: Any) -> dict[str, Any]:
    return {"status": "PASS" if passed else "FAIL", "evidence": evidence}


def _save_figure(fig: Any, path: Path, pyplot: Any) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    pyplot.close(fig)
    return path
