"""AI evaluation and feature projection for controlled V0.4 ground truth."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)

from src.security.ground_truth import assert_no_attack_metadata


DIRECT_FEATURE_MAP: dict[str, str] = {
    "Order Item Quantity": "order_item_quantity",
    "Order Item Total": "order_item_total",
    "Order Item Product Price": "order_item_product_price",
    "Product Price": "product_price",
    "Days for shipment (scheduled)": "days_schedule",
    "quantity": "order_item_quantity",
    "total_amount": "order_item_total",
}
TIMESTAMP_SOURCE_FIELDS: set[str] = {"order date (DateOrders)", "timestamp"}
TEMPORAL_FEATURES: tuple[str, ...] = (
    "year",
    "month",
    "day",
    "day_of_week",
    "hour",
    "is_weekend",
)


def project_attacks_to_feature_space(
    clean_features: pd.DataFrame,
    clean_metadata: pd.DataFrame,
    attacked_metadata: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    reference_features: pd.DataFrame | None = None,
    reference_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Project source edits through the existing V0.2 standardized representation.

    The affine mappings are recovered from aligned clean TRAIN source/processed
    pairs supplied by the caller; no labels, attack metadata, or evaluation
    outcomes are used. Source fields excluded by V0.2 intentionally leave the
    model feature matrix unchanged.
    """
    _validate_alignment(clean_features, clean_metadata, attacked_metadata)
    reference_features = reference_features if reference_features is not None else clean_features
    reference_metadata = reference_metadata if reference_metadata is not None else clean_metadata
    _validate_reference(reference_features, reference_metadata)
    projected = clean_features.copy(deep=True)
    if manifest.empty:
        assert_no_attack_metadata(projected)
        return projected

    row_positions = pd.Series(clean_features.index, index=clean_features["row_id"]).to_dict()
    metadata_positions = pd.Series(
        attacked_metadata.index, index=attacked_metadata["row_id"]
    ).to_dict()

    for source_field, feature_name in DIRECT_FEATURE_MAP.items():
        if (
            source_field not in reference_metadata
            or feature_name not in projected
            or not manifest["original_field"].eq(source_field).any()
        ):
            continue
        slope, intercept = _affine_mapping(
            reference_metadata[source_field], reference_features[feature_name]
        )
        records = manifest.loc[manifest["original_field"].eq(source_field), "record_id"]
        for record_id in records:
            feature_row = row_positions[record_id]
            metadata_row = metadata_positions[record_id]
            raw_value = float(attacked_metadata.at[metadata_row, source_field])
            projected.at[feature_row, feature_name] = slope * raw_value + intercept

    timestamp_records = manifest.loc[
        manifest["original_field"].isin(TIMESTAMP_SOURCE_FIELDS)
    ]
    for source_field, records in timestamp_records.groupby("original_field", sort=False):
        if source_field not in clean_metadata:
            continue
        reference_components = _datetime_components(reference_metadata[source_field])
        attacked_components = _datetime_components(attacked_metadata[source_field])
        for feature_name in TEMPORAL_FEATURES:
            if feature_name not in projected:
                continue
            slope, intercept = _affine_mapping(
                reference_components[feature_name], reference_features[feature_name]
            )
            for record_id in records["record_id"]:
                feature_row = row_positions[record_id]
                metadata_row = metadata_positions[record_id]
                value = attacked_components.at[metadata_row, feature_name]
                projected.at[feature_row, feature_name] = slope * value + intercept

    assert_no_attack_metadata(projected)
    return projected


def evaluate_detection(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate valid binary detection metrics from experimental ground truth."""
    aligned_truth, aligned_predictions = _align_evaluation(ground_truth, predictions)
    y_true = aligned_truth["is_attack"].to_numpy(dtype=int)
    y_pred = aligned_predictions["anomaly_label"].to_numpy(dtype=int)
    scores = aligned_predictions["anomaly_score"].to_numpy(dtype=float)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    has_both_classes = len(np.unique(y_true)) == 2
    if has_both_classes:
        curve_precision, curve_recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = float(auc(curve_recall, curve_precision))
        average_precision = float(average_precision_score(y_true, scores))
    else:
        pr_auc = None
        average_precision = None
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": pr_auc,
        "pr_auc_method": "trapezoidal_precision_recall_curve" if has_both_classes else None,
        "average_precision": average_precision,
        "roc_auc": float(roc_auc_score(y_true, scores)) if has_both_classes else None,
        "true_positives": int(true_positive),
        "true_negatives": int(true_negative),
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "confusion_matrix": matrix.tolist(),
        "evaluated_records": int(len(y_true)),
        "attacked_records": int(y_true.sum()),
    }


def evaluate_paired_detection(
    ground_truth: pd.DataFrame,
    clean_predictions: pd.DataFrame,
    attacked_predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Measure prediction and score changes induced on selected attack records."""
    truth, clean = _align_evaluation(ground_truth, clean_predictions)
    _, attacked = _align_evaluation(ground_truth, attacked_predictions)
    selected = truth["is_attack"].eq(1).to_numpy()
    clean_labels = clean["anomaly_label"].to_numpy(dtype=int)[selected]
    attacked_labels = attacked["anomaly_label"].to_numpy(dtype=int)[selected]
    score_delta = (
        attacked["anomaly_score"].to_numpy(dtype=float)
        - clean["anomaly_score"].to_numpy(dtype=float)
    )[selected]
    count = int(selected.sum())
    newly_detected = int(((clean_labels == 0) & (attacked_labels == 1)).sum())
    lost_detections = int(((clean_labels == 1) & (attacked_labels == 0)).sum())
    return {
        "attacked_records": count,
        "already_flagged_before_attack": int(clean_labels.sum()),
        "flagged_after_attack": int(attacked_labels.sum()),
        "newly_detected_after_attack": newly_detected,
        "lost_detections_after_attack": lost_detections,
        "unchanged_flagged": int(((clean_labels == 1) & (attacked_labels == 1)).sum()),
        "unchanged_not_flagged": int(((clean_labels == 0) & (attacked_labels == 0)).sum()),
        "attack_induced_detection_rate": float(newly_detected / count) if count else None,
        "mean_anomaly_score_delta": float(score_delta.mean()) if count else None,
        "median_anomaly_score_delta": float(np.median(score_delta)) if count else None,
    }


def detector_visible_modified_count(
    clean_features: pd.DataFrame,
    attacked_features: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> int:
    """Count attacked rows that changed at least one legitimate AI feature."""
    changed = detector_visible_mask(clean_features, attacked_features)
    return int((changed & ground_truth["is_attack"].eq(1).to_numpy()).sum())


def detector_visible_mask(
    clean_features: pd.DataFrame, attacked_features: pd.DataFrame
) -> np.ndarray:
    """Return a positional mask for records whose legitimate AI inputs changed."""
    feature_columns = [column for column in clean_features if column != "row_id"]
    return ~np.isclose(
        clean_features[feature_columns].to_numpy(dtype=float),
        attacked_features[feature_columns].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ).all(axis=1)


def evaluate_visible_attacks(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    visible_mask: np.ndarray,
) -> dict[str, Any] | None:
    """Evaluate after excluding attacked records whose model inputs did not change."""
    if len(visible_mask) != len(ground_truth):
        raise ValueError("Visibility mask must align with ground truth.")
    include = ground_truth["is_attack"].eq(0).to_numpy() | visible_mask
    visible_attacks = int((ground_truth["is_attack"].eq(1).to_numpy() & visible_mask).sum())
    if visible_attacks == 0:
        return None
    return evaluate_detection(
        ground_truth.loc[include].reset_index(drop=True),
        predictions.loc[include].reset_index(drop=True),
    )


def save_security_figures(
    attack_rate_analysis: pd.DataFrame,
    severity_analysis: pd.DataFrame,
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path | str,
) -> list[Path]:
    """Create the six curated V0.4 security figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for metric, filename, title in (
        (
            "recall",
            "attack_rate_vs_recall.png",
            "Attack rate vs final-label recall (includes pre-existing flags)",
        ),
        (
            "f1",
            "attack_rate_vs_f1.png",
            "Attack rate vs final-label F1 (includes pre-existing flags)",
        ),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for attack_type, group in attack_rate_analysis.groupby("attack_type", sort=False):
            ax.plot(group["attack_rate"] * 100, group[metric], marker="o", label=attack_type)
        ax.set_title(title)
        ax.set_xlabel("Injected attack rate (%)")
        ax.set_ylabel(metric.upper() if metric == "f1" else metric.title())
        ax.legend(fontsize=7, ncol=2)
        written.append(_save_figure(fig, output_dir / filename, plt))

    severity_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    severity_plot = severity_analysis.copy()
    severity_plot["severity_order"] = severity_plot["severity"].map(severity_order)
    for metric, filename, title in (
        (
            "recall",
            "severity_vs_recall.png",
            "Severity vs final-label recall (includes pre-existing flags)",
        ),
        (
            "f1",
            "severity_vs_f1.png",
            "Severity vs final-label F1 (includes pre-existing flags)",
        ),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for attack_type, group in severity_plot.groupby("attack_type", sort=False):
            group = group.sort_values("severity_order")
            ax.plot(group["severity"], group[metric], marker="o", label=attack_type)
        ax.set_title(title)
        ax.set_xlabel("Controlled attack severity")
        ax.set_ylabel(metric.upper() if metric == "f1" else metric.title())
        ax.legend(fontsize=7, ncol=2)
        written.append(_save_figure(fig, output_dir / filename, plt))

    metrics = evaluate_detection(ground_truth, predictions)
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    ax.set_xticks([0, 1], labels=["Predicted clean", "Predicted attack"])
    ax.set_yticks([0, 1], labels=["Actual clean", "Actual attack"])
    ax.set_title("Final flags vs ground truth (includes pre-existing flags)")
    fig.colorbar(image, ax=ax)
    written.append(_save_figure(fig, output_dir / "confusion_matrix.png", plt))

    labels = ground_truth["is_attack"].to_numpy(dtype=int)
    scores = predictions["anomaly_score"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores[labels == 0], bins=45, alpha=0.7, label="Clean", color="#356859")
    ax.hist(scores[labels == 1], bins=35, alpha=0.7, label="Controlled attack", color="#c14953")
    ax.set_title("Post-injection scores: clean vs selected attack records")
    ax.set_xlabel("Anomaly score (-score_samples); higher is more anomalous")
    ax.set_ylabel("Records")
    ax.legend()
    written.append(_save_figure(fig, output_dir / "score_distribution_clean_vs_attacked.png", plt))
    return written


def _validate_alignment(
    clean_features: pd.DataFrame,
    clean_metadata: pd.DataFrame,
    attacked_metadata: pd.DataFrame,
) -> None:
    for frame in (clean_features, clean_metadata, attacked_metadata):
        if "row_id" not in frame or not frame["row_id"].is_unique:
            raise ValueError("Aligned feature and metadata frames require unique row_id values.")
    expected = clean_features["row_id"].tolist()
    if expected != clean_metadata["row_id"].tolist() or expected != attacked_metadata["row_id"].tolist():
        raise ValueError("Feature and metadata row_id values are misaligned.")


def _validate_reference(features: pd.DataFrame, metadata: pd.DataFrame) -> None:
    if "row_id" not in features or "row_id" not in metadata:
        raise ValueError("Projection references require row_id traceability.")
    if features["row_id"].tolist() != metadata["row_id"].tolist():
        raise ValueError("Projection reference features and metadata are misaligned.")


def _align_evaluation(
    ground_truth: pd.DataFrame, predictions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "record_id" not in ground_truth or "record_id" not in predictions:
        raise ValueError("Ground truth and predictions require record_id alignment.")
    if not ground_truth["record_id"].is_unique or not predictions["record_id"].is_unique:
        raise ValueError("Evaluation record_id values must be unique.")
    if set(ground_truth["record_id"]) != set(predictions["record_id"]):
        raise ValueError("Ground truth and prediction record_id sets differ.")
    truth = ground_truth.reset_index(drop=True)
    aligned = predictions.set_index("record_id").loc[truth["record_id"]].reset_index()
    return truth, aligned


def _affine_mapping(raw: pd.Series, processed: pd.Series) -> tuple[float, float]:
    raw_numeric = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
    processed_numeric = pd.to_numeric(processed, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(raw_numeric) & np.isfinite(processed_numeric)
    if finite.sum() < 2 or np.isclose(np.ptp(raw_numeric[finite]), 0.0):
        raise ValueError("Cannot recover a non-constant train-fitted affine feature mapping.")
    slope, intercept = np.polyfit(raw_numeric[finite], processed_numeric[finite], 1)
    return float(slope), float(intercept)


def _datetime_components(series: pd.Series) -> pd.DataFrame:
    values = pd.to_datetime(series, errors="coerce", format="mixed")
    if values.isna().any():
        raise ValueError("Timestamp attack projection encountered an invalid timestamp.")
    return pd.DataFrame(
        {
            "year": values.dt.year.astype(float),
            "month": values.dt.month.astype(float),
            "day": values.dt.day.astype(float),
            "day_of_week": values.dt.dayofweek.astype(float),
            "hour": values.dt.hour.astype(float),
            "is_weekend": values.dt.dayofweek.ge(5).astype(float),
        },
        index=series.index,
    )


def _save_figure(fig: Any, path: Path, pyplot: Any) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    pyplot.close(fig)
    return path
