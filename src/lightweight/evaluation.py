"""Metrics, deterministic Pareto analysis, and figures for V0.5."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.security.evaluation import evaluate_detection


def calculate_metrics(
    ground_truth: pd.DataFrame, predictions: pd.DataFrame
) -> dict[str, Any]:
    """Calculate binary detection metrics against controlled V0.4 labels."""
    return evaluate_detection(ground_truth, predictions)


def identify_pareto_candidates(
    frame: pd.DataFrame,
    *,
    maximize: Sequence[str] = ("f1",),
    minimize: Sequence[str] = (
        "inference_time_sec",
        "memory_mb",
        "feature_count",
    ),
) -> pd.DataFrame:
    """Return rows not dominated across all stated objectives.

    A row is dominated when another row is at least as good on every objective
    and strictly better on one or more objectives. No evolutionary optimizer is
    used.
    """
    objectives = [*maximize, *minimize]
    missing = [column for column in objectives if column not in frame]
    if missing:
        raise ValueError(f"Pareto input is missing objectives: {missing}")
    if frame[objectives].isna().any().any():
        raise ValueError("Pareto objectives must contain valid finite values.")
    values = frame[objectives].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Pareto objectives must contain valid finite values.")

    keep = np.ones(len(frame), dtype=bool)
    max_count = len(maximize)
    for candidate in range(len(frame)):
        for challenger in range(len(frame)):
            if candidate == challenger:
                continue
            at_least = np.all(values[challenger, :max_count] >= values[candidate, :max_count])
            at_least = at_least and np.all(
                values[challenger, max_count:] <= values[candidate, max_count:]
            )
            strict = np.any(values[challenger, :max_count] > values[candidate, :max_count])
            strict = strict or np.any(
                values[challenger, max_count:] < values[candidate, max_count:]
            )
            if at_least and strict:
                keep[candidate] = False
                break
    result = frame.loc[keep].copy()
    result["pareto_objectives"] = (
        f"maximize {', '.join(maximize)}; minimize {', '.join(minimize)}"
    )
    return result.sort_values(
        [maximize[0], "inference_time_sec", "feature_count"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def save_lightweight_figures(
    feature_results: pd.DataFrame,
    model_comparison: pd.DataFrame,
    pareto_candidates: pd.DataFrame,
    output_dir: Path | str,
) -> list[Path]:
    """Create five single-axis computational trade-off figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    plots = (
        ("inference_time_sec", "f1", "Test inference time (seconds)", "Test F1", "f1_vs_inference_time.png"),
        ("feature_count", "f1", "Number of features", "Test F1", "f1_vs_feature_count.png"),
        ("feature_count", "inference_time_sec", "Number of features", "Test inference time (seconds)", "inference_time_vs_features.png"),
        ("feature_count", "memory_mb", "Number of features", "Test peak process RSS (MB)", "memory_vs_features.png"),
    )
    for x_name, y_name, x_label, y_label, filename in plots:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for model, group in feature_results.groupby("model", sort=False):
            ordered = group.sort_values("feature_count")
            ax.plot(ordered[x_name], ordered[y_name], marker="o", label=model)
        if filename == "f1_vs_inference_time.png" and not pareto_candidates.empty:
            ax.scatter(
                pareto_candidates["inference_time_sec"],
                pareto_candidates["f1"],
                marker="*",
                s=130,
                color="black",
                label="Validation-selected Pareto candidate",
                zorder=5,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{y_label} vs {x_label}")
        ax.legend(fontsize=8)
        written.append(_save(fig, output / filename, plt))

    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = model_comparison.sort_values("f1", ascending=False)
    bars = ax.bar(ordered["model"], ordered["f1"], color="#356859")
    ax.bar_label(bars, fmt="%.3f", padding=2)
    ax.set_ylabel("F1")
    ax.set_title("Full-feature model comparison on the primary V0.4 scenario")
    ax.tick_params(axis="x", rotation=25)
    written.append(_save(fig, output / "model_comparison.png", plt))
    return written


def _save(fig: Any, path: Path, pyplot: Any) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    pyplot.close(fig)
    return path
