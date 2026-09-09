"""Measured model training and inference helpers for V0.5."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import traceback
from typing import Any

import numpy as np
import pandas as pd

from src.lightweight.models import LightweightDetector
from src.lightweight.resource_monitor import (
    ResourceMeasurement,
    measure_call,
    measure_model_size,
)


@dataclass(frozen=True)
class TrainingResult:
    model: LightweightDetector
    artifact_path: Path
    resources: ResourceMeasurement
    model_size_bytes: int
    model_size_kb: float


@dataclass(frozen=True)
class InferenceResult:
    predictions: pd.DataFrame
    resources: ResourceMeasurement


def train_model(
    model: LightweightDetector,
    features: pd.DataFrame,
    labels: pd.Series | np.ndarray | None,
    artifact_path: Path | str,
    *,
    isolated: bool = True,
) -> TrainingResult:
    """Fit one model under the resource monitor and save its actual artifact."""
    path = Path(artifact_path).resolve()
    selected = features[model.feature_names].copy()
    if isolated:
        payload = _run_worker(
            _training_worker,
            model.model_name,
            model.feature_names,
            model.parameters,
            model.random_state,
            selected,
            None if labels is None else np.asarray(labels, dtype=int),
            str(path),
        )
        resources = ResourceMeasurement.from_dict(payload["resources"])
        fitted_model = LightweightDetector.load(path)
    else:
        _, resources = measure_call(
            lambda: model.fit(selected, labels), label=f"v0.5-{model.model_name}-training"
        )
        path = model.save(path)
        fitted_model = model
    size = measure_model_size(path)
    return TrainingResult(
        model=fitted_model,
        artifact_path=path,
        resources=resources,
        model_size_bytes=int(size["model_size_bytes"]),
        model_size_kb=float(size["model_size_kb"]),
    )


def run_inference(
    model: LightweightDetector,
    features: pd.DataFrame,
    *,
    artifact_path: Path | str | None = None,
    repeats: int = 1,
) -> InferenceResult:
    if repeats < 1:
        raise ValueError("Inference repeats must be positive.")
    selected = features[model.feature_names]
    predictions = model.predict_frame(selected)
    if artifact_path is not None:
        resources = measure_artifact_inference(
            artifact_path,
            selected,
            artifact_kind="v05",
            repeats=repeats,
        )
    else:
        _, resources = measure_call(
            lambda: _repeat_prediction(model, selected, repeats),
            label=f"v0.5-{model.model_name}-inference",
        )
        resources = resources.with_per_call_time(repeats)
    return InferenceResult(predictions=predictions, resources=resources)


def measure_artifact_inference(
    artifact_path: Path | str,
    features: pd.DataFrame,
    *,
    artifact_kind: str,
    repeats: int,
) -> ResourceMeasurement:
    """Measure inference in a fresh process so absolute peak RSS is comparable."""
    if artifact_kind not in {"v03", "v05"}:
        raise ValueError(f"Unsupported artifact kind: {artifact_kind}")
    payload = _run_worker(
        _inference_worker,
        str(Path(artifact_path).resolve()),
        artifact_kind,
        features.copy(),
        int(repeats),
    )
    return ResourceMeasurement.from_dict(payload["resources"]).with_per_call_time(repeats)


def reproduce_v03_training(
    feature_names: list[str],
    parameters: dict[str, Any],
    features: pd.DataFrame,
    artifact_path: Path | str,
) -> ResourceMeasurement:
    """Retrain the exact V0.3 methodology in a fresh current-environment process."""
    selected = features[feature_names].copy()
    payload = _run_worker(
        _v03_training_worker,
        feature_names,
        parameters,
        selected,
        str(Path(artifact_path).resolve()),
    )
    return ResourceMeasurement.from_dict(payload["resources"])


def _training_worker(
    queue: Any,
    model_name: str,
    feature_names: list[str],
    parameters: dict[str, Any],
    random_state: int,
    features: pd.DataFrame,
    labels: np.ndarray | None,
    artifact_path: str,
) -> None:
    try:
        model = LightweightDetector(
            model_name, feature_names, parameters, random_state=random_state
        )
        _, resources = measure_call(
            lambda: model.fit(features, labels), label=f"v0.5-{model_name}-training"
        )
        model.save(artifact_path)
        queue.put({"ok": True, "resources": resources.to_dict()})
    except BaseException:
        queue.put({"ok": False, "error": traceback.format_exc()})


def _inference_worker(
    queue: Any,
    artifact_path: str,
    artifact_kind: str,
    features: pd.DataFrame,
    repeats: int,
) -> None:
    try:
        if artifact_kind == "v03":
            from src.ai.anomaly_detector import IsolationForestBaseline

            model = IsolationForestBaseline.load(artifact_path)
        else:
            model = LightweightDetector.load(artifact_path)
        _, resources = measure_call(
            lambda: _repeat_prediction(model, features, repeats),
            label=f"v0.5-{artifact_kind}-inference",
        )
        queue.put({"ok": True, "resources": resources.to_dict()})
    except BaseException:
        queue.put({"ok": False, "error": traceback.format_exc()})


def _v03_training_worker(
    queue: Any,
    feature_names: list[str],
    parameters: dict[str, Any],
    features: pd.DataFrame,
    artifact_path: str,
) -> None:
    try:
        from src.ai.anomaly_detector import IsolationForestBaseline

        model = IsolationForestBaseline(feature_names, **parameters)
        _, resources = measure_call(
            lambda: model.fit(features), label="v0.5-v0.3-reference-training"
        )
        model.save(artifact_path)
        queue.put({"ok": True, "resources": resources.to_dict()})
    except BaseException:
        queue.put({"ok": False, "error": traceback.format_exc()})


def _repeat_prediction(model: Any, features: pd.DataFrame, repeats: int) -> None:
    for _ in range(repeats):
        model.predict_frame(features)


def _run_worker(target: Any, *args: Any) -> dict[str, Any]:
    methods = mp.get_all_start_methods()
    method = "forkserver" if "forkserver" in methods else "spawn"
    context = mp.get_context(method)
    queue = context.Queue()
    process = context.Process(target=target, args=(queue, *args))
    process.start()
    process.join(timeout=300)
    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError("Isolated resource measurement exceeded 300 seconds.")
    try:
        payload = queue.get(timeout=5)
    except Empty as exc:
        raise RuntimeError(
            f"Isolated resource worker exited with code {process.exitcode} without a result."
        ) from exc
    finally:
        queue.close()
        queue.join_thread()
    if not payload.get("ok"):
        raise RuntimeError(f"Isolated resource worker failed:\n{payload.get('error')}")
    if process.exitcode != 0:
        raise RuntimeError(f"Isolated resource worker exited with code {process.exitcode}.")
    return payload
