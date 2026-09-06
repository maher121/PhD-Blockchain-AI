"""End-to-end orchestration of the Prototype V0.1 workflow.

The pipeline composes the data, AI, blockchain, security and energy modules
into one reproducible sequence. The Jupyter notebook calls these same
functions cell-by-cell; it does NOT re-implement any of the steps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai.anomaly_detection import (
    IsolationForestAnomalyDetector,
    report_anomaly_metrics,
)
from src.blockchain.core import build_chain_from_transactions, create_transaction
from src.blockchain.validation import detect_tampering
from src.config import (
    ANOMALY_TRAIN_FRACTION,
    DATA_SOURCE,
    DEFAULT_MODEL_FILE,
    DEFAULT_PROCESSED_DATA_FILE,
    DEFAULT_RAW_DATA_FILE,
    GLOBAL_SEED,
    LABEL_KIND_CONTROLLED,
    LABEL_KIND_NATURAL,
    LABEL_KIND_NONE,
    NATURAL_LABEL_COLUMN,
    PROCESS_MAX_ROWS,
)
from src.data.loading import load_or_download_supply_chain_data
from src.energy.measurement import ExecutionTimer
from src.evaluation.reporting import save_experiment_results
from src.preprocessing.feature_engineering import build_feature_matrix
from src.preprocessing.preprocessing import preprocess_supply_chain_data
from src.security.integrity import modify_transaction_after_sealing

logger = logging.getLogger(__name__)


def load_or_download_dataset(
    raw_path: Path | str = DEFAULT_RAW_DATA_FILE,
    source: str = DATA_SOURCE,
    synthetic_n_rows: int = 2000,
) -> tuple[pd.DataFrame, str]:
    """Resolve the dataset following the configured ``source`` policy.

    Returns ``(dataframe, source_used)``; ``source_used`` is one of
    ``"local"``, ``"kaggle"`` or ``"synthetic"`` (see
    ``src.data.loading.load_or_download_supply_chain_data``).
    """
    return load_or_download_supply_chain_data(
        local_path=raw_path, source=source, synthetic_n_rows=synthetic_n_rows
    )


def _resolve_labels(features: pd.DataFrame, cleaned: pd.DataFrame) -> tuple[pd.Series | None, str]:
    """Pick the label column available in ``cleaned`` and align it to features.

    Resolution order:
    * ``known_anomaly``  -> controlled/synthetic labels (generator-injected).
    * ``natural_label``  -> real ``Late_delivery_risk`` flags from the Kaggle
      table (delivery-risk annotation, NOT a cybersecurity label).
    * otherwise          -> no labels (``LABEL_KIND_NONE``).
    """
    if "known_anomaly" in cleaned.columns:
        labels = cleaned.loc[features.index, "known_anomaly"].astype(int)
        return labels, LABEL_KIND_CONTROLLED

    if NATURAL_LABEL_COLUMN in cleaned.columns:
        raw_labels = pd.to_numeric(
            cleaned.loc[features.index, NATURAL_LABEL_COLUMN], errors="coerce"
        )
        valid_mask = raw_labels.notna() & raw_labels.isin([0, 1])
        return raw_labels[valid_mask].astype(int), LABEL_KIND_NATURAL

    return None, LABEL_KIND_NONE


def prepare_features(
    df: pd.DataFrame,
    processed_path: Path | str = DEFAULT_PROCESSED_DATA_FILE,
    max_rows: int = PROCESS_MAX_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None, str]:
    """Clean the data, persist the processed CSV and build the feature matrix.

    If the cleaned table exceeds ``max_rows`` (as the full 180k-row Kaggle
    table does), a fixed-seed sample is used for feature engineering so the
    notebook stays fast while remaining deterministic.

    Returns
    -------
    ``(processed_df, feature_matrix, labels, label_kind)`` where ``labels`` is
    None (and ``label_kind`` is ``"none"``) when no label column exists.
    """
    cleaned = preprocess_supply_chain_data(df)
    cleaned.to_csv(processed_path, index=False)

    sample = cleaned
    if len(cleaned) > max_rows:
        sample = cleaned.sample(n=max_rows, random_state=GLOBAL_SEED).sort_index()
        logger.info("Sampled %d of %d rows for feature engineering", max_rows, len(cleaned))

    features = build_feature_matrix(sample)
    labels, label_kind = _resolve_labels(features, cleaned)
    if labels is not None and NATURAL_LABEL_COLUMN in cleaned.columns:
        features = features.loc[labels.index]
        logger.info("Aligned %d feature rows with natural labels", len(features))

    return cleaned, features, labels, label_kind


def run_anomaly_detection(
    features: pd.DataFrame,
    labels: pd.Series | None = None,
    model_output_path: Path | str = DEFAULT_MODEL_FILE,
) -> dict[str, Any]:
    """Train Isolation Forest, predict, score and return an artifact bundle.

    ``labels`` is optional. When it is None (or holds a single class) the
    metric dicts are empty -- label-free reporting is still produced via
    ``predicted_anomaly`` and ``anomaly_score`` columns.
    """
    split_index = int(len(features) * ANOMALY_TRAIN_FRACTION)
    train_features = features.iloc[:split_index]
    test_features = features.iloc[split_index:]

    detector = IsolationForestAnomalyDetector()
    detector.fit(train_features)

    detector.save_model(model_output_path)
    train_predictions = detector.predict_anomalies(train_features)
    train_scores = detector.anomaly_scores(train_features)

    test_predictions = detector.predict_anomalies(test_features)
    test_scores = detector.anomaly_scores(test_features)

    train_labels = labels.iloc[:split_index] if labels is not None else None
    test_labels = labels.iloc[split_index:] if labels is not None else None
    train_metrics = report_anomaly_metrics(train_labels, train_predictions, train_scores)
    test_metrics = report_anomaly_metrics(test_labels, test_predictions, test_scores)

    results_df = pd.DataFrame(
        {
            "index": features.index[split_index:],
            "predicted_anomaly": test_predictions.to_numpy(),
            "anomaly_score": test_scores.to_numpy(),
        }
    )
    if test_labels is not None:
        results_df["label"] = test_labels.to_numpy()

    logger.info(
        "Anomaly detection done: %d test rows, %d flagged (train metrics: %s)",
        len(test_predictions),
        int(test_predictions.sum()),
        {k: round(v, 3) for k, v in train_metrics.items()} or "none",
    )
    return {
        "detector": detector,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "results_df": results_df,
    }


def tx_payload_schema(result_row: pd.Series) -> dict[str, Any]:
    """Return the canonical JSON-serializable transaction fields for a row."""
    return {
        "quantity": float(result_row["quantity"]),
        "transaction_status": str(result_row["transaction_status"]),
        "unit_price": float(result_row["unit_price"]),
        "predicted_anomaly": bool(bool(result_row["predicted_anomaly"])),
        "anomaly_score": float(result_row["anomaly_score"]),
    }


def build_blockchain_from_flags(
    scored_records: pd.DataFrame,
    max_transactions: int = 8,
) -> list[dict[str, Any]]:
    """Convert anomaly-flagged records into a simulated blockchain.

    Parameters
    ----------
    scored_records : DataFrame indexed by the original feature index and
        containing columns ``predicted_anomaly``, ``anomaly_score``,
        ``participant_id``, ``product_id``, ``order_id``, ``timestamp``,
        ``quantity`` and ``transaction_status``. Only rows where
        ``predicted_anomaly`` is True become transactions.
    max_transactions : Transaction capacity per block.
    """
    flagged = scored_records[scored_records["predicted_anomaly"]]
    transactions: list[dict[str, Any]] = []
    for index_value, row in flagged.iterrows():
        transaction = create_transaction(
            transaction_id=f"TXN-{int(index_value):06d}",
            participant_id=str(row["participant_id"]),
            product_id=str(row["product_id"]),
            order_id=str(row["order_id"]),
            timestamp=row["timestamp"],
            quantity=float(row["quantity"]),
            transaction_status=str(row["transaction_status"]),
            extra_fields={
                "predicted_anomaly": bool(row["predicted_anomaly"]),
                "anomaly_score": float(row["anomaly_score"]),
            },
        )
        transactions.append(transaction)
    return build_chain_from_transactions(transactions, max_transactions=max_transactions)


def run_prototype_v01(
    raw_path: Path | str = DEFAULT_RAW_DATA_FILE,
    processed_path: Path | str = DEFAULT_PROCESSED_DATA_FILE,
    model_path: Path | str = DEFAULT_MODEL_FILE,
    results_path: Path | str = "results/prototype_v01_results.json",
    source: str = DATA_SOURCE,
    synthetic_n_rows: int = 2000,
    max_rows: int = PROCESS_MAX_ROWS,
) -> dict[str, Any]:
    """Execute the full V0.1 workflow and persist its artifacts.

    Uses an ``ExecutionTimer`` around the whole research run to produce a
    single, reproducible measurement record for the report.
    """
    with ExecutionTimer(label="full-prototype-v01") as timer:
        raw_df, source_used = load_or_download_dataset(
            raw_path, source=source, synthetic_n_rows=synthetic_n_rows
        )
        processed_df, features, labels, label_kind = prepare_features(
            raw_df, processed_path=processed_path, max_rows=max_rows
        )
        ai_bundle = run_anomaly_detection(features, labels, model_output_path=model_path)
        results_df = ai_bundle["results_df"]

        aligned_processed = processed_df.loc[results_df["index"].astype(int)].copy()
        aligned_processed["predicted_anomaly"] = results_df.set_index("index")[
            "predicted_anomaly"
        ].astype(bool)
        aligned_processed["anomaly_score"] = results_df.set_index("index")[
            "anomaly_score"
        ]
        scored_records = aligned_processed[
            [
                "predicted_anomaly",
                "anomaly_score",
                "participant_id",
                "product_id",
                "order_id",
                "timestamp",
                "quantity",
                "transaction_status",
            ]
        ]
        chain = build_blockchain_from_flags(scored_records)

        tamper_scenario: dict[str, Any] | None = None
        if len(chain) >= 2 and chain[1]["transactions"]:
            target_block = chain[1]
            target_transaction = target_block["transactions"][0]
            tamper_scenario = modify_transaction_after_sealing(
                chain,
                block_id=target_block["block_id"],
                transaction_id=target_transaction["transaction_id"],
                field="quantity",
                new_value=-999.0,
            )

    payload = {
        "data_source": source_used,
        "label_kind": label_kind,
        "n_raw_rows": int(len(raw_df)),
        "n_processed_rows": int(len(processed_df)),
        "n_feature_rows": int(len(features)),
        "n_features": int(features.shape[1]),
        "n_flag_anomalies": int(results_df["predicted_anomaly"].sum()),
        "n_blocks": len(chain),
        "n_transactions_in_chain": sum(len(block["transactions"]) for block in chain),
        "train_metrics": ai_bundle["train_metrics"],
        "test_metrics": ai_bundle["test_metrics"],
        "measurement": timer.measurement.to_dict(),
        "tamper_detected_on_demo": bool(
            tamper_scenario is not None
            and detect_tampering(tamper_scenario)["tampered"]
        ),
        "label_note": (
            "Labels are either controlled/synthetic anomalies (synthetic source) "
            "or the Kaggle dataset's real Late_delivery_risk flag; NEITHER is a "
            "cybersecurity attack label."
        ),
    }

    save_experiment_results(payload, output_path=results_path)
    logger.info("Prototype V0.1 pipeline finished (source=%s).", source_used)
    return {
        "raw_df": raw_df,
        "processed_df": processed_df,
        "features": features,
        "labels": labels,
        "label_kind": label_kind,
        "ai_bundle": ai_bundle,
        "chain": chain,
        "tamper_scenario": tamper_scenario,
        "payload": payload,
    }