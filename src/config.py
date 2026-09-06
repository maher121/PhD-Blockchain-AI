"""Global configuration constants for Prototype V0.1.

Centralises paths, reproducibility seeds, schema names and default
estimation constants so that modules never hard-code these values.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR: Path = DATA_DIR / "synthetic"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

DEFAULT_RAW_DATA_FILE: Path = RAW_DATA_DIR / "supply_chain_raw.csv"
DEFAULT_PROCESSED_DATA_FILE: Path = PROCESSED_DATA_DIR / "supply_chain_processed.csv"
DEFAULT_SYNTHETIC_DATA_FILE: Path = SYNTHETIC_DATA_DIR / "supply_chain_synthetic.csv"
DEFAULT_MODEL_FILE: Path = MODELS_DIR / "isolation_forest_v01.joblib"
DEFAULT_RESULTS_FILE: Path = RESULTS_DIR / "prototype_v01_results.json"

GLOBAL_SEED: int = 42

LOGGING_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

RAW_SCHEMA: dict[str, str] = {
    "transaction_id": "object",
    "participant_id": "object",
    "product_id": "object",
    "order_id": "object",
    "timestamp": "datetime64[ns]",
    "quantity": "float64",
    "unit_price": "float64",
    "total_amount": "float64",
    "shipping_days": "float64",
    "location_region": "object",
    "transaction_status": "object",
}

# --- Data source configuration (Kaggle / local / synthetic) ----------------

#: Source resolution policy: "auto" tries local -> Kaggle -> synthetic fallback.
DATA_SOURCE: str = "auto"

#: Optional cap applied (with a fixed seed) before feature engineering /
#: blockchain so the notebook stays fast on the 180k-row Kaggle table.
PROCESS_MAX_ROWS: int = 20000

KAGGLE_DATASET_HANDLE: str = "shashwatwork/dataco-smart-supply-chain-for-big-data-analysis"
KAGGLE_FILE: str = "DataCoSupplyChainDataset.csv"

#: Mapping from the DataCo Kaggle columns to the canonical RAW_SCHEMA.
KAGGLE_TO_CANONICAL: dict[str, str] = {
    "Order Item Id": "transaction_id",
    "Customer Segment": "participant_id",
    "Product Card Id": "product_id",
    "Order Id": "order_id",
    "order date (DateOrders)": "timestamp",
    "Order Item Quantity": "quantity",
    "Order Item Product Price": "unit_price",
    "Sales": "total_amount",
    "Days for shipping (real)": "shipping_days",
    "Market": "location_region",
    "Order Status": "transaction_status",
}

#: Dataset-annotated 0/1 delivery-risk flag. This is a REAL (non-injected)
#: label column describing late-delivery risk -- it is NOT a cybersecurity
#: attack label.
KAGGLE_LABEL_SOURCE: str = "Late_delivery_risk"
NATURAL_LABEL_COLUMN: str = "natural_label"

#: Semantic labels describing the provenance of evaluation labels.
LABEL_KIND_CONTROLLED: str = "controlled_synthetic"
LABEL_KIND_NATURAL: str = "natural_late_delivery_risk"
LABEL_KIND_NONE: str = "none"

FEATURE_COLUMNS: list[str] = [
    "quantity",
    "unit_price",
    "total_amount",
    "shipping_days",
]

TRANSACTION_FIELDS: list[str] = [
    "transaction_id",
    "participant_id",
    "product_id",
    "order_id",
    "timestamp",
    "quantity",
    "transaction_status",
    "data_hash",
]

# Average power drawn by a typical consumer CPU while computing.
# Used ONLY by the estimation interface (Energy = Power x Time). It is an
# estimation convention, not a physical measurement of this machine.
DEFAULT_CPU_POWER_WATTS: float = 65.0
DEFAULT_RAM_POWER_WATTS: float = 3.0

ISOLATION_FOREST_PARAMS: dict[str, float | int | bool] = {
    "n_estimators": 200,
    "contamination": "auto",
    "max_samples": "auto",
    "random_state": GLOBAL_SEED,
}

ANOMALY_TRAIN_FRACTION: float = 0.7

MAX_TRANSACTIONS_PER_BLOCK: int = 8