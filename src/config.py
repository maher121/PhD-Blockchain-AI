"""Global configuration constants for Prototype V0.1 / V0.2.

Centralises paths, reproducibility seeds, schema names and default
estimation constants so that modules never hard-code these values.
V0.2 constants (DataCo dataset pipeline) live in the second half of this
file and do not modify any V0.1 behaviour on their own.
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
DOCS_DIR: Path = PROJECT_ROOT / "docs"

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

# ============================================================================
# V0.2 -- DataCo Smart Supply Chain Dataset Pipeline
# ============================================================================

#: V0.2 output locations (audit reports + figures are written under results/).
DATA_AUDIT_DIR: Path = RESULTS_DIR / "data_audit"
AUDIT_FIGURES_DIR: Path = DATA_AUDIT_DIR / "figures"
DATA_DICTIONARY_FILE: Path = DOCS_DIR / "data_dictionary.md"
DATASET_METADATA_FILE: Path = PROCESSED_DATA_DIR / "dataset_metadata.json"
PIPELINE_REPORT_FILE: Path = RESULTS_DIR / "dataco_pipeline_v02_report.json"

#: Processed-split layout under ``data/processed``.
PROCESSED_TRAIN_DIR: Path = PROCESSED_DATA_DIR / "train"
PROCESSED_VAL_DIR: Path = PROCESSED_DATA_DIR / "validation"
PROCESSED_TEST_DIR: Path = PROCESSED_DATA_DIR / "test"
PROCESSED_SPLIT_DIRS: dict[str, Path] = {
    "train": PROCESSED_TRAIN_DIR,
    "validation": PROCESSED_VAL_DIR,
    "test": PROCESSED_TEST_DIR,
}

#: Default DataCo raw filename (also the Kaggle file name). Discovery looks
#: for a file whose header matches the DataCo signature regardless of name.
DATACO_EXPECTED_FILE: Path = RAW_DATA_DIR / "DataCoSupplyChainDataset.csv"

#: Header signature used to recognise a raw DataCo table unambiguously.
DATACO_SIGNATURE_COLUMNS: tuple[str, ...] = (
    "Order Id",
    "Order Item Id",
    "Order Customer Id",
    "order date (DateOrders)",
    "Late_delivery_risk",
)
DATACO_SUPPORTED_EXTS: tuple[str, ...] = (".csv", ".xlsx")

#: Kaggle documentation of the canonical dataset (used only to *locate* a
#: previously-downloaded local cache; V0.2 never downloads automatically).
KAGGLE_DATASET_HANDLE: str = "shashwatwork/dataco-smart-supply-chain-for-big-data-analysis"
KAGGLE_FILE: str = "DataCoSupplyChainDataset.csv"

#: Local kagglehub cache layout scanned as a fallback when data/raw has no
#: DataCo file (e.g. the file was downloaded during V0.1 and cached at
#: ``~/.cache/kagglehub/datasets/<handle>/versions/1/<file>``).
KAGGLEHUB_CACHE_RELATIVE: tuple[str, ...] = (
    "datasets",
    "shashwatwork/dataco-smart-supply-chain-for-big-data-analysis",
    "versions",
)

# --- Train / validation / test split ---------------------------------------

#: Default ratios. Any change is possible but must be documented in metadata.
SPLIT_RATIOS: dict[str, float] = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

#: Split strategy. "order_grouped" keeps every row of one Order Id in the same
#: split (prevents order-level identity leakage between sets) and is the
#: default for V0.2. Alternatives: "random" (row-level, fixed seed) and
#: "chronological" (time-ordered, no random shuffling).
SPLIT_STRATEGY: str = "order_grouped"

#: Order/customer/product keys used for grouped splits and overlap checks.
SPLIT_GROUP_KEY: str = "Order Id"
LEAKAGE_IDENTITY_KEYS: tuple[str, ...] = (
    "Order Id",
    "Order Customer Id",
    "Product Card Id",
)

# --- Target variables -------------------------------------------------------

#: Primary operational prediction target of the DataCo table (real label:
#: 0/1 late-delivery risk). It is an operational / delivery outcome, NOT a
#: cybersecurity label -- the DataCo dataset provides no native cyber labels.
DATACO_TARGET_COLUMN: str = "Late_delivery_risk"

#: Outcome-adjacent columns that would leak the target if used as ML features.
#: They are kept as traceability metadata only.
DATACO_OUTCOME_COLUMNS: tuple[str, ...] = (
    "Delivery Status",
    "Days for shipping (real)",
    "Order Status",
)

# --- Column roles -----------------------------------------------------------

#: Columns treated as traceability / blockchain metadata (never ML features
#: by default): identifiers, PII, or high-cardinality keys.
DATACO_IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "Order Id",
    "Order Item Id",
    "Order Customer Id",
    "Customer Id",
    "Customer Email",
    "Customer Password",
    "Customer Fname",
    "Customer Lname",
    "Customer Street",
    "Customer Zipcode",
    "Order Zipcode",
    "Product Card Id",
    "Product Category Id",
    "Category Id",
    "Department Id",
    "Product Description",
    "Product Image",
    "Product Name",
    "Latitude",
    "Longitude",
    "Order City",
    "Order Country",
    "Order Region",
    "Order State",
    "Customer City",
    "Customer Country",
    "Customer State",
)

#: Columns that carry numeric information used as ML features.
DATACO_NUMERIC_FEATURES: tuple[str, ...] = (
    "Order Item Quantity",
    "Order Item Product Price",
    "Product Price",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Profit Ratio",
    "Order Item Total",
    "Days for shipment (scheduled)",
)

#: Categorical columns encoded with one-hot (top-k, fitted on TRAIN only).
DATACO_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "Type",
    "Market",
    "Shipping Mode",
    "Customer Segment",
    "Department Name",
)

#: Datetime columns available for temporal feature engineering.
DATACO_DATETIME_COLUMNS: tuple[str, ...] = (
    "order date (DateOrders)",
    "shipping date (DateOrders)",
)

#: Maximum one-hot cardinality fitted from TRAIN; rarer levels collapse to
#: an "OTHER" bucket so validation/test never see unseen levels.
MAX_ONEHOT_CARDINALITY: int = 15

#: Derived temporal features built from the order date (2015-01-01..2017-09-09
#: ranges in the real table).
TEMPORAL_FEATURES: tuple[str, ...] = (
    "year",
    "month",
    "day",
    "day_of_week",
    "hour",
    "is_weekend",
)

#: Maximum fraction of the raw table processed so the notebook stays fast.
#: APPLIED BEFORE the split (never after), so split probabilities are honest.
DATACO_MAX_ROWS: int = 40000

#: Preprocessing options (documented in every metadata record).
DATA_QUALITY_DROP_EMPTY_TARGET: bool = True
NEGATIVE_VALUE_COLUMNS: tuple[str, ...] = (
    "Order Item Quantity",
    "Order Item Product Price",
    "Product Price",
    "Order Item Total",
    "Sales",
)