"""CSV loading and dataset-source resolution for the supply-chain data.

Source-resolution order (``DATA_SOURCE = "auto"``):

1. If a canonical CSV already exists under ``data/raw`` -> load it locally.
2. Otherwise download the DataCo SMART supply-chain dataset from Kaggle with
   ``kagglehub``, normalise it to the canonical ``RAW_SCHEMA`` and cache the
   normalised CSV under ``data/raw`` for offline reuse.
3. If the download fails (missing credentials / no network), fall back to the
   deterministic synthetic generator and log a clear warning.

Downloading is optional: the rest of the framework never needs network access
once ``data/raw`` contains a CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_SOURCE,
    DEFAULT_RAW_DATA_FILE,
    KAGGLE_DATASET_HANDLE,
    KAGGLE_FILE,
    KAGGLE_LABEL_SOURCE,
    KAGGLE_TO_CANONICAL,
    NATURAL_LABEL_COLUMN,
    RAW_SCHEMA,
)
from src.data.generator import generate_supply_chain_dataset, write_supply_chain_dataset

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = tuple(RAW_SCHEMA.keys())

DATA_SOURCE_OPTIONS: tuple[str, ...] = ("auto", "local", "kaggle", "synthetic")


def load_supply_chain_data(file_path: Path | str) -> pd.DataFrame:
    """Load a canonical supply-chain CSV and validate its schema.

    Parameters
    ----------
    file_path : Path to a CSV conforming to ``RAW_SCHEMA`` (either produced by
        the synthetic generator, normalised from Kaggle, or cached locally).

    Returns
    -------
    pandas.DataFrame parsed with the canonical ``RAW_SCHEMA`` dtypes. Extra
    columns such as ``known_anomaly`` or ``natural_label`` are preserved.

    Raises
    ------
    FileNotFoundError if the file does not exist.
    ValueError if required columns are missing or the table is empty.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Found columns: {list(df.columns)}."
        )

    df = df.astype({column: dtype for column, dtype in RAW_SCHEMA.items()})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")

    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")

    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def download_kaggle_dataset(handle: str = KAGGLE_DATASET_HANDLE, file_path: str = KAGGLE_FILE) -> pd.DataFrame:
    """Download one file of a public Kaggle dataset into a pandas DataFrame.

    Uses ``kagglehub.load_dataset`` with the Pandas adapter. Requires either
    public access (no auth) or Kaggle credentials via ``KAGGLE_API_TOKEN`` /
    ``~/.kaggle/kaggle.json`` (see README).

    The DataCo CSV is latin-1 encoded (accented product/customer names), so
    ``pandas_kwargs`` overrides the default UTF-8 decoding.

    The kagglehub dependency is imported lazily so that the rest of the
    prototype works offline once a local CSV exists.
    """
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required to download from Kaggle. Install it with "
            "`pip install kagglehub[pandas-datasets]`."
        ) from exc

    logger.info("Downloading %s/%s from Kaggle ...", handle, file_path)
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        handle,
        file_path,
        pandas_kwargs={"encoding": "latin1", "low_memory": False},
    )
    logger.info("Kaggle download returned %d rows", len(df))
    return df


def normalize_kaggle_supply_chain(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Map the raw DataCo Kaggle columns onto the canonical ``RAW_SCHEMA``.

    The source table has no synthetic ``known_anomaly`` flag. When present,
    the dataset-annotated ``Late_delivery_risk`` column is preserved as
    ``natural_label`` (0/1) for optional evaluation -- it is a delivery-risk
    annotation, NOT a cybersecurity attack label.
    """
    missing = [column for column in KAGGLE_TO_CANONICAL if column not in raw_df.columns]
    if missing:
        raise ValueError(
            f"Kaggle table is missing expected columns: {missing}. "
            f"Found: {list(raw_df.columns)[:30]}"
        )

    canonical = raw_df[list(KAGGLE_TO_CANONICAL)].rename(columns=KAGGLE_TO_CANONICAL).copy()

    canonical["transaction_id"] = canonical["transaction_id"].astype(str)
    canonical["participant_id"] = canonical["participant_id"].astype(str)
    canonical["product_id"] = canonical["product_id"].astype(str)
    canonical["order_id"] = canonical["order_id"].astype(str)

    for column in ("quantity", "unit_price", "total_amount", "shipping_days"):
        canonical[column] = pd.to_numeric(canonical[column], errors="coerce").astype("float64")

    canonical["timestamp"] = pd.to_datetime(canonical["timestamp"], errors="coerce")
    canonical["location_region"] = canonical["location_region"].astype("object")
    canonical["transaction_status"] = canonical["transaction_status"].astype("object")

    if KAGGLE_LABEL_SOURCE in raw_df.columns:
        canonical[NATURAL_LABEL_COLUMN] = pd.to_numeric(
            raw_df[KAGGLE_LABEL_SOURCE], errors="coerce"
        ).astype("Int64")
    else:
        logger.warning("Column %r not found; natural_label omitted.", KAGGLE_LABEL_SOURCE)
        canonical[NATURAL_LABEL_COLUMN] = pd.NA

    canonical = canonical.dropna(subset=["transaction_id", "order_id", "timestamp"])
    canonical = canonical.astype({column: dtype for column, dtype in RAW_SCHEMA.items()})
    canonical = canonical.reset_index(drop=True)

    logger.info("Normalised %d Kaggle rows to canonical schema", len(canonical))
    return canonical


def load_or_download_supply_chain_data(
    local_path: Path | str = DEFAULT_RAW_DATA_FILE,
    source: str = DATA_SOURCE,
    synthetic_n_rows: int = 2000,
) -> tuple[pd.DataFrame, str]:
    """Resolve the dataset following the configured source policy.

    Parameters
    ----------
    local_path : Canonical CSV location checked first in "auto" mode.
    source : one of ``("auto", "local", "kaggle", "synthetic")``.
    synthetic_n_rows : Row count when falling back to the synthetic generator.

    Returns
    -------
    ``(dataframe, source_used)`` where ``source_used`` is ``"local"``,
    ``"kaggle"`` or ``"synthetic"``.

    Raises
    ------
    ValueError for an unknown ``source``.
    FileNotFoundError when ``source="local"`` and the file is absent.
    RuntimeError when ``source="kaggle"`` and the download fails.
    """
    if source not in DATA_SOURCE_OPTIONS:
        raise ValueError(
            f"Unknown source {source!r}; expected one of {DATA_SOURCE_OPTIONS}."
        )

    path = Path(local_path)

    if source in ("auto", "local") and path.exists():
        logger.info("Using local dataset %s", path)
        return load_supply_chain_data(path), "local"

    if source in ("auto", "kaggle"):
        try:
            raw = download_kaggle_dataset()
            canonical = normalize_kaggle_supply_chain(raw)
            path.parent.mkdir(parents=True, exist_ok=True)
            canonical.to_csv(path, index=False)
            logger.info("Cached normalised Kaggle dataset to %s", path)
            return canonical, "kaggle"
        except Exception as exc:
            if source == "kaggle":
                raise RuntimeError(
                    "Kaggle download failed. Provide credentials (see README) or "
                    f"drop a CSV at {path}."
                ) from exc
            logger.warning(
                "Kaggle download failed (%s). Falling back to the synthetic generator.",
                exc,
            )

    logger.warning("No local/Kaggle dataset; generating synthetic data (source=%s).", source)
    dataset = generate_supply_chain_dataset(n_rows=synthetic_n_rows)
    write_supply_chain_dataset(dataset, output_path=path)
    return dataset, "synthetic"