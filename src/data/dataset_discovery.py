"""DataCo dataset discovery for Prototype V0.2.

Task
----
Locate the *real* DataCo SMART Supply Chain dataset in ``data/raw/`` and, if
it is not there, find it in the local ``kagglehub`` cache that the V0.1
documented download populated. V0.2 NEVER downloads the dataset by itself and
NEVER fabricates a stand-in: if no genuine DataCo file can be found it
returns a resolution object that clearly states the expected name/format.

Signature detection
-------------------
A file is recognised as the raw DataCo table when its header contains the
``DATACO_SIGNATURE_COLUMNS`` (e.g. ``Order Id``, ``Order Item Id``,
``Order Customer Id``, ``order date (DateOrders)``, ``Late_delivery_risk``).
The V0.1 synthetic table and the V0.1-normalised 12-column cache do NOT match
this signature and are therefore never mis-selected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import (
    DATACO_EXPECTED_FILE,
    DATACO_SIGNATURE_COLUMNS,
    DATACO_SUPPORTED_EXTS,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class DataCoResolution:
    """Result of dataset discovery.

    Attributes
    ----------
    path : The resolved dataset path, or None when no genuine DataCo table
        could be located anywhere locally.
    found_in_raw : True when an unambiguous DataCo file sat in ``data/raw``.
    candidates_raw : Candidate files found in ``data/raw`` (signature matches).
    cached_candidates : Genuine DataCo files found in the local kagglehub
        cache (fallback source).
    provenance : "raw_dir", "kagglehub_cache" or "none".
    staged_to : Where ``path`` was copied when provenance == "kagglehub_cache"
        (None otherwise).
    errors : Human-readable diagnostics when resolution is ambiguous.
    messages : Informational log messages.
    """

    path: Path | None = None
    found_in_raw: bool = False
    candidates_raw: list[Path] = field(default_factory=list)
    cached_candidates: list[Path] = field(default_factory=list)
    provenance: str = "none"
    staged_to: Path | None = None
    errors: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def _read_header(file_path: Path) -> list[str] | None:
    """Return the column names of a CSV/XLSX, or None when unreadable."""
    try:
        if file_path.suffix.lower() == ".xlsx":
            try:
                import openpyxl  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "XLSX support requires the optional 'openpyxl' package "
                    "(`pip install openpyxl`)."
                ) from exc
            raw = pd.read_excel(file_path, nrows=1)
        else:
            raw = pd.read_csv(file_path, nrows=1, encoding="latin1", low_memory=False)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not read candidate header %s: %s", file_path, exc)
        return None
    return [str(col).strip() for col in raw.columns]


def _matches_signature(columns: list[str]) -> tuple[bool, list[str]]:
    """Check whether a header matches the raw DataCo signature."""
    present = [col for col in DATACO_SIGNATURE_COLUMNS if col in columns]
    missing = [col for col in DATACO_SIGNATURE_COLUMNS if col not in columns]
    return len(missing) == 0, missing


def find_dataco_candidates(raw_dir: Path | str = RAW_DATA_DIR) -> list[Path]:
    """Return genuine DataCo files (signature match) found in a directory."""
    raw_dir = Path(raw_dir)
    candidates: list[Path] = []
    if not raw_dir.is_dir():
        return candidates
    for extension in DATACO_SUPPORTED_EXTS:
        for file_path in sorted(raw_dir.glob(f"*{extension}")):
            if file_path.is_file():
                columns = _read_header(file_path)
                if columns and _matches_signature(columns)[0]:
                    candidates.append(file_path)
    return candidates


def find_kagglehub_cache_candidates() -> list[Path]:
    """Locate genuine DataCo files inside the local kagglehub cache.

    The V0.1 README documents downloading the dataset through
    ``kagglehub.load_dataset(...)``. kagglehub stores the resolved file at
    ``~/.cache/kagglehub/datasets/<handle>/versions/<n>/<file>``. We scan that
    layout so an already-downloaded copy can be reused without the network.
    """
    handle = "shashwatwork/dataco-smart-supply-chain-for-big-data-analysis"
    base = Path.home() / ".cache" / "kagglehub" / "datasets" / handle / "versions"
    matches: list[Path] = []
    if not base.is_dir():
        return matches
    for version_dir in sorted(base.iterdir()):
        candidate = version_dir / "DataCoSupplyChainDataset.csv"
        if candidate.is_file():
            columns = _read_header(candidate)
            if columns and _matches_signature(columns)[0]:
                matches.append(candidate)
    return matches


def resolve_dataco_dataset(
    raw_dir: Path | str = RAW_DATA_DIR,
    stage_to: Path | None = None,
) -> DataCoResolution:
    """Resolve the DataCo dataset using only local resources.

    Discovery order:
    1. ``data/raw`` -- any unambiguous file matching the DataCo signature.
    2. local ``kagglehub`` cache -- a genuine DataCo table downloaded during
       a previous run (README-documented), copied into ``data/raw`` (staged)
       so downstream steps read one canonical location.
    3. none -- a resolution object with ``path is None`` and a clear message
       describing the expected file. No fabricated dataset is produced.

    Parameters
    ----------
    raw_dir : Directory searched for the DataCo table (default ``data/raw``).
    stage_to : Destination used when the file is taken from the kagglehub
        cache. Defaults to ``<raw_dir>/DataCoSupplyChainDataset.csv``.

    Returns
    -------
    DataCoResolution describing the outcome.
    """
    raw_dir = Path(raw_dir)
    if stage_to is None:
        stage_to = raw_dir / "DataCoSupplyChainDataset.csv"
    stage_to = Path(stage_to)

    resolution = DataCoResolution()

    raw_candidates = find_dataco_candidates(raw_dir)
    resolution.candidates_raw = raw_candidates

    if len(raw_candidates) == 1:
        resolution.path = raw_candidates[0]
        resolution.found_in_raw = True
        resolution.provenance = "raw_dir"
        resolution.messages.append(f"Found unambiguous DataCo file in data/raw: {raw_candidates[0]}")
        return resolution

    if len(raw_candidates) > 1:
        resolution.errors.append(
            "Multiple DataCo candidates in data/raw; refusing to pick silently: "
            + ", ".join(str(p) for p in raw_candidates)
        )
        resolution.provenance = "ambiguous_raw"
        return resolution

    cached_candidates = find_kagglehub_cache_candidates()
    resolution.cached_candidates = cached_candidates

    if len(cached_candidates) == 1:
        source = cached_candidates[0]
        raw_dir.mkdir(parents=True, exist_ok=True)
        stage_to.parent.mkdir(parents=True, exist_ok=True)
        # Deterministic byte-for-byte copy (do not touch the cache file).
        stage_to.write_bytes(source.read_bytes())
        resolution.path = stage_to
        resolution.staged_to = stage_to
        resolution.provenance = "kagglehub_cache"
        resolution.messages.append(
            f"DataCo dataset was not in data/raw; reused the genuine copy "
            f"already cached locally at {source} and staged it to {stage_to}."
        )
        return resolution

    if len(cached_candidates) > 1:
        resolution.errors.append(
            "Multiple DataCo copies found in the kagglehub cache; refusing to "
            "pick silently: " + ", ".join(str(p) for p in cached_candidates[:5])
        )
        resolution.provenance = "ambiguous_cache"
        return resolution

    resolution.errors.append(
        "No genuine DataCo dataset found. Expected a CSV/XLSX whose header "
        f"contains {list(DATACO_SIGNATURE_COLUMNS)}, e.g. "
        f"'{DATACO_EXPECTED_FILE.name}' under data/raw (or a previously "
        "downloaded copy in the local kagglehub cache). "
        "V0.2 does not fabricate or auto-download data."
    )
    resolution.provenance = "none"
    return resolution