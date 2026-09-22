"""Canonical serialization, timestamp normalization, and SHA-256 primitives.

Implements the EXACT V1.1-A frozen contract (config `canonical_serialization`,
`hash`, and `genesis`/`order_identity`). This module is a dependency leaf of the
engine: it imports no other engine module.

Frozen contract
---------------
* encoding: utf-8
* ``json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
  allow_nan=False)``, no trailing newline
* floats: finite, Python-round-trippable; NaN/Infinity raise
* times: pre-serialized ISO-8601 UTC strings ``%Y-%m-%dT%H:%M:%SZ`` only
* sentinel: JSON ``null`` for absent/optional items
* semantic hash: ``sha256(canonical_bytes(preimage))``
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from .errors import BlockchainEngineError, InvalidTimestampError

UTF_8 = "utf-8"
SORT_KEYS = True
SEPARATORS = (",", ":")
ENSURE_ASCII = True
ALLOW_NAN = False
NO_TRAILING_NEWLINE = True

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_TIMESTAMP_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")

SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: Any) -> str:
    """Return the canonical compact JSON string for ``value``.

    Raises ``ValueError`` (from ``json.dumps(..., allow_nan=False)``) for
    NaN/Infinity, which is the frozen deterministic NaN policy.
    """
    return json.dumps(
        value,
        sort_keys=SORT_KEYS,
        separators=SEPARATORS,
        ensure_ascii=ENSURE_ASCII,
        allow_nan=ALLOW_NAN,
    )


def canonical_bytes(value: Any) -> bytes:
    """Return the UTF-8 canonical bytes for ``value`` (no trailing newline)."""
    return canonical_json(value).encode(UTF_8)


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_hex(value: Any) -> str:
    """Return ``sha256(canonical_bytes(value)).hexdigest()`` (lowercase)."""
    return sha256_bytes(canonical_bytes(value))


def is_sha256_hex(value: Any) -> bool:
    """True if ``value`` is a lowercase 64-char SHA-256 hex string."""
    return isinstance(value, str) and bool(SHA256_HEX_PATTERN.fullmatch(value))


def normalize_timestamp_utc(value: Any) -> str:
    """Normalize a timestamp to the frozen canonical ``%Y-%m-%dT%H:%M:%SZ`` form.

    Rules (frozen): timestamps are pre-serialized ISO-8601 UTC strings. The
    engine therefore accepts
    * already-canonical ``...Z`` strings (round-trip verified), and
    * timezone-aware ``datetime`` objects (converted to UTC, second precision).

    Rejected deterministically (never silently guessed):
    * naive ``datetime`` (ambiguous timezone),
    * ``datetime`` with sub-second parts (cannot round-trip to the frozen form),
    * offset/fractional/non-Z strings (e.g. ``+00:00``, ``.123``), and
    * any other type.
    """
    if isinstance(value, str):
        if not _TIMESTAMP_RE.fullmatch(value):
            raise InvalidTimestampError(f"Not a canonical UTC timestamp string: {value!r}")
        parsed = datetime.strptime(value, TIMESTAMP_FORMAT)
        if parsed.strftime(TIMESTAMP_FORMAT) != value:
            raise InvalidTimestampError(f"Timestamp does not round-trip: {value!r}")
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise InvalidTimestampError(
                f"Naive datetime is ambiguous; provide UTC-aware value: {value!r}"
            )
        if value.microsecond != 0:
            raise InvalidTimestampError(
                f"Sub-second precision cannot round-trip to frozen format: {value!r}"
            )
        return value.astimezone(timezone.utc).strftime(TIMESTAMP_FORMAT)
    raise InvalidTimestampError(
        f"Unsupported timestamp type {type(value).__name__}: {value!r}"
    )


def is_canonical_timestamp(value: Any) -> bool:
    """True if ``value`` already is a canonical ``...Z`` UTC timestamp string."""
    try:
        return normalize_timestamp_utc(value) == value
    except InvalidTimestampError:
        return False


def timestamps_non_decreasing(previous: str, current: str) -> bool:
    """Chronological comparison of two canonical timestamps (frozen order check).

    Canonical ``YYYY-MM-DDTHH:MM:SSZ`` strings are zero-padded and sort
    lexicographically exactly like datetimes. Both inputs must be canonical.
    """
    if not is_canonical_timestamp(previous) or not is_canonical_timestamp(current):
        raise BlockchainEngineError("timestamps_non_decreasing requires canonical timestamps")
    return previous <= current


def canonical_order_id(value: Any) -> str:
    """Normalize an order identity to the canonical decimal integer string.

    Frozen rule: ``canonical decimal integer string; no sign, no leading
    zeros``. Accepts ``int`` or digit-only ``str`` (whitespace-stripped).
    Represented as a quoted string inside canonical JSON.
    """
    if isinstance(value, bool):
        raise BlockchainEngineError(f"Not a canonical order id: {value!r}")
    if isinstance(value, int):
        digits = str(value)
    elif isinstance(value, str):
        digits = value.strip()
    else:
        raise BlockchainEngineError(
            f"Unsupported order id type {type(value).__name__}: {value!r}"
        )
    if not digits.isdigit():
        raise BlockchainEngineError(f"Not a canonical decimal order id: {value!r}")
    canonical = str(int(digits))
    if canonical != digits:
        raise BlockchainEngineError(
            f"Non-canonical order id (leading zeros/sign): {value!r}"
        )
    return canonical