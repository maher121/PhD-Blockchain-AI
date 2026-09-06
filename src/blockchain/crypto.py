"""Cryptographic hashing for the simulated blockchain.

Prototype V0.1 uses SHA-256 from the standard library ``hashlib``.

IMPORTANT LIMITATION
--------------------
This is a *research simulation*, not a production consensus system. There is
no Proof-of-Work, no digital signatures and no node synchronisation. The
hashes provide tamper-evidence only: any change to a recorded transaction
produces a different digest, which the validation layer then detects.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _json_dumps(record: dict[str, Any]) -> str:
    """Serialize a record deterministically (sorted keys, indent-free)."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)


def calculate_transaction_hash(transaction: dict[str, Any]) -> str:
    """Return the SHA-256 hex digest over the canonical transaction record.

    The self-referential fields ``data_hash`` and ``hash`` are excluded from
    the digest so a (re-)computed hash never depends on the stored hash.
    """
    excluded_keys = {"data_hash", "hash"}
    canonical = {
        key: value for key, value in transaction.items() if key not in excluded_keys
    }
    payload = _json_dumps(canonical).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def calculate_block_hash(block: dict[str, Any]) -> str:
    """Return the SHA-256 hex digest over a block header and its contents.

    The digest covers ``block_id``, ``timestamp``, the hash of every
    contained transaction, and ``previous_block_hash``. ``current_block_hash``
    is excluded to keep the computation self-consistent.
    """
    transactions = block.get("transactions") or []
    if not isinstance(transactions, list):
        transactions = list(transactions)

    transaction_hashes = [
        tx["hash"] if isinstance(tx, dict) and "hash" in tx else calculate_transaction_hash(tx)
        for tx in transactions
    ]

    canonical: dict[str, Any] = {
        "block_id": block.get("block_id"),
        "timestamp": block.get("timestamp"),
        "previous_block_hash": block.get("previous_block_hash"),
        "transaction_hashes": transaction_hashes,
    }

    payload = _json_dumps(canonical).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tamper_evident_digest(transaction: dict[str, Any]) -> str:
    """Alias of :func:`calculate_transaction_hash` for readability."""
    return calculate_transaction_hash(transaction)