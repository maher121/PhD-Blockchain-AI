"""Core data structures and construction logic for the simulated ledger.

Scope and limitations
---------------------
Prototype V0.1 implements a single-chain, single-authority simulation for
research purposes. It does NOT provide consensus, mining, network
synchronisation, digital signatures or adversarial node behaviour. All of
these are out of scope until later research phases.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from src.blockchain.crypto import calculate_block_hash, calculate_transaction_hash
from src.config import MAX_TRANSACTIONS_PER_BLOCK

GENESIS_BLOCK_ID: int = 0
GENESIS_PREVIOUS_HASH: str = "0" * 64


def _fmt_timestamp(value: Any) -> str:
    """Normalize a timestamp-ish value to an ISO-8601 string."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc).isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Cannot format timestamp of type {type(value).__name__}: {value!r}")


def create_transaction(
    transaction_id: str,
    participant_id: str,
    product_id: str,
    order_id: str,
    timestamp: Any,
    quantity: float,
    transaction_status: str,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a transaction record and stamp it with its own content hash.

    Returns a dictionary with the canonical transaction fields plus ``hash``.
    The ``data_hash`` field is included so notebooks can double-check that
    ``hash == calculate_transaction_hash(record)``.
    """
    transaction: dict[str, Any] = {
        "transaction_id": str(transaction_id),
        "participant_id": str(participant_id),
        "product_id": str(product_id),
        "order_id": str(order_id),
        "timestamp": _fmt_timestamp(timestamp),
        "quantity": float(quantity),
        "transaction_status": str(transaction_status),
        "data_hash": None,
    }
    if extra_fields:
        transaction.update(extra_fields)

    transaction["data_hash"] = calculate_transaction_hash(transaction)
    transaction["hash"] = transaction["data_hash"]
    return transaction


def create_genesis_block() -> dict[str, Any]:
    """Create the immutable genesis block (no transactions, zero prev hash)."""
    return create_block(
        block_id=GENESIS_BLOCK_ID,
        transactions=[],
        previous_hash=GENESIS_PREVIOUS_HASH,
    )


def create_block(
    transactions: list[dict[str, Any]],
    block_id: int,
    previous_hash: str,
    timestamp: Any | None = None,
) -> dict[str, Any]:
    """Create a block that batches ``transactions`` into the ledger.

    Parameters
    ----------
    transactions : List of transaction records (see ``create_transaction``).
    block_id : Integer identifier of the block in the chain.
    previous_hash : ``current_block_hash`` of the preceding block.
    timestamp : Optional block creation time (defaults to now, UTC).

    Returns
    -------
    Block dictionary containing ``block_id``, ``timestamp``,
    ``transactions``, ``previous_block_hash`` and ``current_block_hash``.
    """
    if timestamp is None:
        timestamp = dt.datetime.now(dt.timezone.utc)

    block: dict[str, Any] = {
        "block_id": int(block_id),
        "timestamp": _fmt_timestamp(timestamp),
        "transactions": list(transactions),
        "previous_block_hash": str(previous_hash),
        "current_block_hash": None,
    }
    block["current_block_hash"] = calculate_block_hash(block)
    return block


def build_chain_from_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the ``blocks`` list unchanged (helper for symmetric APIs)."""
    return list(blocks)


def append_transaction_to_chain(
    chain: list[dict[str, Any]],
    transaction: dict[str, Any],
    max_transactions: int = MAX_TRANSACTIONS_PER_BLOCK,
) -> list[dict[str, Any]]:
    """Append a transaction to the last block, creating a new block when full.

    The genesis block (``block_id == 0``) is treated as immutable and never
    receives transactions; the first transaction starts the first data block.

    Returns a *new* chain list; the input chain is not mutated.
    """
    new_chain = list(chain)
    if not new_chain:
        new_chain = [create_genesis_block()]

    last_block = new_chain[-1]
    last_is_genesis = last_block["block_id"] == GENESIS_BLOCK_ID
    last_is_full = len(last_block["transactions"]) >= max_transactions

    if last_is_genesis or last_is_full:
        new_chain.append(
            create_block(
                block_id=len(new_chain),
                transactions=[],
                previous_hash=new_chain[-1]["current_block_hash"],
            )
        )
    else:
        new_chain[-1] = dict(new_chain[-1])
        new_chain[-1]["transactions"] = list(new_chain[-1]["transactions"])

    new_chain[-1]["transactions"].append(transaction)
    new_chain[-1]["current_block_hash"] = calculate_block_hash(new_chain[-1])
    return new_chain


def build_chain_from_transactions(
    transactions: list[dict[str, Any]],
    max_transactions: int = MAX_TRANSACTIONS_PER_BLOCK,
) -> list[dict[str, Any]]:
    """Batch a list of transactions into a complete chain starting at genesis."""
    chain: list[dict[str, Any]] = [create_genesis_block()]
    for transaction in transactions:
        chain = append_transaction_to_chain(
            chain, transaction, max_transactions=max_transactions
        )
    return chain