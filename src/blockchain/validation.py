"""Validation and tamper-detection logic for the simulated ledger."""

from __future__ import annotations

import logging
from typing import Any

from src.blockchain.crypto import calculate_block_hash, calculate_transaction_hash

logger = logging.getLogger(__name__)

REQUIRED_TRANSACTION_FIELDS: tuple[str, ...] = (
    "transaction_id",
    "participant_id",
    "product_id",
    "order_id",
    "timestamp",
    "quantity",
    "transaction_status",
)

REQUIRED_BLOCK_FIELDS: tuple[str, ...] = (
    "block_id",
    "timestamp",
    "transactions",
    "previous_block_hash",
    "current_block_hash",
)


def validate_transaction(transaction: dict[str, Any]) -> bool:
    """Validate a transaction record (structure + hash integrity).

    A transaction is valid when every required field is present and
    ``transaction["hash"]`` matches the freshly computed digest.

    Returns
    -------
    bool : True if the transaction is structurally valid and its stored hash
        matches the recomputed hash.
    """
    if not isinstance(transaction, dict):
        logger.error("Transaction is not a dict: %r", type(transaction).__name__)
        return False

    missing = [field for field in REQUIRED_TRANSACTION_FIELDS if field not in transaction]
    if missing:
        logger.error("Transaction missing required fields: %s", missing)
        return False

    stored_hash = transaction.get("hash")
    recalculated_hash = calculate_transaction_hash(transaction)
    if stored_hash is None or stored_hash != recalculated_hash:
        logger.warning(
            "Hash mismatch for transaction %s", transaction.get("transaction_id")
        )
        return False

    try:
        quantity = float(transaction["quantity"])
    except (TypeError, ValueError):
        logger.warning("Invalid quantity for transaction %s", transaction.get("transaction_id"))
        return False
    if quantity <= 0:
        logger.warning("Non-positive quantity for transaction %s", transaction.get("transaction_id"))
        return False

    return True


def validate_block(block: dict[str, Any], verify_transactions: bool = True) -> bool:
    """Validate a single block (structure, self-hash, and content hashes).

    Parameters
    ----------
    block : Block dictionary produced by ``src.blockchain.core``.
    verify_transactions : Whether stored transaction hashes are also checked.

    Returns
    -------
    bool : True if the block header hash matches the recomputed hash and
        (optionally) all contained transactions are valid.
    """
    if not isinstance(block, dict):
        logger.error("Block is not a dict: %r", type(block).__name__)
        return False

    missing = [field for field in REQUIRED_BLOCK_FIELDS if field not in block]
    if missing:
        logger.error("Block missing required fields: %s", missing)
        return False

    stored_hash = block.get("current_block_hash")
    block_without_current_hash = dict(block)
    block_without_current_hash["current_block_hash"] = None
    recalculated_hash = calculate_block_hash(block_without_current_hash)
    if stored_hash != recalculated_hash:
        logger.warning("Block hash mismatch for block %s", block.get("block_id"))
        return False

    if verify_transactions:
        for transaction in block.get("transactions", []):
            if not validate_transaction(transaction):
                return False

    return True


def verify_chain(chain: list[dict[str, Any]]) -> bool:
    """Verify the full chain: link integrity + every block's self-hash.

    Checks that:
    1. The chain is a non-empty list.
    2. Each block passes :func:`validate_block`.
    3. The genesis block has the canonical zero previous-hash.
    4. Every block references the previous block's stored hash.

    Returns
    -------
    bool : True only when the whole ledger is intact.
    """
    if not isinstance(chain, list) or not chain:
        logger.error("Chain is empty or not a list")
        return False

    for block in chain:
        if not validate_block(block):
            return False

    if chain[0]["previous_block_hash"] != "0" * 64:
        logger.error("Chain does not start with a canonical genesis block")
        return False

    for index in range(1, len(chain)):
        expected_previous = chain[index - 1]["current_block_hash"]
        actual_previous = chain[index]["previous_block_hash"]
        if actual_previous != expected_previous:
            logger.warning(
                "Broken link between blocks %d and %d", index - 1, index
            )
            return False

    return True


def detect_tampering(chain: list[dict[str, Any]]) -> dict[str, Any]:
    """Locate tampering inside the chain and return a diagnostic report.

    The report distinguishes between tampering at the block-header level
    (hash mismatch, broken link) and at the transaction level (a transaction
    record changed after its block was sealed).

    Returns
    -------
    dict with keys:

    * ``tampered``               - bool
    * ``tampered_blocks``        - list of affected block ids
    * ``tampered_transactions``  - list of ``transaction_id`` values
    * ``broken_links``           - list of (prev_block, next_block) pairs
    * ``report``                 - human-readable summary string
    """
    report: dict[str, Any] = {
        "tampered": False,
        "tampered_blocks": [],
        "tampered_transactions": [],
        "broken_links": [],
        "report": "",
    }

    if not isinstance(chain, list) or not chain:
        report["tampered"] = True
        report["report"] = "Chain is empty or not a list."
        return report

    transaction_counter = 0
    for index, block in enumerate(chain):
        block_intact = True

        stored_hash = block.get("current_block_hash")
        recomputed = calculate_block_hash(block)
        if stored_hash != recomputed:
            report["tampered_blocks"].append(block.get("block_id"))
            block_intact = False

        for transaction in block.get("transactions", []):
            stored_tx_hash = transaction.get("hash")
            recomputed_tx_hash = calculate_transaction_hash(transaction)
            if stored_tx_hash != recomputed_tx_hash:
                report["tampered_transactions"].append(
                    transaction.get("transaction_id", f"<tx-{transaction_counter}>")
                )
                block_intact = False
            transaction_counter += 1

        if index == 0 and block["previous_block_hash"] != "0" * 64:
            report["tampered_blocks"].append(block.get("block_id"))
            report["broken_links"].append(("genesis", "genesis"))
            block_intact = False

        if index > 0:
            expected = chain[index - 1]["current_block_hash"]
            if block["previous_block_hash"] != expected:
                report["broken_links"].append(
                    (chain[index - 1].get("block_id"), block.get("block_id"))
                )
                block_intact = False

    report["tampered"] = bool(
        report["tampered_blocks"]
        or report["tampered_transactions"]
        or report["broken_links"]
    )

    if report["tampered"]:
        report["tampered_blocks"] = sorted(set(report["tampered_blocks"]))
        report["broken_links"] = sorted(set(report["broken_links"]))
        report["report"] = (
            f"Tampering detected: {len(report['tampered_blocks'])} affected "
            f"block(s) ({report['tampered_blocks']}), "
            f"{len(report['tampered_transactions'])} tampered transaction(s), "
            f"{len(report['broken_links'])} broken link(s)."
        )
    else:
        report["report"] = "Chain integrity verified: no tampering detected."

    logger.info(report["report"])
    return report