"""Security-facing facade: integrity checks and tamper attack simulation.

Prototype V0.1 provides *tamper-evidence*, not attacker resistance: the hashes
let a verifier reliably detect that something was modified after sealing.
"""

from __future__ import annotations

import logging
from typing import Any

from src.blockchain.validation import detect_tampering, verify_chain

logger = logging.getLogger(__name__)


def verify_chain_integrity(chain: list[dict[str, Any]]) -> bool:
    """Public integrity check: return True only for an intact chain."""
    return verify_chain(chain)


def detect_tampering(chain: list[dict[str, Any]]) -> dict[str, Any]:
    """Public tamper-detection entry point; see `validation.detect_tampering`."""
    return detect_tampering(chain)


def modify_transaction_after_sealing(
    chain: list[dict[str, Any]],
    block_id: int,
    transaction_id: str,
    field: str,
    new_value: Any,
) -> list[dict[str, Any]]:
    """Return a deep-ish copy of ``chain`` with one transaction field edited.

    This simulates an attacker silently editing a sealed transaction. The
    hashes are deliberately NOT recomputed, so the resulting chain resembles
    an unsophisticated tamper attempt that validation must catch.

    Notes
    -----
    - ``block_id`` refers to the index within the list version of the chain.
    - The returned chain is a new nested structure; ``chain`` is untouched.
    """
    tampered_chain: list[dict[str, Any]] = []
    found = False
    for block in chain:
        edited_block = dict(block)
        edited_block["transactions"] = list(block["transactions"])
        for index, transaction in enumerate(edited_block["transactions"]):
            tx = dict(transaction)
            if block["block_id"] == block_id and tx["transaction_id"] == transaction_id:
                tx[field] = new_value
                found = True
            edited_block["transactions"][index] = tx
        tampered_chain.append(edited_block)

    if not found:
        raise ValueError(
            f"Transaction {transaction_id!r} not found in block {block_id}."
        )

    logger.warning(
        "Simulated tampering: set %s=%r on %s (block %s); hashes left stale",
        field,
        new_value,
        transaction_id,
        block_id,
    )
    return tampered_chain