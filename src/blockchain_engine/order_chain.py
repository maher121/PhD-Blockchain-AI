"""Order-centric per-order logical chain (frozen V1.1-A architecture).

Each distinct order owns exactly one deterministic, hash-linked chain:
``GENESIS -> ORDER_CREATED -> SHIPMENT_RECORDED -> DELIVERY_STATUS_RECORDED
-> AI_RISK_ASSESSED`` (frozen static sequence). Blocks of unrelated orders
never co-occur in one chain; the engine rejects cross-order appends and
malformed event sequencing deterministically. No DataCo access.
"""

from __future__ import annotations

from typing import Any, Mapping

from .block import (
    FIXED_EVENT_SEQUENCE,
    Block,
    DEFAULT_VALIDATION_POLICY,
    construct_block,
    construct_genesis_block,
)
from .canonical import canonical_order_id, timestamps_non_decreasing
from .errors import InvalidBlockError, InvalidEventTransitionError


class OrderChain:
    """A single per-order logical chain with a deterministic genesis anchor.

    The chain starts frozen-free: it is constructed empty and gains its
    deterministic genesis block through :meth:`seed_genesis` (the order
    creation timestamp is a required, explicit, deterministic input; the
    engine never reads the wall clock).
    """

    def __init__(self, order_id: Any) -> None:
        self._order_id = canonical_order_id(order_id)
        self._blocks: list[Block] = []

    # ------------------------------------------------------------------ #
    # read-only state
    # ------------------------------------------------------------------ #
    @property
    def order_id(self) -> str:
        """Canonical chain identity (frozen canonical decimal integer string)."""
        return self._order_id

    @property
    def blocks(self) -> tuple[Block, ...]:
        """Immutable snapshot of the appended blocks (read-only exposure)."""
        return tuple(self._blocks)

    @property
    def length(self) -> int:
        """Number of appended blocks."""
        return len(self._blocks)

    @property
    def last_block(self) -> Block | None:
        """The head of the chain, or ``None`` when the chain is empty."""
        return self._blocks[-1] if self._blocks else None

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def seed_genesis(self, order_creation_canonical: Any) -> Block:
        """Seed the deterministic genesis block.

        ``order_creation_canonical`` is the canonical ISO-8601 UTC order
        creation timestamp (derived from the governed order date in V1.1-C;
        supplied as a fixed synthetic fixture in V1.1-B tests).
        """
        if self._blocks:
            raise InvalidBlockError("genesis already seeded; chain is not empty")
        genesis = construct_genesis_block(
            self._order_id, order_creation_canonical
        )
        self._blocks = [genesis]
        return genesis

    def next_event_type(self) -> str:
        """The frozen expected event type for the next append, if any.

        Raises ``InvalidEventTransitionError`` when the chain is complete
        (no further frozen event remains).
        """
        position = len(self._blocks)
        if position >= len(FIXED_EVENT_SEQUENCE):
            raise InvalidEventTransitionError(
                f"chain for order {self._order_id} is complete after "
                f"{len(FIXED_EVENT_SEQUENCE)} frozen events"
            )
        expected = FIXED_EVENT_SEQUENCE[position]
        if position == 0:
            raise InvalidEventTransitionError(
                f"genesis must be seeded via seed_genesis before appending "
                f"{expected!r}"
            )
        return expected

    def append_event(
        self,
        event_type: str,
        event_timestamp: Any,
        payload_digest: str,
        *,
        payload_ref: str | None = None,
        ai_risk_reference: Mapping[str, Any] | None = None,
        validation_policy: str = DEFAULT_VALIDATION_POLICY,
    ) -> Block:
        """Append the next frozen event as a linked block.

        The block index, previous-hash link, and expected event type are
        derived deterministically from the chain state; event timestamps must
        be non-decreasing per order.
        """
        expected = self.next_event_type()
        if event_type != expected:
            raise InvalidEventTransitionError(
                f"expected event {expected!r} for index {len(self._blocks)}, "
                f"got {event_type!r}"
            )
        previous_hash = self._blocks[-1].block_hash if self._blocks else None
        block = construct_block(
            order_id=self._order_id,
            block_index=len(self._blocks),
            event_type=event_type,
            event_timestamp=event_timestamp,
            payload_digest=payload_digest,
            validation_policy=validation_policy,
            payload_ref=payload_ref,
            ai_risk_reference=(
                dict(ai_risk_reference) if ai_risk_reference is not None else None
            ),
            previous_hash=previous_hash,
        )
        return self.append_block(block)

    def append_block(self, block: Block) -> Block:
        """Low-level append that enforces chain-local correctness rules.

        Deterministically rejects, in order:
        * cross-order blocks (``block.order_id != chain.order_id``),
        * index gaps / non-monotone indices (``index != chain length``),
        * malformed event sequencing (``event_type`` outside the frozen stem),
        * broken previous-hash linkage,
        * non-monotone event timestamps,
        * hashes that do not match the block content.
        """
        if not isinstance(block, Block):
            raise InvalidBlockError("append_block requires a Block instance")
        if block.order_id != self._order_id:
            raise InvalidBlockError(
                f"cross-order append: block belongs to order {block.order_id!r} "
                f"but chain is for order {self._order_id!r}"
            )
        if block.block_index != len(self._blocks):
            raise InvalidBlockError(
                f"index gap: chain has {len(self._blocks)} block(s), "
                f"block declares index {block.block_index}"
            )
        if block.block_index >= len(FIXED_EVENT_SEQUENCE):
            raise InvalidEventTransitionError(
                f"event {block.event_type!r} at index {block.block_index} is "
                f"outside the frozen sequence of {len(FIXED_EVENT_SEQUENCE)} events"
            )
        expected_type = FIXED_EVENT_SEQUENCE[block.block_index]
        if block.event_type != expected_type:
            raise InvalidEventTransitionError(
                f"malformed event sequencing: expected {expected_type!r} at "
                f"index {block.block_index}, got {block.event_type!r}"
            )
        if block.block_index == 0:
            raise InvalidBlockError(
                "append_block cannot seat a genesis block; use seed_genesis"
            )
        previous = self._blocks[-1]
        if block.previous_hash != previous.block_hash:
            raise InvalidBlockError(
                f"broken previous-hash linkage at index {block.block_index}"
            )
        if not timestamps_non_decreasing(previous.event_timestamp, block.event_timestamp):
            raise InvalidEventTransitionError(
                f"event timestamp not non-decreasing at index {block.block_index}: "
                f"{previous.event_timestamp!r} -> {block.event_timestamp!r}"
            )
        self._blocks.append(block)
        return block