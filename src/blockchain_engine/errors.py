"""Domain-specific exceptions for the V1.1-B lightweight blockchain engine."""

from __future__ import annotations


class BlockchainEngineError(Exception):
    """Base class for all engine-domain errors in the V1.1-B implementation."""


class InvalidBlockError(BlockchainEngineError):
    """A block violates the frozen V1.1-A block schema or construction rules."""


class InvalidEventTransitionError(BlockchainEngineError):
    """An event transition violates the frozen V1.1-A static event sequence."""


class InvalidTimestampError(BlockchainEngineError):
    """A timestamp cannot be deterministically normalized to the frozen UTC form."""


class BlockchainValidationError(BlockchainEngineError):
    """Raised when a caller requests validation against an inconsistent context."""