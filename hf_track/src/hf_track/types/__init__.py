"""Type definitions for HuggingFace progress tracking.

This subpackage groups the data types by concept:

- :mod:`.events` -- progress event vocabulary (enums + ``ProgressEvent``)
- :mod:`.results` -- result envelope types (``TransferResult``, ``TransferError``)
- :mod:`.errors` -- exception classes
- :mod:`.ids`    -- transfer ID generator

All public symbols previously exposed by the flat ``types.py`` module
are re-exported here so existing imports keep working.
"""

from __future__ import annotations

from .errors import (
    TokenError,
    TransferCancelledError,
    TransferProgressError,
)
from .events import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferDirection,
)
from .ids import generate_transfer_id
from .results import TransferError, TransferResult

__all__ = [
    # Event vocabulary
    "EventType",
    "ProgressEvent",
    "ProgressPhase",
    "TransferDirection",
    # Result envelopes
    "TransferError",
    "TransferResult",
    # Exceptions
    "TokenError",
    "TransferCancelledError",
    "TransferProgressError",
    # Utilities
    "generate_transfer_id",
]
