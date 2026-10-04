"""Type definitions for HuggingFace progress tracking.

This subpackage groups the data types by concept:

- :mod:`.events` -- progress event vocabulary (enums + ``ProgressEvent``)
- :mod:`.results` -- result envelope types (``TransferResult``, ``TransferErrorInfo``)
- :mod:`.errors` -- exception classes
- :mod:`.ids`    -- transfer ID generator

All public symbols previously exposed by the flat ``types.py`` module
are re-exported here so existing imports keep working.
"""

from __future__ import annotations

import warnings as _warnings
from typing import Any as _Any

from .errors import (
    TokenError,
    TransferCancelledError,
    TransferProgressError,
)
from .events import (
    TRANSPORT_HTTP,
    TRANSPORT_XET,
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferDirection,
    annotate_transport,
)
from .ids import generate_transfer_id
from .results import TransferErrorInfo, TransferResult

__all__ = [
    # Event vocabulary
    "TRANSPORT_HTTP",
    "TRANSPORT_XET",
    "EventType",
    "ProgressEvent",
    "ProgressPhase",
    "TransferDirection",
    "annotate_transport",
    # Result envelopes
    "TransferErrorInfo",
    "TransferResult",
    # Exceptions
    "TokenError",
    "TransferCancelledError",
    "TransferProgressError",
    # Utilities
    "generate_transfer_id",
]

#: Names that were renamed, mapped to their replacement. Kept for one
#: deprecation cycle; see ``hf_track.__getattr__`` for the same map at the
#: top-level package.
_RENAMED = {
    "TransferError": TransferErrorInfo,
}


def __getattr__(name: str) -> _Any:
    """Serve the pre-rename name, with a warning, for one cycle.

    ``TransferError`` was a dataclass sitting next to
    ``TransferCancelledError`` and ``TransferProgressError``, which are
    exceptions. The name invited ``except TransferError``, which never
    caught anything. It is now ``TransferErrorInfo``; this shim keeps
    ``from hf_track.types import TransferError`` working for one release
    cycle and then goes away.
    """
    replacement = _RENAMED.get(name)
    if replacement is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _warnings.warn(
        f"{name} was renamed to {replacement.__name__}: it is a dataclass, "
        f"not an exception. Import {replacement.__name__} and catch "
        f"TransferCancelledError / TransferProgressError for failures.",
        DeprecationWarning,
        stacklevel=2,
    )
    return replacement
