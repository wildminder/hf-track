"""Subprocess isolation primitives for Xet operations.

Subpackage layout:

- :mod:`.messages` -- ``SubprocessMessage`` IPC contract
- :mod:`.runner`   -- ``XetSubprocessRunner`` lifecycle manager

All public symbols previously exposed by the flat
``subprocess_messages.py`` and ``subprocess_runner.py`` modules are
re-exported here so existing imports keep working.
"""

from __future__ import annotations

from .messages import (
    MSG_CANCELLED,
    MSG_ERROR,
    MSG_EVENT,
    MSG_RESULT,
    SubprocessMessage,
)
from .runner import XetSubprocessRunner

__all__ = [
    "MSG_CANCELLED",
    "MSG_ERROR",
    "MSG_EVENT",
    "MSG_RESULT",
    "SubprocessMessage",
    "XetSubprocessRunner",
]
