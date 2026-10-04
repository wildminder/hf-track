"""Xet authentication and capability detection.

Subpackage layout:

- :mod:`.credentials` -- ``XetCredentials`` value object
- :mod:`.manager`     -- ``XetTokenManager`` (upload + download resolution)
- :mod:`.capabilities` -- ``is_xet_available``, ``has_xet_session``

All public symbols previously exposed by the flat ``token.py`` module
are re-exported here so existing imports keep working.
"""

from __future__ import annotations

from .capabilities import has_xet_session, is_xet_available
from .credentials import XetCredentials
from .manager import XetTokenManager

__all__ = [
    "XetCredentials",
    "XetTokenManager",
    "is_xet_available",
    "has_xet_session",
]
