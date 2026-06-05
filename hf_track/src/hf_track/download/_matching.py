"""Glob pattern matching helper for download path filtering.

Used by ``xet_streaming.py`` to filter files by ``allow_patterns``
and ``ignore_patterns``. Kept in its own tiny module because the
``_`` prefix marks it as internal to the download subpackage.
"""

from __future__ import annotations

import os

def _matches_any(path: str, patterns) -> bool:
    """Return True if path matches any of the glob patterns (matches by basename)."""
    import fnmatch
    base = os.path.basename(path)
    for pat in patterns or []:
        if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(base, pat):
            return True
    return False
