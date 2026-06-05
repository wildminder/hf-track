"""Async wrappers for HfTracker sync methods.

Each ``*_async`` method is a thin ``asyncio.to_thread`` wrapper around
the corresponding sync method. They preserve the exact signature
(``*args, **kwargs``) so callers can use either form interchangeably.

Split from the original ``tracker.py`` (28KB god class) on 2026-06-05
as part of the modular-refactor plan
(docs/plans/2026-06-04-modular-refactor.md, Step 8).
"""

from __future__ import annotations

import asyncio
from typing import Any


class _TrackerAsync:
    """Mixin: async wrappers for the synchronous download/upload methods.

    These methods exist so callers can ``await`` operations from
    inside an event loop without blocking it. The actual work runs
    on a worker thread via :func:`asyncio.to_thread`.
    """

    async def download_file_async(self, *args: Any, **kwargs: Any) -> str:
        """Asynchronous wrapper for :meth:`download_file`."""
        return await asyncio.to_thread(self.download_file, *args, **kwargs)  # type: ignore[attr-defined]

    async def download_snapshot_async(self, *args: Any, **kwargs: Any) -> str:
        """Asynchronous wrapper for :meth:`download_snapshot`."""
        return await asyncio.to_thread(self.download_snapshot, *args, **kwargs)  # type: ignore[attr-defined]

    async def upload_file_async(self, *args: Any, **kwargs: Any) -> str:
        """Asynchronous wrapper for :meth:`upload_file`."""
        return await asyncio.to_thread(self.upload_file, *args, **kwargs)  # type: ignore[attr-defined]

    async def upload_bytes_async(self, *args: Any, **kwargs: Any) -> str:
        """Asynchronous wrapper for :meth:`upload_bytes`."""
        return await asyncio.to_thread(self.upload_bytes, *args, **kwargs)  # type: ignore[attr-defined]

    async def upload_folder_async(self, *args: Any, **kwargs: Any) -> str:
        """Asynchronous wrapper for :meth:`upload_folder`."""
        return await asyncio.to_thread(self.upload_folder, *args, **kwargs)  # type: ignore[attr-defined]
