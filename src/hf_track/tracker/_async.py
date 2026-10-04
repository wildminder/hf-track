"""Async wrappers for the synchronous download/upload methods.

NTH-002 and NTH-011 are one entry, not two. Both described the same gap —
"there is no async API" — on two different transports (download and
upload), and ``docs/reviews/issues-improvements.md:543`` says to merge
them when either is actioned. One ``asyncio.to_thread`` wrapper per sync
method covers both: the transport is chosen inside the sync method, not
here.

Every ``*_async`` method takes the *exact* signature of its sync
counterpart (``*args, **kwargs``), so the two forms are interchangeable::

    path = await tracker.download_file_async("bert-base-uncased", "config.json")
    path = tracker.download_file("bert-base-uncased", "config.json")

Cancellation
------------
``asyncio.to_thread`` cannot stop the thread it started: cancelling the
awaiting task abandons the ``await`` but leaves the transfer running. So
when the task is cancelled, this module calls ``self.cancel(transfer_id)``
before re-raising — the transfer then ends at its next cooperative
checkpoint.

That only works when the caller passed ``transfer_id=`` explicitly. The
sync methods generate one internally when it is absent, and the generated
id is not visible from out here, so a cancelled transfer started without
an explicit id runs to completion. That asymmetry is worth knowing: pass
``transfer_id`` if you intend to cancel.

Split from the original ``tracker.py`` (28KB god class) on 2026-06-05
as part of the modular-refactor plan
(docs/plans/2026-06-04-modular-refactor.md, Step 8).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable


class _TrackerAsync:
    """Mixin: async wrappers for the synchronous download/upload methods.

    These methods exist so callers can ``await`` operations from inside an
    event loop without blocking it. The actual work runs on a worker
    thread via :func:`asyncio.to_thread`.
    """

    async def _run_in_thread(
        self,
        method: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Await ``method`` on a worker thread, honouring task cancellation.

        Shared by every wrapper so the cancellation contract is written
        once: a cancelled ``await`` forwards to ``self.cancel(...)`` and
        then re-raises, so the caller sees the cancellation and the
        transfer stops too.
        """
        transfer_id = kwargs.get("transfer_id")
        task = asyncio.ensure_future(asyncio.to_thread(method, *args, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if transfer_id is not None:
                self.cancel(transfer_id)  # type: ignore[attr-defined]
            # The thread keeps running until it reaches a checkpoint; it
            # is not left unobserved, so no "exception was never
            # retrieved" warning is emitted on the way out.
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            raise

    async def download_file_async(self, *args: Any, **kwargs: Any) -> str:
        """Asynchronous wrapper for :meth:`download_file`."""
        return await self._run_in_thread(self.download_file, *args, **kwargs)  # type: ignore[attr-defined]

    async def download_snapshot_async(self, *args: Any, **kwargs: Any) -> str:
        """Asynchronous wrapper for :meth:`download_snapshot`."""
        return await self._run_in_thread(self.download_snapshot, *args, **kwargs)  # type: ignore[attr-defined]

    async def upload_file_async(self, *args: Any, **kwargs: Any) -> str:
        """Asynchronous wrapper for :meth:`upload_file` (NTH-011)."""
        return await self._run_in_thread(self.upload_file, *args, **kwargs)  # type: ignore[attr-defined]

    async def upload_bytes_async(self, *args: Any, **kwargs: Any) -> str:
        """Asynchronous wrapper for :meth:`upload_bytes`."""
        return await self._run_in_thread(self.upload_bytes, *args, **kwargs)  # type: ignore[attr-defined]

    async def upload_folder_async(self, *args: Any, **kwargs: Any) -> str:
        """Asynchronous wrapper for :meth:`upload_folder`."""
        return await self._run_in_thread(self.upload_folder, *args, **kwargs)  # type: ignore[attr-defined]