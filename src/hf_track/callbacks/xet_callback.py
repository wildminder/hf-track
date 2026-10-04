"""Xet progress callbacks.

This module adapts ``hf_xet``'s detailed progress reports
(``callback(total_update, item_updates)``) into ``ProgressEvent`` rows
that flow through the rest of the package.

Key design choices
==================

- :class:`XetProgressCallback` is a single class that handles BOTH
  upload and download — the only difference is the ``direction`` and
  ``phase`` it tags on emitted events, plus the dedup-aware display rule
  in :meth:`XetProgressCallback._resolve_display_completed`.
- :class:`XetUploadProgressCallback` and
  :class:`XetDownloadProgressCallback` are thin backward-compatible
  subclasses that pre-fill ``direction`` and ``phase``. They exist so
  historical import paths keep working.
- :class:`_DummyFile` is a no-op file handle used by
  :mod:`.tqdm_patch` to silence tqdm output during patcher execution.
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Callable, Optional

from ..types import (
    TRANSPORT_XET,
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferCancelledError,
    TransferDirection,
    annotate_transport,
)

logger = logging.getLogger(__name__)


# ── Internal helpers ────────────────────────────────────────────


class _DummyFile:
    """A dummy file-like object to silently sink tqdm output."""

    def write(self, x: str) -> int:
        return len(x)

    def flush(self) -> None:
        pass


_dummy_file = _DummyFile()


# ── Xet callback family ─────────────────────────────────────────


class XetProgressCallback:
    """Unified progress callback for Xet uploads and downloads.

    Args:
        filename: Name of the file being transferred.
        total_bytes: Expected total bytes for this file.
        event_queue: Queue to emit ProgressEvents into.
        direction: TransferDirection.UPLOAD or DOWNLOAD.
        phase: ProgressPhase.UPLOADING or DOWNLOADING.
        report_interval: Minimum seconds between progress reports (deprecated logic).
        transfer_id: Unique transfer identifier.
        file_index: Index of this file in a multi-file transfer.
        total_files: Total number of files in the transfer.
        is_cancelled: Optional callable returning True to cancel.
    """

    def __init__(
        self,
        filename: str,
        total_bytes: int,
        event_queue: queue.Queue,
        direction: TransferDirection = TransferDirection.UPLOAD,
        phase: ProgressPhase = ProgressPhase.UPLOADING,
        report_interval: float = 0.1,
        transfer_id: str = "",
        file_index: int = 0,
        total_files: int = 1,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ):
        self.filename = filename
        self.total_bytes = total_bytes
        self.event_queue = event_queue
        self.direction = direction
        self.phase = phase
        self.report_interval = report_interval
        self.transfer_id = transfer_id
        self.file_index = file_index
        self.total_files = total_files
        self.is_cancelled = is_cancelled
        self._dropped_count = 0
        self._start_time = time.time()

    def get_wrapper(self):
        """Return a top-level callable suitable for ``inspect.signature()``.

        The Rust runtime checks the callback's signature to decide between
        ``callback(total_update, item_updates)`` (detailed) and
        ``callback(progress_bytes)`` (legacy). This wrapper exposes the
        detailed form.
        """
        def progress_updater(total_update, item_updates):
            return self(total_update, item_updates)
        return progress_updater

    def _resolve_display_completed(
        self,
        bytes_completed: int,
        total_bytes: int,
        transfer_completed: int,
    ) -> int:
        """Choose which byte count to display as progress.

        For uploads, always show per-file bytes_completed.
        For downloads, prefer transfer_completed when per-file
        progress hasn't caught up yet (e.g. dedup-aware display).
        """
        if self.direction == TransferDirection.UPLOAD:
            return bytes_completed
        # Download: show transfer-level progress when per-file is stale
        if bytes_completed >= total_bytes > 0:
            return bytes_completed
        if transfer_completed > 0:
            return transfer_completed
        return bytes_completed

    def __call__(self, total_update, item_updates) -> None:
        if self.is_cancelled and self.is_cancelled():
            raise TransferCancelledError("Transfer cancelled by user")

        item_update = next(
            (item for item in item_updates if getattr(item, "item_name", "") == self.filename),
            None,
        )

        bytes_completed = getattr(total_update, "total_bytes_completed", 0)
        total_bytes = getattr(total_update, "total_bytes", 0) or self.total_bytes
        speed = getattr(total_update, "total_bytes_completion_rate", 0) or 0
        transfer_completed = getattr(total_update, "total_transfer_bytes_completed", 0)
        transfer_total = getattr(total_update, "total_transfer_bytes", 0)
        transfer_speed = getattr(total_update, "total_transfer_bytes_completion_rate", 0) or 0

        if self.total_files > 1 and item_update is not None:
            bytes_completed = getattr(item_update, "bytes_completed", 0)
            total_bytes = getattr(item_update, "total_bytes", 0) or self.total_bytes

        display_completed = self._resolve_display_completed(
            bytes_completed, total_bytes, transfer_completed
        )

        active_speed = transfer_speed or speed

        # NTH-001: Progress Estimation when bytes = 0 but speed > 0
        if display_completed == 0 and active_speed > 0:
            elapsed = time.time() - self._start_time
            estimated = int(active_speed * elapsed)
            if total_bytes > 0:
                # Cap the estimation at 99% to prevent jumping to completion
                estimated = min(estimated, int(total_bytes * 0.99))
            display_completed = estimated

        percentage = ((display_completed / total_bytes * 100) if total_bytes > 0 else 0)
        dedup_saved = (
            max(0, bytes_completed - transfer_completed)
            if self.total_files == 1
            else 0
        )

        self._emit(
            display_completed,
            total_bytes,
            percentage,
            active_speed,
            transfer_completed,
            transfer_total,
            transfer_speed,
            dedup_saved,
        )

    def _emit(
        self,
        bytes_completed: int,
        total_bytes: int,
        percentage: float,
        speed: float,
        transfer_completed: int,
        transfer_total: int,
        transfer_speed: float,
        dedup_saved: int,
    ) -> None:
        # Only include Xet-specific transfer stats for single-file transfers
        include_transfer_stats = self.total_files == 1
        event = ProgressEvent(
            event_type=EventType.PROGRESS,
            transfer_id=self.transfer_id,
            direction=self.direction,
            filename=self.filename,
            phase=self.phase,
            bytes_completed=bytes_completed,
            total_bytes=total_bytes,
            percentage=percentage,
            speed=speed,
            file_index=self.file_index,
            total_files=self.total_files,
            transfer_bytes_completed=transfer_completed if include_transfer_stats else 0,
            transfer_bytes_total=transfer_total if include_transfer_stats else 0,
            transfer_speed=transfer_speed if include_transfer_stats else 0,
            dedup_saved_bytes=dedup_saved,
        )
        annotate_transport(event, TRANSPORT_XET)
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            self._dropped_count += 1
            logger.warning(
                "Progress event queue full — dropping %s event for %s; %d dropped total",
                event.event_type.value,
                event.filename,
                self._dropped_count,
            )


class XetUploadProgressCallback(XetProgressCallback):
    """Backward-compatible subclass for Xet upload progress."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("direction", TransferDirection.UPLOAD)
        kwargs.setdefault("phase", ProgressPhase.UPLOADING)
        super().__init__(*args, **kwargs)


class XetDownloadProgressCallback(XetProgressCallback):
    """Backward-compatible subclass for Xet download progress."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("direction", TransferDirection.DOWNLOAD)
        kwargs.setdefault("phase", ProgressPhase.DOWNLOADING)
        super().__init__(*args, **kwargs)
