"""Progress callback classes for intercepting transfer progress."""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from typing import Callable, Dict, Optional

from tqdm.auto import tqdm as base_tqdm

from .types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferCancelledError,
    TransferDirection,
)

logger = logging.getLogger(__name__)

class TransferStateManager:
    """Thread-safe manager for aggregate transfer states."""
    
    def __init__(self):
        self._states: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def init_download(self, transfer_id: str):
        with self._lock:
            if transfer_id not in self._states:
                self._states[transfer_id] = {
                    "files_completed": 0,
                    "total_files": 0,
                    "bytes_completed": 0,
                    "total_bytes": 0,
                    # Accumulation tracking for sequential byte bars:
                    # Each byte bar only knows its own file's total, so we
                    # track the current bar's total and the sum of all
                    # previously completed bars' totals to compute the
                    # aggregate total_bytes across the entire snapshot.
                    "_current_bar_total": 0,
                    "_committed_bytes": 0,
                }

    def init_upload(self, transfer_id: str, filename: str, total_bytes: int, event_queue: queue.Queue):
        with self._lock:
            if transfer_id not in self._states:
                self._states[transfer_id] = {
                    "filename": filename,
                    "total_bytes": total_bytes,
                    "bytes_completed": 0,
                    "event_queue": event_queue,
                    "completed_emitted": False,
                    "start_time": time.time(),
                }

    def update_download_files(self, transfer_id: str, files_completed: int, total_files: int):
        with self._lock:
            if transfer_id in self._states:
                self._states[transfer_id]["files_completed"] = files_completed
                if total_files > 0:
                    self._states[transfer_id]["total_files"] = total_files

    def reset_byte_bar_state(self, transfer_id: str):
        """Reset the per-byte-bar state for a NEW byte bar.

        Called by ``DownloadProgressTqdm.__init__`` when a new byte bar is
        created. This is how we distinguish:

        - **HTTP per-file byte bars**: each file gets its own
          ``DownloadProgressTqdm`` instance, so ``__init__`` is called
          once per file. Committing the previous bar's total here ensures
          aggregate ``total_bytes`` reflects all files.

        - **xet shared byte bar**: ``huggingface_hub.snapshot_download``
          creates ONE shared byte bar (via ``bytes_progress`` +
          ``_AggregatedTqdm``). ``__init__`` is called only once. The bar
          reports growing totals (cumulative) as files are discovered.
          No commit is needed during its lifetime.

        The function commits the previous bar's total to ``_committed_bytes``
        if there was one, then resets the per-bar state. ``total_bytes``
        is updated to reflect the committed amount.
        """
        with self._lock:
            if transfer_id not in self._states:
                return
            state = self._states[transfer_id]
            prev_bar_total = state["_current_bar_total"]
            if prev_bar_total > 0:
                state["_committed_bytes"] += prev_bar_total
            state["_current_bar_total"] = 0
            state["bytes_completed"] = state["_committed_bytes"]
            state["total_bytes"] = state["_committed_bytes"]

    def update_download_bytes(self, transfer_id: str, bytes_completed: int, total_bytes: int):
        with self._lock:
            if transfer_id not in self._states:
                return
            state = self._states[transfer_id]

            # Detect bar switch within a single bar's lifetime.
            # Three patterns are observed:
            #   1. **xet shared bar** (snapshot_download): same bar, total
            #      GROWS as files are discovered. ``update(n)`` is the
            #      per-file increment. NOT a bar switch.
            #   2. **HTTP per-file bar**: detected via ``reset_byte_bar_state``
            #      in ``__init__`` (commits previous bar before reset).
            #      During the bar's life, total is constant.
            #   3. **HTTP fallback / edge case**: if a new bar's ``__init__``
            #      was not called (e.g., bound class not properly hooked),
            #      we fall back to the legacy heuristic: total changed AND
            #      bytes_completed dropped.
            prev_bar_total = state["_current_bar_total"]
            prev_bar_bytes = state["bytes_completed"] - state["_committed_bytes"]

            # If total GREW, it's the xet-style "more files discovered"
            # pattern — same bar, do NOT commit.
            if total_bytes > prev_bar_total:
                bar_switched = False
            elif total_bytes < prev_bar_total:
                # Total shrank — likely a new bar with smaller per-file size.
                bar_switched = True
            else:
                # Same total. Check if bytes_completed dropped significantly.
                if prev_bar_total > 0 and bytes_completed < (prev_bar_bytes * 0.5):
                    bar_switched = True
                else:
                    bar_switched = False

            if bar_switched and prev_bar_total > 0:
                state["_committed_bytes"] += prev_bar_total
            if bar_switched:
                state["_current_bar_total"] = total_bytes
            elif prev_bar_total == 0 and total_bytes > 0:
                # First non-zero total for this bar.
                state["_current_bar_total"] = total_bytes

            # Aggregate bytes_completed = committed from previous files
            # + current bar's progress. Aggregate total_bytes = committed
            # + current bar's total.
            state["bytes_completed"] = state["_committed_bytes"] + bytes_completed
            state["total_bytes"] = state["_committed_bytes"] + total_bytes

    def add_upload_bytes(self, transfer_id: str, byte_increment: int) -> dict:
        """Accumulates bytes for multipart uploads and returns the current state."""
        with self._lock:
            state = self._states.get(transfer_id)
            if state:
                state["bytes_completed"] += byte_increment
                # Return a copy for safe event emission
                return dict(state)
            return {}

    def mark_upload_completed(self, transfer_id: str):
        with self._lock:
            state = self._states.get(transfer_id)
            if state:
                state["completed_emitted"] = True

    def get_state(self, transfer_id: str) -> dict:
        with self._lock:
            return dict(self._states.get(transfer_id, {}))

    def clear_state(self, transfer_id: str):
        with self._lock:
            self._states.pop(transfer_id, None)


# Global thread-safe state manager
state_manager = TransferStateManager()


class _DummyFile:
    """A dummy file-like object to silently sink tqdm output."""
    def write(self, x: str) -> int:
        return len(x)
    def flush(self) -> None:
        pass


_dummy_file = _DummyFile()


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
        def progress_updater(total_update, item_updates):
            return self(total_update, item_updates)
        return progress_updater

    def _resolve_display_completed(self, bytes_completed, total_bytes, transfer_completed):
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

    def __call__(self, total_update, item_updates):
        if self.is_cancelled and self.is_cancelled():
            raise TransferCancelledError("Transfer cancelled by user")

        item_update = next((item for item in item_updates if getattr(item, "item_name", "") == self.filename), None)

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
        dedup_saved = max(0, bytes_completed - transfer_completed) if self.total_files == 1 else 0

        self._emit(display_completed, total_bytes, percentage, active_speed, transfer_completed, transfer_total, transfer_speed, dedup_saved)

    def _emit(self, bytes_completed, total_bytes, percentage, speed, transfer_completed, transfer_total, transfer_speed, dedup_saved):
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
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            self._dropped_count += 1
            logger.warning("Progress event queue full — dropping %s event for %s; %d dropped total", event.event_type.value, event.filename, self._dropped_count)


class XetUploadProgressCallback(XetProgressCallback):
    """Backward-compatible subclass for Xet upload progress."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("direction", TransferDirection.UPLOAD)
        kwargs.setdefault("phase", ProgressPhase.UPLOADING)
        super().__init__(*args, **kwargs)


class XetDownloadProgressCallback(XetProgressCallback):
    """Backward-compatible subclass for Xet download progress."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("direction", TransferDirection.DOWNLOAD)
        kwargs.setdefault("phase", ProgressPhase.DOWNLOADING)
        super().__init__(*args, **kwargs)


class DownloadProgressTqdm(base_tqdm):
    def __init__(self, *args, **kwargs):
        self._event_queue = kwargs.pop("event_queue", None)
        self._transfer_id = kwargs.pop("transfer_id", "")
        self._filename = kwargs.pop("filename", "")
        self._report_interval = kwargs.pop("report_interval", 0.1)
        self._is_cancelled = kwargs.pop("is_cancelled", None)
        self._start_time = time.time()
        self._closed = False
        # Throttling state: avoid flooding the event queue (and IPC
        # mp.Queue in subprocess mode) with per-chunk progress events.
        self._last_emit_time: float = 0.0
        self._last_emit_bytes: int = 0
        self._first_event_emitted: bool = False
        # Track the previous self.n value to detect when update(n)
        # is called with an absolute position (xet) vs an increment
        # (standard HTTP). See update() for details.
        self._prev_n: int = 0

        self.is_bytes_bar = (kwargs.get("unit", "it") in ("B", "iB"))
        kwargs.pop("name", None)
        super().__init__(*args, **kwargs)

        if not self._filename:
            self._filename = getattr(self, "desc", "unknown") or "unknown"

        state_manager.init_download(self._transfer_id)
        if self.is_bytes_bar:
            state_manager.reset_byte_bar_state(self._transfer_id)
        else:
            state_manager.update_download_files(self._transfer_id, 0, getattr(self, "total", 0) or 0)

        # ── DIAG: temporary instrumentation (Phase 0.1) ──────────────
        import os as _os
        if _os.environ.get("HF_TRACK_DEBUG_XET"):
            try:
                import threading as _thr
                logger.debug(
                    "[DIAG-INIT] pid=%d tid=%s id=%s total=%s unit=%s desc=%s "
                    "is_bytes_bar=%s event_queue_is_none=%s has_cancel_hook=%s",
                    _os.getpid(),
                    _thr.get_ident(),
                    id(self),
                    getattr(self, "total", None),
                    getattr(self, "unit", None),
                    getattr(self, "desc", None),
                    self.is_bytes_bar,
                    self._event_queue is None,
                    self._is_cancelled is not None,
                )
            except Exception as _e:
                logger.debug("[DIAG-INIT] failed to log: %s", _e)
        # ── END DIAG ─────────────────────────────────────────────────

    def _should_throttle(self, agg_bytes_completed: int, agg_total_bytes: int, now: float) -> bool:
        """Return True if this PROGRESS event should be dropped to reduce flooding.

        Throttling rules:
        - Never throttle the very first event (ensures immediate feedback).
        - Never throttle if the transfer appears complete (bytes >= total).
        - Throttle if less than ``_report_interval`` seconds have elapsed
          AND the byte delta since last emit is less than 1% of total
          (or less than 1 KiB for small/unknown totals).
        """
        if not self._first_event_emitted:
            return False
        if agg_total_bytes > 0 and agg_bytes_completed >= agg_total_bytes:
            return False

        time_delta = now - self._last_emit_time
        if time_delta >= self._report_interval:
            return False

        bytes_delta = abs(agg_bytes_completed - self._last_emit_bytes)
        if agg_total_bytes > 0:
            min_delta = max(agg_total_bytes // 100, 1024)
        else:
            min_delta = 1024
        return bytes_delta < min_delta

    def update(self, n=1):
        # ── DIAG: temporary instrumentation (Phase 0.2) ──────────────
        import os as _os
        _diag = _os.environ.get("HF_TRACK_DEBUG_XET")
        if _diag:
            try:
                import threading as _thr
                _total_pre = getattr(self, "total", 0) or 0
                _n_pre = getattr(self, "n", 0)
                logger.debug(
                    "[DIAG-UPDATE-ENTRY] pid=%d tid=%s id=%s n=%s prev_n=%s "
                    "n_pre=%s total=%s is_bytes_bar=%s is_cancelled_hook=%s",
                    _os.getpid(), _thr.get_ident(), id(self),
                    n, self._prev_n, _n_pre, _total_pre,
                    getattr(self, "is_bytes_bar", False),
                    bool(self._is_cancelled and self._is_cancelled()),
                )
            except Exception:
                pass
        # ── END DIAG (entry) ─────────────────────────────────────────

        if self._is_cancelled is not None and self._is_cancelled():
            raise TransferCancelledError("Transfer cancelled by user")

        # ── Xet absolute-position fix ──────────────────────────────
        # huggingface_hub.xet_get() calls progress.update(progress_bytes)
        # where progress_bytes is the TOTAL bytes completed so far (not
        # an increment). But tqdm.update(n) expects n to be an INCREMENT
        # that gets ADDED to self.n. This causes self.n to grow far
        # beyond the actual total, breaking progress tracking.
        #
        # Detection: after super().update(n), if self.n > total AND
        # n > remaining (total - prev_n), then n was likely an absolute
        # position. We correct self.n to the absolute value.
        prev_n = self._prev_n
        result = super().update(n)

        # ── DIAG: post-super().update() ─────────────────────────────
        if _diag:
            try:
                logger.debug(
                    "[DIAG-UPDATE-POST-SUPER] pid=%d n=%s self.n=%s total=%s",
                    _os.getpid(), n, getattr(self, "n", 0), getattr(self, "total", 0),
                )
            except Exception:
                pass
        # ── END DIAG ─────────────────────────────────────────────────

        if n == 0:
            return result

        # For byte bars: detect and correct xet's absolute-position calls
        if getattr(self, "is_bytes_bar", False):
            total_val = getattr(self, "total", 0) or 0
            current_n = getattr(self, "n", 0)
            remaining = total_val - prev_n

            # If n exceeds remaining AND current_n exceeds total,
            # this was an absolute position call from xet_get().
            # Correct: self.n was (prev_n + n), should be just n.
            if total_val > 0 and n > remaining and current_n > total_val:
                # ── DIAG: absolute correction fired ───────────────────
                if _diag:
                    try:
                        logger.debug(
                            "[DIAG-ABS-CORRECTION] pid=%s n=%s prev_n=%s "
                            "old_self_n=%s new_self_n=%s total=%s remaining=%s",
                            _os.getpid(), n, prev_n, current_n, n, total_val, remaining,
                        )
                    except Exception:
                        pass
                # ── END DIAG ─────────────────────────────────────────
                # Reset to the absolute position
                setattr(self, "n", n)
                current_n = n

        self._prev_n = getattr(self, "n", 0)

        if not getattr(self, "is_bytes_bar", False):
            state_manager.update_download_files(self._transfer_id, getattr(self, "n", 0), getattr(self, "total", 0) or 0)
            # Emit a PROGRESS event for file-count bars so consumers
            # (web app, subprocess relay) receive per-file progress.
            # Without this, snapshot downloads show no progress updates
            # because only byte bars emitted events previously.
            state = state_manager.get_state(self._transfer_id)
            files_completed = state.get("files_completed", 0)
            total_files = state.get("total_files", 0)
            bytes_completed = state.get("bytes_completed", 0)
            total_bytes = state.get("total_bytes", 0)
            file_pct = ((files_completed / total_files * 100) if total_files > 0 else 0)
            event = ProgressEvent(
                event_type=EventType.PROGRESS,
                transfer_id=self._transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=self._filename,
                phase=ProgressPhase.DOWNLOADING,
                bytes_completed=bytes_completed,
                total_bytes=total_bytes,
                percentage=file_pct,
                speed=0,
                file_index=files_completed,
                total_files=total_files,
            )
            self._emit_event(event)
            return result

        now = time.time()
        bar_bytes_completed = getattr(self, "n", 0)
        bar_total_bytes = getattr(self, "total", 0) or 0

        # Update state_manager (accumulates across sequential byte bars)
        state_manager.update_download_bytes(self._transfer_id, bar_bytes_completed, bar_total_bytes)

        # Read aggregate values from state_manager for the event
        state = state_manager.get_state(self._transfer_id)
        agg_bytes_completed = state.get("bytes_completed", 0)
        agg_total_bytes = state.get("total_bytes", 0)
        files_completed = state.get("files_completed", 0)
        total_files = state.get("total_files", 0)
        percentage = ((agg_bytes_completed / agg_total_bytes * 100) if agg_total_bytes > 0 else 0)

        # Throttle: skip emitting if too soon and too little change
        if self._should_throttle(agg_bytes_completed, agg_total_bytes, now):
            # ── DIAG: throttled ──────────────────────────────────────
            if _diag:
                try:
                    logger.debug(
                        "[DIAG-THROTTLED] pid=%s agg_bytes=%s agg_total=%s "
                        "first_emitted=%s",
                        _os.getpid(), agg_bytes_completed, agg_total_bytes,
                        self._first_event_emitted,
                    )
                except Exception:
                    pass
            # ── END DIAG ─────────────────────────────────────────────
            return result

        speed = getattr(self, "format_dict", {}).get("rate") or 0
        if not speed and bar_bytes_completed > 0:
            elapsed = now - self._start_time
            speed = bar_bytes_completed / elapsed if elapsed > 0 else 0

        event = ProgressEvent(
            event_type=EventType.PROGRESS,
            transfer_id=self._transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename=self._filename,
            phase=ProgressPhase.DOWNLOADING,
            bytes_completed=agg_bytes_completed,
            total_bytes=agg_total_bytes,
            percentage=percentage,
            speed=speed or 0,
            file_index=files_completed,
            total_files=total_files,
        )
        self._emit_event(event)
        self._last_emit_time = now
        self._last_emit_bytes = agg_bytes_completed
        self._first_event_emitted = True

        # ── DIAG: emit-success ─────────────────────────────────────
        if _diag:
            try:
                logger.debug(
                    "[DIAG-EMIT] pid=%s event_type=PROGRESS bytes=%s total=%s pct=%.2f",
                    _os.getpid(), agg_bytes_completed, agg_total_bytes, percentage,
                )
            except Exception:
                pass
        # ── END DIAG ───────────────────────────────────────────────
        return result

    def _emit_event(self, event: ProgressEvent) -> None:
        """Emit a progress event to the event queue.

        Override this method in subclasses to route events through
        alternative channels (e.g. ``mp.Queue`` in subprocess workers).
        The default implementation puts events into ``self._event_queue``.
        If ``self._event_queue`` is ``None``, the event is silently dropped
        (useful when a subclass overrides this method to route elsewhere).
        """
        # ── DIAG: temporary instrumentation (Phase 0.3) ──────────────
        import os as _os
        _diag = _os.environ.get("HF_TRACK_DEBUG_XET")
        if _diag:
            try:
                import threading as _thr
                _d_size = -1
                try:
                    _d = event.to_dict()
                    _d_size = len(_d)
                except Exception as _td:
                    _d_size = -1
                    logger.debug(
                        "[DIAG-EMIT-EVENT-DICT-FAIL] pid=%s tid=%s err=%s",
                        _os.getpid(), _thr.get_ident(), _td,
                    )
                logger.debug(
                    "[DIAG-EMIT-EVENT] pid=%s tid=%s event_type=%s "
                    "bytes=%s total=%s dict_size=%s queue_is_none=%s",
                    _os.getpid(), _thr.get_ident(),
                    event.event_type.value, event.bytes_completed,
                    event.total_bytes, _d_size,
                    self._event_queue is None,
                )
            except Exception as _e:
                logger.debug("[DIAG-EMIT-EVENT] log failed: %s", _e)
        # ── END DIAG ─────────────────────────────────────────────────

        if self._event_queue is None:
            return

        # ── DIAG: capture put_nowait outcome ────────────────────────
        if _diag:
            try:
                self._event_queue.put_nowait(event)
                logger.debug(
                    "[DIAG-EMIT-PUT-OK] pid=%s event_type=%s",
                    _os.getpid(), event.event_type.value,
                )
            except queue.Full:
                logger.debug(
                    "[DIAG-EMIT-PUT-FULL] pid=%s event_type=%s",
                    _os.getpid(), event.event_type.value,
                )
                logger.warning(
                    "Download progress event queue full — dropping %s event",
                    event.event_type.value,
                )
            except BaseException as _put_err:  # DIAG: catch ALL
                logger.debug(
                    "[DIAG-EMIT-PUT-ERR] pid=%s err_type=%s err=%s",
                    _os.getpid(), type(_put_err).__name__, _put_err,
                )
                # Re-raise so we can see if the in-process path propagates
                raise
            return
        # ── END DIAG (with put) ──────────────────────────────────────

        # Non-DIAG path: original behavior
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            logger.warning("Download progress event queue full — dropping %s event", event.event_type.value)

    def close(self):
        # ── DIAG: temporary instrumentation (Phase 0.4) ──────────────
        import os as _os
        _diag = _os.environ.get("HF_TRACK_DEBUG_XET")
        if _diag:
            try:
                logger.debug(
                    "[DIAG-CLOSE-ENTRY] pid=%s id=%s n=%s total=%s "
                    "is_bytes_bar=%s event_queue_is_none=%s closed=%s",
                    _os.getpid(), id(self),
                    getattr(self, "n", 0), getattr(self, "total", 0),
                    getattr(self, "is_bytes_bar", False),
                    self._event_queue is None, self._closed,
                )
            except Exception:
                pass
        # ── END DIAG ─────────────────────────────────────────────────

        if self._closed:
            super().close()
            return
        self._closed = True

        if self._event_queue is not None and getattr(self, "total", None) is not None and getattr(self, "is_bytes_bar", False):
            n_val = getattr(self, "n", 0)
            total_val = getattr(self, "total", 0)

            is_complete = n_val >= total_val
            is_xet_cached = n_val == 0 and total_val > 0

            # ── DIAG: branch decision ────────────────────────────────
            if _diag:
                try:
                    logger.debug(
                        "[DIAG-CLOSE-BRANCH] pid=%s is_complete=%s "
                        "is_xet_cached=%s n_val=%s total_val=%s",
                        _os.getpid(), is_complete, is_xet_cached, n_val, total_val,
                    )
                except Exception:
                    pass
            # ── END DIAG ─────────────────────────────────────────────

            if is_complete or is_xet_cached:
                final_bytes = total_val if is_xet_cached else n_val
                state = state_manager.get_state(self._transfer_id)
                total_files = state.get("total_files", 0)
                files_completed = total_files if total_files > 0 else state.get("files_completed", 0)

                # ── DIAG: synthesize COMPLETE ──────────────────────────
                if _diag:
                    try:
                        logger.debug(
                            "[DIAG-CLOSE-SYNTH-COMPLETE] pid=%s final_bytes=%s "
                            "total_bytes=%s files_completed=%s total_files=%s",
                            _os.getpid(), final_bytes, total_val,
                            files_completed, total_files,
                        )
                    except Exception:
                        pass
                # ── END DIAG ─────────────────────────────────────────

                event = ProgressEvent(
                    event_type=EventType.COMPLETE,
                    transfer_id=self._transfer_id,
                    direction=TransferDirection.DOWNLOAD,
                    filename=self._filename,
                    phase=ProgressPhase.COMPLETE,
                    bytes_completed=final_bytes,
                    total_bytes=total_val,
                    percentage=100.0,
                    file_index=files_completed,
                    total_files=total_files,
                )
                self._emit_event(event)

        super().close()

    @classmethod
    def bind(
        cls,
        event_queue: queue.Queue,
        transfer_id: str,
        filename: str = "",
        report_interval: float = 0.1,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> type:
        _queue = event_queue
        _tid = transfer_id
        _fname = filename
        _interval = report_interval
        _cancel_hook = is_cancelled

        class BoundDownloadTqdm(cls):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("event_queue", _queue)
                kwargs.setdefault("transfer_id", _tid)
                kwargs.setdefault("filename", _fname)
                kwargs.setdefault("report_interval", _interval)
                kwargs.setdefault("is_cancelled", _cancel_hook)
                kwargs["file"] = _dummy_file
                super().__init__(*args, **kwargs)

        BoundDownloadTqdm.__name__ = f"BoundDownloadTqdm_{transfer_id[:8]}"
        BoundDownloadTqdm.__qualname__ = f"BoundDownloadTqdm_{transfer_id[:8]}"
        return BoundDownloadTqdm


@contextlib.contextmanager
def tqdm_upload_patcher(
    event_queue: queue.Queue,
    transfer_id: str = "",
    filename: str = "",
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
    total_bytes: int = 0,
):
    import tqdm.auto as tqdm_auto_module

    original_tqdm = tqdm_auto_module.tqdm
    _patch_active = True

    # Register this upload in the state manager.
    state_manager.init_upload(transfer_id, filename, total_bytes, event_queue)

    class UploadProgressTqdm(original_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["file"] = _dummy_file
            super().__init__(*args, **kwargs)
            
            self._managed_state = state_manager.get_state(transfer_id)
            
            self._upload_is_file_bar = (
                getattr(self, "total", None) is not None
                and getattr(self, "total", 0) > 0
                and getattr(self, "unit", "") in ("B", "iB")
            )
            
            self._last_n = 0

            if getattr(self, "_upload_is_file_bar", False) and self._managed_state and self._managed_state["bytes_completed"] == 0:
                event = ProgressEvent(
                    event_type=EventType.START,
                    transfer_id=transfer_id,
                    direction=TransferDirection.UPLOAD,
                    filename=self._managed_state["filename"],
                    phase=ProgressPhase.UPLOADING,
                    total_bytes=self._managed_state["total_bytes"],
                )
                try:
                    event_queue.put_nowait(event)
                except queue.Full:
                    logger.warning("Upload progress event queue full — dropping %s event", event.event_type.value)

        def update(self, n=1):
            if is_cancelled is not None and is_cancelled():
                raise TransferCancelledError("Transfer cancelled by user")

            result = super().update(n)

            if not getattr(self, "_upload_is_file_bar", False) or not _patch_active or n == 0:
                return result

            current_n = getattr(self, "n", 0)
            delta = current_n - self._last_n
            self._last_n = current_n

            if delta > 0:
                state = state_manager.add_upload_bytes(transfer_id, delta)
                if state:
                    bytes_completed = state["bytes_completed"]
                    total_bytes = state["total_bytes"]
                    percentage = ((bytes_completed / total_bytes * 100) if total_bytes > 0 else 0)

                    now = time.time()
                    elapsed = now - state["start_time"]
                    speed = bytes_completed / elapsed if elapsed > 0 else 0

                    event = ProgressEvent(
                        event_type=EventType.PROGRESS,
                        transfer_id=transfer_id,
                        direction=TransferDirection.UPLOAD,
                        filename=state["filename"],
                        phase=ProgressPhase.UPLOADING,
                        bytes_completed=bytes_completed,
                        total_bytes=total_bytes,
                        percentage=percentage,
                        speed=speed,
                    )
                    try:
                        state["event_queue"].put_nowait(event)
                    except queue.Full:
                        logger.warning("Upload progress event queue full — dropping %s event", event.event_type.value)
            return result

        def close(self):
            if not getattr(self, "_upload_is_file_bar", False) or not _patch_active:
                super().close()
                return
                
            state = state_manager.get_state(transfer_id)
            if state and not state.get("completed_emitted", False):
                if state["bytes_completed"] >= state["total_bytes"]:
                    state_manager.mark_upload_completed(transfer_id)
                    event = ProgressEvent(
                        event_type=EventType.COMPLETE,
                        transfer_id=transfer_id,
                        direction=TransferDirection.UPLOAD,
                        filename=state["filename"],
                        phase=ProgressPhase.COMPLETE,
                        bytes_completed=state["bytes_completed"],
                        total_bytes=state["total_bytes"],
                        percentage=100.0,
                    )
                    state["event_queue"].put(event)
                    
            super().close()

    tqdm_auto_module.tqdm = UploadProgressTqdm
    try:
        yield
    finally:
        _patch_active = False
        tqdm_auto_module.tqdm = original_tqdm