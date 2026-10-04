"""Shared machinery for every Xet subprocess worker (NTH-016, step S32).

Split out of ``_xet_worker.py`` so that the worker entry points can live
in modules sized for their job instead of all sharing one 1,700-line file.
Nothing here imports ``hf_xet`` at module level, and nothing here is a
worker: it is the plumbing every worker needs.

Contents:

* ``_init_worker`` / ``_safe_put`` / ``_handle_worker_exception`` — the
  process-level rules. ``_safe_put`` is why a worker never blocks or
  raises while reporting, and ``_handle_worker_exception`` is the single
  place a worker failure becomes a terminal message.
* ``_new_xet_session`` — the only place in the package that calls
  ``huggingface_hub.utils._xet.get_xet_session``.
* ``_ProgressThrottler`` — keeps the mp queue from filling with
  per-chunk events.
* ``_serialize_xet_file_data`` / ``_deserialize_xet_file_data`` — the
  picklable form of ``XetFileData`` that crosses the process boundary.
* ``_make_progress_callback`` — builds the callback every download and
  upload worker passes to the Xet runtime.

The workers themselves must stay importable at *module top level*:
``multiprocessing``'s ``spawn`` start method pickles them by qualified
name, so a nested function is not spawnable.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import signal
import time
from typing import Any, Dict, List, Optional

from .subprocess.messages import SubprocessMessage
from .types import EventType, ProgressEvent, ProgressPhase, TransferCancelledError

logger = logging.getLogger(__name__)



# ── Initialization & Safe IO ─────────────────────────────────────

def _new_xet_session():
    """Return a live ``hf_xet`` session handle.

    The single place in this module that touches
    ``huggingface_hub.utils._xet.get_xet_session``. Five worker entry
    points used to open their own session from their own import block, so
    the credential API was pinned in five places — and when
    ``huggingface_hub`` 1.x moved it (it was two other names before), five
    independent edits were needed to follow, which is how the breakage
    reached a spawned subprocess where it could only surface at transfer
    time.

    The import stays inside the function: ``hf_xet`` is a compiled PyO3
    extension, and importing it in the parent would load the Rust runtime
    into a process that must stay terminable. Workers are spawned
    precisely so they can be killed.
    """
    from huggingface_hub.utils._xet import get_xet_session

    return get_xet_session()


def _init_worker() -> None:
    """Initialize worker process state.
    
    Ignores SIGINT so that Ctrl+C is exclusively handled by the main
    process (which will cleanly terminate this worker).
    """
    try:
        if mp.current_process().name != "MainProcess":
            signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass


        pass


def _safe_put(mp_queue: mp.Queue, message: SubprocessMessage) -> None:
    """Put a message to the multiprocessing queue, suppressing all errors.

    Uses ``put_nowait`` so the call never blocks (a blocking ``put`` can
    deadlock if the queue's feeder thread is busy or the buffer is full).
    Catches ``BaseException`` (including ``KeyboardInterrupt``) so that
    a second interrupt during error/cancel handling never produces a
    traceback from the subprocess.
    """
    try:
        mp_queue.put_nowait(message)
    except BaseException:
        pass


        pass


def _handle_worker_exception(
    mp_queue: mp.Queue,
    e: BaseException,
    *,
    transfer_id: str = "",
    direction: str = "download",
    filename: str = "",
) -> None:
    """Safely format and send an exception as a terminal message."""
    try:
        from .types import TransferCancelledError
        if isinstance(e, KeyboardInterrupt):
            _safe_put(mp_queue, SubprocessMessage.cancelled(
                message="Transfer interrupted by user (Ctrl+C)",
                transfer_id=transfer_id,
                direction=direction,
                filename=filename,
            ))
        elif isinstance(e, TransferCancelledError):
            _safe_put(mp_queue, SubprocessMessage.cancelled(
                message="Transfer cancelled by user",
                transfer_id=transfer_id,
                direction=direction,
                filename=filename,
            ))
        else:
            _safe_put(mp_queue, SubprocessMessage.error(
                message=str(e),
                error_type=type(e).__name__,
            ))
    except BaseException:
        pass


# ── Progress Throttler (shared by all xet workers) ───────────────


class _ProgressThrottler:
    """Time + byte-delta based throttling for progress events.

    Used by the active workers (``_download_worker``, ``_snapshot_worker``,
    ``_streaming_download_worker``) and the legacy
    ``_make_progress_callback`` (2-arg mode).

    Behavior:
        - The first event is always emitted (regardless of value).
        - Subsequent events are throttled by:
            * Time elapsed since last emit (``report_interval`` seconds)
            * OR byte delta since last emit (1% of total or 1 KiB,
              whichever is larger)
        - The completion event (display >= total) is always emitted,
            so the user always sees 100% at the end.
        - ``record_emit()`` must be called after each successful emit
            to update internal state.

    .. note::

        Previous throttling used function-attribute hacks
        (``on_progress._last_emit``) which had a subtle bug: the first
        call with no byte change would be silently dropped, and the
        first call with a non-zero byte change would have
        ``_last_emit = 0.0`` set, then any subsequent call with
        ``bytes_change <= 0`` would also be dropped. This caused the
        progress bar to never update.

        See: 2026-06-03 plan "migrate-snapshot-to-xetsession-api".

    Args:
        report_interval: Minimum seconds between consecutive emits.
        min_bytes_delta: Minimum byte delta to bypass time throttling
            (default: 1% of total or 1 KiB, whichever is larger — set
            in ``should_emit``).
    """

    def __init__(self, report_interval: float = 0.1):
        self._report_interval = max(0.0, report_interval)
        self._last_emit_time: float = 0.0
        self._last_emit_value: int = 0
        self._first_event_emitted: bool = False

    def should_emit(self, current_value: int, total: int, now: float) -> bool:
        """Return True if a progress event should be emitted.

        Args:
            current_value: Current bytes_completed (or display value).
            total: Total bytes expected.
            now: Current time (typically ``time.time()``).
        """
        # First event: always emit
        if not self._first_event_emitted:
            return True
        # Completion: always emit (so user sees 100%)
        if total > 0 and current_value >= total:
            return True
        # Time throttle
        time_delta = now - self._last_emit_time
        if time_delta >= self._report_interval:
            return True
        # Byte delta throttle
        bytes_delta = abs(current_value - self._last_emit_value)
        min_bytes_delta = max(total // 100, 1024) if total > 0 else 1024
        if bytes_delta >= min_bytes_delta:
            return True
        # Both throttled — skip
        return False

    def record_emit(self, current_value: int, now: float) -> None:
        """Record that an event was emitted.

        Updates internal state so subsequent ``should_emit`` calls
        correctly throttle.
        """
        self._last_emit_time = now
        self._last_emit_value = current_value
        self._first_event_emitted = True

    def reset(self) -> None:
        """Reset throttler state (for reuse in new transfers)."""
        self._last_emit_time = 0.0
        self._last_emit_value = 0
        self._first_event_emitted = False



# ── Serialization Helpers ────────────────────────────────────────

def _serialize_xet_file_data(xet_file_data: Any) -> Dict[str, Any]:
    """Convert an XetFileData object to a plain dict for pickling.

    ``XetFileData`` is a namedtuple from ``huggingface_hub`` with fields
    ``file_hash`` and ``refresh_route``. We serialize it to a dict so
    it can cross the process boundary.
    """
    if xet_file_data is None:
        return {}
    return {
        "file_hash": getattr(xet_file_data, "file_hash", ""),
        "refresh_route": getattr(xet_file_data, "refresh_route", ""),
    }


def _deserialize_xet_file_data(data: Dict[str, Any]) -> Any:
    """Reconstruct an XetFileData-like object from a dict.

    Returns a simple namespace object that has ``file_hash`` and
    ``refresh_route`` attributes, i.e. what
    ``session.new_file_download_group(token_refresh_url=...)`` consumes.
    """
    if not data:
        return None

    class _XetFileDataProxy:
        __slots__ = ("file_hash", "refresh_route")

        def __init__(self, file_hash: str, refresh_route: str):
            self.file_hash = file_hash
            self.refresh_route = refresh_route

    return _XetFileDataProxy(
        file_hash=data.get("file_hash", ""),
        refresh_route=data.get("refresh_route", ""),
    )



# ── Internal Callback (runs in child process) ────────────────────

def _make_progress_callback(
    filename: str,
    total_bytes: int,
    transfer_id: str,
    mp_queue: mp.Queue,
    cancel_event: mp.Event,
    direction: str = "download",
    file_index: int = 0,
    total_files: int = 1,
    report_interval: float = 0.1,
) -> Any:
    """Create a progress callback suitable for ``hf_xet`` detailed mode.

    The returned callable has the signature ``callback(total_update, item_updates)``
    which matches what the Rust runtime detects via ``inspect.signature()``.

    Events are throttled by ``report_interval`` (seconds) and a minimum
    byte delta (1% of total or 1 KiB, whichever is larger) to prevent
    flooding the IPC queue and deadlocking the Rust runtime's background
    feeder thread.
    """
    _start_time = time.time()
    _last_emit_time: float = 0.0
    _last_emit_bytes: int = 0
    _first_event_emitted: bool = False

    def _should_throttle(display_completed: int, total: int, now: float) -> bool:
        """Return True if this event should be dropped to reduce IPC flooding."""
        nonlocal _first_event_emitted, _last_emit_time, _last_emit_bytes
        if not _first_event_emitted:
            return False
        if total > 0 and display_completed >= total:
            return False
        time_delta = now - _last_emit_time
        if time_delta >= report_interval:
            return False
        bytes_delta = abs(display_completed - _last_emit_bytes)
        if total > 0:
            min_delta = max(total // 100, 1024)
        else:
            min_delta = 1024
        return bytes_delta < min_delta

    def progress_updater(total_update, item_updates):
        nonlocal _last_emit_time, _last_emit_bytes, _first_event_emitted

        # Check cancellation
        if cancel_event.is_set():
            from .types import TransferCancelledError
            raise TransferCancelledError("Transfer cancelled by user")

        # Extract values from Rust PyTotalProgressUpdate
        bytes_completed = getattr(total_update, "total_bytes_completed", 0)
        total = getattr(total_update, "total_bytes", 0) or total_bytes
        speed = getattr(total_update, "total_bytes_completion_rate", 0) or 0
        transfer_completed = getattr(total_update, "total_transfer_bytes_completed", 0)
        transfer_total = getattr(total_update, "total_transfer_bytes", 0)
        transfer_speed = getattr(total_update, "total_transfer_bytes_completion_rate", 0) or 0

        # Per-file progress for multi-file transfers
        if total_files > 1 and item_updates:
            item_update = next(
                (item for item in item_updates if getattr(item, "item_name", "") == filename),
                None,
            )
            if item_update is not None:
                bytes_completed = getattr(item_update, "bytes_completed", 0)
                total = getattr(item_update, "total_bytes", 0) or total_bytes

        # Choose display bytes: prefer transfer_completed for smooth progress
        display_completed = bytes_completed
        if direction == "download" and transfer_completed > 0:
            if bytes_completed < total:
                display_completed = transfer_completed
            elif bytes_completed >= total > 0:
                display_completed = bytes_completed

        active_speed = transfer_speed or speed

        # NTH-001: Progress estimation when bytes = 0 but speed > 0
        if display_completed == 0 and active_speed > 0:
            elapsed = time.time() - _start_time
            estimated = int(active_speed * elapsed)
            if total > 0:
                estimated = min(estimated, int(total * 0.99))
            display_completed = estimated

        # Throttle: skip IPC if too soon and too little change
        now = time.time()
        if _should_throttle(display_completed, total, now):
            return

        percentage = ((display_completed / total * 100) if total > 0 else 0)

        event_dict = {
            "event_type": EventType.PROGRESS.value,
            "transfer_id": transfer_id,
            "direction": direction,
            "filename": filename,
            "phase": ProgressPhase.DOWNLOADING.value if direction == "download" else ProgressPhase.UPLOADING.value,
            "bytes_completed": display_completed,
            "total_bytes": total,
            "percentage": percentage,
            "speed": active_speed,
            "file_index": file_index,
            "total_files": total_files,
            "transfer_bytes_completed": transfer_completed if total_files == 1 else 0,
            "transfer_bytes_total": transfer_total if total_files == 1 else 0,
            "transfer_speed": transfer_speed if total_files == 1 else 0,
        }
        try:
            mp_queue.put_nowait(SubprocessMessage.event(event_dict))
        except BaseException:
            pass # Drop event if queue is full or closed

        _last_emit_time = now
        _last_emit_bytes = display_completed
        _first_event_emitted = True

    return progress_updater
