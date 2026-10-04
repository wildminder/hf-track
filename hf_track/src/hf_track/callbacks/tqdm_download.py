"""``DownloadProgressTqdm`` — a tqdm subclass that emits progress events.

Why a single class (not split further)
======================================

This class is intentionally **kept whole** even though it spans several
hundred lines. The throttling logic (``_should_throttle``), the xet
absolute-position correction (in :meth:`update`), the per-file fallback
emission, and the COMPLETE-event synthesis in :meth:`close` all read
and write the same small set of instance attributes (``self.n``,
``self._last_emit_time``, ``self._last_emit_bytes``,
``self._first_event_emitted``). Splitting them across files would
require passing those values around explicitly and would obscure the
interplay between them.

The class is in its own module to:

1. Make its public surface (``DownloadProgressTqdm``) the only thing
   the rest of the package needs to import.
2. Keep the diagnostic instrumentation co-located with the code it
   instruments. Every record goes through the module-level :func:`_diag`
   helper, which is gated on a single env var
   (``DIAG_ENV_VAR``) — so removing the instrumentation later is one
   deletion instead of an edit inside every instrumented method.
3. Separate it from the unrelated Xet callback family and the upload
   patcher.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Callable, Optional

from tqdm.auto import tqdm as base_tqdm

from ..types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferCancelledError,
    TransferDirection,
)
from .state import state_manager
from .xet_callback import _dummy_file

logger = logging.getLogger(__name__)

#: Name of the environment variable that turns on the temporary
#: instrumentation emitted by :func:`_diag`.
DIAG_ENV_VAR = "HF_TRACK_DEBUG_XET"


def _diag(event: str, **fields: object) -> None:
    """Emit one temporary instrumentation record, gated on the env var.

    Every diagnostic in this module routes through this helper so that
    enabling, disabling, or deleting the instrumentation is a single edit
    (:func:`_diag_enabled` plus this body) rather than an edit inside
    every instrumented method.

    The env var is read on each call rather than captured at import, so
    the gate also works for interpreters that set it after this module is
    loaded (the test suite, and the in-process worker entry points).

    Diagnostics must never break a download: the formatting is wrapped so
    a failure to log is swallowed, and nothing is raised.
    """
    if not _diag_enabled():
        return
    try:
        rendered = " ".join(f"{k}={v}" for k, v in fields.items())
        logger.debug("[DIAG-%s] pid=%d tid=%s %s", event, os.getpid(),
                     threading.get_ident(), rendered)
    except Exception as exc:  # pragma: no cover - diagnostics are best-effort
        logger.debug("[DIAG-%s] failed to log: %s", event, exc)


def _diag_enabled() -> bool:
    """Return True when the instrumentation env var is set to a non-empty value."""
    return bool(os.environ.get(DIAG_ENV_VAR))


class DownloadProgressTqdm(base_tqdm):
    """A ``tqdm`` subclass that emits ``ProgressEvent`` rows.

    Snaps the ``huggingface_hub`` tqdm progress bars (both the byte
    bars emitted by ``snapshot_download``/``hf_hub_download`` and the
    file-count bars emitted for multi-file transfers) and turns them
    into a single stream of ``ProgressEvent`` instances delivered to
    ``self._event_queue``.

    The class is bindable (:meth:`bind`) so callers can pre-configure a
    subclass with the queue/transfer-id/filename and have it silently
    sink its tqdm output.
    """

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
            state_manager.update_download_files(
                self._transfer_id, 0, getattr(self, "total", 0) or 0,
            )

        # ── DIAG: temporary instrumentation (Phase 0.1) ──────────────
        _diag(
            "INIT",
            id=id(self),
            total=getattr(self, "total", None),
            unit=getattr(self, "unit", None),
            desc=getattr(self, "desc", None),
            is_bytes_bar=self.is_bytes_bar,
            event_queue_is_none=self._event_queue is None,
            has_cancel_hook=self._is_cancelled is not None,
        )

    def _should_throttle(
        self,
        agg_bytes_completed: int,
        agg_total_bytes: int,
        now: float,
    ) -> bool:
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
        # ── DIAG: update entry ───────────────────────────────────────
        _diag(
            "UPDATE-ENTRY",
            id=id(self),
            n=n,
            prev_n=self._prev_n,
            total=getattr(self, "total", 0) or 0,
            is_bytes_bar=getattr(self, "is_bytes_bar", False),
            is_cancelled_hook=bool(self._is_cancelled and self._is_cancelled()),
        )

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
        _diag(
            "UPDATE-POST-SUPER",
            n=n,
            self_n=getattr(self, "n", 0),
            total=getattr(self, "total", 0),
        )

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
                _diag(
                    "ABS-CORRECTION",
                    n=n,
                    prev_n=prev_n,
                    old_self_n=current_n,
                    new_self_n=n,
                    total=total_val,
                    remaining=remaining,
                )
                # Reset to the absolute position
                setattr(self, "n", n)
                current_n = n

        self._prev_n = getattr(self, "n", 0)

        if not getattr(self, "is_bytes_bar", False):
            state_manager.update_download_files(
                self._transfer_id,
                getattr(self, "n", 0),
                getattr(self, "total", 0) or 0,
            )
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
        state_manager.update_download_bytes(
            self._transfer_id, bar_bytes_completed, bar_total_bytes,
        )

        # Read aggregate values from state_manager for the event
        state = state_manager.get_state(self._transfer_id)
        agg_bytes_completed = state.get("bytes_completed", 0)
        agg_total_bytes = state.get("total_bytes", 0)
        files_completed = state.get("files_completed", 0)
        total_files = state.get("total_files", 0)
        percentage = (
            (agg_bytes_completed / agg_total_bytes * 100) if agg_total_bytes > 0 else 0
        )

        # Throttle: skip emitting if too soon and too little change
        if self._should_throttle(agg_bytes_completed, agg_total_bytes, now):
            # ── DIAG: throttled ──────────────────────────────────────
            _diag(
                "THROTTLED",
                agg_bytes=agg_bytes_completed,
                agg_total=agg_total_bytes,
                first_emitted=self._first_event_emitted,
            )
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
        _diag(
            "EMIT",
            event_type=EventType.PROGRESS.name,
            bytes=agg_bytes_completed,
            total=agg_total_bytes,
            pct=f"{percentage:.2f}",
        )
        return result

    def _emit_event(self, event: ProgressEvent) -> None:
        """Emit a progress event to the event queue.

        Override this method in subclasses to route events through
        alternative channels (e.g. ``mp.Queue`` in subprocess workers).
        The default implementation puts events into ``self._event_queue``.
        If ``self._event_queue`` is ``None``, the event is silently dropped
        (useful when a subclass overrides this method to route elsewhere).
        """
        # ── DIAG: emission into the queue ──────────────────────────
        dict_size = -1
        try:
            dict_size = len(event.to_dict())
        except Exception as exc:
            _diag("EMIT-EVENT-DICT-FAIL", error=f"{type(exc).__name__}: {exc}")
        _diag(
            "EMIT-EVENT",
            event_type=event.event_type.value,
            bytes=event.bytes_completed,
            total=event.total_bytes,
            dict_size=dict_size,
            queue_is_none=self._event_queue is None,
        )

        if self._event_queue is None:
            return

        try:
            self._event_queue.put_nowait(event)
            _diag("EMIT-PUT-OK", event_type=event.event_type.value)
        except queue.Full:
            _diag("EMIT-PUT-FULL", event_type=event.event_type.value)
            logger.warning(
                "Download progress event queue full — dropping %s event",
                event.event_type.value,
            )
        except BaseException as put_err:
            # Not a full queue — something else is wrong with the queue
            # object itself. Log it, then let it propagate: swallowing it
            # here would turn a broken queue into a silently stalled bar.
            _diag(
                "EMIT-PUT-ERR",
                err_type=type(put_err).__name__,
                err=put_err,
            )
            raise

    def close(self):
        # ── DIAG: close entry ──────────────────────────────────────
        _diag(
            "CLOSE-ENTRY",
            id=id(self),
            n=getattr(self, "n", 0),
            total=getattr(self, "total", 0),
            is_bytes_bar=getattr(self, "is_bytes_bar", False),
            event_queue_is_none=self._event_queue is None,
            closed=self._closed,
        )

        if self._closed:
            super().close()
            return
        self._closed = True

        if (
            self._event_queue is not None
            and getattr(self, "total", None) is not None
            and getattr(self, "is_bytes_bar", False)
        ):
            n_val = getattr(self, "n", 0)
            total_val = getattr(self, "total", 0)

            is_complete = n_val >= total_val
            is_xet_cached = n_val == 0 and total_val > 0

            # ── DIAG: branch decision ────────────────────────────────
            _diag(
                "CLOSE-BRANCH",
                is_complete=is_complete,
                is_xet_cached=is_xet_cached,
                n_val=n_val,
                total_val=total_val,
            )

            if is_complete or is_xet_cached:
                final_bytes = total_val if is_xet_cached else n_val
                state = state_manager.get_state(self._transfer_id)
                total_files = state.get("total_files", 0)
                files_completed = (
                    total_files if total_files > 0 else state.get("files_completed", 0)
                )

                # ── DIAG: synthesized COMPLETE ──────────────────────────
                _diag(
                    "CLOSE-SYNTH-COMPLETE",
                    final_bytes=final_bytes,
                    total_bytes=total_val,
                    files_completed=files_completed,
                    total_files=total_files,
                )

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
        """Create a pre-configured subclass with the given context.

        The returned class has ``_dummy_file`` injected as its
        ``file=`` argument so tqdm output is silently dropped (the
        progress is reported via ``ProgressEvent`` rows instead).
        """
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
