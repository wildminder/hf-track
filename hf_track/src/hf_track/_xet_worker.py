"""Worker functions for Xet subprocess isolation.

These functions run inside child processes spawned by ``XetSubprocessRunner``.
They import ``hf_xet`` locally (never at module level) so the Rust .pyd
extension is loaded only in the child process — safe to terminate without
affecting the main process.

Each worker:
1. Receives a plain ``dict`` of parameters (picklable across process boundary)
2. Receives a ``multiprocessing.Queue`` for sending ``SubprocessMessage`` back
3. Receives a ``multiprocessing.Event`` for cancellation signaling
4. Puts progress event messages during the transfer
5. Puts a terminal message (result/error/cancelled) when done

.. warning::

    These functions MUST be defined at module top-level (not nested)
    so they are picklable by ``multiprocessing`` with the ``spawn`` context.
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
from .types import ProgressEvent

logger = logging.getLogger(__name__)


# ── Initialization & Safe IO ─────────────────────────────────────

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


def _safe_put(mp_queue: mp.Queue, message: SubprocessMessage) -> None:
    """Put a message to the multiprocessing queue, suppressing all errors.

    Catches ``BaseException`` (including ``KeyboardInterrupt``) so that
    a second interrupt during error/cancel handling never produces a
    traceback from the subprocess.
    """
    try:
        mp_queue.put(message)
    except BaseException:
        pass


def _handle_worker_exception(mp_queue: mp.Queue, e: BaseException) -> None:
    """Safely format and send an exception as a terminal message."""
    try:
        from .types import TransferCancelledError
        if isinstance(e, KeyboardInterrupt):
            _safe_put(mp_queue, SubprocessMessage.cancelled(
                message="Transfer interrupted by user (Ctrl+C)",
            ))
        elif isinstance(e, TransferCancelledError):
            _safe_put(mp_queue, SubprocessMessage.cancelled(
                message="Transfer cancelled by user",
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
    ``refresh_route`` attributes, compatible with ``refresh_xet_connection_info``.
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
    from .types import EventType, ProgressPhase, TransferDirection

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


# ── Download Worker ───────────────────────────────────────────────

def _download_worker(params: Dict[str, Any], mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Worker function for single-file Xet downloads.

    Runs in a child process. Imports ``hf_xet`` locally.

    .. deprecated:: 2026-07-09
        This worker uses the legacy ``hf_xet.download_files()`` API
        which hangs indefinitely in some environments. The hybrid
        approach (``download_file_with_xet_hybrid`` /
        ``HybridRunner`` + ``download_hybrid``) replaces it for
        single-file downloads. See
        docs/plans/2026-07-09-xet-single-file-download-fix.md.

    Args:
        params: Dict with keys: file_hash, file_size, dest_path,
            xet_file_data (dict), token, endpoint, transfer_id,
            report_interval, request_headers.
        mp_queue: Queue for sending SubprocessMessage back to main process.
        cancel_event: Event set by main process to signal cancellation.
    """
    _init_worker()
    from .types import EventType, ProgressPhase, TransferCancelledError, TransferDirection

    transfer_id = params["transfer_id"]
    filename = os.path.basename(params["dest_path"])
    file_size = params["file_size"]

    try:
        import hf_xet
        from huggingface_hub.utils._xet import refresh_xet_connection_info
    except ImportError as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=str(e), error_type="ImportError", retryable=False,
        ))
        return

    # Reconstruct XetFileData from serialized dict
    xet_file_data = _deserialize_xet_file_data(params.get("xet_file_data", {}))

    # Get credentials
    try:
        headers = params.get("request_headers", {})
        connection_info = refresh_xet_connection_info(file_data=xet_file_data, headers=headers)

        def token_refresher():
            ci = refresh_xet_connection_info(file_data=xet_file_data, headers=headers)
            return ci.access_token, ci.expiration_unix_epoch
    except Exception as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=f"Failed to get download credentials: {e}",
            error_type=type(e).__name__,
        ))
        return

    # Build download info
    download_info = [
        hf_xet.PyXetDownloadInfo(
            destination_path=str(os.path.abspath(params["dest_path"])),
            hash=params["file_hash"],
            file_size=file_size,
        )
    ]

    # Build progress callback
    callback = _make_progress_callback(
        filename=filename,
        total_bytes=file_size,
        transfer_id=transfer_id,
        mp_queue=mp_queue,
        cancel_event=cancel_event,
        direction="download",
        report_interval=params.get("report_interval", 0.1),
    )

    try:
        kwargs: Dict[str, Any] = dict(
            endpoint=connection_info.endpoint,
            token_info=(connection_info.access_token, connection_info.expiration_unix_epoch),
            token_refresher=token_refresher,
            progress_updater=[callback],
        )
        if params.get("request_headers"):
            kwargs["request_headers"] = params["request_headers"]

        hf_xet.download_files(download_info, **kwargs)

        _safe_put(mp_queue, SubprocessMessage.result(
            filename=filename,
            destination_path=params["dest_path"],
            file_size=file_size,
            transfer_id=transfer_id,
        ))

    except (KeyboardInterrupt, Exception) as e:
        _handle_worker_exception(mp_queue, e)


def _download_batch_worker(params: Dict[str, Any], mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Worker function for multi-file Xet downloads.

    Args:
        params: Dict with keys: file_specs (list of dicts), token,
            endpoint, transfer_id, report_interval, request_headers.
        mp_queue: Queue for sending SubprocessMessage back to main process.
        cancel_event: Event set by main process to signal cancellation.
    """
    _init_worker()
    from .types import EventType, ProgressPhase, TransferCancelledError, TransferDirection

    transfer_id = params["transfer_id"]
    file_specs: List[Dict[str, Any]] = params["file_specs"]
    total_files = len(file_specs)

    try:
        import hf_xet
        from huggingface_hub.utils._xet import refresh_xet_connection_info
    except ImportError as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=str(e), error_type="ImportError", retryable=False,
        ))
        return

    # Use first file's xet_file_data for credentials
    xet_file_data = _deserialize_xet_file_data(file_specs[0].get("xet_file_data", {}))

    try:
        headers = params.get("request_headers", {})
        connection_info = refresh_xet_connection_info(file_data=xet_file_data, headers=headers)

        def token_refresher():
            ci = refresh_xet_connection_info(file_data=xet_file_data, headers=headers)
            return ci.access_token, ci.expiration_unix_epoch
    except Exception as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=f"Failed to get download credentials: {e}",
            error_type=type(e).__name__,
        ))
        return

    # Build download infos and callbacks
    download_infos = []
    callbacks = []
    for i, spec in enumerate(file_specs):
        filename = os.path.basename(spec["dest_path"])
        download_infos.append(
            hf_xet.PyXetDownloadInfo(
                destination_path=str(os.path.abspath(spec["dest_path"])),
                hash=spec["hash"],
                file_size=spec["file_size"],
            )
        )
        callbacks.append(
            _make_progress_callback(
                filename=filename,
                total_bytes=spec["file_size"],
                transfer_id=transfer_id,
                mp_queue=mp_queue,
                cancel_event=cancel_event,
                direction="download",
                file_index=i,
                total_files=total_files,
                report_interval=params.get("report_interval", 0.1),
            )
        )

    try:
        kwargs: Dict[str, Any] = dict(
            endpoint=connection_info.endpoint,
            token_info=(connection_info.access_token, connection_info.expiration_unix_epoch),
            token_refresher=token_refresher,
            progress_updater=callbacks,
        )
        if params.get("request_headers"):
            kwargs["request_headers"] = params["request_headers"]

        hf_xet.download_files(download_infos, **kwargs)

        # Send individual result messages for each file
        for i, spec in enumerate(file_specs):
            filename = os.path.basename(spec["dest_path"])
            _safe_put(mp_queue, SubprocessMessage.result(
                filename=filename,
                destination_path=spec["dest_path"],
                file_size=spec["file_size"],
                transfer_id=transfer_id,
                file_index=i,
                total_files=total_files,
            ))

    except (KeyboardInterrupt, Exception) as e:
        _handle_worker_exception(mp_queue, e)




# ── Shared fsync utilities ────────────────────────────────────────
#
# Used by the hybrid file-download-group worker below. The previous
# streaming worker (broken at the Rust __next__ layer) also relied on
# these; they are kept here as shared helpers.

DEFAULT_FSYNC_INTERVAL: int = 4 * 1024 * 1024


def _open_unbuffered(path: str) -> int:
    """Open ``path`` for writing and return a raw file descriptor.

    Unlike ``open(path, "wb")`` this skips Python's userspace buffer
    (8 KiB by default) so each ``os.write()`` goes straight to the OS
    page cache. Combined with periodic ``os.fsync()`` this guarantees
    that a process killed mid-download leaves a non-empty file on disk.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)


# ── File-Download-Group Worker (HYBRID, plan 2026-06-15) ───────────

DEFAULT_TIER_TIMEOUT_S: float = 60.0
DEFAULT_POLL_INTERVAL_S: float = 0.1
DEFAULT_TIER_PROGRESS_INTERVAL_S: float = 0.1


def download_hybrid(
    file_specs: List[Dict[str, Any]],
    *,
    repo_id: str = "",
    repo_type: str = "model",
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    token: Any = None,
    transfer_id: str = "hybrid",
    report_interval: float = 0.1,
    fsync_interval: int = DEFAULT_FSYNC_INTERVAL,
    disable_fsync: bool = False,
    tier_timeout_s: float = DEFAULT_TIER_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    enable_http_fallback: bool = True,
    use_xet: bool = True,
    cancel_event: Optional[Any] = None,
    progress_queue: Any = None,
    progress_dict: Any = None,
) -> Dict[str, Any]:
    """Run the HYBRID download IN-PROCESS from the parent thread.

    Plan: ``docs/plans/2026-06-15-xet-streaming-hybrid-approach.md``
    (re-designed to run without a subprocess -- the legacy ``spawn``
    worker introduced GIL/CAS issues that were not worth carrying).

    Per-file loop:

    1. ``XetFileDownloadGroup.start_download_file()`` is called via
       ``hf_xet.XetSession()``. We poll ``handle.progress()`` and emit
       progress into ``progress_queue`` (a ``queue.Queue`` of plain
       dict events -- see ``hf_track.types.ProgressEvent.to_dict()``).
    2. If progress reaches ``file_size`` or the status becomes
       ``Completed`` before ``tier_timeout_s`` elapses, Tier 1 wins and
       we move on to the next file.
    3. Otherwise ``group.abort()`` is called and the file is
       downloaded via ``download_file_http`` -- ``requests`` →
       ``iter_content`` → ``os.write`` + periodic ``os.fsync``. This
       path gives full incremental disk visibility independent of the
       Rust runtime.
    4. If both tiers fail the worker raises; the caller turns that
       into an arbitrary ``progress_queue.put({"event_type": "error",
       ...})`` event.

    Progress reporting goes through ``progress_queue`` if provided,
    otherwise it is dropped. ``progress_dict`` is an optional mutable
    container (e.g. ``[int]`` or ``dict``) the caller can inspect
    mid-call for live state (last error, last completed index).

    Args:
        file_specs: List of dicts each with ``hash``, ``file_size``,
            ``dest_path``, optional ``filename`` and ``xet_file_data``
            (serialized ``XetFileData``).
        repo_id, repo_type, revision, endpoint, token: needed for the
            HTTP fallback and for refreshing xet credentials.
        tier_timeout_s: How long (per file) Tier 1 may stall before
            falling back to HTTP.
        poll_interval_s: Loop interval between ``handle.progress()``
            polls.
        report_interval: Seconds between emitted PROGRESS events.
        fsync_interval: Bytes between ``os.fsync`` calls in HTTP path.
        disable_fsync: If True, skip ``os.fsync`` (fastest).
        enable_http_fallback: When False, Tier 1 failure is fatal.
        use_xet: When False, all files go through Tier 3 (HTTP).
        cancel_event: Any object with ``.is_set()`` / ``.set()``. Used
            both between chunks and between polls.
        progress_queue: ``queue.Queue`` to receive ProgressEvent dicts.
        progress_dict: Mutable container inspected by tests.

    Returns:
        A ``dict`` summary: ``{"bytes_completed": int,
        "total_bytes": int, "files_completed": int, "total_files":
        int, "errors": [str] }``.
    """
    import sys as _sys

    from .types import (
        EventType,
        ProgressPhase,
        TransferCancelledError,
        TransferDirection,
    )
    from .download.http_fallback import download_file_http

    def _diag(msg: str) -> None:
        try:
            _sys.stderr.write(f"[HYBRID] {msg}\n")
            _sys.stderr.flush()
        except Exception:
            pass

    def _put_progress(event: Dict[str, Any]) -> None:
        if progress_queue is None:
            return
        try:
            progress_queue.put_nowait(event)
        except Exception:
            pass

    def _is_cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    if not file_specs:
        return {
            "bytes_completed": 0,
            "total_bytes": 0,
            "files_completed": 0,
            "total_files": 0,
            "errors": [],
        }

    total_bytes_all = sum(int(spec["file_size"]) for spec in file_specs)
    bytes_completed_all = 0
    errors: List[str] = []
    files_completed = 0

    # ── Set up Tier 1 (lazily) ─────────────────────────────────────
    session = None
    refresh_xet_connection_info = None
    hf_xet_mod = None
    if use_xet:
        try:
            import hf_xet as _hf_xet
            from huggingface_hub.utils._xet import (
                refresh_xet_connection_info as _refresh,
            )
        except ImportError as e:
            _diag(f"hf_xet-import-failed: {e}")
            errors.append(f"hf_xet unavailable: {e}")
            session = None
            refresh_xet_connection_info = None
            hf_xet_mod = None
        else:
            session = _hf_xet.XetSession()
            refresh_xet_connection_info = _refresh
            hf_xet_mod = _hf_xet
    else:
        _diag("use_xet=False → Tier 1 disabled, all files via HTTP")

    headers: Dict[str, Any] = {}
    start_time = time.time()

    for i, spec in enumerate(file_specs):
        if _is_cancelled():
            _put_progress({
                "event_type": EventType.CANCELLED.value,
                "transfer_id": transfer_id,
                "direction": TransferDirection.DOWNLOAD.value,
                "filename": spec.get("filename") or os.path.basename(spec["dest_path"]),
                "phase": ProgressPhase.CANCELLED.value,
                "bytes_completed": bytes_completed_all,
                "total_bytes": total_bytes_all,
            })
            if progress_dict is not None:
                progress_dict["cancelled"] = True
            return {
                "bytes_completed": bytes_completed_all,
                "total_bytes": total_bytes_all,
                "files_completed": files_completed,
                "total_files": len(file_specs),
                "errors": errors,
            }

        filename = spec.get("filename") or os.path.basename(spec["dest_path"])
        file_hash = spec["hash"]
        file_size = int(spec["file_size"])
        dest_path = spec["dest_path"]

        # ── Tier 1 ────────────────────────────────────────────────
        tier1_ok = False
        if use_xet and session is not None and file_size > 0:
            tier1_ok = _run_tier1_file(
                session=session,
                refresh_xet_connection_info=refresh_xet_connection_info,
                hf_xet=hf_xet_mod,
                filename=filename,
                file_hash=file_hash,
                file_size=file_size,
                dest_path=dest_path,
                xet_file_data=spec.get("xet_file_data", {}),
                headers=headers,
                transfer_id=transfer_id,
                file_index=i,
                total_files=len(file_specs),
                progress_queue=progress_queue,
                cancel_event=cancel_event,
                tier_timeout_s=tier_timeout_s,
                poll_interval_s=poll_interval_s,
                total_bytes_all=total_bytes_all,
                bytes_completed_baseline=bytes_completed_all,
                start_time=start_time,
                diag=_diag,
                report_interval=report_interval,
            )

        if tier1_ok:
            bytes_completed_all += file_size
        elif file_size == 0:
            # Empty file handled in Tier 1, no fallback needed.
            parent = os.path.dirname(dest_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            open(dest_path, "wb").close()
            bytes_completed_all += 0
        elif enable_http_fallback:
            # ── Tier 3 (HTTP) ────────────────────────────────────
            _diag(f"file-{i} tier1-failed, falling back to http")
            if os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except OSError:
                    pass
            try:
                written = download_file_http(
                    repo_id=repo_id,
                    filename=filename,
                    dest_path=dest_path,
                    token=token,
                    repo_type=repo_type,
                    revision=revision or "main",
                    endpoint=endpoint,
                    fsync_interval=fsync_interval,
                    cancel_event=cancel_event,
                    expected_size=file_size,
                )
            except Exception as fb_err:
                err_msg = f"Tier 3 (HTTP) failed for {filename}: {fb_err}"
                _diag(f"file-{i} tier3-failed: {fb_err}")
                errors.append(err_msg)
                _put_progress({
                    "event_type": EventType.ERROR.value,
                    "transfer_id": transfer_id,
                    "direction": TransferDirection.DOWNLOAD.value,
                    "filename": filename,
                    "phase": ProgressPhase.ERROR.value,
                    "bytes_completed": bytes_completed_all,
                    "total_bytes": total_bytes_all,
                    "error": {"message": err_msg, "error_type": "AllTiersFailed"},
                })
                if progress_dict is not None:
                    progress_dict["error"] = err_msg
                break
            if written != file_size:
                err_msg = f"http fallback wrote {written}/{file_size} bytes"
                errors.append(err_msg)
                _put_progress({
                    "event_type": EventType.ERROR.value,
                    "transfer_id": transfer_id,
                    "direction": TransferDirection.DOWNLOAD.value,
                    "filename": filename,
                    "phase": ProgressPhase.ERROR.value,
                    "bytes_completed": bytes_completed_all,
                    "total_bytes": total_bytes_all,
                    "error": {"message": err_msg, "error_type": "SizeMismatch"},
                })
                if progress_dict is not None:
                    progress_dict["error"] = err_msg
                break
            bytes_completed_all += written
        else:
            err_msg = (
                f"Tier 1 (Xet) failed for {filename} but HTTP "
                f"fallback is disabled."
            )
            errors.append(err_msg)
            _put_progress({
                "event_type": EventType.ERROR.value,
                "transfer_id": transfer_id,
                "direction": TransferDirection.DOWNLOAD.value,
                "filename": filename,
                "phase": ProgressPhase.ERROR.value,
                "bytes_completed": bytes_completed_all,
                "total_bytes": total_bytes_all,
                "error": {"message": err_msg, "error_type": "XetTierFailed"},
            })
            if progress_dict is not None:
                progress_dict["error"] = err_msg
            break

        files_completed += 1
        now = time.time()
        elapsed = now - start_time
        transfer_speed = (bytes_completed_all / elapsed) if elapsed > 0 else 0.0
        _put_progress({
            "event_type": EventType.COMPLETE.value,
            "transfer_id": transfer_id,
            "direction": TransferDirection.DOWNLOAD.value,
            "filename": filename,
            "phase": ProgressPhase.COMPLETE.value,
            "bytes_completed": file_size,
            "total_bytes": file_size,
            "percentage": 100.0,
            "speed": 0,
            "file_index": i,
            "total_files": len(file_specs),
            "transfer_bytes_completed": bytes_completed_all,
            "transfer_bytes_total": total_bytes_all,
            "transfer_speed": transfer_speed,
        })

    summary = {
        "bytes_completed": bytes_completed_all,
        "total_bytes": total_bytes_all,
        "files_completed": files_completed,
        "total_files": len(file_specs),
        "errors": errors,
    }
    if progress_dict is not None:
        progress_dict.update(summary)
    return summary


def _run_tier1_file(
    *,
    session: Any,
    refresh_xet_connection_info: Any,
    hf_xet: Any,
    filename: str,
    file_hash: str,
    file_size: int,
    dest_path: str,
    xet_file_data: Dict[str, Any],
    headers: Dict[str, Any],
    transfer_id: str,
    file_index: int,
    total_files: int,
    progress_queue: Any,
    cancel_event: Any,
    tier_timeout_s: float,
    poll_interval_s: float,
    total_bytes_all: int,
    bytes_completed_baseline: int,
    start_time: float,
    diag: Any,
    report_interval: float,
) -> bool:
    """Tier 1 driver: ``XetFileDownloadGroup.start_download_file()`` for one file.

    Returns True on completion, False on timeout/error. Polls
    ``handle.progress()`` every ``poll_interval_s`` and emits
    throttled PROGRESS events into ``progress_queue``. Tier 1
    completes when ``handle.try_result()`` returns, ``status()``
    reports ``Completed``, or the bytes-completed reaches the
    expected file size.
    """
    from .types import EventType, ProgressPhase, TransferCancelledError, TransferDirection

    file_info = hf_xet.XetFileInfo(file_hash, file_size)
    try:
        file_data = _deserialize_xet_file_data(xet_file_data) if xet_file_data else None
        conn_info = refresh_xet_connection_info(file_data=file_data, headers=headers)
        endpoint = conn_info.endpoint
        access_token = conn_info.access_token
        token_expiry = getattr(conn_info, "expiration_unix_epoch", 0)
        refresh_route = getattr(file_data, "refresh_route", "") if file_data else ""
        group = session.new_file_download_group(
            endpoint=endpoint,
            token=access_token,
            token_expiry_unix_secs=token_expiry,
            token_refresh_url=refresh_route,
            token_refresh_headers=headers,
        )
    except Exception as e:
        diag(f"file-{file_index} tier1-setup-failed: {e}")
        return False

    if file_size == 0:
        parent = os.path.dirname(dest_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fd = _open_unbuffered(dest_path)
        try:
            os.fsync(fd)
        except OSError:
            pass
        os.close(fd)
        return True

    try:
        handle = group.start_download_file(file_info, os.path.abspath(dest_path))
    except Exception as e:
        diag(f"file-{file_index} tier1-start-failed: {e}")
        try:
            group.abort()
        except Exception:
            pass
        return False

    deadline = time.time() + tier_timeout_s
    last_progress_bytes = 0
    last_emit_time = 0.0
    last_emit_bytes = 0
    first_event = True

    def _emit_prog(prog_bytes: int, total_bytes: int, now: float) -> None:
        nonlocal last_emit_time, last_emit_bytes, first_event
        transfer_completed = bytes_completed_baseline + prog_bytes
        elapsed = now - start_time
        transfer_speed = (transfer_completed / elapsed) if elapsed > 0 else 0.0
        if progress_queue is None:
            return
        try:
            progress_queue.put_nowait({
                "event_type": EventType.PROGRESS.value,
                "transfer_id": transfer_id,
                "direction": TransferDirection.DOWNLOAD.value,
                "filename": filename,
                "phase": ProgressPhase.DOWNLOADING.value,
                "bytes_completed": prog_bytes,
                "total_bytes": total_bytes,
                "percentage": (prog_bytes / total_bytes * 100.0) if total_bytes else 0.0,
                "speed": transfer_speed,
                "file_index": file_index,
                "total_files": total_files,
                "transfer_bytes_completed": transfer_completed,
                "transfer_bytes_total": total_bytes_all,
                "transfer_speed": transfer_speed,
            })
        except Exception:
            pass
        last_emit_time = now
        last_emit_bytes = prog_bytes
        first_event = False

    try:
        while True:
            now = time.time()

            if cancel_event is not None and cancel_event.is_set():
                diag(f"file-{file_index} tier1-cancelled")
                return False

            prog_bytes = int(file_size)  # default to "done" so we exit if progress() fails
            total_bytes = file_size
            try:
                prog = handle.progress()
                if prog is not None:
                    prog_bytes = int(getattr(prog, "bytes_completed", 0) or 0)
                    total_bytes = int(getattr(prog, "total_bytes", 0) or file_size)
                    last_progress_bytes = prog_bytes
            except Exception:
                pass

            # Throttle PROGRESS emits: first event, completion, or report_interval.
            emit_now = first_event
            if not emit_now and prog_bytes >= total_bytes and total_bytes > 0:
                emit_now = True
            if not emit_now and (now - last_emit_time) >= report_interval:
                emit_now = True
            if emit_now:
                _emit_prog(prog_bytes, total_bytes, now)

            # Short-circuit when bytes_completed reaches full.
            if total_bytes and prog_bytes >= total_bytes:
                diag(f"file-{file_index} tier1-progress-complete ({prog_bytes}/{total_bytes})")
                return True

            # Polling checks.
            try:
                result = handle.try_result()
                if result is not None:
                    diag(f"file-{file_index} tier1-try-result-ok")
                    return True
            except Exception:
                pass

            try:
                status_str = str(handle.status())
                if "Completed" in status_str or status_str.lower() == "complete":
                    diag(f"file-{file_index} tier1-status-completed ({status_str})")
                    return True
                if "Failed" in status_str or "Error" in status_str:
                    diag(f"file-{file_index} tier1-status-failed ({status_str})")
                    return False
            except Exception:
                pass

            if now >= deadline:
                diag(
                    f"file-{file_index} tier1-timeout "
                    f"({tier_timeout_s}s, last_progress={last_progress_bytes}/{file_size})"
                )
                return False

            time.sleep(poll_interval_s)
    finally:
        try:
            group.abort()
        except Exception:
            pass


# Backward compatibility: an alias for any external caller that imported
# the original worker by its old name. Runs synchronously in the calling
# process (not via multiprocessing.spawn) and returns the result dict.
_xet_file_download_worker = download_hybrid


class TranslatingQueue:
    """Queue adapter that converts dict events into ``ProgressEvent`` objects.

    ``download_hybrid`` emits plain dicts (see ``ProgressEvent.to_dict()``)
    into its ``progress_queue``. The public API contract, however, promises
    ``ProgressEvent`` objects in ``HfTracker.event_queue`` — and the example
    ``ConsoleProgressDisplay.update()`` accesses ``.event_type`` etc. directly.

    This adapter wraps the user-supplied queue and transparently translates
    any dict it sees into a ``ProgressEvent`` (via ``ProgressEvent.from_dict``)
    before forwarding. Non-dict items (already-``ProgressEvent`` objects, or
    anything else) pass through unchanged.

    Only ``put`` / ``put_nowait`` are overridden; the consumer side
    (``get`` / ``get_nowait`` / ``empty``) is delegated so the driver loop in
    the caller keeps working unchanged.
    """

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    def _translate(self, item: Any) -> Any:
        if isinstance(item, dict) and "event_type" in item:
            try:
                return ProgressEvent.from_dict(item)
            except Exception:
                # If the dict is malformed, pass it through untouched so
                # the caller can decide (avoids swallowing real errors).
                return item
        return item

    def put(self, item: Any, *args, **kwargs) -> None:
        self._wrapped.put(self._translate(item), *args, **kwargs)

    def put_nowait(self, item: Any, *args, **kwargs) -> None:
        self._wrapped.put_nowait(self._translate(item), *args, **kwargs)

    def get(self, *args, **kwargs) -> Any:
        return self._wrapped.get(*args, **kwargs)

    def get_nowait(self, *args, **kwargs) -> Any:
        return self._wrapped.get_nowait(*args, **kwargs)

    def empty(self) -> bool:
        return self._wrapped.empty()

    def qsize(self) -> int:
        return self._wrapped.qsize()


class HybridRunner:
    """Simple IN-PROCESS runner for ``download_hybrid`` (plan 2026-06-15).

    Uses a daemon ``threading.Thread`` instead of a spawned subprocess.
    No GIL-watching, no ``multiprocessing`` pickling, and no relay
    thread: progress events go straight into the user-supplied
    ``event_queue`` (a ``queue.Queue``). Cancellation is via a
    ``threading.Event``.

    The runner exposes ``wait(timeout)`` like the legacy
    ``XetSubprocessRunner`` so the driver loop in
    ``download_snapshot_streaming`` does not need to be rewritten.
    """

    def __init__(self) -> None:
        import threading as _t
        self._thread: Optional["_t.Thread"] = None
        self._cancel_event = _t.Event()
        self._result: Optional[Dict[str, Any]] = None
        self._error: Optional[BaseException] = None
        self._done_event = _t.Event()

    def start(self, params: Dict[str, Any], event_queue: Any) -> None:
        import threading as _t
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("HybridRunner already started")

        # ``file_specs`` lives inside params; mirror the field into a
        # top-level arg for ``download_hybrid``.
        call_kwargs = dict(params)
        file_specs = call_kwargs.pop("file_specs", None)
        if file_specs is None:
            raise ValueError("HybridRunner.start requires params['file_specs']")

        # ``download_hybrid`` emits plain dict events into its
        # ``progress_queue``. The public API contract promises
        # ``ProgressEvent`` objects in the user's queue (and the example
        # display accesses ``.event_type`` etc. directly), so wrap the
        # queue with a translator that converts dicts → ProgressEvent.
        translating_queue = TranslatingQueue(event_queue)

        def _runner():
            try:
                summary = download_hybrid(
                    file_specs,
                    progress_queue=translating_queue,
                    cancel_event=self._cancel_event,
                    **call_kwargs,
                )
                self._result = summary
            except BaseException as exc:  # noqa: BLE001
                self._error = exc
            finally:
                self._done_event.set()

        self._thread = _t.Thread(target=_runner, daemon=True, name="hybrid-runner")
        self._thread.start()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def wait(self, timeout: Optional[float] = 1.0) -> Optional[Dict[str, Any]]:
        if timeout is None:
            self._done_event.wait()
        else:
            self._done_event.wait(timeout=timeout)
        if not self._done_event.is_set():
            return None
        if self._error is not None:
            return {
                "status": "error",
                "message": str(self._error),
                "error_type": type(self._error).__name__,
            }
        summary = self._result or {}
        if summary.get("errors"):
            return {
                "status": "error",
                "message": "; ".join(summary["errors"]),
                "error_type": "AllTiersFailed",
                "bytes_completed": summary.get("bytes_completed", 0),
                "total_bytes": summary.get("total_bytes", 0),
            }
        if summary.get("cancelled"):
            return {
                "status": "cancelled",
                "message": "Transfer cancelled by user",
                "bytes_completed": summary.get("bytes_completed", 0),
                "total_bytes": summary.get("total_bytes", 0),
            }
        return {
            "status": "success",
            "bytes_completed": summary.get("bytes_completed", 0),
            "total_bytes": summary.get("total_bytes", 0),
            "files_completed": summary.get("files_completed", 0),
            "total_files": summary.get("total_files", 0),
        }

    def terminate(self, grace: Optional[float] = None) -> None:  # noqa: ARG002
        """Best-effort termination: set the cancel flag and wait."""
        self.request_cancel()
        if self._thread is not None:
            self._thread.join(timeout=grace if grace is not None else 1.0)


# ── Snapshot Download Worker ─────────────────────────────────────

def _snapshot_worker(params: Dict[str, Any], mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Worker function for snapshot downloads via ``huggingface_hub.snapshot_download``.

    Runs in a child process. Calls ``snapshot_download()`` which internally
    handles xet routing — if ``hf_xet`` is available, it uses xet; otherwise
    it falls back to HTTP. Either way, the ``hf_xet`` extension is loaded
    only in this child process, safe to terminate.

    Progress events are emitted via a ``DownloadProgressTqdm`` subclass
    that routes events through ``mp_queue`` instead of ``queue.Queue``.

    Args:
        params: Dict with keys: repo_id, token, repo_type, revision,
            local_dir, allow_patterns, ignore_patterns, endpoint,
            transfer_id, report_interval, force_download, use_xet.
        mp_queue: Queue for sending SubprocessMessage back to main process.
        cancel_event: Event set by main process to signal cancellation.
    """
    _init_worker()

    # ── Set xet env var BEFORE any huggingface_hub imports ──────────
    # In a spawned child process, huggingface_hub.constants hasn't been
    # imported yet, so the module-level constant HF_HUB_DISABLE_XET
    # will be cached with the correct value when we import it below.
    # This is critical: the main process cannot toggle this env var at
    # runtime because huggingface_hub caches it at import time. But in
    # a fresh child process, we can set it before the first import.
    use_xet = params.get("use_xet", True)
    if not use_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    else:
        os.environ.pop("HF_HUB_DISABLE_XET", None)

    from .types import EventType, ProgressPhase, TransferCancelledError, TransferDirection
    from .callbacks import DownloadProgressTqdm, _dummy_file, state_manager
    from .subprocess.messages import SubprocessMessage

    transfer_id = params["transfer_id"]
    repo_id = params["repo_id"]

    def _is_cancelled() -> bool:
        return cancel_event.is_set()

    # Subclass DownloadProgressTqdm to route events through mp_queue
    class _SubprocessDownloadTqdm(DownloadProgressTqdm):
        """DownloadProgressTqdm subclass that sends events via mp_queue."""

        def _emit_event(self, event):
            """Override: route events through mp_queue as SubprocessMessage."""
            try:
                mp_queue.put_nowait(SubprocessMessage.event(event.to_dict()))
            except BaseException:
                pass

    # Bind the subclass with transfer parameters
    _BoundTqdm = DownloadProgressTqdm.bind(
        event_queue=None,  # Not used — _emit_event is overridden
        transfer_id=transfer_id,
        filename=repo_id,
        report_interval=params.get("report_interval", 0.1),
        is_cancelled=_is_cancelled,
    )

    # Create a combined class that inherits both _SubprocessDownloadTqdm
    # and _BoundTqdm, so we get both the mp_queue routing and the bound params
    class _SubprocessBoundTqdm(_SubprocessDownloadTqdm, _BoundTqdm):
        pass

    _SubprocessBoundTqdm.__name__ = f"SubprocessBoundTqdm_{transfer_id[:8]}"
    _SubprocessBoundTqdm.__qualname__ = f"SubprocessBoundTqdm_{transfer_id[:8]}"

    try:
        from huggingface_hub import snapshot_download

        download_kwargs = dict(
            repo_id=repo_id,
            repo_type=params.get("repo_type", "model"),
            revision=params.get("revision"),
            allow_patterns=params.get("allow_patterns"),
            ignore_patterns=params.get("ignore_patterns"),
            token=params.get("token"),
            endpoint=params.get("endpoint"),
            tqdm_class=_SubprocessBoundTqdm,
            force_download=params.get("force_download", False),
        )
        if params.get("local_dir"):
            download_kwargs["local_dir"] = params["local_dir"]

        result_path = snapshot_download(**download_kwargs) # nosec B615
    
        # Get final stats from state_manager for the COMPLETE event
        stats = state_manager.get_state(transfer_id)
        bytes_completed = stats.get("bytes_completed", 0)
        total_bytes = stats.get("total_bytes", 0)
        files_completed = stats.get("files_completed", 0)
        total_files = stats.get("total_files", 0)
    
        # Emit a COMPLETE event directly through mp_queue (matching
        # the standard_download pattern) so the main process receives
        # a proper COMPLETE with file_index/total_files.
        complete_event_dict = {
            "event_type": EventType.COMPLETE.value,
            "transfer_id": transfer_id,
            "direction": TransferDirection.DOWNLOAD.value,
            "filename": repo_id,
            "phase": ProgressPhase.COMPLETE.value,
            "bytes_completed": bytes_completed,
            "total_bytes": total_bytes,
            "percentage": 100.0,
            "speed": 0,
            "file_index": files_completed,
            "total_files": total_files,
        }
        _safe_put(mp_queue, SubprocessMessage.event(complete_event_dict))
    
        # Also send the result message so XetSubprocessRunner.wait()
        # knows the worker finished successfully.
        _safe_put(mp_queue, SubprocessMessage.result(
            filename=repo_id,
            destination_path=result_path,
            transfer_id=transfer_id,
            direction="download",
            file_size=bytes_completed,
            bytes_completed=bytes_completed,
            total_bytes=total_bytes,
            files_completed=files_completed,
            total_files=total_files,
        ))

    except (KeyboardInterrupt, Exception) as e:
        _handle_worker_exception(mp_queue, e)
    finally:
        state_manager.clear_state(transfer_id)


# ── Upload Workers ────────────────────────────────────────────────

def _upload_file_worker(params: Dict[str, Any], mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Worker function for Xet file uploads.

    Args:
        params: Dict with keys: file_path, repo_id, token, repo_type,
            revision, endpoint, transfer_id, report_interval.
        mp_queue: Queue for sending SubprocessMessage back to main process.
        cancel_event: Event set by main process to signal cancellation.
    """
    _init_worker()
    from .types import EventType, ProgressPhase, TransferCancelledError, TransferDirection

    transfer_id = params["transfer_id"]
    file_path = params["file_path"]
    filename = os.path.basename(file_path)

    try:
        import hf_xet
        from huggingface_hub.utils._xet import (
            XetTokenType,
            fetch_xet_connection_info_from_repo_info,
        )
    except ImportError as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=str(e), error_type="ImportError", retryable=False,
        ))
        return

    try:
        file_size = os.path.getsize(file_path)
    except OSError as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=f"File not found: {file_path}", error_type=type(e).__name__,
        ))
        return

    # Get upload credentials
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=params["token"], endpoint=params.get("endpoint"))
        headers = api._build_hf_headers()

        connection_info = fetch_xet_connection_info_from_repo_info(
            token_type=XetTokenType.WRITE,
            repo_id=params["repo_id"],
            repo_type=params.get("repo_type", "model"),
            revision=params.get("revision"),
            headers=headers,
            endpoint=params.get("endpoint"),
        )

        def token_refresher():
            info = fetch_xet_connection_info_from_repo_info(
                token_type=XetTokenType.WRITE,
                repo_id=params["repo_id"],
                repo_type=params.get("repo_type", "model"),
                revision=params.get("revision"),
                headers=headers,
                endpoint=params.get("endpoint"),
            )
            return info.access_token, info.expiration_unix_epoch
    except Exception as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=f"Failed to get upload credentials: {e}",
            error_type=type(e).__name__,
        ))
        return

    # Build progress callback
    callback = _make_progress_callback(
        filename=filename,
        total_bytes=file_size,
        transfer_id=transfer_id,
        mp_queue=mp_queue,
        cancel_event=cancel_event,
        direction="upload",
        report_interval=params.get("report_interval", 0.1),
    )

    try:
        results = hf_xet.upload_files(
            [file_path],
            connection_info.endpoint,
            (connection_info.access_token, connection_info.expiration_unix_epoch),
            token_refresher,
            callback,
            params.get("repo_type", "model"),
        )

        result_info = results[0] if results else None
        _safe_put(mp_queue, SubprocessMessage.result(
            filename=filename,
            file_size=file_size,
            transfer_id=transfer_id,
            hash=getattr(result_info, "hash", "") if result_info else "",
            url=getattr(result_info, "url", None) if result_info else None,
        ))

    except (KeyboardInterrupt, Exception) as e:
        _handle_worker_exception(mp_queue, e)


def _upload_bytes_worker(params: Dict[str, Any], mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Worker function for Xet bytes uploads.

    For large payloads (>10MB), the caller should write bytes to a temp file
    and pass ``file_path`` instead of ``file_content`` to avoid excessive
    pickle serialization cost.

    Args:
        params: Dict with keys: file_content (bytes) OR file_path (str),
            filename, repo_id, token, repo_type, revision, endpoint,
            transfer_id, report_interval.
        mp_queue: Queue for sending SubprocessMessage back to main process.
        cancel_event: Event set by main process to signal cancellation.
    """
    _init_worker()
    from .types import EventType, ProgressPhase, TransferCancelledError, TransferDirection

    transfer_id = params["transfer_id"]
    filename = params["filename"]

    # Determine content source
    file_content = params.get("file_content")
    file_path = params.get("file_path")

    if file_content is not None:
        file_size = len(file_content)
    elif file_path is not None:
        file_size = os.path.getsize(file_path)
    else:
        _safe_put(mp_queue, SubprocessMessage.error(
            message="Either file_content or file_path must be provided",
            error_type="ValueError",
        ))
        return

    try:
        import hf_xet
        from huggingface_hub.utils._xet import (
            XetTokenType,
            fetch_xet_connection_info_from_repo_info,
        )
    except ImportError as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=str(e), error_type="ImportError", retryable=False,
        ))
        return

    # Get upload credentials
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=params["token"], endpoint=params.get("endpoint"))
        headers = api._build_hf_headers()

        connection_info = fetch_xet_connection_info_from_repo_info(
            token_type=XetTokenType.WRITE,
            repo_id=params["repo_id"],
            repo_type=params.get("repo_type", "model"),
            revision=params.get("revision"),
            headers=headers,
            endpoint=params.get("endpoint"),
        )

        def token_refresher():
            info = fetch_xet_connection_info_from_repo_info(
                token_type=XetTokenType.WRITE,
                repo_id=params["repo_id"],
                repo_type=params.get("repo_type", "model"),
                revision=params.get("revision"),
                headers=headers,
                endpoint=params.get("endpoint"),
            )
            return info.access_token, info.expiration_unix_epoch
    except Exception as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=f"Failed to get upload credentials: {e}",
            error_type=type(e).__name__,
        ))
        return

    # Build progress callback
    callback = _make_progress_callback(
        filename=filename,
        total_bytes=file_size,
        transfer_id=transfer_id,
        mp_queue=mp_queue,
        cancel_event=cancel_event,
        direction="upload",
        report_interval=params.get("report_interval", 0.1),
    )

    try:
        if file_content is not None:
            results = hf_xet.upload_bytes(
                [file_content],
                connection_info.endpoint,
                (connection_info.access_token, connection_info.expiration_unix_epoch),
                token_refresher,
                callback,
                params.get("repo_type", "model"),
            )
        else:
            results = hf_xet.upload_files(
                [file_path],
                connection_info.endpoint,
                (connection_info.access_token, connection_info.expiration_unix_epoch),
                token_refresher,
                callback,
                params.get("repo_type", "model"),
            )

        result_info = results[0] if results else None
        _safe_put(mp_queue, SubprocessMessage.result(
            filename=filename,
            file_size=file_size,
            transfer_id=transfer_id,
            hash=getattr(result_info, "hash", "") if result_info else "",
            url=getattr(result_info, "url", None) if result_info else None,
        ))

    except (KeyboardInterrupt, Exception) as e:
        _handle_worker_exception(mp_queue, e)