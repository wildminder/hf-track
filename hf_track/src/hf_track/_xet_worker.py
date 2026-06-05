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
import signal
import time
from typing import Any, Dict, List

from .subprocess.messages import SubprocessMessage

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

    Used by both the legacy ``_make_progress_callback`` (2-arg mode) and
    the new ``XetSession`` API callback (``_xet_session_snapshot_worker``
    and ``_xet_session_download_worker``).

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


# ── Streaming Xet Download Worker (NEW 2026-06-04) ────────────────
#
# This worker uses ``hf_xet.XetSession().new_download_stream_group()``
# instead of ``hf_xet.download_files()`` to bypass the buffered API.
# It writes each file's chunks to disk immediately via ``os.write`` on
# a raw file descriptor (no userspace buffer) and calls ``os.fsync``
# periodically so a SIGKILL preserves the in-flight data.
#
# See: docs/plans/2026-06-04-xet-streaming-subprocess.md for the
# rationale and the test coverage matrix.


# Default fsync cadence: commit the OS page cache to disk every N bytes.
# 4 MiB is a good trade-off: cheap on SSD, fast enough on HDD, and
# ensures no more than ~4 MiB of work is lost on SIGKILL.
DEFAULT_FSYNC_INTERVAL = 4 * 1024 * 1024


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


def _xet_streaming_download_worker(
    params: Dict[str, Any],
    mp_queue: mp.Queue,
    cancel_event: mp.Event,
    session_factory: Any = None,
) -> None:
    """Worker function for multi-file Xet downloads via the STREAMING API.

    Runs in a child process. Imports ``hf_xet`` locally.

    Uses ``hf_xet.XetSession().new_download_stream_group().download_stream()``
    to write each file's chunks to disk incrementally. Memory stays
    bounded by chunk size (~4 MB) instead of file size. Each chunk is
    written via ``os.write`` on a raw fd (no userspace buffer) and
    ``os.fsync`` is called every ``fsync_interval`` bytes plus once at
    the end in a ``finally`` block. This guarantees that a SIGKILL
    at any point leaves a non-empty, non-truncated file on disk.

    Args:
        params: Dict with keys: file_specs (list of dicts with
            ``hash``, ``file_size``, ``dest_path``, ``xet_file_data``),
            token, endpoint, transfer_id, report_interval,
            request_headers, fsync_interval (optional, default 4 MiB).
        mp_queue: Queue for sending SubprocessMessage back to main process.
        cancel_event: Event set by main process to signal cancellation.

    Cancellation semantics:
        - Cooperative: ``cancel_event.is_set()`` is checked between
          chunks. If set, the worker's inner loop calls ``stream.cancel()``
          (best-effort), breaks, and emits a CANCELLED message.
        - Hard: the parent process can ``terminate()`` the child at any
          point. The ``finally`` block ensures ``os.fsync`` + ``os.close``
          run on the in-flight file before the OS reaps the process.
    """
    _init_worker()
    from .types import (
        EventType,
        ProgressPhase,
        TransferCancelledError,
        TransferDirection,
    )

    transfer_id = params["transfer_id"]
    file_specs: List[Dict[str, Any]] = params["file_specs"]
    total_files = len(file_specs)
    fsync_interval = params.get("fsync_interval", DEFAULT_FSYNC_INTERVAL)

    try:
        import hf_xet
        from huggingface_hub.utils._xet import refresh_xet_connection_info
    except ImportError as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=str(e), error_type="ImportError", retryable=False,
        ))
        return

    # Use first file's xet_file_data for credentials. We refresh the
    # connection per file (see below) because the access token is scoped
    # to a specific xet file's blocks; downloading a different file with
    # another file's token would either fail or yield zero bytes (the
    # symptom we hit when downloading all three xet files in a snapshot
    # in one subprocess).
    headers = params.get("request_headers", {})

    # Build the XetSession once. The per-file DownloadStreamGroup is
    # rebuilt on each iteration of the file loop, because the group
    # binds to a single (endpoint, token) pair.
    try:
        session = hf_xet.XetSession()
    except Exception as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=f"Failed to build XetSession: {e}",
            error_type=type(e).__name__,
        ))
        return

    # Per-file group: we will re-create this in the loop below.
    group = None

    # Per-file throttler
    throttler = _ProgressThrottler(report_interval=params.get("report_interval", 0.1))

    # Track overall progress across all files
    total_bytes_all = sum(spec["file_size"] for spec in file_specs)
    bytes_completed_all = 0
    start_time = time.time()
    transfer_completed = 0
    transfer_speed = 0.0

    try:
        for i, spec in enumerate(file_specs):
            filename = os.path.basename(spec["dest_path"])
            file_hash = spec["hash"]
            file_size = spec["file_size"]
            dest_path = spec["dest_path"]
            import sys as _sys
            print(f"[DBG] >>> file {i+1}/{total_files} START: {filename} (size={file_size})", file=_sys.stderr, flush=True)
            _t0 = time.time()

            # Cancellation check between files
            if cancel_event.is_set():
                from .types import TransferCancelledError
                raise TransferCancelledError("Transfer cancelled by user")

            # Empty file: create and emit COMPLETE.
            # NOTE: we deliberately do NOT send a per-file SubprocessMessage.result()
            # here. The relay thread treats ``result`` as the terminal message and
            # breaks out of the loop on the first one it sees. If we emitted one
            # per file, the relay would break after file 1 and lose all remaining
            # progress events and the final result. Per-file completion is
            # communicated via the COMPLETE event below; the final terminal
            # ``result`` is sent once after the loop ends.
            if file_size == 0:
                fd = _open_unbuffered(dest_path)
                os.close(fd)
                _safe_put(mp_queue, SubprocessMessage.event({
                    "event_type": EventType.COMPLETE.value,
                    "transfer_id": transfer_id,
                    "direction": TransferDirection.DOWNLOAD.value,
                    "filename": filename,
                    "phase": ProgressPhase.COMPLETE.value,
                    "bytes_completed": 0,
                    "total_bytes": 0,
                    "percentage": 100.0,
                    "speed": 0,
                    "file_index": i,
                    "total_files": total_files,
                    "transfer_bytes_completed": transfer_completed,
                    "transfer_bytes_total": total_bytes_all,
                    "transfer_speed": transfer_speed,
                }))
                continue

            # Open the stream
            file_info = hf_xet.XetFileInfo(file_hash, file_size)

            # Refresh the connection_info + group for this specific file.
            # The Xet access token is scoped to a specific xet file's
            # blocks; if we reuse the previous file's group/token, the
            # stream will return 0 bytes for the new file (the symptom
            # we hit when downloading 3 xet files in a single batch).
            try:
                this_xet_file_data = _deserialize_xet_file_data(
                    spec.get("xet_file_data", {})
                )
                this_conn_info = refresh_xet_connection_info(
                    file_data=this_xet_file_data, headers=headers,
                )
                # _XetFileDataProxy has __slots__, so use getattr
                refresh_route = getattr(this_xet_file_data, "refresh_route", "")
                group = session.new_download_stream_group(
                    endpoint=this_conn_info.endpoint,
                    token=this_conn_info.access_token,
                    token_expiry_unix_secs=this_conn_info.expiration_unix_epoch,
                    token_refresh_url=refresh_route,
                    token_refresh_headers=headers,
                )
                print(f"[DBG]   group built in {time.time()-_t0:.2f}s", file=_sys.stderr, flush=True); _t0 = time.time()
            except Exception as e:
                _safe_put(mp_queue, SubprocessMessage.error(
                    message=f"Failed to refresh credentials for {filename}: {e}",
                    error_type=type(e).__name__,
                ))
                return

            try:
                stream = group.download_stream(file_info)
                print(f"[DBG]   download_stream opened in {time.time()-_t0:.2f}s", file=_sys.stderr, flush=True); _t0 = time.time()
            except Exception as e:
                _safe_put(mp_queue, SubprocessMessage.error(
                    message=f"Failed to open stream for {filename}: {e}",
                    error_type=type(e).__name__,
                ))
                return

            # Iterate chunks, write to disk, emit progress
            fd = _open_unbuffered(dest_path)
            bytes_completed = 0
            bytes_since_fsync = 0
            file_start_time = time.time()
            cancelled = False
            write_failed = False

            try:
                for chunk in stream:
                    # Cooperative cancellation between chunks
                    if cancel_event.is_set():
                        if hasattr(stream, "cancel"):
                            try:
                                stream.cancel()
                            except Exception:
                                pass
                        cancelled = True
                        break
                    if not chunk:
                        continue
                    try:
                        os.write(fd, chunk)
                    except OSError as e:
                        _safe_put(mp_queue, SubprocessMessage.error(
                            message=f"os.write failed for {filename}: {e}",
                            error_type=type(e).__name__,
                        ))
                        write_failed = True
                        break
                    bytes_completed += len(chunk)
                    bytes_since_fsync += len(chunk)
                    if bytes_since_fsync >= fsync_interval:
                        try:
                            os.fsync(fd)
                        except OSError:
                            pass
                        bytes_since_fsync = 0

                    # Emit PROGRESS event (throttled)
                    now = time.time()
                    transfer_completed = bytes_completed_all + bytes_completed
                    elapsed = now - start_time
                    transfer_speed = (transfer_completed / elapsed) if elapsed > 0 else 0.0
                    if throttler.should_emit(transfer_completed, total_bytes_all, now):
                        _safe_put(mp_queue, SubprocessMessage.event({
                            "event_type": EventType.PROGRESS.value,
                            "transfer_id": transfer_id,
                            "direction": TransferDirection.DOWNLOAD.value,
                            "filename": filename,
                            "phase": ProgressPhase.DOWNLOADING.value,
                            "bytes_completed": bytes_completed,
                            "total_bytes": file_size,
                            "percentage": (bytes_completed / file_size * 100.0) if file_size else 0.0,
                            "speed": transfer_speed,
                            "file_index": i,
                            "total_files": total_files,
                            "transfer_bytes_completed": transfer_completed,
                            "transfer_bytes_total": total_bytes_all,
                            "transfer_speed": transfer_speed,
                        }))
                        throttler.record_emit(transfer_completed, now)
            finally:
                # Always fsync + close, even on cancel / error / success.
                try:
                    os.fsync(fd)
                except OSError:
                    pass
                try:
                    os.close(fd)
                except OSError:
                    pass

            if cancelled:
                _safe_put(mp_queue, SubprocessMessage.cancelled(
                    message=f"Cancelled mid-download of {filename} "
                            f"({bytes_completed}/{file_size} bytes on disk)",
                ))
                return

            if write_failed:
                return

            # Verify size match
            if bytes_completed != file_size:
                _safe_put(mp_queue, SubprocessMessage.error(
                    message=f"Size mismatch for {filename}: got {bytes_completed} bytes, "
                            f"expected {file_size}",
                    error_type="SizeMismatch",
                ))
                return

            bytes_completed_all += bytes_completed

            # Emit COMPLETE event for this file.
            # NOTE: we deliberately do NOT send a per-file SubprocessMessage.result()
            # here (see empty-file branch above for the full rationale). The
            # COMPLETE event above is what tells the parent "this file is done";
            # the terminal ``result`` is sent once after the loop ends.
            now = time.time()
            elapsed = now - start_time
            transfer_speed = (bytes_completed_all / elapsed) if elapsed > 0 else 0.0
            _safe_put(mp_queue, SubprocessMessage.event({
                "event_type": EventType.COMPLETE.value,
                "transfer_id": transfer_id,
                "direction": TransferDirection.DOWNLOAD.value,
                "filename": filename,
                "phase": ProgressPhase.COMPLETE.value,
                "bytes_completed": bytes_completed,
                "total_bytes": file_size,
                "percentage": 100.0,
                "speed": 0,
                "file_index": i,
                "total_files": total_files,
                "transfer_bytes_completed": bytes_completed_all,
                "transfer_bytes_total": total_bytes_all,
                "transfer_speed": transfer_speed,
            }))

        # Emit ONE terminal ``result`` message after the loop ends.
        # The relay thread treats this as the signal to stop and return
        # the result to the parent. The payload summarizes the whole
        # transfer (not a single file) so the parent can confirm
        # ``status == "success"`` and look up the per-file dest paths
        # via ``file_specs``.
        _safe_put(mp_queue, SubprocessMessage.result(
            filename=os.path.basename(file_specs[-1]["dest_path"]),
            destination_path=os.path.commonpath(
                [s["dest_path"] for s in file_specs]
            ) if len({os.path.dirname(s["dest_path"]) for s in file_specs}) == 1
            else file_specs[-1]["dest_path"],
            file_size=bytes_completed_all,
            transfer_id=transfer_id,
            file_index=len(file_specs),
            total_files=len(file_specs),
            bytes_completed=bytes_completed_all,
            total_bytes=total_bytes_all,
        ))

    except (KeyboardInterrupt, Exception) as e:
        _handle_worker_exception(mp_queue, e)


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


# ── Legacy XetSession-based Workers (DEPRECATED 2026-06-03) ─────
#
# These two workers (``_xet_session_snapshot_worker`` and
# ``_xet_session_download_worker``) are deprecated as of 2026-06-03
# due to three critical correctness bugs. They are kept here as
# re-exports from ``_xet_worker_legacy`` for backward compatibility
# with direct importers (e.g. ``from hf_track._xet_worker import
# _xet_session_snapshot_worker``).
#
# New code should use ``_snapshot_worker`` and ``_download_worker``
# instead. See :mod:`hf_track._xet_worker_legacy` for the
# deprecation rationale.

from ._xet_worker_legacy import (  # noqa: E402, F401
    _xet_session_download_worker,
    _xet_session_snapshot_worker,
)

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