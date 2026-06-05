"""Legacy XetSession-based workers (DEPRECATED 2026-06-03).

This module contains the **deprecated** ``_xet_session_*`` workers that
use the ``hf_xet.XetSession`` API for snapshot and single-file downloads.

Both workers were **deprecated on 2026-06-03** because of three critical
correctness bugs:

1. **Nested folder layout**: ``HfFileSystem.ls()`` returns names like
   ``openbmb/VoxCPM-0.5B/README.md`` (no ``models/`` prefix). The
   original path parsing did ``full_name.split("/", 1)[1]`` which only
   stripped one segment, leaving ``VoxCPM-0.5B/README.md`` in the
   relative path. Result: every file written to
   ``<local_dir>/VoxCPM-0.5B/<file>`` instead of ``<local_dir>/<file>``.

2. **Missing files**: ``fs.ls(...)`` was non-recursive (so files in
   subdirectories like ``assets/`` were missed), and files without
   ``xet_hash`` (regular LFS files like ``config.json``,
   ``tokenizer.json``) were explicitly skipped.

3. **Zero-length files on cancel**: ``group.start_download_file()``
   creates the destination file with no pre-cleanup, and the xet
   runtime does not use atomic tmp-file + rename. Cancelling leaves
   0-byte files on disk; the next run sees the file "exists" and
   skips re-downloading.

The new ``XetSession`` API is fundamentally a single-session,
single-batch primitive that does not provide the mixed xet/non-xet
routing, subdirectory traversal, or cache-aware resume that snapshot
downloads require.

These workers are kept in the codebase for backward compatibility with
any direct callers, and to allow future re-implementation if a use case
justifies the complexity. New code should use ``_snapshot_worker`` and
``_download_worker`` instead.

See plan ``docs/plans/2026-06-03-revert-broken-snapshot-xetsession-path.md``.

.. note::

    Public re-exports are provided from the top-level ``hf_track`` package
    via :mod:`hf_track._xet_worker`, so ``from hf_track._xet_worker
    import _xet_session_snapshot_worker`` continues to work.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Any, Dict

# NOTE: Helpers from ``hf_track._xet_worker`` (``_init_worker``,
# ``_safe_put``, ``_handle_worker_exception``, ``_ProgressThrottler``,
# ``_download_worker``) are imported lazily inside the worker function
# bodies below. Doing the imports at module-top-level would create a
# circular import: ``_xet_worker`` re-exports the deprecated workers
# from this module so that the historical import path
# ``from hf_track._xet_worker import _xet_session_snapshot_worker``
# continues to work. By the time those names are needed (inside the
# function bodies), both modules are fully loaded and the imports
# succeed.


# ── New XetSession-based Snapshot Worker (DEPRECATED 2026-06-03) ──


def _xet_session_snapshot_worker(params: Dict[str, Any], mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Snapshot download worker using the NEW ``hf_xet.XetSession`` API.

    .. deprecated::
        This worker is **deprecated** as of 2026-06-03. It is no longer
        used by the public ``download_snapshot_with_xet_session`` API
        (which now delegates to the proven ``_snapshot_worker``).

        The worker had three critical correctness bugs:

        1. **Nested folder layout**: ``HfFileSystem.ls()`` returns
           names like ``openbmb/VoxCPM-0.5B/README.md`` (no ``models/``
           prefix). The original path parsing did
           ``full_name.split("/", 1)[1]`` which only stripped one
           segment, leaving ``VoxCPM-0.5B/README.md`` in the relative
           path. Result: every file written to
           ``<local_dir>/VoxCPM-0.5B/<file>`` instead of
           ``<local_dir>/<file>``.

        2. **Missing files**: ``fs.ls(...)`` was non-recursive (so
           files in subdirectories like ``assets/`` were missed), and
           files without ``xet_hash`` (regular LFS files like
           ``config.json``, ``tokenizer.json``) were explicitly
           skipped.

        3. **Zero-length files on cancel**: ``group.start_download_file()``
           creates the destination file with no pre-cleanup, and the
           xet runtime does not use atomic tmp-file + rename. Cancelling
           leaves 0-byte files on disk; the next run sees the file
           "exists" and skips re-downloading.

        The new ``XetSession`` API is fundamentally a single-session,
        single-batch primitive that does not provide the mixed
        xet/non-xet routing, subdirectory traversal, or cache-aware
        resume that snapshot downloads require.

        Use ``_snapshot_worker`` instead. See plan
        ``docs/plans/2026-06-03-revert-broken-snapshot-xetsession-path.md``.

    Note:
        Kept in the codebase to maintain backward compatibility for
        any direct callers, and to allow future re-implementation if
        a use case justifies the complexity.

    Args:
        params: Dict with keys: files (list of {filename, xet_hash, size, refresh_route}),
            transfer_id, report_interval.
        mp_queue: Queue for sending SubprocessMessage back to main process.
        cancel_event: Event set by main process to signal cancellation.
    """
    # Lazy imports: avoid the circular import between this module and
    # ``_xet_worker`` (which re-exports the deprecated workers from
    # here so the historical import path keeps working).
    from ._xet_worker import (
        _ProgressThrottler,
        _handle_worker_exception,
        _init_worker,
        _safe_put,
    )
    _init_worker()

    from .types import EventType, ProgressPhase, TransferCancelledError, TransferDirection
    from .subprocess.messages import SubprocessMessage

    transfer_id = params["transfer_id"]
    files = params["files"]  # list of {"filename", "xet_hash", "size", "refresh_route"}
    report_interval_ms = int(params.get("report_interval", 0.1) * 1000)
    repo_id = params.get("repo_id", "snapshot")

    try:
        import hf_xet
    except ImportError as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=str(e), error_type="ImportError", retryable=False,
        ))
        return

    if not hasattr(hf_xet, "XetSession"):
        _safe_put(mp_queue, SubprocessMessage.error(
            message="hf_xet.XetSession not available (requires hf_xet>=1.5.0).",
            error_type="ImportError",
            retryable=False,
        ))
        return

    # State tracking
    _start_time = time.time()
    _last_transfer_completed = 0
    _files_completed = 0
    _total_transfer_bytes = 0
    _total_logical_bytes = sum(f.get("size", 0) for f in files)
    # Use the shared throttler (Phase 6) for cleaner, testable throttling.
    # The new XetSession API fires the callback every progress_interval_ms,
    # but we only forward to mp_queue at the requested rate.
    _throttler = _ProgressThrottler(
        report_interval=max(0.05, params.get("report_interval", 0.1) / 2.0),
    )

    def _is_cancelled() -> bool:
        return cancel_event.is_set()

    def on_progress(group_report, item_reports):
        """Progress callback for the new XetSession API.

        Receives (GroupProgressReport, dict[UniqueID, ItemProgressReport]).
        Computes the display value from total_transfer_bytes_completed
        (smoother) and total_bytes_completed (more accurate, but jumps
        at file boundaries) and emits a throttled PROGRESS event via
        mp_queue.

        Phase 5 (2026-06-03): Throttling logic rewritten to use the
        ``_ProgressThrottler`` class. Previous implementation had a
        subtle bug where the first event with no byte change was
        silently dropped, and subsequent events with no byte change
        were also dropped. This caused the progress bar to never
        update. Now the first event is ALWAYS emitted, and subsequent
        events are throttled by time and byte delta.
        """
        nonlocal _last_transfer_completed, _files_completed
        nonlocal _total_transfer_bytes

        if _is_cancelled():
            return

        current_transfer_completed = group_report.total_transfer_bytes_completed
        current_logical_completed = group_report.total_bytes_completed
        _last_transfer_completed = current_transfer_completed
        _total_transfer_bytes = group_report.total_transfer_bytes

        # Build the display value:
        # - `total_bytes_completed` advances only at file boundaries (jumpy)
        # - `total_transfer_bytes_completed` advances per-chunk (smooth)
        #   but can be < total_logical_bytes due to dedup
        # We want: smooth per-chunk + reach 100% at the end.
        # Use max() of the two, which gives us the higher of the
        # chunked/transfer view vs the file-boundary/logical view.
        if _total_logical_bytes > 0 and _total_transfer_bytes > 0:
            # Estimate: how much logical content has been "transferred"?
            # Ratio of logical to transfer (close to 1 for unique content,
            # > 1 for deduplicated content).
            logical_per_transfer = _total_logical_bytes / _total_transfer_bytes
            estimated_logical_done = int(current_transfer_completed * logical_per_transfer)
            display_completed = max(current_logical_completed, estimated_logical_done)
        else:
            display_completed = max(current_logical_completed, current_transfer_completed)
        display_total = _total_logical_bytes or _total_transfer_bytes

        # Compute percentage
        if display_total > 0:
            display_percentage = (display_completed / display_total) * 100.0
        else:
            display_percentage = 0.0

        # Count files completed by counting items that reached their total.
        files_done_now = 0
        for uid, item in item_reports.items():
            if item.total_bytes > 0 and item.bytes_completed >= item.total_bytes:
                files_done_now += 1
        if files_done_now > _files_completed:
            _files_completed = files_done_now

        # Throttle: always emit first event, then by time/byte delta.
        now = time.time()
        if not _throttler.should_emit(display_completed, display_total, now):
            return
        _throttler.record_emit(display_completed, now)

        try:
            event_dict = {
                "event_type": EventType.PROGRESS.value,
                "transfer_id": transfer_id,
                "direction": TransferDirection.DOWNLOAD.value,
                "filename": repo_id,
                "phase": ProgressPhase.DOWNLOADING.value,
                "bytes_completed": display_completed,
                "total_bytes": display_total,
                "percentage": display_percentage,
                "speed": int(group_report.total_transfer_bytes_completion_rate or 0),
                "file_index": _files_completed,
                "total_files": len(files),
                "transfer_bytes_completed": current_transfer_completed,
                "transfer_bytes_total": _total_transfer_bytes,
                "transfer_speed": int(group_report.total_transfer_bytes_completion_rate or 0),
            }
            _safe_put(mp_queue, SubprocessMessage.event(event_dict))
        except Exception:
            pass

    try:
        session = hf_xet.XetSession()

        # Use the refresh_route from the first file (they should all be the same repo)
        first_file = files[0] if files else {}
        refresh_route = first_file.get("refresh_route", "")

        if not refresh_route:
            _safe_put(mp_queue, SubprocessMessage.error(
                message="No refresh_route provided for xet session",
                error_type="ValueError",
            ))
            return

        # Get xet connection info
        try:
            from huggingface_hub.utils._xet import refresh_xet_connection_info
        except ImportError:
            # Fallback: fetch directly
            import httpx
            token_resp = httpx.get(refresh_route)
            token_data = token_resp.json()
            cas_url = token_data["casUrl"]
            token = token_data["accessToken"]
            exp = token_data["exp"]
        else:
            # Use the high-level API
            class _FileDataProxy:
                __slots__ = ("file_hash", "refresh_route")
                def __init__(self, fh, rr):
                    self.file_hash = fh
                    self.refresh_route = rr
            proxy = _FileDataProxy(first_file.get("xet_hash", ""), refresh_route)
            conn_info = refresh_xet_connection_info(file_data=proxy, headers={})
            cas_url = conn_info.endpoint
            token = conn_info.access_token
            exp = conn_info.expiration_unix_epoch

        # Create the download group with progress callback
        group = session.new_file_download_group(
            endpoint=cas_url,
            token=token,
            token_expiry_unix_secs=exp,
            token_refresh_url=refresh_route,
            token_refresh_headers={},
            custom_headers={},
            progress_callback=on_progress,
            progress_interval_ms=report_interval_ms,
        )

        # Start downloads for all files
        for f in files:
            if _is_cancelled():
                break
            xet_hash = f.get("xet_hash")
            size = f.get("size", 0)
            dest_path = f.get("dest_path", "")
            if not xet_hash or not dest_path:
                continue
            file_info = hf_xet.XetFileInfo(xet_hash, size)
            group.start_download_file(file_info, dest_path)

        # Wait for completion with cancellation polling.
        # The new XetSession API has group.wait_to_finish() which blocks
        # until all downloads complete. It does NOT support timeout or
        # cancellation. We need to watch cancel_event in a side thread
        # and call group.abort() if cancellation is requested.
        # (Phase 5.4: cancellation poller for XetSession group.)
        import threading as _threading
        _abort_triggered = _threading.Event()

        def _cancellation_watcher():
            """Watch cancel_event; if set, call group.abort()."""
            while not _abort_triggered.is_set():
                if _is_cancelled():
                    try:
                        group.abort()
                    except Exception:
                        pass
                    return
                # Poll at ~10 Hz. cancel_event.is_set() is fast.
                _abort_triggered.wait(timeout=0.1)
                if _abort_triggered.is_set():
                    return

        watcher_thread = _threading.Thread(
            target=_cancellation_watcher,
            name=f"xet-session-cancel-watcher-{transfer_id[:8]}",
            daemon=True,
        )
        watcher_thread.start()

        try:
            if not _is_cancelled():
                # Block until all downloads complete or abort is called
                report = group.wait_to_finish()
            else:
                try:
                    group.abort()
                except Exception:
                    pass
                raise TransferCancelledError("Download cancelled by user")
        finally:
            _abort_triggered.set()
            watcher_thread.join(timeout=0.5)

        # If cancellation was triggered, raise cleanly
        if _is_cancelled():
            raise TransferCancelledError("Download cancelled by user")

        # Count completed files
        _files_completed = len(files)

        # Emit COMPLETE event
        complete_bytes = _total_logical_bytes or _total_transfer_bytes
        complete_event_dict = {
            "event_type": EventType.COMPLETE.value,
            "transfer_id": transfer_id,
            "direction": TransferDirection.DOWNLOAD.value,
            "filename": repo_id,
            "phase": ProgressPhase.COMPLETE.value,
            "bytes_completed": complete_bytes,
            "total_bytes": complete_bytes,
            "percentage": 100.0,
            "speed": 0,
            "file_index": _files_completed,
            "total_files": len(files),
        }
        _safe_put(mp_queue, SubprocessMessage.event(complete_event_dict))

        # Send result message
        _safe_put(mp_queue, SubprocessMessage.result(
            filename=repo_id,
            destination_path=params.get("local_dir", ""),
            transfer_id=transfer_id,
            direction="download",
            file_size=_total_logical_bytes,
            bytes_completed=_total_logical_bytes,
            total_bytes=_total_logical_bytes,
            files_completed=_files_completed,
            total_files=len(files),
        ))

    except (KeyboardInterrupt, Exception) as e:
        _handle_worker_exception(mp_queue, e)


# ── New XetSession-based Single File Worker (Phase 2.K) ─────────


def _xet_session_download_worker(
    params: Dict[str, Any], mp_queue: mp.Queue, cancel_event: mp.Event
) -> None:
    """Single-file Xet download worker using the NEW ``hf_xet.XetSession`` API.

    Replaces the broken ``hf_xet.download_files()`` flow (which only supports
    a 1-arg ``progress_updater`` callback that fires at file completion in
    hf_xet >= 1.0) with the new ``XetSession`` API that has a
    ``progress_callback`` firing every 100ms with actual per-chunk progress.

    Args:
        params: Dict with keys: file_hash, file_size, dest_path, xet_file_data
            (dict), token, endpoint, transfer_id, report_interval,
            request_headers.
        mp_queue: Queue for sending SubprocessMessage back to main process.
        cancel_event: Event set by main process to signal cancellation.
    """
    # Lazy imports: avoid the circular import between this module and
    # ``_xet_worker`` (which re-exports the deprecated workers from
    # here so the historical import path keeps working).
    from ._xet_worker import (
        _download_worker,
        _handle_worker_exception,
        _init_worker,
        _safe_put,
    )
    _init_worker()

    from .types import EventType, ProgressPhase, TransferCancelledError, TransferDirection
    from .subprocess.messages import SubprocessMessage

    transfer_id = params["transfer_id"]
    filename = os.path.basename(params["dest_path"])
    file_size = params["file_size"]
    file_hash = params["file_hash"]
    dest_path = params["dest_path"]

    try:
        import hf_xet
    except ImportError as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=str(e), error_type="ImportError", retryable=False,
        ))
        return

    if not hasattr(hf_xet, "XetSession"):
        # Fall back to old detailed-callback worker
        kwargs = dict(
            params,
            filename=filename,
            direction="download",
        )
        _download_worker(kwargs, mp_queue, cancel_event)
        return

    # Reconstruct XetFileData-like proxy for token refresh
    xet_file_data_dict = params.get("xet_file_data", {})

    def _is_cancelled() -> bool:
        return cancel_event.is_set()

    # State tracking
    _start_time = time.time()
    _last_transfer_completed = 0
    _last_logical_completed = 0
    _total_transfer_bytes = 0
    _last_emit_time = 0.0
    _first_event_emitted = False
    report_interval_s = params.get("report_interval", 0.1)

    def on_progress(group_report, item_reports):
        """Progress callback for the new XetSession API."""
        nonlocal _last_transfer_completed, _last_logical_completed
        nonlocal _total_transfer_bytes, _last_emit_time, _first_event_emitted

        if _is_cancelled():
            return

        current_transfer_completed = group_report.total_transfer_bytes_completed
        current_logical_completed = group_report.total_bytes_completed
        _total_transfer_bytes = group_report.total_transfer_bytes

        # Decide which bytes to display. We want smooth progress (use
        # transfer bytes) but the user expects the bar to reach 100% when
        # the file is fully downloaded. Logical bytes reach the file size
        # at completion; transfer bytes may be smaller due to dedup.
        #
        # Strategy: use logical bytes if they've advanced, otherwise use
        # the transfer bytes * ratio estimate to give a smooth view.
        if current_logical_completed > 0:
            display_completed = current_logical_completed
        else:
            # Estimate logical from transfer rate
            if _total_transfer_bytes > 0 and file_size > 0:
                ratio = file_size / _total_transfer_bytes
                display_completed = min(int(current_transfer_completed * ratio), file_size)
            else:
                display_completed = current_transfer_completed

        display_total = file_size or _total_transfer_bytes
        if display_total > 0:
            display_percentage = (display_completed / display_total) * 100.0
        else:
            display_percentage = 0.0

        # Always emit the first event, then throttle by time.
        now = time.time()
        bytes_change = (
            (current_logical_completed - _last_logical_completed) +
            (current_transfer_completed - _last_transfer_completed)
        )
        if not _first_event_emitted:
            pass  # Always emit first
        elif bytes_change <= 0:
            return
        elif now - _last_emit_time < report_interval_s:
            return

        _last_transfer_completed = current_transfer_completed
        _last_logical_completed = current_logical_completed
        _last_emit_time = now
        _first_event_emitted = True

        try:
            event_dict = {
                "event_type": EventType.PROGRESS.value,
                "transfer_id": transfer_id,
                "direction": TransferDirection.DOWNLOAD.value,
                "filename": filename,
                "phase": ProgressPhase.DOWNLOADING.value,
                "bytes_completed": display_completed,
                "total_bytes": display_total,
                "percentage": display_percentage,
                "speed": int(group_report.total_transfer_bytes_completion_rate or 0),
                "file_index": 0,
                "total_files": 1,
                "transfer_bytes_completed": current_transfer_completed,
                "transfer_bytes_total": _total_transfer_bytes,
                "transfer_speed": int(group_report.total_transfer_bytes_completion_rate or 0),
            }
            _safe_put(mp_queue, SubprocessMessage.event(event_dict))
        except Exception:
            pass

    try:
        session = hf_xet.XetSession()

        # Get xet connection info via huggingface_hub's helper
        # (handles the proper URL/token flow)
        try:
            from huggingface_hub.utils._xet import (
                XetFileData,
                refresh_xet_connection_info,
            )

            # Use the refresh_route from the original XetFileData, or
            # construct from the file's URL as a fallback.
            xet_file_data = XetFileData(
                file_hash=xet_file_data_dict.get("file_hash", file_hash),
                refresh_route=xet_file_data_dict.get("refresh_route", ""),
            )
            ci = refresh_xet_connection_info(file_data=xet_file_data, headers={})
            cas_url = ci.endpoint
            token = ci.access_token
            exp = ci.expiration_unix_epoch
            refresh_route = xet_file_data.refresh_route
        except Exception as e:
            _safe_put(mp_queue, SubprocessMessage.error(
                message=f"Failed to get xet credentials: {e}",
                error_type=type(e).__name__,
            ))
            return

        # Create the download group with progress callback
        try:
            group = session.new_file_download_group(
                endpoint=cas_url,
                token=token,
                token_expiry_unix_secs=exp,
                token_refresh_url=refresh_route,
                token_refresh_headers={},
                custom_headers={},
                progress_callback=on_progress,
                progress_interval_ms=int(report_interval_s * 1000),
            )
        except Exception as e:
            # Some hf_xet versions may have different arg names; fall back
            # to the old detailed-callback worker.
            _safe_put(mp_queue, SubprocessMessage.error(
                message=f"XetSession.new_file_download_group failed: {e}; "
                f"falling back to old API",
                error_type=type(e).__name__,
            ))
            kwargs = dict(params, filename=filename, direction="download")
            _download_worker(kwargs, mp_queue, cancel_event)
            return

        # Start the file download
        try:
            file_info = hf_xet.XetFileInfo(file_hash, file_size)
            group.start_download_file(file_info, dest_path)
        except Exception as e:
            _safe_put(mp_queue, SubprocessMessage.error(
                message=f"Failed to start download: {e}",
                error_type=type(e).__name__,
            ))
            return

        # Wait for completion
        if not _is_cancelled():
            report = group.wait_to_finish()
        else:
            try:
                group.abort()
            except Exception:
                pass
            raise TransferCancelledError("Download cancelled by user")

        # Emit COMPLETE event
        complete_event_dict = {
            "event_type": EventType.COMPLETE.value,
            "transfer_id": transfer_id,
            "direction": TransferDirection.DOWNLOAD.value,
            "filename": filename,
            "phase": ProgressPhase.COMPLETE.value,
            "bytes_completed": file_size,
            "total_bytes": file_size,
            "percentage": 100.0,
            "speed": 0,
            "file_index": 0,
            "total_files": 1,
        }
        _safe_put(mp_queue, SubprocessMessage.event(complete_event_dict))

        # Send result message
        _safe_put(mp_queue, SubprocessMessage.result(
            filename=filename,
            destination_path=dest_path,
            file_size=file_size,
            transfer_id=transfer_id,
        ))

    except (KeyboardInterrupt, Exception) as e:
        _handle_worker_exception(mp_queue, e)
