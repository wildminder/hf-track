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

from .subprocess_messages import SubprocessMessage

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
    from .subprocess_messages import SubprocessMessage

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


# ── New XetSession-based Snapshot Worker (Phase 2.J) ─────────────

def _xet_session_snapshot_worker(params: Dict[str, Any], mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Snapshot download worker using the NEW ``hf_xet.XetSession`` API.

    This replaces the broken ``huggingface_hub.snapshot_download()`` flow
    (which uses a 1-arg ``progress_updater`` callback that only fires at
    file completion) with the new ``XetSession`` API that has a
    ``progress_callback`` firing every 100ms with actual per-chunk progress.

    Key differences from ``_snapshot_worker``:
    - Uses ``hf_xet.XetSession.new_file_download_group(progress_callback=...)``
      instead of ``huggingface_hub.snapshot_download(tqdm_class=...)``.
    - The progress callback receives ``(GroupProgressReport, dict[UniqueID, ItemProgressReport])``
      and computes the increment from ``total_transfer_bytes_completed``.
    - Smooth per-chunk progress: bar advances every ~100ms during file downloads.

    Args:
        params: Dict with keys: files (list of {filename, xet_hash, size, refresh_route}),
            transfer_id, report_interval.
        mp_queue: Queue for sending SubprocessMessage back to main process.
        cancel_event: Event set by main process to signal cancellation.
    """
    _init_worker()

    from .types import EventType, ProgressPhase, TransferCancelledError, TransferDirection
    from .subprocess_messages import SubprocessMessage

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
    _last_logical_completed = 0

    def _is_cancelled() -> bool:
        return cancel_event.is_set()

    def on_progress(group_report, item_reports):
        """Progress callback for the new XetSession API.

        Receives (GroupProgressReport, dict[UniqueID, ItemProgressReport]).
        Computes the increment from total_transfer_bytes_completed and
        emits a PROGRESS event via mp_queue.
        """
        nonlocal _last_transfer_completed, _files_completed, _total_transfer_bytes
        nonlocal _last_logical_completed

        if _is_cancelled():
            return

        current_transfer_completed = group_report.total_transfer_bytes_completed
        current_logical_completed = group_report.total_bytes_completed
        increment = current_transfer_completed - _last_transfer_completed
        _last_transfer_completed = current_transfer_completed
        _total_transfer_bytes = group_report.total_transfer_bytes

        # Count files completed from item_reports
        for uid, item in item_reports.items():
            if item.bytes_completed >= item.total_bytes and item.total_bytes > 0:
                # File just completed - will be counted in _check_file_completion
                pass

        # Use logical bytes for display so the percentage matches what the
        # user expects ("I downloaded N% of the file"). Logical bytes advance
        # in larger jumps at file boundaries but are more meaningful.
        # Transfer bytes advance smoothly but are smaller than logical due
        # to dedup — using them for percentage would make the bar never
        # reach 100% for heavily-deduplicated files.
        #
        # However, we ALSO need smooth progress between file completions.
        # Solution: use max(logical, transfer) as display value so we
        # always show the user's view of progress.
        if _total_logical_bytes > 0 and _total_transfer_bytes > 0:
            # Estimate: how much logical content has been "transferred"?
            # Ratio of logical to transfer (close to 1 for unique content).
            logical_per_transfer = _total_logical_bytes / _total_transfer_bytes
            estimated_logical_done = int(current_transfer_completed * logical_per_transfer)
            display_completed = max(current_logical_completed, estimated_logical_done)
        else:
            display_completed = current_transfer_completed
        display_total = _total_logical_bytes or _total_transfer_bytes

        # Compute percentage
        if display_total > 0:
            display_percentage = (display_completed / display_total) * 100.0
        else:
            display_percentage = 0.0

        # Skip if nothing meaningful changed (no new bytes in either domain)
        bytes_change = (
            (current_logical_completed - _last_logical_completed) +
            (current_transfer_completed - _last_transfer_completed)
        )
        _last_logical_completed = current_logical_completed
        if bytes_change <= 0 and not hasattr(on_progress, "_last_emit"):
            return
        if bytes_change <= 0 and on_progress._last_emit > 0:
            return

        # Emit PROGRESS event
        # Throttle: skip if last emit was very recent
        now = time.time()
        if not hasattr(on_progress, "_last_emit"):
            on_progress._last_emit = 0.0
        if now - on_progress._last_emit < (report_interval_ms / 1000.0) * 0.5:
            return
        on_progress._last_emit = now

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

        # Wait for completion (or cancellation)
        if not _is_cancelled():
            report = group.wait_to_finish()
        else:
            try:
                group.abort()
            except Exception:
                pass
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
    _init_worker()

    from .types import EventType, ProgressPhase, TransferCancelledError, TransferDirection
    from .subprocess_messages import SubprocessMessage

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