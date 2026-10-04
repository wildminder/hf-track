"""Tier-1 file transfer and the fsync helper it needs (NTH-016, step S32).

``_run_tier1_file`` is the per-file half of the hybrid streaming path:
it opens an unbuffered handle on the destination, hands the file's token
route to a ``new_file_download_group``, and polls the progress callback
until the deadline. ``_open_unbuffered`` exists because ``hf_xet`` writes
through the descriptor it is given — a buffered Python file object would
swallow the bytes the progress callback has already reported.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from typing import Any, Dict, Optional

from ._xet_worker_common import _deserialize_xet_file_data, _safe_put
from .subprocess.messages import SubprocessMessage

logger = logging.getLogger(__name__)


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


def _run_tier1_file(
    *,
    session: Any,
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
        from huggingface_hub.utils._xet import xet_headers_without_auth

        file_data = _deserialize_xet_file_data(xet_file_data) if xet_file_data else None
        refresh_route = getattr(file_data, "refresh_route", "") if file_data else ""
        # huggingface_hub 1.x removed ``refresh_xet_connection_info``, which
        # used to mint the CAS token here. The download group now takes the
        # file's refresh route and refreshes the token itself.
        group = session.new_file_download_group(
            token_refresh_url=refresh_route,
            token_refresh_headers=headers,
            custom_headers=xet_headers_without_auth(headers),
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
