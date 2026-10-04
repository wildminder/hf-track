"""Xet subprocess worker entry points.

These functions run inside child processes spawned by
``XetSubprocessRunner``. They import ``hf_xet`` locally (never at module
level) so the Rust .pyd extension is loaded only in the child process —
safe to terminate without affecting the main process.

Each worker:

1. Receives a plain ``dict`` of parameters (picklable across the process
   boundary)
2. Receives a ``multiprocessing.Queue`` for sending ``SubprocessMessage``
   back
3. Receives a ``multiprocessing.Event`` for cancellation signaling
4. Puts progress event messages during the transfer
5. Puts a terminal message (result/error/cancelled) when done

.. warning::

    These functions MUST be defined at module top-level (not nested)
    so they are picklable by ``multiprocessing`` with the ``spawn`` context.

Layout (NTH-016, step S32)
--------------------------
This module used to hold every worker, every runner and all the shared
plumbing — 1,753 lines against a 1,800-line budget, eleven lines of
headroom. It is now the two workers that are the *current* path, plus
re-exports:

============================  =========================================
Module                        Holds
============================  =========================================
``_xet_worker_common``        init/safe-put/throttler/serialization/callback
``_xet_legacy_worker``        the deprecated 2026-07-09 download workers
``_xet_upload_worker``        the upload workers
``_xet_hybrid``               ``download_hybrid``
``_xet_hybrid_tier1``         ``_run_tier1_file`` + ``_open_unbuffered``
``_xet_hybrid_runner``        ``TranslatingQueue`` / ``HybridRunner``
``_xet_worker`` (this file)   ``_xet_file_only_worker``, ``_snapshot_worker``
============================  =========================================

Every name that used to be importable from here still is: the re-exports
at the bottom of this module are what keeps ``from .._xet_worker import
download_hybrid`` (and the other twelve callers) working. They are
re-exports, not aliases that patch — a caller that wants to intercept
``download_hybrid`` must patch it in the module that *defines* it.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from typing import Any, Dict, List, Optional

from ._xet_worker_common import (
    _deserialize_xet_file_data,
    _handle_worker_exception,
    _init_worker,
    _make_progress_callback,
    _new_xet_session,
    _safe_put,
)
from .subprocess.messages import SubprocessMessage
from .types import ProgressEvent

logger = logging.getLogger(__name__)



# ── Single-File Xet Worker (REAL XetSession API, plan 2026-07-16) ──
#
# This worker runs the PROVEN real-API pattern (from xet_file_only.py,
# verified against hf_xet 1.5.0) INSIDE a child process spawned by
# XetSubprocessRunner. Because hf_xet is imported only in the child, the
# Rust .pyd background thread lives only in the child and can be killed
# via runner.terminate() (SIGTERM -> SIGKILL), freeing its memory. This
# is the safe-isolation pattern the snapshot path already uses.
#
# It does NOT use the broken legacy hf_xet.download_files() path with a
# Python-minted token (that is _download_worker, deprecated).


def _xet_file_only_worker(
    params: Dict[str, Any],
    mp_queue: mp.Queue,
    cancel_event: mp.Event,
) -> None:
    """Worker for single-file Xet download via the real XetSession API.

    Runs in a child process. Imports ``hf_xet`` locally. Emits
    SubprocessMessage events (start / progress / result / error /
    cancelled) back to the main process.

    Args:
        params: Dict with keys:
            file_hash, file_size, dest_path, xet_file_data (dict from
            ``_serialize_xet_file_data``), token, endpoint, transfer_id,
            report_interval, request_headers.
        mp_queue: Queue for sending SubprocessMessage back to main process.
        cancel_event: Event set by main process to signal cancellation.
    """
    _init_worker()
    from .types import (
        EventType,
        ProgressPhase,
        TransferCancelledError,
        TransferDirection,
    )

    transfer_id = params["transfer_id"]
    dest_path = params["dest_path"]
    filename = os.path.basename(dest_path)
    file_size = int(params.get("file_size", 0) or 0)
    report_interval = float(params.get("report_interval", 0.1) or 0.1)

    # Reconstruct XetFileData from serialized dict
    xet_file_data = _deserialize_xet_file_data(params.get("xet_file_data", {}))

    if xet_file_data is None or not getattr(xet_file_data, "refresh_route", None):
        _safe_put(mp_queue, SubprocessMessage.error(
            message=(
                f"File '{filename}' is not stored in Xet storage "
                f"(missing xet_file_data / refresh_route)."
            ),
            error_type="TransferProgressError",
            retryable=False,
        ))
        return

    # ── Build headers (Hub auth) ──────────────────────────────────
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils._xet import xet_headers_without_auth

        headers: dict = {}
        try:
            headers = HfApi(
                endpoint=params.get("endpoint"),
                token=params.get("token"),
            )._build_hf_headers()
        except Exception:
            pass
        xet_headers = xet_headers_without_auth(headers)
    except Exception as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=f"Xet header build failed for '{filename}': {e}",
            error_type=type(e).__name__,
            retryable=False,
        ))
        return

    # ── Progress bookkeeping ──────────────────────────────────────
    state = {
        "bytes_completed": 0,
        "last_emit": 0.0,
        "start_time": time.time(),
    }

    def _emit_start() -> None:
        try:
            mp_queue.put_nowait(SubprocessMessage.event({
                "event_type": EventType.START.value,
                "transfer_id": transfer_id,
                "direction": TransferDirection.DOWNLOAD.value,
                "filename": filename,
                "phase": ProgressPhase.DOWNLOADING.value,
                "total_bytes": file_size,
            }))
        except BaseException:
            pass

    def _maybe_emit_progress() -> None:
        now = time.time()
        if (now - state["last_emit"]) < report_interval:
            return
        state["last_emit"] = now
        bc = state["bytes_completed"]
        total = file_size or 0
        pct = (bc / total * 100.0) if total > 0 else 0.0
        pct = max(0.0, min(100.0, pct))
        elapsed = max(1e-6, now - state["start_time"])
        speed = bc / elapsed
        try:
            mp_queue.put_nowait(SubprocessMessage.event({
                "event_type": EventType.PROGRESS.value,
                "transfer_id": transfer_id,
                "direction": TransferDirection.DOWNLOAD.value,
                "filename": filename,
                "phase": ProgressPhase.DOWNLOADING.value,
                "bytes_completed": bc,
                "total_bytes": total,
                "percentage": pct,
                "speed": speed,
                "transfer_bytes_completed": bc,
                "transfer_bytes_total": total,
                "transfer_speed": speed,
            }))
        except BaseException:
            pass

    def progress_callback(total_update, item_updates) -> None:
        # The installed hf_xet wheel calls progress_callback(total_update,
        # item_updates). For Xet, chunks are buffered and only flushed to
        # disk at the end, so total_bytes_completed (disk bytes) stays 0
        # until completion. total_transfer_bytes_completed (network bytes
        # received) grows continuously and is the right live-progress signal.
        # The exact parameter names (total_update, item_updates) are REQUIRED:
        # the Rust WrappedProgressUpdaterImpl inspects the signature and only
        # uses the detailed (2-arg) mode when both names match.
        if cancel_event.is_set():
            raise TransferCancelledError("Transfer cancelled by user")
        if total_update is None:
            return
        completed = getattr(total_update, "total_transfer_bytes_completed", None)
        if not completed:
            completed = getattr(total_update, "total_bytes_completed", None)
        if completed is None:
            return
        state["bytes_completed"] = int(completed)
        _maybe_emit_progress()

    _emit_start()

    try:
        import hf_xet

        session = _new_xet_session()
        with session.new_file_download_group(
            token_refresh_url=xet_file_data.refresh_route,
            token_refresh_headers=headers,
            custom_headers=xet_headers,
            progress_callback=progress_callback,
        ) as group:
            group.start_download_file(
                hf_xet.XetFileInfo(
                    params["file_hash"],
                    file_size if file_size else None,
                ),
                os.path.abspath(dest_path),
            )

        # Final COMPLETE progress event
        final_bytes = state["bytes_completed"] or file_size or 0
        try:
            mp_queue.put_nowait(SubprocessMessage.event({
                "event_type": EventType.PROGRESS.value,
                "transfer_id": transfer_id,
                "direction": TransferDirection.DOWNLOAD.value,
                "filename": filename,
                "phase": ProgressPhase.DOWNLOADING.value,
                "bytes_completed": final_bytes,
                "total_bytes": file_size or final_bytes,
                "percentage": 100.0,
                "speed": 0,
                "transfer_bytes_completed": final_bytes,
                "transfer_bytes_total": file_size or final_bytes,
                "transfer_speed": 0,
            }))
        except BaseException:
            pass

        _safe_put(mp_queue, SubprocessMessage.result(
            filename=filename,
            destination_path=dest_path,
            file_size=file_size,
            transfer_id=transfer_id,
        ))
    except (KeyboardInterrupt, Exception) as e:  # noqa: BLE001
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


# ── Re-exports: the pre-split public surface of this module ─────────
#
# Every one of these used to be *defined* here. They are re-exported so
# the eleven call sites across `download/`, `upload/` and
# `subprocess/runner.py` — and the tests — keep importing them from
# `hf_track._xet_worker`. Delete this block once the deprecation window
# for the legacy download workers closes and the remaining callers have
# been repointed.

from ._xet_hybrid import download_hybrid as download_hybrid  # noqa: E402,F401
from ._xet_hybrid_runner import HybridRunner as HybridRunner  # noqa: E402,F401
from ._xet_hybrid_runner import TranslatingQueue as TranslatingQueue  # noqa: E402,F401
from ._xet_hybrid_tier1 import (  # noqa: E402,F401
    DEFAULT_FSYNC_INTERVAL as DEFAULT_FSYNC_INTERVAL,
    _open_unbuffered as _open_unbuffered,
    _run_tier1_file as _run_tier1_file,
)
from ._xet_legacy_worker import _download_batch_worker as _download_batch_worker  # noqa: E402,F401
from ._xet_legacy_worker import _download_worker as _download_worker  # noqa: E402,F401
from ._xet_upload_worker import _upload_bytes_worker as _upload_bytes_worker  # noqa: E402,F401
from ._xet_upload_worker import _upload_file_worker as _upload_file_worker  # noqa: E402,F401
from ._xet_worker_common import (  # noqa: E402,F401
    _ProgressThrottler as _ProgressThrottler,
    _serialize_xet_file_data as _serialize_xet_file_data,
)

#: Back-compat alias for the streaming path's entry point, used by
#: ``subprocess/runner.py``. It has always pointed at ``download_hybrid``.
_xet_file_download_worker = download_hybrid
_xet_streaming_download_worker = download_hybrid
