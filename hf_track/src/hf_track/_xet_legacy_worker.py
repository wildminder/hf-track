"""The deprecated Xet download workers (NTH-016, step S32).

.. deprecated:: 2026-07-09
    ``_download_worker`` and ``_download_batch_worker`` mint the CAS token
    in Python and hand it to the legacy ``hf_xet.download_files()`` API,
    which hangs indefinitely in some environments. Use
    ``download_file_xet_only`` / ``_xet_file_only_worker``, or pass
    ``use_xet=False`` to ``HfTracker.download_file`` for the HTTP path.

These two moved here from ``_xet_worker.py`` — which re-exports both
names, so every existing ``from .._xet_worker import _download_worker``
keeps working through the deprecation window.

The module is kept alive deliberately: the entry points are still
reachable, still picklable by ``spawn`` (so they must stay at module top
level), and removing them is a separate, breaking change.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from typing import Any, Dict, List

from ._xet_worker_common import (
    _deserialize_xet_file_data,
    _handle_worker_exception,
    _init_worker,
    _make_progress_callback,
    _new_xet_session,
    _safe_put,
)
from .subprocess.messages import SubprocessMessage

logger = logging.getLogger(__name__)



# ── Download Worker ───────────────────────────────────────────────

def _download_worker(params: Dict[str, Any], mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Worker function for single-file Xet downloads.

    Runs in a child process. Imports ``hf_xet`` locally.

    .. deprecated:: 2026-07-09
        This worker used the legacy ``hf_xet.download_files()`` API
        which hangs indefinitely in some environments; it now drives the
        same ``XetSession`` download group as ``_xet_file_only_worker``.
        The hybrid approach (``download_file_with_xet_hybrid`` /
        ``HybridRunner`` + ``download_hybrid``) remains the preferred
        replacement. See
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
        from huggingface_hub.utils._xet import xet_headers_without_auth
    except ImportError as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=str(e), error_type="ImportError", retryable=False,
        ))
        return

    # Reconstruct XetFileData from serialized dict
    xet_file_data = _deserialize_xet_file_data(params.get("xet_file_data", {}))
    headers = params.get("request_headers", {})

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

    # Get credentials: since huggingface_hub 1.x the runtime fetches and
    # refreshes the CAS token itself from the file's refresh route.
    try:
        session = _new_xet_session()
        group = session.new_file_download_group(
            token_refresh_url=xet_file_data.refresh_route,
            token_refresh_headers=headers,
            custom_headers=xet_headers_without_auth(headers),
            progress_callback=callback,
        )
    except Exception as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=f"Failed to get download credentials: {e}",
            error_type=type(e).__name__,
        ))
        return

    try:
        with group:
            group.start_download_file(
                hf_xet.XetFileInfo(
                    params["file_hash"],
                    file_size if file_size else None,
                ),
                os.path.abspath(params["dest_path"]),
            )

        _safe_put(mp_queue, SubprocessMessage.result(
            filename=filename,
            destination_path=params["dest_path"],
            file_size=file_size,
            transfer_id=transfer_id,
        ))

    except (KeyboardInterrupt, Exception) as e:
        _handle_worker_exception(mp_queue, e)


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
        from huggingface_hub.utils._xet import xet_headers_without_auth
    except ImportError as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=str(e), error_type="ImportError", retryable=False,
        ))
        return

    headers = params.get("request_headers", {})
    session = _new_xet_session()

    # One download group per file, each with its own progress callback:
    # this is the per-file shape the hybrid worker uses (see
    # ``_run_tier1_file``), so each callback reports its own bytes.
    for i, spec in enumerate(file_specs):
        filename = os.path.basename(spec["dest_path"])
        file_size = int(spec["file_size"])
        xet_file_data = _deserialize_xet_file_data(spec.get("xet_file_data", {}))
        callback = _make_progress_callback(
            filename=filename,
            total_bytes=file_size,
            transfer_id=transfer_id,
            mp_queue=mp_queue,
            cancel_event=cancel_event,
            direction="download",
            file_index=i,
            total_files=total_files,
            report_interval=params.get("report_interval", 0.1),
        )

        try:
            with session.new_file_download_group(
                token_refresh_url=getattr(xet_file_data, "refresh_route", None),
                token_refresh_headers=headers,
                custom_headers=xet_headers_without_auth(headers),
                progress_callback=callback,
            ) as group:
                group.start_download_file(
                    hf_xet.XetFileInfo(spec["hash"], file_size or None),
                    os.path.abspath(spec["dest_path"]),
                )
        except (KeyboardInterrupt, Exception) as e:
            _handle_worker_exception(mp_queue, e)
            return

        # Send individual result messages for each file
        _safe_put(mp_queue, SubprocessMessage.result(
            filename=filename,
            destination_path=spec["dest_path"],
            file_size=file_size,
            transfer_id=transfer_id,
            file_index=i,
            total_files=total_files,
        ))
