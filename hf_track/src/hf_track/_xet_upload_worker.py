"""The Xet upload workers (NTH-016, step S32).

``_upload_file_worker`` and ``_upload_bytes_worker`` share their shape
exactly — same event stream, same ``new_upload_commit`` call, same
terminal handling — and differ only in where the bytes come from. They
were moved together out of ``_xet_worker.py``, which re-exports both
names, so every existing ``from .._xet_worker import _upload_file_worker``
keeps working.

Both open their commit through ``_new_xet_session`` and pass the repo's
token-refresh URL rather than a Python-minted token: since
``huggingface_hub`` 1.x the runtime owns that exchange.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from typing import Any, Dict

from ._xet_worker_common import (
    _handle_worker_exception,
    _init_worker,
    _make_progress_callback,
    _new_xet_session,
    _safe_put,
)
from .subprocess.messages import SubprocessMessage

logger = logging.getLogger(__name__)



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
            xet_connection_info_refresh_url,
            xet_headers_without_auth,
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

    # Get upload credentials: since huggingface_hub 1.x the write token is
    # not minted in Python — the runtime refreshes it from the repo's
    # token-refresh URL, which is all the commit needs.
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=params["token"], endpoint=params.get("endpoint"))
        headers = api._build_hf_headers()
        session = _new_xet_session()
        commit = session.new_upload_commit(
            token_refresh_url=xet_connection_info_refresh_url(
                token_type=XetTokenType.WRITE,
                repo_id=params["repo_id"],
                repo_type=params.get("repo_type", "model"),
                revision=params.get("revision"),
                endpoint=params.get("endpoint"),
            ),
            token_refresh_headers=headers,
            custom_headers=xet_headers_without_auth(headers),
            progress_callback=callback,
        )
    except Exception as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=f"Failed to get upload credentials: {e}",
            error_type=type(e).__name__,
        ))
        return

    try:
        with commit:
            upload = commit.start_upload_file(file_path)

        metadata = upload.result()
        _safe_put(mp_queue, SubprocessMessage.result(
            filename=filename,
            file_size=file_size,
            transfer_id=transfer_id,
            hash=getattr(getattr(metadata, "xet_info", None), "hash", ""),
            url=getattr(getattr(metadata, "xet_info", None), "url", None),
        ))

    except (KeyboardInterrupt, Exception) as e:
        _handle_worker_exception(mp_queue, e)


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
            xet_connection_info_refresh_url,
            xet_headers_without_auth,
        )
    except ImportError as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=str(e), error_type="ImportError", retryable=False,
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

    # Get upload credentials: see ``_upload_file_worker`` — the runtime
    # refreshes the write token from the repo token-refresh URL.
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=params["token"], endpoint=params.get("endpoint"))
        headers = api._build_hf_headers()
        session = _new_xet_session()
        commit = session.new_upload_commit(
            token_refresh_url=xet_connection_info_refresh_url(
                token_type=XetTokenType.WRITE,
                repo_id=params["repo_id"],
                repo_type=params.get("repo_type", "model"),
                revision=params.get("revision"),
                endpoint=params.get("endpoint"),
            ),
            token_refresh_headers=headers,
            custom_headers=xet_headers_without_auth(headers),
            progress_callback=callback,
        )
    except Exception as e:
        _safe_put(mp_queue, SubprocessMessage.error(
            message=f"Failed to get upload credentials: {e}",
            error_type=type(e).__name__,
        ))
        return

    try:
        with commit:
            if file_content is not None:
                upload = commit.start_upload_bytes(file_content)
            else:
                upload = commit.start_upload_file(file_path)

        metadata = upload.result()
        _safe_put(mp_queue, SubprocessMessage.result(
            filename=filename,
            file_size=file_size,
            transfer_id=transfer_id,
            hash=getattr(getattr(metadata, "xet_info", None), "hash", ""),
            url=getattr(getattr(metadata, "xet_info", None), "url", None),
        ))

    except (KeyboardInterrupt, Exception) as e:
        _handle_worker_exception(mp_queue, e)
