"""Terminable-subprocess path for a single-file Xet download.

Extracted from ``xet_file_only.py``: the in-process Xet download and the
subprocess-isolated one are separate responsibilities, and keeping both in
one module pushed it past the soft size ceiling in
``tests/download/test_download_layout.py``.

``hf_xet`` is imported only in the child, so the Rust background thread
lives only in the child and can be killed with ``runner.terminate()``
(SIGTERM -> SIGKILL), freeing its memory. The progress helpers are shared
with the in-process path and live in ``xet_file_only``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Optional

from ._retry import (
    DEFAULT_RETRY_BASE_DELAY_S,
    DEFAULT_RETRY_MAX_DELAY_S,
    RETRYABLE_ERRORS,
    _sleep_unless_cancelled,
    retry_with_backoff,
)
from .xet_file_only import _emit_error

logger = logging.getLogger(__name__)


def download_file_xet_subprocess(
    *,
    repo_id: str,
    filename: str,
    file_hash: str,
    file_size: int,
    dest_path: str,
    token: Optional[str],
    xet_file_data: Any = None,
    repo_type: str = "model",
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    event_queue: Any = None,
    transfer_id: str = "",
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
    on_spawn: Optional[Callable[[object], None]] = None,
    on_finish: Optional[Callable[[], None]] = None,
    probe_timeout_s: float = 600.0,
    max_retries: int = 3,
    retry_base_delay_s: float = DEFAULT_RETRY_BASE_DELAY_S,
    retry_max_delay_s: float = DEFAULT_RETRY_MAX_DELAY_S,
) -> str:
    """Dedicated xet single-file download run in a TERMINABLE subprocess.

    Plan: docs/plans/2026-07-16-xet-single-file-subprocess-isolation.md

    Spawns ``_xet_file_only_worker`` (which uses the proven real XetSession
    API) inside a child process via ``XetSubprocessRunner``. Because ``hf_xet``
    is imported only in the child, the Rust ``.pyd`` background thread lives
    only in the child and can be killed via ``runner.terminate()``
    (SIGTERM -> SIGKILL), freeing its memory. This is the safe-isolation
    pattern the snapshot path already uses.

    Progress events (START / PROGRESS / COMPLETE / ERROR / CANCELLED) flow
    from the child to ``event_queue`` via the runner's relay thread.

    Args:
        repo_id, filename, file_hash, file_size, dest_path: file identity.
        token: HuggingFace token (may be None for public repos).
        xet_file_data: the ``XetFileData`` from ``get_hf_file_metadata``.
            Required (carries ``file_hash`` and ``refresh_route``).
        repo_type, revision, endpoint: hub routing.
        event_queue: queue to receive ProgressEvents.
        transfer_id: transfer identifier for events.
        report_interval: minimum seconds between PROGRESS events.
        is_cancelled: optional callable; if True we request cancel.
        on_spawn: optional hook called with the runner (for fast cancel).
        on_finish: optional cleanup hook called after the subprocess exits.
        probe_timeout_s: max seconds to wait for the subprocess.

    Returns:
        The destination path on success.

    Raises:
        TransferProgressError: if xet_file_data is missing or the transfer fails.
        TransferCancelledError: if the transfer is cancelled.
    """
    from .._xet_worker import _serialize_xet_file_data, _xet_file_only_worker
    from ..subprocess import XetSubprocessRunner
    from ..types import TransferCancelledError, TransferProgressError

    if xet_file_data is None or not getattr(xet_file_data, "refresh_route", None):
        err = TransferProgressError(
            f"File '{filename}' in '{repo_id}' is not stored in Xet storage "
            f"(missing xet_file_data / refresh_route)."
        )
        _emit_error(event_queue, transfer_id, filename, err)
        raise err

    runner = XetSubprocessRunner()
    if on_spawn is not None:
        try:
            on_spawn(runner)
        except Exception:
            pass  # on_spawn is a best-effort hook

    params = {
        "file_hash": file_hash,
        "file_size": file_size,
        "dest_path": dest_path,
        "xet_file_data": _serialize_xet_file_data(xet_file_data),
        "token": token,
        "endpoint": endpoint,
        "transfer_id": transfer_id,
        "report_interval": report_interval,
        "request_headers": {},
    }

    try:
        runner.start(
            worker_func=_xet_file_only_worker,
            params=params,
            event_queue=event_queue,
        )

        # Cooperative cancel: poll is_cancelled and forward to the runner.
        # The runner also hard-terminates the process on terminate().
        cancelled_flag = {"value": False}

        def _watchdog() -> None:
            while runner.is_alive():
                if is_cancelled is not None and is_cancelled():
                    cancelled_flag["value"] = True
                    runner.request_cancel()
                    runner.terminate(grace=2.0)
                    break
                time.sleep(0.05)

        watchdog = threading.Thread(target=_watchdog, daemon=True)
        watchdog.start()

        result = runner.wait(timeout=probe_timeout_s)
        if result is None:
            # The child exited without a terminal message. Distinguish a
            # user-initiated cancel (watchdog terminated the process) from a
            # genuine stall/timeout.
            if cancelled_flag["value"] or (is_cancelled is not None and is_cancelled()):
                raise TransferCancelledError()
            runner.terminate(grace=2.0)
            err = TransferProgressError(
                f"Xet subprocess download for '{filename}' in '{repo_id}' "
                f"timed out after {probe_timeout_s:.0f}s."
            )
            _emit_error(event_queue, transfer_id, filename, err)
            raise err

        status = result.get("status")
        if status == "cancelled":
            raise TransferCancelledError()
        if status != "success":
            err = TransferProgressError(
                result.get("message", "Xet subprocess download failed")
            )
            _emit_error(event_queue, transfer_id, filename, err)
            raise err

        return result.get("destination_path", dest_path)
    finally:
        runner.terminate()
        if on_finish is not None:
            try:
                on_finish()
            except Exception:
                pass  # on_finish is a best-effort cleanup hook
