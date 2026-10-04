"""Dedicated Xet single-file download path using the real hf_xet Session API.

Plan: docs/plans/2026-07-16-xet-download-real-fix.md

Root cause (verified against the INSTALLED hf_xet 1.5.0 wheel + huggingface_hub):

The previous code called ``hf_xet.XetSession()`` / ``new_file_download_group(
endpoint=..., token=..., token_expiry_unix_secs=..., token_refresh_url=...,
token_refresh_headers=...)`` and never passed a ``progress_callback``. It also relied
on ``refresh_xet_connection_info`` (which does NOT exist in the installed
huggingface_hub) to build credentials, so credential refresh crashed.

The PROVEN working pattern (from ``huggingface_hub/file_download.py`` lines 546-574)
is:

    session = get_xet_session()
    with session.new_file_download_group(
        token_refresh_url=xet_file_data.refresh_route,
        token_refresh_headers=headers,
        custom_headers=xet_headers,            # headers WITHOUT authorization
        progress_callback=progress.update_progress,
    ) as group:
        group.start_download_file(
            XetFileInfo(xet_file_data.file_hash, expected_size),
            str(incomplete_path.absolute()),
        )

This module replicates that, but:
  * emits our own ``ProgressEvent`` objects onto ``event_queue`` via the
    ``progress_callback``,
  * supports cancellation via ``abort_xet_session()`` (the huggingface_hub hook)
    which makes ``start_download_file`` raise ``KeyboardInterrupt``,
  * does NOT fall back to HTTP (dedicated xet path).

No HTTP fallback: if xet auth/transfer fails, we raise a clear
``TransferProgressError``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Optional

from ._retry import (  # noqa: F401  (re-exported for the tests)
    DEFAULT_RETRY_BASE_DELAY_S,
    DEFAULT_RETRY_MAX_DELAY_S,
    RETRYABLE_ERRORS,
    _sleep_unless_cancelled,
    retry_with_backoff,
)

logger = logging.getLogger(__name__)

def download_file_xet_only(
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
    probe_timeout_s: float = 600.0,
    max_retries: int = 3,
    retry_base_delay_s: float = DEFAULT_RETRY_BASE_DELAY_S,
    retry_max_delay_s: float = DEFAULT_RETRY_MAX_DELAY_S,
) -> str:
    """Dedicated xet single-file download using the hf_xet Session API.

    Uses the same proven primitives as ``huggingface_hub.file_download``:
    ``get_xet_session()`` + ``new_file_download_group(...)`` +
    ``start_download_file(...)``. Emits ``ProgressEvent`` s and supports
    cancellation.

    Args:
        repo_id, filename, file_hash, file_size, dest_path: file identity.
        token: HuggingFace token (may be None for public repos).
        xet_file_data: the ``XetFileData`` from ``get_hf_file_metadata`` (carries
            ``file_hash`` and ``refresh_route``). Required.
        repo_type, revision, endpoint: hub routing.
        event_queue: optional queue to receive START/PROGRESS/COMPLETE/ERROR events.
        transfer_id: transfer identifier for events.
        report_interval: minimum seconds between PROGRESS events (throttling).
        is_cancelled: optional callable; if it returns True we cancel the transfer
            (via ``abort_xet_session``) and raise ``TransferCancelledError``.
        probe_timeout_s: max seconds for the whole transfer before we give up and
            raise ``TransferProgressError`` (guards against a stuck transfer).

    Returns:
        The destination path on success.

    Raises:
        TransferProgressError: if xet_file_data is missing or the transfer fails.
        TransferCancelledError: if ``is_cancelled()`` becomes True.
    """
    from ..types import (
        EventType,
        ProgressEvent,
        ProgressPhase,
        TransferDirection,
        TransferErrorInfo,
        TransferProgressError,
        TransferCancelledError,
    )

    if is_cancelled is not None and is_cancelled():
        raise TransferCancelledError()

    if xet_file_data is None or not getattr(xet_file_data, "refresh_route", None):
        err = TransferProgressError(
            f"File '{filename}' in '{repo_id}' is not stored in Xet storage "
            f"(missing xet_file_data / refresh_route)."
        )
        _emit_error(event_queue, transfer_id, filename, err)
        raise err

    # ── Build headers (Hub auth) ────────────────────────────────────────
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils._xet import xet_headers_without_auth

        headers: dict = {}
        try:
            headers = HfApi(endpoint=endpoint, token=token)._build_hf_headers()
        except Exception:
            pass
        xet_headers = xet_headers_without_auth(headers)
    except Exception as e:
        err = TransferProgressError(
            f"Xet header build failed for '{filename}' in '{repo_id}': "
            f"{type(e).__name__}: {e}"
        )
        _emit_error(event_queue, transfer_id, filename, err)
        raise err

    # ── Progress bookkeeping ────────────────────────────────────────────
    dest_abs = os.path.abspath(dest_path)
    os.makedirs(os.path.dirname(dest_abs) or ".", exist_ok=True)

    state = {
        "bytes_completed": 0,
        "last_emit": 0.0,
        "start_time": time.time(),
    }

    def progress_callback(total_update, item_updates) -> None:
        # The installed hf_xet wheel calls progress_callback(total_update, item_updates)
        # where total_update is a PyTotalProgressUpdate. For Xet, chunks are buffered
        # and only flushed to disk at the end, so total_bytes_completed (disk bytes)
        # stays 0 until completion. total_transfer_bytes_completed (network bytes
        # received) grows continuously and is the right live-progress signal.
        # The exact parameter names (total_update, item_updates) are REQUIRED: the
        # Rust WrappedProgressUpdaterImpl inspects the signature and only uses the
        # detailed (2-arg) mode when both names match.
        if total_update is None:
            return
        # Prefer network-transfer bytes for live progress; fall back to disk bytes.
        completed = getattr(total_update, "total_transfer_bytes_completed", None)
        if not completed:
            completed = getattr(total_update, "total_bytes_completed", None)
        if completed is None:
            return
        state["bytes_completed"] = int(completed)
        _maybe_emit_progress(
            event_queue, transfer_id, filename, file_size, state, report_interval
        )

    _emit_start(event_queue, transfer_id, filename, file_size)

    # ── Run the download on a worker thread; watchdog handles cancel ─────
    result_holder: dict = {"path": None}
    exc_holder: dict = {"exc": None}

    def _run() -> None:
        try:
            from huggingface_hub.utils._xet import get_xet_session

            def _open_group() -> Any:
                """One attempt at acquiring the session and the group.

                Both live in the retried unit: a session is cheap to
                rebuild, but a half-opened group is not, so the whole
                ``with`` block is re-entered rather than just the session
                lookup.
                """
                session = get_xet_session()
                return session, session.new_file_download_group(
                    token_refresh_url=xet_file_data.refresh_route,
                    token_refresh_headers=headers,
                    custom_headers=xet_headers,
                    progress_callback=progress_callback,
                )

            session, group_handle = retry_with_backoff(
                _open_group,
                max_retries=max_retries,
                base_delay_s=retry_base_delay_s,
                max_delay_s=retry_max_delay_s,
                sleep=lambda d: _sleep_unless_cancelled(d, is_cancelled),
            )
            with group_handle as group:
                group.start_download_file(
                    _XetFileInfo(file_hash, file_size if file_size else None),
                    dest_abs,
                )
            result_holder["path"] = dest_abs
        except KeyboardInterrupt:
            exc_holder["exc"] = "cancelled"
        except Exception as e:  # noqa: BLE001
            exc_holder["exc"] = e

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()

    deadline = time.time() + probe_timeout_s
    while worker.is_alive():
        if is_cancelled is not None and is_cancelled():
            _abort_xet_session()
            worker.join(timeout=10.0)
            raise TransferCancelledError()
        if time.time() > deadline:
            _abort_xet_session()
            worker.join(timeout=10.0)
            err = TransferProgressError(
                f"Xet download for '{filename}' in '{repo_id}' timed out after "
                f"{probe_timeout_s:.0f}s with no completion."
            )
            _emit_error(event_queue, transfer_id, filename, err)
            raise err
        time.sleep(0.05)

    # Worker finished. Check outcome.
    if exc_holder["exc"] == "cancelled":
        raise TransferCancelledError()
    if isinstance(exc_holder["exc"], Exception):
        e = exc_holder["exc"]
        err = TransferProgressError(
            f"Xet download failed for '{filename}' in '{repo_id}': "
            f"{type(e).__name__}: {e}"
        )
        _emit_error(event_queue, transfer_id, filename, err)
        raise err

    # ── Success: emit a final COMPLETE event ────────────────────────────
    final_bytes = state["bytes_completed"] or file_size or 0
    if event_queue is not None:
        try:
            event = ProgressEvent.complete(
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=filename,
                phase=ProgressPhase.COMPLETE,
                bytes_completed=final_bytes,
                total_bytes=file_size or final_bytes,
            )
            event_queue.put_nowait(event)
        except Exception:
            pass

    return result_holder.get("path", dest_abs)


def _XetFileInfo(hash: str, file_size: Optional[int]):
    """Construct hf_xet.XetFileInfo (hash, file_size=None)."""
    import hf_xet

    return hf_xet.XetFileInfo(hash, file_size)


def _abort_xet_session() -> None:
    """Cancel in-flight xet tasks (makes start_download_file raise KeyboardInterrupt)."""
    try:
        from huggingface_hub.utils._xet import abort_xet_session

        abort_xet_session()
    except Exception:
        try:
            import hf_xet

            hf_xet.force_sigint_shutdown()
        except Exception:
            pass


def _emit_start(event_queue: Any, transfer_id: str, filename: str, file_size: int) -> None:
    if event_queue is None:
        return
    try:
        from ..types import (
            EventType,
            ProgressEvent,
            ProgressPhase,
            TransferDirection,
        )

        event = ProgressEvent.start(
            transfer_id=transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename=filename,
            phase=ProgressPhase.DOWNLOADING,
            total_bytes=file_size or 0,
        )
        event_queue.put_nowait(event)
    except Exception:
        pass


def _maybe_emit_progress(
    event_queue: Any,
    transfer_id: str,
    filename: str,
    file_size: int,
    state: dict,
    report_interval: float,
) -> None:
    if event_queue is None:
        return
    now = time.time()
    if (now - state["last_emit"]) < report_interval:
        return
    state["last_emit"] = now
    try:
        from ..types import (
            EventType,
            ProgressEvent,
            ProgressPhase,
            TransferDirection,
        )

        bc = state["bytes_completed"]
        total = file_size or 0
        pct = (bc / total * 100.0) if total > 0 else 0.0
        pct = max(0.0, min(100.0, pct))
        elapsed = max(1e-6, now - state["start_time"])
        speed = bc / elapsed
        event = ProgressEvent(
            event_type=EventType.PROGRESS,
            transfer_id=transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename=filename,
            phase=ProgressPhase.DOWNLOADING,
            bytes_completed=bc,
            total_bytes=total,
            percentage=pct,
            speed=speed,
            transfer_bytes_completed=bc,
            transfer_bytes_total=total,
            transfer_speed=speed,
        )
        event_queue.put_nowait(event)
    except Exception:
        pass


def _emit_error(event_queue: Any, transfer_id: str, filename: str, err: Exception) -> None:
    """Put an ERROR ProgressEvent onto the queue if one was provided."""
    if event_queue is None:
        return
    try:
        from ..types import (
            EventType,
            ProgressEvent,
            ProgressPhase,
            TransferDirection,
            TransferErrorInfo,
        )

        event = ProgressEvent.error_event(
            transfer_id=transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename=filename,
            error=TransferErrorInfo(
                message=str(err),
                error_type="XetUnavailable",
            ),
            phase=ProgressPhase.ERROR,
        )
        event_queue.put_nowait(event)
    except Exception:
        pass


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
