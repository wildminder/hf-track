"""The hybrid streaming download driver (NTH-016, step S32).

``download_hybrid`` is the entry point ``TranslatingQueue``/
``HybridRunner`` drive: it fans a list of ``file_specs`` out over the
Xet tier (via ``_run_tier1_file``) and falls back to plain HTTP for
whatever the Xet tier did not finish. The two tiers share one progress
stream, which is why the driver — not the runner — owns the event shape.

This module moved out of ``_xet_worker.py`` (which re-exports
``download_hybrid``) purely for size; the behaviour is unchanged.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from typing import Any, Dict, List, Optional

from ._xet_hybrid_tier1 import (
    DEFAULT_FSYNC_INTERVAL,
    _open_unbuffered,
    _run_tier1_file,
)
from ._xet_worker_common import _deserialize_xet_file_data
from .subprocess.messages import SubprocessMessage

logger = logging.getLogger(__name__)

#: How long the Xet tier may take for the whole batch before the driver
#: gives up on it and lets the HTTP fallback finish the remainder.
DEFAULT_TIER_TIMEOUT_S: float = 60.0

#: How often the driver polls the progress channel while a tier runs.
DEFAULT_POLL_INTERVAL_S: float = 0.1

#: How often a running tier asks for fresh progress. Kept separate from
#: ``DEFAULT_POLL_INTERVAL_S`` because they are not required to agree: the
#: poll interval is about how responsive ``cancel()`` is, the progress
#: interval about how much work the callback does per wakeup.
DEFAULT_TIER_PROGRESS_INTERVAL_S: float = 0.1


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
    hf_xet_mod = None
    if use_xet:
        try:
            import hf_xet as _hf_xet
        except ImportError as e:
            _diag(f"hf_xet-import-failed: {e}")
            errors.append(f"hf_xet unavailable: {e}")
            session = None
            hf_xet_mod = None
        else:
            # huggingface_hub 1.x removed ``refresh_xet_connection_info``;
            # the download group now fetches and refreshes the CAS token
            # itself from the file's refresh route. Importing the removed
            # name here raised ImportError, and the handler above then
            # cleared ``session`` — disabling Tier 1 for every file in the
            # run rather than for one file.
            session = _hf_xet.XetSession()
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
                # Record the failure and keep going: one unreachable file
                # should not abandon the rest of the snapshot.
                continue
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
                continue
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
