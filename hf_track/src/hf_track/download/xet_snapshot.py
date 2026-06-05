"""Repository snapshot Xet downloads via the isolated subprocess runner.

Groups the legacy ``download_snapshot_with_xet`` and the deprecated
``download_snapshot_with_xet_session`` (which now delegates to the
legacy path) because they are two implementations of the SAME
snapshot operation. Both are intentionally kept together for review.
"""

from __future__ import annotations

import logging
import os
import queue
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

from ..types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferCancelledError,
    TransferDirection,
    TransferError,
    TransferProgressError,
    generate_transfer_id,
)
from ..subprocess import XetSubprocessRunner
from ..token import is_xet_available
from .._xet_worker import _snapshot_worker, _xet_session_snapshot_worker

def download_snapshot_with_xet(
    repo_id: str,
    token: Optional[str],
    event_queue: queue.Queue,
    allow_patterns=None,
    ignore_patterns=None,
    repo_type: str = "model",
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    local_dir: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
    force_download: bool = False,
    use_xet: bool = True,
) -> str:
    """Download a repository snapshot via an isolated subprocess.

    Spawns a child process that calls ``huggingface_hub.snapshot_download()``.
    Inside the child, ``huggingface_hub`` internally decides whether to use
    xet or HTTP for each file. Either way, ``hf_xet`` is loaded only in the
    child process — safe to terminate without affecting the main process.

    Args:
        repo_id: HuggingFace repository ID.
        token: HuggingFace API token.
        event_queue: Queue for ProgressEvent objects.
        allow_patterns: Optional list of glob patterns to include.
        ignore_patterns: Optional list of glob patterns to exclude.
        repo_type: Repository type (model/dataset/space).
        revision: Optional git revision.
        endpoint: Optional custom API endpoint.
        local_dir: Local directory to download files to.
        transfer_id: Optional pre-existing transfer ID.
        report_interval: Event reporting interval (seconds).
        is_cancelled: Optional cancellation hook (checked in main process).
        force_download: Whether to force re-download even if files exist.
        use_xet: If False, the subprocess worker sets
            ``HF_HUB_DISABLE_XET=1`` before importing
            ``huggingface_hub``, forcing HTTP download inside
            the child. This allows runtime xet toggling without
            restarting the app.

    Returns:
        Path to the local directory containing downloaded files.

    Raises:
        ImportError: If hf_xet is not installed.
        TransferCancelledError: If the transfer is cancelled.
        TransferProgressError: If the download fails.
    """
    if not is_xet_available():
        raise ImportError("hf_xet is not installed.")

    transfer_id = transfer_id or generate_transfer_id()

    # Emit START event from main process
    event_queue.put(
        ProgressEvent(
            event_type=EventType.START,
            transfer_id=transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename=repo_id,
            phase=ProgressPhase.DOWNLOADING,
        )
    )

    params = {
        "repo_id": repo_id,
        "token": token,
        "repo_type": repo_type,
        "revision": revision,
        "local_dir": local_dir,
        "allow_patterns": allow_patterns,
        "ignore_patterns": ignore_patterns,
        "endpoint": endpoint,
        "transfer_id": transfer_id,
        "report_interval": report_interval,
        "force_download": force_download,
        "use_xet": use_xet,
    }

    runner = XetSubprocessRunner()
    runner.start(
        worker_func=_snapshot_worker,
        params=params,
        event_queue=event_queue,
    )

    try:
        while True:
            result = runner.wait(timeout=1.0)
            if result is not None:
                break
            if is_cancelled is not None and is_cancelled():
                runner.terminate()
                event_queue.put(
                    ProgressEvent.cancelled_event(
                        transfer_id=transfer_id,
                        direction=TransferDirection.DOWNLOAD,
                        filename=repo_id,
                    )
                )
                raise TransferCancelledError("Snapshot download cancelled by user")

        if result.get("status") == "success":
            return result.get("destination_path", local_dir or repo_id)
        elif (
            result.get("status") == "cancelled"
            or result.get("error_type") == "TransferCancelledError"
            or "cancelled" in result.get("message", "").lower()
            or "interrupted" in result.get("message", "").lower()
        ):
            raise TransferCancelledError(result.get("message", "Snapshot download cancelled by user"))
        else:
            error_msg = result.get("message", "Snapshot download failed")
            raise TransferProgressError(error_msg)

    except KeyboardInterrupt:
        runner.terminate()
        event_queue.put(
            ProgressEvent.cancelled_event(
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=repo_id,
            )
        )
        raise TransferCancelledError("Snapshot download interrupted by user (Ctrl+C)")
    except TransferCancelledError:
        raise
    except TransferProgressError:
        raise
    except Exception as e:
        runner.terminate()
        event_queue.put(
            ProgressEvent(
                event_type=EventType.ERROR,
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=repo_id,
                phase=ProgressPhase.ERROR,
                error=TransferError(message=str(e), error_type=type(e).__name__),
            )
        )
        raise
    finally:
        runner.terminate()



def download_snapshot_with_xet_session(
    repo_id: str,
    token: Optional[str],
    event_queue: queue.Queue,
    allow_patterns=None,
    ignore_patterns=None,
    repo_type: str = "model",
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    local_dir: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> str:
    """Download a repository snapshot — DEPRECATED: delegates to ``download_snapshot_with_xet``.

    .. deprecated::
        This function used to implement a snapshot downloader on top of
        the new ``hf_xet.XetSession`` API. That implementation was found
        to have three critical correctness bugs (2026-06-03):

        1. Nested folder layout (``local_dir/VoxCPM-0.5B/<files>`` instead
           of ``local_dir/<files>``) caused by incorrect path parsing of
           ``HfFileSystem.ls()`` output.
        2. Missing files: only 2 of 13 files were downloaded because
           (a) ``fs.ls()`` is non-recursive by default (misses
           subdirectories like ``assets/``) and (b) files without
           ``xet_hash`` (regular LFS files like ``config.json``,
           ``tokenizer.json``) were explicitly skipped.
        3. Zero-length files on disk after cancel because the new
           API does not pre-cleanup or use atomic tmp-file + rename
           semantics.

        The new ``XetSession`` API is fundamentally a single-session,
        single-batch primitive that does not provide the mixed
        xet/non-xet routing, subdirectory traversal, or cache-aware
        resume that snapshot downloads require. Re-implementing
        ``huggingface_hub.snapshot_download`` from scratch to work
        around these limitations would duplicate hundreds of lines
        of complex, battle-tested code and introduce more bugs.

        This function now delegates to ``download_snapshot_with_xet``,
        which uses ``huggingface_hub.snapshot_download()`` in an
        isolated subprocess (so ``hf_xet`` is still loaded only in
        the child for safe termination). The progress is per-file
        rather than per-chunk, which is the trade-off for correctness.

        See: ``docs/plans/2026-06-03-revert-broken-snapshot-xetsession-path.md``

    Args:
        repo_id: HuggingFace repository ID.
        token: HuggingFace API token.
        event_queue: Queue for ProgressEvent objects.
        allow_patterns: Optional list of glob patterns to include.
        ignore_patterns: Optional list of glob patterns to exclude.
        repo_type: Repository type (model/dataset/space).
        revision: Optional git revision.
        endpoint: Optional custom API endpoint.
        local_dir: Local directory to download files to.
        transfer_id: Optional pre-existing transfer ID.
        report_interval: Event reporting interval (seconds).
        is_cancelled: Optional cancellation hook (checked in main process).

    Returns:
        Path to the local directory containing downloaded files.

    Raises:
        ImportError: If hf_xet is not installed.
        TransferCancelledError: If the transfer is cancelled.
        TransferProgressError: If the download fails.
    """
    # Delegate to the proven path that uses
    # ``huggingface_hub.snapshot_download`` in an isolated subprocess.
    # This correctly handles:
    # - Mixed xet/non-xet files (huggingface_hub routes per-file)
    # - Subdirectories (huggingface_hub.snapshot_download walks them)
    # - Cache-aware resume (etag-based, tmp-file + rename atomic)
    # - Subprocess isolation (xet is loaded only in the child)
    # - Cancellation (terminate the child process)
    return download_snapshot_with_xet(
        repo_id=repo_id,
        token=token,
        event_queue=event_queue,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        repo_type=repo_type,
        revision=revision,
        endpoint=endpoint,
        local_dir=local_dir,
        transfer_id=transfer_id,
        report_interval=report_interval,
        is_cancelled=is_cancelled,
        use_xet=True,
    )

