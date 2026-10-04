"""Repository snapshot Xet downloads via the isolated subprocess runner.

The XetSession-API variant was removed in 2026-06-05 (had three
critical correctness bugs documented in
docs/plans/2026-06-03-revert-broken-snapshot-xetsession-path.md).
Only the proven ``_snapshot_worker`` subprocess path remains.
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
    TransferErrorInfo,
    TransferProgressError,
    generate_transfer_id,
)
from ..subprocess import XetSubprocessRunner
from ..token import is_xet_available
from .._xet_worker import _snapshot_worker

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

        status = result.get("status")
        if status == "success":
            return result.get("destination_path", local_dir or repo_id)
        elif status == "cancelled":
            # ``status`` is the only discriminator — see the note in
            # ``download/xet_file.py``.
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
                error=TransferErrorInfo(message=str(e), error_type=type(e).__name__),
            )
        )
        raise
    finally:
        runner.terminate()

