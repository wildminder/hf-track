"""Xet file upload via the isolated subprocess runner.

Groups:
  * :class:`XetUploadResult` -- the upload envelope (success/filename/
    hash/file_size/transfer_id/url).
  * :func:`_run_upload_in_subprocess` -- the shared polling/cancellation
    loop used by both file and bytes Xet uploads. Kept here (not in a
    shared module) because the entire Xet file-upload path lives in
    one cohesive concept: a single file going up via hf_xet in a
    subprocess.
  * :func:`upload_file_with_xet` -- public entry point that uploads a
    file on disk.
"""

from __future__ import annotations

import os
import queue
from dataclasses import dataclass
from typing import Callable, Optional

from ..subprocess import XetSubprocessRunner
from ..token import is_xet_available
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
from .._xet_worker import _upload_file_worker


@dataclass
class XetUploadResult:
    success: bool
    filename: str
    hash: str = ""
    file_size: int = 0
    transfer_id: str = ""
    url: Optional[str] = None


def _run_upload_in_subprocess(
    *,
    filename: str,
    file_size: int,
    repo_id: str,
    token: str,
    event_queue: queue.Queue,
    repo_type: str = "model",
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
    worker_func: Callable,
    params: dict,
) -> XetUploadResult:
    """Shared subprocess upload logic for both file and bytes uploads.

    Args:
        filename: Name of the file being uploaded.
        file_size: Size of the file in bytes.
        repo_id: Target repository ID.
        token: HuggingFace API token.
        event_queue: Queue for progress events.
        repo_type: Repository type.
        revision: Optional git revision.
        endpoint: Optional custom endpoint.
        transfer_id: Optional pre-existing transfer ID.
        report_interval: Event reporting interval.
        is_cancelled: Optional cancellation hook.
        worker_func: The worker function to run in subprocess.
        params: Parameters dict for the worker.

    Returns:
        XetUploadResult on success.

    Raises:
        TransferCancelledError: If the transfer is cancelled.
        TransferProgressError: If the upload fails.
    """
    transfer_id = transfer_id or generate_transfer_id()

    # Emit START event from main process
    event_queue.put(
        ProgressEvent(
            event_type=EventType.START,
            transfer_id=transfer_id,
            direction=TransferDirection.UPLOAD,
            filename=filename,
            phase=ProgressPhase.UPLOADING,
            total_bytes=file_size,
        )
    )

    runner = XetSubprocessRunner()
    runner.start(
        worker_func=worker_func,
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
                        direction=TransferDirection.UPLOAD,
                        filename=filename,
                    )
                )
                raise TransferCancelledError("Upload cancelled by user")

        status = result.get("status")
        if status == "success":
            return XetUploadResult(
                success=True,
                filename=result.get("filename", filename),
                hash=result.get("hash", ""),
                file_size=result.get("file_size", file_size),
                transfer_id=transfer_id,
                url=result.get("url"),
            )
        elif status == "cancelled":
            # ``status`` is the only discriminator — see the note in
            # ``download/xet_file.py``.
            raise TransferCancelledError(result.get("message", "Upload cancelled by user"))
        else:
            error_msg = result.get("message", "Upload failed")
        raise TransferProgressError(error_msg)

    except KeyboardInterrupt:
        runner.terminate()
        event_queue.put(
            ProgressEvent.cancelled_event(
                transfer_id=transfer_id,
                direction=TransferDirection.UPLOAD,
                filename=filename,
            )
        )
        raise TransferCancelledError("Upload interrupted by user (Ctrl+C)")
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
                direction=TransferDirection.UPLOAD,
                filename=filename,
                phase=ProgressPhase.ERROR,
                error=TransferErrorInfo(message=str(e), error_type=type(e).__name__),
            )
        )
        raise
    finally:
        runner.terminate()


def upload_file_with_xet(
    file_path: str,
    repo_id: str,
    token: str,
    event_queue: queue.Queue,
    repo_type: str = "model",
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> XetUploadResult:
    """Upload a file via Xet in an isolated subprocess.

    Args:
        file_path: Local path to the file to upload.
        repo_id: Target repository ID.
        token: HuggingFace API token.
        event_queue: Queue for progress events.
        repo_type: Repository type.
        revision: Optional git revision.
        endpoint: Optional custom endpoint.
        transfer_id: Optional pre-existing transfer ID.
        report_interval: Event reporting interval.
        is_cancelled: Optional cancellation hook.

    Returns:
        XetUploadResult on success.

    Raises:
        ImportError: If hf_xet is not installed.
    """
    if not is_xet_available():
        raise ImportError("hf_xet is not installed.")

    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    params = {
        "file_path": file_path,
        "repo_id": repo_id,
        "token": token,
        "repo_type": repo_type,
        "revision": revision,
        "endpoint": endpoint,
        "transfer_id": transfer_id,
        "report_interval": report_interval,
        "filename": filename,
        "direction": "upload",
    }

    return _run_upload_in_subprocess(
        filename=filename,
        file_size=file_size,
        repo_id=repo_id,
        token=token,
        event_queue=event_queue,
        repo_type=repo_type,
        revision=revision,
        endpoint=endpoint,
        transfer_id=transfer_id,
        report_interval=report_interval,
        is_cancelled=is_cancelled,
        worker_func=_upload_file_worker,
        params=params,
    )
