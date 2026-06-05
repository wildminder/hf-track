"""Single-file Xet downloads via the isolated subprocess runner.

Groups both legacy and new-API single-file download functions
because they share the same `XetDownloadResult` envelope and the
same parameter shape (file_hash, file_size, dest_path, xet_file_data).
Splitting them would force callers to import two modules for what
is logically one operation.
"""

from __future__ import annotations

import logging
import os
import queue
from dataclasses import dataclass
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
from .._xet_worker import _download_worker, _serialize_xet_file_data, _xet_session_download_worker

@dataclass
class XetDownloadResult:
    success: bool
    filename: str
    destination_path: str = ""
    file_size: int = 0
    transfer_id: str = ""


def download_file_with_xet(
    file_hash: str,
    file_size: int,
    dest_path: str,
    xet_file_data,
    token: str,
    event_queue: queue.Queue,
    endpoint: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    request_headers: Optional[dict] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> XetDownloadResult:
    """Download a single file via Xet in an isolated subprocess.

    The actual ``hf_xet.download_files()`` call runs in a child process,
    which can be safely terminated via ``runner.terminate()`` if the
    transfer hangs or the user cancels.

    Args:
        file_hash: Xet file hash.
        file_size: Expected file size in bytes.
        dest_path: Local destination path.
        xet_file_data: XetFileData object from HfFileMetadata.
        token: HuggingFace API token.
        event_queue: Queue for ProgressEvent objects.
        endpoint: Optional custom Xet endpoint.
        transfer_id: Optional pre-existing transfer ID.
        report_interval: Event reporting interval (seconds).
        request_headers: Optional HTTP headers for Xet requests.
        is_cancelled: Optional cancellation hook (checked in main process).

    Returns:
        XetDownloadResult on success.

    Raises:
        ImportError: If hf_xet is not installed.
        TransferCancelledError: If the transfer is cancelled.
        TransferError: If the download fails.
    """
    if not is_xet_available():
        raise ImportError("hf_xet is not installed.")

    transfer_id = transfer_id or generate_transfer_id()
    filename = os.path.basename(dest_path)

    # Emit START event from main process
    event_queue.put(
        ProgressEvent(
            event_type=EventType.START,
            transfer_id=transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename=filename,
            phase=ProgressPhase.DOWNLOADING,
            total_bytes=file_size,
        )
    )

    # Serialize xet_file_data for cross-process boundary
    xet_file_data_dict = _serialize_xet_file_data(xet_file_data)

    params = {
        "file_hash": file_hash,
        "file_size": file_size,
        "dest_path": dest_path,
        "xet_file_data": xet_file_data_dict,
        "token": token,
        "endpoint": endpoint,
        "transfer_id": transfer_id,
        "report_interval": report_interval,
        "request_headers": request_headers or {},
        "direction": "download",
        "filename": filename,
    }

    runner = XetSubprocessRunner()
    runner.start(
        worker_func=_download_worker,
        params=params,
        event_queue=event_queue,
    )

    try:
        # Poll for cancellation from main process while waiting
        while True:
            result = runner.wait(timeout=1.0)
            if result is not None:
                break
            # Check main-process cancellation hook
            if is_cancelled is not None and is_cancelled():
                runner.terminate()
                event_queue.put(
                    ProgressEvent.cancelled_event(
                        transfer_id=transfer_id,
                        direction=TransferDirection.DOWNLOAD,
                        filename=filename,
                    )
                )
                raise TransferCancelledError("Download cancelled by user")

        # Process the result
        if result.get("status") == "success":
            return XetDownloadResult(
                success=True,
                filename=result.get("filename", filename),
                destination_path=result.get("destination_path", dest_path),
                file_size=result.get("file_size", file_size),
                transfer_id=transfer_id,
            )
        elif (
            result.get("status") == "cancelled"
            or result.get("error_type") == "TransferCancelledError"
            or "cancelled" in result.get("message", "").lower()
            or "interrupted" in result.get("message", "").lower()
        ):
            raise TransferCancelledError(result.get("message", "Download cancelled by user"))
        else:
            # Error result from worker
            error_msg = result.get("message", "Download failed")
            raise TransferProgressError(error_msg)

    except KeyboardInterrupt:
        runner.terminate()
        event_queue.put(
            ProgressEvent.cancelled_event(
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=filename,
            )
        )
        raise TransferCancelledError("Download interrupted by user (Ctrl+C)")
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
                filename=filename,
                phase=ProgressPhase.ERROR,
                error=TransferError(message=str(e), error_type=type(e).__name__),
            )
        )
        raise
    finally:
        runner.terminate()


def download_file_with_xet_session(
    file_hash: str,
    file_size: int,
    dest_path: str,
    xet_file_data,
    token: str,
    event_queue: queue.Queue,
    endpoint: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    request_headers: Optional[dict] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> XetDownloadResult:
    """Download a single file via Xet using the NEW ``hf_xet.XetSession`` API.

    Replaces ``download_file_with_xet`` when ``hf_xet>=1.0`` is installed and
    ``XetSession`` is available. The old ``download_files()`` API in
    ``hf_xet>=1.0`` only accepts a 1-arg ``progress_updater`` callback that
    fires at file completion, which produces a 0% → 100% jump. The new
    ``XetSession`` API has a 2-arg ``progress_callback`` firing every
    ``progress_interval_ms`` with per-chunk progress.

    Args:
        file_hash: Xet file hash.
        file_size: Expected file size in bytes.
        dest_path: Local destination path.
        xet_file_data: XetFileData object from HfFileMetadata.
        token: HuggingFace API token.
        event_queue: Queue for ProgressEvent objects.
        endpoint: Optional custom Xet endpoint.
        transfer_id: Optional pre-existing transfer ID.
        report_interval: Event reporting interval (seconds).
        request_headers: Optional HTTP headers for Xet requests.
        is_cancelled: Optional cancellation hook (checked in main process).

    Returns:
        XetDownloadResult on success.

    Raises:
        ImportError: If hf_xet or XetSession is not available.
        TransferCancelledError: If the transfer is cancelled.
        TransferError: If the download fails.
    """
    if not is_xet_available():
        raise ImportError("hf_xet is not installed.")

    try:
        import hf_xet
        if not hasattr(hf_xet, "XetSession"):
            raise ImportError("hf_xet.XetSession not available (requires hf_xet>=1.5.0)")
    except ImportError:
        raise ImportError("hf_xet.XetSession not available")

    transfer_id = transfer_id or generate_transfer_id()
    filename = os.path.basename(dest_path)

    # Emit START event from main process
    event_queue.put(
        ProgressEvent(
            event_type=EventType.START,
            transfer_id=transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename=filename,
            phase=ProgressPhase.DOWNLOADING,
            total_bytes=file_size,
        )
    )

    xet_file_data_dict = _serialize_xet_file_data(xet_file_data)

    params = {
        "file_hash": file_hash,
        "file_size": file_size,
        "dest_path": dest_path,
        "xet_file_data": xet_file_data_dict,
        "token": token,
        "endpoint": endpoint,
        "transfer_id": transfer_id,
        "report_interval": report_interval,
        "request_headers": request_headers or {},
        "filename": filename,
        "direction": "download",
    }

    runner = XetSubprocessRunner()
    runner.start(
        worker_func=_xet_session_download_worker,
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
                        filename=filename,
                    )
                )
                raise TransferCancelledError("Download cancelled by user")

        if result.get("status") == "success":
            return XetDownloadResult(
                success=True,
                filename=result.get("filename", filename),
                destination_path=result.get("destination_path", dest_path),
                file_size=result.get("file_size", file_size),
                transfer_id=transfer_id,
            )
        elif (
            result.get("status") == "cancelled"
            or result.get("error_type") == "TransferCancelledError"
            or "cancelled" in result.get("message", "").lower()
            or "interrupted" in result.get("message", "").lower()
        ):
            raise TransferCancelledError(result.get("message", "Download cancelled by user"))
        else:
            error_msg = result.get("message", "Download failed")
            raise TransferProgressError(error_msg)

    except KeyboardInterrupt:
        runner.terminate()
        event_queue.put(
            ProgressEvent.cancelled_event(
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=filename,
            )
        )
        raise TransferCancelledError("Download interrupted by user (Ctrl+C)")
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
                filename=filename,
                phase=ProgressPhase.ERROR,
                error=TransferError(message=str(e), error_type=type(e).__name__),
            )
        )
        raise
    finally:
        runner.terminate()
