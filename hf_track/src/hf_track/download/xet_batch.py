"""Batch Xet download via the isolated subprocess runner.

Downloads a list of files in a single subprocess. Kept as its own
module because the input shape (a list of file_specs dicts) and
the output shape (a list of XetDownloadResult) are distinct from
the single-file download.
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
from .._xet_worker import _download_batch_worker, _serialize_xet_file_data
from .xet_file import XetDownloadResult

def download_files_with_xet(
    file_specs: List[dict],
    token: str,
    event_queue: queue.Queue,
    endpoint: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    request_headers: Optional[dict] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> List[XetDownloadResult]:
    """Download multiple files via Xet in an isolated subprocess.

    Args:
        file_specs: List of dicts with keys: dest_path, hash, file_size, xet_file_data.
        token: HuggingFace API token.
        event_queue: Queue for ProgressEvent objects.
        endpoint: Optional custom Xet endpoint.
        transfer_id: Optional pre-existing transfer ID.
        report_interval: Event reporting interval (seconds).
        request_headers: Optional HTTP headers for Xet requests.
        is_cancelled: Optional cancellation hook.

    Returns:
        List of XetDownloadResult on success.

    Raises:
        ImportError: If hf_xet is not installed.
        TransferCancelledError: If the transfer is cancelled.
    """
    if not is_xet_available():
        raise ImportError("hf_xet is not installed.")

    transfer_id = transfer_id or generate_transfer_id()
    total_files = len(file_specs)

    # Emit START events from main process
    for i, spec in enumerate(file_specs):
        filename = os.path.basename(spec["dest_path"])
        event_queue.put(
            ProgressEvent(
                event_type=EventType.START,
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=filename,
                phase=ProgressPhase.DOWNLOADING,
                total_bytes=spec["file_size"],
                file_index=i,
                total_files=total_files,
            )
        )

    # Serialize xet_file_data for each file spec
    serialized_specs = []
    for spec in file_specs:
        serialized_spec = dict(spec)
        serialized_spec["xet_file_data"] = _serialize_xet_file_data(spec.get("xet_file_data"))
        serialized_specs.append(serialized_spec)

    params = {
        "file_specs": serialized_specs,
        "token": token,
        "endpoint": endpoint,
        "transfer_id": transfer_id,
        "report_interval": report_interval,
        "request_headers": request_headers or {},
    }

    runner = XetSubprocessRunner()
    runner.start(
        worker_func=_download_batch_worker,
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
                for i, spec in enumerate(file_specs):
                    filename = os.path.basename(spec["dest_path"])
                    event_queue.put(
                        ProgressEvent.cancelled_event(
                            transfer_id=transfer_id,
                            direction=TransferDirection.DOWNLOAD,
                            filename=filename,
                            file_index=i,
                            total_files=total_files,
                        )
                    )
                raise TransferCancelledError("Download cancelled by user")

        if result.get("status") == "success":
            # Batch worker sends individual results — reconstruct from result
            download_results = []
            for i, spec in enumerate(file_specs):
                filename = os.path.basename(spec["dest_path"])
                download_results.append(
                    XetDownloadResult(
                        success=True,
                        filename=filename,
                        destination_path=spec["dest_path"],
                        file_size=spec["file_size"],
                        transfer_id=transfer_id,
                    )
                )
            return download_results
        elif (
            result.get("status") == "cancelled"
            or result.get("error_type") == "TransferCancelledError"
            or "cancelled" in result.get("message", "").lower()
            or "interrupted" in result.get("message", "").lower()
        ):
            raise TransferCancelledError(result.get("message", "Download cancelled by user"))
        else:
            error_msg = result.get("message", "Batch download failed")
            error_type = result.get("error_type", "Exception")
            # Emit ERROR events for all files
            for i, spec in enumerate(file_specs):
                filename = os.path.basename(spec["dest_path"])
                event_queue.put(
                    ProgressEvent(
                        event_type=EventType.ERROR,
                        transfer_id=transfer_id,
                        direction=TransferDirection.DOWNLOAD,
                        filename=filename,
                        phase=ProgressPhase.ERROR,
                        error=TransferError(message=error_msg, error_type=error_type),
                        file_index=i,
                        total_files=total_files,
                    )
                )
            raise TransferProgressError(error_msg)

    except KeyboardInterrupt:
        runner.terminate()
        for i, spec in enumerate(file_specs):
            filename = os.path.basename(spec["dest_path"])
            event_queue.put(
                ProgressEvent.cancelled_event(
                    transfer_id=transfer_id,
                    direction=TransferDirection.DOWNLOAD,
                    filename=filename,
                    file_index=i,
                    total_files=total_files,
                )
            )
        raise TransferCancelledError("Download interrupted by user (Ctrl+C)")
    except TransferCancelledError:
        raise
    except TransferProgressError:
        raise
    except Exception as e:
        runner.terminate()
        for i, spec in enumerate(file_specs):
            filename = os.path.basename(spec["dest_path"])
            event_queue.put(
                ProgressEvent(
                    event_type=EventType.ERROR,
                    transfer_id=transfer_id,
                    direction=TransferDirection.DOWNLOAD,
                    filename=filename,
                    phase=ProgressPhase.ERROR,
                    error=TransferError(message=str(e), error_type=type(e).__name__),
                    file_index=i,
                    total_files=total_files,
                )
            )
        raise
    finally:
        runner.terminate()
