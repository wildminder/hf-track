"""Xet direct download functions with progress tracking.

Bypasses ``huggingface_hub``'s download functions to call
``hf_xet.download_files()`` directly, passing per-file progress
callbacks with the **detailed** signature ``(total_update, item_updates)``.

This provides the highest quality download progress data available:
- Byte-level progress with deduplication awareness
- Per-file progress via ``PyItemProgressUpdate``
- Transfer speed (both processing and network)
- Dedup savings (bytes_completed - transfer_bytes_completed)

**Critical**: The ``progress_updater`` callback parameter names MUST be
``total_update`` and ``item_updates`` for the Rust runtime's
``WrappedProgressUpdaterImpl`` to detect the detailed callback signature.
If names don't match, the Rust runtime falls back to simple ``(int)``
mode and all detailed progress data is lost.

The ``progress_updater`` parameter for downloads is a ``List[Py<PyAny>]``
(one callback per file), not a single callback like uploads.
"""

from __future__ import annotations

import os
import queue
from dataclasses import dataclass
from typing import List, Optional

from .callbacks import XetDownloadProgressCallback
from .token import XetTokenManager, is_xet_available
from .types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferDirection,
    generate_transfer_id,
)


@dataclass
class XetDownloadResult:
    """Result of a Xet download operation.

    Attributes:
        success: Whether the download completed successfully.
        filename: Name of the downloaded file.
        destination_path: Local path where the file was saved.
        file_size: Size of the downloaded file in bytes.
        transfer_id: Unique identifier for the transfer.
    """

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
) -> XetDownloadResult:
    """Download a file using hf_xet with detailed progress tracking.

    Calls ``hf_xet.download_files()`` directly with a per-file
    progress callback using the **detailed** signature
    ``(total_update, item_updates)``. This provides speed, dedup
    info, and per-item progress from the Rust runtime.

    Args:
        file_hash: Content hash of the file to download.
        file_size: Expected file size in bytes.
        dest_path: Local path to save the downloaded file.
        xet_file_data: ``XetFileData`` instance for token acquisition.
        token: HuggingFace API token.
        event_queue: Queue for emitting ProgressEvent objects.
        endpoint: Optional custom HuggingFace API endpoint.
        transfer_id: Unique transfer identifier (auto-generated if None).
        report_interval: Minimum seconds between progress events.
        request_headers: Optional dict of HTTP headers to pass to
            ``hf_xet.download_files()`` (e.g. user-agent). Auth
            headers should be stripped before passing.

    Returns:
        XetDownloadResult with download status and metadata.

    Raises:
        ImportError: If ``hf_xet`` is not installed.
    """
    if not is_xet_available():
        raise ImportError(
            "hf_xet is not installed. Install with: "
            'pip install "huggingface_hub[hf_xet]"'
        )

    import hf_xet  # noqa: F811

    transfer_id = transfer_id or generate_transfer_id()
    filename = os.path.basename(dest_path)

    # Get credentials
    token_manager = XetTokenManager(token, endpoint)
    creds = token_manager.get_download_credentials(xet_file_data)

    # Create download info
    download_info = [
        hf_xet.PyXetDownloadInfo(
            destination_path=str(os.path.abspath(dest_path)),
            hash=file_hash,
            file_size=file_size,
        )
    ]

    # Create per-file progress callback
    callback = XetDownloadProgressCallback(
        filename=filename,
        total_bytes=file_size,
        event_queue=event_queue,
        report_interval=report_interval,
        transfer_id=transfer_id,
    )

    # Emit start event
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

    try:
        kwargs = dict(
            download_info=download_info,
            endpoint=creds.endpoint,
            token_info=creds.token_info,
            token_refresher=creds.token_refresher,
            progress_updater=[callback],  # List of per-file callbacks
        )
        if request_headers:
            kwargs["request_headers"] = request_headers

        results = hf_xet.download_files(**kwargs)

        # Emit complete event
        event_queue.put(
            ProgressEvent(
                event_type=EventType.COMPLETE,
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=filename,
                phase=ProgressPhase.COMPLETE,
                bytes_completed=file_size,
                total_bytes=file_size,
                percentage=100.0,
            )
        )

        return XetDownloadResult(
            success=True,
            filename=filename,
            destination_path=dest_path,
            file_size=file_size,
            transfer_id=transfer_id,
        )

    except Exception as e:
        event_queue.put(
            ProgressEvent(
                event_type=EventType.ERROR,
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=filename,
                phase=ProgressPhase.ERROR,
                error=str(e),
            )
        )
        raise


def download_files_with_xet(
    file_specs: List[dict],
    token: str,
    event_queue: queue.Queue,
    endpoint: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    request_headers: Optional[dict] = None,
) -> List[XetDownloadResult]:
    """Download multiple files using hf_xet with detailed per-file progress.

    Each file gets its own progress callback in the ``progress_updater``
    list, matching the ``hf_xet.download_files()`` API. All callbacks
    use the **detailed** signature ``(total_update, item_updates)``.

    Args:
        file_specs: List of dicts, each with keys:
        - ``hash`` (str): Content hash of the file.
        - ``file_size`` (int): Expected file size in bytes.
        - ``dest_path`` (str): Local path to save the file.
        - ``xet_file_data``: XetFileData for token acquisition.
        token: HuggingFace API token.
        event_queue: Queue for emitting ProgressEvent objects.
        endpoint: Optional custom HuggingFace API endpoint.
        transfer_id: Unique transfer identifier (auto-generated if None).
        report_interval: Minimum seconds between progress events.
        request_headers: Optional dict of HTTP headers to pass to
            ``hf_xet.download_files()`` (e.g. user-agent). Auth
            headers should be stripped before passing.

    Returns:
        List of XetDownloadResult objects.

    Raises:
        ImportError: If ``hf_xet`` is not installed.
    """
    if not is_xet_available():
        raise ImportError(
            "hf_xet is not installed. Install with: "
            'pip install "huggingface_hub[hf_xet]"'
        )

    import hf_xet  # noqa: F811

    transfer_id = transfer_id or generate_transfer_id()
    total_files = len(file_specs)

    # Build download info and callbacks
    download_infos = []
    callbacks = []

    for i, spec in enumerate(file_specs):
        filename = os.path.basename(spec["dest_path"])
        download_infos.append(
            hf_xet.PyXetDownloadInfo(
                destination_path=str(os.path.abspath(spec["dest_path"])),
                hash=spec["hash"],
                file_size=spec["file_size"],
            )
        )
        callbacks.append(
            XetDownloadProgressCallback(
                filename=filename,
                total_bytes=spec["file_size"],
                event_queue=event_queue,
                report_interval=report_interval,
                transfer_id=transfer_id,
                file_index=i,
                total_files=total_files,
            )
        )

    # Get credentials from first file's xet_file_data
    # (all files in the same repo share the same endpoint)
    token_manager = XetTokenManager(token, endpoint)
    creds = token_manager.get_download_credentials(
        file_specs[0]["xet_file_data"]
    )

    # Emit start events
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

    try:
        kwargs = dict(
            download_info=download_infos,
            endpoint=creds.endpoint,
            token_info=creds.token_info,
            token_refresher=creds.token_refresher,
            progress_updater=callbacks,
        )
        if request_headers:
            kwargs["request_headers"] = request_headers

        results = hf_xet.download_files(**kwargs)

        # Emit complete events
        download_results = []
        for i, spec in enumerate(file_specs):
            filename = os.path.basename(spec["dest_path"])
            event_queue.put(
                ProgressEvent(
                    event_type=EventType.COMPLETE,
                    transfer_id=transfer_id,
                    direction=TransferDirection.DOWNLOAD,
                    filename=filename,
                    phase=ProgressPhase.COMPLETE,
                    bytes_completed=spec["file_size"],
                    total_bytes=spec["file_size"],
                    percentage=100.0,
                    file_index=i,
                    total_files=total_files,
                )
            )
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

    except Exception as e:
        for i, spec in enumerate(file_specs):
            filename = os.path.basename(spec["dest_path"])
            event_queue.put(
                ProgressEvent(
                    event_type=EventType.ERROR,
                    transfer_id=transfer_id,
                    direction=TransferDirection.DOWNLOAD,
                    filename=filename,
                    phase=ProgressPhase.ERROR,
                    error=str(e),
                    file_index=i,
                    total_files=total_files,
                )
            )
        raise
