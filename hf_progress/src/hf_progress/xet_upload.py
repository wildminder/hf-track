"""Xet direct upload functions with progress tracking.

Bypasses ``huggingface_hub``'s ``HfApi.upload_file()`` to call
``hf_xet.upload_files()`` and ``hf_xet.upload_bytes()`` directly,
passing a custom ``progress_updater`` callback for detailed progress.

This provides the highest quality progress data available:
- Byte-level progress with deduplication awareness
- Per-file progress via ``PyItemProgressUpdate``
- Transfer speed (both processing and network)
- Dedup savings (bytes_completed - transfer_bytes_completed)

**Critical**: The ``progress_updater`` callback parameter names MUST be
``total_update`` and ``item_updates`` for the Rust runtime's
``WrappedProgressUpdaterImpl`` to detect the detailed callback signature.
If names don't match, the Rust runtime falls back to simple ``(int)``
mode and all detailed progress data is lost.
"""

from __future__ import annotations

import os
import queue
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from .callbacks import XetUploadProgressCallback
from .token import XetCredentials, XetTokenManager, is_xet_available
from .types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferDirection,
    TransferResult,
    generate_transfer_id,
)


@dataclass
class XetUploadResult:
    """Result of a Xet upload operation.

    Attributes:
        success: Whether the upload completed successfully.
        filename: Name of the uploaded file.
        hash: Content hash of the uploaded file.
        file_size: Size of the uploaded file in bytes.
        transfer_id: Unique identifier for the transfer.
    """

    success: bool
    filename: str
    hash: str = ""
    file_size: int = 0
    transfer_id: str = ""


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
) -> XetUploadResult:
    """Upload a file from disk using hf_xet with detailed progress.

    Calls ``hf_xet.upload_files()`` directly, bypassing
    ``HfApi.upload_file()`` to get detailed progress callbacks.

    Args:
        file_path: Local path to the file to upload.
        repo_id: Repository ID (e.g. ``"username/model"``).
        token: HuggingFace API token.
        event_queue: Queue for emitting ProgressEvent objects.
        repo_type: Repository type (model, dataset, space).
        revision: Optional git revision.
        endpoint: Optional custom HuggingFace API endpoint.
        transfer_id: Unique transfer identifier (auto-generated if None).
        report_interval: Minimum seconds between progress events.

    Returns:
        XetUploadResult with upload status and metadata.

    Raises:
        ImportError: If ``hf_xet`` is not installed.
        RuntimeError: If the upload fails.
    """
    if not is_xet_available():
        raise ImportError(
            "hf_xet is not installed. Install with: "
            'pip install "huggingface_hub[hf_xet]"'
        )

    import hf_xet  # noqa: F811

    transfer_id = transfer_id or generate_transfer_id()
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    # Get credentials
    token_manager = XetTokenManager(token, endpoint)
    creds = token_manager.get_upload_credentials(repo_id, repo_type, revision)

    # Create progress callback
    callback = XetUploadProgressCallback(
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
            direction=TransferDirection.UPLOAD,
            filename=filename,
            phase=ProgressPhase.UPLOADING,
            total_bytes=file_size,
        )
    )

    try:
        results = hf_xet.upload_files(
            file_paths=[file_path],
            endpoint=creds.endpoint,
            token_info=creds.token_info,
            token_refresher=creds.token_refresher,
            progress_updater=callback,
            _repo_type=repo_type,
        )

        # Emit complete event
        result_info = results[0] if results else None
        event_queue.put(
            ProgressEvent(
                event_type=EventType.COMPLETE,
                transfer_id=transfer_id,
                direction=TransferDirection.UPLOAD,
                filename=filename,
                phase=ProgressPhase.COMPLETE,
                bytes_completed=file_size,
                total_bytes=file_size,
                percentage=100.0,
            )
        )

        return XetUploadResult(
            success=True,
            filename=filename,
            hash=getattr(result_info, "hash", "") if result_info else "",
            file_size=getattr(result_info, "file_size", file_size)
            if result_info
            else file_size,
            transfer_id=transfer_id,
        )

    except Exception as e:
        event_queue.put(
            ProgressEvent(
                event_type=EventType.ERROR,
                transfer_id=transfer_id,
                direction=TransferDirection.UPLOAD,
                filename=filename,
                phase=ProgressPhase.ERROR,
                error=str(e),
            )
        )
        raise


def upload_bytes_with_xet(
    file_content: bytes,
    filename: str,
    repo_id: str,
    token: str,
    event_queue: queue.Queue,
    repo_type: str = "model",
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
) -> XetUploadResult:
    """Upload in-memory bytes using hf_xet with detailed progress.

    Calls ``hf_xet.upload_bytes()`` directly. This is used for
    uploading file content that is already in memory (e.g. wheel
    files, generated content).

    **Note**: ``token_refresher`` is a required parameter in the Rust
    binding even though it can be ``None``. Omitting it causes:
    ``upload_bytes() missing 1 required positional argument: 'token_refresher'``

    Args:
        file_content: File content as bytes.
        filename: Name for the file in the repository.
        repo_id: Repository ID.
        token: HuggingFace API token.
        event_queue: Queue for emitting ProgressEvent objects.
        repo_type: Repository type.
        revision: Optional git revision.
        endpoint: Optional custom HuggingFace API endpoint.
        transfer_id: Unique transfer identifier (auto-generated if None).
        report_interval: Minimum seconds between progress events.

    Returns:
        XetUploadResult with upload status and metadata.

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
    file_size = len(file_content)

    # Get credentials
    token_manager = XetTokenManager(token, endpoint)
    creds = token_manager.get_upload_credentials(repo_id, repo_type, revision)

    # Create progress callback
    callback = XetUploadProgressCallback(
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
            direction=TransferDirection.UPLOAD,
            filename=filename,
            phase=ProgressPhase.UPLOADING,
            total_bytes=file_size,
        )
    )

    try:
        results = hf_xet.upload_bytes(
            file_contents=[file_content],
            endpoint=creds.endpoint,
            token_info=creds.token_info,
            token_refresher=creds.token_refresher,
            progress_updater=callback,
            _repo_type=repo_type,
        )

        # Emit complete event
        result_info = results[0] if results else None
        event_queue.put(
            ProgressEvent(
                event_type=EventType.COMPLETE,
                transfer_id=transfer_id,
                direction=TransferDirection.UPLOAD,
                filename=filename,
                phase=ProgressPhase.COMPLETE,
                bytes_completed=file_size,
                total_bytes=file_size,
                percentage=100.0,
            )
        )

        return XetUploadResult(
            success=True,
            filename=filename,
            hash=getattr(result_info, "hash", "") if result_info else "",
            file_size=getattr(result_info, "file_size", file_size)
            if result_info
            else file_size,
            transfer_id=transfer_id,
        )

    except Exception as e:
        event_queue.put(
            ProgressEvent(
                event_type=EventType.ERROR,
                transfer_id=transfer_id,
                direction=TransferDirection.UPLOAD,
                filename=filename,
                phase=ProgressPhase.ERROR,
                error=str(e),
            )
        )
        raise


def upload_bytes_via_temp_file(
    file_content: bytes,
    filename: str,
    repo_id: str,
    token: str,
    event_queue: queue.Queue,
    repo_type: str = "model",
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
) -> XetUploadResult:
    """Upload bytes by writing to a temp file first (for LFS progress).

    When using the LFS upload path (no Xet), ``tqdm_stream_file()``
    only works with file paths, not bytes. This function writes the
    content to a temporary file and then uploads it, enabling
    tqdm-based progress tracking.

    Also works with Xet: if ``hf_xet`` is available, uses
    ``upload_file_with_xet()`` for the best progress data.

    Args:
        file_content: File content as bytes.
        filename: Name for the file in the repository.
        repo_id: Repository ID.
        token: HuggingFace API token.
        event_queue: Queue for emitting ProgressEvent objects.
        repo_type: Repository type.
        revision: Optional git revision.
        endpoint: Optional custom HuggingFace API endpoint.
        transfer_id: Unique transfer identifier.
        report_interval: Minimum seconds between progress events.

    Returns:
        XetUploadResult with upload status and metadata.
    """
    transfer_id = transfer_id or generate_transfer_id()

    # Write to temp file
    with tempfile.NamedTemporaryFile(
        suffix=f"_{filename}", delete=False
    ) as f:
        f.write(file_content)
        temp_path = f.name

    try:
        if is_xet_available():
            return upload_file_with_xet(
                file_path=temp_path,
                repo_id=repo_id,
                token=token,
                event_queue=event_queue,
                repo_type=repo_type,
                revision=revision,
                endpoint=endpoint,
                transfer_id=transfer_id,
                report_interval=report_interval,
            )
        else:
            # Xet not available — this should be called via the
            # standard_upload path which handles tqdm patching
            raise ImportError(
                "hf_xet not available. Use standard_upload.upload_file() "
                "with tqdm_upload_patcher instead."
            )
    finally:
        os.unlink(temp_path)
