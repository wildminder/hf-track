"""Xet bytes upload + Xet/LFS routing.

Two public functions, both about *bytes going up*:

  * :func:`upload_bytes_with_xet` -- send raw bytes via hf_xet in a
    subprocess. For payloads above 10 MB, the bytes are first written
    to a temp file to avoid expensive pickle of large blobs across the
    process boundary.
  * :func:`upload_bytes_via_temp_file` -- convenience wrapper that
    writes bytes to a temp file and then routes them through Xet
    (when available) or the standard LFS upload (when not).
"""

from __future__ import annotations

import os
import queue
import tempfile
from typing import Callable, Optional

from ..token import is_xet_available
from ..types import generate_transfer_id
from .._xet_worker import _upload_bytes_worker
from .xet_file import XetUploadResult, _run_upload_in_subprocess

# Threshold for auto-writing bytes to temp file instead of pickling across processes
_LARGE_PAYLOAD_THRESHOLD = 10 * 1024 * 1024  # 10 MB


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
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> XetUploadResult:
    """Upload bytes via Xet in an isolated subprocess.

    For payloads larger than 10MB, the bytes are automatically written
    to a temporary file to avoid excessive pickle serialization cost
    across the process boundary.

    Args:
        file_content: Raw bytes to upload.
        filename: Name for the uploaded file.
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

    file_size = len(file_content)
    temp_path = None

    try:
        if file_size > _LARGE_PAYLOAD_THRESHOLD:
            # Write to temp file to avoid expensive pickle of large bytes
            with tempfile.NamedTemporaryFile(
                suffix=f"_{filename}", delete=False
            ) as f:
                f.write(file_content)
                temp_path = f.name

            params = {
                "file_path": temp_path,
                "filename": filename,
                "repo_id": repo_id,
                "token": token,
                "repo_type": repo_type,
                "revision": revision,
                "endpoint": endpoint,
                "transfer_id": transfer_id,
                "report_interval": report_interval,
                "direction": "upload",
            }
        else:
            params = {
                "file_content": file_content,
                "filename": filename,
                "repo_id": repo_id,
                "token": token,
                "repo_type": repo_type,
                "revision": revision,
                "endpoint": endpoint,
                "transfer_id": transfer_id,
                "report_interval": report_interval,
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
            worker_func=_upload_bytes_worker,
            params=params,
        )
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


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
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> XetUploadResult:
    """Upload bytes via temp file, choosing Xet or LFS based on availability.

    This is a convenience wrapper that handles the Xet/LFS routing
    for bytes uploads.
    """
    transfer_id = transfer_id or generate_transfer_id()

    with tempfile.NamedTemporaryFile(
        suffix=f"_{filename}", delete=False
    ) as f:
        f.write(file_content)
        temp_path = f.name

    try:
        if is_xet_available():
            from .xet_file import upload_file_with_xet
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
                is_cancelled=is_cancelled,
            )
        else:
            from .standard import upload_file as _upload_file
            return _upload_file(
                file_path=temp_path,
                repo_id=repo_id,
                token=token,
                event_queue=event_queue,
                path_in_repo=filename,
                repo_type=repo_type,
                revision=revision,
                endpoint=endpoint,
                transfer_id=transfer_id,
                report_interval=report_interval,
                is_cancelled=is_cancelled,
            )
    finally:
        os.unlink(temp_path)
