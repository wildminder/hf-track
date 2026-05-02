"""Standard (non-Xet) upload progress tracking via tqdm monkey-patching."""

from __future__ import annotations

import os
import queue
import tempfile
from typing import Callable, Optional

from .callbacks import tqdm_upload_patcher
from .types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferDirection,
    generate_transfer_id,
)


def upload_file(
    file_path: str,
    repo_id: str,
    token: str,
    event_queue: queue.Queue,
    path_in_repo: Optional[str] = None,
    repo_type: str = "model",
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> str:
    from huggingface_hub import HfApi

    transfer_id = transfer_id or generate_transfer_id()
    filename = os.path.basename(file_path)
    path_in_repo = path_in_repo or filename

    event_queue.put(
        ProgressEvent(
            event_type=EventType.START,
            transfer_id=transfer_id,
            direction=TransferDirection.UPLOAD,
            filename=filename,
            phase=ProgressPhase.UPLOADING,
            total_bytes=os.path.getsize(file_path),
        )
    )

    with tqdm_upload_patcher(
        event_queue=event_queue,
        transfer_id=transfer_id,
        filename=filename,
        report_interval=report_interval,
        is_cancelled=is_cancelled,
    ):
        try:
            api = HfApi(token=token, endpoint=endpoint)
            result = api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
            )

            event_queue.put(
                ProgressEvent(
                    event_type=EventType.COMPLETE,
                    transfer_id=transfer_id,
                    direction=TransferDirection.UPLOAD,
                    filename=filename,
                    phase=ProgressPhase.COMPLETE,
                    percentage=100.0,
                )
            )

            return result

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


def upload_bytes(
    file_content: bytes,
    filename: str,
    repo_id: str,
    token: str,
    event_queue: queue.Queue,
    path_in_repo: Optional[str] = None,
    repo_type: str = "model",
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> str:
    transfer_id = transfer_id or generate_transfer_id()
    path_in_repo = path_in_repo or filename

    with tempfile.NamedTemporaryFile(
        suffix=f"_{filename}", delete=False
    ) as f:
        f.write(file_content)
        temp_path = f.name

    try:
        return upload_file(
            file_path=temp_path,
            repo_id=repo_id,
            token=token,
            event_queue=event_queue,
            path_in_repo=path_in_repo,
            repo_type=repo_type,
            revision=revision,
            endpoint=endpoint,
            transfer_id=transfer_id,
            report_interval=report_interval,
            is_cancelled=is_cancelled,
        )
    finally:
        os.unlink(temp_path)