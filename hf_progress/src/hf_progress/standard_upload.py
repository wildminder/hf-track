"""Standard (non-Xet) upload progress tracking via tqdm monkey-patching.

Since ``HfApi.upload_file()`` has no ``tqdm_class`` or ``progress_callback``
parameter, the only way to extract upload progress is to temporarily
replace the global tqdm class used by ``huggingface_hub``.

The upload path internally:
    ``HfApi.upload_file()``
    → ``create_commit()``
    → ``_upload_files()``
    → ``_upload_lfs_files()``  (when Xet is NOT available)
    → ``thread_map(_wrapped_lfs_upload, ..., tqdm_class=hf_tqdm)``
    → ``lfs_upload()``
    → ``_upload_single_part()`` or ``_upload_multi_part()``
    → ``operation.as_file(with_tqdm=True)``
    → ``tqdm_stream_file(path)``
    → ``pbar = tqdm(total=file_size, desc=filename, unit="B", unit_scale=True)``
    → ``f.read = _inner_read``  (monkey-patched read)
    → ``pbar.update(len(data))`` per chunk

The ``tqdm_upload_patcher`` context manager intercepts these tqdm bars
by filtering file-level bars (``unit="B"``) from file-count bars
(``unit="it"``) created by ``thread_map``.

**WARNING**: This patches the GLOBAL tqdm class. See the warnings
in the ``tqdm_upload_patcher`` docstring.
"""

from __future__ import annotations

import os
import queue
import tempfile
from typing import Optional

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
) -> str:
    """Upload a file with progress tracking via tqdm monkey-patching.

    Uses ``HfApi.upload_file()`` inside a ``tqdm_upload_patcher``
    context manager to intercept LFS upload progress.

    **Limitations**:
    - Only works for file path uploads (not bytes/BytesIO).
    - Patches the global tqdm class — avoid concurrent tqdm operations.
    - No progress for small files uploaded as regular git blobs.
    - Files already present upstream (LFS dedup) skip the upload
      entirely, so no progress events are emitted.

    Args:
        file_path: Local path to the file to upload.
        repo_id: Repository ID (e.g. ``"username/model"``).
        token: HuggingFace API token.
        event_queue: Queue for emitting ProgressEvent objects.
        path_in_repo: Target path in the repository (defaults to filename).
        repo_type: Repository type (model, dataset, space).
        revision: Optional git revision.
        endpoint: Optional custom HuggingFace API endpoint.
        transfer_id: Unique transfer identifier (auto-generated if None).
        report_interval: Minimum seconds between progress events.

    Returns:
        URL of the uploaded file.

    Raises:
        Exception: If the upload fails.
    """
    from huggingface_hub import HfApi

    transfer_id = transfer_id or generate_transfer_id()
    filename = os.path.basename(file_path)
    path_in_repo = path_in_repo or filename

    # Emit start event manually (the patcher will also emit one
    # when it detects the file-level tqdm bar)
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

            # Emit complete event (in case the patcher's close()
            # didn't fire — e.g. file already present upstream)
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
) -> str:
    """Upload bytes with progress tracking by writing to a temp file first.

    When uploading from ``bytes`` or ``io.BufferedIOBase``,
    ``CommitOperationAdd.as_file()`` does NOT use ``tqdm_stream_file()``,
    so there is no progress tracking. This function works around the
    limitation by writing the content to a temporary file and then
    uploading the file path.

    Args:
        file_content: File content as bytes.
        filename: Name for the file in the repository.
        repo_id: Repository ID.
        token: HuggingFace API token.
        event_queue: Queue for emitting ProgressEvent objects.
        path_in_repo: Target path in the repository (defaults to filename).
        repo_type: Repository type.
        revision: Optional git revision.
        endpoint: Optional custom HuggingFace API endpoint.
        transfer_id: Unique transfer identifier.
        report_interval: Minimum seconds between progress events.

    Returns:
        URL of the uploaded file.

    Raises:
        Exception: If the upload fails.
    """
    transfer_id = transfer_id or generate_transfer_id()
    path_in_repo = path_in_repo or filename

    # Write to temp file
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
        )
    finally:
        os.unlink(temp_path)
