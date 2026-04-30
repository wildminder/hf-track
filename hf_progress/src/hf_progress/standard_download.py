"""Standard (non-Xet) download progress tracking via tqdm_class.

Uses the ``tqdm_class`` parameter supported by ``hf_hub_download()``
and ``snapshot_download()`` to intercept HTTP download progress.

This is the cleanest integration path — no monkey-patching required.
The custom tqdm subclass receives ``update(n)`` calls for each HTTP
chunk downloaded, providing byte-level progress with speed tracking.

How it works internally:
    ``hf_hub_download(tqdm_class=...)``
    → ``_hf_hub_download_to_cache(tqdm_class=...)``
    → ``http_get(tqdm_class=...)``
    → ``_get_progress_bar_context(tqdm_class=...)``
    → ``cls(desc=..., total=..., unit="B", unit_scale=True)``
    → ``progress.update(len(chunk))`` per HTTP chunk

Key behavior: If your ``tqdm_class`` is NOT a subclass of
``huggingface_hub.utils.tqdm``, the ``_create_progress_bar()``
function calls ``cls(**kwargs)`` directly without injecting
``disable`` or ``name`` — your class is fully responsible for
its own behavior.
"""

from __future__ import annotations

import queue
from typing import Optional

from .callbacks import DownloadProgressTqdm
from .types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferDirection,
    generate_transfer_id,
)


def download_file(
    repo_id: str,
    filename: str,
    token: Optional[str],
    event_queue: queue.Queue,
    repo_type: str = "model",
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
    local_dir: Optional[str] = None,
    transfer_id: Optional[str] = None,
    report_interval: float = 0.1,
    **kwargs,
) -> str:
    """Download a single file with progress tracking via tqdm_class.

    Uses ``hf_hub_download()`` with a custom ``tqdm_class`` that
    emits ProgressEvent objects to the provided queue.

    This works with both HTTP and Xet downloads — when Xet is
    available, ``hf_hub_download()`` internally uses ``xet_get()``
    which also respects the ``tqdm_class`` parameter.

    Args:
        repo_id: Repository ID (e.g. ``"bert-base-uncased"``).
        filename: Filename within the repository.
        token: HuggingFace API token.
        event_queue: Queue for emitting ProgressEvent objects.
        repo_type: Repository type (model, dataset, space).
        revision: Optional git revision.
        endpoint: Optional custom HuggingFace API endpoint.
        local_dir: Optional local directory to download to. If set,
            the file is saved directly to this directory (not the
            HF cache). Symlinks are disabled in this mode.
        transfer_id: Unique transfer identifier (auto-generated if None).
        report_interval: Minimum seconds between progress events.
        **kwargs: Additional arguments passed to ``hf_hub_download()``.

    Returns:
        Local path to the downloaded file.

    Raises:
        Exception: If the download fails.
    """
    from huggingface_hub import hf_hub_download

    transfer_id = transfer_id or generate_transfer_id()

    # Create a bound tqdm class
    tqdm_class = DownloadProgressTqdm.bind(
        event_queue=event_queue,
        transfer_id=transfer_id,
        filename=filename,
        report_interval=report_interval,
    )

    # Emit start event
    event_queue.put(
        ProgressEvent(
            event_type=EventType.START,
            transfer_id=transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename=filename,
            phase=ProgressPhase.DOWNLOADING,
            total_bytes=0, # Unknown until download starts
        )
    )

    try:
        download_kwargs = dict(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            token=token,
            endpoint=endpoint,
            tqdm_class=tqdm_class,
        )
        # If local_dir is specified, download directly to that directory
        # instead of using the HF cache system
        if local_dir is not None:
            download_kwargs["local_dir"] = local_dir
        download_kwargs.update(kwargs)

        result = hf_hub_download(**download_kwargs)
        return result

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


def download_snapshot(
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
    **kwargs,
) -> str:
    """Download a repository snapshot with progress tracking.

    Uses ``snapshot_download()`` with a custom ``tqdm_class``.

    **Important caveat**: ``snapshot_download()`` provides file-count
    progress (N/M files downloaded), NOT per-file byte-level progress.
    The ``tqdm_class`` controls the outer file-count bar. For per-file
    byte progress, use ``download_file()`` for each file individually.

    Internally, ``snapshot_download()`` uses an ``_AggregatedTqdm``
    class to funnel per-file progress into an aggregate bytes bar.
    The ``tqdm_class`` parameter is passed to the ``thread_map`` that
    coordinates downloads.

    Args:
        repo_id: Repository ID.
        token: HuggingFace API token.
        event_queue: Queue for emitting ProgressEvent objects.
        allow_patterns: Glob patterns for files to include.
        ignore_patterns: Glob patterns for files to exclude.
        repo_type: Repository type.
        revision: Optional git revision.
        endpoint: Optional custom HuggingFace API endpoint.
        local_dir: Optional local directory to download to. If set,
            files are saved directly to this directory (not the HF
            cache). Symlinks are disabled in this mode.
        transfer_id: Unique transfer identifier.
        report_interval: Minimum seconds between progress events.
        **kwargs: Additional arguments passed to ``snapshot_download()``.

    Returns:
        Local path to the downloaded snapshot directory.

    Raises:
        Exception: If the download fails.
    """
    from huggingface_hub import snapshot_download

    transfer_id = transfer_id or generate_transfer_id()

    tqdm_class = DownloadProgressTqdm.bind(
        event_queue=event_queue,
        transfer_id=transfer_id,
        filename=f"snapshot:{repo_id}",
        report_interval=report_interval,
    )

    try:
        download_kwargs = dict(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            token=token,
            endpoint=endpoint,
            tqdm_class=tqdm_class,
        )
        # If local_dir is specified, download directly to that directory
        if local_dir is not None:
            download_kwargs["local_dir"] = local_dir
        download_kwargs.update(kwargs)

        result = snapshot_download(**download_kwargs)
        return result

    except Exception as e:
        event_queue.put(
            ProgressEvent(
                event_type=EventType.ERROR,
                transfer_id=transfer_id,
                direction=TransferDirection.DOWNLOAD,
                filename=f"snapshot:{repo_id}",
                phase=ProgressPhase.ERROR,
                error=str(e),
            )
        )
        raise
