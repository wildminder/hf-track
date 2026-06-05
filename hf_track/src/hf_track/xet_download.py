"""Xet direct download functions with progress tracking via subprocess isolation.

All ``hf_xet`` calls are executed in isolated child processes using
``XetSubprocessRunner``, so they can be safely terminated without
affecting the main process.
"""

from __future__ import annotations

import logging
import os
import queue
from dataclasses import dataclass
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

from ._xet_worker import _download_batch_worker, _download_worker, _serialize_xet_file_data, _snapshot_worker, _xet_session_download_worker, _xet_session_snapshot_worker
from .callbacks import state_manager
from .subprocess import XetSubprocessRunner
from .token import is_xet_available
from .types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferCancelledError,
    TransferDirection,
    TransferError,
    TransferProgressError,
    generate_transfer_id,
)


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

        if result.get("status") == "success":
            return result.get("destination_path", local_dir or repo_id)
        elif (
            result.get("status") == "cancelled"
            or result.get("error_type") == "TransferCancelledError"
            or "cancelled" in result.get("message", "").lower()
            or "interrupted" in result.get("message", "").lower()
        ):
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
                error=TransferError(message=str(e), error_type=type(e).__name__),
            )
        )
        raise
    finally:
        runner.terminate()


def download_snapshot_with_xet_session(
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
) -> str:
    """Download a repository snapshot — DEPRECATED: delegates to ``download_snapshot_with_xet``.

    .. deprecated::
        This function used to implement a snapshot downloader on top of
        the new ``hf_xet.XetSession`` API. That implementation was found
        to have three critical correctness bugs (2026-06-03):

        1. Nested folder layout (``local_dir/VoxCPM-0.5B/<files>`` instead
           of ``local_dir/<files>``) caused by incorrect path parsing of
           ``HfFileSystem.ls()`` output.
        2. Missing files: only 2 of 13 files were downloaded because
           (a) ``fs.ls()`` is non-recursive by default (misses
           subdirectories like ``assets/``) and (b) files without
           ``xet_hash`` (regular LFS files like ``config.json``,
           ``tokenizer.json``) were explicitly skipped.
        3. Zero-length files on disk after cancel because the new
           API does not pre-cleanup or use atomic tmp-file + rename
           semantics.

        The new ``XetSession`` API is fundamentally a single-session,
        single-batch primitive that does not provide the mixed
        xet/non-xet routing, subdirectory traversal, or cache-aware
        resume that snapshot downloads require. Re-implementing
        ``huggingface_hub.snapshot_download`` from scratch to work
        around these limitations would duplicate hundreds of lines
        of complex, battle-tested code and introduce more bugs.

        This function now delegates to ``download_snapshot_with_xet``,
        which uses ``huggingface_hub.snapshot_download()`` in an
        isolated subprocess (so ``hf_xet`` is still loaded only in
        the child for safe termination). The progress is per-file
        rather than per-chunk, which is the trade-off for correctness.

        See: ``docs/plans/2026-06-03-revert-broken-snapshot-xetsession-path.md``

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

    Returns:
        Path to the local directory containing downloaded files.

    Raises:
        ImportError: If hf_xet is not installed.
        TransferCancelledError: If the transfer is cancelled.
        TransferProgressError: If the download fails.
    """
    # Delegate to the proven path that uses
    # ``huggingface_hub.snapshot_download`` in an isolated subprocess.
    # This correctly handles:
    # - Mixed xet/non-xet files (huggingface_hub routes per-file)
    # - Subdirectories (huggingface_hub.snapshot_download walks them)
    # - Cache-aware resume (etag-based, tmp-file + rename atomic)
    # - Subprocess isolation (xet is loaded only in the child)
    # - Cancellation (terminate the child process)
    return download_snapshot_with_xet(
        repo_id=repo_id,
        token=token,
        event_queue=event_queue,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        repo_type=repo_type,
        revision=revision,
        endpoint=endpoint,
        local_dir=local_dir,
        transfer_id=transfer_id,
        report_interval=report_interval,
        is_cancelled=is_cancelled,
        use_xet=True,
    )


def download_snapshot_streaming(
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
    fsync_interval: int = 4 * 1024 * 1024,
) -> List[str]:
    """Download a snapshot using the streaming Xet API.

    Resolves the file list, partitions into xet-stored and non-xet
    files, and:

    * For xet files: spawns one ``XetSubprocessRunner`` running the
      chunk-by-chunk ``_xet_streaming_download_worker`` so each file's
      bytes are flushed to disk via ``os.write`` + ``os.fsync`` and the
      child can be killed mid-file.
    * For non-xet files: downloads in the main process via
      ``huggingface_hub.hf_hub_download`` (these are typically small
      JSON, markdown, .gitignore; bounded memory).

    Returns:
        Sorted list of file paths that were downloaded.

    Raises:
        ImportError: If ``hf_xet`` is not installed.
        TransferCancelledError: If the user cancels.
        TransferProgressError: If a download fails.
    """
    if not is_xet_available():
        raise ImportError("hf_xet is not installed; cannot use streaming snapshot.")

    from huggingface_hub import HfApi
    from ._xet_worker import _xet_streaming_download_worker

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

    api = HfApi(endpoint=endpoint, token=token)

    # Resolve the list of files in the snapshot.
    #
    # We use ``api.repo_info(..., files_metadata=True)`` to enumerate the
    # full sibling list (paths + sizes). We previously tried
    # ``api.get_paths_info(paths=None, expand=True)`` but that endpoint
    # requires an explicit ``paths`` list -- it cannot enumerate "all
    # files in a repo" and silently returns an empty list when called
    # with ``paths=None``. Empirically confirmed against
    # ``openbmb/VoxCPM-0.5B``: ``get_paths_info`` returned 0 entries
    # while ``repo_info(files_metadata=True)`` returned the correct 12
    # siblings. See ``tmp/diag_paths_info.py``.
    try:
        info = api.repo_info(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            files_metadata=True,
        )
    except Exception as e:
        raise TransferProgressError(f"Failed to list files for {repo_id}: {e}")

    file_specs: List[dict] = []
    non_xet_paths: List[str] = []
    for sibling in info.siblings:
        path = sibling.rfilename
        if not path:
            continue
        if allow_patterns and not _matches_any(path, allow_patterns):
            continue
        if ignore_patterns and _matches_any(path, ignore_patterns):
            continue
        sibling_size = sibling.size or 0
        if sibling_size == 0:
            # Zero-byte file: nothing to download via xet. Fall through
            # to the standard path which will materialize an empty file.
            non_xet_paths.append(path)
            continue
        # Resolve xet metadata via a per-file HEAD request.
        try:
            from huggingface_hub import hf_hub_url
            url = hf_hub_url(
                repo_id=repo_id,
                filename=path,
                repo_type=repo_type,
                revision=revision,
                endpoint=endpoint,
            )
            meta = api.get_hf_file_metadata(url=url, token=token)
        except Exception:
            non_xet_paths.append(path)
            continue
        if meta.xet_file_data is None:
            non_xet_paths.append(path)
            continue
        # Authoritative size from the HEAD (sibling.size can lag).
        meta_size = meta.size or sibling_size
        # Skip files that already exist with the right size
        if local_dir:
            dest = os.path.join(local_dir, path)
        else:
            from huggingface_hub import hf_hub_download as _hhd  # noqa: F401
            dest = None  # resolved by hf_hub_download below
        if (
            not force_download
            and dest is not None
            and os.path.exists(dest)
            and os.path.getsize(dest) == meta_size
        ):
            event_queue.put(
                ProgressEvent(
                    event_type=EventType.COMPLETE,
                    transfer_id=transfer_id,
                    direction=TransferDirection.DOWNLOAD,
                    filename=os.path.basename(path),
                    phase=ProgressPhase.COMPLETE,
                    bytes_completed=meta_size,
                    total_bytes=meta_size,
                    percentage=100.0,
                )
            )
            continue
        # Truncate partial files
        if dest is not None and os.path.exists(dest):
            try:
                os.remove(dest)
            except OSError:
                pass
        # Build the dest path expected by the streaming worker
        if dest is None:
            # Default cache layout
            from huggingface_hub.constants import HF_HUB_CACHE
            repo_cache = os.path.join(HF_HUB_CACHE, f"{repo_type}s--{repo_id.replace('/', '--')}")
            os.makedirs(repo_cache, exist_ok=True)
            dest = os.path.join(repo_cache, os.path.basename(path))
        file_specs.append(
            {
                "hash": meta.xet_file_data.file_hash,
                "file_size": meta_size,
                "dest_path": dest,
                "xet_file_data": _serialize_xet_file_data(meta.xet_file_data),
            }
        )

    # Spawn streaming subprocess for xet files
    downloaded_paths: List[str] = []
    if file_specs:
        serialized_specs = []
        for spec in file_specs:
            serialized_specs.append({
                "hash": spec["hash"],
                "file_size": spec["file_size"],
                "dest_path": spec["dest_path"],
                "xet_file_data": spec["xet_file_data"],
            })
        params = {
            "file_specs": serialized_specs,
            "token": token,
            "endpoint": endpoint,
            "transfer_id": transfer_id,
            "report_interval": report_interval,
            "request_headers": {},
            "fsync_interval": fsync_interval,
        }
        runner = XetSubprocessRunner()
        runner.spawn_streaming(params=params, event_queue=event_queue)
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
                    raise TransferCancelledError("Snapshot streaming cancelled by user")
            if result.get("status") == "cancelled":
                raise TransferCancelledError(result.get("message", "cancelled"))
            if result.get("status") != "success":
                raise TransferProgressError(result.get("message", "Streaming snapshot failed"))
        except KeyboardInterrupt:
            runner.terminate()
            event_queue.put(
                ProgressEvent.cancelled_event(
                    transfer_id=transfer_id,
                    direction=TransferDirection.DOWNLOAD,
                    filename=repo_id,
                )
            )
            raise TransferCancelledError("Snapshot streaming interrupted by user (Ctrl+C)")
        finally:
            runner.terminate()
        for spec in file_specs:
            downloaded_paths.append(spec["dest_path"])

    # Download non-xet files in main process
    if non_xet_paths:
        from huggingface_hub import hf_hub_download
        for path in non_xet_paths:
            try:
                p = hf_hub_download(
                    repo_id=repo_id,
                    filename=path,
                    repo_type=repo_type,
                    revision=revision,
                    token=token,
                )
                downloaded_paths.append(p)
            except Exception as e:
                raise TransferProgressError(f"Failed to download {path}: {e}")

    return sorted(downloaded_paths)


def _matches_any(path: str, patterns) -> bool:
    """Return True if path matches any of the glob patterns (matches by basename)."""
    import fnmatch
    base = os.path.basename(path)
    for pat in patterns or []:
        if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(base, pat):
            return True
    return False
