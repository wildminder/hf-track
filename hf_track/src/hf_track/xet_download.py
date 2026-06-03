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
from .subprocess_runner import XetSubprocessRunner
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
    """Download a repository snapshot using the NEW ``hf_xet.XetSession`` API.

    This replaces the broken ``huggingface_hub.snapshot_download()`` flow
    (which uses a 1-arg ``progress_updater`` callback that only fires at
    file completion) with the new ``XetSession`` API that has a
    ``progress_callback`` firing every 100ms with actual per-chunk progress.

    Key differences from ``download_snapshot_with_xet``:
    - Uses ``hf_xet.XetSession.new_file_download_group(progress_callback=...)``
      instead of ``huggingface_hub.snapshot_download(tqdm_class=...)``.
    - The progress callback receives ``(GroupProgressReport, dict[UniqueID, ItemProgressReport])``
      and computes the increment from ``total_transfer_bytes_completed``.
    - Smooth per-chunk progress: bar advances every ~100ms during file downloads.

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
        ImportError: If hf_xet is not installed or XetSession not available.
        TransferCancelledError: If the transfer is cancelled.
        TransferProgressError: If the download fails.
    """
    if not is_xet_available():
        raise ImportError("hf_xet is not installed.")

    # Check if XetSession is available (requires hf_xet >= 1.5.0)
    try:
        import hf_xet
        if not hasattr(hf_xet, "XetSession"):
            raise ImportError("hf_xet.XetSession not available (requires hf_xet>=1.5.0)")
    except ImportError as e:
        raise ImportError(f"hf_xet.XetSession not available: {e}")

    transfer_id = transfer_id or generate_transfer_id()

    # Get the list of files with xet hashes from the repo
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()

    # Build the repo path
    if repo_type == "model":
        prefix = f"models/{repo_id}"
    elif repo_type == "dataset":
        prefix = f"datasets/{repo_id}"
    elif repo_type == "space":
        prefix = f"spaces/{repo_id}"
    else:
        prefix = f"{repo_type}s/{repo_id}"

    if revision:
        prefix = f"{prefix}@{revision}"

    # List all files
    try:
        all_files = fs.ls(prefix, detail=True)
    except Exception as e:
        raise TransferProgressError(f"Failed to list files in {repo_id}: {e}")

    # Filter by patterns
    import fnmatch
    def matches_patterns(name: str) -> bool:
        if allow_patterns:
            if not any(fnmatch.fnmatch(name, p) for p in allow_patterns):
                return False
        if ignore_patterns:
            if any(fnmatch.fnmatch(name, p) for p in ignore_patterns):
                return False
        return True

    # Filter to files only (not directories) and apply patterns
    xet_files = []
    for f in all_files:
        if f.get("type") != "file":
            continue
        # Get relative path
        full_name = f.get("name", "")
        if "/" in full_name.replace("\\", "/"):
            rel_name = full_name.split("/", 1)[1] if "/" in full_name else full_name
            # Handle @revision in path
            if "@" in rel_name:
                rel_name = rel_name.split("@", 1)[0]
        else:
            rel_name = full_name
        if not matches_patterns(rel_name):
            continue
        xet_hash = f.get("xet_hash")
        if not xet_hash:
            continue  # Skip non-xet files for now
        size = f.get("size", 0)
        # Build destination path
        if local_dir:
            dest = os.path.join(local_dir, rel_name)
        else:
            dest = os.path.join(os.getcwd(), rel_name)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        xet_files.append({
            "filename": rel_name,
            "xet_hash": xet_hash,
            "size": size,
            "dest_path": dest,
            "refresh_route": f"https://huggingface.co/api/{repo_type}s/{repo_id}/xet-read-token/{revision or 'main'}",
        })

    if not xet_files:
        # No xet files found, fall back to old method
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

    # Emit START event from main process
    event_queue.put(
        ProgressEvent(
            event_type=EventType.START,
            transfer_id=transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename=repo_id,
            phase=ProgressPhase.DOWNLOADING,
            total_bytes=sum(f["size"] for f in xet_files),
        )
    )

    params = {
        "files": xet_files,
        "transfer_id": transfer_id,
        "report_interval": report_interval,
        "repo_id": repo_id,
        "local_dir": local_dir or "",
    }

    runner = XetSubprocessRunner()
    runner.start(
        worker_func=_xet_session_snapshot_worker,
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
        state_manager.clear_state(transfer_id)
