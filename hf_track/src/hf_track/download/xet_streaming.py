"""Streaming snapshot download via chunk-by-chunk subprocess.

Resolves the file list, partitions into xet-stored and non-xet
files, and downloads xet files via the streaming worker. Kept in
its own module because of the substantial file-resolution logic
and the unique dual-path (xet streaming + main-process hf_hub_download).
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
from .._xet_worker import _serialize_xet_file_data, _xet_streaming_download_worker

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
    disable_fsync: bool = False,
    on_spawn: Optional[Callable[[object], None]] = None,
    on_finish: Optional[Callable[[], None]] = None,
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

    Args:
        repo_id: HuggingFace repository ID.
        token: Optional API token.
        event_queue: Queue used to publish progress events.
        allow_patterns: Optional glob patterns to include.
        ignore_patterns: Optional glob patterns to exclude.
        repo_type: ``"model"``, ``"dataset"`` or ``"space"``.
        revision: Optional git revision.
        endpoint: Optional HF endpoint override.
        local_dir: Optional local directory to write to.
        transfer_id: Pre-existing transfer ID. Auto-generated if None.
        report_interval: Seconds between progress events.
        is_cancelled: Optional callback returning True to cancel.
        force_download: If True, re-download even if files exist.
        fsync_interval: Bytes between ``os.fsync`` calls in the worker.
        disable_fsync: If True, skip ``os.fsync`` entirely (fastest,
            but a SIGKILL may leave zero-byte files on disk).
            Defaults to False (fsync on at the configured interval).
        on_spawn: Optional hook called with the active
            ``XetSubprocessRunner`` as soon as the subprocess is
            spawned. Used by ``HfTracker`` to register for fast
            cancel forwarding.
        on_finish: Optional cleanup hook called after the
            subprocess exits (success, error, or cancel).

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

    # _xet_streaming_download_worker is already imported at module level
    # (line 30, ``from .._xet_worker import ...``). An older version of
    # this file re-imported it here with a wrong relative path
    # (``from ._xet_worker import ...``), which raised
    # ``No module named 'hf_track.download._xet_worker'`` at function
    # call time. The runner (``XetSubprocessRunner.spawn_streaming``)
    # re-imports the worker itself when the subprocess starts, so we
    # never need the local name here.
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
            "disable_fsync": disable_fsync,
        }
        runner = XetSubprocessRunner()
        runner.spawn_streaming(params=params, event_queue=event_queue)
        # Plan 2026-06-05 step 3: notify the caller that the runner
        # is now active so it can register for fast cancel forwarding.
        if on_spawn is not None:
            try:
                on_spawn(runner)
            except Exception:
                pass
        try:
            stall_count = 0
            while True:
                result = runner.wait(timeout=1.0)
                if result is not None:
                    break
                # Safety net: if the subprocess is dead but we still
                # have no result, the _synthesize_result_if_missing
                # fix in the runner should have set self._result.
                # If somehow it didn't, break after 3 consecutive
                # dead checks to avoid an infinite loop.
                if not runner.is_alive():
                    stall_count += 1
                    if stall_count >= 3:
                        logger.error(
                            "Subprocess is dead but no result received "
                            "after %d checks — breaking wait loop",
                            stall_count,
                        )
                        break
                else:
                    stall_count = 0
                if is_cancelled is not None and is_cancelled():
                    # Plan step 3: request_cancel sets the event
                    # directly; step 4: terminate(grace=2.0) gives the
                    # child a 2 s cooperative window before SIGTERM.
                    runner.request_cancel()
                    runner.terminate(grace=2.0)
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
            runner.request_cancel()
            runner.terminate(grace=2.0)
            event_queue.put(
                ProgressEvent.cancelled_event(
                    transfer_id=transfer_id,
                    direction=TransferDirection.DOWNLOAD,
                    filename=repo_id,
                )
            )
            raise TransferCancelledError("Snapshot streaming interrupted by user (Ctrl+C)")
        finally:
            runner.terminate(grace=2.0)
            if on_finish is not None:
                try:
                    on_finish()
                except Exception:
                    pass
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

