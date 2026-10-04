"""Streaming snapshot download via in-process hybrid Xet + HTTP worker.

Plan ``docs/plans/2026-06-15-xet-streaming-hybrid-approach.md``
(2026-06-15 revision): runs HYBRID downloads **in-process** (a daemon
``threading.Thread``) instead of via a spawned subprocess, because the
``hf_xet`` Rust extension's APIs misbehave when called from a separate
process in this environment. A single ``HybridRunner`` wraps the
``download_hybrid`` driver which tries ``XetFileDownloadGroup``
(Tier 1) and falls back to a pure-HTTP download via ``requests``
(Tier 3).

The driver also downloads non-xet files (small JSON, .gitignore,
markdown) via ``huggingface_hub.hf_hub_download`` in the main
process -- these files are small and bounded.
"""

from __future__ import annotations

import logging
import os
import queue
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

from ..types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferCancelledError,
    TransferDirection,
    TransferErrorInfo,
    TransferProgressError,
    generate_transfer_id,
)
from ..token import is_xet_available
from .._xet_worker import HybridRunner, download_hybrid, _serialize_xet_file_data
from ._matching import _matches_any

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
    tier_timeout_s: float = 60.0,
    enable_http_fallback: bool = True,
    use_xet: bool = True,
    on_spawn: Optional[Callable[[object], None]] = None,
    on_finish: Optional[Callable[[], None]] = None,
) -> List[str]:
    """Download a snapshot using the hybrid ``XetFileDownloadGroup`` + HTTP fallback path.

    Plan: ``docs/plans/2026-06-15-xet-streaming-hybrid-approach.md``.

    Resolves the file list, partitions into xet-stored and non-xet
    files, and:

    * For xet files: spawns one ``XetSubprocessRunner`` running the new
      ``_xet_file_download_worker`` which uses
      ``hf_xet.XetSession().new_file_download_group().start_download_file()``
      under the hood. Per file, the worker polls
      ``handle.progress()`` and falls back to a pure-HTTP download
      ``(download.download.http_fallback.download_file_http)`` if the
      xet handle does not complete within ``tier_timeout_s``
      seconds. The HTTP path streams bytes via ``requests`` and writes
      them to disk with ``os.write`` + periodic ``os.fsync``, so the
      file is observable on disk throughout the download.
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
        tier_timeout_s: Seconds to wait on a Tier 1 (xet) download
            before falling back to Tier 3 (HTTP) for one file.
        enable_http_fallback: When False, the worker emits a
            ``XetTierFailed`` error instead of falling back to HTTP.
        use_xet: When False, skips the Tier 1 (xet) path entirely and
            routes every xet file through the HTTP fallback in the
            worker. Equivalent to ``--no-xet`` for the streaming
            snapshot path.
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

    # ``_xet_file_download_worker`` is imported at module level (line 30).
    # The runner (``XetSubprocessRunner.spawn_streaming``) re-imports
    # the worker itself when the subprocess starts, so we never need
    # the local name here.
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
            dest = os.path.join(repo_cache, path)
        file_specs.append(
            {
                "hash": meta.xet_file_data.file_hash,
                "file_size": meta_size,
                # Repo-relative path. Without this the worker falls back to
                # ``os.path.basename(dest_path)``, which drops the directory
                # component — ``onnx/audio_encoder.onnx`` became
                # ``audio_encoder.onnx`` and the HTTP fallback 404'd.
                "filename": path,
                "dest_path": dest,
                "xet_file_data": _serialize_xet_file_data(meta.xet_file_data),
            }
        )

    # Run IN-PROCESS via HybridRunner (daemon thread + threading.Event).
    # Plan 2026-06-15 redesign: dropping the spawned-subprocess path
    # avoids GIL/CAS issues with hf_xet 1.5.0 and lets Xet run normally.
    # Each file uses Tier 1 (XetFileDownloadGroup) with a timeout; if
    # the timeout fires, Tier 3 (HTTP) is used for that file.
    downloaded_paths: List[str] = []
    if file_specs:
        serialized_specs: List[Dict[str, Any]] = []
        for spec in file_specs:
            entry = {
                "hash": spec["hash"],
                "file_size": spec["file_size"],
                "dest_path": spec["dest_path"],
                "xet_file_data": spec["xet_file_data"],
            }
            if spec.get("filename"):
                entry["filename"] = spec["filename"]
            serialized_specs.append(entry)

        runner = HybridRunner()
        if on_spawn is not None:
            try:
                on_spawn(runner)
            except Exception:
                pass
        runner.start(
            params={
                "file_specs": serialized_specs,
                "repo_id": repo_id,
                "repo_type": repo_type,
                "revision": revision,
                "endpoint": endpoint,
                "token": token,
                "transfer_id": transfer_id,
                "report_interval": report_interval,
                "fsync_interval": fsync_interval,
                "tier_timeout_s": tier_timeout_s,
                "poll_interval_s": 0.1,
                "enable_http_fallback": enable_http_fallback,
                "use_xet": use_xet,
            },
            event_queue=event_queue,
        )

        try:
            result = runner.wait(timeout=None)
            assert result is not None  # in-process never times out
            if result.get("status") == "cancelled":
                event_queue.put(
                    ProgressEvent.cancelled_event(
                        transfer_id=transfer_id,
                        direction=TransferDirection.DOWNLOAD,
                        filename=repo_id,
                        bytes_completed=result.get("bytes_completed", 0),
                        total_bytes=result.get("total_bytes", 0),
                    )
                )
                raise TransferCancelledError("Snapshot streaming cancelled by user")
            if result.get("status") != "success":
                raise TransferProgressError(result.get("message", "Streaming snapshot failed"))
        except KeyboardInterrupt:
            runner.request_cancel()
            event_queue.put(
                ProgressEvent.cancelled_event(
                    transfer_id=transfer_id,
                    direction=TransferDirection.DOWNLOAD,
                    filename=repo_id,
                )
            )
            raise TransferCancelledError("Snapshot streaming interrupted by user (Ctrl+C)")
        except TransferCancelledError:
            raise
        finally:
            runner.terminate(grace=1.0)
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

