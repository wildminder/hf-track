"""Batch Xet download via the isolated subprocess runner.

Downloads a list of files. Kept as its own module because the input shape
(a list of ``file_specs`` dicts) and the output shape (a list of
``XetDownloadResult``) are distinct from the single-file download.

Worker layout
-------------
``download_files_with_xet`` picks one of two layouts, both of which run
each file through the *same* single-file worker:

* **one subprocess for the whole batch** (``max_workers == 1``, the
  default) — the historical behaviour. One process, one ``hf_xet``
  session, sequential files.
* **one subprocess per worker, files dealt out** (``max_workers > 1``) —
  independent files download concurrently.

The default stays at 1 on purpose. A pool means N ``hf_xet`` sessions and
N copies of the file data in flight; on a bandwidth-bound link that is
usually *slower*, not faster, and it multiplies peak memory. Callers who
know their bottleneck is per-file latency (many small files, slow per-file
handshake) turn it on explicitly.
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
    TransferErrorInfo,
    TransferProgressError,
    generate_transfer_id,
)
from ..subprocess import XetSubprocessRunner
from ..token import is_xet_available
from .._xet_worker import _download_batch_worker, _serialize_xet_file_data
from .xet_file import XetDownloadResult

def get_default_max_workers() -> int:
    """Number of subprocesses a batch download uses unless told otherwise.

    One, deliberately: a batch is normally bandwidth-bound, and a second
    ``hf_xet`` session competes for the same link rather than adding
    throughput. ``max_workers`` exists for the other case -- many small
    files, where the per-file handshake dominates -- and has to be opted
    into per call so nobody inherits it by accident.
    """
    return 1


def _split_specs(
    file_specs: List[dict], max_workers: int
) -> List[List[dict]]:
    """Deal ``file_specs`` out into at most ``max_workers`` contiguous groups.

    Contiguous rather than round-robin on purpose: it keeps each worker's
    files in the order the caller listed them, so the event stream stays
    readable and a cancelled batch cancels adjacent files together.
    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")
    if max_workers == 1 or len(file_specs) <= 1:
        return [list(file_specs)]

    groups: List[List[dict]] = [[] for _ in range(min(max_workers, len(file_specs)))]
    for index, spec in enumerate(file_specs):
        groups[index % len(groups)].append(spec)
    return [g for g in groups if g]


def _download_one_batch(
    file_specs: List[dict],
    *,
    token: str,
    event_queue: queue.Queue,
    endpoint: Optional[str],
    transfer_id: str,
    report_interval: float,
    request_headers: Optional[dict],
    is_cancelled: Optional[Callable[[], bool]],
    total_files: int,
    file_offset: int,
) -> List[XetDownloadResult]:
    """Run one worker over one group of files and rewrite its event indices.

    ``file_offset`` is where this group starts in the caller's list, so a
    pooled run still reports ``file_index`` in the caller's numbering
    rather than restarting at 0 per worker.
    """
    runner = XetSubprocessRunner()
    runner.start(
        worker_func=_download_batch_worker,
        params={
            "file_specs": [
                {**spec, "xet_file_data": _serialize_xet_file_data(spec.get("xet_file_data"))}
                for spec in file_specs
            ],
            "token": token,
            "endpoint": endpoint,
            "transfer_id": transfer_id,
            "report_interval": report_interval,
            "request_headers": request_headers or {},
        },
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
                    event_queue.put(
                        ProgressEvent.cancelled_event(
                            transfer_id=transfer_id,
                            direction=TransferDirection.DOWNLOAD,
                            filename=os.path.basename(spec["dest_path"]),
                            file_index=file_offset + i,
                            total_files=total_files,
                        )
                    )
                raise TransferCancelledError("Download cancelled by user")

        status = result.get("status")
        if status == "success":
            return [
                XetDownloadResult(
                    success=True,
                    filename=os.path.basename(spec["dest_path"]),
                    destination_path=spec["dest_path"],
                    file_size=spec["file_size"],
                    transfer_id=transfer_id,
                )
                for spec in file_specs
            ]
        if status == "cancelled":
            raise TransferCancelledError(result.get("message", "Download cancelled by user"))

        error_msg = result.get("message", "Batch download failed")
        error_type = result.get("error_type", "Exception")
        for i, spec in enumerate(file_specs):
            event_queue.put(
                ProgressEvent(
                    event_type=EventType.ERROR,
                    transfer_id=transfer_id,
                    direction=TransferDirection.DOWNLOAD,
                    filename=os.path.basename(spec["dest_path"]),
                    phase=ProgressPhase.ERROR,
                    error=TransferErrorInfo(message=error_msg, error_type=error_type),
                    file_index=file_offset + i,
                    total_files=total_files,
                )
            )
        raise TransferProgressError(error_msg)
    except KeyboardInterrupt:
        runner.terminate()
        for i, spec in enumerate(file_specs):
            event_queue.put(
                ProgressEvent.cancelled_event(
                    transfer_id=transfer_id,
                    direction=TransferDirection.DOWNLOAD,
                    filename=os.path.basename(spec["dest_path"]),
                    file_index=file_offset + i,
                    total_files=total_files,
                )
            )
        raise TransferCancelledError("Download interrupted by user (Ctrl+C)")
    except (TransferCancelledError, TransferProgressError):
        raise
    except Exception as e:
        runner.terminate()
        for i, spec in enumerate(file_specs):
            event_queue.put(
                ProgressEvent(
                    event_type=EventType.ERROR,
                    transfer_id=transfer_id,
                    direction=TransferDirection.DOWNLOAD,
                    filename=os.path.basename(spec["dest_path"]),
                    phase=ProgressPhase.ERROR,
                    error=TransferErrorInfo(message=str(e), error_type=type(e).__name__),
                    file_index=file_offset + i,
                    total_files=total_files,
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
    max_workers: int = 1,
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
        max_workers: How many subprocesses may run at once. 1 (the
            default) keeps the original single-process behaviour; higher
            values download independent files concurrently.

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

    groups = _split_specs(file_specs, max_workers)

    if len(groups) == 1:
        return _download_one_batch(
            groups[0],
            token=token,
            event_queue=event_queue,
            endpoint=endpoint,
            transfer_id=transfer_id,
            report_interval=report_interval,
            request_headers=request_headers,
            is_cancelled=is_cancelled,
            total_files=total_files,
            file_offset=0,
        )

    # Pooled: one subprocess per group, each on its own thread. The threads
    # exist only to give each XetSubprocessRunner a loop of its own to poll
    # in -- the transfer itself is in the child process, so the GIL is not
    # the constraint and no Python work is being parallelised.
    import threading

    # Where each group starts in the caller's list, so file_index keeps
    # the caller's numbering instead of restarting at 0 per worker.
    offset_of = {id(spec): i for i, spec in enumerate(file_specs)}
    offsets = [offset_of[id(group[0])] for group in groups]

    results: List[Optional[List[XetDownloadResult]]] = [None] * len(groups)
    errors: List[BaseException] = []
    errors_lock = threading.Lock()

    def _run_one(slot, group, offset):
        try:
            results[slot] = _download_one_batch(
                group,
                token=token,
                event_queue=event_queue,
                endpoint=endpoint,
                transfer_id=transfer_id,
                report_interval=report_interval,
                request_headers=request_headers,
                is_cancelled=is_cancelled,
                total_files=total_files,
                file_offset=offset,
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            with errors_lock:
                errors.append(exc)

    threads = [
        threading.Thread(
            target=_run_one,
            args=(slot, group, offset),
            daemon=True,
            name=f"hf-track-batch-{offset}",
        )
        for slot, (group, offset) in enumerate(zip(groups, offsets))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        # The first failure to arrive is the one the caller wants; the
        # rest are the same failure seen through the other workers.
        raise errors[0]

    # Return in the caller's file order. The groups are dealt out
    # round-robin, so concatenating them -- in any order -- interleaves
    # the results relative to ``file_specs``. Re-ordering by destination
    # is what makes a pooled run's output identical to a serial one's,
    # which is the property callers (and this test) depend on.
    by_destination: dict = {}
    for group_results in results:
        for result in group_results or []:
            by_destination[result.destination_path] = result
    return [
        by_destination[spec["dest_path"]]
        for spec in file_specs
        if spec["dest_path"] in by_destination
    ]
