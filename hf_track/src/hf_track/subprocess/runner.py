"""Subprocess runner for isolating hf_xet operations.

Manages the lifecycle of a child process that runs an Xet worker function.
Provides:

- ``start()``: Spawn a child process with ``multiprocessing.get_context("spawn")``
- ``terminate()``: Kill the child process (SIGTERM → SIGKILL fallback)
- ``wait()``: Block until the worker sends a terminal message
- Event relay: Background thread translates ``mp.Queue`` messages
  into ``queue.Queue[ProgressEvent]`` for the main process

Key design choices:

- **``spawn`` context only**: Avoids inheriting parent state/locks/CUDA
- **``daemon=True``**: Child dies if main process crashes
- **Relay thread**: Bridges ``mp.Queue`` → ``queue.Queue`` so the
  main process keeps its existing ``ProgressEvent`` API
- **``mp.Event`` cancel signal**: Cross-process cancellation that
  the worker's callback checks on each invocation
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import warnings
from typing import Any, Callable, Dict, Optional

from ..types import EventType, ProgressEvent, TransferError
from .messages import MSG_EVENT, SubprocessMessage

logger = logging.getLogger(__name__)


class XetSubprocessRunner:
    """Manages an isolated subprocess for Xet operations.

    Usage::

        runner = XetSubprocessRunner()
        runner.start(
            worker_func=_download_worker,
            params={"file_hash": "abc", ...},
            event_queue=my_queue,
        )
        # ... events flow into my_queue via relay thread ...
        result = runner.wait(timeout=300)
        runner.terminate()  # always call to clean up

    Args:
        terminate_timeout: Seconds to wait after SIGTERM before SIGKILL (default 3).
        kill_timeout: Seconds to wait after SIGKILL before giving up (default 2).
    """

    def __init__(
        self,
        terminate_timeout: float = 3.0,
        kill_timeout: float = 2.0,
    ):
        self._ctx = mp.get_context("spawn")
        self._process: Optional[mp.Process] = None
        self._mp_queue: Optional[mp.Queue] = None
        self._cancel_event: Optional[mp.Event] = None
        self._relay_thread: Optional[threading.Thread] = None
        self._event_queue: Optional[queue.Queue] = None
        self._stop_event = threading.Event()
        self._result: Optional[Dict[str, Any]] = None
        self._terminate_timeout = terminate_timeout
        self._kill_timeout = kill_timeout
        self._lock = threading.Lock()

    def start(
        self,
        worker_func: Callable,
        params: Dict[str, Any],
        event_queue: queue.Queue,
    ) -> None:
        """Spawn child process with the given worker function and start event relay thread.

        Args:
            worker_func: A top-level function that runs in the child process.
                Must accept ``(params: dict, mp_queue: mp.Queue, cancel_event: mp.Event)``.
            params: Picklable dict of parameters for the worker.
            event_queue: Main-process queue to receive ``ProgressEvent`` objects.

        Raises:
            RuntimeError: If a process is already running.
        """
        with self._lock:
            if self._process is not None and self._process.is_alive():
                raise RuntimeError("A subprocess is already running. Call terminate() first.")

            self._event_queue = event_queue
            self._stop_event.clear()
            self._result = None

            # Create multiprocessing objects in the spawn context
            self._mp_queue = self._ctx.Queue()
            self._cancel_event = self._ctx.Event()

            # Spawn child process
            self._process = self._ctx.Process(
                target=worker_func,
                args=(params, self._mp_queue, self._cancel_event),
                daemon=True,
            )
            self._process.start()

            # Start relay thread
            self._relay_thread = threading.Thread(
                target=self._relay_events,
                name=f"xet-relay-{self._process.pid}",
                daemon=True,
            )
            self._relay_thread.start()

            logger.debug(
                "Subprocess started: pid=%d, worker=%s",
                self._process.pid,
                getattr(worker_func, "__name__", str(worker_func)),
            )

    def spawn_streaming(
        self,
        params: Dict[str, Any],
        event_queue: queue.Queue,
    ) -> None:
        """Spawn a streaming-download subprocess using the chunk-by-chunk worker.

        Convenience wrapper that calls ``start()`` with the streaming
        worker function ``_xet_streaming_download_worker``. The worker
        uses ``XetSession().new_download_stream_group().download_stream()``
        to write each file's chunks to disk incrementally — no in-memory
        buffering of whole files.

        Args:
            params: Picklable dict of parameters for the worker. Must
                include the keys required by ``_xet_streaming_download_worker``:
                ``file_specs`` (list of dicts with ``hash``, ``file_size``,
                ``dest_path``, ``xet_file_data``), ``token``, ``endpoint``,
                ``transfer_id``, ``report_interval``, ``request_headers``,
                ``fsync_interval`` (optional).
            event_queue: Main-process queue to receive ``ProgressEvent`` objects.

        Raises:
            RuntimeError: If a process is already running.
        """
        # Local import to avoid a circular dependency at module load time
        # (subprocess_runner is imported by _xet_worker, which is imported by
        # subprocess_messages → runner → worker chain). The streaming worker
        # itself doesn't import the runner.
        from .._xet_worker import _xet_streaming_download_worker
        self.start(
            worker_func=_xet_streaming_download_worker,
            params=params,
            event_queue=event_queue,
        )

    def _relay_events(self) -> None:
        """Background thread: mp.Queue → queue.Queue translation.

        Reads ``SubprocessMessage`` from the multiprocessing queue,
        converts to ``ProgressEvent``, and puts into the main event queue.
        Stops when:
        - A terminal message (result/error/cancelled) is received
        - The stop_event is set (from terminate())
        - The mp_queue is empty for too long after the process exits
        """
        while not self._stop_event.is_set():
            try:
                msg = self._mp_queue.get(timeout=0.2)  # type: ignore[union-attr]
            except Exception:
                # Queue empty or closed — check if process is still alive
                if self._process is not None and not self._process.is_alive():
                    # Drain any remaining messages before exiting
                    self._drain_queue()
                    self._synthesize_result_if_missing()
                    break
                continue

            if msg.is_event:
                # Translate SubprocessMessage → ProgressEvent
                try:
                    event = ProgressEvent.from_dict(msg.payload)
                    if self._event_queue is not None:
                        try:
                            self._event_queue.put_nowait(event)
                        except queue.Full:
                            logger.warning(
                                "Event queue full — dropping %s event for %s",
                                event.event_type.value, event.filename,
                            )
                except Exception as e:
                    logger.warning("Failed to translate event message: %s", e)

            elif msg.is_result:
                self._result = msg.payload
                # Emit COMPLETE event from the result
                self._emit_complete_from_result(msg.payload)
                break

            elif msg.is_error:
                self._result = msg.payload
                # Emit ERROR event
                self._emit_error_from_payload(msg.payload)
                break

            elif msg.is_cancelled:
                self._result = msg.payload
                # Emit CANCELLED event
                self._emit_cancelled_from_payload(msg.payload)
                break

    def _synthesize_result_if_missing(self) -> None:
        """If the worker process exited without sending a terminal message
        (e.g. a segfault, OOM kill, or unhandled exception in the worker
        function that bypassed ``_handle_worker_exception``), synthesize
        an error result so the caller's ``wait()`` returns instead of
        looping forever.

        Without this, the parent thread in ``download_snapshot_streaming``
        (or any other caller of ``runner.wait(timeout=...)``) would loop
        indefinitely on ``result is None``, hiding the real failure
        behind a "stuck" UI.
        """
        if self._result is not None:
            return
        exitcode = None
        try:
            if self._process is not None:
                exitcode = self._process.exitcode
        except Exception:
            pass
        # Negative exit codes indicate the process was killed by a signal
        # (e.g. -9 for SIGKILL, -11 for SIGSEGV). 0 means clean exit but
        # no result — should not normally happen, but treat as an error.
        if exitcode is not None and exitcode < 0:
            msg = (
                f"Subprocess exited with signal {-exitcode} before sending "
                f"a terminal message (likely killed externally, e.g. OOM "
                f"or user SIGKILL)"
            )
        elif exitcode == 0:
            msg = (
                "Subprocess exited cleanly without sending a result message "
                "(possible bug in the worker function)"
            )
        else:
            msg = (
                f"Subprocess exited with code {exitcode} before sending a "
                f"terminal message"
            )
        logger.error(msg)
        # Build a synthetic error payload and emit it through the same
        # path that real errors take.
        payload = {
            "message": msg,
            "error_type": "SubprocessDied",
            "retryable": False,
            "exitcode": exitcode,
        }
        self._result = payload
        self._emit_error_from_payload(payload)

    def _drain_queue(self) -> None:
        """Drain any remaining messages from mp_queue after process exits."""
        while True:
            try:
                msg = self._mp_queue.get_nowait()  # type: ignore[union-attr]
                if msg.is_event:
                    try:
                        event = ProgressEvent.from_dict(msg.payload)
                        if self._event_queue is not None:
                            self._event_queue.put_nowait(event)
                    except Exception:
                        pass
                elif msg.is_terminal:
                    self._result = msg.payload
                    if msg.is_result:
                        self._emit_complete_from_result(msg.payload)
                    elif msg.is_error:
                        self._emit_error_from_payload(msg.payload)
                    elif msg.is_cancelled:
                        self._emit_cancelled_from_payload(msg.payload)
                    break
            except Exception:
                break

    def _emit_complete_from_result(self, payload: Dict[str, Any]) -> None:
        """Emit a COMPLETE ProgressEvent from a result payload.

        For snapshot downloads, the payload may include
        ``bytes_completed``, ``total_bytes``, ``files_completed``,
        and ``total_files`` fields from the subprocess worker's
        ``state_manager``.  For single-file downloads, only
        ``file_size`` is available.
        """
        if self._event_queue is None:
            return
        try:
            from ..types import ProgressPhase, TransferDirection
            # Prefer explicit bytes_completed/total_bytes from snapshot
            # workers; fall back to file_size for single-file workers.
            bytes_completed = payload.get("bytes_completed", 0) or payload.get("file_size", 0)
            total_bytes = payload.get("total_bytes", 0) or bytes_completed
            event = ProgressEvent(
                event_type=EventType.COMPLETE,
                transfer_id=payload.get("transfer_id", ""),
                direction=TransferDirection(payload.get("direction", "download")),
                filename=payload.get("filename", ""),
                phase=ProgressPhase.COMPLETE,
                bytes_completed=bytes_completed,
                total_bytes=total_bytes,
                percentage=100.0,
                file_index=payload.get("files_completed", 0),
                total_files=payload.get("total_files", 0),
            )
            self._event_queue.put(event)
        except Exception as e:
            logger.warning("Failed to emit COMPLETE event: %s", e)

    def _emit_error_from_payload(self, payload: Dict[str, Any]) -> None:
        """Emit an ERROR ProgressEvent from an error payload."""
        if self._event_queue is None:
            return
        try:
            from ..types import ProgressPhase, TransferDirection
            event = ProgressEvent(
                event_type=EventType.ERROR,
                transfer_id=payload.get("transfer_id", ""),
                direction=TransferDirection(payload.get("direction", "download")),
                filename=payload.get("filename", ""),
                phase=ProgressPhase.ERROR,
                error=TransferError(
                    message=payload.get("message", "Unknown error"),
                    error_type=payload.get("error_type", "Exception"),
                    retryable=payload.get("retryable", False),
                ),
            )
            self._event_queue.put(event)
        except Exception as e:
            logger.warning("Failed to emit ERROR event: %s", e)

    def _emit_cancelled_from_payload(self, payload: Dict[str, Any]) -> None:
        """Emit a CANCELLED ProgressEvent from a cancelled payload."""
        if self._event_queue is None:
            return
        try:
            event = ProgressEvent.cancelled_event(
                transfer_id=payload.get("transfer_id", ""),
                direction=payload.get("direction", "download"),
                filename=payload.get("filename", ""),
                bytes_completed=payload.get("bytes_completed", 0),
                total_bytes=payload.get("total_bytes", 0),
            )
            self._event_queue.put(event)
        except Exception as e:
            logger.warning("Failed to emit CANCELLED event: %s", e)

    def request_cancel(self) -> None:
        """Set the cancel_event without terminating the process.

        The child worker observes the event between chunks and breaks
        out of the loop cooperatively. If the child is GIL-stalled
        (e.g. blocked in ``XetDownloadStream.__next__``), the caller
        should follow up with a hard ``terminate(grace=2.0)`` after a
        short grace period.

        See plan ``docs/plans/2026-06-05-xet-streaming-flush-reliability.md``
        (Step 3, layer L3) for the full rationale.

        Always safe to call — no-op if no process is running.
        """
        with self._lock:
            if self._cancel_event is not None:
                try:
                    self._cancel_event.set()
                except Exception:
                    pass

    def terminate(self, grace: Optional[float] = None) -> None:
        """Terminate the child process.

        Two-phase termination (plan 2026-06-05 step 4):

        1. **Cooperative phase** (``grace`` > 0): set the cancel_event
           and wait up to ``grace`` seconds for the child to exit on
           its own. The child breaks out of its loop cooperatively on
           the next chunk boundary. If the child exits within
           ``grace``, no SIGTERM is sent.

        2. **Hard phase**: send SIGTERM, wait ``terminate_timeout``,
           escalate to SIGKILL if necessary. This is the original
           terminate() behavior, used as a fallback when the child is
           GIL-stalled (e.g. blocked in ``XetDownloadStream.__next__``)
           or otherwise unable to observe the cancel_event.

        Args:
            grace: Seconds to wait for cooperative exit. If None
                (default), the cooperative phase is skipped and the
                hard phase is used immediately (preserves original
                behavior for existing callers).

        Always safe to call — no-op if no process is running.
        """
        with self._lock:
            # Stop the relay thread (always, regardless of which phase)
            self._stop_event.set()

            if self._process is not None:
                # === Phase 1: cooperative exit (only if grace > 0) ===
                if grace and grace > 0:
                    # Signal cancellation first
                    if self._cancel_event is not None:
                        try:
                            self._cancel_event.set()
                        except Exception:
                            pass
                    if self._process.is_alive():
                        self._process.join(timeout=grace)
                    if not self._process.is_alive():
                        # Child exited cooperatively — clean up and return.
                        if self._relay_thread is not None and self._relay_thread.is_alive():
                            self._relay_thread.join(timeout=2.0)
                        self._process = None
                        self._mp_queue = None
                        self._cancel_event = None
                        self._relay_thread = None
                        return
                    # Child is still alive after grace period — fall through
                    # to the hard phase (SIGTERM).

                # === Phase 2: hard terminate (SIGTERM → SIGKILL) ===
                # If grace was provided, the cancel_event was already
                # set in phase 1; if grace was None, set it now so the
                # child can still observe the cancel between chunks
                # before SIGTERM lands.
                if self._cancel_event is not None and not (grace and grace > 0):
                    try:
                        self._cancel_event.set()
                    except Exception:
                        pass

                if self._process.is_alive():
                    logger.debug("Terminating subprocess pid=%d", self._process.pid)
                    self._process.terminate()
                    self._process.join(timeout=self._terminate_timeout)

                    if self._process.is_alive():
                        logger.warning(
                            "Subprocess pid=%d did not terminate in %.1fs — killing",
                            self._process.pid,
                            self._terminate_timeout,
                        )
                        self._process.kill()
                        self._process.join(timeout=self._kill_timeout)

                    if self._process.is_alive():
                        logger.error(
                            "Subprocess pid=%d could not be killed!",
                            self._process.pid,
                        )

                # Wait for relay thread to finish
                if self._relay_thread is not None and self._relay_thread.is_alive():
                    self._relay_thread.join(timeout=2.0)

                # Clean up process reference
                self._process = None

            # Clean up mp objects
            self._mp_queue = None
            self._cancel_event = None
            self._relay_thread = None

    def is_alive(self) -> bool:
        """Check if the child process is still running.

        Returns:
            True if the process exists and is alive, False otherwise.
        """
        with self._lock:
            return self._process is not None and self._process.is_alive()

    def wait(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Wait for the worker to complete and return the result.

        Blocks until the relay thread receives a terminal message
        (result/error/cancelled) or the timeout expires.

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            Result payload dict on success/error/cancelled, or None on timeout.
        """
        if self._relay_thread is None:
            return self._result

        self._relay_thread.join(timeout=timeout)

        if self._relay_thread.is_alive():
            # Timeout expired
            return None

        return self._result

    @property
    def pid(self) -> Optional[int]:
        """Process ID of the child, or None if not started."""
        with self._lock:
            if self._process is not None:
                return self._process.pid
            return None

    @property
    def exitcode(self) -> Optional[int]:
        """Exit code of the child process, or None if still running."""
        with self._lock:
            if self._process is not None:
                return self._process.exitcode
            return None

    def __del__(self) -> None:
        """Warn if process wasn't properly joined."""
        if self._process is not None and self._process.is_alive():
            warnings.warn(
                f"XetSubprocessRunner with pid={self._process.pid} was not "
                f"properly terminated. Call terminate() to avoid zombie processes.",
                ResourceWarning,
                stacklevel=1,
            )
