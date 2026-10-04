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

from ..types import EventType, ProgressEvent, TransferErrorInfo
from .messages import MSG_EVENT, SubprocessMessage

logger = logging.getLogger(__name__)


def _signal_cancel(cancel_event: Optional[mp.Event]) -> None:
    """Best-effort ``cancel_event.set()`` on a possibly dead handle.

    The handle belongs to a child that may already have exited, which can
    leave a broken mp primitive behind; a failure here must not abort
    termination.
    """
    if cancel_event is None:
        return
    try:
        cancel_event.set()
    except Exception:
        pass


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
        # Bumped by every ``start()``. ``terminate()`` clears the process
        # references only if the generation it snapshotted is still current,
        # so a ``start()`` issued while ``terminate()`` was joining the old
        # child is not clobbered by that older call (CRIT-012).
        self._generation = 0

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

            self._generation += 1
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
        """Spawn a hybrid FileDownloadGroup + HTTP-fallback subprocess.

        Convenience wrapper that calls ``start()`` with
        ``_xet_file_download_worker`` from ``hf_track._xet_worker``.

        The worker uses
        ``hf_xet.XetSession().new_file_download_group().start_download_file()``
        and falls back to a pure-HTTP download via
        ``hf_track.download.http_fallback.download_file_http`` if the
        xet handle does not complete within ``params["tier_timeout_s"]``
        seconds. Either path delivers an incrementally-growing file to
        disk; cancel via ``request_cancel()``.

        Args:
            params: Picklable dict of parameters for the worker. Must
                include the keys required by ``_xet_file_download_worker``:
                ``file_specs`` (list of dicts with ``hash``, ``file_size``,
                ``dest_path``, ``xet_file_data``, optional ``filename``),
                ``token``, ``endpoint``, ``transfer_id``,
                ``report_interval``, ``request_headers``, ``repo_id``,
                ``repo_type``, ``revision``, ``tier_timeout_s`` (optional,
                default 60), ``enable_http_fallback`` (optional,
                default True), ``use_xet`` (optional, default True),
                ``fsync_interval`` (optional), ``disable_fsync`` (optional).
            event_queue: Main-process queue to receive ``ProgressEvent`` objects.

        Raises:
            RuntimeError: If a process is already running.
        """
        # Local import to avoid a circular dependency at module load time
        # (subprocess_runner is imported by _xet_worker, which is imported by
        # subprocess_messages → runner → worker chain).
        from .._xet_worker import _xet_file_download_worker
        self.start(
            worker_func=_xet_file_download_worker,
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
                    self._enqueue_event(event)
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

    def _enqueue_event(self, event: ProgressEvent) -> None:
        """Put ``event`` on the consumer queue, never blocking the relay.

        A full queue used to mean "log and discard", which silently threw
        away progress updates and -- worse -- a terminal event a final
        progress bar depends on. Instead:

        * a PROGRESS event evicts the superseded PROGRESS event already
          queued for the same transfer, so a slow consumer sees the latest
          byte count rather than a stale one;
        * a terminal event (COMPLETE/ERROR/CANCELLED) evicts whatever is
          queued, because dropping one leaves the UI stuck forever.

        Only when neither eviction applies -- a full queue of unrelated
        events -- is the new event dropped, as before.
        """
        target = self._event_queue
        if target is None:
            return

        try:
            target.put_nowait(event)
            return
        except queue.Full:
            pass

        evicted = self._evict_for(event)
        if not evicted:
            logger.warning(
                "Event queue full — dropping %s event for %s",
                event.event_type.value, event.filename,
            )
            return
        try:
            target.put_nowait(event)
        except queue.Full:
            logger.warning(
                "Event queue refilled — dropping %s event for %s",
                event.event_type.value, event.filename,
            )

    def _evict_for(self, event: ProgressEvent) -> bool:
        """Free one queue slot for ``event`` if the queued event may be lost.

        Returns True when a slot was freed (or was not needed), False when
        the queue head must be preserved and the caller should drop.
        """
        target = self._event_queue
        if target is None:
            return False
        try:
            queued = target.get_nowait()
        except queue.Empty:
            return True

        if event.event_type is not EventType.PROGRESS:
            # Terminal event: the consumer is waiting on this one, so the
            # queued event is the one that goes.
            return True
        if (
            queued.event_type is EventType.PROGRESS
            and queued.transfer_id == event.transfer_id
        ):
            # ``queued`` is a superseded progress report for the same transfer.
            return True
        # Unrelated event: put the slot back and report "no slot".
        try:
            target.put_nowait(queued)
        except queue.Full:  # pragma: no cover - the slot we took is ours
            logger.warning("Lost a %s event for %s while coalescing",
                           queued.event_type.value, queued.filename)
        return False

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
                        self._enqueue_event(event)
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
            self._enqueue_event(event)
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
                error=TransferErrorInfo(
                    message=payload.get("message", "Unknown error"),
                    error_type=payload.get("error_type", "Exception"),
                    retryable=payload.get("retryable", False),
                ),
            )
            self._enqueue_event(event)
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
            self._enqueue_event(event)
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

        The lock is held only while snapshotting the child references and
        setting the stop event; the joins run outside it. Holding it across
        the joins blocked ``pid``, ``exitcode``, ``is_alive()`` and
        ``request_cancel()`` for the whole termination window, including the
        ``terminate_timeout``/``kill_timeout`` waits.
        """
        with self._lock:
            self._stop_event.set()
            process = self._process
            cancel_event = self._cancel_event
            relay_thread = self._relay_thread
            generation = self._generation

        if process is not None:
            # === Phase 1: cooperative exit (only if grace > 0) ===
            cooperative_exit = False
            if grace and grace > 0:
                # Signal cancellation first
                _signal_cancel(cancel_event)
                if process.is_alive():
                    process.join(timeout=grace)
                cooperative_exit = not process.is_alive()

            if not cooperative_exit:
                # === Phase 2: hard terminate (SIGTERM → SIGKILL) ===
                # If grace was provided, the cancel_event was already
                # set in phase 1; if grace was None, set it now so the
                # child can still observe the cancel between chunks
                # before SIGTERM lands.
                if not (grace and grace > 0):
                    _signal_cancel(cancel_event)

                if process.is_alive():
                    logger.debug("Terminating subprocess pid=%d", process.pid)
                    process.terminate()
                    process.join(timeout=self._terminate_timeout)

                    if process.is_alive():
                        logger.warning(
                            "Subprocess pid=%d did not terminate in %.1fs — killing",
                            process.pid,
                            self._terminate_timeout,
                        )
                        process.kill()
                        process.join(timeout=self._kill_timeout)

                    if process.is_alive():
                        logger.error(
                            "Subprocess pid=%d could not be killed!",
                            process.pid,
                        )

        # Wait for relay thread to finish
        if relay_thread is not None and relay_thread.is_alive():
            relay_thread.join(timeout=2.0)

        self._release(generation)

    def _release(self, generation: int) -> None:
        """Drop the child references captured at ``generation``.

        The second, short acquisition is what makes a concurrent
        ``start()`` safe: it bumps ``_generation``, so this call sees the
        mismatch and leaves the freshly started process alone instead of
        clearing references that now belong to it.
        """
        with self._lock:
            if generation != self._generation:
                logger.debug(
                    "Skipping stale cleanup: generation %d superseded by %d",
                    generation,
                    self._generation,
                )
                return
            self._process = None
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
