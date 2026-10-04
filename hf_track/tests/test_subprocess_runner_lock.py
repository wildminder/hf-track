"""CRIT-012 and IMP-019: lock scope and event coalescing in the runner.

``terminate()`` was restructured so the runner lock is held only long
enough to snapshot the child references; every ``join()`` runs outside it.
Before that change the lock was held across the cooperative join, the
SIGTERM join, the SIGKILL join and the relay-thread join, which blocked
``is_alive()``, ``request_cancel()``, ``pid`` and ``exitcode`` for the
whole termination window. A generation counter makes the post-join
cleanup skip stale work so a concurrent ``start()`` is not clobbered.

The same pass replaced "log and discard on a full queue" with an evicting
enqueue, because silently dropping a terminal event leaves a progress bar
stuck forever.

These tests drive the real methods with doubles for the child process and
the relay thread. No subprocess is spawned: the point is the lock scope,
which is only observable if the join blocks.
"""

from __future__ import annotations

import queue
import threading
import time

import pytest

from hf_track.subprocess.messages import SubprocessMessage
from hf_track.subprocess.runner import XetSubprocessRunner
from hf_track.types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferDirection,
)


class _FakeProcess:
    """Child-process double whose ``join`` blocks for a controlled time.

    ``join_delay`` must stay well above the probe timeouts used below: the
    tests assert that a competing thread is blocked out *for longer than
    the join lasts*. A delay shorter than the probe window would let a
    lock-holding ``terminate()`` pass by simply winning the race.
    """

    JOIN_DELAY = 3.0

    def __init__(self, *, exit_after_join: bool = False, join_delay: float = JOIN_DELAY):
        self.pid = 4242
        self._alive = True
        self._exit_after_join = exit_after_join
        self._join_delay = join_delay
        self.terminated = False
        self.killed = False
        self.join_calls: list = []
        self.join_started = threading.Event()
        self.join_finished = threading.Event()

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout=None):
        self.join_calls.append(timeout)
        self.join_started.set()
        time.sleep(self._join_delay)
        if self._exit_after_join:
            self._alive = False
        self.join_finished.set()

    def terminate(self):
        self.terminated = True
        self._alive = False

    def kill(self):
        self.killed = True
        self._alive = False


class _FakeThread:
    """Relay-thread double that is never actually alive."""

    def __init__(self):
        self.join_calls: list = []

    def is_alive(self) -> bool:
        return False

    def join(self, timeout=None):
        self.join_calls.append(timeout)


def _progress_event(transfer_id: str = "t1", filename: str = "a.bin") -> ProgressEvent:
    return ProgressEvent(
        event_type=EventType.PROGRESS,
        transfer_id=transfer_id,
        direction=TransferDirection.DOWNLOAD,
        filename=filename,
        phase=ProgressPhase.DOWNLOADING,
        bytes_completed=1,
        total_bytes=10,
        percentage=10.0,
    )


def _terminal_event(event_type=EventType.COMPLETE, transfer_id: str = "t1") -> ProgressEvent:
    return ProgressEvent(
        event_type=event_type,
        transfer_id=transfer_id,
        direction=TransferDirection.DOWNLOAD,
        filename="a.bin",
        phase=ProgressPhase.COMPLETE,
        percentage=100.0 if event_type is EventType.COMPLETE else 0.0,
    )


class TestTerminateLockScope:
    """The lock must not be held while terminate() waits on the child."""

    def test_terminate_releases_lock_before_joins(self):
        runner = XetSubprocessRunner()
        proc = _FakeProcess()
        runner._process = proc
        runner._cancel_event = threading.Event()
        runner._relay_thread = _FakeThread()
        runner._generation = 7

        worker = threading.Thread(target=runner.terminate)
        worker.start()

        # Wait until terminate() is inside its first join, then prove the
        # lock is free by taking it from another thread. The probe gives up
        # well before the join finishes, so only an actually-free lock wins.
        assert proc.join_started.wait(timeout=5), "terminate() never reached a join"
        acquired = threading.Event()

        def _try_lock():
            with runner._lock:
                acquired.set()

        locker = threading.Thread(target=_try_lock, daemon=True)
        locker.start()
        try:
            assert acquired.wait(timeout=1.0), (
                "runner._lock was still held while terminate() joined -- "
                "is_alive()/request_cancel() would block for the whole window"
            )
        finally:
            worker.join(timeout=30)
            locker.join(timeout=5)

        assert proc.join_finished.is_set()

    def test_accessors_do_not_block_during_terminate(self):
        """is_alive() and request_cancel() must answer while terminating."""
        runner = XetSubprocessRunner()
        proc = _FakeProcess()
        runner._process = proc
        runner._cancel_event = threading.Event()
        runner._relay_thread = _FakeThread()

        worker = threading.Thread(target=runner.terminate)
        worker.start()
        try:
            assert proc.join_started.wait(timeout=5)
            answered = []

            def _probe():
                answered.append(runner.is_alive())
                runner.request_cancel()

            prober = threading.Thread(target=_probe, daemon=True)
            prober.start()
            prober.join(timeout=1.0)
            assert answered, (
                "is_alive() blocked for the whole termination window"
            )
        finally:
            worker.join(timeout=30)

        assert proc.join_finished.is_set()

    def test_terminate_is_a_noop_without_a_process(self):
        runner = XetSubprocessRunner()
        runner.terminate()
        runner.terminate(grace=1.0)

    def test_terminate_clears_references_on_completion(self):
        runner = XetSubprocessRunner()
        proc = _FakeProcess(exit_after_join=True, join_delay=0.0)
        runner._process = proc
        runner._cancel_event = threading.Event()
        runner._relay_thread = _FakeThread()

        runner.terminate(grace=1.0)

        assert runner._process is None
        assert runner._cancel_event is None
        assert runner._relay_thread is None

    def test_concurrent_start_is_not_clobbered_by_stale_release(self):
        """A start() during terminate() must survive the cleanup."""
        runner = XetSubprocessRunner()
        proc = _FakeProcess(join_delay=0.6)
        runner._process = proc
        runner._cancel_event = threading.Event()
        runner._relay_thread = _FakeThread()
        stale_generation = runner._generation

        worker = threading.Thread(target=runner.terminate)
        worker.start()
        try:
            assert proc.join_started.wait(timeout=5)
            # Simulate start() landing mid-termination: it bumps the
            # generation and installs its own child.
            fresh_process = _FakeProcess()
            with runner._lock:
                runner._generation += 1
                runner._process = fresh_process
        finally:
            worker.join(timeout=15)

        assert runner._generation != stale_generation
        assert runner._process is fresh_process, (
            "terminate() cleared references belonging to the newer process"
        )
        fresh_process.terminate()
        runner.terminate()


class TestRelayCoalescing:
    """A full consumer queue must cost a progress event, never a terminal one."""

    def test_progress_evicts_superseded_progress(self):
        runner = XetSubprocessRunner()
        runner._event_queue = queue.Queue(maxsize=1)
        runner._enqueue_event(_progress_event(filename="old.bin"))

        runner._enqueue_event(_progress_event(filename="new.bin"))

        queued = runner._event_queue.get_nowait()
        assert queued.filename == "new.bin", (
            "the latest progress report should replace the stale one"
        )

    def test_terminal_event_evicts_queued_progress(self):
        runner = XetSubprocessRunner()
        runner._event_queue = queue.Queue(maxsize=1)
        runner._enqueue_event(_progress_event())

        runner._enqueue_event(_terminal_event(EventType.COMPLETE))

        queued = runner._event_queue.get_nowait()
        assert queued.event_type is EventType.COMPLETE, (
            "a terminal event must never be dropped -- it strands the UI"
        )

    def test_terminal_event_is_kept_over_an_unrelated_terminal(self):
        """Queue head wins when it is not a supersedable progress event."""
        runner = XetSubprocessRunner()
        runner._event_queue = queue.Queue(maxsize=1)
        runner._enqueue_event(_terminal_event(EventType.COMPLETE, transfer_id="other"))

        runner._enqueue_event(_progress_event(transfer_id="t1"))

        queued = runner._event_queue.get_nowait()
        assert queued.event_type is EventType.COMPLETE
        assert queued.transfer_id == "other"

    def test_unrelated_progress_is_not_evicted(self):
        """Progress for a different transfer is not superseded."""
        runner = XetSubprocessRunner()
        runner._event_queue = queue.Queue(maxsize=1)
        runner._enqueue_event(_progress_event(transfer_id="t1"))

        runner._enqueue_event(_progress_event(transfer_id="t2"))

        queued = runner._event_queue.get_nowait()
        assert queued.transfer_id == "t1", (
            "progress for another transfer was evicted instead of dropped"
        )

    def test_enqueue_without_queue_is_a_noop(self):
        runner = XetSubprocessRunner()
        runner._enqueue_event(_progress_event())

    def test_enqueue_never_blocks_on_a_full_queue(self):
        runner = XetSubprocessRunner()
        runner._event_queue = queue.Queue(maxsize=1)
        runner._enqueue_event(_terminal_event(EventType.COMPLETE, transfer_id="other"))

        finished = threading.Event()

        def _push():
            runner._enqueue_event(_progress_event(transfer_id="t2"))
            finished.set()

        thread = threading.Thread(target=_push, daemon=True)
        thread.start()
        assert finished.wait(timeout=2.0), "_enqueue_event blocked on a full queue"
        thread.join(timeout=2)


class TestCancelledPayloadIdentity:
    """The runner rebuilds a CANCELLED event from payload keys."""

    def test_cancelled_payload_carries_transfer_identity(self):
        msg = SubprocessMessage.cancelled(
            message="stopped",
            transfer_id="t-42",
            direction="upload",
            filename="weights.bin",
        )

        assert msg.payload["transfer_id"] == "t-42"
        assert msg.payload["direction"] == "upload"
        assert msg.payload["filename"] == "weights.bin"

    def test_emit_cancelled_preserves_identity(self):
        runner = XetSubprocessRunner()
        runner._event_queue = queue.Queue()

        runner._emit_cancelled_from_payload({
            "status": "cancelled",
            "message": "stopped",
            "transfer_id": "t-42",
            "direction": "download",
            "filename": "weights.bin",
            "bytes_completed": 5,
            "total_bytes": 10,
        })

        event = runner._event_queue.get_nowait()
        assert event.event_type is EventType.CANCELLED
        assert event.transfer_id == "t-42"
        assert event.filename == "weights.bin"
        assert event.direction == TransferDirection.DOWNLOAD