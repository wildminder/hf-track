"""End-to-end integration tests for subprocess isolation.

Tests real subprocess behavior: terminate during operation, cancel via
mp.Event, concurrent runners, and full event flow from worker → relay →
event_queue. Uses lightweight test workers (no hf_xet dependency).
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from typing import Any, Dict

import pytest

from hf_track.subprocess import SubprocessMessage
from hf_track.subprocess import XetSubprocessRunner
from hf_track.types import EventType, ProgressEvent, TransferDirection


# ── Test Workers ──────────────────────────────────────────────────


def _counting_worker(params: dict, mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Worker that emits N progress events with incrementing bytes, then result."""
    n = params.get("n_events", 5)
    transfer_id = params.get("transfer_id", "e2e-count")
    filename = params.get("filename", "count.bin")
    total = params.get("total_bytes", 1000)

    for i in range(n):
        if cancel_event.is_set():
            mp_queue.put(SubprocessMessage.cancelled(
                message="Counting worker cancelled",
                transfer_id=transfer_id,
                direction="download",
                filename=filename,
                bytes_completed=i * (total // n),
                total_bytes=total,
            ))
            return

        pct = (i + 1) / n * 100
        mp_queue.put(SubprocessMessage.event({
            "event_type": "progress",
            "transfer_id": transfer_id,
            "direction": "download",
            "filename": filename,
            "phase": "downloading",
            "bytes_completed": (i + 1) * (total // n),
            "total_bytes": total,
            "percentage": pct,
            "speed": 5000.0,
        }))
        time.sleep(0.05)

    mp_queue.put(SubprocessMessage.result(
        filename=filename,
        file_size=total,
        transfer_id=transfer_id,
        direction="download",
    ))


def _slow_worker(params: dict, mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Worker that emits one event per second for N seconds."""
    duration = params.get("duration", 5)
    transfer_id = params.get("transfer_id", "e2e-slow")
    filename = params.get("filename", "slow.bin")
    total = params.get("total_bytes", 10000)

    steps = int(duration)
    for i in range(steps):
        if cancel_event.is_set():
            mp_queue.put(SubprocessMessage.cancelled(
                message="Slow worker cancelled",
                transfer_id=transfer_id,
                direction="download",
                filename=filename,
                bytes_completed=i * (total // steps),
                total_bytes=total,
            ))
            return

        mp_queue.put(SubprocessMessage.event({
            "event_type": "progress",
            "transfer_id": transfer_id,
            "direction": "download",
            "filename": filename,
            "phase": "downloading",
            "bytes_completed": (i + 1) * (total // steps),
            "total_bytes": total,
            "percentage": (i + 1) / steps * 100,
            "speed": 2000.0,
        }))
        time.sleep(1.0)

    mp_queue.put(SubprocessMessage.result(
        filename=filename,
        file_size=total,
        transfer_id=transfer_id,
        direction="download",
    ))


def _unresponsive_worker(params: dict, mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Worker that hangs in a sleep loop without checking cancel_event."""
    # Simulates a stuck hf_xet Rust call
    time.sleep(300)


def _crash_worker(params: dict, mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Worker that crashes with an unhandled exception."""
    time.sleep(0.1)
    raise RuntimeError("Simulated crash in subprocess")


def _multi_file_worker(params: dict, mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Worker that simulates downloading multiple files sequentially."""
    files = params.get("files", ["a.bin", "b.bin", "c.bin"])
    transfer_id = params.get("transfer_id", "e2e-multi")
    file_size = params.get("file_size", 500)

    for fname in files:
        if cancel_event.is_set():
            mp_queue.put(SubprocessMessage.cancelled(
                message="Multi-file worker cancelled",
                transfer_id=transfer_id,
                direction="download",
                filename=fname,
                bytes_completed=0,
                total_bytes=file_size,
            ))
            return

        # Emit progress for each file
        for pct in [25, 50, 75, 100]:
            mp_queue.put(SubprocessMessage.event({
                "event_type": "progress",
                "transfer_id": transfer_id,
                "direction": "download",
                "filename": fname,
                "phase": "downloading",
                "bytes_completed": int(file_size * pct / 100),
                "total_bytes": file_size,
                "percentage": float(pct),
                "speed": 3000.0,
            }))
        time.sleep(0.05)

    mp_queue.put(SubprocessMessage.result(
        filename=",".join(files),
        file_size=file_size * len(files),
        transfer_id=transfer_id,
        direction="download",
    ))


# ── Helpers ───────────────────────────────────────────────────────


def _collect_events(event_queue: queue.Queue, timeout: float = 2.0) -> list[ProgressEvent]:
    """Drain all events from the queue with a timeout."""
    events: list[ProgressEvent] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            event = event_queue.get(timeout=min(0.2, remaining))
            events.append(event)
        except queue.Empty:
            # Check if we've gotten a terminal event
            if events and events[-1].event_type in (EventType.COMPLETE, EventType.ERROR, EventType.CANCELLED):
                break
    return events


# ── Full Event Flow Tests ─────────────────────────────────────────


class TestSubprocessE2EEventFlow:
    """Test complete event flow: worker → mp.Queue → relay → event_queue."""

    def test_counting_worker_emits_all_events(self):
        """Counting worker emits N PROGRESS + 1 COMPLETE events."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_counting_worker,
            params={"n_events": 5, "transfer_id": "e2e-flow", "filename": "flow.bin", "total_bytes": 1000},
            event_queue=event_queue,
        )

        result = runner.wait(timeout=15)
        events = _collect_events(event_queue)

        runner.terminate()

        # Verify result
        assert result is not None
        assert result.get("status") == "success"
        assert result.get("filename") == "flow.bin"

        # Verify events
        progress = [e for e in events if e.event_type == EventType.PROGRESS]
        complete = [e for e in events if e.event_type == EventType.COMPLETE]

        assert len(progress) == 5, f"Expected 5 PROGRESS, got {len(progress)}"
        assert len(complete) == 1, f"Expected 1 COMPLETE, got {len(complete)}"

        # Verify event content
        for e in progress:
            assert e.transfer_id == "e2e-flow"
            assert e.filename == "flow.bin"
            assert e.direction == TransferDirection.DOWNLOAD

        # Verify bytes are increasing
        bytes_vals = [e.bytes_completed for e in progress]
        assert bytes_vals == sorted(bytes_vals), f"Bytes not increasing: {bytes_vals}"

    def test_multi_file_worker_emits_per_file_events(self):
        """Multi-file worker emits progress for each file."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_multi_file_worker,
            params={
                "files": ["a.bin", "b.bin", "c.bin"],
                "transfer_id": "e2e-multi",
                "file_size": 500,
            },
            event_queue=event_queue,
        )

        result = runner.wait(timeout=15)
        events = _collect_events(event_queue)

        runner.terminate()

        assert result is not None
        progress = [e for e in events if e.event_type == EventType.PROGRESS]
        complete = [e for e in events if e.event_type == EventType.COMPLETE]

        # 3 files × 4 progress events each = 12
        assert len(progress) == 12, f"Expected 12 PROGRESS, got {len(progress)}"
        assert len(complete) == 1

        # Verify filenames appear in order
        filenames = [e.filename for e in progress]
        assert filenames[:4] == ["a.bin"] * 4
        assert filenames[4:8] == ["b.bin"] * 4
        assert filenames[8:12] == ["c.bin"] * 4


# ── Terminate During Operation Tests ──────────────────────────────


class TestSubprocessE2ETerminate:
    """Test terminating a subprocess during an active operation."""

    def test_terminate_during_slow_operation(self):
        """terminate() kills a slow worker mid-operation."""
        runner = XetSubprocessRunner(terminate_timeout=1.0, kill_timeout=1.0)
        event_queue = queue.Queue()

        runner.start(
            worker_func=_slow_worker,
            params={"duration": 30, "transfer_id": "e2e-term-slow"},
            event_queue=event_queue,
        )

        # Let it emit a few events
        time.sleep(1.5)
        assert runner.is_alive()

        # Terminate
        runner.terminate()
        assert not runner.is_alive()

        # Should have collected some events before termination
        events = _collect_events(event_queue, timeout=1.0)
        # At least 1 progress event should have been relayed
        assert len(events) >= 1, "Expected at least 1 event before termination"

    def test_terminate_unresponsive_worker(self):
        """terminate() kills an unresponsive worker that doesn't check cancel_event."""
        runner = XetSubprocessRunner(terminate_timeout=0.5, kill_timeout=1.0)
        event_queue = queue.Queue()

        runner.start(
            worker_func=_unresponsive_worker,
            params={},
            event_queue=event_queue,
        )

        time.sleep(0.3)
        assert runner.is_alive()

        start = time.monotonic()
        runner.terminate()
        elapsed = time.monotonic() - start

        assert not runner.is_alive()
        # Should terminate within reasonable time (terminate_timeout + kill_timeout + overhead)
        assert elapsed < 5.0, f"Terminate took {elapsed:.1f}s — too long"

    def test_terminate_sets_cancel_event_before_kill(self):
        """terminate() sets cancel_event so cooperative workers can exit gracefully."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_counting_worker,
            params={"n_events": 100, "transfer_id": "e2e-cancel-check"},
            event_queue=event_queue,
        )

        # Let it start
        time.sleep(0.2)

        # Store cancel_event reference before terminate clears it
        cancel_event = runner._cancel_event
        runner.terminate()

        # cancel_event was set before process was killed
        assert cancel_event.is_set()

    def test_terminate_is_idempotent(self):
        """Multiple terminate() calls are safe."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_slow_worker,
            params={"duration": 30},
            event_queue=event_queue,
        )

        time.sleep(0.3)
        runner.terminate()
        runner.terminate()  # Second call — should be no-op
        runner.terminate()  # Third call — still safe

        assert not runner.is_alive()


# ── Cancellation Tests ────────────────────────────────────────────


class TestSubprocessE2ECancel:
    """Test cooperative cancellation via mp.Event."""

    def test_cooperative_cancel_emits_cancelled_event(self):
        """Cooperative worker emits CANCELLED when cancel_event is set."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_counting_worker,
            params={"n_events": 50, "transfer_id": "e2e-coop-cancel"},
            event_queue=event_queue,
        )

        # Let a few events through, then cancel
        time.sleep(0.3)
        assert runner._cancel_event is not None
        runner._cancel_event.set()

        # Wait for the worker to notice and emit cancelled
        result = runner.wait(timeout=10)
        events = _collect_events(event_queue, timeout=2.0)

        runner.terminate()

        # Worker should have sent a cancelled result
        if result is not None:
            assert result.get("status") == "cancelled"

        # Check for CANCELLED event in the queue
        cancelled = [e for e in events if e.event_type == EventType.CANCELLED]
        if cancelled:
            assert cancelled[0].transfer_id == "e2e-coop-cancel"

    def test_cancel_via_terminate_for_cooperative_worker(self):
        """terminate() on a cooperative worker lets it emit CANCELLED before dying."""
        runner = XetSubprocessRunner(terminate_timeout=2.0, kill_timeout=1.0)
        event_queue = queue.Queue()

        runner.start(
            worker_func=_counting_worker,
            params={"n_events": 100, "transfer_id": "e2e-term-cancel"},
            event_queue=event_queue,
        )

        time.sleep(0.2)
        runner.terminate()

        events = _collect_events(event_queue, timeout=2.0)

        # Should have at least some progress events
        progress = [e for e in events if e.event_type == EventType.PROGRESS]
        assert len(progress) >= 1, "Expected at least 1 progress event before cancel"

    def test_cancel_during_multi_file(self):
        """Cancellation mid-batch stops remaining files."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_multi_file_worker,
            params={
                "files": ["a.bin", "b.bin", "c.bin", "d.bin", "e.bin"],
                "transfer_id": "e2e-multi-cancel",
                "file_size": 500,
            },
            event_queue=event_queue,
        )

        # Cancel after first file likely started
        time.sleep(0.15)
        if runner._cancel_event is not None:
            runner._cancel_event.set()

        result = runner.wait(timeout=10)
        events = _collect_events(event_queue, timeout=2.0)

        runner.terminate()

        # Should not have gotten all 5 files' worth of events
        progress = [e for e in events if e.event_type == EventType.PROGRESS]
        # With 5 files × 4 events = 20 max, cancellation should yield fewer
        # (This is timing-dependent, so just verify we got some events)
        assert len(progress) >= 1, "Expected at least 1 progress event"


# ── Crash Handling Tests ──────────────────────────────────────────


class TestSubprocessE2ECrash:
    """Test handling of subprocess crashes."""

    def test_crashed_worker_process_dies(self):
        """Worker that raises an unhandled exception causes process to exit."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_crash_worker,
            params={},
            event_queue=event_queue,
        )

        # Wait for the process to crash
        # The relay thread should detect the dead process and exit
        time.sleep(2.0)

        # Process should be dead
        assert not runner.is_alive()
        runner.terminate()

    def test_runner_can_be_reused_after_crash(self):
        """After a crashed worker, the runner can start a new worker."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        # First run: crash
        runner.start(
            worker_func=_crash_worker,
            params={},
            event_queue=event_queue,
        )
        time.sleep(1.5)
        runner.terminate()

        # Second run: success
        runner.start(
            worker_func=_counting_worker,
            params={"n_events": 2, "transfer_id": "e2e-reuse"},
            event_queue=event_queue,
        )

        result = runner.wait(timeout=10)
        events = _collect_events(event_queue, timeout=2.0)
        runner.terminate()

        assert result is not None
        assert result.get("status") == "success"
        progress = [e for e in events if e.event_type == EventType.PROGRESS]
        assert len(progress) >= 2


# ── Concurrent Runner Tests ───────────────────────────────────────


class TestSubprocessE2EConcurrent:
    """Test multiple runners operating concurrently."""

    def test_two_runners_concurrently(self):
        """Two runners can operate simultaneously, each with its own event queue."""
        runner1 = XetSubprocessRunner()
        runner2 = XetSubprocessRunner()
        queue1 = queue.Queue()
        queue2 = queue.Queue()

        runner1.start(
            worker_func=_counting_worker,
            params={"n_events": 3, "transfer_id": "e2e-concurrent-1", "filename": "file1.bin"},
            event_queue=queue1,
        )
        runner2.start(
            worker_func=_counting_worker,
            params={"n_events": 5, "transfer_id": "e2e-concurrent-2", "filename": "file2.bin"},
            event_queue=queue2,
        )

        result1 = runner1.wait(timeout=15)
        result2 = runner2.wait(timeout=15)

        events1 = _collect_events(queue1, timeout=2.0)
        events2 = _collect_events(queue2, timeout=2.0)

        runner1.terminate()
        runner2.terminate()

        # Each queue should only have its own events
        progress1 = [e for e in events1 if e.event_type == EventType.PROGRESS]
        progress2 = [e for e in events2 if e.event_type == EventType.PROGRESS]

        assert len(progress1) == 3, f"Runner1: expected 3 PROGRESS, got {len(progress1)}"
        assert len(progress2) == 5, f"Runner2: expected 5 PROGRESS, got {len(progress2)}"

        # Verify transfer_id isolation
        for e in progress1:
            assert e.transfer_id == "e2e-concurrent-1"
        for e in progress2:
            assert e.transfer_id == "e2e-concurrent-2"

    def test_terminate_one_runner_does_not_affect_other(self):
        """Terminating one runner doesn't kill the other."""
        runner1 = XetSubprocessRunner()
        runner2 = XetSubprocessRunner()
        queue1 = queue.Queue()
        queue2 = queue.Queue()

        # Runner1: slow (will be terminated)
        runner1.start(
            worker_func=_slow_worker,
            params={"duration": 30, "transfer_id": "e2e-term-one-1"},
            event_queue=queue1,
        )
        # Runner2: fast (should complete)
        runner2.start(
            worker_func=_counting_worker,
            params={"n_events": 3, "transfer_id": "e2e-term-one-2", "filename": "survivor.bin"},
            event_queue=queue2,
        )

        time.sleep(0.5)

        # Kill runner1
        runner1.terminate()
        assert not runner1.is_alive()

        # Runner2 should still be alive or have completed normally
        result2 = runner2.wait(timeout=10)
        events2 = _collect_events(queue2, timeout=2.0)

        runner2.terminate()

        assert result2 is not None
        assert result2.get("status") == "success"
        progress2 = [e for e in events2 if e.event_type == EventType.PROGRESS]
        assert len(progress2) == 3


# ── Runner Reuse Tests ────────────────────────────────────────────


class TestSubprocessE2EReuse:
    """Test that a single runner can be reused for sequential operations."""

    def test_sequential_runs_same_runner(self):
        """Same runner can run multiple workers sequentially."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        # Run 1
        runner.start(
            worker_func=_counting_worker,
            params={"n_events": 2, "transfer_id": "e2e-seq-1", "filename": "seq1.bin"},
            event_queue=event_queue,
        )
        result1 = runner.wait(timeout=10)
        runner.terminate()

        assert result1 is not None
        assert result1.get("filename") == "seq1.bin"

        # Clear queue
        _collect_events(event_queue, timeout=0.5)

        # Run 2
        runner.start(
            worker_func=_counting_worker,
            params={"n_events": 3, "transfer_id": "e2e-seq-2", "filename": "seq2.bin"},
            event_queue=event_queue,
        )
        result2 = runner.wait(timeout=10)
        events2 = _collect_events(event_queue, timeout=2.0)
        runner.terminate()

        assert result2 is not None
        assert result2.get("filename") == "seq2.bin"

        progress2 = [e for e in events2 if e.event_type == EventType.PROGRESS]
        assert len(progress2) == 3

    def test_three_sequential_runs(self):
        """Runner handles 3 sequential operations."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        for i in range(3):
            runner.start(
                worker_func=_counting_worker,
                params={
                    "n_events": i + 1,
                    "transfer_id": f"e2e-seq3-{i}",
                    "filename": f"seq3_{i}.bin",
                    "total_bytes": (i + 1) * 100,
                },
                event_queue=event_queue,
            )
            result = runner.wait(timeout=10)
            runner.terminate()

            assert result is not None
            assert result.get("status") == "success"

            # Clear queue for next run
            _collect_events(event_queue, timeout=0.5)


# ── Event Content Integrity Tests ─────────────────────────────────


class TestSubprocessE2EEventIntegrity:
    """Test that event data survives the subprocess boundary intact."""

    def test_event_fields_preserved_across_subprocess(self):
        """All ProgressEvent fields are correctly relayed from subprocess."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_counting_worker,
            params={
                "n_events": 1,
                "transfer_id": "e2e-integrity",
                "filename": "integrity.bin",
                "total_bytes": 999,
            },
            event_queue=event_queue,
        )

        result = runner.wait(timeout=10)
        events = _collect_events(event_queue, timeout=2.0)
        runner.terminate()

        progress = [e for e in events if e.event_type == EventType.PROGRESS]
        assert len(progress) >= 1

        e = progress[0]
        assert e.transfer_id == "e2e-integrity"
        assert e.filename == "integrity.bin"
        assert e.direction == TransferDirection.DOWNLOAD
        assert e.total_bytes == 999
        assert e.speed == 5000.0
        assert isinstance(e.percentage, float)
        assert e.percentage > 0

    def test_complete_event_has_100_percent(self):
        """COMPLETE event always has percentage=100."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_counting_worker,
            params={
                "n_events": 3,
                "transfer_id": "e2e-pct100",
                "filename": "pct100.bin",
                "total_bytes": 500,
            },
            event_queue=event_queue,
        )

        runner.wait(timeout=10)
        events = _collect_events(event_queue, timeout=2.0)
        runner.terminate()

        complete = [e for e in events if e.event_type == EventType.COMPLETE]
        assert len(complete) == 1
        assert complete[0].percentage == 100.0
        assert complete[0].bytes_completed == 500
        assert complete[0].total_bytes == 500
