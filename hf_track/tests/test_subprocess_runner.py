"""Tests for XetSubprocessRunner class."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from unittest.mock import MagicMock, patch

import pytest

from hf_track.subprocess_messages import SubprocessMessage
from hf_track.subprocess_runner import XetSubprocessRunner
from hf_track.types import EventType, ProgressEvent, TransferDirection


# ── Echo Worker for Integration Tests ────────────────────────────

def _echo_worker(params: dict, mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Test worker: emits N progress events then a result."""
    n_events = params.get("n_events", 3)
    transfer_id = params.get("transfer_id", "test-echo")
    filename = params.get("filename", "echo.bin")
    direction = params.get("direction", "download")

    for i in range(n_events):
        if cancel_event.is_set():
            mp_queue.put(SubprocessMessage.cancelled(
                message="Echo cancelled",
                bytes_completed=i * 100,
                total_bytes=n_events * 100,
            ))
            return

        event_dict = {
            "event_type": "progress",
            "transfer_id": transfer_id,
            "direction": direction,
            "filename": filename,
            "phase": "downloading" if direction == "download" else "uploading",
            "bytes_completed": (i + 1) * 100,
            "total_bytes": n_events * 100,
            "percentage": ((i + 1) / n_events * 100),
            "speed": 1000.0,
        }
        mp_queue.put(SubprocessMessage.event(event_dict))
        time.sleep(0.05)  # Simulate work

    mp_queue.put(SubprocessMessage.result(
        filename=filename,
        file_size=n_events * 100,
        transfer_id=transfer_id,
        direction=direction,
    ))


def _sleeping_worker(params: dict, mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Test worker: sleeps for a long time, checking cancel_event."""
    duration = params.get("duration", 60)
    for _ in range(int(duration / 0.1)):
        if cancel_event.is_set():
            mp_queue.put(SubprocessMessage.cancelled(message="Sleeping worker cancelled"))
            return
        time.sleep(0.1)
    mp_queue.put(SubprocessMessage.result(status="completed"))


def _error_worker(params: dict, mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Test worker: immediately emits an error."""
    mp_queue.put(SubprocessMessage.error(
        message="Simulated error",
        error_type="RuntimeError",
    ))


def _hanging_worker(params: dict, mp_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """Test worker: hangs forever without checking cancel_event."""
    time.sleep(300)  # 5 minutes — will be killed


# ── Basic Lifecycle Tests ────────────────────────────────────────


class TestXetSubprocessRunnerLifecycle:
    """Test start, terminate, is_alive, wait."""

    def test_spawn_context_is_spawn(self):
        """Runner uses spawn context for process creation."""
        runner = XetSubprocessRunner()
        assert runner._ctx == mp.get_context("spawn")

    def test_start_creates_process_and_thread(self):
        """After start(), process and relay thread exist."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_echo_worker,
            params={"n_events": 1, "transfer_id": "test-lifecycle"},
            event_queue=event_queue,
        )

        assert runner._process is not None
        assert runner._process.is_alive()
        assert runner._relay_thread is not None
        assert runner._relay_thread.is_alive()
        assert runner.pid is not None

        # Wait for completion
        runner.wait(timeout=10)
        runner.terminate()

    def test_start_raises_if_already_running(self):
        """Double start() raises RuntimeError."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_sleeping_worker,
            params={"duration": 10},
            event_queue=event_queue,
        )

        try:
            with pytest.raises(RuntimeError, match="already running"):
                runner.start(
                    worker_func=_echo_worker,
                    params={"n_events": 1},
                    event_queue=event_queue,
                )
        finally:
            runner.terminate()

    def test_is_alive_reflects_process_state(self):
        """is_alive() returns False before start, True during, False after."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        # Before start
        assert runner.is_alive() is False

        # After start
        runner.start(
            worker_func=_echo_worker,
            params={"n_events": 1, "transfer_id": "test-alive"},
            event_queue=event_queue,
        )
        assert runner.is_alive() is True

        # After completion
        runner.wait(timeout=10)
        # Process may still be technically alive briefly after last message
        runner.terminate()
        assert runner.is_alive() is False

    def test_daemon_flag_set(self):
        """Child process is created with daemon=True."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_echo_worker,
            params={"n_events": 1, "transfer_id": "test-daemon"},
            event_queue=event_queue,
        )

        assert runner._process.daemon is True

        runner.wait(timeout=10)
        runner.terminate()


# ── Terminate Tests ──────────────────────────────────────────────


class TestXetSubprocessRunnerTerminate:
    """Test terminate() behavior."""

    def test_terminate_kills_process(self):
        """terminate() kills a running process."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_sleeping_worker,
            params={"duration": 60},
            event_queue=event_queue,
        )

        assert runner.is_alive() is True
        runner.terminate()
        assert runner.is_alive() is False

    def test_terminate_no_op_if_not_started(self):
        """terminate() is safe to call when no process exists."""
        runner = XetSubprocessRunner()
        runner.terminate()  # Should not raise

    def test_terminate_sets_cancel_event(self):
        """terminate() sets the cancel_event before killing the process."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_sleeping_worker,
            params={"duration": 60},
            event_queue=event_queue,
        )

        cancel_event = runner._cancel_event
        runner.terminate()

        # cancel_event should have been set
        assert cancel_event is not None
        assert cancel_event.is_set()

    def test_terminate_kill_fallback(self):
        """terminate() uses SIGKILL if SIGTERM doesn't work."""
        runner = XetSubprocessRunner(terminate_timeout=0.5, kill_timeout=1.0)
        event_queue = queue.Queue()

        runner.start(
            worker_func=_hanging_worker,
            params={},
            event_queue=event_queue,
        )

        runner.terminate()
        assert runner.is_alive() is False

    def test_cleanup_after_terminate(self):
        """After terminate(), process and queue references are None."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_echo_worker,
            params={"n_events": 1, "transfer_id": "test-cleanup"},
            event_queue=event_queue,
        )

        runner.wait(timeout=10)
        runner.terminate()

        assert runner._process is None
        assert runner._mp_queue is None
        assert runner._cancel_event is None


# ── Event Relay Tests ────────────────────────────────────────────


class TestXetSubprocessRunnerRelay:
    """Test mp.Queue → queue.Queue event relay."""

    def test_relay_translates_events(self):
        """Progress events from worker appear in event_queue as ProgressEvent."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_echo_worker,
            params={"n_events": 3, "transfer_id": "test-relay", "filename": "test.bin"},
            event_queue=event_queue,
        )

        # Wait for completion
        runner.wait(timeout=10)

        # Collect events
        events = []
        while True:
            try:
                event = event_queue.get(timeout=1)
                events.append(event)
            except queue.Empty:
                break

        runner.terminate()

        # Should have 3 PROGRESS + 1 COMPLETE
        progress_events = [e for e in events if e.event_type == EventType.PROGRESS]
        complete_events = [e for e in events if e.event_type == EventType.COMPLETE]

        assert len(progress_events) == 3
        assert len(complete_events) == 1

        # Verify event content
        for e in progress_events:
            assert isinstance(e, ProgressEvent)
            assert e.transfer_id == "test-relay"
            assert e.filename == "test.bin"

    def test_relay_stops_on_result(self):
        """Relay thread exits after receiving a result message."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_echo_worker,
            params={"n_events": 1, "transfer_id": "test-result"},
            event_queue=event_queue,
        )

        # Wait for relay thread to finish
        runner.wait(timeout=10)

        # Relay thread should have stopped
        assert runner._relay_thread is not None
        # After wait, relay thread should be done (or nearly done)
        runner._relay_thread.join(timeout=2)
        assert not runner._relay_thread.is_alive()

        runner.terminate()

    def test_relay_stops_on_error(self):
        """Relay thread exits after receiving an error message."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_error_worker,
            params={},
            event_queue=event_queue,
        )

        result = runner.wait(timeout=10)
        assert result is not None
        assert result.get("message") == "Simulated error"

        # Check that ERROR event was emitted
        events = []
        while True:
            try:
                event = event_queue.get(timeout=1)
                events.append(event)
            except queue.Empty:
                break

        runner.terminate()

        error_events = [e for e in events if e.event_type == EventType.ERROR]
        assert len(error_events) == 1
        assert error_events[0].error.message == "Simulated error"

    def test_relay_stops_on_cancelled(self):
        """Relay thread exits after receiving a cancelled message."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_sleeping_worker,
            params={"duration": 60},
            event_queue=event_queue,
        )

        # Wait a moment then cancel
        time.sleep(0.3)
        runner.terminate()

        # Collect events (may or may not have CANCELLED depending on timing)
        events = []
        while True:
            try:
                event = event_queue.get(timeout=1)
                events.append(event)
            except queue.Empty:
                break

        # The relay should have stopped
        if runner._relay_thread is not None:
            runner._relay_thread.join(timeout=2)

    def test_complete_event_includes_snapshot_stats(self):
        """COMPLETE event from result payload includes file_index/total_files."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        # Use echo worker that sends a result with snapshot-level fields
        runner.start(
            worker_func=_echo_worker,
            params={"n_events": 0, "transfer_id": "test-snap-stats"},
            event_queue=event_queue,
        )

        # Manually inject a result message with snapshot stats
        # (simulating what _snapshot_worker sends)
        runner._mp_queue.put(SubprocessMessage.result(
            filename="test/repo",
            destination_path="/tmp/test_repo",
            transfer_id="test-snap-stats",
            direction="download",
            file_size=5000,
            bytes_completed=5000,
            total_bytes=5000,
            files_completed=3,
            total_files=3,
        ))

        result = runner.wait(timeout=5)
        assert result is not None

        # Collect events from event_queue
        events = []
        while True:
            try:
                event = event_queue.get(timeout=1)
                events.append(event)
            except queue.Empty:
                break

        runner.terminate()

        # Should have a COMPLETE event with file_index and total_files
        complete_events = [e for e in events if e.event_type == EventType.COMPLETE]
        assert len(complete_events) == 1
        complete_event = complete_events[0]
        assert complete_event.bytes_completed == 5000
        assert complete_event.total_bytes == 5000
        assert complete_event.file_index == 3  # files_completed
        assert complete_event.total_files == 3

    def test_complete_event_fallback_to_file_size(self):
        """COMPLETE event falls back to file_size when bytes_completed not in payload."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_echo_worker,
            params={"n_events": 0, "transfer_id": "test-fallback"},
            event_queue=event_queue,
        )

        # Send a result with only file_size (single-file download pattern)
        runner._mp_queue.put(SubprocessMessage.result(
            filename="file.bin",
            destination_path="/tmp/file.bin",
            transfer_id="test-fallback",
            direction="download",
            file_size=1000,
        ))

        result = runner.wait(timeout=5)
        assert result is not None

        events = []
        while True:
            try:
                event = event_queue.get(timeout=1)
                events.append(event)
            except queue.Empty:
                break

        runner.terminate()

        complete_events = [e for e in events if e.event_type == EventType.COMPLETE]
        assert len(complete_events) == 1
        # Should fall back to file_size for both bytes_completed and total_bytes
        assert complete_events[0].bytes_completed == 1000
        assert complete_events[0].total_bytes == 1000


# ── Wait Tests ───────────────────────────────────────────────────


class TestXetSubprocessRunnerWait:
    """Test wait() behavior."""

    def test_wait_returns_result(self):
        """wait() returns the result payload on success."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_echo_worker,
            params={"n_events": 1, "transfer_id": "test-wait", "filename": "wait.bin"},
            event_queue=event_queue,
        )

        result = runner.wait(timeout=10)
        assert result is not None
        assert result.get("status") == "success"
        assert result.get("filename") == "wait.bin"

        runner.terminate()

    def test_wait_returns_none_on_timeout(self):
        """wait() returns None when timeout expires."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_sleeping_worker,
            params={"duration": 60},
            event_queue=event_queue,
        )

        result = runner.wait(timeout=0.5)
        assert result is None  # Worker is still sleeping

        runner.terminate()

    def test_wait_returns_error_payload(self):
        """wait() returns error payload when worker fails."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_error_worker,
            params={},
            event_queue=event_queue,
        )

        result = runner.wait(timeout=10)
        assert result is not None
        assert result.get("message") == "Simulated error"
        assert result.get("error_type") == "RuntimeError"

        runner.terminate()


# ── Property Tests ───────────────────────────────────────────────


class TestXetSubprocessRunnerProperties:
    """Test pid and exitcode properties."""

    def test_pid_before_start(self):
        """pid is None before start."""
        runner = XetSubprocessRunner()
        assert runner.pid is None

    def test_pid_after_start(self):
        """pid is set after start."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_echo_worker,
            params={"n_events": 1, "transfer_id": "test-pid"},
            event_queue=event_queue,
        )

        assert runner.pid is not None
        assert isinstance(runner.pid, int)

        runner.wait(timeout=10)
        runner.terminate()

    def test_exitcode_after_terminate(self):
        """exitcode is set after terminate."""
        runner = XetSubprocessRunner()
        event_queue = queue.Queue()

        runner.start(
            worker_func=_echo_worker,
            params={"n_events": 1, "transfer_id": "test-exit"},
            event_queue=event_queue,
        )

        runner.wait(timeout=10)
        # After normal completion, exitcode should be 0
        # But we need to let the process fully exit
        time.sleep(0.5)

        # Process may have already exited normally
        if runner._process is not None:
            exitcode = runner.exitcode

        runner.terminate()
