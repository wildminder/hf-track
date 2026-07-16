"""Tests that cancellation terminates the xet subprocess.

Plan: docs/plans/2026-07-16-xet-single-file-subprocess-isolation.md (Step 4)

Verifies the cancellation chain for single-file xet downloads:

  tracker.cancel(tid)  ->  is_cancelled() == True
    ->  download_file_xet_subprocess watchdog observes the flag
    ->  runner.request_cancel() + runner.terminate(grace=2.0)
    ->  child OS process killed (hf_xet .pyd thread dies, memory freed)

We test the watchdog logic in ``download_file_xet_subprocess`` with a fake
runner (no real subprocess), and the tracker-level flag propagation.
"""

from __future__ import annotations

import queue
import threading
import time
from unittest import mock

import pytest

import hf_track.download.xet_file_only as xet_file_only
import hf_track.subprocess as hf_track_subprocess_mod
from hf_track.download import xet_file_only as xfo
from hf_track.tracker import HfTracker
from hf_track.types import TransferCancelledError


class _FakeRunner:
    """Fake runner that simulates a live subprocess."""

    def __init__(self, result=None):
        self._result = result
        self._alive = True
        self.started = False
        self.terminated = False
        self.cancelled = False
        self.params = None
        self.event_queue = None

    def start(self, worker_func, params, event_queue):
        self.started = True
        self.params = params
        self.event_queue = event_queue

    def request_cancel(self):
        self.cancelled = True

    def terminate(self, grace=None):
        self.terminated = True
        self._alive = False

    def is_alive(self):
        return self._alive

    def wait(self, timeout=None):
        # Block until terminated (simulating the child running until killed).
        deadline = time.time() + (timeout or 30)
        while self._alive and time.time() < deadline:
            time.sleep(0.02)
        return self._result


def _fake_xfd():
    class _X:
        file_hash = "abc"
        refresh_route = "https://xet/refresh"
    return _X()


def test_watchdog_terminates_runner_on_cancel():
    """The watchdog in download_file_xet_subprocess must call terminate()
    when is_cancelled() becomes True."""
    runner = _FakeRunner(result={"status": "success", "destination_path": "/tmp/f.pth"})
    cancel_flag = {"v": False}

    def fake_is_cancelled():
        return cancel_flag["v"]

    kwargs = dict(
        repo_id="repo",
        filename="f.pth",
        file_hash="abc",
        file_size=100,
        dest_path="/tmp/f.pth",
        token=None,
        xet_file_data=_fake_xfd(),
        event_queue=queue.Queue(),
        transfer_id="tid-watch",
        is_cancelled=fake_is_cancelled,
    )
    with mock.patch.object(hf_track_subprocess_mod, "XetSubprocessRunner", return_value=runner):
        t = threading.Thread(target=lambda: xet_file_only.download_file_xet_subprocess(**kwargs))
        t.start()
        # Let the watchdog start looping.
        time.sleep(0.2)
        # User cancels.
        cancel_flag["v"] = True
        t.join(timeout=5.0)
        assert not t.is_alive(), "download thread should exit after cancel"
        assert runner.terminated, "watchdog must call runner.terminate() on cancel"
        assert runner.cancelled, "watchdog must call runner.request_cancel() on cancel"


def test_tracker_cancel_propagates_to_is_cancelled():
    """tracker.cancel(tid) makes is_cancelled(tid) True, which the watchdog
    observes to terminate the subprocess."""
    tracker = HfTracker(token="hf_test")
    tid = "tid-propagate"
    tracker.cancel(tid)
    assert tracker.is_cancelled(tid) is True


def test_tracker_cancel_registers_runner_then_terminates():
    """End-to-end: a registered runner is terminated when cancel() flips the
    flag and the watchdog observes it."""
    tracker = HfTracker(token="hf_test")
    tid = "tid-e2e"
    runner = _FakeRunner(result={"status": "success", "destination_path": "/tmp/f.pth"})

    # Register the runner exactly as download_file()'s _on_spawn does.
    with tracker._lock:
        tracker._active_runners[tid] = runner

    cancel_flag = {"v": False}

    def fake_is_cancelled():
        return cancel_flag["v"]

    kwargs = dict(
        repo_id="repo",
        filename="f.pth",
        file_hash="abc",
        file_size=100,
        dest_path="/tmp/f.pth",
        token=None,
        xet_file_data=_fake_xfd(),
        event_queue=queue.Queue(),
        transfer_id=tid,
        is_cancelled=fake_is_cancelled,
    )
    with mock.patch.object(hf_track_subprocess_mod, "XetSubprocessRunner", return_value=runner):
        t = threading.Thread(target=lambda: xet_file_only.download_file_xet_subprocess(**kwargs))
        t.start()
        time.sleep(0.2)
        # This is what the example script / Ctrl+C handler does.
        tracker.cancel(tid)
        cancel_flag["v"] = True
        t.join(timeout=5.0)
        assert not t.is_alive()
        assert runner.terminated
