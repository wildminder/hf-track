"""Tests for ``download_file_xet_subprocess`` orchestration.

Plan: docs/plans/2026-07-16-xet-single-file-subprocess-isolation.md (Step 2)

These tests verify the orchestration in ``download_file_xet_subprocess``:
  * returns destination_path on success
  * raises TransferCancelledError when worker sends CANCELLED
  * raises TransferProgressError when worker sends ERROR
  * raises TransferProgressError when wait() times out (None)
  * always calls runner.terminate() in finally (idempotent)
  * calls on_spawn with the runner instance
  * raises TransferProgressError when xet_file_data is missing

We use a FakeRunner that mimics XetSubprocessRunner's interface so no real
child process is spawned.
"""

from __future__ import annotations

import queue
from unittest import mock

import pytest

import hf_track.subprocess as subprocess_mod
from hf_track.download import xet_file_subprocess
from hf_track.types import TransferCancelledError, TransferProgressError


class _FakeRunner:
    """Minimal stand-in for XetSubprocessRunner."""

    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises
        self.started = False
        self.terminated = False
        self.cancelled = False
        self.params = None
        self.event_queue = None

    def start(self, worker_func, params, event_queue):
        self.started = True
        self.params = params
        self.event_queue = event_queue
        self._worker_func = worker_func

    def request_cancel(self):
        self.cancelled = True

    def terminate(self, grace=None):
        self.terminated = True

    def is_alive(self):
        return False

    def wait(self, timeout=None):
        if self._raises is not None:
            raise self._raises
        return self._result


def _fake_xfd():
    class _X:
        file_hash = "abc"
        refresh_route = "https://xet/refresh"
    return _X()


def _call(runner, **overrides):
    kwargs = dict(
        repo_id="repo",
        filename="f.pth",
        file_hash="abc",
        file_size=100,
        dest_path="/tmp/f.pth",
        token=None,
        xet_file_data=_fake_xfd(),
        event_queue=queue.Queue(),
        transfer_id="tid",
    )
    kwargs.update(overrides)
    with mock.patch.object(subprocess_mod, "XetSubprocessRunner", return_value=runner):
        return xet_file_subprocess.download_file_xet_subprocess(**kwargs)


def test_returns_destination_on_success():
    runner = _FakeRunner(result={
        "status": "success",
        "destination_path": "/tmp/f.pth",
    })
    path = _call(runner)
    assert path == "/tmp/f.pth"
    assert runner.started
    assert runner.terminated  # finally cleanup


def test_raises_cancelled_when_worker_cancelled():
    runner = _FakeRunner(result={"status": "cancelled"})
    with pytest.raises(TransferCancelledError):
        _call(runner)
    assert runner.terminated


def test_raises_progress_error_when_worker_error():
    runner = _FakeRunner(result={"status": "error", "message": "boom"})
    with pytest.raises(TransferProgressError):
        _call(runner)
    assert runner.terminated


def test_raises_progress_error_on_timeout():
    runner = _FakeRunner(result=None)  # wait() returns None -> timeout
    with pytest.raises(TransferProgressError):
        _call(runner)
    assert runner.terminated


def test_raises_cancelled_when_wait_none_but_is_cancelled():
    """Regression: when the watchdog terminates the child (is_cancelled True)
    and wait() returns None, we must raise TransferCancelledError — NOT a
    misleading 'timed out' TransferProgressError."""
    runner = _FakeRunner(result=None)  # wait() returns None (child killed)

    # Make is_alive() True so the watchdog loop runs and observes cancel.
    class _AliveRunner(_FakeRunner):
        def is_alive(self):
            return True

    runner.__class__ = _AliveRunner
    cancel_state = {"flag": False}

    def _is_cancelled():
        # Flip to True on the second poll so the watchdog terminates.
        cancel_state["flag"] = True
        return cancel_state["flag"]

    with pytest.raises(TransferCancelledError):
        _call(runner, is_cancelled=_is_cancelled)
    assert runner.terminated
    assert runner.cancelled  # watchdog called request_cancel()


def test_terminate_called_even_on_unexpected_exception():
    runner = _FakeRunner(raises=RuntimeError("unexpected"))
    with pytest.raises(RuntimeError):
        _call(runner)
    assert runner.terminated


def test_on_spawn_receives_runner():
    runner = _FakeRunner(result={"status": "success", "destination_path": "/tmp/f.pth"})
    captured = {}
    _call(runner, on_spawn=lambda r: captured.setdefault("runner", r))
    assert captured.get("runner") is runner


def test_missing_xet_file_data_raises():
    runner = _FakeRunner(result={"status": "success", "destination_path": "/tmp/f.pth"})
    with pytest.raises(TransferProgressError):
        _call(runner, xet_file_data=None)
    # Runner should NOT have been started (early return before subprocess).
    assert not runner.started


def test_params_serialized_correctly():
    runner = _FakeRunner(result={"status": "success", "destination_path": "/tmp/f.pth"})
    _call(runner)
    assert runner.params["file_hash"] == "abc"
    assert runner.params["file_size"] == 100
    assert runner.params["dest_path"] == "/tmp/f.pth"
    # xet_file_data must be serialized to a picklable dict.
    assert isinstance(runner.params["xet_file_data"], dict)
    assert runner.params["xet_file_data"]["refresh_route"] == "https://xet/refresh"
    assert runner.params["transfer_id"] == "tid"
