"""Tests for the dedicated xet single-file download path (real XetSession API).

Plan: docs/plans/2026-07-16-xet-download-real-fix.md

These tests verify that `download_file_xet_only`:
  * calls get_xet_session() + new_file_download_group(token_refresh_url=...,
    token_refresh_headers=..., custom_headers=..., progress_callback=...) +
    start_download_file(XetFileInfo(hash, size), dest) with the correct args,
  * returns the destination path on success,
  * emits START and COMPLETE ProgressEvents,
  * emits correct PROGRESS events (bytes_completed, percentage, speed, clamped),
  * supports cancellation via abort_xet_session -> TransferCancelledError,
  * raises a clear TransferProgressError when xet_file_data is missing,
  * propagates start_download_file exceptions as TransferProgressError + ERROR event.
"""

from __future__ import annotations

import queue
import threading
import time
from unittest import mock

import pytest

from hf_track.download import xet_file_only
from hf_track.types import (
    TransferProgressError,
    TransferCancelledError,
    ProgressEvent,
    EventType,
)


class _FakeXetFileData:
    def __init__(self, file_hash="abc123", refresh_route="https://huggingface.co/api/xet/refresh"):
        self.file_hash = file_hash
        self.refresh_route = refresh_route


class _XetPatch:
    """Context manager that patches hf_xet + huggingface_hub._xet globals."""

    def __init__(
        self,
        *,
        start_side_effect=None,
        progress_calls=None,
        abort_called=None,
    ):
        self._start_side_effect = start_side_effect
        self._progress_calls = progress_calls or []
        self._abort_called = abort_called if abort_called is not None else {}
        self.captured = {
            "new_group_kwargs": None,
            "start_file_info": None,
            "start_dest": None,
            "progress_callback": None,
            "abort_called": False,
            "session_used": False,
        }

    def __enter__(self):
        # Fake XetFileInfo
        def _XetFileInfo(hash, file_size=None):
            self.captured["start_file_info"] = {"hash": hash, "file_size": file_size}
            return mock.MagicMock()

        # Fake group
        def _start_download_file(file_info, dest_path):
            self.captured["start_dest"] = dest_path
            if self._progress_calls:
                cb = self.captured["progress_callback"]
                # The installed hf_xet wheel calls progress_callback(total_update, item_updates)
                # where total_update.total_transfer_bytes_completed is the cumulative
                # network bytes (the live-progress signal for Xet).
                for completed in self._progress_calls:
                    total_update = mock.MagicMock()
                    total_update.total_transfer_bytes_completed = completed
                    total_update.total_bytes_completed = completed
                    if cb:
                        cb(total_update, [])
            if self._start_side_effect is not None:
                if isinstance(self._start_side_effect, Exception):
                    raise self._start_side_effect
                return self._start_side_effect()
            return None

        group = mock.MagicMock()
        group.start_download_file.side_effect = _start_download_file
        # Capture progress_callback when new_file_download_group is entered.
        group.__enter__.return_value = group

        def _new_file_download_group(**kwargs):
            self.captured["new_group_kwargs"] = kwargs
            # Capture the progress_callback passed by our code.
            self.captured["progress_callback"] = kwargs.get("progress_callback")
            return group

        session = mock.MagicMock()
        session.new_file_download_group.side_effect = _new_file_download_group

        def _get_xet_session():
            self.captured["session_used"] = True
            return session

        def _abort_xet_session():
            self.captured["abort_called"] = True
            self._abort_called["v"] = True

        def _xet_headers_without_auth(headers):
            return {k: v for k, v in (headers or {}).items() if k.lower() != "authorization"}

        fake_hf_xet = mock.MagicMock()
        fake_hf_xet.XetFileInfo = _XetFileInfo
        fake_hf_xet.force_sigint_shutdown = mock.MagicMock()

        import sys

        self._modules_patch = mock.patch.dict(sys.modules, {"hf_xet": fake_hf_xet})
        self._modules_patch.__enter__()

        self._get_patch = mock.patch(
            "huggingface_hub.utils._xet.get_xet_session", _get_xet_session
        )
        self._get_patch.__enter__()

        self._abort_patch = mock.patch(
            "huggingface_hub.utils._xet.abort_xet_session", _abort_xet_session
        )
        self._abort_patch.__enter__()

        self._hdr_patch = mock.patch(
            "huggingface_hub.utils._xet.xet_headers_without_auth", _xet_headers_without_auth
        )
        self._hdr_patch.__enter__()

        return self.captured

    def __exit__(self, *exc):
        self._hdr_patch.__exit__(*exc)
        self._abort_patch.__exit__(*exc)
        self._get_patch.__exit__(*exc)
        self._modules_patch.__exit__(*exc)
        return False


def _call(**kwargs):
    base = dict(
        repo_id="openbmb/VoxCPM-0.5B",
        filename="audiovae.pth",
        file_hash="abc123",
        file_size=301494192,
        dest_path="/tmp/diag_xet.bin",
        token=None,
        xet_file_data=_FakeXetFileData(),
        event_queue=None,
        transfer_id="t1",
    )
    base.update(kwargs)
    return xet_file_only.download_file_xet_only(**base)


# ── Step 1: happy path ──────────────────────────────────────────────────

import os


def test_calls_session_group_and_start_with_correct_args():
    with _XetPatch() as cap:
        _call()
    assert cap["session_used"], "get_xet_session() was not called"
    kw = cap["new_group_kwargs"]
    assert kw is not None, "new_file_download_group was not called"
    assert kw["token_refresh_url"] == "https://huggingface.co/api/xet/refresh"
    assert "authorization" not in (kw.get("custom_headers") or {})
    assert callable(kw["progress_callback"])
    assert cap["start_file_info"]["hash"] == "abc123"
    assert cap["start_file_info"]["file_size"] == 301494192
    assert cap["start_dest"] == os.path.abspath("/tmp/diag_xet.bin")


def test_returns_dest_path_on_success():
    with _XetPatch() as cap:
        result = _call()
    assert result == os.path.abspath("/tmp/diag_xet.bin")


def test_emits_start_and_complete_events():
    q: "queue.Queue" = queue.Queue()
    with _XetPatch(progress_calls=[100, 200]) as cap:
        _call(event_queue=q)
    events = _drain(q)
    assert EventType.START in [e.event_type for e in events]
    completes = [e for e in events if e.event_type == EventType.COMPLETE]
    assert completes, "no COMPLETE event emitted"
    assert completes[0].percentage == 100.0


# ── Step 2: progress event correctness ─────────────────────────────────

def test_progress_events_increment():
    q: "queue.Queue" = queue.Queue()
    # progress_calls are ABSOLUTE total_bytes_completed values (matching the
    # installed hf_xet wheel's progress_callback(total_update, item_updates)
    # where total_update.total_bytes_completed is cumulative).
    with _XetPatch(progress_calls=[100, 200, 700]) as cap:
        _call(event_queue=q, file_size=1000, report_interval=0.0)
    progresses = [e for e in _drain(q) if e.event_type == EventType.PROGRESS]
    assert progresses, "no PROGRESS events emitted"
    assert progresses[-1].bytes_completed == 700
    assert abs(progresses[-1].percentage - 70.0) < 1e-6


def test_percentage_clamped_for_unknown_size():
    q: "queue.Queue" = queue.Queue()
    with _XetPatch(progress_calls=[50, 50]) as cap:
        _call(event_queue=q, file_size=0, report_interval=0.0)
    progresses = [e for e in _drain(q) if e.event_type == EventType.PROGRESS]
    assert progresses
    for e in progresses:
        assert e.percentage == 0.0


def test_speed_computed():
    q: "queue.Queue" = queue.Queue()
    with _XetPatch(progress_calls=[100, 100]) as cap:
        _call(event_queue=q, report_interval=0.0)
    progresses = [e for e in _drain(q) if e.event_type == EventType.PROGRESS]
    assert progresses
    assert progresses[-1].speed >= 0.0


# ── Step 3: cancellation ───────────────────────────────────────────────

def test_cancellation_raises_cancelled_error():
    cancel_flag = {"v": False}

    def _start_slow(*args):
        # Simulate the transfer running; after a short delay the "user" cancels
        # (flips the flag), and the watchdog detects it and aborts.
        time.sleep(0.3)
        cancel_flag["v"] = True
        # Keep the worker alive a bit so the watchdog has a chance to act.
        time.sleep(2.0)
        return None

    with _XetPatch(start_side_effect=_start_slow) as cap:
        with pytest.raises(TransferCancelledError):
            _call(is_cancelled=lambda: cancel_flag["v"], probe_timeout_s=30.0)
    assert cap["abort_called"], "abort_xet_session was not called on cancel"


def test_cancellation_precedence_before_call():
    with _XetPatch() as cap:
        with pytest.raises(TransferCancelledError):
            _call(is_cancelled=lambda: True)
    assert cap["new_group_kwargs"] is None, "download started despite cancel"


# ── Step 4: missing xet_file_data ──────────────────────────────────────

def test_missing_xet_file_data_raises_error():
    q: "queue.Queue" = queue.Queue()
    with _XetPatch() as cap:
        with pytest.raises(TransferProgressError) as exc:
            _call(xet_file_data=None, event_queue=q)
    assert "not stored in Xet" in str(exc.value)
    errors = [e for e in _drain(q) if e.event_type == EventType.ERROR]
    assert errors


# ── Step 5: error propagation from start_download_file ─────────────────

def test_start_exception_emits_error_event():
    q: "queue.Queue" = queue.Queue()
    with _XetPatch(start_side_effect=RuntimeError("CAS boom")) as cap:
        with pytest.raises(TransferProgressError) as exc:
            _call(event_queue=q)
    assert "CAS boom" in str(exc.value)
    errors = [e for e in _drain(q) if e.event_type == EventType.ERROR]
    assert errors, "no ERROR event emitted"
    assert errors[0].error is not None


# ── Helpers ────────────────────────────────────────────────────────────

def _drain(q: "queue.Queue") -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out
