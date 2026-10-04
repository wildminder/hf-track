"""Tests for the subprocess worker ``_xet_file_only_worker``.

Plan: docs/plans/2026-07-16-xet-single-file-subprocess-isolation.md (Step 1)

These tests run the worker FUNCTION directly (in the test process) with a
``multiprocessing.Queue`` + ``multiprocessing.Event`` so we can assert the
``SubprocessMessage`` protocol it speaks, without spawning a real child. The
``hf_xet`` / ``huggingface_hub.utils._xet`` modules are patched so no network
or real Rust extension is involved.

Verifies:
  * worker sends a START event with correct transfer_id / filename / total_bytes
  * worker sends PROGRESS events with bytes_completed > 0 and percentage in [0,100]
  * worker sends a RESULT with destination_path == dest_path on success
  * worker sends CANCELLED when cancel_event is set early
  * worker sends ERROR when xet_file_data is missing
  * worker passes correct args to new_file_download_group
    (token_refresh_url, token_refresh_headers, custom_headers, progress_callback)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
from unittest import mock

import pytest

from hf_track._xet_worker import (
    _deserialize_xet_file_data,
    _serialize_xet_file_data,
    _xet_file_only_worker,
)
from hf_track.subprocess import SubprocessMessage
from hf_track.types import EventType


class _XetPatch:
    """Context manager that patches hf_xet + huggingface_hub._xet globals."""

    def __init__(self, *, start_side_effect=None, progress_calls=None):
        self._start_side_effect = start_side_effect
        self._progress_calls = progress_calls or []
        self.captured = {
            "new_group_kwargs": None,
            "start_file_info": None,
            "start_dest": None,
            "progress_callback": None,
            "session_used": False,
        }

    def __enter__(self):
        def _XetFileInfo(hash, file_size=None):
            self.captured["start_file_info"] = {"hash": hash, "file_size": file_size}
            return mock.MagicMock()

        def _start_download_file(file_info, dest_path):
            self.captured["start_dest"] = dest_path
            cb = self.captured["progress_callback"]
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
        group.__enter__.return_value = group

        def _new_file_download_group(**kwargs):
            self.captured["new_group_kwargs"] = kwargs
            self.captured["progress_callback"] = kwargs.get("progress_callback")
            return group

        session = mock.MagicMock()
        session.new_file_download_group.side_effect = _new_file_download_group

        def _get_xet_session():
            self.captured["session_used"] = True
            return session

        def _xet_headers_without_auth(headers):
            return {k: v for k, v in (headers or {}).items() if k.lower() != "authorization"}

        fake_hf_xet = mock.MagicMock()
        fake_hf_xet.XetFileInfo = _XetFileInfo

        import sys

        self._modules_patch = mock.patch.dict(
            sys.modules, {"hf_xet": fake_hf_xet}
        )
        self._modules_patch.start()

        self._xet_patch = mock.patch(
            "hf_track._xet_worker.get_xet_session"
            if False
            else "huggingface_hub.utils._xet.get_xet_session",
            _get_xet_session,
        )
        # Patch at the import site used by the worker.
        self._xet_patch2 = mock.patch(
            "hf_track._xet_worker.xet_headers_without_auth"
            if False
            else "huggingface_hub.utils._xet.xet_headers_without_auth",
            _xet_headers_without_auth,
        )
        self._xet_patch.start()
        self._xet_patch2.start()
        return self

    def __exit__(self, *exc):
        self._xet_patch.stop()
        self._xet_patch2.stop()
        self._modules_patch.stop()
        return False


def _run_worker(params, cancel_event=None, progress_calls=None):
    """Run the worker in-process, collecting all SubprocessMessages."""
    # A plain thread queue, not mp.Queue: the worker runs in THIS process, and
    # mp.Queue hands writes to a feeder thread, so empty() can report True
    # while messages are still in flight. That made these tests fail or pass
    # depending on suite ordering. With queue.Queue the put is synchronous and
    # the drain below is exact.
    mp_queue: queue.Queue = queue.Queue()
    if cancel_event is None:
        cancel_event = mp.Event()
    with _XetPatch(progress_calls=progress_calls or [100, 200, 300]):
        _xet_file_only_worker(params, mp_queue, cancel_event)
    # Drain the queue
    messages = []
    while not mp_queue.empty():
        messages.append(mp_queue.get_nowait())
    return messages


def _base_params(**overrides):
    xfd = _serialize_xet_file_data(
        _deserialize_xet_file_data(
            {"file_hash": "abc123", "refresh_route": "https://xet/refresh"}
        )
    )
    params = {
        "file_hash": "abc123",
        "file_size": 300,
        "dest_path": "/tmp/audiovae.pth",
        "xet_file_data": xfd,
        "token": None,
        "endpoint": None,
        "transfer_id": "tid-1",
        "report_interval": 0.0,  # emit every progress call
        "request_headers": {},
    }
    params.update(overrides)
    return params


def test_worker_sends_start_event():
    messages = _run_worker(_base_params())
    starts = [m for m in messages if m.is_event and m.payload.get("event_type") == "start"]
    assert len(starts) == 1
    start = starts[0].payload
    assert start["transfer_id"] == "tid-1"
    assert start["filename"] == "audiovae.pth"
    assert start["total_bytes"] == 300


def test_worker_sends_progress_with_positive_bytes():
    messages = _run_worker(_base_params(), progress_calls=[100, 200, 300])
    prog = [m for m in messages if m.is_event and m.payload.get("event_type") == "progress"]
    # At least the final 100% progress event must be present.
    assert any(p.payload["bytes_completed"] > 0 for p in prog)
    for p in prog:
        pct = p.payload["percentage"]
        assert 0.0 <= pct <= 100.0


def test_worker_sends_result_with_dest_path():
    messages = _run_worker(_base_params())
    results = [m for m in messages if m.is_result]
    assert len(results) == 1
    assert results[0].payload["destination_path"] == "/tmp/audiovae.pth"
    assert results[0].payload["filename"] == "audiovae.pth"
    assert results[0].payload["file_size"] == 300


def test_worker_passes_correct_group_args():
    with _XetPatch(progress_calls=[100, 200, 300]) as patch:
        mp_queue: queue.Queue = queue.Queue()
        _xet_file_only_worker(_base_params(), mp_queue, mp.Event())
    kwargs = patch.captured["new_group_kwargs"]
    assert kwargs is not None
    assert kwargs["token_refresh_url"] == "https://xet/refresh"
    assert "token_refresh_headers" in kwargs
    assert "custom_headers" in kwargs
    assert kwargs["progress_callback"] is not None
    # The progress_callback must accept (total_update, item_updates).
    assert "total_update" in kwargs["progress_callback"].__code__.co_varnames
    assert "item_updates" in kwargs["progress_callback"].__code__.co_varnames
    # start_download_file received the right file info + dest.
    assert patch.captured["start_file_info"]["hash"] == "abc123"
    assert patch.captured["start_dest"] == os.path.abspath("/tmp/audiovae.pth")


def test_worker_sends_cancelled_when_event_set():
    cancel = mp.Event()
    cancel.set()  # cancel before the worker starts
    messages = _run_worker(_base_params(), cancel_event=cancel)
    cancelled = [m for m in messages if m.is_cancelled]
    assert len(cancelled) == 1
    # No result on cancel
    assert not any(m.is_result for m in messages)


def test_worker_sends_error_when_xet_file_data_missing():
    params = _base_params()
    params["xet_file_data"] = {}  # empty -> deserialize returns None
    messages = _run_worker(params)
    errors = [m for m in messages if m.is_error]
    assert len(errors) == 1
    assert "not stored in Xet" in errors[0].payload["message"]


def test_worker_sends_error_on_start_failure():
    with _XetPatch(start_side_effect=RuntimeError("boom")) as patch:
        mp_queue: queue.Queue = queue.Queue()
        _xet_file_only_worker(_base_params(), mp_queue, mp.Event())
        messages = []
        while not mp_queue.empty():
            messages.append(mp_queue.get_nowait())
    errors = [m for m in messages if m.is_error]
    assert len(errors) == 1
    assert "boom" in errors[0].payload["message"]
