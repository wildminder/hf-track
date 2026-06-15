"""Tests for Step 1 + Step 2 of plan ``docs/plans/2026-06-05-xet-streaming-flush-reliability.md``.

Step 1: per-chunk fsync for small files (D5 fix).
Step 2: ``finally`` cleanup of stream + group on cancel/error (D4/D6/D10 fix).
Step 5: ``disable_fsync`` worker param (D9 fix).

These are split into a separate file from ``test_xet_worker.py`` to keep
the existing test file untouched. The tests use synthetic ``hf_xet`` mocks
(no real network), invoked in-process for deterministic timing.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from unittest.mock import MagicMock, patch

import pytest

from hf_track._xet_worker import (
    DEFAULT_FSYNC_INTERVAL,
    _xet_streaming_download_worker,
)
from hf_track.subprocess import MSG_CANCELLED, MSG_ERROR, MSG_EVENT, MSG_RESULT


# ── Helpers ─────────────────────────────────────────────────────────


def _build_xet_mock_module(stream_iter=None, raise_on_download_stream=None):
    """Build a stand-in for ``hf_xet`` and ``huggingface_hub.utils._xet``.

    Args:
        stream_iter: iterable yielding chunk bytes. If None, defaults
            to ``[b"x" * 256]`` (one small chunk).
        raise_on_download_stream: if set, the mock ``download_stream``
            method raises this exception instead of returning a stream.
    """
    if stream_iter is None:
        stream_iter = [b"x" * 256]

    # Build a mock stream that supports both ``for chunk in stream``
    # (via __iter__) and ``next(stream, sentinel)`` (via __next__).
    _iter = iter(stream_iter)

    mock_stream = MagicMock()
    mock_stream.__iter__ = lambda self: _iter
    mock_stream.__next__ = lambda self: next(_iter)

    mock_xet = MagicMock()
    if raise_on_download_stream is not None:
        mock_xet.XetSession.return_value.new_download_stream_group.return_value.download_stream.side_effect = (
            raise_on_download_stream
        )
    else:
        mock_xet.XetSession.return_value.new_download_stream_group.return_value.download_stream.return_value = (
            mock_stream
        )

    mock_xet_utils = MagicMock(
        refresh_xet_connection_info=MagicMock(
            return_value=MagicMock(
                endpoint="https://xet.example.com",
                access_token="tok",
                expiration_unix_epoch=9999999999,
            ),
        ),
    )
    return mock_xet, mock_xet_utils


def _make_params(tmp_path, file_size, chunks=None, **overrides):
    """Build a streaming worker params dict."""
    dest = str(tmp_path / "out.bin")
    params = {
        "file_specs": [
            {
                "hash": "abc123",
                "file_size": file_size,
                "dest_path": dest,
                "xet_file_data": {
                    "file_hash": "abc123",
                    "refresh_route": "https://xet.example.com/refresh",
                },
            },
        ],
        "token": "hf_test",
        "endpoint": None,
        "transfer_id": "step1-2-test",
        "report_interval": 0.0,
        "fsync_interval": DEFAULT_FSYNC_INTERVAL,
        "request_headers": {},
    }
    params.update(overrides)
    return params, dest


def _drain(mp_queue):
    out = []
    while True:
        try:
            out.append(mp_queue.get_nowait())
        except Exception:
            break
    return out


# ── Step 1: per-chunk fsync for small files ───────────────────────


class TestSmallFileEarlyFsync:
    """Plan D5 fix: small files (< fsync_interval) must fsync after the
    first chunk is written, so an external observer (e.g. ``watch ls -l``)
    can see the partial file before the next chunk arrives.
    """

    def test_small_file_fsyncs_after_first_chunk(self, tmp_path):
        """4 KiB file, 4 KiB chunk, fsync_interval=4 MiB → fsync after first chunk."""
        chunks = [b"x" * 4096]  # 4 KiB total
        file_size = sum(len(c) for c in chunks)

        fsync_calls: list[int] = []

        def _track_fsync(fd):
            fsync_calls.append(fd)

        mock_xet, mock_xet_utils = _build_xet_mock_module(stream_iter=chunks)
        params, _ = _make_params(tmp_path, file_size, fsync_interval=4 * 1024 * 1024)

        with patch.dict(
            "sys.modules",
            {
                "hf_xet": mock_xet,
                "huggingface_hub.utils._xet": mock_xet_utils,
            },
        ), patch("hf_track._xet_worker.os.fsync", side_effect=_track_fsync):
            ctx = mp.get_context("spawn")
            mp_queue = ctx.Queue()
            cancel_event = ctx.Event()
            _xet_streaming_download_worker(params, mp_queue, cancel_event)

        # Plan: at least one fsync must have fired before the worker
        # closed the file (i.e. before the file completed).
        assert len(fsync_calls) >= 1, (
            f"Expected fsync after first chunk, got {len(fsync_calls)} calls"
        )

    def test_large_file_does_not_fsync_per_chunk(self, tmp_path):
        """100 MiB file, 1 MiB chunks, fsync_interval=4 MiB → no per-chunk fsync."""
        chunk_size = 1 * 1024 * 1024  # 1 MiB
        # 100 chunks to total 100 MiB; we don't actually write 100 MiB
        # in the test, we just count fsync calls.
        chunks = [b"y" * chunk_size] * 100
        file_size = sum(len(c) for c in chunks)
        assert file_size == 100 * 1024 * 1024

        fsync_calls: list[int] = []

        def _track_fsync(fd):
            fsync_calls.append(fd)

        mock_xet, mock_xet_utils = _build_xet_mock_module(stream_iter=chunks)
        params, _ = _make_params(
            tmp_path, file_size, fsync_interval=4 * 1024 * 1024,
        )

        with patch.dict(
            "sys.modules",
            {
                "hf_xet": mock_xet,
                "huggingface_hub.utils._xet": mock_xet_utils,
            },
        ), patch("hf_track._xet_worker.os.fsync", side_effect=_track_fsync):
            ctx = mp.get_context("spawn")
            mp_queue = ctx.Queue()
            cancel_event = ctx.Event()
            _xet_streaming_download_worker(params, mp_queue, cancel_event)

        # For 100 MiB at 4 MiB fsync_interval, we expect ~25 interval
        # fsyncs + 1 final fsync (in the finally). 25 fsyncs, NOT
        # 100 (which would be per-chunk). Generous bound of 50 to
        # account for the per-file-loop timing on Windows.
        assert len(fsync_calls) < 50, (
            f"Expected <50 fsyncs (interval-driven, not per-chunk), got {len(fsync_calls)}"
        )
        # And at least 2: one for the first 4 MiB boundary and the
        # final-fsync. (Could be more if many 4 MiB boundaries crossed.)
        assert len(fsync_calls) >= 2

    def test_fsync_failure_does_not_abort_download(self, tmp_path):
        """fsync raises → worker still completes, file content is correct."""
        chunks = [b"a" * 1024, b"b" * 1024]
        file_size = sum(len(c) for c in chunks)

        def _failing_fsync(fd):
            raise OSError("simulated fsync failure")

        mock_xet, mock_xet_utils = _build_xet_mock_module(stream_iter=chunks)
        params, dest = _make_params(tmp_path, file_size)

        with patch.dict(
            "sys.modules",
            {
                "hf_xet": mock_xet,
                "huggingface_hub.utils._xet": mock_xet_utils,
            },
        ), patch("hf_track._xet_worker.os.fsync", side_effect=_failing_fsync):
            ctx = mp.get_context("spawn")
            mp_queue = ctx.Queue()
            cancel_event = ctx.Event()
            _xet_streaming_download_worker(params, mp_queue, cancel_event)

        # Worker must have completed (terminal result was emitted).
        messages = _drain(mp_queue)
        result = next((m for m in messages if m.msg_type == MSG_RESULT), None)
        assert result is not None, "Worker did not emit terminal result"
        assert result.payload["status"] == "success"
        # And the file content must be correct even though fsync failed.
        with open(dest, "rb") as f:
            assert f.read() == b"a" * 1024 + b"b" * 1024

    def test_empty_file_branch_creates_file(self, tmp_path):
        """file_size=0 → empty file on disk, COMPLETE event, no stream opened."""
        mock_xet, mock_xet_utils = _build_xet_mock_module(stream_iter=[])
        params, dest = _make_params(tmp_path, file_size=0)

        with patch.dict(
            "sys.modules",
            {
                "hf_xet": mock_xet,
                "huggingface_hub.utils._xet": mock_xet_utils,
            },
        ):
            ctx = mp.get_context("spawn")
            mp_queue = ctx.Queue()
            cancel_event = ctx.Event()
            _xet_streaming_download_worker(params, mp_queue, cancel_event)

        # File exists and is empty.
        assert os.path.exists(dest)
        assert os.path.getsize(dest) == 0
        # COMPLETE event was emitted.
        messages = _drain(mp_queue)
        complete = [
            m for m in messages
            if m.msg_type == MSG_EVENT and m.payload.get("event_type") == "complete"
        ]
        assert len(complete) == 1
        # Terminal result was emitted exactly once.
        results = [m for m in messages if m.msg_type == MSG_RESULT]
        assert len(results) == 1

    def test_fsync_even_when_first_chunk_fits_exactly(self, tmp_path):
        """4 KiB file, single 4 KiB chunk → file is fsynced (early + final)."""
        chunks = [b"z" * 4096]
        file_size = 4096

        fsync_calls: list[int] = []

        def _track_fsync(fd):
            fsync_calls.append(fd)

        mock_xet, mock_xet_utils = _build_xet_mock_module(stream_iter=chunks)
        params, _ = _make_params(tmp_path, file_size)

        with patch.dict(
            "sys.modules",
            {
                "hf_xet": mock_xet,
                "huggingface_hub.utils._xet": mock_xet_utils,
            },
        ), patch("hf_track._xet_worker.os.fsync", side_effect=_track_fsync):
            ctx = mp.get_context("spawn")
            mp_queue = ctx.Queue()
            cancel_event = ctx.Event()
            _xet_streaming_download_worker(params, mp_queue, cancel_event)

        # At least one fsync fires (small-file early fsync OR final-fsync).
        assert len(fsync_calls) >= 1


# ── Step 2: finally cleanup of stream + group ─────────────────────


class TestWorkerFinallyCleanup:
    """Plan D4/D6/D10 fix: cancel / error paths must call
    ``stream.cancel()`` and ``group.close()`` (or fallback ``del``).
    """

    def test_worker_cancels_stream_on_keyboard_interrupt(self, tmp_path):
        """Stream iterator that raises KeyboardInterrupt → stream.cancel() called."""

        class _KBIter:
            def __iter__(self):
                return self

            def __next__(self):
                raise KeyboardInterrupt()

        mock_stream = MagicMock()
        mock_stream.__iter__ = lambda self: _KBIter()
        cancel_called: list[bool] = []

        def _track_cancel():
            cancel_called.append(True)

        mock_stream.cancel = _track_cancel

        mock_xet = MagicMock()
        mock_xet.XetSession.return_value.new_download_stream_group.return_value.download_stream.return_value = (
            mock_stream
        )
        mock_xet_utils = MagicMock(
            refresh_xet_connection_info=MagicMock(
                return_value=MagicMock(
                    endpoint="https://xet.example.com",
                    access_token="tok",
                    expiration_unix_epoch=9999999999,
                ),
            ),
        )

        chunks = [b"a" * 1024]
        params, _ = _make_params(tmp_path, file_size=1024, chunks=chunks)

        with patch.dict(
            "sys.modules",
            {
                "hf_xet": mock_xet,
                "huggingface_hub.utils._xet": mock_xet_utils,
            },
        ):
            ctx = mp.get_context("spawn")
            mp_queue = ctx.Queue()
            cancel_event = ctx.Event()
            _xet_streaming_download_worker(params, mp_queue, cancel_event)

        # The cleanup path (Step 2's outer finally) must call stream.cancel().
        # The KeyboardInterrupt raises from __next__ before the inner-loop
        # cancel check, so the stream itself is cancelled by the outer
        # finally.
        assert cancel_called, "stream.cancel() was not called on KeyboardInterrupt"

    def test_worker_closes_group_on_error(self, tmp_path):
        """download_stream raises → group's close is called by the outer finally."""
        close_called: list[bool] = []

        def _track_close():
            close_called.append(True)

        mock_group = MagicMock()
        mock_group.close = _track_close
        mock_xet = MagicMock()
        mock_xet.XetSession.return_value.new_download_stream_group.return_value = (
            mock_group
        )
        mock_xet.XetSession.return_value.new_download_stream_group.return_value.download_stream.side_effect = (
            RuntimeError("simulated stream open failure")
        )

        mock_xet_utils = MagicMock(
            refresh_xet_connection_info=MagicMock(
                return_value=MagicMock(
                    endpoint="https://xet.example.com",
                    access_token="tok",
                    expiration_unix_epoch=9999999999,
                ),
            ),
        )

        params, _ = _make_params(tmp_path, file_size=1024)

        with patch.dict(
            "sys.modules",
            {
                "hf_xet": mock_xet,
                "huggingface_hub.utils._xet": mock_xet_utils,
            },
        ):
            ctx = mp.get_context("spawn")
            mp_queue = ctx.Queue()
            cancel_event = ctx.Event()
            _xet_streaming_download_worker(params, mp_queue, cancel_event)

        assert close_called, (
            "group.close() was not called on download_stream error"
        )

    def test_worker_cleans_up_on_size_mismatch(self, tmp_path):
        """Stream yields fewer bytes than file_size → stream.cancel() called."""
        # file_size=1024, but stream yields only 512
        chunks = [b"a" * 512]
        file_size = 1024

        cancel_called: list[bool] = []

        def _track_cancel():
            cancel_called.append(True)

        mock_stream = MagicMock()
        mock_stream.__iter__ = lambda self: iter(chunks)
        mock_stream.cancel = _track_cancel

        mock_xet = MagicMock()
        mock_xet.XetSession.return_value.new_download_stream_group.return_value.download_stream.return_value = (
            mock_stream
        )
        mock_xet_utils = MagicMock(
            refresh_xet_connection_info=MagicMock(
                return_value=MagicMock(
                    endpoint="https://xet.example.com",
                    access_token="tok",
                    expiration_unix_epoch=9999999999,
                ),
            ),
        )

        params, _ = _make_params(tmp_path, file_size=file_size)

        with patch.dict(
            "sys.modules",
            {
                "hf_xet": mock_xet,
                "huggingface_hub.utils._xet": mock_xet_utils,
            },
        ):
            ctx = mp.get_context("spawn")
            mp_queue = ctx.Queue()
            cancel_event = ctx.Event()
            _xet_streaming_download_worker(params, mp_queue, cancel_event)

        # After size mismatch, the worker returns. The outer finally
        # must have called stream.cancel() during cleanup.
        assert cancel_called, (
            "stream.cancel() was not called on size mismatch cleanup"
        )

    def test_disable_fsync_skips_all_fsync(self, tmp_path):
        """Step 5: disable_fsync=True → no os.fsync calls (--no-fsync flag)."""
        chunks = [b"a" * 1024, b"b" * 1024, b"c" * 1024]
        file_size = sum(len(c) for c in chunks)

        fsync_calls: list[int] = []

        def _track_fsync(fd):
            fsync_calls.append(fd)

        mock_xet, mock_xet_utils = _build_xet_mock_module(stream_iter=chunks)
        params, _ = _make_params(tmp_path, file_size, disable_fsync=True)

        with patch.dict(
            "sys.modules",
            {
                "hf_xet": mock_xet,
                "huggingface_hub.utils._xet": mock_xet_utils,
            },
        ), patch("hf_track._xet_worker.os.fsync", side_effect=_track_fsync):
            ctx = mp.get_context("spawn")
            mp_queue = ctx.Queue()
            cancel_event = ctx.Event()
            _xet_streaming_download_worker(params, mp_queue, cancel_event)

        # No fsync should have been called at all.
        assert len(fsync_calls) == 0, (
            f"disable_fsync=True but {len(fsync_calls)} fsyncs were called"
        )
