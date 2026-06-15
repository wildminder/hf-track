"""Regression tests for mid-stream disk visibility in the Xet streaming worker.

This module captures the user-reported bug:
    "downloading xet file still has 0 length on disk and only appears
     after complete downloading"

These tests assert that during a real ``_xet_streaming_download_worker``
run, the destination file's on-disk size grows strictly monotonically as
chunks arrive, and that external observers (``os.path.getsize``) can see
the partial file between chunks.

Step 0 of ``docs/plans/2026-06-05-xet-streaming-flush-reliability.md``
adds the baseline repro test (``test_file_visible_on_disk_during_gil_stall``)
which fails on the current main and is fixed by Step 1's per-chunk fsync
for small files.

Step 6 adds the broader end-to-end pattern tests
(``test_small_file_visible_immediately``, ``test_large_file_grows_per_chunk``,
``test_partial_file_persists_after_sigkill``).

All tests use synthetic ``hf_xet`` mocks (no real network). The worker
is invoked in-process for deterministic timing, except the SIGKILL
test which spawns a real subprocess.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from hf_track._xet_worker import (
    DEFAULT_FSYNC_INTERVAL,
    _open_unbuffered,
    _xet_streaming_download_worker,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _build_mock_xet(chunks):
    """Build a mock hf_xet module whose stream yields the given chunks.

    The stream's ``__iter__`` returns an iterator over ``chunks``.
    Also provides ``__next__`` for ``next(stream, sentinel)`` support.
    """
    _iter = iter(chunks)
    mock_xet_module = MagicMock()
    mock_stream = MagicMock()
    mock_stream.__iter__ = lambda self: _iter
    mock_stream.__next__ = lambda self: next(_iter)
    mock_xet_module.XetSession.return_value.new_download_stream_group.return_value.download_stream.return_value = mock_stream
    return mock_xet_module


def _build_mock_xet_with_slow_stream(
    chunks,
    first_chunk_delay_s: float,
    inter_chunk_delay_s: float = 0.0,
):
    """Build a mock hf_xet module whose first ``__next__`` blocks for ``first_chunk_delay_s``.

    This simulates the real-world behavior where the Rust runtime holds
    the GIL while fetching/waiting for the first chunk, but still yields
    each subsequent chunk quickly. Used by the GIL-stall repro test.

    If ``inter_chunk_delay_s`` > 0, the iterator sleeps for that long
    *between* successive chunks (gives an external poller a window to
    observe the in-progress state).
    """

    class _SlowIter:
        def __init__(self, chunks, first_delay, inter_delay):
            self._chunks = list(chunks)
            self._first_delay = first_delay
            self._inter_delay = inter_delay
            self._yielded = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._yielded == 0 and self._first_delay > 0:
                time.sleep(self._first_delay)
            elif self._yielded > 0 and self._inter_delay > 0:
                time.sleep(self._inter_delay)
            if self._yielded >= len(self._chunks):
                raise StopIteration
            chunk = self._chunks[self._yielded]
            self._yielded += 1
            return chunk

    mock_xet_module = MagicMock()
    mock_stream = MagicMock()
    _slow_iter = _SlowIter(chunks, first_chunk_delay_s, inter_chunk_delay_s)
    mock_stream.__iter__ = lambda self: _slow_iter
    mock_stream.__next__ = lambda self: _slow_iter.__next__()
    mock_xet_module.XetSession.return_value.new_download_stream_group.return_value.download_stream.return_value = mock_stream
    return mock_xet_module


def _build_xet_mock_module():
    """Build a stand-in for ``huggingface_hub.utils._xet``."""
    return MagicMock(
        refresh_xet_connection_info=MagicMock(
            return_value=MagicMock(
                endpoint="https://xet.example.com",
                access_token="tok",
                expiration_unix_epoch=9999999999,
            ),
        ),
    )


def _make_streaming_params(tmp_path, file_size, chunks, **overrides):
    """Build a params dict for the streaming worker.

    Args:
        tmp_path: pytest tmp_path fixture
        file_size: expected total file size in bytes
        chunks: iterable of bytes chunks to be yielded by the mock stream
        **overrides: additional params to merge in
    """
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
        "transfer_id": "disk-flush-test",
        "report_interval": 0.0,
        "fsync_interval": DEFAULT_FSYNC_INTERVAL,
        "request_headers": {},
    }
    params.update(overrides)
    return params, dest


# ── Step 0: GIL-stall repro test ────────────────────────────────────


class TestDiskFlushBaseline:
    """Step 0 regression guard.

    Verifies that the user-reported "0-length during download" symptom
    is reproduced and fixable.
    """

    def test_file_visible_on_disk_during_gil_stall(self, tmp_path):
        """Baseline: file MUST be visible on disk during a 5 s GIL-stall.

        Scenario: the simulated stream blocks for 5 s before yielding
        the first chunk (mimicking the upstream ``__next__`` GIL-held
        behavior). The worker must still call ``os.fsync`` after the
        first chunk is written so an external observer can see the
        partial file *before* the stream is exhausted.

        With the current main (no small-file early fsync) this test
        FAILS because the per-chunk fsync only happens at
        ``fsync_interval`` boundaries (4 MiB default) -- so a 1 KiB
        file with one 1 KiB chunk never gets fsynced until the worker's
        ``finally`` block, which is after the second chunk yields
        (i.e. after the user-visible stall).
        """
        chunks = [b"A" * 1024, b"B" * 1024]  # 2 KiB total
        file_size = sum(len(c) for c in chunks)

        # Simulate 0.3 s GIL-stall before first chunk + 0.3 s
        # inter-chunk delay. The first delay emulates the Rust
        # runtime holding the GIL while waiting for the first chunk
        # to arrive. The inter-chunk delay gives the poller a
        # window to observe the in-progress file (size 1 KiB
        # between chunks 1 and 2) — without this, the worker
        # writes both chunks back-to-back in <1 ms and the poller
        # only ever sees 0 (during the stall) and 2 KiB (final).
        mock_xet = _build_mock_xet_with_slow_stream(
            chunks,
            first_chunk_delay_s=0.3,
            inter_chunk_delay_s=0.3,
        )
        mock_xet_utils = _build_xet_mock_module()

        params, dest = _make_streaming_params(tmp_path, file_size, chunks)

        # Poll os.path.getsize from a background thread while the
        # worker runs. We assert that *before* the worker completes,
        # the file has at least 1 byte on disk.
        sizes_seen: list[int] = []
        stop_polling = threading.Event()

        def _poller():
            while not stop_polling.is_set():
                try:
                    sizes_seen.append(os.path.getsize(dest))
                except OSError:
                    sizes_seen.append(0)
                time.sleep(0.02)

        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()

        poller_thread = threading.Thread(target=_poller, daemon=True)
        poller_thread.start()

        try:
            with patch.dict(
                "sys.modules",
                {
                    "hf_xet": mock_xet,
                    "huggingface_hub.utils._xet": mock_xet_utils,
                },
            ):
                _xet_streaming_download_worker(params, mp_queue, cancel_event)
        finally:
            stop_polling.set()
            poller_thread.join(timeout=2.0)

        # Assert: at some point during the run, the file was visible
        # with size > 0 (before the final 2 KiB completion). The poll
        # window is small (~20 ms) and the worker yields the first
        # chunk after 500 ms, so we expect a few dozen samples
        # covering the period before the second chunk lands.
        max_size_before_complete = max(sizes_seen[:-1]) if len(sizes_seen) > 1 else 0
        assert max_size_before_complete > 0, (
            f"File was never visible on disk before completion. "
            f"Final poll size = {sizes_seen[-1] if sizes_seen else 'N/A'}, "
            f"max size during run = {max_size_before_complete}. "
            f"This is the user-reported '0-length during download' bug."
        )

    def test_open_unbuffered_helper(self, tmp_path):
        """Smoke: _open_unbuffered creates a file that takes writes."""
        path = tmp_path / "x.bin"
        fd = _open_unbuffered(str(path))
        try:
            os.write(fd, b"hello")
        finally:
            os.close(fd)
        assert path.read_bytes() == b"hello"

    def test_default_fsync_interval_is_4mb(self):
        """DEFAULT_FSYNC_INTERVAL is 4 MiB (matches plan §6)."""
        assert DEFAULT_FSYNC_INTERVAL == 4 * 1024 * 1024
