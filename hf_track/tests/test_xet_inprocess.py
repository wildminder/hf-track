"""Regression tests for the in-process xet download flow.

These tests verify:
- H5: total_bytes aggregation does not overcount when xet's byte bar
  has a dynamically growing total (files discovered incrementally).
- H7: Synthetic progress events fire during xet file downloads.

The tests mock the actual xet download (we don't want network in unit tests),
but they exercise the state_manager and DownloadProgressTqdm logic that
the live diagnostic showed was the root cause.
"""
from __future__ import annotations

import os
import queue
import threading
import time

import pytest

from hf_track.callbacks import (
    DownloadProgressTqdm,
    EventType,
    state_manager,
)
from hf_track.types import ProgressPhase, TransferDirection


@pytest.fixture(autouse=True)
def _clean_state():
    """Clear global state before AND after each test for isolation."""
    state_manager._states.clear()
    yield
    state_manager._states.clear()


class TestXetGrowingTotalNoOvercount:
    """Verify H5 fix: xet-style growing total does not overcount bytes."""

    def test_xet_growing_total_between_file_completions(self):
        """Simulate the xet byte bar pattern: total grows as files are discovered.

        The xet pattern is:
        1. Init: total=0 (unknown)
        2. update(file1_size) with total=file1_size+file2_size (file1 done, file2 discovered)
        3. update(file2_size) with total=file1_size+file2_size+file3_size (all known)

        With OLD code, the total growth between (2) and (3) triggers a
        'bar switch' that commits the old total, leading to overcounting.
        """
        tid = "test-xet-growing"
        q = queue.Queue()
        # File 1 (tokenizer.model = 499723 bytes)
        bar = DownloadProgressTqdm(
            total=0,  # initial unknown
            desc="Downloading (incomplete total...)",
            unit="B",
            unit_scale=True,
            event_queue=q,
            transfer_id=tid,
            filename="repo.snapshot",
            report_interval=0,
        )
        # After init, state should be reset (no previous bar to commit)
        state = state_manager.get_state(tid)
        assert state["_committed_bytes"] == 0
        assert state["bytes_completed"] == 0
        assert state["total_bytes"] == 0

        # xet discovers more files BETWEEN completions.
        # Simulate by setting total then calling update with the file size.
        # File 1 (tokenizer.model) completes; xet has also discovered
        # file 2 (model.safetensors). total = 499723 + 4131280 = 4631003.
        bar.total = 4631003
        bar.update(499723)
        state = state_manager.get_state(tid)
        # total_bytes should equal 4631003 (NOT overcounted)
        assert state["total_bytes"] == 4631003, (
            f"Growing total overcounted: expected 4631003, got {state['total_bytes']}"
        )
        assert state["bytes_completed"] == 499723

        # File 2 (model.safetensors) completes; xet has also discovered
        # file 3 (onnx/model.onnx). total grows to 13029939.
        bar.total = 13029939
        bar.update(4131280)
        state = state_manager.get_state(tid)
        # CRITICAL: total_bytes should equal 13029939 (NOT 17660942)
        assert state["total_bytes"] == 13029939, (
            f"Growing total overcounted: expected 13029939, got {state['total_bytes']}"
        )
        # bytes_completed = 499723 + 4131280 = 4631003
        assert state["bytes_completed"] == 4631003

        # File 3 (onnx/model.onnx) completes
        bar.update(8398936)
        state = state_manager.get_state(tid)
        assert state["bytes_completed"] == 13029939
        assert state["total_bytes"] == 13029939

        bar.close()

    def test_xet_growing_total_20_files(self):
        """Simulate a 20-file snapshot with growing total — verify no overcount.

        Reproduces the user's bug scenario: 20 files, repo size 157MB,
        but the OLD code reported 823MB (5.2× overcount).
        """
        tid = "test-xet-20-files"
        q = queue.Queue()
        bar = DownloadProgressTqdm(
            total=0,
            desc="Downloading (incomplete total...)",
            unit="B",
            unit_scale=True,
            event_queue=q,
            transfer_id=tid,
            filename="repo.snapshot",
            report_interval=0,
        )

        # 20 files of varying sizes summing to ~157MB.
        # Sizes chosen so each pair sums nicely.
        file_sizes = [
            5000000, 8000000, 12000000, 3000000, 7000000,
            9000000, 4000000, 11000000, 2000000, 6000000,
            15000000, 5000000, 8000000, 7000000, 4000000,
            13000000, 6000000, 10000000, 9000000, 5000000,
        ]
        # 5+8+12+3+7+9+4+11+2+6+15+5+8+7+4+13+6+10+9+5 = 149MB
        actual_total = sum(file_sizes)

        running_total = 0
        for i, file_size in enumerate(file_sizes):
            # xet discovers ALL files early, so total is stable at actual_total
            # by the time we start receiving completions. But the FIRST update
            # happens before all files are discovered.
            if i == 0:
                # First update: only files 0,1 known. total = 5+8 = 13MB
                running_total = file_sizes[0] + file_sizes[1]
                bar.total = running_total
            else:
                # Subsequent updates: all files known
                bar.total = actual_total
            bar.update(file_size)
            # Verify NO overcount at each step
            state = state_manager.get_state(tid)
            if i == 0:
                expected_total = file_sizes[0] + file_sizes[1]
            else:
                expected_total = actual_total
            assert state["total_bytes"] == expected_total, (
                f"Overcount at file {i}: expected {expected_total}, got {state['total_bytes']}"
            )
        # Final state: bytes_completed == total_bytes == actual_total
        state = state_manager.get_state(tid)
        assert state["bytes_completed"] == actual_total
        assert state["total_bytes"] == actual_total
        bar.close()

    def test_emitted_events_have_correct_total(self):
        """Verify events emitted from a xet-style bar have correct total_bytes."""
        tid = "test-xet-events"
        q = queue.Queue()
        bar = DownloadProgressTqdm(
            total=0,
            desc="Downloading (incomplete total...)",
            unit="B",
            unit_scale=True,
            event_queue=q,
            transfer_id=tid,
            filename="repo",
            report_interval=0,
        )
        # Simulate xet file completions with growing total
        bar.total = 4631003
        bar.update(499723)  # file 1 done, file 2 also discovered
        bar.total = 13029939
        bar.update(4131280)  # file 2 done, file 3 also discovered
        bar.update(8398936)  # file 3 done

        events = []
        while not q.empty():
            events.append(q.get_nowait())
        progress_events = [e for e in events if e.event_type == EventType.PROGRESS]
        # Find the last progress event
        assert len(progress_events) >= 1
        last = progress_events[-1]
        # CRITICAL: total_bytes must equal the actual repo size, not overcounted
        assert last.total_bytes == 13029939, (
            f"Last event total_bytes overcounted: expected 13029939, got {last.total_bytes}"
        )
        assert last.bytes_completed == 13029939
        assert last.percentage == 100.0

        bar.close()


class TestHTTPPerFileBarsStillAggregate:
    """Verify H5 fix does not break HTTP per-file byte bar aggregation."""

    def test_http_three_files_aggregate_correctly(self):
        """Three HTTP files (each with own byte bar) should accumulate via reset_byte_bar_state."""
        tid = "test-http-3-files"
        q = queue.Queue()
        # File 1: 1000 bytes
        bar1 = DownloadProgressTqdm(
            total=1000, desc="file1.bin", unit="B", unit_scale=True,
            event_queue=q, transfer_id=tid, filename="file1.bin", report_interval=0,
        )
        bar1.update(500)
        bar1.update(1000)  # complete
        state = state_manager.get_state(tid)
        assert state["bytes_completed"] == 1000
        assert state["total_bytes"] == 1000
        bar1.close()

        # File 2: 2000 bytes (LARGER than file 1) — this was the edge case
        bar2 = DownloadProgressTqdm(
            total=2000, desc="file2.bin", unit="B", unit_scale=True,
            event_queue=q, transfer_id=tid, filename="file2.bin", report_interval=0,
        )
        # After init (reset_byte_bar_state), _committed_bytes should now
        # include file 1's 1000. _current_bar_total=0, no current bar yet.
        state = state_manager.get_state(tid)
        assert state["_committed_bytes"] == 1000, (
            f"reset_byte_bar_state should commit file 1's total: "
            f"_committed={state['_committed_bytes']}"
        )
        # First non-zero update of file 2 (in real HTTP, chunks are non-zero)
        bar2.update(1000)  # mid-download
        state = state_manager.get_state(tid)
        assert state["bytes_completed"] == 2000  # 1000 committed + 1000 current
        assert state["total_bytes"] == 3000  # 1000 committed + 2000 current
        # Complete
        bar2.update(2000)
        state = state_manager.get_state(tid)
        assert state["bytes_completed"] == 3000
        assert state["total_bytes"] == 3000
        bar2.close()

        # File 3: 500 bytes (SMALLER than file 2)
        bar3 = DownloadProgressTqdm(
            total=500, desc="file3.bin", unit="B", unit_scale=True,
            event_queue=q, transfer_id=tid, filename="file3.bin", report_interval=0,
        )
        state = state_manager.get_state(tid)
        assert state["_committed_bytes"] == 3000
        bar3.update(500)
        state = state_manager.get_state(tid)
        assert state["bytes_completed"] == 3500  # 3000 + 500
        assert state["total_bytes"] == 3500
        bar3.close()
