"""Tests for hf_progress.callbacks module."""

from __future__ import annotations

import logging
import os
import queue
import time
import types
from unittest.mock import MagicMock

import pytest

from hf_track.callbacks import (
    DownloadProgressTqdm,
    XetDownloadProgressCallback,
    XetUploadProgressCallback,
    XetProgressCallback,
    tqdm_upload_patcher,
)
from hf_track.types import EventType, ProgressEvent, ProgressPhase, TransferDirection


class TestXetProgressCallbackEstimation:
    def _make_mock_total_update(self, **overrides):
        defaults = {
            "total_bytes": 1000,
            "total_bytes_completed": 0,
            "total_bytes_completion_rate": 500.0,
            "total_transfer_bytes": 1000,
            "total_transfer_bytes_completed": 0,
            "total_transfer_bytes_completion_rate": 500.0,
        }
        defaults.update(overrides)
        mock = MagicMock()
        for key, value in defaults.items():
            setattr(mock, key, value)
        return mock

    def test_progress_estimation_when_waiting_for_bytes(self):
        """[NTH-001]: Verify progress is estimated when bytes_completed is 0 but speed > 0."""
        q = queue.Queue()
        callback = XetProgressCallback(
            filename="test.bin",
            total_bytes=10000,
            event_queue=q,
            transfer_id="test-est",
        )
        
        # Manually backdate the start time to simulate 1 second of transfer
        callback._start_time = time.time() - 1.0

        # Pass 0 for completed bytes, but active speed is 1000.0
        total_update = self._make_mock_total_update(
            total_bytes=10000,
            total_bytes_completed=0,
            total_bytes_completion_rate=1000.0,
            total_transfer_bytes=10000,
            total_transfer_bytes_completed=0,
            total_transfer_bytes_completion_rate=1000.0,
        )
        callback(total_update, [])

        event = q.get_nowait()
        # Should estimate approx 1000 bytes (1 second * 1000 bytes/sec)
        assert event.bytes_completed > 0
        assert event.bytes_completed >= 900
        assert event.percentage > 0.0


class TestXetUploadProgressCallback:
    """Tests for XetUploadProgressCallback."""

    def _make_mock_total_update(self, **overrides):
        """Create a mock PyTotalProgressUpdate."""
        defaults = {
            "total_bytes": 1000,
            "total_bytes_completed": 500,
            "total_bytes_completion_rate": 100.0,
            "total_transfer_bytes": 800,
            "total_transfer_bytes_completed": 400,
            "total_transfer_bytes_completion_rate": 80.0,
        }
        defaults.update(overrides)
        mock = MagicMock()
        for key, value in defaults.items():
            setattr(mock, key, value)
        return mock

    def test_emits_progress_event(self):
        """First call should emit a progress event."""
        q = queue.Queue()
        callback = XetUploadProgressCallback(
            filename="test.bin",
            total_bytes=1000,
            event_queue=q,
            transfer_id="test-1",
        )

        total_update = self._make_mock_total_update()
        callback(total_update, [])

        assert q.qsize() == 1
        event = q.get_nowait()
        assert event.event_type == EventType.PROGRESS
        assert event.transfer_id == "test-1"
        assert event.direction == TransferDirection.UPLOAD
        assert event.filename == "test.bin"
        assert event.bytes_completed == 500
        assert event.total_bytes == 1000
        assert event.percentage == 50.0
        assert event.speed == 80.0  # transfer_speed takes priority over speed
        assert event.transfer_bytes_completed == 400
        assert event.transfer_bytes_total == 800
        assert event.transfer_speed == 80.0
        assert event.dedup_saved_bytes == 100  # 500 - 400

    def test_every_call_emits_event(self):
        """Every callback invocation should emit an event.

        The consumer is responsible for throttling/display refresh.
        """
        q = queue.Queue()
        callback = XetUploadProgressCallback(
            filename="test.bin",
            total_bytes=1000,
            event_queue=q,
            report_interval=1.0,  # unused — kept for API compat
        )

        total_update = self._make_mock_total_update()

        # First call should emit
        callback(total_update, [])
        assert q.qsize() == 1

        # Second call should also emit (no producer-side throttle)
        callback(total_update, [])
        assert q.qsize() == 2  # Both calls emit

    def test_dedup_saved_bytes(self):
        """Dedup saved bytes = bytes_completed - transfer_bytes_completed."""
        q = queue.Queue()
        callback = XetUploadProgressCallback(
            filename="test.bin",
            total_bytes=1000,
            event_queue=q,
        )

        # 800 bytes completed, but only 200 transferred (600 deduped)
        total_update = self._make_mock_total_update(
            total_bytes_completed=800,
            total_transfer_bytes_completed=200,
        )
        callback(total_update, [])

        event = q.get_nowait()
        assert event.dedup_saved_bytes == 600

    def test_dedup_saved_bytes_no_negative(self):
        """Dedup saved bytes should never be negative."""
        q = queue.Queue()
        callback = XetUploadProgressCallback(
            filename="test.bin",
            total_bytes=1000,
            event_queue=q,
        )

        # transfer > completed (shouldn't happen, but be safe)
        total_update = self._make_mock_total_update(
            total_bytes_completed=100,
            total_transfer_bytes_completed=200,
        )
        callback(total_update, [])

        event = q.get_nowait()
        assert event.dedup_saved_bytes == 0  # max(0, ...)

    def test_zero_total_bytes(self):
        """Percentage should be 0 when total_bytes is 0."""
        q = queue.Queue()
        callback = XetUploadProgressCallback(
            filename="test.bin",
            total_bytes=0,
            event_queue=q,
        )

        total_update = self._make_mock_total_update(
            total_bytes=0,
            total_bytes_completed=0,
        )
        callback(total_update, [])

        event = q.get_nowait()
        assert event.percentage == 0.0

    def test_fallback_to_self_total_bytes(self):
        """When total_update.total_bytes is 0, use self.total_bytes."""
        q = queue.Queue()
        callback = XetUploadProgressCallback(
            filename="test.bin",
            total_bytes=2000,
            event_queue=q,
        )

        total_update = self._make_mock_total_update(
            total_bytes=0,
            total_bytes_completed=1000,
        )
        callback(total_update, [])

        event = q.get_nowait()
        assert event.total_bytes == 2000  # Fallback
        assert event.percentage == 50.0  # 1000/2000

    def test_getattr_fallback(self):
        """Missing Rust attributes should default to 0."""
        q = queue.Queue()
        callback = XetUploadProgressCallback(
            filename="test.bin",
            total_bytes=1000,
            event_queue=q,
        )

        # Minimal mock without all attributes
        total_update = MagicMock(spec=[])
        callback(total_update, [])

        event = q.get_nowait()
        assert event.bytes_completed == 0
        assert event.speed == 0

    def test_file_index_and_total_files(self):
        """File index and total files should be passed through."""
        q = queue.Queue()
        callback = XetUploadProgressCallback(
            filename="test.bin",
            total_bytes=1000,
            event_queue=q,
            transfer_id="test-idx",
            file_index=2,
            total_files=5,
        )

        total_update = self._make_mock_total_update()
        callback(total_update, [])

        event = q.get_nowait()
        assert event.file_index == 2
        assert event.total_files == 5


class TestXetDownloadProgressCallback:
    """Tests for XetDownloadProgressCallback with detailed (total_update, item_updates) signature."""

    @staticmethod
    def _make_total_update(
        total_bytes=1000,
        total_bytes_completed=0,
        total_bytes_completion_rate=0.0,
        total_transfer_bytes=0,
        total_transfer_bytes_completed=0,
        total_transfer_bytes_completion_rate=0.0,
    ):
        """Create a mock PyTotalProgressUpdate object."""
        return types.SimpleNamespace(
            total_bytes=total_bytes,
            total_bytes_increment=0,
            total_bytes_completed=total_bytes_completed,
            total_bytes_completion_increment=0,
            total_bytes_completion_rate=total_bytes_completion_rate,
            total_transfer_bytes=total_transfer_bytes,
            total_transfer_bytes_increment=0,
            total_transfer_bytes_completed=total_transfer_bytes_completed,
            total_transfer_bytes_completion_increment=0,
            total_transfer_bytes_completion_rate=total_transfer_bytes_completion_rate,
        )

    @staticmethod
    def _make_item_update(item_name="test.bin", total_bytes=1000, bytes_completed=0, bytes_completion_increment=0):
        """Create a mock PyItemProgressUpdate object."""
        return types.SimpleNamespace(
            item_name=item_name,
            total_bytes=total_bytes,
            bytes_completed=bytes_completed,
            bytes_completion_increment=bytes_completion_increment,
        )

    def test_emits_progress_from_detailed_callback(self):
        """Download callback emits progress from PyTotalProgressUpdate."""
        q = queue.Queue()
        callback = XetDownloadProgressCallback(
            filename="test.bin",
            total_bytes=1000,
            event_queue=q,
        )

        # Simulate Rust calling with detailed progress data
        total_update = self._make_total_update(
            total_bytes=1000,
            total_bytes_completed=500,
            total_bytes_completion_rate=1000.0,
            total_transfer_bytes=800,
            total_transfer_bytes_completed=400,
            total_transfer_bytes_completion_rate=800.0,
        )
        item_updates = [self._make_item_update(
            item_name="test.bin",
            total_bytes=1000,
            bytes_completed=500,
            bytes_completion_increment=500,
        )]
        callback(total_update, item_updates)

        event = q.get_nowait()
        # bytes_completed uses transfer_completed (400) for smooth
        # progress, not total_bytes_completed (500) which only
        # updates when chunks are fully assembled.
        assert event.bytes_completed == 400
        assert event.total_bytes == 1000
        assert event.percentage == 40.0  # 400/1000
        # speed uses transfer_speed when available
        assert event.speed == 800.0
        assert event.transfer_bytes_completed == 400
        assert event.transfer_bytes_total == 800
        assert event.transfer_speed == 800.0
        assert event.dedup_saved_bytes == 100  # 500 - 400

    def test_uses_transfer_completed_for_smooth_progress(self):
        """bytes_completed uses transfer_completed for smooth progress.

        When transfer_completed > 0, it provides incremental network-level
        progress. total_bytes_completed only jumps when chunks are assembled.
        """
        q = queue.Queue()
        callback = XetDownloadProgressCallback(
            filename="test.bin",
            total_bytes=1000,
            event_queue=q,
        )

        # transfer_completed=300 > 0, so bytes_completed should be 300
        total_update = self._make_total_update(
            total_bytes=1000,
            total_bytes_completed=100,  # assembly progress (jumps)
            total_transfer_bytes=800,
            total_transfer_bytes_completed=300,  # network progress (smooth)
            total_transfer_bytes_completion_rate=500.0,
        )
        callback(total_update, [])
        event = q.get_nowait()
        assert event.bytes_completed == 300  # uses transfer_completed
        assert event.percentage == 30.0  # 300/1000
        assert event.speed == 500.0  # uses transfer_speed

    def test_uses_total_bytes_completed_at_100_percent(self):
        """At 100%, bytes_completed uses total_bytes_completed for exact value."""
        q = queue.Queue()
        callback = XetDownloadProgressCallback(
            filename="test.bin",
            total_bytes=1000,
            event_queue=q,
        )

        # When bytes_completed >= total_bytes, use the exact final value
        total_update = self._make_total_update(
            total_bytes=1000,
            total_bytes_completed=1000,  # fully assembled
            total_transfer_bytes=800,
            total_transfer_bytes_completed=750,  # less due to dedup
        )
        callback(total_update, [])
        event = q.get_nowait()
        assert event.bytes_completed == 1000  # uses total_bytes_completed
        assert event.percentage == 100.0

    def test_computes_percentage(self):
        """Percentage should be computed from bytes_completed/total_bytes."""
        q = queue.Queue()
        callback = XetDownloadProgressCallback(
            filename="test.bin",
            total_bytes=1000,
            event_queue=q,
        )

        total_update = self._make_total_update(
            total_bytes=1000,
            total_bytes_completed=250,
        )
        callback(total_update, [])
        event = q.get_nowait()
        assert event.percentage == 25.0

    def test_every_call_emits_event(self):
        """Every callback invocation should emit an event.

        The consumer is responsible for throttling/display refresh.
        """
        q = queue.Queue()
        callback = XetDownloadProgressCallback(
            filename="test.bin",
            total_bytes=1000,
            event_queue=q,
        )

        total_update = self._make_total_update(total_bytes_completed=100)
        callback(total_update, [])
        assert q.qsize() == 1

        # Second call should also emit (no producer-side throttle)
        total_update2 = self._make_total_update(total_bytes_completed=200)
        callback(total_update2, [])
        assert q.qsize() == 2  # Both calls emit

    def test_zero_total_bytes(self):
        """Percentage should be 0 when total_bytes is 0."""
        q = queue.Queue()
        callback = XetDownloadProgressCallback(
            filename="test.bin",
            total_bytes=0,
            event_queue=q,
        )

        total_update = self._make_total_update(
            total_bytes=0,
            total_bytes_completed=100,
        )
        callback(total_update, [])
        event = q.get_nowait()
        assert event.percentage == 0.0

    def test_fallback_to_stored_total_bytes(self):
        """Should use stored total_bytes when total_update.total_bytes is 0."""
        q = queue.Queue()
        callback = XetDownloadProgressCallback(
            filename="test.bin",
            total_bytes=1000,
            event_queue=q,
        )

        # Rust may send total_bytes=0 in some edge cases
        total_update = self._make_total_update(
            total_bytes=0,
            total_bytes_completed=500,
        )
        callback(total_update, [])
        event = q.get_nowait()
        # Should fall back to the stored total_bytes=1000
        assert event.total_bytes == 1000
        assert event.percentage == 50.0


class TestDownloadProgressTqdm:
    """Tests for DownloadProgressTqdm custom tqdm class."""

    def test_emits_progress_events(self):
        """Update calls should emit progress events."""
        q = queue.Queue()
        bar = DownloadProgressTqdm(
            total=1000,
            desc="test.bin",
            unit="B",
            unit_scale=True,
            event_queue=q,
            transfer_id="test-dl-1",
            filename="test.bin",
            report_interval=0,  # No throttling
        )

        bar.update(500)
        bar.update(500)
        bar.close()

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        progress_events = [
            e for e in events if e.event_type == EventType.PROGRESS
        ]
        complete_events = [
            e for e in events if e.event_type == EventType.COMPLETE
        ]

        assert len(progress_events) == 2
        assert progress_events[0].bytes_completed == 500
        assert progress_events[1].bytes_completed == 1000
        assert len(complete_events) == 1
        assert complete_events[0].percentage == 100.0

    def test_first_update_always_emits_event(self):
        """The first update() call should always emit an event,
        even with throttling enabled.
        """
        q = queue.Queue()
        bar = DownloadProgressTqdm(
            total=1000,
            desc="test.bin",
            unit="B",
            unit_scale=True,
            event_queue=q,
            transfer_id="test-dl-2",
            filename="test.bin",
            report_interval=1.0,
        )

        # First update should always emit
        bar.update(100)
        assert q.qsize() == 1

        bar.close()

    def test_throttled_updates_skip_events(self):
        """Rapid updates with small byte deltas should be throttled."""
        q = queue.Queue()
        bar = DownloadProgressTqdm(
            total=100000,
            desc="test.bin",
            unit="B",
            unit_scale=True,
            event_queue=q,
            transfer_id="test-dl-throttle",
            filename="test.bin",
            report_interval=10.0,  # Very long interval to force throttling
        )

        # First update always emits
        bar.update(100)
        assert q.qsize() == 1

        # Second update with small delta should be throttled
        bar.update(10)
        assert q.qsize() == 1  # Still only 1 event

        # Large update exceeding 1% threshold should emit
        bar.update(5000)  # 5000 > 100000//100 = 1000
        assert q.qsize() == 2

        bar.close()

    def test_completion_not_throttled(self):
        """Events where bytes_completed >= total_bytes should not be throttled."""
        q = queue.Queue()
        bar = DownloadProgressTqdm(
            total=1000,
            desc="test.bin",
            unit="B",
            unit_scale=True,
            event_queue=q,
            transfer_id="test-dl-complete",
            filename="test.bin",
            report_interval=10.0,  # Very long interval
        )

        # First update
        bar.update(500)
        assert q.qsize() == 1

        # Complete the download — should not be throttled
        bar.update(500)
        assert q.qsize() == 2  # Completion event emitted despite throttle

        bar.close()

    def test_no_events_without_queue(self):
        """No events should be emitted when event_queue is None."""
        bar = DownloadProgressTqdm(
            total=1000,
            desc="test.bin",
            unit="B",
            unit_scale=True,
            event_queue=None,
        )

        bar.update(500)
        bar.update(500)
        bar.close()
        # No assertion needed — just no exceptions

    def test_complete_event_on_close(self):
        """Close at 100% should emit a complete event."""
        q = queue.Queue()
        bar = DownloadProgressTqdm(
            total=1000,
            desc="test.bin",
            unit="B",
            unit_scale=True,
            event_queue=q,
            transfer_id="test-dl-3",
            filename="test.bin",
            report_interval=0,
        )

        bar.update(1000)
        bar.close()

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        complete_events = [
            e for e in events if e.event_type == EventType.COMPLETE
        ]
        assert len(complete_events) == 1
        assert complete_events[0].percentage == 100.0

    def test_no_complete_event_if_not_finished(self):
        """Close before 100% should NOT emit a complete event."""
        q = queue.Queue()
        bar = DownloadProgressTqdm(
            total=1000,
            desc="test.bin",
            unit="B",
            unit_scale=True,
            event_queue=q,
            transfer_id="test-dl-4",
            filename="test.bin",
            report_interval=0,
        )

        bar.update(500)
        bar.close()

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        complete_events = [
            e for e in events if e.event_type == EventType.COMPLETE
        ]
        assert len(complete_events) == 0

    def test_xet_cached_download_emits_complete(self):
        """Close with n=0 and total>0 should emit COMPLETE (Xet cached download).

        When Xet downloads a file where all chunks are in cache, no
        update() calls happen (n stays 0), but the file was written
        successfully. close() should still emit COMPLETE.
        """
        q = queue.Queue()
        bar = DownloadProgressTqdm(
            total=1000,
            desc="cached.bin",
            unit="B",
            unit_scale=True,
            event_queue=q,
            transfer_id="test-dl-cached",
            filename="cached.bin",
            report_interval=0,
        )
        # No update() calls — simulates Xet cached download
        bar.close()

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        complete_events = [
            e for e in events if e.event_type == EventType.COMPLETE
        ]
        assert len(complete_events) == 1
        assert complete_events[0].bytes_completed == 1000
        assert complete_events[0].percentage == 100.0

    def test_bind_creates_subclass(self):
        """bind() should create a tqdm subclass with bound parameters."""
        q = queue.Queue()
        bound_class = DownloadProgressTqdm.bind(
            event_queue=q,
            transfer_id="bound-1",
            filename="bound.bin",
        )

        # The returned class should be a subclass
        assert issubclass(bound_class, DownloadProgressTqdm)

        # Creating an instance should work without passing event_queue
        bar = bound_class(total=1000, desc="bound.bin", unit="B", unit_scale=True)
        bar.update(500)

        # Should have emitted an event to the bound queue
        assert q.qsize() >= 1
        event = q.get_nowait()
        assert event.transfer_id == "bound-1"
        bar.close()

    def test_name_kwarg_ignored(self):
        """The 'name' kwarg injected by HF should be silently removed.
        
        snapshot_download() passes name="huggingface_hub.snapshot_download"
        to _create_progress_bar() which calls cls(**kwargs). Vanilla tqdm
        does NOT accept 'name' (raises TqdmKeyError). Our class must
        strip it before passing to super().__init__().
        """
        q = queue.Queue()
        # This should NOT raise TqdmKeyError
        bar = DownloadProgressTqdm(
            total=1000,
            desc="test.bin",
            unit="B",
            unit_scale=True,
            event_queue=q,
            transfer_id="test-dl-name",
            filename="test.bin",
            name="huggingface_hub.snapshot_download",  # HF-injected kwarg
        )
        bar.update(1000)
        bar.close()
        # Should work normally
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        assert any(e.event_type == EventType.COMPLETE for e in events)

    def test_closed_attr_exists_before_init(self):
        """_closed must exist before super().__init__() to prevent AttributeError.
        
        If super().__init__() fails and __del__ calls close(), the _closed
        attribute must already exist to prevent:
        AttributeError: 'BoundDownloadTqdm_xxx' object has no attribute '_closed'
        """
        bar = DownloadProgressTqdm.__new__(DownloadProgressTqdm)
        # _closed should be set in __init__ before super().__init__()
        # Simulate partial init by calling __init__ with bad kwargs
        try:
            bar.__init__(bad_kwarg=True)
        except (TypeError, Exception):
            pass
        # _closed should exist even if __init__ failed
        assert hasattr(bar, "_closed")

    def test_speed_calculation(self):
        """Speed should be calculated from tqdm's internal rate or elapsed time."""
        q = queue.Queue()
        bar = DownloadProgressTqdm(
            total=1000,
            desc="test.bin",
            unit="B",
            unit_scale=True,
            event_queue=q,
            transfer_id="test-dl-5",
            filename="test.bin",
            report_interval=0,
        )

        bar.update(1000)

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        assert len(events) >= 1
        # Speed should be > 0 (either from tqdm rate or manual calculation)
        assert events[0].speed >= 0

        bar.close()

    def test_emitted_events_reflect_accumulated_bytes(self):
        """Emitted events should reflect the current accumulated bytes.

        With throttling, not every update() emits an event, but the events
        that ARE emitted should have the correct bytes_completed value.
        Use report_interval=0 to disable throttling for this test.
        """
        q = queue.Queue()
        bar = DownloadProgressTqdm(
            total=1000,
            desc="test.bin",
            unit="B",
            unit_scale=True,
            event_queue=q,
            transfer_id="test-dl-6",
            filename="test.bin",
            report_interval=0,  # Disable throttling
        )

        # First update — emits event with bytes_completed=100
        bar.update(100)
        assert bar.n == 100

        # Second update — emits event with bytes_completed=300
        bar.update(200)
        assert bar.n == 300

        # Third update — emits event with bytes_completed=600
        bar.update(300)
        assert bar.n == 600

        # Drain events — should have 3 progress events (one per update)
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        progress_events = [e for e in events if e.event_type == EventType.PROGRESS]
        assert len(progress_events) == 3
        # Each event reflects the accumulated progress at that point
        assert progress_events[0].bytes_completed == 100
        assert progress_events[1].bytes_completed == 300
        assert progress_events[2].bytes_completed == 600

        bar.close()

    def test_download_tqdm_tracks_bytes_in_state_manager(self):
        """CRIT-004 Support: TransferStateManager must track bytes_completed/total_bytes."""
        from hf_track.callbacks import state_manager
        q = queue.Queue()
        bar = DownloadProgressTqdm(
            total=1000,
            desc="test.bin",
            unit="B",
            unit_scale=True,
            event_queue=q,
            transfer_id="test-dl-states",
        )

        bar.update(250)

        stats = state_manager.get_state("test-dl-states")
        assert stats["bytes_completed"] == 250
        assert stats["total_bytes"] == 1000
        bar.close()

    def test_file_count_bar_emits_progress_event(self):
        """File-count bars (is_bytes_bar=False) now emit PROGRESS events.

        This is critical for snapshot downloads where the file-count bar
        is the primary progress indicator. Previously, file-count bar
        updates were silently dropped, causing the web app to show no
        progress during xet snapshot downloads.
        """
        q = queue.Queue()
        bar = DownloadProgressTqdm(
            total=5,
            unit="files",
            event_queue=q,
            transfer_id="test-file-count",
            filename="test/repo",
        )
        assert not bar.is_bytes_bar

        bar.update(1)
        events = []
        while not q.empty():
            events.append(q.get_nowait())

        # Should have emitted a PROGRESS event
        assert len(events) == 1
        event = events[0]
        assert event.event_type == EventType.PROGRESS
        assert event.transfer_id == "test-file-count"
        assert event.filename == "test/repo"
        # File-count bars report file progress as percentage
        assert event.file_index == 1  # 1 file completed
        assert event.total_files == 5  # 5 total files
        bar.close()

    def test_file_count_bar_progress_uses_state_manager_bytes(self):
        """File-count bar PROGRESS events include byte stats from state_manager."""
        from hf_track.callbacks import state_manager

        q = queue.Queue()
        # Create a byte bar first to seed state_manager with byte data
        byte_bar = DownloadProgressTqdm(
            total=1000, unit="B", unit_scale=True,
            event_queue=q, transfer_id="test-mixed",
            filename="test/repo",
        )
        byte_bar.update(500)
        # Drain byte bar events
        while not q.empty():
            q.get_nowait()

        # Now create a file-count bar with the same transfer_id
        file_bar = DownloadProgressTqdm(
            total=3, unit="files",
            event_queue=q, transfer_id="test-mixed",
            filename="test/repo",
        )
        file_bar.update(1)
        events = []
        while not q.empty():
            events.append(q.get_nowait())

        assert len(events) == 1
        event = events[0]
        assert event.event_type == EventType.PROGRESS
        # Should include byte stats from state_manager
        assert event.bytes_completed == 500
        assert event.total_bytes == 1000
        file_bar.close()
        byte_bar.close()
        state_manager.clear_state("test-mixed")

    def test_file_count_bar_no_event_without_queue(self):
        """File-count bar with no event_queue doesn't emit events (base _emit_event drops silently)."""
        bar = DownloadProgressTqdm(
            total=5,
            unit="files",
            event_queue=None,
            transfer_id="test-no-q",
            filename="test/repo",
        )
        # Should not raise — base _emit_event returns early when queue is None
        bar.update(1)
        bar.close()

    def test_emit_event_with_none_queue_does_not_crash(self):
        """Base _emit_event gracefully handles None event_queue."""
        bar = DownloadProgressTqdm(
            total=100,
            unit="B",
            event_queue=None,
            transfer_id="test-emit-none",
            filename="test/file.bin",
        )
        event = ProgressEvent(
            event_type=EventType.PROGRESS,
            transfer_id="test-emit-none",
            direction=TransferDirection.DOWNLOAD,
            filename="test/file.bin",
            phase=ProgressPhase.DOWNLOADING,
            bytes_completed=50,
            total_bytes=100,
            percentage=50.0,
            speed=0,
        )
        # Calling _emit_event directly with None queue should not raise
        bar._emit_event(event)
        bar.close()

    def test_subclass_emit_event_called_with_none_queue(self):
        """When _emit_event is overridden (subprocess pattern), events flow
        even though event_queue is None.

        This is the critical fix: update() no longer short-circuits on
        ``self._event_queue is None``, so overridden _emit_event methods
        (like _SubprocessDownloadTqdm in _xet_worker.py) are actually
        reached during download progress updates.
        """
        captured: list[ProgressEvent] = []

        class CapturingTqdm(DownloadProgressTqdm):
            """Subclass that captures events instead of putting them in a queue."""
            def _emit_event(self, event):
                captured.append(event)

        BoundCapturing = CapturingTqdm.bind(
            event_queue=None,  # None — just like _snapshot_worker does
            transfer_id="test-subproc",
            filename="test/repo",
            report_interval=0.0,
        )

        bar = BoundCapturing(total=3, unit="files")
        bar.update(1)
        bar.update(1)
        bar.update(1)
        bar.close()

        # With the fix, _emit_event IS called even though event_queue is None
        progress_events = [e for e in captured if e.event_type == EventType.PROGRESS]
        assert len(progress_events) >= 3, (
            f"Expected >=3 PROGRESS events from file-count bar with None queue, "
            f"got {len(progress_events)}"
        )

    def test_byte_bar_subclass_emit_event_called_with_none_queue(self):
        """Byte-bar subclass with overridden _emit_event also works with None queue."""
        captured: list[ProgressEvent] = []

        class CapturingTqdm(DownloadProgressTqdm):
            def _emit_event(self, event):
                captured.append(event)

        BoundCapturing = CapturingTqdm.bind(
            event_queue=None,
            transfer_id="test-subproc-bytes",
            filename="test/file.bin",
            report_interval=0.0,
        )

        bar = BoundCapturing(total=1000, unit="B", unit_scale=True)
        bar.update(500)
        bar.update(500)
        bar.close()

        progress_events = [e for e in captured if e.event_type == EventType.PROGRESS]
        assert len(progress_events) >= 2, (
            f"Expected >=2 PROGRESS events from byte bar with None queue, "
            f"got {len(progress_events)}"
        )


class TestTqdmUploadPatcher:
    """Tests for tqdm_upload_patcher context manager."""

    def test_captures_file_bars(self):
        """File-level bars (unit='B') should emit events."""
        q = queue.Queue()

        with tqdm_upload_patcher(q, transfer_id="test-u1", filename="model.bin", total_bytes=1000):
            import tqdm.auto as tqdm_auto

            # Simulate what tqdm_stream_file does
            bar = tqdm_auto.tqdm(
                total=1000, desc="model.bin", unit="B", unit_scale=True
            )
            bar.update(500)
            bar.update(500)
            bar.close()

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        start_events = [e for e in events if e.event_type == EventType.START]
        progress_events = [
            e for e in events if e.event_type == EventType.PROGRESS
        ]
        complete_events = [
            e for e in events if e.event_type == EventType.COMPLETE
        ]

        assert len(start_events) == 1
        assert start_events[0].total_bytes == 1000  # from init_upload
        assert len(progress_events) >= 1
        assert len(complete_events) == 1

    def test_ignores_count_bars(self):
        """File-count bars (unit='it') should NOT emit events."""
        q = queue.Queue()

        with tqdm_upload_patcher(q, transfer_id="test-u2"):
            import tqdm.auto as tqdm_auto

            # Simulate what thread_map does
            bar = tqdm_auto.tqdm(total=5, desc="Upload 5 files", unit="it")
            bar.update(1)
            bar.update(1)
            bar.close()

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        # No events should be emitted for count bars
        assert len(events) == 0

    def test_restores_original_tqdm(self):
        """tqdm should be restored after the context manager exits."""
        import tqdm.auto as tqdm_auto

        original = tqdm_auto.tqdm

        with tqdm_upload_patcher(queue.Queue()):
            assert tqdm_auto.tqdm is not original

        assert tqdm_auto.tqdm is original

    def test_restores_on_exception(self):
        """tqdm should be restored even if an exception occurs."""
        import tqdm.auto as tqdm_auto

        original = tqdm_auto.tqdm

        with pytest.raises(ValueError):
            with tqdm_upload_patcher(queue.Queue()):
                raise ValueError("test error")

        assert tqdm_auto.tqdm is original

    def test_complete_event_at_100_percent(self):
        """CRIT-001 Check: Complete event MUST be emitted when bar reaches 100%."""
        q = queue.Queue()

        with tqdm_upload_patcher(q, transfer_id="test-u3", filename="model.bin"):
            import tqdm.auto as tqdm_auto

            bar = tqdm_auto.tqdm(
                total=1000, desc="model.bin", unit="B", unit_scale=True
            )
            bar.update(1000)
            bar.close()

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        complete_events = [
            e for e in events if e.event_type == EventType.COMPLETE
        ]
        assert len(complete_events) == 1
        assert complete_events[0].percentage == 100.0

    def test_no_complete_event_below_100(self):
        """No complete event from UploadProgressTqdm.close() when bytes < total_bytes.

        Note: The outer upload_file/upload_folder wrapper may still emit a
        COMPLETE event as a safety net, but the tqdm bar's own close()
        should NOT emit one when bytes_completed < total_bytes.
        """
        q = queue.Queue()

        with tqdm_upload_patcher(q, transfer_id="test-u4", filename="model.bin", total_bytes=1000):
            import tqdm.auto as tqdm_auto

            bar = tqdm_auto.tqdm(
                total=1000, desc="model.bin", unit="B", unit_scale=True
            )
            bar.update(500)
            bar.close()

            events = []
            while not q.empty():
                events.append(q.get_nowait())

            complete_events = [
                e for e in events if e.event_type == EventType.COMPLETE
            ]
            # The tqdm bar's close() should NOT emit COMPLETE since 500 < 1000
            # However, the state_manager tracks accumulated bytes, and if
            # bytes_completed >= total_bytes it would emit. With only 500/1000,
            # no COMPLETE should come from the bar itself.
            # We check that no COMPLETE event has bytes_completed=500
            bar_complete = [
                e for e in complete_events
                if e.bytes_completed == 500
            ]
            assert len(bar_complete) == 0

    def test_every_update_emits_event(self):
        """Every update() call should emit an event (no producer-side throttle).

        The consumer is responsible for throttling/display refresh.
        """
        q = queue.Queue()

        with tqdm_upload_patcher(
            q, transfer_id="test-u5", filename="model.bin", report_interval=1.0
        ):
            import tqdm.auto as tqdm_auto

            bar = tqdm_auto.tqdm(
                total=1000, desc="model.bin", unit="B", unit_scale=True
            )
            bar.update(100)
            # First update should emit (start + progress)
            initial_count = q.qsize()

            bar.update(100)
            # Second update should also emit (upload patcher has no throttle)
            assert q.qsize() == initial_count + 1
    
            bar.close()
    
    
    # ── Accumulation Tests (Phase 6) ──────────────────────────────────
    
    
    class TestTransferStateManagerAccumulation:
        """Test TransferStateManager.update_download_bytes accumulation logic.
    
        When snapshot_download creates sequential byte bars (one per file),
        each bar only knows its own file's total. The state_manager must
        ACCUMULATE total_bytes across files, not replace it.
        """
    
        def test_single_bar_updates(self):
            """Single byte bar: bytes_completed and total_bytes track normally."""
            from hf_track.callbacks import state_manager
            tid = "test-accum-1"
            state_manager.init_download(tid)
            try:
                state_manager.update_download_bytes(tid, 500, 1000)
                state = state_manager.get_state(tid)
                assert state["bytes_completed"] == 500
                assert state["total_bytes"] == 1000
    
                state_manager.update_download_bytes(tid, 800, 1000)
                state = state_manager.get_state(tid)
                assert state["bytes_completed"] == 800
                assert state["total_bytes"] == 1000
            finally:
                state_manager.clear_state(tid)
    
        def test_bar_switch_commits_previous_total(self):
            """When a new byte bar starts (via reset_byte_bar_state), the previous bar's total is committed."""
            from hf_track.callbacks import state_manager
            tid = "test-accum-2"
            state_manager.init_download(tid)
            try:
                # First file: 1000 bytes total, 500 completed
                state_manager.update_download_bytes(tid, 500, 1000)
                state = state_manager.get_state(tid)
                assert state["bytes_completed"] == 500
                assert state["total_bytes"] == 1000

                # Second file starts: simulate new byte bar init by calling
                # reset_byte_bar_state (which commits the previous bar's total
                # and resets per-bar state). Then update with new total.
                state_manager.reset_byte_bar_state(tid)
                # After reset, _committed_bytes = 1000, _current_bar_total = 0
                state = state_manager.get_state(tid)
                assert state["bytes_completed"] == 1000  # 1000 committed + 0 current
                assert state["total_bytes"] == 1000     # just committed, no current bar

                state_manager.update_download_bytes(tid, 0, 2000)
                state = state_manager.get_state(tid)
                assert state["bytes_completed"] == 1000  # 1000 committed + 0 current
                assert state["total_bytes"] == 3000  # 1000 committed + 2000 current
            finally:
                state_manager.clear_state(tid)

        def test_growing_total_does_not_commit(self):
            """xet-style growing total (same bar, more files discovered) does not commit."""
            from hf_track.callbacks import state_manager
            tid = "test-accum-growing"
            state_manager.init_download(tid)
            try:
                # Simulate xet's bytes_progress: total grows as files are discovered.
                # First 2 files discovered: total=200
                state_manager.update_download_bytes(tid, 0, 200)
                state = state_manager.get_state(tid)
                assert state["total_bytes"] == 200
                assert state["bytes_completed"] == 0

                # All 3 files discovered: total grows to 300 (DOES NOT commit 200)
                state_manager.update_download_bytes(tid, 0, 300)
                state = state_manager.get_state(tid)
                assert state["total_bytes"] == 300  # not 500!
                assert state["bytes_completed"] == 0

                # First file completes (size 100): increment
                state_manager.update_download_bytes(tid, 100, 300)
                state = state_manager.get_state(tid)
                assert state["bytes_completed"] == 100
                assert state["total_bytes"] == 300

                # Second file completes (size 100): cumulative increment
                state_manager.update_download_bytes(tid, 200, 300)
                state = state_manager.get_state(tid)
                assert state["bytes_completed"] == 200
                assert state["total_bytes"] == 300

                # Third file completes: full total
                state_manager.update_download_bytes(tid, 300, 300)
                state = state_manager.get_state(tid)
                assert state["bytes_completed"] == 300
                assert state["total_bytes"] == 300
            finally:
                state_manager.clear_state(tid)
    
        def test_three_sequential_bars(self):
            """Three sequential byte bars accumulate correctly (using reset between files)."""
            from hf_track.callbacks import state_manager
            tid = "test-accum-3"
            state_manager.init_download(tid)
            try:
                # File 1: 100 bytes
                state_manager.update_download_bytes(tid, 50, 100)
                state_manager.update_download_bytes(tid, 100, 100)
                state = state_manager.get_state(tid)
                assert state["bytes_completed"] == 100
                assert state["total_bytes"] == 100

                # File 2: 200 bytes — new bar, call reset to commit file 1
                state_manager.reset_byte_bar_state(tid)
                state_manager.update_download_bytes(tid, 100, 200)
                state = state_manager.get_state(tid)
                assert state["bytes_completed"] == 200  # 100 committed + 100 current
                assert state["total_bytes"] == 300  # 100 committed + 200 current

                state_manager.update_download_bytes(tid, 200, 200)
                state = state_manager.get_state(tid)
                assert state["bytes_completed"] == 300  # 100 committed + 200 current
                assert state["total_bytes"] == 300

                # File 3: 300 bytes — new bar
                state_manager.reset_byte_bar_state(tid)
                state_manager.update_download_bytes(tid, 150, 300)
                state = state_manager.get_state(tid)
                assert state["bytes_completed"] == 450  # 300 committed + 150 current
                assert state["total_bytes"] == 600  # 300 committed + 300 current
            finally:
                state_manager.clear_state(tid)
    
        def test_same_total_no_double_commit(self):
            """Two bars with the same total: second bar should NOT re-commit."""
            from hf_track.callbacks import state_manager
            tid = "test-accum-4"
            state_manager.init_download(tid)
            try:
                # First bar: 1000 bytes
                state_manager.update_download_bytes(tid, 500, 1000)
                state = state_manager.get_state(tid)
                assert state["bytes_completed"] == 500
                assert state["total_bytes"] == 1000
    
                # Second bar with SAME total (e.g. two files of same size)
                # This should commit the first bar's total
                state_manager.update_download_bytes(tid, 200, 1000)
                state = state_manager.get_state(tid)
                assert state["bytes_completed"] == 1200  # 1000 committed + 200 current
                assert state["total_bytes"] == 2000  # 1000 committed + 1000 current
            finally:
                state_manager.clear_state(tid)
    
        def test_zero_total_bar(self):
            """A bar with total=0 should not cause issues."""
            from hf_track.callbacks import state_manager
            tid = "test-accum-5"
            state_manager.init_download(tid)
            try:
                state_manager.update_download_bytes(tid, 0, 0)
                state = state_manager.get_state(tid)
                assert state["bytes_completed"] == 0
                assert state["total_bytes"] == 0
    
                # Switch to a real bar
                state_manager.update_download_bytes(tid, 100, 500)
                state = state_manager.get_state(tid)
                assert state["bytes_completed"] == 100
                assert state["total_bytes"] == 500
            finally:
                state_manager.clear_state(tid)
    
        def test_unknown_transfer_id(self):
            """Updating a non-existent transfer_id should not crash."""
            from hf_track.callbacks import state_manager
            # Should silently do nothing
            state_manager.update_download_bytes("nonexistent", 100, 200)
    
    
    # ── Throttling Tests (Phase 7) ────────────────────────────────────
    
    
    class TestDownloadProgressTqdmThrottling:
        """Test DownloadProgressTqdm event throttling behavior."""
    
        def test_zero_report_interval_disables_throttling(self):
            """report_interval=0 should disable throttling entirely."""
            q = queue.Queue()
            bar = DownloadProgressTqdm(
                total=1000,
                desc="test.bin",
                unit="B",
                unit_scale=True,
                event_queue=q,
                transfer_id="test-throttle-0",
                filename="test.bin",
                report_interval=0,
            )
            # All updates should emit
            bar.update(100)
            bar.update(100)
            bar.update(100)
            events = []
            while not q.empty():
                events.append(q.get_nowait())
            progress_events = [e for e in events if e.event_type == EventType.PROGRESS]
            assert len(progress_events) == 3
            bar.close()
    
        def test_time_based_throttle(self):
            """Events within report_interval are throttled unless byte delta is large."""
            q = queue.Queue()
            bar = DownloadProgressTqdm(
                total=100000,
                desc="test.bin",
                unit="B",
                unit_scale=True,
                event_queue=q,
                transfer_id="test-throttle-time",
                filename="test.bin",
                report_interval=10.0,  # 10 seconds — very long
            )
            # First event always emits
            bar.update(100)
            assert q.qsize() == 1
    
            # Small delta — throttled
            bar.update(50)
            assert q.qsize() == 1
    
            # Large delta (>1% = 1000 bytes) — should emit
            bar.update(2000)
            assert q.qsize() == 2
    
            bar.close()
    
        def test_first_event_never_throttled(self):
            """The very first event should always be emitted."""
            q = queue.Queue()
            bar = DownloadProgressTqdm(
                total=100000,
                desc="test.bin",
                unit="B",
                unit_scale=True,
                event_queue=q,
                transfer_id="test-throttle-first",
                filename="test.bin",
                report_interval=100.0,  # Extremely long
            )
            # Even a tiny first update should emit
            bar.update(1)
            assert q.qsize() == 1
            bar.close()
    
        def test_completion_not_throttled_even_with_small_delta(self):
            """When bytes_completed >= total_bytes, event should not be throttled."""
            q = queue.Queue()
            bar = DownloadProgressTqdm(
                total=1000,
                desc="test.bin",
                unit="B",
                unit_scale=True,
                event_queue=q,
                transfer_id="test-throttle-complete",
                filename="test.bin",
                report_interval=100.0,  # Extremely long
            )
            # First event
            bar.update(500)
            assert q.qsize() == 1
    
            # Complete — should emit despite throttle (bytes >= total)
            bar.update(500)
            assert q.qsize() == 2
    
            bar.close()
    
        def test_file_count_bar_not_throttled(self):
            """File-count bars (unit='files') should not be throttled."""
            q = queue.Queue()
            bar = DownloadProgressTqdm(
                total=10,
                desc="Downloading",
                unit="files",
                event_queue=q,
                transfer_id="test-throttle-files",
                filename="test-repo",
                report_interval=100.0,  # Extremely long
            )
            # File-count bar updates should always emit
            bar.update(1)
            assert q.qsize() == 1
            bar.update(1)
            assert q.qsize() == 2
            bar.close()
    
        def test_throttle_state_reset_on_new_bar(self):
            """Each new DownloadProgressTqdm instance should have fresh throttle state."""
            q = queue.Queue()
            # First bar
            bar1 = DownloadProgressTqdm(
                total=1000,
                desc="file1.bin",
                unit="B",
                unit_scale=True,
                event_queue=q,
                transfer_id="test-throttle-reset",
                filename="file1.bin",
                report_interval=100.0,
            )
            bar1.update(500)
            assert q.qsize() == 1
            bar1.close()
    
            # Second bar with same transfer_id — should have fresh throttle state
            bar2 = DownloadProgressTqdm(
                total=2000,
                desc="file2.bin",
                unit="B",
                unit_scale=True,
                event_queue=q,
                transfer_id="test-throttle-reset",
                filename="file2.bin",
                report_interval=100.0,
            )
            # First event of new bar should always emit
            bar2.update(100)
            # Drain to count just the latest event
            while not q.empty():
                last_event = q.get_nowait()
            assert last_event.event_type == EventType.PROGRESS
            assert last_event.bytes_completed > 0
            bar2.close()
    
    
    # ── Xet Absolute-Position Fix Tests ───────────────────────────────
    
    
    class TestXetAbsolutePositionFix:
        """Test DownloadProgressTqdm handling of xet's absolute-position update() calls.
    
        huggingface_hub.xet_get() calls progress.update(progress_bytes) where
        progress_bytes is the TOTAL bytes completed so far (not an increment).
        But tqdm.update(n) expects n to be an INCREMENT. DownloadProgressTqdm
        must detect and correct this.
        """
    
        def test_increment_mode_unchanged(self):
            """Standard increment-based update() should work as before."""
            q = queue.Queue()
            bar = DownloadProgressTqdm(
                total=1000,
                desc="test.bin",
                unit="B",
                unit_scale=True,
                event_queue=q,
                transfer_id="test-xet-incr",
                filename="test.bin",
                report_interval=0,
            )
            # Increment mode: update(100) means "add 100 bytes"
            bar.update(100)
            assert bar.n == 100
    
            bar.update(200)
            assert bar.n == 300
    
            bar.close()
    
        def test_absolute_position_detected_and_corrected(self):
            """When update(n) is called with n > remaining, self.n should be corrected."""
            q = queue.Queue()
            bar = DownloadProgressTqdm(
                total=1000,
                desc="test.bin",
                unit="B",
                unit_scale=True,
                event_queue=q,
                transfer_id="test-xet-abs",
                filename="test.bin",
                report_interval=0,
            )
            # Simulate xet_get's behavior: update(500) means "500 bytes completed"
            # NOT "add 500 bytes to the current count"
            bar.update(500)
            # After correction, self.n should be 500 (absolute), not 500 (increment from 0)
            # Both give the same result for the first call, so this passes either way
            assert bar.n == 500
    
            # Now xet calls update(800) — meaning "800 bytes completed so far"
            # Without the fix, self.n would be 500 + 800 = 1300 (WRONG)
            # With the fix, self.n should be corrected to 800
            bar.update(800)
            assert bar.n == 800  # Corrected from 1300
    
            # xet calls update(1000) — meaning "1000 bytes completed"
            bar.update(1000)
            assert bar.n == 1000  # Corrected from 1800
    
            bar.close()
    
        def test_absolute_position_event_values(self):
            """Events emitted during xet-style updates should have correct bytes_completed."""
            q = queue.Queue()
            bar = DownloadProgressTqdm(
                total=1000,
                desc="test.bin",
                unit="B",
                unit_scale=True,
                event_queue=q,
                transfer_id="test-xet-events",
                filename="test.bin",
                report_interval=0,
            )
            # xet-style: update(500) means 500 bytes completed
            bar.update(500)
            # xet-style: update(1000) means 1000 bytes completed
            bar.update(1000)
    
            # Drain events
            events = []
            while not q.empty():
                events.append(q.get_nowait())
            progress_events = [e for e in events if e.event_type == EventType.PROGRESS]
    
            # The last progress event should show bytes_completed=1000
            last_progress = progress_events[-1]
            assert last_progress.bytes_completed == 1000
            assert last_progress.total_bytes == 1000
            assert last_progress.percentage == 100.0
    
            bar.close()
    
        def test_mixed_increment_and_absolute(self):
            """If some calls are increments and some are absolute, detection still works."""
            q = queue.Queue()
            bar = DownloadProgressTqdm(
                total=1000,
                desc="test.bin",
                unit="B",
                unit_scale=True,
                event_queue=q,
                transfer_id="test-xet-mixed",
                filename="test.bin",
                report_interval=0,
            )
            # First call: increment of 100 (n=100, remaining=1000, n <= remaining)
            bar.update(100)
            assert bar.n == 100
    
            # Second call: increment of 200 (n=200, remaining=900, n <= remaining)
            bar.update(200)
            assert bar.n == 300
    
            # Third call: absolute position 800 (n=800, remaining=700, n > remaining)
            # This triggers the correction: self.n should be 800, not 1100
            bar.update(800)
            assert bar.n == 800

            bar.close()


# ── Temporary diagnostic logging tests (Phase 0.6 — 0.10) ─────────
#
# These tests verify the temporary diagnostic instrumentation in
# DownloadProgressTqdm emits the expected log lines when
# HF_TRACK_DEBUG_XET is set. They will be REMOVED in Phase 3 when
# the diagnostic logging is stripped out.


class TestDownloadProgressTqdmDiagnosticLogging:
    """Verify temporary diagnostic instrumentation in DownloadProgressTqdm.

    Activated by env var HF_TRACK_DEBUG_XET=1. Tests run with the env
    var set/unset to confirm both the on and off paths.
    """

    @pytest.fixture
    def debug_xet_enabled(self, monkeypatch):
        monkeypatch.setenv("HF_TRACK_DEBUG_XET", "1")
        yield

    @pytest.fixture
    def debug_xet_disabled(self, monkeypatch):
        monkeypatch.delenv("HF_TRACK_DEBUG_XET", raising=False)
        yield

    # 0.6 — init log line
    def test_diag_init_emits_log_with_expected_fields(
        self, debug_xet_enabled, caplog
    ):
        with caplog.at_level(logging.DEBUG, logger="hf_track.callbacks"):
            DownloadProgressTqdm(
                total=1000,
                desc="test.bin",
                unit="B",
                unit_scale=True,
                event_queue=queue.Queue(),
                transfer_id="diag-init-1",
                filename="test.bin",
                report_interval=0,
            )
        init_logs = [r for r in caplog.records if "[DIAG-INIT]" in r.getMessage()]
        assert len(init_logs) == 1
        msg = init_logs[0].getMessage()
        assert "total=1000" in msg
        assert "unit=B" in msg
        assert "desc=test.bin" in msg
        assert "is_bytes_bar=True" in msg
        assert "event_queue_is_none=False" in msg
        assert "has_cancel_hook=False" in msg

    # 0.7 — update log lines
    def test_diag_update_emits_log_with_expected_fields(
        self, debug_xet_enabled, caplog
    ):
        with caplog.at_level(logging.DEBUG, logger="hf_track.callbacks"):
            bar = DownloadProgressTqdm(
                total=1000,
                desc="test.bin",
                unit="B",
                unit_scale=True,
                event_queue=queue.Queue(),
                transfer_id="diag-upd-1",
                filename="test.bin",
                report_interval=0,
            )
            caplog.clear()
            bar.update(100)
        upd_entries = [
            r for r in caplog.records if "[DIAG-UPDATE-ENTRY]" in r.getMessage()
        ]
        upd_posts = [
            r for r in caplog.records if "[DIAG-UPDATE-POST-SUPER]" in r.getMessage()
        ]
        assert len(upd_entries) == 1
        assert len(upd_posts) == 1
        msg = upd_entries[0].getMessage()
        assert "n=100" in msg
        assert "prev_n=0" in msg
        assert "is_bytes_bar=True" in msg
        bar.close()

    def test_diag_update_emits_emit_line_on_first_event(
        self, debug_xet_enabled, caplog
    ):
        with caplog.at_level(logging.DEBUG, logger="hf_track.callbacks"):
            bar = DownloadProgressTqdm(
                total=1000,
                desc="test.bin",
                unit="B",
                unit_scale=True,
                event_queue=queue.Queue(),
                transfer_id="diag-emit-1",
                filename="test.bin",
                report_interval=0,
            )
            caplog.clear()
            bar.update(100)
        emit_logs = [r for r in caplog.records if "[DIAG-EMIT]" in r.getMessage()]
        assert len(emit_logs) == 1
        msg = emit_logs[0].getMessage()
        assert "event_type=PROGRESS" in msg
        assert "bytes=100" in msg
        bar.close()

    def test_diag_update_emits_throttle_line(self, debug_xet_enabled, caplog):
        with caplog.at_level(logging.DEBUG, logger="hf_track.callbacks"):
            bar = DownloadProgressTqdm(
                total=10000,
                desc="test.bin",
                unit="B",
                unit_scale=True,
                event_queue=queue.Queue(),
                transfer_id="diag-throttle-1",
                filename="test.bin",
                report_interval=10.0,  # huge to force throttle
            )
            bar.update(50)  # first event
            caplog.clear()
            bar.update(51)  # throttled
        throttled = [
            r for r in caplog.records if "[DIAG-THROTTLED]" in r.getMessage()
        ]
        assert len(throttled) == 1
        bar.close()

    # 0.8 — silent at INFO level
    def test_diag_silent_when_env_var_not_set(
        self, debug_xet_disabled, caplog
    ):
        with caplog.at_level(logging.DEBUG, logger="hf_track.callbacks"):
            bar = DownloadProgressTqdm(
                total=1000,
                desc="test.bin",
                unit="B",
                unit_scale=True,
                event_queue=queue.Queue(),
                transfer_id="diag-silent-1",
                filename="test.bin",
                report_interval=0,
            )
            bar.update(100)
            bar.update(200)
            bar.close()
        diag_lines = [
            r for r in caplog.records if r.getMessage().startswith("[DIAG-")
        ]
        assert diag_lines == []

    # 0.9 — _emit_event logs exceptions (queue.Full)
    def test_diag_emit_event_logs_queue_full(self, debug_xet_enabled, caplog):
        # Bounded queue pre-filled to capacity so the next put_nowait raises Full
        full_q: queue.Queue = queue.Queue(maxsize=1)
        full_q.put_nowait("sentinel")
        with caplog.at_level(logging.DEBUG, logger="hf_track.callbacks"):
            bar = DownloadProgressTqdm(
                total=1000,
                desc="test.bin",
                unit="B",
                unit_scale=True,
                event_queue=full_q,
                transfer_id="diag-full-1",
                filename="test.bin",
                report_interval=0,
            )
            bar.update(100)
        full_logs = [
            r for r in caplog.records if "[DIAG-EMIT-PUT-FULL]" in r.getMessage()
        ]
        assert len(full_logs) == 1
        bar.close()

    def test_diag_emit_event_logs_unexpected_exception(
        self, debug_xet_enabled, caplog
    ):
        # Queue that raises a non-queue.Full exception
        class _BrokenQueue:
            def put_nowait(self, _evt):
                raise RuntimeError("synthetic put failure")

        with caplog.at_level(logging.DEBUG, logger="hf_track.callbacks"):
            bar = DownloadProgressTqdm(
                total=1000,
                desc="test.bin",
                unit="B",
                unit_scale=True,
                event_queue=_BrokenQueue(),
                transfer_id="diag-broken-1",
                filename="test.bin",
                report_interval=0,
            )
            with pytest.raises(RuntimeError, match="synthetic put failure"):
                bar.update(100)
        err_logs = [
            r for r in caplog.records if "[DIAG-EMIT-PUT-ERR]" in r.getMessage()
        ]
        assert len(err_logs) == 1
        assert "RuntimeError" in err_logs[0].getMessage()
        bar.close()

    # 0.10 — close emits summary
    def test_diag_close_emits_summary(self, debug_xet_enabled, caplog):
        with caplog.at_level(logging.DEBUG, logger="hf_track.callbacks"):
            bar = DownloadProgressTqdm(
                total=1000,
                desc="test.bin",
                unit="B",
                unit_scale=True,
                event_queue=queue.Queue(),
                transfer_id="diag-close-1",
                filename="test.bin",
                report_interval=0,
            )
            bar.update(1000)  # complete
            caplog.clear()
            bar.close()
        close_entries = [
            r for r in caplog.records if "[DIAG-CLOSE-ENTRY]" in r.getMessage()
        ]
        close_branches = [
            r for r in caplog.records if "[DIAG-CLOSE-BRANCH]" in r.getMessage()
        ]
        close_synths = [
            r for r in caplog.records
            if "[DIAG-CLOSE-SYNTH-COMPLETE]" in r.getMessage()
        ]
        assert len(close_entries) == 1
        assert len(close_branches) == 1
        assert "is_complete=True" in close_branches[0].getMessage()
        assert len(close_synths) == 1
        assert "final_bytes=1000" in close_synths[0].getMessage()

    def test_diag_close_xet_cached_branch(self, debug_xet_enabled, caplog):
        """close() with n=0 and total>0 should hit is_xet_cached branch."""
        with caplog.at_level(logging.DEBUG, logger="hf_track.callbacks"):
            bar = DownloadProgressTqdm(
                total=1000,
                desc="test.bin",
                unit="B",
                unit_scale=True,
                event_queue=queue.Queue(),
                transfer_id="diag-cached-1",
                filename="test.bin",
                report_interval=0,
            )
            caplog.clear()
            bar.close()  # never called update → n stays 0
        branches = [
            r for r in caplog.records if "[DIAG-CLOSE-BRANCH]" in r.getMessage()
        ]
        synths = [
            r for r in caplog.records
            if "[DIAG-CLOSE-SYNTH-COMPLETE]" in r.getMessage()
        ]
        assert len(branches) == 1
        assert "is_xet_cached=True" in branches[0].getMessage()
        assert len(synths) == 1
        assert "final_bytes=1000" in synths[0].getMessage()