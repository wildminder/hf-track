"""Tests for hf_progress.callbacks module."""

from __future__ import annotations

import queue
import time
import types
from unittest.mock import MagicMock

import pytest

from hf_progress.callbacks import (
    DownloadProgressTqdm,
    XetDownloadProgressCallback,
    XetUploadProgressCallback,
    tqdm_upload_patcher,
)
from hf_progress.types import EventType, ProgressPhase, TransferDirection


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
            report_interval=0,  # No throttling
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
        assert event.speed == 100.0
        assert event.transfer_bytes_completed == 400
        assert event.transfer_bytes_total == 800
        assert event.transfer_speed == 80.0
        assert event.dedup_saved_bytes == 100  # 500 - 400

    def test_throttling(self):
        """Rapid calls should be throttled to report_interval."""
        q = queue.Queue()
        callback = XetUploadProgressCallback(
            filename="test.bin",
            total_bytes=1000,
            event_queue=q,
            report_interval=1.0,  # 1 second throttle
        )

        total_update = self._make_mock_total_update()

        # First call should emit
        callback(total_update, [])
        assert q.qsize() == 1

        # Immediate second call should be throttled
        callback(total_update, [])
        assert q.qsize() == 1  # Still 1

    def test_throttle_allows_after_interval(self):
        """After the report_interval, calls should emit again."""
        q = queue.Queue()
        callback = XetUploadProgressCallback(
            filename="test.bin",
            total_bytes=1000,
            event_queue=q,
            report_interval=0.05,  # 50ms throttle
        )

        total_update = self._make_mock_total_update()

        callback(total_update, [])
        assert q.qsize() == 1

        time.sleep(0.1)  # Wait for throttle to expire

        callback(total_update, [])
        assert q.qsize() == 2

    def test_dedup_saved_bytes(self):
        """Dedup saved bytes = bytes_completed - transfer_bytes_completed."""
        q = queue.Queue()
        callback = XetUploadProgressCallback(
            filename="test.bin",
            total_bytes=1000,
            event_queue=q,
            report_interval=0,
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
            report_interval=0,
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
            report_interval=0,
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
            report_interval=0,
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
            report_interval=0,
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
            report_interval=0,
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
            report_interval=0,  # No throttling
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
        assert event.bytes_completed == 500
        assert event.total_bytes == 1000
        assert event.percentage == 50.0
        assert event.speed == 1000.0
        assert event.transfer_bytes_completed == 400
        assert event.transfer_bytes_total == 800
        assert event.transfer_speed == 800.0
        assert event.dedup_saved_bytes == 100  # 500 - 400

    def test_computes_percentage(self):
        """Percentage should be computed from bytes_completed/total_bytes."""
        q = queue.Queue()
        callback = XetDownloadProgressCallback(
            filename="test.bin",
            total_bytes=1000,
            event_queue=q,
            report_interval=0,
        )

        total_update = self._make_total_update(
            total_bytes=1000,
            total_bytes_completed=250,
        )
        callback(total_update, [])
        event = q.get_nowait()
        assert event.percentage == 25.0

    def test_throttling(self):
        """Rapid calls should be throttled."""
        q = queue.Queue()
        callback = XetDownloadProgressCallback(
            filename="test.bin",
            total_bytes=1000,
            event_queue=q,
            report_interval=1.0,  # 1 second throttle
        )

        total_update = self._make_total_update(total_bytes_completed=100)
        callback(total_update, [])
        assert q.qsize() == 1

        # Immediate second call should be throttled
        total_update2 = self._make_total_update(total_bytes_completed=200)
        callback(total_update2, [])
        assert q.qsize() == 1  # Still 1

    def test_zero_total_bytes(self):
        """Percentage should be 0 when total_bytes is 0."""
        q = queue.Queue()
        callback = XetDownloadProgressCallback(
            filename="test.bin",
            total_bytes=0,
            event_queue=q,
            report_interval=0,
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
            report_interval=0,
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

    def test_throttling(self):
        """Throttling should prevent excessive events."""
        q = queue.Queue()
        bar = DownloadProgressTqdm(
            total=1000,
            desc="test.bin",
            unit="B",
            unit_scale=True,
            event_queue=q,
            transfer_id="test-dl-2",
            filename="test.bin",
            report_interval=1.0,  # 1 second throttle
        )

        # First update should emit
        bar.update(100)
        assert q.qsize() == 1

        # Immediate second update should be throttled
        bar.update(100)
        assert q.qsize() == 1  # Still 1

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

    def test_throttle_only_affects_event_emission_not_counter(self):
        """Throttle should only limit event emission, not tqdm's internal counter.

        When update() is called during a throttle period, self.n should still
        be updated (via super().update(n)) even though no event is emitted.
        The next emitted event should reflect the accumulated progress.
        This prevents the "frozen then jump" progress bar behavior.
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
            report_interval=1.0,  # 1 second throttle — long enough to block
        )

        # First update — should emit event (first call always passes)
        bar.update(100)
        assert bar.n == 100  # Counter updated

        # Second update during throttle — event suppressed but counter updated
        bar.update(200)
        assert bar.n == 300  # Counter still updated despite throttle!

        # Third update during throttle — event suppressed but counter updated
        bar.update(300)
        assert bar.n == 600  # Counter still updated despite throttle!

        # Drain events — should have only 1 progress event (from first update)
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        progress_events = [e for e in events if e.event_type == EventType.PROGRESS]
        assert len(progress_events) == 1
        # The first event shows 100 bytes (from the first update)
        assert progress_events[0].bytes_completed == 100

        # Now force the throttle to expire
        bar._last_report_time = 0.0

        # Next update should emit event with ACCUMULATED progress
        bar.update(100)
        assert bar.n == 700

        # Drain new events
        new_events = []
        while not q.empty():
            new_events.append(q.get_nowait())
        new_progress = [e for e in new_events if e.event_type == EventType.PROGRESS]
        assert len(new_progress) == 1
        # The event shows 700 bytes — the full accumulated progress
        assert new_progress[0].bytes_completed == 700

        bar.close()


class TestTqdmUploadPatcher:
    """Tests for tqdm_upload_patcher context manager."""

    def test_captures_file_bars(self):
        """File-level bars (unit='B') should emit events."""
        q = queue.Queue()

        with tqdm_upload_patcher(q, transfer_id="test-u1", filename="model.bin"):
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
        assert start_events[0].total_bytes == 1000
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
        """Complete event should be emitted when bar reaches 100%."""
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
        """No complete event when bar closes below 100%."""
        q = queue.Queue()

        with tqdm_upload_patcher(q, transfer_id="test-u4", filename="model.bin"):
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
        assert len(complete_events) == 0

    def test_throttling(self):
        """Rapid updates should be throttled."""
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
            # Second update should be throttled
            assert q.qsize() == initial_count

            bar.close()
