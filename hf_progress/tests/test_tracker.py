"""Tests for hf_progress.tracker module."""

from __future__ import annotations

import queue
import threading
import time

import pytest

from hf_progress.tracker import HfProgressTracker
from hf_progress.types import EventType, ProgressEvent, TransferDirection


class TestHfProgressTrackerInit:
    """Tests for HfProgressTracker initialization."""

    def test_creates_with_token(self):
        tracker = HfProgressTracker(token="hf_test123")
        assert tracker._token == "hf_test123"
        assert tracker.event_queue is not None
        assert tracker._report_interval == 0.1

    def test_custom_report_interval(self):
        tracker = HfProgressTracker(token="hf_test", report_interval=0.5)
        assert tracker._report_interval == 0.5

    def test_custom_endpoint(self):
        tracker = HfProgressTracker(token="hf_test", endpoint="https://custom.api")
        assert tracker._endpoint == "https://custom.api"
        
    def test_bounded_queue(self):
        """IMP-012: Ensure queue is bounded to prevent OOM."""
        tracker = HfProgressTracker()
        assert getattr(tracker.event_queue, "maxsize", 0) > 0


class TestHfProgressTrackerCancellation:
    """Tests for cancellation logic."""
    
    def test_cancel_flags_transfer(self):
        tracker = HfProgressTracker()
        transfer_id = "test-cancel-1"
        
        assert not tracker.is_cancelled(transfer_id)
        tracker.cancel(transfer_id)
        assert tracker.is_cancelled(transfer_id)


class TestHfProgressTrackerEvents:
    """Tests for HfProgressTracker event consumption methods."""

    def test_get_events_empty(self):
        """get_events should return empty list when queue is empty."""
        tracker = HfProgressTracker(token="hf_test")
        events = tracker.get_events()
        assert events == []

    def test_get_events_returns_all(self):
        """get_events should return all queued events."""
        tracker = HfProgressTracker(token="hf_test")

        # Put some events directly
        for i in range(5):
            tracker.event_queue.put(
                ProgressEvent(
                    event_type=EventType.PROGRESS,
                    transfer_id=f"test-{i}",
                    direction=TransferDirection.UPLOAD,
                    filename="test.bin",
                    phase="uploading",
                    bytes_completed=i * 100,
                    total_bytes=500,
                    percentage=i * 20.0,
                )
            )

        events = tracker.get_events()
        assert len(events) == 5
        assert events[0].transfer_id == "test-0"
        assert events[4].transfer_id == "test-4"

    def test_get_events_non_blocking(self):
        """get_events with timeout=0 should be non-blocking."""
        tracker = HfProgressTracker(token="hf_test")
        events = tracker.get_events(timeout=0)
        assert events == []

    def test_events_generator(self):
        """events() should yield ProgressEvent objects."""
        tracker = HfProgressTracker(token="hf_test")

        # Put events
        tracker.event_queue.put(
            ProgressEvent(
                event_type=EventType.START,
                transfer_id="gen-1",
                direction=TransferDirection.UPLOAD,
                filename="test.bin",
                phase="uploading",
            )
        )
        tracker.event_queue.put(
            ProgressEvent(
                event_type=EventType.COMPLETE,
                transfer_id="gen-1",
                direction=TransferDirection.UPLOAD,
                filename="test.bin",
                phase="complete",
                percentage=100.0,
            )
        )

        # Collect events using generator with stop_on
        events = list(tracker.events(timeout=0.5, stop_on=EventType.COMPLETE))
        assert len(events) == 2
        assert events[0].event_type == EventType.START
        assert events[1].event_type == EventType.COMPLETE

    def test_wait_for_complete_success(self):
        """wait_for_complete should return the COMPLETE event."""
        tracker = HfProgressTracker(token="hf_test")

        def put_events():
            time.sleep(0.1)
            tracker.event_queue.put(
                ProgressEvent(
                    event_type=EventType.PROGRESS,
                    transfer_id="wait-1",
                    direction=TransferDirection.UPLOAD,
                    filename="test.bin",
                    phase="uploading",
                )
            )
            time.sleep(0.1)
            tracker.event_queue.put(
                ProgressEvent(
                    event_type=EventType.COMPLETE,
                    transfer_id="wait-1",
                    direction=TransferDirection.UPLOAD,
                    filename="test.bin",
                    phase="complete",
                    percentage=100.0,
                )
            )

        thread = threading.Thread(target=put_events, daemon=True)
        thread.start()

        result = tracker.wait_for_complete("wait-1", timeout=5)
        assert result is not None
        assert result.event_type == EventType.COMPLETE
        assert result.transfer_id == "wait-1"

    def test_wait_for_complete_timeout(self):
        """wait_for_complete should return None on timeout."""
        tracker = HfProgressTracker(token="hf_test")

        result = tracker.wait_for_complete("nonexistent", timeout=0.2)
        assert result is None

    def test_wait_for_complete_error(self):
        """wait_for_complete should return ERROR events too."""
        tracker = HfProgressTracker(token="hf_test")

        tracker.event_queue.put(
            ProgressEvent(
                event_type=EventType.ERROR,
                transfer_id="err-1",
                direction=TransferDirection.UPLOAD,
                filename="test.bin",
                phase="error",
                error="Connection refused",
            )
        )

        result = tracker.wait_for_complete("err-1", timeout=1)
        assert result is not None
        assert result.event_type == EventType.ERROR
        assert result.error == "Connection refused"

    def test_wait_for_complete_ignores_other_transfers(self):
        """wait_for_complete should only match the specified transfer_id."""
        tracker = HfProgressTracker(token="hf_test")

        # Put events for a different transfer
        tracker.event_queue.put(
            ProgressEvent(
                event_type=EventType.COMPLETE,
                transfer_id="other-transfer",
                direction=TransferDirection.UPLOAD,
                filename="other.bin",
                phase="complete",
                percentage=100.0,
            )
        )

        # Put event for our transfer
        tracker.event_queue.put(
            ProgressEvent(
                event_type=EventType.COMPLETE,
                transfer_id="my-transfer",
                direction=TransferDirection.UPLOAD,
                filename="my.bin",
                phase="complete",
                percentage=100.0,
            )
        )

        result = tracker.wait_for_complete("my-transfer", timeout=1)
        assert result is not None
        assert result.transfer_id == "my-transfer"


class TestHfProgressTrackerUploadRouting:
    """Tests for upload method routing (Xet vs LFS)."""

    def test_upload_file_calls_xet_when_available(self):
        """When hf_xet is available, upload_file should route to Xet."""
        tracker = HfProgressTracker(token="hf_test")

        from unittest.mock import patch

        with patch("hf_progress.tracker.is_xet_available", return_value=True):
            with patch.object(
                tracker, "_upload_file_xet", return_value="abc123"
            ) as mock:
                result = tracker.upload_file(
                    file_path="/tmp/test.bin",
                    repo_id="user/repo",
                    transfer_id="test-1",
                )
                mock.assert_called_once()

    def test_upload_file_calls_lfs_when_xet_unavailable(self):
        """When hf_xet is not available, upload_file should route to LFS."""
        tracker = HfProgressTracker(token="hf_test")

        from unittest.mock import patch

        with patch("hf_progress.tracker.is_xet_available", return_value=False):
            with patch.object(
                tracker, "_upload_file_lfs", return_value="https://..."
            ) as mock:
                result = tracker.upload_file(
                    file_path="/tmp/test.bin",
                    repo_id="user/repo",
                    transfer_id="test-1",
                )
                mock.assert_called_once()

    def test_upload_bytes_calls_xet_when_available(self):
        """When hf_xet is available, upload_bytes should route to Xet."""
        tracker = HfProgressTracker(token="hf_test")

        from unittest.mock import patch

        with patch("hf_progress.tracker.is_xet_available", return_value=True):
            with patch.object(
                tracker, "_upload_bytes_xet", return_value="abc123"
            ) as mock:
                result = tracker.upload_bytes(
                    file_content=b"test data",
                    filename="test.bin",
                    repo_id="user/repo",
                    transfer_id="test-1",
                )
                mock.assert_called_once()

    def test_upload_bytes_calls_temp_when_xet_unavailable(self):
        """When hf_xet is not available, upload_bytes should use temp file."""
        tracker = HfProgressTracker(token="hf_test")

        from unittest.mock import patch

        with patch("hf_progress.tracker.is_xet_available", return_value=False):
            with patch.object(
                tracker, "_upload_bytes_via_temp", return_value="https://..."
            ) as mock:
                result = tracker.upload_bytes(
                    file_content=b"test data",
                    filename="test.bin",
                    repo_id="user/repo",
                    transfer_id="test-1",
                )
                mock.assert_called_once()


class TestHfProgressTrackerEventFlow:
    """Integration tests for event flow through the tracker."""

    def test_manual_event_flow(self):
        """Manually put events and verify consumption."""
        tracker = HfProgressTracker(token="hf_test")

        # Simulate a complete transfer lifecycle
        tracker.event_queue.put(
            ProgressEvent(
                event_type=EventType.START,
                transfer_id="flow-1",
                direction=TransferDirection.UPLOAD,
                filename="model.bin",
                phase="uploading",
                total_bytes=1000,
            )
        )

        for i in range(1, 11):
            tracker.event_queue.put(
                ProgressEvent(
                    event_type=EventType.PROGRESS,
                    transfer_id="flow-1",
                    direction=TransferDirection.UPLOAD,
                    filename="model.bin",
                    phase="uploading",
                    bytes_completed=i * 100,
                    total_bytes=1000,
                    percentage=i * 10.0,
                )
            )

        tracker.event_queue.put(
            ProgressEvent(
                event_type=EventType.COMPLETE,
                transfer_id="flow-1",
                direction=TransferDirection.UPLOAD,
                filename="model.bin",
                phase="complete",
                bytes_completed=1000,
                total_bytes=1000,
                percentage=100.0,
            )
        )

        # Consume all events
        events = tracker.get_events()
        assert len(events) == 12  # 1 start + 10 progress + 1 complete

        # Verify event types
        start_events = [e for e in events if e.event_type == EventType.START]
        progress_events = [
            e for e in events if e.event_type == EventType.PROGRESS
        ]
        complete_events = [
            e for e in events if e.event_type == EventType.COMPLETE
        ]

        assert len(start_events) == 1
        assert len(progress_events) == 10
        assert len(complete_events) == 1

        # Verify progress is monotonically increasing
        percentages = [e.percentage for e in progress_events]
        assert percentages == sorted(percentages)

    def test_concurrent_event_consumption(self):
        """Events from background threads should be consumable from main thread."""
        tracker = HfProgressTracker(token="hf_test")

        def producer():
            for i in range(10):
                tracker.event_queue.put(
                    ProgressEvent(
                        event_type=EventType.PROGRESS,
                        transfer_id="concurrent-1",
                        direction=TransferDirection.UPLOAD,
                        filename="test.bin",
                        phase="uploading",
                        bytes_completed=(i + 1) * 100,
                        total_bytes=1000,
                        percentage=(i + 1) * 10.0,
                    )
                )
                time.sleep(0.01)

            tracker.event_queue.put(
                ProgressEvent(
                    event_type=EventType.COMPLETE,
                    transfer_id="concurrent-1",
                    direction=TransferDirection.UPLOAD,
                    filename="test.bin",
                    phase="complete",
                    percentage=100.0,
                )
            )

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        # Wait for completion
        result = tracker.wait_for_complete("concurrent-1", timeout=5)
        assert result is not None
        assert result.event_type == EventType.COMPLETE

        thread.join()

        # Should have consumed all events
        remaining = tracker.get_events()
        assert len(remaining) == 0