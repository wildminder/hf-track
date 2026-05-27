"""Tests for hf_progress.tracker module."""

from __future__ import annotations

import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from hf_track.tracker import HfTracker
from hf_track.types import (
    EventType,
    ProgressEvent,
    TransferCancelledError,
    TransferDirection,
    TransferError,
)


class TestHfTrackerInit:
    """Tests for HfTracker initialization."""

    def test_creates_with_token(self):
        tracker = HfTracker(token="hf_test123")
        assert tracker._token == "hf_test123"
        assert tracker.event_queue is not None
        assert tracker._report_interval == 0.1

    def test_custom_report_interval(self):
        tracker = HfTracker(token="hf_test", report_interval=0.5)
        assert tracker._report_interval == 0.5

    def test_custom_endpoint(self):
        tracker = HfTracker(token="hf_test", endpoint="https://custom.api")
        assert tracker._endpoint == "https://custom.api"
        
    def test_bounded_queue(self):
        """IMP-012: Ensure queue is bounded to prevent OOM."""
        tracker = HfTracker()
        assert getattr(tracker.event_queue, "maxsize", 0) > 0


class TestHfTrackerCancellation:
    """Tests for cancellation logic."""
    
    def test_cancel_flags_transfer(self):
        tracker = HfTracker()
        transfer_id = "test-cancel-1"
        
        assert not tracker.is_cancelled(transfer_id)
        tracker.cancel(transfer_id)
        assert tracker.is_cancelled(transfer_id)


class TestHfTrackerEvents:
    """Tests for HfTracker event consumption methods."""

    def test_get_events_empty(self):
        """get_events should return empty list when queue is empty."""
        tracker = HfTracker(token="hf_test")
        events = tracker.get_events()
        assert events == []

    def test_get_events_returns_all(self):
        """get_events should return all queued events."""
        tracker = HfTracker(token="hf_test")

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
        tracker = HfTracker(token="hf_test")
        events = tracker.get_events(timeout=0)
        assert events == []

    def test_events_generator(self):
        """events() should yield ProgressEvent objects."""
        tracker = HfTracker(token="hf_test")

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
        tracker = HfTracker(token="hf_test")

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
        tracker = HfTracker(token="hf_test")

        result = tracker.wait_for_complete("nonexistent", timeout=0.2)
        assert result is None

    def test_wait_for_complete_error(self):
        """wait_for_complete should return ERROR events too."""
        tracker = HfTracker(token="hf_test")

        tracker.event_queue.put(
            ProgressEvent(
                event_type=EventType.ERROR,
                transfer_id="err-1",
                direction=TransferDirection.UPLOAD,
                filename="test.bin",
                phase="error",
                error=TransferError(message="Connection refused", error_type="ConnectionError"),
            )
        )

        result = tracker.wait_for_complete("err-1", timeout=1)
        assert result is not None
        assert result.event_type == EventType.ERROR
        assert isinstance(result.error, TransferError)
        assert result.error.message == "Connection refused"

    def test_wait_for_complete_ignores_other_transfers(self):
        """wait_for_complete should only match the specified transfer_id."""
        tracker = HfTracker(token="hf_test")

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


class TestHfTrackerUploadRouting:
    """Tests for upload method routing (Xet vs LFS)."""

    def test_upload_file_calls_xet_when_available(self):
        """When hf_xet is available, upload_file should route to Xet."""
        tracker = HfTracker(token="hf_test")

        from unittest.mock import patch

        with patch("hf_track.tracker.is_xet_available", return_value=True):
            with patch.object(
                tracker, "_upload_file_xet", return_value="abc123"
            ) as mock:
                tracker.upload_file(
                    file_path="/tmp/test.bin",
                    repo_id="user/repo",
                    transfer_id="test-1",
                )
                mock.assert_called_once()

    def test_upload_file_calls_lfs_when_xet_unavailable(self):
        """When hf_xet is not available, upload_file should route to LFS."""
        tracker = HfTracker(token="hf_test")

        from unittest.mock import patch

        with patch("hf_track.tracker.is_xet_available", return_value=False):
            with patch.object(
                tracker, "_upload_file_lfs", return_value="https://..."
            ) as mock:
                tracker.upload_file(
                    file_path="/tmp/test.bin",
                    repo_id="user/repo",
                    transfer_id="test-1",
                )
                mock.assert_called_once()

    def test_upload_bytes_calls_xet_when_available(self):
        """When hf_xet is available, upload_bytes should route to Xet."""
        tracker = HfTracker(token="hf_test")

        from unittest.mock import patch

        with patch("hf_track.tracker.is_xet_available", return_value=True):
            with patch.object(
                tracker, "_upload_bytes_xet", return_value="abc123"
            ) as mock:
                tracker.upload_bytes(
                    file_content=b"test data",
                    filename="test.bin",
                    repo_id="user/repo",
                    transfer_id="test-1",
                )
                mock.assert_called_once()

    def test_upload_bytes_calls_temp_when_xet_unavailable(self):
        """When hf_xet is not available, upload_bytes should use temp file."""
        tracker = HfTracker(token="hf_test")

        from unittest.mock import patch

        with patch("hf_track.tracker.is_xet_available", return_value=False):
            with patch.object(
                tracker, "_upload_bytes_via_temp", return_value="https://..."
            ) as mock:
                tracker.upload_bytes(
                    file_content=b"test data",
                    filename="test.bin",
                    repo_id="user/repo",
                    transfer_id="test-1",
                )
                mock.assert_called_once()


class TestHfTrackerEventFlow:
    """Integration tests for event flow through the tracker."""

    def test_manual_event_flow(self):
        """Manually put events and verify consumption."""
        tracker = HfTracker(token="hf_test")

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
        tracker = HfTracker(token="hf_test")

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


# ── _download_snapshot_xet Tests ─────────────────────────────────


class TestDownloadSnapshotXet:
    """Tests for HfTracker._download_snapshot_xet — thin wrapper that
    delegates to download_snapshot_with_xet().
    """

    @patch("hf_track.xet_download.download_snapshot_with_xet")
    def test_delegates_to_download_snapshot_with_xet(self, mock_dl_snap):
        """_download_snapshot_xet delegates to download_snapshot_with_xet."""
        tracker = HfTracker(token="hf_test")
        mock_dl_snap.return_value = "/tmp/test"

        tracker._download_snapshot_xet(
            repo_id="test/repo",
            allow_patterns=None,
            ignore_patterns=None,
            repo_type="model",
            revision="main",
            local_dir="/tmp/test",
            transfer_id="tid-1",
            is_cancelled=lambda: False,
        )

        mock_dl_snap.assert_called_once()

    @patch("hf_track.xet_download.download_snapshot_with_xet")
    def test_passes_all_params(self, mock_dl_snap):
        """All params are forwarded to download_snapshot_with_xet."""
        tracker = HfTracker(token="hf_test")
        mock_dl_snap.return_value = "/tmp/test"

        hook = lambda: False
        tracker._download_snapshot_xet(
            repo_id="test/repo",
            allow_patterns=["*.bin"],
            ignore_patterns=["*.tmp"],
            repo_type="dataset",
            revision="v1.0",
            local_dir="/tmp/test",
            transfer_id="tid-1",
            is_cancelled=hook,
            force_download=True,
        )

        call_kwargs = mock_dl_snap.call_args.kwargs
        assert call_kwargs["repo_id"] == "test/repo"
        assert call_kwargs["allow_patterns"] == ["*.bin"]
        assert call_kwargs["ignore_patterns"] == ["*.tmp"]
        assert call_kwargs["repo_type"] == "dataset"
        assert call_kwargs["revision"] == "v1.0"
        assert call_kwargs["local_dir"] == "/tmp/test"
        assert call_kwargs["transfer_id"] == "tid-1"
        assert call_kwargs["is_cancelled"] is hook
        assert call_kwargs["force_download"] is True
        assert call_kwargs["token"] == "hf_test"
        assert call_kwargs["event_queue"] is tracker.event_queue

    @patch("hf_track.xet_download.download_snapshot_with_xet")
    def test_passes_force_download_default_false(self, mock_dl_snap):
        """force_download defaults to False."""
        tracker = HfTracker(token="hf_test")
        mock_dl_snap.return_value = "/tmp/test"

        tracker._download_snapshot_xet(
            repo_id="test/repo",
            allow_patterns=None,
            ignore_patterns=None,
            repo_type="model",
            revision=None,
            local_dir="/tmp/test",
            transfer_id="tid-1",
            is_cancelled=lambda: False,
        )

        call_kwargs = mock_dl_snap.call_args.kwargs
        assert call_kwargs["force_download"] is False

    @patch("hf_track.xet_download.download_snapshot_with_xet")
    def test_returns_result_path(self, mock_dl_snap):
        """Returns the path from download_snapshot_with_xet."""
        tracker = HfTracker(token="hf_test")
        mock_dl_snap.return_value = "/my/download/path"

        result = tracker._download_snapshot_xet(
            repo_id="test/repo",
            allow_patterns=None,
            ignore_patterns=None,
            repo_type="model",
            revision=None,
            local_dir="/tmp/test",
            transfer_id="tid-1",
            is_cancelled=lambda: False,
        )

        assert result == "/my/download/path"

    @patch("hf_track.xet_download.download_snapshot_with_xet")
    def test_propagates_cancelled_error(self, mock_dl_snap):
        """TransferCancelledError from download_snapshot_with_xet propagates."""
        tracker = HfTracker(token="hf_test")
        mock_dl_snap.side_effect = TransferCancelledError("cancelled")

        with pytest.raises(TransferCancelledError, match="cancelled"):
            tracker._download_snapshot_xet(
                repo_id="test/repo",
                allow_patterns=None,
                ignore_patterns=None,
                repo_type="model",
                revision=None,
                local_dir="/tmp/test",
                transfer_id="tid-1",
                is_cancelled=lambda: False,
            )

    @patch("hf_track.xet_download.download_snapshot_with_xet")
    def test_propagates_other_errors(self, mock_dl_snap):
        """Other exceptions from download_snapshot_with_xet propagate."""
        tracker = HfTracker(token="hf_test")
        mock_dl_snap.side_effect = RuntimeError("xet crashed")

        with pytest.raises(RuntimeError, match="xet crashed"):
            tracker._download_snapshot_xet(
                repo_id="test/repo",
                allow_patterns=None,
                ignore_patterns=None,
                repo_type="model",
                revision=None,
                local_dir="/tmp/test",
                transfer_id="tid-1",
                is_cancelled=lambda: False,
            )

    # NOTE: Old tests that tested metadata-discovery logic (repo_info,
    # get_hf_file_metadata, fnmatch filtering, etc.) have been removed.
    # The new _download_snapshot_xet is a thin wrapper that delegates
    # to download_snapshot_with_xet(), which runs snapshot_download()
    # in a subprocess. Those internal details are now handled by
    # huggingface_hub internally.


# ── download_snapshot Routing Tests ──────────────────────────────


class TestDownloadSnapshotRouting:
    """Tests for download_snapshot() xet routing logic."""

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_snapshot_xet")
    def test_routes_to_xet_when_available(self, mock_xet, _mock_avail):
        """is_xet_available()=True → _download_snapshot_xet is called."""
        tracker = HfTracker(token="hf_test")
        mock_xet.return_value = "/tmp/repo"

        tracker.download_snapshot(repo_id="test/repo")

        mock_xet.assert_called_once()

    @patch("hf_track.tracker.is_xet_available", return_value=False)
    @patch("hf_track.standard_download.download_snapshot")
    def test_routes_to_standard_when_not_installed(self, mock_standard, _mock_avail):
        """is_xet_available()=False → standard download is called."""
        tracker = HfTracker(token="hf_test")
        mock_standard.return_value = "/tmp/repo"

        tracker.download_snapshot(repo_id="test/repo")

        mock_standard.assert_called_once()

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_snapshot_xet")
    def test_fallback_on_value_error(self, mock_xet, _mock_avail):
        """_download_snapshot_xet raises ValueError → standard download used."""
        tracker = HfTracker(token="hf_test")
        mock_xet.side_effect = ValueError("No xet files")

        with patch("hf_track.standard_download.download_snapshot", return_value="/tmp/repo") as mock_std:
            result = tracker.download_snapshot(repo_id="test/repo")

        mock_xet.assert_called_once()
        mock_std.assert_called_once()
        assert result == "/tmp/repo"

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_snapshot_xet")
    def test_fallback_on_runtime_error(self, mock_xet, _mock_avail):
        """_download_snapshot_xet raises RuntimeError → standard download used."""
        tracker = HfTracker(token="hf_test")
        mock_xet.side_effect = RuntimeError("xet crashed")

        with patch("hf_track.standard_download.download_snapshot", return_value="/tmp/repo") as mock_std:
            result = tracker.download_snapshot(repo_id="test/repo")

        mock_xet.assert_called_once()
        mock_std.assert_called_once()
        assert result == "/tmp/repo"

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_snapshot_xet")
    def test_no_fallback_on_cancelled(self, mock_xet, _mock_avail):
        """_download_snapshot_xet raises TransferCancelledError → propagates."""
        tracker = HfTracker(token="hf_test")
        mock_xet.side_effect = TransferCancelledError("cancelled")

        with pytest.raises(TransferCancelledError):
            tracker.download_snapshot(repo_id="test/repo")

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_snapshot_xet")
    def test_cleanup_on_xet_success(self, mock_xet, _mock_avail):
        """cleanup_transfer() is called after successful xet download."""
        tracker = HfTracker(token="hf_test")
        mock_xet.return_value = "/tmp/repo"

        with patch.object(tracker, "cleanup_transfer") as mock_cleanup:
            tracker.download_snapshot(repo_id="test/repo", transfer_id="tid-1")

        mock_cleanup.assert_called_once_with("tid-1")

    @patch("hf_track.tracker.is_xet_available", return_value=False)
    @patch("hf_track.standard_download.download_snapshot")
    def test_cleanup_on_standard_success(self, mock_standard, _mock_avail):
        """cleanup_transfer() is called after successful standard download."""
        tracker = HfTracker(token="hf_test")
        mock_standard.return_value = "/tmp/repo"

        with patch.object(tracker, "cleanup_transfer") as mock_cleanup:
            tracker.download_snapshot(repo_id="test/repo", transfer_id="tid-1")

        mock_cleanup.assert_called_once_with("tid-1")

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_snapshot_xet")
    def test_cleanup_on_cancelled(self, mock_xet, _mock_avail):
        """cleanup_transfer() is called after TransferCancelledError."""
        tracker = HfTracker(token="hf_test")
        mock_xet.side_effect = TransferCancelledError("cancelled")

        with patch.object(tracker, "cleanup_transfer") as mock_cleanup:
            with pytest.raises(TransferCancelledError):
                tracker.download_snapshot(repo_id="test/repo", transfer_id="tid-1")

        mock_cleanup.assert_called_once_with("tid-1")

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_snapshot_xet")
    def test_fallback_logs_warning(self, mock_xet, _mock_avail, caplog):
        """Warning is logged when xet fails and falls back to standard."""
        tracker = HfTracker(token="hf_test")
        mock_xet.side_effect = ValueError("No xet files")

        with patch("hf_track.standard_download.download_snapshot", return_value="/tmp/repo"):
            with caplog.at_level(logging.WARNING):
                tracker.download_snapshot(repo_id="test/repo")

        assert any("falling back to standard" in r.message.lower() for r in caplog.records)

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_snapshot_xet")
    def test_passes_all_params_to_xet(self, mock_xet, _mock_avail):
        """Xet download receives correct params including force_download."""
        tracker = HfTracker(token="hf_test")
        mock_xet.return_value = "/tmp/repo"

        tracker.download_snapshot(
            repo_id="test/repo",
            allow_patterns=["*.bin"],
            ignore_patterns=["*.tmp"],
            repo_type="dataset",
            revision="v1.0",
            local_dir="/tmp/test",
            transfer_id="tid-1",
            force_download=True,
        )

        call_kwargs = mock_xet.call_args.kwargs
        assert call_kwargs["repo_id"] == "test/repo"
        assert call_kwargs["allow_patterns"] == ["*.bin"]
        assert call_kwargs["ignore_patterns"] == ["*.tmp"]
        assert call_kwargs["repo_type"] == "dataset"
        assert call_kwargs["revision"] == "v1.0"
        assert call_kwargs["local_dir"] == "/tmp/test"
        assert call_kwargs["transfer_id"] == "tid-1"
        assert call_kwargs["force_download"] is True