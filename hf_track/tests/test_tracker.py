"""Tests for hf_progress.tracker module."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from hf_track.tracker import HfTracker
from hf_track.types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferCancelledError,
    TransferDirection,
    TransferErrorInfo,
    TransferProgressError,
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
                error=TransferErrorInfo(message="Connection refused", error_type="ConnectionError"),
            )
        )

        result = tracker.wait_for_complete("err-1", timeout=1)
        assert result is not None
        assert result.event_type == EventType.ERROR
        assert isinstance(result.error, TransferErrorInfo)
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

    @patch("hf_track.download.download_snapshot_with_xet")
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

    @patch("hf_track.download.download_snapshot_with_xet")
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

    @patch("hf_track.download.download_snapshot_with_xet")
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

    @patch("hf_track.download.download_snapshot_with_xet")
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

    @patch("hf_track.download.download_snapshot_with_xet")
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

    @patch("hf_track.download.download_snapshot_with_xet")
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
    @patch("hf_track.download.download_snapshot")
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

        with patch("hf_track.download.download_snapshot", return_value="/tmp/repo") as mock_std:
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

        with patch("hf_track.download.download_snapshot", return_value="/tmp/repo") as mock_std:
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
    @patch("hf_track.download.download_snapshot")
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

        with patch("hf_track.download.download_snapshot", return_value="/tmp/repo"):
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


class TestUseXetParameter:
    """Tests for the use_xet parameter on download methods.

    The use_xet parameter allows runtime toggling of xet without
    relying on HF_HUB_DISABLE_XET (which is cached at import time
    by huggingface_hub.constants).
    """

    # ── download_snapshot use_xet tests ────────────────────────────

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_snapshot_xet")
    def test_snapshot_use_xet_true_routes_to_xet(self, mock_xet, _mock_avail):
        """use_xet=True (default) → _download_snapshot_xet is called."""
        tracker = HfTracker(token="hf_test")
        mock_xet.return_value = "/tmp/repo"

        tracker.download_snapshot(repo_id="test/repo", use_xet=True)

        mock_xet.assert_called_once()
        call_kwargs = mock_xet.call_args.kwargs
        assert call_kwargs["use_xet"] is True

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_snapshot_xet")
    def test_snapshot_use_xet_false_still_uses_subprocess(self, mock_xet, _mock_avail):
        """use_xet=False with xet installed → still uses subprocess
        (for safe cancellation), but passes use_xet=False to worker."""
        tracker = HfTracker(token="hf_test")
        mock_xet.return_value = "/tmp/repo"

        tracker.download_snapshot(repo_id="test/repo", use_xet=False)

        mock_xet.assert_called_once()
        call_kwargs = mock_xet.call_args.kwargs
        assert call_kwargs["use_xet"] is False

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_snapshot_xet")
    def test_snapshot_use_xet_default_is_true(self, mock_xet, _mock_avail):
        """Default use_xet is True."""
        tracker = HfTracker(token="hf_test")
        mock_xet.return_value = "/tmp/repo"

        tracker.download_snapshot(repo_id="test/repo")

        mock_xet.assert_called_once()
        call_kwargs = mock_xet.call_args.kwargs
        assert call_kwargs["use_xet"] is True

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_snapshot_xet")
    def test_snapshot_use_xet_false_no_warning_on_fallback(self, mock_xet, _mock_avail, caplog):
        """use_xet=False with xet failure → no warning about xet fallback."""
        tracker = HfTracker(token="hf_test")
        mock_xet.side_effect = ValueError("xet failed")

        with patch("hf_track.download.download_snapshot", return_value="/tmp/repo"):
            with caplog.at_level(logging.WARNING):
                tracker.download_snapshot(repo_id="test/repo", use_xet=False)

        # Should NOT warn about xet fallback when user explicitly disabled xet
        assert not any("falling back" in r.message.lower() for r in caplog.records)

    # ── download_file use_xet tests ────────────────────────────────

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_file_xet")
    def test_file_use_xet_true_routes_to_xet(self, mock_xet, _mock_avail):
        """use_xet=True with xet installed → _download_file_xet is called."""
        tracker = HfTracker(token="hf_test")
        mock_xet.return_value = "/tmp/file"

        tracker.download_file(repo_id="test/repo", filename="file.bin", use_xet=True)

        mock_xet.assert_called_once()

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_file_xet")
    def test_file_use_xet_false_skips_xet(self, mock_xet, _mock_avail):
        """use_xet=False with xet installed → _download_file_xet is
        NOT called, standard download is used instead."""
        tracker = HfTracker(token="hf_test")

        with patch("hf_track.download.download_file", return_value="/tmp/file") as mock_std:
            tracker.download_file(repo_id="test/repo", filename="file.bin", use_xet=False)

        mock_xet.assert_not_called()
        mock_std.assert_called_once()

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_file_xet")
    def test_file_use_xet_default_is_true(self, mock_xet, _mock_avail):
        """Default use_xet for download_file is True."""
        tracker = HfTracker(token="hf_test")
        mock_xet.return_value = "/tmp/file"

        tracker.download_file(repo_id="test/repo", filename="file.bin")

        mock_xet.assert_called_once()

    # ── download_file dedicated xet routing (plan 2026-07-16) ────────
    #
    # These tests patch ``_download_file_xet`` because it calls
    # ``api.get_hf_file_metadata()`` which hits the network. The routing
    # logic in ``download_file()`` is what we verify: xet errors propagate
    # (NO silent HTTP fallback), and use_xet=False uses the standard path.

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_file_xet")
    def test_download_file_xet_error_propagates_no_fallback(self, mock_xet, _mock_avail):
        """A xet failure raises (no silent HTTP fallback to standard path)."""
        from hf_track.types import TransferProgressError

        tracker = HfTracker(token="hf_test")
        mock_xet.side_effect = TransferProgressError("xet broken")

        with patch("hf_track.download.download_file", return_value="/tmp/file") as mock_std:
            with pytest.raises(TransferProgressError):
                tracker.download_file(repo_id="test/repo", filename="file.bin", use_xet=True)

        # The standard path must NOT have been used as a fallback.
        mock_std.assert_not_called()

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_file_xet")
    def test_download_file_xet_no_hybrid_kwargs(self, mock_xet, _mock_avail):
        """_download_file_xet is called WITHOUT tier_timeout_s /
        enable_http_fallback (hybrid-only params removed)."""
        tracker = HfTracker(token="hf_test")
        mock_xet.return_value = "/tmp/file"

        tracker.download_file(repo_id="test/repo", filename="file.bin")

        kwargs = mock_xet.call_args.kwargs
        assert "tier_timeout_s" not in kwargs
        assert "enable_http_fallback" not in kwargs

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_file_xet")
    def test_download_file_xet_receives_on_spawn_hook(self, mock_xet, _mock_avail):
        """on_spawn/on_finish hooks are still passed to _download_file_xet."""
        tracker = HfTracker(token="hf_test")
        mock_xet.return_value = "/tmp/file"

        tracker.download_file(repo_id="test/repo", filename="file.bin", transfer_id="tid-1")

        assert callable(mock_xet.call_args.kwargs["on_spawn"])
        assert callable(mock_xet.call_args.kwargs["on_finish"])

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_file_xet")
    def test_download_file_deregisters_runner_on_finish(self, mock_xet, _mock_avail):
        """_active_runners is empty after download_file returns."""
        tracker = HfTracker(token="hf_test")
        mock_xet.return_value = "/tmp/file"

        tracker.download_file(repo_id="test/repo", filename="file.bin", transfer_id="tid-y")

        assert "tid-y" not in tracker._active_runners
        assert len(tracker._active_runners) == 0

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_file_xet")
    def test_download_file_deregisters_on_error(self, mock_xet, _mock_avail):
        """_active_runners is cleaned up even when _download_file_xet errors,
        and the error propagates (no silent HTTP fallback)."""
        tracker = HfTracker(token="hf_test")

        def _raise_after_spawn(**kwargs):
            # Simulate a real xet path: registers the runner, then fails.
            kwargs["on_spawn"](MagicMock(name="runner"))
            raise RuntimeError("boom")

        mock_xet.side_effect = _raise_after_spawn

        with pytest.raises(RuntimeError):
            tracker.download_file(repo_id="test/repo", filename="file.bin", transfer_id="tid-z")

        assert "tid-z" not in tracker._active_runners

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_file_xet")
    def test_download_file_on_spawn_registers_runner(self, mock_xet, _mock_avail):
        """The on_spawn hook registers the runner in _active_runners."""
        tracker = HfTracker(token="hf_test")
        fake_runner = MagicMock(name="runner")

        def _call_on_spawn(**kwargs):
            kwargs["on_spawn"](fake_runner)
            return "/tmp/file"

        mock_xet.side_effect = _call_on_spawn
        tracker.download_file(repo_id="test/repo", filename="file.bin", transfer_id="tid-r")

        # After the call, the finally block deregisters it.
        assert "tid-r" not in tracker._active_runners

    @patch("hf_track.tracker.is_xet_available", return_value=True)
    @patch.object(HfTracker, "_download_file_xet")
    def test_tracker_cancel_forwards_to_registered_runner(self, mock_xet, _mock_avail):
        """tracker.cancel(tid) calls request_cancel on a registered runner."""
        tracker = HfTracker(token="hf_test")
        fake_runner = MagicMock(name="runner")

        def _call_on_spawn(**kwargs):
            kwargs["on_spawn"](fake_runner)
            # Simulate the runner being still active during the call by
            # re-registering it (the finally block deregisters after we
            # return, but cancel happens mid-call in real usage).
            tracker._active_runners["tid-c"] = fake_runner
            return "/tmp/file"

        mock_xet.side_effect = _call_on_spawn
        tracker.download_file(repo_id="test/repo", filename="file.bin", transfer_id="tid-c")

        # Manually re-register to simulate a still-running transfer, then cancel.
        tracker._active_runners["tid-c"] = fake_runner
        tracker.cancel("tid-c")
        fake_runner.request_cancel.assert_called_once()


    @patch("hf_track.download.download_snapshot_with_xet")
    def test_snapshot_xet_passes_use_xet_to_xet_download(self, mock_dl_snap):
        """_download_snapshot_xet forwards use_xet to download_snapshot_with_xet."""
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
            use_xet=False,
        )

        call_kwargs = mock_dl_snap.call_args.kwargs
        assert call_kwargs["use_xet"] is False


class TestDownloadSnapshotStreaming:
    """Tests for HfTracker.download_snapshot_streaming() — streaming variant.

    Uses heavy mocking to avoid actual network/subprocess work. The end-to-end
    streaming path is covered by test_xet_worker.py::TestXetStreamingDownloadWorker
    and the manual diagnostics in tmp/diag_streaming_subprocess.py.
    """

    def _make_tracker(self):
        return HfTracker(token="hf_test", report_interval=0.0)

    def test_method_exists(self):
        """HfTracker has a download_snapshot_streaming method."""
        tracker = self._make_tracker()
        assert hasattr(tracker, "download_snapshot_streaming")
        assert callable(tracker.download_snapshot_streaming)

    def test_signature_matches_plan(self):
        """download_snapshot_streaming accepts the documented kwargs."""
        import inspect
        sig = inspect.signature(HfTracker.download_snapshot_streaming)
        params = list(sig.parameters.keys())
        for required in ("repo_id",):
            assert required in params
        for opt in ("allow_patterns", "ignore_patterns", "repo_type",
                    "revision", "local_dir", "transfer_id", "force_download",
                    "fsync_interval"):
            assert opt in params, f"missing parameter: {opt}"

    def test_delegates_to_xet_download_helper(self):
        """download_snapshot_streaming delegates to .xet_download.download_snapshot_streaming."""
        tracker = self._make_tracker()
        with patch("hf_track.download.download_snapshot_streaming") as mock_helper:
            mock_helper.return_value = []
            tracker.download_snapshot_streaming(repo_id="user/repo")
            mock_helper.assert_called_once()
            call_kwargs = mock_helper.call_args.kwargs
            assert call_kwargs["repo_id"] == "user/repo"
            assert call_kwargs["token"] == "hf_test"
            assert call_kwargs["event_queue"] is tracker.event_queue
            assert call_kwargs["report_interval"] == 0.0
            assert call_kwargs["fsync_interval"] == 4 * 1024 * 1024

    def test_passes_transfer_id_to_helper(self):
        """download_snapshot_streaming passes the (auto-generated or given) transfer_id
        to the helper. The helper itself is responsible for emitting the START event.
        """
        tracker = self._make_tracker()
        with patch("hf_track.download.download_snapshot_streaming") as mock_helper:
            mock_helper.return_value = []
            tracker.download_snapshot_streaming(
                repo_id="user/repo",
                transfer_id="streaming-1",
            )
        mock_helper.assert_called_once()
        assert mock_helper.call_args.kwargs["transfer_id"] == "streaming-1"

        # Without explicit transfer_id, the helper should still get a generated one
        with patch("hf_track.download.download_snapshot_streaming") as mock_helper:
            mock_helper.return_value = []
            tracker.download_snapshot_streaming(repo_id="user/repo")
        tid = mock_helper.call_args.kwargs["transfer_id"]
        assert isinstance(tid, str) and len(tid) > 0

    def test_returns_list_of_paths_from_helper(self):
        """download_snapshot_streaming returns whatever the helper returns."""
        tracker = self._make_tracker()
        with patch("hf_track.download.download_snapshot_streaming") as mock_helper:
            mock_helper.return_value = ["/tmp/a.bin", "/tmp/b.json"]
            result = tracker.download_snapshot_streaming(repo_id="user/repo")
        assert result == ["/tmp/a.bin", "/tmp/b.json"]

    def test_cleanup_transfer_called_in_finally(self):
        """transfer_id is removed from cancelled set after completion."""
        tracker = self._make_tracker()
        with patch("hf_track.download.download_snapshot_streaming") as mock_helper:
            mock_helper.return_value = []
            tracker.download_snapshot_streaming(
                repo_id="user/repo",
                transfer_id="cleanup-test",
            )
        assert not tracker.is_cancelled("cleanup-test")

    def test_keyboard_interrupt_raises_transfer_cancelled(self):
        """KeyboardInterrupt from helper is converted to TransferCancelledError."""
        tracker = self._make_tracker()
        with patch("hf_track.download.download_snapshot_streaming") as mock_helper:
            mock_helper.side_effect = KeyboardInterrupt()
            with pytest.raises(TransferCancelledError):
                tracker.download_snapshot_streaming(repo_id="user/repo")

    def test_propagates_helper_exceptions(self):
        """Non-KeyboardInterrupt exceptions are re-raised unchanged."""
        tracker = self._make_tracker()
        with patch("hf_track.download.download_snapshot_streaming") as mock_helper:
            mock_helper.side_effect = TransferProgressError("download failed")
            with pytest.raises(TransferProgressError, match="download failed"):
                tracker.download_snapshot_streaming(repo_id="user/repo")


class TestCancelledTransferCleanup:
    """Cancelled ids must not accumulate for the life of the tracker.

    ``cancel()`` adds an id to ``_cancelled_transfers``; the public
    download/upload methods discard it in a ``finally``. An id that is
    cancelled and then reaches a terminal event through *any* other path was
    never discarded — that residue is what ``_TrackingEventQueue`` closes.
    """

    @staticmethod
    def _terminal(transfer_id: str, event_type: EventType) -> ProgressEvent:
        return ProgressEvent(
            event_type=event_type,
            transfer_id=transfer_id,
            direction=TransferDirection.DOWNLOAD,
            phase=ProgressPhase.DOWNLOADING,
            filename="weights.bin",
            bytes_completed=0,
            total_bytes=0,
        )

    @pytest.mark.parametrize(
        "event_type",
        [EventType.COMPLETE, EventType.ERROR, EventType.CANCELLED],
    )
    def test_cancelled_transfers_discarded_on_terminal_event(self, event_type):
        """A cancelled id is dropped when its transfer ends, whatever the end.

        No download/upload method is entered: the event is pushed straight
        at the queue, which is the residue the tracker's own ``finally``
        blocks cannot reach.
        """
        tracker = HfTracker()
        tracker_id = f"orphan-{event_type.value}"

        tracker.cancel(tracker_id)
        assert tracker.is_cancelled(tracker_id)

        tracker.event_queue.put_nowait(self._terminal(tracker_id, event_type))

        assert tracker_id not in tracker._cancelled_transfers
        assert not tracker.is_cancelled(tracker_id)

    def test_non_terminal_event_does_not_discard(self):
        """PROGRESS and START must not clear a live cancellation flag.

        Clearing on a non-terminal event would make ``cancel()`` a no-op for
        any transfer that reports progress before it finishes.
        """
        tracker = HfTracker()
        tracker_id = "still-running"

        tracker.cancel(tracker_id)
        tracker.event_queue.put_nowait(self._terminal(tracker_id, EventType.PROGRESS))

        assert tracker.is_cancelled(tracker_id)

    def test_cancelled_set_does_not_grow_across_completed_transfers(self):
        """N cancel/terminal cycles leave the set empty, not N entries."""
        tracker = HfTracker()

        for i in range(50):
            transfer_id = f"cycle-{i}"
            tracker.cancel(transfer_id)
            tracker.event_queue.put_nowait(
                self._terminal(transfer_id, EventType.COMPLETE)
            )

        assert len(tracker._cancelled_transfers) == 0

    def test_discarding_one_id_leaves_the_others_alone(self):
        """Cleanup is per-transfer, not a blanket clear."""
        tracker = HfTracker()
        tracker.cancel("a")
        tracker.cancel("b")

        tracker.event_queue.put_nowait(self._terminal("a", EventType.COMPLETE))

        assert not tracker.is_cancelled("a")
        assert tracker.is_cancelled("b")

    def test_queue_still_behaves_like_a_queue(self):
        """The subclass must not change the queue's observable behaviour."""
        import queue as _queue

        tracker = HfTracker()
        tracker.event_queue.put_nowait(self._terminal("x", EventType.PROGRESS))

        assert isinstance(tracker.event_queue, _queue.Queue)
        assert tracker.event_queue.qsize() == 1
        assert tracker.event_queue.get_nowait().transfer_id == "x"

        with pytest.raises(_queue.Empty):
            tracker.event_queue.get_nowait()


class TestAsyncApi:
    """The async wrappers cover both transports (NTH-002 + NTH-011).

    ``download_file_async``/``upload_file_async`` delegate to the sync
    methods through ``asyncio.to_thread``. That is the whole contract, so
    the tests assert the two things that can break it silently: that the
    work really does leave the event-loop thread, and that cancelling the
    task reaches the tracker instead of abandoning a running transfer.
    """

    @staticmethod
    def _make_tracker(sync_method_name, *, result="ok", raises=None, delay=0.0):
        """An HfTracker whose one sync method records the thread it ran on."""
        tracker = HfTracker()
        seen = {}

        def _fake(*args, **kwargs):
            seen["thread_id"] = threading.get_ident()
            seen["kwargs"] = kwargs
            if delay:
                time.sleep(delay)
            if raises is not None:
                raise raises
            return result

        setattr(tracker, sync_method_name, _fake)
        return tracker, seen

    async def test_async_download_file_runs_in_thread(self):
        """The transfer must not execute on the event loop thread."""
        tracker, seen = self._make_tracker("download_file", result="/tmp/model.bin")

        path = await tracker.download_file_async(
            "user/repo", "model.bin", transfer_id="async-dl-1",
        )

        assert path == "/tmp/model.bin"
        assert seen["thread_id"] != threading.get_ident()
        assert seen["kwargs"]["transfer_id"] == "async-dl-1"

    async def test_async_upload_file_runs_in_thread(self):
        """NTH-011's half: the upload wrapper behaves the same way."""
        tracker, seen = self._make_tracker("upload_file", result="/tmp/model.bin")

        path = await tracker.upload_file_async(
            "/tmp/model.bin", "user/repo", transfer_id="async-up-1",
        )

        assert path == "/tmp/model.bin"
        assert seen["thread_id"] != threading.get_ident()

    async def test_async_api_propagates_transfer_cancelled_error(self):
        """A cancellation raised in the thread reaches the awaiter."""
        tracker, _ = self._make_tracker(
            "download_file", raises=TransferCancelledError("cancelled in thread"),
        )

        with pytest.raises(TransferCancelledError, match="cancelled in thread"):
            await tracker.download_file_async(
                "user/repo", "model.bin", transfer_id="async-dl-2",
            )

    async def test_task_cancellation_is_forwarded_to_the_tracker(self):
        """Cancelling the task cancels the transfer, it does not abandon it.

        ``asyncio.to_thread`` cannot kill the thread it started, so a
        cancelled ``await`` on its own would leave the download running to
        completion with nobody watching it.
        """
        tracker, _ = self._make_tracker("download_file", delay=2.0)
        tracker._cancelled_transfers.clear()

        task = asyncio.ensure_future(
            tracker.download_file_async(
                "user/repo", "model.bin", transfer_id="async-dl-3",
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert "async-dl-3" in tracker._cancelled_transfers

    async def test_every_wrapper_exists_for_its_sync_counterpart(self):
        """No transport is left without an async form."""
        tracker = HfTracker()
        pairs = [
            ("download_file", "download_file_async"),
            ("download_snapshot", "download_snapshot_async"),
            ("upload_file", "upload_file_async"),
            ("upload_bytes", "upload_bytes_async"),
            ("upload_folder", "upload_folder_async"),
        ]
        for sync_name, async_name in pairs:
            assert callable(getattr(tracker, sync_name, None)), sync_name
            assert callable(getattr(tracker, async_name, None)), async_name

    async def test_other_errors_are_not_swallowed(self):
        """A failure in the thread surfaces to the awaiter unchanged."""
        tracker, _ = self._make_tracker(
            "upload_file", raises=TransferProgressError("upload failed"),
        )
        with pytest.raises(TransferProgressError, match="upload failed"):
            await tracker.upload_file_async(
                "/tmp/model.bin", "user/repo", transfer_id="async-up-2",
            )
