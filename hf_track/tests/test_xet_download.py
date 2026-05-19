"""Tests for hf_progress.xet_download module — subprocess-isolated version."""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch

import pytest

from hf_track.types import EventType, ProgressPhase, TransferDirection, TransferError, TransferProgressError


# ── download_file_with_xet ────────────────────────────────────


class TestDownloadFileWithXet:
    """Tests for download_file_with_xet function."""

    def test_raises_import_error_when_xet_unavailable(self):
        """Should raise ImportError when hf_xet is not installed."""
        from hf_track.xet_download import download_file_with_xet

        with patch("hf_track.xet_download.is_xet_available", return_value=False):
            with pytest.raises(ImportError, match="hf_xet is not installed"):
                download_file_with_xet(
                    file_hash="abc123",
                    file_size=1024,
                    dest_path="/tmp/test.bin",
                    xet_file_data=MagicMock(),
                    token="test-token",
                    event_queue=queue.Queue(),
                )

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_emits_start_event(self, MockRunner, _mock_xet_avail):
        """Should emit a START event before downloading."""
        from hf_track.xet_download import download_file_with_xet

        # Mock runner to return success immediately
        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "success",
            "filename": "model.bin",
            "destination_path": "/tmp/model.bin",
            "file_size": 2048,
            "transfer_id": "test-transfer-1",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        result = download_file_with_xet(
            file_hash="abc123",
            file_size=2048,
            dest_path="/tmp/model.bin",
            xet_file_data=MagicMock(),
            token="test-token",
            event_queue=event_queue,
            transfer_id="test-transfer-1",
        )

        # First event should be START
        start_event = event_queue.get_nowait()
        assert start_event.event_type == EventType.START
        assert start_event.transfer_id == "test-transfer-1"
        assert start_event.direction == TransferDirection.DOWNLOAD
        assert start_event.filename == "model.bin"
        assert start_event.phase == ProgressPhase.DOWNLOADING
        assert start_event.total_bytes == 2048

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_emits_complete_event_on_success(self, MockRunner, _mock_xet_avail):
        """Should emit a COMPLETE event after successful download."""
        from hf_track.xet_download import download_file_with_xet

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "success",
            "filename": "model.bin",
            "destination_path": "/tmp/model.bin",
            "file_size": 2048,
            "transfer_id": "test-transfer-2",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        result = download_file_with_xet(
            file_hash="abc123",
            file_size=2048,
            dest_path="/tmp/model.bin",
            xet_file_data=MagicMock(),
            token="test-token",
            event_queue=event_queue,
            transfer_id="test-transfer-2",
        )

        assert result.success is True
        assert result.filename == "model.bin"
        assert result.destination_path == "/tmp/model.bin"
        assert result.transfer_id == "test-transfer-2"

        # Verify runner lifecycle
        mock_runner.start.assert_called_once()
        mock_runner.terminate.assert_called()

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_emits_error_event_on_failure(self, MockRunner, _mock_xet_avail):
        """Should raise RuntimeError with TransferError message when worker returns error."""
        from hf_track.xet_download import download_file_with_xet

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "message": "download failed",
            "error_type": "RuntimeError",
            "retryable": False,
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        # TransferProgressError is raised with the error message
        with pytest.raises(TransferProgressError, match="download failed"):
            download_file_with_xet(
                file_hash="abc123",
                file_size=2048,
                dest_path="/tmp/model.bin",
                xet_file_data=MagicMock(),
                token="test-token",
                event_queue=event_queue,
                transfer_id="test-transfer-3",
            )

        mock_runner.terminate.assert_called()

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_generates_transfer_id_when_not_provided(self, MockRunner, _mock_xet_avail):
        """Should generate a transfer_id if not provided."""
        from hf_track.xet_download import download_file_with_xet

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "success",
            "filename": "model.bin",
            "destination_path": "/tmp/model.bin",
            "file_size": 1024,
            "transfer_id": "auto-generated",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        result = download_file_with_xet(
            file_hash="abc123",
            file_size=1024,
            dest_path="/tmp/model.bin",
            xet_file_data=MagicMock(),
            token="test-token",
            event_queue=event_queue,
        )

        # Should have generated a transfer_id (UUID format)
        start_event = event_queue.get_nowait()
        assert start_event.transfer_id  # Not empty
        assert len(start_event.transfer_id) == 36  # UUID format

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_cancel_via_is_cancelled_hook(self, MockRunner, _mock_xet_avail):
        """Should terminate runner and emit CANCELLED when is_cancelled returns True."""
        from hf_track.xet_download import download_file_with_xet
        from hf_track.types import TransferCancelledError

        mock_runner = MagicMock()
        # Simulate wait timing out (worker still running)
        call_count = [0]
        def wait_side_effect(timeout=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # First call: timeout
            return {"status": "success"}  # Shouldn't reach here

        mock_runner.wait.side_effect = wait_side_effect
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        with pytest.raises(TransferCancelledError):
            download_file_with_xet(
                file_hash="abc123",
                file_size=1024,
                dest_path="/tmp/model.bin",
                xet_file_data=MagicMock(),
                token="test-token",
                event_queue=event_queue,
                transfer_id="test-cancel",
                is_cancelled=lambda: call_count[0] >= 1,
            )

        mock_runner.terminate.assert_called()


# ── download_files_with_xet ───────────────────────────────────


class TestDownloadFilesWithXet:
    """Tests for download_files_with_xet function."""

    def test_raises_import_error_when_xet_unavailable(self):
        """Should raise ImportError when hf_xet is not installed."""
        from hf_track.xet_download import download_files_with_xet

        with patch("hf_track.xet_download.is_xet_available", return_value=False):
            with pytest.raises(ImportError, match="hf_xet is not installed"):
                download_files_with_xet(
                    file_specs=[{"dest_path": "/tmp/a.bin", "hash": "a", "file_size": 100, "xet_file_data": MagicMock()}],
                    token="test-token",
                    event_queue=queue.Queue(),
                )

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_emits_start_and_complete_for_each_file(self, MockRunner, _mock_xet_avail):
        """Should emit START events for each file and return results."""
        from hf_track.xet_download import download_files_with_xet

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "success",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        results = download_files_with_xet(
            file_specs=[
                {"dest_path": "/tmp/a.bin", "hash": "a", "file_size": 100, "xet_file_data": MagicMock()},
                {"dest_path": "/tmp/b.bin", "hash": "b", "file_size": 200, "xet_file_data": MagicMock()},
            ],
            token="test-token",
            event_queue=event_queue,
            transfer_id="test-batch-1",
        )

        assert len(results) == 2
        assert results[0].filename == "a.bin"
        assert results[1].filename == "b.bin"

        # Should have 2 START events
        events = []
        while not event_queue.empty():
            events.append(event_queue.get_nowait())
        start_events = [e for e in events if e.event_type == EventType.START]
        assert len(start_events) == 2

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_emits_error_events_for_all_files_on_failure(self, MockRunner, _mock_xet_avail):
        """Should emit ERROR events for all files when batch fails."""
        from hf_track.xet_download import download_files_with_xet

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "message": "batch failed",
            "error_type": "RuntimeError",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        with pytest.raises(TransferProgressError, match="batch failed"):
            download_files_with_xet(
                file_specs=[
                    {"dest_path": "/tmp/a.bin", "hash": "a", "file_size": 100, "xet_file_data": MagicMock()},
                    {"dest_path": "/tmp/b.bin", "hash": "b", "file_size": 200, "xet_file_data": MagicMock()},
                ],
                token="test-token",
                event_queue=event_queue,
                transfer_id="test-batch-err",
            )

        # Should have ERROR events for each file
        events = []
        while not event_queue.empty():
            events.append(event_queue.get_nowait())
        error_events = [e for e in events if e.event_type == EventType.ERROR]
        assert len(error_events) == 2
        assert all(isinstance(e.error, TransferError) for e in error_events)
        assert all(e.error.message == "batch failed" for e in error_events)


# ── XetDownloadResult ─────────────────────────────────────────


class TestXetDownloadResult:
    """Tests for XetDownloadResult dataclass."""

    def test_default_values(self):
        from hf_track.xet_download import XetDownloadResult

        result = XetDownloadResult(success=True, filename="test.bin")
        assert result.success is True
        assert result.filename == "test.bin"
        assert result.destination_path == ""
        assert result.file_size == 0
        assert result.transfer_id == ""

    def test_all_fields(self):
        from hf_track.xet_download import XetDownloadResult

        result = XetDownloadResult(
            success=True,
            filename="model.bin",
            destination_path="/tmp/model.bin",
            file_size=1024,
            transfer_id="test-123",
        )
        assert result.destination_path == "/tmp/model.bin"
        assert result.file_size == 1024
        assert result.transfer_id == "test-123"
