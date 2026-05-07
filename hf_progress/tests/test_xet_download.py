"""Tests for hf_progress.xet_download module."""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch

import pytest

from hf_progress.types import EventType, ProgressPhase, TransferDirection, TransferError


# ── download_file_with_xet ────────────────────────────────────


class TestDownloadFileWithXet:
    """Tests for download_file_with_xet function."""

    def test_raises_import_error_when_xet_unavailable(self):
        """Should raise ImportError when hf_xet is not installed."""
        from hf_progress.xet_download import download_file_with_xet

        with patch("hf_progress.xet_download.is_xet_available", return_value=False):
            with pytest.raises(ImportError, match="hf_xet is not installed"):
                download_file_with_xet(
                    file_hash="abc123",
                    file_size=1024,
                    dest_path="/tmp/test.bin",
                    xet_file_data=MagicMock(),
                    token="test-token",
                    event_queue=queue.Queue(),
                )

    @patch("hf_progress.xet_download.is_xet_available", return_value=True)
    @patch("hf_progress.xet_download.XetTokenManager")
    def test_emits_start_event(self, mock_token_mgr_cls, _mock_xet_avail):
        """Should emit a START event before downloading."""
        from hf_progress.xet_download import download_file_with_xet

        mock_creds = MagicMock()
        mock_token_mgr_cls.return_value.fetch_download_credentials.return_value = mock_creds

        event_queue = queue.Queue()

        mock_xet = MagicMock()
        with patch.dict("sys.modules", {"hf_xet": mock_xet}):
            mock_xet.PyXetDownloadInfo = MagicMock
            mock_xet.download_files = MagicMock()

            download_file_with_xet(
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

    @patch("hf_progress.xet_download.is_xet_available", return_value=True)
    @patch("hf_progress.xet_download.XetTokenManager")
    def test_emits_complete_event_on_success(self, mock_token_mgr_cls, _mock_xet_avail):
        """Should emit a COMPLETE event after successful download."""
        from hf_progress.xet_download import download_file_with_xet

        mock_creds = MagicMock()
        mock_token_mgr_cls.return_value.fetch_download_credentials.return_value = mock_creds

        event_queue = queue.Queue()

        mock_xet = MagicMock()
        with patch.dict("sys.modules", {"hf_xet": mock_xet}):
            mock_xet.PyXetDownloadInfo = MagicMock
            mock_xet.download_files = MagicMock()

            result = download_file_with_xet(
                file_hash="abc123",
                file_size=4096,
                dest_path="/tmp/data.bin",
                xet_file_data=MagicMock(),
                token="test-token",
                event_queue=event_queue,
                transfer_id="test-transfer-2",
            )

        # Drain START event
        event_queue.get_nowait()

        # Next event should be COMPLETE
        complete_event = event_queue.get_nowait()
        assert complete_event.event_type == EventType.COMPLETE
        assert complete_event.transfer_id == "test-transfer-2"
        assert complete_event.bytes_completed == 4096
        assert complete_event.percentage == 100.0

        # Verify result
        assert result.success is True
        assert result.filename == "data.bin"
        assert result.destination_path == "/tmp/data.bin"
        assert result.file_size == 4096
        assert result.transfer_id == "test-transfer-2"

    @patch("hf_progress.xet_download.is_xet_available", return_value=True)
    @patch("hf_progress.xet_download.XetTokenManager")
    def test_emits_error_event_on_failure(self, mock_token_mgr_cls, _mock_xet_avail):
        """Should emit an ERROR event and re-raise when download fails."""
        from hf_progress.xet_download import download_file_with_xet

        mock_creds = MagicMock()
        mock_token_mgr_cls.return_value.fetch_download_credentials.return_value = mock_creds

        event_queue = queue.Queue()

        mock_xet = MagicMock()
        with patch.dict("sys.modules", {"hf_xet": mock_xet}):
            mock_xet.PyXetDownloadInfo = MagicMock
            mock_xet.download_files = MagicMock(side_effect=RuntimeError("download failed"))

            with pytest.raises(RuntimeError, match="download failed"):
                download_file_with_xet(
                    file_hash="abc123",
                    file_size=1024,
                    dest_path="/tmp/fail.bin",
                    xet_file_data=MagicMock(),
                    token="test-token",
                    event_queue=event_queue,
                    transfer_id="test-transfer-3",
            )

            # Drain START event
            event_queue.get_nowait()

            # Next event should be ERROR
            error_event = event_queue.get_nowait()
            assert error_event.event_type == EventType.ERROR
            assert error_event.transfer_id == "test-transfer-3"
            assert isinstance(error_event.error, TransferError)
            assert error_event.error.message == "download failed"

    @patch("hf_progress.xet_download.is_xet_available", return_value=True)
    @patch("hf_progress.xet_download.XetTokenManager")
    def test_generates_transfer_id_when_not_provided(self, mock_token_mgr_cls, _mock_xet_avail):
        """Should auto-generate a transfer_id when none is provided."""
        from hf_progress.xet_download import download_file_with_xet

        mock_creds = MagicMock()
        mock_token_mgr_cls.return_value.fetch_download_credentials.return_value = mock_creds

        event_queue = queue.Queue()

        mock_xet = MagicMock()
        with patch.dict("sys.modules", {"hf_xet": mock_xet}):
            mock_xet.PyXetDownloadInfo = MagicMock
            mock_xet.download_files = MagicMock()

            result = download_file_with_xet(
                file_hash="abc123",
                file_size=1024,
                dest_path="/tmp/auto.bin",
                xet_file_data=MagicMock(),
                token="test-token",
                event_queue=event_queue,
            )

        # transfer_id should have been auto-generated (non-empty string)
        assert result.transfer_id
        assert len(result.transfer_id) > 0


# ── download_files_with_xet ───────────────────────────────────


class TestDownloadFilesWithXet:
    """Tests for download_files_with_xet function."""

    def test_raises_import_error_when_xet_unavailable(self):
        """Should raise ImportError when hf_xet is not installed."""
        from hf_progress.xet_download import download_files_with_xet

        with patch("hf_progress.xet_download.is_xet_available", return_value=False):
            with pytest.raises(ImportError, match="hf_xet is not installed"):
                download_files_with_xet(
                    file_specs=[{"dest_path": "/tmp/a.bin", "hash": "a1", "file_size": 100, "xet_file_data": MagicMock()}],
                    token="test-token",
                    event_queue=queue.Queue(),
                )

    @patch("hf_progress.xet_download.is_xet_available", return_value=True)
    @patch("hf_progress.xet_download.XetTokenManager")
    def test_emits_start_and_complete_for_each_file(self, mock_token_mgr_cls, _mock_xet_avail):
        """Should emit START and COMPLETE events for each file in the batch."""
        from hf_progress.xet_download import download_files_with_xet

        mock_creds = MagicMock()
        mock_token_mgr_cls.return_value.fetch_download_credentials.return_value = mock_creds

        event_queue = queue.Queue()

        file_specs = [
            {"dest_path": "/tmp/file1.bin", "hash": "h1", "file_size": 100, "xet_file_data": MagicMock()},
            {"dest_path": "/tmp/file2.bin", "hash": "h2", "file_size": 200, "xet_file_data": MagicMock()},
        ]

        mock_xet = MagicMock()
        with patch.dict("sys.modules", {"hf_xet": mock_xet}):
            mock_xet.PyXetDownloadInfo = MagicMock
            mock_xet.download_files = MagicMock()

            results = download_files_with_xet(
                file_specs=file_specs,
                token="test-token",
                event_queue=event_queue,
                transfer_id="batch-1",
            )

        # Collect all events
        events = []
        while not event_queue.empty():
            events.append(event_queue.get_nowait())

        start_events = [e for e in events if e.event_type == EventType.START]
        complete_events = [e for e in events if e.event_type == EventType.COMPLETE]

        assert len(start_events) == 2
        assert len(complete_events) == 2
        assert len(results) == 2
        assert all(r.success for r in results)
        assert results[0].filename == "file1.bin"
        assert results[1].filename == "file2.bin"

    @patch("hf_progress.xet_download.is_xet_available", return_value=True)
    @patch("hf_progress.xet_download.XetTokenManager")
    def test_emits_error_events_for_all_files_on_failure(self, mock_token_mgr_cls, _mock_xet_avail):
        """Should emit ERROR events for every file when batch download fails."""
        from hf_progress.xet_download import download_files_with_xet

        mock_creds = MagicMock()
        mock_token_mgr_cls.return_value.fetch_download_credentials.return_value = mock_creds

        event_queue = queue.Queue()

        file_specs = [
            {"dest_path": "/tmp/file1.bin", "hash": "h1", "file_size": 100, "xet_file_data": MagicMock()},
            {"dest_path": "/tmp/file2.bin", "hash": "h2", "file_size": 200, "xet_file_data": MagicMock()},
        ]

        mock_xet = MagicMock()
        with patch.dict("sys.modules", {"hf_xet": mock_xet}):
            mock_xet.PyXetDownloadInfo = MagicMock
            mock_xet.download_files = MagicMock(side_effect=RuntimeError("batch failed"))

            with pytest.raises(RuntimeError, match="batch failed"):
                download_files_with_xet(
                    file_specs=file_specs,
                    token="test-token",
                    event_queue=event_queue,
                    transfer_id="batch-err",
            )

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
        """Should have correct default values."""
        from hf_progress.xet_download import XetDownloadResult

        result = XetDownloadResult(success=True, filename="test.bin")
        assert result.success is True
        assert result.filename == "test.bin"
        assert result.destination_path == ""
        assert result.file_size == 0
        assert result.transfer_id == ""

    def test_all_fields(self):
        """Should accept all fields."""
        from hf_progress.xet_download import XetDownloadResult

        result = XetDownloadResult(
            success=True,
            filename="model.bin",
            destination_path="/tmp/model.bin",
            file_size=4096,
            transfer_id="tid-123",
        )
        assert result.destination_path == "/tmp/model.bin"
        assert result.file_size == 4096
        assert result.transfer_id == "tid-123"
