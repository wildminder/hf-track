"""Tests for hf_progress.xet_upload module."""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch

import pytest

from hf_progress.types import EventType, TransferDirection, TransferError


@pytest.fixture
def event_queue():
    return queue.Queue()


def _make_mock_xet():
    """Create a mock hf_xet module with necessary attributes."""
    mock_xet = MagicMock()
    mock_xet.PyXetUploadInfo = MagicMock
    mock_xet.upload_files = MagicMock()
    mock_xet.upload_bytes = MagicMock()
    return mock_xet


class TestUploadFileWithXet:
    """Tests for upload_file_with_xet function."""

    @patch("hf_progress.xet_upload.is_xet_available", return_value=False)
    def test_raises_when_xet_not_available(self, mock_avail, event_queue):
        from hf_progress.xet_upload import upload_file_with_xet

        with pytest.raises(ImportError, match="hf_xet is not installed"):
            upload_file_with_xet(
                file_path="/tmp/test.bin",
                repo_id="user/repo",
                token="hf_test",
                event_queue=event_queue,
            )

    @patch("hf_progress.xet_upload.is_xet_available", return_value=True)
    @patch("hf_progress.xet_upload.XetTokenManager")
    def test_emits_start_and_complete_events(self, mock_tm, mock_avail, event_queue, tmp_path):
        from hf_progress.xet_upload import upload_file_with_xet

        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"x" * 100)

        mock_creds = MagicMock()
        mock_creds.endpoint = "https://xet.example.com"
        mock_creds.token_info = ("token", 9999)
        mock_creds.token_refresher = lambda: ("token", 9999)
        mock_tm.return_value.fetch_upload_credentials.return_value = mock_creds

        mock_xet = _make_mock_xet()
        mock_result = MagicMock()
        mock_result.hash = "abc123"
        mock_result.file_size = 100
        mock_xet.upload_files.return_value = [mock_result]

        with patch.dict("sys.modules", {"hf_xet": mock_xet}):
            result = upload_file_with_xet(
                file_path=str(test_file),
                repo_id="user/repo",
                token="hf_test",
                event_queue=event_queue,
            )

        assert result.success is True
        assert result.filename == "test.bin"
        assert result.transfer_id != ""

        events = []
        while not event_queue.empty():
            events.append(event_queue.get())

        start_events = [e for e in events if e.event_type == EventType.START]
        complete_events = [e for e in events if e.event_type == EventType.COMPLETE]
        assert len(start_events) == 1
        assert len(complete_events) == 1
        assert start_events[0].direction == TransferDirection.UPLOAD
        assert complete_events[0].percentage == 100.0

    @patch("hf_progress.xet_upload.is_xet_available", return_value=True)
    @patch("hf_progress.xet_upload.XetTokenManager")
    def test_emits_error_event_on_failure(self, mock_tm, mock_avail, event_queue, tmp_path):
        from hf_progress.xet_upload import upload_file_with_xet

        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"x" * 100)

        mock_creds = MagicMock()
        mock_tm.return_value.fetch_upload_credentials.return_value = mock_creds

        mock_xet = _make_mock_xet()
        mock_xet.upload_files.side_effect = RuntimeError("upload failed")

        with patch.dict("sys.modules", {"hf_xet": mock_xet}):
            with pytest.raises(RuntimeError, match="upload failed"):
                upload_file_with_xet(
                    file_path=str(test_file),
                    repo_id="user/repo",
                    token="hf_test",
                    event_queue=event_queue,
            )

            events = []
            while not event_queue.empty():
                events.append(event_queue.get())

            error_events = [e for e in events if e.event_type == EventType.ERROR]
            assert len(error_events) == 1
            assert isinstance(error_events[0].error, TransferError)
            assert "upload failed" in error_events[0].error.message


class TestUploadBytesWithXet:
    """Tests for upload_bytes_with_xet function."""

    @patch("hf_progress.xet_upload.is_xet_available", return_value=False)
    def test_raises_when_xet_not_available(self, mock_avail, event_queue):
        from hf_progress.xet_upload import upload_bytes_with_xet

        with pytest.raises(ImportError, match="hf_xet is not installed"):
            upload_bytes_with_xet(
                file_content=b"test data",
                filename="test.bin",
                repo_id="user/repo",
                token="hf_test",
                event_queue=event_queue,
            )

    @patch("hf_progress.xet_upload.is_xet_available", return_value=True)
    @patch("hf_progress.xet_upload.XetTokenManager")
    def test_emits_start_and_complete_events(self, mock_tm, mock_avail, event_queue):
        from hf_progress.xet_upload import upload_bytes_with_xet

        mock_creds = MagicMock()
        mock_tm.return_value.fetch_upload_credentials.return_value = mock_creds

        mock_xet = _make_mock_xet()
        mock_result = MagicMock()
        mock_result.hash = "abc123"
        mock_result.file_size = 9
        mock_xet.upload_bytes.return_value = [mock_result]

        with patch.dict("sys.modules", {"hf_xet": mock_xet}):
            result = upload_bytes_with_xet(
                file_content=b"test data",
                filename="test.bin",
                repo_id="user/repo",
                token="hf_test",
                event_queue=event_queue,
            )

        assert result.success is True
        assert result.filename == "test.bin"

        events = []
        while not event_queue.empty():
            events.append(event_queue.get())

        start_events = [e for e in events if e.event_type == EventType.START]
        complete_events = [e for e in events if e.event_type == EventType.COMPLETE]
        assert len(start_events) == 1
        assert len(complete_events) == 1


class TestXetUploadResult:
    """Tests for XetUploadResult dataclass."""

    def test_default_values(self):
        from hf_progress.xet_upload import XetUploadResult

        result = XetUploadResult(success=True, filename="test.bin")
        assert result.hash == ""
        assert result.file_size == 0
        assert result.transfer_id == ""

    def test_all_fields(self):
        from hf_progress.xet_upload import XetUploadResult

        result = XetUploadResult(
            success=True,
            filename="model.bin",
            hash="abc123",
            file_size=4096,
            transfer_id="tid-456",
        )
        assert result.hash == "abc123"
        assert result.file_size == 4096
        assert result.transfer_id == "tid-456"
