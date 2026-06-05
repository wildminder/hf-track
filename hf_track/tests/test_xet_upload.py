"""Tests for hf_progress.xet_upload module — subprocess-isolated version."""

from __future__ import annotations

import os
import queue
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from hf_track.types import EventType, TransferDirection, TransferError, TransferProgressError
from hf_track.upload import upload_file_with_xet, upload_bytes_with_xet


@pytest.fixture
def event_queue():
    return queue.Queue()


class TestUploadFileWithXet:
    """Tests for upload_file_with_xet function."""

    @patch("hf_track.upload.xet_file.is_xet_available", return_value=False)
    def test_raises_when_xet_not_available(self, mock_avail, event_queue):
        from hf_track.upload import upload_file_with_xet

        with pytest.raises(ImportError, match="hf_xet is not installed"):
            upload_file_with_xet(
                file_path="/tmp/test.bin",
                repo_id="user/repo",
                token="hf_test",
                event_queue=event_queue,
            )

    @patch("hf_track.upload.xet_file.is_xet_available", return_value=True)
    @patch("hf_track.upload.xet_file.XetSubprocessRunner")
    def test_emits_start_and_complete_events(self, MockRunner, mock_avail, event_queue, tmp_path):
        """Should emit START event and return result on success."""
        from hf_track.upload import upload_file_with_xet

        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"x" * 100)

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "success",
            "filename": "test.bin",
            "file_size": 100,
            "transfer_id": "test-upload-1",
            "hash": "abc123",
            "url": "https://huggingface.co/user/repo/blob/main/test.bin",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        result = upload_file_with_xet(
            file_path=str(test_file),
            repo_id="user/repo",
            token="hf_test",
            event_queue=event_queue,
            transfer_id="test-upload-1",
        )

        assert result.success is True
        assert result.filename == "test.bin"
        assert result.hash == "abc123"
        assert result.url == "https://huggingface.co/user/repo/blob/main/test.bin"

        # Check START event
        start_event = event_queue.get_nowait()
        assert start_event.event_type == EventType.START
        assert start_event.transfer_id == "test-upload-1"
        assert start_event.direction == TransferDirection.UPLOAD

        # Verify runner lifecycle
        mock_runner.start.assert_called_once()
        mock_runner.terminate.assert_called()

    @patch("hf_track.upload.xet_file.is_xet_available", return_value=True)
    @patch("hf_track.upload.xet_file.XetSubprocessRunner")
    def test_emits_error_event_on_failure(self, MockRunner, mock_avail, event_queue, tmp_path):
        """Should raise TransferError when worker returns error."""
        from hf_track.upload import upload_file_with_xet

        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"x" * 100)

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "message": "upload failed",
            "error_type": "ConnectionError",
            "retryable": True,
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        with pytest.raises(TransferProgressError, match="upload failed"):
            upload_file_with_xet(
                file_path=str(test_file),
                repo_id="user/repo",
                token="hf_test",
                event_queue=event_queue,
                transfer_id="test-upload-err",
            )

        mock_runner.terminate.assert_called()


class TestUploadBytesWithXet:
    """Tests for upload_bytes_with_xet function."""

    @patch("hf_track.upload.xet_bytes.is_xet_available", return_value=False)
    def test_raises_when_xet_not_available(self, mock_avail, event_queue):
        from hf_track.upload import upload_bytes_with_xet

        with pytest.raises(ImportError, match="hf_xet is not installed"):
            upload_bytes_with_xet(
                file_content=b"hello",
                filename="test.txt",
                repo_id="user/repo",
                token="hf_test",
                event_queue=event_queue,
            )

    @patch("hf_track.upload.xet_bytes.is_xet_available", return_value=True)
    @patch("hf_track.upload.xet_file.XetSubprocessRunner")
    def test_emits_start_and_complete_events(self, MockRunner, mock_avail, event_queue):
        """Should emit START event and return result on success."""
        from hf_track.upload import upload_bytes_with_xet

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "success",
            "filename": "test.txt",
            "file_size": 5,
            "transfer_id": "test-bytes-1",
            "hash": "def456",
            "url": None,
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        result = upload_bytes_with_xet(
            file_content=b"hello",
            filename="test.txt",
            repo_id="user/repo",
            token="hf_test",
            event_queue=event_queue,
            transfer_id="test-bytes-1",
        )

        assert result.success is True
        assert result.filename == "test.txt"
        assert result.file_size == 5

        # Check START event
        start_event = event_queue.get_nowait()
        assert start_event.event_type == EventType.START
        assert start_event.direction == TransferDirection.UPLOAD

    @patch("hf_track.upload.xet_bytes.is_xet_available", return_value=True)
    @patch("hf_track.upload.xet_file.XetSubprocessRunner")
    def test_large_payload_uses_temp_file(self, MockRunner, mock_avail, event_queue):
        """Payloads >10MB should be written to temp file before subprocess."""
        from hf_track.upload import upload_bytes_with_xet
        from hf_track.upload.xet_bytes import _LARGE_PAYLOAD_THRESHOLD

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "success",
            "filename": "large.bin",
            "file_size": _LARGE_PAYLOAD_THRESHOLD + 1,
            "transfer_id": "test-large-1",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        large_content = b"x" * (_LARGE_PAYLOAD_THRESHOLD + 1)

        result = upload_bytes_with_xet(
            file_content=large_content,
            filename="large.bin",
            repo_id="user/repo",
            token="hf_test",
            event_queue=event_queue,
            transfer_id="test-large-1",
        )

        assert result.success is True
        # Verify the worker was called with _upload_bytes_worker
        mock_runner.start.assert_called_once()
        # The params should contain file_path, not file_content
        call_args = mock_runner.start.call_args
        params = call_args[1]["params"] if "params" in call_args[1] else call_args[0][1]
        assert "file_path" in params
        assert "file_content" not in params


class TestXetUploadResult:
    """Tests for XetUploadResult dataclass."""

    def test_default_values(self):
        from hf_track.upload import XetUploadResult

        result = XetUploadResult(success=True, filename="test.bin")
        assert result.success is True
        assert result.filename == "test.bin"
        assert result.hash == ""
        assert result.file_size == 0
        assert result.transfer_id == ""
        assert result.url is None

    def test_all_fields(self):
        from hf_track.upload import XetUploadResult

        result = XetUploadResult(
            success=True,
            filename="model.bin",
            hash="abc123",
            file_size=1024,
            transfer_id="test-123",
            url="https://huggingface.co/user/repo/blob/main/model.bin",
        )
        assert result.hash == "abc123"
        assert result.file_size == 1024
        assert result.transfer_id == "test-123"
        assert result.url == "https://huggingface.co/user/repo/blob/main/model.bin"
