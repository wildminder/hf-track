"""Tests for hf_progress.standard_upload module."""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch

import pytest

from hf_track.types import EventType, TransferDirection


@pytest.fixture
def event_queue():
    return queue.Queue()


class TestUploadFile:
    """Tests for the upload_file function (standard HTTP upload)."""

    @patch("hf_track.standard_upload.generate_transfer_id", return_value="tid-1")
    @patch("huggingface_hub.HfApi")
    def test_emits_complete_on_success(self, mock_api_cls, mock_tid, event_queue, tmp_path):
        from hf_track.standard_upload import upload_file

        # Create a temp file to upload
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"x" * 100)

        mock_api = MagicMock()
        mock_api.upload_file.return_value = "https://huggingface.co/user/repo/resolve/main/test.bin"
        mock_api_cls.return_value = mock_api

        result = upload_file(
            file_path=str(test_file),
            repo_id="user/repo",
            token="hf_test",
            event_queue=event_queue,
            path_in_repo="test.bin",
        )

        # upload_file returns the URL string from HfApi.upload_file
        assert result == "https://huggingface.co/user/repo/resolve/main/test.bin"

        events = []
        while not event_queue.empty():
            events.append(event_queue.get())

        complete_events = [e for e in events if e.event_type == EventType.COMPLETE]
        assert len(complete_events) == 1
        assert complete_events[0].direction == TransferDirection.UPLOAD
        assert complete_events[0].bytes_completed == 100
        assert complete_events[0].filename == "test.bin"

    @patch("hf_track.standard_upload.generate_transfer_id", return_value="tid-2")
    @patch("huggingface_hub.HfApi")
    def test_emits_error_on_failure(self, mock_api_cls, mock_tid, event_queue, tmp_path):
        from hf_track.standard_upload import upload_file

        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"x" * 100)

        mock_api = MagicMock()
        mock_api.upload_file.side_effect = RuntimeError("upload failed")
        mock_api_cls.return_value = mock_api

        with pytest.raises(RuntimeError, match="upload failed"):
            upload_file(
                file_path=str(test_file),
                repo_id="user/repo",
                token="hf_test",
                event_queue=event_queue,
                path_in_repo="test.bin",
            )

        events = []
        while not event_queue.empty():
            events.append(event_queue.get())

        error_events = [e for e in events if e.event_type == EventType.ERROR]
        assert len(error_events) == 1
