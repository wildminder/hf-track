"""Tests for _xet_worker module — worker functions and serialization helpers."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from hf_progress._xet_worker import (
    _deserialize_xet_file_data,
    _download_batch_worker,
    _download_worker,
    _make_progress_callback,
    _serialize_xet_file_data,
    _upload_bytes_worker,
    _upload_file_worker,
)
from hf_progress.subprocess_messages import MSG_CANCELLED, MSG_ERROR, MSG_RESULT, SubprocessMessage


# ── Serialization Tests ──────────────────────────────────────────


class TestXetFileDataSerialization:
    """Test _serialize_xet_file_data / _deserialize_xet_file_data."""

    def test_serialize_xet_file_data(self):
        """Serialize an XetFileData-like object to dict."""
        xfd = MagicMock()
        xfd.file_hash = "abc123"
        xfd.refresh_route = "https://xet.example.com/refresh"
        result = _serialize_xet_file_data(xfd)
        assert result == {"file_hash": "abc123", "refresh_route": "https://xet.example.com/refresh"}

    def test_serialize_none(self):
        """Serializing None returns empty dict."""
        assert _serialize_xet_file_data(None) == {}

    def test_deserialize_xet_file_data(self):
        """Deserialize a dict back to an object with .file_hash and .refresh_route."""
        data = {"file_hash": "def456", "refresh_route": "https://xet.example.com/refresh2"}
        obj = _deserialize_xet_file_data(data)
        assert obj is not None
        assert obj.file_hash == "def456"
        assert obj.refresh_route == "https://xet.example.com/refresh2"

    def test_deserialize_empty_dict(self):
        """Deserializing empty dict returns None."""
        assert _deserialize_xet_file_data({}) is None

    def test_round_trip(self):
        """Serialize then deserialize preserves data."""
        xfd = MagicMock()
        xfd.file_hash = "xyz789"
        xfd.refresh_route = "https://xet.example.com/refresh3"
        serialized = _serialize_xet_file_data(xfd)
        restored = _deserialize_xet_file_data(serialized)
        assert restored.file_hash == "xyz789"
        assert restored.refresh_route == "https://xet.example.com/refresh3"


# ── Progress Callback Tests ──────────────────────────────────────


class TestMakeProgressCallback:
    """Test _make_progress_callback that runs in child process."""

    def test_callback_emits_event_message(self):
        """Callback puts a SubprocessMessage(event) into mp_queue."""
        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()

        callback = _make_progress_callback(
            filename="model.bin",
            total_bytes=1000,
            transfer_id="test-123",
            mp_queue=mp_queue,
            cancel_event=cancel_event,
            direction="download",
        )

        # Simulate a Rust callback invocation
        total_update = MagicMock()
        total_update.total_bytes_completed = 500
        total_update.total_bytes = 1000
        total_update.total_bytes_completion_rate = 100.0
        total_update.total_transfer_bytes_completed = 600
        total_update.total_transfer_bytes = 1000
        total_update.total_transfer_bytes_completion_rate = 120.0

        item_updates = []

        callback(total_update, item_updates)

        msg = mp_queue.get(timeout=2)
        assert msg.msg_type == "event"
        assert msg.payload["event_type"] == "progress"
        assert msg.payload["transfer_id"] == "test-123"
        assert msg.payload["filename"] == "model.bin"
        assert msg.payload["bytes_completed"] == 600  # transfer_completed preferred for download
        assert msg.payload["total_bytes"] == 1000

    def test_callback_raises_on_cancel_event(self):
        """Callback raises TransferCancelledError when cancel_event is set."""
        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()
        cancel_event.set()

        callback = _make_progress_callback(
            filename="model.bin",
            total_bytes=1000,
            transfer_id="test-123",
            mp_queue=mp_queue,
            cancel_event=cancel_event,
        )

        total_update = MagicMock()
        total_update.total_bytes_completed = 500
        total_update.total_bytes = 1000
        total_update.total_bytes_completion_rate = 0
        total_update.total_transfer_bytes_completed = 0
        total_update.total_transfer_bytes = 0
        total_update.total_transfer_bytes_completion_rate = 0

        with pytest.raises(Exception, match="Transfer cancelled"):
            callback(total_update, [])

    def test_callback_not_cancelled_by_default(self):
        """Callback proceeds normally when cancel_event is not set."""
        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()

        callback = _make_progress_callback(
            filename="model.bin",
            total_bytes=1000,
            transfer_id="test-123",
            mp_queue=mp_queue,
            cancel_event=cancel_event,
        )

        total_update = MagicMock()
        total_update.total_bytes_completed = 100
        total_update.total_bytes = 1000
        total_update.total_bytes_completion_rate = 50.0
        total_update.total_transfer_bytes_completed = 0
        total_update.total_transfer_bytes = 0
        total_update.total_transfer_bytes_completion_rate = 0

        # Should not raise
        callback(total_update, [])

        msg = mp_queue.get(timeout=2)
        assert msg.msg_type == "event"


# ── Download Worker Tests ────────────────────────────────────────


class TestDownloadWorker:
    """Test _download_worker function."""

    def _make_params(self, **overrides):
        """Create default download params with overrides."""
        params = {
            "file_hash": "abc123",
            "file_size": 1000,
            "dest_path": "/tmp/model.bin",
            "xet_file_data": {"file_hash": "abc123", "refresh_route": "https://xet.example.com/refresh"},
            "token": "hf_test",
            "endpoint": None,
            "transfer_id": "test-123",
            "report_interval": 0.1,
            "request_headers": {},
        }
        params.update(overrides)
        return params

    @patch("hf_progress._xet_worker._deserialize_xet_file_data")
    @patch("huggingface_hub.utils._xet.refresh_xet_connection_info")
    @patch("hf_xet.download_files", create=True)
    @patch("hf_xet.PyXetDownloadInfo", create=True)
    def test_download_worker_success(self, mock_info_cls, mock_download, mock_refresh, mock_deserialize):
        """Worker emits result message on successful download."""
        mock_deserialize.return_value = MagicMock(file_hash="abc123", refresh_route="https://xet.example.com/refresh")
        mock_conn = MagicMock()
        mock_conn.endpoint = "https://xet.example.com"
        mock_conn.access_token = "token123"
        mock_conn.expiration_unix_epoch = 9999999999
        mock_refresh.return_value = mock_conn
        mock_download.return_value = None

        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()

        # We need to mock hf_xet at module level since the worker imports it
        with patch.dict("sys.modules", {"hf_xet": MagicMock(
            PyXetDownloadInfo=mock_info_cls,
            download_files=mock_download,
        )}):
            _download_worker(self._make_params(), mp_queue, cancel_event)

        msg = mp_queue.get(timeout=2)
        assert msg.msg_type == MSG_RESULT
        assert msg.payload["status"] == "success"
        assert msg.payload["filename"] == "model.bin"

    @patch("hf_progress._xet_worker._deserialize_xet_file_data")
    @patch("huggingface_hub.utils._xet.refresh_xet_connection_info")
    def test_download_worker_credential_error(self, mock_refresh, mock_deserialize):
        """Worker emits error message when credential fetch fails."""
        mock_deserialize.return_value = MagicMock()
        mock_refresh.side_effect = ConnectionError("Auth failed")

        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()

        _download_worker(self._make_params(), mp_queue, cancel_event)

        msg = mp_queue.get(timeout=2)
        assert msg.msg_type == MSG_ERROR
        assert "credential" in msg.payload["message"].lower() or "Auth" in msg.payload["message"]

    def test_download_worker_import_error(self):
        """Worker emits error message when hf_xet is not importable."""
        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()

        with patch.dict("sys.modules", {"hf_xet": None}):
            _download_worker(self._make_params(), mp_queue, cancel_event)

        msg = mp_queue.get(timeout=2)
        assert msg.msg_type == MSG_ERROR
        assert msg.payload["error_type"] == "ImportError"


# ── Upload Worker Tests ──────────────────────────────────────────


class TestUploadFileWorker:
    """Test _upload_file_worker function."""

    def _make_params(self, **overrides):
        params = {
            "file_path": "/tmp/model.bin",
            "repo_id": "user/repo",
            "token": "hf_test",
            "repo_type": "model",
            "revision": None,
            "endpoint": None,
            "transfer_id": "test-456",
            "report_interval": 0.1,
        }
        params.update(overrides)
        return params

    def test_upload_file_worker_import_error(self):
        """Worker emits error message when hf_xet is not importable."""
        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()

        with patch.dict("sys.modules", {"hf_xet": None}):
            _upload_file_worker(self._make_params(), mp_queue, cancel_event)

        msg = mp_queue.get(timeout=2)
        assert msg.msg_type == MSG_ERROR
        assert msg.payload["error_type"] == "ImportError"


class TestUploadBytesWorker:
    """Test _upload_bytes_worker function."""

    def _make_params(self, **overrides):
        params = {
            "file_content": b"hello world",
            "filename": "test.txt",
            "repo_id": "user/repo",
            "token": "hf_test",
            "repo_type": "model",
            "revision": None,
            "endpoint": None,
            "transfer_id": "test-789",
            "report_interval": 0.1,
        }
        params.update(overrides)
        return params

    def test_upload_bytes_worker_import_error(self):
        """Worker emits error message when hf_xet is not importable."""
        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()

        with patch.dict("sys.modules", {"hf_xet": None}):
            _upload_bytes_worker(self._make_params(), mp_queue, cancel_event)

        msg = mp_queue.get(timeout=2)
        assert msg.msg_type == MSG_ERROR
        assert msg.payload["error_type"] == "ImportError"

    def test_upload_bytes_worker_no_content_no_path(self):
        """Worker emits error when neither file_content nor file_path is provided."""
        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()

        params = self._make_params()
        del params["file_content"]

        _upload_bytes_worker(params, mp_queue, cancel_event)

        msg = mp_queue.get(timeout=2)
        assert msg.msg_type == MSG_ERROR
        assert msg.payload["error_type"] == "ValueError"


# ── Module-Level Import Safety ───────────────────────────────────


class TestWorkerModuleSafety:
    """Test that _xet_worker doesn't import hf_xet at module level."""

    def test_worker_no_top_level_hf_xet_import(self):
        """Verify the module doesn't import hf_xet at load time."""
        import hf_progress._xet_worker as worker_module

        # hf_xet should not be in the module's namespace
        assert not hasattr(worker_module, "hf_xet")

    def test_worker_functions_are_picklable(self):
        """Worker functions must be picklable for multiprocessing spawn."""
        import pickle

        for func in (_download_worker, _download_batch_worker, _upload_file_worker, _upload_bytes_worker):
            # Should not raise
            pickled = pickle.dumps(func)
            unpickled = pickle.loads(pickled)
            assert unpickled is func
