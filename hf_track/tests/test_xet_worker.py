"""Tests for _xet_worker module — worker functions and serialization helpers."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from hf_track._xet_worker import (
    _deserialize_xet_file_data,
    _download_batch_worker,
    _download_worker,
    _make_progress_callback,
    _open_unbuffered,
    _ProgressThrottler,
    _serialize_xet_file_data,
    _snapshot_worker,
    _upload_bytes_worker,
    _upload_file_worker,
    _xet_streaming_download_worker,
    DEFAULT_FSYNC_INTERVAL,
)
from hf_track.subprocess import MSG_CANCELLED, MSG_ERROR, MSG_EVENT, MSG_RESULT, SubprocessMessage


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


# ── Progress Throttler Tests ──────────────────────────────────────


class TestProgressThrottler:
    """Test _ProgressThrottler for shared throttling logic.

    This is the throttler used by all Xet worker progress callbacks.
    It ensures the first event is always emitted, subsequent events
    are throttled by time and byte delta, and the completion event
    (100%) is always emitted.
    """

    def test_first_event_always_emitted_even_with_no_change(self):
        """First call to should_emit returns True regardless of value."""
        t = _ProgressThrottler(report_interval=0.1)
        # Even with value 0 and no time elapsed
        assert t.should_emit(current_value=0, total=1000, now=0.0) is True
        # And with any value
        assert t.should_emit(current_value=500, total=1000, now=0.0) is True

    def test_completion_always_emitted(self):
        """Events at 100% bypass throttling."""
        t = _ProgressThrottler(report_interval=1.0)  # 1s throttle
        t.record_emit(500, now=0.0)
        # Right after, value goes to 100% — should emit
        assert t.should_emit(current_value=1000, total=1000, now=0.0) is True

    def test_time_throttle_blocks_repeated_emits(self):
        """Repeated emits within report_interval are throttled."""
        t = _ProgressThrottler(report_interval=0.5)
        t.record_emit(100, now=0.0)
        # Same time, no change
        assert t.should_emit(100, 1000, now=0.0) is False
        # Slightly later, still within interval
        assert t.should_emit(100, 1000, now=0.1) is False
        # After interval elapsed
        assert t.should_emit(100, 1000, now=0.6) is True

    def test_byte_delta_throttle_blocks_small_changes(self):
        """Tiny byte deltas are throttled when time hasn't elapsed."""
        t = _ProgressThrottler(report_interval=10.0)  # long interval
        t.record_emit(100, now=0.0)
        # Small change (less than 1% of total = 10 bytes, but min 1024)
        # so any change under 1024 bytes is throttled
        assert t.should_emit(100 + 100, 1000, now=0.0) is False
        # Large change (>= 1024 bytes) — emitted (byte delta bypasses time throttle)
        assert t.should_emit(100 + 2000, 1000, now=0.0) is True

    def test_min_bytes_delta_is_1_percent_of_total(self):
        """Min bytes delta is 1% of total (when total > 102400)."""
        t = _ProgressThrottler(report_interval=10.0)  # long interval
        t.record_emit(0, now=0.0)
        # Total = 100_000_000, 1% = 1_000_000
        # Change of 500_000 < 1_000_000 — throttled (within time, byte delta too small)
        assert t.should_emit(500_000, 100_000_000, now=0.0) is False
        # Change of 2_000_000 >= 1_000_000 — emitted (byte delta bypasses)
        assert t.should_emit(2_000_000, 100_000_000, now=0.0) is True

    def test_record_emit_updates_state(self):
        """record_emit properly updates internal state for next call."""
        t = _ProgressThrottler(report_interval=0.5)
        t.record_emit(1000, now=10.0)
        # Immediately after, with same value, should be throttled
        assert t.should_emit(1000, 10000, now=10.0) is False
        # After interval
        assert t.should_emit(1000, 10000, now=10.5) is True

    def test_reset_clears_state(self):
        """reset() allows the throttler to emit first-event again."""
        t = _ProgressThrottler(report_interval=0.5)
        t.record_emit(500, now=10.0)
        # Throttled
        assert t.should_emit(500, 1000, now=10.0) is False
        # Reset
        t.reset()
        # First event again
        assert t.should_emit(500, 1000, now=10.0) is True

    def test_total_zero_uses_1kib_min_delta(self):
        """When total is 0, the min bytes delta is 1024."""
        t = _ProgressThrottler(report_interval=10.0)  # long interval
        t.record_emit(0, now=0.0)
        # Change of 100 bytes — throttled (min_delta = 1024)
        assert t.should_emit(100, 0, now=0.0) is False
        # Change of 2000 bytes — emitted
        assert t.should_emit(2000, 0, now=0.0) is True


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

    @patch("hf_track._xet_worker._deserialize_xet_file_data")
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

    @patch("hf_track._xet_worker._deserialize_xet_file_data")
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
        test_ctx = mp.get_context("spawn")
        test_mp_queue = test_ctx.Queue()
        test_cancel_event = test_ctx.Event()

        test_params = self._make_params()
        del test_params["file_content"]

        _upload_bytes_worker(test_params, test_mp_queue, test_cancel_event)

        test_msg = test_mp_queue.get(timeout=2)
        assert test_msg.msg_type == MSG_ERROR
        assert test_msg.payload["error_type"] == "ValueError"


# ── Snapshot Worker Tests ────────────────────────────────────────


class TestSnapshotWorker:
    """Test _snapshot_worker function."""

    def _make_params(self, **overrides):
        params = {
            "repo_id": "test/repo",
            "token": "hf_test",
            "repo_type": "model",
            "revision": None,
            "local_dir": "/tmp/test_repo",
            "allow_patterns": None,
            "ignore_patterns": None,
            "endpoint": None,
            "transfer_id": "snap-test-001",
            "report_interval": 0.1,
            "force_download": False,
        }
        params.update(overrides)
        return params

    def _drain_messages(self, mp_queue):
        """Drain all messages from mp_queue and return as a list."""
        messages = []
        while True:
            try:
                messages.append(mp_queue.get_nowait())
            except Exception:
                break
        return messages

    def _find_message(self, messages, msg_type):
        """Find the first message of a given type in a list."""
        for msg in messages:
            if msg.msg_type == msg_type:
                return msg
        return None

    @patch("huggingface_hub.snapshot_download")
    def test_snapshot_worker_success(self, mock_snapshot_dl):
        """Worker sends COMPLETE event and result message on successful snapshot_download."""
        mock_snapshot_dl.return_value = "/tmp/test_repo"

        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()

        _snapshot_worker(self._make_params(), mp_queue, cancel_event)

        messages = self._drain_messages(mp_queue)
        # Should have a COMPLETE event followed by a result message
        complete_msg = self._find_message(messages, MSG_EVENT)
        assert complete_msg is not None
        assert complete_msg.payload["event_type"] == "complete"

        result_msg = self._find_message(messages, MSG_RESULT)
        assert result_msg is not None
        assert result_msg.payload["status"] == "success"
        assert result_msg.payload["destination_path"] == "/tmp/test_repo"
        assert result_msg.payload["transfer_id"] == "snap-test-001"

    @patch("huggingface_hub.snapshot_download")
    def test_snapshot_worker_passes_kwargs(self, mock_snapshot_dl):
        """Worker passes all kwargs to snapshot_download."""
        mock_snapshot_dl.return_value = "/tmp/test_repo"

        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()

        params = self._make_params(
            repo_type="dataset",
            revision="v1.0",
            allow_patterns=["*.bin"],
            ignore_patterns=["*.tmp"],
            force_download=True,
            local_dir="/my/dir",
        )
        _snapshot_worker(params, mp_queue, cancel_event)

        # Drain all messages — should include a result
        messages = self._drain_messages(mp_queue)
        result_msg = self._find_message(messages, MSG_RESULT)
        assert result_msg is not None

        # Verify snapshot_download was called with correct kwargs
        mock_snapshot_dl.assert_called_once()
        call_kwargs = mock_snapshot_dl.call_args.kwargs
        assert call_kwargs["repo_id"] == "test/repo"
        assert call_kwargs["repo_type"] == "dataset"
        assert call_kwargs["revision"] == "v1.0"
        assert call_kwargs["allow_patterns"] == ["*.bin"]
        assert call_kwargs["ignore_patterns"] == ["*.tmp"]
        assert call_kwargs["force_download"] is True
        assert call_kwargs["local_dir"] == "/my/dir"
        assert call_kwargs["token"] == "hf_test"

    @patch("huggingface_hub.snapshot_download")
    def test_snapshot_worker_passes_tqdm_class(self, mock_snapshot_dl):
        """Worker passes a tqdm_class subclass to snapshot_download."""
        mock_snapshot_dl.return_value = "/tmp/test_repo"

        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()

        _snapshot_worker(self._make_params(), mp_queue, cancel_event)

        # Drain all messages — should include a result
        messages = self._drain_messages(mp_queue)
        result_msg = self._find_message(messages, MSG_RESULT)
        assert result_msg is not None

        # Verify tqdm_class was passed
        call_kwargs = mock_snapshot_dl.call_args.kwargs
        assert "tqdm_class" in call_kwargs
        tqdm_cls = call_kwargs["tqdm_class"]
        # Should be a subclass of DownloadProgressTqdm
        from hf_track.callbacks import DownloadProgressTqdm
        assert issubclass(tqdm_cls, DownloadProgressTqdm)

    @patch("huggingface_hub.snapshot_download")
    def test_snapshot_worker_handles_exception(self, mock_snapshot_dl):
        """Worker sends error message when snapshot_download raises."""
        mock_snapshot_dl.side_effect = RuntimeError("network error")

        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()

        _snapshot_worker(self._make_params(), mp_queue, cancel_event)

        msg = mp_queue.get(timeout=5)
        assert msg.msg_type == MSG_ERROR
        assert "network error" in msg.payload["message"]
        assert msg.payload["error_type"] == "RuntimeError"

    @patch("huggingface_hub.snapshot_download")
    def test_snapshot_worker_handles_import_error(self, mock_snapshot_dl):
        """Worker sends error message when snapshot_download raises ImportError."""
        mock_snapshot_dl.side_effect = ImportError("no module")

        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()

        _snapshot_worker(self._make_params(), mp_queue, cancel_event)

        msg = mp_queue.get(timeout=5)
        assert msg.msg_type == MSG_ERROR
        assert msg.payload["error_type"] == "ImportError"

    @patch("huggingface_hub.snapshot_download")
    def test_snapshot_worker_handles_keyboard_interrupt(self, mock_snapshot_dl):
        """Worker sends cancelled message on KeyboardInterrupt."""
        mock_snapshot_dl.side_effect = KeyboardInterrupt()

        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()

        _snapshot_worker(self._make_params(), mp_queue, cancel_event)

        msg = mp_queue.get(timeout=5)
        # _handle_worker_exception sends MSG_CANCELLED for KeyboardInterrupt
        assert msg.msg_type == MSG_CANCELLED

    @patch("huggingface_hub.snapshot_download")
    def test_snapshot_worker_omits_local_dir_when_none(self, mock_snapshot_dl):
        """Worker does not pass local_dir to snapshot_download when it is None."""
        mock_snapshot_dl.return_value = "/tmp/cache/test_repo"

        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()

        params = self._make_params(local_dir=None)
        _snapshot_worker(params, mp_queue, cancel_event)

        # Drain all messages — should include a result
        messages = self._drain_messages(mp_queue)
        result_msg = self._find_message(messages, MSG_RESULT)
        assert result_msg is not None

        # local_dir should NOT be in the kwargs
        call_kwargs = mock_snapshot_dl.call_args.kwargs
        assert "local_dir" not in call_kwargs

    @patch("huggingface_hub.snapshot_download")
    def test_snapshot_worker_emits_complete_event(self, mock_snapshot_dl):
        """Worker sends COMPLETE event before result message."""
        mock_snapshot_dl.return_value = "/tmp/test_repo"

        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()

        _snapshot_worker(self._make_params(), mp_queue, cancel_event)

        # Collect all messages
        messages = self._drain_messages(mp_queue)

        # Should have both a COMPLETE event and a result message
        msg_types = [m.msg_type for m in messages]
        assert MSG_EVENT in msg_types
        assert MSG_RESULT in msg_types

        # Find the COMPLETE event
        complete_events = [
            m for m in messages
            if m.msg_type == MSG_EVENT and m.payload.get("event_type") == "complete"
        ]
        assert len(complete_events) == 1
        complete_payload = complete_events[0].payload
        assert complete_payload["transfer_id"] == "snap-test-001"
        assert complete_payload["filename"] == "test/repo"
        assert complete_payload["percentage"] == 100.0

    @patch("huggingface_hub.snapshot_download")
    def test_snapshot_worker_result_includes_snapshot_stats(self, mock_snapshot_dl):
        """Worker result payload includes bytes_completed, total_bytes, files_completed, total_files."""
        mock_snapshot_dl.return_value = "/tmp/test_repo"

        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()

        _snapshot_worker(self._make_params(), mp_queue, cancel_event)

        messages = self._drain_messages(mp_queue)
        result_msg = self._find_message(messages, MSG_RESULT)
        assert result_msg is not None
        # Result should include snapshot-level stats
        assert "bytes_completed" in result_msg.payload
        assert "total_bytes" in result_msg.payload
        assert "files_completed" in result_msg.payload
        assert "total_files" in result_msg.payload

    @patch("huggingface_hub.snapshot_download")
    def test_snapshot_worker_respects_cancel_event(self, mock_snapshot_dl):
        """Worker's _is_cancelled returns True when cancel_event is set."""
        mock_snapshot_dl.return_value = "/tmp/test_repo"

        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()
        cancel_event.set() # Pre-set the cancel event

        _snapshot_worker(self._make_params(), mp_queue, cancel_event)

        # Worker should still complete (snapshot_download doesn't check cancel_event
        # directly — it's checked by the is_cancelled hook in the tqdm class)
        messages = self._drain_messages(mp_queue)
        result_msg = self._find_message(messages, MSG_RESULT)
        assert result_msg is not None


# ── Module-Level Import Safety ───────────────────────────────────


class TestWorkerModuleSafety:
    """Test that _xet_worker doesn't import hf_xet at module level."""

    def test_worker_no_top_level_hf_xet_import(self):
        """Verify the module doesn't import hf_xet at load time."""
        import hf_track._xet_worker as worker_module

        # hf_xet should not be in the module's namespace
        assert not hasattr(worker_module, "hf_xet")

    def test_worker_functions_are_picklable(self):
        """Worker functions must be picklable for multiprocessing spawn."""
        import pickle

        for func in (_download_worker, _download_batch_worker, _snapshot_worker, _upload_file_worker, _upload_bytes_worker):
            # Should not raise
            pickled = pickle.dumps(func)
            unpickled = pickle.loads(pickled)
            assert unpickled is func


class TestSnapshotWorkerUseXet:
    """Tests for _snapshot_worker use_xet parameter and env var handling.

    The _snapshot_worker sets HF_HUB_DISABLE_XET in the child process
    BEFORE importing huggingface_hub, so the cached constant reflects
    the correct value. This is critical for runtime xet toggling.
    """

    def _make_params(self, **overrides):
        params = {
            "repo_id": "test/repo",
            "token": "hf_test",
            "repo_type": "model",
            "revision": None,
            "local_dir": "/tmp/test_repo",
            "allow_patterns": None,
            "ignore_patterns": None,
            "endpoint": None,
            "transfer_id": "snap-use-xet-001",
            "report_interval": 0.1,
            "force_download": False,
            "use_xet": True,
        }
        params.update(overrides)
        return params

    @patch("huggingface_hub.snapshot_download")
    def test_use_xet_false_sets_env_var(self, mock_snapshot_dl):
        """use_xet=False → worker sets HF_HUB_DISABLE_XET=1 before
        huggingface_hub imports occur."""
        mock_snapshot_dl.return_value = "/tmp/test_repo"
        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()
        params = self._make_params(use_xet=False)

        # Clean env before test
        old_val = os.environ.pop("HF_HUB_DISABLE_XET", None)
        try:
            _snapshot_worker(params, mp_queue, cancel_event)

            # After the worker runs, the env var should be set
            assert os.environ.get("HF_HUB_DISABLE_XET") == "1"
        finally:
            # Restore env
            if old_val is not None:
                os.environ["HF_HUB_DISABLE_XET"] = old_val
            else:
                os.environ.pop("HF_HUB_DISABLE_XET", None)

    @patch("huggingface_hub.snapshot_download")
    def test_use_xet_true_clears_env_var(self, mock_snapshot_dl):
        """use_xet=True → worker clears HF_HUB_DISABLE_XET before
        huggingface_hub imports occur."""
        mock_snapshot_dl.return_value = "/tmp/test_repo"
        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()
        params = self._make_params(use_xet=True)

        # Set env var before test to verify it gets cleared
        old_val = os.environ.get("HF_HUB_DISABLE_XET")
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        try:
            _snapshot_worker(params, mp_queue, cancel_event)

            # After the worker runs, the env var should be cleared
            assert "HF_HUB_DISABLE_XET" not in os.environ
        finally:
            # Restore env
            if old_val is not None:
                os.environ["HF_HUB_DISABLE_XET"] = old_val
            else:
                os.environ.pop("HF_HUB_DISABLE_XET", None)

    @patch("huggingface_hub.snapshot_download")
    def test_use_xet_default_is_true(self, mock_snapshot_dl):
        """Default use_xet (not in params) is True."""
        mock_snapshot_dl.return_value = "/tmp/test_repo"
        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()
        params = self._make_params()
        del params["use_xet"]  # Remove to test default

        old_val = os.environ.get("HF_HUB_DISABLE_XET")
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        try:
            _snapshot_worker(params, mp_queue, cancel_event)

            # Default is True → env var should be cleared
            assert "HF_HUB_DISABLE_XET" not in os.environ
        finally:
            if old_val is not None:
                os.environ["HF_HUB_DISABLE_XET"] = old_val
            else:
                os.environ.pop("HF_HUB_DISABLE_XET", None)


# ── Streaming Download Worker Tests ───────────────────────────────


class TestXetStreamingDownloadWorker:
    """Test _xet_streaming_download_worker — chunk-by-chunk file writer.

    This worker uses ``XetSession().new_download_stream_group().download_stream()``
    and writes each chunk directly to disk via ``os.write`` (no userspace buffer).
    Cancellation is checked before every chunk write.
    """

    def _make_params(self, tmp_path, file_size=1000, **overrides):
        """Build default params dict for the worker (multi-file shape)."""
        params = {
            "file_specs": [{
                "hash": "abc123",
                "file_size": file_size,
                "dest_path": str(tmp_path / "model.bin"),
                "xet_file_data": {"file_hash": "abc123", "refresh_route": "https://xet.example.com/refresh"},
            }],
            "token": "hf_test",
            "endpoint": None,
            "transfer_id": "stream-test-001",
            "report_interval": 0.05,
            "fsync_interval": 1024,
            "request_headers": {},
        }
        params.update(overrides)
        return params

    def _drain_messages(self, mp_queue):
        """Drain all messages from queue and return as a list."""
        messages = []
        while True:
            try:
                messages.append(mp_queue.get_nowait())
            except Exception:
                break
        return messages

    def test_open_unbuffered_creates_file(self, tmp_path):
        """_open_unbuffered creates a new file."""
        path = tmp_path / "x.bin"
        fd = _open_unbuffered(str(path))
        try:
            os.write(fd, b"hello")
        finally:
            os.close(fd)
        assert path.read_bytes() == b"hello"

    def test_open_unbuffered_truncates_existing(self, tmp_path):
        """_open_unbuffered truncates an existing file on open."""
        path = tmp_path / "x.bin"
        path.write_bytes(b"old content here")
        fd = _open_unbuffered(str(path))
        try:
            os.write(fd, b"new")
        finally:
            os.close(fd)
        assert path.read_bytes() == b"new"

    def test_open_unbuffered_returns_int(self, tmp_path):
        """_open_unbuffered returns a positive integer fd."""
        fd = _open_unbuffered(str(tmp_path / "x.bin"))
        try:
            assert isinstance(fd, int)
            assert fd > 0
        finally:
            os.close(fd)

    def test_default_fsync_interval_is_4mb(self):
        """DEFAULT_FSYNC_INTERVAL is 4 MB."""
        assert DEFAULT_FSYNC_INTERVAL == 4 * 1024 * 1024

    def test_import_error_emits_error_message(self, tmp_path):
        """Worker emits an error message when hf_xet is not importable."""
        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()
        params = self._make_params(tmp_path)

        with patch.dict("sys.modules", {"hf_xet": None}):
            _xet_streaming_download_worker(params, mp_queue, cancel_event)

        messages = self._drain_messages(mp_queue)
        errors = [m for m in messages if m.msg_type == MSG_ERROR]
        assert len(errors) == 1
        assert errors[0].payload["error_type"] == "ImportError"

    def test_emits_progress_event_per_chunk(self, tmp_path):
        """Worker emits PROGRESS events as chunks are written."""
        chunks = [b"A" * 100, b"B" * 200, b"C" * 300, b"D" * 400]
        full_size = sum(len(c) for c in chunks)  # 1000

        # Build a mock hf_xet module that produces a stream of chunks
        mock_xet_module = MagicMock()
        mock_session = MagicMock()
        mock_group = MagicMock()
        mock_stream = MagicMock()
        _iter = iter(chunks)
        mock_stream.__iter__ = lambda self: _iter
        mock_stream.__next__ = lambda self: next(_iter)

        mock_xet_module.XetSession.return_value = mock_session
        mock_session.new_download_stream_group.return_value = mock_group
        mock_group.download_stream.return_value = mock_stream

        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()
        params = self._make_params(
            tmp_path, file_size=full_size, report_interval=0.0,
        )

        with patch.dict("sys.modules", {
            "hf_xet": mock_xet_module,
            "huggingface_hub.utils._xet": MagicMock(
                refresh_xet_connection_info=MagicMock(return_value=MagicMock(
                    endpoint="https://xet.example.com",
                    access_token="tok",
                    expiration_unix_epoch=9999999999,
                )),
            ),
        }):
            _xet_streaming_download_worker(params, mp_queue, cancel_event)

        messages = self._drain_messages(mp_queue)
        events = [m for m in messages if m.msg_type == MSG_EVENT and m.payload.get("event_type") == "progress"]
        # 4 chunks, 4 events (no throttling since report_interval=0)
        assert len(events) == 4, f"Expected 4 progress events, got {len(events)}"
        # bytes_completed should be cumulative
        bytes_vals = [e.payload["bytes_completed"] for e in events]
        assert bytes_vals == [100, 300, 600, 1000]

    def test_emits_complete_and_result_at_end(self, tmp_path):
        """Worker emits a COMPLETE event and a RESULT message on success."""
        chunks = [b"x" * 250, b"x" * 250, b"x" * 250, b"x" * 250]  # 1000 bytes

        mock_xet_module = MagicMock()
        mock_stream = MagicMock()
        _iter = iter(chunks)
        mock_stream.__iter__ = lambda self: _iter
        mock_stream.__next__ = lambda self: next(_iter)
        mock_xet_module.XetSession.return_value.new_download_stream_group.return_value.download_stream.return_value = mock_stream

        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()
        params = self._make_params(tmp_path)

        with patch.dict("sys.modules", {
            "hf_xet": mock_xet_module,
            "huggingface_hub.utils._xet": MagicMock(
                refresh_xet_connection_info=MagicMock(return_value=MagicMock(
                    endpoint="https://xet.example.com",
                    access_token="tok",
                    expiration_unix_epoch=9999999999,
                )),
            ),
        }):
            _xet_streaming_download_worker(params, mp_queue, cancel_event)

        messages = self._drain_messages(mp_queue)
        complete = [m for m in messages if m.msg_type == MSG_EVENT and m.payload.get("event_type") == "complete"]
        result = [m for m in messages if m.msg_type == MSG_RESULT]
        assert len(complete) == 1
        assert complete[0].payload["percentage"] == 100.0
        assert len(result) == 1
        assert result[0].payload["status"] == "success"
        assert result[0].payload["filename"] == "model.bin"
        assert result[0].payload["file_size"] == 1000

    def test_writes_full_content_to_disk(self, tmp_path):
        """Worker writes all chunks to the destination file."""
        chunks = [bytes([i]) * 100 for i in range(10)]  # 1000 bytes total
        expected = b"".join(chunks)

        mock_xet_module = MagicMock()
        mock_stream = MagicMock()
        _iter = iter(chunks)
        mock_stream.__iter__ = lambda self: _iter
        mock_stream.__next__ = lambda self: next(_iter)
        mock_xet_module.XetSession.return_value.new_download_stream_group.return_value.download_stream.return_value = mock_stream

        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()
        dest = str(tmp_path / "out.bin")
        params = self._make_params(tmp_path, dest_path=dest, file_size=1000)
        params["file_specs"][0]["dest_path"] = dest

        with patch.dict("sys.modules", {
            "hf_xet": mock_xet_module,
            "huggingface_hub.utils._xet": MagicMock(
                refresh_xet_connection_info=MagicMock(return_value=MagicMock(
                    endpoint="https://xet.example.com",
                    access_token="tok",
                    expiration_unix_epoch=9999999999,
                )),
            ),
        }):
            _xet_streaming_download_worker(params, mp_queue, cancel_event)

        with open(dest, "rb") as f:
            data = f.read()
        assert data == expected
        assert len(data) == 1000

    def test_cancel_event_stops_loop(self, tmp_path):
        """Setting cancel_event makes the worker exit cooperatively."""
        # 5 chunks of 100 bytes; cancel after the first chunk
        chunks = [b"a" * 100] * 5

        mock_xet_module = MagicMock()
        mock_stream = MagicMock()

        # Make the iterator raise a fake cancel signal after yielding one chunk
        def iter_then_cancel(self):
            yield chunks[0]
            # After the first chunk, the worker's loop checks cancel_event and breaks

        _gen = iter_then_cancel(None)
        mock_stream.__iter__ = lambda self: _gen
        mock_stream.__next__ = lambda self: next(_gen)

        mock_xet_module.XetSession.return_value.new_download_stream_group.return_value.download_stream.return_value = mock_stream

        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()
        cancel_event.set()  # Pre-cancel
        dest = str(tmp_path / "cancelled.bin")
        params = self._make_params(tmp_path, dest_path=dest, file_size=500)
        params["file_specs"][0]["dest_path"] = dest

        with patch.dict("sys.modules", {
            "hf_xet": mock_xet_module,
            "huggingface_hub.utils._xet": MagicMock(
                refresh_xet_connection_info=MagicMock(return_value=MagicMock(
                    endpoint="https://xet.example.com",
                    access_token="tok",
                    expiration_unix_epoch=9999999999,
                )),
            ),
        }):
            _xet_streaming_download_worker(params, mp_queue, cancel_event)

        # We should see a CANCELLED message
        messages = self._drain_messages(mp_queue)
        cancelled = [m for m in messages if m.msg_type == MSG_CANCELLED]
        assert len(cancelled) == 1

        # The file may be empty or contain at most 1 chunk (since cancel is checked BEFORE write)
        if os.path.exists(dest):
            size = os.path.getsize(dest)
            assert size <= 100, f"Expected ≤100 bytes (cancelled), got {size}"

    def test_fsync_called_at_interval(self, tmp_path):
        """Worker calls os.fsync at the configured fsync_interval boundary."""
        # 10 chunks of 200 bytes = 2000 bytes total; fsync_interval=500 → expect ≥3 fsyncs
        chunks = [b"X" * 200 for _ in range(10)]

        mock_xet_module = MagicMock()
        mock_stream = MagicMock()
        _iter = iter(chunks)
        mock_stream.__iter__ = lambda self: _iter
        mock_stream.__next__ = lambda self: next(_iter)
        mock_xet_module.XetSession.return_value.new_download_stream_group.return_value.download_stream.return_value = mock_stream

        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()
        dest = str(tmp_path / "fsync.bin")
        params = self._make_params(tmp_path, dest_path=dest, file_size=2000, fsync_interval=500)
        params["file_specs"][0]["dest_path"] = dest

        fsync_call_count = [0]
        real_fsync = os.fsync

        def tracking_fsync(fd):
            fsync_call_count[0] += 1
            return real_fsync(fd)

        with patch.dict("sys.modules", {
            "hf_xet": mock_xet_module,
            "huggingface_hub.utils._xet": MagicMock(
                refresh_xet_connection_info=MagicMock(return_value=MagicMock(
                    endpoint="https://xet.example.com",
                    access_token="tok",
                    expiration_unix_epoch=9999999999,
                )),
            ),
        }), patch("hf_track._xet_worker.os.fsync", side_effect=tracking_fsync):
            _xet_streaming_download_worker(params, mp_queue, cancel_event)

        # 2000 bytes / 500 = 4 fsync intervals. With the final fsync in finally,
        # we expect at least 4 fsync calls.
        assert fsync_call_count[0] >= 3, f"Expected ≥3 fsync calls, got {fsync_call_count[0]}"

    def test_credential_error_emits_error(self, tmp_path):
        """Worker emits an error if credentials cannot be fetched."""
        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()
        params = self._make_params(tmp_path)

        with patch.dict("sys.modules", {
            "hf_xet": MagicMock(),
            "huggingface_hub.utils._xet": MagicMock(
                refresh_xet_connection_info=MagicMock(side_effect=ConnectionError("auth failed")),
            ),
        }):
            _xet_streaming_download_worker(params, mp_queue, cancel_event)

        messages = self._drain_messages(mp_queue)
        errors = [m for m in messages if m.msg_type == MSG_ERROR]
        assert len(errors) == 1
        assert "credential" in errors[0].payload["message"].lower() or "auth" in errors[0].payload["message"].lower()

    def test_empty_file_is_handled(self, tmp_path):
        """Worker creates an empty file and emits COMPLETE for zero-byte files."""
        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()
        dest = str(tmp_path / "empty.bin")
        params = self._make_params(tmp_path, file_size=0)
        params["file_specs"][0]["dest_path"] = dest
        params["file_specs"][0]["file_size"] = 0

        # No need to mock hf_xet since empty files short-circuit before opening a stream
        with patch.dict("sys.modules", {
            "hf_xet": MagicMock(),
            "huggingface_hub.utils._xet": MagicMock(
                refresh_xet_connection_info=MagicMock(return_value=MagicMock(
                    endpoint="https://xet.example.com",
                    access_token="tok",
                    expiration_unix_epoch=9999999999,
                )),
            ),
        }):
            _xet_streaming_download_worker(params, mp_queue, cancel_event)

        # Empty file should exist on disk
        assert os.path.exists(dest)
        assert os.path.getsize(dest) == 0

        # COMPLETE event and RESULT message should be emitted
        messages = self._drain_messages(mp_queue)
        complete = [m for m in messages if m.msg_type == MSG_EVENT and m.payload.get("event_type") == "complete"]
        result = [m for m in messages if m.msg_type == MSG_RESULT]
        assert len(complete) == 1
        assert len(result) == 1
        assert result[0].payload["status"] == "success"
        assert result[0].payload["file_size"] == 0

    def test_multi_file_emits_one_result(self, tmp_path):
        """Regression: multi-file download emits exactly ONE terminal result, not one per file.

        The relay thread in ``XetSubprocessRunner._relay_events`` treats
        ``SubprocessMessage.result`` as the terminal signal and breaks out of
        the loop on the first one. Previously, the streaming worker emitted a
        per-file ``result`` for every file in the loop, causing the relay to
        break after file 1 and lose all remaining progress events and the
        final result. Symptom: 0 files / 0 bytes on the parent side even
        though the worker was still running.

        This test verifies the fix: only ONE ``result`` is emitted at the
        very end of the loop, and one COMPLETE per file. Three xet files
        in the input produce three COMPLETEs and exactly one RESULT.
        """
        file_sizes = [100, 200, 300]
        # Each spec gets a unique hash → triggers per-file group rebuild
        specs = []
        for i, sz in enumerate(file_sizes):
            specs.append({
                "hash": f"hash_{i}",
                "file_size": sz,
                "dest_path": str(tmp_path / f"file_{i}.bin"),
                "xet_file_data": {
                    "file_hash": f"hash_{i}",
                    "refresh_route": f"https://xet.example.com/refresh/{i}",
                },
            })

        # We don't know the order in which ``new_download_stream_group`` will
        # be called (depends on which file the worker iterates first), so we
        # bind each call to a size determined at call time. We do this by
        # capturing the file_info passed to download_stream via a side_effect
        # that introspects the spec's hash via a lookup table.
        hash_to_size = {f"hash_{i}": sz for i, sz in enumerate(file_sizes)}

        # Mock hf_xet module. The XetFileInfo returned by ``hf_xet.XetFileInfo(...)``
        # is a MagicMock with arbitrary attribute access — but we can use a
        # namedtuple-like real class to expose file_hash / file_size attributes
        # that the side_effect can read.
        class _FakeFileInfo:
            def __init__(self, file_hash, file_size):
                self.file_hash = file_hash
                self.file_size = file_size

        def fake_xet_file_info(file_hash, file_size):
            return _FakeFileInfo(file_hash, file_size)

        def make_stream(size):
            stream = MagicMock()
            _iter = iter([b"x" * size])
            stream.__iter__ = lambda self: _iter
            stream.__next__ = lambda self: next(_iter)
            return stream

        # The worker re-creates the group each loop iteration via
        # ``new_download_stream_group``, so each call must produce a fresh
        # mock whose ``download_stream`` returns the stream for the
        # currently-iterated file.
        def new_group(**kwargs):
            g = MagicMock()
            g.download_stream = MagicMock(
                side_effect=lambda fi: make_stream(fi.file_size),
            )
            return g

        mock_xet_module = MagicMock()
        mock_xet_module.XetFileInfo = fake_xet_file_info
        mock_xet_module.XetSession.return_value.new_download_stream_group.side_effect = new_group

        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        cancel_event = ctx.Event()
        params = self._make_params(tmp_path)
        params["file_specs"] = specs
        params["report_interval"] = 0.0  # no throttling

        with patch.dict("sys.modules", {
            "hf_xet": mock_xet_module,
            "huggingface_hub.utils._xet": MagicMock(
                refresh_xet_connection_info=MagicMock(return_value=MagicMock(
                    endpoint="https://xet.example.com",
                    access_token="tok",
                    expiration_unix_epoch=9999999999,
                )),
            ),
        }):
            _xet_streaming_download_worker(params, mp_queue, cancel_event)

        messages = self._drain_messages(mp_queue)
        complete_events = [
            m for m in messages
            if m.msg_type == MSG_EVENT and m.payload.get("event_type") == "complete"
        ]
        results = [m for m in messages if m.msg_type == MSG_RESULT]

        # ONE COMPLETE per file (3 files → 3 COMPLETEs)
        assert len(complete_events) == 3, (
            f"Expected 3 COMPLETE events (one per file), got {len(complete_events)}. "
            f"Messages: {[(m.msg_type, m.payload) for m in messages]}"
        )
        # EXACTLY ONE terminal RESULT — the bug was emitting one per file
        assert len(results) == 1, (
            f"Expected exactly 1 terminal RESULT message, got {len(results)}. "
            f"Multiple result messages break the relay thread's terminal-state "
            f"detection in XetSubprocessRunner._relay_events. "
            f"Messages: {[(m.msg_type, m.payload) for m in messages]}"
        )
        # The single result should aggregate the whole transfer
        result = results[0]
        assert result.payload["status"] == "success"
        # All three files should exist on disk with correct sizes
        for i, sz in enumerate(file_sizes):
            p = tmp_path / f"file_{i}.bin"
            assert p.exists(), f"file_{i}.bin missing"
            assert p.stat().st_size == sz, f"file_{i}.bin wrong size"
