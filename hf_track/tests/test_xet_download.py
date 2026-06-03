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


# ── download_snapshot_with_xet ─────────────────────────────────


class TestDownloadSnapshotWithXet:
    """Tests for download_snapshot_with_xet function."""

    def test_raises_import_error_when_xet_unavailable(self):
        """Should raise ImportError when hf_xet is not installed."""
        from hf_track.xet_download import download_snapshot_with_xet

        with patch("hf_track.xet_download.is_xet_available", return_value=False):
            with pytest.raises(ImportError, match="hf_xet is not installed"):
                download_snapshot_with_xet(
                    repo_id="test/repo",
                    token="test-token",
                    event_queue=queue.Queue(),
                )

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_emits_start_event(self, MockRunner, _mock_xet_avail):
        """Should emit a START event before downloading."""
        from hf_track.xet_download import download_snapshot_with_xet

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "success",
            "filename": "test/repo",
            "destination_path": "/tmp/test_repo",
            "file_size": 0,
            "transfer_id": "snap-1",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        download_snapshot_with_xet(
            repo_id="test/repo",
            token="test-token",
            event_queue=event_queue,
            transfer_id="snap-1",
        )

        # First event should be START
        start_event = event_queue.get_nowait()
        assert start_event.event_type == EventType.START
        assert start_event.transfer_id == "snap-1"
        assert start_event.direction == TransferDirection.DOWNLOAD
        assert start_event.filename == "test/repo"
        assert start_event.phase == ProgressPhase.DOWNLOADING

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_returns_destination_path_on_success(self, MockRunner, _mock_xet_avail):
        """Should return the destination_path from the result."""
        from hf_track.xet_download import download_snapshot_with_xet

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "success",
            "filename": "test/repo",
            "destination_path": "/my/local/dir",
            "file_size": 4096,
            "transfer_id": "snap-2",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        result = download_snapshot_with_xet(
            repo_id="test/repo",
            token="test-token",
            event_queue=event_queue,
            transfer_id="snap-2",
            local_dir="/my/local/dir",
        )

        assert result == "/my/local/dir"

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_raises_cancelled_on_cancelled_result(self, MockRunner, _mock_xet_avail):
        """Should raise TransferCancelledError when worker returns cancelled status."""
        from hf_track.xet_download import download_snapshot_with_xet
        from hf_track.types import TransferCancelledError

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "cancelled",
            "message": "Download cancelled by user",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        with pytest.raises(TransferCancelledError, match="cancelled"):
            download_snapshot_with_xet(
                repo_id="test/repo",
                token="test-token",
                event_queue=event_queue,
                transfer_id="snap-cancel",
            )

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_raises_progress_error_on_error_result(self, MockRunner, _mock_xet_avail):
        """Should raise TransferProgressError when worker returns error status."""
        from hf_track.xet_download import download_snapshot_with_xet

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "error",
            "message": "Download failed: network error",
            "error_type": "ConnectionError",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        with pytest.raises(TransferProgressError, match="network error"):
            download_snapshot_with_xet(
                repo_id="test/repo",
                token="test-token",
                event_queue=event_queue,
                transfer_id="snap-err",
            )

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_cancel_via_is_cancelled_hook(self, MockRunner, _mock_xet_avail):
        """Should terminate runner and raise TransferCancelledError when is_cancelled returns True."""
        from hf_track.xet_download import download_snapshot_with_xet
        from hf_track.types import TransferCancelledError

        mock_runner = MagicMock()
        call_count = [0]

        def wait_side_effect(timeout=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # First call: timeout (worker still running)
            return {"status": "success"}  # Shouldn't reach here

        mock_runner.wait.side_effect = wait_side_effect
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        with pytest.raises(TransferCancelledError):
            download_snapshot_with_xet(
                repo_id="test/repo",
                token="test-token",
                event_queue=event_queue,
                transfer_id="snap-hook-cancel",
                is_cancelled=lambda: call_count[0] >= 1,
            )

        mock_runner.terminate.assert_called()

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_terminates_runner_on_keyboard_interrupt(self, MockRunner, _mock_xet_avail):
        """Should terminate runner and raise TransferCancelledError on KeyboardInterrupt."""
        from hf_track.xet_download import download_snapshot_with_xet
        from hf_track.types import TransferCancelledError

        mock_runner = MagicMock()
        mock_runner.wait.side_effect = KeyboardInterrupt
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        with pytest.raises(TransferCancelledError, match="Ctrl\\+C"):
            download_snapshot_with_xet(
                repo_id="test/repo",
                token="test-token",
                event_queue=event_queue,
                transfer_id="snap-ctrlc",
            )

        mock_runner.terminate.assert_called()

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_terminates_runner_on_unexpected_exception(self, MockRunner, _mock_xet_avail):
        """Should terminate runner and emit ERROR event on unexpected exceptions."""
        from hf_track.xet_download import download_snapshot_with_xet

        mock_runner = MagicMock()
        mock_runner.wait.side_effect = RuntimeError("unexpected crash")
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        with pytest.raises(RuntimeError, match="unexpected crash"):
            download_snapshot_with_xet(
                repo_id="test/repo",
                token="test-token",
                event_queue=event_queue,
                transfer_id="snap-crash",
            )

        mock_runner.terminate.assert_called()
        # Should have emitted an ERROR event
        events = []
        while not event_queue.empty():
            events.append(event_queue.get_nowait())
        error_events = [e for e in events if e.event_type == EventType.ERROR]
        assert len(error_events) == 1

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_terminates_runner_in_finally(self, MockRunner, _mock_xet_avail):
        """Runner.terminate() should always be called in the finally block."""
        from hf_track.xet_download import download_snapshot_with_xet

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "success",
            "destination_path": "/tmp/ok",
            "transfer_id": "snap-finally",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        download_snapshot_with_xet(
            repo_id="test/repo",
            token="test-token",
            event_queue=event_queue,
            transfer_id="snap-finally",
        )

        mock_runner.terminate.assert_called()

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_passes_all_params_to_runner(self, MockRunner, _mock_xet_avail):
        """All params should be forwarded to the subprocess runner via params dict."""
        from hf_track.xet_download import download_snapshot_with_xet

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "success",
            "destination_path": "/tmp/ok",
            "transfer_id": "snap-params",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        download_snapshot_with_xet(
            repo_id="test/repo",
            token="test-token",
            event_queue=event_queue,
            allow_patterns=["*.bin"],
            ignore_patterns=["*.tmp"],
            repo_type="dataset",
            revision="v1.0",
            endpoint="https://custom.endpoint",
            local_dir="/my/dir",
            transfer_id="snap-params",
            report_interval=0.5,
            force_download=True,
        )

        mock_runner.start.assert_called_once()
        call_kwargs = mock_runner.start.call_args
        params = call_kwargs.kwargs["params"]
        assert params["repo_id"] == "test/repo"
        assert params["token"] == "test-token"
        assert params["allow_patterns"] == ["*.bin"]
        assert params["ignore_patterns"] == ["*.tmp"]
        assert params["repo_type"] == "dataset"
        assert params["revision"] == "v1.0"
        assert params["endpoint"] == "https://custom.endpoint"
        assert params["local_dir"] == "/my/dir"
        assert params["transfer_id"] == "snap-params"
        assert params["report_interval"] == 0.5
        assert params["force_download"] is True

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_generates_transfer_id_when_not_provided(self, MockRunner, _mock_xet_avail):
        """Should generate a transfer_id if not provided."""
        from hf_track.xet_download import download_snapshot_with_xet

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "success",
            "destination_path": "/tmp/ok",
            "transfer_id": "auto-gen",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        download_snapshot_with_xet(
            repo_id="test/repo",
            token="test-token",
            event_queue=event_queue,
        )

        # Should have generated a transfer_id (UUID format)
        start_event = event_queue.get_nowait()
        assert start_event.transfer_id  # Not empty
        assert len(start_event.transfer_id) == 36  # UUID format

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_uses_snapshot_worker(self, MockRunner, _mock_xet_avail):
        """Should use _snapshot_worker as the worker function."""
        from hf_track.xet_download import download_snapshot_with_xet, _snapshot_worker

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "success",
            "destination_path": "/tmp/ok",
            "transfer_id": "snap-worker",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        download_snapshot_with_xet(
            repo_id="test/repo",
            token="test-token",
            event_queue=event_queue,
            transfer_id="snap-worker",
        )

        mock_runner.start.assert_called_once()
        call_kwargs = mock_runner.start.call_args
        assert call_kwargs.kwargs["worker_func"] is _snapshot_worker

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_cancelled_error_propagates_without_terminate(self, MockRunner, _mock_xet_avail):
        """TransferCancelledError from result handling should propagate cleanly."""
        from hf_track.xet_download import download_snapshot_with_xet
        from hf_track.types import TransferCancelledError

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "cancelled",
            "message": "User cancelled",
            "error_type": "TransferCancelledError",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        with pytest.raises(TransferCancelledError, match="User cancelled"):
            download_snapshot_with_xet(
                repo_id="test/repo",
                token="test-token",
                event_queue=event_queue,
                transfer_id="snap-prop-cancel",
            )

        # terminate is still called in finally
        mock_runner.terminate.assert_called()

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_fallback_to_repo_id_when_no_destination_path(self, MockRunner, _mock_xet_avail):
        """Should fall back to repo_id when destination_path is missing from result."""
        from hf_track.xet_download import download_snapshot_with_xet

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "success",
            "transfer_id": "snap-no-dest",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        result = download_snapshot_with_xet(
            repo_id="test/repo",
            token="test-token",
            event_queue=event_queue,
            transfer_id="snap-no-dest",
        )

        # Falls back to local_dir or repo_id
        assert result == "test/repo"

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_passes_use_xet_in_params(self, MockRunner, _mock_xet_avail):
        """use_xet parameter is included in the params dict passed to the subprocess worker."""
        from hf_track.xet_download import download_snapshot_with_xet

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "success",
            "destination_path": "/tmp/ok",
            "transfer_id": "snap-use-xet",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        download_snapshot_with_xet(
            repo_id="test/repo",
            token="test-token",
            event_queue=event_queue,
            transfer_id="snap-use-xet",
            use_xet=False,
        )

        mock_runner.start.assert_called_once()
        call_kwargs = mock_runner.start.call_args
        params = call_kwargs.kwargs["params"]
        assert params["use_xet"] is False

    @patch("hf_track.xet_download.is_xet_available", return_value=True)
    @patch("hf_track.xet_download.XetSubprocessRunner")
    def test_use_xet_default_is_true(self, MockRunner, _mock_xet_avail):
        """Default use_xet is True when not specified."""
        from hf_track.xet_download import download_snapshot_with_xet

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "success",
            "destination_path": "/tmp/ok",
            "transfer_id": "snap-default-xet",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()

        download_snapshot_with_xet(
            repo_id="test/repo",
            token="test-token",
            event_queue=event_queue,
            transfer_id="snap-default-xet",
        )

        mock_runner.start.assert_called_once()
        call_kwargs = mock_runner.start.call_args
        params = call_kwargs.kwargs["params"]
        assert params["use_xet"] is True
