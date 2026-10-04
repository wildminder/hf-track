"""Tests for hf_progress.xet_download module — subprocess-isolated version."""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from hf_track.types import EventType, ProgressPhase, TransferDirection, TransferErrorInfo, TransferProgressError


# ── download_file_with_xet ────────────────────────────────────


class TestDownloadFileWithXet:
    """Tests for download_file_with_xet function."""

    def test_raises_import_error_when_xet_unavailable(self):
        """Should raise ImportError when hf_xet is not installed."""
        from hf_track.download import download_file_with_xet

        with patch("hf_track.download.xet_file.is_xet_available", return_value=False):
            with pytest.raises(ImportError, match="hf_xet is not installed"):
                download_file_with_xet(
                    file_hash="abc123",
                    file_size=1024,
                    dest_path="/tmp/test.bin",
                    xet_file_data=MagicMock(),
                    token="test-token",
                    event_queue=queue.Queue(),
                )

    @patch("hf_track.download.xet_file.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_file.XetSubprocessRunner")
    def test_emits_start_event(self, MockRunner, _mock_xet_avail):
        """Should emit a START event before downloading."""
        from hf_track.download import download_file_with_xet

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

    @patch("hf_track.download.xet_file.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_file.XetSubprocessRunner")
    def test_emits_complete_event_on_success(self, MockRunner, _mock_xet_avail):
        """Should emit a COMPLETE event after successful download."""
        from hf_track.download import download_file_with_xet

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

    @patch("hf_track.download.xet_file.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_file.XetSubprocessRunner")
    def test_emits_error_event_on_failure(self, MockRunner, _mock_xet_avail):
        """Should raise RuntimeError with TransferErrorInfo message when worker returns error."""
        from hf_track.download import download_file_with_xet

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

    @patch("hf_track.download.xet_file.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_file.XetSubprocessRunner")
    def test_generates_transfer_id_when_not_provided(self, MockRunner, _mock_xet_avail):
        """Should generate a transfer_id if not provided."""
        from hf_track.download import download_file_with_xet

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

    @patch("hf_track.download.xet_file.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_file.XetSubprocessRunner")
    def test_cancel_via_is_cancelled_hook(self, MockRunner, _mock_xet_avail):
        """Should terminate runner and emit CANCELLED when is_cancelled returns True."""
        from hf_track.download import download_file_with_xet
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

    # ── Plan 2026-07-09 step 7: deprecation ──────────────────────

    def test_download_file_with_xet_emits_deprecation_warning(self):
        """download_file_with_xet must emit a DeprecationWarning.

        Plan 2026-07-09 step 7: the legacy ``hf_xet.download_files()``
        path hangs indefinitely in some environments. The function is
        now deprecated in favor of ``download_file_with_xet_hybrid``.
        Calling it must warn so users migrate off it.
        """
        from hf_track.download import download_file_with_xet

        with patch("hf_track.download.xet_file.is_xet_available", return_value=False):
            with pytest.warns(DeprecationWarning, match="deprecated"):
                with pytest.raises(ImportError):
                    download_file_with_xet(
                        file_hash="abc123",
                        file_size=1024,
                        dest_path="/tmp/test.bin",
                        xet_file_data=MagicMock(),
                        token="test-token",
                        event_queue=queue.Queue(),
                    )

    @patch("hf_track.download.xet_file.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_file.XetSubprocessRunner")
    def test_deprecation_warning_mentions_xet_only_alternative(
        self, MockRunner, _mock_xet_avail
    ):
        """The deprecation message must point users to the dedicated path.

        Plan 2026-07-16 step 5: the warning text must mention
        ``download_file_xet_only`` (or ``use_xet=False``) so users know
        the replacement for the broken legacy API.
        """
        from hf_track.download import download_file_with_xet

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "success",
            "filename": "model.bin",
            "destination_path": "/tmp/model.bin",
            "file_size": 1024,
            "transfer_id": "dep-1",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        with pytest.warns(DeprecationWarning) as record:
            download_file_with_xet(
                file_hash="abc123",
                file_size=1024,
                dest_path="/tmp/model.bin",
                xet_file_data=MagicMock(),
                token="test-token",
                event_queue=queue.Queue(),
                transfer_id="dep-1",
            )

        # At least one warning mentions the dedicated-path alternative.
        messages = [str(w.message) for w in record]
        assert any(
            ("download_file_xet_only" in m or "use_xet=False" in m) for m in messages
        ), (
            "DeprecationWarning must mention download_file_xet_only / "
            "use_xet=False as the replacement. Got: " + repr(messages)
        )


# ── download_files_with_xet ───────────────────────────────────


class TestDownloadFilesWithXet:
    """Tests for download_files_with_xet function."""

    def test_raises_import_error_when_xet_unavailable(self):
        """Should raise ImportError when hf_xet is not installed."""
        from hf_track.download import download_files_with_xet

        with patch("hf_track.download.xet_batch.is_xet_available", return_value=False):
            with pytest.raises(ImportError, match="hf_xet is not installed"):
                download_files_with_xet(
                    file_specs=[{"dest_path": "/tmp/a.bin", "hash": "a", "file_size": 100, "xet_file_data": MagicMock()}],
                    token="test-token",
                    event_queue=queue.Queue(),
                )

    @patch("hf_track.download.xet_batch.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_batch.XetSubprocessRunner")
    def test_emits_start_and_complete_for_each_file(self, MockRunner, _mock_xet_avail):
        """Should emit START events for each file and return results."""
        from hf_track.download import download_files_with_xet

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

    @patch("hf_track.download.xet_batch.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_batch.XetSubprocessRunner")
    def test_emits_error_events_for_all_files_on_failure(self, MockRunner, _mock_xet_avail):
        """Should emit ERROR events for all files when batch fails."""
        from hf_track.download import download_files_with_xet

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
        assert all(isinstance(e.error, TransferErrorInfo) for e in error_events)
        assert all(e.error.message == "batch failed" for e in error_events)


# ── XetDownloadResult ─────────────────────────────────────────


class TestXetDownloadResult:
    """Tests for XetDownloadResult dataclass."""

    def test_default_values(self):
        from hf_track.download import XetDownloadResult

        result = XetDownloadResult(success=True, filename="test.bin")
        assert result.success is True
        assert result.filename == "test.bin"
        assert result.destination_path == ""
        assert result.file_size == 0
        assert result.transfer_id == ""

    def test_all_fields(self):
        from hf_track.download import XetDownloadResult

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
        from hf_track.download import download_snapshot_with_xet

        with patch("hf_track.download.xet_snapshot.is_xet_available", return_value=False):
            with pytest.raises(ImportError, match="hf_xet is not installed"):
                download_snapshot_with_xet(
                    repo_id="test/repo",
                    token="test-token",
                    event_queue=queue.Queue(),
                )

    @patch("hf_track.download.xet_snapshot.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_snapshot.XetSubprocessRunner")
    def test_emits_start_event(self, MockRunner, _mock_xet_avail):
        """Should emit a START event before downloading."""
        from hf_track.download import download_snapshot_with_xet

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

    @patch("hf_track.download.xet_snapshot.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_snapshot.XetSubprocessRunner")
    def test_returns_destination_path_on_success(self, MockRunner, _mock_xet_avail):
        """Should return the destination_path from the result."""
        from hf_track.download import download_snapshot_with_xet

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

    @patch("hf_track.download.xet_snapshot.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_snapshot.XetSubprocessRunner")
    def test_raises_cancelled_on_cancelled_result(self, MockRunner, _mock_xet_avail):
        """Should raise TransferCancelledError when worker returns cancelled status."""
        from hf_track.download import download_snapshot_with_xet
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

    @patch("hf_track.download.xet_snapshot.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_snapshot.XetSubprocessRunner")
    def test_raises_progress_error_on_error_result(self, MockRunner, _mock_xet_avail):
        """Should raise TransferProgressError when worker returns error status."""
        from hf_track.download import download_snapshot_with_xet

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

    @patch("hf_track.download.xet_snapshot.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_snapshot.XetSubprocessRunner")
    def test_cancel_via_is_cancelled_hook(self, MockRunner, _mock_xet_avail):
        """Should terminate runner and raise TransferCancelledError when is_cancelled returns True."""
        from hf_track.download import download_snapshot_with_xet
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

    @patch("hf_track.download.xet_snapshot.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_snapshot.XetSubprocessRunner")
    def test_terminates_runner_on_keyboard_interrupt(self, MockRunner, _mock_xet_avail):
        """Should terminate runner and raise TransferCancelledError on KeyboardInterrupt."""
        from hf_track.download import download_snapshot_with_xet
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

    @patch("hf_track.download.xet_snapshot.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_snapshot.XetSubprocessRunner")
    def test_terminates_runner_on_unexpected_exception(self, MockRunner, _mock_xet_avail):
        """Should terminate runner and emit ERROR event on unexpected exceptions."""
        from hf_track.download import download_snapshot_with_xet

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

    @patch("hf_track.download.xet_snapshot.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_snapshot.XetSubprocessRunner")
    def test_terminates_runner_in_finally(self, MockRunner, _mock_xet_avail):
        """Runner.terminate() should always be called in the finally block."""
        from hf_track.download import download_snapshot_with_xet

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

    @patch("hf_track.download.xet_snapshot.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_snapshot.XetSubprocessRunner")
    def test_passes_all_params_to_runner(self, MockRunner, _mock_xet_avail):
        """All params should be forwarded to the subprocess runner via params dict."""
        from hf_track.download import download_snapshot_with_xet

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

    @patch("hf_track.download.xet_snapshot.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_snapshot.XetSubprocessRunner")
    def test_generates_transfer_id_when_not_provided(self, MockRunner, _mock_xet_avail):
        """Should generate a transfer_id if not provided."""
        from hf_track.download import download_snapshot_with_xet

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

    @patch("hf_track.download.xet_snapshot.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_snapshot.XetSubprocessRunner")
    def test_uses_snapshot_worker(self, MockRunner, _mock_xet_avail):
        """Should use _snapshot_worker as the worker function."""
        from hf_track.download import download_snapshot_with_xet
        from hf_track._xet_worker import _snapshot_worker

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

    @patch("hf_track.download.xet_snapshot.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_snapshot.XetSubprocessRunner")
    def test_cancelled_error_propagates_without_terminate(self, MockRunner, _mock_xet_avail):
        """TransferCancelledError from result handling should propagate cleanly."""
        from hf_track.download import download_snapshot_with_xet
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

    @patch("hf_track.download.xet_snapshot.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_snapshot.XetSubprocessRunner")
    def test_fallback_to_repo_id_when_no_destination_path(self, MockRunner, _mock_xet_avail):
        """Should fall back to repo_id when destination_path is missing from result."""
        from hf_track.download import download_snapshot_with_xet

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

    @patch("hf_track.download.xet_snapshot.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_snapshot.XetSubprocessRunner")
    def test_passes_use_xet_in_params(self, MockRunner, _mock_xet_avail):
        """use_xet parameter is included in the params dict passed to the subprocess worker."""
        from hf_track.download import download_snapshot_with_xet

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

    @patch("hf_track.download.xet_snapshot.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_snapshot.XetSubprocessRunner")
    def test_use_xet_default_is_true(self, MockRunner, _mock_xet_avail):
        """Default use_xet is True when not specified."""
        from hf_track.download import download_snapshot_with_xet

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


class TestStatusDiscrimination:
    """``status`` is the sole discriminator for a worker result (IMP-008).

    The dispatch used to read ``status``, *then* ``error_type``, then
    substring-match the message for "cancelled"/"interrupted". That last
    clause is what made a plain failure look like a cancellation: a worker
    that fails with ``{"status": "error", "error_type":
    "TransferCancelledError", "message": "interrupted by peer"}`` raised
    ``TransferCancelledError``, so the caller's retry/abort logic took the
    wrong branch on a transport error.
    """

    @staticmethod
    def _mock_runner(MockRunner, result):
        mock_runner = MagicMock()
        mock_runner.wait.return_value = result
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner
        return mock_runner

    @patch("hf_track.download.xet_file.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_file.XetSubprocessRunner")
    def test_error_type_field_alone_does_not_raise_cancelled(self, MockRunner, _):
        """The payload that used to raise the wrong exception type."""
        from hf_track.download import download_file_with_xet
        from hf_track.types import TransferCancelledError

        self._mock_runner(
            MockRunner,
            {
                "status": "error",
                "error_type": "TransferCancelledError",
                "message": "interrupted by peer",
            },
        )

        with pytest.raises(TransferProgressError) as exc_info:
            download_file_with_xet(
                file_hash="abc123",
                file_size=1024,
                dest_path="/tmp/test.bin",
                xet_file_data=MagicMock(),
                token="test-token",
                event_queue=queue.Queue(),
            )

        assert not isinstance(exc_info.value, TransferCancelledError)
        assert "interrupted by peer" in str(exc_info.value)

    @patch("hf_track.download.xet_file.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_file.XetSubprocessRunner")
    def test_status_cancelled_still_raises_cancelled(self, MockRunner, _):
        """The real cancellation path is unchanged."""
        from hf_track.download import download_file_with_xet
        from hf_track.types import TransferCancelledError

        self._mock_runner(
            MockRunner, {"status": "cancelled", "message": "Transfer cancelled by user"}
        )

        with pytest.raises(TransferCancelledError):
            download_file_with_xet(
                file_hash="abc123",
                file_size=1024,
                dest_path="/tmp/test.bin",
                xet_file_data=MagicMock(),
                token="test-token",
                event_queue=queue.Queue(),
            )

    @patch("hf_track.download.xet_batch.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_batch.XetSubprocessRunner")
    def test_batch_uses_the_same_discriminator(self, MockRunner, _):
        """The batch path must not keep the message matching the file path lost."""
        from hf_track.download import download_files_with_xet
        from hf_track.types import TransferCancelledError

        self._mock_runner(
            MockRunner,
            {
                "status": "error",
                "error_type": "TransferCancelledError",
                "message": "interrupted by peer",
            },
        )

        with pytest.raises(TransferProgressError):
            download_files_with_xet(
                file_specs=[{"dest_path": "/tmp/a.bin", "file_size": 10, "file_hash": "h"}],
                token="test-token",
                event_queue=queue.Queue(),
            )

        self._mock_runner(
            MockRunner, {"status": "cancelled", "message": "Transfer cancelled by user"}
        )
        with pytest.raises(TransferCancelledError):
            download_files_with_xet(
                file_specs=[{"dest_path": "/tmp/a.bin", "file_size": 10, "file_hash": "h"}],
                token="test-token",
                event_queue=queue.Queue(),
            )

    @patch("hf_track.upload.xet_file.is_xet_available", return_value=True)
    @patch("hf_track.upload.xet_file.XetSubprocessRunner")
    def test_upload_uses_the_same_discriminator(self, MockRunner, _, tmp_path):
        """The upload path had the identical chain."""
        from hf_track.upload.xet_file import upload_file_with_xet
        from hf_track.types import TransferCancelledError

        # The upload path stats the file before spawning, so it needs a
        # real one on disk.
        upload_path = tmp_path / "a.bin"
        upload_path.write_bytes(b"x" * 16)

        self._mock_runner(
            MockRunner,
            {
                "status": "error",
                "error_type": "TransferCancelledError",
                "message": "interrupted by peer",
            },
        )

        with pytest.raises(TransferProgressError):
            upload_file_with_xet(
                file_path=str(upload_path),
                repo_id="user/repo",
                token="test-token",
                event_queue=queue.Queue(),
            )

        self._mock_runner(
            MockRunner, {"status": "cancelled", "message": "Transfer cancelled by user"}
        )
        with pytest.raises(TransferCancelledError):
            upload_file_with_xet(
                file_path=str(upload_path),
                repo_id="user/repo",
                token="test-token",
                event_queue=queue.Queue(),
            )

    def test_download_source_has_no_message_substring_matching(self):
        """The string matching is gone from every dispatch site, not just one.

        Asserted on the source because a behavioural test cannot prove the
        *absence* of a heuristic: a payload not exercised by any test would
        still pass.
        """
        import pathlib

        import hf_track.download as download_pkg

        sources = {
            path.name: path.read_text(encoding="utf-8")
            for path in (
                pathlib.Path(download_pkg.__file__).parent / name
                for name in ("xet_file.py", "xet_batch.py", "xet_snapshot.py")
            )
        }
        sources["upload/xet_file.py"] = (
            pathlib.Path(download_pkg.__file__).parent.parent / "upload" / "xet_file.py"
        ).read_text(encoding="utf-8")

        for name, src in sources.items():
            assert '"interrupted" in' not in src, f"{name} still substring-matches"
            assert '"cancelled" in' not in src, f"{name} still substring-matches"
            assert 'result.get("message", "").lower()' not in src, (
                f"{name} still lowercases the message to classify it"
            )


class TestBatchPooling:
    """NTH-008: a batch may run more than one worker subprocess.

    The default stays at one subprocess for the whole batch, so the tests
    here cover both halves: that concurrency is bounded by ``max_workers``
    when asked for, and that the unasked-for case is unchanged. A pooling
    change that silently raised the default would be a regression, not a
    feature.
    """

    class _ProbeRunner:
        """Stands in for XetSubprocessRunner, tracking concurrency."""

        instances = []
        lock = threading.Lock()
        live = 0
        peak = 0
        gate: Optional[threading.Event] = None

        def __init__(self):
            self.started_with = None
            type(self).instances.append(self)

        def start(self, worker_func=None, params=None, event_queue=None):
            self.started_with = params
            cls = type(self)
            with cls.lock:
                cls.live += 1
                cls.peak = max(cls.peak, cls.live)
            # Two rounds of polling so the main loop exercises the
            # wait() path rather than returning on the first call.
            self._calls = 0

        def wait(self, timeout=None):
            self._calls += 1
            if self._calls < 2:
                if type(self).gate is not None:
                    type(self).gate.wait(timeout=5)
                return None
            cls = type(self)
            with cls.lock:
                cls.live -= 1
            return {"status": "success", "message": "ok"}

        def terminate(self, grace=None):
            return None

    @classmethod
    def _specs(cls, count):
        return [
            {
                "dest_path": f"/tmp/file-{i}.bin",
                "file_size": 100 + i,
                "file_hash": f"hash-{i}",
                "xet_file_data": None,
            }
            for i in range(count)
        ]

    @pytest.fixture
    def probe(self):
        self._ProbeRunner.instances = []
        self._ProbeRunner.live = 0
        self._ProbeRunner.peak = 0
        self._ProbeRunner.gate = threading.Event()
        self._ProbeRunner.gate.set()
        yield self._ProbeRunner
        self._ProbeRunner.gate = None

    def test_max_workers_defaults_to_one(self):
        """The default preserves today's single-process behaviour."""
        from hf_track.download.xet_batch import get_default_max_workers

        assert get_default_max_workers() == 1

    def test_default_call_starts_exactly_one_runner(self, probe):
        """One batch, one subprocess, unless the caller asks otherwise."""
        from hf_track.download import download_files_with_xet

        with patch("hf_track.download.xet_batch.is_xet_available", return_value=True), \
             patch("hf_track.download.xet_batch.XetSubprocessRunner", probe):
            results = download_files_with_xet(
                file_specs=self._specs(4),
                token="t",
                event_queue=queue.Queue(),
            )

        assert len(probe.instances) == 1
        assert len(results) == 4

    def test_batch_download_respects_max_workers(self, probe):
        """Peak concurrency never exceeds the requested worker count."""
        from hf_track.download import download_files_with_xet

        probe.gate.clear()
        with patch("hf_track.download.xet_batch.is_xet_available", return_value=True), \
             patch("hf_track.download.xet_batch.XetSubprocessRunner", probe):
            worker = threading.Thread(
                target=lambda: download_files_with_xet(
                    file_specs=self._specs(6),
                    token="t",
                    event_queue=queue.Queue(),
                    max_workers=3,
                ),
                daemon=True,
            )
            worker.start()
            # Let all three workers reach their first wait() before the
            # gate opens, so the peak really is three and not one.
            deadline = time.time() + 5
            while probe.live < 3 and time.time() < deadline:
                time.sleep(0.01)
            assert probe.live == 3, (
                f"only {probe.live} workers started; the pool did not spawn them"
            )
            probe.gate.set()
            worker.join(timeout=15)

        assert not worker.is_alive()
        assert probe.peak <= 3

    def test_max_workers_cannot_exceed_the_file_count(self, probe):
        """Two files and four workers is two workers, not four empty ones."""
        from hf_track.download import download_files_with_xet

        with patch("hf_track.download.xet_batch.is_xet_available", return_value=True), \
             patch("hf_track.download.xet_batch.XetSubprocessRunner", probe):
            download_files_with_xet(
                file_specs=self._specs(2),
                token="t",
                event_queue=queue.Queue(),
                max_workers=4,
            )

        assert len(probe.instances) == 2

    def test_zero_workers_is_rejected(self):
        """A pool of zero would silently download nothing."""
        from hf_track.download.xet_batch import _split_specs

        with pytest.raises(ValueError, match="max_workers"):
            _split_specs(self._specs(3), 0)

    def test_pooled_batch_results_match_unpooled(self, probe):
        """Equivalence, not merely non-crashing: same files, same order."""
        from hf_track.download import download_files_with_xet

        def _run(max_workers):
            probe.instances = []
            with patch("hf_track.download.xet_batch.is_xet_available", return_value=True), \
                 patch("hf_track.download.xet_batch.XetSubprocessRunner", probe):
                return download_files_with_xet(
                    file_specs=self._specs(4),
                    token="t",
                    event_queue=queue.Queue(),
                    max_workers=max_workers,
                )

        serial = _run(1)
        pooled = _run(2)

        assert [r.destination_path for r in pooled] == [
            r.destination_path for r in serial
        ]
        assert all(r.success for r in pooled)

    def test_file_indices_stay_in_the_callers_numbering(self, probe):
        """A pooled run must not renumber file_index from 0 per worker."""
        from hf_track.download import download_files_with_xet

        q: queue.Queue = queue.Queue()
        with patch("hf_track.download.xet_batch.is_xet_available", return_value=True), \
             patch("hf_track.download.xet_batch.XetSubprocessRunner", probe):
            download_files_with_xet(
                file_specs=self._specs(4),
                token="t",
                event_queue=q,
                max_workers=2,
            )

        indices = sorted(
            e.file_index for e in list(q.queue) if e.event_type == EventType.START
        )
        assert indices == [0, 1, 2, 3]
