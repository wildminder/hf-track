"""Tests for hf_progress.standard_download module."""

from __future__ import annotations

import queue
from pathlib import Path
from unittest.mock import MagicMock, patch

from hf_progress.standard_download import patch_xet_get, download_snapshot
from hf_progress.callbacks import state_manager
from hf_progress.types import EventType, TransferDirection, ProgressPhase


def test_patch_xet_get_no_index_error():
    """CRIT-005: Ensure patch_xet_get parses kwargs gracefully and avoids IndexError.

    The key behavior being tested is that patch_xet_get accesses xet_get's
    arguments via kwargs.get() instead of positional args indexing, which
    would throw IndexError on keyword-only functions.
    """
    mock_original_xet_get = MagicMock()

    # We patch inside the module scope to mock the import of xet_get
    with patch("huggingface_hub.file_download.xet_get", mock_original_xet_get, create=True):
        with patch_xet_get():
            import huggingface_hub.file_download as fd

            # Simulated keyword-only args exactly as huggingface_hub passes them.
            # The xet_file_data mock needs a real string for refresh_route
            # because refresh_xet_connection_info is a validated huggingface_hub
            # function that will try to use it as a URL.
            mock_xet_file_data = MagicMock()
            mock_xet_file_data.file_hash = "abc123"

            mock_kwargs = {
                "incomplete_path": Path("/tmp/file.bin.incomplete"),
                "xet_file_data": mock_xet_file_data,
                "headers": {"Authorization": "Bearer test"},
                "expected_size": 1024,
            }

            # Mock the internal _fetch_xet_connection_info_with_url which is
            # the actual function that makes the HTTP call. The public
            # refresh_xet_connection_info is a @validate decorator wrapper
            # that's hard to patch at the module level.
            mock_conn = MagicMock()
            mock_conn.access_token = "test-token"
            mock_conn.expiration_unix_epoch = 9999999999
            mock_conn.endpoint = "https://xet.example.com"

            with patch(
                "huggingface_hub.utils._xet._fetch_xet_connection_info_with_url",
                return_value=mock_conn,
            ):
                with patch("hf_xet.download_files") as mock_dl:
                    fd.xet_get(**mock_kwargs)

            # The patched function should have called download_files (not the original)
            mock_dl.assert_called_once()
            # The original xet_get should NOT have been called
            mock_original_xet_get.assert_not_called()


def test_patch_xet_get_fallback_on_missing_required_args():
    """CRIT-005: Ensure missing critical kwargs trigger a safe fallback."""
    mock_original = MagicMock()

    with patch("huggingface_hub.file_download.xet_get", mock_original, create=True):
        with patch_xet_get():
            import huggingface_hub.file_download as fd

            # Missing `incomplete_path` and `xet_file_data`
            mock_kwargs = {"headers": {}}

            with patch("hf_progress.standard_download.original_xet_get", mock_original, create=True):
                fd.xet_get(**mock_kwargs)
                mock_original.assert_called_once_with(**mock_kwargs)


def test_download_snapshot_emits_complete():
    """CRIT-004: Ensure download_snapshot synthesizes a COMPLETE event."""

    q = queue.Queue()
    mock_snapshot_download = MagicMock(return_value="/tmp/snapshot/repo")

    # Patch the lazy import inside download_snapshot
    with patch("huggingface_hub.snapshot_download", mock_snapshot_download):
        transfer_id = "test-snap-1"

        # Seed the state_manager to simulate the callbacks recording bytes
        state_manager.init_download(transfer_id)
        state_manager.update_download_bytes(transfer_id, 1024, 1024)
        state_manager.update_download_files(transfer_id, 2, 2)

        result = download_snapshot(
            repo_id="user/repo",
            token="hf_test",
            event_queue=q,
            transfer_id=transfer_id,
        )

        assert result == "/tmp/snapshot/repo"

        events = []
        while not q.empty():
            events.append(q.get_nowait())

        # download_snapshot no longer emits its own START event (the tqdm bars
        # handle that), so we only get the synthesized COMPLETE event
        assert len(events) >= 1

        complete_event = events[-1]

        assert complete_event.event_type == EventType.COMPLETE
        assert complete_event.transfer_id == transfer_id
        assert complete_event.direction == TransferDirection.DOWNLOAD
        assert complete_event.phase == ProgressPhase.COMPLETE
        assert complete_event.bytes_completed == 1024
        assert complete_event.total_bytes == 1024
        assert complete_event.percentage == 100.0
        assert complete_event.total_files == 2
