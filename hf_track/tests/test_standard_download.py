"""Tests for hf_progress.standard_download module."""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch

from hf_track.standard_download import download_snapshot
from hf_track.callbacks import state_manager
from hf_track.types import EventType, TransferDirection, ProgressPhase

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
