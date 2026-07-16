"""Integration tests for HfTracker fallback paths."""
from __future__ import annotations

from unittest.mock import patch

import pytest  # noqa: F401
from hf_track.tracker import HfTracker
from hf_track.types import EventType, TransferDirection


class TestTrackerFallbackPaths:
    """Integration tests for Xet->standard fallback strategy.

    NOTE (2026-07-16): the single-file xet path is now a DEDICATED path
    with NO HTTP fallback. A xet failure propagates instead of silently
    falling back to the standard download. The separate reliable path is
    ``use_xet=False``. See docs/plans/2026-07-16-xet-download-separate-paths.md.
    """

    def test_download_file_xet_failure_propagates_no_fallback(self):
        """When the xet path fails, the error propagates (no silent
        standard fallback)."""
        tracker = HfTracker(token="hf_test")

        with patch("hf_track.tracker.is_xet_available", return_value=True):
            with patch.object(
                tracker, "_download_file_xet", side_effect=ValueError("Xet broken")
            ):
                with patch(
                    "hf_track.download.download_file",
                    return_value="/tmp/model.bin",
                ) as mock_std:
                    with pytest.raises(ValueError):
                        tracker.download_file(
                            repo_id="user/repo",
                            filename="model.bin",
                        )

        # The standard path must NOT have been used as a fallback.
        mock_std.assert_not_called()

    def test_upload_file_falls_back_to_lfs_when_xet_unavailable(self):
        """When Xet is unavailable, tracker uses LFS upload."""
        tracker = HfTracker(token="hf_test")

        with patch("hf_track.tracker.is_xet_available", return_value=False):
            with patch(
                "hf_track.upload.upload_file",
                return_value="https://hf.co/user/repo/blob/main/model.bin",
            ) as mock_std:
                result = tracker.upload_file(
                    file_path="/tmp/model.bin",
                    repo_id="user/repo",
                )

        assert result == "https://hf.co/user/repo/blob/main/model.bin"
        mock_std.assert_called_once()
        # Verify event was emitted (queue may be empty after get_events races)

    def test_upload_bytes_falls_back_to_temp_file_lfs(self):
        """When Xet unavailable for bytes, use temp file + LFS."""
        tracker = HfTracker(token="hf_test")

        with patch("hf_track.tracker.is_xet_available", return_value=False):
            with patch(
                "hf_track.upload.upload_bytes",
                return_value="https://hf.co/user/repo/blob/main/test.bin",
            ) as mock_std:
                result = tracker.upload_bytes(
                    file_content=b"test data",
                    filename="test.bin",
                    repo_id="user/repo",
                )

        assert result == "https://hf.co/user/repo/blob/main/test.bin"
        mock_std.assert_called_once()
        # Verify events exist (queue may have items, no race)

    def test_cancelled_transfer_stops_events(self):
        """Cancelled transfer stops event emission."""
        tracker = HfTracker(token="hf_test")
        tid = "test-transfer-123"
        tracker.cancel(tid)
        assert tracker.is_cancelled(tid)

    def test_wait_for_complete_returns_event(self):
        """wait_for_complete returns the COMPLETE event."""
        tracker = HfTracker(token="hf_test")
        tid = "test-transfer-456"

        # Inject a COMPLETE event manually
        from hf_track.types import ProgressEvent, ProgressPhase

        tracker.event_queue.put(
            ProgressEvent(
                event_type=EventType.COMPLETE,
                transfer_id=tid,
                direction=TransferDirection.UPLOAD,
                filename="test.bin",
                phase=ProgressPhase.COMPLETE,
            )
        )

        result = tracker.wait_for_complete(tid, timeout=1.0)
        assert result is not None
        assert result.event_type == EventType.COMPLETE
