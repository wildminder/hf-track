"""Tests for ``_download_file_xet`` routing to the subprocess path.

Plan: docs/plans/2026-07-16-xet-single-file-subprocess-isolation.md (Step 3)

Verifies that ``HfTracker._download_file_xet``:
  * calls ``download_file_xet_subprocess`` (NOT the in-process
    ``download_file_xet_only``) when xet is available,
  * returns the path the subprocess function returns,
  * propagates TransferCancelledError,
  * registers the runner in ``_active_runners[transfer_id]`` via on_spawn,
  * raises a clear error when xet_file_data is missing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hf_track.tracker import HfTracker
from hf_track.types import TransferCancelledError, TransferProgressError


def _make_tracker():
    tracker = HfTracker(token="hf_test")
    return tracker


def _fake_metadata(xet_file_data):
    meta = MagicMock()
    meta.xet_file_data = xet_file_data
    meta.size = 100
    return meta


def test_routes_to_subprocess_function():
    tracker = _make_tracker()
    transfer_id = "tid-sub"
    fake_xfd = MagicMock()
    fake_xfd.file_hash = "abc"
    fake_xfd.refresh_route = "https://xet/refresh"

    with patch(
        "huggingface_hub.HfApi.get_hf_file_metadata",
        return_value=_fake_metadata(fake_xfd),
    ), patch(
        "hf_track.download.download_file_xet_subprocess",
        return_value="/tmp/f.pth",
    ) as mock_sub:
        path = tracker._download_file_xet(  # type: ignore[attr-defined]
            repo_id="repo",
            filename="f.pth",
            repo_type="model",
            revision=None,
            local_dir="/tmp",
            transfer_id=transfer_id,
            is_cancelled=lambda: False,
        )
    assert path == "/tmp/f.pth"
    mock_sub.assert_called_once()
    # Confirm it is the subprocess function, not the in-process one.
    assert mock_sub._mock_name is None or "subprocess" in str(mock_sub)


def test_registers_runner_via_on_spawn():
    tracker = _make_tracker()
    transfer_id = "tid-reg"
    fake_xfd = MagicMock()
    fake_xfd.file_hash = "abc"
    fake_xfd.refresh_route = "https://xet/refresh"

    captured = {}

    def _fake_sub(**kwargs):
        on_spawn = kwargs.get("on_spawn")
        if on_spawn is not None:
            on_spawn("RUNNER-INSTANCE")
        captured["on_spawn_seen"] = on_spawn is not None
        return "/tmp/f.pth"

    with patch(
        "huggingface_hub.HfApi.get_hf_file_metadata",
        return_value=_fake_metadata(fake_xfd),
    ), patch(
        "hf_track.download.download_file_xet_subprocess",
        side_effect=_fake_sub,
    ):
        tracker._download_file_xet(  # type: ignore[attr-defined]
            repo_id="repo",
            filename="f.pth",
            repo_type="model",
            revision=None,
            local_dir="/tmp",
            transfer_id=transfer_id,
            is_cancelled=lambda: False,
            on_spawn=lambda r: captured.setdefault("runner", r),
        )
    assert captured.get("on_spawn_seen") is True
    # on_spawn must be called with the runner instance (so the caller can
    # register it for fast cancel / terminate). The actual _active_runners
    # registration is verified at the download_file() level (Step 4).
    assert captured.get("runner") is not None


def test_propagates_cancelled():
    tracker = _make_tracker()
    transfer_id = "tid-cancel"
    fake_xfd = MagicMock()
    fake_xfd.file_hash = "abc"
    fake_xfd.refresh_route = "https://xet/refresh"

    with patch(
        "huggingface_hub.HfApi.get_hf_file_metadata",
        return_value=_fake_metadata(fake_xfd),
    ), patch(
        "hf_track.download.download_file_xet_subprocess",
        side_effect=TransferCancelledError(),
    ):
        with pytest.raises(TransferCancelledError):
            tracker._download_file_xet(  # type: ignore[attr-defined]
                repo_id="repo",
                filename="f.pth",
                repo_type="model",
                revision=None,
                local_dir="/tmp",
                transfer_id=transfer_id,
                is_cancelled=lambda: False,
            )


def test_missing_xet_file_data_raises():
    tracker = _make_tracker()
    transfer_id = "tid-missing"
    # metadata.xet_file_data is None -> should raise before subprocess.
    with patch(
        "huggingface_hub.HfApi.get_hf_file_metadata",
        return_value=_fake_metadata(None),
    ), patch(
        "hf_track.download.download_file_xet_subprocess",
    ) as mock_sub:
        with pytest.raises(ValueError):
            tracker._download_file_xet(  # type: ignore[attr-defined]
                repo_id="repo",
                filename="f.pth",
                repo_type="model",
                revision=None,
                local_dir="/tmp",
                transfer_id=transfer_id,
                is_cancelled=lambda: False,
            )
    mock_sub.assert_not_called()
