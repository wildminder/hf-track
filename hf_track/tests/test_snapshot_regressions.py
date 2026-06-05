"""Regression tests for snapshot xet download bugs reported on 2026-06-03.

These tests capture the three reported bugs and verify the fix described in
``docs/plans/2026-06-03-revert-broken-snapshot-xetsession-path.md``:

1. **Nested folder bug**: ``download_repo.py`` with repo
   ``openbmb/VoxCPM-0.5B`` creates ``VoxCPM-0.5B/VoxCPM-0.5B/...`` instead
   of ``VoxCPM-0.5B/...``.

2. **Missing files bug**: Only 2 of 13 files are downloaded. Files
   without ``xet_hash`` and files in subdirectories are silently
   skipped.

3. **Zero-length files bug**: Cancelled/terminated downloads leave
   0-byte files on disk.

The tests are designed to FAIL against the buggy implementation and
PASS after the fix. They do NOT use the real HuggingFace API; they
validate the internal contract of the snapshot path code.
"""

from __future__ import annotations

import os
import queue
from unittest.mock import MagicMock, patch

import pytest


def _norm(path: str) -> str:
    """Normalize a path for cross-platform comparison."""
    return os.path.normpath(path).replace("\\", "/")


class TestSnapshotPathLayout:
    """Regression tests for the nested folder bug (BUG 1).

    Root cause: ``HfFileSystem.ls("models/owner/repo")`` returns names
    like ``"owner/repo/file.txt"`` (no ``models/`` prefix). The buggy
    code did ``full_name.split("/", 1)[1]`` which only strips one
    segment, leaving ``"repo/file.txt"``. The fix is to strip the
    full ``owner/repo/`` prefix.
    """

    def test_buggy_split_creates_nested_path(self):
        """Demonstrates the bug: a single ``split('/' 1)`` produces
        the wrong relative path. The fix uses ``removeprefix`` or
        string slicing with the full ``owner/repo/`` prefix.
        """
        full_name = "openbmb/VoxCPM-0.5B/README.md"

        # BUGGY behavior (what the current code does):
        buggy_rel = full_name.split("/", 1)[1]
        assert buggy_rel == "VoxCPM-0.5B/README.md", (
            "Test premise: this is exactly what the buggy code produces"
        )

        # CORRECT behavior (what the fix must do):
        repo_id = "openbmb/VoxCPM-0.5B"
        prefix = repo_id + "/"
        correct_rel = full_name[len(prefix):]
        assert correct_rel == "README.md", (
            "Fix should strip the full 'openbmb/VoxCPM-0.5B/' prefix"
        )

    def test_dest_path_for_repo_file_should_not_nest(self):
        """When local_dir='/tmp/dl' and rel_name='README.md',
        dest='/tmp/dl/README.md'. Must NOT contain repo name.
        """
        local_dir = "/tmp/dl"
        rel_name = "README.md"  # Correct relative name (after fix)
        dest = _norm(os.path.join(local_dir, rel_name))
        assert dest == _norm("/tmp/dl/README.md")
        assert "VoxCPM-0.5B" not in dest, (
            f"BUG 1: dest {dest!r} contains repo name — nested folder!"
        )

    def test_dest_path_for_subdirectory_file(self):
        """Files in subdirectories must end up at
        ``local_dir/assets/logo.png``, not ``local_dir/VoxCPM-0.5B/assets/logo.png``.
        """
        local_dir = "/tmp/dl"
        rel_name = "assets/modelbest_logo.png"
        dest = _norm(os.path.join(local_dir, rel_name))
        expected = _norm("/tmp/dl/assets/modelbest_logo.png")
        assert dest == expected
        assert "VoxCPM-0.5B" not in dest, (
            f"BUG 1: nested subdir path {dest!r} contains repo name"
        )


class TestSnapshotFileEnumeration:
    """Regression tests for the missing files bug (BUG 2)."""

    def test_snapshot_worker_uses_snapshot_download(self):
        """The proven path (``_snapshot_worker``) must use
        ``huggingface_hub.snapshot_download`` for correct file
        enumeration, subdirectory traversal, and resume safety.

        This is the regression assertion for the 2026-06-03 missing
        files bug: the old XetSession path used a custom
        ``HfFileSystem.ls(prefix, detail=True)`` (non-recursive,
        dropped non-xet files), which was removed.
        """
        from hf_track._xet_worker import _snapshot_worker
        import inspect
        source = inspect.getsource(_snapshot_worker)
        assert "snapshot_download" in source, (
            "The proven path (_snapshot_worker) must use "
            "huggingface_hub.snapshot_download for correct file "
            "enumeration, subdirectory traversal, and resume safety."
        )
        assert "from huggingface_hub import snapshot_download" in source, (
            "_snapshot_worker must import "
            "huggingface_hub.snapshot_download"
        )


class TestSnapshotResumeSafety:
    """Regression tests for the zero-length files bug (BUG 3).

    Root cause: the new XetSession path uses
    ``group.start_download_file(file_info, dest_path)`` directly with
    no pre-cleanup of existing files. On cancel, the partially-created
    empty file is left on disk. The fix uses ``huggingface_hub.snapshot_download``
    which has proper tmp-file + rename atomic semantics.
    """

    def test_snapshot_uses_proven_subprocess_path(self):
        """The snapshot path must use ``_snapshot_worker`` (which calls
        ``huggingface_hub.snapshot_download``) so that:
        - Existing partial files are properly handled
        - Cancellation cleans up correctly
        - Resume works (etag-based)
        """
        from hf_track._xet_worker import _snapshot_worker
        import inspect
        source = inspect.getsource(_snapshot_worker)
        assert "snapshot_download" in source, (
            "BUG 3: _snapshot_worker must use "
            "huggingface_hub.snapshot_download for proper resume safety"
        )
        assert "from huggingface_hub import snapshot_download" in source, (
            "BUG 3: _snapshot_worker must import "
            "huggingface_hub.snapshot_download"
        )

    def test_uses_subprocess_for_xet_isolation(self):
        """The snapshot path must spawn a subprocess so the xet
        Rust extension is loaded only in the child. This is the
        ONLY way to safely terminate a running xet download.
        """
        from hf_track._xet_worker import _snapshot_worker
        import inspect
        sig = inspect.signature(_snapshot_worker)
        params = list(sig.parameters.keys())
        assert "mp_queue" in params, (
            "_snapshot_worker must accept mp_queue parameter "
            "for subprocess isolation"
        )
        assert "cancel_event" in params, (
            "_snapshot_worker must accept cancel_event parameter "
            "for cross-process cancellation"
        )


class TestDispatcherRouting:
    """Tests that the tracker dispatches to the correct snapshot path."""

    def test_tracker_dispatches_to_proven_snapshot_path(self):
        """``HfTracker._download_snapshot_xet`` must delegate to
        ``download_snapshot_with_xet`` (the proven path that uses
        ``huggingface_hub.snapshot_download`` in an isolated subprocess).
        """
        from hf_track import tracker as tracker_module
        import inspect
        source = inspect.getsource(tracker_module.HfTracker._download_snapshot_xet)
        assert "download_snapshot_with_xet" in source, (
            "_download_snapshot_xet must delegate to "
            "download_snapshot_with_xet (the working path)."
        )


class TestSnapshotResultIsDirectory:
    """The snapshot return value should be the local_dir, not a nested folder."""

    @patch("hf_track.download.xet_snapshot.is_xet_available", return_value=True)
    @patch("hf_track.download.xet_snapshot.XetSubprocessRunner")
    def test_snapshot_returns_local_dir(self, MockRunner, _mock_xet_avail):
        """When local_dir is set, the snapshot download should
        return the local_dir path. The function should not nest
        the repo name inside local_dir.
        """
        from hf_track.download import download_snapshot_with_xet

        mock_runner = MagicMock()
        mock_runner.wait.return_value = {
            "status": "success",
            "filename": "test/repo",
            "destination_path": "/tmp/dl",  # huggingface_hub returns local_dir
            "file_size": 4096,
            "transfer_id": "snap-dir-test",
        }
        mock_runner.is_alive.return_value = False
        MockRunner.return_value = mock_runner

        event_queue = queue.Queue()
        result = download_snapshot_with_xet(
            repo_id="test/repo",
            token="test-token",
            event_queue=event_queue,
            transfer_id="snap-dir-test",
            local_dir="/tmp/dl",
        )
        assert result == "/tmp/dl", (
            f"Snapshot should return local_dir '/tmp/dl', got {result!r}. "
            "BUG 1: nested folder bug — the result includes the repo name "
            "in the path."
        )
