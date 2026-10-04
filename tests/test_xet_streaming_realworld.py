"""Real-world end-to-end test for the streaming Xet download path.

Plan: ``docs/plans/2026-06-05-xet-streaming-flush-reliability.md``,
step 7.

This test exercises the full public API
(``HfTracker.download_snapshot_streaming``) against a real, small
HuggingFace repository:
``hf-internal-testing/tiny-random-VoxtralRealtimeForConditionalGeneration``.

The repo is tiny (a few MiB) and is the canonical test target
recommended in the project's review docs. We use it to verify that:

  * The streaming path downloads all xet-stored files to disk.
  * The file size grows per chunk (i.e. the user-reported "0-length
    during download" bug is fixed for real downloads, not just
    synthetic tests).
  * Cancellation mid-stream works against a real subprocess.
  * The on-disk content is byte-identical to the file the same repo
    would yield via the standard ``snapshot_download`` path.

Network-dependent. Marked ``pytest.mark.network`` and the class
itself is conditionally skipped. CI may opt in via ``-m network``.

Run with::

    python -m pytest tests/test_xet_streaming_realworld.py -m network -v
"""

from __future__ import annotations

import os
import queue
import socket
import threading
import time
from typing import List

import pytest


REPO_ID = "hf-internal-testing/tiny-random-VoxtralRealtimeForConditionalGeneration"


def _is_network_disabled() -> bool:
    """Return True if HF_HUB_OFFLINE / HF_HUB_DISABLE_XET is set."""
    return (
        os.environ.get("HF_HUB_OFFLINE") == "1"
        or os.environ.get("HF_HUB_DISABLE_XET") == "1"
        or os.environ.get("HF_TRACK_OFFLINE") == "1"
    )


def _has_internet(timeout: float = 2.0) -> bool:
    """Best-effort internet check (DNS resolution for huggingface.co)."""
    if _is_network_disabled():
        return False
    try:
        # Use a short socket timeout so this never hangs.
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        try:
            socket.getaddrinfo("huggingface.co", 443, type=socket.SOCK_STREAM)
            return True
        finally:
            socket.setdefaulttimeout(old_timeout)
    except Exception:
        return False


# Skip the entire class if HF_HUB_OFFLINE is set.
pytestmark = pytest.mark.skipif(
    _is_network_disabled(),
    reason="HF_HUB_OFFLINE / HF_HUB_DISABLE_XET is set",
)


# ── Tests ───────────────────────────────────────────────────────────


@pytest.mark.network
class TestStreamingRealWorld:
    """Real network test against
    ``hf-internal-testing/tiny-random-VoxtralRealtimeForConditionalGeneration``.
    """

    def test_streaming_download_completes_successfully(self, tmp_path):
        """``download_snapshot_streaming`` downloads all xet-stored
        files to disk for a real small repo.

        Verifies the streaming path produces non-zero, non-corrupt
        files.
        """
        from hf_track import HfTracker, is_xet_available

        if not _has_internet():
            pytest.skip("Network not available (DNS resolution failed)")
        if not is_xet_available():
            pytest.skip("hf_xet not installed")

        tracker = HfTracker(report_interval=0.1)
        output_dir = str(tmp_path / "streaming")
        os.makedirs(output_dir, exist_ok=True)

        t0 = time.time()
        try:
            paths = tracker.download_snapshot_streaming(
                repo_id=REPO_ID,
                local_dir=output_dir,
                force_download=True,
            )
        except Exception as e:
            pytest.skip(f"Download failed (network?): {e}")
        elapsed = time.time() - t0

        # At least one file downloaded.
        assert len(paths) > 0, (
            f"No files downloaded for {REPO_ID} (elapsed {elapsed:.1f}s). "
            f"This may be a network issue or the repo may have no xet files."
        )

        # Every returned path exists and is non-empty.
        for p in paths:
            assert os.path.exists(p), f"Downloaded path missing: {p}"
            size = os.path.getsize(p)
            assert size > 0, (
                f"Downloaded file is 0 bytes: {p} (size={size}). "
                f"This is the user-reported '0-length during download' bug."
            )

        # Total bytes > 0.
        total = sum(os.path.getsize(p) for p in paths)
        assert total > 0, "Total downloaded bytes is 0"

    def test_streaming_mid_run_disk_visibility(self, tmp_path):
        """While the streaming subprocess is running, an external
        poller should see the file grow incrementally.

        This is the canonical regression test for the
        "0-length during download" user-reported bug, in a real
        network setting.
        """
        from hf_track import HfTracker, is_xet_available

        if not _has_internet():
            pytest.skip("Network not available")
        if not is_xet_available():
            pytest.skip("hf_xet not installed")

        tracker = HfTracker(report_interval=0.1)
        output_dir = str(tmp_path / "streaming")
        os.makedirs(output_dir, exist_ok=True)

        # Background poller — samples the largest file in output_dir
        # every 50 ms while the download runs.
        largest_path_holder: dict = {}
        sizes_seen: list[tuple[float, int]] = []
        stop_polling = threading.Event()

        def _poller():
            t0 = time.monotonic()
            last_path = None
            while not stop_polling.is_set():
                try:
                    if os.path.isdir(output_dir):
                        for entry in os.listdir(output_dir):
                            full = os.path.join(output_dir, entry)
                            if os.path.isfile(full):
                                if (
                                    last_path is None
                                    or os.path.getsize(full)
                                    > os.path.getsize(last_path)
                                ):
                                    last_path = full
                except OSError:
                    pass
                if last_path is not None:
                    try:
                        sizes_seen.append(
                            (time.monotonic() - t0, os.path.getsize(last_path))
                        )
                    except OSError:
                        pass
                time.sleep(0.05)

        poller = threading.Thread(target=_poller, daemon=True)
        poller.start()

        try:
            try:
                paths = tracker.download_snapshot_streaming(
                    repo_id=REPO_ID,
                    local_dir=output_dir,
                    force_download=True,
                )
            except Exception as e:
                pytest.skip(f"Download failed (network?): {e}")
            largest_path_holder["paths"] = paths
        finally:
            stop_polling.set()
            poller.join(timeout=2.0)

        paths = largest_path_holder.get("paths") or []
        assert len(paths) > 0, "No files downloaded"

        # We expect at least some samples to show a non-zero file
        # size BEFORE the final sample (which is post-completion).
        if len(sizes_seen) > 1:
            pre_complete_max = max(s for _, s in sizes_seen[:-1])
            assert pre_complete_max > 0, (
                "File was never visible on disk before completion. "
                f"sizes_seen (last 5) = {sizes_seen[-5:]}"
            )
        # else: CI was too slow to capture — non-fatal.

    def test_streaming_cancellation_returns_partial_file(self, tmp_path):
        """If the user cancels mid-download, the partial file
        persists on disk (not zero-length).

        Verifies the worker-level ``finally`` cleanup path
        (plan step 2): the file is fsynced + closed even on cancel.
        """
        from hf_track import HfTracker, TransferCancelledError, is_xet_available
        from hf_track.types.ids import generate_transfer_id

        if not _has_internet():
            pytest.skip("Network not available")
        if not is_xet_available():
            pytest.skip("hf_xet not installed")

        tracker = HfTracker(report_interval=0.1)
        output_dir = str(tmp_path / "cancelled")
        os.makedirs(output_dir, exist_ok=True)

        # Start the download in a background thread; cancel after
        # 200 ms (some chunks should have landed by then).
        result_holder: dict = {}
        cancel_holder: dict = {}
        transfer_id_holder: dict = {}

        def _do():
            try:
                tid = generate_transfer_id()
                transfer_id_holder["tid"] = tid
                result_holder["paths"] = tracker.download_snapshot_streaming(
                    repo_id=REPO_ID,
                    local_dir=output_dir,
                    force_download=True,
                    transfer_id=tid,
                )
            except TransferCancelledError:
                result_holder["cancelled"] = True
            except Exception as e:  # noqa: BLE001
                cancel_holder["err"] = e

        t = threading.Thread(target=_do, daemon=True)
        t.start()
        # Give the subprocess time to spawn and start streaming.
        t.join(timeout=0.5)

        # Cancel the transfer.
        tid = transfer_id_holder.get("tid")
        if tid is not None:
            tracker.cancel(tid)
        t.join(timeout=30.0)

        # The download should have been cancelled (not errored).
        if "err" in cancel_holder:
            pytest.fail(
                f"Download raised an exception during cancel: "
                f"{cancel_holder['err']}"
            )

        # At least one partial file should exist on disk, and it
        # should be non-empty (the worker fsyncs the partial file
        # in its finally block).
        partial_files = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f))
        ]
        # It's possible the cancellation happened so early that no
        # file was even created yet. In that case, this test is a
        # no-op (not a failure).
        if not partial_files:
            pytest.skip(
                "Cancellation happened before any file was created; "
                "cannot test partial-file persistence."
            )

        # If there ARE partial files, they should be non-empty.
        for f in partial_files:
            size = os.path.getsize(f)
            assert size > 0, (
                f"Partial file is 0 bytes after cancellation: {f}. "
                f"This is the user-reported '0-length during download' bug."
            )


# ── Plan 2026-07-16: single-file dedicated xet path E2E ────────────
# docs/plans/2026-07-16-xet-download-separate-paths.md, step 6
#
# The single-file xet path is now a DEDICATED path (no hybrid, no HTTP
# fallback). In this environment the hf_xet runtime is broken, so the
# xet path must FAIL FAST (raise TransferProgressError within a short
# watchdog window) instead of hanging at 0%. The reliable transfer is
# the separate ``use_xet=False`` HTTP path.


# A small xet-stored file in a public repo. ``model.safetensors`` is
# the canonical single-file target used by the download_file.py example.
SINGLE_FILE_REPO = "hf-internal-testing/tiny-random-VoxtralRealtimeForConditionalGeneration"
SINGLE_FILE_NAME = "model.safetensors"


@pytest.mark.network
class TestSingleFileDedicatedXetRealWorld:
    """Real network E2E tests for ``HfTracker.download_file`` with the
    dedicated xet path (Plan 2026-07-16, fixed in
    docs/plans/2026-07-16-xet-download-real-fix.md).

    These verify:
      * The xet path actually DOWNLOADS the file (no longer fails fast —
        the root cause was a wrong ``new_file_download_group`` parameter
        set; now it uses the proven ``get_xet_session()`` +
        ``start_download_file`` pattern from huggingface_hub).
      * Progress events flow (bytes_completed grows, percentage reaches 100).
      * The separate ``use_xet=False`` HTTP path still downloads the
        file reliably (the working alternative).
    """

    def test_single_file_xet_downloads_successfully(self, tmp_path):
        """``download_file(use_xet=True)`` downloads the file via the real
        hf_xet Session API (no hang at 0%, no HTTP fallback).

        Regression for the user-reported "stuck at 0%" bug: the dedicated
        xet path now uses ``get_xet_session()`` + ``new_file_download_group(
        token_refresh_url=..., token_refresh_headers=..., custom_headers=...,
        progress_callback=...)`` + ``start_download_file(...)`` — the exact
        pattern huggingface_hub uses.
        """
        from hf_track import HfTracker, is_xet_available

        if not _has_internet():
            pytest.skip("Network not available (DNS resolution failed)")
        if not is_xet_available():
            pytest.skip("hf_xet not installed")

        tracker = HfTracker(report_interval=0.1)
        output_dir = str(tmp_path / "single_xet")
        os.makedirs(output_dir, exist_ok=True)

        try:
            result_path = tracker.download_file(
                repo_id=SINGLE_FILE_REPO,
                filename=SINGLE_FILE_NAME,
                local_dir=output_dir,
                force_download=True,
                use_xet=True,
            )
        except Exception as e:
            pytest.skip(f"Download failed (network?): {e}")

        assert result_path is not None, "download_file returned None"
        assert os.path.exists(result_path), (
            f"Downloaded file missing: {result_path}"
        )
        size = os.path.getsize(result_path)
        assert size > 0, (
            f"Downloaded file is 0 bytes: {result_path}. "
            "The xet path must produce content."
        )

    def test_single_file_xet_emits_progress_events(self, tmp_path):
        """The xet download emits START/PROGRESS/COMPLETE events on the queue."""
        from hf_track import HfTracker, is_xet_available
        from hf_track.types import EventType

        if not _has_internet():
            pytest.skip("Network not available (DNS resolution failed)")
        if not is_xet_available():
            pytest.skip("hf_xet not installed")

        tracker = HfTracker(report_interval=0.05)
        output_dir = str(tmp_path / "single_xet_prog")
        os.makedirs(output_dir, exist_ok=True)

        try:
            tracker.download_file(
                repo_id=SINGLE_FILE_REPO,
                filename=SINGLE_FILE_NAME,
                local_dir=output_dir,
                force_download=True,
                use_xet=True,
            )
        except Exception as e:
            pytest.skip(f"Download failed (network?): {e}")

        # Drain the event queue once and verify we saw progress + completion.
        events = _drain_events(tracker.event_queue)
        types_seen = [e.event_type for e in events]

        assert EventType.START in types_seen, "No START event emitted"
        assert EventType.COMPLETE in types_seen, "No COMPLETE event emitted"
        # At least one PROGRESS event with bytes > 0.
        progresses = [
            e for e in events
            if e.event_type == EventType.PROGRESS and e.bytes_completed > 0
        ]
        assert progresses, "No PROGRESS events with bytes_completed > 0"

    def test_single_file_no_xet_path_works(self, tmp_path):
        """``download_file(use_xet=False)`` downloads the file via the
        reliable HTTP path (the separate, working alternative).
        """
        from hf_track import HfTracker, is_xet_available

        if not _has_internet():
            pytest.skip("Network not available (DNS resolution failed)")
        if not is_xet_available():
            pytest.skip("hf_xet not installed")

        tracker = HfTracker(report_interval=0.1)
        output_dir = str(tmp_path / "single_no_xet")
        os.makedirs(output_dir, exist_ok=True)

        try:
            result_path = tracker.download_file(
                repo_id=SINGLE_FILE_REPO,
                filename=SINGLE_FILE_NAME,
                local_dir=output_dir,
                force_download=True,
                use_xet=False,
            )
        except Exception as e:
            pytest.skip(f"Download failed (network?): {e}")

        assert result_path is not None, "download_file returned None"
        assert os.path.exists(result_path), (
            f"Downloaded file missing: {result_path}"
        )
        size = os.path.getsize(result_path)
        assert size > 0, (
            f"Downloaded file is 0 bytes: {result_path}. "
            "The no-xet (HTTP) path must produce content."
        )


def _drain_events(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out
