#!/usr/bin/env python3
"""Download a single file with subprocess-isolated Xet operations.

Demonstrates the ``HfTracker.download_file`` API with Xet enabled. Because
the single-file Xet download now runs ``hf_xet`` inside a terminable
subprocess (see plan ``docs/plans/2026-07-16-xet-single-file-subprocess-isolation.md``),
the Rust ``.pyd`` background thread lives only in the child process. This
means:

- **Ctrl+C safety**: ``tracker.cancel(transfer_id)`` flips the cancel flag,
  the in-process watchdog observes it, and calls ``runner.terminate()``
  (SIGTERM -> SIGKILL). The child OS process — and the ``hf_xet`` thread
  inside it — is killed directly, freeing its memory. The main process is
  never blocked by the uninterruptible Rust runtime.
- **No zombie processes**: ``XetSubprocessRunner`` cleans up the child in
  its ``finally`` block (``terminate()``), so it never outlives the parent.
- **Clean cancellation**: progress events keep flowing until the child exits;
  a CANCELLED event is emitted and the transfer raises ``TransferCancelledError``.

The subprocess isolation is internal to the tracker — this example just calls
``tracker.download_file(...)`` and ``tracker.cancel(transfer_id)`` like the
regular ``download_file.py`` example. The difference is that with Xet enabled,
the heavy ``hf_xet`` work happens in a child process that can be terminated.

Usage::

    python download_subprocess.py
    python download_subprocess.py --no-xet
    python download_subprocess.py --timeout 10
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import threading
import uuid

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download with subprocess-isolated Xet (Ctrl+C safe)"
    )
    parser.add_argument("--repo", default="openbmb/VoxCPM-0.5B")
    parser.add_argument("--file", default="audiovae.pth")
    parser.add_argument("--output", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--timeout", type=float, default=300, help="Max seconds to wait")
    parser.add_argument("--no-xet", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    # Late import: add project src to path only when run as a script
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from hf_track import (
        HfTracker,
        EventType,
        TransferCancelledError,
        is_xet_available,
    )
    from progress_bar import ConsoleProgressDisplay

    args = parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or None
    if args.no_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    output_dir = args.output or os.path.join(os.path.dirname(__file__), "downloads")
    os.makedirs(output_dir, exist_ok=True)

    print()
    print("=" * 60)
    print(" HuggingFace Subprocess-Isolated Downloader")
    print("=" * 60)
    print(f" Repository : {args.repo}")
    print(f" File       : {args.file}")
    print(f" Timeout    : {args.timeout}s")
    xet_status = (
        "[DISABLED]" if args.no_xet
        else ("[OK] available — Xet runs in a terminable subprocess" if is_xet_available()
              else "[--] not installed (using HTTP)")
    )
    print(f" Xet        : {xet_status}")
    print("=" * 60)
    print()
    print(" Press Ctrl+C to cancel — the subprocess will be terminated.")
    print(" (hf_xet's .pyd thread lives only in the child process.)")
    print()

    tracker = HfTracker(token=token, report_interval=0.1)
    transfer_id = str(uuid.uuid4())

    display = ConsoleProgressDisplay(filename=args.file, is_snapshot=False, bar_width=40)

    result_path = None
    error_occurred = None

    def do_download():
        nonlocal result_path, error_occurred
        try:
            result_path = tracker.download_file(
                repo_id=args.repo,
                filename=args.file,
                local_dir=output_dir,
                force_download=True,
                transfer_id=transfer_id,
            )
        except TransferCancelledError:
            pass
        except Exception as e:
            error_occurred = e

    download_thread = threading.Thread(target=do_download, daemon=True)
    download_thread.start()

    try:
        while download_thread.is_alive() or not tracker.event_queue.empty():
            try:
                event = tracker.event_queue.get(timeout=0.05)
                display.update(event)
            except queue.Empty:
                continue

    except KeyboardInterrupt:
        print("\n\n [STOP] Ctrl+C received — cancelling transfer...")

        # 1. Signal cancellation through the tracker. This flips the
        #    cancel flag; the in-process watchdog in
        #    download_file_xet_subprocess observes it and calls
        #    runner.terminate() (SIGTERM -> SIGKILL) on the child.
        tracker.cancel(transfer_id)

        # 2. Wait for the download thread to exit (the child is killed).
        download_thread.join(timeout=5.0)

        if download_thread.is_alive():
            print(" [WARN] Download thread did not exit within 5s")

        # Drain remaining events (may include CANCELLED)
        while not tracker.event_queue.empty():
            try:
                event = tracker.event_queue.get_nowait()
                display.update(event)
            except queue.Empty:
                break

        print(" Transfer cancelled — subprocess terminated, hf_xet memory freed.")
        return 1

    download_thread.join(timeout=5.0)

    # Drain remaining events
    while not tracker.event_queue.empty():
        try:
            event = tracker.event_queue.get_nowait()
            display.update(event)
        except queue.Empty:
            break

    if error_occurred:
        print(f"\n [ERROR] {error_occurred}")
        return 1

    if result_path:
        print(f"\n [OK] Downloaded to: {result_path}")
        return 0

    print("\n [WARN] Download completed but no path returned")
    return 1


if __name__ == "__main__":
    sys.exit(main())
