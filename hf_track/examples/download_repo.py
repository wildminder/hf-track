#!/usr/bin/env python3
"""Download an entire HuggingFace repository with custom progress bar.

Usage::

    python download_repo.py
    python download_repo.py --force
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
        description="Download a HuggingFace repository with custom progress bar",
    )
    parser.add_argument("--repo", default="hf-internal-testing/tiny-random-VoxtralRealtimeForConditionalGeneration", help="HuggingFace repository ID")
    parser.add_argument("--output", default=None, help="Local directory to download to")
    parser.add_argument("--token", default=None, help="HuggingFace API token")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--allow-patterns", nargs="*", default=None)
    parser.add_argument("--ignore-patterns", nargs="*", default=None)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--no-xet", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    # Late import: add project src to path only when run as a script
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from hf_track import (
        HfTracker, 
        EventType, 
        is_xet_available,
        TransferCancelledError,
    )
    from progress_bar import ConsoleProgressDisplay

    args = parse_args()

    if args.no_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    token = args.token or os.environ.get("HF_TOKEN") or None
    if not token:
        print("[!] No HF_TOKEN provided. Public repos will work, but gated repos require a token.\n")  # nosec: log_sensitive

    repo_name = args.repo.split("/")[-1]
    output_dir = args.output or os.path.join(os.path.dirname(__file__), repo_name)
    os.makedirs(output_dir, exist_ok=True)

    print()
    print("=" * 60)
    print(" HuggingFace Repository Downloader")
    print("=" * 60)
    print(f" Repository : {args.repo}")
    xet_status = "[DISABLED]" if args.no_xet else ("[OK] available" if is_xet_available() else "[--] not installed (using HTTP)")
    print(f" Xet        : {xet_status}")
    print("=" * 60)

    tracker = HfTracker(token=token, report_interval=0.1)

    # Generate explicit transfer_id so we can cancel it cleanly later
    transfer_id = str(uuid.uuid4())

    display = ConsoleProgressDisplay(filename=args.repo, is_snapshot=True, bar_width=40)

    result_path = None
    error_occurred = None

    def do_download():
        nonlocal result_path, error_occurred
        try:
            result_path = tracker.download_snapshot(
                repo_id=args.repo,
                repo_type=args.repo_type,
                allow_patterns=args.allow_patterns,
                ignore_patterns=args.ignore_patterns,
                local_dir=output_dir,
                force_download=args.force,
                transfer_id=transfer_id,
            )
        except TransferCancelledError:
            # Silently exit background thread on user cancellation
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
                logger.debug("Queue empty while waiting for events — polling")
                continue

    except KeyboardInterrupt:
        print("\n\n [STOP] Download interrupted by user.")
        tracker.cancel(transfer_id)
        # Give the background thread a moment to trap the cancel and exit cleanly
        download_thread.join(timeout=2.0)
        return 1

    download_thread.join(timeout=5.0)

    while not tracker.event_queue.empty():
        try:
            event = tracker.event_queue.get_nowait()
            display.update(event)
        except queue.Empty:
            break

    display.close()

    if error_occurred:
        print(f"\n [ERR] Download failed: {error_occurred}")
        return 1

    if result_path:
        print(f"\n [DIR] Files saved to: {result_path}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())