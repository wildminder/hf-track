#!/usr/bin/env python3
"""Download an entire HuggingFace repository with custom progress bar.

Usage::
    python download_repo.py
    python download_repo.py --force
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hf_progress import HfProgressTracker, EventType, is_xet_available
from progress_bar import ConsoleProgressDisplay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace repository with custom progress bar",
    )
    parser.add_argument("--repo", default="openbmb/VoxCPM-0.5B", help="HuggingFace repository ID")
    parser.add_argument("--output", default=None, help="Local directory to download to")
    parser.add_argument("--token", default=None, help="HuggingFace API token")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--allow-patterns", nargs="*", default=None)
    parser.add_argument("--ignore-patterns", nargs="*", default=None)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--no-xet", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    token = args.token or os.environ.get("HF_TOKEN") or None
    if not token:
        print("[!] No HF_TOKEN provided. Public repos will work, but gated repos require a token.\n")

    repo_name = args.repo.split("/")[-1]
    output_dir = args.output or os.path.join(os.path.dirname(__file__), repo_name)
    os.makedirs(output_dir, exist_ok=True)

    print()
    print("=" * 60)
    print(" HuggingFace Repository Downloader")
    print("=" * 60)
    print(f" Repository : {args.repo}")
    print(f" Type       : {args.repo_type}")
    print(f" Output     : {output_dir}")
    xet_status = "[DISABLED]" if args.no_xet else ("[OK] available" if is_xet_available() else "[--] not installed (using HTTP)")
    print(f" Xet        : {xet_status}")
    if args.force:
        print(f" Force      : re-download even if cached")
    print("=" * 60)

    tracker = HfProgressTracker(token=token, report_interval=0.1)
    
    # Generate explicit transfer_id so we can cancel it cleanly later
    transfer_id = str(uuid.uuid4())

    display = ConsoleProgressDisplay(filename=args.repo, is_snapshot=False, bar_width=35)

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
        except Exception as e:
            error_occurred = e
        except BaseException as e:
            # Handles things like KeyboardInterrupt slipping through
            error_occurred = Exception("Cancelled by user")

    download_thread = threading.Thread(target=do_download, daemon=True)
    download_thread.start()

    max_total_bytes = 0

    try:
        while download_thread.is_alive() or not tracker.event_queue.empty():
            try:
                event = tracker.event_queue.get(timeout=0.05)

                if event.event_type == EventType.PROGRESS:
                    if event.total_bytes > max_total_bytes:
                        max_total_bytes = event.total_bytes
                    
                    if max_total_bytes > 10000 and event.total_bytes < max_total_bytes // 2:
                        continue

                display.update(event)

            except queue.Empty:
                continue
                
    except KeyboardInterrupt:
        print("\n\n  [STOP] Download interrupted by user.")
        # Trigger abort to the background threads cleanly
        tracker.cancel(transfer_id)
        # Use os._exit(1) to drop dead instantly and skip ThreadPoolExecutor blocking
        os._exit(1)

    download_thread.join(timeout=5.0)

    while not tracker.event_queue.empty():
        try:
            event = tracker.event_queue.get_nowait()
            display.update(event)
        except queue.Empty:
            break

    display.close()

    if error_occurred:
        print(f"\n  [ERR] Download failed: {error_occurred}")
        sys.exit(1)

    if result_path:
        print(f"\n  [DIR] Files saved to: {result_path}")
    print()

if __name__ == "__main__":
    main()