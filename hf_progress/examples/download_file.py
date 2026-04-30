#!/usr/bin/env python3
"""Download a single file from HuggingFace with custom progress bar.

Downloads a specific file from a HuggingFace repository using the
``hf-progress`` library, showing a byte-level animated progress bar
in the console.

Default file: ``audiovae.pth`` from ``openbmb/VoxCPM-0.5B``
(https://huggingface.co/openbmb/VoxCPM-0.5B/blob/main/audiovae.pth)

Usage::

    # From hf_progress/examples/ directory:
    python download_file.py

    # With a HuggingFace token (for gated repos):
    python download_file.py --token hf_...

    # Custom output directory:
    python download_file.py --output ./downloads

    # Download a different file:
    python download_file.py --repo user/model --file path/to/file.bin

    # Force HTTP fallback (skip Xet direct path):
    python download_file.py --no-xet

This uses ``HfProgressTracker.download_file()`` with a **direct-first**
strategy:

1. If ``hf_xet`` is available, calls ``hf_xet.download_files()`` directly
   with a detailed ``(total_update, item_updates)`` callback. This provides
   speed, dedup info, and per-item progress from the Rust runtime.

2. If Xet is not available (or ``--no-xet`` is set), falls back to
   ``hf_hub_download(tqdm_class=...)`` which works for HTTP downloads
   with basic byte-level progress.

Unlike ``download_repo.py`` (which shows file-count progress),
this example shows real byte-level progress with speed and ETA.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

# Ensure the hf_progress package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hf_progress import HfProgressTracker, EventType, ProgressEvent, is_xet_available
from progress_bar import ConsoleProgressDisplay, format_bytes, format_eta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a single HuggingFace file with custom progress bar",
    )
    parser.add_argument(
        "--repo",
        default="openbmb/VoxCPM-0.5B",
        help="HuggingFace repository ID (default: openbmb/VoxCPM-0.5B)",
    )
    parser.add_argument(
        "--file",
        default="audiovae.pth",
        help="Filename within the repository (default: audiovae.pth)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Local directory to download to (default: ./downloads)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Repository type (default: model)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Git revision (branch, tag, or commit hash)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-download even if file is cached",
    )
    parser.add_argument(
        "--no-xet",
        action="store_true",
        default=False,
        help="Disable Xet storage (use HTTP download instead)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve token — use None for anonymous access (empty string causes
    # "Illegal header value b'Bearer '" error in huggingface_hub)
    token = args.token or os.environ.get("HF_TOKEN") or None
    if not token:
        print("[!] No HF_TOKEN provided. Public repos will work, but gated repos require a token.")
        print("    Set HF_TOKEN env var or use --token hf_...\n")

    # Disable Xet if requested
    if args.no_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    # Resolve output directory
    output_dir = args.output or os.path.join(os.path.dirname(__file__), "downloads")
    os.makedirs(output_dir, exist_ok=True)

    # Print header
    print()
    print("=" * 60)
    print(" HuggingFace Single File Downloader")
    print("=" * 60)
    print(f" Repository : {args.repo}")
    print(f" File : {args.file}")
    print(f" Type : {args.repo_type}")
    print(f" Output : {output_dir}")
    xet_status = "[DISABLED]" if args.no_xet else ("[OK] available" if is_xet_available() else "[--] not installed (using HTTP)")
    print(f" Xet : {xet_status}")
    if args.force:
        print(f" Force : re-download even if cached")
    if args.revision:
        print(f" Revision : {args.revision}")
    print("=" * 60)

    # Create tracker
    tracker = HfProgressTracker(
        token=token,
        report_interval=0.1,  # Update every 100ms for smooth bar
    )

    # Create display
    display = ConsoleProgressDisplay(
        filename=args.file,
        is_snapshot=False,
        bar_width=40,
    )

    # Result holder
    result_path = None
    error_occurred = None

    # Run download in a background thread so we can consume events
    def do_download():
        nonlocal result_path, error_occurred
        try:
            result_path = tracker.download_file(
                repo_id=args.repo,
                filename=args.file,
                repo_type=args.repo_type,
                revision=args.revision,
                local_dir=output_dir,
                force_download=args.force,
            )
        except Exception as e:
            error_occurred = e

    download_thread = threading.Thread(target=do_download, daemon=True)
    download_thread.start()

    # Consume events and update display
    # Use short timeout (50ms) for responsive UI — the callback throttle
    # (100ms) controls event frequency; our timeout just needs to be
    # shorter so we don't miss the timing window.
    try:
        while download_thread.is_alive() or not tracker.event_queue.empty():
            try:
                event = tracker.event_queue.get(timeout=0.05)
                display.update(event)
            except Exception:
                continue
    except KeyboardInterrupt:
        print("\n\n [STOP] Download interrupted by user.")
        sys.exit(1)

    # Wait for thread to finish
    download_thread.join(timeout=5.0)

    # Drain remaining events
    while not tracker.event_queue.empty():
        try:
            event = tracker.event_queue.get_nowait()
            display.update(event)
        except Exception:
            break

    display.close()

    # Print final result
    if error_occurred:
        print(f"\n  [ERR] Download failed: {error_occurred}")
        sys.exit(1)

    if result_path:
        # Get file size
        file_size = 0
        try:
            file_size = os.path.getsize(result_path)
        except OSError:
            pass
        size_str = format_bytes(file_size) if file_size > 0 else ""
        print(f"\n  [FILE] File saved to: {result_path}")
        if size_str:
            print(f"  [SIZE] Size: {size_str}")
    print()


if __name__ == "__main__":
    main()
