#!/usr/bin/env python3
"""Download an entire HuggingFace repository with custom progress bar.

Downloads the repository ``openbmb/VoxCPM-0.5B`` to a local folder
using the ``hf-progress`` library, showing a custom animated progress
bar in the console.

Usage::

    # From hf_progress/examples/ directory:
    python download_repo.py

    # With a HuggingFace token (for gated repos):
    python download_repo.py --token hf_...

    # Custom local directory:
    python download_repo.py --output ./my_models

    # Download a different repo:
    python download_repo.py --repo user/model-name

The download uses ``HfProgressTracker.download_snapshot()`` which
wraps ``huggingface_hub.snapshot_download()`` with a custom
``tqdm_class`` that emits ``ProgressEvent`` objects. The snapshot
progress bar shows file-count progress (N/M files downloaded).

For per-file byte-level progress, see ``download_file.py``.
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
        description="Download a HuggingFace repository with custom progress bar",
    )
    parser.add_argument(
        "--repo",
        default="openbmb/VoxCPM-0.5B",
        help="HuggingFace repository ID (default: openbmb/VoxCPM-0.5B)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Local directory to download to (default: ./<repo-name>)",
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
        "--allow-patterns",
        nargs="*",
        default=None,
        help="Glob patterns for files to include (e.g. '*.safetensors')",
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=None,
        help="Glob patterns for files to exclude (e.g. '*.bin')",
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

    # Disable Xet if requested
    if args.no_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    # Resolve token — use None for anonymous access (empty string causes
    # "Illegal header value b'Bearer '" error in huggingface_hub)
    token = args.token or os.environ.get("HF_TOKEN") or None
    if not token:
        print("[!] No HF_TOKEN provided. Public repos will work, but gated repos require a token.")
        print(" Set HF_TOKEN env var or use --token hf_...\n")

    # Resolve output directory
    repo_name = args.repo.split("/")[-1]
    output_dir = args.output or os.path.join(os.path.dirname(__file__), repo_name)
    os.makedirs(output_dir, exist_ok=True)

    # Print header
    print()
    print("=" * 60)
    print(" HuggingFace Repository Downloader")
    print("=" * 60)
    print(f" Repository : {args.repo}")
    print(f" Type : {args.repo_type}")
    print(f" Output : {output_dir}")
    xet_status = "[DISABLED]" if args.no_xet else ("[OK] available" if is_xet_available() else "[--] not installed (using HTTP)")
    print(f" Xet : {xet_status}")
    if args.allow_patterns:
        print(f" Include : {args.allow_patterns}")
    if args.ignore_patterns:
        print(f" Exclude : {args.ignore_patterns}")
    print("=" * 60)

    # Create tracker
    tracker = HfProgressTracker(
        token=token,
        report_interval=0.15,  # Update every 150ms
    )

    # Create display
    display = ConsoleProgressDisplay(
        filename=args.repo,
        is_snapshot=True,
        bar_width=35,
    )

    # Result holder
    result_path = None
    error_occurred = None

    # Run download in a background thread so we can consume events
    def do_download():
        nonlocal result_path, error_occurred
        try:
            result_path = tracker.download_snapshot(
                repo_id=args.repo,
                repo_type=args.repo_type,
                allow_patterns=args.allow_patterns,
                ignore_patterns=args.ignore_patterns,
                local_dir=output_dir,
            )
        except Exception as e:
            error_occurred = e

    download_thread = threading.Thread(target=do_download, daemon=True)
    download_thread.start()

    # Consume events and update display
    # Use short timeout (50ms) for responsive UI — the callback throttle
    # (100ms) controls event frequency; our timeout just needs to be
    # shorter so we don't miss the timing window.
    last_file_count = 0
    try:
        while download_thread.is_alive() or not tracker.event_queue.empty():
            try:
                event = tracker.event_queue.get(timeout=0.05)
                display.update(event)

                # Show per-file progress for snapshot downloads
                if event.event_type == EventType.PROGRESS:
                    if event.file_index > last_file_count:
                        last_file_count = event.file_index
            except Exception:
                continue
    except KeyboardInterrupt:
        print("\n\n  [STOP] Download interrupted by user.")
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
        print(f"\n  [DIR] Files saved to: {result_path}")
    print()


if __name__ == "__main__":
    main()
