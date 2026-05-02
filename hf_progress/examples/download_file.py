#!/usr/bin/env python3
"""Download a single file from HuggingFace with custom progress bar.

Usage::
    python download_file.py
    python download_file.py --force
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
    parser = argparse.ArgumentParser(description="Download a single HuggingFace file")
    parser.add_argument("--repo", default="openbmb/VoxCPM-0.5B")
    parser.add_argument("--file", default="audiovae.pth")
    parser.add_argument("--output", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--revision", default=None)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--no-xet", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or None
    if args.no_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    output_dir = args.output or os.path.join(os.path.dirname(__file__), "downloads")
    os.makedirs(output_dir, exist_ok=True)

    print()
    print("=" * 60)
    print(" HuggingFace Single File Downloader")
    print("=" * 60)
    print(f" Repository : {args.repo}")
    print(f" File       : {args.file}")
    xet_status = "[DISABLED]" if args.no_xet else ("[OK] available" if is_xet_available() else "[--] not installed (using HTTP)")
    print(f" Xet        : {xet_status}")
    print("=" * 60)

    tracker = HfProgressTracker(token=token, report_interval=0.1)
    
    # Generate explicit transfer_id so we can cancel it cleanly later
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
                repo_type=args.repo_type,
                revision=args.revision,
                local_dir=output_dir,
                force_download=args.force,
                transfer_id=transfer_id,
            )
        except Exception as e:
            error_occurred = e
        except BaseException as e:
            error_occurred = Exception("Cancelled by user")

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
        print("\n\n [STOP] Download interrupted by user.")
        tracker.cancel(transfer_id)
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
        file_size = 0
        try:
            file_size = os.path.getsize(result_path)
        except OSError:
            pass
        from progress_bar import format_bytes
        size_str = format_bytes(file_size) if file_size > 0 else ""
        print(f"\n  [FILE] File saved to: {result_path}")
        if size_str:
            print(f"  [SIZE] Size: {size_str}")
    print()

if __name__ == "__main__":
    main()