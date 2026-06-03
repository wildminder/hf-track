#!/usr/bin/env python3
"""Download a HuggingFace repository IN-PROCESS (no subprocess isolation).

Diagnostic example: calls ``huggingface_hub.snapshot_download()`` directly
with ``DownloadProgressTqdm`` as the ``tqdm_class``, without spawning a
subprocess.  This isolates whether download issues are caused by the
subprocess/IPC layer or by the xet download itself.

Usage::

    python download_xet_inprocess.py
    python download_xet_inprocess.py --no-xet
    python download_xet_inprocess.py --repo openbmb/VoxCPM-0.5B

If xet is available and not disabled, ``snapshot_download()`` will
internally route file downloads through ``hf_xet`` — but the Rust
extension is loaded IN-PROCESS, so Ctrl+C cannot safely interrupt it.

This example is for DIAGNOSTIC purposes only.  For production use,
prefer ``download_repo.py`` which isolates xet in a subprocess.
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
        description="Download a HuggingFace repo IN-PROCESS (diagnostic — no subprocess)",
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
    from hf_track.callbacks import DownloadProgressTqdm, state_manager
    from progress_bar import ConsoleProgressDisplay

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
    print(" HuggingFace IN-PROCESS Downloader (DIAGNOSTIC)")
    print("=" * 60)
    print(f" Repository : {args.repo}")
    xet_status = "[DISABLED]" if args.no_xet else ("[OK] available" if is_xet_available() else "[--] not installed (using HTTP)")
    print(f" Xet        : {xet_status}")
    print(f" Mode       : IN-PROCESS (no subprocess isolation)")
    print("=" * 60)
    print()
    print("WARNING: hf_xet Rust extension will be loaded in-process.")
    print("         Ctrl+C may not work. Use download_repo.py for production.")
    print()

    # We use HfTracker just for its event_queue — the actual download
    # is done by calling snapshot_download() directly.
    tracker = HfTracker(token=token, report_interval=0.1)
    transfer_id = str(uuid.uuid4())

    # Create the bound tqdm_class that routes events to tracker.event_queue
    tqdm_class = DownloadProgressTqdm.bind(
        event_queue=tracker.event_queue,
        transfer_id=transfer_id,
        filename=args.repo,
        report_interval=0.1,
    )

    display = ConsoleProgressDisplay(filename=args.repo, is_snapshot=True, bar_width=40)

    result_path = None
    error_occurred = None

    def do_download():
        nonlocal result_path, error_occurred
        try:
            from huggingface_hub import snapshot_download

            download_kwargs = dict(
                repo_id=args.repo,
                repo_type=args.repo_type,
                allow_patterns=args.allow_patterns,
                ignore_patterns=args.ignore_patterns,
                token=token,
                tqdm_class=tqdm_class,
                force_download=args.force,
            )
            if output_dir:
                download_kwargs["local_dir"] = output_dir

            logger.debug("[DIAG-INPROC] calling snapshot_download(tqdm_class=...)")
            result_path = snapshot_download(**download_kwargs)  # nosec B615
            logger.debug("[DIAG-INPROC] snapshot_download returned: %s", result_path)

            # Synthesize COMPLETE event (matching standard_download pattern)
            stats = state_manager.get_state(transfer_id)
            from hf_track.types import ProgressEvent, ProgressPhase, TransferDirection
            tracker.event_queue.put(
                ProgressEvent(
                    event_type=EventType.COMPLETE,
                    transfer_id=transfer_id,
                    direction=TransferDirection.DOWNLOAD,
                    filename=args.repo,
                    phase=ProgressPhase.COMPLETE,
                    bytes_completed=stats.get("bytes_completed", 0),
                    total_bytes=stats.get("total_bytes", 0),
                    percentage=100.0,
                    file_index=stats.get("files_completed", 0),
                    total_files=stats.get("total_files", 0),
                )
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
                logger.debug(
                    "[DIAG-INPROC-EVENT] type=%s bytes=%s total=%s pct=%.2f",
                    event.event_type.value, event.bytes_completed,
                    event.total_bytes, event.percentage,
                )
                display.update(event)
            except queue.Empty:
                logger.debug("Queue empty while waiting for events — polling")
                continue

    except KeyboardInterrupt:
        print("\n\n [STOP] Download interrupted by user.")
        print(" [WARN] hf_xet is in-process — thread may not respond to cancellation.")
        tracker.cancel(transfer_id)
        download_thread.join(timeout=2.0)
        return 1

    download_thread.join(timeout=5.0)

    # Drain any final events
    final_drained = 0
    while not tracker.event_queue.empty():
        try:
            event = tracker.event_queue.get_nowait()
            final_drained += 1
            logger.debug(
                "[DIAG-INPROC-FINAL-EVENT] type=%s bytes=%s total=%s",
                event.event_type.value, event.bytes_completed, event.total_bytes,
            )
            display.update(event)
        except queue.Empty:
            break
    logger.debug("[DIAG-INPROC] drained %d final events", final_drained)

    display.close()

    # Verify downloaded files
    if result_path:
        print(f"\n [DIR] Files saved to: {result_path}")
        # Check for 0-length files
        zero_files = []
        total_files_count = 0
        for root, dirs, files in os.walk(result_path):
            for f in files:
                total_files_count += 1
                fp = os.path.join(root, f)
                try:
                    if os.path.getsize(fp) == 0:
                        zero_files.append(fp)
                except OSError:
                    pass
        if zero_files:
            print(f"\n [ERR] {len(zero_files)} of {total_files_count} files are 0-length:")
            for f in zero_files[:10]:
                print(f"        0 bytes: {f}")
            if len(zero_files) > 10:
                print(f"        ... and {len(zero_files) - 10} more")
        else:
            print(f" [OK] All {total_files_count} files have non-zero size.")
    else:
        print("\n [ERR] No result path returned.")

    if error_occurred:
        print(f"\n [ERR] Download failed: {error_occurred}")
        return 1

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
