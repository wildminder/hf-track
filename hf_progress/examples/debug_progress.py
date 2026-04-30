#!/usr/bin/env python3
"""Debug script to trace progress events during download.

Supports two modes:

1. **Xet direct path** (default if hf_xet is installed): Traces
   ``XetDownloadProgressCallback.__call__`` which receives
   ``(total_update, item_updates)`` from the Rust runtime with
   detailed progress data (speed, dedup, per-item updates).

2. **HTTP fallback path** (``--no-xet``): Patches
   ``DownloadProgressTqdm`` to trace tqdm update/close calls.
   This is the fallback path used when hf_xet is not available.

Usage::

    # Default: use Xet direct path if available
    python debug_progress.py

    # Force HTTP fallback (tqdm-based) path
    python debug_progress.py --no-xet

    # With a HuggingFace token
    python debug_progress.py --token hf_...
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import types

# Ensure the hf_progress package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hf_progress import HfProgressTracker, EventType, ProgressEvent, is_xet_available
from hf_progress.callbacks import DownloadProgressTqdm, XetDownloadProgressCallback


# ── Patch DownloadProgressTqdm (HTTP fallback path) ───────────────

_original_update = DownloadProgressTqdm.update
_original_close = DownloadProgressTqdm.close
_original_init = DownloadProgressTqdm.__init__

_tqdm_update_count = 0

def _debug_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    print(f"[TQDM-INIT] DownloadProgressTqdm created: total={self.total}, desc={self.desc}, "
          f"filename={self._filename}, report_interval={self._report_interval}", flush=True)

def _debug_update(self, n=1):
    global _tqdm_update_count
    _tqdm_update_count += 1
    result = _original_update(self, n)
    # Only log every 10th update or when percentage changes significantly
    if self.total and self.total > 0:
        pct = self.n / self.total * 100
        if _tqdm_update_count <= 5 or _tqdm_update_count % 10 == 0 or pct >= 99:
            print(f"[TQDM-UPDATE #{_tqdm_update_count}] n=+{n}, self.n={self.n}, "
                  f"total={self.total}, pct={pct:.1f}%, "
                  f"last_report_time={self._last_report_time}", flush=True)
    return result

def _debug_close(self):
    print(f"[TQDM-CLOSE] self.n={self.n}, self.total={self.total}, "
          f"_closed={self._closed}", flush=True)
    return _original_close(self)

DownloadProgressTqdm.__init__ = _debug_init
DownloadProgressTqdm.update = _debug_update
DownloadProgressTqdm.close = _debug_close


# ── Patch XetDownloadProgressCallback (Xet direct path) ───────────

_original_xet_call = XetDownloadProgressCallback.__call__
_xet_call_count = 0

def _debug_xet_call(self, total_update, item_updates):
    global _xet_call_count
    _xet_call_count += 1
    # Extract key fields from the Rust progress objects
    bytes_completed = getattr(total_update, "total_bytes_completed", 0)
    total_bytes = getattr(total_update, "total_bytes", 0) or 0
    speed = getattr(total_update, "total_bytes_completion_rate", 0) or 0
    transfer_completed = getattr(total_update, "total_transfer_bytes_completed", 0)
    transfer_total = getattr(total_update, "total_transfer_bytes", 0)
    transfer_speed = getattr(total_update, "total_transfer_bytes_completion_rate", 0) or 0
    dedup_saved = max(0, bytes_completed - transfer_completed) if transfer_completed else 0

    pct = (bytes_completed / total_bytes * 100) if total_bytes > 0 else 0
    # Log every 10th call, first 5, or near completion
    if _xet_call_count <= 5 or _xet_call_count % 10 == 0 or pct >= 99:
        item_names = [getattr(iu, "item_name", "?") for iu in (item_updates or [])]
        print(f"[XET-CALL #{_xet_call_count}] "
              f"bytes={bytes_completed}/{total_bytes} ({pct:.1f}%) "
              f"speed={speed:.0f} B/s "
              f"transfer={transfer_completed}/{transfer_total} "
              f"transfer_speed={transfer_speed:.0f} B/s "
              f"dedup_saved={dedup_saved} "
              f"items={item_names}",
              flush=True)
    return _original_xet_call(self, total_update, item_updates)

XetDownloadProgressCallback.__call__ = _debug_xet_call


# ── Main ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug progress events during HuggingFace download",
    )
    parser.add_argument(
        "--repo", default="openbmb/VoxCPM-0.5B",
        help="HuggingFace repository ID (default: openbmb/VoxCPM-0.5B)",
    )
    parser.add_argument(
        "--file", default="audiovae.pth",
        help="Filename within the repository (default: audiovae.pth)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Local directory to download to (default: ./debug_downloads)",
    )
    parser.add_argument(
        "--token", default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--no-xet", action="store_true", default=False,
        help="Disable Xet storage (force HTTP fallback path)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Disable Xet if requested
    if args.no_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    output_dir = args.output or os.path.join(os.path.dirname(__file__), "debug_downloads")
    os.makedirs(output_dir, exist_ok=True)

    # Resolve token
    token = args.token or os.environ.get("HF_TOKEN") or None

    xet_available = is_xet_available() and not args.no_xet
    strategy = "Xet direct (detailed callbacks)" if xet_available else "HTTP fallback (tqdm_class)"

    print(f"=== Debug Progress Download ===", flush=True)
    print(f"Repo: {args.repo}", flush=True)
    print(f"File: {args.file}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Strategy: {strategy}", flush=True)
    print(flush=True)

    tracker = HfProgressTracker(
        token=token,
        report_interval=0.1,
    )

    result_path = None
    error_occurred = None
    event_times = []

    def do_download():
        nonlocal result_path, error_occurred
        try:
            result_path = tracker.download_file(
                repo_id=args.repo,
                filename=args.file,
                repo_type="model",
                local_dir=output_dir,
                force_download=True,
            )
        except Exception as e:
            error_occurred = e

    download_thread = threading.Thread(target=do_download, daemon=True)
    start_time = time.time()
    download_thread.start()

    # Consume events with very short timeout
    event_count = 0
    try:
        while download_thread.is_alive() or not tracker.event_queue.empty():
            try:
                event = tracker.event_queue.get(timeout=0.05)
                event_count += 1
                elapsed = time.time() - start_time
                event_times.append(elapsed)
                if event.event_type == EventType.PROGRESS:
                    # Show Xet-specific fields when available
                    xet_info = ""
                    if event.transfer_bytes_completed or event.transfer_bytes_total:
                        xet_info = (f" transfer={event.transfer_bytes_completed}/"
                                    f"{event.transfer_bytes_total}"
                                    f" transfer_speed={event.transfer_speed:.0f} B/s")
                    if event.dedup_saved_bytes:
                        xet_info += f" dedup_saved={event.dedup_saved_bytes}"
                    print(f"[EVENT #{event_count} @ {elapsed:.2f}s] PROGRESS: "
                          f"bytes={event.bytes_completed}/{event.total_bytes} "
                          f"({event.percentage:.1f}%) speed={event.speed:.0f} B/s"
                          f"{xet_info}", flush=True)
                elif event.event_type == EventType.START:
                    print(f"[EVENT #{event_count} @ {elapsed:.2f}s] START", flush=True)
                elif event.event_type == EventType.COMPLETE:
                    xet_info = ""
                    if event.transfer_bytes_completed or event.transfer_bytes_total:
                        xet_info = (f" transfer={event.transfer_bytes_completed}/"
                                    f"{event.transfer_bytes_total}")
                    if event.dedup_saved_bytes:
                        xet_info += f" dedup_saved={event.dedup_saved_bytes}"
                    print(f"[EVENT #{event_count} @ {elapsed:.2f}s] COMPLETE: "
                          f"bytes={event.bytes_completed}/{event.total_bytes}"
                          f"{xet_info}", flush=True)
                elif event.event_type == EventType.ERROR:
                    print(f"[EVENT #{event_count} @ {elapsed:.2f}s] ERROR: {event.error}", flush=True)
            except Exception:
                continue
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted", flush=True)
        sys.exit(1)

    download_thread.join(timeout=5.0)

    # Drain remaining events
    while not tracker.event_queue.empty():
        try:
            event = tracker.event_queue.get_nowait()
            event_count += 1
            elapsed = time.time() - start_time
            print(f"[EVENT #{event_count} @ {elapsed:.2f}s] {event.event_type.value}: "
                  f"bytes={event.bytes_completed}/{event.total_bytes}", flush=True)
        except Exception:
            break

    total_elapsed = time.time() - start_time
    print(f"\n=== Summary ===", flush=True)
    print(f"Strategy: {strategy}", flush=True)
    if xet_available:
        print(f"Xet callback calls: {_xet_call_count}", flush=True)
    else:
        print(f"Tqdm update calls: {_tqdm_update_count}", flush=True)
    print(f"Total events received: {event_count}", flush=True)
    print(f"Total elapsed: {total_elapsed:.1f}s", flush=True)
    if event_times:
        intervals = [event_times[i] - event_times[i-1] for i in range(1, len(event_times))]
        if intervals:
            print(f"Event intervals: min={min(intervals):.3f}s, max={max(intervals):.3f}s, "
                  f"avg={sum(intervals)/len(intervals):.3f}s", flush=True)

    if error_occurred:
        print(f"Error: {error_occurred}", flush=True)
    elif result_path:
        print(f"File saved to: {result_path}", flush=True)


if __name__ == "__main__":
    main()
