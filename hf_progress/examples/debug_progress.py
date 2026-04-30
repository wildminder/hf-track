#!/usr/bin/env python3
"""Debug script to trace progress events during download.

Adds logging to DownloadProgressTqdm to see:
- When update() is called and with what values
- When events are emitted
- When close() is called
"""

from __future__ import annotations

import os
import sys
import threading
import time

# Ensure the hf_progress package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hf_progress import HfProgressTracker, EventType, ProgressEvent, is_xet_available
from hf_progress.callbacks import DownloadProgressTqdm

# Patch DownloadProgressTqdm to add debug logging
_original_update = DownloadProgressTqdm.update
_original_close = DownloadProgressTqdm.close
_original_init = DownloadProgressTqdm.__init__

_update_count = 0
_event_count = 0

def _debug_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    print(f"[DEBUG-INIT] DownloadProgressTqdm created: total={self.total}, desc={self.desc}, "
          f"filename={self._filename}, report_interval={self._report_interval}", flush=True)

def _debug_update(self, n=1):
    global _update_count
    _update_count += 1
    result = _original_update(self, n)
    # Only log every 10th update or when percentage changes significantly
    if self.total and self.total > 0:
        pct = self.n / self.total * 100
        if _update_count <= 5 or _update_count % 10 == 0 or pct >= 99:
            print(f"[DEBUG-UPDATE #{_update_count}] n=+{n}, self.n={self.n}, "
                  f"total={self.total}, pct={pct:.1f}%, "
                  f"last_report_time={self._last_report_time}", flush=True)
    return result

def _debug_close(self):
    print(f"[DEBUG-CLOSE] self.n={self.n}, self.total={self.total}, "
          f"_closed={self._closed}", flush=True)
    return _original_close(self)

DownloadProgressTqdm.__init__ = _debug_init
DownloadProgressTqdm.update = _debug_update
DownloadProgressTqdm.close = _debug_close


def main():
    # Disable Xet for clean HTTP download test
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    repo = "openbmb/VoxCPM-0.5B"
    filename = "audiovae.pth"
    output_dir = os.path.join(os.path.dirname(__file__), "debug_downloads")
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Debug Progress Download ===", flush=True)
    print(f"Repo: {repo}", flush=True)
    print(f"File: {filename}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Xet: DISABLED (using HTTP)", flush=True)
    print(flush=True)

    tracker = HfProgressTracker(
        token=None,
        report_interval=0.1,
    )

    result_path = None
    error_occurred = None
    event_times = []

    def do_download():
        nonlocal result_path, error_occurred
        try:
            result_path = tracker.download_file(
                repo_id=repo,
                filename=filename,
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
                    print(f"[EVENT #{event_count} @ {elapsed:.2f}s] PROGRESS: "
                          f"bytes={event.bytes_completed}/{event.total_bytes} "
                          f"({event.percentage:.1f}%) speed={event.speed:.0f} B/s", flush=True)
                elif event.event_type == EventType.START:
                    print(f"[EVENT #{event_count} @ {elapsed:.2f}s] START", flush=True)
                elif event.event_type == EventType.COMPLETE:
                    print(f"[EVENT #{event_count} @ {elapsed:.2f}s] COMPLETE: "
                          f"bytes={event.bytes_completed}/{event.total_bytes}", flush=True)
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
    print(f"Total updates called: {_update_count}", flush=True)
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
