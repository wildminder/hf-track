#!/usr/bin/env python3
"""Download a HuggingFace repository using the Xet STREAMING API.

This example calls ``HfTracker.download_snapshot_streaming()``, which:

* Resolves the file list via ``HfApi.get_paths_info()``.
* Partitions the files into xet-stored and non-xet-stored.
* For xet files, spawns a child process running
  ``hf_track._xet_worker._xet_streaming_download_worker``. The child uses
  ``hf_xet.XetSession().new_download_stream_group().download_stream()`` and
  flushes each chunk to disk via ``os.write`` + ``os.fsync`` so a SIGKILL
  at any point leaves a non-empty file on disk.
* For non-xet files (small JSON, markdown, .gitignore, etc.) the parent
  process uses ``huggingface_hub.hf_hub_download()`` -- these files are
  small and bounded.

Compared to the old in-process streaming example that was in this file,
this approach is the actual fix for the snapshot bug. The in-process
variant was proven broken: the ``hf_xet`` C extension held the GIL during
file reconstruction, so the Python iterator over the stream delivered
one big chunk at end-of-file (zero mid-stream visibility) and a SIGKILL
could not interrupt the in-flight Rust fetch.

Compare to:
- ``download_repo.py``     -- uses ``HfTracker.download_snapshot()`` which
  spawns a subprocess; the subprocess uses ``hf_xet.download_files()``
  (buffered API). RSS peaks at file size, mid-stream progress is one
  jump per file.
- ``download_xet_inprocess.py`` -- uses ``snapshot_download(tqdm_class=...)``
  in-process. Same buffered API, same memory pattern.
- ``download_xet_inprocess_smooth.py`` -- same as above, but with a
  smooth-progress ticker (separate concern, kept for illustration only).
- ``download_xet_streaming.py`` (this file) -- the recommended path for
  large xet-stored models/datasets. Memory bounded by chunk size, mid-
  stream disk visibility, clean mid-file cancellation via SIGTERM of
  the child.

Key properties of this fix:
- **API path**: ``XetSession().new_download_stream_group().download_stream()``
  yields ``bytes`` chunks as they are reconstructed (vs buffered).
- **Memory**: bounded by chunk size (~4 MB), not file size.
- **Disk**: file size grows incrementally as chunks arrive. Periodic
  ``os.fsync()`` flushes the page cache so a SIGKILL at any point
  leaves a non-zero, non-truncated file on disk.
- **Cancellation**: ``tracker.cancel(transfer_id)`` calls
  ``XetSubprocessRunner.terminate()`` which SIGTERMs the child. The
  child also checks a cancel event between chunks for graceful exit.
- **No workarounds**: the streaming worker runs in a real subprocess
  because there is no way to kill the in-flight C-extension fetch
  from the same process (the GIL is held).

Usage::

    python download_xet_streaming.py
    python download_xet_streaming.py --no-xet
    python download_xet_streaming.py --repo openbmb/VoxCPM-0.5B
    python download_xet_streaming.py --watch-mem   # log RSS every 500ms

Requires: ``hf_xet >= 1.5.0`` (for the streaming API).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import queue
import sys
import threading
import time

logger = logging.getLogger(__name__)


# -- Memory Watcher -----------------------------------------------


class MemoryWatcher:
    """Background thread that logs (elapsed, rss_mb, disk_mb) to a CSV.

    Helps verify that the streaming API truly keeps memory bounded
    (vs the buffered API which grows RSS to file size).
    """

    def __init__(self, output_dir: str, csv_path: str, interval_s: float = 0.5):
        self._output_dir = output_dir
        self._csv_path = csv_path
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._csv_file = None
        self._writer = None

    def start(self) -> None:
        import psutil  # local import (optional dep)
        self._psutil = psutil
        self._csv_file = open(self._csv_path, "w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(["elapsed_s", "rss_mb", "disk_mb"])
        self._csv_file.flush()
        self._thread = threading.Thread(target=self._run, name="MemWatcher", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

    def _total_dir_size(self) -> int:
        total = 0
        if not os.path.isdir(self._output_dir):
            return 0
        for root, _dirs, files in os.walk(self._output_dir):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total

    def _run(self) -> None:
        start = time.time()
        proc = self._psutil.Process(os.getpid())
        while not self._stop.is_set():
            elapsed = time.time() - start
            rss = proc.memory_info().rss / (1024 * 1024)
            disk = self._total_dir_size() / (1024 * 1024)
            self._writer.writerow([f"{elapsed:.2f}", f"{rss:.1f}", f"{disk:.1f}"])
            self._csv_file.flush()
            if self._stop.wait(self._interval):
                break


# -- Argparse ----------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a HF repo using HfTracker.download_snapshot_streaming() "
            "(subprocess-isolated chunk-by-chunk Xet streaming)"
        ),
    )
    parser.add_argument(
        "--repo",
        default="hf-internal-testing/tiny-random-VoxtralRealtimeForConditionalGeneration",
        help="HuggingFace repository ID",
    )
    parser.add_argument("--output", default=None, help="Local directory to download to")
    parser.add_argument("--token", default=None, help="HuggingFace API token")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--allow-patterns", nargs="*", default=None)
    parser.add_argument("--ignore-patterns", nargs="*", default=None)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--no-xet", action="store_true", default=False)
    parser.add_argument(
        "--watch-mem", action="store_true",
        help="Log RSS and disk size to tmp/diag_streaming_mem.csv every 500ms",
    )
    parser.add_argument(
        "--fsync-interval", type=int, default=4 * 1024 * 1024,
        help=(
            "Bytes between os.fsync() calls in the worker (default: 4 MiB). "
            "Set to 0 to fsync after every chunk (safest, slowest). "
            "Set to a large value (e.g. 64 MiB) for max throughput."
        ),
    )
    parser.add_argument(
        "--no-fsync", action="store_true", default=False,
        help=(
            "Disable all os.fsync() calls in the worker. "
            "Data on disk is then only guaranteed by the OS page cache; "
            "a SIGKILL mid-download can leave an empty or truncated file. "
            "Use for max throughput only."
        ),
    )
    parser.add_argument(
        "--chunk-timeout", type=int, default=300,
        help=(
            "Seconds to wait for the next chunk before timing out "
            "(default: 300). If the Rust streaming API blocks longer "
            "than this, the worker cancels the stream and reports a "
            "ChunkTimeout error. Set to 0 to disable timeout (not "
            "recommended)."
        ),
    )
    return parser.parse_args()


# -- Main --------------------------------------------------------


def main() -> int:
    # Late import: add project src to path only when run as a script
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from hf_track import (
        HfTracker,
        TransferCancelledError,
        is_xet_available,
    )
    from progress_bar import ConsoleProgressDisplay

    args = parse_args()

    if args.no_xet:
        # Snapshot streaming is the xet code path. With --no-xet the
        # snapshot is small text files; fall back to download_snapshot.
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        print("[!] --no-xet: falling back to HfTracker.download_snapshot()")
        tracker = HfTracker(token=args.token, report_interval=0.1)
        display = ConsoleProgressDisplay(filename=args.repo, is_snapshot=True, bar_width=40)
        repo_name = args.repo.split("/")[-1]
        output_dir = args.output or os.path.join(os.path.dirname(__file__), repo_name)
        os.makedirs(output_dir, exist_ok=True)
        # Run in a background thread, drain events into the display
        result_holder: dict = {}
        err_holder: dict = {}

        def _do():
            try:
                result_holder["path"] = tracker.download_snapshot(
                    repo_id=args.repo,
                    repo_type=args.repo_type,
                    allow_patterns=args.allow_patterns,
                    ignore_patterns=args.ignore_patterns,
                    local_dir=output_dir,
                    force_download=args.force,
                    use_xet=False,
                )
            except Exception as e:  # noqa: BLE001
                err_holder["err"] = e

        t = threading.Thread(target=_do, daemon=True)
        t.start()
        try:
            while t.is_alive() or not tracker.event_queue.empty():
                try:
                    display.update(tracker.event_queue.get(timeout=0.05))
                except queue.Empty:
                    continue
        except KeyboardInterrupt:
            print("\n[STOP] Interrupted by user.")
            return 1
        display.close()
        if err_holder.get("err"):
            print(f"[ERR] {err_holder['err']}")
            return 1
        print(f"[OK] Files saved to: {result_holder.get('path', output_dir)}")
        return 0

    if not is_xet_available():
        print("[!] hf_xet is not installed. Install with: pip install hf_xet")
        return 1

    token = args.token or os.environ.get("HF_TOKEN") or None
    if not token:
        print("[!] No HF_TOKEN provided. Public repos will work, but gated repos require a token.")

    repo_name = args.repo.split("/")[-1]
    output_dir = args.output or os.path.join(os.path.dirname(__file__), repo_name)
    os.makedirs(output_dir, exist_ok=True)

    print()
    print("=" * 60)
    print(" HuggingFace XET STREAMING Downloader (subprocess)")
    print("=" * 60)
    print(f" Repository       : {args.repo}")
    print(" Xet              : [OK] streaming API available")
    print(" Mode             : SUBPROCESS streaming (child does chunks -> disk)")
    print(f" fsync interval   : {args.fsync_interval} bytes")
    print(f" chunk timeout    : {args.chunk_timeout}s")
    if args.no_fsync:
        print(" fsync            : [DISABLED] (data only in OS page cache)")
    print("=" * 60)
    print()

    # 1. Set up tracker + display
    tracker = HfTracker(token=token, report_interval=0.1)
    display = ConsoleProgressDisplay(filename=args.repo, is_snapshot=True, bar_width=40)

    # 2. Optional: start memory watcher
    watcher: MemoryWatcher | None = None
    if args.watch_mem:
        mem_csv = os.path.join("tmp", "diag_streaming_mem.csv")
        os.makedirs("tmp", exist_ok=True)
        watcher = MemoryWatcher(output_dir=output_dir, csv_path=mem_csv, interval_s=0.5)
        watcher.start()
        print(f"[MEM] Watching RSS -> {mem_csv}")

    # 3. Run download_snapshot_streaming() in a background thread.
    #    The method itself spawns a child subprocess for the xet files
    #    and drains progress events into tracker.event_queue. We just
    #    drain that queue into the display until the thread is done.
    #
    #    Set XET_CHUNK_TIMEOUT env var so the worker subprocess uses
    #    the user's --chunk-timeout value (the worker reads this env
    #    var on startup).
    if args.chunk_timeout:
        os.environ["XET_CHUNK_TIMEOUT"] = str(args.chunk_timeout)

    result_holder: dict = {}
    err_holder: dict = {}

    def _do():
        try:
            result_holder["paths"] = tracker.download_snapshot_streaming(
                repo_id=args.repo,
                allow_patterns=args.allow_patterns,
                ignore_patterns=args.ignore_patterns,
                repo_type=args.repo_type,
                local_dir=output_dir,
                force_download=args.force,
                fsync_interval=args.fsync_interval,
                disable_fsync=args.no_fsync,
            )
        except TransferCancelledError:
            pass  # silent on user cancel
        except Exception as e:  # noqa: BLE001
            err_holder["err"] = e

    t0 = time.time()
    t = threading.Thread(target=_do, daemon=True)
    t.start()

    # When the child worker is blocked in ``XetDownloadStream.__next__``
    # (the GIL is held while the Rust runtime reconstructs the next
    # xet term), the parent sees no events for several seconds. The
    # display bar therefore looks frozen. To make it obvious the
    # process is alive, we render an animated spinner whenever the
    # event queue has been empty for more than ~250 ms.
    SPINNER = "|/-\\"
    last_event_time = time.time()
    last_spinner_idx = 0
    SPINNER_INTERVAL_S = 0.2
    IDLE_THRESHOLD_S = 0.25

    def _show_spinner() -> None:
        nonlocal last_spinner_idx
        if not t.is_alive():
            return
        elapsed_idle = time.time() - last_event_time
        if elapsed_idle < IDLE_THRESHOLD_S:
            return
        idx = int((time.time() / SPINNER_INTERVAL_S)) % len(SPINNER)
        if idx == last_spinner_idx:
            return
        last_spinner_idx = idx
        sys.stderr.write(f"\r  {SPINNER[idx]} streaming... (waiting for next chunk)")
        sys.stderr.flush()

    try:
        while t.is_alive() or not tracker.event_queue.empty():
            try:
                event = tracker.event_queue.get(timeout=0.05)
                last_event_time = time.time()
                display.update(event)
            except queue.Empty:
                _show_spinner()
                continue
        # Plan 2026-06-05 step 5: trailing-drain. The child subprocess
        # can enqueue a final COMPLETE / ERROR / CANCELLED event AFTER
        # the worker thread has exited (the runner.terminate(grace=...)
        # call in download/xet_streaming.py finally: block publishes
        # these from the main thread, but with a possible few-ms lag).
        # Drain for up to 200 ms to give the display a chance to render
        # the terminal state. This avoids the cosmetic bug where the
        # last event seen is "downloading" instead of "complete".
        drain_deadline = time.time() + 0.2
        while time.time() < drain_deadline:
            try:
                event = tracker.event_queue.get_nowait()
                last_event_time = time.time()
                display.update(event)
            except queue.Empty:
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n\n[STOP] Download interrupted by user.")
        # drain thread will exit on its own; tracker.cancel was triggered
        # implicitly when the inner code raised TransferCancelledError.
        t.join(timeout=2.0)
        if watcher is not None:
            watcher.stop()
        display.close()
        return 1

    t.join(timeout=10.0)
    if watcher is not None:
        watcher.stop()
    display.close()
    elapsed = time.time() - t0

    if err_holder.get("err"):
        print(f"\n[ERR] Download failed: {err_holder['err']}")
        return 1

    paths = result_holder.get("paths") or []
    total_bytes = 0
    for p in paths:
        try:
            total_bytes += os.path.getsize(p)
        except OSError:
            pass
    print()
    print("=" * 60)
    print(f" Files downloaded : {len(paths)}")
    print(f" Total bytes      : {total_bytes / (1024*1024):.1f} MB")
    print(f" Elapsed          : {elapsed:.1f}s")
    if total_bytes and elapsed > 0:
        print(f" Throughput       : {total_bytes / (1024*1024) / elapsed:.1f} MB/s")
    print(f" Output dir       : {output_dir}")
    print("=" * 60)
    print()
    return 0 if paths else 1


if __name__ == "__main__":
    raise SystemExit(main())
