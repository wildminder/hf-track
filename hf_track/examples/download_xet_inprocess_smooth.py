#!/usr/bin/env python3
"""Download a HuggingFace repository IN-PROCESS with a SMOOTH progress bar.

Diagnostic example (variant of ``download_xet_inprocess.py``) that
adds a smooth-progress ticker to interpolate the bar between
sparse xet chunk events.

The base ``download_xet_inprocess.py`` example shows jumps in the
progress bar (e.g. 0% -> 5% -> 10% -> ...) because ``hf_xet`` reports
progress in coarse chunk boundaries (typically 1-4 MB) and the
package's throttler suppresses events that arrive too close in
byte-delta terms. This variant adds a 50 ms daemon ticker that
extrapolates the current byte position from the last reported
``bytes_completed`` and ``speed`` so the bar moves smoothly
between real events.

Usage::

    python download_xet_inprocess_smooth.py
    python download_xet_inprocess_smooth.py --no-xet
    python download_xet_inprocess_smooth.py --repo openbmb/VoxCPM-0.5B
    python download_xet_inprocess_smooth.py --interval 20   # 20ms ticker

The smooth ticker is example-level: it does not modify the
``hf_track`` package. It is purely a UI improvement.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import threading
import time
import uuid
from dataclasses import replace
from typing import Optional

logger = logging.getLogger(__name__)


# ── Smooth Ticker ──────────────────────────────────────────────────


class SmoothTicker:
    """Daemon thread that interpolates progress between real events.

    Real progress events arrive sporadically (every few hundred ms
    to seconds) because ``hf_xet`` only reports at chunk boundaries.
    This ticker runs at a high frequency (default 50 ms = 20 fps)
    and renders a synthetic ``ProgressEvent`` whose ``bytes_completed``
    is estimated as::

        interpolated = last_bytes + speed * elapsed_since_last_event

    Real events from the queue ``override`` the interpolated value
    (i.e. they update ``last_bytes`` / ``speed`` and reset the timer).

    The ticker stops when ``stop()`` is called or when the download
    completes (caller signals via ``set_done()``).

    Thread-safety: all state mutations are protected by ``_lock``.
    Reads from the rendering loop are lock-free (atomic ints/floats
    on CPython) but we still acquire the lock for snapshot reads
    to guarantee a consistent view.
    """

    def __init__(
        self,
        display,  # ConsoleProgressDisplay instance
        interval_ms: int = 50,
    ):
        self._display = display
        self._interval = max(0.005, interval_ms / 1000.0)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._done_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Last known real event state (snapshot of ProgressEvent fields).
        self._has_real_event: bool = False
        self._last_bytes: int = 0
        self._last_speed: float = 0.0
        self._last_total_bytes: int = 0
        self._last_event_time: float = 0.0
        self._last_file_index: int = 0
        self._last_total_files: int = 1
        self._last_filename: str = ""
        self._last_transfer_id: str = ""
        self._last_direction = None  # TransferDirection, set on first real event
        self._last_phase = None  # ProgressPhase, set on first real event
        self._last_transfer_bytes_completed: int = 0
        self._last_transfer_bytes_total: int = 0
        self._last_transfer_speed: float = 0.0
        self._last_dedup_saved_bytes: int = 0

    # ── Lifecycle ──────────────────────────────────────────────────

    def start(self) -> None:
        """Start the daemon ticker thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._done_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="SmoothTicker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 0.5) -> None:
        """Stop the ticker and wait briefly for the thread to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def set_done(self) -> None:
        """Signal that the transfer has finished (COMPLETE/ERROR/CANCELLED)."""
        self._done_event.set()

    # ── Real-event update (called from main thread) ────────────────

    def update_from_real_event(
        self,
        bytes_completed: int,
        total_bytes: int,
        speed: float,
        file_index: int,
        total_files: int,
        filename: str,
        transfer_id: str,
        direction,
        phase,
        transfer_bytes_completed: int = 0,
        transfer_bytes_total: int = 0,
        transfer_speed: float = 0.0,
        dedup_saved_bytes: int = 0,
    ) -> None:
        """Record the latest real progress event.

        The ticker uses these values as the basis for interpolation
        until the next real event arrives.
        """
        now = time.time()
        with self._lock:
            self._has_real_event = True
            self._last_bytes = bytes_completed
            self._last_speed = max(0.0, speed)
            self._last_total_bytes = total_bytes
            self._last_event_time = now
            self._last_file_index = file_index
            self._last_total_files = total_files
            self._last_filename = filename
            self._last_transfer_id = transfer_id
            self._last_direction = direction
            self._last_phase = phase
            self._last_transfer_bytes_completed = transfer_bytes_completed
            self._last_transfer_bytes_total = transfer_bytes_total
            self._last_transfer_speed = max(0.0, transfer_speed)
            self._last_dedup_saved_bytes = dedup_saved_bytes

    # ── Snapshot (called from ticker thread) ───────────────────────

    def _snapshot_interpolated(self) -> Optional["ProgressEvent"]:
        """Return a synthetic ProgressEvent with interpolated bytes.

        Returns ``None`` if no real event has been seen yet.
        """
        # Local import to avoid top-level heavy imports.
        from hf_track import ProgressEvent, EventType, ProgressPhase

        with self._lock:
            if not self._has_real_event:
                return None
            if self._last_direction is None or self._last_phase is None:
                return None

            now = time.time()
            elapsed = max(0.0, now - self._last_event_time)
            speed = self._last_speed

            # Cap interpolated bytes at total_bytes (don't overshoot).
            if self._last_total_bytes > 0:
                estimated = self._last_bytes + int(speed * elapsed)
                if estimated > self._last_total_bytes:
                    estimated = self._last_total_bytes
            else:
                estimated = self._last_bytes

            # Interpolate transfer bytes too (network-side, for xet dedup view).
            if self._last_transfer_bytes_total > 0:
                est_transfer = self._last_transfer_bytes_completed + int(
                    self._last_transfer_speed * elapsed
                )
                if est_transfer > self._last_transfer_bytes_total:
                    est_transfer = self._last_transfer_bytes_total
            else:
                est_transfer = self._last_transfer_bytes_completed

            # Compute percentage.
            if self._last_total_bytes > 0:
                percentage = (estimated / self._last_total_bytes) * 100.0
            else:
                percentage = 0.0

            return ProgressEvent(
                event_type=EventType.PROGRESS,
                transfer_id=self._last_transfer_id,
                direction=self._last_direction,
                filename=self._last_filename,
                phase=self._last_phase,
                bytes_completed=estimated,
                total_bytes=self._last_total_bytes,
                percentage=percentage,
                speed=speed,
                file_index=self._last_file_index,
                total_files=self._last_total_files,
                transfer_bytes_completed=est_transfer,
                transfer_bytes_total=self._last_transfer_bytes_total,
                transfer_speed=self._last_transfer_speed,
                dedup_saved_bytes=self._last_dedup_saved_bytes,
            )

    # ── Ticker loop ────────────────────────────────────────────────

    def _run(self) -> None:
        """Main loop: emit interpolated progress events at ``_interval``."""
        from hf_track import EventType  # local import

        # Stagger the ticker to start after the next event loop tick so
        # the main thread has a chance to call update_from_real_event()
        # before the first interpolation.
        while not self._stop_event.is_set():
            if self._done_event.is_set():
                break
            synthetic = self._snapshot_interpolated()
            if synthetic is not None:
                try:
                    self._display.update(synthetic)
                except Exception as e:
                    # Never let a UI error kill the ticker.
                    logger.debug("SmoothTicker display.update error: %s", e)
            # Use Event.wait for responsive shutdown.
            if self._stop_event.wait(self._interval):
                break


# ── Argparse ──────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace repo IN-PROCESS with a smooth progress bar",
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
        "--interval",
        type=int,
        default=50,
        help="Smooth-ticker interval in milliseconds (default: 50 = 20 fps)",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────


def main() -> int:
    # Late import: add project src to path only when run as a script
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from hf_track import (
        HfTracker,
        EventType,
        ProgressEvent,
        ProgressPhase,
        TransferDirection,
        TransferCancelledError,
        is_xet_available,
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
    print(" HuggingFace IN-PROCESS Downloader (SMOOTH BAR)")
    print("=" * 60)
    print(f" Repository : {args.repo}")
    xet_status = "[DISABLED]" if args.no_xet else ("[OK] available" if is_xet_available() else "[--] not installed (using HTTP)")
    print(f" Xet        : {xet_status}")
    print(f" Mode       : IN-PROCESS (no subprocess isolation)")
    print(f" Bar FPS    : {1000 // max(1, args.interval)}")
    print("=" * 60)
    print()
    print("WARNING: hf_xet Rust extension will be loaded in-process.")
    print("         Ctrl+C may not work. Use download_repo.py for production.")
    print()

    tracker = HfTracker(token=token, report_interval=0.1)
    transfer_id = str(uuid.uuid4())

    tqdm_class = DownloadProgressTqdm.bind(
        event_queue=tracker.event_queue,
        transfer_id=transfer_id,
        filename=args.repo,
        report_interval=0.1,
    )

    display = ConsoleProgressDisplay(filename=args.repo, is_snapshot=True, bar_width=40)
    ticker = SmoothTicker(display=display, interval_ms=args.interval)
    ticker.start()

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

            logger.debug("[DIAG-SMOOTH] calling snapshot_download(tqdm_class=...)")
            result_path = snapshot_download(**download_kwargs)  # nosec B615
            logger.debug("[DIAG-SMOOTH] snapshot_download returned: %s", result_path)

            stats = state_manager.get_state(transfer_id)
            complete_event = ProgressEvent(
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
            # Hand the real COMPLETE event to the ticker so it stops cleanly,
            # then forward to the display.
            display.update(complete_event)
            ticker.set_done()
        except TransferCancelledError:
            ticker.set_done()
        except Exception as e:
            error_occurred = e
            ticker.set_done()

    download_thread = threading.Thread(target=do_download, daemon=True)
    download_thread.start()

    try:
        while download_thread.is_alive() or not tracker.event_queue.empty():
            try:
                event = tracker.event_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            # Feed the ticker with real data so its interpolation starts
            # from a known good state.
            ticker.update_from_real_event(
                bytes_completed=event.bytes_completed,
                total_bytes=event.total_bytes,
                speed=event.speed,
                file_index=event.file_index,
                total_files=event.total_files,
                filename=event.filename,
                transfer_id=event.transfer_id,
                direction=event.direction,
                phase=event.phase,
                transfer_bytes_completed=event.transfer_bytes_completed,
                transfer_bytes_total=event.transfer_bytes_total,
                transfer_speed=event.transfer_speed,
                dedup_saved_bytes=event.dedup_saved_bytes,
            )
            # The display is also driven by the ticker (smooth, ~20 fps).
            # We do NOT call display.update(event) here for PROGRESS events
            # so the ticker owns rendering. For START / COMPLETE / ERROR /
            # CANCELLED we forward directly to ensure they appear immediately.
            if event.event_type in (EventType.START, EventType.COMPLETE, EventType.ERROR, EventType.CANCELLED):
                display.update(event)
                if event.event_type in (EventType.COMPLETE, EventType.ERROR, EventType.CANCELLED):
                    ticker.set_done()

    except KeyboardInterrupt:
        print("\n\n [STOP] Download interrupted by user.")
        print(" [WARN] hf_xet is in-process — thread may not respond to cancellation.")
        tracker.cancel(transfer_id)
        ticker.set_done()
        download_thread.join(timeout=2.0)
        ticker.stop()
        return 1

    download_thread.join(timeout=5.0)
    ticker.stop()

    # Drain any final events.
    while not tracker.event_queue.empty():
        try:
            event = tracker.event_queue.get_nowait()
            if event.event_type in (EventType.COMPLETE, EventType.ERROR, EventType.CANCELLED):
                display.update(event)
        except queue.Empty:
            break

    display.close()

    if result_path:
        print(f"\n [DIR] Files saved to: {result_path}")
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
