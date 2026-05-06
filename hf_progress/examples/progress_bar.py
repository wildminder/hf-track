"""Shared console progress bar renderer for hf-progress examples.

Renders a rich, animated progress bar in the terminal using only
standard library (no external deps beyond hf-progress).
"""

from __future__ import annotations

import sys
import time

from hf_progress import ProgressEvent, EventType, ProgressPhase


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            if unit == "B":
                return f"{n:.0f} {unit}"
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def format_speed(bytes_per_sec: float) -> str:
    return f"{format_bytes(bytes_per_sec)}/s"


def format_eta(seconds: float) -> str:
    if seconds <= 0 or seconds == float("inf"):
        return "--:--"
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h{mins:02d}m{secs:02d}s"
    if mins > 0:
        return f"{mins}m{secs:02d}s"
    return f"{secs}s"


BAR_FILL = "#"
BAR_EMPTY = "-"
BAR_LEFT = "["
BAR_RIGHT = "]"


def render_bar(percentage: float, width: int = 30) -> str:
    filled = int(width * min(percentage, 100.0) / 100.0)
    remaining = width - filled
    return f"{BAR_LEFT}{BAR_FILL * filled}{BAR_EMPTY * remaining}{BAR_RIGHT}"


def render_progress_line(event: ProgressEvent, bar_width: int = 30) -> str:
    if event.phase == ProgressPhase.HASHING:
        label = "hash "
    elif event.phase == ProgressPhase.DOWNLOADING:
        label = "down "
    elif event.phase == ProgressPhase.UPLOADING:
        label = "up "
    elif event.phase == ProgressPhase.VERIFYING:
        label = "verify"
    elif event.phase == ProgressPhase.COMPLETE:
        label = "done "
    elif event.phase == ProgressPhase.ERROR:
        label = "error"
    else:
        label = "      "

    bar = render_bar(event.percentage, bar_width)
    pct = f"{event.percentage:5.1f}%"

    completed = format_bytes(event.bytes_completed)
    total = format_bytes(event.total_bytes) if event.total_bytes > 0 else "???"
    byte_str = f"{completed}/{total}"

    # Add file count natively!
    if event.total_files > 1:
        byte_str = f"[{event.file_index}/{event.total_files} files] {byte_str}"

    speed_str = format_speed(event.speed) if event.speed > 0 else ""
    
    net_str = ""
    if event.transfer_speed > 0:
        net_str = f"net:{format_speed(event.transfer_speed)}"

    eta_str = ""
    if event.speed > 0 and event.total_bytes > event.bytes_completed:
        remaining_bytes = event.total_bytes - event.bytes_completed
        eta_secs = remaining_bytes / event.speed
        eta_str = f"ETA {format_eta(eta_secs)}"

    dedup_str = ""
    if event.dedup_saved_bytes > 0:
        dedup_str = f"dedup:{format_bytes(event.dedup_saved_bytes)} saved"

    parts = [f"{label} {bar} {pct} {byte_str}"]
    if speed_str:
        parts.append(speed_str)
    if net_str:
        parts.append(net_str)
    if eta_str:
        parts.append(eta_str)
    if dedup_str:
        parts.append(dedup_str)

    return " ".join(parts)


class ConsoleProgressDisplay:
    def __init__(
        self,
        filename: str,
        is_snapshot: bool = False,
        bar_width: int = 30,
        refresh_interval: float = 0.1,
    ):
        self.filename = filename
        self.bar_width = bar_width
        self._refresh_interval = refresh_interval
        self._last_line_len = 0
        self._start_time = time.time()
        self._last_bytes = 0
        self._last_time = time.time()
        self._last_render_time = 0.0
        self._smoothed_speed = 0.0

        print()
        print(f" >> Downloading: {filename}")
        print()

    def _compute_speed(self, event: ProgressEvent) -> float:
        if event.speed > 0:
            return event.speed
        now = time.time()
        elapsed = now - self._last_time
        if elapsed > 0.2:
            delta_bytes = event.bytes_completed - self._last_bytes
            instant_speed = delta_bytes / elapsed
            alpha = 0.3
            self._smoothed_speed = alpha * instant_speed + (1 - alpha) * self._smoothed_speed
            self._last_bytes = event.bytes_completed
            self._last_time = now
        return self._smoothed_speed

    def update(self, event: ProgressEvent) -> None:
        if event.event_type == EventType.START:
            self._clear_line()
            size_str = format_bytes(event.total_bytes) if event.total_bytes > 0 else "unknown size"
            line = f"  Starting download... ({size_str})"
            sys.stderr.write(f"\r{line}")
            sys.stderr.flush()
            self._last_line_len = len(line)
            return

        if event.event_type == EventType.PROGRESS:
            now = time.time()
            # Render throttling: avoid overwhelming the console IO
            if now - self._last_render_time < self._refresh_interval:
                return
            
            self._last_render_time = now
            speed = event.speed if event.speed > 0 else self._compute_speed(event)
            self._clear_line()
            
            original_speed = event.speed
            event.speed = speed  # type: ignore[attr-defined] — temporarily override dataclass field for rendering
            line = "  " + render_progress_line(event, self.bar_width)
            event.speed = original_speed  # type: ignore[attr-defined] — restore original value

            sys.stderr.write(f"\r{line}")
            sys.stderr.flush()
            self._last_line_len = len(line)
            return

        if event.event_type == EventType.COMPLETE:
            self._clear_line()
            elapsed = time.time() - self._start_time
            avg_speed = event.bytes_completed / elapsed if elapsed > 0 else 0
            size_str = format_bytes(event.bytes_completed)
            
            if event.total_files > 1:
                line = f"  [OK] Complete! {event.total_files} files ({size_str}) downloaded in {format_eta(elapsed)} (avg {format_speed(avg_speed)})"
            else:
                line = f"  [OK] Complete! {size_str} downloaded in {format_eta(elapsed)} (avg {format_speed(avg_speed)})"

            sys.stderr.write(f"\r{line}\n")
            sys.stderr.flush()
            self._last_line_len = 0
            return

        if event.event_type == EventType.CANCELLED:
            self._clear_line()
            size_str = format_bytes(event.bytes_completed)
            line = f" [STOP] Cancelled at {size_str} ({event.percentage:.1f}%)"
            sys.stderr.write(f"\r{line}\n")
            sys.stderr.flush()
            self._last_line_len = 0
            return

        if event.event_type == EventType.ERROR:
            self._clear_line()
            line = f" [ERR] Error: {event.error or 'Unknown error'}"
            sys.stderr.write(f"\r{line}\n")
            sys.stderr.flush()
            self._last_line_len = 0
            return

    def _clear_line(self) -> None:
        if self._last_line_len > 0:
            sys.stderr.write(f"\r{' ' * self._last_line_len}\r")
            sys.stderr.flush()

    def close(self) -> None:
        self._clear_line()