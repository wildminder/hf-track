"""Shared console progress bar renderer for hf-progress examples.

Renders a rich, animated progress bar in the terminal using only
standard library (no external deps beyond hf-progress).

Supports:
- Byte-level progress with percentage
- Transfer speed formatting (KB/s, MB/s, GB/s)
- ETA display
- Xet-specific fields (dedup savings, transfer vs processing bytes)
- Multi-line display for snapshot downloads (file count + bytes)

Windows-compatible: uses only ASCII characters for the progress bar
to avoid encoding issues with non-UTF-8 console code pages.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

from hf_progress import ProgressEvent, EventType, ProgressPhase


# ── Formatting Helpers ──────────────────────────────────────────────

def format_bytes(n: float) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            if unit == "B":
                return f"{n:.0f} {unit}"
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def format_speed(bytes_per_sec: float) -> str:
    """Format transfer speed as human-readable string."""
    return f"{format_bytes(bytes_per_sec)}/s"


def format_eta(seconds: float) -> str:
    """Format estimated time remaining."""
    if seconds <= 0 or seconds == float("inf"):
        return "--:--"
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h{mins:02d}m{secs:02d}s"
    if mins > 0:
        return f"{mins}m{secs:02d}s"
    return f"{secs}s"


def get_terminal_width() -> int:
    """Get terminal width, defaulting to 80."""
    try:
        return os.get_terminal_size(sys.stderr.fileno()).columns
    except (AttributeError, ValueError, OSError):
        return 80


# ── Progress Bar Rendering ──────────────────────────────────────────

# ASCII-safe characters (works on all Windows code pages)
BAR_FILL = "#"
BAR_EMPTY = "-"
BAR_LEFT = "["
BAR_RIGHT = "]"


def render_bar(
    percentage: float,
    width: int = 30,
    fill: str = BAR_FILL,
    empty: str = BAR_EMPTY,
) -> str:
    """Render a visual progress bar string.

    Args:
        percentage: Completion percentage (0-100).
        width: Character width of the bar interior.
        fill: Character for filled portion.
        empty: Character for empty portion.

    Returns:
        Rendered bar string like ``[######------------]``
    """
    filled = int(width * min(percentage, 100.0) / 100.0)
    remaining = width - filled
    return f"{BAR_LEFT}{fill * filled}{empty * remaining}{BAR_RIGHT}"


def render_progress_line(
    event: ProgressEvent,
    bar_width: int = 30,
    phase_label: Optional[str] = None,
) -> str:
    """Render a single progress line from a ProgressEvent.

    Layout::

        down  [##################------------]  67.3%  350.0 MB/520.0 MB  12.1 MB/s  ETA 14s

    For snapshot downloads (file-count bars), the layout is::

        files [##################------------]  5/12 files  12.1 MB/s
    """
    # Phase label
    if phase_label:
        label = phase_label
    elif event.phase == ProgressPhase.HASHING:
        label = "hash "
    elif event.phase == ProgressPhase.DOWNLOADING:
        label = "down "
    elif event.phase == ProgressPhase.UPLOADING:
        label = "up   "
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

    # Byte progress
    completed = format_bytes(event.bytes_completed)
    total = format_bytes(event.total_bytes) if event.total_bytes > 0 else "???"
    byte_str = f"{completed}/{total}"

    # Speed
    speed_str = format_speed(event.speed) if event.speed > 0 else ""

    # ETA (only if we have speed and remaining bytes)
    eta_str = ""
    if event.speed > 0 and event.total_bytes > event.bytes_completed:
        remaining_bytes = event.total_bytes - event.bytes_completed
        eta_secs = remaining_bytes / event.speed
        eta_str = f"ETA {format_eta(eta_secs)}"

    # Xet dedup info
    dedup_str = ""
    if event.dedup_saved_bytes > 0:
        dedup_str = f"dedup:{format_bytes(event.dedup_saved_bytes)} saved"

    parts = [f"{label} {bar} {pct}  {byte_str}"]
    if speed_str:
        parts.append(speed_str)
    if eta_str:
        parts.append(eta_str)
    if dedup_str:
        parts.append(dedup_str)

    return "  ".join(parts)


def render_snapshot_line(
    event: ProgressEvent,
    bar_width: int = 30,
) -> str:
    """Render a snapshot download progress line (file-count based).

    Layout::

        files [##################------------]  5/12 files
    """
    label = "files"
    bar = render_bar(event.percentage, bar_width)

    if event.total_files > 1:
        file_str = f"{event.file_index}/{event.total_files} files"
    else:
        file_str = ""

    speed_str = format_speed(event.speed) if event.speed > 0 else ""

    parts = [f"{label} {bar}"]
    if file_str:
        parts.append(file_str)
    if speed_str:
        parts.append(speed_str)

    return "  ".join(parts)


# ── Console Display ─────────────────────────────────────────────────

class ConsoleProgressDisplay:
    """Animated console progress display for HuggingFace transfers.

    Renders a live-updating progress bar in the terminal with:
    - Visual bar fill
    - Percentage, bytes completed/total
    - Transfer speed
    - ETA
    - Xet dedup savings (when available)

    Supports **timer-based refresh**: even when no new ProgressEvent arrives
    (e.g., between HTTP chunk downloads), the display re-renders at a
    configurable interval using the latest known state. This prevents the
    "frozen then jump" appearance caused by infrequent ``update()`` calls.

    Usage::

        display = ConsoleProgressDisplay(filename="model.safetensors")
        for event in tracker.events(stop_on=EventType.COMPLETE):
            display.update(event)
        display.close()
    """

    def __init__(
        self,
        filename: str,
        is_snapshot: bool = False,
        bar_width: int = 30,
        refresh_interval: float = 0.1,
    ):
        self.filename = filename
        self.is_snapshot = is_snapshot
        self.bar_width = bar_width
        self._last_line_len = 0
        self._start_time = time.time()
        self._last_bytes = 0
        self._last_time = time.time()
        self._smoothed_speed = 0.0
        self._refresh_interval = refresh_interval
        self._last_render_time = 0.0

        # Latest event state for timer-based refresh
        self._latest_event: Optional[ProgressEvent] = None

        # Print header
        print()
        print(f" >> Downloading: {filename}")
        print()

    def _compute_speed(self, event: ProgressEvent) -> float:
        """Compute smoothed speed from event data."""
        # Use the event's speed if available (from Xet callbacks)
        if event.speed > 0:
            return event.speed

        # Otherwise compute from elapsed time and bytes
        now = time.time()
        elapsed = now - self._last_time
        if elapsed > 0.2:  # Update every 200ms
            delta_bytes = event.bytes_completed - self._last_bytes
            instant_speed = delta_bytes / elapsed
            # Exponential moving average
            alpha = 0.3
            self._smoothed_speed = alpha * instant_speed + (1 - alpha) * self._smoothed_speed
            self._last_bytes = event.bytes_completed
            self._last_time = now

        return self._smoothed_speed

    def update(self, event: ProgressEvent) -> None:
        """Update the display with a new progress event."""
        # Store latest event for timer-based refresh
        self._latest_event = event

        if event.event_type == EventType.START:
            # Start event -- just show initial message
            self._clear_line()
            if self.is_snapshot:
                line = f"  Preparing to download repository..."
            else:
                size_str = format_bytes(event.total_bytes) if event.total_bytes > 0 else "unknown size"
                line = f"  Starting download... ({size_str})"
            sys.stderr.write(f"\r{line}")
            sys.stderr.flush()
            self._last_line_len = len(line)
            return

        if event.event_type == EventType.PROGRESS:
            # Compute speed if not provided
            if event.speed == 0:
                speed = self._compute_speed(event)
            else:
                speed = event.speed

            self._clear_line()

            if self.is_snapshot:
                line = "  " + render_snapshot_line(event, self.bar_width)
            else:
                # Use computed speed for rendering if event speed is 0
                if event.speed == 0 and speed > 0:
                    # Temporarily set speed for rendering
                    original_speed = event.speed
                    event.speed = speed  # type: ignore[misc]
                    line = "  " + render_progress_line(event, self.bar_width)
                    event.speed = original_speed  # type: ignore[misc]
                else:
                    line = "  " + render_progress_line(event, self.bar_width)

            sys.stderr.write(f"\r{line}")
            sys.stderr.flush()
            self._last_line_len = len(line)
            return

        if event.event_type == EventType.COMPLETE:
            self._clear_line()
            elapsed = time.time() - self._start_time
            avg_speed = event.bytes_completed / elapsed if elapsed > 0 else 0

            if self.is_snapshot:
                line = f"  [OK] Complete! {event.file_index} files downloaded in {format_eta(elapsed)} (avg {format_speed(avg_speed)})"
            else:
                size_str = format_bytes(event.bytes_completed)
                line = f"  [OK] Complete! {size_str} downloaded in {format_eta(elapsed)} (avg {format_speed(avg_speed)})"

            sys.stderr.write(f"\r{line}\n")
            sys.stderr.flush()
            self._last_line_len = 0
            return

        if event.event_type == EventType.ERROR:
            self._clear_line()
            line = f"  [ERR] Error: {event.error or 'Unknown error'}"
            sys.stderr.write(f"\r{line}\n")
            sys.stderr.flush()
            self._last_line_len = 0
            return

    def _clear_line(self) -> None:
        """Clear the previous line content."""
        if self._last_line_len > 0:
            sys.stderr.write(f"\r{' ' * self._last_line_len}\r")
            sys.stderr.flush()

    def close(self) -> None:
        """Clean up the display."""
        self._clear_line()
