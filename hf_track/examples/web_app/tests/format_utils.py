"""Python reference implementations of the JS formatting functions.

These mirror the logic in app.js. If you change the logic here,
also update app.js.
"""
from __future__ import annotations


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
    if seconds <= 0 or seconds == float("inf") or seconds != seconds:  # NaN check
        return "--:--"
    total_secs = int(seconds)
    hrs = total_secs // 3600
    mins = (total_secs % 3600) // 60
    secs = total_secs % 60
    if hrs > 0:
        return f"{hrs}h{mins:02d}m{secs:02d}s"
    if mins > 0:
        return f"{mins}m{secs:02d}s"
    return f"{secs}s"


def format_percentage(pct: float) -> str:
    return f"{pct:.1f}%"
