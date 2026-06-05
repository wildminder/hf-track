"""Unit tests for the SmoothTicker class.

SmoothTicker is defined in ``examples/download_xet_inprocess_smooth.py``.
To make it testable without running the example as a script, we
import it directly with the example directory added to ``sys.path``.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

# Make the example importable.
_EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "examples",
)
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

# Late import after path manipulation.
from download_xet_inprocess_smooth import SmoothTicker  # noqa: E402

from hf_track import EventType, ProgressEvent, ProgressPhase, TransferDirection  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────


class _FakeDisplay:
    """Stand-in for ConsoleProgressDisplay that records update() calls."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []
        self._lock = threading.Lock()

    def update(self, event: ProgressEvent) -> None:
        with self._lock:
            self.events.append(event)

    @property
    def last_event(self) -> ProgressEvent | None:
        with self._lock:
            return self.events[-1] if self.events else None


def _real_event(
    *,
    bytes_completed: int = 0,
    total_bytes: int = 0,
    speed: float = 0.0,
    file_index: int = 0,
    total_files: int = 1,
    filename: str = "test.bin",
    transfer_id: str = "t1",
    transfer_bytes_completed: int = 0,
    transfer_bytes_total: int = 0,
    transfer_speed: float = 0.0,
) -> ProgressEvent:
    return ProgressEvent(
        event_type=EventType.PROGRESS,
        transfer_id=transfer_id,
        direction=TransferDirection.DOWNLOAD,
        filename=filename,
        phase=ProgressPhase.DOWNLOADING,
        bytes_completed=bytes_completed,
        total_bytes=total_bytes,
        speed=speed,
        file_index=file_index,
        total_files=total_files,
        transfer_bytes_completed=transfer_bytes_completed,
        transfer_bytes_total=transfer_bytes_total,
        transfer_speed=transfer_speed,
    )


def _feed(
    ticker: SmoothTicker,
    *,
    bytes_completed: int,
    total_bytes: int,
    speed: float,
    file_index: int = 0,
    total_files: int = 1,
    transfer_bytes_completed: int = 0,
    transfer_bytes_total: int = 0,
    transfer_speed: float = 0.0,
) -> None:
    """Push a real event into the ticker (does not call display)."""
    ticker.update_from_real_event(
        bytes_completed=bytes_completed,
        total_bytes=total_bytes,
        speed=speed,
        file_index=file_index,
        total_files=total_files,
        filename="test.bin",
        transfer_id="t1",
        direction=TransferDirection.DOWNLOAD,
        phase=ProgressPhase.DOWNLOADING,
        transfer_bytes_completed=transfer_bytes_completed,
        transfer_bytes_total=transfer_bytes_total,
        transfer_speed=transfer_speed,
    )


# ── Tests ────────────────────────────────────────────────────────


def test_ticker_initial_state_has_no_real_event():
    """Before any real event arrives, ticker has no data to interpolate."""
    display = _FakeDisplay()
    ticker = SmoothTicker(display=display, interval_ms=50)
    # No real event yet.
    assert ticker._has_real_event is False  # type: ignore[attr-defined]
    snap = ticker._snapshot_interpolated()
    assert snap is None


def test_ticker_does_not_emit_synthetic_event_before_first_real_event():
    """Without a real event, the ticker should not produce synthetic events."""
    display = _FakeDisplay()
    ticker = SmoothTicker(display=display, interval_ms=20)
    ticker.start()
    time.sleep(0.1)  # Let the ticker loop run ~5 times.
    ticker.stop()
    assert display.events == [], (
        f"Ticker emitted {len(display.events)} synthetic events with no real input"
    )


def test_ticker_interpolates_linearly_between_events():
    """Real event at 50MB/s, total 100MB; after 0.1s, interpolated should be ~5MB."""
    display = _FakeDisplay()
    ticker = SmoothTicker(display=display, interval_ms=20)
    _feed(
        ticker,
        bytes_completed=0,
        total_bytes=100 * 1024 * 1024,  # 100 MB
        speed=50 * 1024 * 1024,  # 50 MB/s
    )
    ticker.start()
    time.sleep(0.1)  # 100ms elapsed → expect ~5MB interpolated
    ticker.stop()

    assert display.events, "Ticker produced no synthetic events after real event"
    last = display.last_event
    assert last is not None
    # Expected: ~5 MB (allow 3-7 MB window for timing jitter).
    interpolated = last.bytes_completed
    assert 3 * 1024 * 1024 <= interpolated <= 7 * 1024 * 1024, (
        f"Interpolated bytes {interpolated} not near expected 5MB"
    )
    # Total bytes must come from the real event (preserved).
    assert last.total_bytes == 100 * 1024 * 1024
    # Speed must come from the real event (preserved).
    assert last.speed == 50 * 1024 * 1024


def test_ticker_caps_interpolation_at_total_bytes():
    """If interpolation would overshoot, it must clamp to total_bytes."""
    display = _FakeDisplay()
    ticker = SmoothTicker(display=display, interval_ms=20)
    _feed(
        ticker,
        bytes_completed=95 * 1024 * 1024,  # 95 MB done
        total_bytes=100 * 1024 * 1024,    # 100 MB total
        speed=50 * 1024 * 1024,           # 50 MB/s
    )
    ticker.start()
    time.sleep(0.5)  # 500ms → 25MB extrapolation, but only 5MB remain
    ticker.stop()

    last = display.last_event
    assert last is not None
    # Must not exceed 100 MB.
    assert last.bytes_completed <= 100 * 1024 * 1024
    # Must be very close to 100 MB (clamped).
    assert last.bytes_completed >= 100 * 1024 * 1024 - 1024  # 1 KB tolerance


def test_ticker_resets_on_real_event_update():
    """A new real event should update last_bytes and reset interpolation base."""
    display = _FakeDisplay()
    ticker = SmoothTicker(display=display, interval_ms=20)
    _feed(
        ticker,
        bytes_completed=0,
        total_bytes=10 * 1024 * 1024,  # 10 MB
        speed=1 * 1024 * 1024,         # 1 MB/s
    )
    ticker.start()
    time.sleep(0.1)  # 100ms → 100KB interpolated
    # Now send a new real event at 5MB.
    _feed(
        ticker,
        bytes_completed=5 * 1024 * 1024,
        total_bytes=10 * 1024 * 1024,
        speed=1 * 1024 * 1024,
    )
    # The next synthetic event should be near 5MB, NOT continuing from old ~100KB+1MB
    time.sleep(0.1)
    ticker.stop()

    last = display.last_event
    assert last is not None
    # Should be near 5MB (real value) or 5MB+0.1s*1MB/s = 5.1MB
    assert 5 * 1024 * 1024 <= last.bytes_completed <= 6 * 1024 * 1024


def test_ticker_stops_on_stop_call():
    """After stop(), no more synthetic events are produced."""
    display = _FakeDisplay()
    ticker = SmoothTicker(display=display, interval_ms=20)
    _feed(ticker, bytes_completed=0, total_bytes=1000, speed=100)
    ticker.start()
    time.sleep(0.1)
    ticker.stop()
    count_after_stop = len(display.events)
    time.sleep(0.1)
    assert len(display.events) == count_after_stop, (
        f"Ticker produced events after stop(): {len(display.events)} vs {count_after_stop}"
    )


def test_ticker_stops_on_done_signal():
    """After set_done(), ticker exits its loop without stop() being called."""
    display = _FakeDisplay()
    ticker = SmoothTicker(display=display, interval_ms=20)
    _feed(ticker, bytes_completed=0, total_bytes=1000, speed=100)
    ticker.start()
    time.sleep(0.1)
    ticker.set_done()
    time.sleep(0.1)  # Give the ticker thread a chance to exit
    count_after_done = len(display.events)
    time.sleep(0.1)
    assert len(display.events) == count_after_done, (
        "Ticker produced events after set_done()"
    )


def test_ticker_handles_zero_speed():
    """With speed=0, interpolated bytes must equal last_bytes (no movement)."""
    display = _FakeDisplay()
    ticker = SmoothTicker(display=display, interval_ms=20)
    _feed(ticker, bytes_completed=500, total_bytes=1000, speed=0)
    ticker.start()
    time.sleep(0.1)
    ticker.stop()

    last = display.last_event
    assert last is not None
    # Speed=0 → no interpolation, bytes stay at 500.
    assert last.bytes_completed == 500


def test_ticker_thread_safety_concurrent_updates():
    """Main thread updates while ticker reads — no exceptions, no corruption."""
    display = _FakeDisplay()
    ticker = SmoothTicker(display=display, interval_ms=10)
    _feed(ticker, bytes_completed=0, total_bytes=10_000_000, speed=1_000_000)
    ticker.start()

    errors: list[BaseException] = []

    def feeder():
        try:
            for i in range(100):
                _feed(
                    ticker,
                    bytes_completed=i * 100_000,
                    total_bytes=10_000_000,
                    speed=1_000_000,
                )
                time.sleep(0.001)
        except BaseException as e:
            errors.append(e)

    t = threading.Thread(target=feeder, daemon=True)
    t.start()
    time.sleep(0.2)
    ticker.stop()
    t.join(timeout=1.0)

    assert not errors, f"Concurrent update raised: {errors}"
    assert display.events, "Ticker produced no events during concurrent updates"


def test_ticker_start_is_idempotent():
    """Calling start() twice should not spawn a second thread."""
    display = _FakeDisplay()
    ticker = SmoothTicker(display=display, interval_ms=20)
    ticker.start()
    first_thread = ticker._thread
    ticker.start()  # Should be a no-op
    assert ticker._thread is first_thread
    ticker.stop()


def test_ticker_zero_interval_uses_floor():
    """interval_ms=0 should clamp to a small positive value (not infinite loop)."""
    display = _FakeDisplay()
    ticker = SmoothTicker(display=display, interval_ms=0)
    assert ticker._interval >= 0.005  # type: ignore[attr-defined]
    _feed(ticker, bytes_completed=0, total_bytes=1000, speed=100)
    ticker.start()
    time.sleep(0.1)
    ticker.stop()
    assert display.events, "Ticker should still emit events with clamped interval"


def test_ticker_preserves_transfer_fields_in_interpolation():
    """Xet-specific transfer_bytes_* fields should be interpolated too."""
    display = _FakeDisplay()
    ticker = SmoothTicker(display=display, interval_ms=20)
    _feed(
        ticker,
        bytes_completed=0,
        total_bytes=100 * 1024 * 1024,
        speed=50 * 1024 * 1024,
        transfer_bytes_completed=0,
        transfer_bytes_total=80 * 1024 * 1024,  # 80MB over the wire (dedup saves 20MB)
        transfer_speed=40 * 1024 * 1024,         # 40MB/s wire speed
    )
    ticker.start()
    time.sleep(0.1)  # 100ms → 4MB wire interpolated
    ticker.stop()

    last = display.last_event
    assert last is not None
    # Wire-side interpolation: 0 + 40MB/s * 0.1s = 4MB
    assert 3 * 1024 * 1024 <= last.transfer_bytes_completed <= 5 * 1024 * 1024
    assert last.transfer_bytes_total == 80 * 1024 * 1024
    assert last.transfer_speed == 40 * 1024 * 1024


def test_ticker_preserves_file_index_and_total_files():
    """file_index/total_files should be carried over from the real event."""
    display = _FakeDisplay()
    ticker = SmoothTicker(display=display, interval_ms=20)
    _feed(
        ticker,
        bytes_completed=0,
        total_bytes=1000,
        speed=100,
        file_index=5,
        total_files=20,
    )
    ticker.start()
    time.sleep(0.1)
    ticker.stop()

    last = display.last_event
    assert last is not None
    assert last.file_index == 5
    assert last.total_files == 20


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
