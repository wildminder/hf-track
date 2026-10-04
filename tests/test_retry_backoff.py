"""Retry with exponential backoff around the Xet session acquisition (NTH-004).

``get_xet_session()`` and the download-group handshake are the first thing
that touches the network on the Xet path, and both fail transiently for
reasons that have nothing to do with the caller: a reset connection, a
proxy timeout, a Hub blip. Before this, any such error surfaced as a hard
``TransferProgressError`` and the caller had to retry the whole download.

The retry is deliberately narrow. Only ``RETRYABLE_ERRORS`` are retried, a
permanent failure propagates on the first attempt, and the backoff is
capped and cancellation-aware — a ``cancel()`` during a backoff must not
wait out the delay.
"""

from __future__ import annotations

from unittest import mock

import pytest

from hf_track.download._retry import (
    DEFAULT_RETRY_BASE_DELAY_S,
    DEFAULT_RETRY_MAX_DELAY_S,
    RETRYABLE_ERRORS,
    _sleep_unless_cancelled,
    retry_with_backoff,
)


class _Recorder:
    """A callable that records its delays and can be told to fail N times."""

    def __init__(self, failures=0, error=ConnectionError("blip")):
        self.failures = failures
        self.error = error
        self.calls = 0
        self.delays = []

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return f"ok-after-{self.calls}"

    def sleep(self, delay):
        self.delays.append(delay)


class TestRetryBackoff:
    """The three behaviours that make a retry loop safe to ship."""

    def test_transient_failure_is_retried(self):
        """Two failures then success, with max_retries=3, is three calls."""
        op = _Recorder(failures=2)
        delays = []

        result = retry_with_backoff(
            op, max_retries=3, sleep=delays.append,
        )

        assert op.calls == 3
        assert result == "ok-after-3"
        assert len(delays) == 2, "one sleep between each pair of attempts"

    def test_permanent_failure_is_not_retried(self):
        """An error outside RETRYABLE_ERRORS propagates on attempt one."""
        op = _Recorder(failures=1, error=ValueError("bad file hash"))
        delays = []

        with pytest.raises(ValueError, match="bad file hash"):
            retry_with_backoff(op, max_retries=5, sleep=delays.append)

        assert op.calls == 1
        assert delays == [], "a permanent failure must not sleep before failing"

    def test_backoff_delays_are_exponential(self):
        """Each retry waits twice as long as the last, up to the cap."""
        op = _Recorder(failures=99)
        delays = []

        with pytest.raises(ConnectionError):
            retry_with_backoff(
                op, max_retries=4, base_delay_s=1.0, sleep=delays.append,
            )

        assert delays == pytest.approx([1.0, 2.0, 4.0, 8.0])

    def test_backoff_delays_are_capped(self):
        """A large retry count cannot turn into a minutes-long wait."""
        op = _Recorder(failures=99)
        delays = []

        with pytest.raises(ConnectionError):
            retry_with_backoff(
                op,
                max_retries=10,
                base_delay_s=1.0,
                max_delay_s=4.0,
                sleep=delays.append,
            )

        assert delays == pytest.approx([1.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
        assert max(delays) <= 4.0

    def test_max_retries_zero_means_one_attempt(self):
        """``max_retries`` counts retries, so 0 is not "no retries allowed"."""
        op = _Recorder(failures=1)
        delays = []

        with pytest.raises(ConnectionError):
            retry_with_backoff(op, max_retries=0, sleep=delays.append)

        assert op.calls == 1
        assert delays == []

    def test_exhausted_retries_reraise_the_last_error(self):
        """The caller sees the real failure, not a wrapper."""
        op = _Recorder(failures=5, error=TimeoutError("hub timeout"))
        delays = []

        with pytest.raises(TimeoutError, match="hub timeout"):
            retry_with_backoff(op, max_retries=2, sleep=delays.append)

        assert op.calls == 3
        assert len(delays) == 2

    def test_every_retryable_class_is_actually_transient(self):
        """The tuple is the network-error set, not something wider."""
        assert ConnectionError in RETRYABLE_ERRORS
        assert TimeoutError in RETRYABLE_ERRORS
        # These must NOT be retried: retrying them turns a clear error
        # into a slow one.
        assert ValueError not in RETRYABLE_ERRORS
        assert TypeError not in RETRYABLE_ERRORS

    def test_defaults_are_ordered(self):
        """A cap below the base would make the schedule meaningless."""
        assert DEFAULT_RETRY_BASE_DELAY_S < DEFAULT_RETRY_MAX_DELAY_S

    def test_uses_the_real_sleep_by_default(self):
        """The seam exists for tests, not to disable the backoff."""
        op = _Recorder(failures=1)
        with mock.patch("hf_track.download._retry.time.sleep") as slept:
            result = retry_with_backoff(op, max_retries=2)
        assert slept.call_count == 1
        assert slept.call_args.args[0] == pytest.approx(DEFAULT_RETRY_BASE_DELAY_S)
        assert result == "ok-after-2"


class TestCancelAwareSleep:
    """A cancel during a backoff must not wait out the delay."""

    def test_sleeps_the_whole_delay_without_a_cancel_hook(self):
        with mock.patch("hf_track.download._retry.time.sleep") as slept:
            _sleep_unless_cancelled(0.5, None)
        assert slept.call_count == 1

    def test_returns_immediately_when_already_cancelled(self):
        with mock.patch("hf_track.download._retry.time.sleep") as slept:
            _sleep_unless_cancelled(5.0, lambda: True)
        assert slept.call_count == 0

    def test_returns_as_soon_as_the_hook_fires(self):
        """Sleeping in slices is what makes the hook observable."""
        state = {"n": 0}

        def _hook():
            state["n"] += 1
            return state["n"] >= 3

        with mock.patch(
            "hf_track.download._retry.time.sleep"
        ) as slept:
            _sleep_unless_cancelled(10.0, _hook)

        assert 0 < slept.call_count < 10, (
            "the delay should be polled in slices, not slept in one call"
        )
        assert slept.call_args.args[0] <= 0.05
