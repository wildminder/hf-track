"""Retry with exponential backoff for transient network failures (NTH-004).

Both helpers live here rather than in ``xet_file_only.py`` because a retry
policy is not a download concern: the Xet batch path, the upload path and
any future transport all need the same three rules, and a copy per caller
is how the "why is this one different" bugs arrive.

The three rules:

1. **Retry only what is transient.** ``RETRYABLE_ERRORS`` is the
   network-error set. A bad file hash or a revoked token is permanent;
   retrying it just spends the caller's timeout before failing the same
   way.
2. **Back off exponentially, with a cap.** Delay before retry *n* is
   ``min(base * 2**(n-1), max)``. Without the cap, a caller who sets
   ``max_retries=20`` waits minutes for a transfer that was never going to
   succeed.
3. **Stay responsive to cancellation.** :func:`_sleep_unless_cancelled`
   polls in slices so a ``cancel()`` issued during a backoff is observed
   immediately instead of after the remaining delay.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: Base delay for the retry backoff, in seconds. The first retry waits
#: this long, the second twice as long, and so on.
DEFAULT_RETRY_BASE_DELAY_S = 0.5

#: Ceiling on a single backoff delay, so a large ``max_retries`` cannot
#: turn one transfer into an hours-long wait.
DEFAULT_RETRY_MAX_DELAY_S = 8.0

#: Exceptions that mean "the network or the Hub blinked, try again".
#: Anything outside this set is permanent (a bad hash, a missing file, a
#: revoked token) and retrying it only wastes the caller's timeout.
RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def retry_with_backoff(
    operation: Callable[[], Any],
    *,
    max_retries: int = 3,
    base_delay_s: float = DEFAULT_RETRY_BASE_DELAY_S,
    max_delay_s: float = DEFAULT_RETRY_MAX_DELAY_S,
    retryable: tuple[type[BaseException], ...] = RETRYABLE_ERRORS,
    sleep: Optional[Callable[[float], None]] = None,
) -> Any:
    """Call ``operation``, retrying transient failures with exponential backoff.

    ``max_retries`` counts *retries*, not attempts: ``max_retries=3`` means
    up to four calls.

    The delay before retry *n* (1-based) is ``min(base_delay_s * 2**(n-1),
    max_delay_s)`` — one second of waiting buys roughly one extra network
    round trip at these magnitudes, and the cap keeps a caller that sets
    ``max_retries=20`` from waiting minutes for a transfer that is never
    going to succeed.

    An exception outside ``retryable`` propagates immediately: a
    ``TransferCancelledError`` must reach the caller on the first attempt,
    not after the whole backoff schedule has been slept through.

    ``sleep`` defaults to :func:`time.sleep` resolved at call time rather
    than captured as a default argument, so a test that patches
    ``time.sleep`` actually intercepts the backoff instead of sleeping.
    """
    sleep = sleep or time.sleep
    attempts = max(1, max_retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except retryable as exc:
            if attempt == attempts:
                raise
            delay = min(base_delay_s * (2 ** (attempt - 1)), max_delay_s)
            logger.warning(
                "xet operation failed (%s: %s); retry %d/%d in %.2fs",
                type(exc).__name__, exc, attempt, attempts - 1, delay,
            )
            sleep(delay)


def _sleep_unless_cancelled(
    delay_s: float, is_cancelled: Optional[Callable[[], bool]]
) -> None:
    """Sleep between retries, but wake immediately on cancellation.

    A plain ``time.sleep`` here would make ``cancel()`` unresponsive for
    the length of the backoff — up to ``DEFAULT_RETRY_MAX_DELAY_S`` per
    attempt — which is the difference between a cancel that feels instant
    and one that appears to hang. The wait is therefore polled in short
    slices and abandoned as soon as the hook fires.
    """
    if is_cancelled is None:
        time.sleep(delay_s)
        return

    deadline = time.monotonic() + delay_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or is_cancelled():
            return
        time.sleep(min(0.05, remaining))
