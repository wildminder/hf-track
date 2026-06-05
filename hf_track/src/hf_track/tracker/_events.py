"""Event-consumer methods for HfTracker.

This mixin provides the consumer-side API for :class:`HfTracker`:
drain the event queue, iterate events as a generator, or block until
a specific transfer completes. The producer-side (pushing events
into ``self.event_queue``) is handled by the download/upload
operations and worker subprocesses — see ``_downloads`` and
``_uploads``.

Split from the original ``tracker.py`` (28KB god class) on 2026-06-05
as part of the modular-refactor plan
(docs/plans/2026-06-04-modular-refactor.md, Step 8).
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Generator, List, Optional

from ..types import EventType, ProgressEvent

logger = logging.getLogger(__name__)


class _TrackerEvents:
    """Mixin: event-consumer methods for HfTracker.

    Provides:
        get_events(timeout=0): Drain the queue (non-blocking by default).
        events(timeout, stop_on): Iterate events as a generator.
        wait_for_complete(transfer_id, timeout): Block until a transfer
            emits COMPLETE/ERROR/CANCELLED.

    Subclasses must define ``self.event_queue`` — provided by
    :class:`_TrackerCore`.
    """

    def get_events(self, timeout: float = 0) -> List[ProgressEvent]:
        """Return all currently queued events.

        Args:
            timeout: Seconds to wait for the FIRST event. Subsequent
                events are drained non-blocking (timeout=0). Pass a
                positive value to block until at least one event
                arrives (or the timeout elapses).
        """
        events: List[ProgressEvent] = []
        while True:
            try:
                event = self.event_queue.get(timeout=timeout)  # type: ignore[attr-defined]
                events.append(event)
                timeout = 0
            except queue.Empty:
                break
        return events

    def events(
        self, timeout: float = 1.0, stop_on: Optional[EventType] = None
    ) -> Generator[ProgressEvent, None, None]:
        """Yield events as they arrive.

        Args:
            timeout: Seconds to wait for the next event. The generator
                loops forever until the consumer breaks out.
            stop_on: Optional event type that ends iteration when
                received (typically ``EventType.COMPLETE``).
        """
        while True:
            try:
                event = self.event_queue.get(timeout=timeout)  # type: ignore[attr-defined]
                yield event
                if stop_on and event.event_type == stop_on:
                    return
            except queue.Empty:
                logger.debug("Queue empty in events generator — polling")
                continue

    def wait_for_complete(
        self,
        transfer_id: str,
        timeout: float = 300,
    ) -> Optional[ProgressEvent]:
        """Block until the given transfer emits a terminal event.

        Returns the COMPLETE/ERROR/CANCELLED :class:`ProgressEvent` for
        the transfer, or ``None`` if the timeout elapses first.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = deadline - time.time()
            try:
                event = self.event_queue.get(timeout=min(remaining, 0.5))  # type: ignore[attr-defined]
                if event.transfer_id == transfer_id:
                    if event.event_type in (
                        EventType.COMPLETE,
                        EventType.ERROR,
                        EventType.CANCELLED,
                    ):
                        return event
            except queue.Empty:
                logger.debug(
                    "Queue empty while waiting for transfer %s — polling",
                    transfer_id,
                )
                continue
        return None
