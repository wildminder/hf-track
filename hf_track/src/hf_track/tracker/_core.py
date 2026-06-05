"""Core state and cancellation logic for HfTracker.

This mixin owns the tracker's state (token, endpoint, event queue,
cancelled-transfers set) and the synchronization primitives. It does
not know anything about download/upload operations — those are added
by the sibling mixins (``_downloads``, ``_uploads``).

Split from the original ``tracker.py`` (28KB god class) on 2026-06-05
as part of the modular-refactor plan
(docs/plans/2026-06-04-modular-refactor.md, Step 8).
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

from ..token import XetTokenManager
from ..types import ProgressEvent, generate_transfer_id

logger = logging.getLogger(__name__)


class _TrackerCore:
    """Mixin: state + cancellation primitives shared by all operations.

    Subclasses are expected to inherit from this alongside the
    operation-specific mixins (``_downloads``, ``uploads``, etc.) to
    form the public :class:`HfTracker` class.

    Attributes set here:
        _token: Optional HuggingFace token used for auth.
        _endpoint: Optional custom endpoint URL.
        _report_interval: Seconds between event reports.
        event_queue: Thread-safe queue of :class:`ProgressEvent`.
        _token_manager: Xet token manager (auto-refresh).
        _cancelled_transfers: Set of transfer IDs the user requested to cancel.
        _lock: Guards ``_cancelled_transfers`` for thread-safety.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        endpoint: Optional[str] = None,
        report_interval: float = 0.1,
    ) -> None:
        self._token = token
        self._endpoint = endpoint
        self._report_interval = report_interval
        self.event_queue: queue.Queue[ProgressEvent] = queue.Queue(maxsize=10000)
        self._token_manager = XetTokenManager(token, endpoint)
        self._cancelled_transfers: set[str] = set()
        # Plan 2026-06-05 step 3: registry of active
        # ``XetSubprocessRunner`` instances keyed by transfer_id. Allows
        # ``cancel(transfer_id)`` to forward the cancel signal to the
        # child process immediately, without waiting for the parent's
        # 1 s poll loop.
        self._active_runners: dict[str, "XetSubprocessRunner"] = {}
        self._lock = threading.Lock()

    def cancel(self, transfer_id: str) -> None:
        """Cancel an active transfer by its transfer_id.

        Marks the transfer for cancellation and, if a runner is
        currently registered for this transfer (i.e. a streaming or
        subprocess-isolated download is in flight), forwards the
        cancel to the child process immediately by calling
        ``runner.request_cancel()``. The child breaks out of its
        loop cooperatively on the next chunk boundary. If the child
        is GIL-stalled, the caller is responsible for following up
        with a hard ``runner.terminate(grace=2.0)`` after a short
        grace period.
        """
        with self._lock:
            self._cancelled_transfers.add(transfer_id)
            runner = self._active_runners.get(transfer_id)
        if runner is not None:
            try:
                runner.request_cancel()
            except Exception:
                # Best-effort: if request_cancel raises, the parent
                # poll loop will catch the cancel on the next iteration
                # (within 1 s). Don't propagate the error.
                pass

    def is_cancelled(self, transfer_id: str) -> bool:
        """Check if a transfer has been cancelled.

        Thread-safe — uses the same lock as :meth:`cancel`.
        """
        with self._lock:
            return transfer_id in self._cancelled_transfers

    def cleanup_transfer(self, transfer_id: str) -> None:
        """Remove a transfer from tracking sets upon completion.

        Called from the ``finally`` block of every public download/upload
        method so the cancelled-transfers set does not grow unbounded.
        """
        with self._lock:
            self._cancelled_transfers.discard(transfer_id)

    def _prepare_transfer(
        self, transfer_id: Optional[str]
    ) -> tuple[str, Callable[[], bool]]:
        """Generate a transfer ID and cancellation hook.

        Returns:
            Tuple of (transfer_id, is_cancelled_hook) — used by all
            public download/upload methods.
        """
        transfer_id = transfer_id or generate_transfer_id()

        def _cancelled_hook() -> bool:
            return self.is_cancelled(transfer_id)

        is_cancelled_hook = _cancelled_hook
        return transfer_id, is_cancelled_hook
