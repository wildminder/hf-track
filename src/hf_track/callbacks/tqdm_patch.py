"""``tqdm_upload_patcher`` — context manager that instruments tqdm for uploads.

While uploads are running, ``huggingface_hub`` instantiates ``tqdm``
bars to display file-upload progress. By default these bars are decoupled
from our ``ProgressEvent`` stream. This module provides a context manager
that:

1. Replaces ``tqdm.auto.tqdm`` with a subclass (``UploadProgressTqdm``)
   that knows about the upload state (transfer_id, total bytes,
   filename) and routes byte-bar updates into the shared
   ``state_manager``.
2. On exit (or exception), restores the original ``tqdm.auto.tqdm``
   class so other unrelated uses of tqdm are unaffected.

The patcher is the mirror of :class:`DownloadProgressTqdm` for the
upload side, but is built as a context manager rather than a class
because the Xet/standard upload flows do not provide a clean
"pre-bound class" hook — they construct ``tqdm()`` directly inside
``huggingface_hub`` code we don't own.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import time
from typing import Callable, Optional

from ..types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferCancelledError,
    TransferDirection,
)
from .state import state_manager
from .xet_callback import _dummy_file

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def tqdm_upload_patcher(
    event_queue: queue.Queue,
    transfer_id: str = "",
    filename: str = "",
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
    total_bytes: int = 0,
):
    """Context manager: route tqdm progress into ``state_manager``.

    Args:
        event_queue: Queue that the patched tqdm will write ``ProgressEvent``
            instances to (START on first byte bar init, PROGRESS for each
            update, COMPLETE on bar close).
        transfer_id: Unique transfer id (used to look up state and tag events).
        filename: Display filename for the upload.
        report_interval: Throttle interval in seconds (currently unused —
            emission is driven by ``update(n)`` calls, not time).
        is_cancelled: Optional callable returning True to abort.
        total_bytes: Expected total upload size (used for START event and
            percentage calculation).

    Yields:
        ``None``. The patch is active for the duration of the ``with``
        block and reverted on exit (even on exceptions).
    """
    import tqdm.auto as tqdm_auto_module

    original_tqdm = tqdm_auto_module.tqdm
    _patch_active = True

    # Register this upload in the state manager.
    state_manager.init_upload(transfer_id, filename, total_bytes, event_queue)

    class UploadProgressTqdm(original_tqdm):
        """Patched tqdm that mirrors byte-bar progress into events.

        Only byte-unit bars (``unit in ("B", "iB")``) are instrumented;
        other tqdm uses (count bars, indeterminate spinners) are left
        untouched to avoid corrupting unrelated progress displays.
        """

        def __init__(self, *args, **kwargs):
            kwargs["file"] = _dummy_file
            super().__init__(*args, **kwargs)

            self._managed_state = state_manager.get_state(transfer_id)

            self._upload_is_file_bar = (
                getattr(self, "total", None) is not None
                and getattr(self, "total", 0) > 0
                and getattr(self, "unit", "") in ("B", "iB")
            )

            self._last_n = 0

            if (
                getattr(self, "_upload_is_file_bar", False)
                and self._managed_state
                and self._managed_state["bytes_completed"] == 0
            ):
                event = ProgressEvent(
                    event_type=EventType.START,
                    transfer_id=transfer_id,
                    direction=TransferDirection.UPLOAD,
                    filename=self._managed_state["filename"],
                    phase=ProgressPhase.UPLOADING,
                    total_bytes=self._managed_state["total_bytes"],
                )
                try:
                    event_queue.put_nowait(event)
                except queue.Full:
                    logger.warning(
                        "Upload progress event queue full — dropping %s event",
                        event.event_type.value,
                    )

        def update(self, n=1):
            if is_cancelled is not None and is_cancelled():
                raise TransferCancelledError("Transfer cancelled by user")

            result = super().update(n)

            if (
                not getattr(self, "_upload_is_file_bar", False)
                or not _patch_active
                or n == 0
            ):
                return result

            current_n = getattr(self, "n", 0)
            delta = current_n - self._last_n
            self._last_n = current_n

            if delta > 0:
                state = state_manager.add_upload_bytes(transfer_id, delta)
                if state:
                    bytes_completed = state["bytes_completed"]
                    total_bytes_local = state["total_bytes"]
                    percentage = (
                        (bytes_completed / total_bytes_local * 100)
                        if total_bytes_local > 0
                        else 0
                    )

                    now = time.time()
                    elapsed = now - state["start_time"]
                    speed = bytes_completed / elapsed if elapsed > 0 else 0

                    event = ProgressEvent(
                        event_type=EventType.PROGRESS,
                        transfer_id=transfer_id,
                        direction=TransferDirection.UPLOAD,
                        filename=state["filename"],
                        phase=ProgressPhase.UPLOADING,
                        bytes_completed=bytes_completed,
                        total_bytes=total_bytes_local,
                        percentage=percentage,
                        speed=speed,
                    )
                    try:
                        state["event_queue"].put_nowait(event)
                    except queue.Full:
                        logger.warning(
                            "Upload progress event queue full — dropping %s event",
                            event.event_type.value,
                        )
            return result

        def close(self):
            if (
                not getattr(self, "_upload_is_file_bar", False)
                or not _patch_active
            ):
                super().close()
                return

            state = state_manager.get_state(transfer_id)
            if state and not state.get("completed_emitted", False):
                if state["bytes_completed"] >= state["total_bytes"]:
                    state_manager.mark_upload_completed(transfer_id)
                    event = ProgressEvent(
                        event_type=EventType.COMPLETE,
                        transfer_id=transfer_id,
                        direction=TransferDirection.UPLOAD,
                        filename=state["filename"],
                        phase=ProgressPhase.COMPLETE,
                        bytes_completed=state["bytes_completed"],
                        total_bytes=state["total_bytes"],
                        percentage=100.0,
                    )
                    state["event_queue"].put(event)

            super().close()

    tqdm_auto_module.tqdm = UploadProgressTqdm
    try:
        yield
    finally:
        _patch_active = False
        tqdm_auto_module.tqdm = original_tqdm
