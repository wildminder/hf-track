"""The in-process driver around ``download_hybrid`` (NTH-016, step S32).

``TranslatingQueue`` adapts ``download_hybrid``'s dictionary progress
channel onto the event queue the rest of the library speaks, and
``HybridRunner`` is the ``run()``/``cancel()``/``terminate()`` surface
that mirrors ``XetSubprocessRunner`` so the streaming path can be swapped
in without the caller noticing.

Separated from ``_xet_hybrid.py`` because the dependency only points one
way: the runner drives the driver, never the reverse.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from typing import Any, Dict, List, Optional

from ._xet_hybrid import download_hybrid
from .subprocess.messages import SubprocessMessage
from .types import ProgressEvent

logger = logging.getLogger(__name__)


_xet_file_download_worker = download_hybrid


class TranslatingQueue:
    """Queue adapter that converts dict events into ``ProgressEvent`` objects.

    ``download_hybrid`` emits plain dicts (see ``ProgressEvent.to_dict()``)
    into its ``progress_queue``. The public API contract, however, promises
    ``ProgressEvent`` objects in ``HfTracker.event_queue`` — and the example
    ``ConsoleProgressDisplay.update()`` accesses ``.event_type`` etc. directly.

    This adapter wraps the user-supplied queue and transparently translates
    any dict it sees into a ``ProgressEvent`` (via ``ProgressEvent.from_dict``)
    before forwarding. Non-dict items (already-``ProgressEvent`` objects, or
    anything else) pass through unchanged.

    Only ``put`` / ``put_nowait`` are overridden; the consumer side
    (``get`` / ``get_nowait`` / ``empty``) is delegated so the driver loop in
    the caller keeps working unchanged.
    """

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    def _translate(self, item: Any) -> Any:
        if isinstance(item, dict) and "event_type" in item:
            try:
                return ProgressEvent.from_dict(item)
            except Exception:
                # If the dict is malformed, pass it through untouched so
                # the caller can decide (avoids swallowing real errors).
                return item
        return item

    def put(self, item: Any, *args, **kwargs) -> None:
        self._wrapped.put(self._translate(item), *args, **kwargs)

    def put_nowait(self, item: Any, *args, **kwargs) -> None:
        self._wrapped.put_nowait(self._translate(item), *args, **kwargs)

    def get(self, *args, **kwargs) -> Any:
        return self._wrapped.get(*args, **kwargs)

    def get_nowait(self, *args, **kwargs) -> Any:
        return self._wrapped.get_nowait(*args, **kwargs)

    def empty(self) -> bool:
        return self._wrapped.empty()

    def qsize(self) -> int:
        return self._wrapped.qsize()


class HybridRunner:
    """Simple IN-PROCESS runner for ``download_hybrid`` (plan 2026-06-15).

    Uses a daemon ``threading.Thread`` instead of a spawned subprocess.
    No GIL-watching, no ``multiprocessing`` pickling, and no relay
    thread: progress events go straight into the user-supplied
    ``event_queue`` (a ``queue.Queue``). Cancellation is via a
    ``threading.Event``.

    The runner exposes ``wait(timeout)`` like the legacy
    ``XetSubprocessRunner`` so the driver loop in
    ``download_snapshot_streaming`` does not need to be rewritten.
    """

    def __init__(self) -> None:
        import threading as _t
        self._thread: Optional["_t.Thread"] = None
        self._cancel_event = _t.Event()
        self._result: Optional[Dict[str, Any]] = None
        self._error: Optional[BaseException] = None
        self._done_event = _t.Event()

    def start(self, params: Dict[str, Any], event_queue: Any) -> None:
        import threading as _t
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("HybridRunner already started")

        # ``file_specs`` lives inside params; mirror the field into a
        # top-level arg for ``download_hybrid``.
        call_kwargs = dict(params)
        file_specs = call_kwargs.pop("file_specs", None)
        if file_specs is None:
            raise ValueError("HybridRunner.start requires params['file_specs']")

        # ``download_hybrid`` emits plain dict events into its
        # ``progress_queue``. The public API contract promises
        # ``ProgressEvent`` objects in the user's queue (and the example
        # display accesses ``.event_type`` etc. directly), so wrap the
        # queue with a translator that converts dicts → ProgressEvent.
        translating_queue = TranslatingQueue(event_queue)

        def _runner():
            try:
                summary = download_hybrid(
                    file_specs,
                    progress_queue=translating_queue,
                    cancel_event=self._cancel_event,
                    **call_kwargs,
                )
                self._result = summary
            except BaseException as exc:  # noqa: BLE001
                self._error = exc
            finally:
                self._done_event.set()

        self._thread = _t.Thread(target=_runner, daemon=True, name="hybrid-runner")
        self._thread.start()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def wait(self, timeout: Optional[float] = 1.0) -> Optional[Dict[str, Any]]:
        if timeout is None:
            self._done_event.wait()
        else:
            self._done_event.wait(timeout=timeout)
        if not self._done_event.is_set():
            return None
        if self._error is not None:
            return {
                "status": "error",
                "message": str(self._error),
                "error_type": type(self._error).__name__,
            }
        summary = self._result or {}
        if summary.get("errors"):
            return {
                "status": "error",
                "message": "; ".join(summary["errors"]),
                "error_type": "AllTiersFailed",
                "bytes_completed": summary.get("bytes_completed", 0),
                "total_bytes": summary.get("total_bytes", 0),
            }
        if summary.get("cancelled"):
            return {
                "status": "cancelled",
                "message": "Transfer cancelled by user",
                "bytes_completed": summary.get("bytes_completed", 0),
                "total_bytes": summary.get("total_bytes", 0),
            }
        return {
            "status": "success",
            "bytes_completed": summary.get("bytes_completed", 0),
            "total_bytes": summary.get("total_bytes", 0),
            "files_completed": summary.get("files_completed", 0),
            "total_files": summary.get("total_files", 0),
        }

    def terminate(self, grace: Optional[float] = None) -> None:  # noqa: ARG002
        """Best-effort termination: set the cancel flag and wait."""
        self.request_cancel()
        if self._thread is not None:
            self._thread.join(timeout=grace if grace is not None else 1.0)
