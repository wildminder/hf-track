"""Thread-safe aggregate state coordinator for transfer progress.

This module owns the ``TransferStateManager`` class and the
``state_manager`` singleton that all progress callbacks (tqdm, Xet,
subprocess workers) read from and write to.

The manager's job is to combine per-bar and per-part signals — which
only know about their own file's bytes — into a single, consistent
aggregate view (files_completed, total_files, bytes_completed, total_bytes)
that can be turned into ``ProgressEvent`` rows for the consumer.

The class is small (one class with a handful of methods) but
deliberately kept in its own module so the threading invariants are
easy to audit and the state is reusable from callbacks, workers, and
the tracker.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Dict


class TransferStateManager:
    """Thread-safe manager for aggregate transfer states.

    Each transfer (identified by ``transfer_id``) carries a dict with:

    - For **downloads**:
        * ``files_completed`` / ``total_files`` — per-file counts
        * ``bytes_completed`` / ``total_bytes`` — aggregate bytes
        * ``_current_bar_total`` / ``_committed_bytes`` — internal
          accumulators used to recover the snapshot-wide total when
          each byte bar only knows about its own file (see
          :meth:`update_download_bytes`).

    - For **uploads**:
        * ``filename`` / ``total_bytes`` / ``bytes_completed``
        * ``event_queue`` — used by patchers to emit completion events
        * ``completed_emitted`` — flag to avoid duplicate COMPLETE events
        * ``start_time`` — for elapsed-time metrics
    """

    def __init__(self) -> None:
        self._states: Dict[str, dict] = {}
        self._lock = threading.Lock()

    # ── Downloads ────────────────────────────────────────────────

    def init_download(self, transfer_id: str) -> None:
        """Initialize (or no-op if already present) the download state."""
        with self._lock:
            if transfer_id not in self._states:
                self._states[transfer_id] = {
                    "files_completed": 0,
                    "total_files": 0,
                    "bytes_completed": 0,
                    "total_bytes": 0,
                    # Accumulation tracking for sequential byte bars:
                    # Each byte bar only knows its own file's total, so we
                    # track the current bar's total and the sum of all
                    # previously completed bars' totals to compute the
                    # aggregate total_bytes across the entire snapshot.
                    "_current_bar_total": 0,
                    "_committed_bytes": 0,
                }

    def init_upload(
        self,
        transfer_id: str,
        filename: str,
        total_bytes: int,
        event_queue: queue.Queue,
    ) -> None:
        """Initialize (or no-op if already present) the upload state."""
        with self._lock:
            if transfer_id not in self._states:
                self._states[transfer_id] = {
                    "filename": filename,
                    "total_bytes": total_bytes,
                    "bytes_completed": 0,
                    "event_queue": event_queue,
                    "completed_emitted": False,
                    "start_time": time.time(),
                }

    def update_download_files(
        self,
        transfer_id: str,
        files_completed: int,
        total_files: int,
    ) -> None:
        """Update per-file progress counters for a download transfer."""
        with self._lock:
            if transfer_id in self._states:
                self._states[transfer_id]["files_completed"] = files_completed
                if total_files > 0:
                    self._states[transfer_id]["total_files"] = total_files

    def reset_byte_bar_state(self, transfer_id: str) -> None:
        """Reset the per-byte-bar state for a NEW byte bar.

        Called by ``DownloadProgressTqdm.__init__`` when a new byte bar is
        created. This is how we distinguish:

        - **HTTP per-file byte bars**: each file gets its own
          ``DownloadProgressTqdm`` instance, so ``__init__`` is called
          once per file. Committing the previous bar's total here ensures
          aggregate ``total_bytes`` reflects all files.

        - **xet shared byte bar**: ``huggingface_hub.snapshot_download``
          creates ONE shared byte bar (via ``bytes_progress`` +
          ``_AggregatedTqdm``). ``__init__`` is called only once. The bar
          reports growing totals (cumulative) as files are discovered.
          No commit is needed during its lifetime.

        The function commits the previous bar's total to ``_committed_bytes``
        if there was one, then resets the per-bar state. ``total_bytes``
        is updated to reflect the committed amount.
        """
        with self._lock:
            if transfer_id not in self._states:
                return
            state = self._states[transfer_id]
            prev_bar_total = state["_current_bar_total"]
            if prev_bar_total > 0:
                state["_committed_bytes"] += prev_bar_total
            state["_current_bar_total"] = 0
            state["bytes_completed"] = state["_committed_bytes"]
            state["total_bytes"] = state["_committed_bytes"]

    def update_download_bytes(
        self,
        transfer_id: str,
        bytes_completed: int,
        total_bytes: int,
    ) -> None:
        """Update aggregate byte counters and detect bar switches.

        Three patterns are observed in the wild:

        1. **xet shared bar** (``snapshot_download``): same bar, total
           GROWS as files are discovered. ``update(n)`` is the per-file
           increment. NOT a bar switch.
        2. **HTTP per-file bar**: detected via ``reset_byte_bar_state``
           in ``__init__`` (commits previous bar before reset). During
           the bar's life, total is constant.
        3. **HTTP fallback / edge case**: if a new bar's ``__init__``
           was not called (e.g., bound class not properly hooked), we
           fall back to the legacy heuristic: total changed AND
           bytes_completed dropped.
        """
        with self._lock:
            if transfer_id not in self._states:
                return
            state = self._states[transfer_id]

            prev_bar_total = state["_current_bar_total"]
            prev_bar_bytes = state["bytes_completed"] - state["_committed_bytes"]

            # If total GREW, it's the xet-style "more files discovered"
            # pattern — same bar, do NOT commit.
            if total_bytes > prev_bar_total:
                bar_switched = False
            elif total_bytes < prev_bar_total:
                # Total shrank — likely a new bar with smaller per-file size.
                bar_switched = True
            else:
                # Same total. Check if bytes_completed dropped significantly.
                if prev_bar_total > 0 and bytes_completed < (prev_bar_bytes * 0.5):
                    bar_switched = True
                else:
                    bar_switched = False

            if bar_switched and prev_bar_total > 0:
                state["_committed_bytes"] += prev_bar_total
            if bar_switched:
                state["_current_bar_total"] = total_bytes
            elif prev_bar_total == 0 and total_bytes > 0:
                # First non-zero total for this bar.
                state["_current_bar_total"] = total_bytes

            # Aggregate bytes_completed = committed from previous files
            # + current bar's progress. Aggregate total_bytes = committed
            # + current bar's total.
            state["bytes_completed"] = state["_committed_bytes"] + bytes_completed
            state["total_bytes"] = state["_committed_bytes"] + total_bytes

    # ── Uploads ─────────────────────────────────────────────────

    def add_upload_bytes(self, transfer_id: str, byte_increment: int) -> dict:
        """Accumulate bytes for multipart uploads and return a state copy."""
        with self._lock:
            state = self._states.get(transfer_id)
            if state:
                state["bytes_completed"] += byte_increment
                # Return a copy for safe event emission
                return dict(state)
            return {}

    def mark_upload_completed(self, transfer_id: str) -> None:
        """Mark the upload as having emitted its COMPLETE event."""
        with self._lock:
            state = self._states.get(transfer_id)
            if state:
                state["completed_emitted"] = True

    # ── Generic accessors ───────────────────────────────────────

    def get_state(self, transfer_id: str) -> dict:
        """Return a shallow copy of the state dict (or empty dict)."""
        with self._lock:
            return dict(self._states.get(transfer_id, {}))

    def clear_state(self, transfer_id: str) -> None:
        """Forget the state for ``transfer_id`` (idempotent)."""
        with self._lock:
            self._states.pop(transfer_id, None)


# Global thread-safe state manager (process-wide singleton)
state_manager = TransferStateManager()
