"""Progress callback classes for intercepting transfer progress.

Four callback mechanisms are provided:

1. ``XetUploadProgressCallback`` — For hf_xet direct upload calls.
Receives (total_update, item_updates) from Rust runtime.
Parameter names MUST match exactly for Rust auto-detection.

2. ``XetDownloadProgressCallback`` — For hf_xet direct download calls.
Receives (total_update, item_updates) from Rust runtime.
Parameter names MUST match exactly for Rust auto-detection.

3. ``DownloadProgressTqdm`` — Custom tqdm subclass for HTTP downloads.
Used via the ``tqdm_class`` parameter of ``hf_hub_download()``.
Fallback when hf_xet is not available.

4. ``tqdm_upload_patcher`` — Context manager for LFS upload progress.
Monkey-patches tqdm globally to intercept upload progress bars.
Fallback when hf_xet is not available.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from typing import Callable, List, Optional

from tqdm.auto import tqdm as base_tqdm

from .types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferDirection,
)


class XetUploadProgressCallback:
    """Thread-safe progress callback for hf_xet uploads.

    hf_xet's Rust runtime calls this with two arguments:
    - ``total_update``: PyTotalProgressUpdate with overall progress
    - ``item_updates``: List[PyItemProgressUpdate] with per-file progress

    **CRITICAL**: Parameter names MUST be ``total_update`` and ``item_updates``
    for the Rust ``WrappedProgressUpdaterImpl`` to detect the detailed callback
    signature. If names don't match, the Rust runtime falls back to simple
    ``(int)`` mode and you lose all detailed progress data.

    The callback is invoked from Rust's ``spawn_blocking`` thread pool,
    which acquires the GIL via ``Python::attach``. It is synchronous from
    Python's perspective but runs on a background thread.

    Args:
        filename: Name of the file being uploaded.
        total_bytes: Expected total bytes.
        event_queue: Thread-safe queue to emit ProgressEvent objects.
        report_interval: Minimum seconds between progress events (throttle).
        transfer_id: Unique identifier for this transfer.
        file_index: Index of this file in a multi-file transfer.
        total_files: Total number of files in the transfer.
    """

    def __init__(
        self,
        filename: str,
        total_bytes: int,
        event_queue: queue.Queue,
        report_interval: float = 0.1,
        transfer_id: str = "",
        file_index: int = 0,
        total_files: int = 1,
    ):
        self.filename = filename
        self.total_bytes = total_bytes
        self.event_queue = event_queue
        self.report_interval = report_interval
        self.transfer_id = transfer_id
        self.file_index = file_index
        self.total_files = total_files
        self._last_report_time = 0.0
        self._lock = threading.Lock()

    def __call__(self, total_update, item_updates):
        """Called by hf_xet Rust runtime from a background thread.

        Args:
            total_update: PyTotalProgressUpdate with fields:
                - total_bytes (int)
                - total_bytes_completed (int)
                - total_bytes_completion_rate (float)
                - total_transfer_bytes (int)
                - total_transfer_bytes_completed (int)
                - total_transfer_bytes_completion_rate (float)
            item_updates: List[PyItemProgressUpdate] with fields:
                - item_name (str)
                - total_bytes (int)
                - bytes_completed (int)
                - bytes_completion_increment (int)
        """
        now = time.time()
        with self._lock:
            if now - self._last_report_time < self.report_interval:
                return
            self._last_report_time = now

        # Extract values immediately — don't store Rust object references
        bytes_completed = getattr(total_update, "total_bytes_completed", 0)
        total_bytes = getattr(total_update, "total_bytes", 0) or self.total_bytes
        speed = getattr(total_update, "total_bytes_completion_rate", 0) or 0
        transfer_completed = getattr(
            total_update, "total_transfer_bytes_completed", 0
        )
        transfer_total = getattr(total_update, "total_transfer_bytes", 0)
        transfer_speed = getattr(
            total_update, "total_transfer_bytes_completion_rate", 0
        ) or 0

        percentage = (
            (bytes_completed / total_bytes * 100) if total_bytes > 0 else 0
        )
        dedup_saved = max(0, bytes_completed - transfer_completed)

        event = ProgressEvent(
            event_type=EventType.PROGRESS,
            transfer_id=self.transfer_id,
            direction=TransferDirection.UPLOAD,
            filename=self.filename,
            phase=ProgressPhase.UPLOADING,
            bytes_completed=bytes_completed,
            total_bytes=total_bytes,
            percentage=percentage,
            speed=speed,
            file_index=self.file_index,
            total_files=self.total_files,
            transfer_bytes_completed=transfer_completed,
            transfer_bytes_total=transfer_total,
            transfer_speed=transfer_speed,
            dedup_saved_bytes=dedup_saved,
        )
        self.event_queue.put(event)


class XetDownloadProgressCallback:
    """Thread-safe progress callback for hf_xet downloads.

    Uses the **detailed** callback signature ``(total_update, item_updates)``
    so the Rust ``WrappedProgressUpdaterImpl`` provides rich progress data
    including speed, dedup info, and per-item progress.

    **CRITICAL**: Parameter names MUST be ``total_update`` and ``item_updates``
    for the Rust ``WrappedProgressUpdaterImpl`` to detect the detailed callback
    signature. If names don't match, the Rust runtime falls back to simple
    ``(int)`` mode and you lose all detailed progress data.

    The callback is invoked from Rust's ``spawn_blocking`` thread pool,
    which acquires the GIL via ``Python::attach``. It is synchronous from
    Python's perspective but runs on a background thread.

    Args:
        filename: Name of the file being downloaded.
        total_bytes: Expected total bytes.
        event_queue: Thread-safe queue to emit ProgressEvent objects.
        report_interval: Minimum seconds between progress events (throttle).
        transfer_id: Unique identifier for this transfer.
        file_index: Index of this file in a multi-file transfer.
        total_files: Total number of files in the transfer.
    """

    def __init__(
        self,
        filename: str,
        total_bytes: int,
        event_queue: queue.Queue,
        report_interval: float = 0.1,
        transfer_id: str = "",
        file_index: int = 0,
        total_files: int = 1,
    ):
        self.filename = filename
        self.total_bytes = total_bytes
        self.event_queue = event_queue
        self.report_interval = report_interval
        self.transfer_id = transfer_id
        self.file_index = file_index
        self.total_files = total_files
        self._last_report_time = 0.0
        self._lock = threading.Lock()

    def __call__(self, total_update, item_updates):
        """Called by hf_xet Rust runtime from a background thread.

        Args:
            total_update: PyTotalProgressUpdate with fields:
                - total_bytes (int)
                - total_bytes_completed (int)
                - total_bytes_completion_rate (float)
                - total_transfer_bytes (int)
                - total_transfer_bytes_completed (int)
                - total_transfer_bytes_completion_rate (float)
            item_updates: List[PyItemProgressUpdate] with fields:
                - item_name (str)
                - total_bytes (int)
                - bytes_completed (int)
                - bytes_completion_increment (int)
        """
        now = time.time()
        with self._lock:
            if now - self._last_report_time < self.report_interval:
                return
            self._last_report_time = now

        # Extract values immediately — don't store Rust object references
        bytes_completed = getattr(total_update, "total_bytes_completed", 0)
        total_bytes = getattr(total_update, "total_bytes", 0) or self.total_bytes
        speed = getattr(total_update, "total_bytes_completion_rate", 0) or 0
        transfer_completed = getattr(
            total_update, "total_transfer_bytes_completed", 0
        )
        transfer_total = getattr(total_update, "total_transfer_bytes", 0)
        transfer_speed = getattr(
            total_update, "total_transfer_bytes_completion_rate", 0
        ) or 0

        percentage = (
            (bytes_completed / total_bytes * 100) if total_bytes > 0 else 0
        )
        dedup_saved = max(0, bytes_completed - transfer_completed)

        event = ProgressEvent(
            event_type=EventType.PROGRESS,
            transfer_id=self.transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename=self.filename,
            phase=ProgressPhase.DOWNLOADING,
            bytes_completed=bytes_completed,
            total_bytes=total_bytes,
            percentage=percentage,
            speed=speed,
            file_index=self.file_index,
            total_files=self.total_files,
            transfer_bytes_completed=transfer_completed,
            transfer_bytes_total=transfer_total,
            transfer_speed=transfer_speed,
            dedup_saved_bytes=dedup_saved,
        )
        self.event_queue.put(event)


class DownloadProgressTqdm(base_tqdm):
    """Custom tqdm that emits ProgressEvent objects to a queue.

    Subclass ``tqdm.auto.tqdm`` (NOT ``huggingface_hub.utils.tqdm``) so
    that ``_create_progress_bar()`` takes the ``cls(**kwargs)`` path
    (line 320 of huggingface_hub/utils/tqdm.py) — no ``disable`` or
    ``name`` injection. Our custom parameters (``event_queue``,
    ``transfer_id``, etc.) are injected via ``bind()`` which uses
    ``kwargs.setdefault()`` in a generated subclass ``__init__``.

    **How it works with HTTP downloads**::

        hf_hub_download(tqdm_class=BoundDownloadTqdm)
        → http_get(tqdm_class=BoundDownloadTqdm)
        → _get_progress_bar_context(tqdm_class=BoundDownloadTqdm)
        → _create_progress_bar(cls=BoundDownloadTqdm, ...)
        → BoundDownloadTqdm(desc=..., total=..., unit="B", ...)
        → progress.update(len(chunk))  per HTTP chunk

    **How it works with Xet downloads**::

        hf_hub_download(tqdm_class=BoundDownloadTqdm)
        → xet_get(tqdm_class=BoundDownloadTqdm)
        → _get_progress_bar_context(tqdm_class=BoundDownloadTqdm)
        → BoundDownloadTqdm(desc=..., total=..., unit="B", ...)
        → progress_updater = lambda bytes: progress.update(bytes)
        → download_files(..., progress_updater=[progress_updater])
        → Rust calls progress_updater(int) per chunk

    **Important**: The ``close()`` method emits a COMPLETE event when
    the download finishes. For Xet downloads where all chunks are cached
    (no ``update()`` calls), ``close()`` still emits COMPLETE with
    ``bytes_completed=total_bytes`` since the file was written successfully.

    Usage::

        from hf_progress.callbacks import DownloadProgressTqdm

        tqdm_class = DownloadProgressTqdm.bind(event_queue, "dl-001")
        hf_hub_download(repo_id="...", filename="...", tqdm_class=tqdm_class)

    Args:
        event_queue: Thread-safe queue for emitting ProgressEvent objects.
        transfer_id: Unique identifier for this transfer.
        filename: Name of the file being downloaded.
        report_interval: Minimum seconds between progress events.
    """

    def __init__(
        self,
        *args,
        event_queue: Optional[queue.Queue] = None,
        transfer_id: str = "",
        filename: str = "",
        report_interval: float = 0.1,
        **kwargs,
    ):
        # Set ALL custom attributes BEFORE super().__init__() so that
        # if __init__ fails and __del__ calls close(), the attributes
        # exist (prevents AttributeError in tqdm.__del__).
        self._closed = False
        self._event_queue = event_queue
        self._transfer_id = transfer_id
        self._filename = filename
        self._report_interval = report_interval
        self._last_report_time = 0.0
        self._lock = threading.Lock()
        self._start_time = time.time()

        # Remove 'name' kwarg that HF injects for its own tqdm subclass.
        # Vanilla tqdm.auto.tqdm does NOT accept 'name' (TqdmKeyError).
        # This happens when snapshot_download() passes name="huggingface_hub.snapshot_download"
        # to _create_progress_bar() → cls(**kwargs).
        kwargs.pop("name", None)

        super().__init__(*args, **kwargs)

        # After super().__init__(), self.desc is available
        if not self._filename:
            self._filename = self.desc or "unknown"

    def update(self, n=1):
        """Override update to emit progress events with throttling.

        Called by ``huggingface_hub`` internals for each chunk downloaded.
        For HTTP downloads: called per ``len(chunk)`` bytes received.
        For Xet downloads: called via ``progress_updater`` closure that
        wraps ``progress.update(progress_bytes)``.

        **Throttling strategy**: Always update tqdm's internal counter
        (via ``super().update(n)``) so ``self.n`` stays accurate.
        Only throttle the *event emission* — when the throttle fires,
        emit the current accumulated progress. This prevents the
        consumer from seeing "frozen then jump" behavior.
        """
        # Always update tqdm's internal counter first
        result = super().update(n)

        if self._event_queue is None:
            return result

        # Throttle only the event emission, not the counter update
        now = time.time()
        with self._lock:
            if now - self._last_report_time < self._report_interval:
                return result
            self._last_report_time = now

        # Calculate progress from current accumulated state
        bytes_completed = self.n
        total_bytes = self.total or 0
        percentage = (
            (bytes_completed / total_bytes * 100) if total_bytes > 0 else 0
        )

        # Calculate speed from tqdm's internal rate or manually
        speed = self.format_dict.get("rate") or 0
        if not speed and self.n > 0:
            elapsed = now - self._start_time
            speed = self.n / elapsed if elapsed > 0 else 0

        event = ProgressEvent(
            event_type=EventType.PROGRESS,
            transfer_id=self._transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename=self._filename,
            phase=ProgressPhase.DOWNLOADING,
            bytes_completed=bytes_completed,
            total_bytes=total_bytes,
            percentage=percentage,
            speed=speed or 0,
        )
        self._event_queue.put(event)

        return result

    def close(self):
        """Emit complete event when the bar closes.

        Emits COMPLETE if the progress bar was opened (has a ``total``)
        and hasn't already been closed. Three cases:

        1. **Normal completion**: ``self.n >= self.total`` — all bytes
           received. Emit COMPLETE with ``bytes_completed=self.n``.
        2. **Xet cached download**: ``self.n == 0`` and ``self.total > 0``
           — no ``update()`` calls because all chunks were in cache, but
           the file was written successfully. The ``with progress_cm``
           block in ``xet_get()`` guarantees success. Emit COMPLETE with
           ``bytes_completed=self.total``.
        3. **Incomplete download**: ``0 < self.n < self.total`` — the
           download was interrupted. Do NOT emit COMPLETE.
        """
        if self._closed:
            super().close()
            return
        self._closed = True

        if self._event_queue is not None and self.total is not None:
            # Case 1: Normal completion (n >= total)
            # Case 2: Xet cached download (n == 0, total > 0)
            # Case 3: Incomplete (0 < n < total) — skip
            is_complete = self.n >= self.total
            is_xet_cached = self.n == 0 and self.total > 0

            if is_complete or is_xet_cached:
                final_bytes = self.total if is_xet_cached else self.n
                self._event_queue.put(
                    ProgressEvent(
                        event_type=EventType.COMPLETE,
                        transfer_id=self._transfer_id,
                        direction=TransferDirection.DOWNLOAD,
                        filename=self._filename,
                        phase=ProgressPhase.COMPLETE,
                        bytes_completed=final_bytes,
                        total_bytes=self.total,
                        percentage=100.0,
                    )
                )
        super().close()

    @classmethod
    def bind(
        cls,
        event_queue: queue.Queue,
        transfer_id: str,
        filename: str = "",
        report_interval: float = 0.1,
    ) -> type:
        """Create a tqdm subclass with pre-bound parameters.

        Returns a class (not an instance) that can be passed as
        ``tqdm_class`` to ``hf_hub_download()`` or ``snapshot_download()``.

        The generated subclass uses ``kwargs.setdefault()`` to inject
        our custom parameters (``event_queue``, ``transfer_id``, etc.)
        BEFORE passing to the parent ``__init__``. This works because
        ``_create_progress_bar()`` calls ``cls(**kwargs)`` where kwargs
        contains ``desc``, ``total``, ``initial``, ``unit``, ``unit_scale``
        — our custom params are NOT in those kwargs, so ``setdefault()``
        fills them in.

        Args:
            event_queue: Thread-safe queue for emitting events.
            transfer_id: Unique identifier for this transfer.
            filename: Name of the file being downloaded.
            report_interval: Minimum seconds between progress events.

        Returns:
            A tqdm subclass with bound parameters.

        Usage::

            tqdm_class = DownloadProgressTqdm.bind(queue, "dl-001")
            hf_hub_download(repo_id="...", filename="...", tqdm_class=tqdm_class)
        """
        _queue = event_queue
        _tid = transfer_id
        _fname = filename
        _interval = report_interval

        class BoundDownloadTqdm(cls):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("event_queue", _queue)
                kwargs.setdefault("transfer_id", _tid)
                kwargs.setdefault("filename", _fname)
                kwargs.setdefault("report_interval", _interval)
                super().__init__(*args, **kwargs)

        BoundDownloadTqdm.__name__ = f"BoundDownloadTqdm_{transfer_id[:8]}"
        BoundDownloadTqdm.__qualname__ = (
            f"BoundDownloadTqdm_{transfer_id[:8]}"
        )
        return BoundDownloadTqdm


@contextlib.contextmanager
def tqdm_upload_patcher(
    event_queue: queue.Queue,
    transfer_id: str = "",
    filename: str = "",
    report_interval: float = 0.1,
):
    """Context manager that intercepts tqdm during LFS uploads.

    Temporarily replaces ``tqdm.auto.tqdm`` with a custom subclass
    that emits ProgressEvent objects to a queue.

    **WARNING**: This patches the GLOBAL tqdm class. It will affect
    ALL tqdm bars created while the context manager is active, not
    just HuggingFace upload bars. Use only when:

    1. ``hf_xet`` is not installed or Xet storage is unavailable.
    2. No other concurrent tqdm-using operations are running.

    The patcher filters file-level bars (``unit="B"``) from file-count
    bars (``unit="it"``) to avoid emitting events for the outer
    ``thread_map`` progress bar.

    Args:
        event_queue: Thread-safe queue for emitting ProgressEvent objects.
        transfer_id: Unique identifier for this transfer.
        filename: Expected filename (used as fallback if tqdm desc is empty).
        report_interval: Minimum seconds between progress events.

    Usage::

        from hf_progress.callbacks import tqdm_upload_patcher

        with tqdm_upload_patcher(event_queue, "upload-001", "model.bin"):
            api.upload_file(
                path_or_fileobj="model.bin",
                path_in_repo="model.bin",
                repo_id="username/repo",
            )
    """
    import tqdm.auto as tqdm_auto_module

    original_tqdm = tqdm_auto_module.tqdm
    _patch_active = True

    class UploadProgressTqdm(original_tqdm):
        """tqdm subclass that emits progress events during uploads."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._upload_event_queue = event_queue
            self._upload_transfer_id = transfer_id
            self._upload_filename = filename or (self.desc or "")
            self._upload_report_interval = report_interval
            self._upload_last_report_time = 0.0
            self._upload_lock = threading.Lock()
            self._upload_start_time = time.time()
            # File-level bars have unit="B" and a positive total;
            # file-count bars from thread_map have unit="it"
            self._upload_is_file_bar = (
                self.total is not None
                and self.total > 0
                and self.unit == "B"
            )

            # Emit start event for file-level bars only
            if self._upload_is_file_bar and event_queue is not None:
                event_queue.put(
                    ProgressEvent(
                        event_type=EventType.START,
                        transfer_id=transfer_id,
                        direction=TransferDirection.UPLOAD,
                        filename=self._upload_filename,
                        phase=ProgressPhase.UPLOADING,
                        total_bytes=self.total or 0,
                    )
                )

        def update(self, n=1):
            result = super().update(n)

            # Only emit events for file-level bars
            if (
                not self._upload_is_file_bar
                or self._upload_event_queue is None
                or not _patch_active
            ):
                return result

            now = time.time()
            with self._upload_lock:
                if (
                    now - self._upload_last_report_time
                    < self._upload_report_interval
                ):
                    return result
                self._upload_last_report_time = now

            bytes_completed = self.n
            total_bytes = self.total or 0
            percentage = (
                (bytes_completed / total_bytes * 100)
                if total_bytes > 0
                else 0
            )

            # Calculate speed
            speed = self.format_dict.get("rate") or 0
            if not speed and self.n > 0:
                elapsed = now - self._upload_start_time
                speed = self.n / elapsed if elapsed > 0 else 0

            event = ProgressEvent(
                event_type=EventType.PROGRESS,
                transfer_id=self._upload_transfer_id,
                direction=TransferDirection.UPLOAD,
                filename=self._upload_filename,
                phase=ProgressPhase.UPLOADING,
                bytes_completed=bytes_completed,
                total_bytes=total_bytes,
                percentage=percentage,
                speed=speed or 0,
            )
            self._upload_event_queue.put(event)

            return result

        def close(self):
            """Emit complete event when file bar closes at 100%."""
            if (
                self._upload_is_file_bar
                and self._upload_event_queue is not None
                and _patch_active
                and self.total is not None
                and self.n >= self.total
            ):
                self._upload_event_queue.put(
                    ProgressEvent(
                        event_type=EventType.COMPLETE,
                        transfer_id=self._upload_transfer_id,
                        direction=TransferDirection.UPLOAD,
                        filename=self._upload_filename,
                        phase=ProgressPhase.COMPLETE,
                        bytes_completed=self.n,
                        total_bytes=self.total or 0,
                        percentage=100.0,
                    )
                )
            super().close()

    # Apply the global patch
    tqdm_auto_module.tqdm = UploadProgressTqdm

    try:
        yield
    finally:
        # Restore original tqdm
        _patch_active = False
        tqdm_auto_module.tqdm = original_tqdm
