"""Progress callback classes for intercepting transfer progress."""

from __future__ import annotations

import contextlib
import queue
import time
from typing import Callable, Optional

from tqdm.auto import tqdm as base_tqdm

from .types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferDirection,
)

_TRANSFER_STATES = {}


class _DummyFile:
    """A dummy file-like object to silently sink tqdm output."""
    def write(self, x): pass
    def flush(self): pass


_dummy_file = _DummyFile()


class XetUploadProgressCallback:
    def __init__(
        self,
        filename: str,
        total_bytes: int,
        event_queue: queue.Queue,
        report_interval: float = 0.1,
        transfer_id: str = "",
        file_index: int = 0,
        total_files: int = 1,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ):
        self.filename = filename
        self.total_bytes = total_bytes
        self.event_queue = event_queue
        self.report_interval = report_interval
        self.transfer_id = transfer_id
        self.file_index = file_index
        self.total_files = total_files
        self.is_cancelled = is_cancelled

    def get_wrapper(self):
        def progress_updater(total_update, item_updates):
            return self(total_update, item_updates)
        return progress_updater

    def __call__(self, total_update, item_updates):
        if self.is_cancelled and self.is_cancelled():
            raise RuntimeError("Transfer cancelled by user")

        item_update = next((item for item in item_updates if getattr(item, "item_name", "") == self.filename), None)

        bytes_completed = getattr(total_update, "total_bytes_completed", 0)
        total_bytes = getattr(total_update, "total_bytes", 0) or self.total_bytes
        speed = getattr(total_update, "total_bytes_completion_rate", 0) or 0
        transfer_completed = getattr(total_update, "total_transfer_bytes_completed", 0)
        transfer_total = getattr(total_update, "total_transfer_bytes", 0)
        transfer_speed = getattr(total_update, "total_transfer_bytes_completion_rate", 0) or 0

        if self.total_files > 1 and item_update is not None:
            bytes_completed = getattr(item_update, "bytes_completed", 0)
            total_bytes = getattr(item_update, "total_bytes", 0) or self.total_bytes
            display_completed = bytes_completed
        else:
            display_completed = bytes_completed

        percentage = ((display_completed / total_bytes * 100) if total_bytes > 0 else 0)
        dedup_saved = max(0, bytes_completed - transfer_completed) if self.total_files == 1 else 0

        event = ProgressEvent(
            event_type=EventType.PROGRESS,
            transfer_id=self.transfer_id,
            direction=TransferDirection.UPLOAD,
            filename=self.filename,
            phase=ProgressPhase.UPLOADING,
            bytes_completed=display_completed,
            total_bytes=total_bytes,
            percentage=percentage,
            speed=transfer_speed or speed,
            file_index=self.file_index,
            total_files=self.total_files,
            transfer_bytes_completed=transfer_completed if self.total_files == 1 else 0,
            transfer_bytes_total=transfer_total if self.total_files == 1 else 0,
            transfer_speed=transfer_speed if self.total_files == 1 else 0,
            dedup_saved_bytes=dedup_saved,
        )
        self.event_queue.put(event)


class XetDownloadProgressCallback:
    def __init__(
        self,
        filename: str,
        total_bytes: int,
        event_queue: queue.Queue,
        report_interval: float = 0.1,
        transfer_id: str = "",
        file_index: int = 0,
        total_files: int = 1,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ):
        self.filename = filename
        self.total_bytes = total_bytes
        self.event_queue = event_queue
        self.report_interval = report_interval
        self.transfer_id = transfer_id
        self.file_index = file_index
        self.total_files = total_files
        self.is_cancelled = is_cancelled

    def get_wrapper(self):
        def progress_updater(total_update, item_updates):
            return self(total_update, item_updates)
        return progress_updater

    def __call__(self, total_update, item_updates):
        if self.is_cancelled and self.is_cancelled():
            raise RuntimeError("Transfer cancelled by user")

        item_update = next((item for item in item_updates if getattr(item, "item_name", "") == self.filename), None)

        bytes_completed = getattr(total_update, "total_bytes_completed", 0)
        total_bytes = getattr(total_update, "total_bytes", 0) or self.total_bytes
        speed = getattr(total_update, "total_bytes_completion_rate", 0) or 0
        transfer_completed = getattr(total_update, "total_transfer_bytes_completed", 0)
        transfer_total = getattr(total_update, "total_transfer_bytes", 0)
        transfer_speed = getattr(total_update, "total_transfer_bytes_completion_rate", 0) or 0

        if self.total_files > 1 and item_update is not None:
            bytes_completed = getattr(item_update, "bytes_completed", 0)
            total_bytes = getattr(item_update, "total_bytes", 0) or self.total_bytes
            display_completed = bytes_completed
        else:
            if bytes_completed >= total_bytes > 0:
                display_completed = bytes_completed
            elif transfer_completed > 0:
                display_completed = transfer_completed
            else:
                display_completed = bytes_completed

        percentage = ((display_completed / total_bytes * 100) if total_bytes > 0 else 0)
        dedup_saved = max(0, bytes_completed - transfer_completed) if self.total_files == 1 else 0

        event = ProgressEvent(
            event_type=EventType.PROGRESS,
            transfer_id=self.transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename=self.filename,
            phase=ProgressPhase.DOWNLOADING,
            bytes_completed=display_completed,
            total_bytes=total_bytes,
            percentage=percentage,
            speed=transfer_speed or speed,
            file_index=self.file_index,
            total_files=self.total_files,
            transfer_bytes_completed=transfer_completed if self.total_files == 1 else 0,
            transfer_bytes_total=transfer_total if self.total_files == 1 else 0,
            transfer_speed=transfer_speed if self.total_files == 1 else 0,
            dedup_saved_bytes=dedup_saved,
        )
        self.event_queue.put(event)


class DownloadProgressTqdm(base_tqdm):
    def __init__(self, *args, **kwargs):
        self._event_queue = kwargs.pop("event_queue", None)
        self._transfer_id = kwargs.pop("transfer_id", "")
        self._filename = kwargs.pop("filename", "")
        self._report_interval = kwargs.pop("report_interval", 0.1)
        self._is_cancelled = kwargs.pop("is_cancelled", None)
        self._start_time = time.time()
        self._closed = False
        
        # Determine if this is a bytes bar BEFORE super().__init__ 
        # so it doesn't fail if super().__init__ fails
        self.is_bytes_bar = (kwargs.get("unit", "it") in ("B", "iB"))
        
        kwargs.pop("name", None)
        super().__init__(*args, **kwargs)
        
        if not self._filename:
            self._filename = getattr(self, "desc", "unknown") or "unknown"
        
        if self._transfer_id not in _TRANSFER_STATES:
            _TRANSFER_STATES[self._transfer_id] = {"files_completed": 0, "total_files": 0}
            
        if not self.is_bytes_bar:
            _TRANSFER_STATES[self._transfer_id]["total_files"] = getattr(self, "total", 0) or 0

    def update(self, n=1):
        if self._is_cancelled is not None and self._is_cancelled():
            raise RuntimeError("Transfer cancelled by user")

        result = super().update(n)

        if self._event_queue is None or n == 0:
            return result

        if not getattr(self, "is_bytes_bar", False):
            _TRANSFER_STATES[self._transfer_id]["files_completed"] = getattr(self, "n", 0)
            return result

        now = time.time()
        bytes_completed = getattr(self, "n", 0)
        total_bytes = getattr(self, "total", 0) or 0
        percentage = ((bytes_completed / total_bytes * 100) if total_bytes > 0 else 0)

        speed = getattr(self, "format_dict", {}).get("rate") or 0
        if not speed and bytes_completed > 0:
            elapsed = now - self._start_time
            speed = bytes_completed / elapsed if elapsed > 0 else 0

        file_stats = _TRANSFER_STATES.get(self._transfer_id, {})
        files_completed = file_stats.get("files_completed", 0)
        total_files = file_stats.get("total_files", 0)

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
            file_index=files_completed,
            total_files=total_files,
        )
        self._event_queue.put(event)
        return result

    def close(self):
        if getattr(self, "_closed", True):
            super().close()
            return
        self._closed = True

        if self._event_queue is not None and getattr(self, "total", None) is not None and getattr(self, "is_bytes_bar", False):
            n_val = getattr(self, "n", 0)
            total_val = getattr(self, "total", 0)
            
            is_complete = n_val >= total_val
            is_xet_cached = n_val == 0 and total_val > 0

            if is_complete or is_xet_cached:
                final_bytes = total_val if is_xet_cached else n_val
                file_stats = _TRANSFER_STATES.get(self._transfer_id, {})
                total_files = file_stats.get("total_files", 0)
                files_completed = total_files if total_files > 0 else file_stats.get("files_completed", 0)

                self._event_queue.put(
                    ProgressEvent(
                        event_type=EventType.COMPLETE,
                        transfer_id=self._transfer_id,
                        direction=TransferDirection.DOWNLOAD,
                        filename=self._filename,
                        phase=ProgressPhase.COMPLETE,
                        bytes_completed=final_bytes,
                        total_bytes=total_val,
                        percentage=100.0,
                        file_index=files_completed,
                        total_files=total_files,
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
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> type:
        _queue = event_queue
        _tid = transfer_id
        _fname = filename
        _interval = report_interval
        _cancel_hook = is_cancelled

        class BoundDownloadTqdm(cls):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("event_queue", _queue)
                kwargs.setdefault("transfer_id", _tid)
                kwargs.setdefault("filename", _fname)
                kwargs.setdefault("report_interval", _interval)
                kwargs.setdefault("is_cancelled", _cancel_hook)
                
                # IMPORTANT: Silence native tqdm rendering without breaking its math
                kwargs["file"] = _dummy_file
                super().__init__(*args, **kwargs)

        BoundDownloadTqdm.__name__ = f"BoundDownloadTqdm_{transfer_id[:8]}"
        BoundDownloadTqdm.__qualname__ = f"BoundDownloadTqdm_{transfer_id[:8]}"
        return BoundDownloadTqdm


@contextlib.contextmanager
def tqdm_upload_patcher(
    event_queue: queue.Queue,
    transfer_id: str = "",
    filename: str = "",
    report_interval: float = 0.1,
    is_cancelled: Optional[Callable[[], bool]] = None,
):
    import tqdm.auto as tqdm_auto_module

    original_tqdm = tqdm_auto_module.tqdm
    _patch_active = True

    class UploadProgressTqdm(original_tqdm):
        def __init__(self, *args, **kwargs):
            # IMPORTANT: Silence native tqdm rendering without breaking its math
            kwargs["file"] = _dummy_file
            super().__init__(*args, **kwargs)
            
            self._upload_event_queue = event_queue
            self._upload_transfer_id = transfer_id
            self._upload_filename = filename or getattr(self, "desc", "") or ""
            self._upload_report_interval = report_interval
            self._upload_start_time = time.time()
            self._upload_is_file_bar = (
                getattr(self, "total", None) is not None
                and getattr(self, "total", 0) > 0
                and getattr(self, "unit", "") in ("B", "iB")
            )

            if getattr(self, "_upload_is_file_bar", False) and event_queue is not None:
                event_queue.put(
                    ProgressEvent(
                        event_type=EventType.START,
                        transfer_id=transfer_id,
                        direction=TransferDirection.UPLOAD,
                        filename=self._upload_filename,
                        phase=ProgressPhase.UPLOADING,
                        total_bytes=getattr(self, "total", 0),
                    )
                )

        def update(self, n=1):
            if is_cancelled is not None and is_cancelled():
                raise RuntimeError("Transfer cancelled by user")

            result = super().update(n)

            if (
                not getattr(self, "_upload_is_file_bar", False)
                or getattr(self, "_upload_event_queue", None) is None
                or not _patch_active
                or n == 0
            ):
                return result

            now = time.time()
            bytes_completed = getattr(self, "n", 0)
            total_bytes = getattr(self, "total", 0) or 0
            percentage = ((bytes_completed / total_bytes * 100) if total_bytes > 0 else 0)

            speed = getattr(self, "format_dict", {}).get("rate") or 0
            if not speed and bytes_completed > 0:
                elapsed = now - getattr(self, "_upload_start_time", now)
                speed = bytes_completed / elapsed if elapsed > 0 else 0

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
            if (
                getattr(self, "_upload_is_file_bar", False)
                and getattr(self, "_upload_event_queue", None) is None
                and _patch_active
                and getattr(self, "total", None) is not None
                and getattr(self, "n", 0) >= self.total
            ):
                self._upload_event_queue.put(
                    ProgressEvent(
                        event_type=EventType.COMPLETE,
                        transfer_id=self._upload_transfer_id,
                        direction=TransferDirection.UPLOAD,
                        filename=self._upload_filename,
                        phase=ProgressPhase.COMPLETE,
                        bytes_completed=self.n,
                        total_bytes=self.total,
                        percentage=100.0,
                    )
                )
            super().close()

    tqdm_auto_module.tqdm = UploadProgressTqdm
    try:
        yield
    finally:
        _patch_active = False
        tqdm_auto_module.tqdm = original_tqdm