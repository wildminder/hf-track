"""Unified progress tracker for HuggingFace uploads and downloads.

Provides a single, plug-and-play interface that automatically selects
the best available progress tracking method using a **direct-first**
strategy:

- **Downloads with Xet**: Uses ``hf_xet.download_files()`` directly
  with detailed ``(total_update, item_updates)`` callbacks for
  speed, dedup info, and per-item progress.
- **Downloads without Xet**: Falls back to ``tqdm_class`` override
  via ``hf_hub_download()``.
- **Uploads with Xet**: Uses ``hf_xet`` direct calls with detailed callbacks.
- **Uploads without Xet**: Uses tqdm monkey-patching.

All progress events are emitted to an internal ``queue.Queue`` that
can be consumed from any thread. Events are typed as ``ProgressEvent``
objects with consistent fields regardless of the underlying method.

Usage::

    from hf_progress import HfProgressTracker

    tracker = HfProgressTracker(token="hf_...")

    # Upload a file
    result = tracker.upload_file(
        file_path="/path/to/model.safetensors",
        repo_id="username/my-model",
    )

    # Download a file
    path = tracker.download_file(
        repo_id="bert-base-uncased",
        filename="config.json",
    )

    # Consume events from any thread
    for event in tracker.events():
        print(f"[{event.event_type.value}] {event.filename}: {event.percentage:.1f}%")
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
from typing import Generator, List, Optional

from .callbacks import (
    DownloadProgressTqdm,
    XetUploadProgressCallback,
    tqdm_upload_patcher,
)
from .token import XetTokenManager, is_xet_available
from .types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferDirection,
    TransferResult,
    generate_transfer_id,
)


class HfProgressTracker:
    """Unified progress tracker for HuggingFace transfers.

    Automatically selects the best available progress tracking method
    based on whether ``hf_xet`` is installed and the type of operation.

    All progress events are emitted to an internal ``queue.Queue``.
    Use ``events()``, ``get_events()``, or ``wait_for_complete()``
    to consume events.

    The tracker is thread-safe: you can run transfers in background
    threads and consume events from the main thread.

    Args:
        token: HuggingFace API token (``hf_...``).
        endpoint: Optional custom HuggingFace API endpoint.
        report_interval: Default minimum seconds between progress events.

    Example::

        tracker = HfProgressTracker(token="hf_...")

        # Run upload in background
        def do_upload():
            tracker.upload_file("model.bin", "user/repo")

        thread = threading.Thread(target=do_upload, daemon=True)
        thread.start()

        # Consume events from main thread
        for event in tracker.events(timeout=1.0):
            if event.event_type == EventType.PROGRESS:
                print(f"{event.filename}: {event.percentage:.1f}%")
            elif event.event_type in (EventType.COMPLETE, EventType.ERROR):
                break
    """

    def __init__(
        self,
        token: Optional[str] = None,
        endpoint: Optional[str] = None,
        report_interval: float = 0.1,
    ):
        self._token = token
        self._endpoint = endpoint
        self._report_interval = report_interval
        self.event_queue: queue.Queue[ProgressEvent] = queue.Queue()
        self._token_manager = XetTokenManager(token, endpoint)
        self._active_transfers: dict = {}
        self._lock = threading.Lock()

    # ── Download Methods ──────────────────────────────────────────

    def download_file(
        self,
        repo_id: str,
        filename: str,
        repo_type: str = "model",
        revision: Optional[str] = None,
        local_dir: Optional[str] = None,
        transfer_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Download a single file with progress tracking.

        Uses a **direct-first** strategy:

        1. If ``hf_xet`` is available, attempts to acquire ``XetFileData``
           (file hash + refresh route) via a HEAD request, then calls
           ``hf_xet.download_files()`` directly with a detailed
           ``(total_update, item_updates)`` callback. This provides
           speed, dedup info, and per-item progress from the Rust runtime.

        2. If Xet is not available, or the file is not stored in Xet
           (no ``XetFileData`` in the HEAD response), falls back to
           ``hf_hub_download(tqdm_class=...)`` which works for both
           HTTP and Xet downloads (but with less detailed progress data).

        Args:
            repo_id: Repository ID (e.g. ``"bert-base-uncased"``).
            filename: Filename within the repository.
            repo_type: Repository type (model, dataset, space).
            revision: Optional git revision.
            local_dir: Optional local directory to download to. If set,
                the file is saved directly to this directory (not the
                HF cache). Symlinks are disabled in this mode.
            transfer_id: Unique transfer identifier (auto-generated if None).
            **kwargs: Additional arguments passed to ``hf_hub_download()``
                (fallback path only).

        Returns:
            Local path to the downloaded file.
        """
        transfer_id = transfer_id or generate_transfer_id()

        # Direct-first: try Xet direct download if hf_xet is available
        if is_xet_available():
            try:
                return self._download_file_xet(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type=repo_type,
                    revision=revision,
                    local_dir=local_dir,
                    transfer_id=transfer_id,
                )
            except Exception as xet_err:
                # If Xet direct fails (e.g. file not in Xet storage,
                # token refresh failure), fall back to standard path
                import logging
                logging.getLogger(__name__).warning(
                    f"Xet direct download failed for {repo_id}/{filename}, "
                    f"falling back to tqdm_class: {xet_err}"
                )

        # Fallback: use hf_hub_download with tqdm_class override
        from .standard_download import download_file as _download_file

        return _download_file(
            repo_id=repo_id,
            filename=filename,
            token=self._token,
            event_queue=self.event_queue,
            repo_type=repo_type,
            revision=revision,
            endpoint=self._endpoint,
            local_dir=local_dir,
            transfer_id=transfer_id,
            report_interval=self._report_interval,
            **kwargs,
        )

    def download_snapshot(
        self,
        repo_id: str,
        allow_patterns=None,
        ignore_patterns=None,
        repo_type: str = "model",
        revision: Optional[str] = None,
        local_dir: Optional[str] = None,
        transfer_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Download a repository snapshot with progress tracking.

        **Note**: ``snapshot_download()`` provides file-count progress
        (N/M files), NOT per-file byte-level progress. For byte-level
        tracking, use ``download_file()`` for each file individually.

        Args:
            repo_id: Repository ID.
            allow_patterns: Glob patterns for files to include.
            ignore_patterns: Glob patterns for files to exclude.
            repo_type: Repository type.
            revision: Optional git revision.
            local_dir: Optional local directory to download to. If set,
                files are saved directly to this directory (not the HF
                cache). Symlinks are disabled in this mode.
            transfer_id: Unique transfer identifier.
            **kwargs: Additional arguments passed to ``snapshot_download()``.

        Returns:
            Local path to the downloaded snapshot directory.
        """
        from .standard_download import download_snapshot as _download_snapshot

        transfer_id = transfer_id or generate_transfer_id()

        return _download_snapshot(
            repo_id=repo_id,
            token=self._token,
            event_queue=self.event_queue,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            repo_type=repo_type,
            revision=revision,
            endpoint=self._endpoint,
            local_dir=local_dir,
            transfer_id=transfer_id,
            report_interval=self._report_interval,
            **kwargs,
        )

    # ── Internal: Xet Download Methods ──────────────────────────

    def _download_file_xet(
        self,
        repo_id: str,
        filename: str,
        repo_type: str,
        revision: Optional[str],
        local_dir: Optional[str],
        transfer_id: str,
    ) -> str:
        """Download a file using hf_xet directly with detailed progress.

        Uses ``HfApi.get_hf_file_metadata()`` to acquire ``XetFileData``
        (file hash + refresh route), then calls ``hf_xet.download_files()``
        with a detailed ``(total_update, item_updates)`` callback.

        Raises:
            ValueError: If the file is not stored in Xet storage
                (no ``XetFileData`` in the metadata response).
            ImportError: If ``hf_xet`` is not installed.
        """
        from huggingface_hub import HfApi, hf_hub_url

        from .xet_download import download_file_with_xet

        # Step 1: Acquire XetFileData via get_hf_file_metadata
        # This is the stable public API that handles HEAD requests,
        # redirects, and Xet header parsing internally.
        api = HfApi(endpoint=self._endpoint, token=self._token)
        url = hf_hub_url(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            endpoint=self._endpoint,
        )
        metadata = api.get_hf_file_metadata(url=url, token=self._token)

        if metadata.xet_file_data is None:
            raise ValueError(
                f"File '{filename}' in '{repo_id}' is not stored in Xet storage. "
                f"Use the standard download path instead."
            )

        xet_file_data = metadata.xet_file_data
        file_size = metadata.size or 0

        # Step 2: Determine destination path
        if local_dir:
            dest_path = os.path.join(local_dir, filename)
        else:
            dest_path = os.path.join(tempfile.gettempdir(), filename)

        # Step 3: Build request headers (strip auth for Xet CAS server)
        headers = api._build_hf_headers()
        xet_headers = dict(headers)
        xet_headers.pop("authorization", None)

        # Step 4: Call hf_xet.download_files() directly
        result = download_file_with_xet(
            file_hash=xet_file_data.file_hash,
            file_size=file_size,
            dest_path=dest_path,
            xet_file_data=xet_file_data,
            token=self._token,
            event_queue=self.event_queue,
            endpoint=api.endpoint,
            transfer_id=transfer_id,
            report_interval=self._report_interval,
            request_headers=xet_headers,
        )

        return result.destination_path

    # ── Upload Methods ────────────────────────────────────────────

    def upload_file(
        self,
        file_path: str,
        repo_id: str,
        path_in_repo: Optional[str] = None,
        repo_type: str = "model",
        revision: Optional[str] = None,
        transfer_id: Optional[str] = None,
    ) -> str:
        """Upload a file with progress tracking.

        Automatically selects the best method:
        - **hf_xet direct call** if Xet is available (detailed progress
          with dedup, transfer speed, per-file data)
        - **tqdm monkey-patching** if Xet is not available (basic
          byte-level progress from LFS upload)

        Args:
            file_path: Local path to the file to upload.
            repo_id: Repository ID (e.g. ``"username/model"``).
            path_in_repo: Target path in the repository (defaults to filename).
            repo_type: Repository type.
            revision: Optional git revision.
            transfer_id: Unique transfer identifier (auto-generated if None).

        Returns:
            URL of the uploaded file (standard upload) or content hash (Xet).
        """
        transfer_id = transfer_id or generate_transfer_id()
        filename = os.path.basename(file_path)
        path_in_repo = path_in_repo or filename

        if is_xet_available():
            return self._upload_file_xet(
                file_path=file_path,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                repo_type=repo_type,
                revision=revision,
                transfer_id=transfer_id,
                filename=filename,
            )
        else:
            return self._upload_file_lfs(
                file_path=file_path,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                repo_type=repo_type,
                revision=revision,
                transfer_id=transfer_id,
                filename=filename,
            )

    def upload_bytes(
        self,
        file_content: bytes,
        filename: str,
        repo_id: str,
        path_in_repo: Optional[str] = None,
        repo_type: str = "model",
        revision: Optional[str] = None,
        transfer_id: Optional[str] = None,
    ) -> str:
        """Upload in-memory bytes with progress tracking.

        For Xet: calls ``hf_xet.upload_bytes()`` directly with a
        progress callback.

        For non-Xet: writes bytes to a temporary file first, then
        uploads the file path (enabling tqdm-based progress tracking).

        Args:
            file_content: File content as bytes.
            filename: Name for the file in the repository.
            repo_id: Repository ID.
            path_in_repo: Target path in the repository (defaults to filename).
            repo_type: Repository type.
            revision: Optional git revision.
            transfer_id: Unique transfer identifier.

        Returns:
            URL of the uploaded file or content hash.
        """
        transfer_id = transfer_id or generate_transfer_id()
        path_in_repo = path_in_repo or filename

        if is_xet_available():
            return self._upload_bytes_xet(
                file_content=file_content,
                filename=filename,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                repo_type=repo_type,
                revision=revision,
                transfer_id=transfer_id,
            )
        else:
            return self._upload_bytes_via_temp(
                file_content=file_content,
                filename=filename,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                repo_type=repo_type,
                revision=revision,
                transfer_id=transfer_id,
            )

    # ── Event Consumer Methods ────────────────────────────────────

    def get_events(self, timeout: float = 0) -> List[ProgressEvent]:
        """Get all pending progress events (call from any thread).

        Args:
            timeout: Maximum seconds to wait for the first event.
                Use 0 for non-blocking (default).

        Returns:
            List of ProgressEvent objects.
        """
        events: List[ProgressEvent] = []
        while True:
            try:
                event = self.event_queue.get(timeout=timeout)
                events.append(event)
                timeout = 0  # Only block on first get
            except queue.Empty:
                break
        return events

    def events(
        self, timeout: float = 1.0, stop_on: Optional[EventType] = None
    ) -> Generator[ProgressEvent, None, None]:
        """Yield progress events as they arrive.

        A convenience generator that yields events from the queue.
        Optionally stops when a specific event type is received.

        Args:
            timeout: Maximum seconds to wait for each event.
            stop_on: Stop iteration when an event of this type is
                received (e.g. ``EventType.COMPLETE`` or ``EventType.ERROR``).

        Yields:
            ProgressEvent objects as they arrive.
        """
        while True:
            try:
                event = self.event_queue.get(timeout=timeout)
                yield event
                if stop_on and event.event_type == stop_on:
                    return
            except queue.Empty:
                continue

    def wait_for_complete(
        self,
        transfer_id: str,
        timeout: float = 300,
    ) -> Optional[ProgressEvent]:
        """Block until a transfer completes or errors.

        Consumes events from the queue until a ``COMPLETE`` or ``ERROR``
        event with the matching ``transfer_id`` is found, or the timeout
        expires.

        Args:
            transfer_id: Transfer identifier to wait for.
            timeout: Maximum seconds to wait.

        Returns:
            The final ProgressEvent (COMPLETE or ERROR), or None if
            the timeout expired.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = deadline - time.time()
            try:
                event = self.event_queue.get(
                    timeout=min(remaining, 0.5)
                )
                if event.transfer_id == transfer_id:
                    if event.event_type in (
                        EventType.COMPLETE,
                        EventType.ERROR,
                    ):
                        return event
            except queue.Empty:
                continue
        return None

    # ── Internal: Xet Upload Methods ──────────────────────────────

    def _upload_file_xet(
        self,
        file_path: str,
        repo_id: str,
        path_in_repo: str,
        repo_type: str,
        revision: Optional[str],
        transfer_id: str,
        filename: str,
    ) -> str:
        """Upload using hf_xet with detailed progress callback."""
        from .xet_upload import upload_file_with_xet

        try:
            result = upload_file_with_xet(
                file_path=file_path,
                repo_id=repo_id,
                token=self._token,
                event_queue=self.event_queue,
                repo_type=repo_type,
                revision=revision,
                endpoint=self._endpoint,
                transfer_id=transfer_id,
                report_interval=self._report_interval,
            )
            return result.hash or result.filename
        except ImportError:
            # Fallback to LFS if hf_xet import fails at runtime
            return self._upload_file_lfs(
                file_path=file_path,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                repo_type=repo_type,
                revision=revision,
                transfer_id=transfer_id,
                filename=filename,
            )

    def _upload_bytes_xet(
        self,
        file_content: bytes,
        filename: str,
        repo_id: str,
        path_in_repo: str,
        repo_type: str,
        revision: Optional[str],
        transfer_id: str,
    ) -> str:
        """Upload bytes using hf_xet with detailed progress callback."""
        from .xet_upload import upload_bytes_with_xet

        try:
            result = upload_bytes_with_xet(
                file_content=file_content,
                filename=filename,
                repo_id=repo_id,
                token=self._token,
                event_queue=self.event_queue,
                repo_type=repo_type,
                revision=revision,
                endpoint=self._endpoint,
                transfer_id=transfer_id,
                report_interval=self._report_interval,
            )
            return result.hash or result.filename
        except ImportError:
            return self._upload_bytes_via_temp(
                file_content=file_content,
                filename=filename,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                repo_type=repo_type,
                revision=revision,
                transfer_id=transfer_id,
            )

    # ── Internal: LFS Upload Methods ──────────────────────────────

    def _upload_file_lfs(
        self,
        file_path: str,
        repo_id: str,
        path_in_repo: str,
        repo_type: str,
        revision: Optional[str],
        transfer_id: str,
        filename: str,
    ) -> str:
        """Upload using LFS with tqdm monkey-patching."""
        from .standard_upload import upload_file as _upload_file

        return _upload_file(
            file_path=file_path,
            repo_id=repo_id,
            token=self._token,
            event_queue=self.event_queue,
            path_in_repo=path_in_repo,
            repo_type=repo_type,
            revision=revision,
            endpoint=self._endpoint,
            transfer_id=transfer_id,
            report_interval=self._report_interval,
        )

    def _upload_bytes_via_temp(
        self,
        file_content: bytes,
        filename: str,
        repo_id: str,
        path_in_repo: str,
        repo_type: str,
        revision: Optional[str],
        transfer_id: str,
    ) -> str:
        """Upload bytes by writing to a temp file first (for LFS progress)."""
        from .standard_upload import upload_bytes as _upload_bytes

        return _upload_bytes(
            file_content=file_content,
            filename=filename,
            repo_id=repo_id,
            token=self._token,
            event_queue=self.event_queue,
            path_in_repo=path_in_repo,
            repo_type=repo_type,
            revision=revision,
            endpoint=self._endpoint,
            transfer_id=transfer_id,
            report_interval=self._report_interval,
        )
