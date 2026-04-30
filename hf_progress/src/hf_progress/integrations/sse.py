"""FastAPI/SSE integration for real-time progress streaming.

Provides ready-to-use FastAPI endpoints that stream progress events
from ``HfProgressTracker`` to frontend clients via Server-Sent Events.

Usage::

    from fastapi import FastAPI
    from hf_progress import HfProgressTracker
    from hf_progress.integrations.sse import create_progress_router

    app = FastAPI()
    tracker = HfProgressTracker(token="hf_...")
    router = create_progress_router(tracker)
    app.include_router(router)

The router adds these endpoints:

- ``POST /hf-progress/upload`` — Start an upload
- ``POST /hf-progress/download`` — Start a download
- ``GET /hf-progress/events/{transfer_id}`` — SSE stream for a transfer
- ``GET /hf-progress/status`` — List active transfers
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from typing import Dict, Optional

from ..types import EventType, ProgressEvent, TransferDirection


def create_progress_router(tracker, prefix: str = "/hf-progress"):
    """Create a FastAPI APIRouter with progress tracking endpoints.

    Args:
        tracker: An ``HfProgressTracker`` instance.
        prefix: URL prefix for all endpoints.

    Returns:
        A ``fastapi.APIRouter`` instance.

    Example::

        from hf_progress import HfProgressTracker
        from hf_progress.integrations.sse import create_progress_router

        tracker = HfProgressTracker(token="hf_...")
        router = create_progress_router(tracker)
        app.include_router(router)
    """
    try:
        from fastapi import APIRouter
        from fastapi.responses import StreamingResponse
    except ImportError:
        raise ImportError(
            "FastAPI is required for SSE integration. "
            "Install with: pip install 'hf-progress[sse]'"
        )

    router = APIRouter(prefix=prefix, tags=["progress"])
    _active_transfers: Dict[str, dict] = {}

    @router.post("/upload")
    async def start_upload(
        repo_id: str,
        file_path: str,
        path_in_repo: Optional[str] = None,
        repo_type: str = "model",
    ):
        """Start a file upload in a background thread.

        Returns a ``transfer_id`` that can be used to stream progress
        events via the ``/events/{transfer_id}`` endpoint.
        """
        transfer_id = str(uuid.uuid4())
        _active_transfers[transfer_id] = {
            "direction": "upload",
            "repo_id": repo_id,
            "filename": file_path,
            "status": "running",
        }

        def do_upload():
            try:
                tracker.upload_file(
                    file_path=file_path,
                    repo_id=repo_id,
                    path_in_repo=path_in_repo,
                    repo_type=repo_type,
                    transfer_id=transfer_id,
                )
                _active_transfers[transfer_id]["status"] = "completed"
            except Exception as e:
                _active_transfers[transfer_id]["status"] = "error"
                _active_transfers[transfer_id]["error"] = str(e)

        thread = threading.Thread(target=do_upload, daemon=True)
        thread.start()

        return {"transfer_id": transfer_id}

    @router.post("/download")
    async def start_download(
        repo_id: str,
        filename: str,
        repo_type: str = "model",
    ):
        """Start a file download in a background thread.

        Returns a ``transfer_id`` for progress streaming.
        """
        transfer_id = str(uuid.uuid4())
        _active_transfers[transfer_id] = {
            "direction": "download",
            "repo_id": repo_id,
            "filename": filename,
            "status": "running",
        }

        def do_download():
            try:
                tracker.download_file(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type=repo_type,
                    transfer_id=transfer_id,
                )
                _active_transfers[transfer_id]["status"] = "completed"
            except Exception as e:
                _active_transfers[transfer_id]["status"] = "error"
                _active_transfers[transfer_id]["error"] = str(e)

        thread = threading.Thread(target=do_download, daemon=True)
        thread.start()

        return {"transfer_id": transfer_id}

    @router.get("/events/{transfer_id}")
    async def stream_progress(transfer_id: str):
        """Stream progress events for a transfer via SSE.

        The stream sends ``data: {...}\\n\\n`` messages containing
        JSON-serialized ``ProgressEvent`` dicts. The stream closes
        when a ``COMPLETE`` or ``ERROR`` event is received for the
        matching ``transfer_id``.
        """
        from sse_starlette.sse import EventSourceResponse

        async def event_stream():
            while True:
                events = tracker.get_events()
                for event in events:
                    if event.transfer_id == transfer_id:
                        data = json.dumps(event.to_dict())
                        yield {"data": data}
                        if event.event_type in (
                            EventType.COMPLETE,
                            EventType.ERROR,
                        ):
                            return
                await asyncio.sleep(0.1)

        return EventSourceResponse(event_stream())

    @router.get("/status")
    async def get_status():
        """List all active and recent transfers."""
        return {
            "active_transfers": _active_transfers,
            "queue_size": tracker.event_queue.qsize(),
        }

    return router


def create_raw_sse_stream(tracker, transfer_id: str):
    """Create a raw SSE event generator without FastAPI dependency.

    Useful for integrating with any ASGI framework or custom
    streaming response class.

    Args:
        tracker: An ``HfProgressTracker`` instance.
        transfer_id: Transfer identifier to filter events for.

    Yields:
        SSE-formatted strings: ``data: {...}\\n\\n``
    """
    import asyncio

    async def event_stream():
        while True:
            events = tracker.get_events()
            for event in events:
                if event.transfer_id == transfer_id:
                    data = json.dumps(event.to_dict())
                    yield f"data: {data}\n\n"
                    if event.event_type in (
                        EventType.COMPLETE,
                        EventType.ERROR,
                    ):
                        return
            await asyncio.sleep(0.1)

    return event_stream()
