"""Web app example: real-time HuggingFace download progress in the browser.

Demonstrates how to integrate ``hf_track`` into a FastAPI web application
with Server-Sent Events (SSE) for live progress streaming.

Run with::

    python app.py

Then open http://localhost:8000 in your browser.

Endpoints:

- ``POST /hf-track/download`` — Start a download (single file or full repo)
- ``POST /hf-track/cancel/{transfer_id}`` — Cancel a running transfer
- ``GET /hf-track/events/{transfer_id}`` — SSE stream for a transfer
- ``GET /hf-track/status`` — List active transfers
- ``DELETE /hf-track/transfer/{transfer_id}`` — Clear transfer state

Xet toggle:
The ``use_xet`` parameter controls whether Xet storage is used.
When disabled, the flag is passed through to the tracker methods,
which forward it to the subprocess worker. The worker sets
``HF_HUB_DISABLE_XET=1`` in the child process BEFORE importing
``huggingface_hub``, so the library's cached constant reflects the
correct value. This approach avoids the problem of the main process's
``huggingface_hub.constants.HF_HUB_DISABLE_XET`` being cached at
import time and ignoring runtime env var changes.

Force re-download:
The ``force_download`` parameter controls whether files are
re-downloaded even if they already exist locally. When True,
the tracker re-downloads all files regardless of cached copies.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from sse_starlette import EventSourceResponse

from hf_track import HfTracker, TransferCancelledError
from hf_track.types import EventType, ProgressEvent, TransferDirection

logger = logging.getLogger("hf_track.web_app")

# ── Create tracker and FastAPI app ────────────────────────────────
tracker = HfTracker()

app = FastAPI(
    title="hf-track Web Demo",
    description="Real-time HuggingFace Hub download progress in the browser",
)

# ── In-memory transfer state ──────────────────────────────────────
_active_transfers: Dict[str, dict] = {}
_transfer_threads: Dict[str, threading.Thread] = {}
_transfer_events: Dict[str, List[ProgressEvent]] = {}
_events_lock = threading.Lock()
_TTL_SECONDS = 3600  # 1 hour expiry for completed transfers


def _cleanup_expired_transfers() -> None:
    """Sweep old, completed transfers from memory."""
    now = time.time()
    expired = [
        tid
        for tid, data in _active_transfers.items()
        if data.get("status") in ("completed", "error", "cancelled")
        and data.get("completed_at", now) < (now - _TTL_SECONDS)
    ]
    for tid in expired:
        _active_transfers.pop(tid, None)
        _transfer_threads.pop(tid, None)
        with _events_lock:
            _transfer_events.pop(tid, None)


# ── Background event router ───────────────────────────────────────
# Drains tracker.event_queue and routes each event to the
# per-transfer buffer (_transfer_events) so SSE streams can read
# them without stealing events from each other.

_event_router_stop = threading.Event()


def _event_router_loop() -> None:
    """Background thread: drain tracker.event_queue → _transfer_events."""
    while not _event_router_stop.is_set():
        try:
            event = tracker.event_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        with _events_lock:
            tid = event.transfer_id
            if tid in _transfer_events:
                _transfer_events[tid].append(event)


_event_router_thread = threading.Thread(
    target=_event_router_loop, daemon=True, name="hf-track-event-router"
)
_event_router_thread.start()


# ── Background worker functions (module-level, not closures) ──────


def _do_download(
    transfer_id: str,
    repo_id: str,
    filename: Optional[str],
    repo_type: str,
    local_dir: Optional[str],
    use_xet: bool,
    force_download: bool = False,
) -> None:
    """Download a single file — runs in a daemon thread.

    The ``use_xet`` flag is passed directly to the tracker method.
    We do NOT set ``HF_HUB_DISABLE_XET`` in the main process because
    ``huggingface_hub`` caches that env var at import time.
    """
    from hf_track.token import is_xet_available
    logger.info(
        "file worker: use_xet=%s, is_xet_available=%s",
        use_xet, is_xet_available(),
    )

    try:
        tracker.download_file(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=local_dir,
            transfer_id=transfer_id,
            use_xet=use_xet,
            force_download=force_download,
        )
        if transfer_id in _active_transfers:
            _active_transfers[transfer_id]["status"] = "completed"
    except TransferCancelledError:
        if transfer_id in _active_transfers:
            _active_transfers[transfer_id]["status"] = "cancelled"
    except (OSError, ConnectionError, ValueError) as e:
        if transfer_id in _active_transfers:
            _active_transfers[transfer_id]["status"] = "error"
            _active_transfers[transfer_id]["error"] = str(e)
    except Exception as e:
        if transfer_id in _active_transfers:
            _active_transfers[transfer_id]["status"] = "error"
            _active_transfers[transfer_id]["error"] = f"Unexpected error: {e}"
    finally:
        if transfer_id in _active_transfers:
            _active_transfers[transfer_id]["completed_at"] = time.time()


def _do_download_snapshot(
    transfer_id: str,
    repo_id: str,
    allow_patterns: Optional[str],
    repo_type: str,
    local_dir: Optional[str],
    use_xet: bool,
    force_download: bool = False,
) -> None:
    """Download an entire repo (or filtered subset) — runs in a daemon thread.

    The ``use_xet`` flag is passed directly to the tracker method.
    We do NOT set ``HF_HUB_DISABLE_XET`` in the main process because
    ``huggingface_hub`` caches that env var at import time.
    """
    from hf_track.token import is_xet_available
    logger.info(
        "snapshot worker: use_xet=%s, is_xet_available=%s",
        use_xet, is_xet_available(),
    )

    try:
        patterns = [allow_patterns] if allow_patterns else None
        tracker.download_snapshot(
            repo_id=repo_id,
            allow_patterns=patterns,
            repo_type=repo_type,
            local_dir=local_dir,
            transfer_id=transfer_id,
            use_xet=use_xet,
            force_download=force_download,
        )
        if transfer_id in _active_transfers:
            _active_transfers[transfer_id]["status"] = "completed"
    except TransferCancelledError:
        if transfer_id in _active_transfers:
            _active_transfers[transfer_id]["status"] = "cancelled"
    except (OSError, ConnectionError, ValueError) as e:
        if transfer_id in _active_transfers:
            _active_transfers[transfer_id]["status"] = "error"
            _active_transfers[transfer_id]["error"] = str(e)
    except Exception as e:
        if transfer_id in _active_transfers:
            _active_transfers[transfer_id]["status"] = "error"
            _active_transfers[transfer_id]["error"] = f"Unexpected error: {e}"
    finally:
        if transfer_id in _active_transfers:
            _active_transfers[transfer_id]["completed_at"] = time.time()


# ── API endpoints (module-level, no closures) ─────────────────────


@app.post("/hf-track/download")
async def start_download(
    repo_id: str,
    filename: Optional[str] = None,
    local_dir: Optional[str] = None,
    use_xet: bool = True,
    repo_type: str = "model",
    allow_patterns: Optional[str] = None,
    force_download: bool = False,
):
    """Start a download in a background thread.

    If *filename* is provided, downloads a single file.
    If *filename* is omitted, downloads the entire repository snapshot.
    If *local_dir* is provided, files are saved there instead of the HF cache.
    If *use_xet* is False, forces standard HTTP download (no Xet).
    If *allow_patterns* is provided (snapshot only), only matching files are downloaded.
    If *force_download* is True, re-downloads files even if they already exist locally.

    .. note:: The ``use_xet`` flag is passed directly to the tracker
        methods. We do NOT set ``HF_HUB_DISABLE_XET`` in the main
        process because ``huggingface_hub`` caches that env var at
        import time — changing it at runtime has no effect on the
        library's internal xet routing. Instead, the flag is forwarded
        to the subprocess worker which sets the env var BEFORE
        importing ``huggingface_hub`` in the child process.
    """
    transfer_id = str(uuid.uuid4())
    display_name = filename or f"{repo_id} (full repo)"
    _active_transfers[transfer_id] = {
        "direction": "download",
        "repo_id": repo_id,
        "filename": display_name,
        "local_dir": local_dir,
        "use_xet": use_xet,
        "is_snapshot": filename is None,
        "status": "running",
        "started_at": time.time(),
    }
    # Initialize per-transfer event buffer so the event router
    # can start recording events for this transfer immediately.
    with _events_lock:
        _transfer_events[transfer_id] = []

    # NOTE: We do NOT set HF_HUB_DISABLE_XET here. That env var is
    # cached at import time by huggingface_hub.constants, so changing
    # it in the main process has no effect on huggingface_hub's
    # internal xet routing. Instead, the use_xet flag is passed
    # through to the subprocess worker, which sets the env var
    # BEFORE importing huggingface_hub in the child process.

    _cleanup_expired_transfers()

    if filename:
        target = _do_download
        args = (transfer_id, repo_id, filename, repo_type, local_dir, use_xet, force_download)
    else:
        target = _do_download_snapshot
        args = (transfer_id, repo_id, allow_patterns, repo_type, local_dir, use_xet, force_download)

    thread = threading.Thread(target=target, args=args, daemon=True)
    _transfer_threads[transfer_id] = thread
    thread.start()
    return {
        "transfer_id": transfer_id,
        "is_snapshot": filename is None,
        "use_xet": use_xet,
        "force_download": force_download,
    }


@app.get("/hf-track/events/{transfer_id}")
async def stream_events(transfer_id: str, request: Request):
    """Stream progress events for a transfer via SSE.

    Reads from the per-transfer event buffer instead of the global
    queue, so concurrent SSE streams don't steal each other's events.
    Uses a cursor-based approach: new events appended by the event
    router thread are picked up on each poll iteration.
    """

    async def event_stream():
        cursor = 0
        poll_count = 0
        while True:
            if await request.is_disconnected():
                return
            # Read new events from the per-transfer buffer
            with _events_lock:
                events = list(_transfer_events.get(transfer_id, []))
                new_events = events[cursor:]
            if new_events:
                for event in new_events:
                    cursor += 1
                    yield {"data": json.dumps(event.to_dict())}
                    if event.event_type in (
                        EventType.COMPLETE,
                        EventType.ERROR,
                        EventType.CANCELLED,
                    ):
                        return
            # If the transfer is no longer running and we've sent all
            # its events, close the stream.
            transfer = _active_transfers.get(transfer_id, {})
            if (
                transfer.get("status") in ("completed", "error", "cancelled")
                and not new_events
            ):
                return
            poll_count += 1
            await asyncio.sleep(0.1)

    return EventSourceResponse(event_stream())


@app.get("/hf-track/status")
async def get_status():
    """List all active and recent transfers."""
    _cleanup_expired_transfers()
    return {
        "active_transfers": _active_transfers,
        "queue_size": tracker.event_queue.qsize(),
    }


@app.post("/hf-track/cancel/{transfer_id}")
async def cancel_transfer(transfer_id: str):
    """Cancel a running transfer.

    Calls ``tracker.cancel()`` which:
    1. Sets the cancellation flag (checked by is_cancelled hooks)
    2. Terminates any Xet subprocess associated with this transfer

    Also emits a CANCELLED event so the SSE stream terminates
    immediately, and waits briefly for the worker thread to finish.
    """
    if transfer_id not in _active_transfers:
        return {"status": "not_found", "message": f"Transfer {transfer_id} not found."}

    transfer = _active_transfers[transfer_id]
    direction_str = transfer.get("direction", "download")
    direction = TransferDirection(direction_str)
    filename = transfer.get("filename", "")

    # tracker.cancel() sets the flag AND terminates the Xet subprocess
    tracker.cancel(transfer_id)

    # Emit a CANCELLED event so the SSE stream terminates immediately
    # (the worker thread may be blocked in a long HTTP read and not
    # check the is_cancelled hook for a while).
    cancelled_event = ProgressEvent.cancelled_event(
        transfer_id=transfer_id,
        direction=direction,
        filename=filename,
    )
    # Put into the global queue (for any direct consumers)
    tracker.event_queue.put(cancelled_event)
    # Also put into the per-transfer buffer (for SSE streams)
    with _events_lock:
        if transfer_id in _transfer_events:
            _transfer_events[transfer_id].append(cancelled_event)

    _active_transfers[transfer_id]["status"] = "cancelled"

    # Wait for the worker thread to finish
    thread = _transfer_threads.get(transfer_id)
    if thread and thread.is_alive():
        thread.join(timeout=3.0)

    return {"status": "success", "message": f"Transfer {transfer_id} cancelled."}


@app.delete("/hf-track/transfer/{transfer_id}")
async def clear_transfer(transfer_id: str):
    """Explicitly clear a transfer from memory."""
    if transfer_id in _active_transfers:
        _active_transfers.pop(transfer_id)
        with _events_lock:
            _transfer_events.pop(transfer_id, None)
        return {"status": "success", "message": "Transfer state cleared."}
    return {"status": "not_found", "message": "Transfer ID not found."}


# ── Serve static frontend files ───────────────────────────────────
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


# ── Run with uvicorn ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
