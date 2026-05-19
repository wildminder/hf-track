"""Web app example: real-time HuggingFace download progress in the browser.

Demonstrates how to integrate ``hf_track`` into a FastAPI web application
with Server-Sent Events (SSE) for live progress streaming.

Run with::

    python app.py

Then open http://localhost:8000 in your browser.

Endpoints:

- ``POST /hf-track/download`` — Start a download
- ``POST /hf-track/upload`` — Start an upload
- ``POST /hf-track/cancel/{transfer_id}`` — Cancel a running transfer
- ``GET  /hf-track/events/{transfer_id}`` — SSE stream for a transfer
- ``GET  /hf-track/status`` — List active transfers
- ``DELETE /hf-track/transfer/{transfer_id}`` — Clear transfer state
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from sse_starlette import EventSourceResponse

from hf_track import HfTracker
from hf_track.types import EventType

# ── Create tracker and FastAPI app ────────────────────────────────
tracker = HfTracker()

app = FastAPI(
    title="hf-track Web Demo",
    description="Real-time HuggingFace Hub download progress in the browser",
)

# ── In-memory transfer state ──────────────────────────────────────
_active_transfers: Dict[str, dict] = {}
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


# ── Background worker functions (module-level, not closures) ──────


def _do_upload(
    transfer_id: str,
    file_path: str,
    repo_id: str,
    path_in_repo: Optional[str],
    repo_type: str,
) -> None:
    """Upload worker — runs in a daemon thread."""
    try:
        tracker.upload_file(
            file_path=file_path,
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            repo_type=repo_type,
            transfer_id=transfer_id,
        )
        if transfer_id in _active_transfers:
            _active_transfers[transfer_id]["status"] = "completed"
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


def _do_download(
    transfer_id: str,
    repo_id: str,
    filename: str,
    repo_type: str,
) -> None:
    """Download worker — runs in a daemon thread."""
    try:
        tracker.download_file(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            transfer_id=transfer_id,
        )
        if transfer_id in _active_transfers:
            _active_transfers[transfer_id]["status"] = "completed"
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


@app.post("/hf-track/upload")
async def start_upload(
    repo_id: str,
    file_path: str,
    path_in_repo: Optional[str] = None,
    repo_type: str = "model",
):
    """Start a file upload in a background thread."""
    transfer_id = str(uuid.uuid4())
    _active_transfers[transfer_id] = {
        "direction": "upload",
        "repo_id": repo_id,
        "filename": file_path,
        "status": "running",
        "started_at": time.time(),
    }
    _cleanup_expired_transfers()

    thread = threading.Thread(
        target=_do_upload,
        args=(transfer_id, file_path, repo_id, path_in_repo, repo_type),
        daemon=True,
    )
    thread.start()
    return {"transfer_id": transfer_id}


@app.post("/hf-track/download")
async def start_download(
    repo_id: str,
    filename: str,
    repo_type: str = "model",
):
    """Start a file download in a background thread."""
    transfer_id = str(uuid.uuid4())
    _active_transfers[transfer_id] = {
        "direction": "download",
        "repo_id": repo_id,
        "filename": filename,
        "status": "running",
        "started_at": time.time(),
    }
    _cleanup_expired_transfers()

    thread = threading.Thread(
        target=_do_download,
        args=(transfer_id, repo_id, filename, repo_type),
        daemon=True,
    )
    thread.start()
    return {"transfer_id": transfer_id}


@app.get("/hf-track/events/{transfer_id}")
async def stream_events(transfer_id: str, request: Request):
    """Stream progress events for a transfer via SSE."""

    async def event_stream():
        while True:
            if await request.is_disconnected():
                return
            events = tracker.get_events()
            for event in events:
                if event.transfer_id == transfer_id:
                    yield {"data": json.dumps(event.to_dict())}
                    if event.event_type in (
                        EventType.COMPLETE,
                        EventType.ERROR,
                        EventType.CANCELLED,
                    ):
                        return
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
    """Cancel a running transfer."""
    tracker.cancel(transfer_id)
    if transfer_id in _active_transfers:
        _active_transfers[transfer_id]["status"] = "cancelled"
    return {"status": "success", "message": f"Transfer {transfer_id} cancelled."}


@app.delete("/hf-track/transfer/{transfer_id}")
async def clear_transfer(transfer_id: str):
    """Explicitly clear a transfer from memory."""
    if transfer_id in _active_transfers:
        _active_transfers.pop(transfer_id)
        return {"status": "success", "message": "Transfer state cleared."}
    return {"status": "not_found", "message": "Transfer ID not found."}


# ── Serve static frontend files ───────────────────────────────────
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


# ── Run with uvicorn ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
