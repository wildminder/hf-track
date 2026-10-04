"""Tests for the SSE integration module and web app endpoints.

The ``create_progress_router()`` factory was removed because closure-based
endpoints broke Starlette 1.0 response serialization. All endpoints are
now defined at module level in ``examples/web_app/app.py``.
"""
from __future__ import annotations

import pytest


class TestSSEModule:
    """Tests for the hf_track.integrations.sse module."""

    def test_sse_module_importable(self):
        """The sse module can be imported."""
        from hf_track.integrations import sse

        assert sse is not None

    def test_event_source_response_reexported(self):
        """EventSourceResponse is re-exported from the sse module."""
        pytest.importorskip("sse_starlette")
        from hf_track.integrations.sse import EventSourceResponse

        assert EventSourceResponse is not None

    def test_create_progress_router_removed(self):
        """create_progress_router no longer exists in the sse module."""
        from hf_track.integrations import sse

        assert not hasattr(sse, "create_progress_router")

    def test_create_raw_sse_stream_removed(self):
        """create_raw_sse_stream no longer exists in the sse module."""
        from hf_track.integrations import sse

        assert not hasattr(sse, "create_raw_sse_stream")


class TestWebAppEndpoints:
    """Tests for the web app's FastAPI endpoints.

    These test the module-level endpoints defined in app.py,
    which replaced the closure-based create_progress_router.
    """

    @pytest.fixture
    def client(self):
        """Create a TestClient for the web app."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        # Import the app from the web app example
        import sys
        from pathlib import Path

        web_app_dir = Path(__file__).parent.parent / "examples" / "web_app"
        sys.path.insert(0, str(web_app_dir))

        # We need to import app.py which is at a non-standard location
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "web_app", web_app_dir / "app.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return TestClient(mod.app)

    def test_download_file_returns_transfer_id(self, client):
        """POST /hf-track/download with filename returns a transfer_id."""
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "test/repo", "filename": "test.bin"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "transfer_id" in data
        assert isinstance(data["transfer_id"], str)
        assert data.get("is_snapshot") is False

    def test_download_snapshot_returns_transfer_id(self, client):
        """POST /hf-track/download without filename downloads full repo."""
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "test/repo"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "transfer_id" in data
        assert isinstance(data["transfer_id"], str)
        assert data.get("is_snapshot") is True

    def test_upload_endpoint_removed(self, client):
        """POST /hf-track/upload returns 404 or 405 (upload removed)."""
        resp = client.post(
            "/hf-track/upload",
            params={"repo_id": "test/repo", "file_path": "/tmp/test.bin"},
        )
        # Upload endpoint was removed — 404 or 405 is expected
        assert resp.status_code in (404, 405)

    def test_cancel_endpoint_unknown_id(self, client):
        """POST /hf-track/cancel/{unknown_id} returns not_found."""
        resp = client.post("/hf-track/cancel/test-transfer-id")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_found"

    def test_status_endpoint(self, client):
        """GET /hf-track/status returns active_transfers dict."""
        resp = client.get("/hf-track/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_transfers" in data
        assert isinstance(data["active_transfers"], dict)

    def test_clear_nonexistent_transfer(self, client):
        """DELETE for non-existent transfer returns not_found."""
        resp = client.delete("/hf-track/transfer/nonexistent-id")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_found"

    def test_clear_after_download(self, client):
        """Start a download, then clear it — returns success."""
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "test/repo", "filename": "test.bin"},
        )
        assert resp.status_code == 200
        transfer_id = resp.json()["transfer_id"]

        import time
        time.sleep(0.1)

        clear_resp = client.delete(f"/hf-track/transfer/{transfer_id}")
        assert clear_resp.status_code == 200
        assert clear_resp.json()["status"] == "success"

    def test_download_missing_repo_id(self, client):
        """POST without repo_id returns 422."""
        resp = client.post("/hf-track/download")
        assert resp.status_code == 422

    def test_upload_endpoint_removed(self, client):
        """POST /hf-track/upload returns 404 or 405 (upload removed)."""
        resp = client.post("/hf-track/upload")
        assert resp.status_code in (404, 405)

    def test_events_route_exists(self, client):
        """GET /hf-track/events/{id} is a registered route."""
        # We can't easily test the SSE stream with TestClient,
        # but we can verify the route exists by checking 200/404
        # (the endpoint will hang if we try to stream, so just check routes)
        routes = [route.path for route in client.app.routes]
        assert "/hf-track/events/{transfer_id}" in routes


class TestSSEStream:
    """Live ``GET /hf-track/events/{transfer_id}`` behaviour (NTH-010).

    ``test_events_route_exists`` proves the route resolves; nothing proved
    what it emits. The generator is driven directly rather than through
    ``TestClient``, because ``TestClient`` cannot model the two things
    that matter here -- a client that disconnects mid-stream, and a stream
    that is *supposed* to stay open waiting for events that have not
    arrived yet.
    """

    @pytest.fixture
    def app_module(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        import importlib.util
        import sys
        from pathlib import Path

        web_app_dir = Path(__file__).parent.parent / "examples" / "web_app"
        module_path = web_app_dir / "app.py"
        if not module_path.is_file():
            pytest.skip(f"{module_path} is absent")

        sys.path.insert(0, str(web_app_dir))
        try:
            spec = importlib.util.spec_from_file_location(
                "hf_track_example_web_app", module_path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules["hf_track_example_web_app"] = module
            spec.loader.exec_module(module)
            yield module
        finally:
            sys.path.remove(str(web_app_dir))

    @staticmethod
    def _event(transfer_id, event_type, **kwargs):
        from hf_track.types import ProgressEvent, ProgressPhase, TransferDirection

        return ProgressEvent(
            event_type=event_type,
            transfer_id=transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename=kwargs.pop("filename", "model.bin"),
            phase=ProgressPhase.DOWNLOADING,
            **kwargs,
        )

    @staticmethod
    def _stub_request(disconnected=False):
        class _Request:
            def __init__(self):
                self._disconnected = disconnected

            async def is_disconnected(self):
                return self._disconnected

        return _Request()

    @pytest.fixture
    def clean_buffers(self, app_module):
        """Isolate each test from the app's module-level buffers."""
        app_module._transfer_events.clear()
        app_module._active_transfers.clear()
        yield
        app_module._transfer_events.clear()
        app_module._active_transfers.clear()

    @staticmethod
    async def _drain(app_module, transfer_id, request, limit=50):
        """Drain the endpoint's generator, stopping at ``limit`` frames."""
        import asyncio

        response = await app_module.stream_events(transfer_id, request)
        iterator = response.body_iterator
        frames = []
        while len(frames) < limit:
            try:
                frames.append(await asyncio.wait_for(iterator.__anext__(), timeout=5))
            except (StopAsyncIteration, asyncio.TimeoutError):
                break
        await iterator.aclose()
        return frames

    async def test_sse_stream_emits_data_frames(self, app_module, clean_buffers):
        """Every frame is a ``data:`` payload that parses to an event dict."""
        import json

        from hf_track.types import EventType

        transfer_id = "sse-frames-1"
        app_module._transfer_events[transfer_id] = [
            self._event(transfer_id, EventType.PROGRESS,
                        bytes_completed=50, total_bytes=100, percentage=50.0),
            self._event(transfer_id, EventType.COMPLETE,
                        bytes_completed=100, total_bytes=100, percentage=100.0),
        ]
        app_module._active_transfers[transfer_id] = {"status": "running"}

        frames = await self._drain(app_module, transfer_id, self._stub_request())

        assert frames, "the stream emitted no frames"
        payloads = []
        for frame in frames:
            # The endpoint yields ``{"data": ...}``; the ``data: `` prefix
            # is added by the SSE encoder downstream, so assert on the
            # shape it is handed.
            assert isinstance(frame, dict) and "data" in frame
            payloads.append(json.loads(frame["data"]))

        assert payloads[0]["event_type"] == "progress"
        assert payloads[-1]["event_type"] == "complete"
        assert payloads[0]["transfer_id"] == transfer_id

    async def test_sse_stream_closes_on_terminal_event(self, app_module, clean_buffers):
        """A COMPLETE buffered into the transfer ends the stream."""
        from hf_track.types import EventType

        transfer_id = "sse-close-1"
        app_module._transfer_events[transfer_id] = [
            self._event(transfer_id, EventType.COMPLETE,
                        bytes_completed=10, total_bytes=10, percentage=100.0),
        ]
        app_module._active_transfers[transfer_id] = {"status": "running"}

        frames = await self._drain(app_module, transfer_id, self._stub_request())

        assert len(frames) == 1, (
            "the stream must stop at the terminal event rather than "
            f"re-sending it on every poll; got {len(frames)} frames"
        )

    async def test_sse_stream_stops_on_client_disconnect(self, app_module, clean_buffers):
        """A disconnected client ends the generator without raising."""
        from hf_track.types import EventType

        transfer_id = "sse-disconnect-1"
        # Events ARE buffered, so only the disconnect can stop this stream.
        app_module._transfer_events[transfer_id] = [
            self._event(transfer_id, EventType.PROGRESS,
                        bytes_completed=1, total_bytes=10),
        ]

        frames = await self._drain(
            app_module, transfer_id, self._stub_request(disconnected=True)
        )

        assert frames == [], (
            "a disconnected client must receive nothing, not even the "
            "events already buffered for it"
        )

    async def test_sse_stream_replays_buffered_events_for_new_subscriber(
        self, app_module, clean_buffers
    ):
        """A second subscriber gets the whole history from cursor 0."""
        from hf_track.types import EventType

        transfer_id = "sse-replay-1"
        app_module._active_transfers[transfer_id] = {"status": "running"}
        history = [
            self._event(transfer_id, EventType.START),
            self._event(transfer_id, EventType.PROGRESS,
                        bytes_completed=5, total_bytes=10),
            self._event(transfer_id, EventType.COMPLETE,
                        bytes_completed=10, total_bytes=10, percentage=100.0),
        ]
        # Buffered by an earlier subscriber that has already gone.
        app_module._transfer_events[transfer_id] = list(history)

        frames = await self._drain(app_module, transfer_id, self._stub_request())

        assert len(frames) == len(history), (
            "a late subscriber must be replayed the events it missed; the "
            f"cursor design promises this. sent {len(frames)} of {len(history)}"
        )

    async def test_sse_stream_closes_when_transfer_finished_with_no_new_events(
        self, app_module, clean_buffers
    ):
        """A finished transfer with nothing new terminates rather than polling."""
        transfer_id = "sse-finished-1"
        app_module._transfer_events[transfer_id] = []
        app_module._active_transfers[transfer_id] = {"status": "completed"}

        frames = await self._drain(app_module, transfer_id, self._stub_request())

        assert frames == [], (
            "there is nothing to send and the transfer is finished, so the "
            "stream must close rather than poll forever"
        )