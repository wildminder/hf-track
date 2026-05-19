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

    def test_download_endpoint_returns_transfer_id(self, client):
        """POST /hf-track/download returns a transfer_id."""
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "test/repo", "filename": "test.bin"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "transfer_id" in data
        assert isinstance(data["transfer_id"], str)

    def test_upload_endpoint_returns_transfer_id(self, client):
        """POST /hf-track/upload returns a transfer_id."""
        resp = client.post(
            "/hf-track/upload",
            params={"repo_id": "test/repo", "file_path": "/tmp/test.bin"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "transfer_id" in data
        assert isinstance(data["transfer_id"], str)

    def test_cancel_endpoint(self, client):
        """POST /hf-track/cancel/{id} returns success."""
        resp = client.post("/hf-track/cancel/test-transfer-id")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

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

    def test_download_missing_params(self, client):
        """POST without required params returns 422."""
        resp = client.post("/hf-track/download")
        assert resp.status_code == 422

    def test_upload_missing_params(self, client):
        """POST without required params returns 422."""
        resp = client.post("/hf-track/upload")
        assert resp.status_code == 422

    def test_events_route_exists(self, client):
        """GET /hf-track/events/{id} is a registered route."""
        # We can't easily test the SSE stream with TestClient,
        # but we can verify the route exists by checking 200/404
        # (the endpoint will hang if we try to stream, so just check routes)
        routes = [route.path for route in client.app.routes]
        assert "/hf-track/events/{transfer_id}" in routes
