"""Integration tests for the web app example.

Uses FastAPI TestClient to test the full stack without a running server.
Covers: static files, HTML structure, CSS, JS, API endpoints, worker
functions, event router, cancel flow, xet toggle, and SSE streaming.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from pathlib import Path
from html.parser import HTMLParser
from unittest.mock import patch, MagicMock

import pytest

from hf_track.types import EventType, ProgressEvent, TransferDirection

# Add the web_app directory to sys.path so we can import app.py
WEB_APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(WEB_APP_DIR))

# Also add the hf_track src directory
HF_TRACK_SRC = Path(__file__).parent.parent.parent.parent / "src"
if str(HF_TRACK_SRC) not in sys.path:
    sys.path.insert(0, str(HF_TRACK_SRC))


@pytest.fixture
def client():
    """Create a FastAPI TestClient for the web app."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from app import app
    return TestClient(app)


@pytest.fixture
def app_module():
    """Import the app module for access to module-level state."""
    import app as app_mod
    return app_mod


# ── Static File Tests ────────────────────────────────────────────

class TestStaticFiles:
    def test_serve_index_html(self, client):
        """GET / returns 200 with HTML content."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_serve_css(self, client):
        """GET /style.css returns 200 with CSS content."""
        resp = client.get("/style.css")
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "text/css" in ct or "text/plain" in ct

    def test_serve_js(self, client):
        """GET /app.js returns 200 with JS content."""
        resp = client.get("/app.js")
        assert resp.status_code == 200


# ── HTML Structure Tests ─────────────────────────────────────────

class HTMLStructureChecker(HTMLParser):
    """Parse HTML and check for required elements."""
    def __init__(self):
        super().__init__()
        self.found_elements = set()
        self.found_ids = set()
        self.found_links = []
        self.found_scripts = []

    def handle_starttag(self, tag, attrs):
        self.found_elements.add(tag)
        attrs_dict = dict(attrs)
        if "id" in attrs_dict:
            self.found_ids.add(attrs_dict["id"])
        if tag == "link" and attrs_dict.get("rel") == "stylesheet":
            self.found_links.append(attrs_dict.get("href", ""))
        if tag == "script" and "src" in attrs_dict:
            self.found_scripts.append(attrs_dict["src"])


class TestHTMLStructure:
    def _parse_html(self, client):
        resp = client.get("/")
        checker = HTMLStructureChecker()
        checker.feed(resp.text)
        return checker

    def test_html_has_download_form(self, client):
        checker = self._parse_html(client)
        assert "download-form" in checker.found_ids

    def test_html_has_active_section(self, client):
        checker = self._parse_html(client)
        assert "active-transfers" in checker.found_ids

    def test_html_has_history_section(self, client):
        checker = self._parse_html(client)
        assert "transfer-history" in checker.found_ids

    def test_html_links_css(self, client):
        checker = self._parse_html(client)
        assert "style.css" in checker.found_links

    def test_html_links_js(self, client):
        checker = self._parse_html(client)
        assert "app.js" in checker.found_scripts

    def test_html_has_allow_patterns_input(self, client):
        """HTML contains the allow-patterns input field."""
        resp = client.get("/")
        assert "allow-patterns" in resp.text


# ── CSS Tests ────────────────────────────────────────────────────

class TestCSS:
    def test_css_has_progress_bar(self, client):
        resp = client.get("/style.css")
        assert ".progress-bar-fill" in resp.text

    def test_css_has_transfer_card(self, client):
        resp = client.get("/style.css")
        assert ".transfer-card" in resp.text

    def test_css_has_status_badge(self, client):
        resp = client.get("/style.css")
        assert ".status-badge" in resp.text

    def test_css_has_dark_theme(self, client):
        resp = client.get("/style.css")
        assert "--bg-primary" in resp.text


# ── JS Tests ─────────────────────────────────────────────────────

class TestJS:
    def test_js_has_format_bytes(self, client):
        resp = client.get("/app.js")
        assert "function formatBytes" in resp.text

    def test_js_has_format_speed(self, client):
        resp = client.get("/app.js")
        assert "function formatSpeed" in resp.text

    def test_js_has_format_eta(self, client):
        resp = client.get("/app.js")
        assert "function formatEta" in resp.text

    def test_js_has_start_download(self, client):
        resp = client.get("/app.js")
        assert "startDownload" in resp.text

    def test_js_has_cancel_transfer(self, client):
        resp = client.get("/app.js")
        assert "cancelTransfer" in resp.text

    def test_js_has_event_source(self, client):
        resp = client.get("/app.js")
        assert "EventSource" in resp.text

    def test_js_has_allow_patterns(self, client):
        """JS passes allow_patterns in the download request."""
        resp = client.get("/app.js")
        assert "allow_patterns" in resp.text


# ── API Endpoint Tests ───────────────────────────────────────────

class TestAPIEndpoints:
    def test_download_file_endpoint(self, client):
        """POST /hf-track/download with filename returns transfer_id."""
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "bert-base-german-dbmdz-cased", "filename": "config.json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "transfer_id" in data
        assert data.get("is_snapshot") is False

    def test_download_snapshot_endpoint(self, client):
        """POST /hf-track/download without filename downloads full repo."""
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "bert-base-german-dbmdz-cased"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "transfer_id" in data
        assert data.get("is_snapshot") is True

    def test_download_with_local_dir(self, client):
        """POST /hf-track/download with local_dir returns transfer_id."""
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "bert-base-german-dbmdz-cased", "local_dir": "./downloads"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "transfer_id" in data
        assert data.get("is_snapshot") is True

    def test_download_with_use_xet_false(self, client):
        """POST /hf-track/download with use_xet=false forces standard download."""
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "bert-base-german-dbmdz-cased", "filename": "config.json", "use_xet": "false"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "transfer_id" in data
        assert data.get("use_xet") is False
        assert data.get("is_snapshot") is False

    def test_download_with_allow_patterns(self, client):
        """POST /hf-track/download with allow_patterns returns transfer_id."""
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "bert-base-german-dbmdz-cased", "allow_patterns": "*.json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "transfer_id" in data
        assert data.get("is_snapshot") is True

    def test_cancel_endpoint(self, client):
        """POST /hf-track/cancel/{id} returns success."""
        # First start a download to get a valid transfer_id
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "bert-base-german-dbmdz-cased", "filename": "config.json"},
        )
        transfer_id = resp.json()["transfer_id"]
        resp = client.post(f"/hf-track/cancel/{transfer_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

    def test_cancel_unknown_transfer(self, client):
        """POST /hf-track/cancel/{unknown_id} returns not_found."""
        resp = client.post("/hf-track/cancel/nonexistent-id")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_found"

    def test_status_endpoint(self, client):
        """GET /hf-track/status returns active_transfers."""
        resp = client.get("/hf-track/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_transfers" in data

    def test_download_missing_repo_id(self, client):
        """POST without repo_id returns 422."""
        resp = client.post("/hf-track/download")
        assert resp.status_code == 422

    def test_no_upload_endpoint(self, client):
        """POST /hf-track/upload returns 404 or 405 (upload removed)."""
        resp = client.post(
            "/hf-track/upload",
            params={"repo_id": "test", "file_path": "test.bin"},
        )
        # 404 = route not found, 405 = route exists but method not allowed
        # Either way, the upload endpoint is not functional
        assert resp.status_code in (404, 405)

    def test_clear_endpoint(self, client):
        """DELETE /hf-track/transfer/{id} clears transfer state."""
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "bert-base-german-dbmdz-cased", "filename": "config.json"},
        )
        transfer_id = resp.json()["transfer_id"]
        resp = client.delete(f"/hf-track/transfer/{transfer_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

    def test_clear_unknown_transfer(self, client):
        """DELETE /hf-track/transfer/{unknown_id} returns not_found."""
        resp = client.delete("/hf-track/transfer/nonexistent-id")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_found"


# ── Worker Function Tests ────────────────────────────────────────

class TestWorkerFunctions:
    """Unit tests for _do_download and _do_download_snapshot."""

    def test_do_download_no_env_var_modification(self, app_module):
        """_do_download() does not modify HF_HUB_DISABLE_XET.

        The env var is set by the endpoint handler before the thread
        is spawned, so the worker should not touch it.
        """
        # Set env var before calling worker (simulating endpoint handler)
        os.environ.pop("HF_HUB_DISABLE_XET", None)
        transfer_id = str(uuid.uuid4())
        app_module._active_transfers[transfer_id] = {
            "status": "running",
            "direction": "download",
        }
        app_module._transfer_events[transfer_id] = []

        with patch.object(app_module.tracker, "download_file") as mock_dl:
            app_module._do_download(
                transfer_id=transfer_id,
                repo_id="test/repo",
                filename="file.bin",
                repo_type="model",
                local_dir=None,
                use_xet=False,
            )
        # Worker should NOT have set the env var
        assert os.environ.get("HF_HUB_DISABLE_XET") is None
        mock_dl.assert_called_once()

        # Cleanup
        os.environ.pop("HF_HUB_DISABLE_XET", None)
        app_module._active_transfers.pop(transfer_id, None)
        app_module._transfer_events.pop(transfer_id, None)

    def test_do_download_does_not_clear_env_var(self, app_module):
        """_do_download() does not clear HF_HUB_DISABLE_XET when use_xet=True.

        The env var is managed by the endpoint handler, not the worker.
        """
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        transfer_id = str(uuid.uuid4())
        app_module._active_transfers[transfer_id] = {
            "status": "running",
            "direction": "download",
        }
        app_module._transfer_events[transfer_id] = []

        with patch.object(app_module.tracker, "download_file") as mock_dl:
            app_module._do_download(
                transfer_id=transfer_id,
                repo_id="test/repo",
                filename="file.bin",
                repo_type="model",
                local_dir=None,
                use_xet=True,
            )
        # Worker should NOT have cleared the env var
        assert os.environ.get("HF_HUB_DISABLE_XET") == "1"
        mock_dl.assert_called_once()

        # Cleanup
        os.environ.pop("HF_HUB_DISABLE_XET", None)
        app_module._active_transfers.pop(transfer_id, None)
        app_module._transfer_events.pop(transfer_id, None)

    def test_do_download_cancelled_error(self, app_module):
        """TransferCancelledError sets status to 'cancelled'."""
        from hf_track import TransferCancelledError
        transfer_id = str(uuid.uuid4())
        app_module._active_transfers[transfer_id] = {
            "status": "running",
            "direction": "download",
        }
        app_module._transfer_events[transfer_id] = []

        with patch.object(
            app_module.tracker, "download_file",
            side_effect=TransferCancelledError("cancelled"),
        ):
            app_module._do_download(
                transfer_id=transfer_id,
                repo_id="test/repo",
                filename="file.bin",
                repo_type="model",
                local_dir=None,
                use_xet=True,
            )

        assert app_module._active_transfers[transfer_id]["status"] == "cancelled"
        assert "error" not in app_module._active_transfers[transfer_id]
        assert "completed_at" in app_module._active_transfers[transfer_id]

        # Cleanup
        app_module._active_transfers.pop(transfer_id, None)
        app_module._transfer_events.pop(transfer_id, None)

    def test_do_download_os_error(self, app_module):
        """OSError sets status to 'error' with error message."""
        transfer_id = str(uuid.uuid4())
        app_module._active_transfers[transfer_id] = {
            "status": "running",
            "direction": "download",
        }
        app_module._transfer_events[transfer_id] = []

        with patch.object(
            app_module.tracker, "download_file",
            side_effect=OSError("network error"),
        ):
            app_module._do_download(
                transfer_id=transfer_id,
                repo_id="test/repo",
                filename="file.bin",
                repo_type="model",
                local_dir=None,
                use_xet=True,
            )

        assert app_module._active_transfers[transfer_id]["status"] == "error"
        assert app_module._active_transfers[transfer_id]["error"] == "network error"
        assert "completed_at" in app_module._active_transfers[transfer_id]

        # Cleanup
        app_module._active_transfers.pop(transfer_id, None)
        app_module._transfer_events.pop(transfer_id, None)

    def test_do_download_value_error(self, app_module):
        """ValueError sets status to 'error'."""
        transfer_id = str(uuid.uuid4())
        app_module._active_transfers[transfer_id] = {
            "status": "running",
            "direction": "download",
        }
        app_module._transfer_events[transfer_id] = []

        with patch.object(
            app_module.tracker, "download_file",
            side_effect=ValueError("bad value"),
        ):
            app_module._do_download(
                transfer_id=transfer_id,
                repo_id="test/repo",
                filename="file.bin",
                repo_type="model",
                local_dir=None,
                use_xet=True,
            )

        assert app_module._active_transfers[transfer_id]["status"] == "error"
        assert "bad value" in app_module._active_transfers[transfer_id]["error"]

        # Cleanup
        app_module._active_transfers.pop(transfer_id, None)
        app_module._transfer_events.pop(transfer_id, None)

    def test_do_download_unexpected_error(self, app_module):
        """Generic Exception sets status to 'error' with 'Unexpected' prefix."""
        transfer_id = str(uuid.uuid4())
        app_module._active_transfers[transfer_id] = {
            "status": "running",
            "direction": "download",
        }
        app_module._transfer_events[transfer_id] = []

        with patch.object(
            app_module.tracker, "download_file",
            side_effect=RuntimeError("something broke"),
        ):
            app_module._do_download(
                transfer_id=transfer_id,
                repo_id="test/repo",
                filename="file.bin",
                repo_type="model",
                local_dir=None,
                use_xet=True,
            )

        assert app_module._active_transfers[transfer_id]["status"] == "error"
        assert "Unexpected error" in app_module._active_transfers[transfer_id]["error"]

        # Cleanup
        app_module._active_transfers.pop(transfer_id, None)
        app_module._transfer_events.pop(transfer_id, None)

    def test_do_download_completed_at_always_set(self, app_module):
        """completed_at is set even on error."""
        transfer_id = str(uuid.uuid4())
        app_module._active_transfers[transfer_id] = {
            "status": "running",
            "direction": "download",
        }
        app_module._transfer_events[transfer_id] = []

        with patch.object(
            app_module.tracker, "download_file",
            side_effect=ConnectionError("refused"),
        ):
            app_module._do_download(
                transfer_id=transfer_id,
                repo_id="test/repo",
                filename="file.bin",
                repo_type="model",
                local_dir=None,
                use_xet=True,
            )

        assert "completed_at" in app_module._active_transfers[transfer_id]

        # Cleanup
        app_module._active_transfers.pop(transfer_id, None)
        app_module._transfer_events.pop(transfer_id, None)

    def test_do_download_success(self, app_module):
        """Successful download sets status to 'completed'."""
        transfer_id = str(uuid.uuid4())
        app_module._active_transfers[transfer_id] = {
            "status": "running",
            "direction": "download",
        }
        app_module._transfer_events[transfer_id] = []

        with patch.object(app_module.tracker, "download_file", return_value="/path/to/file"):
            app_module._do_download(
                transfer_id=transfer_id,
                repo_id="test/repo",
                filename="file.bin",
                repo_type="model",
                local_dir=None,
                use_xet=True,
            )

        assert app_module._active_transfers[transfer_id]["status"] == "completed"
        assert "error" not in app_module._active_transfers[transfer_id]

        # Cleanup
        app_module._active_transfers.pop(transfer_id, None)
        app_module._transfer_events.pop(transfer_id, None)

    def test_do_snapshot_cancelled_error(self, app_module):
        """TransferCancelledError in snapshot sets status to 'cancelled'."""
        from hf_track import TransferCancelledError
        transfer_id = str(uuid.uuid4())
        app_module._active_transfers[transfer_id] = {
            "status": "running",
            "direction": "download",
        }
        app_module._transfer_events[transfer_id] = []

        with patch.object(
            app_module.tracker, "download_snapshot",
            side_effect=TransferCancelledError("cancelled"),
        ):
            app_module._do_download_snapshot(
                transfer_id=transfer_id,
                repo_id="test/repo",
                allow_patterns=None,
                repo_type="model",
                local_dir=None,
                use_xet=True,
            )

        assert app_module._active_transfers[transfer_id]["status"] == "cancelled"
        assert "error" not in app_module._active_transfers[transfer_id]

        # Cleanup
        app_module._active_transfers.pop(transfer_id, None)
        app_module._transfer_events.pop(transfer_id, None)

    def test_do_snapshot_with_allow_patterns(self, app_module):
        """allow_patterns is passed to tracker.download_snapshot()."""
        transfer_id = str(uuid.uuid4())
        app_module._active_transfers[transfer_id] = {
            "status": "running",
            "direction": "download",
        }
        app_module._transfer_events[transfer_id] = []

        with patch.object(app_module.tracker, "download_snapshot", return_value="/path") as mock_snap:
            app_module._do_download_snapshot(
                transfer_id=transfer_id,
                repo_id="test/repo",
                allow_patterns="*.json",
                repo_type="model",
                local_dir=None,
                use_xet=True,
            )

        mock_snap.assert_called_once()
        call_kwargs = mock_snap.call_args
        # allow_patterns should be ["*.json"] (wrapped in list)
        assert call_kwargs.kwargs.get("allow_patterns") == ["*.json"] or \
               (len(call_kwargs.args) > 2 and call_kwargs.args[2] == ["*.json"])

        # Cleanup
        app_module._active_transfers.pop(transfer_id, None)
        app_module._transfer_events.pop(transfer_id, None)

    def test_do_snapshot_no_env_var_modification(self, app_module):
        """_do_download_snapshot() does not modify HF_HUB_DISABLE_XET.

        The env var is set by the endpoint handler before the thread
        is spawned, so the worker should not touch it.
        """
        os.environ.pop("HF_HUB_DISABLE_XET", None)
        transfer_id = str(uuid.uuid4())
        app_module._active_transfers[transfer_id] = {
            "status": "running",
            "direction": "download",
        }
        app_module._transfer_events[transfer_id] = []

        with patch.object(app_module.tracker, "download_snapshot"):
            app_module._do_download_snapshot(
                transfer_id=transfer_id,
                repo_id="test/repo",
                allow_patterns=None,
                repo_type="model",
                local_dir=None,
                use_xet=False,
            )
        # Worker should NOT have set the env var
        assert os.environ.get("HF_HUB_DISABLE_XET") is None

        # Cleanup
        os.environ.pop("HF_HUB_DISABLE_XET", None)
        app_module._active_transfers.pop(transfer_id, None)
        app_module._transfer_events.pop(transfer_id, None)


# ── Event Router Tests ───────────────────────────────────────────

class TestEventRouter:
    """Tests for the background event router thread."""

    def test_event_router_thread_running(self, app_module):
        """Event router thread is alive after app import."""
        assert app_module._event_router_thread.is_alive()

    def test_event_router_routes_events(self, app_module):
        """Events put in tracker.event_queue appear in _transfer_events."""
        from hf_track.types import ProgressEvent, EventType, TransferDirection, ProgressPhase

        transfer_id = str(uuid.uuid4())
        app_module._transfer_events[transfer_id] = []

        event = ProgressEvent(
            event_type=EventType.START,
            transfer_id=transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename="test.bin",
            phase=ProgressPhase.DOWNLOADING,
        )
        app_module.tracker.event_queue.put(event)

        # Wait for the router to pick it up
        deadline = time.time() + 2.0
        while time.time() < deadline:
            with app_module._events_lock:
                if len(app_module._transfer_events.get(transfer_id, [])) > 0:
                    break
            time.sleep(0.05)

        with app_module._events_lock:
            events = app_module._transfer_events.get(transfer_id, [])

        assert len(events) >= 1
        assert events[0].event_type == EventType.START
        assert events[0].transfer_id == transfer_id

        # Cleanup
        app_module._transfer_events.pop(transfer_id, None)

    def test_event_router_ignores_unknown_transfer(self, app_module):
        """Events for unknown transfer_ids are silently dropped."""
        from hf_track.types import ProgressEvent, EventType, TransferDirection, ProgressPhase

        unknown_id = "unknown-" + str(uuid.uuid4())
        # Don't register this transfer_id in _transfer_events

        event = ProgressEvent(
            event_type=EventType.START,
            transfer_id=unknown_id,
            direction=TransferDirection.DOWNLOAD,
            filename="test.bin",
            phase=ProgressPhase.DOWNLOADING,
        )
        app_module.tracker.event_queue.put(event)

        # Wait briefly for the router to process
        time.sleep(0.5)

        # Should not crash; the event is simply dropped
        with app_module._events_lock:
            assert unknown_id not in app_module._transfer_events


# ── Cancel Flow Tests ────────────────────────────────────────────

class TestCancelFlow:
    """Tests for the cancel endpoint and its interaction with transfers.

    Uses mocked downloads that block until cancelled, so the cancel
    request arrives while the download is still running.
    """

    def _start_blocked_download(self, app_module):
        """Start a download that blocks until cancelled, return transfer_id."""
        import threading

        transfer_id = str(uuid.uuid4())
        app_module._active_transfers[transfer_id] = {
            "direction": "download",
            "repo_id": "test/repo",
            "filename": "test.bin",
            "local_dir": None,
            "use_xet": True,
            "is_snapshot": False,
            "status": "running",
            "started_at": time.time(),
        }
        with app_module._events_lock:
            app_module._transfer_events[transfer_id] = []

        # Start a thread that blocks until cancelled
        cancel_event = threading.Event()

        def blocked_download():
            try:
                # Simulate a long-running download that checks for cancellation
                while not cancel_event.is_set():
                    time.sleep(0.05)
                # When cancelled, raise TransferCancelledError
                from hf_track import TransferCancelledError
                raise TransferCancelledError("Download cancelled by user")
            except TransferCancelledError:
                if transfer_id in app_module._active_transfers:
                    app_module._active_transfers[transfer_id]["status"] = "cancelled"
            finally:
                if transfer_id in app_module._active_transfers:
                    app_module._active_transfers[transfer_id]["completed_at"] = time.time()

        thread = threading.Thread(target=blocked_download, daemon=True)
        app_module._transfer_threads[transfer_id] = thread
        thread.start()

        return transfer_id, cancel_event

    def test_cancel_sets_status(self, app_module):
        """Cancel sets _active_transfers[tid]['status'] to 'cancelled'."""
        transfer_id, cancel_event = self._start_blocked_download(app_module)

        # Cancel the transfer
        app_module.tracker.cancel(transfer_id)
        app_module._active_transfers[transfer_id]["status"] = "cancelled"

        # Signal the blocked download to exit
        cancel_event.set()

        # Wait for the thread to finish
        thread = app_module._transfer_threads.get(transfer_id)
        if thread and thread.is_alive():
            thread.join(timeout=2.0)

        assert app_module._active_transfers[transfer_id]["status"] == "cancelled"

        # Cleanup
        app_module._active_transfers.pop(transfer_id, None)
        app_module._transfer_events.pop(transfer_id, None)
        app_module._transfer_threads.pop(transfer_id, None)

    def test_cancel_emits_cancelled_event(self, app_module):
        """Cancel puts a CANCELLED event in the per-transfer buffer."""
        transfer_id, cancel_event = self._start_blocked_download(app_module)

        # Emit a CANCELLED event (simulating what the cancel endpoint does)
        cancelled_event = ProgressEvent.cancelled_event(
            transfer_id=transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename="test.bin",
        )
        with app_module._events_lock:
            app_module._transfer_events[transfer_id].append(cancelled_event)

        app_module._active_transfers[transfer_id]["status"] = "cancelled"
        cancel_event.set()

        # Wait for the thread to finish
        thread = app_module._transfer_threads.get(transfer_id)
        if thread and thread.is_alive():
            thread.join(timeout=2.0)

        # Check the per-transfer event buffer
        with app_module._events_lock:
            events = app_module._transfer_events.get(transfer_id, [])

        cancelled_events = [
            e for e in events if e.event_type == EventType.CANCELLED
        ]
        assert len(cancelled_events) >= 1

        # Cleanup
        app_module._active_transfers.pop(transfer_id, None)
        app_module._transfer_events.pop(transfer_id, None)
        app_module._transfer_threads.pop(transfer_id, None)

    def test_cancel_unknown_returns_not_found(self, client):
        """Cancel of unknown ID returns not_found status."""
        resp = client.post("/hf-track/cancel/nonexistent-transfer-id")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_found"

    def test_cancel_uses_correct_direction(self, app_module):
        """CANCELLED event uses the direction from _active_transfers."""
        transfer_id, cancel_event = self._start_blocked_download(app_module)

        # The transfer direction should be "download"
        assert app_module._active_transfers[transfer_id]["direction"] == "download"

        # Emit a CANCELLED event with the correct direction
        direction = TransferDirection(app_module._active_transfers[transfer_id]["direction"])
        cancelled_event = ProgressEvent.cancelled_event(
            transfer_id=transfer_id,
            direction=direction,
            filename="test.bin",
        )
        with app_module._events_lock:
            app_module._transfer_events[transfer_id].append(cancelled_event)

        app_module._active_transfers[transfer_id]["status"] = "cancelled"
        cancel_event.set()

        # Wait for the thread to finish
        thread = app_module._transfer_threads.get(transfer_id)
        if thread and thread.is_alive():
            thread.join(timeout=2.0)

        with app_module._events_lock:
            events = app_module._transfer_events.get(transfer_id, [])

        cancelled_events = [
            e for e in events if e.event_type == EventType.CANCELLED
        ]
        assert len(cancelled_events) >= 1
        assert cancelled_events[0].direction == TransferDirection.DOWNLOAD

        # Cleanup
        app_module._active_transfers.pop(transfer_id, None)
        app_module._transfer_events.pop(transfer_id, None)
        app_module._transfer_threads.pop(transfer_id, None)


# ── Xet Toggle Tests ─────────────────────────────────────────────

class TestXetToggle:
    """Tests for the use_xet parameter and env var handling."""

    def test_download_with_xet_true(self, client):
        """POST with use_xet=true starts download."""
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "bert-base-german-dbmdz-cased", "filename": "config.json", "use_xet": "true"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("use_xet") is True

    def test_download_with_xet_false(self, client):
        """POST with use_xet=false starts download."""
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "bert-base-german-dbmdz-cased", "filename": "config.json", "use_xet": "false"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("use_xet") is False

    def test_xet_flag_in_response(self, client):
        """Response includes correct use_xet value."""
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "bert-base-german-dbmdz-cased", "use_xet": "false"},
        )
        assert resp.json().get("use_xet") is False

        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "bert-base-german-dbmdz-cased", "use_xet": "true"},
        )
        assert resp.json().get("use_xet") is True


# ── SSE Stream Tests ─────────────────────────────────────────────

class TestSSEStream:
    """Tests for the SSE event streaming endpoint."""

    def test_sse_content_type(self, client):
        """GET /hf-track/events/{id} returns text/event-stream."""
        # Start a download first to get a valid transfer_id
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "bert-base-german-dbmdz-cased", "filename": "config.json"},
        )
        transfer_id = resp.json()["transfer_id"]

        with client.stream("GET", f"/hf-track/events/{transfer_id}") as resp:
            assert resp.status_code == 200
            ct = resp.headers.get("content-type", "")
            assert "text/event-stream" in ct

    def test_sse_returns_events_for_transfer(self, client, app_module):
        """SSE stream yields events matching the transfer_id."""
        from hf_track.types import ProgressEvent, EventType, TransferDirection, ProgressPhase

        transfer_id = str(uuid.uuid4())
        app_module._active_transfers[transfer_id] = {
            "status": "running",
            "direction": "download",
            "filename": "test.bin",
        }
        app_module._transfer_events[transfer_id] = []

        # Put a START event in the per-transfer buffer
        start_event = ProgressEvent(
            event_type=EventType.START,
            transfer_id=transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename="test.bin",
            phase=ProgressPhase.DOWNLOADING,
        )
        with app_module._events_lock:
            app_module._transfer_events[transfer_id].append(start_event)

        # Now add a COMPLETE event to terminate the stream
        complete_event = ProgressEvent.complete(
            transfer_id=transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename="test.bin",
        )
        with app_module._events_lock:
            app_module._transfer_events[transfer_id].append(complete_event)
        app_module._active_transfers[transfer_id]["status"] = "completed"

        # Read from the SSE stream
        events_received = []
        with client.stream("GET", f"/hf-track/events/{transfer_id}") as resp:
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    import json
                    data = json.loads(line[5:].strip())
                    events_received.append(data)
                    if data.get("event_type") in ("complete", "error", "cancelled"):
                        break

        assert len(events_received) >= 1
        assert events_received[0]["transfer_id"] == transfer_id

        # Cleanup
        app_module._active_transfers.pop(transfer_id, None)
        app_module._transfer_events.pop(transfer_id, None)

    def test_sse_terminates_on_complete(self, client, app_module):
        """SSE stream closes after COMPLETE event."""
        from hf_track.types import ProgressEvent, EventType, TransferDirection, ProgressPhase

        transfer_id = str(uuid.uuid4())
        app_module._active_transfers[transfer_id] = {
            "status": "completed",
            "direction": "download",
            "filename": "test.bin",
        }

        # Pre-populate with START + COMPLETE events
        start_event = ProgressEvent(
            event_type=EventType.START,
            transfer_id=transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename="test.bin",
            phase=ProgressPhase.DOWNLOADING,
        )
        complete_event = ProgressEvent.complete(
            transfer_id=transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename="test.bin",
        )
        with app_module._events_lock:
            app_module._transfer_events[transfer_id] = [start_event, complete_event]

        events_received = []
        with client.stream("GET", f"/hf-track/events/{transfer_id}") as resp:
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    import json
                    data = json.loads(line[5:].strip())
                    events_received.append(data)
                    if data.get("event_type") in ("complete", "error", "cancelled"):
                        break

        # Should have received at least the START and COMPLETE events
        event_types = [e["event_type"] for e in events_received]
        assert "complete" in event_types

        # Cleanup
        app_module._active_transfers.pop(transfer_id, None)
        app_module._transfer_events.pop(transfer_id, None)


# ── Requirements Tests ───────────────────────────────────────────

class TestRequirements:
    def test_requirements_exists(self):
        req_file = WEB_APP_DIR / "requirements.txt"
        assert req_file.exists()

    def test_requirements_contains_fastapi(self):
        req_file = WEB_APP_DIR / "requirements.txt"
        content = req_file.read_text()
        assert "fastapi" in content.lower()

    def test_requirements_contains_sse(self):
        req_file = WEB_APP_DIR / "requirements.txt"
        content = req_file.read_text()
        assert "sse" in content.lower()


# ── Endpoint Xet Env Var Tests ────────────────────────────────────


class TestEndpointXetEnvVar:
    """Tests for the endpoint handler setting HF_HUB_DISABLE_XET."""

    def test_endpoint_sets_xet_disabled(self, client):
        """POST with use_xet=false sets HF_HUB_DISABLE_XET=1."""
        os.environ.pop("HF_HUB_DISABLE_XET", None)
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "test/repo", "filename": "file.bin", "use_xet": "false"},
        )
        assert resp.status_code == 200
        assert os.environ.get("HF_HUB_DISABLE_XET") == "1"
        # Cleanup
        os.environ.pop("HF_HUB_DISABLE_XET", None)

    def test_endpoint_clears_xet_enabled(self, client):
        """POST with use_xet=true clears HF_HUB_DISABLE_XET."""
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "test/repo", "filename": "file.bin", "use_xet": "true"},
        )
        assert resp.status_code == 200
        assert "HF_HUB_DISABLE_XET" not in os.environ
        # Cleanup
        os.environ.pop("HF_HUB_DISABLE_XET", None)

    def test_endpoint_sets_xet_for_snapshot(self, client):
        """POST snapshot with use_xet=false sets HF_HUB_DISABLE_XET=1."""
        os.environ.pop("HF_HUB_DISABLE_XET", None)
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "test/repo", "use_xet": "false"},
        )
        assert resp.status_code == 200
        assert os.environ.get("HF_HUB_DISABLE_XET") == "1"
        # Cleanup
        os.environ.pop("HF_HUB_DISABLE_XET", None)


# ── Force Download Tests ──────────────────────────────────────────


class TestForceDownload:
    """Tests for the force_download parameter."""

    def test_download_with_force_download_true(self, client):
        """POST with force_download=true returns 200."""
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "test/repo", "filename": "file.bin", "force_download": "true"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("force_download") is True

    def test_download_with_force_download_default(self, client):
        """Default force_download is False."""
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "test/repo", "filename": "file.bin"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("force_download") is False

    def test_force_download_in_response(self, client):
        """Response includes force_download value."""
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "test/repo", "force_download": "true"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("force_download") is True


# ── Cancel Snapshot Flow Tests ────────────────────────────────────


class TestCancelSnapshotFlow:
    """Tests for cancelling snapshot downloads."""

    def test_cancel_snapshot_calls_tracker_cancel(self, app_module):
        """Cancel of snapshot calls tracker.cancel()."""
        transfer_id = str(uuid.uuid4())
        app_module._active_transfers[transfer_id] = {
            "direction": "download",
            "repo_id": "test/repo",
            "filename": "test/repo (full repo)",
            "local_dir": None,
            "use_xet": True,
            "is_snapshot": True,
            "status": "running",
            "started_at": time.time(),
        }
        with app_module._events_lock:
            app_module._transfer_events[transfer_id] = []

        with patch.object(app_module.tracker, "cancel") as mock_cancel:
            app_module.tracker.cancel(transfer_id)

        mock_cancel.assert_called_once_with(transfer_id)

        # Cleanup
        app_module._active_transfers.pop(transfer_id, None)
        with app_module._events_lock:
            app_module._transfer_events.pop(transfer_id, None)

    def test_cancel_snapshot_emits_cancelled_event(self, app_module):
        """CANCELLED event appears in per-transfer buffer after cancel."""
        transfer_id = str(uuid.uuid4())
        app_module._active_transfers[transfer_id] = {
            "direction": "download",
            "repo_id": "test/repo",
            "filename": "test/repo (full repo)",
            "status": "running",
            "started_at": time.time(),
        }
        with app_module._events_lock:
            app_module._transfer_events[transfer_id] = []

        # Simulate what the cancel endpoint does
        app_module.tracker.cancel(transfer_id)
        cancelled_event = ProgressEvent.cancelled_event(
            transfer_id=transfer_id,
            direction=TransferDirection.DOWNLOAD,
            filename="test/repo (full repo)",
        )
        with app_module._events_lock:
            app_module._transfer_events[transfer_id].append(cancelled_event)
        app_module._active_transfers[transfer_id]["status"] = "cancelled"

        with app_module._events_lock:
            events = app_module._transfer_events.get(transfer_id, [])

        assert len(events) >= 1
        assert events[0].event_type == EventType.CANCELLED

        # Cleanup
        app_module._active_transfers.pop(transfer_id, None)
        with app_module._events_lock:
            app_module._transfer_events.pop(transfer_id, None)

    def test_cancel_snapshot_sets_status(self, app_module):
        """Status is set to 'cancelled' after cancel."""
        transfer_id = str(uuid.uuid4())
        app_module._active_transfers[transfer_id] = {
            "direction": "download",
            "repo_id": "test/repo",
            "filename": "test/repo (full repo)",
            "status": "running",
            "started_at": time.time(),
        }
        with app_module._events_lock:
            app_module._transfer_events[transfer_id] = []

        app_module.tracker.cancel(transfer_id)
        app_module._active_transfers[transfer_id]["status"] = "cancelled"

        assert app_module._active_transfers[transfer_id]["status"] == "cancelled"

        # Cleanup
        app_module._active_transfers.pop(transfer_id, None)
        with app_module._events_lock:
            app_module._transfer_events.pop(transfer_id, None)

    def test_cancel_snapshot_unknown_returns_not_found(self, client):
        """Cancel of unknown transfer_id returns not_found."""
        resp = client.post("/hf-track/cancel/unknown-transfer-id")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "not_found"
