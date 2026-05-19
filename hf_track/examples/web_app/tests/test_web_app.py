"""Integration tests for the web app example.

Uses FastAPI TestClient to test the full stack without a running server.
"""
from __future__ import annotations

import os
import sys
import pytest
from pathlib import Path
from html.parser import HTMLParser

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
    fastapi = pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from app import app
    return TestClient(app)


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
        assert "text/css" in resp.headers.get("content-type", "") or "text/plain" in resp.headers.get("content-type", "")

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


# ── API Endpoint Tests ───────────────────────────────────────────

class TestAPIEndpoints:
    def test_download_endpoint(self, client):
        """POST /hf-track/download returns transfer_id."""
        resp = client.post(
            "/hf-track/download",
            params={"repo_id": "bert-base-uncased", "filename": "config.json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "transfer_id" in data

    def test_cancel_endpoint(self, client):
        """POST /hf-track/cancel/{id} returns success."""
        resp = client.post("/hf-track/cancel/test-id")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

    def test_status_endpoint(self, client):
        """GET /hf-track/status returns active_transfers."""
        resp = client.get("/hf-track/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_transfers" in data

    def test_download_missing_params(self, client):
        """POST without required params returns 422."""
        resp = client.post("/hf-track/download")
        assert resp.status_code == 422


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
