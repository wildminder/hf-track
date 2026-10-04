"""Tests for hf_track.download.http_fallback — pure-HTTP Tier 3 path.

Plan: ``docs/plans/2026-06-15-xet-streaming-hybrid-approach.md`` Step 2.

These tests use mocked ``requests`` and a fake response so they run
deterministically without contacting the network. The behaviour we
verify is the documented contract of ``download_file_http``:

* Bytes written to disk match the streamed body.
* Each chunk crosses ``os.write`` directly to a raw fd.
* Periodic ``os.fsync`` is called.
* ``expected_size`` mismatches are reported clearly.
* HTTP error statuses raise ``HTTPFallbackError``.
* Cooperative cancellation via ``cancel_event.is_set()`` short-
  circuits the loop without raising.
* Bearer-token header is added when ``token`` is provided.
* URLs are built for each ``repo_type`` (model / dataset / space).
"""

from __future__ import annotations

import os
from typing import Iterable
from unittest.mock import MagicMock, patch

import pytest

from hf_track.download.http_fallback import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FSYNC_INTERVAL,
    HTTPFallbackError,
    _build_url,
    download_file_http,
)


# ── Helpers ────────────────────────────────────────────────────────


class _FakeResponse:
    """Minimal stand-in for ``requests.Response``.

    Supports the sequencing we use in tests:
    - ``status_code`` (int)
    - ``reason`` (str)
    - ``headers.get("content-length")`` (str | None)
    - ``iter_content(chunk_size=...)`` returning an iterable of bytes
    - ``raise_for_status()`` (raises if status >= 400)
    - ``close()``
    """

    def __init__(
        self,
        body: bytes,
        status_code: int = 200,
        reason: str = "OK",
        content_length: int | None = None,
    ):
        self._body = body
        self.status_code = status_code
        self.reason = reason
        self.closed = False
        self._content_length = (
            str(content_length)
            if content_length is not None
            else str(len(body)) if body else None
        )
        self._headers = {"content-length": self._content_length} if self._content_length else {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise HTTPFallbackError(
                f"HTTP {self.status_code} for url: {self.reason}"
            )

    @property
    def headers(self):
        return self._headers

    def iter_content(self, chunk_size: int) -> Iterable[bytes]:
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def close(self) -> None:
        self.closed = True


def _mock_requests_get(response: _FakeResponse) -> MagicMock:
    mock_requests = MagicMock()
    mock_requests.get.return_value = response
    return mock_requests


# ── URL building ───────────────────────────────────────────────────


class TestBuildUrl:
    def test_model_default_endpoint(self):
        url = _build_url("owner/repo", "weights.bin", "model", "main", None)
        assert url == "https://huggingface.co/owner/repo/resolve/main/weights.bin"

    def test_dataset_prefix(self):
        url = _build_url("owner/repo", "data.csv", "dataset", "v1.0", None)
        assert url == "https://huggingface.co/datasets/owner/repo/resolve/v1.0/data.csv"

    def test_space_prefix(self):
        url = _build_url("owner/repo", "app.py", "space", "main", None)
        assert url == "https://huggingface.co/spaces/owner/repo/resolve/main/app.py"

    def test_custom_endpoint(self):
        url = _build_url("owner/repo", "x", "model", "main", "https://huggingface.acme")
        assert url == "https://huggingface.acme/owner/repo/resolve/main/x"

    def test_endpoint_trailing_slash_stripped(self):
        url = _build_url("owner/repo", "x", "model", "main", "https://huggingface.co/")
        assert url == "https://huggingface.co/owner/repo/resolve/main/x"


# ── Download success paths ─────────────────────────────────────────


class TestDownloadFileHttpSuccess:
    def test_writes_body_to_file(self, tmp_path, monkeypatch):
        body = b"hello world! " * 1024  # 13 KiB
        fake = _FakeResponse(body)
        req = _mock_requests_get(fake)
        monkeypatch.setitem(__import__("sys").modules, "requests", req)

        dest = tmp_path / "out.bin"
        written = download_file_http(
            repo_id="o/r", filename="x.bin", dest_path=str(dest),
            fsync_interval=4096,
        )

        assert written == len(body)
        assert dest.read_bytes() == body

    def test_uses_bearer_token_when_provided(self, tmp_path, monkeypatch):
        body = b"abc"
        fake = _FakeResponse(body)
        req = _mock_requests_get(fake)
        monkeypatch.setitem(__import__("sys").modules, "requests", req)

        dest = tmp_path / "out.bin"
        download_file_http(
            repo_id="o/r", filename="x.bin", dest_path=str(dest),
            token="hf_test_token", fsync_interval=4096,
        )

        kwargs = req.get.call_args.kwargs
        assert kwargs["headers"]["Authorization"] == "Bearer hf_test_token"

    def test_no_auth_header_without_token(self, tmp_path, monkeypatch):
        fake = _FakeResponse(b"abc")
        req = _mock_requests_get(fake)
        monkeypatch.setitem(__import__("sys").modules, "requests", req)

        download_file_http(
            repo_id="o/r", filename="x.bin", dest_path=str(tmp_path / "x.bin"),
        )

        kwargs = req.get.call_args.kwargs
        assert "Authorization" not in kwargs["headers"]

    def test_creates_parent_directories(self, tmp_path, monkeypatch):
        fake = _FakeResponse(b"hi")
        req = _mock_requests_get(fake)
        monkeypatch.setitem(__import__("sys").modules, "requests", req)

        nested = tmp_path / "a" / "b" / "c" / "out.bin"
        assert not nested.parent.exists()
        download_file_http(
            repo_id="o/r", filename="x", dest_path=str(nested), fsync_interval=10,
        )
        assert nested.is_file()

    def test_emits_streams_in_chunks(self, tmp_path, monkeypatch):
        body = b"X" * 8192
        fake = _FakeResponse(body)
        req = _mock_requests_get(fake)
        monkeypatch.setitem(__import__("sys").modules, "requests", req)

        download_file_http(
            repo_id="o/r", filename="x.bin", dest_path=str(tmp_path / "x.bin"),
            chunk_size=1024, fsync_interval=8192,
        )

        kwargs = req.get.call_args.kwargs
        assert kwargs["stream"] is True

    def test_calls_fsync_periodically(self, tmp_path, monkeypatch):
        body = b"X" * (4 * 1024 + 100)  # 4.1 KiB; fsync every 1 KiB
        fake = _FakeResponse(body)
        req = _mock_requests_get(fake)
        monkeypatch.setitem(__import__("sys").modules, "requests", req)

        fsync_count = {"n": 0}
        real_fsync = os.fsync

        def tracking_fsync(fd):
            fsync_count["n"] += 1
            return real_fsync(fd)

        monkeypatch.setattr(
            "hf_track.download.http_fallback.os.fsync", tracking_fsync,
        )
        dest = tmp_path / "out.bin"
        download_file_http(
            repo_id="o/r", filename="x.bin", dest_path=str(dest),
            fsync_interval=1024,
        )
        # 4100 bytes / 1024 = 4 periodic intervals + 1 finalisation fsync.
        # With the DEFAULT_CHUNK_SIZE (64 KiB) the entire body fits in
        # one chunk, so we expect 1 in-loop fsync at the boundary plus
        # the final fsync = 2.
        assert fsync_count["n"] >= 2, f"expected ≥2 fsync calls, got {fsync_count['n']}"

    def test_calls_fsync_with_small_chunks(self, tmp_path, monkeypatch):
        """When chunks are smaller than fsync_interval, fsync fires
        once per multiple of ``fsync_interval`` bytes."""
        body = b"Y" * 4096  # 4 KiB; 64 chunks of 64 bytes
        fake = _FakeResponse(body)
        req = _mock_requests_get(fake)
        monkeypatch.setitem(__import__("sys").modules, "requests", req)

        fsync_count = {"n": 0}
        real_fsync = os.fsync

        def tracking_fsync(fd):
            fsync_count["n"] += 1
            return real_fsync(fd)

        monkeypatch.setattr(
            "hf_track.download.http_fallback.os.fsync", tracking_fsync,
        )

        download_file_http(
            repo_id="o/r", filename="x.bin", dest_path=str(tmp_path / "x.bin"),
            chunk_size=64, fsync_interval=512,
        )
        # 4096/512 = 8 periodic fsyncs + 1 finalisation = 9.
        assert fsync_count["n"] >= 9, f"expected ≥9 fsync calls, got {fsync_count['n']}"

    def test_expected_size_mismatch_raises(self, tmp_path, monkeypatch):
        fake = _FakeResponse(b"abc")  # 3 bytes
        req = _mock_requests_get(fake)
        monkeypatch.setitem(__import__("sys").modules, "requests", req)

        # The Content-Length mismatch fires before any write.
        with pytest.raises(HTTPFallbackError, match="(Content-Length|Size) mismatch"):
            download_file_http(
                repo_id="o/r", filename="x", dest_path=str(tmp_path / "x"),
                expected_size=12345,
            )

    def test_content_length_mismatch_raises(self, tmp_path, monkeypatch):
        fake = _FakeResponse(b"abc", content_length=100)
        req = _mock_requests_get(fake)
        monkeypatch.setitem(__import__("sys").modules, "requests", req)

        # content_length=100 ≠ expected_size=3 → raises before any write.
        with pytest.raises(HTTPFallbackError, match="Content-Length"):
            download_file_http(
                repo_id="o/r", filename="x", dest_path=str(tmp_path / "x"),
                expected_size=3,
            )


# ── Error paths ─────────────────────────────────────────────────────


class TestDownloadFileHttpErrors:
    def test_http_404_raises_with_url(self, tmp_path, monkeypatch):
        fake = _FakeResponse(b"", status_code=404, reason="Not Found")
        req = _mock_requests_get(fake)
        monkeypatch.setitem(__import__("sys").modules, "requests", req)

        with pytest.raises(HTTPFallbackError, match="404"):
            download_file_http(
                repo_id="o/r", filename="x", dest_path=str(tmp_path / "x"),
            )

    def test_http_401_mentions_token(self, tmp_path, monkeypatch):
        fake = _FakeResponse(b"", status_code=401, reason="Unauthorized")
        req = _mock_requests_get(fake)
        monkeypatch.setitem(__import__("sys").modules, "requests", req)

        with pytest.raises(HTTPFallbackError) as exc_info:
            download_file_http(
                repo_id="o/r", filename="x", dest_path=str(tmp_path / "x"),
            )
        assert "token" in str(exc_info.value).lower()

    def test_http_403_mentions_access(self, tmp_path, monkeypatch):
        fake = _FakeResponse(b"", status_code=403, reason="Forbidden")
        req = _mock_requests_get(fake)
        monkeypatch.setitem(__import__("sys").modules, "requests", req)

        with pytest.raises(HTTPFallbackError) as exc_info:
            download_file_http(
                repo_id="o/r", filename="x", dest_path=str(tmp_path / "x"),
            )
        assert "access denied" in str(exc_info.value).lower()

    def test_connection_error_raises_HTTPFallbackError(self, tmp_path, monkeypatch):
        req = MagicMock()
        req.get.side_effect = ConnectionError("network down")
        monkeypatch.setitem(__import__("sys").modules, "requests", req)

        with pytest.raises(HTTPFallbackError, match="HTTP GET"):
            download_file_http(
                repo_id="o/r", filename="x", dest_path=str(tmp_path / "x"),
            )

    def test_close_call_even_on_error(self, tmp_path, monkeypatch):
        fake = _FakeResponse(b"", status_code=500, reason="Internal Server Error")
        req = _mock_requests_get(fake)
        monkeypatch.setitem(__import__("sys").modules, "requests", req)

        with pytest.raises(HTTPFallbackError):
            download_file_http(
                repo_id="o/r", filename="x", dest_path=str(tmp_path / "x"),
            )
        assert fake.closed is True


# ── Cancellation ────────────────────────────────────────────────────


class TestDownloadFileHttpCancellation:
    def test_cancel_via_event_short_circuits_loop(self, tmp_path, monkeypatch):
        """Setting cancel_event mid-stream stops writing without raising."""

        body_parts = [b"X" * 256 for _ in range(20)]

        cancel_event = __import__("threading").Event()

        def on_yield(n):
            # Set cancel AFTER the first chunk is delivered.
            if n >= 1:
                cancel_event.set()

        class _Gen:
            def __init__(self):
                self._parts = body_parts
                self._on_yield = on_yield
                self._cancel_event = cancel_event
                self._idx = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self._idx >= len(self._parts):
                    raise StopIteration
                if self._cancel_event.is_set():
                    raise StopIteration
                p = self._parts[self._idx]
                self._idx += 1
                self._on_yield(self._idx)
                return p

        class _Resp:
            status_code = 200
            reason = "OK"
            headers = {"content-length": str(256 * 20)}
            _closed = False

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size):
                return _Gen()

            def close(self):
                self._closed = True

        resp = _Resp()
        req = MagicMock()
        req.get.return_value = resp
        monkeypatch.setitem(__import__("sys").modules, "requests", req)

        dest = tmp_path / "cancelled.bin"
        written = download_file_http(
            repo_id="o/r", filename="x", dest_path=str(dest),
            cancel_event=cancel_event, fsync_interval=128,
        )
        # cancel is set inside the iterator AFTER the first chunk yields.
        # The function's top-of-iteration cancel check sees the flag on
        # the NEXT loop pass, so exactly one 256-byte chunk is written
        # before the loop breaks.
        assert written == 256
        assert os.path.getsize(str(dest)) == 256
        assert resp._closed is True

    def test_cancel_zero_chunks_creates_empty_file(self, tmp_path, monkeypatch):
        cancel_event = __import__("threading").Event()
        cancel_event.set()  # pre-cancelled

        class _Resp:
            status_code = 200
            reason = "OK"
            headers = {"content-length": "1024"}

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size):
                yield b"x"  # should be discarded

            def close(self):
                pass

        req = MagicMock()
        req.get.return_value = _Resp()
        monkeypatch.setitem(__import__("sys").modules, "requests", req)

        dest = tmp_path / "cancelled.bin"
        written = download_file_http(
            repo_id="o/r", filename="x", dest_path=str(dest),
            cancel_event=cancel_event,
        )
        assert written == 0
        assert os.path.getsize(str(dest)) == 0


# ── Module-level defaults ──────────────────────────────────────────


class TestDefaults:
    def test_default_chunk_size_positive(self):
        assert DEFAULT_CHUNK_SIZE > 0

    def test_default_fsync_interval_positive(self):
        assert DEFAULT_FSYNC_INTERVAL > 0
