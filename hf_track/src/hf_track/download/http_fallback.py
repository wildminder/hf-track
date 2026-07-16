"""Pure HTTP fallback for Xet downloads.

Plan: ``docs/plans/2026-06-15-xet-streaming-hybrid-approach.md``,
Section 2.4 (Tier 3) and Step 2.

The Xet streaming/download APIs are unreliable in some environments
(stuck at the Rust layer, GIL held during ``__next__``, ``status()``
never transitions to ``Completed``). As a last-resort fallback we
bypass ``hf_xet`` entirely and download the file with regular HTTP
from the HuggingFace CDN, using ``requests`` for streaming
``iter_content`` + raw ``os.write``/``os.fsync`` for incremental disk
visibility.

Trade-offs vs. Xet:

* No block-level deduplication across snapshots -- each file is a
  fresh HTTPS GET. Slower for large multi-file snapshots where the
  same chunk is re-downloaded by several revisions.
* Each file is downloaded as a single HTTP request via
  ``https://huggingface.co/{repo_id}/resolve/{revision}/{filename}``.
  Subject to HF Hub's standard rate limits; gated repos require a
  token with read access.
* ``requests`` is required. It is already an indirect dependency of
  ``huggingface_hub``; if not present, ``download_file_http`` raises
  ``ImportError`` with install instructions.
"""

from __future__ import annotations

import os
from typing import Optional


DEFAULT_CHUNK_SIZE: int = 64 * 1024
DEFAULT_FSYNC_INTERVAL: int = 4 * 1024 * 1024


class HTTPFallbackError(RuntimeError):
    """Raised when the HTTP fallback fails (network / size / auth)."""


def _build_url(repo_id: str, filename: str, repo_type: str, revision: str, endpoint: Optional[str]) -> str:
    base = (endpoint or "https://huggingface.co").rstrip("/")
    if repo_type == "dataset":
        prefix = f"{base}/datasets/{repo_id}"
    elif repo_type == "space":
        prefix = f"{base}/spaces/{repo_id}"
    else:
        prefix = f"{base}/{repo_id}"
    return f"{prefix}/resolve/{revision}/{filename}"


def download_file_http(
    repo_id: str,
    filename: str,
    dest_path: str,
    token: Optional[str] = None,
    repo_type: str = "model",
    revision: str = "main",
    endpoint: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    fsync_interval: int = DEFAULT_FSYNC_INTERVAL,
    cancel_event=None,
    expected_size: Optional[int] = None,
) -> int:
    """Download a single file via HTTP with incremental disk visibility.

    Streams the HTTPS body through ``requests`` and writes each chunk
    to ``dest_path`` via ``os.write`` on a raw file descriptor (no
    userspace buffer). ``os.fsync`` is called every ``fsync_interval``
    bytes and once before the close so a SIGKILL at any point leaves a
    non-empty, non-truncated file on disk.

    Args:
        repo_id: HuggingFace repository ID (``"owner/name"``).
        filename: Path within the repo, e.g. ``"weights/model.bin"``.
        dest_path: Local path to write the file to. Parent directories
            are created as needed.
        token: Optional HF access token for gated repos.
        repo_type: ``"model"``, ``"dataset"`` or ``"space"``.
        revision: Git revision (branch / tag / commit-ish).
        endpoint: Optional HF endpoint override (defaults to
            ``https://huggingface.co``).
        chunk_size: Bytes per read chunk from the HTTP response.
        fsync_interval: Bytes between ``os.fsync`` calls.
        cancel_event: Optional object with ``.is_set()`` / ``.set()``.
            When ``.is_set()`` is True, the loop breaks early.
        expected_size: Optional expected byte total (sanity check vs.
            ``Content-Length``). When ``None``, the Content-Length is
            accepted as authoritative.

    Returns:
        Bytes actually written to ``dest_path``.

    Raises:
        ImportError: If ``requests`` is not installed (``pip install requests``).
        HTTPFallbackError: On any network error, HTTP error, or size
            mismatch.
    """
    try:
        import requests  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "The HTTP fallback requires the 'requests' package. "
            "Install with: pip install requests"
        ) from e

    url = _build_url(repo_id, filename, repo_type, revision, endpoint)
    headers: dict = {"Accept": "application/octet-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    parent = os.path.dirname(dest_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fd = os.open(dest_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)

    bytes_written = 0
    bytes_since_fsync = 0
    cancelled = False
    err: Optional[BaseException] = None
    try:
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=300)
        except Exception as e:
            raise HTTPFallbackError(f"HTTP GET {url} failed: {e}") from e

        try:
            try:
                response.raise_for_status()
            except Exception as e:
                status = getattr(response, "status_code", "?")
                reason = getattr(response, "reason", "?")
                msg = f"HTTP {status} for {url}: {reason}"
                if str(status) == "401":
                    msg += " (token missing or invalid for gated repo)"
                elif str(status) == "403":
                    msg += " (access denied)"
                raise HTTPFallbackError(msg) from e

            content_length_header = response.headers.get("content-length")
            try:
                content_length = int(content_length_header) if content_length_header else 0
            except (TypeError, ValueError):
                content_length = 0

            if expected_size is not None and content_length and expected_size != content_length:
                raise HTTPFallbackError(
                    f"Content-Length mismatch for {filename}: "
                    f"got {content_length}, expected {expected_size}"
                )

            chunk_iter = response.iter_content(chunk_size=chunk_size)
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    break
                if not chunk:
                    continue
                try:
                    os.write(fd, chunk)
                except OSError as e:
                    raise HTTPFallbackError(f"os.write failed for {filename}: {e}") from e
                bytes_written += len(chunk)
                bytes_since_fsync += len(chunk)
                if bytes_since_fsync >= fsync_interval:
                    try:
                        os.fsync(fd)
                    except OSError:
                        pass
                    bytes_since_fsync = 0
        finally:
            try:
                response.close()
            except Exception:
                pass
    except BaseException as e:
        err = e
    finally:
        if not cancelled and err is None:
            try:
                os.fsync(fd)
            except OSError:
                pass
        try:
            os.close(fd)
        except OSError:
            pass

    if err is not None:
        raise err

    if cancelled:
        return bytes_written

    if expected_size is not None and bytes_written != expected_size:
        raise HTTPFallbackError(
            f"Size mismatch for {filename}: got {bytes_written} bytes, "
            f"expected {expected_size}"
        )

    return bytes_written
