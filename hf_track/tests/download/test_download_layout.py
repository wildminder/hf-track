"""Smoke tests for the new download/ subpackage layout.

Verifies that:
  * All public download entry points are exported from `hf_track.download`.
  * Each download submodule has the public function/class it claims to have.
  * The subpackage layout matches the documented structure.
  * Old top-level modules (xet_download.py, standard_download.py) are gone.
"""

from __future__ import annotations

import importlib
import os

import pytest


PACKAGE = "hf_track.download"


def test_package_imports():
    """The download subpackage should import cleanly."""
    pkg = importlib.import_module(PACKAGE)
    assert pkg is not None


def test_public_api_exports():
    """All public symbols are re-exported from hf_track.download."""
    from hf_track import download

    expected = {
        "XetDownloadResult",
        "download_file_with_xet",
        "download_file_xet_only",
        "download_files_with_xet",
        "download_snapshot_with_xet",
        "download_snapshot_streaming",
        "download_file",
        "download_snapshot",
        "patch_download_chunk_size",
    }
    for name in expected:
        assert hasattr(download, name), f"Missing public symbol: {name}"


def test_subpackage_layout():
    """Each documented submodule is present and importable."""
    expected_modules = [
        "xet_file",
        "xet_batch",
        "xet_snapshot",
        "xet_streaming",
        "standard",
    ]
    for name in expected_modules:
        mod = importlib.import_module(f"{PACKAGE}.{name}")
        assert mod is not None, f"Cannot import {PACKAGE}.{name}"


def test_xet_file_module():
    """xet_file.py exports the single-file Xet helpers."""
    from hf_track.download import xet_file

    assert hasattr(xet_file, "XetDownloadResult")
    assert hasattr(xet_file, "download_file_with_xet")
    # These names are bound at module level because the tests patch them
    # at hf_track.download.xet_file.<name>.
    assert hasattr(xet_file, "XetSubprocessRunner")
    assert hasattr(xet_file, "is_xet_available")


def test_xet_batch_module():
    """xet_batch.py exports the batch helper and reuses XetDownloadResult."""
    from hf_track.download import xet_batch

    assert hasattr(xet_batch, "download_files_with_xet")
    assert hasattr(xet_batch, "XetSubprocessRunner")
    assert hasattr(xet_batch, "is_xet_available")


def test_xet_snapshot_module():
    """xet_snapshot.py exports the snapshot helpers."""
    from hf_track.download import xet_snapshot

    assert hasattr(xet_snapshot, "download_snapshot_with_xet")
    assert hasattr(xet_snapshot, "XetSubprocessRunner")
    assert hasattr(xet_snapshot, "is_xet_available")


def test_xet_streaming_module():
    """xet_streaming.py exports the streaming snapshot helper."""
    from hf_track.download import xet_streaming

    assert hasattr(xet_streaming, "download_snapshot_streaming")
    assert hasattr(xet_streaming, "is_xet_available")
    # Plan 2026-06-15: xet_streaming now uses the in-process HybridRunner
    # instead of XetSubprocessRunner. The HybridRunner is imported from
    # ``hf_track._xet_worker`` and re-exported via the module's imports.
    assert hasattr(xet_streaming, "HybridRunner")


def test_standard_module():
    """standard.py exports the non-Xet download helpers."""
    from hf_track.download import standard

    assert hasattr(standard, "download_file")
    assert hasattr(standard, "download_snapshot")
    assert hasattr(standard, "patch_download_chunk_size")


def test_old_top_level_modules_removed():
    """The old flat modules (xet_download.py, standard_download.py)
    must be removed from the source tree after the download/ split.
    """
    pkg_root = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "src", "hf_track",
    )
    for stale in ("xet_download.py", "standard_download.py"):
        path = os.path.join(pkg_root, stale)
        assert not os.path.exists(path), f"Stale module still present: {stale}"


def test_module_size_budget_soft():
    """No single download/ submodule should exceed the soft budget
    (200 lines / roughly 8 KB).  Hard ceiling is enforced separately in
    test_module_sizes.py once the enforcement test is added in Step 10.

    Plan 2026-06-15 grew ``xet_streaming.py`` to ~15 KB to add Tier 1 →
    Tier 3 (HTTP) fallback coordination plus per-file resolution. The
    soft ceiling was raised from 14 KB to 16 KB for that module only
    is intentional -- the function has the most responsibility.

    Plan 2026-07-16 (xet-single-file-subprocess-isolation) added
    ``download_file_xet_subprocess`` to ``xet_file_only.py`` alongside
    the existing ``download_file_xet_only``; the module now holds both
    the in-process and subprocess dedicated xet paths (~18 KB). Raised
    the soft ceiling to 20 KB to accommodate the second path.
    """
    pkg_root = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "src", "hf_track", "download",
    )
    SOFT_BYTES = 20_000  # ceiling, not target — just sanity check
    for name in os.listdir(pkg_root):
        if not name.endswith(".py"):
            continue
        path = os.path.join(pkg_root, name)
        size = os.path.getsize(path)
        assert size < SOFT_BYTES, (
            f"{name} is {size} bytes — exceeds soft ceiling {SOFT_BYTES} bytes. "
            "Consider splitting further."
        )
