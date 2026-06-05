"""Smoke tests for the new upload/ subpackage layout.

Verifies that:
  * All public upload entry points are exported from `hf_track.upload`.
  * Each upload submodule has the public function/class it claims to have.
  * The subpackage layout matches the documented structure.
  * Old top-level modules (xet_upload.py, standard_upload.py) are gone.
"""

from __future__ import annotations

import importlib
import os

PACKAGE = "hf_track.upload"


def test_package_imports():
    """The upload subpackage should import cleanly."""
    pkg = importlib.import_module(PACKAGE)
    assert pkg is not None


def test_public_api_exports():
    """All public symbols are re-exported from hf_track.upload."""
    from hf_track import upload

    expected = {
        "XetUploadResult",
        "upload_file_with_xet",
        "upload_bytes_with_xet",
        "upload_bytes_via_temp_file",
        "upload_file",
        "upload_bytes",
        "upload_folder",
    }
    for name in expected:
        assert hasattr(upload, name), f"Missing public symbol: {name}"


def test_subpackage_layout():
    """Each documented submodule is present and importable."""
    expected_modules = ["xet_file", "xet_bytes", "standard"]
    for name in expected_modules:
        mod = importlib.import_module(f"{PACKAGE}.{name}")
        assert mod is not None, f"Cannot import {PACKAGE}.{name}"


def test_xet_file_module():
    """xet_file.py exports the single-file Xet upload helper."""
    from hf_track.upload import xet_file

    assert hasattr(xet_file, "XetUploadResult")
    assert hasattr(xet_file, "upload_file_with_xet")
    # These names are bound at module level because the tests patch them
    # at hf_track.upload.xet_file.<name>.
    assert hasattr(xet_file, "XetSubprocessRunner")
    assert hasattr(xet_file, "is_xet_available")


def test_xet_bytes_module():
    """xet_bytes.py exports the bytes-via-Xet helpers."""
    from hf_track.upload import xet_bytes

    assert hasattr(xet_bytes, "upload_bytes_with_xet")
    assert hasattr(xet_bytes, "upload_bytes_via_temp_file")
    assert hasattr(xet_bytes, "is_xet_available")
    # Module-private constant used by tests to construct large payloads.
    assert hasattr(xet_bytes, "_LARGE_PAYLOAD_THRESHOLD")


def test_standard_module():
    """standard.py exports the non-Xet upload helpers."""
    from hf_track.upload import standard

    assert hasattr(standard, "upload_file")
    assert hasattr(standard, "upload_bytes")
    assert hasattr(standard, "upload_folder")


def test_old_top_level_modules_removed():
    """The old flat modules (xet_upload.py, standard_upload.py)
    must be removed from the source tree after the upload/ split.
    """
    pkg_root = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "src", "hf_track",
    )
    for stale in ("xet_upload.py", "standard_upload.py"):
        path = os.path.join(pkg_root, stale)
        assert not os.path.exists(path), f"Stale module still present: {stale}"


def test_module_size_budget_soft():
    """No single upload/ submodule should exceed the soft budget
    (14 KB ceiling, 200-line target).
    """
    pkg_root = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "src", "hf_track", "upload",
    )
    SOFT_BYTES = 14_000
    for name in os.listdir(pkg_root):
        if not name.endswith(".py"):
            continue
        path = os.path.join(pkg_root, name)
        size = os.path.getsize(path)
        assert size < SOFT_BYTES, (
            f"{name} is {size} bytes — exceeds soft ceiling {SOFT_BYTES} bytes. "
            "Consider splitting further."
        )
