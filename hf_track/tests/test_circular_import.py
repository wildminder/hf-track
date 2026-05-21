"""Regression tests for huggingface_hub circular import issue.

Verifies that importing hf_track (which triggers partial
huggingface_hub imports via XetTokenManager) does not break
subsequent patch() calls on huggingface_hub lazy attributes.

Background: huggingface_hub v1.8.0 uses __getattr__-based lazy imports.
When utils._xet is imported directly (bypassing the lazy chain),
subsequent __getattr__ calls for hf_api trigger a circular import
in _buckets.py. The conftest.py eager import guard prevents this
by forcing all lazy submodules to load before any test runs.

See: docs/plans/2026-05-21-fix-huggingface-hub-circular-import.md
"""
from __future__ import annotations

import sys
from unittest.mock import patch, MagicMock

import pytest


hf_hub = pytest.importorskip("huggingface_hub")


class TestPatchAfterHfTrackImport:
    """patch() on huggingface_hub lazy attributes must work after hf_track import."""

    def test_patch_hf_api(self):
        """patch('huggingface_hub.HfApi') works after hf_track import."""
        from hf_track import HfTracker  # noqa: F401

        with patch("huggingface_hub.HfApi") as mock_api:
            mock_api.return_value = MagicMock()
            assert mock_api is not None

    def test_patch_snapshot_download(self):
        """patch('huggingface_hub.snapshot_download') works after hf_track import."""
        from hf_track import HfTracker  # noqa: F401

        with patch("huggingface_hub.snapshot_download") as mock_snap:
            mock_snap.return_value = "/tmp/repo"
            assert mock_snap is not None


class TestHuggingfaceHubEagerlyLoaded:
    """Verify that conftest.py eager import guard populated the lazy attributes."""

    def test_hf_api_accessible(self):
        """HfApi is accessible on huggingface_hub without triggering __getattr__."""
        # If conftest.py worked, the attribute was already loaded via getattr
        # and the submodule is in sys.modules
        assert "huggingface_hub.hf_api" in sys.modules

    def test_snapshot_download_accessible(self):
        """snapshot_download is accessible on huggingface_hub."""
        assert "huggingface_hub._snapshot_download" in sys.modules

    def test_logging_accessible(self):
        """huggingface_hub.logging is accessible (the trigger of the circular import)."""
        # The circular import happens when _buckets.py does `from . import logging`
        # If logging submodule is loaded, _buckets can find it
        assert hasattr(hf_hub, "logging")

    def test_buckets_import_succeeds(self):
        """huggingface_hub._buckets can be imported without circular error."""
        import huggingface_hub._buckets  # noqa: F401
        # Should not raise ImportError


class TestConftestExists:
    """Verify the conftest.py file exists and has the eager import guard."""

    def test_conftest_file_exists(self):
        """conftest.py exists in the test directory."""
        from pathlib import Path

        conftest_path = Path(__file__).parent / "conftest.py"
        assert conftest_path.exists()

    def test_conftest_has_eager_import(self):
        """conftest.py contains the eager import logic."""
        from pathlib import Path

        conftest_path = Path(__file__).parent / "conftest.py"
        content = conftest_path.read_text(encoding="utf-8")
        assert "huggingface_hub" in content
        assert "_EAGER_ATTRS" in content
