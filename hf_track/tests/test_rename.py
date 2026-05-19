"""Verify the hf_progress → hf_track rename is complete and consistent."""
from __future__ import annotations

import importlib
import inspect

import pytest


def test_hf_track_importable():
    """hf_track package should be importable."""
    import hf_track
    assert hasattr(hf_track, "__version__")
    assert hf_track.__version__ == "0.1.0"


def test_hf_track_public_api():
    """All public API symbols should be importable from hf_track."""
    import hf_track
    for name in hf_track.__all__:
        assert hasattr(hf_track, name), f"Missing export: {name}"


def test_hf_tracker_is_main_class():
    """HfTracker should be the primary class name."""
    from hf_track import HfTracker
    assert HfTracker.__name__ == "HfTracker"


def test_no_hf_progress_tracker_in_source():
    """No HfProgressTracker class should exist in the new package."""
    import hf_track
    assert not hasattr(hf_track, "HfProgressTracker"), (
        "HfProgressTracker should not exist — use HfTracker"
    )


def test_sse_module_reexports_event_source_response():
    """SSE module re-exports EventSourceResponse from sse_starlette."""
    pytest.importorskip("sse_starlette")
    from hf_track.integrations.sse import EventSourceResponse
    assert EventSourceResponse is not None


def test_package_version():
    """Package version should be preserved."""
    import hf_track
    assert hf_track.__version__ == "0.1.0"


def test_all_exports_present():
    """All __all__ entries should be importable."""
    import hf_track
    for name in hf_track.__all__:
        assert hasattr(hf_track, name), f"Missing export: {name}"
