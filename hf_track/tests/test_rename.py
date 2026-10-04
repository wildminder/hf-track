"""Verify the hf_progress → hf_track rename is complete and consistent."""
from __future__ import annotations

import importlib
import importlib.metadata
import inspect

import pytest


def _declared_version() -> str:
    """The single source of truth for the version: installed metadata.

    These tests used to hardcode "0.1.0" and went red on every version
    bump. ``hf_track.__version__`` is derived from ``importlib.metadata``,
    so asserting it against a literal only ever tested staleness.
    """
    return importlib.metadata.version("hf-track")


def test_hf_track_importable():
    """hf_track package should be importable."""
    import hf_track
    assert hasattr(hf_track, "__version__")
    assert hf_track.__version__ == _declared_version()


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
    """Package version should match the installed distribution."""
    import hf_track
    assert hf_track.__version__ == _declared_version()


def test_all_exports_present():
    """All __all__ entries should be importable."""
    import hf_track
    for name in hf_track.__all__:
        assert hasattr(hf_track, name), f"Missing export: {name}"
