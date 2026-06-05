"""Layout tests for the top-level hf_track/__init__.py public API.

Step 9 of the modular-refactor plan
(docs/plans/2026-06-04-modular-refactor.md).

These tests verify the *public contract* of the package:

* Every name in ``__all__`` is actually importable from the top level.
* Every name in ``__all__`` is non-None (no broken re-exports).
* All documented subpackages can be imported standalone (no
  circular-import traps between subpackages). Verified in a subprocess
  so we never pollute the parent test session's module cache (which
  would break pickling of internal dataclasses).
* The quick-start example in the docstring actually works.
* The ``__version__`` is a valid string.

If any of these tests fails, the package is in a broken state and
external users will see ImportErrors / AttributeErrors.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


PACKAGE = "hf_track"


# ── Existence of public symbols ────────────────────────────────────


class TestPublicApiExports:
    """Every name in __all__ must be importable and non-None."""

    def test_package_imports(self):
        """``import hf_track`` must succeed."""
        # Imported lazily so the import happens at test time (after
        # the rest of the test session is set up), not at module-load
        # time (when a missing circular import would be fatal for the
        # entire test process).
        import hf_track  # noqa: PLC0415  (intentional lazy import)
        assert hf_track is not None

    def test_version_is_string(self):
        """``hf_track.__version__`` must be a non-empty string."""
        import hf_track  # noqa: PLC0415
        assert isinstance(hf_track.__version__, str)
        assert len(hf_track.__version__) > 0

    @pytest.mark.parametrize("name", [
        # Core types
        "EventType", "ProgressEvent", "ProgressPhase",
        "TransferCancelledError", "TransferDirection",
        "TransferProgressError", "TransferResult",
        "generate_transfer_id",
        # Callback classes
        "DownloadProgressTqdm", "XetDownloadProgressCallback",
        "XetProgressCallback", "XetUploadProgressCallback",
        "tqdm_upload_patcher",
        # Token management
        "XetCredentials", "XetTokenManager",
        "is_xet_available", "has_xet_session",
        # Subprocess isolation
        "XetSubprocessRunner", "SubprocessMessage",
        # High-level tracker
        "HfTracker",
    ])
    def test_all_name_is_importable(self, name: str):
        """Each name in __all__ must be importable from hf_track."""
        import hf_track  # noqa: PLC0415
        assert hasattr(hf_track, name), f"Missing public symbol: {name}"

    @pytest.mark.parametrize("name", [
        "EventType", "ProgressEvent", "TransferCancelledError",
        "TransferResult", "TransferProgressError", "TransferDirection",
        "ProgressPhase", "XetProgressCallback", "XetDownloadProgressCallback",
        "XetUploadProgressCallback", "DownloadProgressTqdm",
        "tqdm_upload_patcher", "XetTokenManager", "XetCredentials",
        "XetSubprocessRunner", "SubprocessMessage", "HfTracker",
        "is_xet_available", "has_xet_session", "generate_transfer_id",
    ])
    def test_all_name_is_not_none(self, name: str):
        """Each public name must be non-None (no broken re-exports)."""
        import hf_track  # noqa: PLC0415
        value = getattr(hf_track, name)
        assert value is not None, f"hf_track.{name} is None"

    def test_all_list_matches_actual_symbols(self):
        """``__all__`` must list only symbols that exist on the module.

        Catch typos in __all__ (e.g. a name that was never imported).
        """
        import hf_track  # noqa: PLC0415
        for name in hf_track.__all__:
            assert hasattr(hf_track, name), (
                f"hf_track.__all__ lists {name!r} but it is not on the module"
            )


# ── No circular imports between subpackages ────────────────────────


# Subpackages that must import cleanly on their own. Verified in a
# subprocess so we don't pollute the parent test session's module
# cache (which would create duplicate class objects and break pickling
# of internal dataclasses used by the worker tests).
SUBPACKAGES = [
    "hf_track",
    "hf_track.types",
    "hf_track.token",
    "hf_track.callbacks",
    "hf_track.subprocess",
    "hf_track.download",
    "hf_track.upload",
    "hf_track.tracker",
]


def _check_subpackage_in_subprocess(subpkg: str) -> None:
    """Import ``subpkg`` in a clean Python subprocess.

    Returns ``None`` on success; raises ``AssertionError`` with the
    combined stdout/stderr if the import raised ImportError.
    """
    script = textwrap.dedent(
        f"""
        import sys
        try:
            __import__({subpkg!r})
        except ImportError as e:
            print(f"IMPORT_ERROR: {{e}}", file=sys.stderr)
            sys.exit(1)
        # Also exercise the public re-exports (catches 're-export
        # binds to a partially-initialized module' issues that a bare
        # import would miss).
        from hf_track import HfTracker, SubprocessMessage
        assert HfTracker is not None
        assert SubprocessMessage is not None
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0 or "OK" not in result.stdout:
        raise AssertionError(
            f"Subprocess import of {subpkg!r} failed.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


class TestNoCircularImports:
    """Every subpackage must be importable on its own.

    If subpackage A imports from subpackage B at module top-level,
    and B imports from A, then importing A raises ImportError.
    Subpackages must use lazy/deferred imports (inside function bodies)
    to avoid this.

    We verify this in a *subprocess* so a failing import doesn't tear
    down the entire test session, AND so we don't corrupt the parent
    session's module cache with re-loaded classes (which would break
    pickling of internal dataclasses in the worker tests).
    """

    @pytest.mark.parametrize("subpkg", SUBPACKAGES)
    def test_subpackage_imports_cleanly_in_subprocess(
        self, subpkg: str
    ):
        """Each subpackage can be imported in a clean Python process."""
        _check_subpackage_in_subprocess(subpkg)

    def test_hf_track_does_not_trigger_circular_import(self):
        """``import hf_track`` must complete in a clean process."""
        _check_subpackage_in_subprocess(PACKAGE)


# ── Submodule consistency ───────────────────────────────────────────


class TestSubmoduleConsistency:
    """Each subpackage's __init__.py must follow the same shape."""

    @pytest.mark.parametrize("subpkg", [
        "hf_track.types", "hf_track.token", "hf_track.callbacks",
        "hf_track.subprocess", "hf_track.download", "hf_track.upload",
        "hf_track.tracker",
    ])
    def test_subpackage_has_all(self, subpkg: str):
        """Each subpackage should expose a documented __all__ list."""
        import importlib  # noqa: PLC0415

        mod = importlib.import_module(subpkg)
        assert hasattr(mod, "__all__"), f"{subpkg} missing __all__"
        assert isinstance(mod.__all__, list)
        assert len(mod.__all__) > 0, f"{subpkg} has empty __all__"

    @pytest.mark.parametrize("subpkg", [
        "hf_track.types", "hf_track.token", "hf_track.callbacks",
        "hf_track.subprocess", "hf_track.download", "hf_track.upload",
        "hf_track.tracker",
    ])
    def test_subpackage_all_names_resolve(self, subpkg: str):
        """Every name in __all__ must be a real attribute."""
        import importlib  # noqa: PLC0415

        mod = importlib.import_module(subpkg)
        for name in mod.__all__:
            assert hasattr(mod, name), (
                f"{subpkg}.{name} is in __all__ but missing on the module"
            )


# ── Docstring quick-start works ────────────────────────────────────


class TestDocstringQuickStart:
    """The ``Quick start`` example in the module docstring must work."""

    def test_quick_start_imports_succeed(self):
        """``from hf_track import HfTracker, EventType`` must work."""
        from hf_track import HfTracker, EventType
        assert HfTracker is not None
        assert EventType is not None

    def test_hf_tracker_constructor_runs(self):
        """``HfTracker(token="...")`` must not raise."""
        from hf_track import HfTracker
        tracker = HfTracker(token="hf_test")
        assert tracker is not None
        # It must have the public methods documented in the docstring.
        assert callable(getattr(tracker, "upload_file", None))
        assert callable(getattr(tracker, "download_file", None))
        assert callable(getattr(tracker, "events", None))

    def test_low_level_callbacks_importable(self):
        """The low-level callback example must import cleanly."""
        from hf_track import (
            XetUploadProgressCallback,
            XetDownloadProgressCallback,
            DownloadProgressTqdm,
            tqdm_upload_patcher,
        )
        for cls in (
            XetUploadProgressCallback, XetDownloadProgressCallback,
            DownloadProgressTqdm, tqdm_upload_patcher,
        ):
            assert cls is not None

    def test_sse_integration_importable(self):
        """The SSE integration example must import cleanly."""
        from hf_track.integrations.sse import EventSourceResponse
        assert EventSourceResponse is not None
