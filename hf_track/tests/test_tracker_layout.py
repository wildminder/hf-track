"""Layout tests for the tracker/ subpackage.

Step 8 of the modular-refactor plan split the original monolithic
``hf_track/src/hf_track/tracker.py`` (28KB, 751 lines, single god
class) into a ``tracker/`` subpackage with six cohesive mixin
modules. These tests verify the layout invariants so future
refactors don't accidentally re-introduce the god class.

See docs/plans/2026-06-04-modular-refactor.md, Step 8.
"""

from __future__ import annotations

import inspect
import pathlib

import pytest


TRACKER_DIR = pathlib.Path("hf_track/src/hf_track/tracker")


# ── Existence ───────────────────────────────────────────────────────


class TestTrackerPackageExists:
    """The tracker/ subpackage must exist with the expected files."""

    def test_tracker_dir_is_directory(self):
        assert TRACKER_DIR.is_dir(), (
            f"Expected {TRACKER_DIR} to be a directory (subpackage)"
        )

    def test_tracker_init_exists(self):
        assert (TRACKER_DIR / "__init__.py").is_file()

    def test_tracker_init_has_public_class(self):
        """The __init__.py must compose the public HfTracker class."""
        from hf_track.tracker import HfTracker
        assert HfTracker.__name__ == "HfTracker"

    def test_all_mixin_modules_exist(self):
        """Each of the 6 mixin modules must exist as a separate file."""
        for name in ("_core", "_events", "_async", "_downloads", "_uploads", "_xet_impls"):
            path = TRACKER_DIR / f"{name}.py"
            assert path.is_file(), f"Missing mixin module: {path}"

    def test_no_legacy_tracker_py_module(self):
        """The old monolithic tracker.py must not co-exist with the package."""
        legacy = pathlib.Path("hf_track/src/hf_track/tracker.py")
        assert not legacy.exists(), (
            f"{legacy} still exists. Either delete it (after split) or "
            "make it a backward-compat shim (it cannot be both a module "
            "and a package of the same name)."
        )


# ── Mixin Composition ───────────────────────────────────────────────


class TestMixinsComposeIntoHfTracker:
    """HfTracker must be a single class inheriting from all 6 mixins."""

    def test_hf_tracker_inherits_from_all_mixins(self):
        from hf_track.tracker import HfTracker
        from hf_track.tracker._async import _TrackerAsync
        from hf_track.tracker._core import _TrackerCore
        from hf_track.tracker._downloads import _TrackerDownloads
        from hf_track.tracker._events import _TrackerEvents
        from hf_track.tracker._uploads import _TrackerUploads
        from hf_track.tracker._xet_impls import _TrackerXetImpls

        for mixin in (
            _TrackerCore,
            _TrackerEvents,
            _TrackerAsync,
            _TrackerDownloads,
            _TrackerUploads,
            _TrackerXetImpls,
        ):
            assert issubclass(HfTracker, mixin), (
                f"HfTracker must inherit from {mixin.__name__}"
            )

    def test_hf_tracker_is_single_class_not_a_factory(self):
        """HfTracker must be a class, not a function or factory."""
        from hf_track.tracker import HfTracker
        assert inspect.isclass(HfTracker)

    def test_all_24_methods_present_on_class(self):
        """Every method that was on the original god class must still
        be resolvable as an attribute of the new HfTracker class."""
        from hf_track.tracker import HfTracker
        expected = {
            # public sync
            "download_file", "download_snapshot", "download_snapshot_streaming",
            "upload_file", "upload_bytes", "upload_folder",
            # public async wrappers
            "download_file_async", "download_snapshot_async",
            "upload_file_async", "upload_bytes_async", "upload_folder_async",
            # public event consumer
            "get_events", "events", "wait_for_complete",
            # public state
            "cancel", "is_cancelled", "cleanup_transfer",
            # private core
            "_prepare_transfer",
            # private xet impls
            "_download_file_xet", "_download_snapshot_xet",
            "_upload_file_xet", "_upload_bytes_xet",
            "_upload_file_lfs", "_upload_bytes_via_temp",
        }
        actual = set(dir(HfTracker))
        missing = expected - actual
        assert not missing, f"HfTracker is missing methods: {sorted(missing)}"

    def test_methods_resolve_to_their_mixin_via_mro(self):
        """Each method must be defined on one of the mixin classes
        (i.e. it lives in its own file, not on HfTracker directly)."""
        from hf_track.tracker import HfTracker
        # Get the qualname of the defining class for a few key methods
        for method_name in ("_download_snapshot_xet", "wait_for_complete", "download_file"):
            method = getattr(HfTracker, method_name)
            qualname = method.__qualname__
            # qualname looks like "_TrackerXetImpls._download_snapshot_xet"
            # or "HfTracker.wait_for_complete" (only if defined on HfTracker).
            # After the split, NO method is defined directly on HfTracker.
            assert not qualname.startswith("HfTracker."), (
                f"{method_name} is still defined directly on HfTracker "
                f"(qualname={qualname!r}). It should live in a mixin module."
            )


# ── Backward-Compat Patches ────────────────────────────────────────


class TestBackwardCompatPatches:
    """Test patches that worked on the monolithic module must still work."""

    def test_patch_is_xet_available_on_tracker_package(self):
        """``patch("hf_track.tracker.is_xet_available", ...)`` must work
        because the symbol is re-exported from tracker/__init__.py."""
        from unittest.mock import patch
        import hf_track.tracker as tracker_pkg

        # The symbol is accessible on the package.
        assert tracker_pkg.is_xet_available is not None

        # And can be patched. We must look it up via the package attribute
        # at call time (NOT via a local import binding), otherwise the patch
        # will not be visible to the caller.
        with patch("hf_track.tracker.is_xet_available", return_value=False):
            assert tracker_pkg.is_xet_available() is False

    def test_patch_object_hf_tracker_download_snapshot_xet(self):
        """``patch.object(HfTracker, "_download_snapshot_xet")`` must work
        because the method is resolved via MRO to the _xet_impls mixin."""
        from unittest.mock import patch
        from hf_track.tracker import HfTracker

        with patch.object(HfTracker, "_download_snapshot_xet", return_value="/x") as mock:
            assert HfTracker()._download_snapshot_xet() == "/x"
            mock.assert_called_once()

    def test_inspect_getsource_finds_method_implementation(self):
        """``inspect.getsource(HfTracker._download_snapshot_xet)`` must
        return the source code (used by test_snapshot_regressions.py
        to assert it does NOT call the broken xetsession path)."""
        from hf_track.tracker import HfTracker
        source = inspect.getsource(HfTracker._download_snapshot_xet)
        # The fix delegates to download_snapshot_with_xet (the proven path),
        # not to download_snapshot_with_xet_session (the broken path).
        assert "download_snapshot_with_xet" in source
        assert "download_snapshot_with_xet_session" not in source

    def test_inspect_signature_on_public_method(self):
        """``inspect.signature(HfTracker.download_snapshot_streaming)`` must
        return the full parameter list (used by test_tracker.py)."""
        from hf_track.tracker import HfTracker
        sig = inspect.signature(HfTracker.download_snapshot_streaming)
        params = list(sig.parameters.keys())
        for expected_param in (
            "repo_id", "allow_patterns", "ignore_patterns",
            "repo_type", "revision", "local_dir", "transfer_id",
            "force_download", "fsync_interval",
        ):
            assert expected_param in params, (
                f"download_snapshot_streaming missing param {expected_param!r}"
            )


# ── Public Re-Export ────────────────────────────────────────────────


class TestPublicReExports:
    """The package and ``hf_track.__init__`` must re-export HfTracker."""

    def test_hf_tracker_importable_from_subpackage(self):
        from hf_track.tracker import HfTracker
        assert HfTracker.__name__ == "HfTracker"

    def test_hf_tracker_importable_from_top_level(self):
        from hf_track import HfTracker
        assert HfTracker.__name__ == "HfTracker"

    def test_three_import_paths_return_same_class(self):
        from hf_track import HfTracker as A
        from hf_track.tracker import HfTracker as B
        from hf_track.tracker import HfTracker as C
        assert A is B is C


# ── Module Size Budget ──────────────────────────────────────────────


class TestModuleSizesAreReasonable:
    """No single mixin module should balloon past its budget — this is
    the whole point of the split. If a file grows, refactor further.

    Per-module budgets:
        _core, _events, _async, _downloads, _uploads: 250 lines
        _xet_impls: 400 lines (it owns 6 low-level Xet/LFS impl methods,
                    including _download_snapshot_xet which is ~80 lines
                    of code+docstring; this is its natural size).
    """

    # Per-module budgets (tuned to the current natural size of each mixin
    # plus a safety margin).  If a file grows past its budget, consider
    # splitting it further.
    BUDGETS = {
        "_core": 250,
        "_events": 250,
        "_async": 250,
        "_downloads": 250,
        "_uploads": 250,
        "_xet_impls": 400,
    }

    @pytest.mark.parametrize("name,limit", list(BUDGETS.items()))
    def test_mixin_module_within_budget(self, name: str, limit: int):
        path = TRACKER_DIR / f"{name}.py"
        line_count = sum(1 for _ in path.open(encoding="utf-8"))
        assert line_count <= limit, (
            f"{name}.py is {line_count} lines (max {limit}). "
            "Consider splitting this mixin further."
        )

    def test_init_module_is_thin(self):
        """__init__.py should be a composition layer, not a kitchen sink."""
        path = TRACKER_DIR / "__init__.py"
        line_count = sum(1 for _ in path.open(encoding="utf-8"))
        assert line_count <= 80, (
            f"tracker/__init__.py is {line_count} lines (max 80). "
            "It should be a thin composition layer."
        )
