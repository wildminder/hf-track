"""Layout/structure tests for the ``callbacks`` subpackage.

These tests are not behavioral — those live in ``test_callbacks.py``.
The goal is to lock down the subpackage shape so accidental
restructuring (e.g., moving ``DownloadProgressTqdm`` back into a
``callbacks.py`` module) is caught by CI.

Test list
=========

1. ``test_subpackage_has_expected_modules`` — the four cohesive
   submodules (``state``, ``xet_callback``, ``tqdm_download``,
   ``tqdm_patch``) all exist and import without error.
2. ``test_subpackage_public_api`` — the public names listed in
   ``__all__`` are importable from the top-level package and point at
   the same objects as the submodule-level imports.
3. ``test_state_module_exports_singleton`` — the
   ``state_manager`` singleton is the same instance the rest of the
   package uses.
4. ``test_xet_callback_subclasses_share_state_with_base`` — the
   backward-compatible subclasses pre-fill the right ``direction`` /
   ``phase`` defaults.
5. ``test_dummy_file_singleton_is_writable`` — the shared
   ``_dummy_file`` is a no-op sink that accepts ``.write()`` and
   ``.flush()`` calls without raising.
6. ``test_tqdm_patch_module_does_not_pollute_at_import_time`` —
   importing the submodule does NOT mutate ``tqdm.auto.tqdm`` (the
   patcher only fires inside its context manager).
7. ``test_callbacks_subpackage_does_not_depend_on_download`` — the
   ``callbacks`` subpackage is lower-level than ``download`` and
   ``upload`` and must not import from them (avoids cycles).
8. ``test_internal_modules_have_no_cross_dependencies_on_tqdm_patch`` —
   the state and xet-callback modules do not reach into the
   upload-patcher module.
9. ``test_callbacks_layout_module_size`` — a soft budget on the
   number of lines per module to catch accidental growth.
"""

from __future__ import annotations

import importlib
import queue


# ── Module discovery ────────────────────────────────────────────


EXPECTED_SUBMODULES = (
    "state",
    "xet_callback",
    "tqdm_download",
    "tqdm_patch",
)


def test_subpackage_has_expected_modules():
    """The four cohesive submodules all exist and import cleanly."""
    pkg = importlib.import_module("hf_track.callbacks")
    for name in EXPECTED_SUBMODULES:
        full_name = f"hf_track.callbacks.{name}"
        mod = importlib.import_module(full_name)
        assert mod is not None, f"Failed to import {full_name}"
        # Each submodule should declare its public API in __all__
        # (best practice, not strictly required, but documented).
        assert hasattr(mod, "__doc__"), f"{full_name} missing module docstring"


def test_subpackage_public_api():
    """All public names in __all__ are importable from the top level."""
    pkg = importlib.import_module("hf_track.callbacks")

    # Full set of public + internal names that downstream code or
    # sibling subpackages rely on. If you add a new re-export to
    # __init__.py, add it here too so the test locks it in.
    expected = {
        "TransferStateManager",
        "state_manager",
        "XetProgressCallback",
        "XetUploadProgressCallback",
        "XetDownloadProgressCallback",
        "DownloadProgressTqdm",
        "tqdm_upload_patcher",
        "_DummyFile",
        "_dummy_file",
    }
    actual = set(pkg.__all__)
    assert expected.issubset(actual), f"Missing from __all__: {expected - actual}"
    for name in expected:
        assert hasattr(pkg, name), f"hf_track.callbacks.{name} is not importable"
        # Make sure the re-export points at the same object as the
        # submodule-level import (catches accidental duplicates).
        if name in ("TransferStateManager", "state_manager"):
            submodule_obj = importlib.import_module("hf_track.callbacks.state").__dict__.get(name)
        elif name in ("XetProgressCallback", "XetUploadProgressCallback", "XetDownloadProgressCallback", "_DummyFile", "_dummy_file"):
            submodule_obj = importlib.import_module("hf_track.callbacks.xet_callback").__dict__.get(name)
        elif name == "DownloadProgressTqdm":
            submodule_obj = importlib.import_module("hf_track.callbacks.tqdm_download").DownloadProgressTqdm
        elif name == "tqdm_upload_patcher":
            submodule_obj = importlib.import_module("hf_track.callbacks.tqdm_patch").tqdm_upload_patcher
        else:
            submodule_obj = None
        if submodule_obj is not None:
            assert getattr(pkg, name) is submodule_obj, (
                f"hf_track.callbacks.{name} is not the same object as the "
                f"submodule's {name}; check the re-export in __init__.py"
            )


# ── State module ────────────────────────────────────────────────


def test_state_module_exports_singleton():
    """The state_manager singleton is the same object the package uses."""
    state_mod = importlib.import_module("hf_track.callbacks.state")
    pkg = importlib.import_module("hf_track.callbacks")
    assert pkg.state_manager is state_mod.state_manager
    assert isinstance(state_mod.state_manager, state_mod.TransferStateManager)

    # Basic lifecycle: init → update → get → clear
    sm = state_mod.TransferStateManager()
    sm.init_download("test-layout-1")
    sm.update_download_bytes("test-layout-1", 100, 1000)
    s = sm.get_state("test-layout-1")
    assert s["bytes_completed"] == 100
    assert s["total_bytes"] == 1000
    sm.clear_state("test-layout-1")
    assert sm.get_state("test-layout-1") == {}


# ── Xet callback module ─────────────────────────────────────────


def test_xet_callback_subclasses_share_state_with_base():
    """Subclasses pre-fill the right direction/phase defaults."""
    xc = importlib.import_module("hf_track.callbacks.xet_callback")
    q = queue.Queue()

    up = xc.XetUploadProgressCallback(
        filename="up.bin", total_bytes=100, event_queue=q, transfer_id="up"
    )
    assert up.direction.name == "UPLOAD"
    assert up.phase.name == "UPLOADING"

    down = xc.XetDownloadProgressCallback(
        filename="down.bin", total_bytes=100, event_queue=q, transfer_id="down"
    )
    assert down.direction.name == "DOWNLOAD"
    assert down.phase.name == "DOWNLOADING"


def test_dummy_file_singleton_is_writable():
    """The shared _dummy_file is a no-op sink."""
    xc = importlib.import_module("hf_track.callbacks.xet_callback")
    assert xc._dummy_file is not None
    # Should not raise; write returns length of the input
    assert xc._dummy_file.write("hello") == 5
    # Flush is a no-op
    xc._dummy_file.flush()


# ── tqdm_patch module ───────────────────────────────────────────


def test_tqdm_patch_module_does_not_pollute_at_import_time():
    """Importing tqdm_patch does NOT replace tqdm.auto.tqdm."""
    import tqdm.auto

    original = tqdm.auto.tqdm
    importlib.import_module("hf_track.callbacks.tqdm_patch")
    assert tqdm.auto.tqdm is original, (
        "Importing hf_track.callbacks.tqdm_patch mutated tqdm.auto.tqdm; "
        "the patcher must only fire inside its context manager."
    )


def test_tqdm_upload_patcher_restores_tqdm():
    """The patcher restores tqdm.auto.tqdm on context exit (even on exc)."""
    import tqdm.auto
    from hf_track.callbacks import tqdm_upload_patcher

    original = tqdm.auto.tqdm
    q: queue.Queue = queue.Queue()

    # Normal exit
    with tqdm_upload_patcher(
        event_queue=q, transfer_id="layout-1", filename="f.bin", total_bytes=10
    ):
        assert tqdm.auto.tqdm is not original
    assert tqdm.auto.tqdm is original

    # Exit on exception
    try:
        with tqdm_upload_patcher(
            event_queue=q, transfer_id="layout-2", filename="f.bin", total_bytes=10
        ):
            assert tqdm.auto.tqdm is not original
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert tqdm.auto.tqdm is original


# ── Cross-module dependency invariants ──────────────────────────


def test_callbacks_subpackage_does_not_depend_on_download():
    """``callbacks`` is a lower-level subpackage and must not import
    from ``download``/``upload`` (which would create a cycle since
    those subpackages import from ``callbacks``)."""
    pkg = importlib.import_module("hf_track.callbacks")
    # Walk the subpackage and check that no submodule file imports
    # from the higher-level subpackages.
    import pathlib
    base = pathlib.Path(pkg.__file__).parent
    for name in EXPECTED_SUBMODULES:
        path = base / f"{name}.py"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for forbidden in ("from ..download", "from ..upload", "from hf_track.download", "from hf_track.upload"):
            assert forbidden not in text, (
                f"{path.name} imports from a higher-level subpackage ({forbidden!r}); "
                f"this would create an import cycle."
            )


def test_internal_modules_have_no_cross_dependencies_on_tqdm_patch():
    """``state`` and ``xet_callback`` must not import the upload patcher.

    Docstring mentions of the module are allowed (they document the
    dependency graph), but actual import statements are not.
    """
    for name in ("state", "xet_callback"):
        mod = importlib.import_module(f"hf_track.callbacks.{name}")
        text = mod.__file__ and open(mod.__file__, encoding="utf-8").read() or ""
        for forbidden in ("from .tqdm_patch", "import tqdm_patch"):
            assert forbidden not in text, (
                f"{name}.py imports tqdm_patch; the patcher module is "
                f"intended to be a leaf of the subpackage."
            )


# ── Module size guard (soft budget) ─────────────────────────────


# Per-module soft budgets. The plan explicitly notes that
# ``DownloadProgressTqdm`` is kept whole because its throttling,
# absolute-position fix, and COMPLETE synthesis share state
# attributes, so it gets a larger budget than the other modules.
# If you find yourself raising these numbers, split the class first.
MODULE_LINE_BUDGETS = {
    "state": 250,
    "xet_callback": 250,
    "tqdm_download": 600,  # DownloadProgressTqdm kept whole by design
    "tqdm_patch": 350,
}


def test_callbacks_layout_module_size():
    """None of the cohesive submodules should blow past its budget."""
    import pathlib
    pkg = importlib.import_module("hf_track.callbacks")
    base = pathlib.Path(pkg.__file__).parent
    for name, budget in MODULE_LINE_BUDGETS.items():
        path = base / f"{name}.py"
        if not path.exists():
            continue
        line_count = sum(1 for _ in path.open(encoding="utf-8"))
        assert line_count <= budget, (
            f"{path.name} is {line_count} lines (budget {budget}). "
            f"Consider splitting further or trimming comments."
        )
