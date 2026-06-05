"""Layout/structure tests for the ``_xet_worker_legacy`` module.

Both workers in this module (``_xet_session_snapshot_worker`` and
``_xet_session_download_worker``) are **deprecated as of 2026-06-03**
due to three critical correctness bugs. They are kept here for
backward compatibility with direct importers, and to allow future
re-implementation.

These tests lock down:

1. Both legacy workers remain importable from
   ``hf_track._xet_worker`` (the historical import path used by
   ``download/xet_snapshot.py`` and ``test_snapshot_regressions.py``).
2. Both workers' docstrings contain the ``"deprecated"`` warning so
   future maintainers see the deprecation notice when they inspect
   them in an IDE.
3. The legacy module re-uses the shared helpers from
   ``_xet_worker`` (init, safe_put, exception handler, throttler)
   — verifies the move is a true extraction, not a copy.
4. The main ``_xet_worker`` module shrank by at least 500 lines after
   the legacy extraction (guards against accidental re-inlining).
5. The ``hf_track._xet_worker_legacy`` module is itself importable.
"""

from __future__ import annotations

import importlib


def test_legacy_module_is_importable():
    """The new legacy module can be imported."""
    mod = importlib.import_module("hf_track._xet_worker_legacy")
    assert mod is not None
    assert hasattr(mod, "__doc__")
    assert "deprecated" in mod.__doc__.lower(), (
        "The module docstring should mention the deprecation status "
        "so anyone landing on the module immediately understands it."
    )


def test_legacy_workers_re_exported_from_main_module():
    """``from hf_track._xet_worker import _xet_session_*`` still works."""
    main = importlib.import_module("hf_track._xet_worker")
    legacy = importlib.import_module("hf_track._xet_worker_legacy")

    assert hasattr(main, "_xet_session_snapshot_worker")
    assert hasattr(main, "_xet_session_download_worker")

    # The objects must be the same (not just same name) so the
    # deprecation test's inspect.getsource() call still works.
    assert main._xet_session_snapshot_worker is legacy._xet_session_snapshot_worker
    assert main._xet_session_download_worker is legacy._xet_session_download_worker


def test_legacy_workers_docstring_marks_them_deprecated():
    """The snapshot worker docstring must contain 'deprecated' (regex-free
    check).

    The original ``_xet_session_snapshot_worker`` (the worker that was
    actually broken in production and reverted in 2026-06-03) carries an
    explicit ``.. deprecated::`` note. The single-file
    ``_xet_session_download_worker`` is in the same deprecated family
    (per the module docstring) but historically did not have a per-function
    deprecation note. We only enforce the explicit note on the snapshot
    worker because that matches the original code that the regression
    test (``test_snapshot_regressions.py::test_xet_session_snapshot_worker_marked_deprecated``)
    already locks down.

    To verify the single-file worker is still marked deprecated, we
    instead check the *module* docstring (which we already do in
    ``test_legacy_module_is_importable``).
    """
    legacy = importlib.import_module("hf_track._xet_worker_legacy")

    worker = getattr(legacy, "_xet_session_snapshot_worker")
    assert worker.__doc__ is not None
    assert "deprecated" in worker.__doc__.lower(), (
        f"_xet_session_snapshot_worker docstring must contain 'deprecated' "
        f"to warn future maintainers. This is the same assertion that "
        f"test_snapshot_regressions.py::test_xet_session_snapshot_worker_marked_deprecated "
        f"enforces — keeping it here for defense-in-depth."
    )


def test_legacy_module_uses_shared_helpers_from_main_module():
    """The legacy module imports helpers from ``_xet_worker`` lazily
    inside the worker function bodies (not at module-top-level — that
    would create a circular import, since ``_xet_worker`` re-exports
    the deprecated workers from this module).

    This is verified by checking the module's source contains the
    expected lazy import statement.
    """
    legacy = importlib.import_module("hf_track._xet_worker_legacy")
    assert legacy.__file__ is not None
    with open(legacy.__file__, encoding="utf-8") as f:
        text = f.read()

    # Module-level: must NOT have the circular import.
    # Look at the lines up to the first function definition; imports
    # before that would run at module load and cycle.
    lines = text.splitlines()
    first_def_line = next(
        (i for i, ln in enumerate(lines) if ln.startswith("def ")),
        len(lines),
    )
    module_prefix = "\n".join(lines[:first_def_line])
    assert "from ._xet_worker import (" not in module_prefix, (
        "Module-top-level 'from ._xet_worker import' would create a "
        "circular import. Imports must be inside the function bodies."
    )

    # Function bodies: must use lazy imports.
    expected_imports = (
        "from ._xet_worker import (",
        "_ProgressThrottler",
        "_download_worker",
        "_handle_worker_exception",
        "_init_worker",
        "_safe_put",
    )
    for needle in expected_imports:
        assert needle in text, (
            f"_xet_worker_legacy.py should reference '{needle}' (lazy "
            f"import from _xet_worker). True extraction, not duplication."
        )


def test_main_worker_module_shrank_after_extraction():
    """Sanity guard: the main ``_xet_worker.py`` should be smaller
    than 60 KB after the legacy extraction. The original was 74 KB."""
    import os
    size = os.path.getsize("hf_track/src/hf_track/_xet_worker.py")
    assert size < 60 * 1024, (
        f"_xet_worker.py is {size} bytes; expected < 60 KB after the "
        f"legacy extraction. Either the extraction didn't happen, or "
        f"new code was added that pushes the file back over budget."
    )


def test_legacy_module_size_within_budget():
    """The legacy module should not exceed the soft 40 KB budget.

    Soft budget allows for the deprecated XetSession workers' verbose
    deprecation docstrings (which we keep verbatim for traceability).
    """
    import os
    size = os.path.getsize("hf_track/src/hf_track/_xet_worker_legacy.py")
    assert size < 40 * 1024, (
        f"_xet_worker_legacy.py is {size} bytes; expected < 40 KB. "
        f"If it grew, check that the deprecation docstrings weren't "
        f"expanded or duplicated."
    )
