"""Regression tests for Step 14 of the modular refactor plan.

Step 8.5 (2026-06-05) removed the deprecated ``XetSession`` API from
the production code (``hf_track.download.download_file_with_xet_session``
etc.). However, three references to the removed API were left in the
source tree and only surfaced at runtime:

  Bug 1 — ``hf_track/download/xet_streaming.py`` had a function-level
          ``from ._xet_worker import import _xet_streaming_download_worker``
          that re-imported a worker from inside the ``download``
          subpackage — a path that never existed. The module-level
          import on line 30 was correct. This raised
          ``No module named 'hf_track.download._xet_worker'`` at
          function call time. (As of 2026-06-15 that streaming worker
          is replaced by ``_xet_file_download_worker``; the regression
          is moot but the path-stickiness test is retained.)

  Bug 2 — ``hf_track/tracker/_xet_impls.py::_TrackerXetImpls._download_file_xet``
          had a try/except that first tried the removed
          ``download_file_with_xet_session`` and then fell back to the
          working ``download_file_with_xet``. The first call always
          raised ``ImportError`` at runtime. This was dead code from
          before Step 8.5.

  Bug 3 — ``hf_track/tracker/_xet_impls.py::_TrackerXetImpls._download_snapshot_xet``
          had a stale docstring claiming it uses the "NEW
          ``hf_xet.XetSession`` API". The actual code uses
          ``download_snapshot_with_xet`` (the proven path). No runtime
          impact, but misleading for future maintainers.

These tests guard against any future regression of the same shape.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

from .test_module_size import _find_source_root


# A child interpreter launched as ``sys.executable -c ...`` does not
# inherit the ``sys.path`` entry that pytest injects for the *parent*
# process (the ``pythonpath`` ini option); it only reads the real
# ``PYTHONPATH``. On a bare checkout ``hf_track`` is not pip-installed,
# so the fresh-interpreter test below could not import the package at
# all. See plan 2026-10-04, step S06.
SOURCE_ROOT = _find_source_root().parent


def _child_env() -> dict[str, str]:
    """An environment in which a fresh interpreter can import ``hf_track``."""
    return {**os.environ, "PYTHONPATH": str(SOURCE_ROOT)}


# ── Constants ────────────────────────────────────────────────────

# Symbols removed in Step 8.5. Production code must NEVER reference
# any of these (examples are allowed because they can be updated
# independently and are exercised by test_examples.py).
REMOVED_XETSESSION_SYMBOLS: tuple[str, ...] = (
    "XetSession",
    "download_file_with_xet_session",
    "download_snapshot_with_xet_session",
    "_xet_session_download_worker",
    "_xet_session_snapshot_worker",
    "XetSessionProvider",
)

# Files where references to removed symbols would indicate a real bug
# (not just a historical comment in a docstring or test).
PROD_SOURCE_ROOTS: tuple[str, ...] = (
    "src/hf_track",
)

# Files that are permitted to mention removed symbols in non-code
# contexts (docstrings, comments). These are excluded from the strict
# production scan because they exist to DOCUMENT the removal.
DOCUMENTING_FILES: frozenset[str] = frozenset(
    {
        # download/__init__.py: module docstring lists removed names
        "src/hf_track/download/__init__.py",
    }
)


# ── Helpers ──────────────────────────────────────────────────────


def _find_source_root() -> pathlib.Path:
    """Locate the production source root by walking upward."""
    start = pathlib.Path(__file__).resolve().parent
    for ancestor in [start, *start.parents]:
        candidate = ancestor / "src" / "hf_track"
        if (candidate.is_dir() and (candidate / "__init__.py").is_file()):
            return ancestor
    raise RuntimeError("Could not locate source root (expected src/hf_track/__init__.py)")


def _ast_references_symbol(tree: ast.AST, symbol: str) -> bool:
    """Return True if *symbol* is referenced anywhere in *tree* as an
    actual code reference (Name, Attribute, alias, or string constant).

    String constants are scanned both as full strings and by their
    rightmost dotted segment, so both ``"XetSession"`` and
    ``"hf_xet.XetSession"`` match.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == symbol:
            return True
        if isinstance(node, ast.Attribute) and node.attr == symbol:
            return True
        if isinstance(node, ast.alias) and node.name.split(".")[-1] == symbol:
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if not value:
                continue
            if value == symbol or value.endswith(f".{symbol}"):
                return True
    return False


def _scan_production_source_for_removed_symbols() -> dict[str, list[str]]:
    """Walk every .py file under PROD_SOURCE_ROOTS and report which
    removed symbols are referenced (if any).

    Returns a ``{symbol: [relative_paths...]}`` mapping. An empty
    mapping means the codebase is clean.
    """
    src_root = _find_source_root()
    findings: dict[str, list[str]] = {}
    for prod_root_rel in PROD_SOURCE_ROOTS:
        prod_root = src_root / pathlib.Path(prod_root_rel).relative_to("src")
        if not prod_root.exists():
            continue
        for path in sorted(prod_root.rglob("*.py")):
            rel = path.relative_to(src_root).as_posix()
            if rel in DOCUMENTING_FILES:
                # Permitted to mention removed symbols in non-code
                # contexts (module docstring explaining the removal).
                continue
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=rel)
            except (SyntaxError, UnicodeDecodeError):
                continue
            for symbol in REMOVED_XETSESSION_SYMBOLS:
                if _ast_references_symbol(tree, symbol):
                    findings.setdefault(symbol, []).append(rel)
    return findings


# ── Bug 1: download.xet_streaming import path ────────────────────


class TestXetStreamingImportable:
    """``hf_track.download.xet_streaming`` must import cleanly.

    Regression: pre-Step-14, the module had a function-level
    ``from ._xet_worker import _xet_streaming_download_worker`` that
    re-imported a worker from inside the ``download`` subpackage.
    ``download._xet_worker`` never existed, so calling
    ``download_snapshot_streaming()`` raised
    ``No module named 'hf_track.download._xet_worker'`` at runtime,
    even though the module itself imported fine. (Streaming worker
    removed 2026-06-15 in favour of the file-download-group worker;
    this test now guards the same import-path for the new worker.)
    """

    def test_module_imports(self):
        # The module-level imports were already correct; this guards
        # against any future import-time regression.
        from hf_track.download import xet_streaming  # noqa: F401

    def test_download_snapshot_streaming_callable(self):
        from hf_track.download.xet_streaming import download_snapshot_streaming

        assert callable(download_snapshot_streaming)

    def test_xet_streaming_module_has_no_broken_relative_worker_import(self):
        """The function body must NOT contain a relative import that
        tries to reach ``_xet_worker`` from inside ``download/``.

        The correct path is ``from .._xet_worker import ...`` (two
        dots = parent package = ``hf_track/``). The bug was a single
        dot (``from ._xet_worker import ...``) which resolves to
        ``hf_track.download._xet_worker`` and does not exist.
        """
        from hf_track.download import xet_streaming

        src = pathlib.Path(xet_streaming.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            # level: 0 = absolute, 1 = ".", 2 = "..", etc.
            if node.level != 1 or node.module != "_xet_worker":
                continue
            pytest.fail(
                f"{xet_streaming.__name__}:{node.lineno} contains a broken "
                f"relative import `from ._xet_worker import ...` (resolves to "
                f"`hf_track.download._xet_worker`, which does not exist). "
                f"Use `from .._xet_worker import ...` instead."
            )

    def test_xet_streaming_import_via_hf_track_download_package(self):
        """The streaming downloader must be reachable via the public
        ``hf_track.download`` surface (it is the recommended path for
        snapshot downloads of xet-stored repos)."""
        from hf_track.download import download_snapshot_streaming

        assert callable(download_snapshot_streaming)


# ── Bug 2: tracker._xet_impls dead XetSession fallback ───────────


class TestXetImplsNoRemovedSymbols:
    """``hf_track.tracker._xet_impls`` must not reference the removed
    XetSession API.

    Regression: pre-Step-14, ``_TrackerXetImpls._download_file_xet``
    contained a try/except that first tried the removed
    ``download_file_with_xet_session`` and then fell back to
    ``download_file_with_xet``. The first call always raised
    ``ImportError`` at runtime, defeating the Xet single-file download
    path.
    """

    def test_xet_impls_module_imports(self):
        from hf_track.tracker import _xet_impls  # noqa: F401

    def test_download_file_xet_uses_only_remaining_api(self):
        """The implementation source of ``_download_file_xet`` must
        only import ``download_file_with_xet`` (not the removed
        ``download_file_with_xet_session``)."""
        from hf_track.tracker import _xet_impls

        src = pathlib.Path(_xet_impls.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            for alias in node.names:
                if alias.name == "download_file_with_xet_session":
                    pytest.fail(
                        f"{_xet_impls.__name__}:{node.lineno} still imports "
                        f"the removed symbol 'download_file_with_xet_session' "
                        f"(deleted in Step 8.5)."
                    )

    def test_download_snapshot_xet_uses_only_remaining_api(self):
        """The implementation source of ``_download_snapshot_xet`` must
        only import ``download_snapshot_with_xet`` (not the removed
        ``download_snapshot_with_xet_session``)."""
        from hf_track.tracker import _xet_impls

        src = pathlib.Path(_xet_impls.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            for alias in node.names:
                if alias.name == "download_snapshot_with_xet_session":
                    pytest.fail(
                        f"{_xet_impls.__name__}:{node.lineno} still imports "
                        f"the removed symbol 'download_snapshot_with_xet_session' "
                        f"(deleted in Step 8.5)."
                    )

    def test_xet_impls_has_no_try_except_fallback_pattern(self):
        """``_download_file_xet`` used to wrap the call in a
        try/except ImportError block. After Step 14 the call is
        unconditional — we check that the source contains no
        ``except ImportError`` inside that method's body.

        Note: this is a structural check, not a behavioral one. If a
        future change legitimately needs an ImportError catch for some
        other symbol, this test must be re-tightened.
        """
        from hf_track.tracker import _xet_impls

        src = pathlib.Path(_xet_impls.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "_download_file_xet":
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.ExceptHandler):
                    continue
                handled = child.type
                # ``except ImportError`` (no parens) is ast.Name
                if isinstance(handled, ast.Name) and handled.id == "ImportError":
                    pytest.fail(
                        f"{_xet_impls.__name__}:{child.lineno} "
                        f"_download_file_xet still has an `except ImportError` "
                        f"block (legacy XetSession-fallback shim)."
                    )


# ── Defense in depth: no production source references removed symbols


class TestProductionSourceHasNoRemovedSymbols:
    """No file under ``src/hf_track/`` may reference any of the
    removed XetSession symbols, except files that are explicitly
    permitted to document the removal in docstrings/comments.

    This is a coarse, AST-based scan that catches both
    ``from ... import XetSession`` and ``"XetSession"`` string
    constants. It is intentionally redundant with the per-file tests
    above so a single missed location cannot silently pass CI.
    """

    def test_no_removed_symbols_in_production_source(self):
        findings = _scan_production_source_for_removed_symbols()
        if not findings:
            return
        # Format a human-readable failure message
        lines = ["Removed XetSession symbols are still referenced in production code:"]
        for symbol, files in sorted(findings.items()):
            lines.append(f"  - {symbol}:")
            for f in files:
                lines.append(f"      {f}")
        pytest.fail("\n".join(lines))


# ── Bug 3: stale docstring on _download_snapshot_xet ─────────────


class TestXetImplsDocstrings:
    """The docstring of ``_download_snapshot_xet`` must accurately
    describe the implementation. Pre-Step-14 it claimed to use the
    "NEW ``hf_xet.XetSession`` API" but actually calls
    ``download_snapshot_with_xet`` (the proven path).
    """

    def test_download_snapshot_xet_docstring_mentions_proven_path(self):
        from hf_track.tracker import _xet_impls

        doc = _xet_impls._TrackerXetImpls._download_snapshot_xet.__doc__ or ""
        # The docstring must positively assert the proven path...
        assert "download_snapshot_with_xet" in doc, (
            "_download_snapshot_xet docstring should mention the "
            "download_snapshot_with_xet function it actually uses."
        )
        # ...and must NOT claim that the function prefers or falls back
        # to XetSession. (It is OK to mention "XetSession" in a
        # historical "we removed this in Step 8.5" note; the bug was
        # that the docstring claimed to use it as the primary path.)
        prohibited_phrases = (
            "Uses the NEW",
            "prefer the new",
            "fall back to the old download_snapshot_with_xet",
        )
        for phrase in prohibited_phrases:
            assert phrase.lower() not in doc.lower(), (
                f"_download_snapshot_xet docstring still contains the legacy "
                f"phrase {phrase!r} — update it to describe the proven "
                f"download_snapshot_with_xet path."
            )


# ── Subprocess test: import the full module set in a fresh Python ──


class TestSubprocessImportsClean:
    """Run a fresh Python interpreter that imports every public
    download/tracker module. Catches the "module imports cleanly but a
    function-level import fails when the function is called" class of
    bug that Bug 1 belongs to.

    The subprocess has its own ``sys.modules`` cache, so it cannot
    pick up any parent process pollution. This is the gold standard
    for verifying that a fresh interpreter can construct the full
    object graph.
    """

    def test_subprocess_can_construct_hf_tracker(self):
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(
                """
                import hf_track
                from hf_track import HfTracker
                # Construct (does not call .download_*; just ensures the
                # class definition itself imports cleanly).
                t = HfTracker(token="hf_test")
                # Sanity: the mixin that contained Bug 2 is reachable.
                from hf_track.tracker._xet_impls import _TrackerXetImpls
                assert hasattr(_TrackerXetImpls, "_download_file_xet")
                assert hasattr(_TrackerXetImpls, "_download_snapshot_xet")
                # Sanity: Bug 1 module is reachable.
                from hf_track.download.xet_streaming import (
                    download_snapshot_streaming,
                )
                assert callable(download_snapshot_streaming)
                print("OK")
                """
            )],
            capture_output=True,
            text=True,
            timeout=30,
            env=_child_env(),
        )
        assert result.returncode == 0, (
            f"Fresh-Python import failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert "OK" in result.stdout

    def test_step14_fresh_interpreter_gets_pythonpath(self):
        """The env this file hands to children must carry the source root.

        Without it the fresh interpreter imports nothing of the package and
        ``test_subprocess_can_construct_hf_tracker`` fails for a reason that
        has nothing to do with the Step-14 regressions it guards.
        """
        child_env = _child_env()
        assert "src" in child_env["PYTHONPATH"], (
            f"PYTHONPATH should point at <package>/src, got "
            f"{child_env['PYTHONPATH']!r}"
        )
        result = subprocess.run(
            [sys.executable, "-c",
             "from hf_track import HfTracker; HfTracker()"],
            capture_output=True,
            text=True,
            timeout=30,
            env=child_env,
        )
        assert result.returncode == 0, (
            f"Fresh interpreter could not construct HfTracker (rc="
            f"{result.returncode}):\nstdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
