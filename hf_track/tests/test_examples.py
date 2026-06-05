"""Smoke tests for example scripts verifying they are importable, structurally
correct, and use only the current public API (no removed/deprecated symbols).

Step 12 of the modular-refactor plan
(docs/plans/2026-06-04-modular-refactor.md).

This test file is the single source of truth for "is the examples/
directory healthy?". It runs four kinds of checks against every
``*.py`` file in the examples directory:

  1. *Import check* — the module loads without raising (catches
     missing imports, syntax errors, etc.).
  2. *Public-API check* — every example exposes the conventional
     ``parse_args()`` and ``main()`` entry points used by the
     project README and the web-app example.
  3. *No-deprecated-API check* — the AST must not reference any
     of the symbols that were removed in Step 8.5 (XetSession,
     download_file_with_xet_session, etc.). This is a hard
     contract: if an example imports a removed name, the import
     would actually fail at step 1, so this check is mostly
     about catching *re-imports* of the same names after they
     reappear in the codebase.
  4. *Class-export check* — the two examples that define a
     helper class (``SmoothTicker`` for the smooth ticker, and
     ``MemoryWatcher`` for the streaming example) must still
     expose it, because their ``--watch-mem`` / ``--smooth``
     CLI flags depend on it.

If a new example is added, the discovery loop picks it up
automatically (as long as it follows the parse_args/main
convention).
"""

from __future__ import annotations

import ast
from importlib import util
from pathlib import Path
from typing import Iterable

import pytest


# Import from project root
# These examples live outside the src/ tree, so import via spec_from_file_location.
EXAMPLES_ROOT = Path(__file__).parents[1] / "examples"


def _import_example(name: str):
    """Import an example by filename; return the loaded module."""
    spec = util.spec_from_file_location(name, EXAMPLES_ROOT / f"{name}.py")
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _all_example_names() -> list[str]:
    """Every .py file in examples/, sorted, excluding private (_*) files."""
    return sorted(
        p.stem
        for p in EXAMPLES_ROOT.glob("*.py")
        if not p.name.startswith("_")
    )


# Examples that follow the CLI shape (must expose ``parse_args`` + ``main``).
# Library-style examples (formatting helpers, display classes) are NOT
# required to have these — they expose a different public surface.
CLI_EXAMPLES: frozenset[str] = frozenset(
    name
    for name in _all_example_names()
    if name.startswith("download_")
)

# Examples that are *library* style (formatting helpers, reusable
# classes) — they must expose a documented public surface instead of
# a CLI entry point.
LIBRARY_EXAMPLES: frozenset[str] = frozenset(
    name
    for name in _all_example_names()
    if name not in CLI_EXAMPLES
)


# Symbols that were removed in Step 8.5 ("remove all deprecated code").
# If any example references one of these, the test fails. The
# corresponding top-level import would also fail at module load, so
# this is a belt-and-braces check.
REMOVED_SYMBOLS: tuple[str, ...] = (
    "XetSession",
    "download_file_with_xet_session",
    "download_snapshot_with_xet_session",
    "_xet_session_download_worker",
    "_xet_session_snapshot_worker",
    "XetSessionProvider",
    "has_xet_session",
)

# Per-example class exports (mapping example -> expected class name).
# These are classes that an example defines and exposes for its
# own CLI flags to use.
EXAMPLE_CLASSES: dict[str, str] = {
    "download_xet_inprocess_smooth": "SmoothTicker",
    "download_xet_streaming": "MemoryWatcher",
}


# =============================================================================
# Discovery
# =============================================================================


def test_examples_root_exists() -> None:
    """The examples/ directory must exist and contain at least one .py file."""
    assert EXAMPLES_ROOT.is_dir(), f"{EXAMPLES_ROOT} is not a directory"
    examples = _all_example_names()
    assert len(examples) > 0, f"No example scripts found in {EXAMPLES_ROOT}"


# =============================================================================
# Per-example import + structural tests (parametrized)
# =============================================================================


@pytest.mark.parametrize("name", _all_example_names())
def test_example_imports(name: str) -> None:
    """The example module must load without raising.

    Catches syntax errors, missing imports, circular imports.
    """
    mod = _import_example(name)
    assert mod is not None


@pytest.mark.parametrize("name", sorted(CLI_EXAMPLES))
def test_example_has_parse_args(name: str) -> None:
    """Every CLI example exposes ``parse_args()`` for CLI integration.

    Only the CLI-style examples are required to have this — library
    examples (formatting helpers, etc.) follow a different shape.

    The convention is established in the project README and is
    used by the example-launcher in the docs/guides/integration-guide.md.
    """
    mod = _import_example(name)
    fn = getattr(mod, "parse_args", None)
    assert callable(fn), (
        f"{name}.py is missing a parse_args() function "
        f"(required for the standard CLI shape)"
    )


@pytest.mark.parametrize("name", sorted(CLI_EXAMPLES))
def test_example_has_main(name: str) -> None:
    """Every CLI example exposes ``main()`` as the script entry point.

    Same rationale as ``parse_args``.
    """
    mod = _import_example(name)
    fn = getattr(mod, "main", None)
    assert callable(fn), (
        f"{name}.py is missing a main() function "
        f"(required for the standard CLI shape)"
    )


@pytest.mark.parametrize("name", _all_example_names())
def test_example_no_removed_symbols(name: str) -> None:
    """The example must not reference any symbol removed in Step 8.5.

    This is a *regression* test: if a removed symbol re-appears in
    the codebase (e.g. as part of a buggy re-export), this test
    catches it even before the example is run.

    The check scans all Name/Attribute references AND all string
    constants — the latter matters because a sneaky reference like
    ``__import__("hf_xet").XetSession`` or ``_FAKE = "XetSession"``
    would otherwise slip through.
    """
    src_path = EXAMPLES_ROOT / f"{name}.py"
    src = src_path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Collect every Name, Attribute, and string-Constant reference
    # from the module's body.
    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            # Only the rightmost attr name, plus a dotted form for
            # `from x.y import z as w` we can detect via ast.alias
            referenced.add(node.attr)
        elif isinstance(node, ast.alias):
            referenced.add(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # String literals might encode an attribute or module name.
            value = node.value
            if value:
                referenced.add(value)
                # Also add the last dotted segment.
                referenced.add(value.rsplit(".", 1)[-1])

    bad = sorted(set(REMOVED_SYMBOLS) & referenced)
    assert not bad, (
        f"{name}.py references removed symbols: {bad}\n"
        f"Step 8.5 removed these from the public API; examples must "
        f"be updated to use the current API."
    )


@pytest.mark.parametrize(
    "name,class_name",
    list(EXAMPLE_CLASSES.items()),
    ids=list(EXAMPLE_CLASSES.keys()),
)
def test_example_exposes_documented_class(
    name: str, class_name: str
) -> None:
    """Examples that define a helper class must still expose it.

    Specifically:
      * download_xet_inprocess_smooth.SmoothTicker
      * download_xet_streaming.MemoryWatcher

    These are referenced by the example's ``--smooth`` /
    ``--watch-mem`` CLI flags; if the class is renamed or removed
    without updating the example, the CLI breaks.
    """
    mod = _import_example(name)
    cls = getattr(mod, class_name, None)
    assert cls is not None, (
        f"{name}.py is missing the documented class {class_name!r}. "
        f"If you renamed it, update CLI references in the same file."
    )
    assert callable(cls), (
        f"{name}.{class_name} should be callable, got {type(cls).__name__}"
    )


# =============================================================================
# Stream-API guard (kept from the pre-Step-12 test_examples.py)
# =============================================================================


def test_download_xet_streaming_uses_subprocess_api() -> None:
    """The streaming example must delegate to the public subprocess-based API.

    This guards against a regression where someone re-introduces the
    in-process ``XetSession().download_stream()`` loop directly. The
    in-process path is broken (no mid-stream visibility, the C extension
    holds the GIL, kill of an in-flight Rust fetch is impossible).

    We analyze the AST of executable code only, ignoring the docstring
    (which legitimately mentions the API for context).
    """
    mod = _import_example("download_xet_streaming")
    src = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Collect every Name and Attribute access from the module's body.
    # We walk all statements + function bodies recursively.
    def _walk(node: ast.AST) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                # Flatten the call into a string like "tracker.download_snapshot_streaming"
                parts: list[str] = []
                cur: ast.AST = child.func
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                full = ".".join(reversed(parts))
                names.append(full)
            elif isinstance(child, ast.Attribute):
                parts = []
                cur = child
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                names.append(".".join(reversed(parts)))

    names: list[str] = []
    for stmt in tree.body:
        _walk(stmt)

    # Public API call must be present somewhere in the executable code.
    assert any("download_snapshot_streaming" in n for n in names), (
        "Example must call HfTracker.download_snapshot_streaming() in its "
        f"executable code. Found references: {names}"
    )

    # The in-process API from the previous (broken) version must be gone.
    forbidden_substrings = (
        "XetSession",
        ".download_stream(",
        "new_download_stream_group",
    )
    for token in forbidden_substrings:
        offenders = [n for n in names if token in n]
        assert not offenders, (
            f"Example must not use the in-process streaming API ({token!r}); "
            f"offending references in executable code: {offenders}"
        )


# =============================================================================
# progress_bar.py detailed structure (kept from the pre-Step-12 test)
# =============================================================================


def test_progress_bar_formatting_helpers() -> None:
    """The progress-bar example exposes the documented formatting helpers.

    These are reused by the web-app example via the public formatting
    surface; if the names change, the web-app test breaks.
    """
    mod = _import_example("progress_bar")
    assert hasattr(mod, "ConsoleProgressDisplay")
    assert callable(mod.ConsoleProgressDisplay)
    for name in (
        "render_bar",
        "render_progress_line",
        "format_bytes",
        "format_speed",
        "format_eta",
    ):
        assert callable(getattr(mod, name)), (
            f"progress_bar.py is missing the {name!r} helper"
        )
