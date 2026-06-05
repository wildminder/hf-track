"""Smoke tests for example scripts verifying they are importable and structurally correct."""
from __future__ import annotations

from importlib import util
from pathlib import Path


# Import from project root
# These examples live outside the src/ tree, so import via spec_from_file_location.
EXAMPLES_ROOT = Path(__file__).parents[1] / "examples"


def _import_example(name: str):
    spec = util.spec_from_file_location(name, EXAMPLES_ROOT / f"{name}.py")
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Download-file example
# =============================================================================


def test_download_file_importable_and_runnable() -> None:
    mod = _import_example("download_file")
    assert callable(mod.parse_args)
    assert callable(mod.main)


# =============================================================================
# Download-repo example
# =============================================================================


def test_download_repo_importable_and_runnable() -> None:
    mod = _import_example("download_repo")
    assert callable(mod.parse_args)
    assert callable(mod.main)


# =============================================================================
# Download-xet-streaming example (subprocess-isolated streaming)
# =============================================================================


def test_download_xet_streaming_importable_and_runnable() -> None:
    """Smoke-test the new subprocess-streaming example.

    The example is the recommended path for downloading large xet-stored
    repos: it must use ``HfTracker.download_snapshot_streaming()`` (not
    the broken in-process path) so that mid-stream disk visibility and
    clean cancellation are guaranteed.
    """
    mod = _import_example("download_xet_streaming")
    assert callable(mod.parse_args)
    assert callable(mod.main)
    # MemoryWatcher should still be exposed so --watch-mem keeps working.
    assert callable(mod.MemoryWatcher)


def test_download_xet_streaming_uses_subprocess_api() -> None:
    """The example must delegate to the public subprocess-based API.

    This guards against a regression where someone re-introduces the
    in-process ``XetSession().download_stream()`` loop directly. The
    in-process path is broken (no mid-stream visibility, the C extension
    holds the GIL, kill of an in-flight Rust fetch is impossible).

    We analyze the AST of executable code only, ignoring the docstring
    (which legitimately mentions the API for context).
    """
    import ast

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
# Progress-bar example
# =============================================================================


def test_progress_bar_importable_and_runnable() -> None:
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
        assert callable(getattr(mod, name))

