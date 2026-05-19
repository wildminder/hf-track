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

