"""Enforce a maximum line count on every source module.

Step 10 of the modular-refactor plan
(docs/plans/2026-06-04-modular-refactor.md).

The whole point of the refactor was to stop creating 28 KB+ god
modules that are hard to maintain. This test enforces that invariant
going forward: any source file that grows past its budget is reported
as a test failure.

Why line count (not byte count)
--------------------------------
Developers think in lines. A 500-line file is a "moderate" module;
a 1,500-line file is a "growing concern"; a 3,000-line file is a
god module that needs splitting.

Why one test that lists ALL violators
-------------------------------------
It's tempting to split this into a parametrize-per-file, but then
adding a new file would silently be allowed to grow forever. A single
test that scans ``hf_track/src/hf_track/`` and reports every file
that is over its budget gives a complete picture in one failure
message — and adding a new file automatically subjects it to the
default budget.

What is exempt
--------------
* ``__init__.py`` files — re-export shims, expected to be small.
* ``py.typed`` — empty marker file.
* Per-file overrides (in ``OVERRIDES``) for modules that legitimately
  need more room (e.g. the worker module).
"""

from __future__ import annotations

import os
import pathlib
from typing import Dict, List, Tuple

import pytest


# ── Configuration ─────────────────────────────────────────────────


def _find_source_root() -> pathlib.Path:
    """Locate the source root by walking upward from this test file.

    The test file lives at ``<package>/tests/test_module_size.py``,
    where ``<package>`` is the directory that contains
    ``pyproject.toml`` and ``src/``. We find the source root by
    looking for the first ancestor that contains a ``src/hf_track``
    directory with a ``__init__.py`` inside.

    This is robust to the test being moved within the package
    (e.g. into a sub-folder) and to running from any working
    directory.
    """
    start = pathlib.Path(__file__).resolve().parent
    for ancestor in [start, *start.parents]:
        candidate = ancestor / "src" / "hf_track"
        if (
            candidate.is_dir()
            and (candidate / "__init__.py").is_file()
        ):
            return candidate
    raise RuntimeError(
        f"Could not locate 'src/hf_track' from {__file__!r}"
    )


SOURCE_ROOT = _find_source_root()

# Default budget for any file without an explicit override.
DEFAULT_BUDGET = 500

# Per-file overrides. Keys are POSIX-style paths relative to
# SOURCE_ROOT (use forward slashes even on Windows for portability).
OVERRIDES: Dict[str, int] = {
    # The worker module consolidates all the subprocess workers. It
    # is allowed to be the largest source file in the package, but
    # not larger than the smallest original god module (callbacks.py
    # was ~1,400 lines, tracker.py was ~750). Bumped from 1500 to
    # 1550 after adding deprecation docstrings for the legacy
    # _download_worker path, then to 1600 after adding the
    # TranslatingQueue adapter (plan 2026-07-09, step 10: dict ->
    # ProgressEvent translation for the hybrid runner).
    "_xet_worker.py": 1600,
    # tqdm_download is the heaviest of the callback helpers (it
    # contains three progress-bar adapter classes). Still under the
    # original callbacks.py size of 1,400.
    "callbacks/tqdm_download.py": 700,
    # subprocess/runner is the only module that has both a multiprocessing
    # queue and a thread-based relay implementation; it is the most
    # complex single module left after the refactor. Bumped from 500
    # to 550 after adding _synthesize_result_if_missing (stall safety).
    "subprocess/runner.py": 550,
}

# Files that are exempt from the budget entirely.
EXEMPT_PATHS: set[str] = {
    # Re-export shims — expected to be small but not required to fit
    # the default budget. (We still want __init__.py to be small, but
    # the budget mechanism is wrong for a flat list of imports.)
    "hf_track/__init__.py",
    "hf_track/types/__init__.py",
    "hf_track/token/__init__.py",
    "hf_track/callbacks/__init__.py",
    "hf_track/subprocess/__init__.py",
    "hf_track/download/__init__.py",
    "hf_track/upload/__init__.py",
    "hf_track/tracker/__init__.py",
    "hf_track/integrations/__init__.py",
}


# ── Helpers ───────────────────────────────────────────────────────


def _all_source_files() -> List[pathlib.Path]:
    """Yield every .py file under SOURCE_ROOT, sorted by size (largest first)."""
    out: List[Tuple[int, pathlib.Path]] = []
    for p in SOURCE_ROOT.rglob("*.py"):
        rel = p.relative_to(SOURCE_ROOT).as_posix()
        if rel in EXEMPT_PATHS:
            continue
        with open(p, encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        out.append((line_count, p))
    out.sort(key=lambda x: -x[0])
    return [p for _, p in out]


def _budget_for(rel_posix: str) -> int:
    """Return the line budget for a given file (POSIX path)."""
    return OVERRIDES.get(rel_posix, DEFAULT_BUDGET)


# ── Tests ─────────────────────────────────────────────────────────


class TestModuleSize:
    """No source file may exceed its configured line budget."""

    def test_no_module_exceeds_budget(self):
        """Every .py file is at or below its budget.

        Reports the full sorted list of violators in one message so a
        maintainer can see all problems at once and prioritise the
        biggest ones first.
        """
        violators: List[Tuple[int, int, str]] = []
        # (line_count, budget, rel_path) sorted by (line_count - budget)
        for p in _all_source_files():
            rel = p.relative_to(SOURCE_ROOT).as_posix()
            budget = _budget_for(rel)
            with open(p, encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            if line_count > budget:
                violators.append((line_count, budget, rel))

        if not violators:
            return

        # Sort: largest excess first (most urgent to fix).
        violators.sort(key=lambda x: -(x[0] - x[1]))

        lines = [
            f"{len(violators)} source file(s) exceed their line budget:",
            "",
        ]
        for lc, b, rel in violators:
            excess = lc - b
            lines.append(
                f"  {rel}: {lc} lines (budget {b}, excess {excess})"
            )
        lines.append("")
        lines.append(
            "Either split the module or, if the size is justified, "
            "add an entry to OVERRIDES in tests/test_module_size.py."
        )
        pytest.fail("\n".join(lines))

    def test_every_source_file_appears_inventory(self):
        """Inventory check: every file in SOURCE_ROOT is enumerated.

        This is a cheap guard against silent breakage of the test
        (e.g. a path typo in SOURCE_ROOT would make every later test
        pass while the actual modules go un-enforced).
        """
        files = _all_source_files()
        assert len(files) > 0, (
            f"No .py files found under {SOURCE_ROOT} — is SOURCE_ROOT correct?"
        )

    def test_no_module_is_unreasonably_small(self):
        """A source file with 5 or fewer lines is almost certainly a mistake.

        Catches the 'I split too aggressively' failure mode: a module
        that has nothing in it (a one-liner) is just a re-export with
        extra steps.
        """
        TOO_SMALL = 5
        tiny: List[str] = []
        for p in _all_source_files():
            with open(p, encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            if line_count <= TOO_SMALL:
                rel = p.relative_to(SOURCE_ROOT).as_posix()
                tiny.append(f"  {rel}: {line_count} lines")
        assert not tiny, (
            "Source file(s) are suspiciously small (<=5 lines). "
            "Consider inlining or moving to __init__.py:\n"
            + "\n".join(tiny)
        )

    def test_largest_files_are_documented(self):
        """Largest files print as a warning-free report (always passes).

        This is a developer-friendly test: it prints a sorted list of
        the largest files so you can see the size distribution at a
        glance. Use ``pytest -s`` to see the output.
        """
        report: List[Tuple[int, int, str]] = []
        for p in _all_source_files():
            rel = p.relative_to(SOURCE_ROOT).as_posix()
            budget = _budget_for(rel)
            with open(p, encoding="utf-8") as f:
                lc = sum(1 for _ in f)
            report.append((lc, budget, rel))
        report.sort(key=lambda x: -x[0])

        # Print as a report (visible with -s, ignored otherwise).
        msg_lines = ["Largest source modules (lines / budget):"]
        for lc, b, rel in report[:10]:
            marker = " **" if lc > b else ""
            msg_lines.append(f"  {lc:5d} / {b:5d}  {rel}{marker}")
        # Use os.devnull if not in -s mode — pytest will still collect
        # the message but the test always passes.
        print("\n".join(msg_lines))
        assert True  # informational only
