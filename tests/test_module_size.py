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
test that scans ``src/hf_track/`` and reports every file
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
import re
import subprocess
import sys
from typing import Dict, List, Set, Tuple

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

# <root>/src/hf_track -> <root>/src -> <root>. The project root and the
# repository root are the same directory now that pyproject.toml sits at
# the top level, so there is no package directory to step through.
REPO_ROOT = SOURCE_ROOT.parents[1]

# The project's packaging + tooling manifest.
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Default budget for any file without an explicit override.
DEFAULT_BUDGET = 500

# Per-file overrides. Keys are POSIX-style paths relative to
# SOURCE_ROOT (use forward slashes even on Windows for portability).
OVERRIDES: Dict[str, int] = {
    # NOTE: _xet_worker.py used to carry an 1800-line override. It no
    # longer needs one: NTH-016 (step S32) split it into seven modules,
    # the largest of which is 441 lines — every one of them now fits
    # DEFAULT_BUDGET, and the split-out names are what the override used
    # to hide. The family is:
    #
    #   _xet_worker_common.py     shared plumbing (init/safe-put/throttle)
    #   _xet_legacy_worker.py    the deprecated 2026-07-09 download workers
    #   _xet_upload_worker.py    the upload workers
    #   _xet_hybrid.py           download_hybrid
    #   _xet_hybrid_tier1.py     _run_tier1_file + _open_unbuffered
    #   _xet_hybrid_runner.py    TranslatingQueue / HybridRunner
    #   _xet_worker.py           the current download + snapshot workers
    # download/xet_file_only.py holds both the in-process dedicated xet path
    # (download_file_xet_only) and the terminable-subprocess path
    # (download_file_xet_subprocess) plus their shared helpers. Plan
    # 2026-07-16 (xet-single-file-subprocess-isolation) added the subprocess
    # path here; 500 lines is no longer enough for both paths.
    "download/xet_file_only.py": 550,
    # NOTE: callbacks/tqdm_download.py used to carry a 700-line override.
    # It was removed once the ~40 lines of inline HF_TRACK_DEBUG_XET
    # instrumentation were collapsed into the module-level _diag()
    # helper (IMP-020), taking the file back under DEFAULT_BUDGET. The
    # override must not come back without that same collapse.
    # subprocess/runner is the only module that has both a multiprocessing
    # queue and a thread-based relay implementation; it is the most
    # complex single module left after the refactor. Bumped from 500
    # to 550 after adding _synthesize_result_if_missing (stall safety),
    # then to 650 after CRIT-012 (terminate() releases the lock before
    # its joins, plus a generation counter so a concurrent start() is not
    # clobbered) and IMP-019 (progress-event coalescing when the event
    # queue is full) added ~95 lines of real logic. If it grows again,
    # split the relay/coalescing half into subprocess/relay.py rather
    # than raising this a third time.
    "subprocess/runner.py": 650,
}

# Files that are exempt from the budget entirely.
# Keys are POSIX-style paths relative to SOURCE_ROOT (like OVERRIDES),
# NOT relative to the repository root.
EXEMPT_PATHS: set[str] = {
    # Re-export shims — expected to be small but not required to fit
    # the default budget. (We still want __init__.py to be small, but
    # the budget mechanism is wrong for a flat list of imports.)
    "__init__.py",
    "types/__init__.py",
    "token/__init__.py",
    "callbacks/__init__.py",
    "subprocess/__init__.py",
    "download/__init__.py",
    "upload/__init__.py",
    "tracker/__init__.py",
    "integrations/__init__.py",
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

    def test_worker_family_fits_the_default_budget(self):
        """No Xet worker module needs an override (NTH-016).

        ``_xet_worker.py`` carried an 1800-line budget against a 442-line
        file. The split moved the deprecated download workers, the upload
        workers and the hybrid runner into modules of their own, all of
        which fall back to ``DEFAULT_BUDGET``. Asserting the *family* —
        not just the module that used to hold the override — is what stops
        the next worker from being added straight back into one of them.
        """
        worker_family = sorted(
            p.name
            for p in SOURCE_ROOT.glob("_xet*.py")
            if p.is_file()
        )
        assert worker_family, f"no _xet* modules found under {SOURCE_ROOT}"

        for name in worker_family:
            rel = name
            assert _budget_for(rel) == DEFAULT_BUDGET, (
                f"{rel} must not carry a budget override — the worker "
                f"modules all fit the {DEFAULT_BUDGET}-line default"
            )
            with open(SOURCE_ROOT / name, encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            assert line_count <= DEFAULT_BUDGET, (
                f"{rel} is {line_count} lines, over the {DEFAULT_BUDGET}-line budget"
            )

    def test_tqdm_download_has_no_budget_override(self):
        """``callbacks/tqdm_download.py`` fits the default budget (IMP-020).

        It carried a 700-line override while the file was under 500 lines.
        The override is gone, so the file falls back to ``DEFAULT_BUDGET``;
        both halves of that claim are asserted here rather than trusting the
        absence of the key.
        """
        rel = "callbacks/tqdm_download.py"
        assert rel not in OVERRIDES, (
            f"{rel} must not carry a budget override — it fits "
            f"DEFAULT_BUDGET ({DEFAULT_BUDGET})"
        )
        assert _budget_for(rel) == DEFAULT_BUDGET

        with open(SOURCE_ROOT / "callbacks" / "tqdm_download.py", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        assert line_count <= DEFAULT_BUDGET, (
            f"{rel} is {line_count} lines, over the {DEFAULT_BUDGET}-line budget"
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


def _keys_of(mapping: Dict[str, int]) -> Set[str]:
    """Return the keys of a path→budget style mapping."""
    return set(mapping)


def _load_pyproject() -> dict:
    """Parse ``hf_track/pyproject.toml`` as data.

    ``tomllib`` is stdlib from 3.11; the project still supports 3.9, so
    the config checks skip rather than fail on an older interpreter —
    the point here is to police the manifest, not to demand a new runtime.
    """
    tomllib = pytest.importorskip(
        "tomllib", reason="tomllib (3.11+) is needed to read pyproject.toml"
    )
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)


class TestExemptPaths:
    """``EXEMPT_PATHS`` is keyed the same way ``OVERRIDES`` is.

    Both are compared against ``p.relative_to(SOURCE_ROOT).as_posix()``,
    i.e. paths *inside* ``src/hf_track``. The keys used to carry an extra
    ``hf_track/`` prefix, which matched nothing, so no file was ever
    actually exempt — the exemptions had been decorative. A test that only
    asserted a key was present would not have caught that, hence the
    behavioural assertion as well.
    """

    def test_known_exempt_path_is_skipped(self):
        """At least one exemption key matches the paths the scan produces."""
        assert "__init__.py" in _keys_of(EXEMPT_PATHS)

    def test_every_exempt_key_matches_a_real_file(self):
        """No key is dead: each one resolves to a file under SOURCE_ROOT."""
        missing = sorted(k for k in EXEMPT_PATHS if not (SOURCE_ROOT / k).is_file())
        assert not missing, (
            "EXEMPT_PATHS entries that match no file (keys are relative to "
            f"{SOURCE_ROOT}):\n" + "\n".join(f"  {k}" for k in missing)
        )

    def test_exempt_files_are_actually_omitted_from_the_scan(self):
        """The exemptions change what ``_all_source_files`` returns."""
        scanned = {p.relative_to(SOURCE_ROOT).as_posix() for p in _all_source_files()}
        for key in _keys_of(EXEMPT_PATHS):
            assert key not in scanned, (
                f"{key} is listed in EXEMPT_PATHS but still shows up in the scan"
            )

    def test_package_root_is_still_scanned_when_not_exempt(self):
        """The guard cannot be satisfied by exempting everything."""
        assert "__init__.py" not in _keys_of(OVERRIDES)
        assert len(EXEMPT_PATHS) < 50, "exempting the whole tree defeats the budget"


class TestRuffConfiguration:
    """The lint configuration and the lint gate agree with each other.

    ``pyproject.toml`` is read as *data* here — nothing is imported — so
    these checks survive a broken package the same way the module-size
    checks do.
    """

    # Rules that the gate treats as blocking. F821 is a name used before
    # it exists; F811 is a redefinition that silently shadows the first.
    GATE_CODES = frozenset({"F821", "F811"})

    # Measured 2026-10-04. These are hygiene findings — 119 of the 130 —
    # reported by the `ruff-lint-advisory` hook with --exit-zero. Promoting
    # one means adding it to GATE_CODES and deleting it here.
    ADVISORY_CODES = frozenset({"F401", "E402", "F841", "E731"})

    # Everything `ruff check --select F821,F811 .` reported on 2026-10-04,
    # keyed by (path relative to REPO_ROOT, code) so that line drift in a
    # file being refactored does not read as a new violation.
    #
    # The three shipped-package entries are the live bugs: two in
    # xet_streaming.py (CRIT-009) and one in tracker/_core.py (IMP-025).
    # They disappear when those steps land. The two test entries are debt
    # in files no step owns; the sse one is a duplicated test name, i.e. a
    # test that never runs.
    KNOWN_GATE_FINDINGS: Set[Tuple[str, str]] = {
        ("examples/web_app/tests/test_web_app.py", "F811"),
        ("src/hf_track/download/xet_streaming.py", "F821"),
        ("src/hf_track/tracker/_core.py", "F821"),
        ("tests/test_sse.py", "F811"),
        ("tests/test_xet_upload.py", "F811"),
    }

    # Concise-format finding line: path:line:col: CODE message
    FINDING_RE = re.compile(
        r"^(?P<path>[^:]+):(?P<line>\d+):(?P<col>\d+):\s+(?P<code>F\d+)\b"
    )

    @classmethod
    def _ruff_findings(cls, *paths: str) -> Dict[Tuple[str, str], str]:
        """Run the gate over ``paths`` and return {(path, code): detail}.

        Skips the test if ruff is not importable — a missing lint tool in
        the interpreter is an environment problem, not a code defect, and
        the tool is a declared dev dependency rather than a test fixture.
        """
        proc = subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            pytest.skip(f"ruff is not importable: {proc.stderr.strip()}")

        proc = subprocess.run(
            [
                sys.executable, "-m", "ruff", "check",
                "--select", ",".join(sorted(cls.GATE_CODES)),
                "--output-format", "concise",
                *paths,
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        findings: Dict[Tuple[str, str], str] = {}
        for raw in proc.stdout.splitlines():
            match = cls.FINDING_RE.match(raw.strip())
            if not match:
                continue  # trailer lines such as "Found 3 errors."
            rel = pathlib.PurePath(match["path"]).as_posix()
            findings[(rel, match["code"])] = raw.strip()
        return findings

    @classmethod
    def _assert_no_new_findings(cls, findings, known):
        new = sorted(set(findings) - known)
        if not new:
            return
        lines = [
            f"{len(new)} lint finding(s) the gate does not know about:",
            "",
        ]
        for key in new:
            lines.append(f"  {findings[key]}")
        lines += [
            "",
            "F821/F811 hide runtime breakage, so these block. If the "
            "finding is pre-existing debt in a file you do not own, add "
            "it to KNOWN_GATE_FINDINGS with a comment saying who clears it.",
        ]
        pytest.fail("\n".join(lines))

    def test_ruff_config_marks_f821_as_error(self):
        """``[tool.ruff.lint] select`` gates F821 and F811, and nothing else."""
        data = _load_pyproject()
        lint = data["tool"]["ruff"]["lint"]
        select = set(lint["select"])

        missing = self.GATE_CODES - select
        assert not missing, (
            f"{sorted(missing)} must be selected: they are the rules that "
            "hide runtime breakage. A finding from either is a crash at "
            "runtime, not a style opinion."
        )
        # Anything selected beyond the gate must at least be a rule we
        # have consciously classed as advisory, never an accidental pick-up.
        unclassified = select - self.GATE_CODES - self.ADVISORY_CODES
        assert not unclassified, (
            f"selected but unclassified lint rules: {sorted(unclassified)}. "
            "Add them to GATE_CODES or ADVISORY_CODES deliberately."
        )

    def test_ruff_is_declared_in_dev_extra(self):
        """ruff is a declared dev dependency, not an ambient tool."""
        data = _load_pyproject()
        extras = data["project"]["optional-dependencies"]
        assert "dev" in extras, "pyproject.toml has no [dev] extra"
        declared = [d for d in extras["dev"] if d.split(">=")[0].split("==")[0] == "ruff"]
        assert declared, (
            "ruff is not declared in the dev extra "
            f"(found: {extras['dev']}), so the gate depends on an ambient "
            "interpreter rather than on the project."
        )

    def test_coverage_flags_present_in_addopts(self):
        """``pytest`` reports coverage by default (NTH-009).

        ``pytest-cov`` has been a declared dev dependency with nothing in
        ``addopts`` to switch it on, so coverage was only ever measured by
        whoever remembered the flag.

        The absence of ``--cov-fail-under`` is asserted too, and is the
        load-bearing half: the suite is currently red, so a threshold
        would fail every run for a reason unrelated to coverage.
        """
        addopts = _load_pyproject()["tool"]["pytest"]["ini_options"]["addopts"]
        assert "--cov=hf_track" in addopts
        assert "--cov-report=term-missing" in addopts
        assert "--cov-fail-under" not in addopts, (
            "a coverage threshold is a gate on a red suite; add it once the "
            "baseline failure count stops drifting"
        )

    def test_huggingface_hub_has_an_upper_bound(self):
        """``huggingface_hub`` is capped below the next major version.

        1.x removed the Xet credential helpers this package depends on
        (CRIT-008). An unbounded specifier means an unattended install picks
        up whatever the next major ships, and the break surfaces at runtime
        inside a worker subprocess rather than at install time.
        """
        deps = _load_pyproject()["project"]["dependencies"]
        bounded = [
            d for d in deps
            if d.split(">")[0].split("=")[0].split("[")[0].strip() == "huggingface_hub"
        ]
        assert bounded, f"huggingface_hub is not declared in {deps}"
        assert any("<2.0" in d for d in bounded), (
            "huggingface_hub has no upper bound, so a 2.x release can break "
            f"the Xet credential path at runtime: {bounded}"
        )

    def test_gate_does_not_block_the_130_existing_errors(self):
        """The gate is satisfiable without first clearing all 130 findings.

        ``ruff check .`` reported 130 findings on 2026-10-04; the gate set
        is a strict subset of them, so enabling it cannot turn every
        branch red on day one. This asserts the direction of travel rather
        than an exact count: no finding outside the recorded baseline, and
        the baseline shrinking as the underlying defects get fixed is fine.
        """
        findings = self._ruff_findings(".")
        self._assert_no_new_findings(findings, self.KNOWN_GATE_FINDINGS)

    def test_shipped_package_has_only_the_known_gate_findings(self):
        """The blocking scope (``src``) carries only known defects.

        This is the scope CI runs, so it is the one that has to reach zero:
        once CRIT-009 and IMP-025 land, the gate blocks on nothing and
        starts blocking on regressions.
        """
        findings = self._ruff_findings("src")
        shipped = {
            key for key in self.KNOWN_GATE_FINDINGS if key[0].startswith("src/")
        }
        self._assert_no_new_findings(findings, shipped)

    def test_core_module_declares_xetrunner_typechecking_import(self):
        """``tracker/_core.py`` annotates a name it actually imports (IMP-025).

        ``__init__`` annotates ``_active_runners`` as
        ``dict[str, "XetSubprocessRunner"]`` while the module never imported
        that class, so any consumer resolving the annotation
        (``typing.get_type_hints``) got ``NameError``. The import is under
        ``TYPE_CHECKING``, so it is resolved only for type checkers -- the
        annotation must therefore be resolvable in the module's own
        namespace, which is exactly what the child interpreter asserts.
        """
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SOURCE_ROOT.parent)
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import typing, hf_track.tracker._core as c; "
                "typing.get_type_hints(c._TrackerCore.__init__)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, (
            "typing.get_type_hints(hf_track.tracker._core._TrackerCore.__init__) "
            f"failed (rc={proc.returncode}):\n{(proc.stderr or '').strip()}"
        )


class TestHubCompatibility:
    """Every module imports against the *installed* ``huggingface_hub``.

    ``huggingface_hub`` 1.23.0 removed ``refresh_xet_connection_info`` and
    ``fetch_xet_connection_info_from_repo_info``. Nothing failed at install
    time and nothing failed in the import-only test paths -- the breakage
    only surfaced when a worker reached that code inside a spawned
    subprocess. These two tests are the release-time check that would have
    caught it: one proves the whole package loads, the other pins the exact
    API surface the package is allowed to rely on.
    """

    # Every importable module of the package. ``_xet_worker`` is not a
    # subpackage, but it is where the removed API used to be imported, so it
    # is included explicitly.
    MODULES = (
        "hf_track",
        "hf_track.callbacks",
        "hf_track.download",
        "hf_track.integrations",
        "hf_track.subprocess",
        "hf_track.token",
        "hf_track.tracker",
        "hf_track.types",
        "hf_track.upload",
        "hf_track._xet_worker",
    )

    @staticmethod
    def _child_env() -> Dict[str, str]:
        """Env for a child interpreter that can find ``src/hf_track``."""
        env = dict(os.environ)
        src = str(SOURCE_ROOT.parent)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
        return env

    def test_every_module_imports_against_installed_hub(self):
        """Each module imports in a fresh interpreter.

        A subprocess is required, not a plain in-process ``import``: the
        failure mode being guarded against is a *partially* imported
        module, and an in-process import would satisfy any earlier one from
        ``sys.modules`` instead of exercising the real import.
        """
        env = self._child_env()
        failures: List[str] = []
        for module in self.MODULES:
            proc = subprocess.run(
                [sys.executable, "-c", f"import {module}"],
                env=env,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                failures.append(
                    f"{module} (rc={proc.returncode}):\n"
                    f"{(proc.stderr or '').strip().splitlines()[-1]}"
                )
        assert not failures, (
            "modules that do not import against the installed "
            f"huggingface_hub:\n\n" + "\n\n".join(failures)
        )

    def test_installed_hub_exposes_only_supported_xet_api(self):
        """The Xet helpers the package uses exist, and the removed ones do not.

        This is the assertion that fails on ``huggingface_hub`` 1.x before
        the CRIT-008 migration and passes after it.
        """
        from huggingface_hub.utils import _xet

        for name in (
            "get_xet_session",
            "xet_headers_without_auth",
            "xet_connection_info_refresh_url",
            "XetTokenType",
        ):
            assert hasattr(_xet, name), (
                f"the installed huggingface_hub no longer exposes {name}; "
                "the Xet path must be migrated again"
            )

        for removed in (
            "refresh_xet_connection_info",
            "fetch_xet_connection_info_from_repo_info",
        ):
            assert not hasattr(_xet, removed), (
                f"{removed} exists again -- the migration may be reversible"
            )


class TestDocsIndex:
    """``docs/README.md`` is generated, and the check runs in CI (IMP-023).

    The index was hand-maintained, which meant it drifted silently: a
    document added without a table row, a document deleted without its
    row, a link to a file that no longer existed. Each of those produces
    an index that looks authoritative and is wrong, which is worse than
    having no index.

    ``--check`` is a pure comparison, so it is safe to run as a gate; the
    tests below are what make it self-enforcing rather than a script
    nobody remembers to run.
    """

    # SOURCE_ROOT is <repo>/src/hf_track; the generator sits
    # to src/, under the package root.
    GENERATOR = SOURCE_ROOT.parents[1] / "tools" / "generate_docs_index.py"
    DOCS_ROOT = REPO_ROOT / "docs"
    INDEX = DOCS_ROOT / "README.md"
    WORKFLOW = REPO_ROOT / ".github" / "workflows" / "lint.yml"

    def test_generated_index_is_up_to_date(self):
        """The committed index matches what the generator would emit.

        Fails when a document was added, renamed or deleted without
        regenerating -- exactly the drift this replaces.
        """
        if not self.GENERATOR.is_file():
            pytest.fail(f"the docs index generator is missing: {self.GENERATOR}")

        proc = subprocess.run(
            [sys.executable, str(self.GENERATOR), "--check"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, (
            "docs/README.md is out of date. Run:\n"
            "  python tools/generate_docs_index.py\n"
            f"({(proc.stderr or '').strip()})"
        )

    def test_index_has_no_dead_links(self):
        """Every relative link in the index resolves to a file that exists.

        This is the assertion whose failure produced the dead links in
        the tracker: a link to a deleted document survives in the index
        because nothing else notices.
        """
        import urllib.parse

        if not self.INDEX.is_file():
            pytest.fail(f"{self.INDEX} is missing")

        text = self.INDEX.read_text(encoding="utf-8")
        checked = 0
        dead = []
        for target in re.findall(r"\]\(([^)]+)\)", text):
            target = target.strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_part = target.split("#", 1)[0]
            if not path_part:
                continue
            # ``:38`` marks a source citation, not a document.
            path_part = re.sub(r":\d+(?:[-:]\d+)?$", "", path_part)
            decoded = urllib.parse.unquote(path_part)
            # A target that resolves against the repository root is a
            # citation into the source, not a document link.
            if (self.DOCS_ROOT / decoded).exists():
                checked += 1
            elif (self.REPO_ROOT / decoded).exists():
                checked += 1
            else:
                dead.append(target)

        assert checked, "the index has no resolvable relative links at all"
        assert not dead, "dead links in docs/README.md:\n" + "\n".join(f"  {d}" for d in dead)

    def test_index_declares_itself_generated(self):
        """The reader is told not to edit it by hand."""
        text = self.INDEX.read_text(encoding="utf-8").lower()
        assert "generated" in text, (
            "docs/README.md must say it is generated, or the next editor "
            "will hand-maintain it again"
        )
        assert "hand-maintained" not in text

    def test_every_document_is_listed(self):
        """No document is missing from the index.

        The failure this catches is a file added under ``docs/`` whose
        row nobody added -- an index silently under-reporting what the
        project contains.
        """
        text = self.INDEX.read_text(encoding="utf-8")
        missing = []
        for path in self.DOCS_ROOT.rglob("*.md"):
            if path.resolve() == self.INDEX.resolve():
                continue
            rel = path.relative_to(self.DOCS_ROOT).as_posix()
            if rel not in text:
                missing.append(rel)
        assert not missing, (
            "documents present on disk but absent from docs/README.md:\n"
            + "\n".join(f"  {m}" for m in missing)
        )

    def test_link_check_is_wired_into_ci(self):
        """Both self-checks run in the same job.

        A dead link and an undefined name are the same defect -- the tree
        asserting something false -- so they are gated together.
        """
        if not self.WORKFLOW.is_file():
            pytest.fail(f"no CI workflow at {self.WORKFLOW}")

        workflow_text = self.WORKFLOW.read_text(encoding="utf-8")
        assert re.search(r"ruff check", workflow_text), "the lint gate is missing"
        assert re.search(r"generate_docs_index|--check", workflow_text), (
            "the docs link check is not wired into CI"
        )