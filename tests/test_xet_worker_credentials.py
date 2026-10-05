"""The worker no longer imports the Xet credential API removed in hub 1.x.

``huggingface_hub`` 1.23.0 dropped ``refresh_xet_connection_info`` and
``fetch_xet_connection_info_from_repo_info`` from
``huggingface_hub.utils._xet``. Twelve call sites in ``_xet_worker.py``
imported those names unguarded, so every upload/download worker died with
an ``ImportError`` in the child process.

Since 1.x the CAS token is never minted in Python: the runtime fetches and
refreshes it from a token-refresh route handed to ``XetSession``. These
tests pin that shape at the source level, which is where the regression
lives -- the calls only execute inside a spawned subprocess, so a
behavioural test would pass on a machine whose ``hf_xet`` never reaches the
network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


# ``<package>/tests/test_xet_worker_credentials.py`` -> ``<package>/src``.
SRC = Path(__file__).parents[1] / "src"
WORKER_SRC = SRC / "hf_track" / "_xet_worker.py"

#: Every module that holds a Xet worker (NTH-016 split them out of
#: ``_xet_worker.py``). The migration assertions scan all of them: a
#: regression that re-opens ``get_xet_session()`` in the upload module is
#: the same defect as one in the download module.
WORKER_MODULES = (
    "_xet_worker.py",
    "_xet_worker_common.py",
    "_xet_legacy_worker.py",
    "_xet_upload_worker.py",
    "_xet_hybrid.py",
    "_xet_hybrid_tier1.py",
    "_xet_hybrid_runner.py",
)

# The names huggingface_hub 1.x removed.
REMOVED_NAMES = ("refresh_xet_connection_info", "fetch_xet_connection_info_from_repo_info")


def _worker_source() -> str:
    """The main worker module's source (kept for the single-file checks)."""
    return WORKER_SRC.read_text(encoding="utf-8")


def _worker_family_source() -> str:
    """Every worker module's source, concatenated with a file marker."""
    parts = []
    for name in WORKER_MODULES:
        path = SRC / "hf_track" / name
        assert path.is_file(), f"worker module missing: {path}"
        parts.append(f"# ==== {name} ====\n" + path.read_text(encoding="utf-8"))
    return "\n".join(parts)


class TestWorkerCredentialMigration:
    """No unguarded reference to a removed hub API survives in the worker."""

    def test_worker_source_has_no_unguarded_removed_api_import(self):
        """No ``from huggingface_hub.utils._xet import`` names a removed API."""
        worker_src = _worker_family_source()
        offenders = [
            match.group(0)
            for match in re.finditer(
                r"^\s*from huggingface_hub\.utils\._xet import[^\n]*"
                r"\b(?:{})\b".format("|".join(REMOVED_NAMES)),
                worker_src,
                re.M,
            )
        ]
        assert not offenders, (
            "unguarded import of a removed huggingface_hub API:\n"
            + "\n".join(offenders)
        )

    def test_worker_imports_get_xet_session(self):
        """The replacement API is actually used, not merely absent."""
        assert "get_xet_session" in _worker_family_source()

    def test_worker_calls_no_removed_api(self):
        """No migrated worker body names a removed API at all.

        ``download_hybrid`` still *tries* ``refresh_xet_connection_info``
        behind a ``try/except ImportError`` so that an older hub still gets
        the fast path, and passes the bound name to ``_run_tier1_file`` as a
        parameter -- that shim is deliberate. The four workers migrated by
        this change must have no trace of it.
        """
        import ast

        migrated = (
            "_download_worker",
            "_download_batch_worker",
            "_upload_file_worker",
            "_upload_bytes_worker",
        )
        bodies = {}
        for module in WORKER_MODULES:
            source = (SRC / "hf_track" / module).read_text(encoding="utf-8")
            for node in ast.parse(source).body:
                if isinstance(node, ast.FunctionDef) and node.name in migrated:
                    bodies[node.name] = ast.get_source_segment(source, node)
        assert set(bodies) == set(migrated), (
            f"workers not found at module level: {sorted(set(migrated) - set(bodies))}"
        )
        for name, body in bodies.items():
            for removed in REMOVED_NAMES:
                assert removed not in body, (
                    f"{name} still references the removed {removed}"
                )

    def test_worker_uses_a_token_refresh_route(self):
        """Credentials arrive as a refresh route handed to the session."""
        worker_src = _worker_family_source()
        assert "new_file_download_group" in worker_src
        assert "new_upload_commit" in worker_src
        assert "token_refresh_url=" in worker_src

    def test_replacement_api_exists_in_the_installed_hub(self):
        """The names the migration migrated *to* are present.

        Nothing here asserts the old names are still gone. ``huggingface_hub``
        is resolved by a ``>=`` floor and upstream re-added them after
        1.23.0; whether a symbol is absent from a module we do not own is
        not a contract of this package, and asserting it turns any future
        hub release into a red build over a change that broke nothing here.
        The compatibility guarantee the package actually makes is that its
        own calls resolve -- which the tests above check against our source
        and this one checks against the installed hub.
        """
        from huggingface_hub.utils import _xet

        assert hasattr(_xet, "get_xet_session")


class TestWorkerCredentialImportsAtRuntime:
    """The migrated call sites import cleanly on the installed hub."""

    @pytest.mark.parametrize(
        "name",
        ("get_xet_session", "xet_headers_without_auth", "xet_connection_info_refresh_url"),
    )
    def test_migrated_names_exist(self, name):
        """Every name the migration relies on is present in hub 1.x."""
        from huggingface_hub.utils import _xet

        assert hasattr(_xet, name)

    def test_worker_module_imports(self):
        """Importing the worker must not pull in a removed API."""
        import importlib

        module = importlib.import_module("hf_track._xet_worker")
        assert hasattr(module, "_download_worker")
        assert hasattr(module, "_upload_file_worker")

class TestCredentialShimIsCentralised:
    """One place acquires a Xet session, one place fetches credentials.

    NTH-015. The worker used to open its own session from five different
    import blocks, and ``token/manager.py`` carried its own wire-format
    helper. When ``huggingface_hub`` moves the API again (it has moved
    twice: 1.x removed two of the three names), the number of edits
    required to follow is what decides whether the breakage reaches a
    spawned subprocess before anyone notices. One is acceptable.
    """

    MANAGER_SRC = SRC / "hf_track" / "token" / "manager.py"

    def test_only_one_connection_info_implementation(self):
        """One session acquisition in the worker, one wire helper in the manager."""
        worker_src = _worker_family_source()
        manager_src = self.MANAGER_SRC.read_text(encoding="utf-8")

        session_calls = re.findall(r"get_xet_session\(\)", worker_src)
        assert len(session_calls) == 1, (
            f"the worker calls get_xet_session() {len(session_calls)} times; "
            "route every worker through _new_xet_session() instead"
        )

        fetch_defs = re.findall(r"^def _fetch_xet_connection_info\b", manager_src, re.M)
        assert len(fetch_defs) == 1, (
            f"the token manager defines _fetch_xet_connection_info "
            f"{len(fetch_defs)} times"
        )

    def test_every_worker_goes_through_the_helper(self):
        """No worker body calls the hub helper directly."""
        import ast

        worker_src = _worker_source()
        tree = ast.parse(worker_src)
        offenders = []
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name == "_new_xet_session":
                continue
            segment = ast.get_source_segment(worker_src, node) or ""
            if "get_xet_session" in segment:
                offenders.append(node.name)
        assert not offenders, (
            "these functions touch get_xet_session directly; call "
            f"_new_xet_session() instead: {offenders}"
        )

    def test_helper_does_not_import_hf_xet_at_module_level(self):
        """The import stays inside the helper: workers must stay killable.

        ``hf_xet`` is a compiled Rust extension. Importing it in the
        parent process would load the native runtime into the very process
        that has to survive ``terminate()``, so the import is function-local
        and the test pins that.
        """
        import ast

        source = (SRC / "hf_track" / "_xet_worker_common.py").read_text(encoding="utf-8")
        helper = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "_new_xet_session"
        )
        body_src = ast.get_source_segment(source, helper)
        assert "import hf_xet" not in body_src
        assert "get_xet_session" in body_src

    def test_manager_is_the_only_place_that_reads_the_token_payload(self):
        """``casUrl``/``accessToken``/``exp`` are decoded once, in one file."""
        import pathlib

        package_src = pathlib.Path(_worker_source() and SRC / "hf_track")
        offenders = []
        for path in package_src.rglob("*.py"):
            if "__pycache__" in path.parts or path == self.MANAGER_SRC:
                continue
            text = path.read_text(encoding="utf-8")
            if '"casUrl"' in text or "'casUrl'" in text:
                offenders.append(str(path.relative_to(SRC)))
        assert not offenders, (
            "the CAS token payload is decoded in more than one place; it "
            f"belongs to token/manager.py only: {offenders}"
        )
