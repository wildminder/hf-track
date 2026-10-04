"""``hf_track.__version__`` and ``pyproject.toml`` cannot disagree (IMP-018).

The version used to be declared twice — ``0.2.4`` in
``pyproject.toml`` and ``0.1.0`` in ``src/hf_track/__init__.py`` — so
``pip show hf-track`` and ``hf_track.__version__`` answered different
questions with different numbers. ``__init__.py`` now reads the installed
distribution metadata and ``pyproject.toml`` is the only literal.

These tests read the files as data, so they hold even when the package
cannot be imported.
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import re

import pytest

import hf_track

PACKAGE_ROOT = pathlib.Path(hf_track.__file__).resolve().parents[2]
PYPROJECT = PACKAGE_ROOT / "pyproject.toml"
INIT_SRC = pathlib.Path(hf_track.__file__).resolve().parent / "__init__.py"


def _load_pyproject() -> dict:
    tomllib = pytest.importorskip(
        "tomllib", reason="tomllib (3.11+) is needed to read pyproject.toml"
    )
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)


class TestVersionConsistency:
    """One declaration in the manifest, one derived value in the package."""

    def test_version_matches_distribution_metadata(self):
        """``__version__`` is whatever the installed distribution says.

        Skipped on a bare checkout: with ``pythonpath = ["src"]`` the
        package imports without ever being installed, and there is then
        no distribution metadata to compare against. The fallback is
        covered by ``test_bare_checkout_fallback_is_a_literal`` instead.
        """
        try:
            installed = importlib.metadata.version("hf-track")
        except importlib.metadata.PackageNotFoundError:
            pytest.skip(
                "hf-track is not installed as a distribution, so there is no "
                "metadata to compare __version__ against"
            )
        assert hf_track.__version__ == installed

    def test_pyproject_version_is_the_only_literal(self):
        """No hard-coded version anywhere in ``src/hf_track`` but the fallback.

        The manifest declares the version once; ``__init__.py`` derives it.
        The one permitted literal is the ``PackageNotFoundError`` fallback,
        which is not a version claim — it says "unknown", which is false
        of no released version.
        """
        declared = _load_pyproject()["project"]["version"]
        assert declared == "0.2.4"

        init_src = INIT_SRC.read_text(encoding="utf-8")
        literals = re.findall(r'__version__\s*=\s*"([^"]*)"', init_src)
        assert literals == ["0.0.0+unknown"], (
            "src/hf_track/__init__.py must not declare a version literal "
            f"other than the not-installed fallback; found {literals}. The "
            f"declared version is {declared} (pyproject.toml)."
        )

    def test_declared_version_is_the_one_manifest_carries(self):
        """``[project] version`` exists, is a string, and has no build suffix."""
        declared = _load_pyproject()["project"]["version"]
        assert isinstance(declared, str)
        assert re.fullmatch(r"\d+\.\d+\.\d+", declared), (
            f"unexpected version shape {declared!r}"
        )

    def test_bare_checkout_fallback_is_a_literal(self):
        """The fallback string is deliberately not a real version."""
        init_src = INIT_SRC.read_text(encoding="utf-8")
        assert "0.0.0+unknown" in init_src
        # ``+`` marks it as a local/dev version, so no packaging tool can
        # mistake it for a release.
        assert hf_track.__version__ != "0.0.0+unknown" or not importlib.metadata.distribution(  # noqa: E501
            "hf-track"
        )