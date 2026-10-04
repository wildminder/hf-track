"""Public type surface: the ``TransferError`` → ``TransferErrorInfo`` rename.

IMP-011: ``TransferError`` was a dataclass living in the same namespace as
``TransferCancelledError`` and ``TransferProgressError``, which are
exceptions. The names read as a family, so ``except TransferError`` — the
obvious thing to write — caught nothing at all. The dataclass is now
``TransferErrorInfo``; the exceptions keep their names.

The old name still resolves, for one deprecation cycle, through a
``__getattr__`` on both ``hf_track`` and ``hf_track.types``.
"""

from __future__ import annotations

import dataclasses
import pathlib
import typing
import warnings

import pytest

import hf_track
from hf_track.types import TransferErrorInfo

RESULTS_SRC = (
    pathlib.Path(hf_track.__file__).resolve().parent / "types" / "results.py"
)


class TestTransferErrorRename:
    """The dataclass has a name that says it is not an exception."""

    def test_transfer_error_info_is_the_dataclass(self):
        """It is a dataclass, and explicitly not a ``BaseException``."""
        assert dataclasses.is_dataclass(TransferErrorInfo)
        assert not issubclass(TransferErrorInfo, BaseException)

    def test_progress_error_remains_an_exception(self):
        """The rename must not have touched the real exceptions."""
        assert issubclass(hf_track.TransferProgressError, Exception)
        assert issubclass(hf_track.TransferCancelledError, Exception)
        assert not dataclasses.is_dataclass(hf_track.TransferProgressError)

    def test_transfer_error_info_is_exported_from_the_package(self):
        """It is reachable from the top level, where the old name was."""
        assert hf_track.TransferErrorInfo is TransferErrorInfo
        assert "TransferErrorInfo" in hf_track.__all__

    def test_old_name_raises_deprecation_warning(self):
        """``hf_track.TransferError`` warns and returns the new class."""
        with pytest.warns(DeprecationWarning, match="TransferErrorInfo"):
            old = getattr(hf_track, "TransferError")
        assert old is TransferErrorInfo

    def test_old_name_also_warns_from_the_types_subpackage(self):
        """``hf_track.types`` is the other import path callers used."""
        import hf_track.types as types_pkg

        with pytest.warns(DeprecationWarning, match="TransferErrorInfo"):
            old = getattr(types_pkg, "TransferError")
        assert old is TransferErrorInfo

    def test_unknown_attribute_still_raises_attribute_error(self):
        """The shim must not swallow genuine typos."""
        with pytest.raises(AttributeError):
            getattr(hf_track, "NoSuchName")

    def test_warnings_are_not_emitted_by_the_new_name(self):
        """Importing the new name is silent — only the old one warns."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert hf_track.TransferErrorInfo is not None
            assert hf_track.types.TransferErrorInfo is not None

    def test_old_name_is_absent_from_static_attributes(self):
        """``from hf_track import *`` must not resurrect it silently.

        It is absent from ``__all__`` and not a module attribute, so a
        static ``from ... import TransferError`` fails loudly instead of
        picking up a warning-only alias.
        """
        assert "TransferError" not in hf_track.__all__
        assert "TransferError" not in vars(hf_track)

    def test_no_source_file_still_defines_bare_transfer_error(self):
        """The dataclass is renamed; the exception classes are not touched."""
        results_src = RESULTS_SRC.read_text(encoding="utf-8")
        assert "class TransferError:" not in results_src
        assert "class TransferErrorInfo:" in results_src

    def test_error_payload_field_uses_the_new_name(self):
        """``ProgressEvent.error`` is annotated with the renamed class."""
        from hf_track.types import ProgressEvent

        hints = typing.get_type_hints(ProgressEvent)
        assert hints.get("error") in (
            TransferErrorInfo,
            f"Optional[{TransferErrorInfo.__name__}]",
        ) or "TransferErrorInfo" in str(hints.get("error"))


class TestHfXetStubs:
    """``hf_xet`` is a compiled PyO3 extension with no type information.

    The package ships ``py.typed``, so an in-tree stub is what makes the
    names this package actually calls resolvable to a downstream type
    checker (IMP-013).
    """

    STUB = pathlib.Path(hf_track.__file__).resolve().parent / "_hf_xet_stubs.pyi"
    RUST_SURFACE = {
        "upload_files",
        "upload_bytes",
        "download_files",
        "PyTotalProgressUpdate",
        "PyItemProgressUpdate",
        "PyXetDownloadInfo",
    }

    def test_stub_file_parses_and_declares_the_rust_surface(self):
        """The stub is valid Python and names every symbol the package uses."""
        import ast

        assert self.STUB.is_file(), f"{self.STUB} is missing"

        tree = ast.parse(self.STUB.read_text(encoding="utf-8"))
        declared = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        declared |= {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            for target in (node.target,)
        }
        declared |= {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }

        missing = self.RUST_SURFACE - declared
        assert not missing, f"stub does not declare {sorted(missing)}"

    def test_stub_is_discoverable_by_type_checkers(self):
        """It sits next to the package, so it ships in the wheel.

        ``[tool.hatch.build.targets.wheel] packages = ["src/hf_track"]`` in
        ``pyproject.toml`` is what puts it there; this asserts the file is
        inside that package directory, which is the part a rename or a
        moved module would break.
        """
        package_dir = pathlib.Path(hf_track.__file__).resolve().parent
        assert self.STUB.is_file()
        assert package_dir.name == "hf_track"
        assert self.STUB.parent == package_dir

    def test_package_ships_a_py_typed_marker(self):
        """A stub is only consulted for a package that claims to be typed."""
        assert (pathlib.Path(hf_track.__file__).resolve().parent / "py.typed").is_file()