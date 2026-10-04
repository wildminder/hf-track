"""Type stubs for the ``hf_xet`` PyO3 extension (IMP-013).

``hf_xet`` is a compiled Rust extension: it ships no ``.pyi`` and no
docstrings a type checker can read. ``hf_track`` ships ``py.typed``, so
every downstream checker resolves names inside this package — and any
``hf_xet`` name that is not declared anywhere turns into an
``reportMissingTypeStubs`` error in *their* build, caused by *our*
dependency.

This file declares only the surface ``hf_track`` actually calls. It is a
``.pyi``, not an inline ``TYPE_CHECKING`` block, precisely so that the
information reaches consumers: an inline block is invisible outside the
module that declares it.

Every ``...`` here is a deliberate stub body. None of these stubs are
loaded at runtime — ``hf_xet`` is imported normally — so nothing in this
file can affect behaviour; a mistake costs a wrong type, never a
crash.
"""

import os
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

# The two progress-update records the Rust side pushes into the
# callbacks below. Both are plain data: the field names here are the ones
# ``callbacks/xet_callback.py`` reads with ``getattr(item_update, ...)``,
# so a rename on the Rust side shows up as a type error at our call site
# rather than as a silent ``AttributeError`` at transfer time.

class PyTotalProgressUpdate:
    """Aggregate progress across every item in a transfer."""

    bytes_completed: int
    total_bytes: int

class PyItemProgressUpdate:
    """Progress for one item within a transfer."""

    bytes_completed: int
    total_bytes: int
    filename: str

class PyXetDownloadInfo:
    """Per-file destination and size decided by ``hf_xet`` before download."""

    destination_path: str
    size: int
    file_hash: str
    xet_file_data: Any

# ``progress_callback`` is invoked as ``callback(total_update,
# item_updates)``. The library passes two callables; ``hf_track`` passes
# exactly one shape of each, spelled out in ``callbacks/xet_callback.py``.
ProgressCallback = Callable[[PyTotalProgressUpdate, Sequence[PyItemProgressUpdate]], None]

def upload_files(
    paths: Iterable[str],
    repo_id: str,
    repo_type: str,
    token: str,
    progress_callback: Optional[ProgressCallback] = ...,
    endpoint: Optional[str] = ...,
    revision: Optional[str] = ...,
    create_xet_files: bool = ...,
    xet_file_data: Any = ...,
) -> None:
    """Upload local files to a repo through the Xet path.

    Returns ``None``: per-file hashes are reported through the progress
    callback, not the return value.
    """

def upload_bytes(
    file_paths: Iterable[str],
    data: Iterable[bytes],
    repo_id: str,
    repo_type: str,
    token: str,
    progress_callback: Optional[ProgressCallback] = ...,
    endpoint: Optional[str] = ...,
    revision: Optional[str] = ...,
    create_xet_files: bool = ...,
    xet_file_data: Any = ...,
    num_bytes_threshold: Optional[int] = ...,
) -> None:
    """Upload in-memory bytes alongside their on-disk names.

    Raises ``ValueError`` when ``file_paths`` and ``data`` differ in
    length — the one precondition the Rust side checks before touching
    the network.
    """

def download_files(
    files: Sequence[PyXetDownloadInfo],
    token: str,
    progress_callback: Optional[ProgressCallback] = ...,
    endpoint: Optional[str] = ...,
    xet_connection_info: Optional[Any] = ...,
) -> Tuple[int, int]:
    """Download the given files through the Xet path.

    Returns ``(files_completed, total_files)``.
    """

def get_lib_version() -> str:
    """Version string of the compiled ``hf_xet`` runtime."""