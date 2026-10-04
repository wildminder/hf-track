"""hf-track: Plug-and-play progress tracking for HuggingFace Hub.

A library that provides real-time upload/download progress tracking
for HuggingFace Hub operations. Uses a **direct-first** strategy:

- **Xet uploads**: Direct ``hf_xet`` calls with detailed callbacks
  (dedup, transfer speed, per-file progress)
- **Xet downloads**: Direct ``hf_xet.download_files()`` with
  detailed ``(total_update, item_updates)`` callbacks for speed,
  dedup info, and per-item progress
- **HTTP downloads** (fallback): ``tqdm_class`` override for ``hf_hub_download()``
- **LFS uploads** (fallback): tqdm monkey-patching for ``HfApi.upload_file()``

Quick start::

    from hf_track import HfTracker, EventType

    tracker = HfTracker(token="hf_...")

    # Upload with progress
    result = tracker.upload_file("model.bin", "username/repo")

    # Download with progress
    path = tracker.download_file("bert-base-uncased", "config.json")

    # Consume events
    for event in tracker.events(stop_on=EventType.COMPLETE):
        print(f"{event.filename}: {event.percentage:.1f}%")

Low-level callbacks::

    from hf_track import (
        XetUploadProgressCallback,
        XetDownloadProgressCallback,
        DownloadProgressTqdm,
        tqdm_upload_patcher,
    )

SSE integration::

    from hf_track.integrations.sse import EventSourceResponse

    See ``examples/web_app/app.py`` for a full working example.
"""

from __future__ import annotations

import importlib.metadata
import warnings

# Core types
from .types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferCancelledError,
    TransferDirection,
    TransferErrorInfo,
    TransferProgressError,
    TransferResult,
    TokenError,
    generate_transfer_id,
)

# Callback classes
from .callbacks import (
    DownloadProgressTqdm,
    XetDownloadProgressCallback,
    XetProgressCallback,
    XetUploadProgressCallback,
    tqdm_upload_patcher,
)

# Token management
from .token import XetCredentials, XetTokenManager, is_xet_available, has_xet_session

# Subprocess isolation
from .subprocess import SubprocessMessage, XetSubprocessRunner

# High-level tracker
from .tracker import HfTracker

# The single source of truth for the version is ``[project] version`` in
# ``pyproject.toml``, which is what the installed distribution metadata is
# built from. A second literal here used to drift from it silently (it read
# "0.1.0" against a declared "0.2.4"), so this reads the metadata instead.
# The fallback covers a bare checkout where the package is importable via
# ``pythonpath = ["src"]`` but was never installed, so there is no
# distribution to ask.
try:
    __version__ = importlib.metadata.version("hf-track")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    # Core types
    "EventType",
    "ProgressEvent",
    "ProgressPhase",
    "TransferCancelledError",
    "TransferDirection",
    "TransferErrorInfo",
    "TransferProgressError",
    "TransferResult",
    "generate_transfer_id",
    # Callback classes
    "DownloadProgressTqdm",
    "XetDownloadProgressCallback",
    "XetProgressCallback",
    "XetUploadProgressCallback",
    "tqdm_upload_patcher",
    # Token management
    "XetCredentials",
    "XetTokenManager",
    "is_xet_available",
    "has_xet_session",
    # Subprocess isolation
    "XetSubprocessRunner",
    "SubprocessMessage",
    # High-level tracker
    "HfTracker",
]

#: Names renamed in this release, mapped to their replacement. Kept for one
#: deprecation cycle, then deleted along with ``__getattr__`` below.
_RENAMED = {
    # A dataclass that sat next to two exceptions of similar name, which
    # invited ``except TransferError`` — a clause that caught nothing.
    "TransferError": TransferErrorInfo,
}


def __getattr__(name: str):
    """Serve a pre-rename name with a ``DeprecationWarning``, for one cycle."""
    replacement = _RENAMED.get(name)
    if replacement is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"hf_track.{name} was renamed to {replacement.__name__}: it is a "
        f"dataclass carrying an error payload, not an exception. Import "
        f"{replacement.__name__} to read it, and catch "
        f"TransferCancelledError / TransferProgressError for failures.",
        DeprecationWarning,
        stacklevel=2,
    )
    return replacement
