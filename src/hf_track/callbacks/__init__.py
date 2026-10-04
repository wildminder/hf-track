"""Progress callback classes for intercepting transfer progress.

This subpackage provides the runtime callbacks used to translate low-level
``hf_xet``/``huggingface_hub`` progress signals into structured
``ProgressEvent`` objects that can be consumed by trackers and the web UI.

Cohesive modules
================

- :mod:`.state` — ``TransferStateManager``: thread-safe aggregate-state
  coordinator shared by byte bars and upload parts. Owns the
  ``state_manager`` singleton.
- :mod:`.xet_callback` — ``XetProgressCallback`` and its backward-compatible
  ``XetUploadProgressCallback`` / ``XetDownloadProgressCallback`` subclasses
  that adapt ``hf_xet``'s ``(total_update, item_updates)`` progress reports
  into ``ProgressEvent`` instances.
- :mod:`.tqdm_download` — ``DownloadProgressTqdm``: a ``tqdm`` subclass that
  captures byte/file-count progress emitted by ``huggingface_hub``,
  applies throttling, and produces a single stream of events.
- :mod:`.tqdm_patch` — ``tqdm_upload_patcher`` context manager that
  monkey-patches ``tqdm.auto.tqdm`` to a UPLOAD-aware variant for the
  duration of an upload so that ``huggingface_hub.upload_file``/large-file
  flows emit the same kind of events.

Public re-exports
=================

The public names listed in ``__all__`` keep the historical flat
``from hf_track.callbacks import ...`` import paths working, so no
downstream consumer needs to change its imports.
"""

from __future__ import annotations

from .state import TransferStateManager, state_manager
from .tqdm_download import DownloadProgressTqdm
from .tqdm_patch import tqdm_upload_patcher
from .xet_callback import (
    XetDownloadProgressCallback,
    XetProgressCallback,
    XetUploadProgressCallback,
    _DummyFile,
    _dummy_file,
)

__all__ = [
    # State coordination
    "TransferStateManager",
    "state_manager",
    # Xet callback family
    "XetProgressCallback",
    "XetUploadProgressCallback",
    "XetDownloadProgressCallback",
    # tqdm
    "DownloadProgressTqdm",
    "tqdm_upload_patcher",
    # Internal helpers (kept accessible to sibling subpackages)
    "_DummyFile",
    "_dummy_file",
]
