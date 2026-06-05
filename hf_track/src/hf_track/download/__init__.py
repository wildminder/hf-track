"""Public download entry points for hf-track.

Subpackage layout (grouped by *what kind of download*):

- :mod:`.xet_file`     -- single-file Xet (legacy + new XetSession API)
- :mod:`.xet_batch`    -- multi-file batch Xet download
- :mod:`.xet_snapshot` -- repository snapshot Xet (legacy + deprecated XetSession)
- :mod:`.xet_streaming`-- streaming snapshot via chunk-by-chunk subprocess
- :mod:`.standard`     -- non-Xet HTTP / tqdm_class downloads

All public symbols previously exposed by ``xet_download.py`` and
``standard_download.py`` are re-exported here so existing imports
keep working.
"""

from __future__ import annotations

from .xet_batch import download_files_with_xet
from .xet_file import (
    XetDownloadResult,
    download_file_with_xet,
    download_file_with_xet_session,
)
from .xet_snapshot import (
    download_snapshot_with_xet,
    download_snapshot_with_xet_session,
)
from .xet_streaming import download_snapshot_streaming
from .standard import download_file, download_snapshot, patch_download_chunk_size

__all__ = [
    # Xet single-file
    "XetDownloadResult",
    "download_file_with_xet",
    "download_file_with_xet_session",
    # Xet batch
    "download_files_with_xet",
    # Xet snapshot
    "download_snapshot_with_xet",
    "download_snapshot_with_xet_session",
    "download_snapshot_streaming",
    # Standard (non-Xet)
    "download_file",
    "download_snapshot",
    "patch_download_chunk_size",
]
