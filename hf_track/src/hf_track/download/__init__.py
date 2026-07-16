"""Public download entry points for hf-track.

Subpackage layout (grouped by *what kind of download*):

- :mod:`.xet_file`     -- single-file Xet
- :mod:`.xet_batch`    -- multi-file batch Xet download
- :mod:`.xet_snapshot` -- repository snapshot Xet
- :mod:`.xet_streaming`-- streaming snapshot via chunk-by-chunk subprocess
- :mod:`.standard`     -- non-Xet HTTP / tqdm_class downloads

The XetSession-API variants (``download_file_with_xet_session``,
``download_snapshot_with_xet_session``) and the underlying
``_xet_session_*`` workers were removed on 2026-06-05 (had three
critical correctness bugs documented in
docs/plans/2026-06-03-revert-broken-snapshot-xetsession-path.md).
Only the proven subprocess path remains.
"""

from __future__ import annotations

from .xet_batch import download_files_with_xet
from .xet_file import XetDownloadResult, download_file_with_xet
from .xet_file_only import download_file_xet_only, download_file_xet_subprocess
from .xet_snapshot import download_snapshot_with_xet
from .xet_streaming import download_snapshot_streaming
from .standard import download_file, download_snapshot, patch_download_chunk_size

__all__ = [
    # Xet single-file (legacy subprocess path — deprecated, see xet_file)
    "XetDownloadResult",
    "download_file_with_xet",
    # Xet single-file (dedicated path, in-process, NO HTTP fallback)
    "download_file_xet_only",
    # Xet single-file (dedicated path, TERMINABLE subprocess, NO HTTP fallback)
    "download_file_xet_subprocess",
    # Xet batch
    "download_files_with_xet",
    # Xet snapshot
    "download_snapshot_with_xet",
    "download_snapshot_streaming",
    # Standard (non-Xet)
    "download_file",
    "download_snapshot",
    "patch_download_chunk_size",
]
