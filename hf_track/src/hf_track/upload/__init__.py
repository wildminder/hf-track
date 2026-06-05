"""Public upload entry points for hf-track.

Subpackage layout (grouped by *what kind of upload*):

- :mod:`.xet_file`   -- single-file Xet upload + XetUploadResult +
                        shared subprocess runner
- :mod:`.xet_bytes`  -- bytes-via-Xet + Xet/LFS routing wrapper
- :mod:`.standard`   -- non-Xet HfApi.upload_* via tqdm patching

All public symbols previously exposed by ``xet_upload.py`` and
``standard_upload.py`` are re-exported here so existing imports
keep working.
"""

from __future__ import annotations

from .standard import upload_bytes, upload_file, upload_folder
from .xet_bytes import upload_bytes_via_temp_file, upload_bytes_with_xet
from .xet_file import XetUploadResult, upload_file_with_xet

__all__ = [
    # Xet single-file
    "XetUploadResult",
    "upload_file_with_xet",
    # Xet bytes
    "upload_bytes_with_xet",
    "upload_bytes_via_temp_file",
    # Standard (non-Xet)
    "upload_file",
    "upload_bytes",
    "upload_folder",
]
