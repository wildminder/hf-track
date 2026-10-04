"""Result-related types for HuggingFace progress tracking.

This module groups all *result envelope* types -- the structured objects
returned from a transfer operation. ``TransferResult`` is the universal
result; ``TransferErrorInfo`` is the structured error payload embedded in
``ProgressEvent`` and returned from failed transfers.

Note: ``XetDownloadResult`` and ``XetUploadResult`` live alongside their
respective domain modules (``download/xet_*.py`` and ``upload/xet_*.py``)
and are NOT imported from here, because they carry transfer-specific
fields (file_hash, dedup info) tied to those modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# NOTE: We deliberately do NOT import ``TransferDirection`` from
# ``.events`` here, to avoid a circular import. ``TransferDirection`` is
# a ``str`` subclass, so accepting it as ``str`` keeps the dataclass
# inter-operable with both the enum and plain string values.


@dataclass
class TransferErrorInfo:
    """Structured error information for failed transfers."""

    message: str
    error_type: str = "Exception"
    retryable: bool = False

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "error_type": self.error_type,
            "retryable": self.retryable,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TransferErrorInfo:
        return cls(
            message=data.get("message", "Unknown error"),
            error_type=data.get("error_type", "Exception"),
            retryable=data.get("retryable", False),
        )


@dataclass
class TransferResult:
    """Result of a completed transfer operation.

    Attributes:
        success: Whether the transfer completed successfully.
        transfer_id: Unique identifier for the transfer.
        filename: Name of the transferred file.
        url: URL of the uploaded/downloaded resource (if available).
        hash: Content hash of the file (Xet uploads only).
        file_size: Size of the file in bytes.
        direction: Upload or download.
        local_path: Local file path (for downloads).
    """

    success: bool
    transfer_id: str
    filename: str
    url: Optional[str] = None
    hash: Optional[str] = None
    file_size: int = 0
    # Default is the string ``"upload"`` (same value as
    # ``TransferDirection.UPLOAD``). Typed as ``str`` to avoid a circular
    # import from ``.events``; the field is structurally compatible with
    # ``TransferDirection`` because the enum subclasses ``str``.
    direction: str = "upload"
    local_path: Optional[str] = None
