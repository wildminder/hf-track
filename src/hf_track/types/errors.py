"""Exception types raised by HuggingFace progress tracking.

All custom exceptions emitted by the library live here so that callers
can catch them with a single import and a single base hierarchy review.
"""

from __future__ import annotations


class TransferCancelledError(Exception):
    """Raised when a transfer is cancelled by the user.

    Callers can catch this specific exception to distinguish
    user-initiated cancellation from other runtime errors.
    """


class TransferProgressError(Exception):
    """Raised when a transfer operation encounters a recoverable error."""


class TokenError(Exception):
    """Raised when token resolution or authentication fails."""
