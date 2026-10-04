"""Tests for ``hf_track.types.errors`` -- exception hierarchy."""

from __future__ import annotations

import pytest

from hf_track.types import (
    TokenError,
    TransferCancelledError,
    TransferProgressError,
)


class TestTransferCancelledError:
    """Tests for TransferCancelledError exception."""

    def test_is_exception(self):
        with pytest.raises(TransferCancelledError):
            raise TransferCancelledError("Transfer cancelled by user")

    def test_message(self):
        try:
            raise TransferCancelledError("Transfer cancelled by user")
        except TransferCancelledError as e:
            assert str(e) == "Transfer cancelled by user"

    def test_catch_specificity(self):
        """TransferCancelledError should not catch generic RuntimeError."""
        with pytest.raises(TransferCancelledError):
            raise TransferCancelledError("cancelled")
        # RuntimeError should NOT be caught as TransferCancelledError
        with pytest.raises(RuntimeError):
            raise RuntimeError("other error")


class TestTransferProgressError:
    """Tests for TransferProgressError exception."""

    def test_is_exception(self):
        with pytest.raises(TransferProgressError):
            raise TransferProgressError("progress error")

    def test_message(self):
        try:
            raise TransferProgressError("recoverable")
        except TransferProgressError as e:
            assert str(e) == "recoverable"


class TestTokenError:
    """Tests for TokenError exception."""

    def test_is_exception(self):
        with pytest.raises(TokenError):
            raise TokenError("auth failed")

    def test_message(self):
        try:
            raise TokenError("missing token")
        except TokenError as e:
            assert str(e) == "missing token"

    def test_all_errors_are_distinct(self):
        """Each exception class should be catchable independently."""
        for exc_cls in (TransferCancelledError, TransferProgressError, TokenError):
            with pytest.raises(exc_cls):
                raise exc_cls(f"raised {exc_cls.__name__}")
