"""Tests for ``hf_track.types.results`` -- TransferResult and TransferErrorInfo."""

from __future__ import annotations

from hf_track.types import (
    TransferDirection,
    TransferErrorInfo,
    TransferResult,
)


class TestTransferError:
    """Tests for TransferErrorInfo dataclass."""

    def test_basic_creation(self):
        err = TransferErrorInfo(message="Network timeout", error_type="TimeoutError", retryable=True)
        assert err.message == "Network timeout"
        assert err.error_type == "TimeoutError"
        assert err.retryable is True
        assert str(err) == "Network timeout"

    def test_to_from_dict(self):
        err = TransferErrorInfo(message="Connection refused", error_type="ConnectionError", retryable=False)
        d = err.to_dict()
        assert d["message"] == "Connection refused"
        assert d["error_type"] == "ConnectionError"
        assert d["retryable"] is False

        restored = TransferErrorInfo.from_dict(d)
        assert restored.message == "Connection refused"
        assert restored.error_type == "ConnectionError"
        assert restored.retryable is False


class TestTransferResult:
    """Tests for TransferResult dataclass."""

    def test_basic_creation(self):
        result = TransferResult(
            success=True,
            transfer_id="test-1",
            filename="model.bin",
            url="https://huggingface.co/user/model/resolve/main/model.bin",
            hash="abc123",
            file_size=1000,
            direction=TransferDirection.UPLOAD,
        )
        assert result.success is True
        assert result.url is not None
        assert result.hash == "abc123"
        assert result.direction == TransferDirection.UPLOAD
        # String default is the same value as TransferDirection.UPLOAD
        assert result.direction == "upload"

    def test_default_direction_is_upload(self):
        result = TransferResult(
            success=False,
            transfer_id="test-2",
            filename="model.bin",
        )
        assert result.direction == "upload"
        assert result.direction == TransferDirection.UPLOAD
