"""Tests for hf_progress.types module."""

from __future__ import annotations

import time

import pytest

from hf_progress.types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferCancelledError,
    TransferDirection,
    TransferError,
    TransferResult,
    generate_transfer_id,
)


class TestEventType:
    """Tests for EventType enum."""

    def test_values(self):
        assert EventType.START.value == "start"
        assert EventType.PROGRESS.value == "progress"
        assert EventType.COMPLETE.value == "complete"
        assert EventType.ERROR.value == "error"

    def test_from_string(self):
        assert EventType("start") == EventType.START
        assert EventType("progress") == EventType.PROGRESS
        assert EventType("complete") == EventType.COMPLETE
        assert EventType("error") == EventType.ERROR


class TestProgressPhase:
    """Tests for ProgressPhase enum."""

    def test_values(self):
        assert ProgressPhase.HASHING.value == "hashing"
        assert ProgressPhase.UPLOADING.value == "uploading"
        assert ProgressPhase.DOWNLOADING.value == "downloading"
        assert ProgressPhase.VERIFYING.value == "verifying"
        assert ProgressPhase.COMPLETE.value == "complete"
        assert ProgressPhase.ERROR.value == "error"


class TestTransferDirection:
    """Tests for TransferDirection enum."""

    def test_values(self):
        assert TransferDirection.UPLOAD.value == "upload"
        assert TransferDirection.DOWNLOAD.value == "download"


class TestTransferError:
    """Tests for TransferError dataclass."""
    
    def test_basic_creation(self):
        err = TransferError(message="Network timeout", error_type="TimeoutError", retryable=True)
        assert err.message == "Network timeout"
        assert err.error_type == "TimeoutError"
        assert err.retryable is True
        assert str(err) == "Network timeout"
        
    def test_to_from_dict(self):
        err = TransferError(message="Connection refused", error_type="ConnectionError", retryable=False)
        d = err.to_dict()
        assert d["message"] == "Connection refused"
        assert d["error_type"] == "ConnectionError"
        assert d["retryable"] is False
        
        restored = TransferError.from_dict(d)
        assert restored.message == "Connection refused"
        assert restored.error_type == "ConnectionError"
        assert restored.retryable is False


class TestProgressEvent:
    """Tests for ProgressEvent dataclass."""

    def test_basic_creation(self):
        event = ProgressEvent(
            event_type=EventType.START,
            transfer_id="test-1",
            direction=TransferDirection.UPLOAD,
            filename="model.bin",
            phase=ProgressPhase.UPLOADING,
            total_bytes=1000,
        )
        assert event.event_type == EventType.START
        assert event.transfer_id == "test-1"
        assert event.direction == TransferDirection.UPLOAD
        assert event.filename == "model.bin"
        assert event.total_bytes == 1000
        assert event.bytes_completed == 0
        assert event.percentage == 0.0

    def test_start_factory(self):
        event = ProgressEvent.start(
            transfer_id="test-factory",
            direction=TransferDirection.UPLOAD,
            filename="model.bin",
            phase=ProgressPhase.UPLOADING,
            total_bytes=2000,
        )
        assert event.event_type == EventType.START
        assert event.total_bytes == 2000
        assert event.bytes_completed == 0

    def test_complete_factory(self):
        event = ProgressEvent.complete(
            transfer_id="test-factory",
            direction=TransferDirection.DOWNLOAD,
            filename="config.json",
            bytes_completed=1000,
            total_bytes=1000,
        )
        assert event.event_type == EventType.COMPLETE
        assert event.percentage == 100.0
        assert event.bytes_completed == 1000

    def test_error_event_factory(self):
        err = TransferError(message="Connection refused", error_type="ConnectionError")
        event = ProgressEvent.error_event(
            transfer_id="test-factory",
            direction=TransferDirection.UPLOAD,
            filename="model.bin",
            error=err,
        )
        assert event.event_type == EventType.ERROR
        assert event.error.message == "Connection refused"
        assert event.error.error_type == "ConnectionError"
        assert event.phase == ProgressPhase.ERROR

    def test_to_dict_basic(self):
        event = ProgressEvent(
            event_type=EventType.PROGRESS,
            transfer_id="test-2",
            direction=TransferDirection.DOWNLOAD,
            filename="config.json",
            phase=ProgressPhase.DOWNLOADING,
            bytes_completed=500,
            total_bytes=1000,
            percentage=50.0,
            speed=1024.0,
        )
        d = event.to_dict()
        assert d["event_type"] == "progress"
        assert d["transfer_id"] == "test-2"
        assert d["direction"] == "download"
        assert d["filename"] == "config.json"
        assert d["phase"] == "downloading"
        assert d["bytes_completed"] == 500
        assert d["total_bytes"] == 1000
        assert d["percentage"] == 50.0
        assert d["speed"] == 1024.0

    def test_to_dict_xet_fields(self):
        """Xet-specific fields are only included when they have values."""
        # Without Xet fields
        event = ProgressEvent(
            event_type=EventType.PROGRESS,
            transfer_id="test-3",
            direction=TransferDirection.UPLOAD,
            filename="model.bin",
            phase=ProgressPhase.UPLOADING,
        )
        d = event.to_dict()
        assert "transfer_bytes_completed" not in d
        assert "dedup_saved_bytes" not in d

        # With Xet fields
        event = ProgressEvent(
            event_type=EventType.PROGRESS,
            transfer_id="test-4",
            direction=TransferDirection.UPLOAD,
            filename="model.bin",
            phase=ProgressPhase.UPLOADING,
            transfer_bytes_completed=400,
            transfer_bytes_total=800,
            transfer_speed=100.0,
            dedup_saved_bytes=100,
        )
        d = event.to_dict()
        assert d["transfer_bytes_completed"] == 400
        assert d["transfer_bytes_total"] == 800
        assert d["transfer_speed"] == 100.0
        assert d["dedup_saved_bytes"] == 100

    def test_to_dict_error(self):
        """Error field is serialized to dict."""
        err = TransferError(message="Connection refused", error_type="ConnectionError")
        event = ProgressEvent(
            event_type=EventType.ERROR,
            transfer_id="test-5",
            direction=TransferDirection.UPLOAD,
            filename="model.bin",
            phase=ProgressPhase.ERROR,
            error=err,
        )
        d = event.to_dict()
        assert isinstance(d["error"], dict)
        assert d["error"]["message"] == "Connection refused"

        event_no_error = ProgressEvent(
            event_type=EventType.PROGRESS,
            transfer_id="test-6",
            direction=TransferDirection.UPLOAD,
            filename="model.bin",
            phase=ProgressPhase.UPLOADING,
        )
        d = event_no_error.to_dict()
        assert "error" not in d

    def test_from_dict_roundtrip(self):
        event = ProgressEvent(
            event_type=EventType.PROGRESS,
            transfer_id="test-7",
            direction=TransferDirection.UPLOAD,
            filename="model.bin",
            phase=ProgressPhase.UPLOADING,
            bytes_completed=750,
            total_bytes=1000,
            percentage=75.0,
            speed=500.0,
            transfer_bytes_completed=600,
            transfer_bytes_total=900,
            dedup_saved_bytes=150,
        )
        d = event.to_dict()
        restored = ProgressEvent.from_dict(d)
        assert restored.event_type == event.event_type
        assert restored.transfer_id == event.transfer_id
        assert restored.direction == event.direction
        assert restored.filename == event.filename
        assert restored.bytes_completed == event.bytes_completed
        assert restored.total_bytes == event.total_bytes
        assert restored.percentage == event.percentage
        assert restored.speed == event.speed
        
    def test_from_dict_legacy_error_string(self):
        """Should parse older string-based error formats seamlessly into a TransferError object."""
        legacy_data = {
            "event_type": "error",
            "transfer_id": "test-legacy",
            "direction": "upload",
            "filename": "model.bin",
            "error": "Legacy string error",
        }
        restored = ProgressEvent.from_dict(legacy_data)
        assert isinstance(restored.error, TransferError)
        assert restored.error.message == "Legacy string error"
        assert restored.error.error_type == "Exception"

    def test_from_dict_default_phase_infers_from_direction(self):
        """from_dict should infer default phase from direction when phase is missing."""
        d_upload = {
            "event_type": "progress",
            "transfer_id": "test-d1",
            "direction": "upload",
            "filename": "model.bin",
        }
        restored_upload = ProgressEvent.from_dict(d_upload)
        assert restored_upload.phase == ProgressPhase.UPLOADING

        d_download = {
            "event_type": "progress",
            "transfer_id": "test-d2",
            "direction": "download",
            "filename": "data.bin",
        }
        restored_download = ProgressEvent.from_dict(d_download)
        assert restored_download.phase == ProgressPhase.DOWNLOADING

    def test_percentage_rounding(self):
        event = ProgressEvent(
            event_type=EventType.PROGRESS,
            transfer_id="test-8",
            direction=TransferDirection.UPLOAD,
            filename="model.bin",
            phase=ProgressPhase.UPLOADING,
            bytes_completed=333,
            total_bytes=1000,
            percentage=33.3333,
        )
        d = event.to_dict()
        assert d["percentage"] == 33.33

    def test_timestamp_auto_set(self):
        before = time.time()
        event = ProgressEvent(
            event_type=EventType.START,
            transfer_id="test-9",
            direction=TransferDirection.UPLOAD,
            filename="model.bin",
            phase=ProgressPhase.UPLOADING,
        )
        after = time.time()
        assert before <= event.timestamp <= after


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


class TestGenerateTransferId:
    """Tests for generate_transfer_id function."""

    def test_unique(self):
        ids = {generate_transfer_id() for _ in range(100)}
        assert len(ids) == 100

    def test_is_string(self):
        tid = generate_transfer_id()
        assert isinstance(tid, str)
        assert len(tid) > 0