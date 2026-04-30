"""Tests for hf_progress.types module."""

from __future__ import annotations

import time

from hf_progress.types import (
    EventType,
    ProgressEvent,
    ProgressPhase,
    TransferDirection,
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
        """Error field is only included when set."""
        event = ProgressEvent(
            event_type=EventType.ERROR,
            transfer_id="test-5",
            direction=TransferDirection.UPLOAD,
            filename="model.bin",
            phase=ProgressPhase.ERROR,
            error="Connection refused",
        )
        d = event.to_dict()
        assert d["error"] == "Connection refused"

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
