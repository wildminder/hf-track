"""Tests for SubprocessMessage dataclass and factory methods."""

from __future__ import annotations

import pytest

from hf_progress.subprocess_messages import (
    MSG_CANCELLED,
    MSG_ERROR,
    MSG_EVENT,
    MSG_RESULT,
    SubprocessMessage,
)


class TestSubprocessMessageCreation:
    """Test basic SubprocessMessage construction."""

    def test_subprocess_message_creation(self):
        msg = SubprocessMessage(msg_type=MSG_EVENT, payload={"key": "value"})
        assert msg.msg_type == MSG_EVENT
        assert msg.payload == {"key": "value"}

    def test_subprocess_message_default_payload(self):
        """Payload is required — no default."""
        with pytest.raises(TypeError):
            SubprocessMessage(msg_type=MSG_EVENT)  # type: ignore[call-arg]

    def test_msg_type_constants(self):
        assert MSG_EVENT == "event"
        assert MSG_RESULT == "result"
        assert MSG_ERROR == "error"
        assert MSG_CANCELLED == "cancelled"


class TestSubprocessMessageSerialization:
    """Test to_dict / from_dict round-trip."""

    def test_round_trip_event(self):
        original = SubprocessMessage.event({"event_type": "progress", "transfer_id": "abc"})
        d = original.to_dict()
        restored = SubprocessMessage.from_dict(d)
        assert restored.msg_type == original.msg_type
        assert restored.payload == original.payload

    def test_round_trip_result(self):
        original = SubprocessMessage.result(filename="model.bin", file_size=1024)
        d = original.to_dict()
        restored = SubprocessMessage.from_dict(d)
        assert restored.msg_type == MSG_RESULT
        assert restored.payload["status"] == "success"
        assert restored.payload["filename"] == "model.bin"

    def test_round_trip_error(self):
        original = SubprocessMessage.error("disk full", error_type="OSError", retryable=False)
        d = original.to_dict()
        restored = SubprocessMessage.from_dict(d)
        assert restored.msg_type == MSG_ERROR
        assert restored.payload["message"] == "disk full"

    def test_round_trip_cancelled(self):
        original = SubprocessMessage.cancelled(bytes_completed=500, total_bytes=1000)
        d = original.to_dict()
        restored = SubprocessMessage.from_dict(d)
        assert restored.msg_type == MSG_CANCELLED
        assert restored.payload["bytes_completed"] == 500

    def test_to_dict_is_plain_dict(self):
        msg = SubprocessMessage.event({"event_type": "start"})
        d = msg.to_dict()
        assert isinstance(d, dict)
        assert not isinstance(d, SubprocessMessage)

    def test_from_dict_missing_key_raises(self):
        with pytest.raises(KeyError):
            SubprocessMessage.from_dict({"payload": {}})


class TestSubprocessMessageFactories:
    """Test factory methods."""

    def test_event_message(self):
        event_dict = {
            "event_type": "progress",
            "transfer_id": "test-123",
            "direction": "download",
            "filename": "model.bin",
            "phase": "downloading",
            "bytes_completed": 500,
            "total_bytes": 1000,
            "percentage": 50.0,
            "speed": 100.0,
        }
        msg = SubprocessMessage.event(event_dict)
        assert msg.msg_type == MSG_EVENT
        assert msg.payload == event_dict
        assert msg.is_event is True
        assert msg.is_terminal is False

    def test_result_message(self):
        msg = SubprocessMessage.result(
            filename="model.bin",
            destination_path="/tmp/model.bin",
            file_size=1000,
            transfer_id="test-123",
        )
        assert msg.msg_type == MSG_RESULT
        assert msg.payload["status"] == "success"
        assert msg.payload["filename"] == "model.bin"
        assert msg.payload["destination_path"] == "/tmp/model.bin"
        assert msg.is_result is True
        assert msg.is_terminal is True

    def test_error_message(self):
        msg = SubprocessMessage.error(
            message="Connection refused",
            error_type="ConnectionError",
            retryable=True,
        )
        assert msg.msg_type == MSG_ERROR
        assert msg.payload["message"] == "Connection refused"
        assert msg.payload["error_type"] == "ConnectionError"
        assert msg.payload["retryable"] is True
        assert msg.is_error is True
        assert msg.is_terminal is True

    def test_error_message_defaults(self):
        msg = SubprocessMessage.error(message="something failed")
        assert msg.payload["error_type"] == "Exception"
        assert msg.payload["retryable"] is False

    def test_cancelled_message(self):
        msg = SubprocessMessage.cancelled(
            message="User pressed Ctrl+C",
            bytes_completed=750,
            total_bytes=1000,
        )
        assert msg.msg_type == MSG_CANCELLED
        assert msg.payload["message"] == "User pressed Ctrl+C"
        assert msg.payload["bytes_completed"] == 750
        assert msg.payload["total_bytes"] == 1000
        assert msg.is_cancelled is True
        assert msg.is_terminal is True

    def test_cancelled_message_defaults(self):
        msg = SubprocessMessage.cancelled()
        assert msg.payload["message"] == "Transfer cancelled by user"
        assert msg.payload["bytes_completed"] == 0
        assert msg.payload["total_bytes"] == 0


class TestSubprocessMessageTypeChecks:
    """Test is_event, is_result, is_error, is_cancelled, is_terminal properties."""

    def test_event_is_not_terminal(self):
        msg = SubprocessMessage.event({"event_type": "progress"})
        assert msg.is_event is True
        assert msg.is_result is False
        assert msg.is_error is False
        assert msg.is_cancelled is False
        assert msg.is_terminal is False

    def test_result_is_terminal(self):
        msg = SubprocessMessage.result()
        assert msg.is_event is False
        assert msg.is_result is True
        assert msg.is_terminal is True

    def test_error_is_terminal(self):
        msg = SubprocessMessage.error("fail")
        assert msg.is_error is True
        assert msg.is_terminal is True

    def test_cancelled_is_terminal(self):
        msg = SubprocessMessage.cancelled()
        assert msg.is_cancelled is True
        assert msg.is_terminal is True

    def test_unknown_type_is_not_terminal(self):
        msg = SubprocessMessage(msg_type="unknown", payload={})
        assert msg.is_event is False
        assert msg.is_result is False
        assert msg.is_error is False
        assert msg.is_cancelled is False
        assert msg.is_terminal is False
