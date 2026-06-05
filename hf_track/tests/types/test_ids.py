"""Tests for ``hf_track.types.ids`` -- transfer ID generator."""

from __future__ import annotations

import re
import uuid

from hf_track.types import generate_transfer_id


class TestGenerateTransferId:
    """Tests for generate_transfer_id function."""

    def test_unique(self):
        ids = {generate_transfer_id() for _ in range(100)}
        assert len(ids) == 100

    def test_is_string(self):
        tid = generate_transfer_id()
        assert isinstance(tid, str)
        assert len(tid) > 0

    def test_is_valid_uuid(self):
        """Transfer IDs are UUID4 strings."""
        tid = generate_transfer_id()
        # uuid.UUID() will validate the format
        parsed = uuid.UUID(tid)
        assert parsed.version == 4

    def test_no_collisions_in_10k(self):
        """Smoke test: 10k IDs should all be distinct."""
        ids = {generate_transfer_id() for _ in range(10_000)}
        assert len(ids) == 10_000
