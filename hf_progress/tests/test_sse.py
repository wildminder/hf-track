"""Minimal tests for the SSE integration module."""
from __future__ import annotations


def test_sse_importable():
    from hf_progress.integrations import sse

    assert sse is not None
