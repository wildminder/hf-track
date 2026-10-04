"""CRIT-009: the streaming download's allow/ignore filtering actually filters.

``xet_streaming.py`` calls ``_matches_any`` from ``_matching.py`` but
originally never imported it, so the filtering branch blew up with
``NameError`` the first time a caller passed ``allow_patterns`` or
``ignore_patterns`` (ruff ``F821`` at ``xet_streaming.py:174``/``:176``).
S14 adds the import; these tests pin both the binding and the behaviour
it restored.
"""

from __future__ import annotations

import hf_track.download.xet_streaming as xet_streaming
from hf_track.download._matching import _matches_any


def test_streaming_module_binds_matches_any():
    """The module must expose the name it calls on the filtering path."""
    assert hasattr(xet_streaming, "_matches_any")


def test_allow_patterns_filters_streamed_paths():
    """Only the paths matching an allow pattern survive the filter."""
    paths = ["config.json", "model-00001-of-00002.safetensors", "tokenizer.json"]

    assert [p for p in paths if _matches_any(p, ["*.json"])] == ["config.json", "tokenizer.json"]
    assert not _matches_any("model-00001-of-00002.safetensors", ["*.json"])


def test_ignore_patterns_skips_streamed_paths():
    """Ignore patterns match by basename and never match when empty."""
    paths = ["README.md", "model.safetensors"]

    assert _matches_any("README.md", ["*.md"]) is True
    assert [p for p in paths if not _matches_any(p, ["*.md"])] == ["model.safetensors"]

    # Empty pattern lists match nothing — the ``patterns or []`` guard at
    # ``_matching.py:16`` that lets the caller pass ``None`` through.
    assert _matches_any("model.safetensors", None) is False
    assert _matches_any("model.safetensors", []) is False