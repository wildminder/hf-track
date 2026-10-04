"""Tests for the ``_diag`` diagnostic shim (IMP-020 + NTH-014).

``callbacks/tqdm_download.py`` used to carry ~40 lines of inline
``HF_TRACK_DEBUG_XET`` instrumentation spread across 11 regions. Every one
of them is now routed through the module-level ``_diag`` helper, so the
gate lives in one place. These tests pin that collapse structurally and
behaviourally.
"""

from __future__ import annotations

import logging
import queue
import re
from pathlib import Path

import pytest

from hf_track.callbacks import tqdm_download
from hf_track.callbacks.tqdm_download import DownloadProgressTqdm

SOURCE_ROOT = Path(tqdm_download.__file__).resolve().parent
MODULE_SRC = SOURCE_ROOT / "tqdm_download.py"


def _make_bar(transfer_id: str) -> DownloadProgressTqdm:
    """Build a byte-bar tqdm subclass driven through the instrumented path."""
    q: queue.Queue = queue.Queue()
    return DownloadProgressTqdm(
        total=1000,
        desc="diag.bin",
        unit="B",
        unit_scale=True,
        event_queue=q,
        transfer_id=transfer_id,
        filename="diag.bin",
        report_interval=0,  # never throttle: every update is instrumented
    )


class TestDiagShim:
    """The instrumentation is one helper, one env var, two behaviours."""

    def test_diag_blocks_collapse_to_single_helper(self):
        """The env var is named exactly once — in the gate itself."""
        src = MODULE_SRC.read_text(encoding="utf-8")

        assert len(re.findall(r"HF_TRACK_DEBUG_XET", src)) <= 1

    def test_diag_is_a_single_module_level_helper(self):
        """One definition, and no inline ``import os`` inside methods."""
        src = MODULE_SRC.read_text(encoding="utf-8")

        assert len(re.findall(r"^def _diag\(", src, re.M)) == 1
        assert "import os as _os" not in src

    def test_diag_disabled_by_default_emits_nothing(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ):
        """With the env var unset a full bar lifecycle logs nothing."""
        monkeypatch.delenv(tqdm_download.DIAG_ENV_VAR, raising=False)

        with caplog.at_level(logging.DEBUG, logger=tqdm_download.logger.name):
            bar = _make_bar("diag-off")
            bar.update(500)
            bar.update(500)
            bar.close()

        assert caplog.records == []

    def test_diag_enabled_emits_debug_record(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ):
        """With the env var set the same lifecycle emits DEBUG records."""
        monkeypatch.setenv(tqdm_download.DIAG_ENV_VAR, "1")

        with caplog.at_level(logging.DEBUG, logger=tqdm_download.logger.name):
            bar = _make_bar("diag-on")
            bar.update(500)
            bar.update(500)
            bar.close()

        assert any(r.levelname == "DEBUG" for r in caplog.records)

    def test_diag_records_carry_the_pid_and_thread(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ):
        """The shim keeps the process/thread context the inline blocks had."""
        monkeypatch.setenv(tqdm_download.DIAG_ENV_VAR, "1")

        with caplog.at_level(logging.DEBUG, logger=tqdm_download.logger.name):
            bar = _make_bar("diag-ctx")
            bar.update(100)
            bar.close()

        messages = [r.getMessage() for r in caplog.records]
        assert any("tid=" in m for m in messages)

    def test_diag_never_raises(self):
        """A field whose ``__str__`` blows up must not break the download."""
        class _Hostile:
            def __str__(self) -> str:
                raise RuntimeError("boom")

            __repr__ = __str__

        tqdm_download._diag("HOSTILE", value=_Hostile())  # must not raise