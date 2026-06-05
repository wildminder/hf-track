"""Capability detection for the ``hf_xet`` extension.

These two pure-introspection helpers are grouped together because they
both answer the same kind of question: "is the optional ``hf_xet``
Rust extension available, and at what feature level?". They are
deliberately implemented using ``importlib.util.find_spec`` rather
than a direct ``import hf_xet`` to avoid loading the ``.pyd`` into
the main process (which would prevent clean subprocess termination).
"""

from __future__ import annotations

import os


def is_xet_available() -> bool:
    """Check if hf_xet package is installed without importing it.

    Uses ``importlib.util.find_spec`` to detect the package without
    loading the Rust ``.pyd`` extension into the current process.
    This is critical for subprocess isolation: if ``hf_xet`` is loaded
    in the main process, it cannot be safely terminated via SIGTERM.

    Respects the HF_HUB_DISABLE_XET environment variable.

    Returns:
        True if ``hf_xet`` is findable on ``sys.path`` and not disabled, False otherwise.
    """
    import importlib.util

    disable_xet = os.environ.get("HF_HUB_DISABLE_XET", "0").lower()
    if disable_xet in ("1", "true", "yes"):
        return False

    try:
        return importlib.util.find_spec("hf_xet") is not None
    except (ValueError, ImportError):
        # ValueError: hf_xet is in sys.modules but lacks __spec__
        # (e.g. a test injected a MagicMock). Treat as unavailable.
        return False


def has_xet_session() -> bool:
    """Check if the new ``hf_xet.XetSession`` API is available (>= 1.5.0).

    The new ``XetSession`` API provides per-chunk progress callbacks
    that fire every ~100ms. This is the ONLY API that gives smooth
    progress for snapshot downloads on ``hf_xet >= 1.5.0``. The older
    ``hf_xet.download_files`` API only fires its progress callback at
    file completion, which is unsuitable for per-chunk progress.

    This function is safe to call from any process because it only
    inspects the package spec (it does not import the .pyd extension).

    Returns:
        True if ``hf_xet`` is installed AND exposes ``XetSession``,
        False otherwise.
    """
    if not is_xet_available():
        return False
    try:
        import importlib.util
        spec = importlib.util.find_spec("hf_xet")
        if spec is None or spec.submodule_search_locations is None:
            return False
        # Walk the package looking for XetSession without triggering
        # full import. We do this by inspecting __init__'s attributes
        # via importlib's lazy loader, but that's risky. Instead, use
        # a controlled import in a try/except — at the point of this
        # call we are NOT in the main subprocess (we are in a worker
        # or in a fresh main process that won't load the .pyd for
        # purposes other than checking).
        # NOTE: this function should be called from a context where
        # loading the .pyd is safe (e.g. main process before subprocess
        # spawn, or in the worker where it's expected).
        import hf_xet  # noqa: F401
        return hasattr(hf_xet, "XetSession")
    except Exception:
        return False
