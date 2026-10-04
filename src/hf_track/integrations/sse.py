"""FastAPI/SSE integration utilities for real-time progress streaming.

.. deprecated::
    The ``create_progress_router()`` factory function has been removed.
    It used closure-based endpoint definitions that broke Starlette 1.0's
    response serialization (returning ``null`` instead of JSON).

    For a working FastAPI + SSE integration, see the web app example at
    ``examples/web_app/app.py`` which defines endpoints at module level
    following the official ``sse-starlette`` patterns.

This module re-exports :class:`sse_starlette.EventSourceResponse` for
convenience so you can import it as::

    from hf_track.integrations.sse import EventSourceResponse
"""

from __future__ import annotations

try:
    from sse_starlette import EventSourceResponse
except ImportError as err:
    raise ImportError(
        "sse-starlette is required for SSE integration. "
        "Install with: pip install 'hf-track[sse]'"
    ) from err

__all__ = ["EventSourceResponse"]
