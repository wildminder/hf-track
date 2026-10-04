"""Integration helpers for external frameworks.

Provides optional integrations with web frameworks and real-time
streaming protocols. Each integration is an opt-in dependency
controlled by extras in pyproject.toml:

- **sse**: Re-exports :class:`sse_starlette.EventSourceResponse` for
  convenience. For a full working example, see
  ``examples/web_app/app.py``.
  Install with ``pip install 'hf-track[sse]'``.
"""

from __future__ import annotations
