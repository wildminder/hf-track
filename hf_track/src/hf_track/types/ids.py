"""Transfer ID generation.

A small, single-purpose utility kept in its own module because it is
the only function in the type system that produces a value (rather
than a type), and it is logically independent from events, results,
and errors.
"""

from __future__ import annotations

import uuid


def generate_transfer_id() -> str:
    """Generate a unique transfer ID."""
    return str(uuid.uuid4())
