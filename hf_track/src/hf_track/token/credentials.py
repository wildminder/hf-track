"""Xet authentication credentials -- a small immutable envelope.

``XetCredentials`` is the value object passed to the Rust ``hf_xet``
runtime. It carries:

- the storage endpoint URL
- a (token, expiration) tuple
- an optional refresher callable the Rust side calls when the token expires

Kept in its own module because it is the public contract between
``hf_track`` and the ``hf_xet`` extension; the rest of the token
package manipulates instances of this type but does not add behaviour
to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass
class XetCredentials:
    """Xet authentication credentials for a single operation.

    Attributes:
        endpoint: Xet storage endpoint URL.
        token_info: Tuple of (access_token, expiration_unix_epoch).
        token_refresher: Optional callable that returns a fresh
            (access_token, expiration_unix_epoch) tuple.
    """

    endpoint: str
    token_info: Tuple[str, int]
    token_refresher: Optional[Callable[[], Tuple[str, int]]] = None
