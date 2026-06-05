"""HfTracker: unified progress tracker for HuggingFace uploads and downloads.

This subpackage is a MIXIN-BASED DECOMPOSITION of the original 28KB
``tracker.py`` god class. The public class :class:`HfTracker` is
composed of six cohesive mixins, each in its own module:

    _core       (state, cancel/cleanup, _prepare_transfer)
    _events     (get_events, events, wait_for_complete)
    _async      (download_*_async, upload_*_async wrappers)
    _downloads  (download_file, download_snapshot, download_snapshot_streaming)
    _uploads    (upload_file, upload_bytes, upload_folder)
    _xet_impls  (_download_*_xet, _upload_*_xet, _upload_*_lfs, _upload_*_via_temp)

The split was done on 2026-06-05 as Step 8 of the modular-refactor
plan (docs/plans/2026-06-04-modular-refactor.md). The public class
is preserved as a single class with all methods, so:

* ``patch.object(HfTracker, "_download_snapshot_xet")`` still works
  in tests (the method is resolved to the class via MRO).
* ``inspect.getsource(HfTracker._download_snapshot_xet)`` still
  returns the implementation (read from the mixin's ``__code__``).
* ``inspect.signature(HfTracker.download_snapshot_streaming)`` still
  returns the public signature.
* ``from hf_track.tracker import HfTracker`` and
  ``from hf_track import HfTracker`` both work unchanged.
* ``is_xet_available`` is imported here (not in the mixins) so
  ``patch("hf_track.tracker.is_xet_available", ...)`` keeps working.
"""

from __future__ import annotations

import logging

from ._async import _TrackerAsync
from ._core import _TrackerCore
from ._downloads import _TrackerDownloads
from ._events import _TrackerEvents
from ._uploads import _TrackerUploads
from ._xet_impls import _TrackerXetImpls

# Re-exported for tests that ``patch("hf_track.tracker.is_xet_available")``.
# Was previously imported in the monolithic tracker.py at module top-level.
from ..token import is_xet_available  # noqa: F401

logger = logging.getLogger(__name__)


class HfTracker(
    _TrackerCore,
    _TrackerEvents,
    _TrackerAsync,
    _TrackerDownloads,
    _TrackerUploads,
    _TrackerXetImpls,
):
    """Unified progress tracker for HuggingFace uploads and downloads.

    Composed of six mixins — see module docstring. The class is
    intended to be used as a single object; mixins are an
    implementation detail for keeping the source modular and
    testable.

    Example:
        >>> tracker = HfTracker(token="hf_xxx")
        >>> path = tracker.download_file("gpt2", "config.json")
    """

    pass


__all__ = ["HfTracker", "is_xet_available"]
