"""Shared test configuration for hf_track test suite.

Eagerly imports huggingface_hub and forces its lazy submodules to load,
preventing circular import errors when unittest.mock.patch() is used
on lazy-loaded attributes.

Background: huggingface_hub v1.8.0 uses __getattr__-based lazy imports.
When a submodule (e.g. utils._xet) is imported directly before the
full package is loaded, subsequent __getattr__ calls for other
submodules (e.g. hf_api) can trigger a circular import in _buckets.py.

Simply doing ``import huggingface_hub`` is NOT enough — the lazy
attributes are not populated into ``__dict__`` until they are accessed.
We must explicitly access each attribute that our tests patch, so that
``patch()`` finds them in ``__dict__`` and does not trigger ``__getattr__``.
"""
import logging

logger = logging.getLogger(__name__)

try:
    import huggingface_hub  # noqa: F401

    # Force lazy submodules to load by accessing them.
    # This populates sys.modules and huggingface_hub.__dict__ so that
    # subsequent patch() calls find the attributes without triggering
    # __getattr__ (which can hit the circular import in _buckets.py).
    _EAGER_ATTRS = ("HfApi", "snapshot_download", "logging")
    for _attr in _EAGER_ATTRS:
        try:
            getattr(huggingface_hub, _attr)
        except (AttributeError, ImportError):
            logger.debug("huggingface_hub.%s not available", _attr)

except ImportError:
    logger.debug("huggingface_hub not installed")
