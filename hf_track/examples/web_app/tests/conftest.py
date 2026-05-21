"""Shared test configuration for web app tests.

Eagerly imports huggingface_hub and forces its lazy submodules to load,
preventing circular import errors when unittest.mock.patch() is used.

See hf_track/tests/conftest.py for detailed background.
"""
import logging

logger = logging.getLogger(__name__)

try:
    import huggingface_hub  # noqa: F401

    # Force lazy submodules to load by accessing them.
    _EAGER_ATTRS = ("HfApi", "snapshot_download", "logging")
    for _attr in _EAGER_ATTRS:
        try:
            getattr(huggingface_hub, _attr)
        except (AttributeError, ImportError):
            logger.debug("huggingface_hub.%s not available — skipping", _attr)

except ImportError:
    logger.debug("huggingface_hub not installed — skipping eager import")
