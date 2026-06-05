"""Synchronous download methods for HfTracker.

This mixin owns the routing logic for ``download_file``,
``download_snapshot``, and ``download_snapshot_streaming`` — i.e. the
"try Xet, fall back to standard" decision tree and the cancellation
plumbing around it. The actual Xet and standard implementations live
in :mod:`hf_track.download` and are invoked by the
``_*_xet``/``_*_standard`` mixin methods on
:class:`_TrackerXetImpls` and direct imports here.

``is_xet_available`` is referenced through the ``hf_track.tracker``
package namespace (``from . import is_xet_available``) so that tests
using ``patch("hf_track.tracker.is_xet_available", ...)`` keep
working — the symbol is re-exported in ``tracker/__init__.py``.

Split from the original ``tracker.py`` (28KB god class) on 2026-06-05
as part of the modular-refactor plan
(docs/plans/2026-06-04-modular-refactor.md, Step 8).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from ..types import TransferCancelledError

logger = logging.getLogger(__name__)


class _TrackerDownloads:
    """Mixin: synchronous download methods (file + snapshot + streaming).

    Subclasses must define (provided by :class:`_TrackerCore`):
        _token, _endpoint, _report_interval, event_queue, _prepare_transfer,
        cleanup_transfer, _download_file_xet, _download_snapshot_xet.
    """

    def download_file(
        self,
        repo_id: str,
        filename: str,
        repo_type: str = "model",
        revision: Optional[str] = None,
        local_dir: Optional[str] = None,
        transfer_id: Optional[str] = None,
        use_xet: bool = True,
        **kwargs,
    ) -> str:
        transfer_id, is_cancelled_hook = self._prepare_transfer(transfer_id)  # type: ignore[attr-defined]

        # NOTE: route through the ``tracker`` package namespace so tests can
        # ``patch("hf_track.tracker.is_xet_available", ...)``. This was the
        # import path in the original monolithic ``tracker.py``.
        from . import is_xet_available  # noqa: PLC0415
        try:
            if is_xet_available() and use_xet:
                try:
                    return self._download_file_xet(  # type: ignore[attr-defined]
                        repo_id=repo_id,
                        filename=filename,
                        repo_type=repo_type,
                        revision=revision,
                        local_dir=local_dir,
                        transfer_id=transfer_id,
                        is_cancelled=is_cancelled_hook,
                    )
                except TransferCancelledError:
                    raise
                except Exception as xet_err:
                    logger.warning(
                        f"Xet direct download failed for {repo_id}/{filename}, "
                        f"falling back to tqdm_class: {xet_err}"
                    )

            from ..download import download_file as _download_file

            return _download_file(
                repo_id=repo_id,
                filename=filename,
                token=self._token,  # type: ignore[attr-defined]
                event_queue=self.event_queue,  # type: ignore[attr-defined]
                repo_type=repo_type,
                revision=revision,
                endpoint=self._endpoint,  # type: ignore[attr-defined]
                local_dir=local_dir,
                transfer_id=transfer_id,
                report_interval=self._report_interval,  # type: ignore[attr-defined]
                is_cancelled=is_cancelled_hook,
                **kwargs,
            )
        except KeyboardInterrupt:
            raise TransferCancelledError("Download interrupted by user (Ctrl+C)")
        finally:
            self.cleanup_transfer(transfer_id)  # type: ignore[attr-defined]

    def download_snapshot_streaming(
        self,
        repo_id: str,
        allow_patterns=None,
        ignore_patterns=None,
        repo_type: str = "model",
        revision: Optional[str] = None,
        local_dir: Optional[str] = None,
        transfer_id: Optional[str] = None,
        force_download: bool = False,
        fsync_interval: int = 4 * 1024 * 1024,
    ) -> List[str]:
        """Download a repository snapshot using the streaming Xet API.

        Alternative to :meth:`download_snapshot` that uses the chunk-by-chunk
        ``XetSession().new_download_stream_group().download_stream()`` API for
        Xet-stored files. Each chunk is flushed to disk via ``os.write`` +
        ``os.fsync`` and the child subprocess can be killed cleanly mid-file.

        Non-Xet files (small JSON, markdown, etc.) are downloaded via the
        standard ``huggingface_hub.hf_hub_download`` call in the parent
        process -- these files are small and bounded.

        Args:
            repo_id: HuggingFace repository ID.
            allow_patterns: Optional list of glob patterns to include.
            ignore_patterns: Optional list of glob patterns to exclude.
            repo_type: Repository type (model/dataset/space).
            revision: Optional git revision.
            local_dir: Local directory to download files to. Defaults to
                the HuggingFace cache (``HF_HOME``/hub).
            transfer_id: Pre-existing transfer ID. Auto-generated if None.
            force_download: If True, re-download even if files exist.
            fsync_interval: Bytes between ``os.fsync`` calls in the worker.

        Returns:
            Sorted list of file paths that were downloaded.

        Raises:
            ImportError: If ``hf_xet`` is not installed.
            TransferCancelledError: If the user cancels mid-stream.
            TransferProgressError: If a download fails.
        """
        transfer_id, is_cancelled_hook = self._prepare_transfer(transfer_id)  # type: ignore[attr-defined]

        try:
            from ..download import download_snapshot_streaming as _download_snapshot_streaming
            return _download_snapshot_streaming(
                repo_id=repo_id,
                token=self._token,  # type: ignore[attr-defined]
                event_queue=self.event_queue,  # type: ignore[attr-defined]
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                repo_type=repo_type,
                revision=revision,
                endpoint=self._endpoint,  # type: ignore[attr-defined]
                local_dir=local_dir,
                transfer_id=transfer_id,
                report_interval=self._report_interval,  # type: ignore[attr-defined]
                is_cancelled=is_cancelled_hook,
                force_download=force_download,
                fsync_interval=fsync_interval,
            )
        except KeyboardInterrupt:
            raise TransferCancelledError("Download interrupted by user (Ctrl+C)")
        finally:
            self.cleanup_transfer(transfer_id)  # type: ignore[attr-defined]

    def download_snapshot(
        self,
        repo_id: str,
        allow_patterns=None,
        ignore_patterns=None,
        repo_type: str = "model",
        revision: Optional[str] = None,
        local_dir: Optional[str] = None,
        transfer_id: Optional[str] = None,
        force_download: bool = False,
        use_xet: bool = True,
        **kwargs,
    ) -> str:
        from ..download import download_snapshot as _download_snapshot

        transfer_id, is_cancelled_hook = self._prepare_transfer(transfer_id)  # type: ignore[attr-defined]

        # Route through the ``tracker`` package namespace (see ``download_file``).
        from . import is_xet_available  # noqa: PLC0415
        try:
            if is_xet_available():
                # Always use subprocess when xet is installed (for safe
                # cancellation). The worker handles xet disabling internally
                # via the use_xet param — it sets HF_HUB_DISABLE_XET in
                # the child process BEFORE huggingface_hub is imported,
                # so the cached constant reflects the correct value.
                try:
                    return self._download_snapshot_xet(  # type: ignore[attr-defined]
                        repo_id=repo_id,
                        allow_patterns=allow_patterns,
                        ignore_patterns=ignore_patterns,
                        repo_type=repo_type,
                        revision=revision,
                        local_dir=local_dir,
                        transfer_id=transfer_id,
                        is_cancelled=is_cancelled_hook,
                        force_download=force_download,
                        use_xet=use_xet,
                    )
                except TransferCancelledError:
                    raise  # Never fallback on cancellation
                except Exception as xet_err:
                    if not use_xet:
                        # User explicitly disabled xet — don't warn
                        pass
                    else:
                        logger.warning(
                            "Xet snapshot download failed for %s, "
                            "falling back to standard: %s",
                            repo_id, xet_err,
                        )

            return _download_snapshot(
                repo_id=repo_id,
                token=self._token,  # type: ignore[attr-defined]
                event_queue=self.event_queue,  # type: ignore[attr-defined]
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                repo_type=repo_type,
                revision=revision,
                endpoint=self._endpoint,  # type: ignore[attr-defined]
                local_dir=local_dir,
                transfer_id=transfer_id,
                report_interval=self._report_interval,  # type: ignore[attr-defined]
                is_cancelled=is_cancelled_hook,
                force_download=force_download,
                **kwargs,
            )
        except KeyboardInterrupt:
            raise TransferCancelledError("Download interrupted by user (Ctrl+C)")
        finally:
            self.cleanup_transfer(transfer_id)  # type: ignore[attr-defined]
