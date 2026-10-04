"""Synchronous upload methods for HfTracker.

This mixin owns the routing logic for ``upload_file``, ``upload_bytes``,
and ``upload_folder`` — i.e. the "use Xet if available, else LFS"
decision tree. The actual implementations live in :mod:`hf_track.upload`
and are invoked by the ``_*_xet``/``_*_lfs``/``_*_via_temp`` mixin
methods on :class:`_TrackerXetImpls` and direct imports here.

``is_xet_available`` is referenced through the ``hf_track.tracker``
package namespace (``from . import is_xet_available``) so tests using
``patch("hf_track.tracker.is_xet_available", ...)`` keep working — the
symbol is re-exported in ``tracker/__init__.py``.

Split from the original ``tracker.py`` (28KB god class) on 2026-06-05
as part of the modular-refactor plan
(docs/plans/2026-06-04-modular-refactor.md, Step 8).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from ..types import TransferCancelledError

logger = logging.getLogger(__name__)


class _TrackerUploads:
    """Mixin: synchronous upload methods (file + bytes + folder).

    Subclasses must define (provided by :class:`_TrackerCore`):
        _token, _endpoint, _report_interval, event_queue, _prepare_transfer,
        cleanup_transfer, _upload_file_xet, _upload_bytes_xet,
        _upload_file_lfs, _upload_bytes_via_temp.
    """

    def upload_file(
        self,
        file_path: str,
        repo_id: str,
        path_in_repo: Optional[str] = None,
        repo_type: str = "model",
        revision: Optional[str] = None,
        transfer_id: Optional[str] = None,
    ) -> str:
        transfer_id, is_cancelled_hook = self._prepare_transfer(transfer_id)  # type: ignore[attr-defined]
        filename = os.path.basename(file_path)
        path_in_repo = path_in_repo or filename

        # Route through the ``tracker`` package namespace (see
        # ``_downloads.download_file`` for rationale).
        from . import is_xet_available  # noqa: PLC0415
        try:
            if is_xet_available():
                return self._upload_file_xet(  # type: ignore[attr-defined]
                    file_path=file_path,
                    repo_id=repo_id,
                    path_in_repo=path_in_repo,
                    repo_type=repo_type,
                    revision=revision,
                    transfer_id=transfer_id,
                    filename=filename,
                    is_cancelled=is_cancelled_hook,
                )
            else:
                return self._upload_file_lfs(  # type: ignore[attr-defined]
                    file_path=file_path,
                    repo_id=repo_id,
                    path_in_repo=path_in_repo,
                    repo_type=repo_type,
                    revision=revision,
                    transfer_id=transfer_id,
                    filename=filename,
                    is_cancelled=is_cancelled_hook,
                )
        except KeyboardInterrupt:
            raise TransferCancelledError("Upload interrupted by user (Ctrl+C)")
        finally:
            self.cleanup_transfer(transfer_id)  # type: ignore[attr-defined]

    def upload_bytes(
        self,
        file_content: bytes,
        filename: str,
        repo_id: str,
        path_in_repo: Optional[str] = None,
        repo_type: str = "model",
        revision: Optional[str] = None,
        transfer_id: Optional[str] = None,
    ) -> str:
        transfer_id, is_cancelled_hook = self._prepare_transfer(transfer_id)  # type: ignore[attr-defined]
        path_in_repo = path_in_repo or filename

        # Route through the ``tracker`` package namespace (see
        # ``_downloads.download_file`` for rationale).
        from . import is_xet_available  # noqa: PLC0415
        try:
            if is_xet_available():
                return self._upload_bytes_xet(  # type: ignore[attr-defined]
                    file_content=file_content,
                    filename=filename,
                    repo_id=repo_id,
                    path_in_repo=path_in_repo,
                    repo_type=repo_type,
                    revision=revision,
                    transfer_id=transfer_id,
                    is_cancelled=is_cancelled_hook,
                )
            else:
                return self._upload_bytes_via_temp(  # type: ignore[attr-defined]
                    file_content=file_content,
                    filename=filename,
                    repo_id=repo_id,
                    path_in_repo=path_in_repo,
                    repo_type=repo_type,
                    revision=revision,
                    transfer_id=transfer_id,
                    is_cancelled=is_cancelled_hook,
                )
        except KeyboardInterrupt:
            raise TransferCancelledError("Upload interrupted by user (Ctrl+C)")
        finally:
            self.cleanup_transfer(transfer_id)  # type: ignore[attr-defined]

    def upload_folder(
        self,
        folder_path: str,
        repo_id: str,
        path_in_repo: Optional[str] = None,
        repo_type: str = "model",
        revision: Optional[str] = None,
        allow_patterns: Optional[list[str] | str] = None,
        ignore_patterns: Optional[list[str] | str] = None,
        delete_patterns: Optional[list[str] | str] = None,
        transfer_id: Optional[str] = None,
    ) -> str:
        from ..upload import upload_folder as _upload_folder

        transfer_id, is_cancelled_hook = self._prepare_transfer(transfer_id)  # type: ignore[attr-defined]

        try:
            return _upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                token=self._token,  # type: ignore[attr-defined]
                event_queue=self.event_queue,  # type: ignore[attr-defined]
                path_in_repo=path_in_repo,
                repo_type=repo_type,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                delete_patterns=delete_patterns,
                endpoint=self._endpoint,  # type: ignore[attr-defined]
                transfer_id=transfer_id,
                report_interval=self._report_interval,  # type: ignore[attr-defined]
                is_cancelled=is_cancelled_hook,
            )
        except KeyboardInterrupt:
            raise TransferCancelledError("Upload interrupted by user (Ctrl+C)")
        finally:
            self.cleanup_transfer(transfer_id)  # type: ignore[attr-defined]
