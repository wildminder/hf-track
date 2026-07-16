"""Internal Xet implementation methods for HfTracker.

These are the private ``_*_xet`` / ``_*_lfs`` / ``_*_via_temp``
methods called from the public download/upload routing in
``_downloads`` and ``_uploads``. They are kept separate because they
deal with low-level Xet API details (HfApi metadata, XetSession
fallback, xet-stored file size, etc.) and change more often than
the routing logic.

Split from the original ``tracker.py`` (28KB god class) on 2026-06-05
as part of the modular-refactor plan
(docs/plans/2026-06-04-modular-refactor.md, Step 8).
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class _TrackerXetImpls:
    """Mixin: private ``_*_xet`` / ``_*_lfs`` implementation methods.

    These methods are called from the public routing in
    :class:`_TrackerDownloads` and :class:`_TrackerUploads` and are
    NOT part of the public API. They are kept private (single
    underscore) for two reasons:

    1. The routing logic (which path to take) is owned by the public
       methods, and the actual Xet/LFS plumbing is an implementation
       detail.
    2. ``_download_snapshot_xet`` is the one method that MUST stay on
       the ``HfTracker`` class for ``test_snapshot_regressions`` to
       verify its source via ``inspect.getsource()`` (this is a
       regression test for the 2026-06-03 nested-folder bug).

    Subclasses must define the attributes set by :class:`_TrackerCore`:
    ``_token``, ``_endpoint``, ``_report_interval``, ``event_queue``.
    """

    # ── Xet Download Implementations ──────────────────────────────

    def _download_file_xet(
        self,
        repo_id: str,
        filename: str,
        repo_type: str,
        revision: Optional[str],
        local_dir: Optional[str],
        transfer_id: str,
        is_cancelled: Callable[[], bool],
        on_spawn: Optional[Callable[[object], None]] = None,
        on_finish: Optional[Callable[[], None]] = None,
    ) -> str:
        from huggingface_hub import HfApi, hf_hub_url
        from ..download import download_file_xet_only

        api = HfApi(endpoint=self._endpoint, token=self._token)  # type: ignore[attr-defined]
        url = hf_hub_url(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            endpoint=self._endpoint,
        )
        metadata = api.get_hf_file_metadata(url=url, token=self._token)

        if metadata.xet_file_data is None:
            raise ValueError(
                f"File '{filename}' in '{repo_id}' is not stored in Xet storage."
            )

        xet_file_data = metadata.xet_file_data
        file_size = metadata.size or 0

        if local_dir:
            dest_path = os.path.join(local_dir, filename)
        else:
            dest_path = os.path.join(tempfile.gettempdir(), filename)

        # Plan 2026-07-16: dedicated xet path (NO HTTP fallback). In this
        # environment (hf_xet 1.5.0) all xet download APIs are broken, so
        # ``download_file_xet_only`` raises TransferProgressError fast with a
        # clear message telling the caller to use use_xet=False. We do NOT
        # catch-and-fall-back to HTTP here — the user wants separate paths,
        # and a silent fallback would hide the xet failure.
        return download_file_xet_only(
            repo_id=repo_id,
            filename=filename,
            file_hash=getattr(xet_file_data, "file_hash", ""),
            file_size=file_size,
            dest_path=dest_path,
            xet_file_data=xet_file_data,
            token=self._token,
            repo_type=repo_type,
            revision=revision,
            endpoint=api.endpoint,
            event_queue=self.event_queue,  # type: ignore[attr-defined]
            transfer_id=transfer_id,
            report_interval=self._report_interval,  # type: ignore[attr-defined]
            is_cancelled=is_cancelled,
        )

    def _download_snapshot_xet(
        self,
        repo_id: str,
        allow_patterns: Optional[List[str]],
        ignore_patterns: Optional[List[str]],
        repo_type: str,
        revision: Optional[str],
        local_dir: Optional[str],
        transfer_id: str,
        is_cancelled: Callable[[], bool],
        force_download: bool = False,
        use_xet: bool = True,
    ) -> str:
        """Download repo snapshot via Xet in an isolated subprocess.

        Uses the proven ``download_snapshot_with_xet`` path, which
        invokes ``huggingface_hub.snapshot_download`` inside an isolated
        subprocess. The ``hf_xet`` C extension is therefore loaded
        only in the child process, which makes the transfer safely
        terminable without affecting the main process state.

        Step 8.5 of the modular refactor (2026-06-05) removed the
        earlier ``XetSession`` API variant of this function. See the
        in-body comment below for the rationale and the historical
        plan references.

        Args:
            repo_id: HuggingFace repository ID.
            allow_patterns: Optional list of glob patterns to include.
            ignore_patterns: Optional list of glob patterns to exclude.
            repo_type: Repository type (model/dataset/space).
            revision: Optional git revision.
            local_dir: Local directory to download files to.
            transfer_id: Pre-existing transfer ID.
            is_cancelled: Cancellation hook checked by the subprocess runner.
            force_download: Whether to force re-download even if files exist.
            use_xet: If False, the subprocess worker sets
                ``HF_HUB_DISABLE_XET=1`` before importing
                ``huggingface_hub``, forcing HTTP download inside
                the child. This allows runtime xet toggling without
                restarting the app.

        Returns:
            Path to the local directory containing downloaded files.

        Raises:
            TransferCancelledError: If the transfer is cancelled.
            TransferProgressError: If the download fails.
        """
        from ..download import download_snapshot_with_xet

        # Use the proven path: ``huggingface_hub.snapshot_download``
        # running in an isolated subprocess (so ``hf_xet`` is loaded
        # only in the child for safe termination).
        #
        # The new ``hf_xet.XetSession`` API was tried (see plan
        # ``2026-06-03-migrate-snapshot-to-xetsession-api.md``) but
        # had three critical correctness bugs:
        #   1. Nested folder layout in destination path
        #   2. Missing files (non-xet files and subdirectory contents)
        #   3. Zero-length partial files on cancel
        # The new API is fundamentally a single-batch primitive that
        # does not provide the mixed xet/non-xet routing, subdirectory
        # traversal, or cache-aware resume that snapshot downloads
        # require. See plan
        # ``2026-06-03-revert-broken-snapshot-xetsession-path.md``.
        #
        # ``use_xet`` is passed through so the child subprocess can
        # set ``HF_HUB_DISABLE_XET=1`` to force HTTP downloads for
        # the no-xet case (allows runtime xet toggling without
        # restarting the app).
        return download_snapshot_with_xet(
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
            is_cancelled=is_cancelled,
            force_download=force_download,
            use_xet=use_xet,
        )

    # ── Xet Upload Implementations ────────────────────────────────

    def _upload_file_xet(
        self,
        file_path: str,
        repo_id: str,
        path_in_repo: str,
        repo_type: str,
        revision: Optional[str],
        transfer_id: str,
        filename: str,
        is_cancelled: Callable[[], bool],
    ) -> str:
        from ..upload import upload_file_with_xet

        try:
            result = upload_file_with_xet(
                file_path=file_path,
                repo_id=repo_id,
                token=self._token,  # type: ignore[attr-defined]
                event_queue=self.event_queue,  # type: ignore[attr-defined]
                repo_type=repo_type,
                revision=revision,
                endpoint=self._endpoint,  # type: ignore[attr-defined]
                transfer_id=transfer_id,
                report_interval=self._report_interval,  # type: ignore[attr-defined]
                is_cancelled=is_cancelled,
            )
            return result.url or f"xet://{repo_id}/{result.filename}"
        except ImportError:
            return self._upload_file_lfs(  # type: ignore[attr-defined]
                file_path=file_path,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                repo_type=repo_type,
                revision=revision,
                transfer_id=transfer_id,
                filename=filename,
                is_cancelled=is_cancelled,
            )

    def _upload_bytes_xet(
        self,
        file_content: bytes,
        filename: str,
        repo_id: str,
        path_in_repo: str,
        repo_type: str,
        revision: Optional[str],
        transfer_id: str,
        is_cancelled: Callable[[], bool],
    ) -> str:
        from ..upload import upload_bytes_with_xet

        try:
            result = upload_bytes_with_xet(
                file_content=file_content,
                filename=filename,
                repo_id=repo_id,
                token=self._token,  # type: ignore[attr-defined]
                event_queue=self.event_queue,  # type: ignore[attr-defined]
                repo_type=repo_type,
                revision=revision,
                endpoint=self._endpoint,  # type: ignore[attr-defined]
                transfer_id=transfer_id,
                report_interval=self._report_interval,  # type: ignore[attr-defined]
                is_cancelled=is_cancelled,
            )
            return result.url or f"xet://{repo_id}/{result.filename}"
        except ImportError:
            return self._upload_bytes_via_temp(  # type: ignore[attr-defined]
                file_content=file_content,
                filename=filename,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                repo_type=repo_type,
                revision=revision,
                transfer_id=transfer_id,
                is_cancelled=is_cancelled,
            )

    # ── LFS Upload Implementations ────────────────────────────────

    def _upload_file_lfs(
        self,
        file_path: str,
        repo_id: str,
        path_in_repo: str,
        repo_type: str,
        revision: Optional[str],
        transfer_id: str,
        filename: str,
        is_cancelled: Callable[[], bool],
    ) -> str:
        from ..upload import upload_file as _upload_file

        return _upload_file(
            file_path=file_path,
            repo_id=repo_id,
            token=self._token,  # type: ignore[attr-defined]
            event_queue=self.event_queue,  # type: ignore[attr-defined]
            path_in_repo=path_in_repo,
            repo_type=repo_type,
            revision=revision,
            endpoint=self._endpoint,  # type: ignore[attr-defined]
            transfer_id=transfer_id,
            report_interval=self._report_interval,  # type: ignore[attr-defined]
            is_cancelled=is_cancelled,
        )

    def _upload_bytes_via_temp(
        self,
        file_content: bytes,
        filename: str,
        repo_id: str,
        path_in_repo: str,
        repo_type: str,
        revision: Optional[str],
        transfer_id: str,
        is_cancelled: Callable[[], bool],
    ) -> str:
        from ..upload import upload_bytes as _upload_bytes

        return _upload_bytes(
            file_content=file_content,
            filename=filename,
            repo_id=repo_id,
            token=self._token,  # type: ignore[attr-defined]
            event_queue=self.event_queue,  # type: ignore[attr-defined]
            path_in_repo=path_in_repo,
            repo_type=repo_type,
            revision=revision,
            endpoint=self._endpoint,  # type: ignore[attr-defined]
            transfer_id=transfer_id,
            report_interval=self._report_interval,  # type: ignore[attr-defined]
            is_cancelled=is_cancelled,
        )
