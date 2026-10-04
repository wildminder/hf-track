"""XetTokenManager -- unified upload + download token resolution.

The ``XetTokenManager`` wraps the HuggingFace Xet authentication flow
for **both** uploads and downloads. It lazily initializes the
``HfApi`` instance, exposes fetch methods for both directions, and
produces token-refresher callables the Rust runtime can call when a
token expires.

The class is kept whole in this single module (not split into
``UploadTokenManager`` + ``DownloadTokenManager``) because:

- both paths share the same cached ``HfApi`` and HTTP headers
- the refresher-callable construction is symmetric across the two
  directions and benefits from being co-located for review
- callers consume a single ``XetTokenManager`` instance, never one or
  the other

The token lifecycle:
- **Uploads**: Use ``XetTokenType.WRITE`` to build a repo-level
  token-refresh URL.
- **Downloads**: Use file-level ``XetFileData`` to get a per-file
  ``refresh_route``.
- **Fetching**: Both paths GET that URL through ``huggingface_hub``'s
  shared HTTP session, which is exactly what the ``*_xet_connection_info``
  helpers did before ``huggingface_hub`` 1.23.0 removed them.
- **Token refresh**: Both paths support a ``token_refresher`` callable
  that the Rust runtime calls when the token expires.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

from .credentials import XetCredentials


def _fetch_xet_connection_info(refresh_route: str, headers: dict) -> Any:
    """GET a Xet token-refresh route and return the decoded payload.

    Replaces ``refresh_xet_connection_info`` /
    ``fetch_xet_connection_info_from_repo_info``, both removed in
    ``huggingface_hub`` 1.23.0. The response is the same JSON the removed
    helpers parsed: ``{"accessToken": ..., "exp": ..., "casUrl": ...}``.

    Args:
        refresh_route: Absolute URL the Hub serves a CAS token from.
        headers: Authorization headers for that request.

    Returns:
        The decoded JSON payload (a dict).
    """
    from huggingface_hub.utils._http import get_session

    response = get_session().get(refresh_route, headers=headers or None)
    response.raise_for_status()
    return response.json()


class XetTokenManager:
    """Manages Xet authentication tokens for uploads and downloads.

    This class wraps the complex HuggingFace Xet authentication flow
    into a simple interface. It handles:

    - Repo-level token acquisition for uploads (WRITE scope)
    - File-level token acquisition for downloads (READ scope)
    - Token refresher callable creation for Rust runtime

    Usage::

        from hf_track.token import XetTokenManager

        manager = XetTokenManager(token="hf_...")
        creds = manager.fetch_upload_credentials("username/repo")
        # creds.endpoint, creds.token_info, creds.token_refresher

    Args:
        token: HuggingFace API token (``hf_...``).
        endpoint: Optional custom HuggingFace API endpoint.
    """

    def __init__(self, token: Optional[str] = None, endpoint: Optional[str] = None):
        self._token = token
        self._endpoint = endpoint
        self._api = None
        self._headers = None

    def _ensure_api(self):
        """Lazily initialize the HfApi instance and headers."""
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self._token, endpoint=self._endpoint)
            self._headers = self._api._build_hf_headers()

    # ── Credential acquisition (single path for every direction) ──

    def _repo_refresh_route(self, repo_id: str, repo_type: str, revision: Optional[str]) -> str:
        """Build the repo-level token-refresh URL for an upload."""
        from huggingface_hub.utils._xet import (
            XetTokenType,
            xet_connection_info_refresh_url,
        )

        return xet_connection_info_refresh_url(
            token_type=XetTokenType.WRITE,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            endpoint=self._endpoint,
        )

    def _fetch(self, refresh_route: str) -> XetCredentials:
        """Exchange a refresh route for a populated ``XetCredentials``.

        The single place every public fetch method goes through, so a
        change to the token wire format happens once.
        """
        payload = _fetch_xet_connection_info(refresh_route, self._headers or {})

        return XetCredentials(
            endpoint=payload.get("casUrl") or "",
            token_info=(
                payload.get("accessToken") or "",
                payload.get("exp") or 0,
            ),
        )

    def fetch_upload_credentials(
        self,
        repo_id: str,
        repo_type: str = "model",
        revision: Optional[str] = None,
    ) -> XetCredentials:
        """Get credentials for uploading to a repo.

        Uses ``XetTokenType.WRITE`` to obtain a repo-level token
        with write access.

        Args:
            repo_id: Repository ID (e.g. ``"username/model"``).
            repo_type: Repository type (``"model"``, ``"dataset"``, ``"space"``).
            revision: Optional git revision.

        Returns:
            XetCredentials with endpoint, token_info, and token_refresher.
        """
        self._ensure_api()

        creds = self._fetch(
            self._repo_refresh_route(repo_id, repo_type, revision)
        )
        creds.token_refresher = self.fetch_upload_token_refresher(
            repo_id, repo_type, revision
        )
        return creds

    def fetch_upload_token_refresher(
        self,
        repo_id: str,
        repo_type: str = "model",
        revision: Optional[str] = None,
    ) -> Callable[[], Tuple[str, int]]:
        """Create a token refresher callable for uploads.

        The returned callable can be passed as ``token_refresher``
        to ``hf_xet.upload_files()`` or ``hf_xet.upload_bytes()``.
        The Rust runtime calls it when the current token expires.

        Args:
            repo_id: Repository ID.
            repo_type: Repository type.
            revision: Optional git revision.

        Returns:
            A callable that returns (access_token, expiration_unix_epoch).
        """
        self._ensure_api()
        refresh_route = self._repo_refresh_route(repo_id, repo_type, revision)

        def token_refresher():
            return self._fetch(refresh_route).token_info

        return token_refresher

    def fetch_download_credentials(self, xet_file_data) -> XetCredentials:
        """Get credentials for downloading a file.

        Uses file-level ``XetFileData`` to obtain a per-file token
        with read access.

        Args:
            xet_file_data: An ``XetFileData`` instance obtained from
                ``parse_xet_file_data_from_response()`` or from the
                repo's file metadata.

        Returns:
            XetCredentials with endpoint, token_info, and token_refresher.
        """
        self._ensure_api()

        creds = self._fetch(xet_file_data.refresh_route)
        creds.token_refresher = self.fetch_download_token_refresher(xet_file_data)
        return creds

    def fetch_download_token_refresher(
        self, xet_file_data
    ) -> Callable[[], Tuple[str, int]]:
        """Create a token refresher callable for downloads.

        Args:
            xet_file_data: An ``XetFileData`` instance.

        Returns:
            A callable that returns (access_token, expiration_unix_epoch).
        """
        self._ensure_api()
        refresh_route = xet_file_data.refresh_route

        def token_refresher():
            return self._fetch(refresh_route).token_info

        return token_refresher
