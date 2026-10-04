"""Tests for ``hf_track.token.manager`` -- XetTokenManager."""

from __future__ import annotations

import ast
import pathlib
from unittest.mock import MagicMock, patch

from hf_track.token import XetTokenManager

MANAGER_SRC = (
    pathlib.Path(__file__).parents[2] / "src" / "hf_track" / "token" / "manager.py"
)


class TestXetTokenManager:
    """Tests for XetTokenManager."""

    def test_init(self):
        manager = XetTokenManager(token="hf_test123")
        assert manager._token == "hf_test123"
        assert manager._endpoint is None
        assert manager._api is None  # Lazy init

    def test_init_with_endpoint(self):
        manager = XetTokenManager(
            token="hf_test", endpoint="https://custom.api"
        )
        assert manager._endpoint == "https://custom.api"

    def test_ensure_api_lazy_init(self):
        """_ensure_api should lazily create HfApi instance."""
        manager = XetTokenManager(token="hf_test")
        assert manager._api is None

        # Mock HfApi at the huggingface_hub module level
        with patch("huggingface_hub.HfApi") as mock_api_class:
            mock_instance = MagicMock()
            mock_instance._build_hf_headers.return_value = {
                "Authorization": "Bearer hf_test"
            }
            mock_api_class.return_value = mock_instance

            manager._ensure_api()
            assert manager._api is not None
            mock_api_class.assert_called_once_with(
                token="hf_test", endpoint=None
            )

    def test_fetch_upload_credentials(self):
        """fetch_upload_credentials should return XetCredentials."""
        manager = XetTokenManager(token="hf_test")

        mock_connection_info = MagicMock()
        mock_connection_info.endpoint = "https://xet.example.com"
        mock_connection_info.access_token = "xet_token_123"
        mock_connection_info.expiration_unix_epoch = 1234567890

        with patch.object(manager, "_ensure_api"):
            manager._headers = {"Authorization": "Bearer hf_test"}
            # Mock at the token-fetch helper the manager routes through.
            with patch(
                "hf_track.token.manager._fetch_xet_connection_info",
                return_value={
                    "casUrl": "https://xet.example.com",
                    "accessToken": "xet_token_123",
                    "exp": 1234567890,
                },
            ) as mock_fetch:
                creds = manager.fetch_upload_credentials("user/repo")

                assert creds.endpoint == "https://xet.example.com"
                assert creds.token_info == ("xet_token_123", 1234567890)
                assert creds.token_refresher is not None
                mock_fetch.assert_called_once()
                route, headers = mock_fetch.call_args[0]
                assert route.endswith("/api/models/user/repo/xet-write-token/None")
                assert headers == {"Authorization": "Bearer hf_test"}

    def test_fetch_upload_token_refresher(self):
        """Token refresher should return a callable."""
        manager = XetTokenManager(token="hf_test")

        with patch.object(manager, "_ensure_api"):
            manager._headers = {"Authorization": "Bearer hf_test"}
            with patch(
                "hf_track.token.manager._fetch_xet_connection_info",
                return_value={"accessToken": "refreshed_token", "exp": 9999999999},
            ):
                refresher = manager.fetch_upload_token_refresher("user/repo")
                assert callable(refresher)

                token, exp = refresher()
                assert token == "refreshed_token"
                assert exp == 9999999999

    def test_fetch_download_credentials(self):
        """fetch_download_credentials should return XetCredentials."""
        manager = XetTokenManager(token="hf_test")

        mock_file_data = MagicMock(refresh_route="https://hub/xet-read-token/main")

        with patch.object(manager, "_ensure_api"):
            manager._headers = {"Authorization": "Bearer hf_test"}
            with patch(
                "hf_track.token.manager._fetch_xet_connection_info",
                return_value={
                    "casUrl": "https://xet.example.com",
                    "accessToken": "dl_token_123",
                    "exp": 1234567890,
                },
            ) as mock_fetch:
                creds = manager.fetch_download_credentials(mock_file_data)

                assert creds.endpoint == "https://xet.example.com"
                assert creds.token_info == ("dl_token_123", 1234567890)
                assert creds.token_refresher is not None
                assert mock_fetch.call_args[0] == (
                    "https://hub/xet-read-token/main",
                    {"Authorization": "Bearer hf_test"},
                )

    def test_fetch_download_token_refresher(self):
        """Download refresher re-fetches the file's refresh route."""
        manager = XetTokenManager(token="hf_test")
        mock_file_data = MagicMock(refresh_route="https://hub/xet-read-token/main")

        with patch.object(manager, "_ensure_api"):
            manager._headers = {}
            with patch(
                "hf_track.token.manager._fetch_xet_connection_info",
                return_value={"accessToken": "dl_token_456", "exp": 42},
            ):
                refresher = manager.fetch_download_token_refresher(mock_file_data)
                assert refresher() == ("dl_token_456", 42)

    def test_manager_source_has_no_removed_api_import(self):
        """No removed hub API survives as live code in the manager.

        Both names may still be *mentioned* in the module docstring, which
        explains what replaced them -- so this walks the AST rather than
        grepping the text.
        """
        tree = ast.parse(MANAGER_SRC.read_text(encoding="utf-8"))
        removed = {
            "refresh_xet_connection_info",
            "fetch_xet_connection_info_from_repo_info",
        }
        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                offenders += [a.name for a in node.names if a.name in removed]
            elif isinstance(node, ast.Import):
                offenders += [a.name for a in node.names if a.name in removed]
            elif isinstance(node, ast.Name) and node.id in removed:
                offenders.append(node.id)
            elif isinstance(node, ast.Attribute) and node.attr in removed:
                offenders.append(node.attr)
        assert not offenders, (
            "removed huggingface_hub API still referenced in manager.py: "
            f"{sorted(set(offenders))}"
        )
