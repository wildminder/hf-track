"""Tests for ``hf_track.token.manager`` -- XetTokenManager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hf_track.token import XetTokenManager


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
            # Mock at the huggingface_hub.utils._xet module level
            with patch(
                "huggingface_hub.utils._xet.fetch_xet_connection_info_from_repo_info",
                return_value=mock_connection_info,
            ) as mock_fetch:
                creds = manager.fetch_upload_credentials("user/repo")

                assert creds.endpoint == "https://xet.example.com"
                assert creds.token_info == ("xet_token_123", 1234567890)
                assert creds.token_refresher is not None
                mock_fetch.assert_called_once()

    def test_fetch_upload_token_refresher(self):
        """Token refresher should return a callable."""
        manager = XetTokenManager(token="hf_test")

        mock_connection_info = MagicMock()
        mock_connection_info.access_token = "refreshed_token"
        mock_connection_info.expiration_unix_epoch = 9999999999

        with patch.object(manager, "_ensure_api"):
            manager._headers = {"Authorization": "Bearer hf_test"}
            with patch(
                "huggingface_hub.utils._xet.fetch_xet_connection_info_from_repo_info",
                return_value=mock_connection_info,
            ):
                refresher = manager.fetch_upload_token_refresher("user/repo")
                assert callable(refresher)

                token, exp = refresher()
                assert token == "refreshed_token"
                assert exp == 9999999999

    def test_fetch_download_credentials(self):
        """fetch_download_credentials should return XetCredentials."""
        manager = XetTokenManager(token="hf_test")

        mock_connection_info = MagicMock()
        mock_connection_info.endpoint = "https://xet.example.com"
        mock_connection_info.access_token = "dl_token_123"
        mock_connection_info.expiration_unix_epoch = 1234567890

        mock_file_data = MagicMock()

        with patch.object(manager, "_ensure_api"):
            manager._headers = {"Authorization": "Bearer hf_test"}
            with patch(
                "huggingface_hub.utils._xet.refresh_xet_connection_info",
                return_value=mock_connection_info,
            ) as mock_refresh:
                creds = manager.fetch_download_credentials(mock_file_data)

                assert creds.endpoint == "https://xet.example.com"
                assert creds.token_info == ("dl_token_123", 1234567890)
                assert creds.token_refresher is not None
                mock_refresh.assert_called_once_with(
                    file_data=mock_file_data,
                    headers={"Authorization": "Bearer hf_test"},
                )
