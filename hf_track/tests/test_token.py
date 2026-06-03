"""Tests for hf_progress.token module."""

from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch

from hf_track.token import XetCredentials, XetTokenManager, is_xet_available, has_xet_session


class TestIsXetAvailable:
    """Tests for is_xet_available function.

    Uses ``importlib.util.find_spec`` instead of ``import hf_xet``,
    so we mock ``find_spec`` rather than ``builtins.__import__``.
    """

    @patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "0"})
    @patch("importlib.util.find_spec")
    def test_returns_true_when_xet_is_available_and_not_disabled(self, mock_find_spec):
        """Should return True if hf_xet is findable and not disabled."""
        mock_find_spec.return_value = MagicMock()  # non-None = found
        assert is_xet_available() is True
        mock_find_spec.assert_called_once_with("hf_xet")

    @patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "1"})
    def test_returns_false_when_xet_is_disabled_by_env_var(self):
        """Should return False if HF_HUB_DISABLE_XET is set."""
        assert is_xet_available() is False

    @patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "true"})
    def test_returns_false_when_xet_is_disabled_by_env_var_true(self):
        """Should return False if HF_HUB_DISABLE_XET is set to 'true'."""
        assert is_xet_available() is False

    @patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "yes"})
    def test_returns_false_when_xet_is_disabled_by_env_var_yes(self):
        """Should return False if HF_HUB_DISABLE_XET is set to 'yes'."""
        assert is_xet_available() is False

    @patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "FALSE"})  # Ensure case-insensitivity
    @patch("importlib.util.find_spec")
    def test_returns_true_when_xet_is_available_and_env_var_is_false(self, mock_find_spec):
        """Should return True if HF_HUB_DISABLE_XET is false (case-insensitive)."""
        mock_find_spec.return_value = MagicMock()  # non-None = found
        assert is_xet_available() is True

    @patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "0"})
    @patch("importlib.util.find_spec")
    def test_returns_false_when_xet_is_not_installed(self, mock_find_spec):
        """Should return False if hf_xet is not installed (find_spec returns None)."""
        mock_find_spec.return_value = None
        assert is_xet_available() is False
        mock_find_spec.assert_called_once_with("hf_xet")

    @patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "0"})
    def test_does_not_import_hf_xet_module(self):
        """is_xet_available must NOT load hf_xet into sys.modules.

        This is the core fix: using find_spec instead of import ensures
        the Rust .pyd extension is never loaded in the main process,
        preserving subprocess isolation for safe termination.
        """
        import sys
        # Remove hf_xet from sys.modules if it was previously loaded
        had_xet = "hf_xet" in sys.modules
        sys.modules.pop("hf_xet", None)

        # find_spec for hf_xet will return None (not installed in test env)
        is_xet_available()

        # hf_xet must NOT appear in sys.modules after the check
        assert "hf_xet" not in sys.modules, (
            "is_xet_available() must not import hf_xet into the main process"
        )

        # Restore if it was there before
        if had_xet:
            sys.modules["hf_xet"] = MagicMock()

    @patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "1"})
    @patch("importlib.util.find_spec")
    def test_does_not_call_find_spec_when_disabled(self, mock_find_spec):
        """Should not call find_spec at all when HF_HUB_DISABLE_XET is set."""
        is_xet_available()
        mock_find_spec.assert_not_called()

    @patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "0"})
    def test_returns_false_when_hf_xet_in_sys_modules_without_spec(self):
        """Returns False if hf_xet was injected into sys.modules without __spec__.

        Regression for test pollution: some tests (e.g. test_sse.py web app
        tests) import the web app which loads hf_xet. Other tests inject
        MagicMock placeholders into sys.modules["hf_xet"] for mocking.
        find_spec raises ValueError when __spec__ is missing; we must
        treat this as "not available" rather than crashing.
        """
        # Use a plain object with no __spec__ attribute to simulate a
        # module entry that lacks a valid import spec.
        class _Placeholder:
            pass

        placeholder = _Placeholder()
        with patch.dict("sys.modules", {"hf_xet": placeholder}):
            assert is_xet_available() is False


class TestXetCredentials:
    """Tests for XetCredentials dataclass."""

    def test_basic_creation(self):
        creds = XetCredentials(
            endpoint="https://xet.example.com",
            token_info=("token123", 1234567890),
        )
        assert creds.endpoint == "https://xet.example.com"
        assert creds.token_info == ("token123", 1234567890)
        assert creds.token_refresher is None

    def test_with_refresher(self):
        def refresher():
            return ("new_token", 1234567999)

        creds = XetCredentials(
            endpoint="https://xet.example.com",
            token_info=("token123", 1234567890),
            token_refresher=refresher,
        )
        assert creds.token_refresher is not None
        new_token, new_exp = creds.token_refresher()
        assert new_token == "new_token"
        assert new_exp == 1234567999


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


class TestHasXetSession:
    """Tests for has_xet_session function.

    The ``has_xet_session`` helper detects whether the new
    ``hf_xet.XetSession`` API is available (requires hf_xet >= 1.5.0).
    This is the ONLY API that gives smooth per-chunk progress on
    snapshot downloads. The old ``hf_xet.download_files`` API only
    fires its progress callback at file completion.
    """

    @patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "1"})
    def test_returns_false_when_xet_is_disabled_by_env(self):
        """Should return False if HF_HUB_DISABLE_XET is set."""
        assert has_xet_session() is False

    @patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "0"})
    @patch("importlib.util.find_spec")
    def test_returns_false_when_hf_xet_not_installed(self, mock_find_spec):
        """Should return False if hf_xet is not findable."""
        mock_find_spec.return_value = None
        assert has_xet_session() is False

    @patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "0"})
    @patch("importlib.util.find_spec")
    def test_returns_true_when_hf_xet_with_xetsession(self, mock_find_spec):
        """Should return True when hf_xet.XetSession is available."""
        # Mock the find_spec result for hf_xet (package spec)
        mock_spec = MagicMock()
        mock_spec.submodule_search_locations = ["some/path"]
        mock_find_spec.return_value = mock_spec
        # Mock hf_xet module to have XetSession attribute
        mock_hf_xet = MagicMock()
        mock_hf_xet.XetSession = MagicMock()
        with patch.dict("sys.modules", {"hf_xet": mock_hf_xet}):
            assert has_xet_session() is True

    @patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "0"})
    @patch("importlib.util.find_spec")
    def test_returns_false_when_hf_xet_without_xetsession(self, mock_find_spec):
        """Should return False when hf_xet is installed but XetSession is missing."""
        # Mock find_spec result for hf_xet
        mock_spec = MagicMock()
        mock_spec.submodule_search_locations = ["some/path"]
        mock_find_spec.return_value = mock_spec
        # Mock hf_xet module WITHOUT XetSession attribute (old version)
        mock_hf_xet = MagicMock(spec=[])  # no attributes
        del mock_hf_xet.XetSession  # ensure attribute doesn't exist
        with patch.dict("sys.modules", {"hf_xet": mock_hf_xet}):
            assert has_xet_session() is False

    @patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "0"})
    @patch("importlib.util.find_spec")
    def test_returns_false_when_import_raises(self, mock_find_spec):
        """Should return False (not crash) if importing hf_xet raises."""
        # find_spec returns a spec, but actual import fails
        mock_spec = MagicMock()
        mock_spec.submodule_search_locations = ["some/path"]
        mock_find_spec.return_value = mock_spec

        # Make `import hf_xet` fail by injecting a bad module
        class _BrokenModule:
            def __getattr__(self, name):
                raise ImportError("simulated import failure")

        with patch.dict("sys.modules", {"hf_xet": _BrokenModule()}):
            # Even if XetSession attribute access fails, function should return False
            assert has_xet_session() is False