"""Tests for ``hf_track.token.credentials`` -- XetCredentials dataclass.

Imports go through the public ``hf_track.token`` re-exports to keep
this test validating the shim layer as well as the underlying module.
"""

from __future__ import annotations

from hf_track.token import XetCredentials


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

    def test_dataclass_equality(self):
        """Two credentials with the same fields are equal."""
        c1 = XetCredentials(endpoint="https://x", token_info=("t", 1))
        c2 = XetCredentials(endpoint="https://x", token_info=("t", 1))
        assert c1 == c2
