"""Tests for DNS rebinding SSRF protection in validate_url_and_host."""

from unittest.mock import patch

import pytest

from app.mcp.ssrf import validate_url_and_host


class TestValidateUrlAndHost:
    def test_allows_public_url(self) -> None:
        # Should not raise — example.com resolves to a public IP.
        with patch("app.mcp.ssrf.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("93.184.216.34", 0))]
            result = validate_url_and_host("https://example.com/page")
        assert result == "https://example.com/page"

    def test_blocks_hostname_resolving_to_loopback(self) -> None:
        with patch("app.mcp.ssrf.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("127.0.0.1", 0))]
            with pytest.raises(ValueError, match="resolves to blocked IP"):
                validate_url_and_host("https://rebind.attacker.com/secret")

    def test_blocks_hostname_resolving_to_aws_metadata(self) -> None:
        with patch("app.mcp.ssrf.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("169.254.169.254", 0))]
            with pytest.raises(ValueError, match="resolves to blocked IP"):
                validate_url_and_host("https://rebind.attacker.com/meta")

    def test_blocks_hostname_resolving_to_private_10_range(self) -> None:
        with patch("app.mcp.ssrf.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(None, None, None, None, ("10.0.0.1", 0))]
            with pytest.raises(ValueError, match="resolves to blocked IP"):
                validate_url_and_host("https://internal.corp.com/api")

    def test_blocks_unresolvable_hostname(self) -> None:
        import socket

        with patch("app.mcp.ssrf.socket.getaddrinfo", side_effect=socket.gaierror("Name not found")):
            with pytest.raises(ValueError, match="Cannot resolve hostname"):
                validate_url_and_host("https://this-does-not-exist.invalid/page")

    def test_skips_dns_resolution_for_literal_public_ip(self) -> None:
        # Literal IPs validated by _validate_url; no getaddrinfo call needed.
        with patch("app.mcp.ssrf.socket.getaddrinfo") as mock_gai:
            validate_url_and_host("https://93.184.216.34/path")
        mock_gai.assert_not_called()

    def test_still_blocks_literal_private_ip(self) -> None:
        # _validate_url blocks private literal IPs before DNS resolution.
        with pytest.raises(ValueError, match="Blocked non-public IP"):
            validate_url_and_host("http://10.0.0.1/internal")

    def test_still_blocks_private_scheme(self) -> None:
        with pytest.raises(ValueError, match="Disallowed URL scheme"):
            validate_url_and_host("file:///etc/passwd")
