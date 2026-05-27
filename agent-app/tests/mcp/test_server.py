"""Tests for the MCP server using in-process fastmcp Client (no network required)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Client

from app.mcp.server import _validate_url, fetch_url, mcp, web_search


class TestMcpSchema:
    async def test_server_exposes_web_search_tool(self) -> None:
        async with Client(mcp) as client:
            tools = await client.list_tools()
            names = [t.name for t in tools]
        assert "web_search" in names

    async def test_server_exposes_fetch_url_tool(self) -> None:
        async with Client(mcp) as client:
            tools = await client.list_tools()
            names = [t.name for t in tools]
        assert "fetch_url" in names

    async def test_web_search_tool_has_query_parameter(self) -> None:
        async with Client(mcp) as client:
            tools = await client.list_tools()
        tool = next(t for t in tools if t.name == "web_search")
        assert "query" in tool.inputSchema.get("properties", {})

    async def test_fetch_url_tool_has_url_parameter(self) -> None:
        async with Client(mcp) as client:
            tools = await client.list_tools()
        tool = next(t for t in tools if t.name == "fetch_url")
        assert "url" in tool.inputSchema.get("properties", {})


class TestWebSearch:
    async def test_returns_string_with_results(self) -> None:
        fake_results = [
            {"title": "Python docs", "body": "Official Python documentation."},
            {"title": "PyPI", "body": "Package index."},
        ]
        with patch("app.mcp.server.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text.return_value = fake_results
            result = await web_search("python")
        assert isinstance(result, str)
        assert "Python docs" in result

    async def test_returns_no_results_message_when_empty(self) -> None:
        with patch("app.mcp.server.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text.return_value = []
            result = await web_search("zzz_nonexistent_xyz")
        assert "No results found" in result


def _make_httpx_mock(text: str) -> AsyncMock:
    """Return a mock httpx.AsyncClient whose get() returns a response with the given text."""
    mock_response = AsyncMock()
    mock_response.text = text
    mock_response.raise_for_status = MagicMock(return_value=None)
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_ctx.get = AsyncMock(return_value=mock_response)
    return mock_ctx


class TestValidateUrl:
    def test_allows_https(self) -> None:
        _validate_url("https://example.com/path")  # must not raise

    def test_allows_http(self) -> None:
        _validate_url("http://example.com/path")  # must not raise

    def test_rejects_file_scheme(self) -> None:
        with pytest.raises(ValueError, match="Disallowed URL scheme"):
            _validate_url("file:///etc/passwd")

    def test_rejects_ftp_scheme(self) -> None:
        with pytest.raises(ValueError, match="Disallowed URL scheme"):
            _validate_url("ftp://example.com/file")

    def test_rejects_loopback_ip(self) -> None:
        with pytest.raises(ValueError, match="Blocked non-public IP"):
            _validate_url("http://127.0.0.1/admin")

    def test_rejects_aws_metadata_ip(self) -> None:
        with pytest.raises(ValueError, match="Blocked non-public IP"):
            _validate_url("http://169.254.169.254/latest/meta-data/")

    def test_rejects_private_ip_10_range(self) -> None:
        with pytest.raises(ValueError, match="Blocked non-public IP"):
            _validate_url("http://10.0.0.1/internal")

    def test_rejects_localhost_hostname(self) -> None:
        with pytest.raises(ValueError, match="Blocked private hostname"):
            _validate_url("http://localhost/admin")

    def test_rejects_local_hostname(self) -> None:
        with pytest.raises(ValueError, match="Blocked private hostname"):
            _validate_url("http://myservice.local/api")


class TestFetchUrl:
    async def test_returns_string_content(self) -> None:
        with patch("app.mcp.server.httpx.AsyncClient", return_value=_make_httpx_mock("<html>Hello world</html>")):
            result = await fetch_url("https://example.com")
        assert isinstance(result, str)
        assert "Hello world" in result

    async def test_truncates_to_4000_chars(self) -> None:
        with patch("app.mcp.server.httpx.AsyncClient", return_value=_make_httpx_mock("x" * 10_000)):
            result = await fetch_url("https://example.com")
        assert len(result) <= 4000

    async def test_rejects_ssrf_url_before_making_request(self) -> None:
        with pytest.raises(ValueError, match="Blocked non-public IP"):
            await fetch_url("http://169.254.169.254/latest/meta-data/")
