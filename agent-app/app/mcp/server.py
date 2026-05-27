"""MCP tool server: exposes web_search and fetch_url for the ReAct researcher.

Serves on http://localhost:8001 by default (AGENT_MCP_SERVER_URL).

Run standalone:
    uv run python -m app.mcp.server      # default port 8001
    uv run task mcp                      # same via taskipy

Tools exposed:
    web_search(query)  — DuckDuckGo text search; returns a formatted snippet list.
    fetch_url(url)     — HTTP GET + return text content (first 4000 chars).
"""

import asyncio
import ipaddress
from urllib.parse import urlparse

import httpx
from duckduckgo_search import DDGS
from fastmcp import FastMCP

mcp = FastMCP("research-tools")

_ALLOWED_SCHEMES = {"http", "https"}


def _validate_url(url: str) -> None:
    """Reject non-HTTP(S) schemes and private/loopback/link-local hosts.

    Prevents SSRF: the LLM can be prompt-injected via search results to call
    fetch_url with internal addresses (e.g. 169.254.169.254 AWS metadata).
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"Disallowed URL scheme: {parsed.scheme!r}")
    host = parsed.hostname or ""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        # Not an IP literal — check private hostname patterns
        if host == "localhost" or host.endswith(".local") or host.endswith(".internal"):
            raise ValueError(f"Blocked private hostname: {host}")  # noqa: B904
        return
    if not addr.is_global:
        raise ValueError(f"Blocked non-global IP: {host}")


@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo and return a summary of the top results.

    Args:
        query: The search query string.
        max_results: The maximum number of results to return.
    """
    results = await asyncio.to_thread(DDGS().text, query, max_results=max_results)
    if not results:
        return f"No results found for: {query}"
    snippets = [f"- {r['title']}: {r['body']}" for r in results]
    return "\n".join(snippets)


@mcp.tool()
async def fetch_url(url: str, max_char: int = 4000, timeout: float = 15.0) -> str:
    """Fetch the text content of a URL.

    Args:
        url: The URL to fetch.
        max_char: The maximum number of characters to fetch.
        timeout: The timeout in seconds.
    """
    _validate_url(url)
    async with httpx.AsyncClient(follow_redirects=False, timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text[:max_char]


if __name__ == "__main__":
    import uvicorn

    from app.config import settings

    uvicorn.run(mcp.http_app(), host=settings.mcp_host, port=settings.mcp_port)
