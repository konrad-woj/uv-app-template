"""MCP tool server: exposes web_search, fetch_url, and fact_check for the ReAct researcher.

Serves on http://localhost:8001 by default (AGENT_MCP_SERVER_URL).

Run standalone:
    uv run python -m app.mcp.server      # default port 8001
    uv run task mcp                      # same via taskipy

Tools exposed:
    web_search(query)   — DuckDuckGo text search; returns a formatted snippet list.
    fetch_url(url)      — HTTP GET + return text content (first 4000 chars).
    fact_check(claim)   — DuckDuckGo search + top-result fetch for claim verification.
"""

import asyncio
import ipaddress
from urllib.parse import urlparse

import httpx
from duckduckgo_search import DDGS
from fastmcp import FastMCP

from app.config import settings

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
    if (
        not addr.is_global
        or addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
    ):
        raise ValueError(f"Blocked non-public IP: {host}")


@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo and return a summary of the top results.

    Args:
        query: The search query string.
        max_results: The maximum number of results to return.
    """
    results = await asyncio.to_thread(DDGS().text, query, max_results=min(max_results, settings.web_search_max_results))
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


@mcp.tool()
async def fact_check(claim: str) -> str:
    """Search for evidence supporting or refuting a factual claim.

    Runs a DuckDuckGo search for the claim and fetches the full content of the
    top result. Returns snippets from all results plus the top-source content.

    Args:
        claim: The specific factual claim to verify.
    """
    results = await asyncio.to_thread(DDGS().text, f"fact check: {claim}", max_results=3)
    if not results:
        return f"No evidence found for: {claim}"
    snippets = "\n".join(f"- {r['title']}: {r['body']}" for r in results)
    top_url = results[0].get("href", "")
    extra = ""
    if top_url:
        try:
            _validate_url(top_url)
            async with httpx.AsyncClient(follow_redirects=False, timeout=10.0) as client:
                resp = await client.get(top_url)
                resp.raise_for_status()
                extra = f"\n\nFull content from top source ({top_url}):\n{resp.text[:2000]}"
        except Exception:
            pass
    return snippets + extra


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(mcp.http_app(), host=settings.mcp_host, port=settings.mcp_port)
