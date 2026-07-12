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

import httpx
from duckduckgo_search import DDGS
from fastmcp import FastMCP

from app.config import settings
from app.mcp.ssrf import validate_url_and_host

mcp = FastMCP("research-tools")
# Shared client so fetch_url/fact_check reuse one connection pool instead of
# paying a fresh TCP/TLS handshake per call under load.
_http_client = httpx.AsyncClient(follow_redirects=False)


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
    validate_url_and_host(url)
    response = await _http_client.get(url, timeout=timeout)
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
            validate_url_and_host(top_url)
            resp = await _http_client.get(top_url, timeout=10.0)
            resp.raise_for_status()
            extra = f"\n\nFull content from top source ({top_url}):\n{resp.text[:2000]}"
        except Exception:
            pass
    return snippets + extra


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(mcp.http_app(), host=settings.mcp_host, port=settings.mcp_port)
