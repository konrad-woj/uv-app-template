"""Agent App — FastAPI entry point.

Lifecycle:
  startup  → load MCP tools, create AsyncPostgresSaver, compile graph.
  shutdown → checkpointer connection pool is closed via async context manager.

The compiled graph is stored on app.state.graph and injected into endpoints
via the get_graph() dependency (app/dependencies.py).

Quick start:
    # terminal 1: MCP server
    uv run python -m app.mcp.server

    # terminal 2: FastAPI app
    uv run python -m app
    # → http://localhost:8000/docs
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from logger import configure_logging, get_logger

from app.config import settings
from app.graph.mcp_client import load_mcp_tools
from app.graph.nodes._llm_invoke import NodeLLMConfig
from app.graph.workflow import compile_graph
from app.routers import router

configure_logging()
logger = get_logger(__name__)


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    logger.info("Agent App starting")

    mcp_tools = await load_mcp_tools(settings.mcp_server_url)
    logger.info("MCP tools loaded", tool_count=len(mcp_tools))

    node_llm_configs: dict[str, NodeLLMConfig] = {
        "input_guard": NodeLLMConfig(temperature=0.0),
        "planner": NodeLLMConfig(temperature=0.0),
        "react_researcher": NodeLLMConfig(temperature=0.0),
        "writer": NodeLLMConfig(temperature=0.3),
        "reflection": NodeLLMConfig(temperature=0.2),
        "output_guard": NodeLLMConfig(temperature=0.0),
    }

    async with AsyncPostgresSaver.from_conn_string(settings.db_uri) as checkpointer:
        await checkpointer.setup()
        app.state.graph = compile_graph(checkpointer, mcp_tools, node_llm_configs)
        logger.info("Graph compiled and ready")
        yield

    logger.info("Agent App shutting down")


app = FastAPI(
    title="Agent App API",
    description=(
        "LangGraph research-assistant demonstrating time-travel, interrupts, "
        "subgraphs, ReAct, MCP tools, and SSE token streaming."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
app.add_exception_handler(Exception, _unhandled_exception_handler)
