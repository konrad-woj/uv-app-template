"""ReAct researcher node: model ↔ ToolNode loop using MCP tools.

The model is called with a list of available MCP tools bound via bind_tools().
If the model emits tool_calls, the graph routes to ToolNode which executes them
and appends ToolMessages to state. The loop continues until the model stops
calling tools OR react_steps reaches the MAX_REACT_STEPS ceiling (set via
AGENT_MAX_REACT_STEPS).

react_condition() in workflow.py implements the routing — this node only
does the single model invocation.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool

from app.graph.nodes._dead_letter import with_dead_letter
from app.graph.nodes._llm_invoke import llm_invoke_with_retry

if TYPE_CHECKING:
    from app.graph.state import AgentState


def make_react_researcher_node(llm_with_tools: BaseChatModel | Runnable) -> Callable:  # type: ignore[type-arg]
    """Return an async react_researcher node with tools already bound to the LLM.

    Args:
        llm_with_tools: LLM with MCP tools bound via llm.bind_tools(mcp_tools).
    """

    @with_dead_letter("react_researcher")
    async def react_researcher(state: "AgentState", config: RunnableConfig) -> dict:
        response = await llm_invoke_with_retry(llm_with_tools, state["messages"], config)  # type: ignore[arg-type]
        return {"messages": [response], "react_steps": state.get("react_steps", 0) + 1}

    return react_researcher


def make_react_researcher_node_from_llm(llm: BaseChatModel, mcp_tools: list[BaseTool]) -> Callable:
    """Convenience factory: bind tools to the LLM then return the node.

    Args:
        llm: Base LLM without tools.
        mcp_tools: List of MCP BaseTool instances to bind.
    """
    llm_with_tools = llm.bind_tools(mcp_tools)
    return make_react_researcher_node(llm_with_tools)
