"""Search subgraph: fan-out/fan-in using the Send API.

Demonstrates:
  - Fan-out: ``route_to_searchers`` returns one ``Send("searcher", ...)`` per plan step.
    LangGraph spawns each branch in parallel.
  - Fan-in: ``SearchState.results`` carries ``Annotated[list[str], operator.add]``.
    LangGraph calls operator.add to merge results from all parallel branches before
    the subgraph exits.

The subgraph is invoked from the parent graph via a wrapper node (``run_search``)
in workflow.py that maps AgentState keys ↔ SearchState keys.  Direct add_node of
the compiled subgraph is not used because the state schemas are disjoint.

SearchState.query is the per-branch key injected by Send — it is NOT part of the
initial subgraph input and is only set on the individual searcher nodes.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send


class SearchState(TypedDict):
    queries: list[str]  # fan-out source: one query per plan step
    query: str  # per-branch: set by Send, consumed by searcher_node
    results: Annotated[list[str], operator.add]  # fan-in target


async def searcher_node(state: SearchState, config: RunnableConfig) -> dict:
    """Execute a single search query; result is accumulated via operator.add."""
    _ = config
    query = state.get("query", "")
    result = f"[Search result for: {query}]"
    return {"results": [result]}


def route_to_searchers(state: SearchState) -> list[Send]:
    """Fan-out: spawn one searcher per query in the plan."""
    return [Send("searcher", {"queries": [], "query": q, "results": []}) for q in state["queries"]]


def _build_search_graph() -> StateGraph:
    graph: StateGraph = StateGraph(SearchState)
    graph.add_node("router", lambda state: state)
    graph.add_node("searcher", searcher_node)
    graph.add_conditional_edges("router", route_to_searchers)
    graph.add_edge(START, "router")
    graph.add_edge("searcher", END)
    return graph


search_subgraph: CompiledStateGraph = _build_search_graph().compile()
