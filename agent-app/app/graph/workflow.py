"""Graph assembly for the research-assistant agent.

Exposes two functions:
  create_graph()   — returns the uncompiled StateGraph (used by LangGraph Studio).
  compile_graph()  — compiles with a checkpointer (used by main.py lifespan).

Phase 1 graph: START → planner → writer → END (stub nodes, no LLM).

Phase 2 will replace stub nodes with the full node set and wire react_condition
(with the MAX_REACT_STEPS ceiling) between react_researcher and tools/writer.
"""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config import settings
from app.graph.state import AgentState


def react_condition(state: AgentState) -> Literal["tools", "writer"]:
    """Route after react_researcher: loop back to tools or proceed to writer.

    Exits to writer when the model stops emitting tool_calls OR when
    react_steps reaches the configured ceiling (AGENT_MAX_REACT_STEPS).
    The ceiling prevents runaway tool loops that never self-terminate.
    """
    last = state["messages"][-1]
    ceiling_hit = state.get("react_steps", 0) >= settings.max_react_steps
    return "tools" if getattr(last, "tool_calls", None) and not ceiling_hit else "writer"


async def planner_node(state: AgentState, config: RunnableConfig) -> dict:
    last_human = next(
        (msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        None,
    )
    content = last_human.content if last_human else ""
    plan = [f"Research: {content}"]
    return {
        "plan": plan,
        "status": "searching",
        "messages": [AIMessage(content=f"Plan created for: {content}")],
    }


async def writer_node(state: AgentState, config: RunnableConfig) -> dict:
    plan_summary = " | ".join(state.get("plan", []))  # type: ignore[arg-type]
    answer = f"Draft answer for: {plan_summary}"
    return {
        "draft_answer": answer,
        "final_answer": answer,
        "status": "done",
        "messages": [AIMessage(content=answer)],
    }


def _build_graph() -> StateGraph:
    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("writer", writer_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "writer")
    graph.add_edge("writer", END)
    return graph


def create_graph() -> StateGraph:
    """Return the uncompiled StateGraph for LangGraph Studio."""
    return _build_graph()


def compile_graph(checkpointer: AsyncPostgresSaver) -> CompiledStateGraph:
    """Compile the graph with a Postgres checkpointer for production use."""
    return _build_graph().compile(checkpointer=checkpointer)
