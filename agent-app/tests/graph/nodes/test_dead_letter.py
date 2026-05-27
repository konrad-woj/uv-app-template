"""Tests for _dead_letter.py: decorator, routing helper, and terminal node."""

from datetime import datetime

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt
from langgraph.types import Interrupt

from app.graph.nodes._dead_letter import DeadLetterInfo, after, dead_letter_node, with_dead_letter
from app.graph.state import AgentState


def _base_state(
    dead_letter: DeadLetterInfo | None = None,
    status: str = "planning",
) -> AgentState:
    return {
        "messages": [HumanMessage(content="test")],
        "plan": [],
        "plan_approved": False,
        "claims": [],
        "verification_results": [],
        "react_steps": 0,
        "draft_answer": "",
        "reflection_attempts": 0,
        "reflection_passed": False,
        "final_answer": "",
        "status": status,
        "guard_reason": None,
        "dead_letter": dead_letter,
    }


_CONFIG: RunnableConfig = {"configurable": {"thread_id": "test"}}


class TestWithDeadLetterDecorator:
    async def test_passes_through_on_success(self) -> None:
        @with_dead_letter("good_node")
        async def good_node(state: AgentState, config: RunnableConfig) -> dict:
            return {"status": "done"}

        result = await good_node(_base_state(), _CONFIG)
        assert result == {"status": "done"}

    async def test_catches_exception_and_populates_dead_letter(self) -> None:
        @with_dead_letter("bad_node")
        async def bad_node(state: AgentState, config: RunnableConfig) -> dict:
            raise ValueError("boom")

        result = await bad_node(_base_state(), _CONFIG)
        assert result["status"] == "dead_lettered"
        dl: DeadLetterInfo = result["dead_letter"]
        assert dl["failed_node"] == "bad_node"
        assert dl["error_type"] == "ValueError"
        assert dl["error_message"] == "boom"
        assert "ValueError" in dl["traceback"]
        assert dl["timestamp"]  # ISO-8601, non-empty

    async def test_dead_letter_timestamp_is_iso8601(self) -> None:
        @with_dead_letter("ts_node")
        async def ts_node(state: AgentState, config: RunnableConfig) -> dict:
            raise RuntimeError("ts")

        result = await ts_node(_base_state(), _CONFIG)
        # Should not raise
        datetime.fromisoformat(result["dead_letter"]["timestamp"])

    async def test_graph_interrupt_propagates_and_is_not_caught(self) -> None:
        @with_dead_letter("interrupt_node")
        async def interrupt_node(state: AgentState, config: RunnableConfig) -> dict:
            raise GraphInterrupt([Interrupt({"plan": ["step 1"]})])

        with pytest.raises(GraphInterrupt):
            await interrupt_node(_base_state(), _CONFIG)

    async def test_preserves_function_name(self) -> None:
        @with_dead_letter("named_node")
        async def my_node(state: AgentState, config: RunnableConfig) -> dict:
            return {}

        assert my_node.__name__ == "my_node"


class TestAfterRoutingHelper:
    def test_routes_to_next_when_no_dead_letter(self) -> None:
        route = after("writer")
        assert route(_base_state()) == "writer"

    def test_routes_to_dead_letter_when_field_is_set(self) -> None:
        route = after("writer")
        dl: DeadLetterInfo = {
            "failed_node": "planner",
            "error_type": "RuntimeError",
            "error_message": "oops",
            "traceback": "",
            "timestamp": "2024-01-01T00:00:00+00:00",
        }
        assert route(_base_state(dead_letter=dl)) == "dead_letter"

    def test_function_name_reflects_next_node(self) -> None:
        route = after("search_subgraph")
        assert "search_subgraph" in route.__name__


class TestDeadLetterNode:
    async def test_returns_empty_dict(self) -> None:
        dl: DeadLetterInfo = {
            "failed_node": "writer",
            "error_type": "IOError",
            "error_message": "io fail",
            "traceback": "",
            "timestamp": "2024-01-01T00:00:00+00:00",
        }
        state = _base_state(dead_letter=dl, status="dead_lettered")  # type: ignore[arg-type]
        result = await dead_letter_node(state, _CONFIG)
        assert result == {}
