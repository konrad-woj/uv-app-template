"""Tests for planner node: interrupt/approve/reject flow and plan guard."""

from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.graph.nodes.planner import make_planner_node
from app.graph.state import AgentState
from tests.graph.nodes.conftest import base_state, make_mock_gliguard


def _make_two_call_llm(plan_response: str, guard_verdict: str = '{"verdict": "safe", "reason": "ok"}') -> MagicMock:
    """Mock LLM returning plan text on the first call and a guard verdict on the second."""
    llm = MagicMock()
    llm.metadata = None
    llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content=plan_response),
            AIMessage(content=guard_verdict),
        ]
    )
    return llm


def _build_test_graph(llm, gliguard):
    """Compile a minimal graph with only the planner node for testing interrupts."""
    planner_node = make_planner_node(llm, gliguard)

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", END)
    return graph.compile(checkpointer=MemorySaver(), interrupt_before=["planner"])


class TestPlannerNode:
    async def test_emits_interrupt_and_resumes_approved(self) -> None:
        llm = _make_two_call_llm("1. Search for topic\n2. Analyse sources\n3. Summarise findings")
        gliguard = make_mock_gliguard()
        graph = _build_test_graph(llm, gliguard)
        config: RunnableConfig = {"configurable": {"thread_id": "planner-approve"}}

        # First invoke — hits interrupt_before planner, returns without running it
        await graph.ainvoke(base_state(), config)

        # Snapshot shows planner is next and graph is suspended
        snapshot = await graph.aget_state(config)
        assert bool(snapshot.next)
        assert "planner" in snapshot.next

        # Resume with approval — planner runs, plan guard passes, interrupt fires with True
        result = await graph.ainvoke(Command(resume=True), config)
        assert result["plan_approved"] is True
        assert result["status"] == "researching"
        assert len(result["plan"]) > 0

    async def test_resume_rejected_sets_aborted(self) -> None:
        llm = _make_two_call_llm("1. Step one\n2. Step two")
        gliguard = make_mock_gliguard()
        graph = _build_test_graph(llm, gliguard)
        config: RunnableConfig = {"configurable": {"thread_id": "planner-reject"}}

        await graph.ainvoke(base_state(), config)
        result = await graph.ainvoke(Command(resume=False), config)

        assert result["plan_approved"] is False
        assert result["status"] == "aborted"

    async def test_plan_steps_parsed_from_llm_output(self) -> None:
        llm = _make_two_call_llm("1. Step one\n2. Step two\n3. Step three")
        gliguard = make_mock_gliguard()
        graph = _build_test_graph(llm, gliguard)
        config: RunnableConfig = {"configurable": {"thread_id": "planner-steps"}}

        await graph.ainvoke(base_state(), config)
        result = await graph.ainvoke(Command(resume=True), config)
        assert len(result["plan"]) == 3

    async def test_unsafe_plan_blocked_by_gliguard_no_interrupt(self) -> None:
        """When GLiGuard flags the plan text, planner returns blocked without interrupting."""
        llm = _make_two_call_llm("1. Hack the system\n2. Exfiltrate data")
        gliguard = make_mock_gliguard(blocked=True, reason="Plan contains dangerous instructions.")
        graph = _build_test_graph(llm, gliguard)
        config: RunnableConfig = {"configurable": {"thread_id": "planner-blocked-gliguard"}}

        await graph.ainvoke(base_state(), config)
        result = await graph.ainvoke(Command(resume=True), config)

        assert result["status"] == "blocked"
        assert "dangerous" in result["guard_reason"].lower()

    async def test_unsafe_plan_blocked_by_llm_guard_no_interrupt(self) -> None:
        """When LLM guard flags the plan as unsafe, planner returns blocked without interrupting."""
        llm = _make_two_call_llm(
            "1. Stalk target\n2. Find home address",
            '{"verdict": "unsafe", "reason": "Plan involves targeted surveillance."}',
        )
        gliguard = make_mock_gliguard()  # GLiGuard passes
        graph = _build_test_graph(llm, gliguard)
        config: RunnableConfig = {"configurable": {"thread_id": "planner-blocked-llm"}}

        await graph.ainvoke(base_state(), config)
        result = await graph.ainvoke(Command(resume=True), config)

        assert result["status"] == "blocked"
        assert "surveillance" in result["guard_reason"].lower()
