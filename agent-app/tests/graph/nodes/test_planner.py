"""Tests for planner node: interrupt/approve/reject flow."""

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.graph.nodes.planner import make_planner_node
from app.graph.state import AgentState
from tests.graph.nodes.conftest import base_state, make_mock_llm


def _build_test_graph(llm):
    """Compile a minimal graph with only the planner node for testing interrupts."""
    planner_node = make_planner_node(llm)

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", END)
    return graph.compile(checkpointer=MemorySaver(), interrupt_before=["planner"])


class TestPlannerNode:
    async def test_emits_interrupt_and_resumes_approved(self) -> None:
        llm = make_mock_llm("1. Search for topic\n2. Analyse sources\n3. Summarise findings")
        graph = _build_test_graph(llm)
        config: RunnableConfig = {"configurable": {"thread_id": "planner-approve"}}

        # First invoke — hits interrupt_before planner, returns without running it
        await graph.ainvoke(base_state(), config)

        # Snapshot shows planner is next and graph is suspended
        snapshot = await graph.aget_state(config)
        assert bool(snapshot.next)
        assert "planner" in snapshot.next

        # Resume with approval
        result = await graph.ainvoke(Command(resume=True), config)
        assert result["plan_approved"] is True
        assert result["status"] == "searching"
        assert len(result["plan"]) > 0

    async def test_resume_rejected_sets_aborted(self) -> None:
        llm = make_mock_llm("1. Step one\n2. Step two")
        graph = _build_test_graph(llm)
        config: RunnableConfig = {"configurable": {"thread_id": "planner-reject"}}

        await graph.ainvoke(base_state(), config)
        result = await graph.ainvoke(Command(resume=False), config)

        assert result["plan_approved"] is False
        assert result["status"] == "aborted"

    async def test_plan_steps_parsed_from_llm_output(self) -> None:
        llm = make_mock_llm("1. Step one\n2. Step two\n3. Step three")
        graph = _build_test_graph(llm)
        config: RunnableConfig = {"configurable": {"thread_id": "planner-steps"}}

        await graph.ainvoke(base_state(), config)
        result = await graph.ainvoke(Command(resume=True), config)
        assert len(result["plan"]) == 3
