"""Shared fixtures and helpers for graph/nodes tests."""

from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.graph.nodes._dead_letter import DeadLetterInfo
from app.graph.state import AgentState


def base_state(
    messages: list[AnyMessage] | None = None,
    plan: list[str] | None = None,
    status: str = "planning",
    search_results: list[str] | None = None,
    react_steps: int = 0,
    draft_answer: str = "",
    final_answer: str = "",
    reflection_attempts: int = 0,
    reflection_passed: bool = False,
    plan_approved: bool = False,
    guard_reason: str | None = None,
    dead_letter: DeadLetterInfo | None = None,
) -> AgentState:
    return {
        "messages": messages if messages is not None else [HumanMessage(content="test question")],
        "plan": plan if plan is not None else [],
        "plan_approved": plan_approved,
        "search_results": search_results if search_results is not None else [],
        "react_steps": react_steps,
        "draft_answer": draft_answer,
        "reflection_attempts": reflection_attempts,
        "reflection_passed": reflection_passed,
        "final_answer": final_answer,
        "status": status,
        "guard_reason": guard_reason,
        "dead_letter": dead_letter,
    }


def make_mock_llm(response_content: str, tool_calls: list | None = None) -> MagicMock:
    """Return a mock BaseChatModel that returns a fixed AIMessage."""
    response = AIMessage(content=response_content, tool_calls=tool_calls or [])
    llm = MagicMock()
    llm.metadata = None  # ensures (llm.metadata or {}) falls back to settings in llm_invoke
    llm.ainvoke = AsyncMock(return_value=response)
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


CONFIG: RunnableConfig = {"configurable": {"thread_id": "test-node"}}
