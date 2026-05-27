"""Tests for react_researcher node."""

from langchain_core.messages import AIMessage, ToolCall

from app.graph.nodes.react_researcher import make_react_researcher_node
from tests.graph.nodes.conftest import CONFIG, base_state, make_mock_llm


class TestReactResearcherNode:
    async def test_increments_react_steps(self) -> None:
        llm = make_mock_llm("Thinking...")
        node = make_react_researcher_node(llm)
        result = await node(base_state(react_steps=3), CONFIG)
        assert result["react_steps"] == 4

    async def test_appends_response_to_messages(self) -> None:
        llm = make_mock_llm("I gathered some context.")
        node = make_react_researcher_node(llm)
        result = await node(base_state(), CONFIG)
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "I gathered some context."

    async def test_response_with_tool_calls_is_preserved(self) -> None:
        tool_calls = [
            ToolCall(name="web_search", args={"query": "test"}, id="call_1"),
        ]
        response = AIMessage(content="", tool_calls=tool_calls)
        from unittest.mock import AsyncMock, MagicMock

        llm = MagicMock()
        llm.metadata = None
        llm.ainvoke = AsyncMock(return_value=response)
        llm.bind_tools = MagicMock(return_value=llm)

        node = make_react_researcher_node(llm)
        result = await node(base_state(), CONFIG)
        assert len(result["messages"][0].tool_calls) == 1
        assert result["messages"][0].tool_calls[0]["name"] == "web_search"
        assert result["messages"][0].tool_calls[0]["args"] == {"query": "test"}

    async def test_dead_letter_on_llm_error(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from app.exceptions import LLMServiceUnavailableError

        llm = MagicMock()
        llm.metadata = None
        llm.ainvoke = AsyncMock(side_effect=LLMServiceUnavailableError("down"))
        node = make_react_researcher_node(llm)
        result = await node(base_state(), CONFIG)
        assert result["status"] == "dead_lettered"
        assert result["dead_letter"]["failed_node"] == "react_researcher"
