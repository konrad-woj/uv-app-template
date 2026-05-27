"""Tests for input_guard node."""

from app.graph.nodes.input_guard import make_input_guard_node
from tests.graph.nodes.conftest import CONFIG, base_state, make_mock_llm


class TestInputGuard:
    async def test_passes_safe_request(self) -> None:
        llm = make_mock_llm('{"verdict": "safe", "reason": "Valid research question."}')
        node = make_input_guard_node(llm)
        result = await node(base_state(), CONFIG)
        assert result["status"] == "planning"
        assert "guard_reason" not in result or result.get("guard_reason") is None

    async def test_blocks_unsafe_request(self) -> None:
        llm = make_mock_llm('{"verdict": "unsafe", "reason": "Request contains harmful content."}')
        node = make_input_guard_node(llm)
        result = await node(base_state(), CONFIG)
        assert result["status"] == "blocked"
        assert result["guard_reason"] == "Request contains harmful content."

    async def test_blocks_on_unparseable_llm_response(self) -> None:
        llm = make_mock_llm("I cannot determine the safety of this request.")
        node = make_input_guard_node(llm)
        result = await node(base_state(), CONFIG)
        assert result["status"] == "blocked"
        assert "guard_reason" in result

    async def test_dead_letter_on_llm_error(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from app.exceptions import LLMServiceUnavailableError

        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=LLMServiceUnavailableError("down"))
        node = make_input_guard_node(llm)
        result = await node(base_state(), CONFIG)
        assert result["status"] == "dead_lettered"
        assert result["dead_letter"]["failed_node"] == "input_guard"
