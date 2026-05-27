"""Tests for output_guard node."""

from app.graph.nodes.output_guard import make_output_guard_node
from tests.graph.nodes.conftest import CONFIG, base_state, make_mock_llm


class TestOutputGuardNode:
    async def test_passes_clean_answer(self) -> None:
        llm = make_mock_llm('{"verdict": "safe", "reason": "Well-grounded answer."}')
        node = make_output_guard_node(llm)
        state = base_state(final_answer="Clean research answer.", search_results=["Source data."])
        result = await node(state, CONFIG)
        assert result["status"] == "done"
        assert result["final_answer"] == "Clean research answer."

    async def test_blocks_harmful_answer_with_safe_fallback(self) -> None:
        llm = make_mock_llm('{"verdict": "unsafe", "reason": "Contains misleading claims."}')
        node = make_output_guard_node(llm)
        state = base_state(final_answer="Dangerous answer.", search_results=["Some data."])
        result = await node(state, CONFIG)
        assert result["status"] == "blocked"
        assert "unable to produce" in result["final_answer"].lower()
        assert result["guard_reason"] == "Contains misleading claims."

    async def test_blocks_on_unparseable_response(self) -> None:
        llm = make_mock_llm("This looks okay to me.")
        node = make_output_guard_node(llm)
        result = await node(base_state(final_answer="some answer"), CONFIG)
        assert result["status"] == "blocked"

    async def test_dead_letter_on_llm_error(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from app.exceptions import LLMServiceUnavailableError

        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=LLMServiceUnavailableError("down"))
        node = make_output_guard_node(llm)
        result = await node(base_state(final_answer="some answer"), CONFIG)
        assert result["status"] == "dead_lettered"
        assert result["dead_letter"]["failed_node"] == "output_guard"
