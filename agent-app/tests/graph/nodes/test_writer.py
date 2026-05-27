"""Tests for writer node."""

from app.graph.nodes.writer import make_writer_node
from tests.graph.nodes.conftest import CONFIG, base_state, make_mock_llm


class TestWriterNode:
    async def test_drafts_answer_from_context(self) -> None:
        llm = make_mock_llm("This is a comprehensive answer to the question.")
        node = make_writer_node(llm)
        state = base_state(
            plan=["1. Search for X", "2. Analyse Y"],
            search_results=["Result A", "Result B"],
        )
        result = await node(state, CONFIG)
        assert result["draft_answer"] == "This is a comprehensive answer to the question."
        assert result["status"] == "reflecting"

    async def test_dead_letter_on_llm_error(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from app.exceptions import LLMServiceUnavailableError

        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=LLMServiceUnavailableError("down"))
        node = make_writer_node(llm)
        result = await node(base_state(), CONFIG)
        assert result["status"] == "dead_lettered"
        assert result["dead_letter"]["failed_node"] == "writer"
