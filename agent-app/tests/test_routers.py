"""Router-level tests for /health, /v1/chat, and /v1/threads/* endpoints.

Uses a standalone FastAPI app (no lifespan) with the graph dependency overridden
by a mock, so no Postgres or MCP server is required.
"""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.dependencies import get_graph
from app.routers import router

_app = FastAPI()
_app.include_router(router)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_graph() -> MagicMock:
    g = MagicMock()
    g.aget_state = AsyncMock()
    g.ainvoke = AsyncMock()
    return g


@pytest.fixture
async def client(mock_graph: MagicMock) -> AsyncGenerator[AsyncClient]:
    _app.dependency_overrides[get_graph] = lambda: mock_graph
    async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
        yield ac
    _app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot(
    next_nodes: tuple = (),
    values: dict | None = None,
    tasks: list | None = None,
    config: dict | None = None,
    metadata: dict | None = None,
) -> MagicMock:
    s = MagicMock()
    s.next = next_nodes
    s.values = values if values is not None else {}
    s.tasks = tasks or []
    s.config = config or {"configurable": {"checkpoint_id": "cp-1"}}
    s.metadata = metadata or {"step": 0, "source": "loop"}
    return s


def _interrupted_snapshot(plan_value: dict | None = None) -> MagicMock:
    raw = MagicMock()
    raw.value = plan_value or {"plan": ["Step 1", "Step 2"]}
    task = MagicMock()
    task.interrupts = [raw]
    return _snapshot(
        next_nodes=("planner",),
        values={"status": "planning"},
        tasks=[task],
    )


async def _async_history(*snaps: MagicMock):
    for s in snaps:
        yield s


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    async def test_returns_ok(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /v1/chat — fresh turn (not interrupted)
# ---------------------------------------------------------------------------


class TestChatFreshTurn:
    async def test_invokes_and_returns_result(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state = AsyncMock(return_value=_snapshot())
        mock_graph.ainvoke = AsyncMock(return_value={"status": "done", "final_answer": "Research result."})

        response = await client.post("/v1/chat", json={"thread_id": "t-1", "message": "Research AI"})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "done"
        assert data["final_answer"] == "Research result."
        assert data["is_interrupted"] is False

    async def test_thread_id_passed_in_config(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state = AsyncMock(return_value=_snapshot())
        mock_graph.ainvoke = AsyncMock(return_value={"status": "done"})

        await client.post("/v1/chat", json={"thread_id": "my-thread", "message": "hi"})

        config = mock_graph.aget_state.call_args_list[0][0][0]
        assert config["configurable"]["thread_id"] == "my-thread"

    async def test_guard_blocked_propagates_reason(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state = AsyncMock(return_value=_snapshot())
        mock_graph.ainvoke = AsyncMock(return_value={"status": "blocked", "guard_reason": "Off-topic request."})

        response = await client.post("/v1/chat", json={"thread_id": "t-2", "message": "bad"})

        data = response.json()
        assert data["status"] == "blocked"
        assert data["guard_reason"] == "Off-topic request."

    async def test_new_interrupt_after_invocation_sets_is_interrupted(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        """Graph suspends mid-run (planner fires interrupt); response reflects the new interrupt."""
        mock_graph.aget_state = AsyncMock(side_effect=[_snapshot(), _interrupted_snapshot({"plan": ["Step A"]})])
        mock_graph.ainvoke = AsyncMock(return_value={"status": "planning"})

        response = await client.post("/v1/chat", json={"thread_id": "t-3", "message": "Research topic"})

        data = response.json()
        assert data["is_interrupted"] is True
        assert data["interrupt_value"] == {"plan": ["Step A"]}

    async def test_missing_status_defaults_to_done(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state = AsyncMock(return_value=_snapshot())
        mock_graph.ainvoke = AsyncMock(return_value={})

        response = await client.post("/v1/chat", json={"thread_id": "t-4", "message": "hello"})

        assert response.json()["status"] == "done"


# ---------------------------------------------------------------------------
# POST /v1/chat — interrupted thread
# ---------------------------------------------------------------------------


class TestChatInterruptedThread:
    async def test_approve_none_returns_interrupted_without_invoking(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        mock_graph.aget_state = AsyncMock(return_value=_interrupted_snapshot())

        response = await client.post("/v1/chat", json={"thread_id": "t-5", "message": "whatever"})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "interrupted"
        assert data["is_interrupted"] is True
        mock_graph.ainvoke.assert_not_called()

    async def test_approve_none_returns_interrupt_value(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state = AsyncMock(return_value=_interrupted_snapshot({"plan": ["Search for X", "Analyse Y"]}))

        response = await client.post("/v1/chat", json={"thread_id": "t-6", "message": "hi"})

        assert response.json()["interrupt_value"] == {"plan": ["Search for X", "Analyse Y"]}

    async def test_approve_none_with_no_tasks_returns_null_interrupt_value(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        snap = _snapshot(next_nodes=("planner",), values={"status": "planning"}, tasks=[])
        mock_graph.aget_state = AsyncMock(return_value=snap)

        response = await client.post("/v1/chat", json={"thread_id": "t-7", "message": "hi"})

        assert response.json()["interrupt_value"] is None

    async def test_approve_true_resumes_with_command(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state = AsyncMock(side_effect=[_interrupted_snapshot(), _snapshot()])
        mock_graph.ainvoke = AsyncMock(return_value={"status": "done", "final_answer": "Answer"})

        response = await client.post("/v1/chat", json={"thread_id": "t-8", "message": "hi", "approve": True})

        assert response.status_code == 200
        invoked_input = mock_graph.ainvoke.call_args[0][0]
        assert isinstance(invoked_input, Command)
        assert invoked_input.resume is True

    async def test_approve_false_resumes_with_reject_command(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state = AsyncMock(side_effect=[_interrupted_snapshot(), _snapshot()])
        mock_graph.ainvoke = AsyncMock(return_value={"status": "aborted"})

        response = await client.post("/v1/chat", json={"thread_id": "t-9", "message": "hi", "approve": False})

        assert response.status_code == 200
        invoked_input = mock_graph.ainvoke.call_args[0][0]
        assert isinstance(invoked_input, Command)
        assert invoked_input.resume is False


# ---------------------------------------------------------------------------
# POST /v1/chat — request validation
# ---------------------------------------------------------------------------


class TestChatValidation:
    async def test_blank_message_returns_422(self, client: AsyncClient) -> None:
        response = await client.post("/v1/chat", json={"thread_id": "t-1", "message": "   "})
        assert response.status_code == 422

    async def test_blank_thread_id_returns_422(self, client: AsyncClient) -> None:
        response = await client.post("/v1/chat", json={"thread_id": "   ", "message": "hello"})
        assert response.status_code == 422

    async def test_missing_message_returns_422(self, client: AsyncClient) -> None:
        response = await client.post("/v1/chat", json={"thread_id": "t-1"})
        assert response.status_code == 422

    async def test_missing_thread_id_returns_422(self, client: AsyncClient) -> None:
        response = await client.post("/v1/chat", json={"message": "hello"})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /v1/threads/{thread_id}/history
# ---------------------------------------------------------------------------


class TestHistory:
    async def test_empty_history_returns_empty_list(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state_history = MagicMock(return_value=_async_history())

        response = await client.get("/v1/threads/t-1/history")

        assert response.status_code == 200
        assert response.json() == []

    async def test_returns_correct_checkpoint_info_shape(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        snap = _snapshot(
            next_nodes=("writer",),
            values={"status": "writing", "messages": [HumanMessage("hi")]},
            config={"configurable": {"checkpoint_id": "cp-abc"}},
            metadata={"step": 3, "source": "loop"},
        )
        mock_graph.aget_state_history = MagicMock(return_value=_async_history(snap))

        response = await client.get("/v1/threads/t-1/history")

        assert response.status_code == 200
        items = response.json()
        assert len(items) == 1
        item = items[0]
        assert item["checkpoint_id"] == "cp-abc"
        assert item["step"] == 3
        assert item["source"] == "loop"
        assert item["next"] == ["writer"]
        assert item["status"] == "writing"
        assert item["messages_count"] == 1

    async def test_returns_all_snapshots_in_order(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        snaps = [
            _snapshot(
                config={"configurable": {"checkpoint_id": f"cp-{i}"}},
                metadata={"step": i, "source": "loop"},
            )
            for i in range(3)
        ]
        mock_graph.aget_state_history = MagicMock(return_value=_async_history(*snaps))

        response = await client.get("/v1/threads/t-1/history")

        items = response.json()
        assert len(items) == 3
        assert [item["checkpoint_id"] for item in items] == ["cp-0", "cp-1", "cp-2"]

    async def test_passes_thread_id_in_config(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state_history = MagicMock(return_value=_async_history())

        await client.get("/v1/threads/specific-thread/history")

        config = mock_graph.aget_state_history.call_args[0][0]
        assert config["configurable"]["thread_id"] == "specific-thread"


# ---------------------------------------------------------------------------
# POST /v1/threads/{thread_id}/replay
# ---------------------------------------------------------------------------


class TestReplay:
    async def test_invokes_with_none_input_and_correct_config(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.ainvoke = AsyncMock(return_value={"status": "done", "final_answer": "Replayed."})

        await client.post("/v1/threads/t-1/replay", json={"checkpoint_id": "cp-xyz"})

        args, _ = mock_graph.ainvoke.call_args
        assert args[0] is None  # None = replay from stored state, no new input
        assert args[1]["configurable"]["thread_id"] == "t-1"
        assert args[1]["configurable"]["checkpoint_id"] == "cp-xyz"

    async def test_returns_result_with_correct_thread_id(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.ainvoke = AsyncMock(return_value={"status": "done", "final_answer": "The replayed answer."})

        response = await client.post("/v1/threads/t-2/replay", json={"checkpoint_id": "cp-1"})

        data = response.json()
        assert data["thread_id"] == "t-2"
        assert data["status"] == "done"
        assert data["final_answer"] == "The replayed answer."

    async def test_missing_checkpoint_id_returns_422(self, client: AsyncClient) -> None:
        response = await client.post("/v1/threads/t-1/replay", json={})
        assert response.status_code == 422
