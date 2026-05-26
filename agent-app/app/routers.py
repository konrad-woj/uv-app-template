"""API routes for the research-assistant agent.

LangGraph thread and checkpoint model
--------------------------------------
Every graph invocation is tied to a *thread* identified by ``thread_id``.
The ``configurable`` dict inside ``RunnableConfig`` is how callers communicate
thread identity (and optionally a specific checkpoint) to the graph runtime:

    config = {"configurable": {"thread_id": "abc"}}

LangGraph persists a checkpoint after each node completes.  The checkpoint
stores the full ``AgentState`` at that point in time, keyed by thread_id.  On
the next ``ainvoke`` with the same thread_id, the graph loads the latest
checkpoint and merges the new input into it — that is how multi-turn
conversations accumulate messages without the client re-sending history.

Replay vs. fork
---------------
Passing ``checkpoint_id`` alongside ``thread_id`` tells LangGraph to restore
that specific historical state and continue from there:

  - ``ainvoke(None, config)``  — ``None`` input means "use the state already
    stored at this checkpoint; do not inject any new input."  The graph runs
    forward from that snapshot, producing the same result as the original run
    (unless the node logic itself is non-deterministic).

  - ``ainvoke({...new_input...}, checkpoint_config)`` — supplying new input at
    a historical checkpoint creates a *fork*: the new branch diverges from that
    point while the original branch remains intact in checkpoint history.

``snapshot.metadata["source"]`` values
---------------------------------------
  - ``"input"``  — carry-over state at the start of a new invoke (before
    ``__start__`` processes the new input).  Use as the fork point.
  - ``"loop"``   — state saved after a node completed.  Use for replay.
  - ``"update"`` — state after ``aupdate_state()`` was called externally.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from app.config import settings
from app.dependencies import get_graph
from app.models import ChatRequest, ChatResponse, CheckpointInfo, ReplayRequest

router = APIRouter()


@router.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok"}


@router.post("/v1/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    request: ChatRequest,
    graph: Annotated[CompiledStateGraph, Depends(get_graph)],
) -> ChatResponse:
    """Invoke the graph for a new turn in a conversation thread.

    On the first call for a thread_id, the graph starts from scratch.
    On subsequent calls, LangGraph loads the latest checkpoint for that thread
    and merges the new HumanMessage into the existing state — no client-side
    history management needed.
    """
    config: RunnableConfig = {
        "configurable": {"thread_id": request.thread_id},
        "recursion_limit": settings.max_pipeline_steps,
    }
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=request.message)], "status": "planning"},
        config,
    )
    return ChatResponse(
        thread_id=request.thread_id,
        status=result.get("status", "done"),
        final_answer=result.get("final_answer"),
    )


@router.get("/v1/threads/{thread_id}/history", response_model=list[CheckpointInfo], tags=["threads"])
async def get_history(
    thread_id: str,
    graph: Annotated[CompiledStateGraph, Depends(get_graph)],
) -> list[CheckpointInfo]:
    """Return all checkpoints for a thread, newest first.

    Each CheckpointInfo.checkpoint_id can be passed to POST /replay to
    re-execute the graph from that exact state snapshot.

    CheckpointInfo fields:
      source         — "input" | "loop" | "update" (see module docstring)
      next           — node(s) scheduled to run next; empty list means the
                       graph reached END at this checkpoint
      messages_count — total messages accumulated so far (useful for finding
                       fork/replay targets without fetching full state)
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    history = []
    async for snapshot in graph.aget_state_history(config):
        cfg = snapshot.config.get("configurable") or {}
        metadata = snapshot.metadata or {}
        history.append(
            CheckpointInfo(
                checkpoint_id=cfg.get("checkpoint_id", ""),
                step=metadata.get("step", 0),
                source=metadata.get("source", ""),
                next=list(snapshot.next),
                status=snapshot.values.get("status"),
                messages_count=len(snapshot.values.get("messages", [])),
            )
        )
    return history


@router.post("/v1/threads/{thread_id}/replay", response_model=ChatResponse, tags=["threads"])
async def replay(
    thread_id: str,
    request: ReplayRequest,
    graph: Annotated[CompiledStateGraph, Depends(get_graph)],
) -> ChatResponse:
    """Re-execute the graph from a historical checkpoint (time-travel).

    ``ainvoke(None, config)`` — passing None as the input is the LangGraph
    idiom for "restore state from this checkpoint and run forward without
    injecting new input."  The graph continues from the node listed in
    snapshot.next at that checkpoint.

    To *fork* instead of replay, call POST /v1/chat with a new message while
    pointing at a historical checkpoint_id — the new input diverges from that
    point and the original branch is preserved.
    """
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": request.checkpoint_id,
        },
        "recursion_limit": settings.max_pipeline_steps,
    }
    # None input = replay from checkpoint state; no new message injected.
    result = await graph.ainvoke(None, config)
    return ChatResponse(
        thread_id=thread_id,
        status=result.get("status", "done"),
        final_answer=result.get("final_answer"),
    )
