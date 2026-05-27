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
checkpoint and merges the new HumanMessage into the existing state — that is
how multi-turn conversations accumulate messages without the client re-sending
history.

Interrupt / resume flow
-----------------------
When the planner calls interrupt(), the graph suspends and checkpoints. The
``/v1/chat`` endpoint detects this via ``snapshot.next``:
  - ``bool(snapshot.next)`` is True → graph is suspended at an interrupt.
  - Response includes ``is_interrupted=True`` and ``interrupt_value`` with the plan.
  - On the *next* call with the same thread_id and ``approve=True/False`` in the
    request, the endpoint resumes with ``Command(resume=...)`` instead of a fresh
    HumanMessage injection.

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

import json
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.config import settings
from app.dependencies import get_graph
from app.exceptions import LLMRateLimitError, LLMServiceError, LLMServiceUnavailableError
from app.models import ChatRequest, ChatResponse, CheckpointInfo, ReplayRequest

router = APIRouter()


def _extract_interrupt_value(snapshot) -> dict | None:
    """Extract the interrupt payload from the first suspended task in a snapshot."""
    tasks = snapshot.tasks
    if not tasks:
        return None
    raw = getattr(tasks[0], "interrupts", [None])[0]
    if raw is None:
        return None
    return raw.value if hasattr(raw, "value") else None


@router.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok"}


@router.post("/v1/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    request: ChatRequest,
    graph: Annotated[CompiledStateGraph, Depends(get_graph)],
) -> ChatResponse:
    """Invoke the graph for a new turn or resume from an interrupt.

    On the first call for a thread_id, the graph starts from scratch.
    On subsequent calls, LangGraph loads the latest checkpoint for that thread.

    If the thread is suspended at an interrupt (planner waiting for approval),
    the caller must include ``approve`` in the request. The endpoint then
    resumes with Command(resume=True/False) instead of injecting a new message.

    If not interrupted, the new HumanMessage is injected and the graph runs
    forward.
    """
    config: RunnableConfig = {
        "configurable": {"thread_id": request.thread_id},
        "recursion_limit": settings.max_pipeline_steps,
    }

    # Check if the thread is currently suspended at an interrupt.
    snapshot = await graph.aget_state(config)
    is_interrupted = bool(snapshot.next) and snapshot.values

    if is_interrupted:
        if request.approve is None:
            return ChatResponse(
                thread_id=request.thread_id,
                status="interrupted",
                is_interrupted=True,
                interrupt_value=_extract_interrupt_value(snapshot),
            )
        # Write the resume message to state so resume_guard can inspect it.
        if request.message:
            await graph.aupdate_state(config, {"messages": [HumanMessage(content=request.message)]})
        result = await graph.ainvoke(Command(resume=request.approve), config)
    else:
        # Fresh turn: inject the new human message.
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=request.message)], "status": "planning"},
            config,
        )

    # After invocation, check if the graph is now at a new interrupt.
    post_snapshot = await graph.aget_state(config)
    now_interrupted = bool(post_snapshot.next) and bool(post_snapshot.values)

    return ChatResponse(
        thread_id=request.thread_id,
        status=result.get("status", "done"),
        is_interrupted=now_interrupted,
        interrupt_value=_extract_interrupt_value(post_snapshot) if now_interrupted else None,
        final_answer=result.get("final_answer"),
        guard_reason=result.get("guard_reason"),
        dead_letter=result.get("dead_letter"),
    )


def _classify_error(exc: Exception) -> tuple[int, str]:
    if isinstance(exc, LLMRateLimitError):
        return 429, "LLM rate limit exceeded"
    if isinstance(exc, LLMServiceUnavailableError):
        return 503, "LLM service unavailable"
    if isinstance(exc, LLMServiceError):
        return 502, "LLM service error"
    if isinstance(exc, GraphRecursionError):
        return 500, "Pipeline step limit exceeded"
    return 500, "Internal server error"


async def _emit_interrupt(interrupt_value: dict | None) -> AsyncGenerator[str]:
    yield f"event: interrupt\ndata: {json.dumps({'interrupt_value': interrupt_value})}\n\n"


async def _generate(
    graph: CompiledStateGraph, graph_input, config: RunnableConfig, request: Request
) -> AsyncGenerator[str]:
    try:
        async for event in graph.astream_events(graph_input, config, version="v2"):
            if await request.is_disconnected():
                return
            if event["event"] == "on_chat_model_stream" and "writer" in event.get("tags", []):
                token = event["data"].get("chunk").content  # type: ignore[union-attr]
                if isinstance(token, list):
                    token = "".join(b.get("text", "") for b in token if isinstance(b, dict) and b.get("type") == "text")
                if token:
                    yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
    except Exception as exc:
        code, detail = _classify_error(exc)
        yield f"event: error\ndata: {json.dumps({'code': code, 'detail': detail})}\n\n"
        return

    snapshot = await graph.aget_state(config)
    if bool(snapshot.next):
        interrupt_value = _extract_interrupt_value(snapshot)
        yield f"event: interrupt\ndata: {json.dumps({'interrupt_value': interrupt_value})}\n\n"
    else:
        state = snapshot.values
        status = state.get("status", "done")
        if status == "dead_lettered":
            yield f"event: error\ndata: {json.dumps({'status': status, 'dead_letter': state.get('dead_letter')})}\n\n"
        else:
            yield f"event: done\ndata: {json.dumps({'status': status, 'final_answer': state.get('final_answer')})}\n\n"


@router.post("/v1/chat/stream", tags=["chat"])
async def chat_stream(
    request: Request,
    body: ChatRequest,
    graph: Annotated[CompiledStateGraph, Depends(get_graph)],
) -> StreamingResponse:
    """Token-streaming variant of POST /v1/chat.

    Returns a text/event-stream response. Frames:
      event: token       — one LLM token from the writer node
      event: interrupt   — graph paused at planner interrupt
      event: done        — graph reached END
      event: error       — unhandled exception escaped the graph

    Resume an interrupt: call this endpoint again with the same thread_id
    and approve=true or approve=false in the request body.
    """
    config: RunnableConfig = {
        "configurable": {"thread_id": body.thread_id},
        "recursion_limit": settings.max_pipeline_steps,
    }

    snapshot = await graph.aget_state(config)
    is_interrupted = bool(snapshot.next) and snapshot.values

    if is_interrupted and body.approve is None:
        interrupt_value = _extract_interrupt_value(snapshot)
        return StreamingResponse(
            _emit_interrupt(interrupt_value),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if is_interrupted:
        # Write the resume message to state so resume_guard can inspect it.
        if body.message:
            await graph.aupdate_state(config, {"messages": [HumanMessage(content=body.message)]})
        graph_input = Command(resume=body.approve)
    else:
        graph_input = {"messages": [HumanMessage(content=body.message)], "status": "planning"}

    return StreamingResponse(
        _generate(graph, graph_input, config, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
    try:
        result = await graph.ainvoke(None, config)
    except Exception as exc:
        # Postgres raises when checkpoint_id is not a valid UUID or doesn't exist.
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {request.checkpoint_id}") from exc
    return ChatResponse(
        thread_id=thread_id,
        status=result.get("status", "done"),
        final_answer=result.get("final_answer"),
        guard_reason=result.get("guard_reason"),
        dead_letter=result.get("dead_letter"),
    )
