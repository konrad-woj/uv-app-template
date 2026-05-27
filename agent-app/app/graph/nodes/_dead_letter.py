"""Dead-letter pattern for graph nodes.

Any unhandled exception in a decorated node is caught, serialised into a
``DeadLetterInfo`` record, written to state, and the graph is routed to the
terminal ``dead_letter_node`` via the ``after()`` routing helper.

This is analogous to a dead-letter queue in messaging: the execution does not
crash or silently disappear — it lands in an observable, structured record that
can be inspected via checkpoint history or replayed via time-travel.

Usage:
    @with_dead_letter("my_node")
    async def my_node(state: AgentState, config: RunnableConfig) -> dict:
        ...

    # In workflow.py, replace add_edge with:
    graph.add_conditional_edges("my_node", after("next_node"))
"""

import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict

from langchain_core.runnables import RunnableConfig
from logger import get_logger

if TYPE_CHECKING:
    from app.graph.state import AgentState

logger = get_logger(__name__)


class DeadLetterInfo(TypedDict):
    failed_node: str
    error_type: str
    error_message: str
    traceback: str
    timestamp: str  # ISO-8601


def with_dead_letter(node_name: str) -> Callable:
    """Decorator that catches unhandled exceptions and writes DeadLetterInfo to state.

    Args:
        node_name: The name of the node being decorated, included in the record for traceability.
    """

    def decorator(fn: Callable) -> Callable:
        async def wrapper(state: dict, config: RunnableConfig) -> dict:
            try:
                return await fn(state, config)
            except Exception as exc:
                info: DeadLetterInfo = {
                    "failed_node": node_name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                }
                logger.error(
                    "Node raised unhandled exception, routing to dead_letter",
                    node=node_name,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                return {"dead_letter": info, "status": "dead_lettered"}

        wrapper.__name__ = fn.__name__
        return wrapper

    return decorator


def after(next_node: str) -> Callable:
    """Return a routing function that checks for dead_letter before routing to next_node.

    Used as the condition in graph.add_conditional_edges() for every node that
    can raise — this replaces add_edge for those nodes.

    Args:
        next_node: The node to route to when dead_letter is not set.
    """

    def _route(state: dict) -> str:
        return "dead_letter" if state.get("dead_letter") else next_node

    _route.__name__ = f"after_{next_node}"
    return _route


async def dead_letter_node(state: "AgentState", config: RunnableConfig) -> dict:
    """Terminal node: structured-log the DeadLetterInfo and return.

    The checkpointer persists the full state automatically — the record is
    replayable via time-travel from this checkpoint.
    """
    _ = config
    logger.error("dead_letter", dead_letter=state.get("dead_letter"))
    return {}
