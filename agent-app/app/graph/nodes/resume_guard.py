"""Resume guard node: validates the human's resume message before search begins.

Positioned between planner (approved path) and search_subgraph. Applies layers 1
and 2 of the guard pipeline to the message the user sent alongside approve=True/False.
Layer 3 (LLM topic check) is omitted — topic was already validated by input_guard on
the first turn.

The resume message is written to state["messages"] by the router via aupdate_state
before Command(resume=...) is sent, so this node reads state["messages"][-1].

On safe:   routes to react_researcher via _resume_guard_condition in workflow.py.
On block:  sets status="blocked", guard_reason=<reason>, routes to END.
On error:  with_dead_letter catches the exception and routes to dead_letter.
"""

from collections.abc import Callable

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from logger import get_logger

from app.graph.nodes._dead_letter import with_dead_letter
from app.graph.nodes._prompt_utils import sanitize_user_text
from app.guards.gliguard import GLiGuardClient

logger = get_logger(__name__)


def make_resume_guard_node(gliguard: GLiGuardClient) -> Callable:
    """Return an async resume_guard node bound to the given GLiGuardClient."""

    @with_dead_letter("resume_guard")
    async def resume_guard(state: "AgentState", config: RunnableConfig) -> dict:  # type: ignore[name-defined]  # noqa: F821
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        raw_text = str(last_human.content) if last_human else ""
        logger.info("resume_guard.inputs", input_text_length=len(raw_text))

        # Layer 1: regex sanitisation.
        try:
            clean_text = sanitize_user_text(raw_text)
        except ValueError as exc:
            logger.info("resume_guard.layer1_blocked", reason=str(exc))
            return {"status": "blocked", "guard_reason": f"Resume message rejected by sanitiser: {exc}"}
        logger.info("resume_guard.layer1_passed")

        # Layer 2: GLiGuard injection/jailbreak check.
        guard_result = gliguard.check_input(clean_text)
        if guard_result.blocked:
            logger.info("resume_guard.layer2_blocked", reason=guard_result.reason)
            return {
                "status": "blocked",
                "guard_reason": guard_result.reason or "Resume message failed injection check.",
            }
        logger.info("resume_guard.layer2_passed")

        return {}

    return resume_guard
