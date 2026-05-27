"""Planner node: generates a research plan, guards it, then pauses for human approval.

Demonstrates the LangGraph interrupt() pattern:
  1. LLM generates a plan.
  2. Plan text is checked by GLiGuard (injection/content check) + LLM quality check.
     If the plan is unsafe: returns status="blocked" WITHOUT calling interrupt() —
     the unsafe plan is never surfaced to the user.
  3. If the plan is safe: calls interrupt({"plan": plan_steps}) — graph suspends here,
     checkpointed. Resume guard and search run on resumption.
  4. Caller resumes with Command(resume=True) → plan_approved=True → routes to resume_guard.
  5. Caller resumes with Command(resume=False) → plan_approved=False → routes to END (aborted).

The planner is wrapped with with_dead_letter so any LLM error routes to dead_letter.
"""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from logger import get_logger
from pydantic import ValidationError

from app.graph.nodes._dead_letter import with_dead_letter
from app.graph.nodes._guard_verdict import GuardVerdict
from app.graph.nodes._llm_invoke import llm_invoke_with_retry
from app.guards.gliguard import GLiGuardClient

logger = get_logger(__name__)

_PLAN_SYSTEM_PROMPT = """You are a research planning assistant.
Given the user's research question, produce a numbered list of 3-5 concrete research steps.
Each step should be a specific query or action to gather relevant information.
Format: return only the numbered list, one step per line."""

_PLAN_GUARD_PROMPT = """You are a quality and safety checker for research plans.
Evaluate whether the research plan is appropriate and focused on legitimate information gathering.

Disallowed: plans that involve accessing private systems, illegal activities, targeted harassment,
or gathering information to cause harm.

Respond with a JSON object:
{"verdict": "safe" or "unsafe", "reason": "<one sentence>"}
Do not include any other text."""


def make_planner_node(llm: BaseChatModel, gliguard: GLiGuardClient) -> Callable:
    """Return an async planner node bound to the given LLM and GLiGuardClient."""

    @with_dead_letter("planner")
    async def planner(state: "AgentState", config: RunnableConfig) -> dict:  # type: ignore[name-defined]  # noqa: F821
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        user_text = last_human.content if last_human else ""
        logger.info("planner.inputs", question_length=len(str(user_text)), message_count=len(state["messages"]))

        messages = [
            SystemMessage(content=_PLAN_SYSTEM_PROMPT),
            HumanMessage(content=str(user_text)),
        ]
        response = await llm_invoke_with_retry(llm, messages, config)
        plan_text = str(response.content)
        plan_steps = [line.strip() for line in plan_text.splitlines() if line.strip()]
        logger.info("planner.plan_generated", step_count=len(plan_steps))

        # Guard the generated plan before surfacing it to the user.
        plan_as_text = "\n".join(plan_steps)

        # GLiGuard injection check on plan text.
        guard_result = gliguard.check_input(plan_as_text)
        if guard_result.blocked:
            logger.info("planner.guard_gliguard_blocked", reason=guard_result.reason)
            return {
                "plan": plan_steps,
                "status": "blocked",
                "guard_reason": guard_result.reason or "Generated plan failed safety check.",
            }

        # LLM quality + safety check on plan text.
        guard_messages = [
            SystemMessage(content=_PLAN_GUARD_PROMPT),
            HumanMessage(content=f"Research plan:\n{plan_as_text}"),
        ]
        guard_response = await llm_invoke_with_retry(llm, guard_messages, config)
        try:
            verdict = GuardVerdict.model_validate_json(str(guard_response.content))
        except (ValidationError, ValueError):
            logger.warning("planner.guard_llm_parse_failed")
            return {
                "plan": plan_steps,
                "status": "blocked",
                "guard_reason": "Plan guard could not parse LLM response — treating as unsafe.",
            }

        if verdict.verdict != "safe":
            logger.info("planner.guard_llm_blocked", reason=verdict.reason)
            return {
                "plan": plan_steps,
                "status": "blocked",
                "guard_reason": verdict.reason,
            }

        logger.info("planner.guard_passed", awaiting_approval=True)
        # Plan is safe — pause here for human approval.
        approved: bool = interrupt({"plan": plan_steps})

        logger.info("planner.resumed", approved=approved)
        if not approved:
            return {"plan": plan_steps, "plan_approved": False, "status": "aborted"}
        return {"plan": plan_steps, "plan_approved": True, "status": "researching"}

    return planner
