"""Planner node: generates a research plan and pauses for human approval.

Demonstrates the LangGraph interrupt() pattern:
  1. Node calls interrupt({"plan": plan_text}) — graph suspends here, checkpointed.
  2. Caller resumes with Command(resume=True) → plan_approved=True → routes to search.
  3. Caller resumes with Command(resume=False) → plan_approved=False → routes to END (aborted).

The planner is wrapped with with_dead_letter so any LLM error routes to dead_letter.
"""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from app.graph.nodes._dead_letter import with_dead_letter
from app.graph.nodes._llm_invoke import llm_invoke_with_retry

_SYSTEM_PROMPT = """You are a research planning assistant.
Given the user's research question, produce a numbered list of 3-5 concrete research steps.
Each step should be a specific query or action to gather relevant information.
Format: return only the numbered list, one step per line."""


def make_planner_node(llm: BaseChatModel) -> Callable:
    """Return an async planner node bound to the given LLM."""

    @with_dead_letter("planner")
    async def planner(state: "AgentState", config: RunnableConfig) -> dict:  # type: ignore[name-defined]  # noqa: F821
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        user_text = last_human.content if last_human else ""
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=str(user_text)),
        ]
        response = await llm_invoke_with_retry(llm, messages, config)
        plan_text = str(response.content)
        plan_steps = [line.strip() for line in plan_text.splitlines() if line.strip()]

        # Pause here — caller must resume with Command(resume=True/False)
        approved: bool = interrupt({"plan": plan_steps})

        if not approved:
            return {"plan": plan_steps, "plan_approved": False, "status": "aborted"}
        return {"plan": plan_steps, "plan_approved": True, "status": "searching"}

    return planner
