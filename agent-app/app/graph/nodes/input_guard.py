"""Input guardrail node: LLM-based safety and topic check.

Runs first in the graph (START → input_guard). Asks the LLM to classify the
user's request as safe or unsafe/off-topic for a research assistant.

On safe:   routes to planner via after("planner") in workflow.py.
On unsafe: sets status="blocked", guard_reason=<reason>, routes to END.
On error:  with_dead_letter catches the exception and routes to dead_letter.

The LLM response is parsed with Pydantic; any parse failure is treated as
unsafe to avoid silent pass-through of bad input.
"""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from app.graph.nodes._dead_letter import with_dead_letter
from app.graph.nodes._guard_verdict import GuardVerdict
from app.graph.nodes._llm_invoke import llm_invoke_with_retry

_SYSTEM_PROMPT = """You are a content safety classifier for a research assistant.
Classify the user's request.

Allowed topics: factual research, technical questions, analysis of publicly available information.
Disallowed: requests for harmful content, illegal activities, personal attacks, or topics
entirely unrelated to research or information gathering.

Respond with a JSON object:
{"verdict": "safe" or "unsafe", "reason": "<one sentence>"}
Do not include any other text."""


def make_input_guard_node(llm: BaseChatModel) -> Callable:
    """Return an async input_guard node bound to the given LLM."""

    @with_dead_letter("input_guard")
    async def input_guard(state: "AgentState", config: RunnableConfig) -> dict:  # type: ignore[name-defined]  # noqa: F821
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
        try:
            verdict = GuardVerdict.model_validate_json(str(response.content))
        except (ValidationError, ValueError):
            return {
                "status": "blocked",
                "guard_reason": "Input guard could not parse LLM response — treating as unsafe.",
            }

        if verdict.verdict == "safe":
            return {"status": "planning"}
        return {"status": "blocked", "guard_reason": verdict.reason}

    return input_guard
