"""Input guardrail node: three-layer safety and topic check.

Runs first in the graph (START → input_guard). Three layers applied in order,
short-circuiting on block:

  1. Regex blocklist (_prompt_utils.sanitize_user_text) — null bytes, XML injection
     markers, tool-call syntax; <1ms, zero model cost.
  2. GLiGuard (fastino/gliguard-LLMGuardrails-300M) — prompt injection and jailbreak;
     ~15ms GPU; no topic knowledge.
  3. LLM topic check — research-domain relevance only (safety is owned by layer 2);
     ~300ms.

On safe:   routes to planner via after("planner") in workflow.py.
On block:  sets status="blocked", guard_reason=<reason>, routes to END.
On error:  with_dead_letter catches the exception and routes to dead_letter.
"""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from logger import get_logger
from pydantic import ValidationError

from app.config import settings
from app.graph.nodes._dead_letter import with_dead_letter
from app.graph.nodes._guard_verdict import GuardVerdict
from app.graph.nodes._llm_invoke import llm_invoke_with_retry
from app.graph.nodes._prompt_utils import sanitize_user_text
from app.guards.gliguard import GLiGuardClient

logger = get_logger(__name__)

_TOPIC_CHECK_PROMPT = """You are a topic classifier for a research assistant.
Classify whether the user's request is relevant to research or information gathering.

Allowed: factual research, technical questions, analysis of publicly available information,
summaries, comparisons, and similar information-gathering tasks.
Disallowed: requests entirely unrelated to research or information gathering (e.g., asking
the assistant to perform actions, generate creative content unrelated to research, or engage
in role-play).

Note: safety classification is handled separately — only assess topic relevance here.

Respond with a JSON object:
{"verdict": "safe" or "unsafe", "reason": "<one sentence>"}
Do not include any other text."""


def make_input_guard_node(llm: BaseChatModel, gliguard: GLiGuardClient) -> Callable:
    """Return an async input_guard node bound to the given LLM and GLiGuardClient."""

    @with_dead_letter("input_guard")
    async def input_guard(state: "AgentState", config: RunnableConfig) -> dict:  # type: ignore[name-defined]  # noqa: F821
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        raw_text = str(last_human.content) if last_human else ""
        logger.info("input_guard.inputs", input_text_length=len(raw_text))

        # Layer 1: regex sanitisation — blocks null bytes, XML injection markers, tool-call syntax.
        try:
            user_text = sanitize_user_text(raw_text)
        except ValueError as exc:
            logger.info("input_guard.layer1_blocked", reason=str(exc))
            return {"status": "blocked", "guard_reason": f"Input rejected by sanitiser: {exc}"}
        logger.info("input_guard.layer1_passed")

        # Layer 2: GLiGuard — injection and jailbreak detection.
        guard_result = await gliguard.acheck_input(user_text, settings.guard_timeout_seconds)
        if guard_result.blocked:
            logger.info("input_guard.layer2_blocked", reason=guard_result.reason)
            return {"status": "blocked", "guard_reason": guard_result.reason or "Potential prompt injection detected."}
        logger.info("input_guard.layer2_passed")

        # Layer 3: LLM topic relevance check (not safety — GLiGuard owns that).
        messages = [
            SystemMessage(content=_TOPIC_CHECK_PROMPT),
            HumanMessage(content=user_text),
        ]
        response = await llm_invoke_with_retry(llm, messages, config)
        try:
            verdict = GuardVerdict.model_validate_json(str(response.content))
        except (ValidationError, ValueError):
            logger.warning("input_guard.layer3_parse_failed")
            return {
                "status": "blocked",
                "guard_reason": "Input guard could not parse LLM response — treating as off-topic.",
            }

        logger.info("input_guard.layer3_verdict", verdict=verdict.verdict, reason=verdict.reason)
        if verdict.verdict == "safe":
            return {"status": "planning"}
        return {"status": "blocked", "guard_reason": verdict.reason}

    return input_guard
