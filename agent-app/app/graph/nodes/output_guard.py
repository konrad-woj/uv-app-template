"""Output guardrail node: validates the draft answer before returning it to the caller.

Runs as the last node before END. Checks the final_answer for:
  - Factual grounding in search_results (no unsupported claims).
  - Absence of harmful, biased, or misleading content.

On safe:   sets status="done", preserves final_answer.
On unsafe: sets status="blocked", replaces final_answer with a safe fallback message,
           sets guard_reason — does NOT loop back to refiner (keeps graph acyclic).
On error:  with_dead_letter catches and routes to dead_letter.
"""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from app.graph.nodes._dead_letter import with_dead_letter
from app.graph.nodes._guard_verdict import GuardVerdict
from app.graph.nodes._llm_invoke import llm_invoke_with_retry

_SAFE_FALLBACK = (
    "I was unable to produce a verified answer to your question. "
    "Please try rephrasing or narrowing the scope of your query."
)

_SYSTEM_PROMPT = """You are a content quality and safety checker for a research assistant.
Evaluate the draft answer against the provided research results.

Check for:
1. Factual grounding: claims are supported by the search results provided.
2. Safety: no harmful, illegal, or misleading content.

Respond with a JSON object:
{"verdict": "safe" or "unsafe", "reason": "<one sentence>"}
Do not include any other text."""


def make_output_guard_node(llm: BaseChatModel) -> Callable:
    """Return an async output_guard node bound to the given LLM."""

    @with_dead_letter("output_guard")
    async def output_guard(state: "AgentState", config: RunnableConfig) -> dict:  # type: ignore[name-defined]  # noqa: F821
        search_context = "\n".join(state.get("search_results", []))  # type: ignore[arg-type]
        answer = state.get("final_answer", state.get("draft_answer", ""))
        prompt = f"Search results:\n{search_context}\n\nDraft answer:\n{answer}"
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = await llm_invoke_with_retry(llm, messages, config)
        try:
            verdict = GuardVerdict.model_validate_json(str(response.content))
        except (ValidationError, ValueError):
            return {
                "status": "blocked",
                "final_answer": _SAFE_FALLBACK,
                "guard_reason": "Output guard could not parse LLM response — treating as unsafe.",
            }

        if verdict.verdict == "safe":
            return {"status": "done", "final_answer": answer}
        return {
            "status": "blocked",
            "final_answer": _SAFE_FALLBACK,
            "guard_reason": verdict.reason,
        }

    return output_guard
