"""Writer node: drafts a final answer and extracts verifiable claims.

Receives the research context from react_researcher (via state["messages"]) and
the research plan. Produces draft_answer (prose) and claims (list of specific
verifiable facts), which flow into verify_subgraph and then reflection_subgraph.

The LLM is prompted to return JSON with two keys: "answer" and "claims".
On parse failure the raw content is used as draft_answer and claims is empty —
verify_subgraph fans out to zero branches, which is valid.
"""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from logger import get_logger
from pydantic import BaseModel, ValidationError

from app.graph.nodes._dead_letter import with_dead_letter
from app.graph.nodes._llm_invoke import llm_invoke_with_retry

logger = get_logger(__name__)

# Hard ceiling on claims forwarded to verify_subgraph: each claim spawns a parallel
# Send branch (tool call + LLM call), so this bounds fan-out width regardless of
# what the writer LLM actually returns (the "3-5" in the prompt is not enforced by it).
_MAX_CLAIMS = 5

_SYSTEM_PROMPT = """You are a research writer. Given the user's research question,
the research plan, and the gathered research context, write a comprehensive, well-structured answer.

Be specific and ensure the answer directly addresses the question.
Write in clear prose — no bullet points unless listing genuinely enumerable items.

Return a JSON object with exactly two keys:
{
  "answer": "<full research answer in prose>",
  "claims": ["<specific verifiable factual claim>", ...]
}

List 3–5 specific factual claims made in the answer that can be independently verified
(e.g. dates, statistics, names, events). Do not include opinions or methodology as claims."""


class _WriterOutput(BaseModel):
    answer: str
    claims: list[str]


def make_writer_node(llm: BaseChatModel) -> Callable:
    """Return an async writer node bound to the given LLM."""

    @with_dead_letter("writer")
    async def writer(state: "AgentState", config: RunnableConfig) -> dict:  # type: ignore[name-defined]  # noqa: F821
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        question = str(last_human.content) if last_human else ""
        plan: list[str] = state.get("plan", [])  # type: ignore[assignment]
        logger.info(
            "writer.inputs",
            question_length=len(question),
            plan_step_count=len(plan),
            message_count=len(state["messages"]),
        )
        plan_summary = "\n".join(plan)
        context = f"Question: {question}\n\nResearch plan:\n{plan_summary}"
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
        response = await llm_invoke_with_retry(llm, messages, config)
        raw = str(response.content)
        try:
            parsed = _WriterOutput.model_validate_json(raw)
            draft = parsed.answer
            claims = parsed.claims
        except (ValidationError, ValueError):
            logger.warning("writer.parse_failed", raw_length=len(raw))
            draft = raw
            claims = []
        if len(claims) > _MAX_CLAIMS:
            logger.warning("writer.claims_truncated", claim_count=len(claims), max_claims=_MAX_CLAIMS)
            claims = claims[:_MAX_CLAIMS]
        logger.info("writer.draft_produced", draft_length=len(draft), claim_count=len(claims))
        return {"draft_answer": draft, "claims": claims, "status": "writing"}

    return writer
