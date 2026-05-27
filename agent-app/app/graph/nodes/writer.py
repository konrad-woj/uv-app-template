"""Writer node: drafts a final answer from accumulated research context.

Receives search_results (from search_subgraph) and messages (from react_researcher).
Produces draft_answer, which flows into the reflection_subgraph for quality refinement.
"""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.graph.nodes._dead_letter import with_dead_letter
from app.graph.nodes._llm_invoke import llm_invoke_with_retry

_SYSTEM_PROMPT = """You are a research writer. Given the user's research question,
the research plan, and gathered search results, write a comprehensive, well-structured answer.

Be specific, cite the information gathered, and ensure the answer directly addresses the question.
Write in clear prose — no bullet points unless listing genuinely enumerable items."""


def make_writer_node(llm: BaseChatModel) -> Callable:
    """Return an async writer node bound to the given LLM."""

    @with_dead_letter("writer")
    async def writer(state: "AgentState", config: RunnableConfig) -> dict:  # type: ignore[name-defined]  # noqa: F821
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        question = str(last_human.content) if last_human else ""
        search_summary = "\n".join(state.get("search_results", []))  # type: ignore[arg-type]
        plan_summary = "\n".join(state.get("plan", []))  # type: ignore[arg-type]
        context = f"Question: {question}\n\nPlan:\n{plan_summary}\n\nResearch:\n{search_summary}"
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
        response = await llm_invoke_with_retry(llm, messages, config)
        draft = str(response.content)
        return {"draft_answer": draft, "status": "reflecting"}

    return writer
