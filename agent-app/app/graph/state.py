"""Shared state definition for the research-assistant graph.

LangGraph passes the full state dict into every node and merges each node's
returned dict back into state after the node completes.  Fields that carry
a *reducer* annotation are merged with that function instead of overwritten:

  messages: Annotated[list[AnyMessage], operator.add]

The ``operator.add`` reducer *appends* the new messages list to the existing
one, so each node only returns the messages it produced — not the whole
history.  Without this annotation every node invocation would replace the
messages list with its own output.

All other fields (plain TypedDict entries) are last-write-wins: whichever node
writes them last wins. That is intentional for scalar fields like ``status``.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage


class AgentState(TypedDict):
    # operator.add reducer: nodes append to this list, never replace it.
    messages: Annotated[list[AnyMessage], operator.add]
    plan: list[str]  # planner output; each entry is one research step
    draft_answer: str  # writer output before reflection
    final_answer: str  # output_guard-approved answer returned to the caller
    # Lifecycle: "planning" → "searching" → "researching" → "writing" →
    #            "reflecting" → "done" | "aborted" | "blocked" | "dead_lettered"
    status: str
