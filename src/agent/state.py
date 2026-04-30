"""TypedDict-state для langgraph."""

from __future__ import annotations

from typing import Literal, TypedDict

from src.index.retriever import Plan, RetrievedDoc

FormatKind = Literal["markdown", "numbered"]


class AgentState(TypedDict, total=False):
    question: str
    question_index: int
    format_kind: FormatKind
    article_readme: str
    plan: Plan
    docs: list[RetrievedDoc]
    draft_answer: str
    final_text: str
    deadline_ts: float
    error: str | None
    timings: dict
