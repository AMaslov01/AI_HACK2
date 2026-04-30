"""Сборка langgraph StateGraph. Условный переход на visual_augment."""

from __future__ import annotations

from src.agent.nodes import (
    formatter_node,
    make_compose,
    make_planner,
    make_retrieve,
    make_visual_augment,
)
from src.agent.state import AgentState


def _needs_vision(state: AgentState) -> str:
    plan = state.get("plan")
    if plan and getattr(plan, "needs_vision", False) and getattr(plan, "target_label", None):
        return "visual"
    return "compose"


def build_graph(*, gigachat_chat, gigachat_planner, gigachat_vision,
                retriever, embeddings, figure_records: dict,
                deferred: bool = False, embed_cache: dict | None = None):
    from langgraph.graph import StateGraph, START, END

    embed_cache = embed_cache if embed_cache is not None else {}

    g = StateGraph(AgentState)
    g.add_node("planner", make_planner(gigachat_planner, retriever))
    g.add_node("retrieve", make_retrieve(retriever, embeddings, embed_cache))
    g.add_node("visual", make_visual_augment(gigachat_vision, figure_records))
    g.add_node("compose", make_compose(gigachat_chat, deferred=deferred))
    g.add_node("formatter", formatter_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "retrieve")
    g.add_conditional_edges("retrieve", _needs_vision,
                            {"visual": "visual", "compose": "compose"})
    g.add_edge("visual", "compose")
    g.add_edge("compose", "formatter")
    g.add_edge("formatter", END)
    return g.compile()
