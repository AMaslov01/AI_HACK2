"""Узлы langgraph: planner, retrieve, visual_augment, compose, formatter."""

from __future__ import annotations

import json
import time

from src.agent.prompts import (
    COMPOSE_GLOBAL_HINT,
    COMPOSE_PROMPT,
    PLANNER_PROMPT,
    VISION_AUGMENT_PROMPT,
)
from src.agent.state import AgentState
from src.config import (
    COMPOSE_BUDGET_S,
    PLANNER_BUDGET_S,
    PER_VISION_CALL_TIMEOUT_S,
)
from src.index.retriever import Plan
from src.io.answers import format_block
from src.utils.logging import log
from src.utils.timing import run_with_timeout


def _parse_planner_json(s: str) -> dict:
    s = s.strip().strip("`")
    if s.lower().startswith("json"):
        s = s[4:]
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1:
        return {}
    try:
        return json.loads(s[a:b + 1])
    except json.JSONDecodeError:
        return {}


def make_planner(gigachat_planner, retriever):
    """Замыкание над клиентами. Возвращает функцию-узла."""

    def node(state: AgentState) -> dict:
        t0 = time.time()
        prompt = PLANNER_PROMPT.format(
            readme=state["article_readme"], question=state["question"]
        )
        try:
            resp = run_with_timeout(gigachat_planner.invoke, PLANNER_BUDGET_S, prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            data = _parse_planner_json(content)
        except Exception as e:
            log.warning(f"planner failed: {e}")
            data = {}

        plan = Plan(
            section_scope=list(data.get("section_scope") or []),
            is_figure_question=bool(data.get("is_figure_question") or False),
            target_label=data.get("target_label"),
            needs_vision=bool(data.get("needs_vision") or False),
            reasoning=str(data.get("reasoning") or ""),
        )
        plan = retriever.validate_scope(plan)

        timings = state.get("timings") or {}
        timings["planner"] = time.time() - t0
        return {"plan": plan, "timings": timings}

    return node


def make_retrieve(retriever, embeddings, embed_cache: dict[str, list[float]]):
    """Использует эмбеддинги вопроса; кеширует их."""

    def node(state: AgentState) -> dict:
        t0 = time.time()
        q = state["question"]
        if q in embed_cache:
            emb = embed_cache[q]
        else:
            try:
                emb = embeddings.embed_query(q)
                embed_cache[q] = emb
            except Exception as e:
                log.warning(f"retrieve embed failed: {e}")
                emb = []
        plan: Plan = state["plan"]
        docs = retriever.retrieve(emb, plan) if emb else []

        timings = state.get("timings") or {}
        timings["retrieve"] = time.time() - t0
        return {"docs": docs, "timings": timings}

    return node


def make_visual_augment(gigachat_vision, figure_records: dict):
    def node(state: AgentState) -> dict:
        t0 = time.time()
        plan: Plan = state["plan"]
        if not plan.needs_vision or not plan.target_label:
            return {}
        rec = figure_records.get(plan.target_label)
        if not rec or not rec.rendered_png:
            return {}
        try:
            from langchain_core.messages import HumanMessage
            import base64
            from pathlib import Path

            img_b64 = base64.b64encode(Path(rec.rendered_png).read_bytes()).decode("ascii")
            text = VISION_AUGMENT_PROMPT.format(question=state["question"], caption=rec.caption)
            msg = HumanMessage(content=[
                {"type": "text", "text": text},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ])
            resp = run_with_timeout(gigachat_vision.invoke, PER_VISION_CALL_TIMEOUT_S, [msg])
            content = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            log.warning(f"visual_augment failed: {e}")
            return {}

        # Заменяем figure_desc-документ
        new_docs = []
        replaced = False
        for d in state.get("docs") or []:
            if (d.metadata.get("kind") == "figure_desc"
                    and f"|{plan.target_label}|" in (d.metadata.get("figure_refs") or "")):
                d.text = f"Visual answer for {plan.target_label}: {content.strip()}"
                replaced = True
            new_docs.append(d)
        if not replaced:
            from src.index.retriever import RetrievedDoc
            new_docs.append(RetrievedDoc(
                chunk_id=f"visual:{plan.target_label}",
                text=f"Visual answer for {plan.target_label}: {content.strip()}",
                metadata={"kind": "figure_desc",
                          "figure_refs": f"|{plan.target_label}|"},
                score=0.99,
            ))
        timings = state.get("timings") or {}
        timings["visual_augment"] = time.time() - t0
        return {"docs": new_docs, "timings": timings}

    return node


def _format_sources(docs) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs[:8], start=1):
        section = d.metadata.get("section_path") or d.metadata.get("section_number") or "?"
        kind = d.metadata.get("kind") or "body"
        body = d.text.strip()
        if len(body) > 1500:
            body = body[:1500] + "..."
        parts.append(f"[{i}] (kind={kind}, секция {section}) {body}")
    return "\n\n".join(parts) if parts else "(нет источников)"


def make_compose(gigachat_chat, deferred: bool = False):
    def node(state: AgentState) -> dict:
        t0 = time.time()
        sources = _format_sources(state.get("docs") or [])
        prompt = COMPOSE_PROMPT.format(
            readme=state["article_readme"][:8000],
            sources=sources, question=state["question"],
        )
        if deferred:
            prompt += COMPOSE_GLOBAL_HINT
        try:
            resp = run_with_timeout(gigachat_chat.invoke, COMPOSE_BUDGET_S, prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            log.warning(f"compose failed: {e}")
            content = "no answer"
        timings = state.get("timings") or {}
        timings["compose"] = time.time() - t0
        return {"draft_answer": (content or "").strip(), "timings": timings}

    return node


def formatter_node(state: AgentState) -> dict:
    t0 = time.time()
    text = state.get("draft_answer") or "no answer"
    fmt = state["format_kind"]
    idx = state["question_index"]
    final = format_block(idx, text, fmt)
    timings = state.get("timings") or {}
    timings["formatter"] = time.time() - t0
    return {"final_text": final, "timings": timings}
