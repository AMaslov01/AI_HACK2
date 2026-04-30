"""Гибридный retrieval с учётом плана: cosine-порог, fallback, +N глобальных."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import (
    MIN_COSINE,
    MIN_SCOPED_DOCS,
    RETRIEVE_FETCH_K,
    RETRIEVE_GLOBAL_BONUS,
    RETRIEVE_TOP_K,
)


@dataclass
class RetrievedDoc:
    chunk_id: str
    text: str
    metadata: dict
    score: float = 0.0


@dataclass
class Plan:
    section_scope: list[str] | None = None  # None = глобально; [] = unscoped после валидации
    is_figure_question: bool = False
    target_label: str | None = None
    needs_vision: bool = False
    reasoning: str = ""
    unscoped: bool = False
    fallback_used: bool = False
    extra: dict = field(default_factory=dict)


class Retriever:
    """Знает Chroma collection + множества known_numbers/known_top.
    Принимает Plan, отдаёт <= RETRIEVE_TOP_K + RETRIEVE_GLOBAL_BONUS документов."""

    def __init__(self, collection, known_numbers: set[str], known_top: set[str]):
        self.coll = collection
        self.known_numbers = known_numbers
        self.known_top = known_top

    def validate_scope(self, plan: Plan) -> Plan:
        """Фильтрует scope по known_numbers ∪ known_top. Пустой scope → unscoped."""
        if not plan.section_scope:
            plan.section_scope = None
            plan.unscoped = True
            return plan
        normalized: list[str] = []
        for s in plan.section_scope:
            s = s.strip().lstrip("§").lstrip("Section ").strip()
            if not s:
                continue
            if s in self.known_numbers or s in self.known_top:
                normalized.append(s)
        if not normalized:
            plan.section_scope = None
            plan.unscoped = True
        else:
            plan.section_scope = normalized
            plan.unscoped = False
        return plan

    def _build_where(self, scope: list[str] | None,
                     is_figure: bool) -> dict | None:
        clauses: list[dict] = []
        if scope:
            clauses.append({"$or": [
                {"section_number": {"$in": scope}},
                {"section_top": {"$in": scope}},
            ]})
        if is_figure:
            clauses.append({"kind": {"$in": ["caption", "figure_desc", "body"]}})
            clauses.append({"has_figure": True})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _query(self, embedding: list[float], where: dict | None,
               n_results: int) -> list[RetrievedDoc]:
        kwargs: dict = {"query_embeddings": [embedding], "n_results": n_results,
                        "include": ["documents", "metadatas", "distances"]}
        if where is not None:
            kwargs["where"] = where
        try:
            res = self.coll.query(**kwargs)
        except Exception:
            # фолбэк: без фильтра
            kwargs.pop("where", None)
            try:
                res = self.coll.query(**kwargs)
            except Exception:
                return []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: list[RetrievedDoc] = []
        for cid, t, m, d in zip(ids, docs, metas, dists):
            sim = max(0.0, 1.0 - float(d))   # cosine distance → similarity
            out.append(RetrievedDoc(chunk_id=cid, text=t, metadata=m or {}, score=sim))
        return out

    def _rerank(self, docs: list[RetrievedDoc], plan: Plan) -> list[RetrievedDoc]:
        scope = set(plan.section_scope or [])
        for d in docs:
            bonus = 0.0
            if scope and d.metadata.get("section_number") in scope:
                bonus += 0.05
            if scope and d.metadata.get("section_top") in scope:
                bonus += 0.03
            if plan.is_figure_question and d.metadata.get("kind") in ("caption", "figure_desc"):
                bonus += 0.05
            if plan.target_label:
                fr = d.metadata.get("figure_refs") or ""
                er = d.metadata.get("equation_refs") or ""
                tl = f"|{plan.target_label}|"
                if tl in fr or tl in er:
                    bonus += 0.1
            d.score += bonus
        docs.sort(key=lambda x: x.score, reverse=True)
        return docs

    def _dedup(self, docs: list[RetrievedDoc], k: int) -> list[RetrievedDoc]:
        seen: set[tuple[str, str]] = set()
        out: list[RetrievedDoc] = []
        for d in docs:
            key = (d.metadata.get("section_number", ""), d.metadata.get("kind", ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
            if len(out) >= k:
                break
        return out

    def retrieve(self, question_embedding: list[float], plan: Plan) -> list[RetrievedDoc]:
        """Главный entry. Возвращает ≤ RETRIEVE_TOP_K + RETRIEVE_GLOBAL_BONUS docs."""
        # Scoped pass
        where = self._build_where(plan.section_scope, plan.is_figure_question)
        scoped = self._query(question_embedding, where, RETRIEVE_FETCH_K)
        scoped = [d for d in scoped if d.score >= MIN_COSINE]
        scoped = self._rerank(scoped, plan)
        scoped = self._dedup(scoped, RETRIEVE_TOP_K)

        # Fallback при < 3
        if len(scoped) < MIN_SCOPED_DOCS:
            plan.fallback_used = True
            global_where = self._build_where(None, plan.is_figure_question)
            global_docs = self._query(question_embedding, global_where, RETRIEVE_FETCH_K)
            global_docs = [d for d in global_docs if d.score >= MIN_COSINE]
            global_docs = self._rerank(global_docs, plan)
            return self._dedup(global_docs, RETRIEVE_TOP_K + RETRIEVE_GLOBAL_BONUS)

        # +N глобальных к scope-результатам
        seen_ids = {d.chunk_id for d in scoped}
        global_pure = self._query(question_embedding, None, RETRIEVE_FETCH_K // 2)
        global_pure = [d for d in global_pure
                       if d.score >= MIN_COSINE and d.chunk_id not in seen_ids]
        global_pure.sort(key=lambda x: x.score, reverse=True)
        merged = scoped + global_pure[:RETRIEVE_GLOBAL_BONUS]
        return merged
