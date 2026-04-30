"""Семантический Q&A-кэш per article. Точный и semantic (cosine ≥ 0.92) лукап."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import QA_CACHE_FILE, QA_CACHE_SIM_THRESHOLD

_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_question(q: str) -> str:
    s = q.lower().strip()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


@dataclass
class QAEntry:
    question: str
    question_norm: str
    question_embedding: list[float]
    answer: str
    doc_ids: list[str]
    plan: dict
    ts: float
    source: str = "fresh"

    def to_json(self) -> str:
        return json.dumps({
            "question": self.question,
            "question_norm": self.question_norm,
            "question_embedding": self.question_embedding,
            "answer": self.answer,
            "doc_ids": self.doc_ids,
            "plan": self.plan,
            "ts": self.ts,
            "source": self.source,
        }, ensure_ascii=False)


class QACache:
    def __init__(self, cache_dir: Path):
        self.path = cache_dir / QA_CACHE_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[QAEntry] = []
        self._load()
        self.hits_exact = 0
        self.hits_semantic = 0
        self.misses = 0
        self._sim_records: list[float] = []

    def _load(self) -> None:
        if not self.path.is_file():
            return
        for ln in self.path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                d = json.loads(ln)
                self._entries.append(QAEntry(
                    question=d["question"],
                    question_norm=d["question_norm"],
                    question_embedding=d.get("question_embedding") or [],
                    answer=d["answer"],
                    doc_ids=d.get("doc_ids") or [],
                    plan=d.get("plan") or {},
                    ts=d.get("ts", 0.0),
                    source=d.get("source", "fresh"),
                ))
            except (json.JSONDecodeError, KeyError):
                continue

    def lookup(self, question: str, embedding: list[float] | None) -> QAEntry | None:
        norm = normalize_question(question)
        # exact
        for e in self._entries:
            if e.question_norm == norm:
                self.hits_exact += 1
                return e
        # semantic
        if not embedding:
            self.misses += 1
            return None
        best: tuple[float, QAEntry] | None = None
        for e in self._entries:
            if not e.question_embedding:
                continue
            sim = _cosine(embedding, e.question_embedding)
            if best is None or sim > best[0]:
                best = (sim, e)
        if best and best[0] >= QA_CACHE_SIM_THRESHOLD:
            self.hits_semantic += 1
            self._sim_records.append(best[0])
            return best[1]
        self.misses += 1
        return None

    def save(self, question: str, answer: str, embedding: list[float],
             doc_ids: list[str], plan: dict) -> None:
        entry = QAEntry(
            question=question,
            question_norm=normalize_question(question),
            question_embedding=embedding,
            answer=answer, doc_ids=doc_ids, plan=plan,
            ts=time.time(), source="fresh",
        )
        self._entries.append(entry)
        # append-mode line-buffered
        with self.path.open("a", encoding="utf-8", buffering=1) as f:
            f.write(entry.to_json() + "\n")

    def metrics(self) -> dict:
        mean_sim = sum(self._sim_records) / len(self._sim_records) if self._sim_records else 0.0
        return {
            "n_cache_hits_exact": self.hits_exact,
            "n_cache_hits_semantic": self.hits_semantic,
            "n_cache_misses": self.misses,
            "mean_hit_similarity": mean_sim,
        }
