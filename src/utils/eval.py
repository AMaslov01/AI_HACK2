"""Локальный харнес метрик: retrieval / synthetic / self-check / latency.

Запуск: python -m src.utils.eval --article-dir data --mode all
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from src.agent.runner import _build_chat_client
from src.config import (
    CACHE_ROOT,
    TEMPERATURE_CHAT,
)
from src.index.embeddings import build_embeddings
from src.index.retriever import Plan, Retriever
from src.index.vectorstore import get_or_create_collection, has_documents, upsert_chunks
from src.ingest.pipeline import IngestArtifacts, run_ingest
from src.utils.logging import log


# --- Retrieval metrics ---

def retrieval_metrics(art: IngestArtifacts) -> dict:
    sections = [s for s in art.structure.sections if s.number != "0"]
    secs_in_chunks = {c.metadata.get("section_number") for c in art.chunks
                      if c.metadata.get("kind") == "body"}
    coverage = (len(secs_in_chunks & {s.number for s in sections}) /
                max(1, len(sections)))
    sizes = [len(c.text) for c in art.chunks if c.metadata.get("kind") == "body"]
    fig_resolved = sum(1 for r in art.figures.values() if r.resolved_path)
    fig_total = len(art.figures)
    referenced = set(art.structure.label_referenced_by.keys())
    defined = set(art.structure.label_to_target.keys())
    return {
        "n_sections": len(sections),
        "section_coverage": round(coverage, 3),
        "n_chunks_total": len(art.chunks),
        "chunk_size_p50": int(statistics.median(sizes)) if sizes else 0,
        "chunk_size_p95": int(_pctl(sizes, 0.95)) if sizes else 0,
        "figure_resolution_rate": round(fig_resolved / max(1, fig_total), 3),
        "label_undefined_referenced": sorted(referenced - defined)[:20],
        "label_defined_unreferenced": sorted(defined - referenced)[:20],
    }


def _pctl(xs: list[float], p: float) -> float:
    if not xs:
        return 0
    s = sorted(xs)
    idx = int(p * (len(s) - 1))
    return s[idx]


# --- Synthetic eval set ---

def _gen_factual(art: IngestArtifacts, gc) -> list[dict]:
    body_chunks = [c for c in art.chunks if c.metadata.get("kind") == "body"]
    if not body_chunks:
        return []
    import random
    random.seed(42)
    sample = random.sample(body_chunks, min(20, len(body_chunks)))
    out: list[dict] = []
    for c in sample:
        prompt = (
            "Сгенерируй 1 specific factual question и ответ из этого текста. "
            "Только JSON: {\"q\": \"...\", \"a\": \"...\"}\n\n"
            f"Текст: {c.text[:1500]}"
        )
        try:
            resp = gc.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            data = _extract_json(content)
            if data and data.get("q") and data.get("a"):
                out.append({"q": data["q"], "a": data["a"],
                            "target_section": c.metadata.get("section_number"),
                            "kind": "factual"})
        except Exception as e:
            log.warning(f"gen_factual failed: {e}")
    return out


def _gen_figure(art: IngestArtifacts, gc) -> list[dict]:
    figs = list(art.figures.values())
    if not figs:
        return []
    import random
    random.seed(43)
    sample = random.sample(figs, min(5, len(figs)))
    out: list[dict] = []
    for r in sample:
        prompt = (
            "Сгенерируй 1 вопрос, требующий смотреть на рисунок (а не только caption). "
            "Только JSON: {\"q\": \"...\", \"a\": \"...\"}\n\n"
            f"Caption: {r.caption}\nDescription: {r.description_text[:1000]}"
        )
        try:
            resp = gc.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            data = _extract_json(content)
            if data and data.get("q") and data.get("a"):
                out.append({"q": data["q"], "a": data["a"],
                            "target_figure": r.figure_id, "kind": "figure"})
        except Exception as e:
            log.warning(f"gen_figure failed: {e}")
    return out


def _extract_json(s: str) -> dict:
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


def generate_or_load_synthetic(art: IngestArtifacts, gc) -> list[dict]:
    eval_dir = Path("eval/synthetic")
    eval_dir.mkdir(parents=True, exist_ok=True)
    fp = art.cache_dir.name
    path = eval_dir / f"{fp}.jsonl"
    if path.is_file():
        return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    items = _gen_factual(art, gc) + _gen_figure(art, gc)
    path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
                    encoding="utf-8")
    return items


def evaluate_retrieval_recall(items: list[dict], retriever: Retriever,
                              embeddings) -> dict:
    """Доля синтетических вопросов, у которых target_section/target_figure
    встречается в top-5 retrieved."""
    if not items:
        return {}
    sec_hits = sec_total = 0
    fig_hits = fig_total = 0
    for it in items:
        try:
            emb = embeddings.embed_query(it["q"])
        except Exception:
            continue
        plan = Plan(section_scope=None, is_figure_question=(it["kind"] == "figure"),
                    target_label=it.get("target_figure"))
        plan = retriever.validate_scope(plan)
        docs = retriever.retrieve(emb, plan)[:5]
        if it.get("target_section") is not None:
            sec_total += 1
            tops = {d.metadata.get("section_number") for d in docs}
            if it["target_section"] in tops:
                sec_hits += 1
        if it.get("target_figure") is not None:
            fig_total += 1
            for d in docs:
                fr = d.metadata.get("figure_refs") or ""
                if f"|{it['target_figure']}|" in fr:
                    fig_hits += 1
                    break
    return {
        "section_recall@5": round(sec_hits / max(1, sec_total), 3),
        "figure_hit_rate@5": round(fig_hits / max(1, fig_total), 3),
        "n_synthetic_factual": sec_total,
        "n_synthetic_figure": fig_total,
    }


# --- Self-check rubric ---

JUDGE_PROMPT = """\
Оцени ответ. Верни ТОЛЬКО JSON:
{{"relevance": 0|1|2, "specificity": 0|1|2, "consistency": 0|1|2, "comment": "..."}}

Карта статьи:
{readme}

Вопрос: {q}
Ответ: {a}
"""


def selfcheck(answers_path: Path, questions_path: Path, readme: str, gc) -> dict:
    """Применяет judge-rubric ко всем парам (вопрос, ответ)."""
    from src.io.questions import load_questions
    from src.utils.check_submission import split_markdown_answers, split_numbered_answers

    qdoc = load_questions(questions_path)
    atext = answers_path.read_text(encoding="utf-8-sig")
    if qdoc.fmt == "markdown":
        bodies = split_markdown_answers(atext)
    else:
        items = split_numbered_answers(atext)
        bodies = [re.sub(r"^\s*\d+[\.)]\s*", "", b, count=1) for b in items]
    scores: list[dict] = []
    refusals = 0
    grounding_density: list[int] = []
    lengths: list[int] = []
    for q, a in zip(qdoc.items, bodies):
        a_clean = a.strip()
        if not a_clean or "no answer" in a_clean.lower():
            refusals += 1
        grounding = len(re.findall(r"§\d|Figure \d|fig:|eq:", a_clean))
        grounding_density.append(grounding)
        lengths.append(len(a_clean))
        try:
            resp = gc.invoke(JUDGE_PROMPT.format(readme=readme[:6000], q=q.text, a=a_clean))
            content = resp.content if hasattr(resp, "content") else str(resp)
            d = _extract_json(content)
            if d:
                scores.append(d)
        except Exception as e:
            log.warning(f"judge q{q.idx} failed: {e}")
    n = len(scores)
    avg = lambda key: round(sum(s.get(key, 0) for s in scores) / max(1, n), 2)  # noqa: E731
    return {
        "n_answers": len(bodies),
        "refusal_rate": round(refusals / max(1, len(bodies)), 3),
        "answer_len_p50": int(statistics.median(lengths)) if lengths else 0,
        "grounding_density_mean": round(statistics.mean(grounding_density), 2)
            if grounding_density else 0,
        "judge_relevance": avg("relevance"),
        "judge_specificity": avg("specificity"),
        "judge_consistency": avg("consistency"),
    }


def render_report(report: dict) -> str:
    lines = ["# Eval Report", ""]
    for section, payload in report.items():
        lines.append(f"## {section}")
        if isinstance(payload, dict):
            for k, v in payload.items():
                lines.append(f"- **{k}**: {v}")
        else:
            lines.append(str(payload))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--article-dir", default="data")
    ap.add_argument("--mode", choices=["retrieval", "synthetic", "self-check", "all"],
                    default="all")
    args = ap.parse_args()

    gc = _build_chat_client(TEMPERATURE_CHAT)
    # vision-клиенту обязателен auto_upload_attachments=True (см. runner._build_chat_client).
    gc_vision = _build_chat_client(TEMPERATURE_CHAT, vision=True)
    embeddings = build_embeddings()

    art = run_ingest(Path(args.article_dir), gigachat_chat=gc, gigachat_vision=gc_vision,
                     cache_root=CACHE_ROOT)

    coll = get_or_create_collection(art.cache_dir / "chroma", embeddings)
    if not has_documents(coll):
        upsert_chunks(coll, art.chunks)
    known_numbers = {c.metadata.get("section_number") for c in art.chunks}
    known_top = {c.metadata.get("section_top") for c in art.chunks}
    retriever = Retriever(coll, known_numbers, known_top)

    report: dict = {}
    if args.mode in ("retrieval", "all"):
        report["retrieval"] = retrieval_metrics(art)
    if args.mode in ("synthetic", "all"):
        items = generate_or_load_synthetic(art, gc)
        report["synthetic_recall"] = evaluate_retrieval_recall(items, retriever, embeddings)
        report["synthetic_kinds"] = dict(Counter(it.get("kind") for it in items))
    if args.mode in ("self-check", "all"):
        ans_path = Path("output/answers.txt")
        q_path = Path(args.article_dir) / "questions.txt"
        if ans_path.is_file() and q_path.is_file():
            report["self_check"] = selfcheck(ans_path, q_path, art.readme, gc)
        else:
            report["self_check"] = "answers.txt or questions.txt missing"

    out_dir = Path("eval/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{int(time.time())}_{art.cache_dir.name}.md"
    out_path.write_text(render_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
