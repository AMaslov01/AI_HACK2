"""Сборка article_readme.md — карты статьи для агента."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from src.config import (
    CHAT_CONCURRENCY,
    README_MAX_CHARS,
    SECTION_SUMMARY_MAX_SECTIONS,
)
from src.ingest.figures import FigureRecord
from src.ingest.structure import StructureDoc
from src.utils.logging import log
from src.utils.timing import run_with_timeout


SUMMARY_PROMPT = """\
Дай два саммари этой секции — JSON с полями:
- "title_hint": ≤ 25 слов, уйдёт в карту статьи
- "dense_summary": 80–100 слов, уйдёт как поисковый чанк

Текст:
{body}

Только JSON без обёрток.
"""


def _parse_json(s: str) -> dict:
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


def fill_section_summaries(structure: StructureDoc, gigachat_client) -> None:
    """Параллельно заполняет section.title_hint и section.dense_summary.
    Если секций > SECTION_SUMMARY_MAX_SECTIONS — пропускаем (только заголовки)."""
    if gigachat_client is None:
        return
    sections = [s for s in structure.sections
                if s.char_end - s.char_start > 200 and s.number != "0"]
    if not sections or len(sections) > SECTION_SUMMARY_MAX_SECTIONS:
        log.info(f"section_summaries: skip ({len(sections)} sections)")
        return

    def call(sec) -> tuple[str, dict | None, str | None]:
        body = structure.expanded_text[sec.char_start: sec.char_start + 3000]
        prompt = SUMMARY_PROMPT.format(body=body)
        try:
            resp = run_with_timeout(gigachat_client.invoke, 30.0, prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            return sec.number, _parse_json(content), None
        except Exception as e:
            return sec.number, None, str(e)

    with ThreadPoolExecutor(max_workers=CHAT_CONCURRENCY) as ex:
        futs = {ex.submit(call, s): s for s in sections}
        for fut in as_completed(futs):
            sec = futs[fut]
            num, parsed, err = fut.result()
            if err or not parsed:
                if err:
                    log.warning(f"summary[{num}] failed: {err}")
                continue
            sec.title_hint = (parsed.get("title_hint") or "").strip()[:300]
            sec.dense_summary = (parsed.get("dense_summary") or "").strip()[:1000]


def build_readme_markdown(structure: StructureDoc,
                          figures: dict[str, FigureRecord]) -> str:
    lines: list[str] = []
    lines.append("# Article Map")

    # Identity
    lines.append("\n## Identity")
    lines.append(f"- **Title**: {structure.title or '(unknown)'}")
    if structure.authors:
        lines.append(f"- **Authors**: {', '.join(structure.authors)}")
    n_chunks_est = max(1, structure.length_chars // 1800)
    lines.append(f"- **Length**: {structure.length_chars} chars / ~{n_chunks_est} chunks")
    if structure.abstract:
        abs_one_line = re.sub(r"\s+", " ", structure.abstract).strip()[:1500]
        lines.append(f"- **Abstract**: {abs_one_line}")

    # Outline
    lines.append("\n## Outline")
    for s in structure.sections:
        if s.number == "0":
            continue
        indent = "  " * (s.level - 1)
        synth_mark = " *(synthetic)*" if s.synthetic else ""
        hint = f" — {s.title_hint}" if s.title_hint else ""
        lines.append(f"{indent}- **{s.number} {s.title}**{synth_mark}{hint}")
        if s.level == 1 and s.keywords:
            kws = ", ".join(s.keywords[:8])
            lines.append(f"{indent}  - Keywords: {kws}")

    # Figures
    if figures:
        lines.append("\n## Figures")
        for fid, rec in figures.items():
            cap = (rec.caption or "(no caption)").replace("\n", " ")[:200]
            desc = rec.description_text.replace("\n", " ")[:300] if rec.description_text else ""
            lines.append(f"- **{fid}** (Section {rec.section_path}) — *Caption*: {cap}"
                         + (f" — *Description*: {desc}" if desc else ""))

    # Equations
    if structure.equations:
        lines.append("\n## Equations of Interest")
        for e in structure.equations[:30]:
            latex_one = re.sub(r"\s+", " ", e.latex).strip()[:200]
            lines.append(f"- **{e.label}** (Section {e.section}) — `{latex_one}`")

    # Cross-Reference Index
    if structure.label_referenced_by:
        lines.append("\n## Cross-Reference Index")
        for label, secs in list(structure.label_referenced_by.items())[:60]:
            secs_str = ", ".join(f"§{s}" for s in secs[:8])
            lines.append(f"- {label} ссылается из: {secs_str}")

    # Defined Macros
    if structure.defined_macros:
        lines.append("\n## Defined Macros")
        for name, body in list(structure.defined_macros.items())[:40]:
            body_one = re.sub(r"\s+", " ", body).strip()[:120]
            lines.append(f"- `\\{name}` → `{body_one}`")

    md = "\n".join(lines)
    if len(md) > README_MAX_CHARS:
        md = md[:README_MAX_CHARS] + "\n\n[...труncated...]"
    return md
