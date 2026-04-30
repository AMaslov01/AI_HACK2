"""Чанкинг через LatexTextSplitter с посекционным проходом."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.ingest.latex_clean import strip_format_wrappers
from src.ingest.structure import (
    INCLUDEGFX,
    LABEL_RE,
    REF_RE,
    Section,
    StructureDoc,
)

if TYPE_CHECKING:  # noqa
    pass


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _make_splitter():
    from langchain_text_splitters import LatexTextSplitter
    return LatexTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def _refs_in(span: str) -> list[str]:
    return list({m.group(1) for m in REF_RE.finditer(span)})


def _figures_in(span: str, figure_labels: set[str]) -> list[str]:
    return [r for r in _refs_in(span) if r in figure_labels]


def _equations_in(span: str, equation_labels: set[str]) -> list[str]:
    return [r for r in _refs_in(span) if r in equation_labels]


def _has_includegraphics(span: str) -> bool:
    return INCLUDEGFX.search(span) is not None


def _section_path(structure: StructureDoc, sec: Section) -> str:
    parts: list[str] = []
    cur: Section | None = sec
    seen: set[str] = set()
    while cur is not None and cur.number not in seen:
        seen.add(cur.number)
        parts.append(f"{cur.number} {cur.title}")
        if cur.parent_number is None:
            break
        cur = structure.section_by_number(cur.parent_number)
    return " / ".join(reversed(parts))


def _section_top(num: str) -> str:
    return num.split(".", 1)[0] if num else "0"


def build_chunks(structure: StructureDoc) -> list[Chunk]:
    """Body-чанки через LatexTextSplitter; псевдочанки caption/equation/section_summary
    добавляются отдельными вызовами (см. add_caption_chunks / add_equation_chunks /
    add_section_summary_chunks / add_figure_desc_chunks)."""
    splitter = _make_splitter()
    figure_labels = {f.label for f in structure.figures}
    equation_labels = {e.label for e in structure.equations}
    out: list[Chunk] = []

    for sec in structure.sections:
        body = structure.expanded_text[sec.char_start: sec.char_end]
        if not body.strip():
            continue
        body_clean = strip_format_wrappers(body)
        try:
            pieces = splitter.split_text(body_clean) or [body_clean]
        except Exception:
            pieces = [body_clean]
        section_path = _section_path(structure, sec)
        section_top = _section_top(sec.number)
        for i, piece in enumerate(pieces):
            if not piece.strip():
                continue
            # offsets вычислять точно невозможно после strip — сохраняем приблизительные
            ch_start = sec.char_start + sum(len(p) for p in pieces[:i])
            ch_end = ch_start + len(piece)
            label_refs = _refs_in(piece)
            fig_refs = [r for r in label_refs if r in figure_labels]
            eq_refs = [r for r in label_refs if r in equation_labels]
            meta = {
                "kind": "body",
                "section_path": section_path,
                "section_number": sec.number,
                "section_top": section_top,
                "char_start": ch_start,
                "char_end": ch_end,
                "label_refs": _pipe(label_refs),
                "figure_refs": _pipe(fig_refs),
                "equation_refs": _pipe(eq_refs),
                "has_figure": bool(fig_refs) or _has_includegraphics(piece),
                "has_equation": bool(eq_refs),
                "synthetic": sec.synthetic,
            }
            cid = f"body:{sec.number}#{i}"
            out.append(Chunk(chunk_id=cid, text=piece.strip(), metadata=meta))
    return out


def add_caption_chunks(structure: StructureDoc, base: list[Chunk]) -> list[Chunk]:
    out = list(base)
    for f in structure.figures:
        if not f.caption:
            continue
        sec = structure.section_by_number(f.section)
        section_path = _section_path(structure, sec) if sec else f.section
        meta = {
            "kind": "caption",
            "section_path": section_path,
            "section_number": f.section,
            "section_top": _section_top(f.section),
            "char_start": f.char_pos, "char_end": f.char_pos + 1,
            "label_refs": _pipe([f.label]),
            "figure_refs": _pipe([f.label]),
            "equation_refs": _pipe([]),
            "has_figure": True, "has_equation": False, "synthetic": False,
        }
        out.append(Chunk(chunk_id=f"caption:{f.label}",
                         text=f"Figure {f.label}: {f.caption}", metadata=meta))
    return out


def add_equation_chunks(structure: StructureDoc, base: list[Chunk]) -> list[Chunk]:
    """Equation-чанки: латех + сжатый контекст. Жёсткий лимит на размер,
    чтобы не превысить токен-лимит эмбеддингов GigaChat (514 tok)."""
    from src.config import EMBED_MAX_CHARS
    out = list(base)
    text = structure.expanded_text
    # ±150 символов с каждой стороны — короче, чем раньше (300), чтобы итоговый
    # текст уверенно влезал в 1500 chars.
    ctx_span = 150
    for e in structure.equations:
        latex = e.latex.strip()
        if len(latex) > 800:
            latex = latex[:800] + " ..."
        ctx_a = text[max(0, e.char_pos - ctx_span): e.char_pos].strip()
        ctx_b = text[e.char_pos + len(e.latex): e.char_pos + len(e.latex) + ctx_span].strip()
        sec = structure.section_by_number(e.section)
        section_path = _section_path(structure, sec) if sec else e.section
        body = f"eq {e.label}: {latex}\nBefore: {ctx_a}\nAfter: {ctx_b}"
        if len(body) > EMBED_MAX_CHARS:
            body = body[:EMBED_MAX_CHARS]
        meta = {
            "kind": "equation",
            "section_path": section_path,
            "section_number": e.section,
            "section_top": _section_top(e.section),
            "char_start": e.char_pos, "char_end": e.char_pos + len(e.latex),
            "label_refs": _pipe([e.label]),
            "figure_refs": _pipe([]),
            "equation_refs": _pipe([e.label]),
            "has_figure": False, "has_equation": True, "synthetic": False,
        }
        out.append(Chunk(chunk_id=f"equation:{e.label}", text=body, metadata=meta))
    return out


def add_section_summary_chunks(structure: StructureDoc, base: list[Chunk]) -> list[Chunk]:
    out = list(base)
    for s in structure.sections:
        if not s.dense_summary:
            continue
        section_path = _section_path(structure, s)
        meta = {
            "kind": "section_summary",
            "section_path": section_path,
            "section_number": s.number,
            "section_top": _section_top(s.number),
            "char_start": s.char_start, "char_end": s.char_end,
            "label_refs": _pipe([]),
            "figure_refs": _pipe([]),
            "equation_refs": _pipe([]),
            "has_figure": False, "has_equation": False, "synthetic": s.synthetic,
        }
        body = f"Section {s.number} {s.title}: {s.dense_summary}"
        out.append(Chunk(chunk_id=f"section_summary:{s.number}", text=body, metadata=meta))
    return out


def add_figure_desc_chunks(structure: StructureDoc, base: list[Chunk],
                           figure_records: dict[str, dict]) -> list[Chunk]:
    out = list(base)
    for label, rec in figure_records.items():
        desc = rec.get("description_text") or ""
        if not desc.strip():
            continue
        section = rec.get("section_path") or ""
        section_number = section.split()[0] if section else ""
        meta = {
            "kind": "figure_desc",
            "section_path": section,
            "section_number": section_number,
            "section_top": _section_top(section_number),
            "char_start": 0, "char_end": 0,
            "label_refs": _pipe([label]),
            "figure_refs": _pipe([label]),
            "equation_refs": _pipe([]),
            "has_figure": True, "has_equation": False, "synthetic": False,
        }
        out.append(Chunk(chunk_id=f"figure_desc:{label}",
                         text=f"Figure {label} description: {desc}", metadata=meta))
    return out


def _pipe(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    return "|" + "|".join(items) + "|"
