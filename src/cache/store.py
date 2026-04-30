"""Атомарная сериализация артефактов ингеста."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from src.ingest.chunker import Chunk
from src.ingest.figures import (
    FigureRecord,
    deserialize_records,
    serialize_records,
)
from src.ingest.structure import (
    EquationRaw,
    FigureRaw,
    Section,
    StructureDoc,
)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_structure(cache_dir: Path, structure: StructureDoc) -> None:
    payload = {
        "title": structure.title,
        "authors": structure.authors,
        "abstract": structure.abstract,
        "sections": [asdict(s) for s in structure.sections],
        "figures": [asdict(f) for f in structure.figures],
        "equations": [asdict(e) for e in structure.equations],
        "label_to_target": structure.label_to_target,
        "label_referenced_by": structure.label_referenced_by,
        "defined_macros": structure.defined_macros,
    }
    _atomic_write_text(cache_dir / "structure.json",
                       json.dumps(payload, ensure_ascii=False, indent=2))
    _atomic_write_text(cache_dir / "expanded.tex", structure.expanded_text)


def load_structure(cache_dir: Path) -> StructureDoc | None:
    sp = cache_dir / "structure.json"
    ep = cache_dir / "expanded.tex"
    if not (sp.is_file() and ep.is_file()):
        return None
    data = json.loads(sp.read_text(encoding="utf-8"))
    expanded = ep.read_text(encoding="utf-8")
    sections = [Section(**s) for s in data["sections"]]
    figures = [FigureRaw(**f) for f in data["figures"]]
    equations = [EquationRaw(**e) for e in data["equations"]]
    return StructureDoc(
        title=data["title"], authors=data["authors"], abstract=data["abstract"],
        sections=sections, figures=figures, equations=equations,
        label_to_target=data["label_to_target"],
        label_referenced_by=data["label_referenced_by"],
        defined_macros=data.get("defined_macros", {}),
        expanded_text=expanded,
    )


def save_chunks(cache_dir: Path, chunks: list[Chunk]) -> None:
    lines = [json.dumps({"chunk_id": c.chunk_id, "text": c.text,
                         "metadata": c.metadata}, ensure_ascii=False)
             for c in chunks]
    _atomic_write_text(cache_dir / "chunks.jsonl", "\n".join(lines) + "\n")


def load_chunks(cache_dir: Path) -> list[Chunk] | None:
    p = cache_dir / "chunks.jsonl"
    if not p.is_file():
        return None
    out: list[Chunk] = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        d = json.loads(ln)
        out.append(Chunk(chunk_id=d["chunk_id"], text=d["text"],
                         metadata=d.get("metadata", {})))
    return out


def save_figures(cache_dir: Path, figures: dict[str, FigureRecord]) -> None:
    items = serialize_records(figures)
    p = cache_dir / "figures" / "descriptions.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(p, json.dumps(items, ensure_ascii=False, indent=2))


def load_figures(cache_dir: Path) -> dict[str, FigureRecord] | None:
    p = cache_dir / "figures" / "descriptions.json"
    if not p.is_file():
        return None
    items = json.loads(p.read_text(encoding="utf-8"))
    return deserialize_records(items)


def save_readme(cache_dir: Path, text: str) -> None:
    _atomic_write_text(cache_dir / "article_readme.md", text)


def load_readme(cache_dir: Path) -> str | None:
    p = cache_dir / "article_readme.md"
    if not p.is_file():
        return None
    return p.read_text(encoding="utf-8")
