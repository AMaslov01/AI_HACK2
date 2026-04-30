"""Top-level оркестратор ингеста: data/ → IngestArtifacts (с кэшем)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from src.cache.fingerprint import fingerprint, read_fingerprint_meta, write_fingerprint_meta
from src.cache.store import (
    load_chunks,
    load_figures,
    load_readme,
    load_structure,
    save_chunks,
    save_figures,
    save_readme,
    save_structure,
)
from src.config import CACHE_ROOT, SCHEMA_VERSION
from src.ingest.chunker import (
    Chunk,
    add_caption_chunks,
    add_equation_chunks,
    add_figure_desc_chunks,
    add_section_summary_chunks,
    build_chunks,
)
from src.ingest.discovery import expand_inputs, find_tex_files, find_tex_root, pick_entrypoint
from src.ingest.figures import FigureRecord, describe_figures
from src.ingest.latex_clean import (
    clean_pipeline,
    collect_macros,
    gather_macro_sources,
)
from src.ingest.readme_builder import build_readme_markdown, fill_section_summaries
from src.ingest.structure import StructureDoc, parse_structure
from src.utils.logging import log


@dataclass
class IngestArtifacts:
    cache_dir: Path
    structure: StructureDoc
    chunks: list[Chunk]
    figures: dict[str, FigureRecord]
    readme: str
    timings: dict
    cache_hit: bool


def _verify_cache(cache_dir: Path) -> bool:
    meta = read_fingerprint_meta(cache_dir)
    if not meta or meta.get("schema_version") != SCHEMA_VERSION:
        return False
    needed = ["structure.json", "expanded.tex", "chunks.jsonl", "article_readme.md"]
    return all((cache_dir / n).is_file() for n in needed)


def _load_from_cache(cache_dir: Path) -> IngestArtifacts | None:
    structure = load_structure(cache_dir)
    if structure is None:
        return None
    chunks = load_chunks(cache_dir)
    if chunks is None:
        return None
    readme = load_readme(cache_dir)
    if readme is None:
        return None
    figures = load_figures(cache_dir) or {}
    log.info(f"cache hit: {cache_dir}")
    return IngestArtifacts(
        cache_dir=cache_dir, structure=structure, chunks=chunks,
        figures=figures, readme=readme, timings={"cache_hit": 0.0}, cache_hit=True,
    )


def run_ingest(data_dir: Path, *, gigachat_chat=None, gigachat_vision=None,
               cache_root: Path = CACHE_ROOT) -> IngestArtifacts:
    """Главный entry. Если есть валидный кэш — грузит и возвращает."""
    timings: dict = {}
    t0 = time.time()

    tex_root = find_tex_root(data_dir)
    tex_files = find_tex_files(tex_root)
    fp = fingerprint(tex_files, tex_root)
    cache_dir = cache_root / fp
    timings["discovery"] = time.time() - t0

    if _verify_cache(cache_dir):
        cached = _load_from_cache(cache_dir)
        if cached is not None:
            return cached

    log.info(f"cache miss / partial: {cache_dir} — full ingest")
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_fingerprint_meta(cache_dir, fp, tex_files, tex_root)

    # 1. Expansion + cleaning
    t = time.time()
    entry = pick_entrypoint(tex_files)
    raw = expand_inputs(entry, tex_root)
    macro_sources = gather_macro_sources(tex_root, entry)
    macros = collect_macros(macro_sources + [raw])
    expanded = clean_pipeline(raw, macros)
    timings["expand_clean"] = time.time() - t

    # 2. Structure
    t = time.time()
    structure = parse_structure(expanded, macros)
    timings["structure"] = time.time() - t
    log.info(f"structure: {len(structure.sections)} sections, "
             f"{len(structure.figures)} figures, {len(structure.equations)} equations")

    # 3. Section summaries (требует LLM)
    t = time.time()
    if gigachat_chat is not None:
        try:
            fill_section_summaries(structure, gigachat_chat)
        except Exception as e:
            log.warning(f"section_summaries failed: {e}")
    timings["section_summaries"] = time.time() - t

    # 4. Figures + vision
    t = time.time()
    figures = describe_figures(
        figures_raw=structure.figures, structure=structure,
        tex_root=tex_root, cache_dir=cache_dir, gigachat_client=gigachat_vision,
    )
    timings["figures"] = time.time() - t

    # 5. Chunking
    t = time.time()
    chunks = build_chunks(structure)
    chunks = add_caption_chunks(structure, chunks)
    chunks = add_equation_chunks(structure, chunks)
    chunks = add_section_summary_chunks(structure, chunks)
    figs_for_chunks = {fid: {"description_text": rec.description_text,
                             "section_path": rec.section_path}
                       for fid, rec in figures.items()}
    chunks = add_figure_desc_chunks(structure, chunks, figs_for_chunks)
    timings["chunking"] = time.time() - t
    log.info(f"chunks: {len(chunks)}")

    # 6. README
    t = time.time()
    readme_md = build_readme_markdown(structure, figures)
    timings["readme"] = time.time() - t

    # 7. Persist
    t = time.time()
    save_structure(cache_dir, structure)
    save_chunks(cache_dir, chunks)
    save_figures(cache_dir, figures)
    save_readme(cache_dir, readme_md)
    (cache_dir / "ingest_log.json").write_text(
        json.dumps({"timings": timings, "ts": time.time()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    timings["persist"] = time.time() - t

    return IngestArtifacts(
        cache_dir=cache_dir, structure=structure, chunks=chunks,
        figures=figures, readme=readme_md, timings=timings, cache_hit=False,
    )
