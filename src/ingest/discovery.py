"""Поиск .tex, выбор entry-point, рекурсивное раскрытие \\input/\\include/\\subfile."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

DOCCLASS_RE = re.compile(r"^\s*\\documentclass\b", re.MULTILINE)
INPUT_RE = re.compile(r"\\(?:input|include|subfile)\s*\{([^}]+)\}")
GRAPHICSPATH_RE = re.compile(r"\\graphicspath\s*\{((?:\{[^}]*\})+)\}")
_BAD_NAMES = re.compile(r"^(macros|preamble|defs|appendix\d*|references)$", re.IGNORECASE)


def find_tex_root(data_dir: Path) -> Path:
    """Находит корневую папку с .tex. Приоритет — подпапка вида 'tex source'."""
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    # 1. подпапки с именем 'tex source' (case-insensitive)
    for sub in data_dir.rglob("*"):
        if sub.is_dir() and sub.name.lower().replace("_", " ") in ("tex source", "tex-source", "texsource"):
            if any(sub.rglob("*.tex")):
                return sub
    # 2. родитель любого .tex
    tex_files = list(data_dir.rglob("*.tex"))
    if not tex_files:
        raise FileNotFoundError(f"В {data_dir} не найдено ни одного .tex")
    # выбираем тот корень, под которым максимум .tex
    candidates: dict[Path, int] = {}
    for f in tex_files:
        for parent in f.parents:
            if data_dir in parent.parents or parent == data_dir:
                candidates[parent] = candidates.get(parent, 0) + 1
            if parent == data_dir:
                break
    if not candidates:
        return tex_files[0].parent
    # самый «глубокий», у которого ещё есть .tex
    best = max(candidates.items(), key=lambda kv: (kv[1], -len(kv[0].parts)))
    return best[0]


def find_tex_files(tex_root: Path) -> list[Path]:
    return sorted(tex_root.rglob("*.tex"))


def pick_entrypoint(tex_files: list[Path]) -> Path:
    if not tex_files:
        raise FileNotFoundError("нет .tex файлов")
    if len(tex_files) == 1:
        return tex_files[0]

    def has_documentclass(p: Path) -> bool:
        try:
            return bool(DOCCLASS_RE.search(p.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            return False

    with_class = [p for p in tex_files if has_documentclass(p)]
    if len(with_class) == 1:
        return with_class[0]
    pool = with_class or tex_files

    def n_inputs(p: Path) -> int:
        try:
            return len(INPUT_RE.findall(p.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            return 0

    pool_sorted = sorted(pool, key=lambda p: (-n_inputs(p), p.stat().st_size * -1))
    # если есть несколько с равным числом inputs — отсеять «плохие имена» и подпапки figures/sections
    def looks_main(p: Path) -> bool:
        if _BAD_NAMES.match(p.stem):
            return False
        for part in p.parts:
            if part.lower() in ("figures", "sections", "fig", "img", "images"):
                return False
        return True

    for p in pool_sorted:
        if looks_main(p):
            return p
    return pool_sorted[0]


def expand_inputs(entry: Path, tex_root: Path) -> str:
    """Рекурсивно раскрывает \\input/\\include/\\subfile, защита от циклов."""
    seen: set[Path] = set()

    def _read(p: Path) -> str:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        if text and text[0] == "﻿":
            text = text[1:]
        return unicodedata.normalize("NFC", text)

    def _resolve(target: str, current_dir: Path) -> Path | None:
        t = target.strip()
        candidates: list[Path] = []
        names = [t] if t.endswith(".tex") else [t + ".tex", t]
        for name in names:
            candidates += [
                (current_dir / name).resolve(),
                (tex_root / name).resolve(),
            ]
            if not Path(name).is_absolute():
                # если name содержит подпапку — пробуем её относительно root
                candidates.append((tex_root / name).resolve())
        for c in candidates:
            if c.exists() and c.is_file():
                return c
        return None

    def _expand(p: Path) -> str:
        if p in seen:
            return ""
        seen.add(p)
        text = _read(p)

        def repl(m: re.Match[str]) -> str:
            target = m.group(1).strip()
            child = _resolve(target, p.parent)
            if child is None:
                return ""
            return _expand(child)

        return INPUT_RE.sub(repl, text)

    return _expand(entry)


def find_graphicspaths(text: str, tex_root: Path) -> list[Path]:
    out: list[Path] = []
    for m in GRAPHICSPATH_RE.finditer(text):
        for sub in re.findall(r"\{([^}]+)\}", m.group(1)):
            cand = (tex_root / sub).resolve()
            if cand.exists():
                out.append(cand)
    return out
