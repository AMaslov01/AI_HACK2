"""Извлечение структуры из expanded LaTeX. Регекс + balanced-brace сканер."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

TITLE_RE = re.compile(r"\\title\s*(?:\[[^\]]*\])?\s*\{", re.DOTALL)
AUTHOR_RE = re.compile(r"\\author\s*(?:\[[^\]]*\])?\s*\{", re.DOTALL)
ABSTRACT_RE = re.compile(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", re.DOTALL)
HEADING_RE = re.compile(
    r"\\(section|subsection|subsubsection|paragraph)\*?\s*"
    r"(?:\[[^\]]*\])?\s*\{([^}]*)\}"
)
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:ref|eqref|autoref|cref|Cref)\{([^}]+)\}")
INCLUDEGFX = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")
EQ_ENV_RE = re.compile(
    r"\\begin\{(equation|align|gather|multline)\*?\}(.+?)\\end\{\1\*?\}", re.DOTALL
)
CAPTION_RE = re.compile(r"\\caption\s*(?:\[[^\]]*\])?\s*\{")
FIGURE_ENV_RE = re.compile(r"\\begin\{figure\*?\}(.+?)\\end\{figure\*?\}", re.DOTALL)
TABLE_ENV_RE = re.compile(r"\\begin\{table\*?\}(.+?)\\end\{table\*?\}", re.DOTALL)

# keywords-источники
NUM_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?(?:\s*%)?")
TEXTTT_RE = re.compile(r"\\text(?:tt|sf|sc)\{([^{}]+)\}")
VERB_RE = re.compile(r"\\verb\|([^|]+)\|")
PARAM_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


@dataclass
class Section:
    number: str
    title: str
    char_start: int
    char_end: int
    level: int
    label: str | None = None
    parent_number: str | None = None
    synthetic: bool = False
    title_hint: str = ""
    dense_summary: str = ""
    keywords: list[str] = field(default_factory=list)


@dataclass
class FigureRaw:
    label: str
    tex_path: str
    section: str
    caption: str
    char_pos: int


@dataclass
class EquationRaw:
    label: str
    section: str
    latex: str
    char_pos: int


@dataclass
class StructureDoc:
    title: str
    authors: list[str]
    abstract: str
    sections: list[Section]
    figures: list[FigureRaw]
    equations: list[EquationRaw]
    label_to_target: dict[str, str]
    label_referenced_by: dict[str, list[str]]
    defined_macros: dict[str, str]
    expanded_text: str

    @property
    def length_chars(self) -> int:
        return len(self.expanded_text)

    def section_by_number(self, num: str) -> Section | None:
        for s in self.sections:
            if s.number == num:
                return s
        return None


def _balanced_brace_extract(text: str, open_pos: int) -> tuple[str, int] | None:
    """Извлекает содержимое { ... } начиная с позиции открывающей скобки.
    Возвращает (тело, end_pos). end_pos — индекс ПОСЛЕ закрывающей скобки."""
    depth = 0
    i = open_pos
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if c == "{":
            depth += 1
            if depth == 1:
                start = i + 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i], i + 1
        i += 1
    return None


def _extract_braced(text: str, regex: re.Pattern[str]) -> str:
    m = regex.search(text)
    if not m:
        return ""
    open_brace = text.find("{", m.start())
    if open_brace == -1:
        return ""
    body = _balanced_brace_extract(text, open_brace)
    return (body[0] if body else "").strip()


def _build_section_number(stack: list[int]) -> str:
    return ".".join(str(x) for x in stack if x > 0)


def extract_sections(text: str) -> list[Section]:
    """Строит дерево секций по последовательности заголовков."""
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [Section(number="0", title="(document)", char_start=0,
                        char_end=len(text), level=1)]
    levels = {"section": 1, "subsection": 2, "subsubsection": 3, "paragraph": 4}
    sections: list[Section] = []
    counters = [0, 0, 0, 0]  # 1..4
    parent_stack: list[str] = [None] * 5  # type: ignore
    for i, m in enumerate(matches):
        kind = m.group(1)
        level = levels[kind]
        # сброс глубже идущих счётчиков
        for j in range(level, 4):
            counters[j] = 0
        counters[level - 1] += 1
        number = _build_section_number(counters[:level])
        title = m.group(2).strip()
        # извлекаем label сразу после заголовка — ищем в небольшом окне ДО ближайшего
        # \begin{...} или нового \section/\subsection (чтобы не подобрать label из
        # equation/figure внутри секции)
        char_start = m.start()
        char_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body_after = text[m.end(): m.end() + 120]
        cut = re.search(r"\\(begin|section|subsection|subsubsection|paragraph|input|include)\b",
                        body_after)
        if cut is not None:
            body_after = body_after[: cut.start()]
        lbl_m = LABEL_RE.search(body_after)
        label = lbl_m.group(1) if lbl_m else None
        parent_stack[level] = number
        parent_number = parent_stack[level - 1] if level > 1 else None
        sections.append(Section(
            number=number, title=title, char_start=char_start, char_end=char_end,
            level=level, label=label, parent_number=parent_number,
        ))
    return sections


def _find_section_for_pos(sections: list[Section], pos: int) -> str:
    """Возвращает наиболее глубокий section.number, в чьём диапазоне лежит pos."""
    best = "0"
    best_level = 0
    for s in sections:
        if s.char_start <= pos < s.char_end and s.level > best_level:
            best = s.number
            best_level = s.level
    return best


def extract_figures(text: str, sections: list[Section]) -> list[FigureRaw]:
    out: list[FigureRaw] = []
    auto_idx = 0
    for env in FIGURE_ENV_RE.finditer(text):
        env_text = env.group(1)
        env_pos = env.start()
        # \includegraphics
        gfx_m = INCLUDEGFX.search(env_text)
        if not gfx_m:
            continue
        tex_path = gfx_m.group(1).strip()
        # caption через balanced
        caption = ""
        cap_m = CAPTION_RE.search(env_text)
        if cap_m:
            ob = env_text.find("{", cap_m.start())
            if ob != -1:
                cb = _balanced_brace_extract(env_text, ob)
                if cb:
                    caption = cb[0].strip()
        # label
        lbl_m = LABEL_RE.search(env_text)
        if lbl_m:
            label = lbl_m.group(1)
        else:
            auto_idx += 1
            label = f"fig:auto-{auto_idx}"
        section = _find_section_for_pos(sections, env_pos)
        out.append(FigureRaw(label=label, tex_path=tex_path, section=section,
                             caption=caption, char_pos=env_pos))
    # включения вне figure-env (одиночные \includegraphics)
    for gfx in INCLUDEGFX.finditer(text):
        # пропускаем те, что уже в figure-env
        if any(f.char_pos <= gfx.start() < f.char_pos + 4000 for f in out):
            continue
        auto_idx += 1
        label = f"fig:auto-{auto_idx}"
        section = _find_section_for_pos(sections, gfx.start())
        out.append(FigureRaw(label=label, tex_path=gfx.group(1).strip(),
                             section=section, caption="", char_pos=gfx.start()))
    return out


def extract_equations(text: str, sections: list[Section]) -> list[EquationRaw]:
    out: list[EquationRaw] = []
    auto_idx = 0
    for m in EQ_ENV_RE.finditer(text):
        body = m.group(2)
        lbl_m = LABEL_RE.search(body)
        if lbl_m:
            label = lbl_m.group(1)
        else:
            auto_idx += 1
            label = f"eq:auto-{auto_idx}"
        section = _find_section_for_pos(sections, m.start())
        out.append(EquationRaw(label=label, section=section,
                               latex=body.strip(), char_pos=m.start()))
    return out


def build_label_index(text: str, figures: list[FigureRaw], equations: list[EquationRaw],
                      sections: list[Section]) -> tuple[dict[str, str], dict[str, list[str]]]:
    # Порядок: сначала секции, потом equations и figures — последние более специфичны
    # и должны переопределять секционную атрибуцию label-а.
    label_to_target: dict[str, str] = {}
    for s in sections:
        if s.label:
            label_to_target[s.label] = "section"
    for e in equations:
        label_to_target[e.label] = "equation"
    for f in figures:
        label_to_target[f.label] = "figure"

    # обратная: где ссылаются?
    referenced_by: dict[str, list[str]] = {}
    for m in REF_RE.finditer(text):
        target = m.group(1)
        ref_section = _find_section_for_pos(sections, m.start())
        referenced_by.setdefault(target, []).append(ref_section)
    for k, v in list(referenced_by.items()):
        # дедуп с сохранением порядка
        seen: set[str] = set()
        uniq: list[str] = []
        for s in v:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        referenced_by[k] = uniq
    return label_to_target, referenced_by


def extract_keywords(section_text: str, max_n: int = 15) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        s = s.strip()
        if not s or len(s) > 40 or s in seen:
            return
        seen.add(s)
        out.append(s)

    for m in PARAM_RE.finditer(section_text):
        add(f"{m.group(1)}={m.group(2)}")
    for m in TEXTTT_RE.finditer(section_text):
        add(m.group(1))
    for m in VERB_RE.finditer(section_text):
        add(m.group(1))
    # значимые числа: > 1 знака или с десятичной частью
    for m in NUM_RE.finditer(section_text):
        v = m.group(0).strip()
        if "." in v or "," in v or len(v.replace("%", "").strip()) > 1:
            add(v)
        if len(out) >= max_n:
            break
    return out[:max_n]


def parse_structure(expanded: str, defined_macros: dict[str, str]) -> StructureDoc:
    title = _extract_braced(expanded, TITLE_RE)
    authors_blob = _extract_braced(expanded, AUTHOR_RE)
    authors = _split_authors(authors_blob) if authors_blob else []
    abstract = ""
    am = ABSTRACT_RE.search(expanded)
    if am:
        abstract = am.group(1).strip()
    sections = extract_sections(expanded)
    figures = extract_figures(expanded, sections)
    equations = extract_equations(expanded, sections)
    label_to_target, referenced_by = build_label_index(expanded, figures, equations, sections)
    # keywords per section
    for s in sections:
        body = expanded[s.char_start:s.char_end]
        s.keywords = extract_keywords(body)
    return StructureDoc(
        title=title, authors=authors, abstract=abstract,
        sections=sections, figures=figures, equations=equations,
        label_to_target=label_to_target, label_referenced_by=referenced_by,
        defined_macros={k: v for k, v in defined_macros.items() if not k.startswith("__")},
        expanded_text=expanded,
    )


def _split_authors(blob: str) -> list[str]:
    """Очень грубо: режем по \\and / запятой / переносу строки. Чистим \\thanks."""
    s = re.sub(r"\\thanks\s*\{[^{}]*\}", "", blob)
    s = re.sub(r"\\(?:textsuperscript|footnote)\s*\{[^{}]*\}", "", s)
    parts = re.split(r"\\and|\n+", s)
    out = [re.sub(r"\\[a-zA-Z]+\*?\{?", "", p).replace("{", "").replace("}", "").strip()
           for p in parts]
    out = [re.sub(r"\s+", " ", p).strip(",;. ") for p in out if p.strip()]
    return [p for p in out if p]


def chunks_metadata_view(structure: StructureDoc) -> Iterable[dict]:
    """Удобный обзор для дебага."""
    for s in structure.sections:
        yield {"section_number": s.number, "title": s.title,
               "level": s.level, "label": s.label, "synthetic": s.synthetic}
