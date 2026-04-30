"""Чистка LaTeX: комментарии, макросы, обёртки форматирования, библиография."""

from __future__ import annotations

import re
from pathlib import Path

NEWCMD_RE = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\\(\w+)\}?"
    r"\s*(?:\[(\d+)\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}"
)
DEF_RE = re.compile(r"\\def\\(\w+)\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
COMMENT_RE = re.compile(r"(?<!\\)%.*$")
BIBLIO_RE = re.compile(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", re.DOTALL)
IFFALSE_RE = re.compile(r"\\iffalse\b.*?\\fi\b", re.DOTALL)

# strip-обёртки — один проход
WRAPPER_RE = re.compile(r"\\[a-zA-Z]+\*?\{([^{}]*)\}")

# окружения, в которых не схлопываем пустые строки
PROTECTED_BEGIN = re.compile(r"\\begin\{(verbatim|lstlisting|align\*?|equation\*?|gather\*?|multline\*?)\}")
PROTECTED_END = re.compile(r"\\end\{(verbatim|lstlisting|align\*?|equation\*?|gather\*?|multline\*?)\}")


def strip_comments(text: str) -> str:
    return "\n".join(COMMENT_RE.sub("", ln) for ln in text.splitlines())


def collect_macros(texts: list[str]) -> dict[str, str]:
    """Собирает 0-арг и многоарг макросы из всех source-текстов.
    Возвращает {macro_name: expansion_body}. n_args не сохраняем — для ingest достаточно."""
    out: dict[str, str] = {}
    for t in texts:
        for m in NEWCMD_RE.finditer(t):
            name, n_args, body = m.group(1), m.group(2), m.group(3)
            out[name] = body  # храним только тело (для glossary)
            if not n_args:
                out.setdefault(f"__zero__:{name}", body)
        for m in DEF_RE.finditer(t):
            name, body = m.group(1), m.group(2)
            out[name] = body
            out.setdefault(f"__zero__:{name}", body)
    return out


def inline_zero_arg_macros(text: str, macros: dict[str, str]) -> str:
    """Один проход подстановки 0-арг макросов. Не рекурсивно."""
    zero = {k.split(":", 1)[1]: v for k, v in macros.items() if k.startswith("__zero__:")}
    if not zero:
        return text

    # \name (без аргументов) → body. Используем границу не-буква, чтобы \alpha не съело \alphabet.
    def replace(name: str, body: str, t: str) -> str:
        return re.sub(r"\\" + re.escape(name) + r"(?![a-zA-Z@])", lambda _m: body, t)

    for name, body in zero.items():
        text = replace(name, body, text)
    return text


def strip_format_wrappers(text: str) -> str:
    """Убирает \\bfdelta{X} → X. Один проход; смысловые \\caption/\\label сохраняем извне."""
    # smart-protect: временно маскируем критичные команды, чтобы не съесть.
    SAFE = {"caption", "label", "ref", "eqref", "autoref", "cref", "Cref",
            "section", "subsection", "subsubsection", "paragraph",
            "title", "author", "abstract", "includegraphics",
            "begin", "end", "input", "include", "subfile",
            "newcommand", "renewcommand", "providecommand", "def",
            "graphicspath", "cite"}

    placeholders: dict[str, str] = {}
    counter = [0]

    def mask(m: re.Match[str]) -> str:
        cmd = re.match(r"\\([a-zA-Z]+)", m.group(0)).group(1)  # type: ignore[union-attr]
        if cmd in SAFE:
            counter[0] += 1
            tok = f"\x00MASK{counter[0]}\x00"
            placeholders[tok] = m.group(0)
            return tok
        return m.group(0)

    masked = re.sub(r"\\[a-zA-Z]+\*?\{[^{}]*\}", mask, text)
    stripped = WRAPPER_RE.sub(r"\1", masked)
    for tok, original in placeholders.items():
        stripped = stripped.replace(tok, original)
    return stripped


def remove_bibliography(text: str) -> str:
    return BIBLIO_RE.sub("", text)


def remove_iffalse(text: str) -> str:
    return IFFALSE_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    """3+ пустые строки → 2, кроме защищённых окружений."""
    lines = text.splitlines()
    out: list[str] = []
    in_protected = 0
    blank_run = 0
    for ln in lines:
        if PROTECTED_BEGIN.search(ln):
            in_protected += 1
        if not ln.strip():
            if in_protected > 0:
                out.append(ln)
            else:
                blank_run += 1
                if blank_run <= 2:
                    out.append("")
        else:
            blank_run = 0
            out.append(ln.rstrip())
        if PROTECTED_END.search(ln):
            in_protected = max(0, in_protected - 1)
    return "\n".join(out)


def gather_macro_sources(tex_root: Path, entry: Path) -> list[str]:
    """Собирает тексты для парсинга макросов: macros.tex, preamble.tex, .sty, начало entry."""
    out: list[str] = []
    for name in ("macros.tex", "preamble.tex", "defs.tex", "commands.tex"):
        p = tex_root / name
        if p.is_file():
            try:
                out.append(p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
    for sty in tex_root.rglob("*.sty"):
        try:
            out.append(sty.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    # начало entry (до \begin{document})
    try:
        entry_text = entry.read_text(encoding="utf-8", errors="replace")
        idx = entry_text.find(r"\begin{document}")
        out.append(entry_text[: idx if idx > 0 else len(entry_text)])
    except OSError:
        pass
    return out


def clean_pipeline(raw: str, macros: dict[str, str]) -> str:
    """Основной пайплайн чистки. Возвращает 'expanded.tex' стиль."""
    t = raw
    t = remove_iffalse(t)
    t = strip_comments(t)
    t = inline_zero_arg_macros(t, macros)
    t = remove_bibliography(t)
    t = normalize_whitespace(t)
    return t
