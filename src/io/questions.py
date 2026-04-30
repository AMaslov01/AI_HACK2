"""Парсинг data/questions.txt. Зеркалит логику src/utils/check_submission.py:
если есть строки `## Question N` — формат markdown, иначе — numbered."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FormatKind = Literal["markdown", "numbered"]

_MD_QUESTION = re.compile(r"^##\s*Question\s+(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_NUMBERED = re.compile(r"^\s*(\d+)[\).]\s+(.*\S)\s*$")


@dataclass
class Question:
    idx: int           # 1-based
    text: str


@dataclass
class QuestionsDoc:
    items: list[Question]
    fmt: FormatKind

    def __len__(self) -> int:
        return len(self.items)


def _has_md_questions(text: str) -> bool:
    return any(_MD_QUESTION.match(ln.strip()) for ln in text.splitlines())


def _parse_markdown(text: str) -> list[Question]:
    matches = list(_MD_QUESTION.finditer(text))
    if not matches:
        raise ValueError("Markdown-режим: не нашёл ни одного `## Question N`")
    items: list[Question] = []
    for i, m in enumerate(matches):
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        items.append(Question(idx=int(m.group(1)), text=body))
    return items


def _parse_numbered(text: str) -> list[Question]:
    items: list[Question] = []
    current_idx: int | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        m = _NUMBERED.match(line)
        if m:
            if current_idx is not None and current_lines:
                items.append(Question(idx=current_idx, text="\n".join(current_lines).strip()))
            current_idx = int(m.group(1))
            current_lines = [m.group(2)]
        else:
            if line.strip() and current_idx is not None:
                current_lines.append(line)
    if current_idx is not None and current_lines:
        items.append(Question(idx=current_idx, text="\n".join(current_lines).strip()))
    if not items:
        raise ValueError("Numbered-режим: не нашёл ни одной строки `1. ...`/`1) ...`")
    return items


def load_questions(path: Path) -> QuestionsDoc:
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"{path}: файл пуст")
    if _has_md_questions(text):
        return QuestionsDoc(items=_parse_markdown(text), fmt="markdown")
    return QuestionsDoc(items=_parse_numbered(text), fmt="numbered")
