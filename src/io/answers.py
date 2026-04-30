"""Запись output/answers.txt. Гарантирует ровно N блоков по числу вопросов."""

from __future__ import annotations

from pathlib import Path

from src.io.questions import FormatKind


def format_block(idx: int, text: str, fmt: FormatKind) -> str:
    body = (text or "").strip() or "no answer"
    if fmt == "markdown":
        return f"## Answer {idx}\n{body}"
    return f"{idx}. {body}"


def stub_block(idx: int, fmt: FormatKind) -> str:
    return format_block(idx, "no answer", fmt)


def write_answers(path: Path, blocks: list[str], fmt: FormatKind) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "\n\n" if fmt == "markdown" else "\n"
    path.write_text(sep.join(blocks) + "\n", encoding="utf-8")
