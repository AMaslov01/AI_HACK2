from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
ENV_PATH = ROOT / ".env"
DEFAULT_QUESTIONS_PATH = DATA_DIR / "questions.txt"
DEFAULT_ANSWERS_PATH = OUTPUT_DIR / "answers.txt"

# «N. » / «N) » — после точки/скобки пробел + текст (не путать с «2.1» в тексте)
_NUMBERED = re.compile(r"^\d+[\).]\s+\S")

_MD_QUESTION = re.compile(r"^##\s*Question\s+(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_MD_ANSWER = re.compile(r"^##\s*Answer\s+(\d+)\s*$", re.IGNORECASE | re.MULTILINE)


def _is_numbered_start(line_stripped: str) -> bool:
    return _NUMBERED.match(line_stripped) is not None


def ensure_env_file() -> None:
    assert ENV_PATH.is_file(), "Нет файла .env"
    content = ENV_PATH.read_text(encoding="utf-8-sig").splitlines()
    assert any(ln.strip().startswith("GIGACHAT_CREDENTIALS") for ln in content), (
        "В .env нет GIGACHAT_CREDENTIALS"
    )
    assert any(ln.strip().startswith("GIGACHAT_SCOPE") for ln in content), (
        "В .env нет GIGACHAT_SCOPE"
    )


def _has_md_questions(text: str) -> bool:
    return any(_MD_QUESTION.match(ln.strip()) for ln in text.splitlines())


def _has_md_answers(text: str) -> bool:
    return any(_MD_ANSWER.match(ln.strip()) for ln in text.splitlines())


def count_questions_line_based(path: Path) -> int:
    n = len([ln for ln in path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()])
    assert n > 0, f"В {path} нет непустых строк с вопросами"
    return n


def count_questions_markdown(text: str) -> int:
    nums: list[int] = []
    for ln in text.splitlines():
        m = _MD_QUESTION.match(ln.strip())
        if m:
            nums.append(int(m.group(1)))
    assert nums, "Ожидались заголовки ## Question N"
    return len(nums)


def split_markdown_answers(text: str) -> list[str]:
    """Блоки между ## Answer N и следующим ## Answer или концом файла."""
    matches = list(_MD_ANSWER.finditer(text))
    if not matches:
        raise AssertionError("Нет заголовков ## Answer N")
    preamble = text[: matches[0].start()].strip()
    if preamble:
        raise AssertionError(f"Перед первым ## Answer не должно быть текста: {preamble[:120]!r}…")
    items: list[str] = []
    for i, m in enumerate(matches):
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        items.append(body)
    return items


def split_numbered_answers(text: str) -> list[str]:
    """Пункт — строка «N. » / «N) »; строки без такого префикса (напр. 2.1) — продолжение."""
    items: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        head = line.strip()
        if _is_numbered_start(head):
            if current:
                items.append("\n".join(current).strip())
            current = [line]
        else:
            if not current and line.strip():
                raise AssertionError(
                    f"Сначала должен быть пронумерованный ответ (строка «N. …»/«N) …»), не: {line[:100]!r}"
                )
            if current:
                current.append(line)
    if current:
        items.append("\n".join(current).strip())
    return [b for b in items if b]


def assert_numbered_items(answers: list[str]) -> None:
    for s in answers:
        if not _is_numbered_start(s.strip()):
            raise AssertionError(
                "Каждый ответ — пронумерованный пункт (начало «N.»/«N)» с непустым текстом), "
                f"фрагмент: {s[:180]!r}…"
            )


def assert_md_answer_blocks_non_empty(blocks: list[str]) -> None:
    for i, b in enumerate(blocks, start=1):
        if not b.strip():
            raise AssertionError(f"Пустой блок ответа после ## Answer {i}")


def validate_pair(questions_path: Path, answers_path: Path) -> None:
    ensure_env_file()
    assert questions_path.is_file(), f"Нет файла вопросов: {questions_path}"
    assert answers_path.is_file(), f"Нет файла ответов: {answers_path}"

    qtext = questions_path.read_text(encoding="utf-8-sig")
    atext = answers_path.read_text(encoding="utf-8-sig")
    assert atext.strip(), f"{answers_path} пуст"

    md_q = _has_md_questions(qtext)
    md_a = _has_md_answers(atext)

    if md_q != md_a:
        raise AssertionError(
            "Формат не сочетается: если вопросы с заголовками ## Question N, "
            "ответы должны быть с ## Answer N (и наоборот); "
            "либо оба файла в простом виде (пронумерованные строки «1. …»)."
        )

    if md_q and md_a:
        nq = count_questions_markdown(qtext)
        blocks = split_markdown_answers(atext)
        assert_md_answer_blocks_non_empty(blocks)
        assert len(blocks) == nq, (
            f"Число блоков ## Answer ({len(blocks)}) не совпадает с числом ## Question ({nq})"
        )
        return

    nq = count_questions_line_based(questions_path)
    items = split_numbered_answers(atext)
    assert_numbered_items(items)
    assert len(items) == nq, (
        f"Число ответов ({len(items)}) не совпадает с непустыми строками вопросов ({nq})"
    )


def main() -> None:
    assert DATA_DIR.is_dir(), "Нет папки data/"
    assert OUTPUT_DIR.is_dir(), "Нет папки output/"
    validate_pair(DEFAULT_QUESTIONS_PATH, DEFAULT_ANSWERS_PATH)
    print("OK: проверки пройдены ( .env, пронумерованные ответы, количество )")


if __name__ == "__main__":
    main()
