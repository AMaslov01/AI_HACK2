from __future__ import annotations

from pathlib import Path


def _count_questions(questions_path: Path) -> int:
    if not questions_path.exists():
        raise FileNotFoundError(questions_path)
    lines = [ln for ln in questions_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError("data/questions.txt: нет непустых строк с вопросами")
    return len(lines)


def make_baseline_submission() -> None:
    """Пишет заглушку `output/answers.txt` (по одному пронумерованному ответу на вопрос)."""
    n = _count_questions(Path("data/questions.txt"))
    Path("output").mkdir(parents=True, exist_ok=True)
    out = "\n".join(
        f"{i + 1}. [замените на реальный ответ: реализуйте агента на GigaChat-2-Max]"
        for i in range(n)
    )
    Path("output/answers.txt").write_text(out + "\n", encoding="utf-8")
