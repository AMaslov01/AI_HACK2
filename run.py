"""Точка входа: `python run.py` — генерация `output/answers.txt`.

Принципы:
- Всегда выдаёт `output/answers.txt` ровно с N блоками (по числу вопросов),
  даже если агент упал, deps не установлены, или истёк тайм-бюджет.
- Формат блоков (markdown `## Answer N` или numbered `1. ...`) подбирается
  по формату `data/questions.txt` (см. src/utils/check_submission.py).
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

from src.io.answers import stub_block, write_answers
from src.io.questions import QuestionsDoc, load_questions

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
QUESTIONS_PATH = DATA_DIR / "questions.txt"
ANSWERS_PATH = OUTPUT_DIR / "answers.txt"

GLOBAL_BUDGET_S = 870


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception as e:
        print(f"[run] python-dotenv недоступен: {e}", flush=True)


def _emit_stubs(qdoc: QuestionsDoc) -> None:
    blocks = [stub_block(q.idx, qdoc.fmt) for q in qdoc.items]
    write_answers(ANSWERS_PATH, blocks, qdoc.fmt)


def _try_run_agent(qdoc: QuestionsDoc, deadline_ts: float) -> bool:
    try:
        from src.agent.runner import run_all
    except Exception as e:
        print(f"[run] agent unavailable (import error): {e}", flush=True)
        return False
    try:
        blocks = run_all(qdoc=qdoc, deadline_ts=deadline_ts)
    except Exception:
        traceback.print_exc()
        return False
    if not blocks or len(blocks) != len(qdoc.items):
        print("[run] agent returned wrong number of blocks; keeping stubs", flush=True)
        return False
    write_answers(ANSWERS_PATH, blocks, qdoc.fmt)
    return True


def main() -> None:
    _load_dotenv_if_available()
    start_ts = time.time()
    deadline_ts = start_ts + GLOBAL_BUDGET_S

    if not QUESTIONS_PATH.is_file():
        raise FileNotFoundError(f"Не найден {QUESTIONS_PATH}")
    qdoc = load_questions(QUESTIONS_PATH)

    # Гарантия: на диске уже валидный файл стабов до любых дальнейших действий.
    _emit_stubs(qdoc)

    ok = _try_run_agent(qdoc, deadline_ts=deadline_ts)
    if not ok:
        print("[run] agent unavailable or failed; submitted stubs", flush=True)


if __name__ == "__main__":
    main()
