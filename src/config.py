"""Центральные константы. Не импортирует тяжёлые зависимости."""

from __future__ import annotations

from pathlib import Path

# --- Paths ---
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
CACHE_ROOT = Path(".cache")

# --- Кэш ---
SCHEMA_VERSION = 1

# --- Модель ---
MODEL_NAME = "GigaChat-2-Max"
TEMPERATURE_CHAT = 0.2
TEMPERATURE_PLANNER = 0.0
GIGACHAT_TIMEOUT_S = 120

# --- Чанкинг ---
# GigaChat-эмбеддинги лимитят 514 токенов/документ (~1 токен ≈ 2–3 символа для RU).
# Берём с запасом: 1200 символов → ~300–500 токенов.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# --- Эмбеддинги: pre-truncation guard ---
EMBED_MAX_CHARS = 1500          # мягкий лимит — обрезаем перед отправкой
EMBED_MAX_CHARS_FALLBACK = 800  # после 413 — обрезаем агрессивнее и retry

# --- Retrieval ---
RETRIEVE_TOP_K = 6                 # scope-pass итог
RETRIEVE_FETCH_K = 24              # перед re-rank
RETRIEVE_GLOBAL_BONUS = 2          # +N глобальных к scope
MIN_COSINE = 0.45                  # cosine-порог: ниже — шум
MIN_SCOPED_DOCS = 3                # < N → fallback на глобальный pass

# --- QA cache ---
QA_CACHE_SIM_THRESHOLD = 0.92
QA_CACHE_FILE = "qa_cache.jsonl"

# --- Тайм-бюджет (сек) ---
GLOBAL_BUDGET_S = 870
PLANNER_BUDGET_S = 15
RETRIEVE_BUDGET_S = 5
VISUAL_BUDGET_S = 30
COMPOSE_BUDGET_S = 45
FORMATTER_BUDGET_S = 1
MAX_PER_QUESTION_S = 90
MIN_PER_QUESTION_S = 8             # меньше — сразу стаб

VISION_TOTAL_BUDGET_S = 240        # после — caption-only фолбэк
PER_VISION_CALL_TIMEOUT_S = 60

# --- Concurrency ---
VISION_CONCURRENCY = 2
CHAT_CONCURRENCY = 4
EMBED_BATCH = 32

# --- README / структура ---
README_MAX_CHARS = 16000
SECTION_SUMMARY_MAX_SECTIONS = 60      # > → пропускаем шаг
SYNTH_SUBSECTION_MIN_CHARS = 8000      # секция длиннее → возможно делим
SYNTH_SUBSECTION_MAX_NEW = 6           # лимит секций для синтетики

# --- Backoff ---
LLM_RETRY_ATTEMPTS = 3
LLM_RETRY_BASE_DELAY_S = 2.0
