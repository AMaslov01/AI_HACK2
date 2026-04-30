"""Все промпты в одном месте."""

from __future__ import annotations

PLANNER_PROMPT = """\
Ты планируешь retrieval для агента-QA по научной статье.

Карта статьи:
{readme}

Вопрос: {question}

Реши:
1. Какие секции (по номеру: "2", "2.1") наиболее вероятно содержат ответ? До 3-х.
   Только цифры (и опционально s для synthetic), без префиксов "§"/"Section".
2. Является ли вопрос figure-related? (Слова "figure", "diagram", "plot", "shows",
   или совпадение с описаниями рисунков из карты.)
3. Если figure-related, какой конкретно label рисунка из карты?
4. Нужно ли смотреть на само изображение или хватит сохранённого описания?
   По умолчанию — хватит.

Верни ТОЛЬКО JSON без обёрток:
{{"section_scope": ["2", "2.1"], "is_figure_question": false,
  "target_label": null, "needs_vision": false,
  "reasoning": "одно предложение"}}
"""


COMPOSE_PROMPT = """\
Answer in English only.
Answer the question ONLY from the provided sources.
If the answer is not present in the sources, write exactly "no answer".

Article map (navigation only, do not cite):
{readme}

Sources:
{sources}

Question: {question}

Rules:
- Be concise: 1-4 sentences for factual answers, up to 6 for explanations.
- If a figure is relevant, name it (for example, "Figure 3" or its label).
- When using a specific source, mention the section: "(§2.1)".
- Keep numbers and equations verbatim.
- If sources conflict, prefer the more specific section.
"""


COMPOSE_GLOBAL_HINT = (
    "\nThis is a deferred question: the sources cover the whole article without a section filter. "
    "If the answer is still absent, write exactly \"no answer\"."
)


VISION_AUGMENT_PROMPT = """\
The user asks: {question}
Figure caption: "{caption}".
Answer in English, specifically addressing the question. If the answer is not visible in the figure, write "not visible".
"""
