"""Главный runner: per-question loop с QA-кэшем, deferred-очередью, deadline-guard."""

from __future__ import annotations

import os
import time
from pathlib import Path

from src.agent.graph import build_graph
from src.agent.state import AgentState
from src.cache.qa_cache import QACache, normalize_question
from src.config import (
    CACHE_ROOT,
    COMPOSE_BUDGET_S,
    GIGACHAT_TIMEOUT_S,
    GLOBAL_BUDGET_S,
    MIN_PER_QUESTION_S,
    MODEL_NAME,
    TEMPERATURE_CHAT,
    TEMPERATURE_PLANNER,
)
from src.index.embeddings import build_embeddings
from src.index.retriever import Plan, Retriever
from src.index.vectorstore import get_or_create_collection, has_documents, upsert_chunks
from src.ingest.pipeline import IngestArtifacts, run_ingest
from src.io.answers import format_block, stub_block
from src.io.questions import QuestionsDoc
from src.utils.logging import log
from src.utils.timing import run_with_timeout

FALLBACK_UNCERTAINTY_NOTE = (
    "Note: This answer may be inaccurate because no strong matches were found in the article."
)
FALLBACK_CONTEXT_MAX_CHARS = 70000


def _build_chat_client(temperature: float, *, vision: bool = False):
    """Vision-клиент должен иметь auto_upload_attachments=True — иначе GigaChat
    не принимает inline base64-картинки (он требует загрузить их через /files
    и передать attachment_id)."""
    from langchain_gigachat.chat_models import GigaChat
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    scope = os.getenv("GIGACHAT_SCOPE")
    if not credentials or not scope:
        raise RuntimeError("GIGACHAT_CREDENTIALS / GIGACHAT_SCOPE не установлены")
    kwargs = dict(
        credentials=credentials, scope=scope, model=MODEL_NAME,
        temperature=temperature, timeout=GIGACHAT_TIMEOUT_S, verify_ssl_certs=False,
    )
    if vision:
        kwargs["auto_upload_attachments"] = True
    try:
        return GigaChat(**kwargs)
    except TypeError:
        # старая версия без auto_upload_attachments — fallback
        kwargs.pop("auto_upload_attachments", None)
        return GigaChat(**kwargs)


def _known_sections(chunks) -> tuple[set[str], set[str]]:
    known_numbers: set[str] = set()
    known_top: set[str] = set()
    for c in chunks:
        sn = c.metadata.get("section_number")
        st = c.metadata.get("section_top")
        if sn:
            known_numbers.add(sn)
        if st:
            known_top.add(st)
    return known_numbers, known_top


def _try_qa_cache(qa: QACache, question: str, embeddings) -> tuple[str, list[float]] | None:
    """Возвращает (cached_answer, embedding) если хит, иначе None.
    Эмбеддинг возвращается всегда — переиспользуется retriever-ом при промахе."""
    try:
        emb = embeddings.embed_query(question)
    except Exception as e:
        log.warning(f"embed_query failed: {e}")
        emb = []
    hit = qa.lookup(question, emb)
    if hit is not None:
        return hit.answer, emb
    return None


def _no_answer(idx: int, fmt) -> str:
    return stub_block(idx, fmt)


def _is_no_answer(text: str) -> bool:
    normalized = (text or "").strip().lower().strip(" .\"'`")
    return not normalized or normalized == "no answer"


def _has_cyrillic(text: str) -> bool:
    return any("\u0400" <= ch <= "\u04ff" for ch in text or "")


def _invoke_chat_text(gigachat_chat, prompt: str, timeout_s: float) -> str:
    resp = run_with_timeout(gigachat_chat.invoke, timeout_s, prompt)
    return (resp.content if hasattr(resp, "content") else str(resp)).strip()


def _append_uncertainty_note(text: str) -> str:
    text = (text or "").strip()
    if not text:
        text = "No answer."
    if FALLBACK_UNCERTAINTY_NOTE in text:
        return text
    return f"{text}\n\n{FALLBACK_UNCERTAINTY_NOTE}"


def _build_fallback_article_context(chunks) -> str:
    """Pack already-loaded article chunks without running another retrieval/search step."""
    parts: list[str] = []
    total = 0
    sorted_chunks = sorted(
        chunks,
        key=lambda c: (
            int(c.metadata.get("char_start") or 0),
            c.chunk_id,
        ),
    )
    for chunk in sorted_chunks:
        kind = chunk.metadata.get("kind") or "body"
        section = chunk.metadata.get("section_path") or chunk.metadata.get("section_number") or "?"
        text = (chunk.text or "").strip()
        if not text:
            continue
        block = f"[{kind}; {section}]\n{text}"
        if total + len(block) + 2 > FALLBACK_CONTEXT_MAX_CHARS:
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n".join(parts)


def _fallback_no_answer_from_context(
    gigachat_chat,
    *,
    question: str,
    article_readme: str,
    article_context: str,
    previous_context: str,
    deadline_ts: float,
) -> str:
    """Best-effort answer without another retrieval/search pass."""
    remaining = deadline_ts - time.time()
    if remaining < 10:
        return "no answer"
    timeout_s = min(COMPOSE_BUDGET_S, max(5.0, remaining - 2.0))
    prompt = f"""\
Answer in English only.

The retrieval-based answer for this question was "no answer".
Do not run a new search. Use only your own model knowledge, the previous Q/A,
and the already loaded article context below.
Give the best short answer you can. If the question depends on a previous
question, use the previous Q/A to resolve that reference.
Do not answer "No answer" unless there is absolutely no possible answer.

Loaded article context/map:
{article_readme[:12000]}

Previous Q/A:
{previous_context or "(none)"}

Loaded article chunks:
{article_context}

Question: {question}
"""
    try:
        return _invoke_chat_text(gigachat_chat, prompt, timeout_s) or "no answer"
    except Exception as e:
        log.warning(f"fallback no-answer failed: {e}")
        return "no answer"


def _ensure_english_answer(
    gigachat_chat,
    *,
    question: str,
    answer: str,
    deadline_ts: float,
) -> str:
    if _is_no_answer(answer) or not _has_cyrillic(answer):
        return answer
    remaining = deadline_ts - time.time()
    if remaining < 8:
        return answer
    prompt = f"""\
Translate the answer to English only.
Preserve all formulas, numbers, section references, citations, and formatting.
Do not add new facts and do not answer the question again.
Return only the translated answer, without any preamble or explanation.

Question: {question}

Answer:
{answer}
"""
    try:
        translated = _invoke_chat_text(gigachat_chat, prompt, min(COMPOSE_BUDGET_S, remaining - 2.0))
        return translated or answer
    except Exception as e:
        log.warning(f"english postprocess failed: {e}")
        return answer


def _postprocess_answers(
    answers: list[str],
    *,
    qdoc: QuestionsDoc,
    gigachat_chat,
    article_readme: str,
    article_context: str,
    deadline_ts: float,
) -> list[str]:
    processed: list[str] = []
    for answer, q in zip(answers, qdoc.items):
        text = (answer or "").strip()
        fallback_used = False
        if _is_no_answer(text):
            fallback_used = True
            text = _fallback_no_answer_from_context(
                gigachat_chat,
                question=q.text,
                article_readme=article_readme,
                article_context=article_context,
                previous_context="\n\n".join(
                    f"Question {prev_q.idx}: {prev_q.text}\nAnswer {prev_q.idx}: {prev_answer}"
                    for prev_q, prev_answer in zip(qdoc.items[:len(processed)], processed)
                ),
                deadline_ts=deadline_ts,
            )
        text = _ensure_english_answer(
            gigachat_chat,
            question=q.text,
            answer=text,
            deadline_ts=deadline_ts,
        )
        if fallback_used:
            text = _append_uncertainty_note(text)
        processed.append(text or "no answer")
    return processed


def run_all(qdoc: QuestionsDoc, deadline_ts: float | None = None) -> list[str]:
    """Полный прогон: ингест + двухпроходный цикл вопросов.
    Возвращает list[str] длиной len(qdoc) с финальными блоками ответов."""
    start_ts = time.time()
    if deadline_ts is None:
        deadline_ts = start_ts + GLOBAL_BUDGET_S

    # GigaChat clients
    gc_chat = _build_chat_client(TEMPERATURE_CHAT)
    gc_planner = _build_chat_client(TEMPERATURE_PLANNER)
    # vision-клиенту нужен auto_upload_attachments=True — иначе GigaChat не примет
    # inline base64-картинки, требует загрузить через /files и передать attachment_id.
    gc_vision = _build_chat_client(TEMPERATURE_CHAT, vision=True)
    embeddings = build_embeddings()

    # Ingest
    log.info("=== Phase 1: ingest ===")
    artifacts: IngestArtifacts = run_ingest(
        data_dir=Path("data"), gigachat_chat=gc_chat, gigachat_vision=gc_vision,
        cache_root=CACHE_ROOT,
    )
    log.info(f"ingest done in {time.time() - start_ts:.1f}s "
             f"(cache_hit={artifacts.cache_hit})")

    # Index
    log.info("=== Phase 1.5: build index ===")
    coll = get_or_create_collection(artifacts.cache_dir / "chroma", embeddings)
    if not has_documents(coll):
        upsert_chunks(coll, artifacts.chunks)

    known_numbers, known_top = _known_sections(artifacts.chunks)
    retriever = Retriever(coll, known_numbers, known_top)

    qa = QACache(artifacts.cache_dir)

    # Pre-fill стабами — гарантия валидного output даже при early-exit
    final_answers: list[str] = ["no answer" for _ in qdoc.items]

    embed_cache: dict[str, list[float]] = {}

    # ---- Pass 1: scoped ----
    log.info("=== Phase 2 Pass 1 (scoped) ===")
    graph_p1 = build_graph(
        gigachat_chat=gc_chat, gigachat_planner=gc_planner, gigachat_vision=gc_vision,
        retriever=retriever, embeddings=embeddings,
        figure_records=artifacts.figures, deferred=False, embed_cache=embed_cache,
    )

    deferred: list[tuple[int, str, int]] = []   # (slot_index, question_text, original_idx)

    for slot, q in enumerate(qdoc.items):
        remaining = deadline_ts - time.time()
        if remaining < MIN_PER_QUESTION_S:
            log.warning(f"q{q.idx}: deadline reached, leaving stub")
            continue
        # QA-cache lookup
        cached = _try_qa_cache(qa, q.text, embeddings)
        emb_for_q: list[float] = []
        if cached is not None:
            ans, emb_for_q = cached
            final_answers[slot] = ans or "no answer"
            embed_cache[q.text] = emb_for_q  # переиспользуем
            log.info(f"q{q.idx}: QA-cache hit")
            continue
        if cached is None and q.text in embed_cache:
            emb_for_q = embed_cache[q.text]

        try:
            state: AgentState = {
                "question": q.text, "question_index": q.idx,
                "format_kind": qdoc.fmt, "article_readme": artifacts.readme,
                "deadline_ts": deadline_ts, "timings": {},
            }
            result = graph_p1.invoke(state, config={"recursion_limit": 25})
            plan: Plan | None = result.get("plan")
            if plan is not None and plan.unscoped:
                deferred.append((slot, q.text, q.idx))
                # стаб уже в final_answers[slot]
                log.info(f"q{q.idx}: deferred (unscoped)")
                continue
            txt = (result.get("draft_answer") or "").strip()
            final_answers[slot] = txt or "no answer"
            # save в QA-кэш (эмбеддинг и doc_ids)
            doc_ids = [d.chunk_id for d in (result.get("docs") or [])]
            try:
                qa.save(q.text, txt or "no answer",
                        embed_cache.get(q.text) or [], doc_ids,
                        plan.__dict__ if plan else {})
            except Exception as e:
                log.warning(f"qa.save failed: {e}")
        except Exception as e:
            log.exception(f"q{q.idx}: pass1 error: {e}")

    # ---- Pass 2: deferred ----
    if deferred and (deadline_ts - time.time()) > 30:
        log.info(f"=== Phase 2 Pass 2 (deferred): {len(deferred)} questions ===")
        graph_p2 = build_graph(
            gigachat_chat=gc_chat, gigachat_planner=gc_planner, gigachat_vision=gc_vision,
            retriever=retriever, embeddings=embeddings,
            figure_records=artifacts.figures, deferred=True, embed_cache=embed_cache,
        )
        for slot, q_text, q_idx in deferred:
            remaining = deadline_ts - time.time()
            if remaining < MIN_PER_QUESTION_S:
                log.warning(f"q{q_idx}: pass2 skipped, deadline")
                break
            try:
                state: AgentState = {
                    "question": q_text, "question_index": q_idx,
                    "format_kind": qdoc.fmt, "article_readme": artifacts.readme,
                    "deadline_ts": deadline_ts, "timings": {},
                }
                result = graph_p2.invoke(state, config={"recursion_limit": 25})
                txt = (result.get("draft_answer") or "").strip()
                final_answers[slot] = txt or "no answer"
                doc_ids = [d.chunk_id for d in (result.get("docs") or [])]
                try:
                    qa.save(q_text, txt or "no answer",
                            embed_cache.get(q_text) or [], doc_ids,
                            (result.get("plan").__dict__ if result.get("plan") else {}))
                except Exception:
                    pass
            except Exception as e:
                log.exception(f"q{q_idx}: pass2 error: {e}")

    # Метрики
    metrics = qa.metrics()
    log.info(f"qa-cache: {metrics}")
    log.info("=== Phase 3: final answer postprocess ===")
    final_answers = _postprocess_answers(
        final_answers,
        qdoc=qdoc,
        gigachat_chat=gc_chat,
        article_readme=artifacts.readme,
        article_context=_build_fallback_article_context(artifacts.chunks),
        deadline_ts=deadline_ts,
    )
    log.info(f"total wall time: {time.time() - start_ts:.1f}s")
    return [format_block(q.idx, answer, qdoc.fmt) for q, answer in zip(qdoc.items, final_answers)]
