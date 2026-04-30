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
    final_text: list[str] = [stub_block(q.idx, qdoc.fmt) for q in qdoc.items]

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
            final_text[slot] = format_block(q.idx, ans, qdoc.fmt)
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
                # стаб уже в final_text[slot]
                log.info(f"q{q.idx}: deferred (unscoped)")
                continue
            txt = (result.get("draft_answer") or "").strip()
            final_text[slot] = format_block(q.idx, txt or "no answer", qdoc.fmt)
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
                final_text[slot] = format_block(q_idx, txt or "no answer", qdoc.fmt)
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
    log.info(f"total wall time: {time.time() - start_ts:.1f}s")
    return final_text
