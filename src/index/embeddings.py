"""GigaChat embeddings с pre-truncation, 413-handling и retry/backoff.

Лимит GigaChat-эмбеддингов — 514 токенов на документ.
Стратегия:
- pre-truncate каждый текст до EMBED_MAX_CHARS до отправки (защита от длинных
  equation-чанков и section_summary);
- при 413 — обрезаем до EMBED_MAX_CHARS_FALLBACK и пробуем ещё раз;
- если упорно — для этого батча возвращаем zero-векторы (одиночные плохие
  чанки не должны валить весь индекс)."""

from __future__ import annotations

import os
import time

from src.config import (
    EMBED_MAX_CHARS,
    EMBED_MAX_CHARS_FALLBACK,
    LLM_RETRY_ATTEMPTS,
    LLM_RETRY_BASE_DELAY_S,
)
from src.utils.logging import log


def build_embeddings(verify_ssl_certs: bool = False):
    from langchain_gigachat.embeddings import GigaChatEmbeddings

    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    scope = os.getenv("GIGACHAT_SCOPE")
    if not credentials or not scope:
        raise RuntimeError("GIGACHAT_CREDENTIALS / GIGACHAT_SCOPE не установлены")
    emb = GigaChatEmbeddings(
        credentials=credentials, scope=scope,
        verify_ssl_certs=verify_ssl_certs,
    )
    return EmbeddingsWithRetry(emb)


def _truncate(text: str, n: int) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text[:n]


def _is_too_large(exc: Exception) -> bool:
    """413 от GigaChat. Распознаём и по типу, и по тексту (на случай разных версий пакета)."""
    name = type(exc).__name__.lower()
    if "requestentitytoolarge" in name or "toolarge" in name:
        return True
    s = str(exc).lower()
    return "413" in s or "tokens limit exceeded" in s or "too large" in s


class EmbeddingsWithRetry:
    def __init__(self, inner):
        self._inner = inner
        self._dim: int | None = None  # узнаем из первого успешного embed_query

    # --- public API, ожидаемое langchain / chromadb ---

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Первый pass: мягкое усечение
        soft = [_truncate(t, EMBED_MAX_CHARS) for t in texts]
        try:
            result = self._inner.embed_documents(soft)
            self._capture_dim(result)
            return result
        except Exception as e:
            if not _is_too_large(e):
                # стандартный backoff (не 413)
                return self._retry_default(self._inner.embed_documents, soft)
            log.warning(f"embeddings 413 on batch — aggressive truncate to {EMBED_MAX_CHARS_FALLBACK}")

        # Второй pass: жёсткое усечение всего батча
        hard = [_truncate(t, EMBED_MAX_CHARS_FALLBACK) for t in texts]
        try:
            result = self._inner.embed_documents(hard)
            self._capture_dim(result)
            return result
        except Exception as e:
            if not _is_too_large(e):
                return self._retry_default(self._inner.embed_documents, hard)
            log.warning(f"embeddings 413 even after hard truncate — per-doc fallback")

        # Третий pass: документ за документом, чтобы плохие чанки не валили хорошие
        return self._per_doc_fallback(hard)

    def embed_query(self, text: str) -> list[float]:
        truncated = _truncate(text, EMBED_MAX_CHARS)
        try:
            v = self._inner.embed_query(truncated)
            if isinstance(v, list):
                self._dim = len(v)
            return v
        except Exception as e:
            if not _is_too_large(e):
                return self._retry_default(self._inner.embed_query, truncated)
            shorter = _truncate(text, EMBED_MAX_CHARS_FALLBACK)
            try:
                v = self._inner.embed_query(shorter)
                if isinstance(v, list):
                    self._dim = len(v)
                return v
            except Exception as e2:
                log.warning(f"embed_query failed even after hard truncate: {e2}")
                return self._zero_vector()

    # --- internals ---

    def _per_doc_fallback(self, hard: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i, t in enumerate(hard):
            try:
                v = self._inner.embed_documents([t])[0]
                self._dim = len(v) if isinstance(v, list) else self._dim
                out.append(v)
            except Exception as e:
                # последний шанс — ещё раз обрезаем и one-shot embed
                shorter = _truncate(t, EMBED_MAX_CHARS_FALLBACK // 2)
                if shorter and shorter != t:
                    try:
                        v = self._inner.embed_documents([shorter])[0]
                        out.append(v)
                        continue
                    except Exception:
                        pass
                log.warning(f"embeddings: doc {i} failed permanently ({e}); using zero-vector")
                out.append(self._zero_vector())
        return out

    def _retry_default(self, fn, *args, **kwargs):
        last: Exception | None = None
        for attempt in range(LLM_RETRY_ATTEMPTS):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last = e
                if _is_too_large(e):
                    raise
                delay = LLM_RETRY_BASE_DELAY_S * (2 ** attempt)
                log.warning(
                    f"embedding retry {attempt + 1}/{LLM_RETRY_ATTEMPTS} after {delay}s: {e}"
                )
                time.sleep(delay)
        if last is not None:
            raise last

    def _capture_dim(self, batch) -> None:
        if batch and isinstance(batch[0], list):
            self._dim = len(batch[0])

    def _zero_vector(self) -> list[float]:
        # Если размер ещё неизвестен — берём 1024 (типичный размер GigaChat-эмб).
        return [0.0] * (self._dim or 1024)
