"""ChromaDB persistent collection. Кладём чанки батчами."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import EMBED_BATCH
from src.ingest.chunker import Chunk
from src.utils.logging import log


COLLECTION_NAME = "paper"


def _scalar_metadata(meta: dict) -> dict:
    """Chroma metadata требует scalar — приводим bool/None/list к допустимым типам."""
    out: dict = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, list):
            out[k] = "|" + "|".join(str(x) for x in v) + "|" if v else ""
        else:
            out[k] = str(v)
    return out


def get_or_create_collection(persist_dir: Path, embeddings) -> Any:
    """Возвращает Chroma collection (с client). Если уже есть — переиспользует."""
    import chromadb
    from chromadb.config import Settings

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )

    class _EmbedFn:
        def __init__(self, e):
            self.e = e
        def __call__(self, input):  # noqa: A002
            return self.e.embed_documents(list(input))
        def name(self):  # chromadb >= 0.5.x
            return "gigachat"

    coll = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=_EmbedFn(embeddings),
    )
    return coll


def upsert_chunks(coll, chunks: list[Chunk]) -> None:
    if not chunks:
        return
    n = len(chunks)
    log.info(f"chroma: upsert {n} chunks")
    for i in range(0, n, EMBED_BATCH):
        batch = chunks[i:i + EMBED_BATCH]
        ids = [c.chunk_id for c in batch]
        docs = [c.text for c in batch]
        metas = [_scalar_metadata(c.metadata) for c in batch]
        coll.upsert(ids=ids, documents=docs, metadatas=metas)


def has_documents(coll) -> bool:
    try:
        return coll.count() > 0
    except Exception:
        return False
