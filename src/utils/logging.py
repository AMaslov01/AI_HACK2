"""Тонкая обёртка над loguru. Если loguru не установлен — fallback на stdlib logging."""

from __future__ import annotations

try:
    from loguru import logger as _log
    log = _log
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("agent")
