"""SHA-256 от сырых .tex байтов → 16-символьный ключ кэша."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from src.config import SCHEMA_VERSION


def fingerprint(tex_files: list[Path], tex_root: Path) -> str:
    h = hashlib.sha256()
    h.update(f"v{SCHEMA_VERSION}\0".encode())
    for p in sorted(tex_files):
        try:
            rel = p.relative_to(tex_root).as_posix()
        except ValueError:
            rel = p.as_posix()
        h.update(rel.encode())
        h.update(b"\0")
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"<unreadable>")
        h.update(b"\0")
    return h.hexdigest()[:16]


def write_fingerprint_meta(cache_dir: Path, fp: str, tex_files: list[Path], tex_root: Path) -> None:
    payload = {
        "fingerprint": fp,
        "schema_version": SCHEMA_VERSION,
        "ts": time.time(),
        "files": [str(p.relative_to(tex_root)) for p in tex_files
                  if str(p).startswith(str(tex_root))],
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "fingerprint.txt").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_fingerprint_meta(cache_dir: Path) -> dict | None:
    p = cache_dir / "fingerprint.txt"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
