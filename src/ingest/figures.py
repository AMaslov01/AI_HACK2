"""Резолв путей рисунков, рендер PDF→PNG, vision-описания."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from src.config import (
    PER_VISION_CALL_TIMEOUT_S,
    VISION_CONCURRENCY,
    VISION_TOTAL_BUDGET_S,
)
from src.ingest.discovery import find_graphicspaths
from src.ingest.structure import FigureRaw, StructureDoc
from src.utils.logging import log
from src.utils.timing import run_with_timeout

EXTENSIONS = ("", ".png", ".jpg", ".jpeg", ".pdf", ".PNG", ".JPG", ".PDF")


@dataclass
class FigureRecord:
    figure_id: str
    tex_path: str
    resolved_path: str | None
    rendered_png: str | None
    caption: str
    section_path: str
    context_before: str = ""
    context_after: str = ""
    description: dict = field(default_factory=dict)
    description_text: str = ""
    vision_ok: bool = False


def _resolve(tex_path: str, tex_root: Path, current_dir: Path,
             graphicspaths: list[Path]) -> Path | None:
    candidates: list[Path] = []
    base_paths: list[Path] = [tex_root, current_dir, *graphicspaths]
    for base in base_paths:
        for ext in EXTENSIONS:
            candidates.append((base / (tex_path + ext)).resolve())
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    # последний шанс — rglob по stem
    stem = Path(tex_path).stem
    matches = list(tex_root.rglob(f"{stem}.*"))
    image_exts = {".png", ".jpg", ".jpeg", ".pdf"}
    for m in matches:
        if m.suffix.lower() in image_exts:
            return m
    return None


def _pdf_to_png(pdf_path: Path, out_dir: Path, dpi: int = 200) -> Path | None:
    try:
        import pymupdf  # type: ignore
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore
        except ImportError:
            log.warning("pymupdf не установлен — PDF-рисунки не будут конвертированы")
            return None
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        log.warning(f"pymupdf.open({pdf_path}) failed: {e}")
        return None
    try:
        if doc.page_count != 1:
            log.warning(f"{pdf_path}: page_count={doc.page_count}, ожидается 1; пропускаем")
            return None
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=dpi)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (pdf_path.stem + ".png")
        pix.save(str(out_path))
        return out_path
    finally:
        doc.close()


def _ensure_png(resolved: Path, out_dir: Path) -> Path | None:
    if resolved.suffix.lower() == ".pdf":
        return _pdf_to_png(resolved, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / resolved.name
    if not target.exists():
        try:
            shutil.copy2(resolved, target)
        except OSError as e:
            log.warning(f"copy {resolved} → {target} failed: {e}")
            return None
    return target


def _surrounding_context(text: str, char_pos: int, span: int = 800) -> tuple[str, str]:
    before = text[max(0, char_pos - span): char_pos]
    after = text[char_pos: char_pos + span]
    return before.strip(), after.strip()


def _png_to_base64(p: Path) -> str:
    import base64
    return base64.b64encode(p.read_bytes()).decode("ascii")


def _build_vision_messages(rec: FigureRecord) -> list:
    """Готовит messages для langchain GigaChat с image content."""
    from langchain_core.messages import HumanMessage
    img_b64 = _png_to_base64(Path(rec.rendered_png))  # type: ignore[arg-type]
    text_part = (
        "Опиши этот рисунок для downstream-агента, который не видит изображение.\n"
        f"Подпись: {rec.caption or '(нет)'}\n"
        f"Окружающий текст ДО:\n{rec.context_before[:600]}\n\n"
        f"Окружающий текст ПОСЛЕ:\n{rec.context_after[:600]}\n\n"
        "Верни JSON с полями:\n"
        "- subject: одно предложение, что изображено\n"
        "- axes_or_layout: оси/единицы или layout панелей\n"
        "- key_visual_elements: список\n"
        "- numbers_visible: список\n"
        "- takeaway: одно предложение, главный вывод\n"
        "Только JSON без обёрток."
    )
    return [HumanMessage(content=[
        {"type": "text", "text": text_part},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
    ])]


def _parse_vision_json(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    # ищем первый { и последний }
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1:
        return {}
    try:
        return json.loads(s[a:b + 1])
    except json.JSONDecodeError:
        return {}


def _flatten_description(d: dict, caption: str) -> str:
    parts: list[str] = []
    if caption:
        parts.append(f"Caption: {caption}")
    for key in ("subject", "axes_or_layout", "takeaway"):
        v = d.get(key)
        if v:
            parts.append(f"{key}: {v}")
    for key in ("key_visual_elements", "numbers_visible"):
        v = d.get(key)
        if isinstance(v, list) and v:
            parts.append(f"{key}: {', '.join(str(x) for x in v)}")
    return " | ".join(parts)


def describe_figures(figures_raw: list[FigureRaw], structure: StructureDoc,
                     tex_root: Path, cache_dir: Path, gigachat_client,
                     time_budget_s: float = VISION_TOTAL_BUDGET_S) -> dict[str, FigureRecord]:
    """Резолв путей, рендер PNG, vision-описание для каждого рисунка.
    gigachat_client — экземпляр langchain GigaChat (с поддержкой vision)."""
    import time
    figures_dir = cache_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    graphicspaths = find_graphicspaths(structure.expanded_text, tex_root)

    # Резолв путей и рендер
    records: dict[str, FigureRecord] = {}
    for f in figures_raw:
        sec = structure.section_by_number(f.section)
        section_path = f"{sec.number} {sec.title}" if sec else f.section
        before, after = _surrounding_context(structure.expanded_text, f.char_pos)
        resolved = _resolve(f.tex_path, tex_root, tex_root, graphicspaths)
        png_path: Path | None = None
        if resolved is not None:
            png_path = _ensure_png(resolved, figures_dir)
        rec = FigureRecord(
            figure_id=f.label, tex_path=f.tex_path,
            resolved_path=str(resolved) if resolved else None,
            rendered_png=str(png_path) if png_path else None,
            caption=f.caption, section_path=section_path,
            context_before=before, context_after=after,
        )
        if png_path is None:
            rec.description_text = (
                f"Figure {f.label}: {f.caption} (файл рисунка не найден или не сконвертирован). "
                f"Контекст ДО: {before[:300]} ... Контекст ПОСЛЕ: {after[:300]}"
            ).strip()
        records[f.label] = rec

    if gigachat_client is None:
        # vision недоступен — фолбэк на caption+context
        for rec in records.values():
            if rec.rendered_png and not rec.description_text:
                rec.description_text = f"Caption: {rec.caption} | Context: {rec.context_before[:500]} {rec.context_after[:500]}"
        return records

    # Параллельный vision
    deadline = time.time() + time_budget_s
    todo = [rec for rec in records.values() if rec.rendered_png]
    log.info(f"vision: {len(todo)} рисунков на описание, бюджет {time_budget_s:.0f}s")

    def call_one(rec: FigureRecord) -> tuple[str, dict | None, str | None]:
        try:
            msgs = _build_vision_messages(rec)
            resp = run_with_timeout(gigachat_client.invoke, PER_VISION_CALL_TIMEOUT_S, msgs)
            content = resp.content if hasattr(resp, "content") else str(resp)
            parsed = _parse_vision_json(content)
            return rec.figure_id, parsed, None
        except Exception as e:
            return rec.figure_id, None, str(e)

    with ThreadPoolExecutor(max_workers=VISION_CONCURRENCY) as ex:
        futures = {ex.submit(call_one, rec): rec for rec in todo}
        for fut in as_completed(futures):
            rec = futures[fut]
            if time.time() > deadline:
                fut.cancel()
                rec.description_text = rec.description_text or f"Caption: {rec.caption}"
                rec.vision_ok = False
                continue
            fid, parsed, err = fut.result()
            if err or not parsed:
                rec.description_text = (
                    f"Caption: {rec.caption} | Context: {rec.context_before[:300]}"
                ).strip()
                rec.vision_ok = False
                if err:
                    log.warning(f"vision[{fid}] failed: {err}")
            else:
                rec.description = parsed
                rec.description_text = _flatten_description(parsed, rec.caption)
                rec.vision_ok = True

    # caption-only фолбэк для оставшихся (например, без rendered_png)
    for rec in records.values():
        if not rec.description_text:
            rec.description_text = f"Caption: {rec.caption}"
    return records


def serialize_records(records: dict[str, FigureRecord]) -> list[dict]:
    out: list[dict] = []
    for r in records.values():
        out.append({
            "figure_id": r.figure_id, "tex_path": r.tex_path,
            "resolved_path": r.resolved_path, "rendered_png": r.rendered_png,
            "caption": r.caption, "section_path": r.section_path,
            "context_before": r.context_before, "context_after": r.context_after,
            "description": r.description, "description_text": r.description_text,
            "vision_ok": r.vision_ok,
        })
    return out


def deserialize_records(items: list[dict]) -> dict[str, FigureRecord]:
    out: dict[str, FigureRecord] = {}
    for it in items:
        r = FigureRecord(
            figure_id=it["figure_id"], tex_path=it.get("tex_path", ""),
            resolved_path=it.get("resolved_path"), rendered_png=it.get("rendered_png"),
            caption=it.get("caption", ""), section_path=it.get("section_path", ""),
            context_before=it.get("context_before", ""),
            context_after=it.get("context_after", ""),
            description=it.get("description") or {},
            description_text=it.get("description_text", ""),
            vision_ok=bool(it.get("vision_ok", False)),
        )
        out[r.figure_id] = r
    return out
