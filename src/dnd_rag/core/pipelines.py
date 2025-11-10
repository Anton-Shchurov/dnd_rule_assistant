from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .io import (
    Section,
    chunk_sections,
    iter_markdown_sections,
    normalize_markdown,
    deep_normalize_markdown,
    parse_pdf_to_markdown,
    save_jsonl,
    save_markdown,
)
from .config import IngestConfig, load_ingest_config, DEFAULT_CONFIG_PATH


def _detect_book_code(pdf_name: str) -> str:
    name = pdf_name.lower()
    if "player" in name or "handbook" in name:
        return "PHB"
    if "dungeon" in name or "master" in name or "dmg" in name:
        return "DMG"
    return Path(pdf_name).stem.upper()


def parse_docs_pipeline(
    raw_dir: str | Path,
    out_md_dir: str | Path,
    *,
    ocr: bool = False,
    ocr_langs: Optional[Sequence[str]] = None,
) -> List[Path]:
    raw_p = Path(raw_dir)
    out_p = Path(out_md_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    produced: List[Path] = []
    for pdf in sorted(raw_p.glob("*.pdf")):
        book = _detect_book_code(pdf.name)
        md = parse_pdf_to_markdown(pdf, ocr=ocr, ocr_langs=ocr_langs)
        md = normalize_markdown(md)
        out_file = out_p / f"{book}.md"
        save_markdown(md, out_file)
        produced.append(out_file)

    return produced


def normalize_md_dir_pipeline(md_dir: str | Path, out_md_dir: str | Path) -> List[Path]:
    """Normalize all Markdown files in a directory and write to another.

    Parameters
    ----------
    md_dir: str | Path
        Directory with input .md files.
    out_md_dir: str | Path
        Destination directory for normalized .md files (created if missing).
    """
    in_p = Path(md_dir)
    out_p = Path(out_md_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    produced: List[Path] = []
    for md_file in sorted(in_p.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        clean = deep_normalize_markdown(text)
        out_file = out_p / md_file.name
        save_markdown(clean, out_file)
        produced.append(out_file)

    return produced


def _slugify(text: str) -> str:
    t = re.sub(r"[^A-Za-z0-9А-Яа-я\-\s:_]", "", text)
    t = re.sub(r"\s+", "-", t).strip("-")
    return t[:60]


def chunk_docs_pipeline(
    md_dir: str | Path,
    out_chunks_dir: str | Path,
    *,
    config_path: Optional[str | Path] = None,
) -> List[Path]:
    cfg: IngestConfig = load_ingest_config(config_path or DEFAULT_CONFIG_PATH)

    md_p = Path(md_dir)
    out_p = Path(out_chunks_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    produced: List[Path] = []
    for md_file in sorted(md_p.glob("*.md")):
        book = md_file.stem.upper()
        md = md_file.read_text(encoding="utf-8")
        sections: Iterable[Section] = iter_markdown_sections(md)
        chunks = chunk_sections(
            sections,
            max_tokens=cfg.chunk_size_tokens,
            overlap=cfg.chunk_overlap_tokens,
        )

        rows = []
        for i, ch in enumerate(chunks, start=1):
            chapter = ch.chapter or ""
            sec = ch.section or ""
            page_range = (
                f"{ch.page_start}-{ch.page_end}"
                if ch.page_start is not None and ch.page_end is not None
                else ""
            )
            cid = f"{book}:{_slugify(chapter)}:{_slugify(sec)}:{page_range}:{i:04d}"
            rows.append(
                {
                    "id": cid,
                    "book": book,
                    "chapter": chapter or None,
                    "section": sec or None,
                    "page_start": ch.page_start,
                    "page_end": ch.page_end,
                    "text": ch.text,
                    "tokens": ch.tokens,
                    "source_md": str(md_file.as_posix()),
                }
            )

        out_file = out_p / f"{book}.jsonl"
        save_jsonl(rows, out_file)
        produced.append(out_file)

    return produced

