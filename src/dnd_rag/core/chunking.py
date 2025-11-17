from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import IngestConfig


@dataclass
class SectionRecord:
    """
    Запись секции, соответствующая строке из sections JSONL.
    """
    book_title: str
    chapter_title: str
    section_path: List[str]
    text: str


def _slug(text: str) -> str:
    """
    Простой slug: нижний регистр, небуквенно-цифровое → '_', схлопывание подряд и обрезка.
    """
    import re

    t = text.strip().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t, flags=re.IGNORECASE)
    t = re.sub(r"_+", "_", t)
    return t.strip("_")


def _book_slug_from_path(path: Path) -> str:
    """
    Получает slug книги по имени файла Markdown или JSONL: DMG.md → dmg, PHB.jsonl → phb.
    """
    stem = Path(path).stem
    return _slug(stem)


def _rcts(encoding_name: str, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Токеновый RecursiveCharacterTextSplitter (tiktoken cl100k_base).
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )


def split_sections_into_chunks(
    sections: Iterable[SectionRecord],
    *,
    source_file: Path,
    ingest_cfg: IngestConfig,
) -> List[Dict[str, Any]]:
    """
    Делит секции на чанки через RCTS. Присваивает chunk_index по главам (H2).
    chunk_id = {book_slug}_ch{chapter_index:02d}_{chunk_index:04d}
    """
    # Готовим RCTS
    splitter = _rcts(
        encoding_name="cl100k_base",
        chunk_size=ingest_cfg.chunk_size_tokens,
        chunk_overlap=ingest_cfg.chunk_overlap_tokens,
    )

    # Переиндексация глав в порядке появления H2
    chapter_to_idx: Dict[str, int] = {}
    next_ch_idx = 1

    # Счётчик чанков по текущей главе
    per_chapter_chunk_counter: Dict[int, int] = {}

    book_slug = _book_slug_from_path(source_file)

    out: List[Dict[str, Any]] = []

    # Идем в порядке секций (как в документе)
    for rec in sections:
        ch = rec.chapter_title or ""
        if ch not in chapter_to_idx:
            chapter_to_idx[ch] = next_ch_idx
            per_chapter_chunk_counter[chapter_to_idx[ch]] = 0
            next_ch_idx += 1

        ch_idx = chapter_to_idx[ch]

        # Разбиваем текст секции
        # В качестве страницы/метаданных RCTS не используется — только текст
        splits = splitter.split_text(rec.text or "")

        for split_text in splits:
            per_chapter_chunk_counter[ch_idx] += 1
            chunk_index = per_chapter_chunk_counter[ch_idx]
            chunk_id = f"{book_slug}_ch{ch_idx:02d}_{chunk_index:04d}"

            out.append(
                {
                    "chunk_id": chunk_id,
                    "book_title": rec.book_title or "",
                    "chapter_title": ch,
                    "section_path": list(rec.section_path),
                    "chunk_index": chunk_index,
                    "text": split_text,
                }
            )

    return out


