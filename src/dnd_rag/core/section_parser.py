from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Dict, Any

# Используем уже реализованный парсер секций из io.py как низкоуровневый
from .io import iter_markdown_sections


@dataclass
class StructuralSection:
    """
    Логическая секция Markdown без резки по длине.
    """
    book_title: str
    chapter_title: str
    section_path: List[str]
    text: str


def _titles_from_path(path: Sequence[Tuple[int, str]]) -> List[str]:
    """
    Преобразует путь [(level, title), ...] в список заголовков по уровню.
    """
    return [title for _, title in path]


def _book_title_from_path(path: Sequence[Tuple[int, str]]) -> Optional[str]:
    for level, title in path:
        if level == 1:
            return title
    return None


def _chapter_title_from_path(path: Sequence[Tuple[int, str]], current_level: int, current_title: str) -> str:
    """
    Возвращает заголовок главы (H2). Для H2 — это текущий заголовок,
    для H3+ — ближайший предок уровня 2. Если не найдено, возвращает пустую строку.
    """
    if current_level == 2:
        return current_title
    # Ищем ближайший H2 среди предков
    for level, title in path:
        if level == 2:
            return title
    return ""


def iter_structural_sections(md_text: str) -> Iterator[StructuralSection]:
    """
    Итератор по логическим секциям:
    - H1 трактуется как название книги (в секции не возвращается)
    - H2 = глава; H3+ = подразделы.
    - section_path: [H1, H2, H3, ...] как в плане.
    """
    # Проходим по всем секциям Markdown
    for sec in iter_markdown_sections(md_text):
        # Пропускаем H1 как самостоятельную секцию
        if sec.level <= 1:
            continue

        titles = _titles_from_path(sec.path)
        book_title = _book_title_from_path(sec.path) or ""
        chapter_title = _chapter_title_from_path(sec.path, sec.level, sec.title)

        yield StructuralSection(
            book_title=book_title,
            chapter_title=chapter_title,
            section_path=titles,  # По плану включаем H1 в path
            text=sec.text or "",
        )


def to_dict(row: StructuralSection) -> Dict[str, Any]:
    """
    Преобразует StructuralSection в dict для JSONL.
    """
    return {
        "book_title": row.book_title,
        "chapter_title": row.chapter_title,
        "section_path": list(row.section_path),
        "text": row.text,
    }


