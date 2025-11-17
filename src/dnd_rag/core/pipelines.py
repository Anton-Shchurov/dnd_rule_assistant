"""
================================================================================
EN: Document Processing Pipelines for D&D RAG System
RU: Конвейеры обработки документов для D&D RAG системы
================================================================================

EN: This module contains three main pipelines for processing D&D rulebooks:
RU: Этот модуль содержит три основных конвейера для обработки книг правил D&D:

EN: 1. parse_docs_pipeline: Convert PDF files to Markdown format
RU: 1. parse_docs_pipeline: Конвертирует PDF файлы в формат Markdown

EN: 2. normalize_md_dir_pipeline: Clean and standardize Markdown files
RU: 2. normalize_md_dir_pipeline: Очищает и стандартизирует Markdown файлы

EN: 3. chunk_docs_pipeline: Split documents into searchable chunks with metadata
RU: 3. chunk_docs_pipeline: Разбивает документы на чанки для поиска с метаданными

EN: These pipelines are designed to work sequentially:
    PDF → Markdown → Normalized Markdown → JSON chunks
RU: Эти конвейеры предназначены для последовательной работы:
    PDF → Markdown → Нормализованный Markdown → JSON чанки
    
EN: Each pipeline is idempotent and can be run multiple times safely.
RU: Каждый конвейер идемпотентен и может быть запущен несколько раз безопасно.
================================================================================
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# EN: Import functions for working with files and text processing
# RU: Импортируем функции для работы с файлами и обработки текста
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
# EN: Import configuration loading utilities
# RU: Импортируем утилиты для загрузки конфигурации
from .config import (
    DEFAULT_CONFIG_PATH,
    IngestConfig,
    load_ingest_config,
)
from .section_parser import iter_structural_sections, to_dict as section_to_dict
from .chunking import SectionRecord, split_sections_into_chunks


# ============================================================================
# EN: Helper function to detect book code from PDF filename
# RU: Вспомогательная функция для определения кода книги по имени PDF файла
# ============================================================================
def _detect_book_code(pdf_name: str) -> str:
    """
    EN: Automatically detect book abbreviation from PDF filename.
    RU: Автоматически определяет аббревиатуру книги из имени PDF файла.
    
    EN: For example: "player_handbook.pdf" → "PHB", "dungeon_master_guide.pdf" → "DMG"
    RU: Например: "player_handbook.pdf" → "PHB", "dungeon_master_guide.pdf" → "DMG"
    """
    # EN: Convert filename to lowercase for easier matching
    # RU: Преобразуем имя файла в нижний регистр для упрощения поиска
    name = pdf_name.lower()
    
    # EN: Check if filename contains "player" or "handbook" - these are Player's Handbook
    # RU: Проверяем, содержит ли имя "player" или "handbook" - это книга игрока
    if "player" in name or "handbook" in name:
        return "PHB"
    
    # EN: Check if filename contains "dungeon", "master" or "dmg" - these are Dungeon Master's Guide
    # RU: Проверяем, содержит ли имя "dungeon", "master" или "dmg" - это руководство мастера
    if "dungeon" in name or "master" in name or "dmg" in name:
        return "DMG"
    
    # EN: If no match found, use the filename (without extension) in uppercase
    # RU: Если совпадений не найдено, используем имя файла (без расширения) в верхнем регистре
    return Path(pdf_name).stem.upper()


# ============================================================================
# EN: PIPELINE 1: Convert PDF documents to normalized Markdown files
# RU: КОНВЕЙЕР 1: Конвертация PDF документов в нормализованные Markdown файлы
# ============================================================================
def parse_docs_pipeline(
    raw_dir: str | Path,
    out_md_dir: str | Path,
    *,
    ocr: bool = False,
    ocr_langs: Optional[Sequence[str]] = None,
) -> List[Path]:
    """
    EN: Parse all PDF files in a directory and convert them to Markdown format.
    RU: Парсит все PDF файлы в директории и конвертирует их в формат Markdown.
    
    Parameters / Параметры:
    ----------------------
    raw_dir: EN: Directory containing PDF files to process
             RU: Директория с PDF файлами для обработки
    out_md_dir: EN: Directory where Markdown files will be saved
                RU: Директория, куда будут сохранены Markdown файлы
    ocr: EN: Whether to use OCR (Optical Character Recognition) for scanned PDFs
         RU: Использовать ли OCR (оптическое распознавание текста) для сканированных PDF
    ocr_langs: EN: Languages to use for OCR (e.g., ['eng', 'rus'])
               RU: Языки для использования в OCR (например, ['eng', 'rus'])
    
    Returns / Возвращает:
    --------------------
    EN: List of paths to created Markdown files
    RU: Список путей к созданным Markdown файлам
    """
    # EN: Convert string paths to Path objects for easier manipulation
    # RU: Конвертируем строковые пути в объекты Path для удобной работы
    raw_p = Path(raw_dir)
    out_p = Path(out_md_dir)
    
    # EN: Create output directory if it doesn't exist (parents=True creates parent dirs too)
    # RU: Создаём выходную директорию, если её нет (parents=True создаёт и родительские директории)
    out_p.mkdir(parents=True, exist_ok=True)

    # EN: List to store paths of all created files
    # RU: Список для хранения путей всех созданных файлов
    produced: List[Path] = []
    
    # EN: Loop through all PDF files in the input directory (sorted for consistent order)
    # RU: Проходим по всем PDF файлам во входной директории (сортируем для постоянного порядка)
    for pdf in sorted(raw_p.glob("*.pdf")):
        # EN: Automatically detect which book this is (e.g., PHB, DMG)
        # RU: Автоматически определяем, какая это книга (например, PHB, DMG)
        book = _detect_book_code(pdf.name)
        
        # EN: Convert PDF to Markdown text using OCR if needed
        # RU: Конвертируем PDF в текст Markdown, используя OCR при необходимости
        md = parse_pdf_to_markdown(pdf, ocr=ocr, ocr_langs=ocr_langs)
        
        # EN: Clean up and standardize the Markdown formatting
        # RU: Очищаем и стандартизируем форматирование Markdown
        md = normalize_markdown(md)
        
        # EN: Create output filename using the detected book code
        # RU: Создаём имя выходного файла, используя определённый код книги
        out_file = out_p / f"{book}.md"
        
        # EN: Save the cleaned Markdown text to file
        # RU: Сохраняем очищенный текст Markdown в файл
        save_markdown(md, out_file)
        
        # EN: Add the created file path to our list
        # RU: Добавляем путь созданного файла в наш список
        produced.append(out_file)

    # EN: Return the list of all created files
    # RU: Возвращаем список всех созданных файлов
    return produced


# ============================================================================
# EN: PIPELINE 2: Deep normalization of existing Markdown files
# RU: КОНВЕЙЕР 2: Глубокая нормализация существующих Markdown файлов
# ============================================================================
def normalize_md_dir_pipeline(
    md_dir: str | Path,
    out_md_dir: str | Path,
    *,
    config_path: Optional[str | Path] = None,
) -> List[Path]:
    """
    EN: Normalize all Markdown files in a directory and write to another.
    RU: Нормализует все Markdown файлы в директории и записывает в другую.
    
    EN: This applies deep cleaning: removes extra spaces, fixes formatting issues,
        standardizes headers, etc.
    RU: Применяет глубокую очистку: удаляет лишние пробелы, исправляет проблемы форматирования,
        стандартизирует заголовки и т.д.

    Parameters / Параметры:
    ----------------------
    md_dir: EN: Directory with input .md files
            RU: Директория с входными .md файлами
    out_md_dir: EN: Destination directory for normalized .md files (created if missing)
                RU: Директория назначения для нормализованных .md файлов (создаётся, если отсутствует)
    config_path: EN: Optional path to configuration file
                 RU: Необязательный путь к файлу конфигурации
    
    Returns / Возвращает:
    --------------------
    EN: List of paths to created normalized Markdown files
    RU: Список путей к созданным нормализованным Markdown файлам
    """
    # EN: Load configuration file (or use default if not specified)
    # RU: Загружаем файл конфигурации (или используем стандартный, если не указан)
    ingest_cfg = load_ingest_config(config_path or DEFAULT_CONFIG_PATH)

    # EN: Convert paths to Path objects
    # RU: Конвертируем пути в объекты Path
    in_p = Path(md_dir)
    out_p = Path(out_md_dir)
    
    # EN: Create output directory if needed
    # RU: Создаём выходную директорию при необходимости
    out_p.mkdir(parents=True, exist_ok=True)

    # EN: List to store all created file paths
    # RU: Список для хранения всех созданных путей файлов
    produced: List[Path] = []
    
    # EN: Process each Markdown file in the input directory
    # RU: Обрабатываем каждый Markdown файл во входной директории
    for md_file in sorted(in_p.glob("*.md")):
        # EN: Read the file content as text (UTF-8 encoding for international characters)
        # RU: Читаем содержимое файла как текст (кодировка UTF-8 для международных символов)
        text = md_file.read_text(encoding="utf-8")
        
        # EN: Apply deep normalization (removes artifacts, standardizes formatting)
        # RU: Применяем глубокую нормализацию (удаляет артефакты, стандартизирует форматирование)
        clean = deep_normalize_markdown(text)
        
        # EN: Create output file path with same name as input
        # RU: Создаём путь выходного файла с таким же именем, как у входного
        out_file = out_p / md_file.name
        
        # EN: Save the cleaned Markdown to file
        # RU: Сохраняем очищенный Markdown в файл
        save_markdown(clean, out_file)
        
        # EN: Add to list of created files
        # RU: Добавляем в список созданных файлов
        produced.append(out_file)

    # EN: Return all created file paths
    # RU: Возвращаем все созданные пути файлов
    return produced


# ============================================================================
# EN: Helper function to create URL-friendly text (slug)
# RU: Вспомогательная функция для создания URL-дружественного текста (slug)
# ============================================================================
def _slugify(text: str) -> str:
    """
    EN: Convert text to a clean slug (URL-friendly format).
    RU: Преобразует текст в чистый slug (формат, подходящий для URL).
    
    EN: Example: "Chapter 1: Magic Items!" → "Chapter-1-Magic-Items"
    RU: Пример: "Глава 1: Магические предметы!" → "Глава-1-Магические-предметы"
    """
    # EN: Remove all special characters except letters, numbers, hyphens, spaces, colons, underscores
    # RU: Удаляем все специальные символы кроме букв, цифр, дефисов, пробелов, двоеточий, подчёркиваний
    t = re.sub(r"[^A-Za-z0-9А-Яа-я\-\s:_]", "", text)
    
    # EN: Replace all whitespace sequences with a single hyphen
    # RU: Заменяем все последовательности пробелов на один дефис
    t = re.sub(r"\s+", "-", t).strip("-")
    
    # EN: Limit to 60 characters for reasonable ID length
    # RU: Ограничиваем до 60 символов для разумной длины ID
    return t[:60]


# ============================================================================
# EN: PIPELINE 3: Split Markdown documents into chunks for search/retrieval
# RU: КОНВЕЙЕР 3: Разбиение Markdown документов на чанки для поиска/извлечения
# ============================================================================
def chunk_docs_pipeline(
    md_dir: str | Path,
    out_chunks_dir: str | Path,
    *,
    config_path: Optional[str | Path] = None,
) -> List[Path]:
    """
    EN: Split Markdown documents into smaller overlapping chunks for better search.
    RU: Разбивает Markdown документы на меньшие перекрывающиеся чанки для лучшего поиска.
    
    EN: This is essential for RAG (Retrieval-Augmented Generation) systems.
        Each chunk gets a unique ID and metadata (chapter, section, pages, etc.).
    RU: Это необходимо для RAG (генерация с дополнением извлечением) систем.
        Каждый чанк получает уникальный ID и метаданные (глава, раздел, страницы и т.д.).
    
    Parameters / Параметры:
    ----------------------
    md_dir: EN: Directory containing normalized Markdown files
            RU: Директория с нормализованными Markdown файлами
    out_chunks_dir: EN: Directory where JSONL chunk files will be saved
                    RU: Директория, куда будут сохранены JSONL файлы с чанками
    config_path: EN: Optional configuration file path (defines chunk size, overlap, etc.)
                 RU: Необязательный путь к файлу конфигурации (определяет размер чанка, перекрытие и т.д.)
    
    Returns / Возвращает:
    --------------------
    EN: List of paths to created JSONL files (one per book)
    RU: Список путей к созданным JSONL файлам (по одному на книгу)
    """
    # EN: Load configuration to get chunk size and overlap settings
    # RU: Загружаем конфигурацию для получения настроек размера чанка и перекрытия
    cfg: IngestConfig = load_ingest_config(config_path or DEFAULT_CONFIG_PATH)

    # EN: Convert paths to Path objects
    # RU: Конвертируем пути в объекты Path
    md_p = Path(md_dir)
    out_p = Path(out_chunks_dir)
    
    # EN: Create output directory if it doesn't exist
    # RU: Создаём выходную директорию, если её нет
    out_p.mkdir(parents=True, exist_ok=True)

    # EN: List to track all created files
    # RU: Список для отслеживания всех созданных файлов
    produced: List[Path] = []
    
    # EN: Process each Markdown file (each represents one book)
    # RU: Обрабатываем каждый Markdown файл (каждый представляет одну книгу)
    for md_file in sorted(md_p.glob("*.md")):
        # EN: Extract book code from filename (e.g., "PHB.md" → "PHB")
        # RU: Извлекаем код книги из имени файла (например, "PHB.md" → "PHB")
        book = md_file.stem.upper()
        
        # EN: Read the entire Markdown file content
        # RU: Читаем всё содержимое Markdown файла
        md = md_file.read_text(encoding="utf-8")
        
        # EN: Parse the Markdown into logical sections (by headers)
        # RU: Парсим Markdown в логические разделы (по заголовкам)
        sections: Iterable[Section] = iter_markdown_sections(md)
        
        # EN: Split sections into smaller chunks with overlap for better context
        # RU: Разбиваем разделы на меньшие чанки с перекрытием для лучшего контекста
        chunks = chunk_sections(
            sections,
            max_tokens=cfg.chunk_size_tokens,  # EN: Max size per chunk | RU: Макс. размер на чанк
            overlap=cfg.chunk_overlap_tokens,  # EN: Tokens to overlap between chunks | RU: Токены для перекрытия между чанками
        )

        # EN: List to store all chunk data as dictionaries
        # RU: Список для хранения всех данных чанков в виде словарей
        rows = []
        
        # EN: Process each chunk and create metadata
        # RU: Обрабатываем каждый чанк и создаём метаданные
        for i, ch in enumerate(chunks, start=1):
            # EN: Extract chapter name (or empty string if not available)
            # RU: Извлекаем название главы (или пустую строку, если недоступно)
            chapter = ch.chapter or ""
            
            # EN: Extract section name (or empty string if not available)
            # RU: Извлекаем название раздела (или пустую строку, если недоступно)
            sec = ch.section or ""
            
            # EN: Create page range string (e.g., "15-17") if pages are available
            # RU: Создаём строку диапазона страниц (например, "15-17"), если страницы доступны
            page_range = (
                f"{ch.page_start}-{ch.page_end}"
                if ch.page_start is not None and ch.page_end is not None
                else ""
            )
            
            # EN: Create unique chunk ID with format: BOOK:chapter-slug:section-slug:pages:index
            # RU: Создаём уникальный ID чанка с форматом: КНИГА:slug-главы:slug-раздела:страницы:индекс
            # EN: Example: "PHB:Chapter-3-Classes:Barbarian:45-47:0001"
            # RU: Пример: "PHB:Глава-3-Классы:Варвар:45-47:0001"
            cid = f"{book}:{_slugify(chapter)}:{_slugify(sec)}:{page_range}:{i:04d}"
            
            # EN: Create a dictionary with all chunk information
            # RU: Создаём словарь со всей информацией о чанке
            rows.append(
                {
                    "id": cid,                      # EN: Unique identifier | RU: Уникальный идентификатор
                    "book": book,                   # EN: Book code | RU: Код книги
                    "chapter": chapter or None,     # EN: Chapter name | RU: Название главы
                    "section": sec or None,         # EN: Section name | RU: Название раздела
                    "page_start": ch.page_start,    # EN: First page | RU: Первая страница
                    "page_end": ch.page_end,        # EN: Last page | RU: Последняя страница
                    "text": ch.text,                # EN: Actual chunk text | RU: Фактический текст чанка
                    "tokens": ch.tokens,            # EN: Token count | RU: Количество токенов
                    "source_md": str(md_file.as_posix()),  # EN: Source file path | RU: Путь к исходному файлу
                }
            )

        # EN: Save all chunks for this book to a JSONL file (one JSON object per line)
        # RU: Сохраняем все чанки для этой книги в JSONL файл (один JSON объект на строку)
        out_file = out_p / f"{book}.jsonl"
        save_jsonl(rows, out_file)
        
        # EN: Add to list of created files
        # RU: Добавляем в список созданных файлов
        produced.append(out_file)

    # EN: Return all created JSONL file paths
    # RU: Возвращаем все созданные пути JSONL файлов
    return produced


# ============================================================================
# RU: НОВЫЙ ПАЙПЛАЙН — Секции (без резки): MD → sections/*.jsonl
# EN: NEW PIPELINE — Structural sections: MD → sections/*.jsonl
# ============================================================================
def sections_from_md_pipeline(
    md_dir: str | Path,
    out_sections_dir: str | Path,
) -> List[Path]:
    """
    Разбирает Markdown по заголовкам и сохраняет логические секции в JSONL.
    На выходе строки формата:
    { book_title, chapter_title, section_path: [...], text }
    """
    in_p = Path(md_dir)
    out_p = Path(out_sections_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    produced: List[Path] = []
    for md_file in sorted(in_p.glob("*.md")):
        md = md_file.read_text(encoding="utf-8")
        rows = (section_to_dict(s) for s in iter_structural_sections(md))
        out_file = out_p / f"{md_file.stem}.jsonl"
        save_jsonl(rows, out_file)
        produced.append(out_file)
    return produced


# ============================================================================
# RU: НОВЫЙ ПАЙПЛАЙН — Чанки: sections/*.jsonl → chunks/*.jsonl
# EN: NEW PIPELINE — Chunks: sections/*.jsonl → chunks/*.jsonl
# ============================================================================
def chunks_from_sections_pipeline(
    sections_dir: str | Path,
    out_chunks_dir: str | Path,
    *,
    config_path: Optional[str | Path] = None,
) -> List[Path]:
    """
    Делит каждую секцию через RecursiveCharacterTextSplitter и сохраняет чанки:
    { chunk_id, book_title, chapter_title, section_path, chunk_index, text }
    """
    cfg: IngestConfig = load_ingest_config(config_path or DEFAULT_CONFIG_PATH)

    in_p = Path(sections_dir)
    out_p = Path(out_chunks_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    produced: List[Path] = []
    for sec_file in sorted(in_p.glob("*.jsonl")):
        # Загружаем секции
        raw_rows: List[dict] = []
        with sec_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_rows.append(json.loads(line))

        secs: List[SectionRecord] = [
            SectionRecord(
                book_title=(r.get("book_title") or ""),
                chapter_title=(r.get("chapter_title") or ""),
                section_path=list(r.get("section_path") or []),
                text=(r.get("text") or ""),
            )
            for r in raw_rows
        ]

        chunks = split_sections_into_chunks(
            secs,
            source_file=sec_file,
            ingest_cfg=cfg,
        )

        out_file = out_p / f"{sec_file.stem}.jsonl"
        save_jsonl(chunks, out_file)
        produced.append(out_file)

    return produced

