"""
================================================================================
EN: Document Processing Pipelines for D&D RAG System
RU: Конвейеры обработки документов для D&D RAG системы
================================================================================

EN: This module contains the ingestion pipelines used in the modern two-step flow:
RU: Этот модуль содержит конвейеры загрузки, используемые в современном двухэтапном процессе:

EN: 1. parse_docs_pipeline: Convert PDF files to Markdown format
RU: 1. parse_docs_pipeline: Конвертирует PDF файлы в формат Markdown

EN: 2. normalize_md_dir_pipeline: Clean and standardize Markdown files
RU: 2. normalize_md_dir_pipeline: Очищает и стандартизирует Markdown файлы

EN: 3. sections_from_md_pipeline: Split normalized Markdown into structural sections
RU: 3. sections_from_md_pipeline: Делит нормализованный Markdown на структурные секции

EN: 4. chunks_from_sections_pipeline: Apply token-based chunking to sections
RU: 4. chunks_from_sections_pipeline: Применяет токеновый чанкинг к секциям

EN: These pipelines are designed to work sequentially:
    PDF → Markdown → Normalized Markdown → Sections JSONL → Chunks JSONL
RU: Эти конвейеры предназначены для последовательной работы:
    PDF → Markdown → Нормализованный Markdown → JSONL секции → JSONL чанки
    
EN: Each pipeline is idempotent and can be run multiple times safely.
RU: Каждый конвейер идемпотентен и может быть запущен несколько раз безопасно.
================================================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# EN: Import functions for working with files and text processing
# RU: Импортируем функции для работы с файлами и обработки текста
from .io import (
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
from .retriever import FilterLike, RetrievedChunk, Retriever
from .prompts import get_system_prompt
from dnd_rag.providers.embeddings import embed_texts
from dnd_rag.providers.llm import ChatMessage, LLMClient


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


@dataclass
class AnswerResult:
    """LLM answer together with retrieved chunks."""

    answer: str
    model: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    chunks: List[RetrievedChunk]


def _safe_truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def _format_source_title(payload: Dict[str, Any]) -> str:
    book = payload.get("book_title") or payload.get("book") or ""
    chapter = payload.get("chapter_title") or payload.get("chapter") or ""
    sections = payload.get("section_path") or []
    if isinstance(sections, list):
        sections_str = " › ".join([s for s in sections if s])
    else:
        sections_str = str(sections)

    parts = [part for part in (book, chapter, sections_str) if part]
    chunk_id = payload.get("chunk_id")
    if chunk_id:
        parts.append(f"id={chunk_id}")
    return " • ".join(parts) if parts else (chunk_id or "fragment")


def _render_context(chunks: List[RetrievedChunk], *, max_chars_per_chunk: int) -> str:
    blocks: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        text = (chunk.text or "").strip()
        if not text:
            text = "⚠️ Текст отсутствует в payload (переиндексируйте с полем text)."
        text = _safe_truncate(text, max_chars_per_chunk)
        title = _format_source_title(chunk.payload)
        blocks.append(f"[{idx}] {title}\n{text}")
    return "\n\n".join(blocks)


def _build_user_prompt(question: str, context_block: str) -> str:
    return (
        "Ответь на вопрос, используя только приведённые ниже выдержки из правил.\n"
        "Если ответа нет, скажи об этом напрямую.\n"
        "Каждое утверждение подтверждай ссылкой на источник вида [номер], где номер — блок из раздела «Контекст».\n\n"
        f"Контекст:\n{context_block}\n\n"
        f"Вопрос: {question}\n\n"
        "Формат ответа: связный текст на языке вопроса с пометками [номер]. "
        "После ответа можешь кратко перечислить использованные источники."
    )


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


def answer_query_pipeline(
    question: str,
    *,
    collection: str = "dnd_rule_assistant",
    host: str = "localhost",
    port: int = 6333,
    k: int = 5,
    config_path: Optional[str | Path] = None,
    filters: FilterLike = None,
    retriever: Optional[Retriever] = None,
    llm_client: Optional[LLMClient] = None,
    system_prompt: Optional[str] = None,
    prompts_path: Optional[str | Path] = None,
    embedding_model: str = "text-embedding-3-small",
    temperature: Optional[float] = None,
    max_chars_per_chunk: int = 1500,
) -> AnswerResult:
    """
    High-level pipeline: question → retrieval → LLM answer.

    RU: Высокоуровневый пайплайн: вопрос → поиск → ответ LLM.
    """

    if not question.strip():
        raise ValueError("Вопрос не может быть пустым.")

    cfg = load_ingest_config(config_path or DEFAULT_CONFIG_PATH)
    retr = retriever or Retriever(collection=collection, host=host, port=port)
    llm = llm_client or LLMClient(model=cfg.llm_model_name)

    query_vec = embed_texts([question], model=embedding_model)[0]
    retrieved = retr.search(query_vec, limit=k, query_filter=filters)

    if not retrieved:
        return AnswerResult(
            answer="Не удалось найти релевантные фрагменты в Qdrant.",
            model=llm.model,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            chunks=[],
        )

    context_block = _render_context(retrieved, max_chars_per_chunk=max_chars_per_chunk)
    sys_prompt = system_prompt if system_prompt is not None else get_system_prompt(prompts_path)
    messages: List[ChatMessage] = []
    if sys_prompt:
        messages.append(ChatMessage(role="system", content=sys_prompt))
    messages.append(ChatMessage(role="user", content=_build_user_prompt(question, context_block)))

    llm_response = llm.generate(messages, temperature=temperature)

    return AnswerResult(
        answer=llm_response.content,
        model=llm_response.model,
        prompt_tokens=llm_response.prompt_tokens,
        completion_tokens=llm_response.completion_tokens,
        total_tokens=llm_response.total_tokens,
        chunks=retrieved,
    )

