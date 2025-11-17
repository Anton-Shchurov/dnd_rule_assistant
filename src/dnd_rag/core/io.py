"""
Input/Output module for DnD RAG system.
Модуль ввода/вывода для системы RAG по D&D.

This module handles:
Этот модуль обрабатывает:
- PDF to Markdown conversion / Конвертация PDF в Markdown
- Text normalization and OCR error correction / Нормализация текста и исправление ошибок OCR
- Document structure parsing / Парсинг структуры документа
- Text chunking for embeddings / Разбиение текста на чанки для эмбеддингов
"""
from __future__ import annotations

# Standard library imports / Импорты стандартной библиотеки
import json  # For JSONL output / Для вывода в формате JSONL
import re  # Regular expressions for text processing / Регулярные выражения для обработки текста
import html  # HTML entity decoding / Декодирование HTML-сущностей
import unicodedata  # Unicode normalization / Нормализация Unicode
import shutil  # File operations / Операции с файлами
import os  # OS-level operations / Операции на уровне ОС
from dataclasses import dataclass  # For data classes / Для классов данных
from pathlib import Path  # Modern path handling / Современная работа с путями
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from functools import lru_cache  # Caching decorator / Декоратор кэширования

# Third-party imports / Импорты сторонних библиотек
from docling.document_converter import DocumentConverter  # PDF conversion / Конвертация PDF
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import TesseractCliOcrOptions  # OCR options / Настройки OCR
import tiktoken  # OpenAI tokenizer / Токенизатор OpenAI

# Optional morphology library for Russian language processing
# Опциональная библиотека морфологии для обработки русского языка
try:
    import pymorphy2
except ImportError:  # pragma: no cover - отсутствие зависимости не критично
    pymorphy2 = None  # type: ignore[assignment]


# =============================================================================
# TOKENIZATION HELPERS / ПОМОЩНИКИ ДЛЯ ТОКЕНИЗАЦИИ
# =============================================================================

# Initialize the tokenizer used by OpenAI (cl100k_base encoding)
# Инициализируем токенизатор, используемый OpenAI (кодировка cl100k_base)
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Initialize morphological analyzer if pymorphy2 is available
# Инициализируем морфологический анализатор, если доступен pymorphy2
if pymorphy2 is not None:  # pragma: no cover - зависит от окружения
    try:
        _MORPH_ANALYZER = pymorphy2.MorphAnalyzer()
    except Exception:  # pragma: no cover - безопасный фолбэк
        _MORPH_ANALYZER = None
else:
    _MORPH_ANALYZER = None


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in text using OpenAI's tokenizer.
    Подсчитывает количество токенов в тексте с помощью токенизатора OpenAI.
    
    Parameters / Параметры:
        text: Input text / Входной текст
        
    Returns / Возвращает:
        Number of tokens / Количество токенов
    """
    return len(_ENCODING.encode(text or ""))


# =============================================================================
# DATA MODELS / МОДЕЛИ ДАННЫХ
# =============================================================================

@dataclass
class Section:
    """
    Represents a document section with hierarchical structure.
    Представляет секцию документа с иерархической структурой.
    
    Attributes / Атрибуты:
        title: Section heading / Заголовок секции
        level: Heading level (1-6 for # to ######) / Уровень заголовка (1-6 для # до ######)
        text: Section content / Содержимое секции
        page_start: First page of section / Первая страница секции
        page_end: Last page of section / Последняя страница секции
        path: Breadcrumb trail of parent headings / Цепочка родительских заголовков
    """
    title: str
    level: int
    text: str
    page_start: Optional[int]
    page_end: Optional[int]
    path: Tuple[Tuple[int, str], ...]  # breadcrumb of headings up to this section


@dataclass
class SuspiciousParagraph:
    """
    Paragraph flagged as potentially containing OCR errors.
    Параграф, помеченный как потенциально содержащий ошибки OCR.
    
    Attributes / Атрибуты:
        text: Paragraph text / Текст параграфа
        score: Suspicion score (0-1, higher = more suspicious) / Оценка подозрительности (0-1, выше = более подозрительный)
        reasons: List of detected issues / Список обнаруженных проблем
        start_line: First line number in document / Номер первой строки в документе
        end_line: Last line number in document / Номер последней строки в документе
    """
    text: str
    score: float
    reasons: List[str]
    start_line: int
    end_line: int


# =============================================================================
# PDF PARSING & BASIC NORMALIZATION / ПАРСИНГ PDF И БАЗОВАЯ НОРМАЛИЗАЦИЯ
# =============================================================================

def parse_pdf_to_markdown(
    pdf_path: str | Path,
    *,
    ocr: bool = False,
    ocr_langs: Optional[Sequence[str]] = None,
) -> str:
    """
    Convert a PDF into Markdown using Docling library.
    Конвертирует PDF в Markdown с помощью библиотеки Docling.

    Parameters / Параметры
    ----------
    pdf_path: str | Path
        Path to PDF file / Путь к PDF файлу
    ocr: bool
        If False, Docling использует встроенный текст PDF без OCR.
        Если True, Docling включает OCR-пайплайн (PaddleOCR/Tesseract).
        
        If False, Docling uses embedded PDF text without OCR.
        If True, Docling enables OCR pipeline (PaddleOCR/Tesseract).
    ocr_langs: Optional[Sequence[str]]
        Список языковых подсказок для OCR (например, ("rus", "eng")).
        По умолчанию используется ("rus", "eng"), если ocr=True.
        
        List of language hints for OCR (e.g., ("rus", "eng")).
        Defaults to ("rus", "eng") if ocr=True.
    
    Returns / Возвращает
    -------
    str
        Markdown text extracted from PDF / Текст Markdown, извлеченный из PDF
    """
    # Step 1: Initialize the document converter
    # Шаг 1: Инициализируем конвертер документов
    converter = DocumentConverter()
    
    # Step 2: Get PDF-specific options
    # Шаг 2: Получаем настройки для PDF
    pdf_option = converter.format_to_options.get(InputFormat.PDF)
    pipeline_options = None
    if pdf_option is not None:
        pipeline_options = getattr(pdf_option, "pipeline_options", None)

    # Step 3: Configure OCR settings if requested
    # Шаг 3: Настраиваем OCR, если требуется
    if pipeline_options is not None:
        # Enable/disable OCR mode / Включаем/выключаем режим OCR
        if hasattr(pipeline_options, "do_ocr"):
            pipeline_options.do_ocr = bool(ocr)
        
        if ocr:
            # Set default languages if not provided / Устанавливаем языки по умолчанию
            if ocr_langs is None:
                ocr_langs = ("rus", "eng")
            
            # Process language list: remove duplicates and normalize
            # Обрабатываем список языков: удаляем дубликаты и нормализуем
            lang_list = list(dict.fromkeys(ocr_langs)) if ocr_langs else []
            lang_list = [str(lang).strip().lower() for lang in lang_list if lang]
            
            # Convert ISO 639-1 codes to ISO 639-2 (en→eng, ru→rus)
            # Конвертируем коды ISO 639-1 в ISO 639-2 (en→eng, ru→rus)
            if lang_list:
                _ISO639_2 = {"en": "eng", "ru": "rus"}
                lang_list = [_ISO639_2.get(code, code) for code in lang_list]
            
            # Fallback to English if no languages specified
            # Используем английский по умолчанию, если языки не указаны
            if not lang_list:
                lang_list = ["eng"]
            
            # Apply language settings to various OCR parameters
            # Применяем языковые настройки к различным параметрам OCR
            if hasattr(pipeline_options, "ocr_lang_hint"):
                pipeline_options.ocr_lang_hint = lang_list
            if hasattr(pipeline_options, "ocr_languages"):
                pipeline_options.ocr_languages = lang_list
            
            if hasattr(pipeline_options, "ocr_options"):
                # Prefer Tesseract CLI: RapidOCR often hangs on PDFs with Cyrillic
                # Предпочитаем tesseract CLI: RapidOCR часто зависает на PDF с кириллицей
                tess_cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
                if tess_cmd is None:
                    raise RuntimeError(
                        "Tesseract OCR не найден (исполняемый файл tesseract.exe отсутствует в PATH). "
                        "Установите Tesseract и добавьте его в PATH или задайте переменную окружения "
                        "TESSERACT_CMD с полным путём до tesseract.exe."
                    )
                
                # Configure Tesseract with our language list
                # Настраиваем Tesseract с нашим списком языков
                options = TesseractCliOcrOptions(
                    lang=lang_list,
                    tesseract_cmd=tess_cmd,
                )
                options.force_full_page_ocr = True  # OCR entire page / OCR всей страницы
                pipeline_options.ocr_options = options
            
            # Set page segmentation mode to auto
            # Устанавливаем режим сегментации страницы на автоматический
            if hasattr(pipeline_options, "ocr_psm") and getattr(pipeline_options, "ocr_psm") is None:
                pipeline_options.ocr_psm = "auto"
    
    # Step 4: Convert PDF to Markdown
    # Шаг 4: Конвертируем PDF в Markdown
    result = converter.convert(str(pdf_path))
    md: str = result.document.export_to_markdown()
    return md


def save_markdown(md: str, out_path: str | Path) -> None:
    """
    Save Markdown text to a file.
    Сохраняет текст Markdown в файл.
    
    Parameters / Параметры:
        md: Markdown text / Текст Markdown
        out_path: Output file path / Путь к выходному файлу
    """
    out_p = Path(out_path)
    # Create parent directories if they don't exist
    # Создаем родительские директории, если их нет
    out_p.parent.mkdir(parents=True, exist_ok=True)
    # Write text with UTF-8 encoding / Записываем текст в кодировке UTF-8
    out_p.write_text(md, encoding="utf-8")


def save_jsonl(rows: Iterable[Dict], out_path: str | Path) -> None:
    """
    Save data as JSON Lines format (one JSON object per line).
    Сохраняет данные в формате JSON Lines (один JSON объект на строку).
    
    Parameters / Параметры:
        rows: Iterable of dictionaries / Итерируемая коллекция словарей
        out_path: Output file path / Путь к выходному файлу
    """
    out_p = Path(out_path)
    # Create parent directories if they don't exist
    # Создаем родительские директории, если их нет
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    # Write each row as a separate JSON line
    # Записываем каждую строку как отдельную JSON строку
    with out_p.open("w", encoding="utf-8") as f:
        for row in rows:
            # ensure_ascii=False preserves Unicode characters (Cyrillic, etc.)
            # ensure_ascii=False сохраняет Unicode символы (кириллицу и т.д.)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_markdown(md: str) -> str:
    """
    Light-weight normalization: EOL hyphenation, whitespace, blank lines.
    Легкая нормализация: переносы строк, пробелы, пустые строки.

    Operations / Операции:
    - Fix soft hyphenation across line breaks: "перено-\nсы" → "переносы"
      Исправляет мягкий перенос через разрыв строки: "перено-\nсы" → "переносы"
    - Trim trailing spaces / Удаляет конечные пробелы
    - Unify newlines / Унифицирует символы новой строки
    - Collapse excessive blank lines / Схлопывает избыточные пустые строки
    
    Parameters / Параметры:
        md: Input Markdown text / Входной текст Markdown
        
    Returns / Возвращает:
        Normalized text / Нормализованный текст
    """
    if not md:
        return md

    # Step 1: Unify line endings to Unix style (\n)
    # Шаг 1: Унифицируем окончания строк в стиль Unix (\n)
    text = md.replace("\r\n", "\n").replace("\r", "\n")

    # Step 2: Remove hyphenation only when surrounded by word chars
    # This avoids breaking bullet lists like "- item"
    # Шаг 2: Удаляем переносы только когда они окружены буквами слов
    # Это избегает поломки маркированных списков типа "- пункт"
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)

    # Step 3: Trim trailing spaces and tabs from each line
    # Шаг 3: Удаляем конечные пробелы и табуляции из каждой строки
    text = re.sub(r"[ \t]+\n", "\n", text)

    # Step 4: Collapse more than 2 blank lines into 2
    # Шаг 4: Схлопываем более 2 пустых строк в 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# =============================================================================
# DEEP NORMALIZATION (for noisy OCR/Docling output)
# ГЛУБОКАЯ НОРМАЛИЗАЦИЯ (для шумного вывода OCR/Docling)
# =============================================================================

def _decode_uni_escapes(text: str) -> str:
    """
    Convert sequences like '/uni041F' to actual Unicode chars (e.g., 'П').
    Конвертирует последовательности типа '/uni041F' в реальные Unicode символы (например, 'П').
    
    OCR sometimes produces escape sequences instead of actual characters.
    OCR иногда производит escape-последовательности вместо реальных символов.
    
    Example / Пример: '/uni041F' → 'П'
    """
    return re.sub(r"/uni([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), text)


def _html_unescape(text: str) -> str:
    """
    Decode HTML entities (&amp; → &, &lt; → <, etc.).
    Декодирует HTML-сущности (&amp; → &, &lt; → <, и т.д.).
    
    Useful when PDF text contains HTML-encoded characters.
    Полезно, когда текст PDF содержит HTML-кодированные символы.
    """
    return html.unescape(text)


# Valid single-letter words in Russian (prepositions, pronouns, conjunctions)
# Допустимые однобуквенные слова в русском (предлоги, местоимения, союзы)
_SINGLE_LETTER_WORDS_CYR = {"В", "К", "С", "Я", "О", "И", "А", "У", "Э", "Е", "Ю", "Ы"}

# Valid single-letter words in English
# Допустимые однобуквенные слова в английском
_SINGLE_LETTER_WORDS_LAT = {"A", "I"}

# Pattern: single capital Cyrillic letter + space + Cyrillic word
# Шаблон: одна заглавная кириллическая буква + пробел + кириллическое слово
_RE_SINGLE_LETTER_CYR = re.compile(r"\b([А-ЯЁ])\s+([А-ЯЁа-яё]{2,})\b")

# Pattern: single capital Latin letter + space + Latin word
# Шаблон: одна заглавная латинская буква + пробел + латинское слово
_RE_SINGLE_LETTER_LAT = re.compile(r"\b([A-Z])\s+([A-Za-z]{2,})\b")


def _join_single_letter_splits(text: str) -> str:
    """
    Join words incorrectly split by OCR: 'П оследующие' → 'Последующие'.
    Склеивает слова, неправильно разбитые OCR: 'П оследующие' → 'Последующие'.
    
    OCR sometimes splits the first letter from the rest of the word.
    This function rejoins them unless the single letter is a valid word.
    
    OCR иногда отделяет первую букву от остальной части слова.
    Эта функция склеивает их, если только одиночная буква не является допустимым словом.
    
    Examples / Примеры:
    - 'П оследующие' → 'Последующие' (rejoined / склеено)
    - 'В огромном' → 'В огромном' (kept, 'В' is a valid word / оставлено, 'В' - допустимое слово)
    - 'D UNGEONS' → 'DUNGEONS' (rejoined / склеено)
    """

    def _join_cyr(m: re.Match[str]) -> str:
        """Process Cyrillic matches / Обрабатывает кириллические совпадения"""
        first, rest = m.group(1), m.group(2)
        # Don't join if first letter is a valid single-letter word and rest is not all caps
        # Не соединяем, если первая буква - допустимое однобуквенное слово и остаток не в верхнем регистре
        if first in _SINGLE_LETTER_WORDS_CYR and not rest.isupper():
            return m.group(0)  # Keep original / Оставляем оригинал
        return first + rest  # Join / Склеиваем

    def _join_lat(m: re.Match[str]) -> str:
        """Process Latin matches / Обрабатывает латинские совпадения"""
        first, rest = m.group(1), m.group(2)
        # Don't join if first letter is a valid single-letter word and rest is not all caps
        # Не соединяем, если первая буква - допустимое однобуквенное слово и остаток не в верхнем регистре
        if first in _SINGLE_LETTER_WORDS_LAT and not rest.isupper():
            return m.group(0)  # Keep original / Оставляем оригинал
        return first + rest  # Join / Склеиваем

    # Apply both patterns / Применяем оба шаблона
    text = _RE_SINGLE_LETTER_CYR.sub(_join_cyr, text)
    text = _RE_SINGLE_LETTER_LAT.sub(_join_lat, text)
    return text


def _fix_hyphenation_and_linebreaks(text: str) -> str:
    """
    Fix line-break hyphenation and stray in-word newlines.
    Исправляет переносы через разрыв строки и случайные переводы строк внутри слов.

    Operations / Операции:
    1. Join hyphenated words split across lines: 'пер-\nенос' → 'перенос'
       Соединяет слова с переносом, разбитые на строки: 'пер-\nенос' → 'перенос'
    2. Replace mid-word newlines with spaces (preserving structure)
       Заменяет переводы строк внутри слов пробелами (сохраняя структуру)
    
    Parameters / Параметры:
        text: Input text / Входной текст
        
    Returns / Возвращает:
        Fixed text / Исправленный текст
    """
    # Step 1: Remove hyphen + newline between word characters
    # Шаг 1: Удаляем дефис + перевод строки между буквами слов
    text = re.sub(r"([A-Za-zА-Яа-яЁё])-\s*\n\s*([A-Za-zА-Яа-яЁё])", r"\1\2", text)

    def _join_non_structural(match: re.Match[str]) -> str:
        """
        Join single newlines that are not structural (tables, lists, etc.)
        Соединяет одиночные переводы строк, которые не являются структурными (таблицы, списки и т.д.)
        """
        before = match.group(1)
        newline = match.group(2)
        after = match.group(3)
        
        # Keep newlines in tables (lines containing '|')
        # Сохраняем переводы строк в таблицах (строки с '|')
        if before.rstrip().endswith("|") or after.lstrip().startswith("|"):
            return before + newline + after
        
        # Keep newlines before list items or headings
        # Сохраняем переводы строк перед элементами списков или заголовками
        if after.lstrip().startswith(("#", "-", "*", "+", "•")):
            return before + newline + after
        
        # Otherwise, replace newline with space
        # Иначе заменяем перевод строки на пробел
        return before + " " + after.lstrip()

    # Step 2: Process single newlines
    # Шаг 2: Обрабатываем одиночные переводы строк
    text = re.sub(r"([^\n])(\n)([^\n])", _join_non_structural, text)
    return text


def _normalize_dashes_and_spaces(text: str) -> str:
    """
    Normalize hyphen vs em-dash and collapse spaces around punctuation.
    Нормализует дефис и тире, убирает лишние пробелы вокруг пунктуации.

    Operations / Операции:
    1. Inside words: remove spaces around hyphens / В словах: убирает пробелы вокруг дефисов
    2. Between tokens: convert ' - ' to ' — ' / Между токенами: конвертирует ' - ' в ' — '
    3. Collapse excessive spaces / Убирает избыточные пробелы
    4. Trim space before punctuation / Убирает пробелы перед знаками препинания
    
    Parameters / Параметры:
        text: Input text / Входной текст
        
    Returns / Возвращает:
        Normalized text / Нормализованный текст
    """
    # Step 1: Hyphen inside words (avoid touching bullet lists)
    # Remove spaces around hyphens when between word characters
    # Шаг 1: Дефис внутри слов (не трогаем маркированные списки)
    # Убираем пробелы вокруг дефисов между буквами
    text = re.sub(r"(?<=\w)\s*-\s*(?=\w)", "-", text)
    
    # Step 2: Em-dash between tokens (not at line start)
    # Convert single hyphen surrounded by spaces to em-dash
    # Шаг 2: Тире между токенами (не в начале строки)
    # Конвертируем одиночный дефис с пробелами в тире
    text = re.sub(r"(?m)(?<!^)\s-\s(?!-)", " — ", text)
    
    # Step 3: Remove extra spaces before punctuation marks
    # Шаг 3: Убираем лишние пробелы перед знаками препинания
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    
    # Step 4: Normalize multiple spaces to single space
    # Шаг 4: Нормализуем множественные пробелы в одиночные
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


# Homoglyph translation table: Latin → Cyrillic
# Таблица перевода омографов: латиница → кириллица
# These Latin letters look identical to Cyrillic letters and are often confused by OCR
# Эти латинские буквы выглядят идентично кириллическим и часто путаются OCR
_LAT_TO_CYR = str.maketrans({
    "A": "А",  # Latin A → Cyrillic А
    "a": "а",
    "B": "В",  # Latin B → Cyrillic В
    "C": "С",  # Latin C → Cyrillic С
    "c": "с",
    "E": "Е",  # Latin E → Cyrillic Е
    "e": "е",
    "H": "Н",  # Latin H → Cyrillic Н
    "K": "К",  # Latin K → Cyrillic К
    "k": "к",
    "M": "М",  # Latin M → Cyrillic М
    "m": "м",
    "O": "О",  # Latin O → Cyrillic О
    "o": "о",
    "P": "Р",  # Latin P → Cyrillic Р
    "p": "р",
    "T": "Т",  # Latin T → Cyrillic Т
    "t": "т",
    "X": "Х",  # Latin X → Cyrillic Х
    "x": "х",
    "Y": "У",  # Latin Y → Cyrillic У
    "y": "у",
    "h": "н",  # Latin h → Cyrillic н
})

# Regex patterns for detecting noise lines
# Регулярные выражения для обнаружения шумовых строк

# Short lines with only punctuation/digits: "...", "123", etc.
# Короткие строки только с пунктуацией/цифрами: "...", "123" и т.д.
_NOISE_SHORT_LINE_RE = re.compile(r"^[\W\d_]{1,3}$")

# Source file references: "@filename.md (2-11)"
# Ссылки на исходные файлы: "@filename.md (2-11)"
_NOISE_SOURCE_LINE_RE = re.compile(r"^@[A-Za-z0-9_.-]+\s*\(\d+(?:-\d+)?\)$")

# HTML comments: "<!-- comment -->"
# HTML комментарии: "<!-- комментарий -->"
_NOISE_HTML_COMMENT_RE = re.compile(r"^<!--.*-->$")

# Short letter sequences (1-3 chars), but keep common words like "да", "нет", "yes", "no"
# Короткие буквенные последовательности (1-3 символа), но сохраняем обычные слова типа "да", "нет", "yes", "no"
_NOISE_SHORT_LETTERS_RE = re.compile(
    r"^(?!(?:да|нет|yes|no)$)[A-Za-zА-Яа-яЁё]{1,3}$", re.IGNORECASE
)


def _drop_noise_lines(text: str) -> str:
    """
    Remove stray OCR artefacts like isolated punctuation or source markers.
    Удаляет случайные артефакты OCR, такие как изолированная пунктуация или маркеры источников.
    
    This function removes:
    Эта функция удаляет:
    - HTML comments / HTML комментарии
    - Incomplete HTML tags / Неполные HTML теги
    - Lines with only punctuation/digits / Строки только с пунктуацией/цифрами
    - Source file markers like '@file (pages)' / Маркеры исходных файлов типа '@файл (страницы)'
    - Very short letter sequences (likely OCR noise) / Очень короткие буквенные последовательности (вероятно, шум OCR)
    
    Parameters / Параметры:
        text: Input text / Входной текст
        
    Returns / Возвращает:
        Cleaned text / Очищенный текст
    """
    lines = text.splitlines()
    cleaned: List[str] = []
    
    for line in lines:
        stripped = line.strip()
        
        # Keep blank lines / Сохраняем пустые строки
        if not stripped:
            cleaned.append(line)
            continue
        
        # Drop HTML comments / Удаляем HTML комментарии
        if _NOISE_HTML_COMMENT_RE.match(stripped):
            continue
        
        # Drop incomplete HTML tags / Удаляем неполные HTML теги
        if stripped.startswith("<") and ">" not in stripped:
            continue
        
        # If line has no letters, check if it's just punctuation/digits
        # Если в строке нет букв, проверяем, не является ли она просто пунктуацией/цифрами
        if not re.search(r"[A-Za-zА-Яа-яЁё]", stripped):
            if _NOISE_SHORT_LINE_RE.match(stripped):
                continue
        
        # Drop source file markers / Удаляем маркеры исходных файлов
        if _NOISE_SOURCE_LINE_RE.match(stripped):
            continue
        
        # Drop very short letter sequences (but keep common words)
        # Удаляем очень короткие буквенные последовательности (но сохраняем обычные слова)
        if _NOISE_SHORT_LETTERS_RE.match(stripped):
            continue
        
        # Keep this line / Сохраняем эту строку
        cleaned.append(line)
    
    return "\n".join(cleaned)


# Regex patterns for analyzing suspicious paragraphs
# Регулярные выражения для анализа подозрительных параграфов

# Matches words (sequences of letters and apostrophes)
# Совпадает со словами (последовательности букв и апострофов)
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё']+")

# Matches Latin letters / Совпадает с латинскими буквами
_LAT_RE = re.compile(r"[A-Za-z]")

# Matches Cyrillic letters / Совпадает с кириллическими буквами
_CYR_RE = re.compile(r"[А-Яа-яЁё]")

# Matches words containing BOTH Latin AND Cyrillic (mixed-script words)
# These are almost always OCR errors
# Совпадает со словами, содержащими И латиницу, И кириллицу (слова смешанного алфавита)
# Это почти всегда ошибки OCR
_MIXED_SCRIPT_WORD_RE = re.compile(r"(?=.*[A-Za-z])(?=.*[А-Яа-яЁё])")

# Set of allowed symbols in clean text
# Набор допустимых символов в чистом тексте
_ALLOWED_SYMBOLS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"  # Latin / Латиница
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"  # Cyrillic uppercase / Кириллица заглавная
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"  # Cyrillic lowercase / Кириллица строчная
    "0123456789"  # Digits / Цифры
    " .,:;!?\"'«»„""()[]{}<>-–—/%&+#°•…"  # Punctuation / Пунктуация
)


def _evaluate_paragraph(
    lines: List[str],
    start_idx: int,
    end_idx: int,
    *,
    min_score: float,
    dominant_script: str | None = None,
) -> Optional[SuspiciousParagraph]:
    """
    Evaluate a paragraph for OCR errors and return suspicion score.
    Оценивает параграф на наличие ошибок OCR и возвращает оценку подозрительности.
    
    This function analyzes text for common OCR problems:
    Эта функция анализирует текст на распространенные проблемы OCR:
    - Mixed-script words (Latin+Cyrillic in one word) / Слова со смешанным алфавитом
    - Unknown Cyrillic words (not in dictionary) / Неизвестные кириллические слова
    - Weird characters / Странные символы
    - Excessive Latin in Russian text / Избыточная латиница в русском тексте
    
    Parameters / Параметры:
        lines: Paragraph lines / Строки параграфа
        start_idx: Starting line number / Номер начальной строки
        end_idx: Ending line number / Номер конечной строки
        min_score: Minimum score to flag as suspicious / Минимальная оценка для пометки как подозрительного
        dominant_script: Expected script ('cyr', 'lat', or None for auto-detect)
                         Ожидаемый алфавит ('cyr', 'lat', или None для автоопределения)
    
    Returns / Возвращает:
        SuspiciousParagraph if suspicious, None otherwise
        SuspiciousParagraph если подозрителен, None в противном случае
    """
    # Step 1: Join lines and validate / Шаг 1: Соединяем строки и валидируем
    text = "\n".join(lines).strip()
    if not text:
        return None
    
    # Skip headings (they often have unusual capitalization)
    # Пропускаем заголовки (у них часто необычная капитализация)
    first_line = lines[0].lstrip()
    if first_line.startswith("#"):
        return None

    # Extract words / Извлекаем слова
    words = _WORD_RE.findall(text)
    if not words:
        return None

    # Step 2: Determine dominant script (Cyrillic vs Latin)
    # Шаг 2: Определяем доминирующий алфавит (кириллица vs латиница)
    cyr_chars = sum(1 for ch in text if _CYR_RE.match(ch))
    lat_chars = sum(1 for ch in text if _LAT_RE.match(ch))
    if dominant_script is None:
        # Auto-detect based on character counts / Автоопределение на основе подсчета символов
        if cyr_chars >= lat_chars:
            dominant_script = "cyr"
        elif lat_chars:
            dominant_script = "lat"
        else:
            dominant_script = "unknown"

    # Step 3: Handle very short fragments (≤10 chars)
    # Шаг 3: Обрабатываем очень короткие фрагменты (≤10 символов)
    if len(text) <= 10:
        reasons = ["short-fragment"]
        if _LAT_RE.search(text):
            reasons.append("latin-only")
        return SuspiciousParagraph(
            text=text,
            score=0.6 if "latin-only" in reasons else 0.3,
            reasons=sorted(reasons),
            start_line=start_idx + 1,
            end_line=end_idx + 1,
        )

    # Step 4: Flag HTML comments as suspicious
    # Шаг 4: Помечаем HTML комментарии как подозрительные
    if _NOISE_HTML_COMMENT_RE.match(text):
        return SuspiciousParagraph(
            text=text,
            score=0.7,
            reasons=["html-comment"],
            start_line=start_idx + 1,
            end_line=end_idx + 1,
        )

    # Step 5: Analyze individual words
    # Шаг 5: Анализируем отдельные слова
    total_words = len(words)
    suspicious_words = 0  # Count of suspicious words / Количество подозрительных слов
    latin_words = 0  # Words containing Latin / Слова, содержащие латиницу
    cyrillic_words = 0  # Words containing Cyrillic / Слова, содержащие кириллицу
    latin_only_words = 0  # Words with only Latin / Слова только с латиницей
    reasons: set[str] = set()  # Reasons for suspicion / Причины подозрительности
    extra_component = 0.0  # Extra score for severe issues / Дополнительная оценка за серьёзные проблемы
    has_mixed_token = False  # Flag for mixed-script words / Флаг для слов смешанного алфавита

    for raw_word in words:
        # Strip apostrophes from word edges / Убираем апострофы с краёв слов
        word = raw_word.strip("'")
        if not word:
            continue
        
        # Check what scripts the word contains / Проверяем, какие алфавиты содержит слово
        has_lat = bool(_LAT_RE.search(word))
        has_cyr = bool(_CYR_RE.search(word))
        if has_lat:
            latin_words += 1
        if has_cyr:
            cyrillic_words += 1

        word_flag = False  # Flag if this specific word is suspicious / Флаг, если конкретное слово подозрительно

        # Check 1: Mixed-script word (Latin + Cyrillic in same word) - ALWAYS suspicious
        # Проверка 1: Слово смешанного алфавита (латиница + кириллица в одном слове) - ВСЕГДА подозрительно
        if _MIXED_SCRIPT_WORD_RE.search(word):
            word_flag = True
            reasons.add("mixed-script")
            has_mixed_token = True

        # Check 2: Mixed case (multiple uppercase letters, but not all caps)
        # Example: "MiXeD" is suspicious
        # Проверка 2: Смешанный регистр (несколько заглавных букв, но не все)
        # Пример: "ПеРеМеШаН" подозрительно
        upper_count = sum(1 for ch in word if ch.isupper())
        if upper_count >= 2 and not word.isupper():
            word_flag = True
            reasons.add("mixed-case")

        # Check 3: Unknown Cyrillic word (if morphology analyzer available)
        # Long Cyrillic words (≥5 chars) not in dictionary are suspicious
        # Проверка 3: Неизвестное кириллическое слово (если доступен морфологический анализатор)
        # Длинные кириллические слова (≥5 символов), не найденные в словаре, подозрительны
        if has_cyr and len(word) >= 5 and _MORPH_ANALYZER is not None:
            lower = word.lower()
            if not _is_known_word(lower):
                word_flag = True
                reasons.add("unknown-cyr")

        # Check 4: Latin words in predominantly Cyrillic text
        # Проверка 4: Латинские слова в преимущественно кириллическом тексте
        if has_lat and not has_cyr:
            latin_only_words += 1
        if has_lat and not has_cyr and dominant_script == "cyr":
            # Flag Latin words in Cyrillic context (unless they're abbreviations in uppercase)
            # Помечаем латинские слова в кириллическом контексте (если только это не аббревиатуры в верхнем регистре)
            if len(word) >= 4 or (len(word) <= 3 and cyrillic_words >= 3):
                if not word.isupper():  # Allow uppercase abbreviations / Разрешаем аббревиатуры в верхнем регистре
                    word_flag = True
                    reasons.add("latin-in-cyr")
                    extra_component = max(extra_component, 0.5)

        if word_flag:
            suspicious_words += 1

    # Step 6: Analyze character-level anomalies
    # Шаг 6: Анализируем аномалии на уровне символов
    total_chars = sum(1 for ch in text if not ch.isspace())
    weird_chars = sum(
        1 for ch in text if not (ch.isspace() or ch in _ALLOWED_SYMBOLS)
    )
    weird_ratio = (weird_chars / total_chars) if total_chars else 0.0
    char_component = 0.0
    # If >5% of characters are weird, add to suspicion score
    # Если >5% символов странные, добавляем к оценке подозрительности
    if weird_ratio > 0.05:
        reasons.add("weird-chars")
        char_component = min(0.4, (weird_ratio - 0.05) * 4.0)

    # Step 7: Check Latin-to-Cyrillic ratio in mixed text
    # Шаг 7: Проверяем соотношение латиницы к кириллице в смешанном тексте
    latin_component = 0.0
    if cyrillic_words >= 4 and latin_words > 0:
        latin_ratio = latin_words / (latin_words + cyrillic_words)
        # If >20% of words are Latin in a Cyrillic text, that's suspicious
        # Если >20% слов - латинские в кириллическом тексте, это подозрительно
        if latin_ratio > 0.2:
            reasons.add("latin-ratio")
            latin_component = min(0.4, (latin_ratio - 0.2) * 2.0)

    # Step 8: Penalize Latin-only words in Cyrillic context
    # Шаг 8: Штрафуем слова только на латинице в кириллическом контексте
    if dominant_script == "cyr" and latin_only_words > 0:
        reasons.add("latin-only")
        extra_component = max(extra_component, min(0.6, 0.4 + 0.05 * latin_only_words))

    # Step 9: Calculate final suspicion score (0.0 to 1.0)
    # Шаг 9: Вычисляем итоговую оценку подозрительности (0.0 до 1.0)
    word_ratio = suspicious_words / total_words if total_words else 0.0
    score = min(1.0, word_ratio + char_component + latin_component + extra_component)

    # Step 10: Apply binary rule for mixed-script words
    # BINARY RULE: If there's a mixed-script token (Cyrillic+Latin in one word),
    # return the paragraph regardless of min_score threshold.
    # Шаг 10: Применяем бинарное правило для слов смешанного алфавита
    # БИНАРНОЕ ПРАВИЛО: Если есть токен смешанного алфавита (кириллица+латиница в одном слове),
    # возвращаем параграф независимо от порога min_score.
    candidate = SuspiciousParagraph(
        text=text,
        score=score,
        reasons=sorted(reasons),
        start_line=start_idx + 1,
        end_line=end_idx + 1,
    )
    if has_mixed_token:
        return candidate

    # Step 11: Filter by minimum score
    # Шаг 11: Фильтруем по минимальной оценке
    if score < min_score or not reasons:
        return None

    return candidate


def find_suspicious_paragraphs(
    text: str,
    *,
    min_score: float = 0.45,
    max_results: Optional[int] = None,
) -> List[SuspiciousParagraph]:
    """
    Scan Markdown text and flag paragraphs likely containing OCR artefacts.
    Сканирует текст Markdown и помечает параграфы, вероятно содержащие артефакты OCR.
    
    This function splits text into paragraphs (blank-line separated) and evaluates
    each one for OCR errors. Results are sorted by priority (most severe first).
    
    Эта функция разбивает текст на параграфы (разделённые пустыми строками) и оценивает
    каждый на наличие ошибок OCR. Результаты сортируются по приоритету (самые серьёзные первыми).
    
    Parameters / Параметры:
        text: Input Markdown text / Входной текст Markdown
        min_score: Minimum suspicion score (0.0-1.0) to include / Минимальная оценка подозрительности для включения
        max_results: Maximum number of results to return / Максимальное количество результатов для возврата
        
    Returns / Возвращает:
        List of suspicious paragraphs, sorted by priority / Список подозрительных параграфов, отсортированных по приоритету
    """
    lines = text.splitlines()
    results: List[SuspiciousParagraph] = []
    buffer: List[str] = []  # Accumulator for current paragraph / Аккумулятор для текущего параграфа
    start_idx = 0  # Starting line of current paragraph / Начальная строка текущего параграфа

    # Step 1: Split text into paragraphs and evaluate each one
    # Шаг 1: Разбиваем текст на параграфы и оцениваем каждый
    for idx, line in enumerate(lines):
        if line.strip():  # Non-empty line / Непустая строка
            if not buffer:
                start_idx = idx
            buffer.append(line)
        else:  # Empty line marks paragraph boundary / Пустая строка обозначает границу параграфа
            if buffer:
                cand = _evaluate_paragraph(buffer, start_idx, idx - 1, min_score=min_score)
                if cand is not None:
                    results.append(cand)
                buffer = []

    # Process last paragraph if exists / Обрабатываем последний параграф, если он есть
    if buffer:
        cand = _evaluate_paragraph(buffer, start_idx, len(lines) - 1, min_score=min_score)
        if cand is not None:
            results.append(cand)

    # Step 2: Sort results by priority (most severe issues first)
    # Шаг 2: Сортируем результаты по приоритету (самые серьёзные проблемы первыми)
    def _priority(p: SuspiciousParagraph) -> Tuple[int, float, int]:
        """
        Calculate sort priority for a suspicious paragraph.
        Вычисляет приоритет сортировки для подозрительного параграфа.
        
        Returns (priority_level, -score, line_number)
        Priority levels: 0=highest, 4=lowest
        
        Возвращает (уровень_приоритета, -оценка, номер_строки)
        Уровни приоритета: 0=наивысший, 4=наименьший
        """
        reasons = set(p.reasons)
        if "latin-only" in reasons or "latin-in-cyr" in reasons:
            priority = 0  # Highest: Latin in Cyrillic / Наивысший: латиница в кириллице
        elif "mixed-script" in reasons or "latin-ratio" in reasons:
            priority = 1  # High: Mixed scripts / Высокий: смешанные алфавиты
        elif "unknown-cyr" in reasons or "weird-chars" in reasons:
            priority = 2  # Medium: Unknown words or weird chars / Средний: неизвестные слова или странные символы
        elif "short-fragment" in reasons:
            priority = 4  # Lowest: Just short fragments / Наименьший: просто короткие фрагменты
        else:
            priority = 3  # Low: Other issues / Низкий: другие проблемы
        return (priority, -p.score, p.start_line)

    results.sort(key=_priority)
    
    # Step 3: Limit results if requested / Шаг 3: Ограничиваем результаты, если запрошено
    if max_results is not None:
        results = results[:max_results]
    return results


def _fix_homoglyphs_in_russian_context(text: str) -> str:
    """Translate Latin homoglyphs to Cyrillic only inside tokens that already contain Cyrillic.

    This avoids corrupting pure English words while fixing cases like "C1e" etc.
    """
    def repl(m: re.Match[str]) -> str:
        token = m.group(0)
        if re.search(r"[А-Яа-яЁё]", token):
            return token.translate(_LAT_TO_CYR)
        return token

    return re.sub(r"[A-Za-zА-Яа-яЁё\-]+", repl, text)


def _is_russian_dominant_line(line: str) -> bool:
    cyr = sum(1 for ch in line if re.match(r"[А-Яа-яЁё]", ch))
    lat = sum(1 for ch in line if re.match(r"[A-Za-z]", ch))
    return cyr > 0 and cyr >= lat


# Set of Latin letters that can be safely translated to Cyrillic (homoglyphs)
# Набор латинских букв, которые можно безопасно перевести в кириллицу (омографы)
_SAFE_LAT_KEYS = set(chr(k) for k in _LAT_TO_CYR.keys())

# Pattern for short Latin words (1-5 characters)
# Шаблон для коротких латинских слов (1-5 символов)
_SHORT_LATIN_RE = re.compile(r"\b[A-Za-z]{1,5}\b")

# Whitelist of short Latin sequences that should be converted to Cyrillic
# when found in Russian text (without morphology checker)
# Белый список коротких латинских последовательностей для конвертации в кириллицу
# при нахождении в русском тексте (без морфологического анализатора)
_SHORT_LATIN_WHITELIST = {
    "KAK", "NA", "HE", "HA", "TOM", "ACT", "KTOMY", "KTOMU", "C",
}


def _fix_short_latin_words_in_russian_lines(text: str) -> str:
    """
    Replace short Latin-only tokens in Russian-dominant lines.
    Заменяет короткие латинские токены в русскоязычных строках.
    
    This function fixes OCR errors where short Russian words are misrecognized
    as Latin. For example, "Kak" → "Как", "NA" → "НА".
    
    Эта функция исправляет ошибки OCR, где короткие русские слова распознаются
    как латинские. Например, "Kak" → "Как", "NA" → "НА".
    
    Criteria / Критерии:
    - Length ≤ 5 / Длина ≤ 5
    - Token letters are all safely mappable (subset of _LAT_TO_CYR keys)
      Все буквы токена безопасно переводятся (подмножество ключей _LAT_TO_CYR)
    - If pymorphy2 available: replace only if candidate is a known word
      Если доступен pymorphy2: заменяем только если кандидат - известное слово
    - Otherwise: restrict to a small whitelist (Kak, Na, He, Ha, TOM, ACT, KTOMY/KTOMU, C)
      Иначе: ограничиваемся небольшим белым списком
    """
    lines = text.splitlines()

    def replace_token(m: re.Match[str]) -> str:
        """
        Callback to replace a single Latin token if appropriate.
        Колбэк для замены одного латинского токена, если это уместно.
        """
        token = m.group(0)
        
        # Fast path: ensure token consists only of safe-mappable letters
        # Быстрый путь: убеждаемся, что токен состоит только из безопасно переводимых букв
        if any(ch not in _SAFE_LAT_KEYS for ch in token):
            return token
        
        # Translate to Cyrillic and convert to lowercase for dictionary check
        # Переводим в кириллицу и в нижний регистр для проверки по словарю
        candidate_lower = token.translate(_LAT_TO_CYR).lower()
        allow = False
        
        # Strategy 1: If morphology analyzer available, check if it's a known Russian word
        # Стратегия 1: Если доступен морфологический анализатор, проверяем, известное ли это русское слово
        if _MORPH_ANALYZER is not None and len(candidate_lower) >= 2:
            allow = _is_known_word(candidate_lower)
        # Strategy 2: Use whitelist for common short words
        # Стратегия 2: Используем белый список для обычных коротких слов
        else:
            allow = token.upper() in _SHORT_LATIN_WHITELIST
        
        if not allow:
            return token
        
        # Apply original casing pattern (preserve uppercase/lowercase/title case)
        # Применяем исходный шаблон регистра (сохраняем верхний/нижний/заглавный регистр)
        replaced = _apply_case(token, candidate_lower)
        return replaced

    # Apply replacement only to Russian-dominant lines
    # Применяем замену только к строкам с доминирующей кириллицей
    for i, line in enumerate(lines):
        if _is_russian_dominant_line(line):
            lines[i] = _SHORT_LATIN_RE.sub(replace_token, line)
    return "\n".join(lines)


def _drop_f_prefix_before_rus_vowel(text: str) -> str:
    """
    Remove stray Latin 'f' preceding a Russian vowel at word start.
    Удаляет случайную латинскую 'f' перед русской гласной в начале слова.
    
    OCR sometimes misrecognizes Russian lowercase б/в as Latin 'f'.
    OCR иногда неправильно распознает русские строчные б/в как латинскую 'f'.
    
    Example / Пример: 'fа' → 'а', 'fизика' → 'изика'
    """
    return re.sub(r"\bf(?=[аеиоуыэюя])", "", text, flags=re.IGNORECASE)


# Pattern for alphanumeric tokens (letters, digits, hyphens)
# Шаблон для алфавитно-цифровых токенов (буквы, цифры, дефисы)
_TOKEN_ALNUM_RE = re.compile(r"\b[0-9A-Za-zА-Яа-яЁё\-]+\b")

# Patterns for dice notation in D&D (e.g., "2к6", "1d20+5")
# These should NOT be modified by digit replacement
# Шаблоны для обозначения кубиков в D&D (например, "2к6", "1d20+5")
# Они НЕ должны изменяться при замене цифр
_RE_DICE_PATTERNS = [
    re.compile(r"(?i)^\d+[кk]\d+(?:\+\d+)?$"),  # "2к6", "1K4+3"
    re.compile(r"(?i)^[кk]\d+(?:\+\d+)?$"),     # "к10", "K6+2"
    re.compile(r"(?i)^\d+d\d+(?:\+\d+)?$"),     # "2d6", "1d20+5"
]


def _fix_digits_in_russian_words(text: str) -> str:
    """
    Replace digits misrecognized as Cyrillic letters inside Russian words.
    Заменяет цифры, ошибочно распознанные как кириллические буквы в русских словах.
    
    OCR often confuses digits with Cyrillic letters due to visual similarity.
    OCR часто путает цифры с кириллическими буквами из-за визуального сходства.
    
    Rules / Правила:
    - Only in Russian-dominant lines / Только в строках с доминирующей кириллицей
    - Skip fenced code lines and Markdown tables / Пропускаем код-блоки и строки таблиц
    - Skip dice notations like '1к4', 'к10', '2d6' / Пропускаем обозначения кубиков типа '1к4', 'к10', '2d6'
    - Operate only on tokens that contain at least one Cyrillic letter
      Обрабатываем только токены, содержащие хотя бы одну кириллическую букву
    - Replace: 3→З/з, 6→Б/б, 0→О/о only if the digit is adjacent to a Cyrillic
      letter and neither neighbor is a digit (avoid '20кг'→'2окг')
      Заменяем 3/6/0 только если рядом есть кириллическая буква и ни один сосед не цифра
    - Replace 4→й/Й only at the end of a token, when preceded by a Cyrillic letter
      and the token has at least two Cyrillic letters (avoid 'к4'/'в4')
      Заменяем 4→й/Й только в конце токена, слева кириллическая буква и в токене ≥2 кириллических
    - After digit replacement, also fix remaining Latin homoglyphs
      После замены цифр также исправляем оставшиеся латинские омографы
    
    Examples / Примеры:
    - "б0льше" → "больше" (0→о)
    - "пре3ид" → "презид" (3→з)
    - "зада4" → "задай" (4→й at end / 4→й в конце)
    """
    def is_dice_token(tok: str) -> bool:
        return any(p.match(tok) for p in _RE_DICE_PATTERNS)

    def choose_case_for_digit(idx: int, token: str, line: str, span_start: int) -> str:
        # Uppercase if the whole token is uppercase
        if token.isupper():
            return "upper"
        # Heuristic: uppercase at sentence start (preceded by start or [.?!:;—(«] and optional space)
        if idx == 0:
            j = span_start - 1
            while j >= 0 and line[j] == " ":
                j -= 1
            if j < 0 or line[j] in ".?!:;—(\"«":
                return "upper"
        return "lower"

    def replace_in_token(m: re.Match[str]) -> str:
        tok = m.group(0)
        # Skip tokens that are dice notations or contain no digits at all
        if is_dice_token(tok) or not any(ch.isdigit() for ch in tok):
            return tok
        # Process only tokens that contain at least one Cyrillic letter
        if not _CYR_RE.search(tok):
            return tok
        out_chars: List[str] = []
        start = m.start()
        for idx, ch in enumerate(tok):
            repl = None
            mode = None
            if ch in ("3", "6", "0"):
                left = tok[idx - 1] if idx > 0 else ""
                right = tok[idx + 1] if idx + 1 < len(tok) else ""
                left_is_digit = left.isdigit() if left else False
                right_is_digit = right.isdigit() if right else False
                left_is_cyr = bool(_CYR_RE.match(left)) if left else False
                right_is_cyr = bool(_CYR_RE.match(right)) if right else False
                # Replace only if not inside a numeric run and adjacent to Cyrillic
                if (not left_is_digit and not right_is_digit) and (left_is_cyr or right_is_cyr):
                    mode = choose_case_for_digit(idx, tok, line_ref, start)
                    if ch == "3":
                        repl = "З" if mode == "upper" else "з"
                    elif ch == "6":
                        repl = "Б" if mode == "upper" else "б"
                    elif ch == "0":
                        repl = "О" if mode == "upper" else "о"
            elif ch == "4" and idx == len(tok) - 1:
                left = tok[idx - 1] if idx > 0 else ""
                # Preceded by Cyrillic, not a digit; token has at least two Cyrillic letters
                cyr_count = sum(1 for c in tok if _CYR_RE.match(c))
                if left and not left.isdigit() and _CYR_RE.match(left) and cyr_count >= 2:
                    mode = choose_case_for_digit(idx, tok, line_ref, start)
                    repl = "Й" if mode == "upper" else "й"
            out_chars.append(repl if repl is not None else ch)
        replaced = "".join(out_chars)
        # After digit fix, convert safe Latin homoglyphs inside the same token
        replaced = re.sub(r"[A-Za-zА-Яа-яЁё\-]+", lambda mm: mm.group(0).translate(_LAT_TO_CYR), replaced)
        return replaced

    lines = text.splitlines()
    in_fence = False
    for i, line_ref in enumerate(lines):
        # Toggle fenced code regions and skip their lines
        if _FENCE_RE.match(line_ref):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # Skip obvious Markdown table rows (multiple pipes)
        if line_ref.count("|") >= 2:
            continue
        if not _is_russian_dominant_line(line_ref):
            continue
        lines[i] = _TOKEN_ALNUM_RE.sub(replace_in_token, line_ref)
    return "\n".join(lines)

def _dedupe_repeated_words(text: str) -> str:
    """
    Collapse duplicated words produced after merges.
    Схлопывает дублированные слова, образованные после слияний.
    
    OCR corrections can sometimes create duplicates when joining split words.
    Исправления OCR иногда создают дубликаты при соединении разбитых слов.
    
    Example / Пример: 'персонажи персонажи' → 'персонажи'
    """
    pattern = re.compile(r"\b(\w{4,})\b(?:[ \t]+)\1\b", flags=re.IGNORECASE)
    prev = None
    # Repeat until no more duplicates found / Повторяем, пока дубликаты не исчезнут
    while prev != text:
        prev = text
        text = pattern.sub(r"\1", text)
    return text


# Russian alphabet for drop cap restoration / Русский алфавит для восстановления буквиц
_RUS_INITIALS = tuple("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
# Latin alphabet for drop cap restoration / Латинский алфавит для восстановления буквиц
_LAT_INITIALS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
# Pattern to detect potential drop caps (missing first letter)
# Шаблон для обнаружения потенциальных буквиц (отсутствующая первая буква)
_DROP_CAP_RE = re.compile(r"(?m)^(?P<prefix>[#>\-\*\s]{0,5})(?P<word>[A-Za-zА-Яа-яЁё]{2,})(?=\b)")
# Patterns for Cyrillic and Latin characters
# Шаблоны для кириллических и латинских символов
_RE_CYR = re.compile(r"[А-Яа-яЁё]")
_RE_LAT = re.compile(r"[A-Za-z]")


@lru_cache(maxsize=8192)
def _is_known_word(word: str) -> bool:
    """
    Check if a word exists in the Russian morphology dictionary.
    Проверяет, существует ли слово в словаре русской морфологии.
    
    Uses LRU cache for performance (caches up to 8192 words).
    Использует LRU кэш для производительности (кэширует до 8192 слов).
    
    Parameters / Параметры:
        word: Word to check / Слово для проверки
        
    Returns / Возвращает:
        True if word is known / True если слово известно
    """
    if not word or _MORPH_ANALYZER is None:
        return False
    try:
        return any(parse.is_known for parse in _MORPH_ANALYZER.parse(word))
    except Exception:  # pragma: no cover - защищаемся от неожиданных ошибок морфологии
        return False


def _apply_case(template: str, replacement: str) -> str:
    """
    Apply the casing pattern from template to replacement string.
    Применяет схему регистра из шаблона к строке замены.
    
    Examples / Примеры:
    - template="ABC", replacement="xyz" → "XYZ" (all uppercase / весь верхний регистр)
    - template="abc", replacement="XYZ" → "xyz" (all lowercase / весь нижний регистр)
    - template="Abc", replacement="xyz" → "Xyz" (title case / заглавная буква)
    
    Parameters / Параметры:
        template: String with desired casing / Строка с желаемым регистром
        replacement: String to apply casing to / Строка для применения регистра
        
    Returns / Возвращает:
        Replacement with applied casing / Замена с применённым регистром
    """
    if not replacement:
        return replacement
    if template.isupper():
        return replacement.upper()
    if template.islower():
        return replacement.lower()
    if template[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _restore_dropcap_first_letter(text: str) -> str:
    """
    Attempt to restore drop caps (missing first letters) using morphology.
    Попытка восстановить буквицы (отсутствующие первые буквы) через морфологию.
    
    Drop caps are decorative first letters that OCR sometimes fails to recognize.
    This function tries to prepend each letter of the alphabet and checks if
    the result is a known word.
    
    Буквицы - это декоративные первые буквы, которые OCR иногда не распознаёт.
    Эта функция пытается добавить каждую букву алфавита и проверяет, является ли
    результат известным словом.
    
    Example / Пример: "ерсонаж" → "Персонаж" (restored 'П' / восстановлена 'П')
    """

    if _MORPH_ANALYZER is None:
        return text

    def _replace(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        word = match.group("word")

        if len(word) < 2:
            return match.group(0)

        if _is_known_word(word) or _is_known_word(word.lower()):
            return match.group(0)

        if _RE_CYR.search(word):
            alphabet = _RUS_INITIALS
        elif _RE_LAT.search(word):
            alphabet = _LAT_INITIALS
        else:
            return match.group(0)

        lower_rest = word.lower()
        for letter in alphabet:
            candidate_lower = (letter + lower_rest).lower()
            if _is_known_word(candidate_lower):
                candidate = _apply_case(word, candidate_lower)
                return prefix + candidate

        return match.group(0)

    return _DROP_CAP_RE.sub(_replace, text)


def deep_normalize_markdown(md: str) -> str:
    """
    Robust normalization for Docling/OCR Markdown with comprehensive error fixing.
    Надёжная нормализация Markdown из Docling/OCR с комплексным исправлением ошибок.
    
    This is the main normalization pipeline that applies all OCR fixes in sequence.
    Each step builds on the previous ones to progressively clean the text.
    
    Это главный пайплайн нормализации, который последовательно применяет все исправления OCR.
    Каждый шаг основывается на предыдущих для постепенной очистки текста.

    Combined normalization steps:
    Объединённые шаги нормализации:
    
    1) HTML entities → Unicode; `/uniXXXX` → символ
       Decode HTML entities and Unicode escapes / Декодируем HTML-сущности и Unicode escape-последовательности
    2) Базовая нормализация переносов и пустых строк
       Basic normalization of line breaks and blank lines / Базовая нормализация переносов строк и пустых строк
    3) Удаление шумовых строк (`@DMG.md (2-11)`, одиночные символы)
       Remove noise lines like source markers / Удаляем шумовые строки типа маркеров источников
    4) Склейка "П оследующие" / "D UNGEONS"
       Join split words like "P ostfix" → "Postfix" / Склеиваем разбитые слова
    5) Починка переносов и внутрисловных переводов строк
       Fix hyphenation and mid-word line breaks / Исправляем переносы и разрывы строк внутри слов
    6) Нормализация дефиса/тире и пробелов вокруг знаков препинания
       Normalize hyphens/dashes and spaces around punctuation / Нормализуем дефисы/тире и пробелы
    7) Попытка восстановить буквицы через pymorphy2 (если библиотека доступна)
       Restore drop caps using morphology / Восстанавливаем буквицы через морфологию
    8) Латиница→кириллица для омографов в русском окружении
       Latin→Cyrillic for homoglyphs in Russian context / Латиница→кириллица для омографов в русском контексте
    9) Короткие латинские слова в русских строках (безопасная замена/морфология)
       Short Latin words in Russian lines / Короткие латинские слова в русских строках
    10) Замена цифр в русских словах (3/6/0; 4 на конце)
        Replace digits misread as Cyrillic letters / Заменяем цифры, прочитанные как кириллические буквы
    11) Чистка артефакта 'f' перед русской гласной
        Clean 'f' artifact before Russian vowels / Очищаем артефакт 'f' перед русскими гласными
    12) Удаление сдвоенных слов
        Remove duplicated words / Удаляем сдвоенные слова
    13) Unicode NFC и уплотнение пробелов
        Unicode NFC normalization and space cleanup / Unicode NFC нормализация и очистка пробелов
    
    Parameters / Параметры:
        md: Input Markdown text from OCR / Входной текст Markdown из OCR
        
    Returns / Возвращает:
        Deeply normalized text / Глубоко нормализованный текст
    """
    if not md:
        return md

    # Step 1: Unify line endings / Шаг 1: Унифицируем окончания строк
    text = md.replace("\r\n", "\n").replace("\r", "\n")
    
    # Step 2: Decode HTML entities and Unicode escapes
    # Шаг 2: Декодируем HTML-сущности и Unicode escape-последовательности
    text = _html_unescape(text)
    text = _decode_uni_escapes(text)
    
    # Step 3: Base lightweight fixes first
    # Шаг 3: Сначала базовые лёгкие исправления
    text = normalize_markdown(text)
    
    # Steps 4-12: Deep OCR fixes (applied in specific order for best results)
    # Шаги 4-12: Глубокие исправления OCR (применяются в определённом порядке для лучших результатов)
    text = _drop_noise_lines(text)
    text = _join_single_letter_splits(text)
    text = _fix_hyphenation_and_linebreaks(text)
    text = _normalize_dashes_and_spaces(text)
    text = _restore_dropcap_first_letter(text)
    text = _fix_homoglyphs_in_russian_context(text)
    text = _fix_short_latin_words_in_russian_lines(text)
    text = _fix_digits_in_russian_words(text)
    text = _drop_f_prefix_before_rus_vowel(text)
    text = _dedupe_repeated_words(text)
    
    # Step 13: Unicode normalization & final spacing cleanup
    # Шаг 13: Unicode нормализация и финальная очистка пробелов
    text = unicodedata.normalize("NFC", text)  # Canonical composition / Каноническая композиция
    text = re.sub(r"[ \t]+", " ", text)  # Collapse multiple spaces / Схлопываем множественные пробелы
    text = re.sub(r"[ \t]+\n", "\n", text)  # Trim trailing spaces / Убираем конечные пробелы
    text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 blank lines / Максимум 2 пустых строки
    
    return text.strip() + "\n"


# =============================================================================
# MARKDOWN STRUCTURE PARSING / ПАРСИНГ СТРУКТУРЫ MARKDOWN
# =============================================================================

# Pattern for Markdown headings: # Heading, ## Heading, etc. (1-6 levels)
# Шаблон для заголовков Markdown: # Заголовок, ## Заголовок и т.д. (1-6 уровней)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")

# Pattern for page markers in text: "Page 189" or "Страница 189"
# Шаблон для маркеров страниц в тексте: "Page 189" или "Страница 189"
_PAGE_RE = re.compile(r"^(?:Page|Страница)\s+(\d+)\s*$", re.IGNORECASE)


def iter_markdown_sections(md: str) -> Iterator[Section]:
    """
    Yield sections split by Markdown headings with best-effort page detection.
    Выдаёт секции, разбитые по заголовкам Markdown с попыткой определения страниц.
    
    This function parses Markdown into hierarchical sections based on headings.
    It maintains a path stack to track the heading hierarchy (breadcrumb trail).
    
    Эта функция парсит Markdown в иерархические секции на основе заголовков.
    Она поддерживает стек путей для отслеживания иерархии заголовков (цепочки навигации).

    Heuristics / Эвристики:
    - Headings are lines starting with 1..6 '#' / Заголовки - строки, начинающиеся с 1..6 '#'
    - Page markers are lines like "Page 189" or "Страница 189"
      Маркеры страниц - строки типа "Page 189" или "Страница 189"
    - Section text includes everything between headings
      Текст секции включает всё между заголовками
    - Page ranges are tracked for each section
      Диапазоны страниц отслеживаются для каждой секции
    
    Parameters / Параметры:
        md: Input Markdown text / Входной текст Markdown
        
    Yields / Выдаёт:
        Section objects with title, level, text, pages, and hierarchy path
        Объекты Section с заголовком, уровнем, текстом, страницами и путём иерархии
    """
    lines = md.splitlines()
    current_page: Optional[int] = None
    path_stack: List[Tuple[int, str]] = []  # list of (level, title)

    buf_lines: List[str] = []
    sec_title: Optional[str] = None
    sec_level: Optional[int] = None
    sec_first_page: Optional[int] = None
    sec_last_page: Optional[int] = None

    def flush() -> Optional[Section]:
        nonlocal buf_lines, sec_title, sec_level, sec_first_page, sec_last_page
        if sec_title is None or sec_level is None:
            buf_lines = []
            return None
        text = "\n".join(buf_lines).strip()
        s = Section(
            title=sec_title,
            level=sec_level,
            text=text,
            page_start=sec_first_page,
            page_end=sec_last_page,
            path=tuple(path_stack),
        )
        buf_lines = []
        return s

    for line in lines:
        m_page = _PAGE_RE.match(line)
        if m_page:
            current_page = int(m_page.group(1))
            if sec_title is not None:
                sec_last_page = current_page
            continue

        m_head = _HEADING_RE.match(line)
        if m_head:
            # close previous section
            prev = flush()
            if prev is not None:
                yield prev

            level = len(m_head.group(1))
            title = m_head.group(2).strip()

            # update path stack
            while path_stack and path_stack[-1][0] >= level:
                path_stack.pop()
            path_stack.append((level, title))

            sec_title = title
            sec_level = level
            sec_first_page = current_page
            sec_last_page = current_page
            continue

        buf_lines.append(line)

    # last section
    last = flush()
    if last is not None:
        yield last


# Helper regex for fenced code blocks used across normalization utilities
_FENCE_RE = re.compile(r"^```")
