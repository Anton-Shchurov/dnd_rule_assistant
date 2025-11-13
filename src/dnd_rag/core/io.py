from __future__ import annotations

import json
import re
import html
import unicodedata
import shutil
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from functools import lru_cache

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import TesseractCliOcrOptions
import tiktoken

try:  # Опциональная морфология для восстановления буквиц
    import pymorphy2
except ImportError:  # pragma: no cover - отсутствие зависимости не критично
    pymorphy2 = None  # type: ignore[assignment]


# --- Tokenization helpers ---

_ENCODING = tiktoken.get_encoding("cl100k_base")

if pymorphy2 is not None:  # pragma: no cover - зависит от окружения
    try:
        _MORPH_ANALYZER = pymorphy2.MorphAnalyzer()
    except Exception:  # pragma: no cover - безопасный фолбэк
        _MORPH_ANALYZER = None
else:
    _MORPH_ANALYZER = None


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text or ""))


# --- Data models ---

@dataclass
class Section:
    title: str
    level: int
    text: str
    page_start: Optional[int]
    page_end: Optional[int]
    path: Tuple[Tuple[int, str], ...]  # breadcrumb of headings up to this section


@dataclass
class Chunk:
    text: str
    chapter: Optional[str]
    section: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    tokens: int


@dataclass
class SuspiciousParagraph:
    text: str
    score: float
    reasons: List[str]
    start_line: int
    end_line: int


# --- Parsing & normalization ---

def parse_pdf_to_markdown(
    pdf_path: str | Path,
    *,
    ocr: bool = False,
    ocr_langs: Optional[Sequence[str]] = None,
) -> str:
    """Convert a PDF into Markdown using Docling.

    Parameters
    ----------
    pdf_path: str | Path
        Path to PDF file.
    ocr: bool
        If False, Docling использует встроенный текст PDF без OCR.
        Если True, Docling включает OCR-пайплайн (PaddleOCR/Tesseract).
    ocr_langs: Optional[Sequence[str]]
        Список языковых подсказок для OCR (например, ("rus", "eng")).
        По умолчанию используется ("rus", "eng"), если ocr=True.
    """
    converter = DocumentConverter()
    pdf_option = converter.format_to_options.get(InputFormat.PDF)
    pipeline_options = None
    if pdf_option is not None:
        pipeline_options = getattr(pdf_option, "pipeline_options", None)

    if pipeline_options is not None:
        if hasattr(pipeline_options, "do_ocr"):
            pipeline_options.do_ocr = bool(ocr)
        if ocr:
            if ocr_langs is None:
                ocr_langs = ("rus", "eng")
            lang_list = list(dict.fromkeys(ocr_langs)) if ocr_langs else []
            lang_list = [str(lang).strip().lower() for lang in lang_list if lang]
            if lang_list:
                _ISO639_2 = {"en": "eng", "ru": "rus"}
                lang_list = [_ISO639_2.get(code, code) for code in lang_list]
            if not lang_list:
                lang_list = ["eng"]
            if hasattr(pipeline_options, "ocr_lang_hint"):
                pipeline_options.ocr_lang_hint = lang_list
            if hasattr(pipeline_options, "ocr_languages"):
                pipeline_options.ocr_languages = lang_list
            if hasattr(pipeline_options, "ocr_options"):
                # Предпочитаем tesseract CLI: RapidOCR часто зависает на PDF с кириллицей.
                tess_cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
                if tess_cmd is None:
                    raise RuntimeError(
                        "Tesseract OCR не найден (исполняемый файл tesseract.exe отсутствует в PATH). "
                        "Установите Tesseract и добавьте его в PATH или задайте переменную окружения "
                        "TESSERACT_CMD с полным путём до tesseract.exe."
                    )
                options = TesseractCliOcrOptions(
                    lang=lang_list,
                    tesseract_cmd=tess_cmd,
                )
                options.force_full_page_ocr = True
                pipeline_options.ocr_options = options
            if hasattr(pipeline_options, "ocr_psm") and getattr(pipeline_options, "ocr_psm") is None:
                pipeline_options.ocr_psm = "auto"
    result = converter.convert(str(pdf_path))
    md: str = result.document.export_to_markdown()
    return md


def save_markdown(md: str, out_path: str | Path) -> None:
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(md, encoding="utf-8")


def save_jsonl(rows: Iterable[Dict], out_path: str | Path) -> None:
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_markdown(md: str) -> str:
    """Light-weight normalization: EOL hyphenation, whitespace, blank lines.

    - Fix soft hyphenation across line breaks: "перено-\nсы" → "переносы".
    - Trim trailing spaces, unify newlines, collapse excessive blank lines.
    """
    if not md:
        return md

    text = md.replace("\r\n", "\n").replace("\r", "\n")

    # Remove hyphenation only when surrounded by word chars (to not touch bullet lists)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)

    # Trim trailing spaces
    text = re.sub(r"[ \t]+\n", "\n", text)

    # Collapse >2 blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# --- Deep normalization (for noisy OCR/Docling output) ---

def _decode_uni_escapes(text: str) -> str:
    """Convert sequences like '/uni041F' to actual Unicode chars (e.g., 'П')."""
    return re.sub(r"/uni([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), text)


def _html_unescape(text: str) -> str:
    """Decode HTML entities (&amp; → &, etc.)."""
    return html.unescape(text)


_SINGLE_LETTER_WORDS_CYR = {"В", "К", "С", "Я", "О", "И", "А", "У", "Э", "Е", "Ю", "Ы"}
_SINGLE_LETTER_WORDS_LAT = {"A", "I"}
_RE_SINGLE_LETTER_CYR = re.compile(r"\b([А-ЯЁ])\s+([А-ЯЁа-яё]{2,})\b")
_RE_SINGLE_LETTER_LAT = re.compile(r"\b([A-Z])\s+([A-Za-z]{2,})\b")


def _join_single_letter_splits(text: str) -> str:
    """Склейка слов вида 'П оследующие', 'D UNGEONS', 'М АСТЕР'."""

    def _join_cyr(m: re.Match[str]) -> str:
        first, rest = m.group(1), m.group(2)
        if first in _SINGLE_LETTER_WORDS_CYR and not rest.isupper():
            return m.group(0)
        return first + rest

    def _join_lat(m: re.Match[str]) -> str:
        first, rest = m.group(1), m.group(2)
        if first in _SINGLE_LETTER_WORDS_LAT and not rest.isupper():
            return m.group(0)
        return first + rest

    text = _RE_SINGLE_LETTER_CYR.sub(_join_cyr, text)
    text = _RE_SINGLE_LETTER_LAT.sub(_join_lat, text)
    return text


def _fix_hyphenation_and_linebreaks(text: str) -> str:
    """Fix line-break hyphenation and stray in-word newlines.

    - ([А-Яа-яЁё]) -\n ([А-Яа-яЁё]) → склейка
    - Внутрисловные переводы строк: (?<=\S)\n(?=\S) → пробел
    """
    text = re.sub(r"([A-Za-zА-Яа-яЁё])-\s*\n\s*([A-Za-zА-Яа-яЁё])", r"\1\2", text)

    def _join_non_structural(match: re.Match[str]) -> str:
        before = match.group(1)
        newline = match.group(2)
        after = match.group(3)
        if before.rstrip().endswith("|") or after.lstrip().startswith("|"):
            return before + newline + after
        if after.lstrip().startswith(("#", "-", "*", "+", "•")):
            return before + newline + after
        return before + " " + after.lstrip()

    text = re.sub(r"([^\n])(\n)([^\n])", _join_non_structural, text)
    return text


def _normalize_dashes_and_spaces(text: str) -> str:
    """Normalize hyphen vs em-dash and collapse spaces around punctuation.

    - Inside words: (\w) \s*-\s* (\w) → \1-\2
    - Between tokens: ' - ' → ' — '
    - Collapse excessive spaces and trim space before punctuation.
    """
    # Hyphen inside words (avoid touching bullet lists)
    text = re.sub(r"(?<=\w)\s*-\s*(?=\w)", "-", text)
    # Em-dash between tokens (not at line start)
    text = re.sub(r"(?m)(?<!^)\s-\s(?!-)", " — ", text)
    # Remove extra spaces before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # Normalize multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


_LAT_TO_CYR = str.maketrans({
    "A": "А",
    "a": "а",
    "B": "В",
    "C": "С",
    "c": "с",
    "E": "Е",
    "e": "е",
    "H": "Н",
    "K": "К",
    "k": "к",
    "M": "М",
    "O": "О",
    "o": "о",
    "P": "Р",
    "p": "р",
    "T": "Т",
    "X": "Х",
    "x": "х",
    "Y": "У",
    "y": "у",
})


_NOISE_SHORT_LINE_RE = re.compile(r"^[\W\d_]{1,3}$")
_NOISE_SOURCE_LINE_RE = re.compile(r"^@[A-Za-z0-9_.-]+\s*\(\d+(?:-\d+)?\)$")
_NOISE_HTML_COMMENT_RE = re.compile(r"^<!--.*-->$")
_NOISE_SHORT_LETTERS_RE = re.compile(
    r"^(?!(?:да|нет|yes|no)$)[A-Za-zА-Яа-яЁё]{1,3}$", re.IGNORECASE
)


def _drop_noise_lines(text: str) -> str:
    """Remove stray OCR artefacts like isolated punctuation or '@file (p-range)' markers."""
    lines = text.splitlines()
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        if _NOISE_HTML_COMMENT_RE.match(stripped):
            continue
        if stripped.startswith("<") and ">" not in stripped:
            continue
        if not re.search(r"[A-Za-zА-Яа-яЁё]", stripped):
            if _NOISE_SHORT_LINE_RE.match(stripped):
                continue
        if _NOISE_SOURCE_LINE_RE.match(stripped):
            continue
        if _NOISE_SHORT_LETTERS_RE.match(stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё']+")
_LAT_RE = re.compile(r"[A-Za-z]")
_CYR_RE = re.compile(r"[А-Яа-яЁё]")
_MIXED_SCRIPT_WORD_RE = re.compile(r"(?=.*[A-Za-z])(?=.*[А-Яа-яЁё])")
_ALLOWED_SYMBOLS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "0123456789"
    " .,:;!?\"'«»„“”()[]{}<>-–—/%&+#°•…"
)


def _evaluate_paragraph(
    lines: List[str],
    start_idx: int,
    end_idx: int,
    *,
    min_score: float,
    dominant_script: str | None = None,
) -> Optional[SuspiciousParagraph]:
    text = "\n".join(lines).strip()
    if not text:
        return None
    first_line = lines[0].lstrip()
    if first_line.startswith("#"):
        return None

    words = _WORD_RE.findall(text)
    if not words:
        return None

    cyr_chars = sum(1 for ch in text if _CYR_RE.match(ch))
    lat_chars = sum(1 for ch in text if _LAT_RE.match(ch))
    if dominant_script is None:
        if cyr_chars >= lat_chars:
            dominant_script = "cyr"
        elif lat_chars:
            dominant_script = "lat"
        else:
            dominant_script = "unknown"

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

    if _NOISE_HTML_COMMENT_RE.match(text):
        return SuspiciousParagraph(
            text=text,
            score=0.7,
            reasons=["html-comment"],
            start_line=start_idx + 1,
            end_line=end_idx + 1,
        )

    total_words = len(words)
    suspicious_words = 0
    latin_words = 0
    cyrillic_words = 0
    latin_only_words = 0
    reasons: set[str] = set()
    extra_component = 0.0
    has_mixed_token = False

    for raw_word in words:
        word = raw_word.strip("'")
        if not word:
            continue
        has_lat = bool(_LAT_RE.search(word))
        has_cyr = bool(_CYR_RE.search(word))
        if has_lat:
            latin_words += 1
        if has_cyr:
            cyrillic_words += 1

        word_flag = False

        if _MIXED_SCRIPT_WORD_RE.search(word):
            word_flag = True
            reasons.add("mixed-script")
            has_mixed_token = True

        upper_count = sum(1 for ch in word if ch.isupper())
        if upper_count >= 2 and not word.isupper():
            word_flag = True
            reasons.add("mixed-case")

        if has_cyr and len(word) >= 5 and _MORPH_ANALYZER is not None:
            lower = word.lower()
            if not _is_known_word(lower):
                word_flag = True
                reasons.add("unknown-cyr")

        if has_lat and not has_cyr:
            latin_only_words += 1
        if has_lat and not has_cyr and dominant_script == "cyr":
            if len(word) >= 4 or (len(word) <= 3 and cyrillic_words >= 3):
                if not word.isupper():
                    word_flag = True
                    reasons.add("latin-in-cyr")
                    extra_component = max(extra_component, 0.5)

        if word_flag:
            suspicious_words += 1

    total_chars = sum(1 for ch in text if not ch.isspace())
    weird_chars = sum(
        1 for ch in text if not (ch.isspace() or ch in _ALLOWED_SYMBOLS)
    )
    weird_ratio = (weird_chars / total_chars) if total_chars else 0.0
    char_component = 0.0
    if weird_ratio > 0.05:
        reasons.add("weird-chars")
        char_component = min(0.4, (weird_ratio - 0.05) * 4.0)

    latin_component = 0.0
    if cyrillic_words >= 4 and latin_words > 0:
        latin_ratio = latin_words / (latin_words + cyrillic_words)
        if latin_ratio > 0.2:
            reasons.add("latin-ratio")
            latin_component = min(0.4, (latin_ratio - 0.2) * 2.0)

    if dominant_script == "cyr" and latin_only_words > 0:
        reasons.add("latin-only")
        extra_component = max(extra_component, min(0.6, 0.4 + 0.05 * latin_only_words))

    word_ratio = suspicious_words / total_words if total_words else 0.0
    score = min(1.0, word_ratio + char_component + latin_component + extra_component)

    # БИНАРНОЕ ПРАВИЛО: если есть смешанный токен (кириллица+латиница в одном слове),
    # отправляем абзац в LLM независимо от min_score.
    candidate = SuspiciousParagraph(
        text=text,
        score=score,
        reasons=sorted(reasons),
        start_line=start_idx + 1,
        end_line=end_idx + 1,
    )
    if has_mixed_token:
        return candidate

    if score < min_score or not reasons:
        return None

    return candidate


def find_suspicious_paragraphs(
    text: str,
    *,
    min_score: float = 0.45,
    max_results: Optional[int] = None,
) -> List[SuspiciousParagraph]:
    """Scan Markdown text and flag paragraphs likely containing OCR artefacts."""
    lines = text.splitlines()
    results: List[SuspiciousParagraph] = []
    buffer: List[str] = []
    start_idx = 0

    for idx, line in enumerate(lines):
        if line.strip():
            if not buffer:
                start_idx = idx
            buffer.append(line)
        else:
            if buffer:
                cand = _evaluate_paragraph(buffer, start_idx, idx - 1, min_score=min_score)
                if cand is not None:
                    results.append(cand)
                buffer = []

    if buffer:
        cand = _evaluate_paragraph(buffer, start_idx, len(lines) - 1, min_score=min_score)
        if cand is not None:
            results.append(cand)

    def _priority(p: SuspiciousParagraph) -> Tuple[int, float, int]:
        reasons = set(p.reasons)
        if "latin-only" in reasons or "latin-in-cyr" in reasons:
            priority = 0
        elif "mixed-script" in reasons or "latin-ratio" in reasons:
            priority = 1
        elif "unknown-cyr" in reasons or "weird-chars" in reasons:
            priority = 2
        elif "short-fragment" in reasons:
            priority = 4
        else:
            priority = 3
        return (priority, -p.score, p.start_line)

    results.sort(key=_priority)
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


def _drop_f_prefix_before_rus_vowel(text: str) -> str:
    """Remove stray Latin 'f' preceding a Russian vowel at word start: 'fа' → 'а'."""
    return re.sub(r"\bf(?=[аеиоуыэюя])", "", text, flags=re.IGNORECASE)


def _dedupe_repeated_words(text: str) -> str:
    """Collapse duplicated words produced after merges: 'персонажи персонажи' → 'персонажи'."""
    pattern = re.compile(r"\b(\w{4,})\b(?:[ \t]+)\1\b", flags=re.IGNORECASE)
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub(r"\1", text)
    return text


_RUS_INITIALS = tuple("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
_LAT_INITIALS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_DROP_CAP_RE = re.compile(r"(?m)^(?P<prefix>[#>\-\*\s]{0,5})(?P<word>[A-Za-zА-Яа-яЁё]{2,})(?=\b)")
_RE_CYR = re.compile(r"[А-Яа-яЁё]")
_RE_LAT = re.compile(r"[A-Za-z]")


@lru_cache(maxsize=8192)
def _is_known_word(word: str) -> bool:
    if not word or _MORPH_ANALYZER is None:
        return False
    try:
        return any(parse.is_known for parse in _MORPH_ANALYZER.parse(word))
    except Exception:  # pragma: no cover - защищаемся от неожиданных ошибок морфологии
        return False


def _apply_case(template: str, replacement: str) -> str:
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
    """Попытка восстановить буквицы через морфологию (если доступна)."""

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
    """Robust normalization for Docling/OCR Markdown.

    Объединённые шаги нормализации:
    1) HTML entities → Unicode; `/uniXXXX` → символ.
    2) Базовая нормализация переносов и пустых строк.
    3) Удаление шумовых строк (`@DMG.md (2-11)`, одиночные символы).
    4) Склейка "П оследующие" / "D UNGEONS".
    5) Починка переносов и внутрисловных переводов строк.
    6) Нормализация дефиса/тире и пробелов вокруг знаков препинания.
    7) Попытка восстановить буквицы через pymorphy2 (если библиотека доступна).
    8) Латиница→кириллица для омографов в русском окружении.
    9) Чистка артефакта 'f' перед русской гласной.
    10) Удаление сдвоенных слов.
    11) Unicode NFC и уплотнение пробелов.
    """
    if not md:
        return md

    text = md.replace("\r\n", "\n").replace("\r", "\n")
    text = _html_unescape(text)
    text = _decode_uni_escapes(text)
    # Base lightweight fixes first
    text = normalize_markdown(text)
    # Deep fixes
    text = _drop_noise_lines(text)
    text = _join_single_letter_splits(text)
    text = _fix_hyphenation_and_linebreaks(text)
    text = _normalize_dashes_and_spaces(text)
    text = _restore_dropcap_first_letter(text)
    text = _fix_homoglyphs_in_russian_context(text)
    text = _drop_f_prefix_before_rus_vowel(text)
    text = _dedupe_repeated_words(text)
    # Unicode normalization & final spacing
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


# --- Markdown structure parsing ---

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_PAGE_RE = re.compile(r"^(?:Page|Страница)\s+(\d+)\s*$", re.IGNORECASE)


def iter_markdown_sections(md: str) -> Iterator[Section]:
    """Yield sections split by headings. Best-effort page detection.

    Heuristics:
      - Headings are lines starting with 1..6 '#'.
      - Page markers are lines like "Page 189" or "Страница 189".
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


# --- Chunking ---

_FENCE_RE = re.compile(r"^```")


def _split_into_blocks(text: str) -> List[str]:
    """Split text into logical blocks: paragraphs, tables, fenced code, lists.

    Simple heuristics: keep fenced blocks and tables atomic; split on blank lines.
    """
    lines = text.splitlines()
    blocks: List[str] = []
    cur: List[str] = []
    in_fence = False
    in_table = False

    def push():
        nonlocal cur
        if cur:
            blocks.append("\n".join(cur).strip())
            cur = []

    for ln in lines:
        if _FENCE_RE.match(ln):
            in_fence = not in_fence
            cur.append(ln)
            if not in_fence:
                push()
            continue

        # naive table detection: lines with multiple '|' characters
        if not in_fence:
            pipe_count = ln.count("|")
            if pipe_count >= 2:
                in_table = True
                cur.append(ln)
                continue
            if in_table and ln.strip() == "":
                in_table = False
                push()
                continue

        if not in_fence and not in_table and ln.strip() == "":
            push()
            continue

        cur.append(ln)

    push()
    return [b for b in blocks if b]


def chunk_sections(
    sections: Iterable[Section], *, max_tokens: int = 800, overlap: int = 120
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for sec in sections:
        header = f"{'#' * sec.level} {sec.title}".strip()
        blocks = _split_into_blocks(sec.text)

        window_tail_tokens: List[int] = []
        cur_blocks: List[str] = []
        cur_tokens = 0

        def flush_chunk():
            nonlocal window_tail_tokens, cur_blocks, cur_tokens
            if not cur_blocks:
                return
            body = "\n\n".join(cur_blocks).strip()
            prefix = _ENCODING.decode(window_tail_tokens) if window_tail_tokens else ""
            text = (header + "\n\n" + (prefix + ("\n" if prefix else "") + body).strip()).strip()
            tok = count_tokens(text)
            chunks.append(
                Chunk(
                    text=text,
                    chapter=_chapter_from_path(sec.path),
                    section=sec.title,
                    page_start=sec.page_start,
                    page_end=sec.page_end,
                    tokens=tok,
                )
            )
            # prepare overlap tail
            all_token_ids = _ENCODING.encode(body)
            tail = all_token_ids[-overlap:] if overlap > 0 else []
            window_tail_tokens = tail
            cur_blocks = []
            cur_tokens = 0

        for b in blocks if blocks else [""]:
            b_tokens = count_tokens(b)
            if b_tokens > max_tokens:
                # too large atomic block: flush current and store alone
                flush_chunk()
                text = header + "\n\n" + b
                chunks.append(
                    Chunk(
                        text=text,
                        chapter=_chapter_from_path(sec.path),
                        section=sec.title,
                        page_start=sec.page_start,
                        page_end=sec.page_end,
                        tokens=count_tokens(text),
                    )
                )
                window_tail_tokens = _ENCODING.encode(b)[-overlap:] if overlap > 0 else []
                continue

            if cur_tokens + b_tokens <= max_tokens:
                cur_blocks.append(b)
                cur_tokens += b_tokens
            else:
                flush_chunk()
                cur_blocks.append(b)
                cur_tokens = b_tokens

        flush_chunk()

    return chunks


def _chapter_from_path(path: Sequence[Tuple[int, str]]) -> Optional[str]:
    if not path:
        return None
    # chapter: the highest-level heading encountered so far (level == min)
    min_level = min(lv for lv, _ in path)
    for lv, title in path:
        if lv == min_level:
            return title
    return None

