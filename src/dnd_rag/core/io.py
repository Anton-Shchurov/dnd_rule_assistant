from __future__ import annotations

import json
import re
import html
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
import tiktoken


# --- Tokenization helpers ---

_ENCODING = tiktoken.get_encoding("cl100k_base")


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


# --- Parsing & normalization ---

def parse_pdf_to_markdown(pdf_path: str | Path, *, ocr: bool = False) -> str:
    """Convert a PDF into Markdown using Docling.

    Parameters
    ----------
    pdf_path: str | Path
        Path to PDF file.
    ocr: bool
        If False, Docling keeps OCR disabled and relies on embedded PDF text;
        if True, Docling may invoke OCR on problematic pages automatically.
    """
    converter = DocumentConverter()
    if not ocr:
        pdf_option = converter.format_to_options.get(InputFormat.PDF)
        if pdf_option is not None and getattr(pdf_option, "pipeline_options", None):
            pdf_option.pipeline_options.do_ocr = False
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

_RE_SPACED_CAPS_CYR = re.compile(r"(?<!\S)([А-ЯЁ](?:\s[А-ЯЁ]){1,})(?!\S)")
_RE_SPACED_CAPS_LAT = re.compile(r"(?<!\S)([A-Z](?:\s[A-Z]){2,})(?!\S)")


def _decode_uni_escapes(text: str) -> str:
    """Convert sequences like '/uni041F' to actual Unicode chars (e.g., 'П')."""
    return re.sub(r"/uni([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), text)


def _html_unescape(text: str) -> str:
    """Decode HTML entities (&amp; → &, etc.)."""
    return html.unescape(text)


def _join_spaced_caps(text: str) -> str:
    """Collapse spaced all-caps words like 'Г ЛАВА' / 'D U N G E O N S'."""
    def _join(m: re.Match[str]) -> str:
        return m.group(0).replace(" ", "")

    text = _RE_SPACED_CAPS_CYR.sub(_join, text)
    text = _RE_SPACED_CAPS_LAT.sub(_join, text)
    return text


def _fix_hyphenation_and_linebreaks(text: str) -> str:
    """Fix line-break hyphenation and stray in-word newlines.

    - ([А-Яа-яЁё]) -\n ([А-Яа-яЁё]) → склейка
    - Внутрисловные переводы строк: (?<=\S)\n(?=\S) → пробел
    """
    text = re.sub(r"([А-Яа-яЁё])-\s*\n\s*([А-Яа-яЁё])", r"\1\2", text)
    text = re.sub(r"(?<=\S)\n(?=\S)", " ", text)
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
    pattern = re.compile(r"\b(\w{4,})\b\s+\1\b", flags=re.IGNORECASE)
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub(r"\\1", text)
    return text


def deep_normalize_markdown(md: str) -> str:
    """Robust normalization for Docling/OCR Markdown.

    Объединённые шаги нормализации:
    1) HTML entities → Unicode; `/uniXXXX` → символ.
    2) Склейка разнесённых капслоком слов (кириллица/латиница).
    3) Починка переносов и внутрисловных переводов строк.
    4) Нормализация дефиса/тире и пробелов вокруг знаков препинания.
    5) Латиница→кириллица для омографов в русском окружении.
    6) Чистка артефакта 'f' перед русской гласной.
    7) Удаление сдвоенных слов.
    8) Unicode NFC и уплотнение пробелов.
    """
    if not md:
        return md

    text = md.replace("\r\n", "\n").replace("\r", "\n")
    text = _html_unescape(text)
    text = _decode_uni_escapes(text)
    # Base lightweight fixes first
    text = normalize_markdown(text)
    # Deep fixes
    text = _join_spaced_caps(text)
    text = _fix_hyphenation_and_linebreaks(text)
    text = _normalize_dashes_and_spaces(text)
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

