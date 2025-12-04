"""Telegram bot entrypoint for DnD Rule Assistant."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import random
import re
from typing import Optional, Set, List
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import find_dotenv, load_dotenv

from dnd_rag.core.config import DEFAULT_CONFIG_PATH
from dnd_rag.core.pipelines import AnswerResult, answer_query_pipeline

_dotenv_path = find_dotenv(filename=".env", usecwd=True)
if _dotenv_path:
    load_dotenv(_dotenv_path, override=False)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "dnd_rule_assistant")
DEBUG_LOG = os.environ.get("DEBUG_LOG", "").lower() not in {"", "0", "false", "no"}

dp = Dispatcher()


def _extract_citation_indices(text: str) -> Set[int]:
    """Extracts citation indices like [1], [2] from text."""
    matches = re.findall(r"\[(\d+)\]", text)
    return {int(m) for m in matches}


def _format_meta(result: AnswerResult) -> str:
    if not result.chunks:
        return "Источники: нет данных."
    
    # Filter chunks based on citations in the answer
    cited_indices = _extract_citation_indices(result.answer)
    
    # If citations are found, filter chunks. Otherwise show all (fallback).
    if cited_indices:
        # Filter and keep original indices for display if needed, 
        # or just re-enumerate. Let's keep original indices to match text citations.
        # But wait, if text says [1] and we filter, we should ensure chunk [1] is displayed as [1].
        # So we iterate over all chunks and include only those whose index (1-based) is in cited_indices.
        display_chunks = []
        for idx, chunk in enumerate(result.chunks, start=1):
            if idx in cited_indices:
                display_chunks.append((idx, chunk))
    else:
        # Fallback: show all chunks
        display_chunks = [(idx, chunk) for idx, chunk in enumerate(result.chunks, start=1)]

    if not display_chunks:
         # Should not happen if result.chunks was not empty, unless citations point to non-existent chunks
         # In that case, maybe fallback to all? Or show none? 
         # Let's fallback to all if filtering resulted in nothing but we have chunks.
         display_chunks = [(idx, chunk) for idx, chunk in enumerate(result.chunks, start=1)]

    lines = ["Источники:"]
    for idx, chunk in display_chunks:
        payload = chunk.payload
        book = payload.get("book_title") or payload.get("book") or ""
        chapter = payload.get("chapter_title") or payload.get("chapter") or ""
        sec_path = payload.get("section_path") or []
        if isinstance(sec_path, list):
            sec_path = " › ".join([s for s in sec_path if s])
        parts = [part for part in (book, chapter, sec_path) if part]
        chunk_id = payload.get("chunk_id")
        title = " / ".join(parts) if parts else (chunk_id or "fragment")
        lines.append(f"[{idx}] {title}")
    return "\n".join(lines)


def _format_debug_block(result: AnswerResult) -> str:
    diag = result.diagnostics
    if not diag:
        return "Диагностика недоступна."

    def _fmt_score(value: Optional[float]) -> str:
        return f"{value:.3f}" if value is not None else "—"

    header = (
        f"🛠 Диагностика\n"
        f"retrieved={len(diag.retrieved)} initial_k={diag.initial_k} "
        f"rerank={'on' if diag.rerank_enabled else 'off'} "
        f"duration={int(diag.duration_ms or 0)}ms"
    )
    lines = [header, "Контекст:"]
    if not diag.final_chunks:
        lines.append("—")
        return "\n".join(lines)

    for chunk in diag.final_chunks:
        sections = " › ".join(chunk.section_path) if chunk.section_path else ""
        source_parts = [p for p in (chunk.book_title, chunk.chapter_title, sections) if p]
        lines.append(
            f"[{chunk.rank}] {chunk.chunk_id} "
            f"vec={_fmt_score(chunk.vector_score)} rer={_fmt_score(chunk.rerank_score)} "
            f"{' / '.join(source_parts) or ''}"
        )
    return "\n".join(lines)





@dp.message(Command("start", "help"))
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет! Я D&D Rule Assistant.\n"
        "Просто напиши мне свой вопрос, и я найду ответ в правилах."
    )


@dp.message(F.text)
async def handle_message(message: Message) -> None:
    question = (message.text or "").strip()
    if not question:
        return

    waiting_messages = [
        "🧙‍♂️ Вглядываюсь в хрустальный шар…",
        "📜 Листаю древние манускрипты…",
        "✨ Советуюсь с мудрецами…",
        "🐉 Спрашиваю у дракона…",
        "🎲 Бросаю кубик на мудрость…",
    ]
    await message.answer(random.choice(waiting_messages))
    try:
        # Now calling async pipeline directly
        result = await answer_query_pipeline(
            question,
            collection=QDRANT_COLLECTION,
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            config_path=DEFAULT_CONFIG_PATH,
            log_queries=DEBUG_LOG,
            include_diagnostics=DEBUG_LOG,
        )
    except Exception as exc:  # pragma: no cover - сеть/ключи
        logger.exception("Failed to answer via pipeline", exc_info=exc)
        await message.answer("Произошла ошибка при обращении к ассистенту.")
        return

    text = result.answer.strip() or "Ответ пуст."
    sources = _format_meta(result)
    await message.answer(f"{text}\n\n{sources}", disable_web_page_preview=True)

    if DEBUG_LOG and result.diagnostics:
        await message.answer(_format_debug_block(result), disable_web_page_preview=True)


async def _run_bot(token: str) -> None:
    bot = Bot(token=token)
    await dp.start_polling(bot)


def main() -> None:  # pragma: no cover
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан.")
    asyncio.run(_run_bot(token))


if __name__ == "__main__":  # pragma: no cover
    main()
