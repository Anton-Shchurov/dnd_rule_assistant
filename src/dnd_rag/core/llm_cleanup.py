from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

from .io import SuspiciousParagraph

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты — редактор OCR-текста по правилам Dungeons & Dragons. "
    "Исправляй только явные опечатки, ошибки распознавания и омографы, "
    "не переписывай стилистику и не добавляй новое содержание. "
    "Сохраняй Markdown-разметку и форматирование абзаца."
)


@dataclass
class LLMCleanupConfig:
    """Настройки LLM-постобработки."""

    model: Optional[str] = "gpt-4o-mini"
    temperature: float = 0.0
    max_retries: int = 2
    request_timeout: Optional[float] = 30.0
    api_key_env: str = "LLM_API_KEY"
    model_env: str = "MODEL"
    base_url_env: Optional[str] = "URL_MODEL"
    env_path: Optional[Path | str] = None


@dataclass
class LLMPostprocessOptions:
    enabled: bool = False
    min_score: float = 0.45
    max_paragraphs: Optional[int] = 20
    llm: LLMCleanupConfig = field(default_factory=LLMCleanupConfig)


def _load_env(env_path: Optional[Path | str]) -> None:
    """Загружает переменные окружения из .env, если файл найден."""
    if env_path is not None:
        load_dotenv(dotenv_path=Path(env_path), override=False)
        return

    # Попытка найти .env в корне репозитория.
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / ".env"
    if candidate.exists():
        load_dotenv(dotenv_path=candidate, override=False)
    else:
        # Fallback: стандартный механизм python-dotenv (ищет по цепочке родительских директорий).
        load_dotenv(override=False)


def _build_chain(cfg: LLMCleanupConfig) -> RunnableSequence:
    _load_env(cfg.env_path)

    api_key = os.environ.get(cfg.api_key_env or "", "").strip() if cfg.api_key_env else ""
    if not api_key:
        # фолбэк на прежнее имя
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(f"Не найден ключ API (ожидалось {cfg.api_key_env} или OPENAI_API_KEY).")

    model_name = os.environ.get(cfg.model_env or "", "").strip() if cfg.model_env else ""
    if not model_name:
        # фолбэк на прежнее имя
        model_name = os.environ.get("OPENAI_MODEL", "").strip()
    if not model_name:
        model_name = (cfg.model or "").strip()
    if not model_name:
        raise RuntimeError("Не удалось определить модель (ожидалось MODEL/OPENAI_MODEL или config.model).")

    base_url = os.environ.get(cfg.base_url_env or "", "").strip() if cfg.base_url_env else ""

    kwargs = dict(
        model=model_name,
        temperature=cfg.temperature,
        max_retries=cfg.max_retries,
        timeout=cfg.request_timeout,
        api_key=api_key,
    )
    if base_url:
        kwargs["base_url"] = base_url

    llm = ChatOpenAI(**kwargs)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Исправь OCR-ошибки в абзаце.\n\n"
                "Обнаруженные проблемы: {reasons}\n\n"
                "Требования:\n"
                "- Сохраняй исходный смысл и форматирование.\n"
                "- Вноси только минимальные правки.\n"
                "- Не добавляй комментарии и пояснения.\n\n"
                "Абзац:\n```markdown\n{paragraph}\n```\n\n"
                "Верни только исправленный текст без дополнительных пояснений.",
            ),
        ]
    )

    return prompt | llm | StrOutputParser()


def apply_llm_cleanup(
    text: str,
    paragraphs: Sequence[SuspiciousParagraph],
    *,
    cfg: Optional[LLMCleanupConfig] = None,
) -> Dict[str, object]:
    """Выполняет точечное исправление абзацев при помощи LLM.

    Parameters
    ----------
    text:
        Полный Markdown-текст исходного документа.
    paragraphs:
        Набор подозрительных абзацев для исправления.
    cfg:
        Настройки модели и доступа к OpenAI.

    Returns
    -------
    Dict[str, object]
        Возвращает словарь с двумя ключами:
        - ``text``: обновлённый Markdown.
        - ``changes``: список словарей с описанием применённых правок.
    """
    if not paragraphs:
        return {"text": text, "changes": []}

    config = cfg or LLMCleanupConfig()
    chain = _build_chain(config)

    lines = text.split("\n")
    changes: List[Dict[str, object]] = []
    offset = 0

    for para in sorted(paragraphs, key=lambda p: (p.start_line, -p.score)):
        start = para.start_line - 1 + offset
        end = para.end_line - 1 + offset
        if start < 0 or start >= len(lines):
            logger.warning(
                "Пропускаем абзац вне диапазона: start=%s end=%s total=%s",
                start,
                end,
                len(lines),
            )
            continue
        end = min(end, len(lines) - 1)
        original = "\n".join(lines[start : end + 1])

        reasons = ", ".join(para.reasons) if para.reasons else "нет данных"
        try:
            corrected = chain.invoke({"paragraph": original, "reasons": reasons}).strip()
        except Exception as exc:  # pragma: no cover - сетевые ошибки
            logger.warning("LLM не обработал абзац (%s:%s): %s", para.start_line, para.end_line, exc)
            continue

        if not corrected:
            continue

        normalized_original = original.strip()
        normalized_corrected = corrected.strip()
        if normalized_corrected == normalized_original:
            continue

        replacement_lines = corrected.split("\n")
        lines[start : end + 1] = replacement_lines
        delta = len(replacement_lines) - (end - start + 1)
        offset += delta

        changes.append(
            {
                "start_line": para.start_line,
                "end_line": para.end_line,
                "score": para.score,
                "reasons": para.reasons,
                "before": original,
                "after": corrected,
            }
        )

    return {"text": "\n".join(lines), "changes": changes}

