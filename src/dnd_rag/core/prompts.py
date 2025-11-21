from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _resolve_default_prompts_path() -> Path:
    start = Path(__file__).resolve()
    for p in [start] + list(start.parents):
        cand = p / "configs" / "prompts.yaml"
        if cand.exists():
            return cand
    try:
        root = Path(__file__).resolve().parents[3]
        return root / "configs" / "prompts.yaml"
    except Exception:
        return Path("configs/prompts.yaml")


DEFAULT_PROMPTS_PATH = _resolve_default_prompts_path()

_FALLBACK_SYSTEM_PROMPT = (
    "Ты — эксперт по правилам Dungeons & Dragons 5e. Отвечай только фактами из "
    "предоставленного контекста. Если информации недостаточно, честно скажи об этом. "
    "Всегда добавляй ссылки на источники в формате [номер]."
)


@lru_cache(maxsize=8)
def _load_prompt_data(path: Optional[str | Path]) -> Dict[str, Any]:
    target = Path(path) if path is not None else DEFAULT_PROMPTS_PATH
    if not target.exists():
        return {}
    with target.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def get_system_prompt(path: Optional[str | Path] = None) -> str:
    data = _load_prompt_data(path)
    if "system_prompt" in data:
        return str(data["system_prompt"])
    prompts = data.get("prompts")
    if isinstance(prompts, dict) and "system" in prompts:
        return str(prompts["system"])
    return _FALLBACK_SYSTEM_PROMPT


def get_eval_prompt(path: Optional[str | Path] = None) -> str:
    data = _load_prompt_data(path)
    if "eval_prompt" in data:
        return str(data["eval_prompt"])
    return (
        "Оцени качество ответа по шкале 0-1. "
        "Ответь JSON: {\"score\": <float>, \"reasoning\": \"...\"}"
    )





















