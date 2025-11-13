from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMPostprocessConfig(BaseModel):
    enabled: bool = False
    min_score: float = 0.45
    max_paragraphs: int | None = 20
    temperature: float = 0.0
    model: str | None = None
    api_key_env: str = "LLM_API_KEY"
    model_env: str = "MODEL"
    base_url_env: str | None = "URL_MODEL"
    max_retries: int = 2
    request_timeout: float | None = 30.0
    env_path: str | None = None


class IngestConfig(BaseModel):
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120
    min_paragraph_chars: int = 40
    keep_tables_as_blocks: bool = True
    include_image_captions: bool = True
    llm_postprocess: LLMPostprocessConfig = LLMPostprocessConfig()


class EnvIngestOverrides(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INGEST_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    chunk_size_tokens: Optional[int] = None
    chunk_overlap_tokens: Optional[int] = None
    min_paragraph_chars: Optional[int] = None
    keep_tables_as_blocks: Optional[bool] = None
    include_image_captions: Optional[bool] = None
    llm_postprocess__enabled: Optional[bool] = None
    llm_postprocess__min_score: Optional[float] = None
    llm_postprocess__max_paragraphs: Optional[int] = None
    llm_postprocess__temperature: Optional[float] = None
    llm_postprocess__model: Optional[str] = None
    llm_postprocess__api_key_env: Optional[str] = None
    llm_postprocess__model_env: Optional[str] = None
    llm_postprocess__base_url_env: Optional[str] = None
    llm_postprocess__max_retries: Optional[int] = None
    llm_postprocess__request_timeout: Optional[float] = None
    llm_postprocess__env_path: Optional[str] = None


def _resolve_default_ingest_path() -> Path:
    """Найти configs/ingest.yaml, поднимаясь вверх от текущего файла."""
    start = Path(__file__).resolve()
    for p in [start] + list(start.parents):
        cand = p / "configs" / "ingest.yaml"
        if cand.exists():
            return cand
    # Fallback: попытка подняться на 3 уровня (корень репо) и собрать путь
    try:
        root = Path(__file__).resolve().parents[3]
        return root / "configs" / "ingest.yaml"
    except Exception:
        return Path("configs/ingest.yaml")


DEFAULT_CONFIG_PATH = _resolve_default_ingest_path()


def load_ingest_config(path: Optional[str | Path] = None) -> IngestConfig:
    file_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    # Если путь относительный — пробуем найти относительно ближайшего родителя с configs/
    if not file_path.is_absolute():
        for p in [Path(__file__).resolve()] + list(Path(__file__).resolve().parents):
            cand = p / file_path
            if cand.exists():
                file_path = cand
                break

    if not file_path.exists():
        # Диагностика: сообщаем куда смотрим
        print(f"[CFG] ingest.yaml не найден по пути: {file_path}")
        data = {}
    else:
        print(f"[CFG] ingest.yaml: {file_path} (exists=True)")
        with file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    params = data.get("ingest", data) if isinstance(data, dict) else {}
    cfg = IngestConfig(**params)

    overrides = EnvIngestOverrides()
    override_dict = overrides.model_dump(exclude_none=True)
    if override_dict:
        cfg = cfg.model_copy(update=override_dict)

    return cfg

