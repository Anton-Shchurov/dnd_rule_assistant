from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestConfig(BaseModel):
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120
    min_paragraph_chars: int = 40
    keep_tables_as_blocks: bool = True
    include_image_captions: bool = True


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


DEFAULT_CONFIG_PATH = Path("configs/ingest.yaml")


def load_ingest_config(path: Optional[str | Path] = None) -> IngestConfig:
    file_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    data = {}
    if file_path.exists():
        with file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    params = data.get("ingest", data) if isinstance(data, dict) else {}
    cfg = IngestConfig(**params)

    overrides = EnvIngestOverrides()
    override_dict = overrides.model_dump(exclude_none=True)
    if override_dict:
        cfg = cfg.model_copy(update=override_dict)

    return cfg

