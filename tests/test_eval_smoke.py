from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pytest

from dnd_rag.core.pipelines import answer_query_pipeline

DATASET_PATH = Path("data/eval/dataset.jsonl")


def _load_first_sample() -> dict:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Файл {DATASET_PATH} не найден. Сгенерируйте датасет перед запуском smoke-теста."
        )
    with DATASET_PATH.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            return json.loads(line)
    raise RuntimeError("Датасет пуст, нет примеров для smoke-теста.")


def _chunk_ids(chunks: List) -> List[str]:
    ids: List[str] = []
    for chunk in chunks:
        if getattr(chunk, "chunk_id", None):
            ids.append(chunk.chunk_id)
    return ids


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_RAG_E2E") is None,
    reason="Установите переменную окружения RUN_RAG_E2E=1 для запуска e2e smoke-теста.",
)


@pytest.mark.asyncio
async def test_eval_smoke_recall_and_answer() -> None:
    sample = _load_first_sample()
    question = sample["question"]
    expected_answer = sample["answer"]
    expected_chunks = {ref["chunk_id"] for ref in sample.get("references", []) if ref.get("chunk_id")}

    result = await answer_query_pipeline(
        question,
        include_diagnostics=True,
    )

    assert result.answer.strip(), "LLM вернул пустой ответ."
    assert expected_answer.split()[0].lower()[:3] in result.answer.lower(), (
        "Ответ не содержит ожидаемого топика. "
        "Проверьте промпт или полноту контекста."
    )

    retrieved_chunk_ids = set(_chunk_ids(result.chunks))
    overlap = len(expected_chunks & retrieved_chunk_ids)
    assert overlap >= 1, (
        f"Ожидаем хотя бы один релевантный chunk (всего {len(expected_chunks)}), "
        f"но пересечение пусто. Настройте retriever/ингест."
    )

    assert result.diagnostics is not None, "Диагностика должна быть включена для smoke-теста."

