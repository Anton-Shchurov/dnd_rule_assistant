from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Sequence

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
import tiktoken

# Автоматически подгружаем переменные окружения из .env в корне проекта (если найден)
_dotenv_path = find_dotenv(filename=".env", usecwd=True)
if _dotenv_path:
    load_dotenv(_dotenv_path, override=False)

_OPENAI_REQUEST_TOKEN_LIMIT = 300_000
_DEFAULT_TOKEN_BUDGET = 250_000
_MAX_TEXTS_PER_BATCH = 512


@lru_cache(maxsize=None)
def _encoding_for_model(model: str) -> tiktoken.Encoding:
    """
    Возвращает подходящее кодирование для модели эмбеддингов.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    """
    Подсчитывает количество токенов в тексте для заданного кодировщика.
    """
    if not text:
        return 0
    return len(encoding.encode_ordinary(text))


def embed_texts(
    texts: Sequence[str],
    *,
    model: str = "text-embedding-3-small",
) -> List[List[float]]:
    """
    Создаёт эмбеддинги для списка текстов через OpenAI.
    Требуется переменная окружения OPENAI_API_KEY.
    """
    texts_list = list(texts)
    if not texts_list:
        return []

    encoding = _encoding_for_model(model)
    token_counts = [_count_tokens(text, encoding) for text in texts_list]

    for idx, token_count in enumerate(token_counts):
        if token_count > _OPENAI_REQUEST_TOKEN_LIMIT:
            raise ValueError(
                f"Текст под индексом {idx} содержит {token_count} токенов — это превышает "
                f"предел { _OPENAI_REQUEST_TOKEN_LIMIT } токенов на один запрос OpenAI. "
                "Сократите размер чанка или увеличьте степень разбиения."
            )

    batches = []
    batch_indices: List[int] = []
    batch_texts: List[str] = []
    batch_tokens = 0

    for idx, (text, token_count) in enumerate(zip(texts_list, token_counts)):
        if batch_indices and (
            len(batch_indices) >= _MAX_TEXTS_PER_BATCH
            or batch_tokens + token_count > _DEFAULT_TOKEN_BUDGET
        ):
            batches.append((batch_indices, batch_texts))
            batch_indices = []
            batch_texts = []
            batch_tokens = 0

        batch_indices.append(idx)
        batch_texts.append(text)
        batch_tokens += token_count

    if batch_indices:
        batches.append((batch_indices, batch_texts))

    client = OpenAI()
    vectors: List[Optional[List[float]]] = [None] * len(texts_list)

    for indices, batch in batches:
        resp = client.embeddings.create(
            model=model,
            input=batch,
        )
        for target_idx, data in zip(indices, resp.data):
            vectors[target_idx] = data.embedding

    ordered_vectors: List[List[float]] = []
    for vec in vectors:
        if vec is None:
            raise RuntimeError(
                "Не удалось получить эмбеддинг для одного из чанков: OpenAI вернул "
                "меньше векторов, чем ожидалось."
            )
        ordered_vectors.append(vec)

    return ordered_vectors
