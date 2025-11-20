"""Vector retriever built on top of Qdrant.

RU: Ретривер поверх Qdrant для поиска релевантных чанков.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, SearchParams


FilterLike = Union[Filter, Dict[str, Any], None]


@dataclass
class RetrievedChunk:
    """Container for retrieved chunk metadata."""

    chunk_id: str
    text: str
    score: float
    payload: Dict[str, Any]


def _coerce_filter(filter_like: FilterLike) -> Optional[Filter]:
    """Convert dict filters to Filter objects."""

    if filter_like is None:
        return None
    if isinstance(filter_like, Filter):
        return filter_like
    if isinstance(filter_like, dict):
        return Filter.model_validate(filter_like)
    raise TypeError("filter must be None, Filter или dict совместимого формата.")


class Retriever:
    """Thin wrapper around Qdrant client search primitives (Async)."""

    def __init__(
        self,
        *,
        client: Optional[AsyncQdrantClient] = None,
        collection: str = "dnd_rule_assistant",
        host: str = "localhost",
        port: int = 6333,
        default_search_params: Optional[SearchParams] = None,
    ) -> None:
        """
        Parameters
        ----------
        client: Optional[AsyncQdrantClient]
            Existing client instance (для тестов). Если не задан, будет создан.
        collection: str
            Название коллекции Qdrant.
        host, port: str, int
            Параметры подключения при создании клиента.
        default_search_params: Optional[SearchParams]
            Значения по умолчанию для SearchParams (например, hnsw_ef).
        """

        self._client = client or AsyncQdrantClient(host=host, port=port)
        self.collection = collection
        self.default_search_params = default_search_params

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        query_filter: FilterLike = None,
        with_payload: bool = True,
        search_params: Optional[SearchParams] = None,
    ) -> List[RetrievedChunk]:
        """
        Perform a vector search and return RetrievedChunk objects.

        RU: Выполняет векторный поиск и возвращает список RetrievedChunk.
        """

        if not query_vector:
            raise ValueError("query_vector не может быть пустым.")

        filter_obj = _coerce_filter(query_filter)
        params = search_params or self.default_search_params

        # AsyncQdrantClient always has query_points (v1.7+)
        response = await self._client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=limit,
            with_payload=with_payload,
            search_params=params,
            query_filter=filter_obj,
            score_threshold=score_threshold,
        )
        points = response.points

        retrieved: List[RetrievedChunk] = []
        for point in points:
            score = point.score if point.score is not None else 0.0
            if score_threshold is not None and score < score_threshold:
                continue

            payload = point.payload or {}
            chunk_id = str(payload.get("chunk_id") or point.id)
            text = payload.get("text", "")

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    score=score,
                    payload=payload,
                )
            )
        return retrieved

