from __future__ import annotations

from typing import Dict, Any, Sequence, Union
from uuid import UUID, uuid5, NAMESPACE_URL

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

_MAX_POINTS_PER_BATCH = 128


def get_client(host: str = "localhost", port: int = 6333) -> QdrantClient:
    return QdrantClient(host=host, port=port)


def ensure_collection(
    client: QdrantClient,
    collection: str,
    *,
    vector_size: int,
    distance: Distance = Distance.COSINE,
) -> None:
    if client.collection_exists(collection):
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=distance),
    )


def _normalize_point_id(raw_id: Union[str, int]) -> Union[str, int]:
    """
    Qdrant принимает идентификаторы только в виде неотрицательных целых чисел
    или UUID. Это хелпер, который приводит произвольную строку к допустимому
    формату (детерминистический UUID5), а числовые строки — к int.
    """
    if isinstance(raw_id, int):
        if raw_id < 0:
            raise ValueError(f"Point ID должен быть неотрицательным, получено {raw_id}.")
        return raw_id

    if isinstance(raw_id, str):
        stripped = raw_id.strip()
        if not stripped:
            raise ValueError("Point ID не может быть пустой строкой.")

        if stripped.isdigit():
            return int(stripped)

        try:
            parsed = UUID(stripped)
        except ValueError:
            parsed = uuid5(NAMESPACE_URL, stripped)
        return str(parsed)

    raise TypeError(f"Point ID должен быть строкой или int, получено {type(raw_id)!r}.")


def upsert_vectors(
    client: QdrantClient,
    collection: str,
    *,
    ids: Sequence[str],
    vectors: Sequence[Sequence[float]],
    payloads: Sequence[Dict[str, Any]],
) -> None:
    if not (len(ids) == len(vectors) == len(payloads)):
        raise ValueError(
            "Размеры списков ids, vectors и payloads должны совпадать для upsert_vectors."
        )

    normalized_ids = [_normalize_point_id(i) for i in ids]

    total = len(normalized_ids)
    if total == 0:
        return

    for start in range(0, total, _MAX_POINTS_PER_BATCH):
        end = min(start + _MAX_POINTS_PER_BATCH, total)
        batch_points = [
            PointStruct(
                id=normalized_ids[i],
                vector=vectors[i],
                payload=payloads[i],
            )
            for i in range(start, end)
        ]
        client.upsert(collection_name=collection, points=batch_points)
