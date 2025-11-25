from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from qdrant_client.models import Filter

from dnd_rag.core.config import DEFAULT_CONFIG_PATH
from dnd_rag.core.pipelines import answer_query_pipeline
from dnd_rag.core.retriever import RetrievedChunk, Retriever


class FakeQdrantClient:
    def __init__(self) -> None:
        self.kwargs = None

    async def query_points(self, **kwargs):
        self.kwargs = kwargs
        points = [
            SimpleNamespace(
                score=0.9,
                payload={"chunk_id": "a", "text": "foo"},
                id="1",
            ),
            SimpleNamespace(
                score=0.2,
                payload={"chunk_id": "b", "text": "bar"},
                id="2",
            ),
        ]
        return SimpleNamespace(points=points)


@pytest.mark.asyncio
async def test_retriever_filters_and_threshold():
    client = FakeQdrantClient()
    retriever = Retriever(client=client, collection="col")

    results = await retriever.search(
        [0.1, 0.2],
        limit=5,
        score_threshold=0.5,
        query_filter={"must": []},
    )

    assert len(results) == 1
    assert isinstance(client.kwargs["query_filter"], Filter)
    assert client.kwargs["score_threshold"] == 0.5
    assert results[0].chunk_id == "a"


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.last_vector = None

    async def search(self, query_vector, **kwargs):
        self.last_vector = list(query_vector)
        return self.chunks


class FakeLLM:
    def __init__(self):
        self.model = "fake-llm"
        self.calls = 0

    async def generate(self, messages, temperature=None):
        self.calls += 1
        return SimpleNamespace(
            content="Ответ [1]",
            model=self.model,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )


# Sample chunk fixture for the answer pipeline tests
@pytest.fixture
def sample_chunk():
    return RetrievedChunk(
        chunk_id="phb_ch01_0001",
        text="Содержимое правила",
        score=0.92,
        payload={"chunk_id": "phb_ch01_0001", "book_title": "PHB"},
    )


@pytest.mark.asyncio
async def test_answer_pipeline_uses_retrieved_chunks(sample_chunk):
    fake_retriever = FakeRetriever([sample_chunk])
    fake_llm = FakeLLM()
    with patch("dnd_rag.core.pipelines.embed_texts", return_value=[[0.1, 0.2]]):
        result = await answer_query_pipeline(
            "Что такое спасбросок?",
            retriever=fake_retriever,
            llm_client=fake_llm,
            config_path=DEFAULT_CONFIG_PATH,
        )

    assert result.answer == "Ответ [1]"
    assert len(result.chunks) == 1
    assert fake_llm.calls == 1


@pytest.mark.asyncio
async def test_answer_pipeline_handles_empty_results():
    fake_retriever = FakeRetriever([])
    fake_llm = FakeLLM()
    with patch("dnd_rag.core.pipelines.embed_texts", return_value=[[0.4]]):
        result = await answer_query_pipeline(
            "Вопрос без ответа",
            retriever=fake_retriever,
            llm_client=fake_llm,
            config_path=DEFAULT_CONFIG_PATH,
        )

    assert "Не удалось найти" in result.answer
    assert len(result.chunks) == 0
    assert fake_llm.calls == 0

