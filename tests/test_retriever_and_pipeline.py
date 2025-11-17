from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qdrant_client.models import Filter

from dnd_rag.core.config import DEFAULT_CONFIG_PATH
from dnd_rag.core.pipelines import answer_query_pipeline
from dnd_rag.core.retriever import RetrievedChunk, Retriever


class FakeQdrantClient:
    def __init__(self) -> None:
        self.kwargs = None

    def search(self, **kwargs):
        self.kwargs = kwargs
        return [
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


class RetrieverTests(unittest.TestCase):
    def test_retriever_filters_and_threshold(self):
        client = FakeQdrantClient()
        retriever = Retriever(client=client, collection="col")

        results = retriever.search(
            [0.1, 0.2],
            limit=5,
            score_threshold=0.5,
            query_filter={"must": []},
        )

        self.assertEqual(len(results), 1)
        self.assertIsInstance(client.kwargs["query_filter"], Filter)
        self.assertEqual(client.kwargs["score_threshold"], 0.5)
        self.assertEqual(results[0].chunk_id, "a")


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.last_vector = None

    def search(self, query_vector, **kwargs):
        self.last_vector = list(query_vector)
        return self.chunks


class FakeLLM:
    def __init__(self):
        self.model = "fake-llm"
        self.calls = 0

    def generate(self, messages, temperature=None):
        self.calls += 1
        return SimpleNamespace(
            content="Ответ [1]",
            model=self.model,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )


class AnswerPipelineTests(unittest.TestCase):
    def setUp(self):
        self.sample_chunk = RetrievedChunk(
            chunk_id="phb_ch01_0001",
            text="Содержимое правила",
            score=0.92,
            payload={"chunk_id": "phb_ch01_0001", "book_title": "PHB"},
        )

    def test_answer_pipeline_uses_retrieved_chunks(self):
        fake_retriever = FakeRetriever([self.sample_chunk])
        fake_llm = FakeLLM()
        with patch("dnd_rag.core.pipelines.embed_texts", return_value=[[0.1, 0.2]]):
            result = answer_query_pipeline(
                "Что такое спасбросок?",
                retriever=fake_retriever,
                llm_client=fake_llm,
                config_path=DEFAULT_CONFIG_PATH,
            )

        self.assertEqual(result.answer, "Ответ [1]")
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(fake_llm.calls, 1)

    def test_answer_pipeline_handles_empty_results(self):
        fake_retriever = FakeRetriever([])
        fake_llm = FakeLLM()
        with patch("dnd_rag.core.pipelines.embed_texts", return_value=[[0.4]]):
            result = answer_query_pipeline(
                "Вопрос без ответа",
                retriever=fake_retriever,
                llm_client=fake_llm,
                config_path=DEFAULT_CONFIG_PATH,
            )

        self.assertIn("Не удалось найти", result.answer)
        self.assertEqual(len(result.chunks), 0)
        self.assertEqual(fake_llm.calls, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

