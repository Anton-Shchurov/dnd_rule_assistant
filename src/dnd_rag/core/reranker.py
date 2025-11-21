"""
Reranker module using CrossEncoder.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from sentence_transformers import CrossEncoder

from dnd_rag.core.retriever import RetrievedChunk


class Reranker:
    """
    Reranks retrieved chunks using a Cross-Encoder model.
    Singleton-like usage recommended to avoid reloading model.
    """

    _instance: Optional[Reranker] = None

    def __new__(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Reranker:
        if cls._instance is None:
            cls._instance = super(Reranker, cls).__new__(cls)
            cls._instance.model = CrossEncoder(model_name)
        return cls._instance

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        # Initialization handled in __new__ to ensure singleton pattern for model loading
        pass

    def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_n: int = 5,
    ) -> List[RetrievedChunk]:
        """
        Rerank chunks based on relevance to the query.
        """
        if not chunks:
            return []

        # Prepare pairs for CrossEncoder
        pairs = [[query, chunk.text] for chunk in chunks]
        
        # Predict scores
        scores = self.model.predict(pairs)

        # Attach scores to chunks (temporarily or update them)
        # We will create a list of (score, chunk) tuples
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            # Update the score of the chunk to the reranker score
            # Note: This overwrites the vector similarity score. 
            # If we want to keep both, we might need to store it elsewhere, 
            # but for RAG generation, relevance is what matters.
            chunk.score = float(scores[i])
            scored_chunks.append(chunk)

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x.score, reverse=True)

        # Return top_n
        return scored_chunks[:top_n]
