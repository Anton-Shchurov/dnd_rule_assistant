from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _safe_preview(text: str, limit: int = 320) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "…"


@dataclass
class ChunkDiagnostics:
    """
    Snapshot with metadata for a chunk participating in retrieval/rerank.
    """

    rank: int
    chunk_id: str
    vector_score: Optional[float]
    rerank_score: Optional[float]
    book_title: Optional[str] = None
    chapter_title: Optional[str] = None
    section_path: Sequence[str] = field(default_factory=list)
    chunk_index: Optional[int] = None
    text_preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "vector_score": self.vector_score,
            "rerank_score": self.rerank_score,
            "book_title": self.book_title,
            "chapter_title": self.chapter_title,
            "section_path": list(self.section_path or []),
            "chunk_index": self.chunk_index,
            "text_preview": self.text_preview,
        }

    @classmethod
    def from_chunk(
        cls,
        chunk: "RetrievedChunk",
        *,
        rank: int,
        vector_score: Optional[float],
        rerank_score: Optional[float],
        preview_chars: int = 320,
    ) -> "ChunkDiagnostics":
        payload = chunk.payload or {}
        return cls(
            rank=rank,
            chunk_id=str(payload.get("chunk_id") or chunk.chunk_id),
            vector_score=vector_score,
            rerank_score=rerank_score,
            book_title=payload.get("book_title"),
            chapter_title=payload.get("chapter_title"),
            section_path=payload.get("section_path") or [],
            chunk_index=payload.get("chunk_index"),
            text_preview=_safe_preview(payload.get("text") or chunk.text or "", preview_chars),
        )


@dataclass
class QueryDiagnostics:
    """
    Aggregated diagnostics for a single user question.
    """

    question: str
    answer: str
    answer_found: bool
    requested_k: int
    initial_k: int
    rerank_enabled: bool
    filters: Optional[Dict[str, Any]]
    embedding_model: str
    llm_model: str
    timestamp_utc: str
    duration_ms: Optional[float]
    retrieved: List[ChunkDiagnostics] = field(default_factory=list)
    reranked: List[ChunkDiagnostics] = field(default_factory=list)
    final_chunks: List[ChunkDiagnostics] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "answer_found": self.answer_found,
            "requested_k": self.requested_k,
            "initial_k": self.initial_k,
            "rerank_enabled": self.rerank_enabled,
            "filters": self.filters,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "timestamp_utc": self.timestamp_utc,
            "duration_ms": self.duration_ms,
            "retrieved": [c.to_dict() for c in self.retrieved],
            "reranked": [c.to_dict() for c in self.reranked],
            "final_chunks": [c.to_dict() for c in self.final_chunks],
            "extra": self.extra,
            "error": self.error,
        }


def write_query_log(diagnostics: QueryDiagnostics, directory: str | Path) -> Path:
    """
    Append diagnostics as JSONL into logs/queries/YYYY-MM-DD.jsonl.
    """

    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc)
    log_path = out_dir / f"{ts:%Y-%m-%d}.jsonl"
    payload = diagnostics.to_dict()

    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return log_path

