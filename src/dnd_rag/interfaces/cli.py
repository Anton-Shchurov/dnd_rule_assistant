from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich import print
from rich.table import Table

from dnd_rag.core.pipelines import (
    parse_docs_pipeline,
    sections_from_md_pipeline,
    chunks_from_sections_pipeline,
    answer_query_pipeline,
    AnswerResult,
)
from dnd_rag.core.config import DEFAULT_CONFIG_PATH
from dnd_rag.providers.vectorstore import get_client, ensure_collection, upsert_vectors
from dnd_rag.providers.embeddings import embed_texts
from dnd_rag.providers.llm import LLMClient


app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("docs-parse")
def docs_parse(
    in_dir: Path = typer.Option(Path("data/raw"), "--in", help="Папка с PDF"),
    out_dir: Path = typer.Option(Path("data/processed/md"), "--out", help="Папка для Markdown"),
    ocr: bool = typer.Option(False, help="Включить OCR (обычно не требуется)"),
):
    produced = parse_docs_pipeline(in_dir, out_dir, ocr=ocr)
    for p in produced:
        print(f"[green]MD сохранён[/green]: {p}")


@app.command("sections")
def sections_cmd(
    in_dir: Path = typer.Option(Path("data/processed/md_clean"), "--in", help="Папка с Markdown"),
    out_dir: Path = typer.Option(Path("data/processed/sections"), "--out", help="Папка для JSONL секций"),
):
    produced = sections_from_md_pipeline(in_dir, out_dir)
    for p in produced:
        print(f"[green]Sections сохранены[/green]: {p}")


@app.command("chunks")
def chunks_cmd(
    in_dir: Path = typer.Option(Path("data/processed/sections"), "--in", help="Папка с JSONL секциями"),
    out_dir: Path = typer.Option(Path("data/processed/chunks"), "--out", help="Папка для JSONL чанков"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Путь к YAML-конфигу"),
):
    produced = chunks_from_sections_pipeline(in_dir, out_dir, config_path=config)
    for p in produced:
        print(f"[green]Chunks сохранены[/green]: {p}")


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _format_meta(payload: dict) -> str:
    book = payload.get("book_title") or payload.get("book") or ""
    chapter = payload.get("chapter_title") or payload.get("chapter") or ""
    sec_path = payload.get("section_path") or []
    if isinstance(sec_path, list):
        sec_path = " › ".join([s for s in sec_path if s])
    parts = [part for part in (book, chapter, sec_path) if part]
    chunk_id = payload.get("chunk_id")
    if chunk_id:
        parts.append(f"id={chunk_id}")
    return " / ".join(parts) if parts else (chunk_id or "—")


def _print_answer(result: AnswerResult, *, show_debug: bool = False) -> None:
    print("[bold cyan]Ответ[/bold cyan]:")
    print(result.answer.strip())
    print()

    if result.chunks:
        table = Table(title="Источники")
        table.add_column("#", justify="right", style="cyan")
        table.add_column("Описание")
        table.add_column("Фрагмент")
        for idx, chunk in enumerate(result.chunks, start=1):
            preview = (chunk.text or "").replace("\n", " ").strip()
            if len(preview) > 140:
                preview = preview[:140].rstrip() + "…"
            table.add_row(str(idx), _format_meta(chunk.payload), preview or "—")
        print(table)
    else:
        print("[yellow]Источники не найдены[/yellow]")

    if show_debug:
        diag = result.diagnostics
        if diag and diag.final_chunks:
            dbg = Table(title="Диагностика контекста")
            dbg.add_column("#", justify="right", style="cyan")
            dbg.add_column("chunk_id")
            dbg.add_column("Vector", justify="right")
            dbg.add_column("Rerank", justify="right")
            dbg.add_column("Источник")

            def _fmt_score(value: Optional[float]) -> str:
                return f"{value:.3f}" if value is not None else "—"

            for chunk in diag.final_chunks:
                sections = " › ".join(chunk.section_path) if chunk.section_path else ""
                source_parts = [p for p in (chunk.book_title, chunk.chapter_title, sections) if p]
                dbg.add_row(
                    str(chunk.rank),
                    chunk.chunk_id,
                    _fmt_score(chunk.vector_score),
                    _fmt_score(chunk.rerank_score),
                    " / ".join(source_parts) or chunk.chunk_id,
                )
            print(dbg)
            print(
                f"[dim]retrieved={len(diag.retrieved)} reranked={len(diag.reranked)} "
                f"duration={int(diag.duration_ms or 0)}ms[/dim]"
            )
        else:
            print("[yellow]Диагностика недоступна[/yellow]")

    if result.total_tokens is not None:
        print(
            f"[dim]Tokens: prompt={result.prompt_tokens} "
            f"completion={result.completion_tokens} total={result.total_tokens}[/dim]"
        )


@app.command("docs-sample")
def docs_sample(
    in_path: Path = typer.Option(Path("data/processed/chunks"), "--in", help="Папка или файл JSONL"),
    q: str = typer.Option(..., "--q", help="Запрос"),
    k: int = typer.Option(5, "-k", help="Количество результатов"),
):
    files: List[Path] = []
    if in_path.is_dir():
        files = sorted(in_path.glob("*.jsonl"))
    else:
        files = [in_path]

    rows: List[dict] = []
    for fp in files:
        rows.extend(_read_jsonl(fp))

    q_terms = [t for t in q.lower().split() if t]
    def score(text: str) -> int:
        low = text.lower()
        return sum(low.count(t) for t in q_terms)

    rows.sort(key=lambda r: score(r.get("text", "")), reverse=True)
    top = rows[:k]

    table = Table(title=f"Top-{k} по запросу: {q}")
    table.add_column("book")
    table.add_column("chapter")
    table.add_column("section")
    table.add_column("pages")
    table.add_column("preview")

    for r in top:
        pages = (
            f"{r.get('page_start')}-{r.get('page_end')}"
            if r.get("page_start") is not None and r.get("page_end") is not None
            else ""
        )
        preview = (r.get("text", "").replace("\n", " ")[:140] + "…") if r.get("text") else ""
        table.add_row(
            str(r.get("book", "")),
            str(r.get("chapter", "")),
            str(r.get("section", "")),
            pages,
            preview,
        )

    print(table)


@app.command("ask")
def ask_cmd(
    question: str = typer.Argument(..., help="Вопрос пользователя"),
    collection: str = typer.Option("dnd_rule_assistant", "--collection", help="Имя коллекции Qdrant"),
    host: str = typer.Option("localhost", "--host", envvar="QDRANT_HOST"),
    port: int = typer.Option(6333, "--port", envvar="QDRANT_PORT"),
    k: int = typer.Option(5, "--k", help="Количество фрагментов контекста"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Путь к ingest.yaml"),
    embedding_model: str = typer.Option(
        "text-embedding-3-small", "--embedding-model", help="Модель эмбеддингов OpenAI"
    ),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", help="Переопределить модель LLM"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="Температура LLM"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Использовать переранжирование"),
    debug_log: bool = typer.Option(False, "--debug-log", help="Логировать запрос и показать служебную таблицу"),
):
    if not question.strip():
        typer.secho("Вопрос не должен быть пустым.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    llm_client = LLMClient(model=llm_model) if llm_model else None
    
    import asyncio
    result = asyncio.run(answer_query_pipeline(
        question,
        collection=collection,
        host=host,
        port=port,
        k=k,
        config_path=config,
        llm_client=llm_client,
        embedding_model=embedding_model,
        temperature=temperature,
        rerank=rerank,
        log_queries=debug_log,
        include_diagnostics=debug_log,
    ))
    _print_answer(result, show_debug=debug_log)


@app.command("init-qdrant")
def init_qdrant(
    collection: str = typer.Option("dnd_rule_assistant", "--collection", help="Имя коллекции"),
    host: str = typer.Option("localhost", "--host", envvar="QDRANT_HOST"),
    port: int = typer.Option(6333, "--port", envvar="QDRANT_PORT"),
    dim: int = typer.Option(1536, "--dim", help="Размерность эмбеддинга"),
):
    client = get_client(host=host, port=port)
    ensure_collection(client, collection, vector_size=dim)
    print(f"[green]Qdrant коллекция готова[/green]: {collection} (dim={dim})")


@app.command("index")
def index_cmd(
    chunks: List[Path] = typer.Argument(..., help="Список путей к JSONL с чанками"),
    collection: str = typer.Option("dnd_rule_assistant", "--collection", help="Имя коллекции"),
    host: str = typer.Option("localhost", "--host", envvar="QDRANT_HOST"),
    port: int = typer.Option(6333, "--port", envvar="QDRANT_PORT"),
    model: str = typer.Option("text-embedding-3-small", "--model", help="Модель эмбеддингов OpenAI"),
):
    client = get_client(host=host, port=port)
    # размерность 1536 для text-embedding-3-small
    ensure_collection(client, collection, vector_size=1536)

    # Считываем чанки
    rows: List[dict] = []
    for fp in chunks:
        rows.extend(_read_jsonl(fp))

    texts = [r.get("text", "") for r in rows]
    ids = [r.get("chunk_id", "") for r in rows]
    payloads: List[dict] = []
    for r in rows:
        payloads.append(
            {
                "chunk_id": r.get("chunk_id", ""),
                "book_title": r.get("book_title", ""),
                "chapter_title": r.get("chapter_title", ""),
                "section_path": r.get("section_path", []),
                "chunk_index": r.get("chunk_index", 0),
                "text": r.get("text", ""),
            }
        )

    vectors = embed_texts(texts, model=model)
    upsert_vectors(client, collection, ids=ids, vectors=vectors, payloads=payloads)

    print(f"[green]Загружено в Qdrant[/green]: {len(rows)} точек → {collection}")


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

